"""Reset events for SO-101 Bench."""

from __future__ import annotations

import math
import random
from pathlib import Path

import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaacsim.core.simulation_manager import SimulationManager

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import get_current_stage
from isaacsim.core.prims import XFormPrim as XformPrimView

from so101_bench.benchmark import (
    DIRECTIONS,
    MAX_GRASP_ATTEMPTS,
    TASK_BETWEEN,
    TASK_BIN,
    TASK_FAMILIES,
    TASK_MIXED,
    TASK_MOVE,
    TASK_NEXT_TO,
    load_object_move_footprint_boxes,
    object_rigid_body_child_names,
    task_instruction,
)

ROBOT_COLORS = {
    "orange": (0.95, 0.15, 0.02),
    "teal": (0.0, 0.8, 0.502),
    "white": (0.95, 0.95, 0.95),
    "black": (0.08, 0.08, 0.08),
}


def _env_step_dt(env) -> float:
    step_dt = getattr(env, "step_dt", env.cfg.sim.dt * env.cfg.decimation)
    return float(step_dt)


def _episode_age_s(env) -> torch.Tensor:
    return env.episode_length_buf.to(dtype=torch.float32) * _env_step_dt(env)


def _env_ids_tensor(env, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    env_ids = env_ids.to(device=env.device)
    if env_ids.dtype == torch.bool:
        return torch.nonzero(env_ids, as_tuple=False).flatten()
    return env_ids.to(dtype=torch.long)


BenchmarkObject = RigidObject | Articulation | DeformableObject | XformPrimView
BinPose = tuple[tuple[float, float, float], tuple[float, float, float]]
DEFAULT_HALF_EXTENTS = (0.02, 0.02, 0.02)
DEFAULT_FOOTPRINT_HALF_EXTENTS = (0.02, 0.02)
DEFAULT_FOOTPRINT_CENTER_OFFSET = (0.0, 0.0)


def benchmark_object_positions(env, object_asset_names: list[str]) -> torch.Tensor:
    positions = []
    multi_info = getattr(env, "_so101_multi_rigid_body_info", {}) or {}
    for name in object_asset_names:
        asset = env.scene[name]
        if isinstance(asset, XformPrimView):
            views = multi_info.get(name)
            if views:
                # Use the first inner rigid body's physics position; subtract its
                # cached local offset to recover the wrapper-frame position.
                view_info = views[0]
                transforms = view_info["view"].get_transforms()
                if not isinstance(transforms, torch.Tensor):
                    transforms = torch.as_tensor(transforms)
                transforms = transforms.to(env.device)
                child_pos = transforms[..., :3]
                child_quat_xyzw = transforms[..., 3:7]
                child_quat_wxyz = math_utils.convert_quat(child_quat_xyzw, to="wxyz")
                local_pos = view_info["local_pos"].to(env.device).expand_as(child_pos)
                # wrapper_pos = child_pos - R(child_quat) * R(local_quat)^-1 * local_pos
                # But child_quat = wrapper_quat * local_quat, so
                # wrapper_quat = child_quat * local_quat^-1, and wrapper_pos = child_pos - R(wrapper_quat) * local_pos.
                local_quat_inv = math_utils.quat_inv(
                    view_info["local_quat"].to(env.device).expand_as(child_quat_wxyz)
                )
                wrapper_quat = math_utils.quat_mul(child_quat_wxyz, local_quat_inv)
                wrapper_pos = child_pos - math_utils.quat_apply(wrapper_quat, local_pos)
                positions.append(wrapper_pos)
            else:
                positions.append(asset.get_world_poses()[0])
        else:
            positions.append(asset.data.root_pos_w)
    return torch.stack(positions, dim=1)


def benchmark_object_yaws(env, object_asset_names: list[str]) -> torch.Tensor:
    yaws = []
    multi_info = getattr(env, "_so101_multi_rigid_body_info", {}) or {}
    for name in object_asset_names:
        asset = env.scene[name]
        if isinstance(asset, XformPrimView):
            views = multi_info.get(name)
            if views:
                view_info = views[0]
                transforms = view_info["view"].get_transforms()
                if not isinstance(transforms, torch.Tensor):
                    transforms = torch.as_tensor(transforms)
                transforms = transforms.to(env.device)
                child_quat_xyzw = transforms[..., 3:7]
                child_quat_wxyz = math_utils.convert_quat(child_quat_xyzw, to="wxyz")
                local_quat_inv = math_utils.quat_inv(
                    view_info["local_quat"].to(env.device).expand_as(child_quat_wxyz)
                )
                wrapper_quat = math_utils.quat_mul(child_quat_wxyz, local_quat_inv)
                yaws.append(_quat_yaw(wrapper_quat))
            else:
                quat = asset.get_world_poses()[1]
                if not isinstance(quat, torch.Tensor):
                    quat = torch.as_tensor(quat)
                yaws.append(_quat_yaw(quat.to(env.device)))
        else:
            yaws.append(_quat_yaw(asset.data.root_quat_w))
    return torch.stack(yaws, dim=1)


def _quat_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _prim_bbox_half_extents(
    stage: Usd.Stage,
    bbox_cache: UsdGeom.BBoxCache,
    prim_path: str,
) -> tuple[float, float, float]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return DEFAULT_HALF_EXTENTS

    bbox_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    minimum = bbox_range.GetMin()
    maximum = bbox_range.GetMax()
    extents = tuple(max(0.5 * abs(float(maximum[i] - minimum[i])), 0.002) for i in range(3))
    if not all(math.isfinite(extent) for extent in extents):
        return DEFAULT_HALF_EXTENTS
    return extents


def _scene_half_extents(env, asset_name: str, bbox_cache: UsdGeom.BBoxCache, stage: Usd.Stage) -> torch.Tensor:
    asset_cfg = getattr(env.scene.cfg, asset_name)
    prim_paths = sim_utils.find_matching_prim_paths(asset_cfg.prim_path)
    extents = [_prim_bbox_half_extents(stage, bbox_cache, prim_path) for prim_path in prim_paths]
    if len(extents) != env.num_envs:
        extents = [*extents[: env.num_envs], *[DEFAULT_HALF_EXTENTS] * max(0, env.num_envs - len(extents))]
    return torch.tensor(extents, dtype=torch.float32, device=env.device)


def _coerce_footprint_pair(
    raw_value,
    fallback: tuple[float, float],
    *,
    min_value: float | None = None,
) -> tuple[float, float]:
    if isinstance(raw_value, torch.Tensor):
        raw_value = raw_value.detach().cpu().tolist()
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) < 2:
        return fallback
    first = float(raw_value[0])
    second = float(raw_value[1])
    if min_value is not None:
        first = max(first, min_value)
        second = max(second, min_value)
    if not math.isfinite(first) or not math.isfinite(second):
        return fallback
    return (first, second)


def _usd_footprint(
    usd_path: str | Path | None,
    fallback_half_extents: tuple[float, float] = DEFAULT_FOOTPRINT_HALF_EXTENTS,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if usd_path is None:
        return fallback_half_extents, DEFAULT_FOOTPRINT_CENTER_OFFSET
    try:
        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise RuntimeError(f"could not open {usd_path}")
        prim = stage.GetDefaultPrim()
        if prim is None or not prim.IsValid():
            prim = stage.GetPseudoRoot()
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        bbox_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum = bbox_range.GetMin()
        maximum = bbox_range.GetMax()
        half_extents = (
            max(0.5 * abs(float(maximum[0] - minimum[0])), 0.002),
            max(0.5 * abs(float(maximum[1] - minimum[1])), 0.002),
        )
        center_offset = (
            0.5 * (float(minimum[0]) + float(maximum[0])),
            0.5 * (float(minimum[1]) + float(maximum[1])),
        )
        if not all(math.isfinite(value) for value in (*half_extents, *center_offset)):
            raise RuntimeError(f"non-finite footprint for {usd_path}")
        return half_extents, center_offset
    except Exception:
        return fallback_half_extents, DEFAULT_FOOTPRINT_CENTER_OFFSET


def _asset_usd_path(env, asset_name: str) -> str | None:
    asset_cfg = getattr(env.scene.cfg, asset_name, None)
    spawn_cfg = getattr(asset_cfg, "spawn", None)
    usd_path = getattr(spawn_cfg, "usd_path", None)
    return str(usd_path) if usd_path is not None else None


def _record_benchmark_geometry(
    env,
    object_asset_names: list[str],
    bin_name: str,
    object_labels: list[str] | None = None,
) -> None:
    """Cache per-slot AABB and USD footprint geometry used by task checks."""

    stage = get_current_stage()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    env._so101_object_half_extents = torch.stack(
        [_scene_half_extents(env, asset_name, bbox_cache, stage) for asset_name in object_asset_names],
        dim=1,
    )
    env._so101_bin_half_extents = _scene_half_extents(env, bin_name, bbox_cache, stage)
    object_footprints = [
        _usd_footprint(
            _asset_usd_path(env, asset_name),
            _coerce_footprint_pair(
                env._so101_object_half_extents[:, object_id, :2].amax(dim=0),
                DEFAULT_FOOTPRINT_HALF_EXTENTS,
                min_value=0.002,
            ),
        )
        for object_id, asset_name in enumerate(object_asset_names)
    ]
    env._so101_object_footprint_half_extents = torch.tensor(
        [footprint[0] for footprint in object_footprints],
        dtype=torch.float32,
        device=env.device,
    ).unsqueeze(0).repeat(env.num_envs, 1, 1)
    env._so101_object_footprint_center_offsets = torch.tensor(
        [footprint[1] for footprint in object_footprints],
        dtype=torch.float32,
        device=env.device,
    ).unsqueeze(0).repeat(env.num_envs, 1, 1)
    env._so101_object_move_footprint_boxes = [
        torch.tensor(
            load_object_move_footprint_boxes(object_label, required=False),
            dtype=torch.float32,
            device=env.device,
        ).reshape(-1, 4)
        if object_labels is not None and object_id < len(object_labels)
        else torch.empty((0, 4), dtype=torch.float32, device=env.device)
        for object_id, object_label in enumerate(object_labels or object_asset_names)
    ]
    bin_footprint_half, bin_footprint_offset = _usd_footprint(
        _asset_usd_path(env, bin_name),
        _coerce_footprint_pair(
            env._so101_bin_half_extents[:, :2].amax(dim=0),
            DEFAULT_FOOTPRINT_HALF_EXTENTS,
            min_value=0.002,
        ),
    )
    env._so101_bin_footprint_half_extents = torch.tensor(
        bin_footprint_half,
        dtype=torch.float32,
        device=env.device,
    ).unsqueeze(0).repeat(env.num_envs, 1)
    env._so101_bin_footprint_center_offsets = torch.tensor(
        bin_footprint_offset,
        dtype=torch.float32,
        device=env.device,
    ).unsqueeze(0).repeat(env.num_envs, 1)
    _ensure_multi_rigid_body_views(env, object_asset_names, object_labels)


def _gf_quat_to_wxyz(quat: Gf.Quatd) -> tuple[float, float, float, float]:
    imaginary = quat.GetImaginary()
    return (
        float(quat.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    )


def _ensure_multi_rigid_body_views(
    env,
    object_asset_names: list[str],
    object_labels: list[str] | None = None,
) -> None:
    """Discover inner rigid bodies and create PhysX views for multi-rigid-body assets.

    Assets that wrap multiple PhysX rigid bodies (e.g., shoes loaded as
    ``AssetBaseCfg``) cannot be teleported by writing the wrapper's xformOp,
    because PhysX overwrites the USD xform each step. We instead teleport each
    inner rigid body via a PhysX ``RigidBodyView`` so the layout pose is applied
    to the actual physics state. This is done once and cached on ``env``.
    """

    if getattr(env, "_so101_multi_rigid_body_info_built", False):
        return

    info: dict[str, list[dict]] = {}
    stage = get_current_stage()
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    physics_sim_view = SimulationManager.get_physics_sim_view()

    for asset_id, asset_name in enumerate(object_asset_names):
        asset = env.scene[asset_name]
        if not isinstance(asset, XformPrimView):
            continue
        asset_cfg = getattr(env.scene.cfg, asset_name)
        wrapper_prim_paths = sim_utils.find_matching_prim_paths(asset_cfg.prim_path)
        if not wrapper_prim_paths:
            continue
        first_path = wrapper_prim_paths[0]
        first_prim = stage.GetPrimAtPath(first_path)
        if not first_prim.IsValid():
            continue

        wrapper_to_world = xform_cache.GetLocalToWorldTransform(first_prim)
        wrapper_to_world.Orthonormalize()
        wrapper_world_inv = wrapper_to_world.GetInverse()

        object_label = (
            object_labels[asset_id] if object_labels is not None and asset_id < len(object_labels) else ""
        )
        split_child_names = object_rigid_body_child_names(object_label) if object_label else ()
        rigid_body_prims = []
        for child_name in split_child_names:
            child_prim = stage.GetPrimAtPath(f"{first_path}/{child_name}")
            if child_prim.IsValid() and child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body_prims.append(child_prim)
        if not rigid_body_prims:
            rigid_body_prims = [
                prim
                for prim in Usd.PrimRange(first_prim)
                if prim.HasAPI(UsdPhysics.RigidBodyAPI)
                and (not split_child_names or prim.GetPath() != first_prim.GetPath())
            ]

        sub_records: list[dict] = []
        for prim in rigid_body_prims:
            body_path = prim.GetPath().pathString
            rel_path = body_path[len(first_path):]
            body_to_world = xform_cache.GetLocalToWorldTransform(prim)
            body_to_world.Orthonormalize()
            body_to_wrapper = body_to_world * wrapper_world_inv
            local_pos = body_to_wrapper.ExtractTranslation()
            local_quat = body_to_wrapper.ExtractRotationQuat()
            sub_records.append(
                {
                    "rel_path": rel_path,
                    "local_pos": (
                        float(local_pos[0]),
                        float(local_pos[1]),
                        float(local_pos[2]),
                    ),
                    "local_quat_wxyz": _gf_quat_to_wxyz(local_quat),
                }
            )

        if not sub_records:
            continue

        views: list[dict] = []
        for record in sub_records:
            pattern = asset_cfg.prim_path + record["rel_path"]
            physx_pattern = pattern.replace(".*", "*")
            view = physics_sim_view.create_rigid_body_view(physx_pattern)
            views.append(
                {
                    "view": view,
                    "rel_path": record["rel_path"],
                    "local_pos": torch.tensor(record["local_pos"], dtype=torch.float32, device=env.device),
                    "local_quat": torch.tensor(record["local_quat_wxyz"], dtype=torch.float32, device=env.device),
                }
            )
        info[asset_name] = views

    env._so101_multi_rigid_body_info = info
    env._so101_multi_rigid_body_info_built = True


def _write_multi_rigid_body_pose(
    env,
    asset_name: str,
    env_id: int,
    root_pos_w: torch.Tensor,
    quat_wxyz: torch.Tensor,
) -> bool:
    """Teleport each inner rigid body of a multi-rigid-body asset to match the wrapper pose."""

    info = getattr(env, "_so101_multi_rigid_body_info", None)
    if not info:
        return False
    views = info.get(asset_name)
    if not views:
        return False

    device = root_pos_w.device
    wrapper_pos = root_pos_w.unsqueeze(0)  # (1, 3)
    wrapper_quat = quat_wxyz.unsqueeze(0)  # (1, 4) wxyz
    indices = torch.tensor([env_id], dtype=torch.int32, device=device)

    for view_info in views:
        view = view_info["view"]
        local_pos = view_info["local_pos"].to(device).unsqueeze(0)
        local_quat = view_info["local_quat"].to(device).unsqueeze(0)
        child_pos = wrapper_pos + math_utils.quat_apply(wrapper_quat, local_pos)
        child_quat_wxyz = math_utils.quat_mul(wrapper_quat, local_quat)
        child_quat_xyzw = math_utils.convert_quat(child_quat_wxyz, to="xyzw")

        data = torch.zeros((view.count, 7), dtype=torch.float32, device=device)
        data[env_id, :3] = child_pos[0]
        data[env_id, 3:7] = child_quat_xyzw[0]
        view.set_transforms(data, indices)
        velocities = torch.zeros((view.count, 6), dtype=torch.float32, device=device)
        view.set_velocities(velocities, indices)
    return True


def _ensure_robot_start_buffers(env) -> None:
    if not hasattr(env, "_so101_robot_started_moving"):
        env._so101_robot_started_moving = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_so101_robot_start_step"):
        env._so101_robot_start_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    if not hasattr(env, "_so101_robot_start_time_s"):
        env._so101_robot_start_time_s = torch.full(
            (env.num_envs,), float("nan"), dtype=torch.float32, device=env.device
        )


def mark_benchmark_robot_start(
    env,
    object_asset_names: list[str],
    bin_name: str,
    env_ids: torch.Tensor | None = None,
    force_robot_start_time: bool = False,
) -> None:
    """Record settled displacement baselines and stamp when robot control starts."""

    if not hasattr(env, "_so101_initial_object_pos_w"):
        return

    env_ids = _env_ids_tensor(env, env_ids)
    if env_ids.numel() == 0:
        return

    if not hasattr(env, "_so101_failure_object_pos_w"):
        env._so101_failure_object_pos_w = env._so101_initial_object_pos_w.clone()
    if not hasattr(env, "_so101_failure_bin_pos_w"):
        env._so101_failure_bin_pos_w = env._so101_initial_bin_pos_w.clone()
    if not hasattr(env, "_so101_failure_baseline_recorded"):
        env._so101_failure_baseline_recorded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    baseline_env_ids = env_ids[~env._so101_failure_baseline_recorded[env_ids]]
    if baseline_env_ids.numel() > 0:
        env._so101_failure_object_pos_w[baseline_env_ids] = benchmark_object_positions(env, object_asset_names)[
            baseline_env_ids
        ]
        env._so101_failure_bin_pos_w[baseline_env_ids] = env.scene[bin_name].data.root_pos_w[baseline_env_ids]
        env._so101_failure_baseline_recorded[baseline_env_ids] = True

    _ensure_robot_start_buffers(env)
    if force_robot_start_time:
        start_env_ids = env_ids
    else:
        start_env_ids = env_ids[~env._so101_robot_started_moving[env_ids]]
    if start_env_ids.numel() == 0:
        return

    env._so101_robot_started_moving[start_env_ids] = True
    env._so101_robot_start_step[start_env_ids] = env.episode_length_buf[start_env_ids]
    env._so101_robot_start_time_s[start_env_ids] = _episode_age_s(env)[start_env_ids]


def randomize_robot_color(
    env,
    env_ids: torch.Tensor | None,
    color_names: list[str] | None = None,
):
    """Set the robot visual material to one of the supported SO-101 colors."""

    if color_names is None:
        color_names = list(ROBOT_COLORS.keys())
    color = ROBOT_COLORS[random.choice(color_names)]

    with Sdf.ChangeBlock():
        robot = env.scene["robot"]
        material_prim_path = robot.cfg.prim_path + "/Looks/material_a_3d_printed/Shader"
        material_prims = sim_utils.find_matching_prims(material_prim_path)
        if not material_prims:
            return
        material_prims[0].GetAttribute("inputs:diffuse_color_constant").Set(color)


def _sample_task_family(task_family: str) -> str:
    if task_family == TASK_MIXED:
        return random.choice(TASK_FAMILIES)
    return task_family


def _required_object_count(task_family: str, object_count_range: tuple[int, int]) -> int:
    low, high = object_count_range
    if task_family == TASK_BIN:
        supported_bin_counts = [count for count in (1, 4) if low <= count <= high]
        if not supported_bin_counts:
            raise ValueError(f"Bin tasks need one or four active objects, got range {object_count_range}.")
        return random.choice(supported_bin_counts)
    if task_family == TASK_NEXT_TO:
        low = max(low, 4)
    elif task_family == TASK_BETWEEN:
        low = max(low, 4)
    elif task_family == TASK_MOVE:
        if low > 4 or high < 4:
            raise ValueError(f"Move tasks need four active objects, got range {object_count_range}.")
        return 4
    high = max(high, low)
    return random.randint(low, high)


def _active_object_ids(
    num_objects: int,
    active_count: int,
    selection: str,
    fixed_active_object_ids: tuple[int, ...] | None = None,
) -> list[int]:
    if selection == "prefix":
        return list(range(active_count))
    if selection == "random":
        return random.sample(range(num_objects), active_count)
    if selection == "fixed":
        if fixed_active_object_ids is None:
            raise ValueError("fixed_active_object_ids must be set when active_object_selection='fixed'.")
        if len(fixed_active_object_ids) != active_count:
            raise ValueError(
                f"Expected {active_count} fixed active object ids, got {len(fixed_active_object_ids)}."
            )
        active_ids = list(fixed_active_object_ids)
        invalid_ids = [object_id for object_id in active_ids if object_id < 0 or object_id >= num_objects]
        if invalid_ids:
            raise ValueError(f"Fixed active object ids are out of range for {num_objects} objects: {invalid_ids}.")
        if len(set(active_ids)) != len(active_ids):
            raise ValueError(f"Fixed active object ids must be unique, got {active_ids}.")
        return active_ids
    raise ValueError(
        f"Unknown active object selection mode: {selection!r}. Expected 'prefix', 'random', or 'fixed'."
    )


def _sample_positions(
    count: int,
    table_bounds: dict[str, tuple[float, float]],
    min_spacing: float,
) -> list[tuple[float, float]]:
    positions: list[tuple[float, float]] = []
    x_min, x_max = table_bounds["x"]
    y_min, y_max = table_bounds["y"]

    for _ in range(count):
        candidate = (0.0, 0.0)
        for _attempt in range(100):
            candidate = (random.uniform(x_min, x_max), random.uniform(y_min, y_max))
            if all(math.dist(candidate, existing) >= min_spacing for existing in positions):
                break
        positions.append(candidate)
    return positions


def _point_in_xy_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y)
        if intersects:
            x_on_edge = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_on_edge:
                inside = not inside
        j = i
    return inside


def _sample_positions_in_polygon(
    count: int,
    polygon_vertices: list[tuple[float, float, float]] | tuple[tuple[float, float, float], ...],
    min_spacing: float,
) -> list[tuple[float, float]]:
    polygon = [(float(vertex[0]), float(vertex[1])) for vertex in polygon_vertices]
    if len(polygon) < 3:
        raise ValueError(f"Expected a spawn polygon with at least 3 vertices, got {polygon_vertices!r}.")

    x_values = [point[0] for point in polygon]
    y_values = [point[1] for point in polygon]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    positions: list[tuple[float, float]] = []

    for _ in range(count):
        candidate = polygon[0]
        fallback_candidate: tuple[float, float] | None = None
        for _attempt in range(500):
            candidate = (random.uniform(x_min, x_max), random.uniform(y_min, y_max))
            if not _point_in_xy_polygon(candidate, polygon):
                continue
            fallback_candidate = candidate
            if all(math.dist(candidate, existing) >= min_spacing for existing in positions):
                break
        else:
            if fallback_candidate is None:
                raise RuntimeError(f"Could not sample an object position inside spawn polygon {polygon_vertices!r}.")
            candidate = fallback_candidate
        positions.append(candidate)
    return positions


def _fixed_positions(
    active_object_ids: list[int],
    object_fixed_poses: tuple[tuple[float, float, float], ...],
) -> list[tuple[float, float]]:
    return [
        (object_fixed_poses[object_id][0], object_fixed_poses[object_id][1])
        for object_id in active_object_ids
    ]


def _inactive_position(
    base_pos: tuple[float, float, float],
    spacing: float,
    object_id: int,
) -> tuple[float, float, float]:
    return (base_pos[0] + spacing * object_id, base_pos[1], base_pos[2])


def _yaw_quat(yaw: float, device: str) -> torch.Tensor:
    return math_utils.quat_from_euler_xyz(
        torch.zeros(1, device=device),
        torch.zeros(1, device=device),
        torch.tensor([yaw], device=device),
    )[0]


def _rpy_quat(rpy: tuple[float, float, float], device: str) -> torch.Tensor:
    return math_utils.quat_from_euler_xyz(
        torch.tensor([rpy[0]], device=device),
        torch.tensor([rpy[1]], device=device),
        torch.tensor([rpy[2]], device=device),
    )[0]


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind()
    w2, x2, y2, z2 = q2.unbind()
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def _bin_quat(yaw: float, root_rotation: tuple[float, float, float], device: str) -> torch.Tensor:
    return _quat_mul(_yaw_quat(yaw, device), _rpy_quat(root_rotation, device))


def _write_pose(
    env,
    asset: BenchmarkObject,
    env_id: int,
    pos: tuple[float, float, float],
    quat: torch.Tensor,
    asset_name: str | None = None,
):
    env_ids = torch.tensor([env_id], dtype=torch.long, device=asset.device)
    root_pos_w = torch.tensor(pos, dtype=torch.float32, device=asset.device).unsqueeze(0)
    root_pos_w += env.scene.env_origins[env_ids]

    if isinstance(asset, XformPrimView):
        quat_w = quat.to(asset.device)
        asset.set_world_poses(root_pos_w, quat_w.unsqueeze(0), indices=env_ids)
        if asset_name is not None:
            _write_multi_rigid_body_pose(env, asset_name, env_id, root_pos_w[0], quat_w)
        return root_pos_w[0]

    if isinstance(asset, DeformableObject):
        nodal_state = asset.data.default_nodal_state_w[env_ids].clone()
        default_root_pos_w = nodal_state[..., :3].mean(dim=1)
        translation = root_pos_w - default_root_pos_w
        nodal_state[..., :3] = asset.transform_nodal_pos(
            nodal_state[..., :3],
            pos=translation,
            quat=quat.to(asset.device).unsqueeze(0),
        )
        nodal_state[..., 3:] = 0.0
        asset.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)
        return root_pos_w[0]

    root_pose = torch.zeros((1, 7), device=asset.device)
    root_pose[:, :3] = root_pos_w
    root_pose[:, 3:7] = quat.to(asset.device).unsqueeze(0)
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=asset.device), env_ids=env_ids)
    return root_pose[0, :3]


def _default_root_z(asset: BenchmarkObject, env_id: int) -> float:
    if isinstance(asset, XformPrimView):
        return float(asset.get_world_poses(indices=[env_id])[0][0, 2].item())
    if isinstance(asset, DeformableObject):
        return float(asset.data.default_nodal_state_w[env_id, :, 2].mean().item())
    return float(asset.data.default_root_state[env_id, 2].item())


def _layout_object_entries(episode_layout: dict | None) -> dict[int, dict]:
    if episode_layout is None:
        return {}
    entries: dict[int, dict] = {}
    for entry in episode_layout.get("objects", []):
        object_id = int(entry["slot"])
        entries[object_id] = entry
    return entries


def _layout_object_position(entry: dict) -> tuple[float, float, float]:
    position = entry.get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        raise ValueError(f"Episode layout object entry is missing a 3D position: {entry!r}.")
    return (float(position[0]), float(position[1]), float(position[2]))


def _layout_object_yaw(entry: dict) -> float:
    if "yaw" in entry:
        return float(entry["yaw"])
    rpy = entry.get("rpy")
    if isinstance(rpy, (list, tuple)) and len(rpy) >= 3:
        return float(rpy[2])
    raise ValueError(f"Episode layout object entry is missing a yaw/rpy rotation: {entry!r}.")


def _layout_bin_pose(
    episode_layout: dict | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if episode_layout is None:
        return None
    bin_entry = episode_layout.get("bin")
    if not isinstance(bin_entry, dict):
        raise ValueError("Episode layout is missing a 'bin' pose entry.")
    position = bin_entry.get("position")
    rpy = bin_entry.get("rpy")
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        raise ValueError(f"Episode layout bin entry is missing a 3D position: {bin_entry!r}.")
    if not isinstance(rpy, (list, tuple)) or len(rpy) < 3:
        raise ValueError(f"Episode layout bin entry is missing an RPY rotation: {bin_entry!r}.")
    return (
        (float(position[0]), float(position[1]), float(position[2])),
        (float(rpy[0]), float(rpy[1]), float(rpy[2])),
    )


def _reset_tensor_rows(
    env,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    env_ids: torch.Tensor,
    fill_value: float | int | bool,
) -> torch.Tensor:
    value = getattr(env, name, None)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        value = torch.full(shape, fill_value, dtype=dtype, device=env.device)
        setattr(env, name, value)
    else:
        value[env_ids] = fill_value
    return value


def _reset_list_rows(env, name: str, env_ids: torch.Tensor, fill_value):
    values = getattr(env, name, None)
    if not isinstance(values, list) or len(values) != env.num_envs:
        values = [fill_value for _ in range(env.num_envs)]
        setattr(env, name, values)
    else:
        for env_id in env_ids.tolist():
            values[env_id] = fill_value
    return values


def reset_benchmark_scene(
    env,
    env_ids: torch.Tensor,
    object_asset_names: list[str],
    bin_name: str,
    object_labels: list[str],
    task_family: str = TASK_MIXED,
    object_count_range: tuple[int, int] = (1, 4),
    table_bounds: dict[str, tuple[float, float]] | None = None,
    table_top_z: float = 0.032,
    min_object_spacing: float = 0.105,
    bin_fixed_pose: tuple[float, float, float] = (0.34, 0.16, 0.0),
    bin_root_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    bin_z: float | None = None,
    object_fixed_poses: tuple[tuple[float, float, float], ...] | None = None,
    randomize_bin_for_bin_task: bool = False,
    bin_random_poses: tuple[BinPose, ...] = (),
    valid_spawn_regions: list[list[tuple[float, float, float]]] | None = None,
    active_object_selection: str = "prefix",
    fixed_active_object_ids: tuple[int, ...] | None = None,
    shuffle_object_labels: bool = True,
    force_bin_all_objects_instruction: bool = False,
    episode_spec: dict | None = None,
    episode_layout: dict | None = None,
    inactive_object_base_pos: tuple[float, float, float] = (20.0, 20.0, -10.0),
    inactive_object_spacing: float = 0.25,
):
    """Reset the benchmark task, plastic bin, and 1-4 tabletop objects.

    The function stores per-episode metadata on the environment under
    ``so101_bench_instruction`` and private tensor buffers consumed by the
    success/failure termination terms. By default active objects are the first
    ``N`` object slots; ``active_object_selection="random"`` samples the active
    slots from all configured objects, and ``active_object_selection="fixed"``
    uses ``fixed_active_object_ids``. Random bin placement samples a full pose
    from ``bin_random_poses`` instead of sampling a continuous position range.
    When ``episode_layout`` is provided, the saved object and bin poses are
    replayed exactly.
    """

    if table_bounds is None:
        table_bounds = {"x": (0.08, 0.45), "y": (-0.20, 0.20)}
    if bin_z is None:
        bin_z = table_top_z

    num_envs = env.num_envs
    num_objects = len(object_asset_names)
    device = env.device

    env._so101_task_family = _reset_list_rows(env, "_so101_task_family", env_ids, TASK_BIN)
    env._so101_instruction_text = _reset_list_rows(env, "_so101_instruction_text", env_ids, "")
    env._so101_active_object_mask = _reset_tensor_rows(
        env, "_so101_active_object_mask", (num_envs, num_objects), torch.bool, env_ids, False
    )
    env._so101_target_object_ids = _reset_tensor_rows(
        env, "_so101_target_object_ids", (num_envs,), torch.long, env_ids, 0
    )
    env._so101_referent_object_ids = _reset_tensor_rows(
        env, "_so101_referent_object_ids", (num_envs, 2), torch.long, env_ids, 0
    )
    env._so101_direction_ids = _reset_tensor_rows(
        env, "_so101_direction_ids", (num_envs,), torch.long, env_ids, 0
    )
    env._so101_initial_object_pos_w = _reset_tensor_rows(
        env, "_so101_initial_object_pos_w", (num_envs, num_objects, 3), torch.float32, env_ids, 0.0
    )
    env._so101_initial_object_yaws = _reset_tensor_rows(
        env, "_so101_initial_object_yaws", (num_envs, num_objects), torch.float32, env_ids, 0.0
    )
    env._so101_initial_bin_pos_w = _reset_tensor_rows(
        env, "_so101_initial_bin_pos_w", (num_envs, 3), torch.float32, env_ids, 0.0
    )
    env._so101_initial_bin_yaws = _reset_tensor_rows(
        env, "_so101_initial_bin_yaws", (num_envs,), torch.float32, env_ids, 0.0
    )
    env._so101_bin_pose_indices = _reset_tensor_rows(
        env, "_so101_bin_pose_indices", (num_envs,), torch.long, env_ids, -1
    )
    env._so101_failure_object_pos_w = _reset_tensor_rows(
        env, "_so101_failure_object_pos_w", (num_envs, num_objects, 3), torch.float32, env_ids, 0.0
    )
    env._so101_failure_bin_pos_w = _reset_tensor_rows(
        env, "_so101_failure_bin_pos_w", (num_envs, 3), torch.float32, env_ids, 0.0
    )
    env._so101_failure_baseline_recorded = _reset_tensor_rows(
        env, "_so101_failure_baseline_recorded", (num_envs,), torch.bool, env_ids, False
    )
    env._so101_robot_started_moving = _reset_tensor_rows(
        env, "_so101_robot_started_moving", (num_envs,), torch.bool, env_ids, False
    )
    env._so101_robot_start_step = _reset_tensor_rows(
        env, "_so101_robot_start_step", (num_envs,), torch.long, env_ids, -1
    )
    env._so101_robot_start_time_s = _reset_tensor_rows(
        env, "_so101_robot_start_time_s", (num_envs,), torch.float32, env_ids, float("nan")
    )
    env._so101_grasp_attempt_counts = _reset_tensor_rows(
        env, "_so101_grasp_attempt_counts", (num_envs, num_objects), torch.long, env_ids, 0
    )
    env._so101_max_object_lift = _reset_tensor_rows(
        env, "_so101_max_object_lift", (num_envs, num_objects), torch.float32, env_ids, 0.0
    )
    env._so101_max_grasp_attempts = MAX_GRASP_ATTEMPTS
    env._so101_grasp_armed = _reset_tensor_rows(
        env, "_so101_grasp_armed", (num_envs,), torch.bool, env_ids, False
    )
    env._so101_grasped_object_ids = _reset_tensor_rows(
        env, "_so101_grasped_object_ids", (num_envs,), torch.long, env_ids, -1
    )
    grasp_arm_jaw_pos = getattr(env, "_so101_grasp_arm_jaw_pos", None)
    if isinstance(grasp_arm_jaw_pos, torch.Tensor) and tuple(grasp_arm_jaw_pos.shape) == (num_envs,):
        grasp_arm_jaw_pos[env_ids] = 0.0
    else:
        env._so101_grasp_arm_jaw_pos = None
    env._so101_bin_success_counter = _reset_tensor_rows(
        env, "_so101_bin_success_counter", (num_envs,), torch.long, env_ids, 0
    )
    env._so101_next_to_success_counter = _reset_tensor_rows(
        env, "_so101_next_to_success_counter", (num_envs,), torch.long, env_ids, 0
    )
    env._so101_between_success_counter = _reset_tensor_rows(
        env, "_so101_between_success_counter", (num_envs,), torch.long, env_ids, 0
    )
    env._so101_move_success_counter = _reset_tensor_rows(
        env, "_so101_move_success_counter", (num_envs,), torch.long, env_ids, 0
    )
    env._so101_move_straightness_failure_counter = _reset_tensor_rows(
        env, "_so101_move_straightness_failure_counter", (num_envs,), torch.long, env_ids, 0
    )
    env._so101_timeout_success_confirmation_active = _reset_tensor_rows(
        env, "_so101_timeout_success_confirmation_active", (num_envs,), torch.bool, env_ids, False
    )
    env._so101_timeout_success_confirmation_failed = _reset_tensor_rows(
        env, "_so101_timeout_success_confirmation_failed", (num_envs,), torch.bool, env_ids, False
    )
    env._so101_grasped_object_contact_steps = _reset_tensor_rows(
        env, "_so101_grasped_object_contact_steps", (num_envs,), torch.long, env_ids, 0
    )
    env._so101_grasped_object_contact_last_episode_steps = _reset_tensor_rows(
        env, "_so101_grasped_object_contact_last_episode_steps", (num_envs,), torch.long, env_ids, -1
    )
    env.so101_bench_episodes = _reset_list_rows(env, "so101_bench_episodes", env_ids, {})
    for cache_name in (
        "_so101_move_boundary_coords",
        "_so101_move_boundary_ids",
        "_so101_grasped_object_made_contact_override",
        "_so101_termination_step_state_cache",
    ):
        if hasattr(env, cache_name):
            delattr(env, cache_name)

    bin_asset: RigidObject = env.scene[bin_name]
    object_assets: list[BenchmarkObject] = [env.scene[name] for name in object_asset_names]
    geometry_signature = (tuple(object_asset_names), tuple(object_labels), bin_name)
    if getattr(env, "_so101_benchmark_geometry_signature", None) != geometry_signature:
        _record_benchmark_geometry(env, object_asset_names, bin_name, object_labels)
        env._so101_default_object_footprint_half_extents = env._so101_object_footprint_half_extents.clone()
        env._so101_default_object_footprint_center_offsets = env._so101_object_footprint_center_offsets.clone()
        env._so101_default_bin_footprint_half_extents = env._so101_bin_footprint_half_extents.clone()
        env._so101_default_bin_footprint_center_offsets = env._so101_bin_footprint_center_offsets.clone()
        env._so101_benchmark_geometry_signature = geometry_signature
    env._so101_object_footprint_half_extents[env_ids] = env._so101_default_object_footprint_half_extents[env_ids]
    env._so101_object_footprint_center_offsets[env_ids] = env._so101_default_object_footprint_center_offsets[env_ids]
    env._so101_bin_footprint_half_extents[env_ids] = env._so101_default_bin_footprint_half_extents[env_ids]
    env._so101_bin_footprint_center_offsets[env_ids] = env._so101_default_bin_footprint_center_offsets[env_ids]
    layout_objects = _layout_object_entries(episode_layout)
    layout_bin_pose = _layout_bin_pose(episode_layout)

    for env_id_tensor in env_ids:
        env_id = int(env_id_tensor.item())
        if episode_spec is None:
            selected_task = _sample_task_family(task_family)
            active_count = _required_object_count(selected_task, object_count_range)
            active_count = min(active_count, num_objects)
            if active_count < 1:
                raise ValueError(f"Expected at least one active object, got {active_count}.")
            active_object_ids = _active_object_ids(
                num_objects,
                active_count,
                active_object_selection,
                fixed_active_object_ids=fixed_active_object_ids,
            )
        else:
            selected_task = str(episode_spec["task_family"])
            active_object_ids = [int(object_id) for object_id in episode_spec["active_object_ids"]]
            active_count = len(active_object_ids)
            if not active_object_ids:
                raise ValueError("episode_spec must activate at least one object.")
            invalid_ids = [object_id for object_id in active_object_ids if object_id < 0 or object_id >= num_objects]
            if invalid_ids:
                raise ValueError(f"episode_spec active object ids are out of range: {invalid_ids}.")
        active_object_order = {object_id: order for order, object_id in enumerate(active_object_ids)}

        bin_pos = (bin_fixed_pose[0], bin_fixed_pose[1], bin_z)
        bin_quat = _bin_quat(bin_fixed_pose[2], bin_root_rotation, device)
        bin_yaw = bin_fixed_pose[2] + bin_root_rotation[2]
        selected_bin_pose_index: int | None = None
        if layout_bin_pose is not None:
            bin_pos, bin_rpy = layout_bin_pose
            bin_quat = _rpy_quat(bin_rpy, device)
            bin_yaw = bin_rpy[2]
            layout_bin_entry = episode_layout.get("bin") if isinstance(episode_layout, dict) else None
            if isinstance(layout_bin_entry, dict) and "pose_index" in layout_bin_entry:
                selected_bin_pose_index = int(layout_bin_entry["pose_index"])
            if isinstance(layout_bin_entry, dict):
                env._so101_bin_footprint_half_extents[env_id] = torch.tensor(
                    _coerce_footprint_pair(
                        layout_bin_entry.get("footprint_half_extents") or layout_bin_entry.get("half_extents"),
                        tuple(env._so101_bin_footprint_half_extents[env_id].tolist()),
                        min_value=0.002,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                env._so101_bin_footprint_center_offsets[env_id] = torch.tensor(
                    _coerce_footprint_pair(
                        layout_bin_entry.get("footprint_center_offset") or layout_bin_entry.get("center_offset"),
                        tuple(env._so101_bin_footprint_center_offsets[env_id].tolist()),
                    ),
                    dtype=torch.float32,
                    device=device,
                )
        elif selected_task == TASK_BIN and randomize_bin_for_bin_task and bin_random_poses:
            selected_bin_pose_index = random.randrange(len(bin_random_poses))
            bin_pos, bin_rpy = bin_random_poses[selected_bin_pose_index]
            bin_quat = _rpy_quat(bin_rpy, device)
            bin_yaw = bin_rpy[2]
        elif selected_task == TASK_MOVE and bin_random_poses:
            selected_bin_pose_index = 0
            bin_pos, bin_rpy = bin_random_poses[selected_bin_pose_index]
            bin_quat = _rpy_quat(bin_rpy, device)
            bin_yaw = bin_rpy[2]

        using_fixed_object_poses = False
        if episode_layout is not None:
            missing_layout_ids = [object_id for object_id in active_object_ids if object_id not in layout_objects]
            if missing_layout_ids:
                raise ValueError(f"Episode layout is missing active object slot(s): {missing_layout_ids}.")
            sampled_positions = []
        elif (
            selected_task == TASK_BIN
            and selected_bin_pose_index is not None
            and valid_spawn_regions is not None
        ):
            if selected_bin_pose_index >= len(valid_spawn_regions):
                raise ValueError(
                    f"Selected bin pose index {selected_bin_pose_index} but only "
                    f"{len(valid_spawn_regions)} valid spawn region(s) are configured."
                )
            sampled_positions = _sample_positions_in_polygon(
                active_count,
                valid_spawn_regions[selected_bin_pose_index],
                min_object_spacing,
            )
        elif object_fixed_poses is not None:
            required_pose_count = max(active_object_ids) + 1
            if len(object_fixed_poses) < required_pose_count:
                raise ValueError(
                    f"Need at least {required_pose_count} fixed object poses, got {len(object_fixed_poses)}."
                )
            sampled_positions = _fixed_positions(active_object_ids, object_fixed_poses)
            using_fixed_object_poses = True
        else:
            sampled_positions = _sample_positions(active_count, table_bounds, min_object_spacing)

        if episode_spec is not None:
            active_labels = [object_labels[object_id] for object_id in active_object_ids]
            target_object_id = int(episode_spec["target_object_id"])
            referents = [int(object_id) for object_id in episode_spec["referent_object_ids"]]
            if len(referents) != 2:
                raise ValueError(f"episode_spec must carry two referent ids, got {referents}.")
            first_referent_id, second_referent_id = referents
            direction = str(episode_spec.get("direction") or DIRECTIONS[0])
        else:
            if shuffle_object_labels:
                label_perm = list(object_labels)
                random.shuffle(label_perm)
                active_labels = label_perm[:active_count]
            else:
                active_labels = [object_labels[object_id] for object_id in active_object_ids]
            direction = random.choice(DIRECTIONS)
            target_object_id = active_object_ids[0]
            first_referent_id = active_object_ids[1] if active_count > 1 else target_object_id
            if active_count > 2:
                second_referent_id = active_object_ids[2]
            elif active_count > 1:
                second_referent_id = active_object_ids[1]
            else:
                second_referent_id = min(target_object_id + 1, num_objects - 1)

        env._so101_task_family[env_id] = selected_task
        env._so101_active_object_mask[env_id, active_object_ids] = True
        env._so101_target_object_ids[env_id] = target_object_id
        env._so101_referent_object_ids[env_id, 0] = first_referent_id
        env._so101_referent_object_ids[env_id, 1] = second_referent_id
        env._so101_direction_ids[env_id] = DIRECTIONS.index(direction)

        env._so101_initial_bin_pos_w[env_id] = _write_pose(
            env,
            bin_asset,
            env_id,
            bin_pos,
            bin_quat,
        )
        env._so101_initial_bin_yaws[env_id] = bin_yaw
        env._so101_bin_pose_indices[env_id] = -1 if selected_bin_pose_index is None else selected_bin_pose_index
        env._so101_failure_bin_pos_w[env_id] = env._so101_initial_bin_pos_w[env_id]

        for object_id, asset in enumerate(object_assets):
            is_active = object_id in active_object_order
            default_z = _default_root_z(asset, env_id)
            if is_active:
                layout_entry = layout_objects.get(object_id)
                if layout_entry is not None:
                    x, y, z = _layout_object_position(layout_entry)
                    yaw = _layout_object_yaw(layout_entry)
                    env._so101_object_footprint_half_extents[env_id, object_id] = torch.tensor(
                        _coerce_footprint_pair(
                            layout_entry.get("footprint_half_extents") or layout_entry.get("half_extents"),
                            tuple(env._so101_object_footprint_half_extents[env_id, object_id].tolist()),
                            min_value=0.002,
                        ),
                        dtype=torch.float32,
                        device=device,
                    )
                    env._so101_object_footprint_center_offsets[env_id, object_id] = torch.tensor(
                        _coerce_footprint_pair(
                            layout_entry.get("footprint_center_offset") or layout_entry.get("center_offset"),
                            tuple(env._so101_object_footprint_center_offsets[env_id, object_id].tolist()),
                        ),
                        dtype=torch.float32,
                        device=device,
                    )
                else:
                    pose_id = active_object_order[object_id]
                    x, y = sampled_positions[pose_id]
                    z = default_z if default_z > table_top_z else table_top_z + 0.025
                    yaw = (
                        object_fixed_poses[object_id][2]
                        if using_fixed_object_poses
                        else random.uniform(-math.pi, math.pi)
                    )
            else:
                x, y, z = _inactive_position(inactive_object_base_pos, inactive_object_spacing, object_id)
                yaw = 0.0

            env._so101_initial_object_pos_w[env_id, object_id] = _write_pose(
                env,
                asset,
                env_id,
                (x, y, z),
                _yaw_quat(yaw, device),
                asset_name=object_asset_names[object_id],
            )
            env._so101_initial_object_yaws[env_id, object_id] = yaw
            env._so101_failure_object_pos_w[env_id, object_id] = env._so101_initial_object_pos_w[
                env_id, object_id
            ]

        if episode_spec is not None:
            instruction = str(episode_spec["instruction"])
        elif selected_task == TASK_BIN and force_bin_all_objects_instruction:
            instruction = "Place each object in the plastic bin."
        else:
            instruction = task_instruction(selected_task, active_labels, direction)
        env._so101_instruction_text[env_id] = instruction
        env.so101_bench_episodes[env_id] = {
            "env_id": env_id,
            "task_family": selected_task,
            "instruction": instruction,
            "active_object_count": active_count,
            "active_object_ids": active_object_ids,
            "active_asset_names": [object_asset_names[object_id] for object_id in active_object_ids],
            "active_labels": active_labels,
            "bin_pose_index": selected_bin_pose_index,
            "direction": direction if selected_task == TASK_MOVE else None,
            "metadata": dict(episode_spec.get("metadata", {})) if episode_spec is not None else {},
        }

    env.so101_bench_instruction = env._so101_instruction_text[0]
