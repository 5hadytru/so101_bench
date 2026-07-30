"""Isaac Lab ManagerBased environment configs for SO-101 Bench."""

from __future__ import annotations

import math
import os

from isaacsim.core.utils.rotations import euler_angles_to_quat

import numpy as np
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.utils import configclass

from so101_bench import assets, mdp
from so101_bench.assets.so101 import SO101_CFG
from so101_bench.benchmark import (
    BenchmarkEpisodeSpec,
    FOUR_OBJECT_BIN_EPISODE_LENGTH_S,
    MOVE_STRAIGHTNESS_TOLERANCE_M,
    SPATIAL_SUCCESS_DISTANCE_M,
    TASK_BETWEEN,
    TASK_BIN,
    TASK_MIXED,
    TASK_MOVE,
    TASK_NAMED_BIN,
    TASK_NEXT_TO,
    object_metadata,
    object_rigid_body_child_names,
    object_usd_stem,
)

ASSETS_PATH = os.path.dirname(os.path.abspath(assets.__file__))

UI_CPU_PHYSICS_ENV_VAR = "SO101_BENCH_UI_CPU_PHYSICS"

TABLE_TOP_Z = 0.0
TABLE_OBJECT_Z = TABLE_TOP_Z + 0.001
BEDROOM_TABLETOP_USD = f"{ASSETS_PATH}/usd/room_scan.usdc"
BEDROOM_TABLETOP_SCALE = (1.0, 1.0, 1.0)

OBJECT_ASSET_NAMES = ["object_1", "object_2", "object_3", "object_4"]
OBJECT_LABELS = ["blue bowl", "silver glasses", "yellow screwdriver", "black tape"]
TABLE_BOUNDS = {"x": (-0.14, 0.25), "y": (-0.1, 0.155)}
INACTIVE_OBJECT_BASE_POS = (20.06628, 20.0, -10.0)
INACTIVE_OBJECT_SPACING = 0.25
ROBOT_BASE_TRANSLATION = (0.05209, 0.18061, -0.03102)
ROBOT_BASE_YAW_DEG = 6.195
OBJECT_FIXED_POSES = (
    (0.05128, 0.1, math.radians(90.0)),
    (0.18828, -0.09, math.radians(90.0)),
    (-0.03372, 0.08, math.radians(90.0)),
    (0.24228, 0.066, math.radians(90.0)),
)

BIN_FIXED_TRANSLATION = (-0.12917, -0.16276, 0.02)
BIN_FIXED_YAW_DEG = -66.023
BIN_ROOT_ROTATION_RPY_DEG = (-0.007, -0.009, 0.0)
BIN_FIXED_POSE = (
    BIN_FIXED_TRANSLATION[0],
    BIN_FIXED_TRANSLATION[1],
    math.radians(BIN_FIXED_YAW_DEG),
)
BIN_RANDOM_POSES_RPY_DEG = (
    ((-0.12917, -0.16276, 0.02), (0.0, 0.0, -66.023)),
    ((-0.15984, -0.06498, 0.02), (0.0, 0.0, -90.0)),
    ((-0.15984, 0.02407, 0.02), (0.0, 0.0, -90.0)),
    ((0.37129, 0.03595, 0.02), (0.0, 0.0, -90.0)),
    ((0.37129, -0.07184, 0.02), (0.0, 0.0, -90.0)),
    ((0.32627, -0.17514, 0.02), (0.0, 0.0, -120.541)),
)
BIN_RANDOM_POSES = tuple(
    (translation, tuple(math.radians(angle) for angle in orientation_rpy_deg))
    for translation, orientation_rpy_deg in BIN_RANDOM_POSES_RPY_DEG
)

VALID_OBJECT_SPAWN_REGIONS = [ # for each bin position, define a valid region for object spawning (as a counter-clockwise set of points for a polygon)
    [(-.11, 0.15, 0.0), (-0.0512, 0.04475, 0.0), (0.01521, -0.099, 0.0), (0.25, -0.099, 0.0), (0.25, 0.15, 0.0), (0.185, 0.15, 0.0), (0.147, 0.07, 0.0), (0.145, 0.05146, 0.0), (0.141, -0.004, 0.0), (0.01, -.0005, 0.0), (-0.047, 0.15, 0.0)],
    [(-0.02, -0.008, 0.0), (-0.02, -0.099, 0.0), (0.25, -0.099, 0.0), (0.25, 0.15, 0.0), (0.19, 0.15, 0.0), (0.165, 0.09, 0.0), (0.14491, 0.053, 0.0), (0.137, -0.006, 0.0)],
    [(-0.02, -0.008, 0.0), (-0.02, -0.099, 0.0), (0.25, -0.099, 0.0), (0.25, 0.15, 0.0), (0.19, 0.15, 0.0), (0.165, 0.09, 0.0), (0.14491, 0.053, 0.0), (0.137, -0.006, 0.0)],
    [(-.11, 0.15, 0.0), (-.1022, -0.084, 0.0), (0.211568, -0.099, -0.0), (0.211568, 0.008515, 0.0), (0.211568, 0.15, 0.0), (0.185, 0.15, 0.0), (0.155, 0.08, 0.0), (0.155, 0.08, 0.0), (0.137, -0.01, 0.0), (0.015, -0.01, 0.0), (-0.0085, 0.05435, 0.0), (-0.043, 0.128006, 0.0), (-0.043, 0.15, 0.0)],
    [(-.11, 0.15, 0.0), (-.1022, -0.084, 0.0), (0.211568, -0.099, -0.0), (0.211568, 0.008515, 0.0), (0.211568, 0.15, 0.0), (0.185, 0.15, 0.0), (0.155, 0.08, 0.0), (0.155, 0.08, 0.0), (0.137, -0.01, 0.0), (0.015, -0.01, 0.0), (-0.0085, 0.05435, 0.0), (-0.043, 0.128006, 0.0), (-0.043, 0.15, 0.0)],
    [(-.11, 0.15, 0.0), (-.1022, -0.084, 0.0), (0.186416, -0.099, -0.0), (0.25, 0.008515, 0.0), (0.25, 0.15, 0.0), (0.185, 0.15, 0.0), (0.155, 0.08, 0.0), (0.155, 0.08, 0.0), (0.137, -0.01, 0.0), (0.015, -0.01, 0.0), (-0.0085, 0.05435, 0.0), (-0.043, 0.128006, 0.0), (-0.043, 0.15, 0.0)],
]

SO101_BOUNDING_BOX = [
    (0.012, 0.241, 0.0),
    (0.0113, 0.1267, 0.0),
    (0.0437, 0.1262, 0.0),
    (0.0431, 0.0182, 0.0),
    (0.0939, 0.0179, 0.0),
    (0.0939, 0.1262, 0.0),
    (0.1256, 0.126, 0.0),
    (0.1263, 0.2403, 0.0),
]

MIN_RESET_TIME_S = 0.5
MIN_FAILURE_TIME_S = 1.0
SUCCESS_CONFIRM_TIME_S = 3.0
FAILURE_CONFIRM_TIME_S = 5.0
CONTACT_GRACE_TIME_S = 5.0
PHYSICS_DT = 1.0 / 240.0
CONTROL_DT = 1.0 / 30.0
CONTROL_DECIMATION = int(round(CONTROL_DT / PHYSICS_DT))
CONTACT_OFFSET = 0.002
REST_OFFSET = 0.0
CONTACT_SOLVER_POSITION_ITERATIONS = 64
CONTACT_SOLVER_VELOCITY_ITERATIONS = 4
MAX_DEPENETRATION_VELOCITY = 0.25
MAX_BIN_LINEAR_VELOCITY = 1.0
MAX_BIN_ANGULAR_VELOCITY = 360.0
MAX_OBJECT_LINEAR_VELOCITY = 2.0
MAX_OBJECT_ANGULAR_VELOCITY = 720.0
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FOCAL_LENGTH = 13.5
DEFAULT_CAMERA_HORIZONTAL_APERTURE = 20.955
IPHONE_17_PRO_MAIN_CAMERA_HORIZONTAL_FOV_DEG = 74.0
IPHONE_17_PRO_MAIN_CAMERA_HORIZONTAL_APERTURE = 34.61329224445431
IPHONE_17_PRO_MAIN_CAMERA_FOCAL_LENGTH = IPHONE_17_PRO_MAIN_CAMERA_HORIZONTAL_APERTURE / (
    2.0 * math.tan(math.radians(IPHONE_17_PRO_MAIN_CAMERA_HORIZONTAL_FOV_DEG) / 2.0)
)

INNOMAKER_WRIST_CAMERA_WIDTH = DEFAULT_CAMERA_WIDTH
INNOMAKER_WRIST_CAMERA_HEIGHT = DEFAULT_CAMERA_HEIGHT
INNOMAKER_WRIST_CAMERA_UPDATE_PERIOD = CONTROL_DT
INNOMAKER_WRIST_CAMERA_HORIZONTAL_FOV_DEG = 52.5
INNOMAKER_WRIST_CAMERA_HORIZONTAL_APERTURE = 20.955
INNOMAKER_WRIST_CAMERA_FOCAL_LENGTH = INNOMAKER_WRIST_CAMERA_HORIZONTAL_APERTURE / (
    2.0 * math.tan(math.radians(INNOMAKER_WRIST_CAMERA_HORIZONTAL_FOV_DEG) / 2.0)
)
INNOMAKER_WRIST_CAMERA_CLIPPING_RANGE = (0.02, 3.0)
INNOMAKER_WRIST_CAMERA_FOCUS_DISTANCE = 0.5
INNOMAKER_WRIST_CAMERA_F_STOP = 0.0
INNOMAKER_WRIST_CAMERA_POS = (0.00283, 0.05937, -0.06408)
INNOMAKER_WRIST_CAMERA_RPY_DEG = (-45.0, 0.0, 0.0)
OVERHEAD_CAMERA_POS = (0.0, -0.34, 0.45722)
OVERHEAD_CAMERA_RPY_DEG = (51.361, -0.169, 0.603)
BEDROOM_LIGHT_TEMPERATURE_K = 6000.0
BEDROOM_LIGHT_INTENSITY = 12000.0
BEDROOM_LIGHT_RADIUS = 0.4


def _validate_fixed_object_poses() -> None:
    x_min, x_max = TABLE_BOUNDS["x"]
    y_min, y_max = TABLE_BOUNDS["y"]
    for index, (x, y, _yaw) in enumerate(OBJECT_FIXED_POSES, start=1):
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            raise ValueError(
                f"OBJECT_FIXED_POSES[{index - 1}] for object_{index} is outside TABLE_BOUNDS: "
                f"(x={x}, y={y}), bounds x={TABLE_BOUNDS['x']} y={TABLE_BOUNDS['y']}."
            )


_validate_fixed_object_poses()


def inactive_object_pos(object_id: int) -> tuple[float, float, float]:
    return (
        INACTIVE_OBJECT_BASE_POS[0] + INACTIVE_OBJECT_SPACING * object_id,
        INACTIVE_OBJECT_BASE_POS[1],
        INACTIVE_OBJECT_BASE_POS[2],
    )


def _robot_cfg() -> ArticulationCfg:
    cfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.init_state.pos = ROBOT_BASE_TRANSLATION
    cfg.init_state.rot = euler_angles_to_quat(np.array([0.0, 0.0, ROBOT_BASE_YAW_DEG]), degrees=True)
    return cfg


def _contact_rigid_props(max_linear_velocity: float, max_angular_velocity: float) -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=CONTACT_SOLVER_POSITION_ITERATIONS,
        solver_velocity_iteration_count=CONTACT_SOLVER_VELOCITY_ITERATIONS,
        max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
        max_linear_velocity=max_linear_velocity,
        max_angular_velocity=max_angular_velocity,
    )


def _contact_collision_props() -> sim_utils.CollisionPropertiesCfg:
    return sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=CONTACT_OFFSET,
        rest_offset=REST_OFFSET,
    )


def _enable_body_ccd(root_prim: Usd.Prim) -> None:
    """Enable sweep and speculative CCD on rigid bodies under the spawned asset."""

    for prim in Usd.PrimRange(root_prim):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_rigid_body_api.CreateEnableCCDAttr().Set(True)
        physx_rigid_body_api.CreateEnableSpeculativeCCDAttr().Set(True)


def _spawn_ccd_usd(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a rigid USD and enable CCD on all rigid bodies under it."""

    prim = sim_utils.spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )

    stage = sim_utils.get_current_stage()
    for spawned_prim_path in sim_utils.find_matching_prim_paths(prim_path):
        spawned_prim = stage.GetPrimAtPath(spawned_prim_path)
        if spawned_prim.IsValid():
            _enable_body_ccd(spawned_prim)
    return prim


def _physics_mesh_prims_below(body_prim: Usd.Prim) -> list[Usd.Prim]:
    physics_mesh_prims = []
    mesh_prims = []
    body_path = body_prim.GetPath().pathString
    for prim in Usd.PrimRange(body_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_prims.append(prim)
        relative_path = prim.GetPath().pathString[len(body_path) :].lower()
        if any(part == "physics" or part.endswith("_physics") for part in relative_path.split("/")):
            physics_mesh_prims.append(prim)
    return physics_mesh_prims or mesh_prims


def _apply_split_rigid_body_physics(
    prim_path: str,
    object_name: str,
    rigid_props: sim_utils.RigidBodyPropertiesCfg | None,
    collision_props: sim_utils.CollisionPropertiesCfg | None,
    mass_props: sim_utils.MassPropertiesCfg | None,
) -> None:
    """Author missing physics schemas on split-body object children."""

    stage = sim_utils.get_current_stage()
    mesh_collision_props = sim_utils.SDFMeshPropertiesCfg()
    for child_name in object_rigid_body_child_names(object_name):
        body_path = f"{prim_path}/{child_name}"
        body_prim = stage.GetPrimAtPath(body_path)
        if not body_prim.IsValid():
            continue
        if rigid_props is not None:
            sim_utils.define_rigid_body_properties(body_path, rigid_props, stage=stage)
        if mass_props is not None:
            sim_utils.define_mass_properties(body_path, mass_props, stage=stage)
        if collision_props is None:
            continue
        for mesh_prim in _physics_mesh_prims_below(body_prim):
            mesh_path = mesh_prim.GetPath().pathString
            sim_utils.define_collision_properties(mesh_path, collision_props, stage=stage)
            sim_utils.define_mesh_collision_properties(mesh_path, mesh_collision_props, stage=stage)


def _spawn_split_rigid_body_usd(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a USD whose independent rigid bodies live under left/right child prims."""

    object_name = os.path.splitext(os.path.basename(cfg.usd_path))[0].replace("_", " ")
    rigid_props = cfg.rigid_props
    collision_props = cfg.collision_props
    mass_props = cfg.mass_props
    activate_contact_sensors = cfg.activate_contact_sensors

    cfg.rigid_props = None
    cfg.collision_props = None
    cfg.mass_props = None
    cfg.activate_contact_sensors = False
    try:
        prim = sim_utils.spawn_from_usd(
            prim_path,
            cfg,
            translation=translation,
            orientation=orientation,
            **kwargs,
        )
    finally:
        cfg.rigid_props = rigid_props
        cfg.collision_props = collision_props
        cfg.mass_props = mass_props
        cfg.activate_contact_sensors = activate_contact_sensors

    for spawned_prim_path in sim_utils.find_matching_prim_paths(prim_path):
        _apply_split_rigid_body_physics(
            spawned_prim_path,
            object_name,
            rigid_props,
            collision_props,
            mass_props,
        )
        spawned_prim = sim_utils.get_current_stage().GetPrimAtPath(spawned_prim_path)
        if spawned_prim.IsValid():
            _enable_body_ccd(spawned_prim)
        if activate_contact_sensors:
            sim_utils.activate_contact_sensors(spawned_prim_path)
    return prim


def _patch_bin_visual_material(root_prim: Usd.Prim) -> None:
    """Keep the Blender material look, but remove camera-hostile transparency."""

    for prim in Usd.PrimRange(root_prim):
        opacity_attr = prim.GetAttribute("inputs:opacity")
        if opacity_attr.IsValid():
            opacity_attr.ClearConnections()
            opacity_attr.Set(1.0)
        for attr_name in ("inputs:transmission", "inputs:transmission_weight"):
            attr = prim.GetAttribute(attr_name)
            if attr.IsValid():
                attr.ClearConnections()
                attr.Set(0.0)


def _spawn_sensor_safe_bin_usd(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the bin USD and make its authored material opaque for camera sensors."""

    prim = sim_utils.spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )

    stage = sim_utils.get_current_stage()
    for spawned_prim_path in sim_utils.find_matching_prim_paths(prim_path):
        spawned_prim = stage.GetPrimAtPath(spawned_prim_path)
        if spawned_prim.IsValid():
            _patch_bin_visual_material(spawned_prim)
            _enable_body_ccd(spawned_prim)
    return prim


def _camera_cfg(
    width: int = DEFAULT_CAMERA_WIDTH,
    height: int = DEFAULT_CAMERA_HEIGHT,
    update_period: float = 0.0,
    focal_length: float = DEFAULT_CAMERA_FOCAL_LENGTH,
    horizontal_aperture: float = DEFAULT_CAMERA_HORIZONTAL_APERTURE,
    vertical_aperture: float | None = None,
    clipping_range: tuple[float, float] = (0.01, 5.0),
    focus_distance: float = 0.35,
    f_stop: float = 0.0,
) -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path="",
        update_period=update_period,
        height=height,
        width=width,
        data_types=["rgb"],
        # colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            projection_type="pinhole",
            focal_length=focal_length,
            focus_distance=focus_distance,
            f_stop=f_stop,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            clipping_range=clipping_range,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=euler_angles_to_quat(np.array([0.0, 0.0, 0.0]), degrees=True),
            convention="opengl",
        ),
    )


def _object_spawn(
    color: tuple[float, float, float],
    size: tuple[float, float, float],
    mass: float = 0.035,
) -> sim_utils.CuboidCfg:
    """
    object_4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_4",
        spawn=_object_spawn((0.90, 0.16, 0.10), (0.095, 0.014, 0.014), mass=0.020),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(OBJECT_FIXED_POSES[3][0], OBJECT_FIXED_POSES[3][1], TABLE_OBJECT_Z)
        ),
    )
    """

    return sim_utils.CuboidCfg(
        size=size,
        activate_contact_sensors=True,
        rigid_props=_contact_rigid_props(MAX_OBJECT_LINEAR_VELOCITY, MAX_OBJECT_ANGULAR_VELOCITY),
        collision_props=_contact_collision_props(),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


def _object_initial_pose(object_id: int) -> tuple[tuple[float, float, float], float]:
    if object_id < len(OBJECT_FIXED_POSES):
        x, y, yaw = OBJECT_FIXED_POSES[object_id]
        return (x, y, TABLE_OBJECT_Z), yaw
    return inactive_object_pos(object_id), 0.0


def _benchmark_object_cfg(object_id: int, object_name: str) -> RigidObjectCfg | AssetBaseCfg | DeformableObjectCfg:
    """Build one scene slot from the object registry and its USD filename convention."""

    metadata = object_metadata(object_name)
    init_pos, init_yaw = _object_initial_pose(object_id)
    usd_path = f"{ASSETS_PATH}/usd/objects/{object_usd_stem(object_name)}.usdc"
    init_rot = euler_angles_to_quat(np.array([0.0, 0.0, init_yaw]), degrees=False)

    if metadata.get("deformable", False):
        if metadata["multiple_rigid_bodies"]:
            raise ValueError(f"Object {object_name!r} cannot be both deformable and multi-rigid-body.")
        return DeformableObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Object_{object_id + 1}",
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
            init_state=DeformableObjectCfg.InitialStateCfg(
                pos=init_pos,
                rot=init_rot,
            ),
            debug_vis=False,
        )

    cfg_type = AssetBaseCfg if metadata["multiple_rigid_bodies"] else RigidObjectCfg
    init_state_type = cfg_type.InitialStateCfg
    spawn_kwargs = {
        "usd_path": usd_path,
        "activate_contact_sensors": True,
        "rigid_props": _contact_rigid_props(MAX_OBJECT_LINEAR_VELOCITY, MAX_OBJECT_ANGULAR_VELOCITY),
        "collision_props": _contact_collision_props(),
        "func": _spawn_ccd_usd,
    }
    if metadata["multiple_rigid_bodies"]:
        spawn_kwargs["func"] = _spawn_split_rigid_body_usd
    return cfg_type(
        prim_path=f"{{ENV_REGEX_NS}}/Object_{object_id + 1}",
        spawn=sim_utils.UsdFileCfg(**spawn_kwargs),
        init_state=init_state_type(
            pos=init_pos,
            rot=init_rot,
        ),
    )


def _object_body_prim_paths(object_id: int, object_name: str) -> list[str]:
    root_path = f"{{ENV_REGEX_NS}}/Object_{object_id + 1}"
    child_names = object_rigid_body_child_names(object_name)
    return [f"{root_path}/{child_name}" for child_name in child_names] if child_names else [root_path]


def _object_contact_sensor_cfgs(object_names: list[str] | tuple[str, ...]) -> dict[str, ContactSensorCfg]:
    """Create per-body sensors filtered to contacts with other tabletop objects only."""

    body_paths_by_object = [
        []
        if object_metadata(object_name).get("deformable", False)
        else _object_body_prim_paths(object_id, object_name)
        for object_id, object_name in enumerate(object_names)
    ]
    sensor_cfgs = {}
    for object_id, body_paths in enumerate(body_paths_by_object):
        filter_paths = [
            body_path
            for other_object_id, other_body_paths in enumerate(body_paths_by_object)
            if other_object_id != object_id
            for body_path in other_body_paths
        ]
        if not filter_paths:
            continue
        for body_path in body_paths:
            body_name = body_path.rsplit("/", maxsplit=1)[-1]
            suffix = "" if body_name == f"Object_{object_id + 1}" else f"_{body_name}"
            sensor_cfgs[f"object_{object_id + 1}{suffix}_contacts"] = ContactSensorCfg(
                prim_path=body_path,
                update_period=0.0,
                filter_prim_paths_expr=filter_paths,
            )
    return sensor_cfgs


def _configure_scene_object_contact_sensors(
    scene_cfg: So101BenchSceneCfg,
    object_names: list[str] | tuple[str, ...],
) -> None:
    for attr_name in list(scene_cfg.__dict__):
        if attr_name.startswith("object_") and attr_name.endswith("_contacts"):
            setattr(scene_cfg, attr_name, None)
    for attr_name, sensor_cfg in _object_contact_sensor_cfgs(object_names).items():
        setattr(scene_cfg, attr_name, sensor_cfg)


_DEFAULT_OBJECT_CONTACT_SENSOR_CFGS = _object_contact_sensor_cfgs(OBJECT_LABELS)


@configclass
class So101BenchSceneCfg(InteractiveSceneCfg):
    """Bedroom tabletop scene with SO-101, bin, cameras, and benchmark object slots."""

    env_spacing = 5.0
    num_envs = 1
    replicate_physics = False # False for deformables

    robot: ArticulationCfg = _robot_cfg()

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper",
                name="gripper",
            ),
        ],
    )

    bedroom_tabletop = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BedroomTabletop",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BEDROOM_TABLETOP_USD,
            scale=BEDROOM_TABLETOP_SCALE,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    plastic_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PlasticBin",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSETS_PATH}/usd/plastic_bin.usdc",
            rigid_props=_contact_rigid_props(MAX_BIN_LINEAR_VELOCITY, MAX_BIN_ANGULAR_VELOCITY),
            collision_props=_contact_collision_props(),
            func=_spawn_sensor_safe_bin_usd,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BIN_FIXED_TRANSLATION,
            rot=euler_angles_to_quat(
                np.array(
                    [
                        BIN_ROOT_ROTATION_RPY_DEG[0],
                        BIN_ROOT_ROTATION_RPY_DEG[1],
                        BIN_ROOT_ROTATION_RPY_DEG[2] + BIN_FIXED_YAW_DEG,
                    ]
                ),
                degrees=True,
            ),
        ),
    )

    object_1 = _benchmark_object_cfg(0, OBJECT_LABELS[0])
    object_2 = _benchmark_object_cfg(1, OBJECT_LABELS[1])
    object_3 = _benchmark_object_cfg(2, OBJECT_LABELS[2])
    object_4 = _benchmark_object_cfg(3, OBJECT_LABELS[3])
    object_1_contacts = _DEFAULT_OBJECT_CONTACT_SENSOR_CFGS["object_1_contacts"]
    object_2_contacts = _DEFAULT_OBJECT_CONTACT_SENSOR_CFGS["object_2_contacts"]
    object_3_contacts = _DEFAULT_OBJECT_CONTACT_SENSOR_CFGS["object_3_contacts"]
    object_4_contacts = _DEFAULT_OBJECT_CONTACT_SENSOR_CFGS["object_4_contacts"]

    camera_wrist = _camera_cfg(
        width=INNOMAKER_WRIST_CAMERA_WIDTH,
        height=INNOMAKER_WRIST_CAMERA_HEIGHT,
        update_period=INNOMAKER_WRIST_CAMERA_UPDATE_PERIOD,
        focal_length=INNOMAKER_WRIST_CAMERA_FOCAL_LENGTH,
        horizontal_aperture=INNOMAKER_WRIST_CAMERA_HORIZONTAL_APERTURE,
        vertical_aperture=None,
        clipping_range=INNOMAKER_WRIST_CAMERA_CLIPPING_RANGE,
        focus_distance=INNOMAKER_WRIST_CAMERA_FOCUS_DISTANCE,
        f_stop=INNOMAKER_WRIST_CAMERA_F_STOP,
    )
    camera_wrist.prim_path = "{ENV_REGEX_NS}/Robot/gripper/gripper_cam"
    camera_wrist.offset.pos = INNOMAKER_WRIST_CAMERA_POS
    camera_wrist.offset.rot = euler_angles_to_quat(np.array(INNOMAKER_WRIST_CAMERA_RPY_DEG), degrees=True)

    camera_overhead = _camera_cfg(
        update_period=CONTROL_DT,
        focal_length=IPHONE_17_PRO_MAIN_CAMERA_FOCAL_LENGTH,
        horizontal_aperture=IPHONE_17_PRO_MAIN_CAMERA_HORIZONTAL_APERTURE,
        vertical_aperture=None,
    )
    camera_overhead.prim_path = "{ENV_REGEX_NS}/CameraOverhead"
    camera_overhead.offset.pos = OVERHEAD_CAMERA_POS
    camera_overhead.offset.rot = euler_angles_to_quat(np.array(OVERHEAD_CAMERA_RPY_DEG), degrees=True)
    camera_overhead.offset.convention = "opengl"

    bedroom_domelight = AssetBaseCfg(
        prim_path="/World/BedroomDomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=800,
            color=(1.0, 0.96, 0.88),
            enable_color_temperature=True,
            color_temperature=6500,
        ),
    )


@configclass
class ActionsCfg:
    """Absolute joint-position actions matching the SO-101 workshop and GR00T runner."""

    joint_positions = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """Robot proprioception plus wrist and overhead camera observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_obs = ObsTerm(func=mdp.joint_pos)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        ee_frame_state = ObsTerm(
            func=mdp.ee_frame_state,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class VisualCfg(ObsGroup):
        rgb_wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera_wrist"), "data_type": "rgb", "normalize": False},
        )
        rgb_overhead = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera_overhead"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    visual: VisualCfg = VisualCfg()


def _reset_scene_event(
    task_family: str,
    object_count_range: tuple[int, int],
    active_object_selection: str = "prefix",
    fixed_active_object_ids: tuple[int, ...] | None = None,
    shuffle_object_labels: bool = False,
    force_bin_all_objects_instruction: bool = False,
) -> EventTerm:
    return EventTerm(
        func=mdp.reset_benchmark_scene,
        mode="reset",
        params={
            "object_asset_names": OBJECT_ASSET_NAMES,
            "bin_name": "plastic_bin",
            "object_labels": OBJECT_LABELS,
            "task_family": task_family,
            "object_count_range": object_count_range,
            "table_bounds": TABLE_BOUNDS,
            "table_top_z": TABLE_TOP_Z,
            "min_object_spacing": 0.105,
            "bin_fixed_pose": BIN_FIXED_POSE,
            "bin_root_rotation": tuple(math.radians(angle) for angle in BIN_ROOT_ROTATION_RPY_DEG),
            "bin_z": BIN_FIXED_TRANSLATION[2],
            "object_fixed_poses": OBJECT_FIXED_POSES,
            "randomize_bin_for_bin_task": True,
            "bin_random_poses": BIN_RANDOM_POSES,
            "valid_spawn_regions": VALID_OBJECT_SPAWN_REGIONS,
            "active_object_selection": active_object_selection,
            "fixed_active_object_ids": fixed_active_object_ids,
            "shuffle_object_labels": shuffle_object_labels,
            "force_bin_all_objects_instruction": force_bin_all_objects_instruction,
            "episode_layout": None,
            "inactive_object_base_pos": INACTIVE_OBJECT_BASE_POS,
            "inactive_object_spacing": INACTIVE_OBJECT_SPACING,
        },
    )


def _single_object_reset_scene_event(fixed_active_object_id: int | None = None) -> EventTerm:
    if fixed_active_object_id is None:
        return _reset_scene_event(
            TASK_BIN,
            (1, 1),
            active_object_selection="random",
            shuffle_object_labels=False,
            force_bin_all_objects_instruction=True,
        )
    return _reset_scene_event(
        TASK_BIN,
        (1, 1),
        active_object_selection="fixed",
        fixed_active_object_ids=(fixed_active_object_id,),
        shuffle_object_labels=False,
        force_bin_all_objects_instruction=True,
    )


def _park_inactive_scene_objects(scene_cfg: So101BenchSceneCfg, active_object_id: int) -> None:
    for object_id, asset_name in enumerate(OBJECT_ASSET_NAMES):
        if object_id == active_object_id:
            continue
        getattr(scene_cfg, asset_name).init_state.pos = inactive_object_pos(object_id)


def object_asset_names_for_count(object_count: int) -> list[str]:
    if object_count < 1:
        raise ValueError(f"Expected at least one object asset, got {object_count}.")
    return [f"object_{object_id + 1}" for object_id in range(object_count)]


def configure_scene_objects(scene_cfg: So101BenchSceneCfg, object_names: list[str] | tuple[str, ...]) -> list[str]:
    """Replace the four scene slots with JSONL-selected object USDs."""

    if not 1 <= len(object_names) <= len(OBJECT_ASSET_NAMES):
        raise ValueError(f"Expected 1-{len(OBJECT_ASSET_NAMES)} episode objects, got {len(object_names)}.")
    slot_labels = [*object_names, *OBJECT_LABELS[len(object_names) :]]
    for object_id, asset_name in enumerate(OBJECT_ASSET_NAMES):
        setattr(scene_cfg, asset_name, _benchmark_object_cfg(object_id, slot_labels[object_id]))
    _configure_scene_object_contact_sensors(scene_cfg, slot_labels)
    return slot_labels


def configure_scene_object_pool(scene_cfg: So101BenchSceneCfg, object_names: list[str] | tuple[str, ...]) -> list[str]:
    """Pre-spawn one scene asset for each unique benchmark object name."""

    if not object_names:
        raise ValueError("Expected at least one object to pre-spawn.")
    if len(set(object_names)) != len(object_names):
        raise ValueError(f"Object pool names must be unique, got {list(object_names)}.")

    asset_names = object_asset_names_for_count(len(object_names))
    default_or_pool_count = max(len(OBJECT_ASSET_NAMES), len(asset_names))
    for object_id, asset_name in enumerate(object_asset_names_for_count(default_or_pool_count)):
        if object_id < len(object_names):
            setattr(scene_cfg, asset_name, _benchmark_object_cfg(object_id, object_names[object_id]))
        else:
            setattr(scene_cfg, asset_name, None)
    _configure_scene_object_contact_sensors(scene_cfg, object_names)
    return asset_names


def configure_env_cfg_for_object_pool(
    env_cfg: So101BenchEnvCfg,
    object_names: list[str] | tuple[str, ...],
) -> list[str]:
    """Configure the scene and task terms for a reusable pool of pre-spawned objects."""

    object_asset_names = configure_scene_object_pool(env_cfg.scene, object_names)
    object_labels = list(object_names)

    reset_params = env_cfg.events.reset_benchmark_scene.params
    reset_params["object_asset_names"] = object_asset_names
    reset_params["object_labels"] = object_labels
    reset_params["object_count_range"] = (1, min(4, len(object_asset_names)))
    reset_params["active_object_selection"] = "fixed"
    reset_params["fixed_active_object_ids"] = (0,)
    reset_params["shuffle_object_labels"] = False
    reset_params["force_bin_all_objects_instruction"] = False
    reset_params["object_fixed_poses"] = None

    env_cfg.terminations.success.params["object_asset_names"] = object_asset_names
    env_cfg.terminations.failure.params["object_asset_names"] = object_asset_names
    return object_asset_names


def configure_env_cfg_for_episode(
    env_cfg: So101BenchEnvCfg,
    episode: BenchmarkEpisodeSpec,
    episode_layout: dict | None = None,
) -> None:
    """Configure object slots and reset metadata for one validated JSONL episode."""

    slot_labels = configure_scene_objects(env_cfg.scene, episode.objects)
    reset_params = env_cfg.events.reset_benchmark_scene.params
    reset_params["task_family"] = episode.task_family
    reset_params["object_count_range"] = (len(episode.objects), len(episode.objects))
    reset_params["active_object_selection"] = "fixed"
    reset_params["fixed_active_object_ids"] = tuple(range(len(episode.objects)))
    reset_params["object_labels"] = slot_labels
    reset_params["shuffle_object_labels"] = False
    reset_params["force_bin_all_objects_instruction"] = False
    reset_params["episode_spec"] = episode.reset_payload()
    reset_params["episode_layout"] = episode_layout


@configclass
class EventCfg:
    """Reset events for robot appearance, robot pose, and benchmark layout."""

    reset_robot_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
            ),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_set_robot_visual_material = EventTerm(
        func=mdp.randomize_robot_color,
        mode="reset",
        params={"color_names": ["orange"]},
    )

    reset_benchmark_scene = _reset_scene_event(TASK_MIXED, (1, 4))


@configclass
class TerminationsCfg:
    """Paper-derived success and measurable failure terms."""

    time_out = DoneTerm(
        func=mdp.task_time_out,
        time_out=True,
        params={"confirm_time_s": SUCCESS_CONFIRM_TIME_S},
    )

    failure = DoneTerm(
        func=mdp.benchmark_failure,
        time_out=False,
        params={
            "object_asset_names": OBJECT_ASSET_NAMES,
            "bin_name": "plastic_bin",
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "min_episode_time_s": MIN_FAILURE_TIME_S,
            "displacement_baseline_time_s": MIN_FAILURE_TIME_S,
            # Keep counting grasp attempts for diagnostics, but do not terminate on the cap.
            # To restore the original cap, set this back to True.
            "enforce_max_grasp_attempts": False,
            "failure_confirm_time_s": FAILURE_CONFIRM_TIME_S,
            "move_straightness_failure_confirm_time_s": FAILURE_CONFIRM_TIME_S,
            "move_past_boundary_failure_confirm_time_s": FAILURE_CONFIRM_TIME_S,
            "move_straightness_tolerance": MOVE_STRAIGHTNESS_TOLERANCE_M,
            "contact_grace_time_s": CONTACT_GRACE_TIME_S,
            "table_bounds": TABLE_BOUNDS,
        },
    )

    # Failure must run before success so the raw failure-condition mask blocks
    # success while a five-second failure confirmation window is in progress.
    success = DoneTerm(
        func=mdp.task_success,
        time_out=False,
        params={
            "object_asset_names": OBJECT_ASSET_NAMES,
            "bin_name": "plastic_bin",
            "table_bounds": TABLE_BOUNDS,
            "min_episode_time_s": MIN_RESET_TIME_S,
            "confirm_time_s": SUCCESS_CONFIRM_TIME_S,
            "contact_grace_time_s": CONTACT_GRACE_TIME_S,
            "move_straightness_tolerance": MOVE_STRAIGHTNESS_TOLERANCE_M,
        },
    )


@configclass
class So101BenchEnvCfg(ManagerBasedRLEnvCfg):
    """Base SO-101 Bench environment suitable for teleop, GR00T rollout, or eval."""

    scene: So101BenchSceneCfg = So101BenchSceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    rewards = None
    commands = None
    curriculum = None

    def __post_init__(self) -> None:
        self.decimation = CONTROL_DECIMATION
        self.episode_length_s = FOUR_OBJECT_BIN_EPISODE_LENGTH_S
        self.scene.num_envs = 1
        self.sim.dt = PHYSICS_DT
        self.sim.physx.enable_ccd = True
        self.sim.render_interval = self.decimation
        self.sim.render.rendering_mode = "quality"
        if os.environ.get(UI_CPU_PHYSICS_ENV_VAR, "").lower() in {"1", "true", "yes", "on"}:
            print("------- CPU SIM ENABLED")
            self.sim.device = "cpu"
            self.sim.use_fabric = False
        self.viewer.eye = (0.04, -0.72, 0.42)
        self.viewer.lookat = (0.04, 0.0, 0.03)


@configclass
class So101BenchBinEnvCfg(So101BenchEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_benchmark_scene = _reset_scene_event(TASK_BIN, (1, 4))


@configclass
class So101BenchNamedBinEnvCfg(So101BenchEnvCfg):
    """Place one named target in the bin while leaving three distractors undisturbed."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_benchmark_scene = _reset_scene_event(TASK_NAMED_BIN, (4, 4))


@configclass
class So101BenchBinSingleObjectEnvCfg(So101BenchEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_benchmark_scene = _single_object_reset_scene_event()
        _park_inactive_scene_objects(self.scene, 0)


@configclass
class So101BenchBinObject1EnvCfg(So101BenchEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_benchmark_scene = _single_object_reset_scene_event(0)
        _park_inactive_scene_objects(self.scene, 0)


@configclass
class So101BenchBinObject2EnvCfg(So101BenchEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_benchmark_scene = _single_object_reset_scene_event(1)
        _park_inactive_scene_objects(self.scene, 1)


@configclass
class So101BenchBinObject3EnvCfg(So101BenchEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_benchmark_scene = _single_object_reset_scene_event(2)
        _park_inactive_scene_objects(self.scene, 2)


@configclass
class So101BenchBinObject4EnvCfg(So101BenchEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_benchmark_scene = _single_object_reset_scene_event(3)
        _park_inactive_scene_objects(self.scene, 3)


@configclass
class So101BenchNextToEnvCfg(So101BenchEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_benchmark_scene = _reset_scene_event(TASK_NEXT_TO, (4, 4))


@configclass
class So101BenchBetweenEnvCfg(So101BenchEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_benchmark_scene = _reset_scene_event(TASK_BETWEEN, (4, 4))


@configclass
class So101BenchMoveEnvCfg(So101BenchEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_benchmark_scene = _reset_scene_event(TASK_MOVE, (1, 4))
