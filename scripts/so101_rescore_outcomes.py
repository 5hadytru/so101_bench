# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Faithfully re-evaluate saved SO-101 Bench replay outcomes against current rule code.

Consumes the outputs of ``so101_lerobot_collect_outcomes.py``:

    outcomes_dir/
        episodes.jsonl     -- per-episode metadata, initial/final scene state, eval_setup
        state/episode_*.npz -- per-step object/bin/robot/ee state and held-object contact (trajectory_stride must be 1)
        frames/*.png        -- unused here

For each saved episode, it builds an in-process environment stub that exposes the same
attributes and ``scene[...].data`` interface the termination functions read, then walks
the trajectory step by step calling ``task_success`` and ``benchmark_failure`` from
``so101_bench.mdp.terminations`` -- so the per-step confirmation counters, displacement
baseline bootstrap, grasp-attempt accumulator, and move-boundary cache all behave the
same way they would in a live env. Whatever the *current* rule code decides is the
new label.

Outputs ``episodes_rescored.jsonl`` and ``summary_rescored.json`` alongside the inputs.

This script does NOT start Isaac Sim. It does need to ``import so101_bench.mdp`` so it
must run under a Python environment that has ``isaaclab`` (and ``isaaclab_tasks``)
installed -- the same one that runs the collector.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _bootstrap_offline_isaaclab() -> None:
    """Make the scoring code importable without a running Omniverse app.

    Rescoring only replays saved trajectories through ``task_success`` /
    ``benchmark_failure`` and never touches the simulator, but newer IsaacLab eagerly
    imports ``omni`` from ``isaaclab.managers`` / ``isaaclab.assets`` at module load,
    which is unavailable unless the app is initialized (``isaaclab.sh -p`` alone does
    not). When ``omni`` is absent we register lightweight stand-ins -- mirroring
    source/so101_bench/test/test_terminations.py -- so the pure-torch scoring runs on
    CPU. When ``omni`` is present the real modules are left untouched.
    """

    try:
        import omni  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    def _register(name: str, **attrs: Any) -> types.ModuleType:
        module = sys.modules.get(name) or types.ModuleType(name)
        sys.modules[name] = module
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    # Only quat_inv / quat_apply are used by terminations.py (bin/next_to/between
    # scoring). Implementations copied verbatim from IsaacLab's isaaclab.utils.math to
    # preserve the (w, x, y, z) quaternion convention.
    def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
        shape = q.shape
        q = q.reshape(-1, 4)
        return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1).view(shape)

    def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)

    def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shape = vec.shape
        quat = quat.reshape(-1, 4)
        vec = vec.reshape(-1, 3)
        xyz = quat[:, 1:]
        t = xyz.cross(vec, dim=-1) * 2
        return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

    _register("isaaclab").__path__ = []
    _register("isaaclab.utils").__path__ = []
    _register("isaaclab.utils.math", quat_conjugate=quat_conjugate, quat_inv=quat_inv, quat_apply=quat_apply)
    _register("isaaclab.assets", RigidObject=object)
    _register("isaaclab.envs", ManagerBasedRLEnv=object)

    class SceneEntityCfg:
        def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
            self.name = name

    _register("isaaclab.managers", SceneEntityCfg=SceneEntityCfg)

    # so101_bench.mdp.resets stub: position/yaw/baseline helpers that read the rescore
    # StubEnv's per-frame state. The StubEnv uses a single env and plain assets (no
    # multi-rigid-body XformPrimView), so these match the real functions' simple branch.
    def benchmark_object_positions(env, object_asset_names):
        return torch.stack([env.scene[name].data.root_pos_w for name in object_asset_names], dim=1)

    def benchmark_object_yaws(env, object_asset_names):
        quats = torch.stack([env.scene[name].data.root_quat_w for name in object_asset_names], dim=1)
        w, x, y, z = quats.unbind(dim=-1)
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def mark_benchmark_robot_start(env, object_asset_names, bin_name, env_ids=None, force_robot_start_time=False):
        if not hasattr(env, "_so101_initial_object_pos_w"):
            return
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        elif env_ids.dtype == torch.bool:
            env_ids = torch.nonzero(env_ids, as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        if not hasattr(env, "_so101_failure_object_pos_w"):
            env._so101_failure_object_pos_w = env._so101_initial_object_pos_w.clone()
        if not hasattr(env, "_so101_failure_bin_pos_w"):
            env._so101_failure_bin_pos_w = env._so101_initial_bin_pos_w.clone()
        if not hasattr(env, "_so101_failure_baseline_recorded"):
            env._so101_failure_baseline_recorded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        new_ids = env_ids[~env._so101_failure_baseline_recorded[env_ids]]
        if new_ids.numel() > 0:
            env._so101_failure_object_pos_w[new_ids] = benchmark_object_positions(env, object_asset_names)[new_ids]
            env._so101_failure_bin_pos_w[new_ids] = env.scene[bin_name].data.root_pos_w[new_ids]
            env._so101_failure_baseline_recorded[new_ids] = True

    package_root = Path(__file__).resolve().parents[1] / "source" / "so101_bench" / "so101_bench"
    _register("so101_bench").__path__ = [str(package_root)]
    _register("so101_bench.mdp").__path__ = [str(package_root / "mdp")]
    _register(
        "so101_bench.mdp.resets",
        benchmark_object_positions=benchmark_object_positions,
        benchmark_object_yaws=benchmark_object_yaws,
        mark_benchmark_robot_start=mark_benchmark_robot_start,
    )

    def _load_from_file(module_name: str, path: Path) -> None:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    _load_from_file("so101_bench.benchmark", package_root / "benchmark.py")
    _load_from_file("so101_bench.mdp.terminations", package_root / "mdp" / "terminations.py")


_bootstrap_offline_isaaclab()

from isaaclab.managers import SceneEntityCfg

from so101_bench.benchmark import load_object_move_footprint_boxes
from so101_bench.mdp.terminations import (
    benchmark_failure,
    task_condition_diagnostics,
    task_success,
    task_time_out,
)


SCHEMA_VERSION = 1
SUCCESS_LABEL_FIELDS = ("success", "failure_reason", "reason", "eval")


# ---------------------------------------------------------------------------
# Env stub: matches the duck-typed interface that terminations.py reads from.
# ---------------------------------------------------------------------------


@dataclass
class _Data:
    root_pos_w: torch.Tensor | None = None
    root_quat_w: torch.Tensor | None = None
    joint_pos: torch.Tensor | None = None
    joint_pos_limits: torch.Tensor | None = None
    target_pos_w: torch.Tensor | None = None


class _Asset:
    def __init__(self) -> None:
        self.data = _Data()


class _Robot(_Asset):
    """Asset with the find_joints / joint_names interface terminations.py expects."""

    def __init__(self, joint_names: list[str]) -> None:
        super().__init__()
        self.joint_names = list(joint_names)

    def find_joints(self, name_pattern: str) -> tuple[list[int], list[str]]:
        if name_pattern in self.joint_names:
            return [self.joint_names.index(name_pattern)], [name_pattern]
        matches = [
            (index, name)
            for index, name in enumerate(self.joint_names)
            if name_pattern.lower() in name.lower()
        ]
        if not matches:
            raise KeyError(f"No joint matches pattern {name_pattern!r}; have {self.joint_names}")
        return [index for index, _ in matches], [name for _, name in matches]


class _Scene:
    def __init__(self, env_origins: torch.Tensor) -> None:
        self._assets: dict[str, _Asset] = {}
        self.env_origins = env_origins

    def __setitem__(self, name: str, asset: _Asset) -> None:
        self._assets[name] = asset

    def __getitem__(self, name: str) -> _Asset:
        return self._assets[name]


class _SimCfg:
    def __init__(self, dt: float) -> None:
        self.dt = dt


class _Cfg:
    def __init__(self, physics_dt: float, decimation: int) -> None:
        self.sim = _SimCfg(physics_dt)
        self.decimation = decimation


class StubEnv:
    """Minimal stand-in for ``ManagerBasedRLEnv`` that the termination functions read."""

    def __init__(
        self,
        *,
        device: torch.device,
        control_dt: float,
        physics_dt: float,
        decimation: int,
        env_origins: torch.Tensor,
        joint_names: list[str],
        action_joint_pos_limits: torch.Tensor,
    ) -> None:
        self.num_envs = 1
        self.device = device
        self.step_dt = float(control_dt)
        self.cfg = _Cfg(physics_dt, decimation)
        self.scene = _Scene(env_origins)
        self.episode_length_buf = torch.zeros(1, dtype=torch.long, device=device)

        robot = _Robot(joint_names)
        robot.data.joint_pos = torch.zeros((1, len(joint_names)), dtype=torch.float32, device=device)
        robot.data.joint_pos_limits = action_joint_pos_limits.unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        self.scene["robot"] = robot

        ee_frame = _Asset()
        ee_frame.data.target_pos_w = torch.zeros((1, 1, 3), dtype=torch.float32, device=device)
        self.scene["ee_frame"] = ee_frame


# ---------------------------------------------------------------------------
# Eval term identical in shape to collect_outcomes._manual_term_eval output.
# ---------------------------------------------------------------------------


@dataclass
class TermEval:
    step: int
    time_s: float
    success: bool
    failure: bool
    timed_out: bool
    reason: str

    @property
    def done(self) -> bool:
        return self.success or self.failure or self.timed_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _yaw_to_quat_wxyz(yaws: torch.Tensor) -> torch.Tensor:
    """Convert an (..., ) yaw tensor to (..., 4) wxyz quaternion."""
    half = 0.5 * yaws
    w = torch.cos(half)
    z = torch.sin(half)
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    return torch.stack((w, x, y, z), dim=-1)


def _tensor(value: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(value, dtype=dtype, device=device)


def _stack_object_field(initial_scene: dict, field: str, *, fallback: Any | None = None) -> list[Any]:
    values = []
    for entry in initial_scene["objects"]:
        item = entry.get(field, None)
        if item is None:
            if fallback is None:
                raise ValueError(f"initial_scene.objects[*].{field} is None; cannot rescore.")
            item = fallback
        values.append(item)
    return values


def _deserialize_scene_entity_cfg(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__scene_entity_cfg__"):
        kwargs: dict[str, Any] = {"name": value["name"]}
        if "joint_names" in value:
            kwargs["joint_names"] = value["joint_names"]
        if "body_names" in value:
            kwargs["body_names"] = value["body_names"]
        return SceneEntityCfg(**kwargs)
    if isinstance(value, dict):
        return {key: _deserialize_scene_entity_cfg(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deserialize_scene_entity_cfg(item) for item in value]
    return value


def _coerce_param_value_for_signature(raw_value: Any, default: Any) -> Any:
    """Coerce saved JSON value back to the shape the function signature expects."""
    if isinstance(default, tuple):
        if not isinstance(raw_value, (list, tuple)):
            return raw_value
        return tuple(_coerce_param_value_for_signature(item, default[i] if i < len(default) else item)
                     for i, item in enumerate(raw_value))
    if isinstance(default, dict) and isinstance(raw_value, dict):
        return {
            key: _coerce_param_value_for_signature(raw_value[key], default[key])
            if key in default
            else raw_value[key]
            for key in raw_value
        }
    return raw_value


def _build_term_params(
    saved: dict[str, Any],
    *,
    overrides: dict[str, Any],
    defaults_for_coercion: dict[str, Any],
) -> dict[str, Any]:
    params = _deserialize_scene_entity_cfg(saved)
    for key, value in params.items():
        if key in defaults_for_coercion:
            params[key] = _coerce_param_value_for_signature(value, defaults_for_coercion[key])
    for key, value in overrides.items():
        params[key] = value
    return params


def _now_stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Per-episode rescoring
# ---------------------------------------------------------------------------


def _initialize_env_state(
    env: StubEnv,
    *,
    initial_scene: dict,
    eval_setup: dict,
    object_asset_names: list[str],
    device: torch.device,
) -> None:
    """Set the static ``_so101_*`` env attributes and bin/object scene assets."""

    num_objects = len(object_asset_names)
    objects = initial_scene["objects"]
    if len(objects) != num_objects:
        raise ValueError(
            f"initial_scene has {len(objects)} object entries but state_schema lists "
            f"{num_objects} ({object_asset_names})."
        )

    # Object scene assets -- give each a stub with .data.root_pos_w / .root_quat_w
    for asset_name in object_asset_names:
        env.scene[asset_name] = _Asset()

    # Bin scene asset
    env.scene[eval_setup["bin_name"]] = _Asset()

    # Active mask
    active_ids = set(initial_scene["active_object_ids"])
    active_mask = torch.zeros((1, num_objects), dtype=torch.bool, device=device)
    for object_id in range(num_objects):
        active_mask[0, object_id] = object_id in active_ids
    env._so101_active_object_mask = active_mask

    # Half extents (fall back to terminations.py defaults if absent)
    half_extents_list = _stack_object_field(initial_scene, "half_extents", fallback=[0.02, 0.02, 0.02])
    env._so101_object_half_extents = _tensor(
        [half_extents_list], dtype=torch.float32, device=device
    )
    env._so101_bin_half_extents = _tensor(
        [initial_scene["bin"].get("half_extents") or [0.125, 0.095, 0.08]],
        dtype=torch.float32,
        device=device,
    )

    object_fp_he = _stack_object_field(
        initial_scene, "footprint_half_extents", fallback=[0.02, 0.02]
    )
    env._so101_object_footprint_half_extents = _tensor(
        [object_fp_he], dtype=torch.float32, device=device
    )
    object_fp_offset = _stack_object_field(
        initial_scene, "footprint_center_offset", fallback=[0.0, 0.0]
    )
    env._so101_object_footprint_center_offsets = _tensor(
        [object_fp_offset], dtype=torch.float32, device=device
    )
    env._so101_object_move_footprint_boxes = [
        _tensor(
            load_object_move_footprint_boxes(str(object_entry["label"]), required=False),
            dtype=torch.float32,
            device=device,
        ).reshape(-1, 4)
        for object_entry in objects
    ]

    bin_fp_he = initial_scene["bin"].get("footprint_half_extents") or [0.125, 0.095]
    env._so101_bin_footprint_half_extents = _tensor([bin_fp_he], dtype=torch.float32, device=device)
    bin_fp_offset = initial_scene["bin"].get("footprint_center_offset") or [0.0, 0.0]
    env._so101_bin_footprint_center_offsets = _tensor(
        [bin_fp_offset], dtype=torch.float32, device=device
    )

    # Task family / target / referent / direction
    env._so101_task_family = [str(initial_scene["task_family"])]
    env._so101_target_object_ids = _tensor(
        [int(initial_scene["target_object_id"])], dtype=torch.long, device=device
    )
    referents = list(initial_scene.get("referent_object_ids", [0, 0]))
    while len(referents) < 2:
        referents.append(0)
    env._so101_referent_object_ids = _tensor(
        [referents[:2]], dtype=torch.long, device=device
    )
    env._so101_direction_ids = _tensor(
        [int(initial_scene.get("direction_id", 0))], dtype=torch.long, device=device
    )
    env.so101_bench_episodes = [
        {
            "env_id": 0,
            "active_object_ids": list(initial_scene["active_object_ids"]),
            "active_labels": [
                str(objects[object_id]["label"])
                for object_id in initial_scene["active_object_ids"]
            ],
        }
    ]

    # Initial positions / yaws (used by displacement baseline + move boundary cache)
    init_object_pos = _tensor(
        [[o["position"] for o in objects]], dtype=torch.float32, device=device
    )
    init_object_yaws = _tensor(
        [[o["yaw"] for o in objects]], dtype=torch.float32, device=device
    )
    init_bin_pos = _tensor([initial_scene["bin"]["position"]], dtype=torch.float32, device=device)
    init_bin_yaws = _tensor(
        [float(initial_scene["bin"].get("yaw", 0.0))], dtype=torch.float32, device=device
    )
    env._so101_initial_object_pos_w = init_object_pos
    env._so101_initial_object_yaws = init_object_yaws
    env._so101_initial_bin_pos_w = init_bin_pos
    env._so101_initial_bin_yaws = init_bin_yaws


def _set_scene_state_for_step(
    env: StubEnv,
    *,
    step_index: int,
    object_pos_w: np.ndarray,
    object_yaw: np.ndarray,
    bin_pos_w: np.ndarray,
    bin_quat_wxyz: np.ndarray | None,
    bin_yaw: float,
    grasped_object_made_contact: bool,
    joint_pos: np.ndarray,
    ee_pos_w: np.ndarray,
    object_asset_names: list[str],
    bin_name: str,
    device: torch.device,
) -> None:
    """Write the per-step state from the trajectory .npz into the stub assets."""

    for object_id, asset_name in enumerate(object_asset_names):
        asset = env.scene[asset_name]
        asset.data.root_pos_w = torch.tensor(
            object_pos_w[object_id], dtype=torch.float32, device=device
        ).unsqueeze(0)
        yaw_tensor = torch.tensor(float(object_yaw[object_id]), dtype=torch.float32, device=device)
        asset.data.root_quat_w = _yaw_to_quat_wxyz(yaw_tensor).unsqueeze(0)

    bin_asset = env.scene[bin_name]
    bin_asset.data.root_pos_w = torch.tensor(bin_pos_w, dtype=torch.float32, device=device).unsqueeze(0)
    if bin_quat_wxyz is not None:
        bin_quat_tensor = torch.tensor(bin_quat_wxyz, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        bin_quat_tensor = _yaw_to_quat_wxyz(
            torch.tensor(float(bin_yaw), dtype=torch.float32, device=device)
        ).unsqueeze(0)
    bin_asset.data.root_quat_w = bin_quat_tensor
    env._so101_grasped_object_made_contact_override = torch.tensor(
        [grasped_object_made_contact],
        dtype=torch.bool,
        device=device,
    )

    robot = env.scene["robot"]
    robot.data.joint_pos = torch.tensor(joint_pos, dtype=torch.float32, device=device).unsqueeze(0)

    ee_frame = env.scene["ee_frame"]
    ee_frame.data.target_pos_w = (
        torch.tensor(ee_pos_w, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    )

    env.episode_length_buf = torch.tensor([step_index], dtype=torch.long, device=device)


def _term_eval_from(
    *,
    step: int,
    control_dt: float,
    success_tensor: torch.Tensor,
    failure_tensor: torch.Tensor,
    timed_out_tensor: torch.Tensor,
    failure_reasons: list[str] | None,
) -> TermEval:
    success = bool(success_tensor[0].item())
    failure = bool(failure_tensor[0].item())
    timed_out = bool(timed_out_tensor[0].item())
    if success:
        reason = "success"
    elif failure:
        reason = failure_reasons[0] if failure_reasons and failure_reasons[0] != "none" else "failure"
    elif timed_out:
        reason = "time_out"
    else:
        reason = "none"
    return TermEval(
        step=step,
        time_s=step * control_dt,
        success=success,
        failure=failure,
        timed_out=timed_out,
        reason=reason,
    )


def _label_from_eval(term_eval: TermEval | None, *, missing_reason: str) -> dict[str, Any]:
    if term_eval is None:
        return {"success": False, "failure_reason": missing_reason, "reason": missing_reason, "eval": None}
    if term_eval.success:
        failure_reason = "none"
    elif term_eval.failure or term_eval.timed_out:
        failure_reason = term_eval.reason
    else:
        failure_reason = missing_reason
    return {
        "success": bool(term_eval.success),
        "failure_reason": failure_reason,
        "reason": term_eval.reason if term_eval.reason != "none" else failure_reason,
        "eval": asdict(term_eval),
    }


def _final_condition_diagnostics(
    env: StubEnv,
    *,
    object_asset_names: list[str],
    success_params: dict[str, Any],
    failure_params: dict[str, Any],
) -> dict[str, Any]:
    snapshots = task_condition_diagnostics(
        env,
        object_asset_names=object_asset_names,
        bin_name=success_params["bin_name"],
        table_bounds=success_params.get("table_bounds"),
        success_min_episode_time_s=success_params.get("min_episode_time_s", 5.0),
        confirm_time_s=success_params.get("confirm_time_s", 3.0),
        move_straightness_tolerance=success_params.get("move_straightness_tolerance", 0.04445),
        failure_min_episode_time_s=failure_params.get("min_episode_time_s", 5.0),
        max_grasp_attempts=failure_params.get("max_grasp_attempts", 3),
        enforce_max_grasp_attempts=failure_params.get("enforce_max_grasp_attempts", True),
        bin_displacement_limit=failure_params.get("bin_displacement_limit", 0.0254),
        non_target_displacement_limit=failure_params.get("non_target_displacement_limit", 0.0127),
        boundary_displacement_limit=failure_params.get("boundary_displacement_limit", 0.0127),
        contact_grace_time_s=failure_params.get(
            "contact_grace_time_s",
            success_params.get("contact_grace_time_s", 3.0),
        ),
    )
    return asdict(snapshots[0])


def _rescore_episode(
    record: dict,
    *,
    outcomes_dir: Path,
    overrides: dict[str, dict[str, Any]],
    success_defaults: dict[str, Any],
    failure_defaults: dict[str, Any],
    device: torch.device,
) -> dict:
    state_path_str = record.get("paths", {}).get("state_trajectory")
    if state_path_str is None:
        raise ValueError(
            "Record has no paths.state_trajectory; rescore requires --save_trajectory output."
        )
    state_path = outcomes_dir / state_path_str
    if not state_path.exists():
        raise FileNotFoundError(f"Trajectory file missing: {state_path}")

    state_schema = record["state_schema"]
    stride = state_schema.get("trajectory_stride")
    if stride is not None and stride != 1:
        raise ValueError(
            f"Episode {record['dataset']['episode_index']} was saved with trajectory_stride="
            f"{stride}; faithful rescoring needs stride 1 so every confirmation step is replayed."
        )

    object_asset_names = list(state_schema["object_asset_names"])
    eval_setup = record["eval_setup"]
    bin_name = eval_setup["bin_name"]
    control_dt = float(eval_setup["control_dt"])
    physics_dt = float(eval_setup["physics_dt"])
    decimation = int(eval_setup.get("decimation") or max(1, round(control_dt / max(physics_dt, 1.0e-9))))

    env_origins = _tensor([eval_setup["env_origins"]], dtype=torch.float32, device=device)
    action_joint_pos_limits = _tensor(
        eval_setup["action_joint_pos_limits"], dtype=torch.float32, device=device
    )
    joint_names = list(eval_setup["action_joint_names"])
    env = StubEnv(
        device=device,
        control_dt=control_dt,
        physics_dt=physics_dt,
        decimation=decimation,
        env_origins=env_origins,
        joint_names=joint_names,
        action_joint_pos_limits=action_joint_pos_limits,
    )
    _initialize_env_state(
        env,
        initial_scene=record["initial_scene"],
        eval_setup=eval_setup,
        object_asset_names=object_asset_names,
        device=device,
    )

    success_params = _build_term_params(
        eval_setup["success_params"],
        overrides=overrides.get("success", {}),
        defaults_for_coercion=success_defaults,
    )
    failure_params = _build_term_params(
        eval_setup["failure_params"],
        overrides=overrides.get("failure", {}),
        defaults_for_coercion=failure_defaults,
    )

    trajectory = np.load(state_path)
    num_steps = int(trajectory["step"].shape[0])
    saved_steps = trajectory["step"]
    has_bin_quat = "bin_quat_wxyz" in trajectory.files
    if "grasped_object_made_contact" not in trajectory.files:
        raise ValueError(
            f"Trajectory {state_path} predates physical contact capture; recollect it before rescoring "
            "against the held-object contact failure rule."
        )

    first_terminal: TermEval | None = None
    final_eval: TermEval | None = None

    for frame in range(num_steps):
        step_index = int(saved_steps[frame])
        _set_scene_state_for_step(
            env,
            step_index=step_index,
            object_pos_w=trajectory["object_pos_w"][frame],
            object_yaw=trajectory["object_yaw"][frame],
            bin_pos_w=trajectory["bin_pos_w"][frame],
            bin_quat_wxyz=trajectory["bin_quat_wxyz"][frame] if has_bin_quat else None,
            bin_yaw=float(trajectory["bin_yaw"][frame]),
            grasped_object_made_contact=bool(trajectory["grasped_object_made_contact"][frame]),
            joint_pos=trajectory["joint_pos"][frame],
            ee_pos_w=trajectory["ee_pos_w"][frame],
            object_asset_names=object_asset_names,
            bin_name=bin_name,
            device=device,
        )
        with torch.inference_mode():
            timed_out_tensor = task_time_out(env)
            failure_tensor = benchmark_failure(env, **failure_params)
            success_tensor = task_success(env, **success_params)
        final_eval = _term_eval_from(
            step=step_index,
            control_dt=control_dt,
            success_tensor=success_tensor,
            failure_tensor=failure_tensor,
            timed_out_tensor=timed_out_tensor,
            failure_reasons=getattr(env, "_so101_failure_reasons", None),
        )
        if final_eval.done and first_terminal is None:
            first_terminal = final_eval

    rescored = dict(record)
    label_source = record.get("label", {}).get("source", "final")
    first_terminal_label = _label_from_eval(
        first_terminal,
        missing_reason="no_terminal_condition_before_action_stream_exhausted",
    )
    final_label = _label_from_eval(final_eval, missing_reason="no_success_condition_at_final_state")
    label = final_label if label_source == "final" else first_terminal_label
    if (
        record.get("episode_length", {}).get("action_stream_exhausted")
        and final_eval is not None
        and not final_eval.done
        and final_eval.reason == "none"
    ):
        final_label["reason"] = "action_stream_exhausted"
        if label_source == "final":
            label = final_label

    rescored["first_terminal_eval"] = first_terminal_label
    rescored["final_eval"] = final_label
    rescored["final_diagnostics"] = _final_condition_diagnostics(
        env,
        object_asset_names=object_asset_names,
        success_params=success_params,
        failure_params=failure_params,
    )
    rescored["label"] = {"source": label_source, **label}
    rescored["rescore"] = {
        "rescored_at": _now_stamp(),
        "schema_version": SCHEMA_VERSION,
        "label_source": label_source,
        "overrides": {kind: dict(items) for kind, items in overrides.items() if items},
        "original_label": {key: record.get("label", {}).get(key) for key in SUCCESS_LABEL_FIELDS}
        | {"source": label_source},
    }
    return rescored


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_overrides(raw_overrides: list[str]) -> dict[str, dict[str, Any]]:
    """Parse ``--override success.confirm_time_s=1.0`` style flags into nested dicts."""
    result: dict[str, dict[str, Any]] = {"success": {}, "failure": {}}
    for raw in raw_overrides:
        if "=" not in raw:
            raise argparse.ArgumentTypeError(
                f"Override {raw!r} must have the form KIND.NAME=VALUE (KIND in success|failure)."
            )
        path, value_str = raw.split("=", 1)
        if "." not in path:
            raise argparse.ArgumentTypeError(
                f"Override {raw!r} must have the form KIND.NAME=VALUE (KIND in success|failure)."
            )
        kind, name = path.split(".", 1)
        if kind not in ("success", "failure"):
            raise argparse.ArgumentTypeError(
                f"Override KIND must be 'success' or 'failure', got {kind!r}."
            )
        try:
            value = json.loads(value_str)
        except json.JSONDecodeError:
            value = value_str
        result[kind][name] = value
    return result


def _parse_episode_indices(raw_indices: str) -> set[int]:
    indices = set()
    for raw_index in raw_indices.split(","):
        raw_index = raw_index.strip()
        if not raw_index:
            continue
        try:
            indices.add(int(raw_index))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid dataset episode index {raw_index!r} in {raw_indices!r}."
            ) from exc
    if not indices:
        raise argparse.ArgumentTypeError("--episode_indices did not contain any indices.")
    return indices


def _signature_defaults(func) -> dict[str, Any]:
    import inspect

    signature = inspect.signature(func)
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outcomes_dir",
        type=Path,
        required=True,
        help="Directory containing episodes.jsonl and state/episode_*.npz produced by the collector.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write episodes_rescored.jsonl and summary_rescored.json. Defaults to --outcomes_dir.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Override a success/failure parameter as KIND.NAME=VALUE, e.g. "
            "--override success.confirm_time_s=1.0 --override failure.bin_displacement_limit=0.05. "
            "VALUE is parsed as JSON when possible (numbers, true/false, [lists]); else as a string."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for the stub env. CPU is fine and avoids GPU contention with sim.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally rescore only the first N episodes (useful while debugging overrides).",
    )
    parser.add_argument(
        "--episode_indices",
        type=_parse_episode_indices,
        default=None,
        help="Comma-separated dataset episode indices to rescore, such as 6,32,70.",
    )
    args = parser.parse_args()

    outcomes_dir: Path = args.outcomes_dir
    if not outcomes_dir.exists():
        raise FileNotFoundError(f"--outcomes_dir does not exist: {outcomes_dir}")
    episodes_path = outcomes_dir / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episodes.jsonl under {outcomes_dir}.")

    output_dir = args.output_dir or outcomes_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rescored_path = output_dir / "episodes_rescored.jsonl"
    summary_path = output_dir / "summary_rescored.json"

    overrides = _parse_overrides(args.override)
    success_defaults = _signature_defaults(task_success)
    failure_defaults = _signature_defaults(benchmark_failure)

    device = torch.device(args.device)

    records = []
    with episodes_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{episodes_path}:{line_no}: invalid JSON: {exc}") from exc

    if args.limit is not None:
        records = records[: args.limit]
    if args.episode_indices is not None:
        records_by_index = {int(record["dataset"]["episode_index"]): record for record in records}
        missing_indices = sorted(args.episode_indices - records_by_index.keys())
        if missing_indices:
            raise ValueError(f"Dataset episode indices are not present in {episodes_path}: {missing_indices}")
        records = [records_by_index[index] for index in sorted(args.episode_indices)]

    summary_records = []
    print(f"[INFO]: Rescoring {len(records)} episode(s) from {episodes_path}")
    if overrides["success"] or overrides["failure"]:
        print(f"[INFO]: Overrides: {overrides}")

    with rescored_path.open("w", encoding="utf-8") as out:
        for index, record in enumerate(records):
            rescored = _rescore_episode(
                record,
                outcomes_dir=outcomes_dir,
                overrides=overrides,
                success_defaults=success_defaults,
                failure_defaults=failure_defaults,
                device=device,
            )
            out.write(json.dumps(rescored, separators=(",", ":")) + "\n")
            out.flush()
            summary_records.append(rescored)
            label = rescored["label"]
            original_success = record.get("label", {}).get("success")
            change_marker = ""
            if original_success is not None and bool(original_success) != bool(label["success"]):
                change_marker = "  [flip vs original]"
            print(
                f"[INFO]: Episode {index + 1}/{len(records)} "
                f"dataset_ep={rescored['dataset']['episode_index']} "
                f"benchmark_row={rescored['benchmark']['episode_index']} "
                f"success={label['success']} reason={label['failure_reason']}{change_marker}"
            )

    successes = sum(1 for entry in summary_records if entry["label"]["success"])
    failures = len(summary_records) - successes
    failure_counts: dict[str, int] = {}
    for entry in summary_records:
        reason = entry["label"]["failure_reason"]
        failure_counts[reason] = failure_counts.get(reason, 0) + 1
    flips = sum(
        1
        for entry in summary_records
        if entry["rescore"]["original_label"]["success"] is not None
        and bool(entry["rescore"]["original_label"]["success"]) != bool(entry["label"]["success"])
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "rescored_at": _now_stamp(),
        "source_episodes_path": str(episodes_path),
        "rescored_episodes_path": str(rescored_path),
        "completed_episodes": len(summary_records),
        "successes": successes,
        "failures": failures,
        "success_rate": successes / max(len(summary_records), 1),
        "failure_reason_counts": failure_counts,
        "label_flips_vs_original": flips,
        "overrides": {kind: dict(items) for kind, items in overrides.items() if items},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[INFO]: Rescore summary: success={successes}/{len(summary_records)} "
        f"({100.0 * summary['success_rate']:.1f}%), failures={failures}, "
        f"flips_vs_original={flips}"
    )
    print(f"[INFO]: Wrote {rescored_path} and {summary_path}")


if __name__ == "__main__":
    main()
