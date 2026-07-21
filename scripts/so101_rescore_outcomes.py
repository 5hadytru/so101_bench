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
import math
import sys
import types
from dataclasses import asdict, dataclass, is_dataclass
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

from so101_bench.benchmark import (
    episode_spec_from_json,
    infer_task_family,
    load_object_move_footprint_boxes,
)
from so101_bench.mdp.terminations import (
    benchmark_failure,
    task_condition_diagnostics,
    task_success,
    task_time_out,
)


# Version of the *rescore annotation*, not the source collection record.  The
# collector's top-level ``schema_version`` is intentionally preserved when a
# record is copied below so schema-v1 and schema-v2 collections remain
# distinguishable after rescoring.
RESCORE_SCHEMA_VERSION = 2
SUCCESS_LABEL_FIELDS = ("success", "failure_reason", "reason", "eval")


# ---------------------------------------------------------------------------
# Env stub: matches the duck-typed interface that terminations.py reads from.
# ---------------------------------------------------------------------------


@dataclass
class _Data:
    root_pos_w: torch.Tensor | None = None
    root_quat_w: torch.Tensor | None = None
    root_lin_vel_w: torch.Tensor | None = None
    root_ang_vel_w: torch.Tensor | None = None
    joint_pos: torch.Tensor | None = None
    joint_vel: torch.Tensor | None = None
    applied_torque: torch.Tensor | None = None
    joint_pos_limits: torch.Tensor | None = None
    target_pos_w: torch.Tensor | None = None
    target_quat_w: torch.Tensor | None = None
    target_lin_vel_w: torch.Tensor | None = None
    target_ang_vel_w: torch.Tensor | None = None


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


def _jsonable(value: Any) -> Any:
    """Recursively convert scorer state into plain JSON-compatible values.

    Failure diagnostics are dataclasses today, but keeping this conversion
    deliberately permissive lets a richer classifier add tensors, NumPy
    scalars, mappings, or nested dataclasses without coupling this script to a
    particular diagnostic schema.
    """

    if value.__class__.__name__ == "SceneEntityCfg" and hasattr(value, "name"):
        payload = {"__scene_entity_cfg__": True, "name": str(value.name)}
        for field_name in ("joint_names", "body_names"):
            field_value = getattr(value, field_name, None)
            if field_value is not None:
                payload[field_name] = _jsonable(field_value)
        return payload
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _postmortem_failure_diagnostics(env: StubEnv) -> list[dict[str, Any]]:
    """Return the latest diagnostics produced by ``benchmark_failure``.

    Old rule implementations and episodes lacking sufficient evidence may not
    create the attribute.  Treat that as an empty result rather than making
    schema-v1 trajectories unrescorable.
    """

    raw = getattr(env, "_so101_postmortem_failure_diagnostics", None)
    if raw is None:
        return []
    converted = _jsonable(raw)
    if isinstance(converted, list):
        return converted
    return [converted]


def _postmortem_failure_type(value: Any) -> str | None:
    """Extract either the legacy or richer classifier's primary label."""

    if isinstance(value, list):
        value = value[0] if len(value) == 1 else None
    if not isinstance(value, dict):
        return None
    for field in ("primary_failure", "failure_type", "primary_failure_type"):
        label = value.get(field)
        if label:
            return str(label)
    return None


def _single_env_postmortem_raw(diagnostics: list[dict[str, Any]]) -> Any:
    """Match the collector's single-environment postmortem extraction."""

    if not diagnostics:
        return None
    if len(diagnostics) == 1:
        return diagnostics[0]
    return diagnostics


def _failure_attribution_from_raw(
    raw: Any,
    *,
    label: dict[str, Any],
    basis: str,
) -> dict[str, Any]:
    """Gate behavioral attribution against the label whose state it explains."""

    raw = _jsonable(raw)
    raw_type = _postmortem_failure_type(raw)
    applicable = bool(
        not label["success"]
        and raw is not None
        and raw_type not in {"none", "not_applicable"}
    )
    return {
        "basis": basis,
        "live_failure_reason": label["failure_reason"],
        "applicable": applicable,
        "postmortem": raw if applicable else None,
        "postmortem_raw": raw,
        "suppressed_reason": (
            "episode_succeeded"
            if label["success"]
            else "classifier_not_applicable"
            if not applicable
            else None
        ),
    }


def _optional_trajectory_frame(
    trajectory: np.lib.npyio.NpzFile,
    frame: int,
    *field_names: str,
) -> np.ndarray | None:
    """Read the first available optional schema-v2 field for one frame."""

    available = set(trajectory.files)
    for field_name in field_names:
        if field_name in available:
            return trajectory[field_name][frame]
    return None


def _optional_finite_trajectory_frame(
    trajectory: np.lib.npyio.NpzFile,
    frame: int,
    *field_names: str,
) -> np.ndarray | None:
    """Read the first available all-finite field, allowing saved fallbacks."""

    available = set(trajectory.files)
    for field_name in field_names:
        if field_name not in available:
            continue
        value = trajectory[field_name][frame]
        try:
            if np.all(np.isfinite(value)):
                return value
        except TypeError:
            continue
    return None


def _trajectory_action_phase(
    trajectory: np.lib.npyio.NpzFile,
    frame: int,
) -> str | None:
    """Return a schema-v2 sample phase, preserving schema-v1 behavior.

    Schema-v2 collections include a reset snapshot for telemetry only.  It was
    never passed through the collector's live termination functions, so the
    rescorer must not treat it as a real evaluation step.  Schema-v1 files do
    not have ``action_phase`` and intentionally retain their original replay
    semantics.
    """

    if "action_phase" not in trajectory.files:
        return None
    value: Any = np.asarray(trajectory["action_phase"][frame]).reshape(-1)[0]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value)


def _trajectory_termination_evaluation_applied(
    trajectory: np.lib.npyio.NpzFile,
    frame: int,
    *,
    source_schema_version: int,
    action_phase: str | None,
) -> bool:
    """Whether the collector ran termination rules for this saved frame.

    New schema-v2 traces state this directly.  Early schema-v2 traces identify
    the telemetry-only reset snapshot by phase; the step-zero fallback covers
    transitional traces that predate both flags.  Schema v1 is never inferred
    from step number, preserving its historical replay behavior.
    """

    if "termination_evaluation_applied" in trajectory.files:
        value = np.asarray(trajectory["termination_evaluation_applied"][frame]).reshape(-1)[0]
        return bool(value)
    if "is_reset_sample" in trajectory.files:
        value = np.asarray(trajectory["is_reset_sample"][frame]).reshape(-1)[0]
        return not bool(value)
    if action_phase is not None:
        return action_phase != "reset"
    if source_schema_version >= 2:
        step = int(np.asarray(trajectory["step"][frame]).reshape(-1)[0])
        return not (frame == 0 and step == 0)
    return True


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
    recorded_task_family = str(initial_scene["task_family"])
    instruction = str(initial_scene.get("instruction", "")).strip()
    env._so101_task_family = [
        infer_task_family(instruction) if instruction else recorded_task_family
    ]
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
    env._so101_failure_object_pos_w = init_object_pos.clone()
    env._so101_failure_bin_pos_w = init_bin_pos.clone()
    env._so101_failure_baseline_recorded = torch.zeros(1, dtype=torch.bool, device=device)


def _set_scene_state_for_step(
    env: StubEnv,
    *,
    step_index: int,
    object_pos_w: np.ndarray,
    object_yaw: np.ndarray,
    object_quat_wxyz: np.ndarray | None,
    object_lin_vel_w: np.ndarray | None,
    object_ang_vel_w: np.ndarray | None,
    bin_pos_w: np.ndarray,
    bin_quat_wxyz: np.ndarray | None,
    bin_lin_vel_w: np.ndarray | None,
    bin_ang_vel_w: np.ndarray | None,
    bin_yaw: float,
    grasped_object_made_contact: bool,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray | None,
    joint_effort: np.ndarray | None,
    ee_pos_w: np.ndarray,
    ee_quat_wxyz: np.ndarray | None,
    ee_lin_vel_w: np.ndarray | None,
    ee_ang_vel_w: np.ndarray | None,
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
        if object_quat_wxyz is not None and np.all(np.isfinite(object_quat_wxyz[object_id])):
            asset.data.root_quat_w = torch.tensor(
                object_quat_wxyz[object_id], dtype=torch.float32, device=device
            ).unsqueeze(0)
        else:
            yaw_tensor = torch.tensor(float(object_yaw[object_id]), dtype=torch.float32, device=device)
            asset.data.root_quat_w = _yaw_to_quat_wxyz(yaw_tensor).unsqueeze(0)
        if object_lin_vel_w is not None and np.all(np.isfinite(object_lin_vel_w[object_id])):
            asset.data.root_lin_vel_w = torch.tensor(
                object_lin_vel_w[object_id], dtype=torch.float32, device=device
            ).unsqueeze(0)
        else:
            asset.data.root_lin_vel_w = None
        if object_ang_vel_w is not None and np.all(np.isfinite(object_ang_vel_w[object_id])):
            asset.data.root_ang_vel_w = torch.tensor(
                object_ang_vel_w[object_id], dtype=torch.float32, device=device
            ).unsqueeze(0)
        else:
            asset.data.root_ang_vel_w = None

    bin_asset = env.scene[bin_name]
    bin_asset.data.root_pos_w = torch.tensor(bin_pos_w, dtype=torch.float32, device=device).unsqueeze(0)
    if bin_quat_wxyz is not None:
        bin_quat_tensor = torch.tensor(bin_quat_wxyz, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        bin_quat_tensor = _yaw_to_quat_wxyz(
            torch.tensor(float(bin_yaw), dtype=torch.float32, device=device)
        ).unsqueeze(0)
    bin_asset.data.root_quat_w = bin_quat_tensor
    if bin_lin_vel_w is not None and np.all(np.isfinite(bin_lin_vel_w)):
        bin_asset.data.root_lin_vel_w = torch.tensor(
            bin_lin_vel_w, dtype=torch.float32, device=device
        ).unsqueeze(0)
    else:
        bin_asset.data.root_lin_vel_w = None
    if bin_ang_vel_w is not None and np.all(np.isfinite(bin_ang_vel_w)):
        bin_asset.data.root_ang_vel_w = torch.tensor(
            bin_ang_vel_w, dtype=torch.float32, device=device
        ).unsqueeze(0)
    else:
        bin_asset.data.root_ang_vel_w = None
    env._so101_grasped_object_made_contact_override = torch.tensor(
        [grasped_object_made_contact],
        dtype=torch.bool,
        device=device,
    )

    robot = env.scene["robot"]
    robot.data.joint_pos = torch.tensor(joint_pos, dtype=torch.float32, device=device).unsqueeze(0)
    if joint_vel is not None and np.all(np.isfinite(joint_vel)):
        robot.data.joint_vel = torch.tensor(joint_vel, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        robot.data.joint_vel = None
    if joint_effort is not None and np.all(np.isfinite(joint_effort)):
        robot.data.applied_torque = torch.tensor(
            joint_effort, dtype=torch.float32, device=device
        ).unsqueeze(0)
    else:
        robot.data.applied_torque = None

    ee_frame = env.scene["ee_frame"]
    ee_frame.data.target_pos_w = (
        torch.tensor(ee_pos_w, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    )
    if ee_quat_wxyz is not None and np.all(np.isfinite(ee_quat_wxyz)):
        ee_frame.data.target_quat_w = (
            torch.tensor(ee_quat_wxyz, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        )
    else:
        ee_frame.data.target_quat_w = None
    if ee_lin_vel_w is not None and np.all(np.isfinite(ee_lin_vel_w)):
        ee_frame.data.target_lin_vel_w = (
            torch.tensor(ee_lin_vel_w, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        )
    else:
        ee_frame.data.target_lin_vel_w = None
    if ee_ang_vel_w is not None and np.all(np.isfinite(ee_ang_vel_w)):
        ee_frame.data.target_ang_vel_w = (
            torch.tensor(ee_ang_vel_w, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        )
    else:
        ee_frame.data.target_ang_vel_w = None

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


def _rescored_rule_analysis(
    trace: list[dict[str, Any]],
    *,
    selected_label: dict[str, Any],
    final_label: dict[str, Any],
    label_source: str,
    final_success_confirm_time_disabled: bool,
    final_confirmation_waiver_applied: bool,
    control_dt: float,
) -> dict[str, Any]:
    """Summarize rule-dependent evidence recomputed during offline replay.

    Physical extrema in the collector's explanation remain useful source facts,
    but confirmation windows, terminal transitions, closest-miss status, and
    success quality can change under an override.  Keep this compact trace as the
    authoritative rescore analysis instead of silently reusing source-policy
    conclusions.
    """

    if not trace:
        return {
            "available": False,
            "reason": "no_scorable_trajectory_frames",
            "rule_trace": [],
            "rule_event_ledger": [],
        }

    counters = np.asarray([point["success_counter"] for point in trace], dtype=np.int64)
    required = np.maximum(
        np.asarray([point["success_required_steps"] for point in trace], dtype=np.int64),
        1,
    )
    confirmation_multiples = counters / required
    fractions = np.minimum(confirmation_multiples, 1.0)
    best_id = int(np.argmax(confirmation_multiples))
    live_success = np.asarray(
        [point["live_success_confirmed"] for point in trace], dtype=np.bool_
    )
    first_live_id = int(np.flatnonzero(live_success)[0]) if np.any(live_success) else None
    selected_success_ever = bool(np.any(live_success) or final_confirmation_waiver_applied)
    first_selected_id = first_live_id
    if first_selected_id is None and final_confirmation_waiver_applied:
        first_selected_id = len(trace) - 1
    success_was_transient = bool(
        np.any(live_success)
        and (
            not bool(final_label["success"])
            or not np.all(live_success[first_live_id:])
        )
    )
    stable_live_success = bool(
        first_live_id is not None
        and np.all(live_success[first_live_id:])
        and bool(final_label["success"])
    )
    if not bool(final_label["success"]):
        success_quality = "failure"
    elif success_was_transient:
        success_quality = "transient_success"
    elif final_confirmation_waiver_applied:
        success_quality = "confirmation_waived_success"
    elif stable_live_success:
        success_quality = "stable_success"
    else:
        success_quality = "confirmed_success"

    events: list[dict[str, Any]] = []

    def add(point: dict[str, Any], event_type: str, **payload: Any) -> None:
        events.append(
            {
                "type": event_type,
                "step": int(point["step"]),
                "time_s": float(point["time_s"]),
                **payload,
            }
        )

    previous_candidate = False
    previous_live_success = False
    previous_failure = False
    previous_timed_out = False
    previous_failure_reason: str | None = None
    previous_postmortem_type: str | None = None
    for point in trace:
        candidate = int(point["success_counter"]) > 0
        if candidate and not previous_candidate:
            add(
                point,
                "success_candidate_started",
                held_steps=int(point["success_counter"]),
                required_steps=int(point["success_required_steps"]),
            )
        elif not candidate and previous_candidate:
            add(point, "success_candidate_lost")
        previous_candidate = candidate

        current_live_success = bool(point["live_success_confirmed"])
        if current_live_success and not previous_live_success:
            add(point, "success_confirmed", basis="rescored_live_benchmark_rule")
        elif not current_live_success and previous_live_success:
            add(point, "success_lost", basis="rescored_live_benchmark_rule")
        previous_live_success = current_live_success

        current_failure = bool(point["failure"])
        current_failure_reason = str(point["failure_reason"])
        if current_failure and (
            not previous_failure or current_failure_reason != previous_failure_reason
        ):
            add(point, "failure_confirmed", reason=current_failure_reason)
        elif not current_failure and previous_failure:
            add(point, "failure_no_longer_confirmed", previous_reason=previous_failure_reason)
        previous_failure = current_failure
        previous_failure_reason = current_failure_reason

        current_timed_out = bool(point["timed_out"])
        if current_timed_out and not previous_timed_out:
            add(point, "time_out")
        previous_timed_out = current_timed_out

        postmortem_type = point.get("postmortem_failure_type")
        if (
            bool(point.get("policy_control_active", False))
            and postmortem_type
            and postmortem_type != previous_postmortem_type
        ):
            add(
                point,
                "behavioral_classification_changed",
                previous_failure_type=previous_postmortem_type,
                failure_type=postmortem_type,
                confidence=point.get("postmortem_confidence"),
                rationale=point.get("postmortem_rationale"),
            )
            previous_postmortem_type = str(postmortem_type)

    if final_confirmation_waiver_applied:
        point = trace[-1]
        add(
            point,
            "final_confirmation_waiver_applied",
            held_steps=int(point["success_counter"]),
            required_steps=int(point["success_required_steps"]),
            live_success_confirmed=bool(point["live_success_confirmed"]),
            final_scoring_success=bool(final_label["success"]),
        )
    for event_id, event in enumerate(events):
        event["event_id"] = event_id

    best_confirmation = {
        "maximum_held_steps": int(counters[best_id]),
        "required_steps": int(required[best_id]),
        "maximum_fraction": float(fractions[best_id]),
        "maximum_multiple_of_required_hold": float(confirmation_multiples[best_id]),
        "best_step": int(trace[best_id]["step"]),
        "best_time_s": float(trace[best_id]["time_s"]),
        "longest_candidate_duration_s": float(counters[best_id] * control_dt),
        "basis": "rescored_live_benchmark_rule",
    }
    closest_miss = None
    if not bool(final_label["success"]):
        closest_miss = {
            "condition": (
                "success_confirmed_then_lost"
                if bool(np.any(live_success))
                else "success_confirmation"
            ),
            "best_step": int(trace[best_id]["step"]),
            "best_time_s": float(trace[best_id]["time_s"]),
            "held_steps": int(counters[best_id]),
            "required_steps": int(required[best_id]),
            "shortfall_steps": max(0, int(required[best_id] - counters[best_id])),
            "shortfall_s": max(
                0.0,
                float((required[best_id] - counters[best_id]) * control_dt),
            ),
            "confirmation_fraction": float(fractions[best_id]),
            "basis": "rescored_live_benchmark_rule",
        }

    return {
        "available": True,
        "analysis_schema_version": 1,
        "basis": "offline_replay_with_effective_rescore_parameters",
        "label_source": label_source,
        "rule_trace": trace,
        "rule_event_ledger": events,
        "best_achieved": {"success_confirmation": best_confirmation},
        "closest_miss": closest_miss,
        "outcome_quality": {
            "final_success": bool(final_label["success"]),
            "selected_label_success": bool(selected_label["success"]),
            "success_ever_confirmed": bool(np.any(live_success)),
            "live_success_ever_confirmed": bool(np.any(live_success)),
            "selected_scoring_success_ever_true": selected_success_ever,
            "first_success_step": (
                int(trace[first_live_id]["step"]) if first_live_id is not None else None
            ),
            "first_success_time_s": (
                float(trace[first_live_id]["time_s"]) if first_live_id is not None else None
            ),
            "first_live_confirmed_success_step": (
                int(trace[first_live_id]["step"]) if first_live_id is not None else None
            ),
            "first_live_confirmed_success_time_s": (
                float(trace[first_live_id]["time_s"]) if first_live_id is not None else None
            ),
            "first_selected_scoring_success_step": (
                int(trace[first_selected_id]["step"]) if first_selected_id is not None else None
            ),
            "first_selected_scoring_success_time_s": (
                float(trace[first_selected_id]["time_s"])
                if first_selected_id is not None
                else None
            ),
            "stable_from_first_success_through_end": stable_live_success,
            "success_was_transient": success_was_transient,
            "success_quality": success_quality,
            "clean_stable_success": success_quality == "stable_success",
            "final_success_confirmation_time_disabled": bool(
                final_success_confirm_time_disabled
            ),
            "final_confirmation_waiver_applied": bool(
                final_confirmation_waiver_applied
            ),
            "maximum_confirmation_fraction": float(fractions[best_id]),
            "maximum_confirmation_multiple_of_required_hold": float(
                confirmation_multiples[best_id]
            ),
            "basis": "rescored_rule_trace",
        },
    }


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
            f"{stride}; faithful rescoring of confirmation, failure-hold, and postmortem "
            "timers requires trajectory_stride=1 so every evaluated control step is replayed."
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
    initial_scene = dict(record["initial_scene"])
    benchmark_record = dict(record.get("benchmark") or {})
    canonical_episode = None
    if benchmark_record.get("objects") and benchmark_record.get("instruction"):
        episode_row = dict(benchmark_record.get("metadata") or {})
        # Old outcome files may say task_family=bin because named-bin did not yet
        # exist. Re-infer the family and target from the instruction instead.
        episode_row.pop("task_family", None)
        episode_row["objects"] = list(benchmark_record["objects"])
        episode_row["instruction"] = str(benchmark_record["instruction"])
        canonical_episode = episode_spec_from_json(
            episode_row,
            source=f"outcome episode {record.get('dataset', {}).get('episode_index', '?')}",
        )
        target_label = canonical_episode.objects[canonical_episode.target_object_id]
        matching_scene_ids = [
            int(object_entry["slot"])
            for object_entry in initial_scene.get("objects", [])
            if str(object_entry.get("label")) == target_label
        ]
        if len(matching_scene_ids) != 1:
            raise ValueError(
                f"Could not uniquely map canonical target {target_label!r} into initial_scene: "
                f"matches={matching_scene_ids}."
            )
        initial_scene["task_family"] = canonical_episode.task_family
        initial_scene["target_object_id"] = matching_scene_ids[0]

    _initialize_env_state(
        env,
        initial_scene=initial_scene,
        eval_setup=eval_setup,
        object_asset_names=object_asset_names,
        device=device,
    )

    success_params = _build_term_params(
        eval_setup["success_params"],
        overrides=overrides.get("success", {}),
        defaults_for_coercion=success_defaults,
    )
    persisted_final_success_confirm_time_disabled = bool(
        eval_setup.get("final_success_confirm_time_disabled", False)
    )
    if persisted_final_success_confirm_time_disabled and "final_success_params" not in eval_setup:
        raise ValueError(
            f"Episode {record['dataset']['episode_index']} waived final success confirmation, "
            "but its eval_setup predates separate live/final success parameters. The original "
            "live confirmation window cannot be reconstructed faithfully; recollect this episode."
        )
    final_success_params = _build_term_params(
        eval_setup.get("final_success_params", eval_setup["success_params"]),
        overrides=overrides.get("success", {}),
        defaults_for_coercion=success_defaults,
    )
    # A rescore override applies to both live and final success rules.  If it
    # restores a non-zero final confirmation window, the persisted collector
    # policy is no longer active for this rescore.
    final_success_confirm_time_disabled = bool(
        persisted_final_success_confirm_time_disabled
        and float(final_success_params.get("confirm_time_s", 3.0)) <= 0.0
    )
    failure_params = _build_term_params(
        eval_setup["failure_params"],
        overrides=overrides.get("failure", {}),
        defaults_for_coercion=failure_defaults,
    )
    # ``task_time_out`` extends the nominal timeout by the same confirmation
    # window used by the success rule.  Reuse the effective (possibly
    # overridden) success parameter so rescore timing matches the configured
    # term rather than silently falling back to its module default.
    timeout_confirm_time_s = float(success_params.get("confirm_time_s", 3.0))

    trajectory = np.load(state_path)
    num_steps = int(trajectory["step"].shape[0])
    saved_steps = trajectory["step"]
    trajectory_schema_version = (
        int(np.asarray(trajectory["schema_version"]).reshape(-1)[0])
        if "schema_version" in trajectory.files
        else int(state_schema.get("schema_version", record.get("schema_version", 1)))
    )
    action_phases = [
        _trajectory_action_phase(trajectory, frame)
        for frame in range(num_steps)
    ]
    if trajectory_schema_version >= 2:
        env._so101_policy_control_active = torch.zeros(
            env.num_envs, dtype=torch.bool, device=device
        )
    evaluation_mask_source = (
        "termination_evaluation_applied"
        if "termination_evaluation_applied" in trajectory.files
        else "is_reset_sample"
        if "is_reset_sample" in trajectory.files
        else "action_phase"
        if "action_phase" in trajectory.files
        else "schema_v2_step_zero_fallback"
        if trajectory_schema_version >= 2
        else "schema_v1_all_frames"
    )
    evaluation_applied = [
        _trajectory_termination_evaluation_applied(
            trajectory,
            frame,
            source_schema_version=trajectory_schema_version,
            action_phase=action_phases[frame],
        )
        for frame in range(num_steps)
    ]
    scorable_steps = np.asarray(
        [
            int(saved_steps[frame])
            for frame, applied in enumerate(evaluation_applied)
            if applied
        ],
        dtype=np.int64,
    )
    if scorable_steps.size == 0:
        trajectory.close()
        raise ValueError(
            f"Episode {record['dataset']['episode_index']} has no evaluated trajectory frames; "
            "faithful rescoring requires the complete evaluated trace beginning at step 1."
        )
    if int(scorable_steps[0]) != 1:
        first_saved_step = int(scorable_steps[0])
        trajectory.close()
        raise ValueError(
            f"Episode {record['dataset']['episode_index']} begins its evaluated trajectory at "
            f"step {first_saved_step}, not step 1; temporal counters cannot be reconstructed "
            "from a truncated prefix."
        )
    if scorable_steps.size > 1:
        step_gaps = np.diff(scorable_steps)
        if np.any(step_gaps != 1):
            first_bad_gap = int(np.flatnonzero(step_gaps != 1)[0])
            previous_step = int(scorable_steps[first_bad_gap])
            next_step = int(scorable_steps[first_bad_gap + 1])
            trajectory.close()
            raise ValueError(
                f"Episode {record['dataset']['episode_index']} has non-consecutive evaluated "
                f"trajectory steps {previous_step}->{next_step}; faithful rescoring of temporal "
                "rules requires one saved sample per evaluated control step (trajectory_stride=1)."
            )
    has_bin_quat = "bin_quat_wxyz" in trajectory.files
    if "grasped_object_made_contact" not in trajectory.files:
        raise ValueError(
            f"Trajectory {state_path} predates physical contact capture; recollect it before rescoring "
            "against the held-object contact failure rule."
        )

    first_terminal: TermEval | None = None
    first_terminal_postmortem_raw: Any = None
    final_eval: TermEval | None = None
    telemetry_only_frames_skipped = 0
    rescored_rule_trace: list[dict[str, Any]] = []

    for frame in range(num_steps):
        step_index = int(saved_steps[frame])
        _set_scene_state_for_step(
            env,
            step_index=step_index,
            object_pos_w=trajectory["object_pos_w"][frame],
            object_yaw=trajectory["object_yaw"][frame],
            object_quat_wxyz=_optional_trajectory_frame(
                trajectory, frame, "object_quat_wxyz", "object_quat_w"
            ),
            object_lin_vel_w=_optional_finite_trajectory_frame(
                trajectory, frame, "object_lin_vel_w", "object_lin_vel_fd_w"
            ),
            object_ang_vel_w=_optional_finite_trajectory_frame(
                trajectory, frame, "object_ang_vel_w"
            ),
            bin_pos_w=trajectory["bin_pos_w"][frame],
            bin_quat_wxyz=trajectory["bin_quat_wxyz"][frame] if has_bin_quat else None,
            bin_lin_vel_w=_optional_finite_trajectory_frame(
                trajectory, frame, "bin_lin_vel_w", "bin_lin_vel_fd_w"
            ),
            bin_ang_vel_w=_optional_finite_trajectory_frame(trajectory, frame, "bin_ang_vel_w"),
            bin_yaw=float(trajectory["bin_yaw"][frame]),
            grasped_object_made_contact=bool(trajectory["grasped_object_made_contact"][frame]),
            joint_pos=trajectory["joint_pos"][frame],
            joint_vel=_optional_finite_trajectory_frame(trajectory, frame, "joint_vel"),
            joint_effort=_optional_finite_trajectory_frame(
                trajectory, frame, "joint_effort", "joint_applied_torque"
            ),
            ee_pos_w=trajectory["ee_pos_w"][frame],
            ee_quat_wxyz=_optional_trajectory_frame(
                trajectory, frame, "ee_quat_wxyz", "ee_quat_w"
            ),
            ee_lin_vel_w=_optional_finite_trajectory_frame(
                trajectory, frame, "ee_lin_vel_w", "ee_lin_vel_fd_w"
            ),
            ee_ang_vel_w=_optional_finite_trajectory_frame(trajectory, frame, "ee_ang_vel_w"),
            object_asset_names=object_asset_names,
            bin_name=bin_name,
            device=device,
        )
        if "failure_baseline_recorded" in trajectory.files:
            baseline_recorded = bool(trajectory["failure_baseline_recorded"][frame])
            if baseline_recorded:
                if "object_failure_baseline_pos_w" in trajectory.files:
                    object_baseline = trajectory["object_failure_baseline_pos_w"][frame]
                else:
                    object_baseline = (
                        trajectory["object_pos_w"][frame]
                        - trajectory["object_displacement_from_failure_baseline_w"][frame]
                    )
                if "bin_failure_baseline_pos_w" in trajectory.files:
                    bin_baseline = trajectory["bin_failure_baseline_pos_w"][frame]
                else:
                    bin_baseline = (
                        trajectory["bin_pos_w"][frame]
                        - trajectory["bin_displacement_from_failure_baseline_w"][frame]
                    )
                env._so101_failure_object_pos_w[0] = torch.as_tensor(
                    object_baseline, dtype=torch.float32, device=device
                )
                env._so101_failure_bin_pos_w[0] = torch.as_tensor(
                    bin_baseline, dtype=torch.float32, device=device
                )
                env._so101_failure_baseline_recorded[0] = True
        if hasattr(env, "_so101_policy_control_active"):
            if "policy_control_active" in trajectory.files:
                env._so101_policy_control_active[0] = bool(
                    trajectory["policy_control_active"][frame]
                )
            elif action_phases[frame] in {"dataset", "final_hold"}:
                env._so101_policy_control_active[0] = True
        if not evaluation_applied[frame]:
            telemetry_only_frames_skipped += 1
            continue
        with torch.inference_mode():
            timed_out_tensor = task_time_out(
                env,
                confirm_time_s=timeout_confirm_time_s,
            )
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
        current_postmortem_raw = _single_env_postmortem_raw(
            _postmortem_failure_diagnostics(env)
        )
        task_family = str(env._so101_task_family[0])
        success_counter_attr = {
            "bin": "_so101_bin_success_counter",
            "named_bin": "_so101_bin_success_counter",
            "next_to": "_so101_next_to_success_counter",
            "between": "_so101_between_success_counter",
            "move": "_so101_move_success_counter",
        }.get(task_family)
        success_counter_state = (
            getattr(env, success_counter_attr, None)
            if success_counter_attr is not None
            else None
        )
        success_counter = (
            int(success_counter_state[0].item())
            if isinstance(success_counter_state, torch.Tensor)
            else 0
        )
        required_state = getattr(env, "_so101_success_confirmation_required_steps", None)
        success_required_steps = (
            int(required_state[0].item())
            if isinstance(required_state, torch.Tensor)
            else max(1, math.ceil(float(success_params.get("confirm_time_s", 3.0)) / control_dt))
        )
        failure_reasons = getattr(env, "_so101_failure_reasons", None)
        failure_reason = (
            str(failure_reasons[0])
            if failure_reasons
            else "failure"
            if final_eval.failure
            else "none"
        )
        policy_control_state = getattr(env, "_so101_policy_control_active", None)
        policy_control_active = bool(
            policy_control_state[0].item()
            if isinstance(policy_control_state, torch.Tensor)
            else True
        )
        rescored_rule_trace.append(
            {
                "step": step_index,
                "time_s": step_index * control_dt,
                "success_counter": success_counter,
                "success_required_steps": success_required_steps,
                "live_success_confirmed": bool(success_tensor[0].item()),
                "failure": bool(failure_tensor[0].item()),
                "failure_reason": failure_reason,
                "timed_out": bool(timed_out_tensor[0].item()),
                "terminal_reason": final_eval.reason,
                "policy_control_active": policy_control_active,
                "postmortem_failure_type": _postmortem_failure_type(
                    current_postmortem_raw
                ),
                "postmortem_confidence": (
                    current_postmortem_raw.get("confidence")
                    if isinstance(current_postmortem_raw, dict)
                    else None
                ),
                "postmortem_rationale": (
                    current_postmortem_raw.get("rationale")
                    if isinstance(current_postmortem_raw, dict)
                    else None
                ),
            }
        )
        if final_eval.done and first_terminal is None:
            first_terminal = final_eval
            # Freeze the classifier state at the same point as the collector.
            # Later replay steps may add attempts, transport, drops, or goal
            # history and must not rewrite first-terminal attribution.
            first_terminal_postmortem_raw = current_postmortem_raw

    recomputed_postmortem = _postmortem_failure_diagnostics(env)
    trajectory_fields = list(trajectory.files)
    trajectory.close()

    rescored = dict(record)
    effective_task_family = str(env._so101_task_family[0])
    benchmark_payload = dict(record.get("benchmark") or {})
    original_task_family = benchmark_payload.get("task_family")
    benchmark_payload["task_family"] = effective_task_family
    if canonical_episode is not None:
        benchmark_payload["target_object_id"] = canonical_episode.target_object_id
    rescored["benchmark"] = benchmark_payload
    for scene_key in ("initial_scene", "final_scene"):
        if isinstance(record.get(scene_key), dict):
            scene_payload = dict(record[scene_key])
            scene_payload["task_family"] = effective_task_family
            scene_payload["target_object_id"] = int(env._so101_target_object_ids[0].item())
            rescored[scene_key] = scene_payload
    final_diagnostics = (
        _final_condition_diagnostics(
            env,
            object_asset_names=object_asset_names,
            success_params=final_success_params,
            failure_params=failure_params,
        )
        if final_eval is not None
        else {
            "available": False,
            "reason": "no_scorable_trajectory_frames",
        }
    )
    final_confirmation_waiver_applied = False
    if (
        final_success_confirm_time_disabled
        and record.get("episode_length", {}).get("action_stream_exhausted")
        and final_eval is not None
        and not final_eval.success
    ):
        final_confirmation_waiver_applied = any(
            condition.get("kind") == "success" and bool(condition.get("met"))
            for condition in final_diagnostics.get("conditions", [])
            if isinstance(condition, dict)
        )
        if final_confirmation_waiver_applied:
            final_eval = TermEval(
                step=final_eval.step,
                time_s=final_eval.time_s,
                success=True,
                failure=False,
                timed_out=False,
                reason="success",
            )

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
    rescored["final_diagnostics"] = final_diagnostics
    # Keep the rule-level terminal reason separate from behavioral attribution.
    # The latter is recomputed by benchmark_failure's temporal accumulator and
    # may evolve independently as that classifier becomes more informative.
    rescored["postmortem_failure_diagnostics"] = recomputed_postmortem
    final_postmortem_raw = _single_env_postmortem_raw(recomputed_postmortem)
    final_failure_attribution = _failure_attribution_from_raw(
        final_postmortem_raw,
        label=final_label,
        basis="final_state_standard_confirmation",
    )
    if first_terminal is None:
        first_terminal_failure_attribution = _failure_attribution_from_raw(
            final_postmortem_raw,
            label=first_terminal_label,
            basis="no_first_terminal_before_action_stream_end",
        )
    else:
        first_terminal_failure_attribution = _failure_attribution_from_raw(
            first_terminal_postmortem_raw,
            label=first_terminal_label,
            basis="first_terminal_state",
        )
    failure_attribution = (
        final_failure_attribution
        if label_source == "final"
        else first_terminal_failure_attribution
    )
    rescored["failure_attribution"] = failure_attribution
    rescored["first_terminal_failure_attribution"] = first_terminal_failure_attribution
    rescored["final_failure_attribution"] = final_failure_attribution
    # Keep the convenience alias synchronized with the recomputed attribution;
    # retaining the source value here would silently expose two answers for the
    # same rescored episode.
    rescored["behavioral_outcome"] = failure_attribution["postmortem"]
    rescored_analysis = _rescored_rule_analysis(
        rescored_rule_trace,
        selected_label=label,
        final_label=final_label,
        label_source=label_source,
        final_success_confirm_time_disabled=final_success_confirm_time_disabled,
        final_confirmation_waiver_applied=final_confirmation_waiver_applied,
        control_dt=control_dt,
    )
    rescored["rescored_analysis"] = rescored_analysis

    # Preserve physical extrema and rich object/contact summaries from the
    # source collection, while replacing confirmation conclusions with values
    # recomputed under the effective rescore rules.
    best_achieved = dict(record.get("best_achieved") or {})
    rescored_best = rescored_analysis.get("best_achieved") or {}
    if "success_confirmation" in rescored_best:
        best_achieved["success_confirmation"] = rescored_best["success_confirmation"]
    best_achieved["analysis_basis"] = (
        "source_collection_extrema_and_source_parameter_metrics_plus_rescored_success_confirmation"
    )
    rescored["best_achieved"] = best_achieved
    rescored["closest_miss"] = rescored_analysis.get("closest_miss")

    outcome_quality = dict(record.get("outcome_quality") or {})
    outcome_quality.update(rescored_analysis.get("outcome_quality") or {})
    if (
        bool(outcome_quality.get("final_success"))
        and bool(outcome_quality.get("collateral_event_detected"))
        and outcome_quality.get("success_quality") == "stable_success"
    ):
        outcome_quality["success_quality"] = "success_with_source_observed_collateral"
        outcome_quality["clean_stable_success"] = False
    outcome_quality["analysis_basis"] = (
        "rescored_rule_trace_plus_source_collection_physical_data_quality_and_source_parameter_metrics"
    )
    selected_raw = failure_attribution.get("postmortem_raw")
    outcome_quality["behavioral_attribution_confidence"] = (
        selected_raw.get("confidence") if isinstance(selected_raw, dict) else None
    )
    outcome_quality["behavioral_attribution_version"] = (
        selected_raw.get("classification_version") if isinstance(selected_raw, dict) else None
    )
    outcome_quality["behavioral_attribution_basis"] = failure_attribution.get("basis")
    outcome_quality["behavioral_attribution_applicable"] = bool(
        failure_attribution.get("applicable")
    )
    rescored["outcome_quality"] = outcome_quality
    rescored["analysis_provenance"] = {
        "event_ledger": {
            "basis": "source_collection",
            "note": (
                "physical events remain valid; source rule-transition events may reflect the "
                "collection parameters. Use rescored_analysis.rule_event_ledger for current rules."
            ),
        },
        "best_achieved": {
            "basis": "mixed",
            "source_fields": (
                "collection-time physical extrema, contacts, per-object progress, and "
                "source-parameter-derived goal geometry"
            ),
            "rescored_fields": ["success_confirmation"],
        },
        "closest_miss": {"basis": "rescored_rule_trace"},
        "outcome_quality": {
            "basis": "mixed",
            "rescored_fields": sorted((rescored_analysis.get("outcome_quality") or {}).keys()),
            "remaining_fields": (
                "source collection physical/data-quality metrics plus source-parameter-derived "
                "goal, collateral, contact, and attempt-threshold fields"
            ),
        },
    }
    rescored["label"] = {"source": label_source, **label}
    rescored["rescore"] = {
        "rescored_at": _now_stamp(),
        "schema_version": RESCORE_SCHEMA_VERSION,
        "source_record_schema_version": record.get("schema_version", 1),
        "source_state_schema_version": trajectory_schema_version,
        "trajectory_fields": trajectory_fields,
        "trajectory_frame_count": num_steps,
        "evaluated_frame_count": int(scorable_steps.size),
        "evaluation_mask_source": evaluation_mask_source,
        "telemetry_only_frames_skipped": telemetry_only_frames_skipped,
        "timeout_confirm_time_s": timeout_confirm_time_s,
        "live_success_confirm_time_s": success_params.get("confirm_time_s"),
        "final_success_confirm_time_s": final_success_params.get("confirm_time_s"),
        "final_success_confirm_time_disabled": final_success_confirm_time_disabled,
        "final_confirmation_waiver_applied": final_confirmation_waiver_applied,
        "label_source": label_source,
        "overrides": {kind: dict(items) for kind, items in overrides.items() if items},
        "effective_success_params": _jsonable(success_params),
        "effective_final_success_params": _jsonable(final_success_params),
        "effective_failure_params": _jsonable(failure_params),
        "source_derived_fields": {
            "event_ledger": "source_collection",
            "best_achieved": "mixed; success_confirmation recomputed",
            "closest_miss": "recomputed",
            "outcome_quality": "mixed; rule-derived fields recomputed",
        },
        "original_label": {key: record.get("label", {}).get(key) for key in SUCCESS_LABEL_FIELDS}
        | {"source": label_source},
        "original_task_family": original_task_family,
        "effective_task_family": effective_task_family,
        "original_postmortem_failure_diagnostics": record.get(
            "postmortem_failure_diagnostics"
        ),
        "original_failure_attribution": record.get("failure_attribution"),
        "original_first_terminal_failure_attribution": record.get(
            "first_terminal_failure_attribution"
        ),
        "original_final_failure_attribution": record.get(
            "final_failure_attribution"
        ),
        "original_behavioral_outcome": record.get("behavioral_outcome"),
        "original_best_achieved": record.get("best_achieved"),
        "original_closest_miss": record.get("closest_miss"),
        "original_outcome_quality": record.get("outcome_quality"),
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
            out.write(json.dumps(_jsonable(rescored), separators=(",", ":"), allow_nan=False) + "\n")
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
    postmortem_counts: dict[str, int] = {}
    for entry in summary_records:
        reason = entry["label"]["failure_reason"]
        failure_counts[reason] = failure_counts.get(reason, 0) + 1
        failure_type = _postmortem_failure_type(
            entry.get("failure_attribution", {}).get("postmortem")
        )
        if failure_type is not None:
            postmortem_counts[failure_type] = postmortem_counts.get(failure_type, 0) + 1
    flips = sum(
        1
        for entry in summary_records
        if entry["rescore"]["original_label"]["success"] is not None
        and bool(entry["rescore"]["original_label"]["success"]) != bool(entry["label"]["success"])
    )
    summary = {
        "schema_version": RESCORE_SCHEMA_VERSION,
        "rescored_at": _now_stamp(),
        "source_episodes_path": str(episodes_path),
        "rescored_episodes_path": str(rescored_path),
        "completed_episodes": len(summary_records),
        "successes": successes,
        "failures": failures,
        "success_rate": successes / max(len(summary_records), 1),
        "failure_reason_counts": failure_counts,
        "postmortem_failure_counts": postmortem_counts,
        "label_flips_vs_original": flips,
        "overrides": {kind: dict(items) for kind, items in overrides.items() if items},
    }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(
        f"[INFO]: Rescore summary: success={successes}/{len(summary_records)} "
        f"({100.0 * summary['success_rate']:.1f}%), failures={failures}, "
        f"flips_vs_original={flips}"
    )
    print(f"[INFO]: Wrote {rescored_path} and {summary_path}")


if __name__ == "__main__":
    main()
