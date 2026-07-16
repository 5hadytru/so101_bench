# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay LeRobot SO-101 sim episodes and save reusable success-evaluation artifacts.

This script intentionally disables Isaac Lab's automatic success/failure reset and
evaluates those same termination terms manually. That preserves the terminal scene
state so later success/failure rule revisions can be run against saved states
without replaying the robot through simulation again.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
import inspect
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

from isaaclab.app import AppLauncher


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


parser = argparse.ArgumentParser(
    description=(
        "Replay LeRobot dataset episodes in SO-101 Bench and save per-episode success/failure labels, "
        "initial/final overhead frames, final scene state, and compact trajectories for offline relabeling."
    )
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Number of Isaac Lab environments to replay in parallel. Defaults to the task config.",
)
parser.add_argument("--task", type=str, default="So101Bench-Bin-v0", help="Isaac Lab task name.")
parser.add_argument("--seed", type=int, default=1984, help="Environment seed.")
parser.add_argument(
    "--episodes_jsonl",
    type=Path,
    default=Path("tasks/teleop_1.jsonl"),
    help="Benchmark episode JSONL matching the original teleop/eval run.",
)
parser.add_argument(
    "--episode_layouts_jsonl",
    "--layouts_jsonl",
    type=Path,
    default=Path("tasks/layouts/teleop_1_layouts.jsonl"),
    help=(
        "JSONL file with object and bin poses from the original run. Rows are matched by trial_id when present; "
        "otherwise by benchmark row index."
    ),
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="5hadytru/so101_bench_sim_1_v3.0",
    help="LeRobot dataset repo id. A local --repo_root is used when provided.",
)
parser.add_argument(
    "--repo_root",
    type=Path,
    default=Path("data/lerobot/so101_bench_sim_1_v3.0"),
    help="Local root directory for the LeRobot dataset.",
)
parser.add_argument(
    "--dataset_episode_index",
    "--episode",
    type=int,
    default=0,
    help="First LeRobot dataset episode index to replay.",
)
parser.add_argument(
    "--benchmark_episode_index",
    type=int,
    default=None,
    help=(
        "First benchmark JSONL/layout row to reset before replay. Defaults to --dataset_episode_index, matching "
        "sequential teleop recordings with no skipped/cancelled episodes."
    ),
)
parser.add_argument(
    "--benchmark_episode_indices",
    type=str,
    default=None,
    help=(
        "Comma-separated benchmark JSONL/layout rows to use for each replayed dataset episode. "
        "Overrides --benchmark_episode_index and --num_episodes."
    ),
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=None,
    help="Number of consecutive LeRobot dataset episodes to replay. Defaults to all available selected episodes.",
)
parser.add_argument(
    "--output_dir",
    type=Path,
    default=None,
    help="Directory for episodes.jsonl, summary.json, frames, and state arrays.",
)
parser.add_argument(
    "--frame_source",
    choices=("dataset", "sim", "none"),
    default="none",
    help=(
        "Where overhead initial/final frames are saved from. 'dataset' reads the recorded LeRobot overhead video "
        "and avoids Isaac camera sensors; 'sim' enables Isaac cameras and renders replay frames."
    ),
)
parser.add_argument("--overwrite", action="store_true", default=False, help="Allow writing into an existing output dir.")
parser.add_argument(
    "--initial_hold_time_s",
    type=float,
    default=0.5,
    help="Seconds to hold the initial sim joint pose before replaying the first recorded action.",
)
parser.add_argument(
    "--hold_last_action_time_s",
    type=float,
    default=0.0,
    help="Seconds to hold the final recorded action after the action stream is exhausted.",
)
parser.add_argument(
    "--no_success_confirm_time",
    action="store_true",
    default=False,
    help=(
        "When the recorded action stream ends, score the final scene state with no success confirmation window. "
        "This keeps short teleop demos from being marked failed only because the success pose did not persist "
        "for the usual confirm_time_s after the demo ended."
    ),
)
parser.add_argument(
    "--stop_on_done",
    action="store_true",
    default=False,
    help=(
        "Stop replay when the current success/failure/timeout logic first fires. By default all recorded actions "
        "are played so the saved final state matches the dataset episode end."
    ),
)
parser.add_argument(
    "--label_source",
    choices=("final", "first_terminal"),
    default="final",
    help="Which current evaluation to expose as the top-level label in episodes.jsonl.",
)
parser.add_argument(
    "--save_trajectory",
    nargs="?",
    const=True,
    default=True,
    type=_str_to_bool,
    help="Save compact per-step state arrays. Accepts '--save_trajectory false'.",
)
parser.add_argument(
    "--trajectory_stride",
    type=int,
    default=1,
    help="Save every Nth state sample in the trajectory NPZ. Use 1 for full offline relabeling fidelity.",
)
parser.add_argument(
    "--render_warmup_frames",
    type=int,
    default=16,
    help=(
        "Number of RTX render() passes to accumulate before saving an overhead frame. The 'quality' "
        "renderer denoises image-based DomeLight sampling over several frames, so a single render after a "
        "scene reset leaves the frame under-converged (dark) once the tiled multi-env render target grows. "
        "Set to 0 to capture after a single render."
    ),
)
parser.add_argument(
    "--real_time",
    nargs="?",
    const=True,
    default=False,
    type=_str_to_bool,
    help="Throttle replay to wall-clock time. Accepts either '--real_time' or '--real_time false'.",
)
parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help="Wall-clock replay speed multiplier used with --real_time. 1.0 means dataset/env time.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Isaac camera rendering is only needed when replay frames come from sim. For
# --frame_source dataset/none, _make_env() nulls out the camera sensors and visual
# observations, so skip the RTX renderer entirely: faster startup and no per-step render.
args_cli.enable_cameras = args_cli.frame_source == "sim"

# Workaround: the headless camera kit (isaaclab.python.headless.rendering.kit) fails to
# produce the LdrColorSD render var on this setup, crashing TiledCamera annotator.attach()
# with "Unable to write from unknown dtype, kind=f, size=0". The GUI rendering kit renders
# the same cameras fine, so force it even under --headless unless the user overrides
# --experience explicitly. (AppLauncher resolves the bare name against IsaacLab's apps/ dir.)
if args_cli.enable_cameras and args_cli.headless and not getattr(args_cli, "experience", ""):
    args_cli.experience = "isaaclab.python.rendering.kit"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg

import so101_bench.tasks  # noqa: F401
from so101_bench.benchmark import BenchmarkEpisodeSpec, load_episode_jsonl
from so101_bench.layouts import normalize_layout_object_slots
from so101_bench.mdp import (
    benchmark_failure,
    benchmark_object_positions,
    benchmark_object_yaws,
    mark_benchmark_robot_start,
    task_time_out,
    task_success,
    task_condition_diagnostics,
    grasped_object_made_contact,
)
from so101_bench.tasks.direct.so101_bench.so101_bench_env_cfg import (
    configure_env_cfg_for_object_pool,
)
from so101_bench.utils.lerobot_calibration import (
    LEROBOT_INITIAL_JOINT_POS,
    LEROBOT_JOINT_FEATURE_ORDER,
    LEROBOT_JOINT_ORDER,
    REAL_SO101_CALIBRATION,
    SIM_LIMIT_MARGIN_DEG,
    STS3215_CENTER_POSITION,
    STS3215_DEGREES_PER_TICK,
    USD_SIM_JOINT_LIMITS_DEG,
    lerobot_position_bounds,
    lerobot_pose_to_sim_joint_pos,
)


ACTION = "action"
ACTION_JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
INITIAL_ROBOT_JOINT_POS = lerobot_pose_to_sim_joint_pos(LEROBOT_INITIAL_JOINT_POS)
BIN_NAME = "plastic_bin"
SCHEMA_VERSION = 1


class SO101ReplayActionMapper:
    """Convert calibrated LeRobot `.pos` actions into SO-101 USD joint radians."""

    def __init__(self, device: str):
        self.device = device
        self.joint_names = LEROBOT_JOINT_ORDER
        self.lerobot_mins = torch.tensor(
            [lerobot_position_bounds(name)[0] for name in self.joint_names],
            dtype=torch.float32,
            device=self.device,
        )
        self.lerobot_maxs = torch.tensor(
            [lerobot_position_bounds(name)[1] for name in self.joint_names],
            dtype=torch.float32,
            device=self.device,
        )
        self.calibration_mins = torch.tensor(
            [REAL_SO101_CALIBRATION[name].range_min for name in self.joint_names],
            dtype=torch.float32,
            device=self.device,
        )
        self.calibration_maxs = torch.tensor(
            [REAL_SO101_CALIBRATION[name].range_max for name in self.joint_names],
            dtype=torch.float32,
            device=self.device,
        )
        self.usd_mins_deg = torch.tensor(
            [USD_SIM_JOINT_LIMITS_DEG[name][0] for name in self.joint_names],
            dtype=torch.float32,
            device=self.device,
        )
        self.usd_maxs_deg = torch.tensor(
            [USD_SIM_JOINT_LIMITS_DEG[name][1] for name in self.joint_names],
            dtype=torch.float32,
            device=self.device,
        )
        self.is_gripper = torch.tensor([name == "gripper" for name in self.joint_names], device=self.device)

    def clamp_lerobot_positions(self, values: torch.Tensor) -> torch.Tensor:
        return torch.minimum(torch.maximum(values, self.lerobot_mins), self.lerobot_maxs)

    def sim_radians_to_lerobot_positions(self, sim_values: torch.Tensor) -> torch.Tensor:
        mapped_deg = sim_values * 180.0 / torch.pi
        mapped_deg = torch.minimum(torch.maximum(mapped_deg, self.usd_mins_deg), self.usd_maxs_deg)

        motor_positions = mapped_deg / STS3215_DEGREES_PER_TICK + STS3215_CENTER_POSITION
        body_normalized = (motor_positions - self.calibration_mins) / (
            self.calibration_maxs - self.calibration_mins
        )
        body_positions = body_normalized * 200.0 - 100.0

        gripper_normalized = (mapped_deg - self.usd_mins_deg) / (self.usd_maxs_deg - self.usd_mins_deg)
        gripper_positions = gripper_normalized * 100.0

        lerobot_positions = torch.where(self.is_gripper, gripper_positions, body_positions)
        return self.clamp_lerobot_positions(lerobot_positions)

    def lerobot_positions_to_sim_radians(self, lerobot_positions: torch.Tensor) -> torch.Tensor:
        bounded_positions = self.clamp_lerobot_positions(lerobot_positions)
        body_normalized = (bounded_positions + 100.0) / 200.0
        gripper_normalized = bounded_positions / 100.0

        motor_positions = body_normalized * (self.calibration_maxs - self.calibration_mins) + self.calibration_mins
        body_degrees = (motor_positions - STS3215_CENTER_POSITION) * STS3215_DEGREES_PER_TICK
        gripper_degrees = self.usd_mins_deg + gripper_normalized * (self.usd_maxs_deg - self.usd_mins_deg)

        mapped_deg = torch.where(self.is_gripper, gripper_degrees, body_degrees)
        mapped_deg = torch.minimum(
            torch.maximum(mapped_deg, self.usd_mins_deg + SIM_LIMIT_MARGIN_DEG),
            self.usd_maxs_deg - SIM_LIMIT_MARGIN_DEG,
        )
        return mapped_deg * torch.pi / 180.0


@dataclass(frozen=True)
class LeRobotActionEpisode:
    episode_index: int
    fps: float
    action_names: tuple[str, ...]
    actions: torch.Tensor

    @property
    def num_frames(self) -> int:
        return int(self.actions.shape[0])


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


@dataclass(frozen=True)
class DatasetVideoSpan:
    video_path: Path
    from_timestamp: float
    to_timestamp: float


@dataclass
class ReplayLane:
    env_id: int
    offset: int
    episode: BenchmarkEpisodeSpec
    benchmark_index: int
    episode_layout: dict | None
    dataset_episode_index: int
    action_episode: LeRobotActionEpisode
    setup: dict[str, Any]
    initial_scene: dict[str, Any]
    initial_frame_path: Path | None
    final_frame_path: Path | None
    last_action_lerobot: torch.Tensor
    last_action_sim: torch.Tensor
    step: int = 0
    frame_index: int = 0
    robot_control_started: bool = False
    first_terminal: TermEval | None = None
    final_eval: TermEval | None = None
    trajectory_samples: list[dict[str, Any]] = field(default_factory=list)
    action_stream_exhausted: bool = False


def _now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _json_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def _canonical_action_name(name: str) -> str:
    name = str(name)
    if name.endswith(".pos"):
        return name
    if name in LEROBOT_JOINT_ORDER:
        return f"{name}.pos"
    return name


def _coerce_action_feature_names(raw_names: Any) -> list[str]:
    if raw_names is None:
        return []
    if isinstance(raw_names, dict):
        raw_names = raw_names.get("names") or raw_names.get("action") or raw_names.values()
    if isinstance(raw_names, (list, tuple)):
        names: list[str] = []
        for entry in raw_names:
            if isinstance(entry, (list, tuple)):
                names.extend(str(value) for value in entry)
            else:
                names.append(str(entry))
        return names
    return []


def _dataset_fps(dataset) -> float:
    fps = getattr(dataset, "fps", None)
    if fps is None:
        meta = getattr(dataset, "meta", None)
        fps = getattr(meta, "fps", None)
    if fps is None:
        return 30.0
    return float(fps)


def _open_lerobot_dataset(repo_id: str, root: Path | None, episode_index: int):
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot is required for replay. Install it in the Isaac Lab Python environment, "
                "then rerun this script."
            ) from exc

    dataset_kwargs: dict[str, Any] = {}
    signature = inspect.signature(LeRobotDataset)
    if "root" in signature.parameters:
        dataset_kwargs["root"] = root
    if "episodes" in signature.parameters:
        dataset_kwargs["episodes"] = [episode_index]
    if "download_videos" in signature.parameters:
        dataset_kwargs["download_videos"] = False
    return LeRobotDataset(repo_id, **dataset_kwargs)


def _raw_action_to_tensor(
    raw_action: Any,
    source_names: list[str],
    *,
    device: str,
    episode_index: int,
    frame_index: int,
) -> torch.Tensor:
    if isinstance(raw_action, dict):
        raw_names = list(raw_action.keys())
        raw_values = np.asarray([raw_action[name] for name in raw_names], dtype=np.float32).reshape(-1)
        source_names = raw_names
    elif isinstance(raw_action, torch.Tensor):
        raw_values = raw_action.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    else:
        raw_values = np.asarray(raw_action, dtype=np.float32).reshape(-1)

    if len(source_names) != len(raw_values):
        if len(raw_values) == len(LEROBOT_JOINT_FEATURE_ORDER):
            source_names = list(LEROBOT_JOINT_FEATURE_ORDER)
        else:
            raise ValueError(
                f"Dataset episode {episode_index} frame {frame_index} has action shape {raw_values.shape}, "
                f"but action feature names are {source_names!r}."
            )

    index_by_name = {_canonical_action_name(name): index for index, name in enumerate(source_names)}
    missing = [name for name in LEROBOT_JOINT_FEATURE_ORDER if name not in index_by_name]
    if missing:
        raise ValueError(
            f"Dataset episode {episode_index} action names are missing {missing}. "
            f"Found {source_names!r}."
        )

    ordered = [float(raw_values[index_by_name[name]]) for name in LEROBOT_JOINT_FEATURE_ORDER]
    return torch.tensor(ordered, dtype=torch.float32, device=device)


def _load_lerobot_action_episode(
    *,
    repo_id: str,
    root: Path | None,
    episode_index: int,
    device: str,
) -> LeRobotActionEpisode:
    dataset = _open_lerobot_dataset(repo_id, root, episode_index)
    features = getattr(dataset, "features", {})
    if ACTION not in features:
        raise ValueError(f"LeRobot dataset has no {ACTION!r} feature. Found features: {list(features)}")

    feature_names = _coerce_action_feature_names(features[ACTION].get("names"))
    if hasattr(dataset, "select_columns"):
        action_rows = dataset.select_columns(ACTION)
    else:
        action_rows = getattr(dataset, "hf_dataset").select_columns(ACTION)

    num_frames = int(getattr(dataset, "num_frames", len(action_rows)))
    if num_frames <= 0:
        raise ValueError(f"LeRobot dataset episode {episode_index} has no frames.")

    actions = []
    for frame_index in range(num_frames):
        row = action_rows[frame_index]
        actions.append(
            _raw_action_to_tensor(
                row[ACTION],
                feature_names,
                device=device,
                episode_index=episode_index,
                frame_index=frame_index,
            )
        )

    return LeRobotActionEpisode(
        episode_index=episode_index,
        fps=_dataset_fps(dataset),
        action_names=tuple(LEROBOT_JOINT_FEATURE_ORDER),
        actions=torch.stack(actions, dim=0),
    )


def _dataset_total_episodes(root: Path | None) -> int | None:
    if root is None:
        return None
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return None
    try:
        return int(json.loads(info_path.read_text(encoding="utf-8"))["total_episodes"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _load_dataset_video_spans(
    root: Path | None,
    video_key: str = "observation.images.overhead",
) -> dict[int, DatasetVideoSpan]:
    if root is None:
        raise ValueError("--frame_source dataset requires --repo_root.")
    meta_root = root / "meta" / "episodes"
    if not meta_root.exists():
        raise FileNotFoundError(f"LeRobot episode metadata directory does not exist: {meta_root}")

    columns = [
        "episode_index",
        f"videos/{video_key}/chunk_index",
        f"videos/{video_key}/file_index",
        f"videos/{video_key}/from_timestamp",
        f"videos/{video_key}/to_timestamp",
    ]
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("--frame_source dataset requires pyarrow in this Python environment.") from exc

    spans: dict[int, DatasetVideoSpan] = {}
    parquet_paths = sorted(meta_root.glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No episode metadata parquet files found under {meta_root}")

    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path, columns=columns)
        data = table.to_pydict()
        for row_id, episode_index in enumerate(data["episode_index"]):
            chunk_index = int(data[f"videos/{video_key}/chunk_index"][row_id])
            file_index = int(data[f"videos/{video_key}/file_index"][row_id])
            video_path = root / "videos" / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
            spans[int(episode_index)] = DatasetVideoSpan(
                video_path=video_path,
                from_timestamp=float(data[f"videos/{video_key}/from_timestamp"][row_id]),
                to_timestamp=float(data[f"videos/{video_key}/to_timestamp"][row_id]),
            )

    return spans


def _serialize_param_value(value: Any) -> Any:
    """Convert a termination-term parameter value to a JSON-friendly representation."""
    if isinstance(value, SceneEntityCfg):
        payload: dict[str, Any] = {"__scene_entity_cfg__": True, "name": value.name}
        joint_names = getattr(value, "joint_names", None)
        body_names = getattr(value, "body_names", None)
        if joint_names is not None:
            payload["joint_names"] = list(joint_names) if not isinstance(joint_names, str) else joint_names
        if body_names is not None:
            payload["body_names"] = list(body_names) if not isinstance(body_names, str) else body_names
        return payload
    if isinstance(value, dict):
        return {str(key): _serialize_param_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serialize_param_value(item) for item in value]
    if isinstance(value, list):
        return [_serialize_param_value(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _serialize_term_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: _serialize_param_value(value) for key, value in params.items()}


def _capture_eval_setup(
    env,
    *,
    env_id: int = 0,
    control_dt: float,
    physics_dt: float,
    success_params: dict[str, Any],
    failure_params: dict[str, Any],
) -> dict[str, Any]:
    """Capture per-run scalars/arrays the rescorer needs to rebuild an env stub."""
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    action_joint_ids = [robot.joint_names.index(joint_name) for joint_name in ACTION_JOINT_NAMES]
    joint_pos_limits = robot.data.joint_pos_limits[env_id, action_joint_ids].detach().cpu().tolist()
    env_origins = unwrapped.scene.env_origins[env_id].detach().cpu().tolist()
    decimation = int(getattr(unwrapped.cfg, "decimation", max(1, round(control_dt / max(physics_dt, 1.0e-9)))))
    return {
        "control_dt": float(control_dt),
        "physics_dt": float(physics_dt),
        "decimation": decimation,
        "bin_name": BIN_NAME,
        "action_joint_names": list(ACTION_JOINT_NAMES),
        "jaw_action_index": ACTION_JOINT_NAMES.index("Jaw"),
        "action_joint_pos_limits": joint_pos_limits,
        "env_origins": list(env_origins),
        "success_params": _serialize_term_params(success_params),
        "failure_params": _serialize_term_params(failure_params),
        "final_success_confirm_time_disabled": bool(
            args_cli.no_success_confirm_time and success_params.get("confirm_time_s") == 0.0
        ),
    }


def _load_dataset_episode_instructions(root: Path | None) -> dict[int, str]:
    """Return ``dataset_episode_index -> first task instruction`` for every recorded episode."""
    if root is None:
        raise ValueError("Dataset verification requires --repo_root.")
    meta_root = root / "meta" / "episodes"
    if not meta_root.exists():
        raise FileNotFoundError(f"LeRobot episode metadata directory does not exist: {meta_root}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Dataset verification requires pyarrow in this Python environment.") from exc
    parquet_paths = sorted(meta_root.glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No episode metadata parquet files found under {meta_root}")

    instructions: dict[int, str] = {}
    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path, columns=["episode_index", "tasks"])
        data = table.to_pydict()
        for episode_index, tasks in zip(data["episode_index"], data["tasks"]):
            entries = list(tasks) if tasks is not None else []
            instructions[int(episode_index)] = str(entries[0]) if entries else ""
    return instructions


def _verify_jsonl_matches_dataset(
    *,
    episode_specs: list[BenchmarkEpisodeSpec],
    episode_plan: list[BenchmarkEpisodeSpec],
    benchmark_indices: list[int],
    dataset_episode_indices: list[int],
    dataset_instructions: dict[int, str],
) -> None:
    """Assert teleop JSONL and the LeRobot dataset agree on episode count and per-episode instructions.

    Checked invariants:
      1. ``len(episode_specs)`` equals the dataset's total episode count.
      2. For every planned (benchmark row, dataset episode) pair, instruction text matches exactly.
    """
    total_teleop = len(episode_specs)
    total_dataset = len(dataset_instructions)
    if total_teleop != total_dataset:
        raise ValueError(
            f"JSONL/dataset episode count mismatch: teleop has {total_teleop} row(s) but dataset has "
            f"{total_dataset} episode(s). Delete/duplicate teleop rows so they align with the dataset, "
            "or pass --benchmark_episode_indices explicitly to override."
        )

    mismatches: list[tuple[int, int, str, str]] = []
    for episode, benchmark_idx, dataset_ep in zip(
        episode_plan, benchmark_indices, dataset_episode_indices, strict=True
    ):
        dataset_instruction = dataset_instructions.get(dataset_ep)
        if dataset_instruction is None:
            mismatches.append((benchmark_idx, dataset_ep, episode.instruction, "<missing in dataset>"))
        elif dataset_instruction != episode.instruction:
            mismatches.append((benchmark_idx, dataset_ep, episode.instruction, dataset_instruction))

    if mismatches:
        lines = [f"Found {len(mismatches)} instruction mismatch(es) between teleop JSONL and dataset:"]
        for benchmark_idx, dataset_ep, teleop_instruction, dataset_instruction in mismatches[:20]:
            lines.append(
                f"  teleop row {benchmark_idx} <-> dataset ep {dataset_ep}: "
                f"teleop={teleop_instruction!r} dataset={dataset_instruction!r}"
            )
        if len(mismatches) > 20:
            lines.append(f"  ... and {len(mismatches) - 20} more.")
        raise ValueError("\n".join(lines))

    print(
        f"[INFO]: Verified teleop JSONL is consistent with dataset: {total_teleop} rows == "
        f"{total_dataset} episodes; {len(episode_plan)} planned row(s) have matching instructions."
    )


def _write_video_frame(path: Path, video_path: Path, timestamp_s: float) -> Path:
    if not video_path.exists():
        raise FileNotFoundError(f"Dataset video does not exist: {video_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp_s = max(float(timestamp_s), 0.0)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{timestamp_s:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(path),
    ]
    subprocess.run(command, check=True)
    return path


def _write_dataset_overhead_frames(
    *,
    output_dir: Path,
    dataset_episode_index: int,
    action_fps: float,
    video_spans: dict[int, DatasetVideoSpan],
) -> tuple[Path, Path]:
    try:
        span = video_spans[dataset_episode_index]
    except KeyError as exc:
        raise KeyError(f"No overhead video metadata found for dataset episode {dataset_episode_index}.") from exc

    initial_path = output_dir / "frames" / f"episode_{dataset_episode_index:06d}_overhead_initial.png"
    final_path = output_dir / "frames" / f"episode_{dataset_episode_index:06d}_overhead_final.png"
    frame_dt = 1.0 / max(action_fps, 1.0e-6)
    final_timestamp = max(span.from_timestamp, span.to_timestamp - frame_dt)
    _write_video_frame(initial_path, span.video_path, span.from_timestamp)
    _write_video_frame(final_path, span.video_path, final_timestamp)
    return initial_path, final_path


def _episode_trial_id(episode: BenchmarkEpisodeSpec, episode_index: int) -> object:
    metadata = episode.metadata or {}
    return metadata.get("trial_id", episode_index)


def _trial_id_key(trial_id: object) -> str:
    return str(trial_id)


def _load_layout_jsonl(path: Path) -> list[dict]:
    layouts = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            layout = json.loads(line)
            if not isinstance(layout, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object per line.")
            layouts.append(layout)
    if not layouts:
        raise ValueError(f"No layout rows found in {path}.")
    return layouts


def _load_episode_layouts(
    episode_plan: list[BenchmarkEpisodeSpec],
    benchmark_indices: list[int],
    layout_path: Path | None,
) -> list[dict | None]:
    if layout_path is None:
        print("[WARN]: No --episode_layouts_jsonl provided; initial scene will be sampled by the env reset.")
        return [None for _ in episode_plan]
    if not layout_path.exists():
        raise FileNotFoundError(f"Episode layout JSONL does not exist: {layout_path}")

    available_layouts = _load_layout_jsonl(layout_path)
    requested_trial_ids = [
        _episode_trial_id(episode, benchmark_index)
        for episode, benchmark_index in zip(episode_plan, benchmark_indices, strict=True)
    ]
    layouts_with_trial_ids = [layout for layout in available_layouts if "trial_id" in layout]

    if layouts_with_trial_ids:
        layouts_by_trial_id = {}
        for layout in layouts_with_trial_ids:
            trial_id = layout["trial_id"]
            trial_id_key = _trial_id_key(trial_id)
            if trial_id_key in layouts_by_trial_id:
                raise ValueError(f"{layout_path} contains duplicate layout rows for trial_id={trial_id!r}.")
            layouts_by_trial_id[trial_id_key] = layout
        missing_trial_ids = [
            trial_id for trial_id in requested_trial_ids if _trial_id_key(trial_id) not in layouts_by_trial_id
        ]
        if missing_trial_ids:
            raise ValueError(f"{layout_path} is missing layout rows for trial_id(s): {missing_trial_ids}.")
        episode_layouts = [layouts_by_trial_id[_trial_id_key(trial_id)] for trial_id in requested_trial_ids]
    else:
        max_index = max(benchmark_indices)
        if len(available_layouts) <= max_index:
            raise ValueError(
                f"{layout_path} contains {len(available_layouts)} layout row(s), "
                f"but benchmark index {max_index} was requested."
            )
        episode_layouts = [available_layouts[index] for index in benchmark_indices]

    normalized_layouts = []
    for episode, benchmark_index, layout in zip(episode_plan, benchmark_indices, episode_layouts, strict=True):
        normalized_layouts.append(
            normalize_layout_object_slots(layout, episode.objects, episode_index=benchmark_index)
        )
    print(f"[INFO]: Loaded provided initial layouts for {len(normalized_layouts)} episode(s): {layout_path}")
    return normalized_layouts


def _episode_object_pool(episode_plan: list[BenchmarkEpisodeSpec]) -> list[str]:
    object_pool = []
    seen = set()
    for episode in episode_plan:
        for object_name in episode.objects:
            if object_name in seen:
                continue
            seen.add(object_name)
            object_pool.append(object_name)
    return object_pool


def _episode_pool_payload(episode: BenchmarkEpisodeSpec, pool_index_by_name: dict[str, int]) -> dict[str, Any]:
    payload = episode.reset_payload()
    local_to_pool = [pool_index_by_name[object_name] for object_name in episode.objects]
    payload["active_object_ids"] = local_to_pool
    payload["target_object_id"] = local_to_pool[episode.target_object_id]
    payload["referent_object_ids"] = [local_to_pool[object_id] for object_id in episode.referent_object_ids]
    return payload


def _episode_pool_layout(
    episode: BenchmarkEpisodeSpec,
    episode_layout: dict | None,
    pool_index_by_name: dict[str, int],
) -> dict | None:
    if episode_layout is None:
        return None

    remapped_layout = dict(episode_layout)
    remapped_objects = []
    for entry in episode_layout.get("objects", []):
        remapped_entry = dict(entry)
        local_slot = int(remapped_entry["slot"])
        object_name = str(remapped_entry.get("name") or episode.objects[local_slot])
        pool_slot = pool_index_by_name[object_name]
        remapped_entry["slot"] = pool_slot
        remapped_entry["asset_name"] = f"object_{pool_slot + 1}"
        remapped_objects.append(remapped_entry)
    remapped_layout["objects"] = remapped_objects
    return remapped_layout


def _episode_reset_params(
    episode: BenchmarkEpisodeSpec,
    episode_layout: dict | None,
    object_pool: list[str],
    object_asset_names: list[str],
) -> dict[str, Any]:
    pool_index_by_name = {object_name: object_id for object_id, object_name in enumerate(object_pool)}
    payload = _episode_pool_payload(episode, pool_index_by_name)
    return {
        "object_asset_names": object_asset_names,
        "object_labels": object_pool,
        "task_family": episode.task_family,
        "object_count_range": (len(episode.objects), len(episode.objects)),
        "active_object_selection": "fixed",
        "fixed_active_object_ids": tuple(payload["active_object_ids"]),
        "shuffle_object_labels": False,
        "force_bin_all_objects_instruction": False,
        "episode_spec": payload,
        "episode_layout": _episode_pool_layout(episode, episode_layout, pool_index_by_name),
    }


def _configure_env_for_episode(
    env,
    episode: BenchmarkEpisodeSpec,
    episode_layout: dict | None,
    object_pool: list[str],
    object_asset_names: list[str],
) -> None:
    params = _episode_reset_params(episode, episode_layout, object_pool, object_asset_names)
    env.unwrapped.cfg.events.reset_benchmark_scene.params.update(params)
    env.unwrapped.event_manager.get_term_cfg("reset_benchmark_scene").params.update(params)


def _make_env(
    object_pool: list[str],
    first_episode: BenchmarkEpisodeSpec,
    first_episode_layout: dict | None,
) -> tuple[gym.Env, list[str], dict[str, Any], dict[str, Any]]:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.scene.robot.init_state.joint_pos = dict(INITIAL_ROBOT_JOINT_POS)
    if args_cli.frame_source != "sim":
        env_cfg.scene.camera_wrist = None
        env_cfg.scene.camera_overhead = None
        env_cfg.observations.visual = None
    object_asset_names = configure_env_cfg_for_object_pool(env_cfg, object_pool)
    env_cfg.events.reset_benchmark_scene.params.update(
        _episode_reset_params(first_episode, first_episode_layout, object_pool, object_asset_names)
    )
    success_params = dict(env_cfg.terminations.success.params)
    failure_params = dict(env_cfg.terminations.failure.params)
    env_cfg.terminations.success = None
    env_cfg.terminations.failure = None
    env_cfg.terminations.time_out = None
    print("[INFO]: Env auto-reset disabled for success, failure, and timeout; terms are evaluated manually.")
    return gym.make(args_cli.task, cfg=env_cfg), object_asset_names, success_params, failure_params


def _initial_robot_action(env) -> torch.Tensor:
    return torch.tensor(
        [INITIAL_ROBOT_JOINT_POS[joint_name] for joint_name in ACTION_JOINT_NAMES],
        dtype=torch.float32,
        device=env.unwrapped.device,
    )


def _env_ids_tensor(env, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device)
    return env_ids.to(dtype=torch.long, device=env.unwrapped.device)


def _restore_robot_initial_pose(env, env_ids: torch.Tensor | None = None) -> None:
    env_ids = _env_ids_tensor(env, env_ids)
    robot = env.unwrapped.scene["robot"]
    joint_ids = [robot.joint_names.index(joint_name) for joint_name in ACTION_JOINT_NAMES]
    joint_pos = _initial_robot_action(env).unsqueeze(0).repeat(len(env_ids), 1)
    joint_vel = torch.zeros_like(joint_pos)
    robot.data.default_joint_pos[env_ids.unsqueeze(1), joint_ids] = joint_pos
    robot.data.default_joint_vel[env_ids.unsqueeze(1), joint_ids] = joint_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, joint_ids=joint_ids, env_ids=env_ids)
    robot.write_data_to_sim()
    grasp_arm_jaw_pos = getattr(env.unwrapped, "_so101_grasp_arm_jaw_pos", None)
    if isinstance(grasp_arm_jaw_pos, torch.Tensor):
        grasp_arm_jaw_pos[env_ids] = joint_pos[:, ACTION_JOINT_NAMES.index("Jaw")]


def _reset_env(env, env_ids: torch.Tensor | None = None) -> tuple[dict, dict]:
    reset_all_envs = env_ids is None
    env_ids = _env_ids_tensor(env, env_ids)
    with torch.inference_mode():
        if reset_all_envs:
            obs, info = env.reset()
        else:
            obs, info = env.unwrapped.reset(env_ids=env_ids)
        _restore_robot_initial_pose(env, env_ids)
        unwrapped = env.unwrapped
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.forward()
        num_rerenders = getattr(unwrapped.cfg, "num_rerenders_on_reset", 0)
        if unwrapped.sim.has_rtx_sensors() and num_rerenders > 0:
            for _ in range(num_rerenders):
                unwrapped.sim.render()
        obs = unwrapped.observation_manager.compute(update_history=True)
        unwrapped.obs_buf = obs
    return obs, info


def _begin_robot_control(env, object_asset_names: list[str], env_ids: torch.Tensor | None = None) -> None:
    mark_benchmark_robot_start(
        env.unwrapped,
        object_asset_names=object_asset_names,
        bin_name=BIN_NAME,
        env_ids=env_ids,
        force_robot_start_time=True,
    )


def _quat_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _tensor_list(value: torch.Tensor | np.ndarray | list | tuple | float | int | bool) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _active_mask(env, object_asset_names: list[str]) -> torch.Tensor:
    return getattr(
        env.unwrapped,
        "_so101_active_object_mask",
        torch.ones((env.unwrapped.num_envs, len(object_asset_names)), dtype=torch.bool, device=env.unwrapped.device),
    )


def _scene_state(env, object_asset_names: list[str], object_labels: list[str], *, env_id: int = 0) -> dict[str, Any]:
    unwrapped = env.unwrapped
    object_pos = benchmark_object_positions(unwrapped, object_asset_names)[env_id]
    object_yaw = benchmark_object_yaws(unwrapped, object_asset_names)[env_id]
    active = _active_mask(env, object_asset_names)[env_id]
    target_id = int(getattr(unwrapped, "_so101_target_object_ids", torch.zeros(1))[env_id].item())
    referent_ids = _tensor_list(
        getattr(unwrapped, "_so101_referent_object_ids", torch.zeros((1, 2), dtype=torch.long))[env_id]
    )
    direction_id = int(getattr(unwrapped, "_so101_direction_ids", torch.zeros(1, dtype=torch.long))[env_id].item())

    initial_object_pos = getattr(unwrapped, "_so101_initial_object_pos_w", None)
    failure_object_pos = getattr(unwrapped, "_so101_failure_object_pos_w", None)
    object_half_extents = getattr(unwrapped, "_so101_object_half_extents", None)
    object_footprint_half_extents = getattr(unwrapped, "_so101_object_footprint_half_extents", None)
    object_footprint_center_offsets = getattr(unwrapped, "_so101_object_footprint_center_offsets", None)

    objects = []
    for object_id, asset_name in enumerate(object_asset_names):
        objects.append(
            {
                "slot": object_id,
                "asset_name": asset_name,
                "label": object_labels[object_id] if object_id < len(object_labels) else asset_name,
                "active": bool(active[object_id].item()),
                "is_target": object_id == target_id,
                "is_referent": object_id in referent_ids,
                "position": _tensor_list(object_pos[object_id]),
                "yaw": float(object_yaw[object_id].item()),
                "initial_position": (
                    _tensor_list(initial_object_pos[env_id, object_id]) if initial_object_pos is not None else None
                ),
                "failure_baseline_position": (
                    _tensor_list(failure_object_pos[env_id, object_id]) if failure_object_pos is not None else None
                ),
                "half_extents": (
                    _tensor_list(object_half_extents[env_id, object_id]) if object_half_extents is not None else None
                ),
                "footprint_half_extents": (
                    _tensor_list(object_footprint_half_extents[env_id, object_id])
                    if object_footprint_half_extents is not None
                    else None
                ),
                "footprint_center_offset": (
                    _tensor_list(object_footprint_center_offsets[env_id, object_id])
                    if object_footprint_center_offsets is not None
                    else None
                ),
            }
        )

    bin_asset = unwrapped.scene[BIN_NAME]
    bin_pos = bin_asset.data.root_pos_w[env_id]
    bin_quat = bin_asset.data.root_quat_w[env_id]
    bin_half_extents = getattr(unwrapped, "_so101_bin_half_extents", None)
    bin_footprint_half_extents = getattr(unwrapped, "_so101_bin_footprint_half_extents", None)
    bin_footprint_center_offsets = getattr(unwrapped, "_so101_bin_footprint_center_offsets", None)

    return {
        "task_family": getattr(unwrapped, "_so101_task_family", ["unknown"])[env_id],
        "instruction": getattr(unwrapped, "_so101_instruction_text", [""])[env_id],
        "active_object_ids": torch.nonzero(active, as_tuple=False).flatten().detach().cpu().tolist(),
        "target_object_id": target_id,
        "referent_object_ids": referent_ids,
        "direction_id": direction_id,
        "objects": objects,
        "bin": {
            "position": _tensor_list(bin_pos),
            "quaternion_wxyz": _tensor_list(bin_quat),
            "yaw": float(_quat_yaw(bin_quat).item()),
            "initial_position": _tensor_list(
                getattr(unwrapped, "_so101_initial_bin_pos_w", bin_pos.unsqueeze(0))[env_id]
            ),
            "failure_baseline_position": _tensor_list(
                getattr(unwrapped, "_so101_failure_bin_pos_w", bin_pos.unsqueeze(0))[env_id]
            ),
            "half_extents": _tensor_list(bin_half_extents[env_id]) if bin_half_extents is not None else None,
            "footprint_half_extents": (
                _tensor_list(bin_footprint_half_extents[env_id]) if bin_footprint_half_extents is not None else None
            ),
            "footprint_center_offset": (
                _tensor_list(bin_footprint_center_offsets[env_id]) if bin_footprint_center_offsets is not None else None
            ),
        },
        "move_boundary": {
            "coords": _tensor_list(
                getattr(unwrapped, "_so101_move_boundary_coords", torch.empty(0))[env_id : env_id + 1]
            ),
            "ids": _tensor_list(
                getattr(unwrapped, "_so101_move_boundary_ids", torch.empty(0, dtype=torch.long))[env_id : env_id + 1]
            ),
        },
        "robot_start": {
            "started_moving": _tensor_list(
                getattr(unwrapped, "_so101_robot_started_moving", torch.empty(0))[env_id : env_id + 1]
            ),
            "start_step": _tensor_list(
                getattr(unwrapped, "_so101_robot_start_step", torch.empty(0, dtype=torch.long))[env_id : env_id + 1]
            ),
            "start_time_s": _tensor_list(
                getattr(unwrapped, "_so101_robot_start_time_s", torch.empty(0))[env_id : env_id + 1]
            ),
        },
        "grasp_attempt_counts": _tensor_list(
            getattr(
                unwrapped,
                "_so101_grasp_attempt_counts",
                torch.zeros((unwrapped.num_envs, len(object_asset_names)), dtype=torch.long, device=unwrapped.device),
            )[env_id]
        ),
    }


def _trajectory_sample(
    env,
    object_asset_names: list[str],
    *,
    step: int,
    time_s: float,
    frame_index: int,
    action_lerobot: torch.Tensor,
    action_sim: torch.Tensor,
    term_eval: TermEval,
    env_id: int = 0,
) -> dict[str, Any]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    joint_ids = [robot.joint_names.index(joint_name) for joint_name in ACTION_JOINT_NAMES]
    ee_frame = unwrapped.scene["ee_frame"]
    bin_asset = unwrapped.scene[BIN_NAME]
    return {
        "step": step,
        "time_s": time_s,
        "dataset_frames_played": frame_index,
        "object_pos_w": benchmark_object_positions(unwrapped, object_asset_names)[env_id].detach().cpu().numpy(),
        "object_yaw": benchmark_object_yaws(unwrapped, object_asset_names)[env_id].detach().cpu().numpy(),
        "bin_pos_w": bin_asset.data.root_pos_w[env_id].detach().cpu().numpy(),
        "bin_quat_wxyz": bin_asset.data.root_quat_w[env_id].detach().cpu().numpy(),
        "bin_yaw": float(_quat_yaw(bin_asset.data.root_quat_w[env_id]).item()),
        "grasped_object_made_contact": bool(
            grasped_object_made_contact(unwrapped, object_asset_names)[env_id].item()
        ),
        "ee_pos_w": ee_frame.data.target_pos_w[env_id, 0, :].detach().cpu().numpy(),
        "joint_pos": robot.data.joint_pos[env_id, joint_ids].detach().cpu().numpy(),
        "action_lerobot": action_lerobot.detach().cpu().numpy(),
        "action_sim": action_sim.detach().cpu().numpy(),
        "success": term_eval.success,
        "failure": term_eval.failure,
        "timed_out": term_eval.timed_out,
    }


def _write_trajectory(path: Path, samples: list[dict[str, Any]]) -> None:
    if not samples:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        step=np.asarray([sample["step"] for sample in samples], dtype=np.int64),
        time_s=np.asarray([sample["time_s"] for sample in samples], dtype=np.float32),
        dataset_frames_played=np.asarray([sample["dataset_frames_played"] for sample in samples], dtype=np.int64),
        object_pos_w=np.stack([sample["object_pos_w"] for sample in samples]).astype(np.float32),
        object_yaw=np.stack([sample["object_yaw"] for sample in samples]).astype(np.float32),
        bin_pos_w=np.stack([sample["bin_pos_w"] for sample in samples]).astype(np.float32),
        bin_quat_wxyz=np.stack([sample["bin_quat_wxyz"] for sample in samples]).astype(np.float32),
        bin_yaw=np.asarray([sample["bin_yaw"] for sample in samples], dtype=np.float32),
        grasped_object_made_contact=np.asarray(
            [sample["grasped_object_made_contact"] for sample in samples],
            dtype=np.bool_,
        ),
        ee_pos_w=np.stack([sample["ee_pos_w"] for sample in samples]).astype(np.float32),
        joint_pos=np.stack([sample["joint_pos"] for sample in samples]).astype(np.float32),
        action_lerobot=np.stack([sample["action_lerobot"] for sample in samples]).astype(np.float32),
        action_sim=np.stack([sample["action_sim"] for sample in samples]).astype(np.float32),
        success=np.asarray([sample["success"] for sample in samples], dtype=np.bool_),
        failure=np.asarray([sample["failure"] for sample in samples], dtype=np.bool_),
        timed_out=np.asarray([sample["timed_out"] for sample in samples], dtype=np.bool_),
    )


def _render_for_capture(env) -> None:
    """Accumulate enough RTX frames for image-based DomeLight sampling to converge.

    The 'quality' renderer denoises the DomeLight over consecutive frames, and the temporal history is
    invalidated by each scene reset. A single render() leaves the overhead frame under-converged (dark)
    once the tiled multi-env render target is large enough, so render a fixed warmup burst instead.
    """
    warmup_frames = max(1, int(getattr(args_cli, "render_warmup_frames", 1) or 1))
    sim = env.unwrapped.sim
    for _ in range(warmup_frames):
        sim.render()


def _camera_rgb(env, camera_name: str = "camera_overhead", *, env_id: int = 0) -> np.ndarray:
    sensor = env.unwrapped.scene[camera_name]
    rgb = sensor.data.output["rgb"]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim == 4:
        rgb = rgb[env_id]
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = (rgb * 255.0).round().astype(np.uint8)
    elif rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _write_rgb_image(path: Path, rgb: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
        return path
    except ImportError:
        fallback_path = path.with_suffix(".npy")
        np.save(fallback_path, rgb)
        print(f"[WARN]: Pillow is unavailable; saved frame as NumPy array instead: {fallback_path}")
        return fallback_path


def _manual_term_evals(
    env,
    *,
    steps_by_env_id: dict[int, int],
    control_dt: float,
    success_params: dict[str, Any],
    failure_params: dict[str, Any],
) -> dict[int, TermEval]:
    with torch.inference_mode():
        timed_out_tensor = task_time_out(env.unwrapped)
        failure_tensor = benchmark_failure(env.unwrapped, **failure_params)
        success_tensor = task_success(env.unwrapped, **success_params)

    failure_reasons = getattr(env.unwrapped, "_so101_failure_reasons", None)
    evals = {}
    for env_id, step in steps_by_env_id.items():
        success = bool(success_tensor[env_id].item())
        failure = bool(failure_tensor[env_id].item())
        timed_out = bool(timed_out_tensor[env_id].item())
        if success:
            reason = "success"
        elif failure:
            reason = (
                failure_reasons[env_id]
                if failure_reasons and failure_reasons[env_id] != "none"
                else "failure"
            )
        elif timed_out:
            reason = "time_out"
        else:
            reason = "none"
        evals[env_id] = TermEval(
            step=step,
            time_s=step * control_dt,
            success=success,
            failure=failure,
            timed_out=timed_out,
            reason=reason,
        )
    return evals


def _success_params_for_final_eval(
    success_params: dict[str, Any],
    *,
    action_stream_exhausted: bool,
) -> tuple[dict[str, Any], bool]:
    params = dict(success_params)
    confirm_time_disabled = args_cli.no_success_confirm_time and action_stream_exhausted
    if confirm_time_disabled:
        params["confirm_time_s"] = 0.0
    return params, confirm_time_disabled


def _episode_window(
    episode_specs: list[BenchmarkEpisodeSpec],
    *,
    start_index: int,
    count: int,
) -> tuple[list[BenchmarkEpisodeSpec], list[int]]:
    if count < 1:
        raise ValueError(f"Expected --num_episodes >= 1, got {count}.")
    if start_index < 0:
        raise ValueError(f"Expected benchmark episode index >= 0, got {start_index}.")
    end_index = start_index + count
    if end_index > len(episode_specs):
        raise ValueError(
            f"Requested benchmark episode indices [{start_index}, {end_index}), "
            f"but {args_cli.episodes_jsonl} contains {len(episode_specs)} validated row(s)."
        )
    benchmark_indices = list(range(start_index, end_index))
    return episode_specs[start_index:end_index], benchmark_indices


def _parse_episode_indices(raw_indices: str) -> list[int]:
    indices = []
    for raw_index in raw_indices.split(","):
        raw_index = raw_index.strip()
        if not raw_index:
            continue
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError(f"Invalid benchmark episode index {raw_index!r} in {raw_indices!r}.") from exc
        indices.append(index)
    if not indices:
        raise ValueError("--benchmark_episode_indices was provided but no indices were parsed.")
    return indices


def _episode_selection(
    episode_specs: list[BenchmarkEpisodeSpec],
    benchmark_indices: list[int],
) -> list[BenchmarkEpisodeSpec]:
    invalid_indices = [index for index in benchmark_indices if index < 0 or index >= len(episode_specs)]
    if invalid_indices:
        raise ValueError(
            f"Requested benchmark episode indices {invalid_indices}, but "
            f"{args_cli.episodes_jsonl} contains {len(episode_specs)} validated row(s)."
        )
    return [episode_specs[index] for index in benchmark_indices]


def _planned_count(episode_specs: list[BenchmarkEpisodeSpec], benchmark_start: int) -> int:
    if args_cli.num_episodes is not None:
        return args_cli.num_episodes

    dataset_total = _dataset_total_episodes(args_cli.repo_root)
    benchmark_remaining = len(episode_specs) - benchmark_start
    if dataset_total is None:
        return benchmark_remaining

    dataset_remaining = dataset_total - args_cli.dataset_episode_index
    if dataset_remaining <= 0:
        raise ValueError(
            f"Dataset total_episodes={dataset_total}, but --dataset_episode_index={args_cli.dataset_episode_index}."
        )
    return min(dataset_remaining, benchmark_remaining)


def _make_output_dir() -> Path:
    if args_cli.output_dir is not None:
        output_dir = args_cli.output_dir
    else:
        root = args_cli.repo_root if args_cli.repo_root is not None else Path("outputs")
        output_dir = root / "eval" / f"sim_replay_outcomes_{_now_stamp()}"

    episodes_path = output_dir / "episodes.jsonl"
    if output_dir.exists() and episodes_path.exists() and not args_cli.overwrite:
        raise FileExistsError(f"Output dir already contains episodes.jsonl; use --overwrite or a new dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frames").mkdir(exist_ok=True)
    (output_dir / "state").mkdir(exist_ok=True)
    return output_dir


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
    env,
    *,
    env_id: int,
    object_asset_names: list[str],
    success_params: dict[str, Any],
    failure_params: dict[str, Any],
) -> dict[str, Any]:
    snapshots = task_condition_diagnostics(
        env.unwrapped,
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
    return asdict(snapshots[env_id])


def _episode_setup(env, *, env_id: int = 0) -> dict[str, Any]:
    episodes = getattr(env.unwrapped, "so101_bench_episodes", [])
    return dict(episodes[env_id]) if len(episodes) > env_id else {}


def _start_replay_lane(
    env,
    *,
    env_id: int,
    offset: int,
    episode_plan: list[BenchmarkEpisodeSpec],
    benchmark_indices: list[int],
    episode_layouts: list[dict | None],
    object_pool: list[str],
    object_asset_names: list[str],
    output_dir: Path,
    video_spans: dict[int, DatasetVideoSpan],
    control_dt: float,
    mapper: SO101ReplayActionMapper,
    actions: torch.Tensor,
    hold_action: torch.Tensor,
    hold_action_lerobot: torch.Tensor,
) -> ReplayLane:
    episode = episode_plan[offset]
    benchmark_index = benchmark_indices[offset]
    episode_layout = episode_layouts[offset]
    dataset_episode_index = args_cli.dataset_episode_index + offset
    print(
        f"[INFO]: Lane {env_id}: loading LeRobot dataset episode {dataset_episode_index} "
        f"from {args_cli.repo_root or args_cli.repo_id}..."
    )
    action_episode = _load_lerobot_action_episode(
        repo_id=args_cli.repo_id,
        root=args_cli.repo_root,
        episode_index=dataset_episode_index,
        device=env.unwrapped.device,
    )
    dataset_dt = 1.0 / max(action_episode.fps, 1.0e-6)
    if abs(dataset_dt - control_dt) > 1.0e-3:
        print(
            "[WARN]: Dataset fps does not match env control rate: "
            f"dataset_fps={action_episode.fps:.3f}, env_fps={1.0 / control_dt:.3f}. "
            "Replay will apply one dataset action per env step."
        )

    print(f"[INFO]: Lane {env_id}: resetting benchmark row {benchmark_index} ({offset + 1}/{len(episode_plan)})...")
    _configure_env_for_episode(env, episode, episode_layout, object_pool, object_asset_names)
    env_ids = torch.tensor([env_id], dtype=torch.long, device=env.unwrapped.device)
    _reset_env(env, env_ids)
    actions[env_id] = hold_action
    setup = _episode_setup(env, env_id=env_id)
    instruction = getattr(env.unwrapped, "_so101_instruction_text", [""])[env_id]
    print(f"[INFO]: Lane {env_id}: episode instruction: {instruction}")
    print(f"[INFO]: Lane {env_id}: replaying {action_episode.num_frames} frame(s).")

    if args_cli.frame_source == "sim":
        _render_for_capture(env)
        initial_frame_path = _write_rgb_image(
            output_dir / "frames" / f"episode_{dataset_episode_index:06d}_overhead_initial.png",
            _camera_rgb(env, env_id=env_id),
        )
        final_frame_path = None
    elif args_cli.frame_source == "dataset":
        initial_frame_path, final_frame_path = _write_dataset_overhead_frames(
            output_dir=output_dir,
            dataset_episode_index=dataset_episode_index,
            action_fps=action_episode.fps,
            video_spans=video_spans,
        )
    else:
        initial_frame_path = None
        final_frame_path = None

    return ReplayLane(
        env_id=env_id,
        offset=offset,
        episode=episode,
        benchmark_index=benchmark_index,
        episode_layout=episode_layout,
        dataset_episode_index=dataset_episode_index,
        action_episode=action_episode,
        setup=setup,
        initial_scene=_scene_state(env, object_asset_names, object_pool, env_id=env_id),
        initial_frame_path=initial_frame_path,
        final_frame_path=final_frame_path,
        last_action_lerobot=hold_action_lerobot.clone(),
        last_action_sim=hold_action.clone(),
    )


def _prepare_lane_action(
    env,
    lane: ReplayLane,
    *,
    object_asset_names: list[str],
    mapper: SO101ReplayActionMapper,
    actions: torch.Tensor,
    hold_action: torch.Tensor,
    hold_action_lerobot: torch.Tensor,
    initial_hold_steps: int,
    hold_last_steps: int,
) -> None:
    if lane.step < initial_hold_steps:
        actions[lane.env_id] = hold_action
        lane.last_action_lerobot = hold_action_lerobot.clone()
        lane.last_action_sim = hold_action.clone()
        return

    if not lane.robot_control_started:
        env_ids = torch.tensor([lane.env_id], dtype=torch.long, device=env.unwrapped.device)
        _begin_robot_control(env, object_asset_names, env_ids=env_ids)
        lane.robot_control_started = True

    replay_step = lane.step - initial_hold_steps
    if replay_step < lane.action_episode.num_frames:
        action_lerobot = mapper.clamp_lerobot_positions(lane.action_episode.actions[lane.frame_index])
        action_sim = mapper.lerobot_positions_to_sim_radians(action_lerobot)
        actions[lane.env_id] = action_sim
        lane.last_action_lerobot = action_lerobot.clone()
        lane.last_action_sim = action_sim.clone()
        lane.frame_index += 1
    elif replay_step >= lane.action_episode.num_frames + hold_last_steps:
        raise RuntimeError(f"Lane {lane.env_id} was stepped after replay episode {lane.dataset_episode_index} finished.")


def _append_trajectory_sample(
    env,
    lane: ReplayLane,
    *,
    object_asset_names: list[str],
    control_dt: float,
) -> None:
    if lane.final_eval is None:
        raise RuntimeError(f"Lane {lane.env_id} has no termination evaluation for trajectory capture.")
    lane.trajectory_samples.append(
        _trajectory_sample(
            env,
            object_asset_names,
            step=lane.step,
            time_s=lane.step * control_dt,
            frame_index=lane.frame_index,
            action_lerobot=lane.last_action_lerobot,
            action_sim=lane.last_action_sim,
            term_eval=lane.final_eval,
            env_id=lane.env_id,
        )
    )


def _upsert_final_trajectory_sample(
    env,
    lane: ReplayLane,
    *,
    object_asset_names: list[str],
    control_dt: float,
) -> None:
    if lane.trajectory_samples and int(lane.trajectory_samples[-1]["step"]) == lane.step:
        lane.trajectory_samples[-1] = _trajectory_sample(
            env,
            object_asset_names,
            step=lane.step,
            time_s=lane.step * control_dt,
            frame_index=lane.frame_index,
            action_lerobot=lane.last_action_lerobot,
            action_sim=lane.last_action_sim,
            term_eval=lane.final_eval,
            env_id=lane.env_id,
        )
    else:
        _append_trajectory_sample(env, lane, object_asset_names=object_asset_names, control_dt=control_dt)


def _finalize_replay_lane(
    env,
    lane: ReplayLane,
    *,
    object_pool: list[str],
    object_asset_names: list[str],
    output_dir: Path,
    control_dt: float,
    physics_dt: float,
    initial_hold_steps: int,
    hold_last_steps: int,
    success_params: dict[str, Any],
    failure_params: dict[str, Any],
) -> dict[str, Any]:
    if lane.final_eval is None:
        raise RuntimeError(f"Lane {lane.env_id} finished without a termination evaluation.")

    final_success_params, final_confirm_time_disabled = _success_params_for_final_eval(
        success_params,
        action_stream_exhausted=lane.action_stream_exhausted,
    )
    if final_confirm_time_disabled:
        lane.final_eval = _manual_term_evals(
            env,
            steps_by_env_id={lane.env_id: lane.step},
            control_dt=control_dt,
            success_params=final_success_params,
            failure_params=failure_params,
        )[lane.env_id]
        if lane.final_eval.success:
            lane.final_eval.failure = False
            lane.final_eval.timed_out = False

    if args_cli.save_trajectory:
        _upsert_final_trajectory_sample(env, lane, object_asset_names=object_asset_names, control_dt=control_dt)

    if args_cli.frame_source == "sim":
        _render_for_capture(env)
        lane.final_frame_path = _write_rgb_image(
            output_dir / "frames" / f"episode_{lane.dataset_episode_index:06d}_overhead_final.png",
            _camera_rgb(env, env_id=lane.env_id),
        )
    final_scene = _scene_state(env, object_asset_names, object_pool, env_id=lane.env_id)

    trajectory_path = None
    if args_cli.save_trajectory:
        trajectory_path = output_dir / "state" / f"episode_{lane.dataset_episode_index:06d}.npz"
        _write_trajectory(trajectory_path, lane.trajectory_samples)

    first_terminal_label = _label_from_eval(
        lane.first_terminal,
        missing_reason="no_terminal_condition_before_action_stream_exhausted",
    )
    final_label = _label_from_eval(lane.final_eval, missing_reason="no_success_condition_at_final_state")
    label = final_label if args_cli.label_source == "final" else first_terminal_label
    if lane.action_stream_exhausted and not lane.final_eval.done and lane.final_eval.reason == "none":
        final_label["failure_reason"] = "no_success_condition_at_final_state"
        final_label["reason"] = "action_stream_exhausted"
        if args_cli.label_source == "final":
            label = final_label

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _json_now(),
        "dataset": {
            "repo_id": args_cli.repo_id,
            "repo_root": str(args_cli.repo_root) if args_cli.repo_root is not None else None,
            "episode_index": lane.dataset_episode_index,
        },
        "benchmark": {
            "episodes_jsonl": str(args_cli.episodes_jsonl),
            "episode_layouts_jsonl": (
                str(args_cli.episode_layouts_jsonl) if args_cli.episode_layouts_jsonl is not None else None
            ),
            "episode_index": lane.benchmark_index,
            "trial_id": _episode_trial_id(lane.episode, lane.benchmark_index),
            "task_family": lane.episode.task_family,
            "instruction": lane.episode.instruction,
            "objects": list(lane.episode.objects),
            "target_object_id": lane.episode.target_object_id,
            "referent_object_ids": list(lane.episode.referent_object_ids),
            "direction": lane.episode.direction,
            "metadata": dict(lane.episode.metadata or {}),
            "env_setup": lane.setup,
        },
        "label": {
            "source": args_cli.label_source,
            **label,
        },
        "first_terminal_eval": first_terminal_label,
        "final_eval": final_label,
        "final_diagnostics": _final_condition_diagnostics(
            env,
            env_id=lane.env_id,
            object_asset_names=object_asset_names,
            success_params=final_success_params,
            failure_params=failure_params,
        ),
        "episode_length": {
            "dataset_frames": lane.action_episode.num_frames,
            "dataset_seconds": lane.action_episode.num_frames / max(lane.action_episode.fps, 1.0e-6),
            "frames_played": lane.frame_index,
            "sim_steps": lane.step,
            "sim_seconds": lane.step * control_dt,
            "initial_hold_steps": initial_hold_steps,
            "hold_last_steps": hold_last_steps,
            "action_stream_exhausted": lane.action_stream_exhausted,
        },
        "paths": {
            "overhead_initial": (
                str(lane.initial_frame_path.relative_to(output_dir)) if lane.initial_frame_path is not None else None
            ),
            "overhead_final": (
                str(lane.final_frame_path.relative_to(output_dir)) if lane.final_frame_path is not None else None
            ),
            "state_trajectory": str(trajectory_path.relative_to(output_dir)) if trajectory_path is not None else None,
        },
        "state_schema": {
            "object_asset_names": object_asset_names,
            "object_labels": object_pool,
            "action_joint_names": ACTION_JOINT_NAMES,
            "trajectory_stride": args_cli.trajectory_stride if args_cli.save_trajectory else None,
            "includes_grasped_object_made_contact": bool(args_cli.save_trajectory),
        },
        "eval_setup": _capture_eval_setup(
            env,
            env_id=lane.env_id,
            control_dt=control_dt,
            physics_dt=physics_dt,
            success_params=final_success_params,
            failure_params=failure_params,
        ),
        "initial_scene": lane.initial_scene,
        "final_scene": final_scene,
    }


def main():
    if args_cli.num_envs is not None and args_cli.num_envs < 1:
        raise ValueError(f"Expected --num_envs >= 1, got {args_cli.num_envs}.")
    if args_cli.dataset_episode_index < 0:
        raise ValueError(f"Expected --dataset_episode_index >= 0, got {args_cli.dataset_episode_index}.")
    if args_cli.speed <= 0.0:
        raise ValueError(f"Expected --speed > 0, got {args_cli.speed}.")
    if args_cli.initial_hold_time_s < 0.0:
        raise ValueError(f"Expected --initial_hold_time_s >= 0, got {args_cli.initial_hold_time_s}.")
    if args_cli.hold_last_action_time_s < 0.0:
        raise ValueError(f"Expected --hold_last_action_time_s >= 0, got {args_cli.hold_last_action_time_s}.")
    if args_cli.trajectory_stride < 1:
        raise ValueError(f"Expected --trajectory_stride >= 1, got {args_cli.trajectory_stride}.")

    episode_specs = load_episode_jsonl(args_cli.episodes_jsonl)
    if args_cli.benchmark_episode_indices:
        benchmark_indices = _parse_episode_indices(args_cli.benchmark_episode_indices)
        episode_plan = _episode_selection(episode_specs, benchmark_indices)
        planned_count = len(episode_plan)
    else:
        benchmark_start = (
            args_cli.dataset_episode_index
            if args_cli.benchmark_episode_index is None
            else args_cli.benchmark_episode_index
        )
        planned_count = _planned_count(episode_specs, benchmark_start)
        episode_plan, benchmark_indices = _episode_window(
            episode_specs,
            start_index=benchmark_start,
            count=planned_count,
        )
    episode_layouts = _load_episode_layouts(episode_plan, benchmark_indices, args_cli.episode_layouts_jsonl)

    dataset_episode_indices = [args_cli.dataset_episode_index + i for i in range(len(episode_plan))]
    dataset_instructions = _load_dataset_episode_instructions(args_cli.repo_root)
    _verify_jsonl_matches_dataset(
        episode_specs=episode_specs,
        episode_plan=episode_plan,
        benchmark_indices=benchmark_indices,
        dataset_episode_indices=dataset_episode_indices,
        dataset_instructions=dataset_instructions,
    )

    output_dir = _make_output_dir()
    video_spans = (
        _load_dataset_video_spans(args_cli.repo_root)
        if args_cli.frame_source == "dataset"
        else {}
    )

    print(f"[INFO]: Loaded {len(episode_specs)} validated JSONL episode(s) from {args_cli.episodes_jsonl}.")
    print(
        "[INFO]: Replay mapping: "
        f"dataset episodes {args_cli.dataset_episode_index}"
        f"..{args_cli.dataset_episode_index + planned_count - 1} -> "
        f"benchmark rows {benchmark_indices[0]}..{benchmark_indices[-1]}"
    )
    print(f"[INFO]: Saving outcome artifacts to {output_dir}")
    if args_cli.frame_source == "dataset":
        print("[INFO]: Saving overhead frames from recorded LeRobot videos; Isaac camera sensors are disabled.")
    if args_cli.no_success_confirm_time:
        print("[INFO]: Final exhausted action streams will be scored with success confirm_time_s=0.0.")

    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    np.random.seed(args_cli.seed)

    object_pool = _episode_object_pool(episode_plan)
    print(f"[INFO]: Pre-spawning {len(object_pool)} benchmark object asset(s): {', '.join(object_pool)}")

    env, object_asset_names, success_params, failure_params = _make_env(
        object_pool,
        episode_plan[0],
        episode_layouts[0],
    )
    control_dt = float(env.unwrapped.step_dt)
    physics_dt = float(env.unwrapped.cfg.sim.dt)
    initial_hold_steps = max(0, math.ceil(args_cli.initial_hold_time_s / control_dt))
    hold_last_steps = max(0, math.ceil(args_cli.hold_last_action_time_s / control_dt))
    num_parallel_envs = int(env.unwrapped.num_envs)
    print(
        "[INFO]: Timing: "
        f"physics_dt={physics_dt:.6f}s, control_dt={control_dt:.6f}s, "
        f"max_episode_length_s={env.unwrapped.cfg.episode_length_s:.1f}"
    )
    print(f"[INFO]: Native Isaac Lab replay lanes: {num_parallel_envs}")

    mapper = SO101ReplayActionMapper(device=env.unwrapped.device)
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    hold_action = _initial_robot_action(env)
    hold_action_lerobot = mapper.sim_radians_to_lerobot_positions(hold_action)
    actions[:] = hold_action
    _reset_env(env)

    episodes_path = output_dir / "episodes.jsonl"
    summary_records: list[dict[str, Any]] = []

    try:
        with episodes_path.open("w", encoding="utf-8") as episodes_file:
            active_lanes: dict[int, ReplayLane] = {}
            pending_records: dict[int, dict[str, Any]] = {}
            next_offset = 0
            next_write_offset = 0
            completed_count = 0
            collection_start_s = time.perf_counter()

            def start_lane(env_id: int) -> None:
                nonlocal next_offset
                active_lanes[env_id] = _start_replay_lane(
                    env,
                    env_id=env_id,
                    offset=next_offset,
                    episode_plan=episode_plan,
                    benchmark_indices=benchmark_indices,
                    episode_layouts=episode_layouts,
                    object_pool=object_pool,
                    object_asset_names=object_asset_names,
                    output_dir=output_dir,
                    video_spans=video_spans,
                    control_dt=control_dt,
                    mapper=mapper,
                    actions=actions,
                    hold_action=hold_action,
                    hold_action_lerobot=hold_action_lerobot,
                )
                next_offset += 1

            def flush_ready_records() -> None:
                nonlocal next_write_offset
                while next_write_offset in pending_records:
                    record = pending_records.pop(next_write_offset)
                    episodes_file.write(json.dumps(record, separators=(",", ":")) + "\n")
                    episodes_file.flush()
                    summary_records.append(record)
                    next_write_offset += 1

            for env_id in range(min(num_parallel_envs, planned_count)):
                start_lane(env_id)

            while active_lanes and simulation_app.is_running():
                replay_step_start = time.perf_counter()
                with torch.inference_mode():
                    for lane in active_lanes.values():
                        _prepare_lane_action(
                            env,
                            lane,
                            object_asset_names=object_asset_names,
                            mapper=mapper,
                            actions=actions,
                            hold_action=hold_action,
                            hold_action_lerobot=hold_action_lerobot,
                            initial_hold_steps=initial_hold_steps,
                            hold_last_steps=hold_last_steps,
                        )

                    env.step(actions)
                    for lane in active_lanes.values():
                        lane.step += 1
                    evals = _manual_term_evals(
                        env,
                        steps_by_env_id={env_id: lane.step for env_id, lane in active_lanes.items()},
                        control_dt=control_dt,
                        success_params=success_params,
                        failure_params=failure_params,
                    )

                finished_env_ids = []
                for env_id, lane in active_lanes.items():
                    lane.final_eval = evals[env_id]
                    if args_cli.save_trajectory and (
                        lane.step % args_cli.trajectory_stride == 0 or lane.final_eval.done
                    ):
                        _append_trajectory_sample(
                            env,
                            lane,
                            object_asset_names=object_asset_names,
                            control_dt=control_dt,
                        )

                    if lane.final_eval.done and lane.first_terminal is None:
                        lane.first_terminal = lane.final_eval
                        print(
                            f"[INFO]: Lane {lane.env_id}: first terminal condition for dataset episode "
                            f"{lane.dataset_episode_index} at {lane.final_eval.time_s:.2f}s: "
                            f"success={lane.final_eval.success}, reason={lane.final_eval.reason}"
                        )

                    natural_end_step = initial_hold_steps + lane.action_episode.num_frames + hold_last_steps
                    if lane.step >= natural_end_step:
                        lane.action_stream_exhausted = True
                    if (
                        lane.action_stream_exhausted
                        or lane.final_eval.timed_out
                        or (args_cli.stop_on_done and lane.final_eval.done)
                    ):
                        finished_env_ids.append(env_id)

                for env_id in finished_env_ids:
                    lane = active_lanes.pop(env_id)
                    record = _finalize_replay_lane(
                        env,
                        lane,
                        object_pool=object_pool,
                        object_asset_names=object_asset_names,
                        output_dir=output_dir,
                        control_dt=control_dt,
                        physics_dt=physics_dt,
                        initial_hold_steps=initial_hold_steps,
                        hold_last_steps=hold_last_steps,
                        success_params=success_params,
                        failure_params=failure_params,
                    )
                    pending_records[lane.offset] = record
                    completed_count += 1
                    elapsed_s = time.perf_counter() - collection_start_s
                    episodes_per_minute = 60.0 * completed_count / max(elapsed_s, 1.0e-6)
                    remaining_s = elapsed_s * (planned_count - completed_count) / completed_count
                    expected_completion = datetime.now().astimezone() + timedelta(seconds=remaining_s)
                    print(
                        f"[INFO]: Episode {completed_count}/{planned_count} finished on lane {env_id}: "
                        f"dataset_episode={lane.dataset_episode_index}, "
                        f"label_success={record['label']['success']}, "
                        f"reason={record['label']['failure_reason']}, "
                        f"sim_seconds={lane.step * control_dt:.2f}, "
                        f"frames_played={lane.frame_index}/{lane.action_episode.num_frames}, "
                        f"elapsed={_format_duration(elapsed_s)}, "
                        f"rate={episodes_per_minute:.2f} episodes/min, "
                        f"eta={_format_duration(remaining_s)}, "
                        f"expected_completion={expected_completion.isoformat(timespec='seconds')}"
                    )
                    if next_offset < planned_count:
                        start_lane(env_id)

                flush_ready_records()
                if args_cli.real_time:
                    dt_s = time.perf_counter() - replay_step_start
                    time.sleep(max((control_dt / args_cli.speed) - dt_s, 0.0))

            for offset in sorted(pending_records):
                record = pending_records[offset]
                episodes_file.write(json.dumps(record, separators=(",", ":")) + "\n")
                summary_records.append(record)
            episodes_file.flush()

        successes = sum(1 for record in summary_records if record["label"]["success"])
        failures = len(summary_records) - successes
        failure_counts: dict[str, int] = {}
        for record in summary_records:
            reason = record["label"]["failure_reason"]
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
        summary = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _json_now(),
            "episodes_path": str(episodes_path),
            "label_source": args_cli.label_source,
            "completed_episodes": len(summary_records),
            "successes": successes,
            "failures": failures,
            "success_rate": successes / max(len(summary_records), 1),
            "failure_reason_counts": failure_counts,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args_cli).items()
                if key != "app_launcher"
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            "[INFO]: Outcome summary: "
            f"success={successes}/{len(summary_records)} ({100.0 * summary['success_rate']:.1f}%), "
            f"failures={failures}"
        )
        print(f"[INFO]: Wrote {episodes_path} and {output_dir / 'summary.json'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
