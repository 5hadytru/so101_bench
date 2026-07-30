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
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timedelta
import hashlib
import importlib.metadata as importlib_metadata
import inspect
import json
import math
from pathlib import Path
import platform
import signal
import subprocess
import sys
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
    "--object_pool_episodes_jsonl",
    type=Path,
    default=None,
    help=(
        "Optional canonical episode JSONL used only to define the pre-spawned object-pool order. "
        "Use the original full-run JSONL when scoring a pruned, reordered, or merged dataset."
    ),
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
dataset_selection_group = parser.add_mutually_exclusive_group()
dataset_selection_group.add_argument(
    "--dataset_episode_index",
    "--episode",
    type=int,
    default=0,
    help="First LeRobot dataset episode index to replay.",
)
dataset_selection_group.add_argument(
    "--dataset_episode_indices",
    type=str,
    default=None,
    help=(
        "Comma-separated LeRobot dataset episode indices to replay, in the requested output order. "
        "By default each dataset episode is paired with the benchmark JSONL/layout row having the same index; "
        "pass --benchmark_episode_indices to provide a different one-to-one mapping."
    ),
)
benchmark_selection_group = parser.add_mutually_exclusive_group()
benchmark_selection_group.add_argument(
    "--benchmark_episode_index",
    type=int,
    default=None,
    help=(
        "First benchmark JSONL/layout row to reset before replay. Defaults to --dataset_episode_index, matching "
        "sequential teleop recordings with no skipped/cancelled episodes."
    ),
)
benchmark_selection_group.add_argument(
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
    "--record_dataset",
    action="store_true",
    default=False,
    help=(
        "Record the replayed simulated observations, source actions, and camera frames to a new LeRobot dataset. "
        "Dataset recording currently requires --num_envs 1 so episode/video writes remain strictly ordered."
    ),
)
parser.add_argument(
    "--record_repo_id",
    type=str,
    default=None,
    help="Repo id stored in the metadata of the simulated output LeRobot dataset.",
)
parser.add_argument(
    "--record_repo_root",
    type=Path,
    default=None,
    help="Local root for the simulated output LeRobot dataset. Must differ from the source --repo_root.",
)
parser.add_argument(
    "--dataset_streaming_encoding",
    nargs="?",
    const=True,
    default=True,
    type=_str_to_bool,
    help="Encode output LeRobot videos while replay is running.",
)
parser.add_argument(
    "--dataset_vcodec",
    type=str,
    default="libsvtav1",
    help="Output video codec. libsvtav1 preserves compatibility with the real SO-101 datasets.",
)
parser.add_argument("--dataset_encoder_threads", type=int, default=2)
parser.add_argument("--dataset_encoder_queue_size", type=int, default=300)
parser.add_argument("--dataset_image_writer_processes", type=int, default=0)
parser.add_argument("--dataset_image_writer_threads_per_camera", type=int, default=4)
parser.add_argument("--dataset_video_files_size_mb", type=int, default=200)
parser.add_argument(
    "--retime_reference_repo_root",
    type=Path,
    default=None,
    help=(
        "Optional local LeRobot dataset whose mean frames per episode is matched by uniformly time-warping "
        "every replayed source action path. The output remains at the environment control rate and, with "
        "--record_dataset, contains newly simulated images, states, and actions."
    ),
)
parser.add_argument(
    "--retime_scale",
    type=float,
    default=None,
    help=(
        "Explicit uniform trajectory-duration multiplier. Values must be >= 1.0. This is mutually exclusive "
        "with --retime_reference_repo_root."
    ),
)
parser.add_argument(
    "--retime_strategy",
    choices=("tracking_compensated", "action_path"),
    default="tracking_compensated",
    help=(
        "How slowed arm targets are constructed. 'tracking_compensated' follows the recorded physical "
        "joint-state path and retains a time-scaled fraction of the original action/state tracking offset; "
        "'action_path' directly time-warps the original commands. The gripper always follows the original "
        "command path so grasp/release semantics are preserved."
    ),
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
    "--resume",
    action="store_true",
    default=False,
    help=(
        "Resume a sequential replay from an existing --output_dir and recorded LeRobot dataset. "
        "Completed outcome rows and recorded episodes must be contiguous and have identical counts."
    ),
)
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

if args_cli.dataset_episode_indices is not None:
    if args_cli.num_episodes is not None:
        parser.error("--dataset_episode_indices cannot be combined with --num_episodes.")
    if args_cli.benchmark_episode_index is not None:
        parser.error(
            "--dataset_episode_indices cannot be combined with --benchmark_episode_index; "
            "use --benchmark_episode_indices for an explicit one-to-one mapping."
        )

# Fail on recording-only CLI mistakes before starting Isaac Sim.  Raising these
# from main() happens after Kit has launched, and Kit shutdown can hide the Python
# exception behind several seconds of renderer startup messages.
if args_cli.record_dataset:
    if args_cli.num_envs != 1:
        parser.error("--record_dataset requires --num_envs 1 so video frames remain episode-ordered.")
    if args_cli.record_repo_id is None:
        parser.error("--record_dataset requires --record_repo_id.")
    if args_cli.record_repo_root is None:
        parser.error("--record_dataset requires --record_repo_root.")
    if args_cli.repo_root is not None and args_cli.record_repo_root.resolve() == args_cli.repo_root.resolve():
        parser.error("--record_repo_root must differ from the source --repo_root.")

# Isaac camera rendering is only needed when replay frames come from sim. For
# --frame_source dataset/none, _make_env() nulls out the camera sensors and visual
# observations, so skip the RTX renderer entirely: faster startup and no per-step render.
args_cli.enable_cameras = args_cli.frame_source == "sim" or args_cli.record_dataset

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
import so101_bench.mdp as so101_mdp
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
from so101_bench.utils.lerobot_dataset import (
    LeRobotSimDatasetRecorder,
    dataset_cameras,
    real_compatible_camera_sources,
    recording_images,
)


ACTION = "action"
OBSERVATION_STATE = "observation.state"
ACTION_JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
INITIAL_ROBOT_JOINT_POS = lerobot_pose_to_sim_joint_pos(LEROBOT_INITIAL_JOINT_POS)
BIN_NAME = "plastic_bin"
SCHEMA_VERSION = 2

# These are deliberately stable schema names rather than implementation details of
# the termination module.  Missing counters are written as -1, which lets older or
# task-specific environments use the same NPZ schema without pretending that a
# condition was observed to be false.
CONDITION_COUNTER_SPECS = (
    ("success_bin", "_so101_bin_success_counter"),
    ("success_next_to", "_so101_next_to_success_counter"),
    ("success_between", "_so101_between_success_counter"),
    ("success_move", "_so101_move_success_counter"),
    ("failure_max_grasp_attempts", "_so101_attempt_failure_counter"),
    ("failure_bin_displaced", "_so101_bin_failure_counter"),
    ("failure_non_target_moved", "_so101_non_target_failure_counter"),
    ("failure_move_boundary_moved", "_so101_move_boundary_failure_counter"),
    ("failure_move_past_boundary", "_so101_move_past_boundary_failure_counter"),
    ("failure_move_not_straight", "_so101_move_straightness_failure_counter"),
    ("failure_grasped_object_contact", "_so101_grasped_object_contact_steps"),
)

ACTION_PHASE_IDS = {"reset": 0, "initial_hold": 1, "dataset": 2, "final_hold": 3}
STOP_REQUESTED = False


def _request_graceful_stop(signum, _frame) -> None:
    """Finish active replay lanes, persist them, and then stop scheduling episodes."""

    global STOP_REQUESTED
    if not STOP_REQUESTED:
        signal_name = signal.Signals(signum).name
        print(
            f"\n[INFO]: Received {signal_name}; finishing the active episode before pausing...",
            flush=True,
        )
    STOP_REQUESTED = True


for _stop_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(_stop_signal, _request_graceful_stop)


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
    observed_states: torch.Tensor | None = None

    @property
    def num_frames(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class UniformRetimingPlan:
    """Smooth time warp used to lengthen a replay while preserving its demonstrated path."""

    scale: float
    strategy: str
    source_mean_frames: float
    target_mean_frames: float
    source_total_episodes: int
    source_total_frames: int
    reference_repo_root: str | None


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
    source_action_num_frames: int
    setup: dict[str, Any]
    initial_scene: dict[str, Any]
    initial_frame_path: Path | None
    final_frame_path: Path | None
    last_action_lerobot_raw: torch.Tensor
    last_action_lerobot: torch.Tensor
    last_action_sim: torch.Tensor
    last_action_clamped_mask: torch.Tensor
    last_action_delta_lerobot: torch.Tensor
    last_action_frame_index: int = -1
    last_action_phase: str = "reset"
    step: int = 0
    frame_index: int = 0
    robot_control_started: bool = False
    first_terminal: TermEval | None = None
    first_terminal_postmortem_raw: Any = None
    final_eval: TermEval | None = None
    trajectory_samples: list[dict[str, Any]] = field(default_factory=list)
    action_stream_exhausted: bool = False
    recorded_dataset_episode_index: int | None = None


def _now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _json_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_PROVENANCE_CACHE: dict[str, Any] | None = None


def _run_provenance() -> dict[str, Any]:
    """Capture enough immutable run identity to reproduce or audit an outcome."""
    global _PROVENANCE_CACHE
    if _PROVENANCE_CACHE is not None:
        return _PROVENANCE_CACHE

    def command_output(command: list[str]) -> str | None:
        try:
            result = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except OSError:
            return None

    versions = {}
    for distribution in ("isaaclab", "isaacsim", "torch", "numpy", "gymnasium", "lerobot"):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = None

    tracked_status = command_output(["git", "status", "--porcelain", "--untracked-files=no"])
    metadata_info_path = args_cli.repo_root / "meta" / "info.json" if args_cli.repo_root is not None else None
    retime_reference_info_path = (
        args_cli.retime_reference_repo_root / "meta" / "info.json"
        if args_cli.retime_reference_repo_root is not None
        else None
    )
    _PROVENANCE_CACHE = {
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_tracked_worktree_dirty": bool(tracked_status),
        "git_untracked_files_checked": False,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_runtime": torch.version.cuda,
        "versions": versions,
        "seed": int(args_cli.seed),
        "task": args_cli.task,
        "input_sha256": {
            "episodes_jsonl": _file_sha256(args_cli.episodes_jsonl),
            "object_pool_episodes_jsonl": _file_sha256(args_cli.object_pool_episodes_jsonl),
            "episode_layouts_jsonl": _file_sha256(args_cli.episode_layouts_jsonl),
            "dataset_info_json": _file_sha256(metadata_info_path),
            "retime_reference_info_json": _file_sha256(retime_reference_info_path),
        },
        "collector_script_sha256": _file_sha256(Path(__file__).resolve()),
    }
    return _PROVENANCE_CACHE


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


def _raw_joint_positions_to_tensor(
    raw_positions: Any,
    source_names: list[str],
    *,
    feature_name: str,
    device: str,
    episode_index: int,
    frame_index: int,
) -> torch.Tensor:
    if isinstance(raw_positions, dict):
        raw_names = list(raw_positions.keys())
        raw_values = np.asarray([raw_positions[name] for name in raw_names], dtype=np.float32).reshape(-1)
        source_names = raw_names
    elif isinstance(raw_positions, torch.Tensor):
        raw_values = raw_positions.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    else:
        raw_values = np.asarray(raw_positions, dtype=np.float32).reshape(-1)

    if len(source_names) != len(raw_values):
        if len(raw_values) == len(LEROBOT_JOINT_FEATURE_ORDER):
            source_names = list(LEROBOT_JOINT_FEATURE_ORDER)
        else:
            raise ValueError(
                f"Dataset episode {episode_index} frame {frame_index} has {feature_name} shape "
                f"{raw_values.shape}, but its feature names are {source_names!r}."
            )

    index_by_name = {_canonical_action_name(name): index for index, name in enumerate(source_names)}
    missing = [name for name in LEROBOT_JOINT_FEATURE_ORDER if name not in index_by_name]
    if missing:
        raise ValueError(
            f"Dataset episode {episode_index} {feature_name} names are missing {missing}. "
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
    load_observed_states: bool = False,
) -> LeRobotActionEpisode:
    dataset = _open_lerobot_dataset(repo_id, root, episode_index)
    features = getattr(dataset, "features", {})
    if ACTION not in features:
        raise ValueError(f"LeRobot dataset has no {ACTION!r} feature. Found features: {list(features)}")
    if load_observed_states and OBSERVATION_STATE not in features:
        raise ValueError(
            f"Trajectory-preserving retiming requires {OBSERVATION_STATE!r}. "
            f"Found features: {list(features)}. Use --retime_strategy action_path for this dataset."
        )

    feature_names = _coerce_action_feature_names(features[ACTION].get("names"))
    state_feature_names = (
        _coerce_action_feature_names(features[OBSERVATION_STATE].get("names"))
        if load_observed_states
        else []
    )
    if hasattr(dataset, "select_columns"):
        action_rows = dataset.select_columns(ACTION)
        state_rows = dataset.select_columns(OBSERVATION_STATE) if load_observed_states else None
    else:
        action_rows = getattr(dataset, "hf_dataset").select_columns(ACTION)
        state_rows = (
            getattr(dataset, "hf_dataset").select_columns(OBSERVATION_STATE)
            if load_observed_states
            else None
        )

    num_frames = int(getattr(dataset, "num_frames", len(action_rows)))
    if num_frames <= 0:
        raise ValueError(f"LeRobot dataset episode {episode_index} has no frames.")

    actions = []
    observed_states = [] if load_observed_states else None
    for frame_index in range(num_frames):
        row = action_rows[frame_index]
        actions.append(
            _raw_joint_positions_to_tensor(
                row[ACTION],
                feature_names,
                feature_name=ACTION,
                device=device,
                episode_index=episode_index,
                frame_index=frame_index,
            )
        )
        if observed_states is not None:
            assert state_rows is not None
            state_row = state_rows[frame_index]
            observed_states.append(
                _raw_joint_positions_to_tensor(
                    state_row[OBSERVATION_STATE],
                    state_feature_names,
                    feature_name=OBSERVATION_STATE,
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
        observed_states=(
            torch.stack(observed_states, dim=0) if observed_states is not None else None
        ),
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


def _dataset_frame_totals(root: Path, *, option_name: str) -> tuple[int, int]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"{option_name} is missing LeRobot metadata: {info_path}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        total_episodes = int(info["total_episodes"])
        total_frames = int(info["total_frames"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{info_path} must contain integer total_episodes and total_frames fields."
        ) from exc
    if total_episodes <= 0 or total_frames <= 0:
        raise ValueError(
            f"{info_path} has invalid dataset totals: episodes={total_episodes}, frames={total_frames}."
        )
    return total_episodes, total_frames


def _build_uniform_retiming_plan(source_root: Path | None) -> UniformRetimingPlan | None:
    if args_cli.retime_reference_repo_root is None and args_cli.retime_scale is None:
        return None
    if args_cli.retime_reference_repo_root is not None and args_cli.retime_scale is not None:
        raise ValueError("--retime_reference_repo_root and --retime_scale are mutually exclusive.")
    if source_root is None:
        raise ValueError("Action retiming requires a local --repo_root.")

    source_episodes, source_frames = _dataset_frame_totals(source_root, option_name="--repo_root")
    source_mean = source_frames / source_episodes
    reference_root = args_cli.retime_reference_repo_root
    if reference_root is not None:
        reference_episodes, reference_frames = _dataset_frame_totals(
            reference_root,
            option_name="--retime_reference_repo_root",
        )
        target_mean = reference_frames / reference_episodes
        scale = target_mean / source_mean
    else:
        scale = float(args_cli.retime_scale)
        target_mean = source_mean * scale

    if not math.isfinite(scale) or scale < 1.0:
        raise ValueError(
            "Retiming only supports slowing trajectories: expected a finite scale >= 1.0, "
            f"got {scale:.9g} from source_mean={source_mean:.3f}, target_mean={target_mean:.3f}."
        )
    return UniformRetimingPlan(
        scale=scale,
        strategy=args_cli.retime_strategy,
        source_mean_frames=source_mean,
        target_mean_frames=target_mean,
        source_total_episodes=source_episodes,
        source_total_frames=source_frames,
        reference_repo_root=str(reference_root) if reference_root is not None else None,
    )


def smoothly_retime_waypoint_path(
    source_waypoints: np.ndarray,
    *,
    initial_waypoint: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Time-warp a joint path with shape-preserving C1 interpolation.

    PCHIP avoids the coordinate overshoot of an unconstrained cubic spline while
    removing the velocity discontinuities introduced by repeated frames or
    piecewise-linear targets. The final waypoint is restored exactly.
    """

    from scipy.interpolate import PchipInterpolator

    waypoints = np.asarray(source_waypoints, dtype=np.float32)
    initial = np.asarray(initial_waypoint, dtype=np.float32)
    if waypoints.ndim != 2 or waypoints.shape[0] < 1:
        raise ValueError(f"Expected non-empty [frames, joints] waypoints, got shape {waypoints.shape}.")
    if initial.shape != waypoints.shape[1:]:
        raise ValueError(
            f"Initial waypoint shape {initial.shape} does not match joint shape {waypoints.shape[1:]}."
        )
    if not math.isfinite(scale) or scale < 1.0:
        raise ValueError(f"Expected finite retime scale >= 1.0, got {scale!r}.")

    source_frames = int(waypoints.shape[0])
    target_frames = max(source_frames, int(round(source_frames * scale)))
    if target_frames == source_frames:
        return waypoints.copy()

    path = np.concatenate((initial[None, :], waypoints), axis=0)
    source_times = np.arange(source_frames + 1, dtype=np.float64)
    target_times = np.arange(1, target_frames + 1, dtype=np.float64) * (
        source_frames / target_frames
    )
    result = PchipInterpolator(source_times, path, axis=0)(target_times)
    result[-1] = waypoints[-1]
    return np.asarray(result, dtype=np.float32)


def trajectory_preserving_retime_actions(
    source_actions: np.ndarray,
    *,
    source_observed_states: np.ndarray | None,
    initial_action: np.ndarray,
    scale: float,
    strategy: str,
) -> np.ndarray:
    """Build smooth slowed commands that follow the demonstrated physical path.

    Directly stretching absolute position commands gives the simulator more time to
    converge on command excursions that the original robot only partially reached.
    That changes the spatial path and can cause overshoot. The tracking-compensated
    strategy instead uses the recorded joint-state path as the arm reference and
    retains 1/scale of the original action/state offset as feed-forward. At scale
    1.0 this reduces exactly to the original arm commands; as duration increases it
    smoothly reduces the dynamic lead that would otherwise cause overshoot.

    The gripper keeps the source command path (rather than the observed jaw state)
    so grasp and release intent remains coordinated with arm progress.
    """

    actions = np.asarray(source_actions, dtype=np.float32)
    initial = np.asarray(initial_action, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] < 1:
        raise ValueError(f"Expected non-empty [frames, joints] actions, got shape {actions.shape}.")
    if initial.shape != actions.shape[1:]:
        raise ValueError(
            f"Initial action shape {initial.shape} does not match action joint shape {actions.shape[1:]}."
        )
    if not math.isfinite(scale) or scale < 1.0:
        raise ValueError(f"Expected finite retime scale >= 1.0, got {scale!r}.")

    if strategy == "action_path":
        waypoints = actions
    elif strategy == "tracking_compensated":
        if source_observed_states is None:
            raise ValueError(
                "tracking_compensated retiming requires source observation.state waypoints."
            )
        states = np.asarray(source_observed_states, dtype=np.float32)
        if states.shape != actions.shape:
            raise ValueError(
                f"Observed-state shape {states.shape} does not match action shape {actions.shape}."
            )
        waypoints = states + (actions - states) / scale
        # Jaw observations contain actuator/contact lag. Preserve the demonstrated
        # command path so close/open intent is not weakened by feeding that lag back.
        waypoints[:, -1] = actions[:, -1]
    else:
        raise ValueError(f"Unknown retime strategy: {strategy!r}.")

    return smoothly_retime_waypoint_path(
        waypoints,
        initial_waypoint=initial,
        scale=scale,
    )


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
    """Convert simulator/config/diagnostic values to a JSON-friendly representation.

    This is intentionally also used for postmortem diagnostics.  The richer
    classifier has evolved from a small dataclass into nested evidence records; a
    recursive serializer keeps the collector compatible with either representation.
    """
    if isinstance(value, SceneEntityCfg):
        payload: dict[str, Any] = {"__scene_entity_cfg__": True, "name": value.name}
        joint_names = getattr(value, "joint_names", None)
        body_names = getattr(value, "body_names", None)
        if joint_names is not None:
            payload["joint_names"] = list(joint_names) if not isinstance(joint_names, str) else joint_names
        if body_names is not None:
            payload["body_names"] = list(body_names) if not isinstance(body_names, str) else body_names
        return payload
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize_param_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _serialize_param_value(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_serialize_param_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, tuple):
        return [_serialize_param_value(item) for item in value]
    if isinstance(value, list):
        return [_serialize_param_value(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _serialize_param_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _serialize_param_value(value.tolist())
    if isinstance(value, np.generic):
        return _serialize_param_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _compact_json(value: Any) -> str:
    return json.dumps(
        _serialize_param_value(value),
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


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
    final_success_params: dict[str, Any] | None = None,
    final_success_confirm_time_disabled: bool = False,
) -> dict[str, Any]:
    """Capture live evaluation parameters separately from final-label policy.

    ``success_params`` always describes the rules that advanced counters during
    the trajectory.  A confirmation waiver is a one-time final scoring policy;
    persisting its zero-second window as the live rule would corrupt offline
    replay of every preceding frame.
    """
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    action_joint_ids = [robot.joint_names.index(joint_name) for joint_name in ACTION_JOINT_NAMES]
    joint_pos_limits = robot.data.joint_pos_limits[env_id, action_joint_ids].detach().cpu().tolist()
    env_origins = unwrapped.scene.env_origins[env_id].detach().cpu().tolist()
    decimation = int(getattr(unwrapped.cfg, "decimation", max(1, round(control_dt / max(physics_dt, 1.0e-9)))))
    record = {
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
        "final_success_params": _serialize_term_params(final_success_params or success_params),
        "final_success_confirm_time_disabled": bool(final_success_confirm_time_disabled),
    }
    return record


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
    if args_cli.frame_source != "sim" and not args_cli.record_dataset:
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


def _discover_cameras(env) -> dict[str, dict[str, int]]:
    cameras = {}
    for scene_key in env.unwrapped.scene.keys():
        if not scene_key.startswith("camera_"):
            continue
        camera_cfg = getattr(env.unwrapped.scene.cfg, scene_key)
        camera_name = scene_key.removeprefix("camera_")
        cameras[camera_name] = {"height": camera_cfg.height, "width": camera_cfg.width}
        print(f"[INFO]: Found camera '{camera_name}' ({camera_cfg.width}x{camera_cfg.height})")
    return cameras


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
    env_ids = _env_ids_tensor(env, env_ids)
    mark_benchmark_robot_start(
        env.unwrapped,
        object_asset_names=object_asset_names,
        bin_name=BIN_NAME,
        env_ids=env_ids,
        force_robot_start_time=True,
    )
    active = getattr(env.unwrapped, "_so101_policy_control_active", None)
    if not isinstance(active, torch.Tensor) or tuple(active.shape) != (env.unwrapped.num_envs,):
        active = torch.zeros(
            env.unwrapped.num_envs,
            dtype=torch.bool,
            device=env.unwrapped.device,
        )
    active[env_ids] = True
    env.unwrapped._so101_policy_control_active = active


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


def _numpy_vector(value: Any, *, length: int, fill: float = np.nan) -> np.ndarray:
    """Best-effort conversion of an Isaac value to one fixed-size float vector."""
    if value is None:
        return np.full(length, fill, dtype=np.float32)
    try:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return np.full(length, fill, dtype=np.float32)
    if array.size < length:
        result = np.full(length, fill, dtype=np.float32)
        result[: array.size] = array
        return result
    return array[:length].astype(np.float32, copy=False)


def _asset_data_vector(
    asset: Any,
    names: tuple[str, ...],
    *,
    env_id: int,
    length: int,
) -> tuple[np.ndarray, bool]:
    """Read the first available per-env asset-data vector without assuming an Isaac version."""
    data = getattr(asset, "data", None)
    for name in names:
        value = getattr(data, name, None) if data is not None else None
        if value is None:
            continue
        try:
            row = value[env_id]
            vector = _numpy_vector(row, length=length)
            if np.any(np.isfinite(vector)):
                return vector, True
        except (IndexError, TypeError, RuntimeError):
            continue
    return np.full(length, np.nan, dtype=np.float32), False


def _object_physics_state(
    env,
    object_asset_names: list[str],
    *,
    env_id: int,
) -> dict[str, np.ndarray]:
    """Return fixed-shape object root state with validity masks for heterogeneous assets."""
    unwrapped = env.unwrapped
    positions = benchmark_object_positions(unwrapped, object_asset_names)[env_id].detach().cpu().numpy()
    yaws = benchmark_object_yaws(unwrapped, object_asset_names)[env_id].detach().cpu().numpy()
    count = len(object_asset_names)
    quaternions = np.full((count, 4), np.nan, dtype=np.float32)
    quaternion_valid = np.zeros(count, dtype=np.bool_)
    linear_velocity = np.full((count, 3), np.nan, dtype=np.float32)
    angular_velocity = np.full((count, 3), np.nan, dtype=np.float32)
    velocity_valid = np.zeros(count, dtype=np.bool_)

    for object_id, asset_name in enumerate(object_asset_names):
        asset = unwrapped.scene[asset_name]
        quat, quat_valid = _asset_data_vector(
            asset,
            ("root_quat_w", "root_quat_wxyz"),
            env_id=env_id,
            length=4,
        )
        if not quat_valid:
            # Xform-backed/multi-rigid objects expose world poses rather than a
            # RigidObjectData root quaternion.  It is safe to try and harmless to
            # fall back to the already-supported yaw representation.
            try:
                _pos, world_quat = asset.get_world_poses()
                quat = _numpy_vector(world_quat[env_id], length=4)
                quat_valid = bool(np.all(np.isfinite(quat)))
            except (AttributeError, IndexError, TypeError, RuntimeError):
                quat_valid = False
        if not quat_valid and np.isfinite(yaws[object_id]):
            half_yaw = 0.5 * float(yaws[object_id])
            quat = np.asarray([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], dtype=np.float32)
        quaternions[object_id] = quat
        quaternion_valid[object_id] = quat_valid

        lin_vel, lin_valid = _asset_data_vector(
            asset,
            ("root_lin_vel_w",),
            env_id=env_id,
            length=3,
        )
        ang_vel, ang_valid = _asset_data_vector(
            asset,
            ("root_ang_vel_w",),
            env_id=env_id,
            length=3,
        )
        if not (lin_valid and ang_valid):
            root_velocity, root_velocity_valid = _asset_data_vector(
                asset,
                ("root_vel_w",),
                env_id=env_id,
                length=6,
            )
            if root_velocity_valid:
                if not lin_valid:
                    lin_vel, lin_valid = root_velocity[:3], True
                if not ang_valid:
                    ang_vel, ang_valid = root_velocity[3:6], True
        linear_velocity[object_id] = lin_vel
        angular_velocity[object_id] = ang_vel
        velocity_valid[object_id] = lin_valid and ang_valid

    return {
        "position": positions.astype(np.float32, copy=False),
        "yaw": yaws.astype(np.float32, copy=False),
        "quaternion": quaternions,
        "quaternion_valid": quaternion_valid,
        "linear_velocity": linear_velocity,
        "angular_velocity": angular_velocity,
        "velocity_valid": velocity_valid,
    }


def _object_contact_evidence(
    env,
    object_asset_names: list[str],
    *,
    env_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate contact force evidence per object across split contact sensors."""
    max_force = np.zeros(len(object_asset_names), dtype=np.float32)
    available = np.zeros(len(object_asset_names), dtype=np.bool_)
    sensors = getattr(env.unwrapped.scene, "sensors", {})
    for object_id, asset_name in enumerate(object_asset_names):
        exact_name = f"{asset_name}_contacts"
        split_prefix = f"{asset_name}_"
        for sensor_name, sensor in sensors.items():
            if sensor_name != exact_name and not (
                sensor_name.startswith(split_prefix) and sensor_name.endswith("_contacts")
            ):
                continue
            force_matrix = getattr(getattr(sensor, "data", None), "force_matrix_w", None)
            if force_matrix is None:
                continue
            try:
                forces = force_matrix[env_id]
                if not isinstance(forces, torch.Tensor):
                    forces = torch.as_tensor(forces)
                magnitude = torch.linalg.vector_norm(forces, dim=-1)
                max_force[object_id] = max(max_force[object_id], float(torch.max(magnitude).item()))
                available[object_id] = True
            except (IndexError, TypeError, RuntimeError):
                continue
    return max_force, available


def _contact_channel_catalog(env, object_asset_names: list[str]) -> list[dict[str, Any]]:
    """Describe filtered object-object contact channels and their pair identities."""
    unwrapped = env.unwrapped
    cache_key = tuple(object_asset_names)
    cached = getattr(unwrapped, "_so101_collector_contact_channel_catalog", None)
    if isinstance(cached, tuple) and cached[0] == cache_key:
        return cached[1]

    sensors = getattr(unwrapped.scene, "sensors", {})
    catalog: list[dict[str, Any]] = []
    for source_object_id, asset_name in enumerate(object_asset_names):
        exact_name = f"{asset_name}_contacts"
        split_prefix = f"{asset_name}_"
        matching_sensor_names = sorted(
            sensor_name
            for sensor_name in sensors
            if sensor_name == exact_name
            or (sensor_name.startswith(split_prefix) and sensor_name.endswith("_contacts"))
        )
        for sensor_name in matching_sensor_names:
            sensor = sensors[sensor_name]
            filter_paths = getattr(getattr(sensor, "cfg", None), "filter_prim_paths_expr", None) or []
            if isinstance(filter_paths, str):
                filter_paths = [filter_paths]
            for filter_index, filter_path in enumerate(filter_paths):
                filter_path = str(filter_path)
                target_object_id = -1
                marker = "/Object_"
                if marker in filter_path:
                    suffix = filter_path.split(marker, maxsplit=1)[1]
                    digits = ""
                    for character in suffix:
                        if not character.isdigit():
                            break
                        digits += character
                    if digits:
                        target_object_id = int(digits) - 1
                catalog.append(
                    {
                        "channel_id": len(catalog),
                        "sensor_name": sensor_name,
                        "filter_index": filter_index,
                        "filter_prim_path": filter_path,
                        "source_object_id": source_object_id,
                        "target_object_id": target_object_id,
                    }
                )
    unwrapped._so101_collector_contact_channel_catalog = (cache_key, catalog)
    return catalog


def _contact_channel_evidence(
    env,
    object_asset_names: list[str],
    *,
    env_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return max force magnitude for each catalogued source/filter contact pair."""
    catalog = _contact_channel_catalog(env, object_asset_names)
    force = np.zeros(len(catalog), dtype=np.float32)
    available = np.zeros(len(catalog), dtype=np.bool_)
    if not catalog:
        return force, available
    sensors = getattr(env.unwrapped.scene, "sensors", {})
    channels_by_sensor: dict[str, list[dict[str, Any]]] = {}
    for channel in catalog:
        channels_by_sensor.setdefault(channel["sensor_name"], []).append(channel)
    for sensor_name, channels in channels_by_sensor.items():
        force_matrix = getattr(getattr(sensors[sensor_name], "data", None), "force_matrix_w", None)
        if force_matrix is None:
            continue
        try:
            row = force_matrix[env_id]
            if not isinstance(row, torch.Tensor):
                row = torch.as_tensor(row)
            magnitude = torch.linalg.vector_norm(row, dim=-1)
            filter_count = len(channels)
            exact_attribution = False
            if magnitude.ndim == 0:
                per_filter = magnitude.reshape(1).repeat(filter_count)
            elif magnitude.shape[-1] == filter_count:
                reduce_dims = tuple(range(magnitude.ndim - 1))
                per_filter = torch.amax(magnitude, dim=reduce_dims) if reduce_dims else magnitude
                exact_attribution = True
            elif magnitude.numel() == filter_count:
                per_filter = magnitude.reshape(filter_count)
                exact_attribution = True
            else:
                # Isaac contact tensor layouts have changed across releases.  A
                # sensor aggregate is still truthful; repeat it while the catalog
                # marks that filter attribution was unavailable.
                per_filter = torch.max(magnitude).reshape(1).repeat(filter_count)
            for local_index, channel in enumerate(channels):
                channel_id = int(channel["channel_id"])
                force[channel_id] = float(per_filter[local_index].item())
                available[channel_id] = exact_attribution
        except (IndexError, KeyError, TypeError, RuntimeError):
            continue
    return force, available


def _env_scalar(unwrapped: Any, attr_name: str, env_id: int, default: int | float | bool) -> Any:
    value = getattr(unwrapped, attr_name, None)
    if value is None:
        return default
    try:
        value = value[env_id]
        if isinstance(value, torch.Tensor):
            return value.item()
        if isinstance(value, np.generic):
            return value.item()
        return value
    except (IndexError, TypeError, RuntimeError, ValueError):
        return default


def _scene_state(env, object_asset_names: list[str], object_labels: list[str], *, env_id: int = 0) -> dict[str, Any]:
    unwrapped = env.unwrapped
    physics = _object_physics_state(env, object_asset_names, env_id=env_id)
    object_pos = physics["position"]
    object_yaw = physics["yaw"]
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
                "quaternion_wxyz": _tensor_list(physics["quaternion"][object_id]),
                "quaternion_fully_observed": bool(physics["quaternion_valid"][object_id]),
                "yaw": float(object_yaw[object_id]),
                "linear_velocity_w": _tensor_list(physics["linear_velocity"][object_id]),
                "angular_velocity_w": _tensor_list(physics["angular_velocity"][object_id]),
                "velocity_observed": bool(physics["velocity_valid"][object_id]),
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
        "grasped_object_id": int(_env_scalar(unwrapped, "_so101_grasped_object_ids", env_id, -1)),
        "max_object_lift_m": _tensor_list(
            getattr(
                unwrapped,
                "_so101_max_object_lift",
                torch.zeros((unwrapped.num_envs, len(object_asset_names)), device=unwrapped.device),
            )[env_id]
        ),
    }


def _diagnostic_call_kwargs(
    object_asset_names: list[str],
    success_params: dict[str, Any],
    failure_params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "object_asset_names": object_asset_names,
        "bin_name": success_params["bin_name"],
        "table_bounds": success_params.get("table_bounds"),
        "success_min_episode_time_s": success_params.get("min_episode_time_s", 5.0),
        "confirm_time_s": success_params.get("confirm_time_s", 3.0),
        "move_straightness_tolerance": success_params.get("move_straightness_tolerance", 0.04445),
        "move_past_boundary_tolerance": failure_params.get("move_past_boundary_tolerance", 0.0254),
        "failure_min_episode_time_s": failure_params.get("min_episode_time_s", 5.0),
        "max_grasp_attempts": failure_params.get("max_grasp_attempts", 3),
        "enforce_max_grasp_attempts": failure_params.get("enforce_max_grasp_attempts", True),
        "bin_displacement_limit": failure_params.get("bin_displacement_limit", 0.0254),
        "non_target_displacement_limit": failure_params.get("non_target_displacement_limit", 0.0127),
        "boundary_displacement_limit": failure_params.get("boundary_displacement_limit", 0.0127),
        "contact_grace_time_s": failure_params.get(
            "contact_grace_time_s",
            success_params.get("contact_grace_time_s", 3.0),
        ),
    }


def _optional_task_condition_metrics(
    env,
    *,
    env_id: int,
    kwargs: dict[str, Any],
) -> Any:
    """Use a richer side-effect-free numeric condition API when the MDP exports one."""
    metrics_fn = getattr(so101_mdp, "task_condition_metrics", None)
    if not callable(metrics_fn):
        return None
    try:
        signature = inspect.signature(metrics_fn)
        supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
        metrics = metrics_fn(env.unwrapped, **supported)
        if isinstance(metrics, (list, tuple)):
            return metrics[env_id] if env_id < len(metrics) else None
        if isinstance(metrics, dict) and env_id in metrics:
            return metrics[env_id]
        return metrics
    except (AttributeError, IndexError, TypeError, RuntimeError, ValueError):
        # Numeric metrics are an optional compatibility path.  The standard
        # diagnostic snapshot and raw state/counters below remain authoritative.
        return None


def _condition_trace(
    env,
    object_asset_names: list[str],
    *,
    env_id: int,
    control_dt: float,
    success_params: dict[str, Any],
    failure_params: dict[str, Any],
) -> dict[str, Any]:
    """Capture typed condition state without advancing any termination counters."""
    unwrapped = env.unwrapped
    kwargs = _diagnostic_call_kwargs(object_asset_names, success_params, failure_params)
    snapshots = task_condition_diagnostics(unwrapped, **kwargs)
    snapshot = snapshots[env_id] if env_id < len(snapshots) else None
    snapshot_payload = _serialize_param_value(snapshot) if snapshot is not None else {}
    condition_payloads = list(snapshot_payload.get("conditions", [])) if isinstance(snapshot_payload, dict) else []
    live_success_confirmed = any(
        str(condition.get("kind", "")) == "success" and bool(condition.get("met", False))
        for condition in condition_payloads
    )

    counter_values = np.asarray(
        [int(_env_scalar(unwrapped, attr_name, env_id, -1)) for _name, attr_name in CONDITION_COUNTER_SPECS],
        dtype=np.int32,
    )
    task_family = str(
        snapshot_payload.get("task_family", "unknown")
        if isinstance(snapshot_payload, dict)
        else "unknown"
    )
    success_counter_attr = {
        "bin": "_so101_bin_success_counter",
        "named_bin": "_so101_bin_success_counter",
        "next_to": "_so101_next_to_success_counter",
        "between": "_so101_between_success_counter",
        "move": "_so101_move_success_counter",
    }.get(task_family)
    success_counter = int(
        _env_scalar(unwrapped, success_counter_attr, env_id, -1) if success_counter_attr is not None else -1
    )
    confirm_time_s = float(success_params.get("confirm_time_s", 3.0))
    success_required_steps = max(1, math.ceil(confirm_time_s / max(control_dt, 1.0e-9)))
    episode_age_s = float(snapshot_payload.get("episode_age_s", 0.0)) if isinstance(snapshot_payload, dict) else 0.0
    numeric_metrics = _serialize_param_value(
        _optional_task_condition_metrics(env, env_id=env_id, kwargs=kwargs)
    )
    goal_metrics = numeric_metrics.get("goal", {}) if isinstance(numeric_metrics, dict) else {}

    def numeric_goal_value(name: str) -> float:
        value = goal_metrics.get(name) if isinstance(goal_metrics, dict) else None
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else math.nan

    return {
        "condition_names": np.asarray(
            [str(condition.get("name", "unknown")) for condition in condition_payloads],
            dtype=np.str_,
        ),
        "condition_kinds": np.asarray(
            [str(condition.get("kind", "unknown")) for condition in condition_payloads],
            dtype=np.str_,
        ),
        "condition_met": np.asarray(
            [bool(condition.get("met", False)) for condition in condition_payloads],
            dtype=np.bool_,
        ),
        "condition_snapshot": snapshot_payload,
        "numeric_metrics": numeric_metrics,
        "goal_metric_name": str(goal_metrics.get("name", "unknown")),
        "goal_value": numeric_goal_value("value"),
        "goal_threshold": numeric_goal_value("threshold"),
        "goal_margin": numeric_goal_value("margin"),
        "goal_instant": bool(goal_metrics.get("instant", False)),
        "goal_overshot": bool(numeric_metrics.get("overshot", False)) if isinstance(numeric_metrics, dict) else False,
        "inside_bin_object_count": int(
            len(numeric_metrics.get("inside_bin_object_ids", []))
            if isinstance(numeric_metrics, dict)
            else 0
        ),
        "counter_values": counter_values,
        "success_counter": success_counter,
        "success_required_steps": success_required_steps,
        "success_candidate": success_counter > 0,
        "live_success_confirmed": live_success_confirmed,
        "success_age_ready": episode_age_s >= float(success_params.get("min_episode_time_s", 5.0)),
        "failure_age_ready": episode_age_s >= float(failure_params.get("min_episode_time_s", 5.0)),
        "failure_baseline_recorded": bool(
            _env_scalar(unwrapped, "_so101_failure_baseline_recorded", env_id, False)
        ),
        "timeout_confirmation_active": bool(
            _env_scalar(unwrapped, "_so101_timeout_success_confirmation_active", env_id, False)
        ),
        "timeout_confirmation_failed": bool(
            _env_scalar(unwrapped, "_so101_timeout_success_confirmation_failed", env_id, False)
        ),
    }


def _trajectory_sample(
    env,
    object_asset_names: list[str],
    *,
    step: int,
    time_s: float,
    frame_index: int,
    action_frame_index: int,
    action_phase: str,
    dataset_fps: float,
    action_lerobot_raw: torch.Tensor,
    action_lerobot: torch.Tensor,
    action_sim: torch.Tensor,
    action_clamped_mask: torch.Tensor,
    action_delta_lerobot: torch.Tensor,
    term_eval: TermEval,
    control_dt: float,
    success_params: dict[str, Any],
    failure_params: dict[str, Any],
    final_scoring_override_applied: bool,
    env_id: int = 0,
) -> dict[str, Any]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    joint_ids = [robot.joint_names.index(joint_name) for joint_name in ACTION_JOINT_NAMES]
    ee_frame = unwrapped.scene["ee_frame"]
    bin_asset = unwrapped.scene[BIN_NAME]
    physics = _object_physics_state(env, object_asset_names, env_id=env_id)
    object_pos = physics["position"]
    object_count = len(object_asset_names)
    active = _active_mask(env, object_asset_names)[env_id].detach().cpu().numpy().astype(np.bool_)
    target_object_id = int(_env_scalar(unwrapped, "_so101_target_object_ids", env_id, 0))

    initial_pos_value = getattr(unwrapped, "_so101_initial_object_pos_w", None)
    failure_pos_value = getattr(unwrapped, "_so101_failure_object_pos_w", None)
    failure_baseline_recorded = bool(
        _env_scalar(unwrapped, "_so101_failure_baseline_recorded", env_id, False)
    )
    initial_pos = (
        initial_pos_value[env_id].detach().cpu().numpy()
        if isinstance(initial_pos_value, torch.Tensor)
        else object_pos.copy()
    )
    failure_baseline_pos = (
        failure_pos_value[env_id].detach().cpu().numpy()
        if failure_baseline_recorded and isinstance(failure_pos_value, torch.Tensor)
        else initial_pos.copy()
    )
    displacement_from_initial = object_pos - initial_pos
    displacement_from_failure_baseline = object_pos - failure_baseline_pos
    table_bounds = success_params.get("table_bounds") or {"x": (0.08, 0.45), "y": (-0.20, 0.20)}
    object_out_of_table_xy_bounds = (
        (object_pos[:, 0] < float(table_bounds["x"][0]))
        | (object_pos[:, 0] > float(table_bounds["x"][1]))
        | (object_pos[:, 1] < float(table_bounds["y"][0]))
        | (object_pos[:, 1] > float(table_bounds["y"][1]))
    )
    object_fell_below_baseline = object_pos[:, 2] < (failure_baseline_pos[:, 2] - 0.05)
    max_object_lift_value = getattr(unwrapped, "_so101_max_object_lift", None)
    max_object_lift = (
        max_object_lift_value[env_id].detach().cpu().numpy()
        if isinstance(max_object_lift_value, torch.Tensor)
        else np.maximum(displacement_from_failure_baseline[:, 2], 0.0)
    )

    ee_pos = ee_frame.data.target_pos_w[env_id, 0, :].detach().cpu().numpy().astype(np.float32)
    ee_quat_value = getattr(ee_frame.data, "target_quat_w", None)
    ee_quat = (
        _numpy_vector(ee_quat_value[env_id, 0, :], length=4)
        if ee_quat_value is not None
        else np.full(4, np.nan, dtype=np.float32)
    )
    ee_lin_vel, ee_lin_vel_valid = _asset_data_vector(
        ee_frame,
        ("target_lin_vel_w", "target_linear_velocity_w"),
        env_id=env_id,
        length=3,
    )
    ee_ang_vel, ee_ang_vel_valid = _asset_data_vector(
        ee_frame,
        ("target_ang_vel_w", "target_angular_velocity_w"),
        env_id=env_id,
        length=3,
    )
    ee_object_distance = np.linalg.norm(object_pos - ee_pos.reshape(1, 3), axis=1).astype(np.float32)
    active_distance = np.where(active, ee_object_distance, np.inf)
    nearest_object_id = int(np.argmin(active_distance)) if np.any(active) else -1
    nearest_object_distance = float(active_distance[nearest_object_id]) if nearest_object_id >= 0 else math.inf

    object_contact_force, object_contact_available = _object_contact_evidence(
        env,
        object_asset_names,
        env_id=env_id,
    )
    contact_channel_force, contact_channel_attribution_available = _contact_channel_evidence(
        env,
        object_asset_names,
        env_id=env_id,
    )
    grasp_attempt_counts_value = getattr(unwrapped, "_so101_grasp_attempt_counts", None)
    grasp_attempt_counts = (
        grasp_attempt_counts_value[env_id].detach().cpu().numpy().astype(np.int32)
        if isinstance(grasp_attempt_counts_value, torch.Tensor)
        else np.zeros(object_count, dtype=np.int32)
    )

    def postmortem_object_values(
        attr_name: str,
        dtype: Any,
        default_value: int | float | bool,
    ) -> np.ndarray:
        # A reused lane's temporal tracker row is reset lazily on its first
        # evaluated step.  The reset-only telemetry sample must never inherit
        # the preceding episode's behavioral evidence.
        if action_phase == "reset":
            return np.full(object_count, default_value, dtype=dtype)
        value = getattr(unwrapped, attr_name, None)
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and env_id < value.shape[0]:
            row = value[env_id].detach().cpu().numpy()
            if row.shape == (object_count,):
                return row.astype(dtype)
        return np.full(object_count, default_value, dtype=dtype)

    postmortem_attempt_counts_all = postmortem_object_values(
        "_so101_pm_attempt_counts_all", np.int32, 0
    )
    postmortem_acquisition_counts = postmortem_object_values(
        "_so101_pm_acquisition_counts", np.int32, 0
    )
    postmortem_acquired_objects = postmortem_object_values(
        "_so101_pm_acquired_objects", np.bool_, False
    )
    postmortem_manipulated_objects = postmortem_object_values(
        "_so101_pm_manipulated_objects", np.bool_, False
    )
    postmortem_drop_counts = postmortem_object_values(
        "_so101_pm_drop_counts", np.int32, 0
    )
    postmortem_release_counts = postmortem_object_values(
        "_so101_pm_release_counts", np.int32, 0
    )
    postmortem_associated_transport_m = postmortem_object_values(
        "_so101_pm_associated_transport_m", np.float32, 0.0
    )
    postmortem_post_acquisition_transport_m = postmortem_object_values(
        "_so101_pm_post_acquisition_transport_m", np.float32, 0.0
    )
    postmortem_max_interaction_segment_transport_m = postmortem_object_values(
        "_so101_pm_max_interaction_segment_transport_m", np.float32, 0.0
    )

    bin_pos = bin_asset.data.root_pos_w[env_id].detach().cpu().numpy().astype(np.float32)
    bin_quat = bin_asset.data.root_quat_w[env_id].detach().cpu().numpy().astype(np.float32)
    bin_lin_vel, bin_lin_vel_valid = _asset_data_vector(
        bin_asset,
        ("root_lin_vel_w",),
        env_id=env_id,
        length=3,
    )
    bin_ang_vel, bin_ang_vel_valid = _asset_data_vector(
        bin_asset,
        ("root_ang_vel_w",),
        env_id=env_id,
        length=3,
    )
    initial_bin_value = getattr(unwrapped, "_so101_initial_bin_pos_w", None)
    failure_bin_value = getattr(unwrapped, "_so101_failure_bin_pos_w", None)
    initial_bin_pos = (
        initial_bin_value[env_id].detach().cpu().numpy()
        if isinstance(initial_bin_value, torch.Tensor)
        else bin_pos
    )
    failure_bin_pos = (
        failure_bin_value[env_id].detach().cpu().numpy()
        if failure_baseline_recorded and isinstance(failure_bin_value, torch.Tensor)
        else initial_bin_pos
    )

    joint_pos = robot.data.joint_pos[env_id, joint_ids].detach().cpu().numpy().astype(np.float32)
    joint_vel = robot.data.joint_vel[env_id, joint_ids].detach().cpu().numpy().astype(np.float32)
    effort_source = getattr(robot.data, "applied_torque", None)
    if effort_source is None:
        effort_source = getattr(robot.data, "computed_torque", None)
    joint_effort = (
        effort_source[env_id, joint_ids].detach().cpu().numpy().astype(np.float32)
        if isinstance(effort_source, torch.Tensor)
        else np.full(len(joint_ids), np.nan, dtype=np.float32)
    )
    joint_limits = robot.data.joint_pos_limits[env_id, joint_ids].detach().cpu().numpy().astype(np.float32)
    joint_limit_margin = np.minimum(joint_pos - joint_limits[:, 0], joint_limits[:, 1] - joint_pos)
    action_sim_np = action_sim.detach().cpu().numpy().astype(np.float32)
    condition_trace = _condition_trace(
        env,
        object_asset_names,
        env_id=env_id,
        control_dt=control_dt,
        success_params=success_params,
        failure_params=failure_params,
    )
    non_target_ids = [
        object_id for object_id in np.flatnonzero(active).tolist() if object_id != target_object_id
    ]
    max_non_target_displacement = max(
        (
            float(np.linalg.norm(displacement_from_failure_baseline[object_id, :2]))
            for object_id in non_target_ids
        ),
        default=0.0,
    )
    boundary_object_id = int(_env_scalar(unwrapped, "_so101_move_boundary_ids", env_id, -1))
    boundary_displacement = (
        float(np.linalg.norm(displacement_from_failure_baseline[boundary_object_id, :2]))
        if 0 <= boundary_object_id < object_count
        else math.nan
    )
    numeric_goal = (
        condition_trace["numeric_metrics"].get("goal", {})
        if isinstance(condition_trace["numeric_metrics"], dict)
        else {}
    )

    def goal_evidence_float(name: str) -> float:
        value = numeric_goal.get(name) if isinstance(numeric_goal, dict) else None
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else math.nan

    postmortem_payload: Any = None
    if action_phase != "reset":
        cached_postmortem = getattr(unwrapped, "_so101_postmortem_failure_diagnostics", None)
        if isinstance(cached_postmortem, (list, tuple)) and env_id < len(cached_postmortem):
            postmortem_payload = _serialize_param_value(cached_postmortem[env_id])
        elif isinstance(cached_postmortem, dict):
            postmortem_payload = _serialize_param_value(
                cached_postmortem.get(env_id, cached_postmortem.get(str(env_id)))
            )
    postmortem_type = (
        postmortem_payload.get("failure_type", "unavailable")
        if isinstance(postmortem_payload, dict)
        else "unavailable"
    )
    postmortem_confidence = (
        postmortem_payload.get("confidence")
        if isinstance(postmortem_payload, dict)
        else None
    )
    postmortem_version = (
        postmortem_payload.get("classification_version")
        if isinstance(postmortem_payload, dict)
        else None
    )

    return {
        "step": step,
        "time_s": time_s,
        "dataset_frames_played": frame_index,
        "dataset_action_frame_index": action_frame_index,
        "dataset_action_timestamp_s": (
            action_frame_index / max(dataset_fps, 1.0e-9) if action_frame_index >= 0 else np.nan
        ),
        "action_phase": action_phase,
        "action_phase_id": ACTION_PHASE_IDS.get(action_phase, -1),
        # The reset sample is an observation of the exact initial scene, not a
        # call to the termination functions.  Offline rescorers must skip it or
        # every temporal counter/classifier history is advanced by one frame.
        "is_reset_sample": action_phase == "reset",
        "termination_evaluation_applied": action_phase != "reset",
        "policy_control_active": bool(
            _env_scalar(unwrapped, "_so101_policy_control_active", env_id, False)
        ),
        "robot_control_started": bool(
            _env_scalar(unwrapped, "_so101_robot_started_moving", env_id, False)
        ),
        "robot_control_start_step": int(
            _env_scalar(unwrapped, "_so101_robot_start_step", env_id, -1)
        ),
        "robot_control_start_time_s": float(
            _env_scalar(unwrapped, "_so101_robot_start_time_s", env_id, math.nan)
        ),
        "object_pos_w": object_pos,
        "object_quat_wxyz": physics["quaternion"],
        "object_quat_fully_observed": physics["quaternion_valid"],
        "object_yaw": physics["yaw"],
        "object_lin_vel_w": physics["linear_velocity"],
        "object_ang_vel_w": physics["angular_velocity"],
        "object_velocity_observed": physics["velocity_valid"],
        "object_active": active,
        "object_displacement_from_initial_w": displacement_from_initial.astype(np.float32),
        "object_failure_baseline_pos_w": failure_baseline_pos.astype(np.float32),
        "object_displacement_from_failure_baseline_w": displacement_from_failure_baseline.astype(np.float32),
        "object_lift_from_failure_baseline": displacement_from_failure_baseline[:, 2].astype(np.float32),
        "object_out_of_table_xy_bounds": object_out_of_table_xy_bounds.astype(np.bool_),
        "object_fell_below_baseline": object_fell_below_baseline.astype(np.bool_),
        "max_object_lift": np.asarray(max_object_lift, dtype=np.float32),
        "object_contact_force_max": object_contact_force,
        "object_contact_impulse_step": (object_contact_force * control_dt).astype(np.float32),
        "object_contact_force_available": object_contact_available,
        "contact_channel_force_max": contact_channel_force,
        "contact_channel_impulse_step": (contact_channel_force * control_dt).astype(np.float32),
        "contact_channel_attribution_available": contact_channel_attribution_available,
        "bin_pos_w": bin_pos,
        "bin_quat_wxyz": bin_quat,
        "bin_yaw": float(_quat_yaw(bin_asset.data.root_quat_w[env_id]).item()),
        "bin_lin_vel_w": bin_lin_vel,
        "bin_ang_vel_w": bin_ang_vel,
        "bin_velocity_observed": bin_lin_vel_valid and bin_ang_vel_valid,
        "bin_displacement_from_initial_w": (bin_pos - initial_bin_pos).astype(np.float32),
        "bin_failure_baseline_pos_w": np.asarray(failure_bin_pos, dtype=np.float32),
        "bin_displacement_from_failure_baseline_w": (bin_pos - failure_bin_pos).astype(np.float32),
        "grasped_object_made_contact": bool(
            grasped_object_made_contact(unwrapped, object_asset_names)[env_id].item()
        ),
        "grasped_object_contact_steps": int(
            _env_scalar(unwrapped, "_so101_grasped_object_contact_steps", env_id, 0)
        ),
        "grasped_object_id": int(_env_scalar(unwrapped, "_so101_grasped_object_ids", env_id, -1)),
        "grasp_attempt_counts": grasp_attempt_counts,
        # The legacy counter above intentionally excludes distractors for
        # instruction tasks because it drives a benchmark rule.  These typed
        # postmortem channels cover every active object and preserve the
        # evidence used by the behavioral classifier.
        "postmortem_attempt_counts_all": postmortem_attempt_counts_all,
        "postmortem_unassociated_attempt_count": int(
            0
            if action_phase == "reset"
            else _env_scalar(unwrapped, "_so101_pm_unassociated_attempt_counts", env_id, 0)
        ),
        "postmortem_acquisition_counts": postmortem_acquisition_counts,
        "postmortem_acquired_objects": postmortem_acquired_objects,
        "postmortem_manipulated_objects": postmortem_manipulated_objects,
        "postmortem_drop_counts": postmortem_drop_counts,
        "postmortem_release_counts": postmortem_release_counts,
        "postmortem_associated_transport_m": postmortem_associated_transport_m,
        "postmortem_post_acquisition_transport_m": postmortem_post_acquisition_transport_m,
        "postmortem_max_interaction_segment_transport_m": (
            postmortem_max_interaction_segment_transport_m
        ),
        "postmortem_current_candidate_object_id": int(
            -1
            if action_phase == "reset"
            else _env_scalar(unwrapped, "_so101_pm_candidate_object_ids", env_id, -1)
        ),
        "postmortem_current_candidate_acquired": bool(
            False
            if action_phase == "reset"
            else _env_scalar(unwrapped, "_so101_pm_candidate_acquired", env_id, False)
        ),
        "nearest_ee_object_id": nearest_object_id,
        "nearest_ee_object_distance": nearest_object_distance,
        "ee_object_distance": ee_object_distance,
        "ee_pos_w": ee_pos,
        "ee_quat_wxyz": ee_quat,
        "ee_lin_vel_w": ee_lin_vel,
        "ee_ang_vel_w": ee_ang_vel,
        "ee_velocity_observed": ee_lin_vel_valid and ee_ang_vel_valid,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "joint_effort": joint_effort,
        "joint_effort_observed": effort_source is not None,
        "joint_limit_margin": joint_limit_margin.astype(np.float32),
        "action_lerobot_raw": action_lerobot_raw.detach().cpu().numpy().astype(np.float32),
        "action_lerobot": action_lerobot.detach().cpu().numpy(),
        "action_lerobot_clamped": action_lerobot.detach().cpu().numpy().astype(np.float32),
        "action_clamped_mask": action_clamped_mask.detach().cpu().numpy().astype(np.bool_),
        "action_delta_lerobot": action_delta_lerobot.detach().cpu().numpy().astype(np.float32),
        "action_sim": action_sim_np,
        "action_tracking_error": (joint_pos - action_sim_np).astype(np.float32),
        "condition_names": condition_trace["condition_names"],
        "condition_kinds": condition_trace["condition_kinds"],
        "condition_met": condition_trace["condition_met"],
        "condition_snapshot_json": _compact_json(condition_trace["condition_snapshot"]),
        "condition_numeric_metrics_json": _compact_json(condition_trace["numeric_metrics"]),
        "condition_numeric_metrics_available": isinstance(condition_trace["numeric_metrics"], dict),
        "goal_metric_name": condition_trace["goal_metric_name"],
        "goal_value": condition_trace["goal_value"],
        "goal_threshold": condition_trace["goal_threshold"],
        "goal_margin": condition_trace["goal_margin"],
        "goal_instant": condition_trace["goal_instant"],
        "goal_overshot": condition_trace["goal_overshot"],
        "inside_bin_object_count": condition_trace["inside_bin_object_count"],
        "condition_counter_values": condition_trace["counter_values"],
        "failure_raw_max_grasp_attempt_count": int(np.max(grasp_attempt_counts)) if object_count else 0,
        "failure_threshold_max_grasp_attempts": int(failure_params.get("max_grasp_attempts", 3)),
        "failure_raw_bin_displacement_xy_m": float(np.linalg.norm(bin_pos[:2] - failure_bin_pos[:2])),
        "failure_threshold_bin_displacement_m": float(failure_params.get("bin_displacement_limit", 0.0254)),
        "failure_raw_max_non_target_displacement_xy_m": max_non_target_displacement,
        "failure_threshold_non_target_displacement_m": float(
            failure_params.get("non_target_displacement_limit", 0.0127)
        ),
        "failure_raw_boundary_root_displacement_xy_m": boundary_displacement,
        "failure_threshold_boundary_displacement_m": float(
            failure_params.get("boundary_displacement_limit", 0.0127)
        ),
        "failure_raw_contact_duration_s": float(
            _env_scalar(unwrapped, "_so101_grasped_object_contact_steps", env_id, 0) * control_dt
        ),
        "failure_threshold_contact_duration_s": float(
            failure_params.get("contact_grace_time_s", success_params.get("contact_grace_time_s", 5.0))
        ),
        "failure_raw_move_boundary_gap_m": goal_evidence_float("boundary_gap_m"),
        "failure_raw_move_lateral_error_m": goal_evidence_float("lateral_error_m"),
        "postmortem_diagnostic_json": _compact_json(postmortem_payload),
        "postmortem_failure_type": str(postmortem_type),
        "postmortem_confidence": (
            float(postmortem_confidence)
            if isinstance(postmortem_confidence, (int, float))
            else math.nan
        ),
        "postmortem_classification_version": (
            int(postmortem_version)
            if isinstance(postmortem_version, (int, float))
            else -1
        ),
        "success_counter": condition_trace["success_counter"],
        "success_required_steps": condition_trace["success_required_steps"],
        "success_candidate": condition_trace["success_candidate"],
        # This is the standard live rule, independent of any final-only
        # confirmation waiver used to assign the selected episode label.
        "live_success_confirmed": condition_trace["live_success_confirmed"],
        "final_scoring_override_applied": bool(final_scoring_override_applied),
        "success_age_ready": condition_trace["success_age_ready"],
        "failure_age_ready": condition_trace["failure_age_ready"],
        "failure_baseline_recorded": failure_baseline_recorded,
        "timeout_confirmation_active": condition_trace["timeout_confirmation_active"],
        "timeout_confirmation_failed": condition_trace["timeout_confirmation_failed"],
        "success": term_eval.success,
        "failure": term_eval.failure,
        "timed_out": term_eval.timed_out,
        "terminal_reason": term_eval.reason,
    }


def _write_trajectory(path: Path, samples: list[dict[str, Any]]) -> None:
    if not samples:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    time_s = np.asarray([sample["time_s"] for sample in samples], dtype=np.float32)

    def stack(name: str, dtype: Any = np.float32) -> np.ndarray:
        return np.stack([sample[name] for sample in samples]).astype(dtype)

    def values(name: str, dtype: Any) -> np.ndarray:
        return np.asarray([sample[name] for sample in samples], dtype=dtype)

    def finite_difference(array: np.ndarray) -> np.ndarray:
        result = np.zeros_like(array, dtype=np.float32)
        if len(array) <= 1:
            return result
        dt = np.maximum(np.diff(time_s), 1.0e-9)
        reshape = (len(dt),) + (1,) * (array.ndim - 1)
        result[1:] = np.diff(array, axis=0) / dt.reshape(reshape)
        result[0] = result[1]
        return result

    object_pos_w = stack("object_pos_w")
    object_yaw = stack("object_yaw")
    ee_pos_w = stack("ee_pos_w")
    bin_pos_w = stack("bin_pos_w")
    bin_yaw = values("bin_yaw", np.float32)
    np.savez_compressed(
        path,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
        step=np.asarray([sample["step"] for sample in samples], dtype=np.int64),
        time_s=time_s,
        dataset_frames_played=values("dataset_frames_played", np.int64),
        dataset_action_frame_index=values("dataset_action_frame_index", np.int64),
        dataset_action_timestamp_s=values("dataset_action_timestamp_s", np.float32),
        action_phase=values("action_phase", np.str_),
        action_phase_id=values("action_phase_id", np.int8),
        is_reset_sample=values("is_reset_sample", np.bool_),
        termination_evaluation_applied=values("termination_evaluation_applied", np.bool_),
        policy_control_active=values("policy_control_active", np.bool_),
        robot_control_started=values("robot_control_started", np.bool_),
        robot_control_start_step=values("robot_control_start_step", np.int64),
        robot_control_start_time_s=values("robot_control_start_time_s", np.float32),
        object_pos_w=object_pos_w,
        object_quat_wxyz=stack("object_quat_wxyz"),
        object_quat_fully_observed=stack("object_quat_fully_observed", np.bool_),
        object_yaw=object_yaw,
        object_lin_vel_w=stack("object_lin_vel_w"),
        object_lin_vel_fd_w=finite_difference(object_pos_w),
        object_ang_vel_w=stack("object_ang_vel_w"),
        object_yaw_rate_fd=finite_difference(np.unwrap(object_yaw, axis=0)),
        object_velocity_observed=stack("object_velocity_observed", np.bool_),
        object_active=stack("object_active", np.bool_),
        object_displacement_from_initial_w=stack("object_displacement_from_initial_w"),
        object_failure_baseline_pos_w=stack("object_failure_baseline_pos_w"),
        object_displacement_from_failure_baseline_w=stack("object_displacement_from_failure_baseline_w"),
        object_lift_from_failure_baseline=stack("object_lift_from_failure_baseline"),
        object_out_of_table_xy_bounds=stack("object_out_of_table_xy_bounds", np.bool_),
        object_fell_below_baseline=stack("object_fell_below_baseline", np.bool_),
        max_object_lift=stack("max_object_lift"),
        object_contact_force_max=stack("object_contact_force_max"),
        object_contact_impulse_step=stack("object_contact_impulse_step"),
        object_contact_force_available=stack("object_contact_force_available", np.bool_),
        contact_channel_force_max=stack("contact_channel_force_max"),
        contact_channel_impulse_step=stack("contact_channel_impulse_step"),
        contact_channel_attribution_available=stack("contact_channel_attribution_available", np.bool_),
        bin_pos_w=bin_pos_w,
        bin_quat_wxyz=stack("bin_quat_wxyz"),
        bin_yaw=bin_yaw,
        bin_lin_vel_w=stack("bin_lin_vel_w"),
        bin_lin_vel_fd_w=finite_difference(bin_pos_w),
        bin_ang_vel_w=stack("bin_ang_vel_w"),
        bin_yaw_rate_fd=finite_difference(np.unwrap(bin_yaw)),
        bin_velocity_observed=values("bin_velocity_observed", np.bool_),
        bin_displacement_from_initial_w=stack("bin_displacement_from_initial_w"),
        bin_failure_baseline_pos_w=stack("bin_failure_baseline_pos_w"),
        bin_displacement_from_failure_baseline_w=stack("bin_displacement_from_failure_baseline_w"),
        grasped_object_made_contact=values("grasped_object_made_contact", np.bool_),
        grasped_object_contact_steps=values("grasped_object_contact_steps", np.int32),
        grasped_object_id=values("grasped_object_id", np.int32),
        grasp_attempt_counts=stack("grasp_attempt_counts", np.int32),
        postmortem_attempt_counts_all=stack("postmortem_attempt_counts_all", np.int32),
        postmortem_unassociated_attempt_count=values(
            "postmortem_unassociated_attempt_count", np.int32
        ),
        postmortem_acquisition_counts=stack("postmortem_acquisition_counts", np.int32),
        postmortem_acquired_objects=stack("postmortem_acquired_objects", np.bool_),
        postmortem_manipulated_objects=stack("postmortem_manipulated_objects", np.bool_),
        postmortem_drop_counts=stack("postmortem_drop_counts", np.int32),
        postmortem_release_counts=stack("postmortem_release_counts", np.int32),
        postmortem_associated_transport_m=stack("postmortem_associated_transport_m"),
        postmortem_post_acquisition_transport_m=stack(
            "postmortem_post_acquisition_transport_m"
        ),
        postmortem_max_interaction_segment_transport_m=stack(
            "postmortem_max_interaction_segment_transport_m"
        ),
        postmortem_current_candidate_object_id=values(
            "postmortem_current_candidate_object_id", np.int32
        ),
        postmortem_current_candidate_acquired=values(
            "postmortem_current_candidate_acquired", np.bool_
        ),
        nearest_ee_object_id=values("nearest_ee_object_id", np.int32),
        nearest_ee_object_distance=values("nearest_ee_object_distance", np.float32),
        ee_object_distance=stack("ee_object_distance"),
        ee_pos_w=ee_pos_w,
        ee_quat_wxyz=stack("ee_quat_wxyz"),
        ee_lin_vel_w=stack("ee_lin_vel_w"),
        ee_lin_vel_fd_w=finite_difference(ee_pos_w),
        ee_ang_vel_w=stack("ee_ang_vel_w"),
        ee_velocity_observed=values("ee_velocity_observed", np.bool_),
        joint_pos=stack("joint_pos"),
        joint_vel=stack("joint_vel"),
        joint_effort=stack("joint_effort"),
        joint_effort_observed=values("joint_effort_observed", np.bool_),
        joint_limit_margin=stack("joint_limit_margin"),
        action_lerobot_raw=stack("action_lerobot_raw"),
        # Backwards-compatible name: this has always meant the command after
        # calibration-range clamping.
        action_lerobot=stack("action_lerobot"),
        action_lerobot_clamped=stack("action_lerobot_clamped"),
        action_clamped_mask=stack("action_clamped_mask", np.bool_),
        action_delta_lerobot=stack("action_delta_lerobot"),
        action_sim=stack("action_sim"),
        action_tracking_error=stack("action_tracking_error"),
        condition_names=np.asarray(samples[0]["condition_names"], dtype=np.str_),
        condition_kinds=np.asarray(samples[0]["condition_kinds"], dtype=np.str_),
        condition_met=stack("condition_met", np.bool_),
        condition_snapshot_json=values("condition_snapshot_json", np.str_),
        condition_numeric_metrics_json=values("condition_numeric_metrics_json", np.str_),
        condition_numeric_metrics_available=values("condition_numeric_metrics_available", np.bool_),
        goal_metric_name=values("goal_metric_name", np.str_),
        goal_value=values("goal_value", np.float32),
        goal_threshold=values("goal_threshold", np.float32),
        goal_margin=values("goal_margin", np.float32),
        goal_instant=values("goal_instant", np.bool_),
        goal_overshot=values("goal_overshot", np.bool_),
        inside_bin_object_count=values("inside_bin_object_count", np.int32),
        condition_counter_names=np.asarray([name for name, _attr in CONDITION_COUNTER_SPECS], dtype=np.str_),
        condition_counter_values=stack("condition_counter_values", np.int32),
        failure_raw_max_grasp_attempt_count=values("failure_raw_max_grasp_attempt_count", np.int32),
        failure_threshold_max_grasp_attempts=values("failure_threshold_max_grasp_attempts", np.int32),
        failure_raw_bin_displacement_xy_m=values("failure_raw_bin_displacement_xy_m", np.float32),
        failure_threshold_bin_displacement_m=values("failure_threshold_bin_displacement_m", np.float32),
        failure_raw_max_non_target_displacement_xy_m=values(
            "failure_raw_max_non_target_displacement_xy_m", np.float32
        ),
        failure_threshold_non_target_displacement_m=values(
            "failure_threshold_non_target_displacement_m", np.float32
        ),
        failure_raw_boundary_root_displacement_xy_m=values(
            "failure_raw_boundary_root_displacement_xy_m", np.float32
        ),
        failure_threshold_boundary_displacement_m=values(
            "failure_threshold_boundary_displacement_m", np.float32
        ),
        failure_raw_contact_duration_s=values("failure_raw_contact_duration_s", np.float32),
        failure_threshold_contact_duration_s=values("failure_threshold_contact_duration_s", np.float32),
        failure_raw_move_boundary_gap_m=values("failure_raw_move_boundary_gap_m", np.float32),
        failure_raw_move_lateral_error_m=values("failure_raw_move_lateral_error_m", np.float32),
        postmortem_diagnostic_json=values("postmortem_diagnostic_json", np.str_),
        postmortem_failure_type=values("postmortem_failure_type", np.str_),
        postmortem_confidence=values("postmortem_confidence", np.float32),
        postmortem_classification_version=values("postmortem_classification_version", np.int32),
        success_counter=values("success_counter", np.int32),
        success_required_steps=values("success_required_steps", np.int32),
        success_candidate=values("success_candidate", np.bool_),
        live_success_confirmed=values("live_success_confirmed", np.bool_),
        final_scoring_override_applied=values("final_scoring_override_applied", np.bool_),
        success_age_ready=values("success_age_ready", np.bool_),
        failure_age_ready=values("failure_age_ready", np.bool_),
        failure_baseline_recorded=values("failure_baseline_recorded", np.bool_),
        timeout_confirmation_active=values("timeout_confirmation_active", np.bool_),
        timeout_confirmation_failed=values("timeout_confirmation_failed", np.bool_),
        success=values("success", np.bool_),
        failure=values("failure", np.bool_),
        timed_out=values("timed_out", np.bool_),
        terminal_reason=values("terminal_reason", np.str_),
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
    timeout_scale: float = 1.0,
) -> dict[int, TermEval]:
    with torch.inference_mode():
        timed_out_tensor = task_time_out(
            env.unwrapped,
            confirm_time_s=float(success_params.get("confirm_time_s", 3.0)),
            duration_scale=timeout_scale,
        )
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


def _confirmation_waived_success_for_lane(
    env,
    *,
    env_id: int,
    object_asset_names: list[str],
    success_params: dict[str, Any],
    failure_params: dict[str, Any],
) -> bool:
    """Read the exact success diagnostic without advancing any environment lane.

    ``task_success`` and the other termination functions operate on every row in
    a vectorized environment.  Calling them just to waive confirmation for one
    exhausted replay lane would increment counters and rewrite classifier state
    for other episodes that are still running.  The diagnostic API implements
    the same task-specific predicate and is explicitly side-effect free.
    """

    kwargs = _diagnostic_call_kwargs(object_asset_names, success_params, failure_params)
    snapshots = task_condition_diagnostics(env.unwrapped, **kwargs)
    if env_id >= len(snapshots):
        return False
    return any(
        condition.kind == "success" and bool(condition.met)
        for condition in snapshots[env_id].conditions
    )


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


def _parse_episode_indices(raw_indices: str, *, option_name: str = "--benchmark_episode_indices") -> list[int]:
    indices = []
    for raw_index in raw_indices.split(","):
        raw_index = raw_index.strip()
        if not raw_index:
            continue
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError(f"Invalid episode index {raw_index!r} in {option_name}={raw_indices!r}.") from exc
        indices.append(index)
    if not indices:
        raise ValueError(f"{option_name} was provided but no indices were parsed.")
    duplicate_indices = sorted({index for index in indices if indices.count(index) > 1})
    if duplicate_indices:
        raise ValueError(f"{option_name} contains duplicate episode indices: {duplicate_indices}.")
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


def _load_resume_records(output_dir: Path) -> list[dict[str, Any]]:
    if not args_cli.resume:
        return []
    episodes_path = output_dir / "episodes.jsonl"
    if not episodes_path.is_file():
        return []

    records: list[dict[str, Any]] = []
    with episodes_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Cannot resume: invalid JSON in {episodes_path} at line {line_number}."
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Cannot resume: expected an object in {episodes_path} at line {line_number}."
                )
            records.append(record)
    return records


def _validate_resume_state(records: list[dict[str, Any]]) -> None:
    if not args_cli.resume:
        return
    if args_cli.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if args_cli.output_dir is None:
        raise ValueError("--resume requires an explicit --output_dir.")
    if args_cli.dataset_episode_indices is not None or args_cli.benchmark_episode_indices is not None:
        raise ValueError("--resume currently supports only contiguous sequential episode selection.")
    if args_cli.benchmark_episode_index not in (None, args_cli.dataset_episode_index):
        raise ValueError(
            "--resume requires --benchmark_episode_index to equal --dataset_episode_index when provided."
        )
    if not args_cli.record_dataset or args_cli.record_repo_root is None:
        raise ValueError("--resume requires --record_dataset and --record_repo_root.")

    expected_source_indices = list(
        range(args_cli.dataset_episode_index, args_cli.dataset_episode_index + len(records))
    )
    actual_source_indices = []
    actual_recorded_indices = []
    for row_id, record in enumerate(records):
        try:
            actual_source_indices.append(int(record["dataset"]["episode_index"]))
            actual_recorded_indices.append(int(record["recorded_sim_dataset"]["episode_index"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot resume: outcome record {row_id} lacks valid source/recorded episode indices."
            ) from exc
    if actual_source_indices != expected_source_indices:
        raise ValueError(
            "Cannot resume: outcome source episode indices are not contiguous from "
            f"{args_cli.dataset_episode_index}: {actual_source_indices[:10]}..."
        )
    expected_recorded_indices = list(range(len(records)))
    if actual_recorded_indices != expected_recorded_indices:
        raise ValueError(
            "Cannot resume: recorded dataset episode indices are not contiguous from zero: "
            f"{actual_recorded_indices[:10]}..."
        )

    recorded_total = _dataset_total_episodes(args_cli.record_repo_root)
    if recorded_total is None:
        if records:
            raise ValueError(
                f"Cannot resume: recorded dataset metadata is missing under {args_cli.record_repo_root}."
            )
    elif recorded_total != len(records):
        raise ValueError(
            "Cannot resume: outcome/recorded dataset counts differ: "
            f"outcomes={len(records)}, recorded_episodes={recorded_total}."
        )

    if records:
        prior_hashes = records[0].get("provenance", {}).get("input_sha256", {})
        current_hashes = {
            "episodes_jsonl": _file_sha256(args_cli.episodes_jsonl),
            "episode_layouts_jsonl": _file_sha256(args_cli.episode_layouts_jsonl),
            "dataset_info_json": _file_sha256(
                args_cli.repo_root / "meta" / "info.json" if args_cli.repo_root is not None else None
            ),
            "retime_reference_info_json": _file_sha256(
                args_cli.retime_reference_repo_root / "meta" / "info.json"
                if args_cli.retime_reference_repo_root is not None
                else None
            ),
        }
        mismatches = {
            key: {"previous": prior_hashes.get(key), "current": value}
            for key, value in current_hashes.items()
            if prior_hashes.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Cannot resume because replay inputs changed: {mismatches}")


def _make_output_dir() -> Path:
    if args_cli.output_dir is not None:
        output_dir = args_cli.output_dir
    else:
        # Outcome artifacts are not LeRobot dataset files.  Keep them with the
        # source evaluation (or under outputs/) so creating this directory before
        # recorder initialization cannot make a new recording root look corrupt.
        root = args_cli.repo_root if args_cli.repo_root is not None else Path("outputs")
        output_dir = root / "eval" / f"sim_replay_outcomes_{_now_stamp()}"

    if args_cli.record_dataset and args_cli.record_repo_root is not None:
        resolved_output = output_dir.resolve()
        resolved_recording = args_cli.record_repo_root.resolve()
        if resolved_output == resolved_recording or resolved_output.is_relative_to(resolved_recording):
            raise ValueError(
                "--output_dir must be outside --record_repo_root; outcome artifacts are not LeRobot dataset files."
            )

    episodes_path = output_dir / "episodes.jsonl"
    if output_dir.exists() and episodes_path.exists() and not (args_cli.overwrite or args_cli.resume):
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
    kwargs = _diagnostic_call_kwargs(object_asset_names, success_params, failure_params)
    snapshots = task_condition_diagnostics(env.unwrapped, **kwargs)
    snapshot = snapshots[env_id] if env_id < len(snapshots) else None
    payload = _serialize_param_value(snapshot)
    if not isinstance(payload, dict):
        payload = {"snapshot": payload}
    payload["numeric_metrics"] = _serialize_param_value(
        _optional_task_condition_metrics(env, env_id=env_id, kwargs=kwargs)
    )
    return payload


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
    dataset_episode_indices: list[int],
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
    success_params: dict[str, Any],
    failure_params: dict[str, Any],
    retiming_plan: UniformRetimingPlan | None,
) -> ReplayLane:
    episode = episode_plan[offset]
    benchmark_index = benchmark_indices[offset]
    episode_layout = episode_layouts[offset]
    dataset_episode_index = dataset_episode_indices[offset]
    print(
        f"[INFO]: Lane {env_id}: loading LeRobot dataset episode {dataset_episode_index} "
        f"from {args_cli.repo_root or args_cli.repo_id}..."
    )
    action_episode = _load_lerobot_action_episode(
        repo_id=args_cli.repo_id,
        root=args_cli.repo_root,
        episode_index=dataset_episode_index,
        device=env.unwrapped.device,
        load_observed_states=(
            retiming_plan is not None and retiming_plan.strategy == "tracking_compensated"
        ),
    )
    source_action_num_frames = action_episode.num_frames
    if retiming_plan is not None:
        source_actions = mapper.clamp_lerobot_positions(action_episode.actions)
        source_observed_states = (
            mapper.clamp_lerobot_positions(action_episode.observed_states)
            if action_episode.observed_states is not None
            else None
        )
        retimed_actions = trajectory_preserving_retime_actions(
            source_actions.detach().cpu().numpy(),
            source_observed_states=(
                source_observed_states.detach().cpu().numpy()
                if source_observed_states is not None
                else None
            ),
            initial_action=hold_action_lerobot.detach().cpu().numpy(),
            scale=retiming_plan.scale,
            strategy=retiming_plan.strategy,
        )
        action_episode = LeRobotActionEpisode(
            episode_index=action_episode.episode_index,
            fps=1.0 / control_dt,
            action_names=action_episode.action_names,
            actions=torch.as_tensor(retimed_actions, dtype=torch.float32, device=env.unwrapped.device),
        )
        print(
            f"[INFO]: Lane {env_id}: retimed {source_action_num_frames} source action frame(s) to "
            f"{action_episode.num_frames} frame(s) "
            f"(scale={retiming_plan.scale:.6f}, strategy={retiming_plan.strategy})."
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
    policy_control_active = getattr(env.unwrapped, "_so101_policy_control_active", None)
    if not isinstance(policy_control_active, torch.Tensor) or tuple(policy_control_active.shape) != (
        env.unwrapped.num_envs,
    ):
        policy_control_active = torch.zeros(
            env.unwrapped.num_envs,
            dtype=torch.bool,
            device=env.unwrapped.device,
        )
    policy_control_active[env_ids] = False
    env.unwrapped._so101_policy_control_active = policy_control_active
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

    lane = ReplayLane(
        env_id=env_id,
        offset=offset,
        episode=episode,
        benchmark_index=benchmark_index,
        episode_layout=episode_layout,
        dataset_episode_index=dataset_episode_index,
        action_episode=action_episode,
        source_action_num_frames=source_action_num_frames,
        setup=setup,
        initial_scene=_scene_state(env, object_asset_names, object_pool, env_id=env_id),
        initial_frame_path=initial_frame_path,
        final_frame_path=final_frame_path,
        last_action_lerobot_raw=hold_action_lerobot.clone(),
        last_action_lerobot=hold_action_lerobot.clone(),
        last_action_sim=hold_action.clone(),
        last_action_clamped_mask=torch.zeros_like(hold_action_lerobot, dtype=torch.bool),
        last_action_delta_lerobot=torch.zeros_like(hold_action_lerobot),
    )
    if args_cli.save_trajectory:
        lane.final_eval = TermEval(
            step=0,
            time_s=0.0,
            success=False,
            failure=False,
            timed_out=False,
            reason="none",
        )
        _append_trajectory_sample(
            env,
            lane,
            object_asset_names=object_asset_names,
            control_dt=control_dt,
            success_params=success_params,
            failure_params=failure_params,
        )
    return lane


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
        lane.last_action_lerobot_raw = hold_action_lerobot.clone()
        lane.last_action_lerobot = hold_action_lerobot.clone()
        lane.last_action_sim = hold_action.clone()
        lane.last_action_clamped_mask.zero_()
        lane.last_action_delta_lerobot.zero_()
        lane.last_action_frame_index = -1
        lane.last_action_phase = "initial_hold"
        return

    if not lane.robot_control_started:
        env_ids = torch.tensor([lane.env_id], dtype=torch.long, device=env.unwrapped.device)
        _begin_robot_control(env, object_asset_names, env_ids=env_ids)
        lane.robot_control_started = True

    replay_step = lane.step - initial_hold_steps
    if replay_step < lane.action_episode.num_frames:
        action_frame_index = lane.frame_index
        action_lerobot_raw = lane.action_episode.actions[action_frame_index]
        action_lerobot = mapper.clamp_lerobot_positions(action_lerobot_raw)
        action_sim = mapper.lerobot_positions_to_sim_radians(action_lerobot)
        actions[lane.env_id] = action_sim
        lane.last_action_delta_lerobot = action_lerobot - lane.last_action_lerobot
        lane.last_action_lerobot_raw = action_lerobot_raw.clone()
        lane.last_action_lerobot = action_lerobot.clone()
        lane.last_action_sim = action_sim.clone()
        lane.last_action_clamped_mask = ~torch.isclose(action_lerobot_raw, action_lerobot, rtol=0.0, atol=1.0e-6)
        lane.last_action_frame_index = action_frame_index
        lane.last_action_phase = "dataset"
        lane.frame_index += 1
    elif replay_step < lane.action_episode.num_frames + hold_last_steps:
        # The last recorded command remains applied during the configured hold.
        lane.last_action_delta_lerobot.zero_()
        lane.last_action_phase = "final_hold"
    elif replay_step >= lane.action_episode.num_frames + hold_last_steps:
        raise RuntimeError(
            f"Lane {lane.env_id} was stepped after replay episode {lane.dataset_episode_index} finished."
        )


def _append_trajectory_sample(
    env,
    lane: ReplayLane,
    *,
    object_asset_names: list[str],
    control_dt: float,
    success_params: dict[str, Any],
    failure_params: dict[str, Any],
    final_scoring_override_applied: bool = False,
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
            action_frame_index=lane.last_action_frame_index,
            action_phase=lane.last_action_phase,
            dataset_fps=lane.action_episode.fps,
            action_lerobot_raw=lane.last_action_lerobot_raw,
            action_lerobot=lane.last_action_lerobot,
            action_sim=lane.last_action_sim,
            action_clamped_mask=lane.last_action_clamped_mask,
            action_delta_lerobot=lane.last_action_delta_lerobot,
            term_eval=lane.final_eval,
            control_dt=control_dt,
            success_params=success_params,
            failure_params=failure_params,
            final_scoring_override_applied=final_scoring_override_applied,
            env_id=lane.env_id,
        )
    )


def _upsert_final_trajectory_sample(
    env,
    lane: ReplayLane,
    *,
    object_asset_names: list[str],
    control_dt: float,
    success_params: dict[str, Any],
    failure_params: dict[str, Any],
    final_scoring_override_applied: bool = False,
) -> None:
    if lane.trajectory_samples and int(lane.trajectory_samples[-1]["step"]) == lane.step:
        lane.trajectory_samples[-1] = _trajectory_sample(
            env,
            object_asset_names,
            step=lane.step,
            time_s=lane.step * control_dt,
            frame_index=lane.frame_index,
            action_frame_index=lane.last_action_frame_index,
            action_phase=lane.last_action_phase,
            dataset_fps=lane.action_episode.fps,
            action_lerobot_raw=lane.last_action_lerobot_raw,
            action_lerobot=lane.last_action_lerobot,
            action_sim=lane.last_action_sim,
            action_clamped_mask=lane.last_action_clamped_mask,
            action_delta_lerobot=lane.last_action_delta_lerobot,
            term_eval=lane.final_eval,
            control_dt=control_dt,
            success_params=success_params,
            failure_params=failure_params,
            final_scoring_override_applied=final_scoring_override_applied,
            env_id=lane.env_id,
        )
    else:
        _append_trajectory_sample(
            env,
            lane,
            object_asset_names=object_asset_names,
            control_dt=control_dt,
            success_params=success_params,
            failure_params=failure_params,
            final_scoring_override_applied=final_scoring_override_applied,
        )


def _event_ledger(
    lane: ReplayLane,
    *,
    object_asset_names: list[str],
    target_object_id: int,
    control_dt: float,
    contact_channel_catalog: list[dict[str, Any]],
    lift_threshold_m: float = 0.0127,
) -> list[dict[str, Any]]:
    """Derive a compact transition ledger; continuous evidence stays in the NPZ."""
    samples = lane.trajectory_samples
    if not samples:
        return []
    events: list[dict[str, Any]] = []

    def add(sample: dict[str, Any], event_type: str, **evidence: Any) -> None:
        events.append(
            {
                "type": event_type,
                "step": int(sample["step"]),
                "time_s": float(sample["time_s"]),
                **_serialize_param_value(evidence),
            }
        )

    previous: dict[str, Any] | None = None
    previous_condition_met: dict[str, bool] = {}
    previous_counter_positive = np.zeros(len(CONDITION_COUNTER_SPECS), dtype=np.bool_)
    previous_lifted = np.zeros(len(object_asset_names), dtype=np.bool_)
    previous_out_of_bounds = np.zeros(len(object_asset_names), dtype=np.bool_)
    previous_fallen = np.zeros(len(object_asset_names), dtype=np.bool_)
    previous_target_moved = False
    previous_gripper_motion = "steady"
    previous_contact_channels = np.zeros(len(contact_channel_catalog), dtype=np.bool_)
    previous_unattributed_contact_sensors: set[str] = set()
    previous_goal_instant = False
    previous_goal_overshot = False
    previous_inside_bin_count = 0
    previous_postmortem_type = "unavailable"

    for sample in samples:
        if previous is None or sample["action_phase"] != previous["action_phase"]:
            add(sample, f"{sample['action_phase']}_started")

        postmortem_type = str(sample.get("postmortem_failure_type", "unavailable"))
        if (
            bool(sample.get("policy_control_active", False))
            and postmortem_type != "unavailable"
            and postmortem_type != previous_postmortem_type
        ):
            try:
                postmortem_snapshot = json.loads(sample["postmortem_diagnostic_json"])
            except (KeyError, TypeError, json.JSONDecodeError):
                postmortem_snapshot = None
            add(
                sample,
                "behavioral_classification_changed",
                previous_failure_type=(
                    previous_postmortem_type
                    if previous_postmortem_type != "unavailable"
                    else None
                ),
                failure_type=postmortem_type,
                confidence=(
                    postmortem_snapshot.get("confidence")
                    if isinstance(postmortem_snapshot, dict)
                    else None
                ),
                rationale=(
                    postmortem_snapshot.get("rationale")
                    if isinstance(postmortem_snapshot, dict)
                    else None
                ),
            )
        if bool(sample.get("policy_control_active", False)):
            previous_postmortem_type = postmortem_type

        clamped = np.asarray(sample["action_clamped_mask"], dtype=np.bool_)
        was_clamped = (
            np.asarray(previous["action_clamped_mask"], dtype=np.bool_)
            if previous is not None
            else np.zeros_like(clamped)
        )
        if np.any(clamped) and not np.any(was_clamped):
            joint_ids = np.flatnonzero(clamped).tolist()
            add(
                sample,
                "action_saturation_started",
                joint_ids=joint_ids,
                joint_names=[ACTION_JOINT_NAMES[index] for index in joint_ids],
                raw_action=np.asarray(sample["action_lerobot_raw"])[joint_ids].tolist(),
                clamped_action=np.asarray(sample["action_lerobot_clamped"])[joint_ids].tolist(),
            )
        elif not np.any(clamped) and np.any(was_clamped):
            add(sample, "action_saturation_ended")

        jaw_delta = float(np.asarray(sample["action_delta_lerobot"])[ACTION_JOINT_NAMES.index("Jaw")])
        gripper_motion = "closing" if jaw_delta < -0.5 else "opening" if jaw_delta > 0.5 else "steady"
        if gripper_motion != "steady" and gripper_motion != previous_gripper_motion:
            add(sample, f"gripper_{gripper_motion}_command_started", command_delta=jaw_delta)
        previous_gripper_motion = gripper_motion

        attempt_counts = np.asarray(
            sample.get("postmortem_attempt_counts_all", sample["grasp_attempt_counts"]),
            dtype=np.int64,
        )
        previous_attempt_counts = (
            np.asarray(
                previous.get("postmortem_attempt_counts_all", previous["grasp_attempt_counts"]),
                dtype=np.int64,
            )
            if previous is not None
            else np.zeros_like(attempt_counts)
        )
        for object_id in np.flatnonzero(attempt_counts > previous_attempt_counts).tolist():
            for attempt_number in range(
                int(previous_attempt_counts[object_id]) + 1,
                int(attempt_counts[object_id]) + 1,
            ):
                add(
                    sample,
                    "grasp_attempt",
                    object_id=object_id,
                    object_name=object_asset_names[object_id],
                    is_target=object_id == target_object_id,
                    is_intended_object=(
                        lane.episode.task_family == "bin" or object_id == target_object_id
                    ),
                    attempt_number=attempt_number,
                    ee_object_distance_m=float(np.asarray(sample["ee_object_distance"])[object_id]),
                    attribution_source="postmortem_all_object_tracker",
                )

        unassociated_attempts = int(sample.get("postmortem_unassociated_attempt_count", 0))
        previous_unassociated_attempts = (
            int(previous.get("postmortem_unassociated_attempt_count", 0))
            if previous is not None
            else 0
        )
        for attempt_number in range(previous_unassociated_attempts + 1, unassociated_attempts + 1):
            add(
                sample,
                "unassociated_grasp_attempt",
                attempt_number=attempt_number,
                nearest_object_id=int(sample["nearest_ee_object_id"]),
                nearest_object_distance_m=float(sample["nearest_ee_object_distance"]),
            )

        for field_name, event_type in (
            ("postmortem_acquisition_counts", "object_acquired"),
            ("postmortem_drop_counts", "object_dropped"),
            ("postmortem_release_counts", "object_released"),
        ):
            counts = np.asarray(sample.get(field_name, np.zeros_like(attempt_counts)), dtype=np.int64)
            previous_counts = (
                np.asarray(previous.get(field_name, np.zeros_like(counts)), dtype=np.int64)
                if previous is not None
                else np.zeros_like(counts)
            )
            for object_id in np.flatnonzero(counts > previous_counts).tolist():
                for event_number in range(
                    int(previous_counts[object_id]) + 1,
                    int(counts[object_id]) + 1,
                ):
                    add(
                        sample,
                        event_type,
                        object_id=object_id,
                        object_name=object_asset_names[object_id],
                        is_target=object_id == target_object_id,
                        is_intended_object=(
                            lane.episode.task_family == "bin" or object_id == target_object_id
                        ),
                        event_number=event_number,
                        associated_transport_m=float(
                            np.asarray(
                                sample.get(
                                    "postmortem_associated_transport_m",
                                    np.zeros_like(attempt_counts, dtype=np.float32),
                                )
                            )[object_id]
                        ),
                    )

        grasped_id = int(sample["grasped_object_id"])
        previous_grasped_id = int(previous["grasped_object_id"]) if previous is not None else -1
        if grasped_id != previous_grasped_id:
            if previous_grasped_id >= 0:
                add(
                    sample,
                    "grasp_association_ended",
                    object_id=previous_grasped_id,
                    object_name=object_asset_names[previous_grasped_id],
                    is_target=previous_grasped_id == target_object_id,
                    release_position_w=np.asarray(sample["object_pos_w"])[previous_grasped_id].tolist(),
                    release_lift_m=float(
                        np.asarray(sample["object_lift_from_failure_baseline"])[previous_grasped_id]
                    ),
                )
            if grasped_id >= 0:
                add(
                    sample,
                    "grasp_association_started",
                    object_id=grasped_id,
                    object_name=object_asset_names[grasped_id],
                    is_target=grasped_id == target_object_id,
                )

        contact = bool(sample["grasped_object_made_contact"])
        previous_contact = bool(previous["grasped_object_made_contact"]) if previous is not None else False
        if contact and not previous_contact:
            add(
                sample,
                "grasped_object_contact_started",
                grasped_object_id=grasped_id,
                continuous_contact_steps=int(sample["grasped_object_contact_steps"]),
            )
        elif not contact and previous_contact:
            add(
                sample,
                "grasped_object_contact_ended",
                continuous_contact_steps=int(previous["grasped_object_contact_steps"]),
            )

        contact_channel_force = np.asarray(sample["contact_channel_force_max"], dtype=np.float32)
        contact_attribution = np.asarray(
            sample["contact_channel_attribution_available"], dtype=np.bool_
        )
        contact_channels = (contact_channel_force > 0.0) & contact_attribution
        for channel_id in np.flatnonzero(contact_channels & ~previous_contact_channels).tolist():
            channel = contact_channel_catalog[channel_id]
            add(
                sample,
                "object_pair_contact_started",
                channel_id=channel_id,
                source_object_id=channel["source_object_id"],
                target_object_id=channel["target_object_id"],
                force=float(contact_channel_force[channel_id]),
                pair_attribution_available=True,
            )
        for channel_id in np.flatnonzero(~contact_channels & previous_contact_channels).tolist():
            channel = contact_channel_catalog[channel_id]
            add(
                sample,
                "object_pair_contact_ended",
                channel_id=channel_id,
                source_object_id=channel["source_object_id"],
                target_object_id=channel["target_object_id"],
            )

        # Some Isaac versions expose only a sensor-wide aggregate.  Never
        # repeat that value under several invented pair identities: emit one
        # explicitly unattributed transition per physical sensor instead.
        unattributed_by_sensor: dict[str, dict[str, Any]] = {}
        for channel_id in np.flatnonzero(
            (contact_channel_force > 0.0) & ~contact_attribution
        ).tolist():
            channel = contact_channel_catalog[channel_id]
            sensor_name = str(channel["sensor_name"])
            aggregate = unattributed_by_sensor.setdefault(
                sensor_name,
                {
                    "force": 0.0,
                    "source_object_ids": set(),
                    "candidate_target_object_ids": set(),
                },
            )
            aggregate["force"] = max(aggregate["force"], float(contact_channel_force[channel_id]))
            aggregate["source_object_ids"].add(int(channel["source_object_id"]))
            if int(channel["target_object_id"]) >= 0:
                aggregate["candidate_target_object_ids"].add(int(channel["target_object_id"]))
        unattributed_sensors = set(unattributed_by_sensor)
        for sensor_name in sorted(unattributed_sensors - previous_unattributed_contact_sensors):
            aggregate = unattributed_by_sensor[sensor_name]
            add(
                sample,
                "object_contact_aggregate_started",
                sensor_name=sensor_name,
                force=float(aggregate["force"]),
                source_object_ids=sorted(aggregate["source_object_ids"]),
                candidate_target_object_ids=sorted(aggregate["candidate_target_object_ids"]),
                pair_attribution_available=False,
            )
        for sensor_name in sorted(previous_unattributed_contact_sensors - unattributed_sensors):
            add(
                sample,
                "object_contact_aggregate_ended",
                sensor_name=sensor_name,
                pair_attribution_available=False,
            )
        previous_contact_channels = contact_channels
        previous_unattributed_contact_sensors = unattributed_sensors

        lifted = np.asarray(sample["object_lift_from_failure_baseline"]) >= lift_threshold_m
        for object_id in np.flatnonzero(lifted & ~previous_lifted).tolist():
            add(
                sample,
                "object_lift_threshold_crossed",
                object_id=object_id,
                object_name=object_asset_names[object_id],
                is_target=object_id == target_object_id,
                lift_m=float(np.asarray(sample["object_lift_from_failure_baseline"])[object_id]),
                threshold_m=lift_threshold_m,
            )
        for object_id in np.flatnonzero(~lifted & previous_lifted).tolist():
            add(
                sample,
                "object_returned_below_lift_threshold",
                object_id=object_id,
                object_name=object_asset_names[object_id],
                is_target=object_id == target_object_id,
            )
        previous_lifted = lifted

        out_of_bounds = np.asarray(sample["object_out_of_table_xy_bounds"], dtype=np.bool_)
        for object_id in np.flatnonzero(out_of_bounds & ~previous_out_of_bounds).tolist():
            add(
                sample,
                "object_left_table_xy_bounds",
                object_id=object_id,
                object_name=object_asset_names[object_id],
                is_target=object_id == target_object_id,
                position_w=np.asarray(sample["object_pos_w"])[object_id].tolist(),
            )
        for object_id in np.flatnonzero(~out_of_bounds & previous_out_of_bounds).tolist():
            add(sample, "object_returned_to_table_xy_bounds", object_id=object_id)
        previous_out_of_bounds = out_of_bounds

        fallen = np.asarray(sample["object_fell_below_baseline"], dtype=np.bool_)
        for object_id in np.flatnonzero(fallen & ~previous_fallen).tolist():
            add(
                sample,
                "object_fell_below_table_baseline",
                object_id=object_id,
                object_name=object_asset_names[object_id],
                is_target=object_id == target_object_id,
                position_w=np.asarray(sample["object_pos_w"])[object_id].tolist(),
            )
        previous_fallen = fallen

        if 0 <= target_object_id < len(object_asset_names):
            target_displacement = float(
                np.linalg.norm(np.asarray(sample["object_displacement_from_initial_w"])[target_object_id, :2])
            )
            target_moved = target_displacement >= 0.005
            if target_moved and not previous_target_moved:
                add(sample, "target_manipulation_started", displacement_xy_m=target_displacement, threshold_m=0.005)
            previous_target_moved = target_moved

        success_candidate = bool(sample["success_candidate"])
        previous_success_candidate = bool(previous["success_candidate"]) if previous is not None else False
        if success_candidate and not previous_success_candidate:
            add(
                sample,
                "success_candidate_started",
                held_steps=int(sample["success_counter"]),
                required_steps=int(sample["success_required_steps"]),
            )
        elif not success_candidate and previous_success_candidate:
            add(
                sample,
                "success_candidate_lost",
                held_steps=int(previous["success_counter"]),
                required_steps=int(previous["success_required_steps"]),
            )

        goal_instant = bool(sample["goal_instant"])
        if goal_instant and not previous_goal_instant:
            add(
                sample,
                "goal_geometry_reached",
                metric_name=sample["goal_metric_name"],
                value=float(sample["goal_value"]),
                threshold=float(sample["goal_threshold"]),
                margin=float(sample["goal_margin"]),
            )
        elif not goal_instant and previous_goal_instant:
            add(
                sample,
                "goal_geometry_lost",
                metric_name=sample["goal_metric_name"],
                value=float(sample["goal_value"]),
                threshold=float(sample["goal_threshold"]),
                margin=float(sample["goal_margin"]),
            )
        previous_goal_instant = goal_instant

        goal_overshot = bool(sample["goal_overshot"])
        if goal_overshot and not previous_goal_overshot:
            add(
                sample,
                "goal_overshot",
                metric_name=sample["goal_metric_name"],
                value=float(sample["goal_value"]),
                margin=float(sample["goal_margin"]),
            )
        previous_goal_overshot = goal_overshot

        inside_bin_count = int(sample["inside_bin_object_count"])
        if inside_bin_count != previous_inside_bin_count:
            add(
                sample,
                "bin_containment_count_changed",
                previous_count=previous_inside_bin_count,
                current_count=inside_bin_count,
            )
        previous_inside_bin_count = inside_bin_count

        counter_values = np.asarray(sample["condition_counter_values"], dtype=np.int64)
        counter_positive = counter_values > 0
        for counter_id in np.flatnonzero(counter_positive != previous_counter_positive).tolist():
            counter_name = CONDITION_COUNTER_SPECS[counter_id][0]
            if not counter_name.startswith("failure_"):
                continue
            if counter_positive[counter_id]:
                add(
                    sample,
                    "failure_candidate_started",
                    condition=counter_name.removeprefix("failure_"),
                    held_steps=int(counter_values[counter_id]),
                )
            else:
                add(
                    sample,
                    "failure_candidate_lost",
                    condition=counter_name.removeprefix("failure_"),
                    held_steps=int(np.asarray(previous["condition_counter_values"])[counter_id]) if previous else 0,
                )
        previous_counter_positive = counter_positive

        condition_names = np.asarray(sample["condition_names"]).tolist()
        condition_met = np.asarray(sample["condition_met"], dtype=np.bool_).tolist()
        current_condition_met = dict(zip(condition_names, condition_met, strict=False))
        for condition_name, is_met in current_condition_met.items():
            was_met = previous_condition_met.get(condition_name, False)
            if is_met and not was_met:
                add(sample, "condition_confirmed", condition=condition_name)
            elif not is_met and was_met:
                add(sample, "condition_no_longer_met", condition=condition_name)
        previous_condition_met = current_condition_met

        live_success = bool(sample.get("live_success_confirmed", sample["success"]))
        previous_live_success = (
            bool(previous.get("live_success_confirmed", previous["success"]))
            if previous is not None
            else False
        )
        if live_success and not previous_live_success:
            add(sample, "success_confirmed", basis="live_benchmark_rule")
        elif not live_success and previous_live_success:
            add(sample, "success_lost", basis="live_benchmark_rule")

        final_override = bool(sample.get("final_scoring_override_applied", False))
        previous_final_override = (
            bool(previous.get("final_scoring_override_applied", False))
            if previous is not None
            else False
        )
        if final_override and not previous_final_override:
            add(
                sample,
                "final_confirmation_waiver_applied",
                held_steps=int(sample["success_counter"]),
                required_steps=int(sample["success_required_steps"]),
                live_success_confirmed=live_success,
                selected_scoring_success=bool(sample["success"]),
            )
        failure = bool(sample["failure"])
        previous_failure = bool(previous["failure"]) if previous is not None else False
        if failure and (not previous_failure or sample["terminal_reason"] != previous["terminal_reason"]):
            add(sample, "failure_confirmed", reason=sample["terminal_reason"])
        timed_out = bool(sample["timed_out"])
        previous_timed_out = bool(previous["timed_out"]) if previous is not None else False
        if timed_out and not previous_timed_out:
            add(sample, "time_out")
        if bool(sample["timeout_confirmation_failed"]):
            was_failed = bool(previous["timeout_confirmation_failed"]) if previous is not None else False
            if not was_failed:
                add(sample, "success_confirmation_breached_at_timeout")

        previous = sample

    if lane.action_stream_exhausted:
        final_sample = samples[-1]
        add(
            final_sample,
            "action_stream_exhausted",
            dataset_frames_played=lane.frame_index,
            dataset_frames=lane.action_episode.num_frames,
        )
    for event_id, event in enumerate(events):
        event["event_id"] = event_id
        event["time_resolution_s"] = float(max(control_dt, args_cli.trajectory_stride * control_dt))
    return events


def _numeric_leaf_values(value: Any, prefix: str = "") -> dict[str, float]:
    leaves: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_numeric_leaf_values(item, child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            leaves.update(_numeric_leaf_values(item, child_prefix))
    elif isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
        numeric = float(value)
        if math.isfinite(numeric):
            leaves[prefix] = numeric
    return leaves


def _numeric_metric_envelopes(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize explicit numeric diagnostic fields; human-readable strings are never parsed."""
    extrema: dict[str, dict[str, Any]] = {}
    for sample in samples:
        payloads: list[tuple[str, Any]] = []
        for payload_name, key in (
            ("diagnostic", "condition_snapshot_json"),
            ("numeric_metrics", "condition_numeric_metrics_json"),
        ):
            try:
                payloads.append((payload_name, json.loads(sample[key])))
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
        for payload_name, payload in payloads:
            for path, value in _numeric_leaf_values(payload, payload_name).items():
                record = extrema.setdefault(
                    path,
                    {
                        "minimum": value,
                        "minimum_step": int(sample["step"]),
                        "maximum": value,
                        "maximum_step": int(sample["step"]),
                    },
                )
                if value < record["minimum"]:
                    record["minimum"] = value
                    record["minimum_step"] = int(sample["step"])
                if value > record["maximum"]:
                    record["maximum"] = value
                    record["maximum_step"] = int(sample["step"])
    return extrema


def _task_progress_summary(
    lane: ReplayLane,
    final_scene: dict[str, Any],
) -> dict[str, Any]:
    """Compute transparent root-geometry progress metrics in addition to authoritative diagnostics."""
    samples = lane.trajectory_samples
    if not samples:
        return {"available": False, "reason": "trajectory_not_saved"}
    target_id = int(final_scene["target_object_id"])
    referent_ids = [int(value) for value in final_scene.get("referent_object_ids", [])]
    positions = np.stack([sample["object_pos_w"] for sample in samples]).astype(np.float64)
    steps = np.asarray([sample["step"] for sample in samples], dtype=np.int64)
    task_family = str(final_scene.get("task_family", lane.episode.task_family))
    result: dict[str, Any] = {
        "available": True,
        "task_family": task_family,
        "geometry": "object_root_centers; authoritative footprint metrics are in condition_numeric_metrics",
    }

    if task_family == "next_to" and referent_ids:
        distance = np.linalg.norm(positions[:, target_id, :2] - positions[:, referent_ids[0], :2], axis=1)
        best = int(np.argmin(distance))
        result.update(
            root_center_distance_m={
                "minimum": float(distance[best]),
                "best_step": int(steps[best]),
                "final": float(distance[-1]),
            }
        )
    elif task_family == "between" and len(referent_ids) >= 2:
        target = positions[:, target_id, :2]
        first = positions[:, referent_ids[0], :2]
        second = positions[:, referent_ids[1], :2]
        segment = second - first
        segment_len_sq = np.maximum(np.sum(segment * segment, axis=1), 1.0e-9)
        fraction = np.sum((target - first) * segment, axis=1) / segment_len_sq
        projection = first + fraction[:, None] * segment
        perpendicular = np.linalg.norm(target - projection, axis=1)
        centered_shortfall = np.maximum.reduce((0.1 - fraction, fraction - 0.9, np.zeros_like(fraction)))
        score = perpendicular + centered_shortfall
        best = int(np.argmin(score))
        result.update(
            between_root_geometry={
                "minimum_perpendicular_distance_m": float(np.min(perpendicular)),
                "best_step": int(steps[best]),
                "segment_fraction_at_best": float(fraction[best]),
                "centered_shortfall_at_best": float(centered_shortfall[best]),
                "final_perpendicular_distance_m": float(perpendicular[-1]),
                "final_segment_fraction": float(fraction[-1]),
            }
        )
    elif task_family == "move":
        direction = str(lane.episode.direction or "")
        axis_sign = {
            "left": (0, 1.0),
            "right": (0, -1.0),
            "forward": (1, -1.0),
            "backward": (1, 1.0),
        }.get(direction, (0, 1.0))
        axis, sign = axis_sign
        lateral_axis = 1 - axis
        initial_target = np.asarray(lane.initial_scene["objects"][target_id]["position"], dtype=np.float64)
        delta = positions[:, target_id, :2] - initial_target[:2]
        progress = sign * delta[:, axis]
        lateral = np.abs(delta[:, lateral_axis])
        best = int(np.argmax(progress))
        result.update(
            move_root_geometry={
                "direction": direction,
                "maximum_directional_progress_m": float(progress[best]),
                "best_progress_step": int(steps[best]),
                "lateral_error_at_best_progress_m": float(lateral[best]),
                "maximum_lateral_error_m": float(np.max(lateral)),
                "final_directional_progress_m": float(progress[-1]),
                "final_lateral_error_m": float(lateral[-1]),
            }
        )
    elif task_family in {"bin", "named_bin"}:
        active_ids = (
            [target_id]
            if task_family == "named_bin"
            else [int(value) for value in final_scene.get("active_object_ids", [])]
        )
        bin_half_extents = final_scene.get("bin", {}).get("footprint_half_extents")
        if active_ids and bin_half_extents is not None:
            bin_pos = np.stack([sample["bin_pos_w"] for sample in samples]).astype(np.float64)
            bin_yaw = np.asarray([sample["bin_yaw"] for sample in samples], dtype=np.float64)
            rel = positions[:, active_ids, :2] - bin_pos[:, None, :2]
            cos_yaw = np.cos(-bin_yaw)[:, None]
            sin_yaw = np.sin(-bin_yaw)[:, None]
            local_x = cos_yaw * rel[..., 0] - sin_yaw * rel[..., 1]
            local_y = sin_yaw * rel[..., 0] + cos_yaw * rel[..., 1]
            offset = np.asarray(final_scene["bin"].get("footprint_center_offset") or [0.0, 0.0])
            half = np.asarray(bin_half_extents, dtype=np.float64)
            inside = (np.abs(local_x - offset[0]) <= half[0]) & (np.abs(local_y - offset[1]) <= half[1])
            counts = np.sum(inside, axis=1)
            best = int(np.argmax(counts))
            result.update(
                bin_root_containment={
                    "maximum_simultaneously_inside": int(counts[best]),
                    "required_objects": len(active_ids),
                    "best_step": int(steps[best]),
                    "final_inside_count": int(counts[-1]),
                    "object_ever_inside": {
                        str(object_id): bool(np.any(inside[:, index]))
                        for index, object_id in enumerate(active_ids)
                    },
                    "scoring_mode": "target_only" if task_family == "named_bin" else "all_active_objects",
                }
            )
    return result


def _episode_explanation(
    lane: ReplayLane,
    *,
    final_scene: dict[str, Any],
    final_label: dict[str, Any],
    object_asset_names: list[str],
    contact_channel_catalog: list[dict[str, Any]],
    control_dt: float,
    final_confirm_time_disabled: bool,
    final_confirmation_waiver_applied: bool,
    failure_params: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    samples = lane.trajectory_samples
    target_id = int(final_scene["target_object_id"])
    active_ids = [int(value) for value in final_scene.get("active_object_ids", [])]
    events = _event_ledger(
        lane,
        object_asset_names=object_asset_names,
        target_object_id=target_id,
        control_dt=control_dt,
        contact_channel_catalog=contact_channel_catalog,
    )
    if not samples:
        quality = {
            "trajectory_available": False,
            "final_success": bool(final_label["success"]),
            "final_success_confirmation_time_disabled": bool(final_confirm_time_disabled),
            "final_confirmation_waiver_applied": bool(final_confirmation_waiver_applied),
            "explanation_limited": True,
        }
        return events, {"available": False}, None, quality

    steps = np.asarray([sample["step"] for sample in samples], dtype=np.int64)
    selected_success = np.asarray([sample["success"] for sample in samples], dtype=np.bool_)
    live_success = np.asarray(
        [sample.get("live_success_confirmed", sample["success"]) for sample in samples],
        dtype=np.bool_,
    )
    counters = np.asarray([sample["success_counter"] for sample in samples], dtype=np.int64)
    required = np.maximum(
        np.asarray([sample["success_required_steps"] for sample in samples], dtype=np.int64),
        1,
    )
    confirmation_multiple = counters / required
    confirmation_fraction = np.minimum(confirmation_multiple, 1.0)
    best_confirmation_id = int(np.argmax(confirmation_multiple))
    numeric_metrics: list[dict[str, Any] | None] = []
    goal_margins = np.full(len(samples), np.nan, dtype=np.float64)
    for sample_id, sample in enumerate(samples):
        try:
            payload = json.loads(sample["condition_numeric_metrics_json"])
        except (KeyError, TypeError, json.JSONDecodeError):
            payload = None
        numeric_metrics.append(payload if isinstance(payload, dict) else None)
        goal = payload.get("goal") if isinstance(payload, dict) else None
        margin = goal.get("margin") if isinstance(goal, dict) else None
        if isinstance(margin, (int, float)) and not isinstance(margin, bool) and math.isfinite(float(margin)):
            goal_margins[sample_id] = float(margin)
    best_goal_id = int(np.nanargmax(goal_margins)) if np.any(np.isfinite(goal_margins)) else None
    positions_initial = np.stack(
        [sample["object_displacement_from_initial_w"] for sample in samples]
    ).astype(np.float64)
    positions_baseline = np.stack(
        [sample["object_displacement_from_failure_baseline_w"] for sample in samples]
    ).astype(np.float64)
    ee_distance = np.stack([sample["ee_object_distance"] for sample in samples]).astype(np.float64)
    contact_force = np.stack([sample["object_contact_force_max"] for sample in samples]).astype(np.float64)
    contact_impulse = np.stack([sample["object_contact_impulse_step"] for sample in samples]).astype(np.float64)
    contact_channel_force = np.stack([sample["contact_channel_force_max"] for sample in samples]).astype(np.float64)
    contact_channel_impulse = np.stack(
        [sample["contact_channel_impulse_step"] for sample in samples]
    ).astype(np.float64)
    contact_channel_available = np.stack(
        [sample["contact_channel_attribution_available"] for sample in samples]
    ).astype(np.bool_)
    object_positions_trace = np.stack([sample["object_pos_w"] for sample in samples]).astype(np.float64)
    object_lift_trace = np.stack(
        [sample["object_lift_from_failure_baseline"] for sample in samples]
    ).astype(np.float64)
    grasped_object_trace = np.asarray([sample["grasped_object_id"] for sample in samples], dtype=np.int64)
    max_lift = np.max(np.stack([sample["max_object_lift"] for sample in samples]), axis=0)
    legacy_rule_attempts = np.asarray(samples[-1]["grasp_attempt_counts"], dtype=np.int64)
    attempts = np.asarray(
        samples[-1].get("postmortem_attempt_counts_all", legacy_rule_attempts),
        dtype=np.int64,
    )
    unassociated_attempts = int(samples[-1].get("postmortem_unassociated_attempt_count", 0))

    per_object = []
    for object_id, object_name in enumerate(object_asset_names):
        xy_initial = np.linalg.norm(positions_initial[:, object_id, :2], axis=1)
        xy_baseline = np.linalg.norm(positions_baseline[:, object_id, :2], axis=1)
        min_ee_id = int(np.argmin(ee_distance[:, object_id]))
        max_initial_id = int(np.argmax(xy_initial))
        max_lift_id = int(np.argmax(object_lift_trace[:, object_id]))
        lifted = object_lift_trace[:, object_id] >= 0.0127
        lift_episode_starts = lifted & ~np.concatenate(([False], lifted[:-1]))
        first_lift_id = int(np.flatnonzero(lifted)[0]) if np.any(lifted) else None
        transport_delta = np.linalg.norm(
            np.diff(object_positions_trace[:, object_id, :], axis=0),
            axis=1,
        )
        associated_transport_mask = (grasped_object_trace[1:] == object_id) | (
            grasped_object_trace[:-1] == object_id
        )
        lifted_transport_mask = lifted[1:] | lifted[:-1]
        per_object.append(
            {
                "object_id": object_id,
                "object_name": object_name,
                "active": object_id in active_ids,
                "is_target": object_id == target_id,
                "grasp_attempt_count": int(attempts[object_id]),
                "maximum_lift_m": float(max_lift[object_id]),
                "maximum_sampled_lift_m": float(object_lift_trace[max_lift_id, object_id]),
                "maximum_sampled_lift_step": int(steps[max_lift_id]),
                "first_lift_step": int(steps[first_lift_id]) if first_lift_id is not None else None,
                "lift_episode_count": int(np.count_nonzero(lift_episode_starts)),
                "maximum_xy_displacement_from_initial_m": float(xy_initial[max_initial_id]),
                "maximum_xy_displacement_step": int(steps[max_initial_id]),
                "maximum_xy_displacement_from_failure_baseline_m": float(np.max(xy_baseline)),
                "minimum_ee_distance_m": float(ee_distance[min_ee_id, object_id]),
                "minimum_ee_distance_step": int(steps[min_ee_id]),
                "sampled_transport_path_length_m": float(np.sum(transport_delta)),
                "sampled_transport_while_grasp_associated_m": float(
                    np.sum(transport_delta[associated_transport_mask])
                ),
                "sampled_transport_while_lifted_m": float(np.sum(transport_delta[lifted_transport_mask])),
                "final_displacement_from_initial_w": positions_initial[-1, object_id].tolist(),
                "maximum_contact_force": float(np.max(contact_force[:, object_id])),
                "approximate_accumulated_contact_impulse": float(np.sum(contact_impulse[:, object_id])),
                "ever_outside_table_xy_bounds": bool(
                    any(bool(sample["object_out_of_table_xy_bounds"][object_id]) for sample in samples)
                ),
                "ever_fell_below_table_baseline": bool(
                    any(bool(sample["object_fell_below_baseline"][object_id]) for sample in samples)
                ),
            }
        )

    contact_pair_summary = []
    for channel_id, channel in enumerate(contact_channel_catalog):
        attributed_force = np.where(
            contact_channel_available[:, channel_id],
            contact_channel_force[:, channel_id],
            0.0,
        )
        attributed_impulse = np.where(
            contact_channel_available[:, channel_id],
            contact_channel_impulse[:, channel_id],
            0.0,
        )
        maximum_force = float(np.max(attributed_force))
        accumulated_impulse = float(np.sum(attributed_impulse))
        if maximum_force <= 0.0 and accumulated_impulse <= 0.0:
            continue
        maximum_id = int(np.argmax(attributed_force))
        contact_pair_summary.append(
            {
                **channel,
                "maximum_force": maximum_force,
                "maximum_force_step": int(steps[maximum_id]),
                "approximate_accumulated_impulse": accumulated_impulse,
            }
        )

    unattributed_contact_summary = []
    contact_channel_ids_by_sensor: dict[str, list[int]] = {}
    for channel_id, channel in enumerate(contact_channel_catalog):
        contact_channel_ids_by_sensor.setdefault(str(channel["sensor_name"]), []).append(channel_id)
    for sensor_name, channel_ids in sorted(contact_channel_ids_by_sensor.items()):
        unavailable = ~contact_channel_available[:, channel_ids]
        aggregate_force = np.max(
            np.where(unavailable, contact_channel_force[:, channel_ids], 0.0), axis=1
        )
        aggregate_impulse = np.max(
            np.where(unavailable, contact_channel_impulse[:, channel_ids], 0.0), axis=1
        )
        maximum_force = float(np.max(aggregate_force))
        accumulated_impulse = float(np.sum(aggregate_impulse))
        if maximum_force <= 0.0 and accumulated_impulse <= 0.0:
            continue
        maximum_id = int(np.argmax(aggregate_force))
        source_ids = sorted(
            {int(contact_channel_catalog[channel_id]["source_object_id"]) for channel_id in channel_ids}
        )
        candidate_target_ids = sorted(
            {
                int(contact_channel_catalog[channel_id]["target_object_id"])
                for channel_id in channel_ids
                if int(contact_channel_catalog[channel_id]["target_object_id"]) >= 0
            }
        )
        unattributed_contact_summary.append(
            {
                "sensor_name": sensor_name,
                "source_object_ids": source_ids,
                "candidate_target_object_ids": candidate_target_ids,
                "pair_attribution_available": False,
                "maximum_force": maximum_force,
                "maximum_force_step": int(steps[maximum_id]),
                "approximate_accumulated_impulse": accumulated_impulse,
            }
        )

    task_progress = _task_progress_summary(lane, final_scene)
    best_achieved = {
        "available": True,
        "success_confirmation": {
            "maximum_held_steps": int(counters[best_confirmation_id]),
            "required_steps": int(required[best_confirmation_id]),
            "maximum_fraction": float(confirmation_fraction[best_confirmation_id]),
            "maximum_multiple_of_required_hold": float(
                confirmation_multiple[best_confirmation_id]
            ),
            "best_step": int(steps[best_confirmation_id]),
            "longest_candidate_duration_s": float(int(counters[best_confirmation_id]) * control_dt),
            "numeric_metrics_at_best_confirmation": numeric_metrics[best_confirmation_id],
        },
        "goal_geometry": (
            {
                "best_step": int(steps[best_goal_id]),
                "best_margin": float(goal_margins[best_goal_id]),
                "final_margin": float(goal_margins[-1]) if np.isfinite(goal_margins[-1]) else None,
                "numeric_metrics_at_best_margin": numeric_metrics[best_goal_id],
                "final_numeric_metrics": numeric_metrics[-1],
            }
            if best_goal_id is not None
            else {"available": False}
        ),
        "per_object": per_object,
        "object_pair_contacts": contact_pair_summary,
        "unattributed_object_contact_aggregates": unattributed_contact_summary,
        "task_progress": task_progress,
        "condition_numeric_envelopes": _numeric_metric_envelopes(samples),
    }
    closest_miss = None
    if not bool(final_label["success"]):
        closest_miss = {
            "condition": "success_confirmation",
            "best_step": int(steps[best_confirmation_id]),
            "held_steps": int(counters[best_confirmation_id]),
            "required_steps": int(required[best_confirmation_id]),
            "shortfall_steps": max(0, int(required[best_confirmation_id] - counters[best_confirmation_id])),
            "shortfall_s": max(
                0.0,
                float((required[best_confirmation_id] - counters[best_confirmation_id]) * control_dt),
            ),
            "confirmation_fraction": float(confirmation_fraction[best_confirmation_id]),
            "numeric_metrics_at_best_confirmation": numeric_metrics[best_confirmation_id],
            "best_goal_geometry": (
                {
                    "best_step": int(steps[best_goal_id]),
                    "margin": float(goal_margins[best_goal_id]),
                    "metrics": numeric_metrics[best_goal_id],
                }
                if best_goal_id is not None
                else None
            ),
            "task_progress_at_best": task_progress,
        }

    first_selected_success_id = (
        int(np.flatnonzero(selected_success)[0]) if np.any(selected_success) else None
    )
    first_live_success_id = int(np.flatnonzero(live_success)[0]) if np.any(live_success) else None
    final_success = bool(final_label["success"])
    clamped_mask = np.stack([sample["action_clamped_mask"] for sample in samples]).astype(np.bool_)
    tracking_error = np.abs(np.stack([sample["action_tracking_error"] for sample in samples]))
    action_delta = np.stack([sample["action_delta_lerobot"] for sample in samples]).astype(np.float64)
    action_jump_norm = np.linalg.norm(action_delta, axis=1)
    maximum_action_jump_id = int(np.argmax(action_jump_norm))
    joint_margin = np.stack([sample["joint_limit_margin"] for sample in samples])
    non_target_ids = [object_id for object_id in active_ids if object_id != target_id]
    max_non_target_displacement = max(
        (
            entry["maximum_xy_displacement_from_initial_m"]
            for entry in per_object
            if entry["object_id"] in non_target_ids
        ),
        default=0.0,
    )
    bin_displacement_xy = np.asarray(
        [np.linalg.norm(sample["bin_displacement_from_initial_w"][:2]) for sample in samples],
        dtype=np.float64,
    )
    max_bin_displacement = float(np.max(bin_displacement_xy))
    max_contact_steps = int(max(int(sample["grasped_object_contact_steps"]) for sample in samples))
    task_family = str(final_scene.get("task_family", lane.episode.task_family))
    non_target_collateral = bool(
        task_family != "bin"
        and max_non_target_displacement > float(failure_params.get("non_target_displacement_limit", 0.0127))
    )
    bin_collateral = bool(
        max_bin_displacement > float(failure_params.get("bin_displacement_limit", 0.0254))
    )
    excessive_attempts = bool(
        failure_params.get("enforce_max_grasp_attempts", True)
        and np.any(legacy_rule_attempts > int(failure_params.get("max_grasp_attempts", 3)))
    )
    prolonged_contact = bool(
        max_contact_steps * control_dt
        > float(failure_params.get("contact_grace_time_s", 5.0))
    )
    out_of_bounds_object_ids = [
        int(entry["object_id"])
        for entry in per_object
        if entry["active"] and entry["ever_outside_table_xy_bounds"]
    ]
    fallen_object_ids = [
        int(entry["object_id"])
        for entry in per_object
        if entry["active"] and entry["ever_fell_below_table_baseline"]
    ]
    non_target_out_of_bounds_ids = [
        object_id for object_id in out_of_bounds_object_ids if object_id in non_target_ids
    ]
    non_target_fallen_ids = [
        object_id for object_id in fallen_object_ids if object_id in non_target_ids
    ]
    out_of_bounds_event = bool(out_of_bounds_object_ids)
    fallen_event = bool(fallen_object_ids)
    # In bin tasks every active object is intentionally transported off its
    # original tabletop region and may enter a bin outside TABLE_BOUNDS.  Those
    # factual transitions remain recorded, but they are not collateral damage.
    unexpected_non_target_workspace_event = bool(
        task_family != "bin" and (non_target_out_of_bounds_ids or non_target_fallen_ids)
    )
    prolonged_forbidden_contact = bool(task_family != "bin" and prolonged_contact)
    transient_success = bool(
        np.any(live_success)
        and (not final_success or not np.all(live_success[first_live_success_id:]))
    )
    collateral = (
        non_target_collateral
        or bin_collateral
        or prolonged_forbidden_contact
        or unexpected_non_target_workspace_event
    )
    success_quality = (
        "failure"
        if not final_success
        else "transient_success"
        if transient_success
        else "confirmation_waived_success"
        if final_confirmation_waiver_applied
        else "success_with_collateral"
        if collateral
        else "stable_success"
    )
    sample_step_gaps = np.diff(steps)
    raw_actions = np.stack([sample["action_lerobot_raw"] for sample in samples])
    object_positions = np.stack([sample["object_pos_w"] for sample in samples])
    joint_positions = np.stack([sample["joint_pos"] for sample in samples])
    dataset_sample_ids = [
        sample_id
        for sample_id, sample in enumerate(samples)
        if sample["action_phase"] == "dataset" and sample["dataset_action_frame_index"] >= 0
    ]
    max_action_time_error = None
    if dataset_sample_ids:
        first_id = dataset_sample_ids[0]
        sim_elapsed = np.asarray(
            [samples[sample_id]["time_s"] - samples[first_id]["time_s"] for sample_id in dataset_sample_ids]
        )
        dataset_elapsed = np.asarray(
            [
                samples[sample_id]["dataset_action_timestamp_s"]
                - samples[first_id]["dataset_action_timestamp_s"]
                for sample_id in dataset_sample_ids
            ]
        )
        max_action_time_error = float(np.max(np.abs(sim_elapsed - dataset_elapsed)))
    quality = {
        "trajectory_available": True,
        "trajectory_stride": int(args_cli.trajectory_stride),
        "temporal_resolution_s": float(args_cli.trajectory_stride * control_dt),
        "final_success": final_success,
        # The compatibility names now explicitly mean confirmation under the
        # live benchmark rule; selected-scoring fields expose final-only policy
        # overrides without conflating the two notions of success.
        "success_ever_confirmed": bool(np.any(live_success)),
        "live_success_ever_confirmed": bool(np.any(live_success)),
        "selected_scoring_success_ever_true": bool(np.any(selected_success)),
        "first_success_step": (
            int(steps[first_live_success_id]) if first_live_success_id is not None else None
        ),
        "first_success_time_s": (
            float(samples[first_live_success_id]["time_s"])
            if first_live_success_id is not None
            else None
        ),
        "first_live_confirmed_success_step": (
            int(steps[first_live_success_id]) if first_live_success_id is not None else None
        ),
        "first_live_confirmed_success_time_s": (
            float(samples[first_live_success_id]["time_s"])
            if first_live_success_id is not None
            else None
        ),
        "first_selected_scoring_success_step": (
            int(steps[first_selected_success_id])
            if first_selected_success_id is not None
            else None
        ),
        "first_selected_scoring_success_time_s": (
            float(samples[first_selected_success_id]["time_s"])
            if first_selected_success_id is not None
            else None
        ),
        "stable_from_first_success_through_end": bool(
            first_live_success_id is not None
            and np.all(live_success[first_live_success_id:])
            and final_success
        ),
        "success_was_transient": transient_success,
        "success_quality": success_quality,
        "clean_stable_success": success_quality == "stable_success",
        "final_success_confirmation_time_disabled": bool(final_confirm_time_disabled),
        "final_confirmation_waiver_applied": bool(final_confirmation_waiver_applied),
        "maximum_confirmation_fraction": float(confirmation_fraction[best_confirmation_id]),
        "maximum_confirmation_multiple_of_required_hold": float(
            confirmation_multiple[best_confirmation_id]
        ),
        "best_goal_margin": float(goal_margins[best_goal_id]) if best_goal_id is not None else None,
        "final_goal_margin": float(goal_margins[-1]) if np.isfinite(goal_margins[-1]) else None,
        "minimum_goal_margin_after_first_success": (
            float(np.nanmin(goal_margins[first_live_success_id:]))
            if first_live_success_id is not None
            and np.any(np.isfinite(goal_margins[first_live_success_id:]))
            else None
        ),
        "action_saturated_sample_count": int(np.count_nonzero(np.any(clamped_mask, axis=1))),
        "action_saturated_joint_counts": {
            ACTION_JOINT_NAMES[index]: int(np.count_nonzero(clamped_mask[:, index]))
            for index in range(len(ACTION_JOINT_NAMES))
        },
        "maximum_absolute_joint_tracking_error_rad": float(np.nanmax(tracking_error)),
        "maximum_absolute_tracking_error_by_joint_rad": {
            ACTION_JOINT_NAMES[index]: float(np.nanmax(tracking_error[:, index]))
            for index in range(len(ACTION_JOINT_NAMES))
        },
        "maximum_lerobot_action_jump_l2": float(action_jump_norm[maximum_action_jump_id]),
        "maximum_lerobot_action_jump_step": int(steps[maximum_action_jump_id]),
        "minimum_joint_limit_margin_rad": float(np.nanmin(joint_margin)),
        "grasp_attempt_count": int(np.sum(attempts)),
        "object_associated_grasp_attempt_count": int(np.sum(attempts)),
        "unassociated_grasp_attempt_count": unassociated_attempts,
        "all_jaw_close_attempt_count": int(np.sum(attempts) + unassociated_attempts),
        "legacy_rule_eligible_grasp_attempt_count": int(np.sum(legacy_rule_attempts)),
        "target_grasp_attempt_count": int(attempts[target_id]) if target_id < len(attempts) else 0,
        "maximum_continuous_contact_steps": max_contact_steps,
        "maximum_continuous_contact_s": float(max_contact_steps * control_dt),
        "prolonged_contact_detected": prolonged_contact,
        "prolonged_forbidden_contact_detected": prolonged_forbidden_contact,
        "excessive_grasp_attempts_detected": excessive_attempts,
        "maximum_non_target_xy_displacement_m": float(max_non_target_displacement),
        "non_target_collateral_displacement_detected": non_target_collateral,
        "maximum_bin_xy_displacement_m": max_bin_displacement,
        "bin_collateral_displacement_detected": bin_collateral,
        "collateral_event_detected": collateral,
        "unexpected_non_target_workspace_event_detected": unexpected_non_target_workspace_event,
        "active_object_ids_that_left_table_xy_bounds": out_of_bounds_object_ids,
        "active_object_ids_that_fell_below_table_baseline": fallen_object_ids,
        "non_target_object_ids_that_left_table_xy_bounds": non_target_out_of_bounds_ids,
        "non_target_object_ids_that_fell_below_table_baseline": non_target_fallen_ids,
        "target_left_table_xy_bounds": target_id in out_of_bounds_object_ids,
        "target_fell_below_table_baseline": target_id in fallen_object_ids,
        "any_active_object_left_table_xy_bounds": out_of_bounds_event,
        "any_active_object_fell_below_table_baseline": fallen_event,
        "missing_full_object_quaternion_fraction": float(
            1.0 - np.mean(np.stack([sample["object_quat_fully_observed"] for sample in samples]))
        ),
        "missing_object_velocity_fraction": float(
            1.0 - np.mean(np.stack([sample["object_velocity_observed"] for sample in samples]))
        ),
        "missing_contact_force_fraction": float(
            1.0 - np.mean(np.stack([sample["object_contact_force_available"] for sample in samples]))
        ),
        "missing_condition_numeric_metrics_fraction": float(
            1.0
            - np.mean(
                np.asarray(
                    [sample["condition_numeric_metrics_available"] for sample in samples],
                    dtype=np.float32,
                )
            )
        ),
        "missing_postmortem_diagnostic_fraction": float(
            np.mean(
                np.asarray(
                    [sample["postmortem_failure_type"] == "unavailable" for sample in samples],
                    dtype=np.float32,
                )
            )
        ),
        "data_quality": {
            "trajectory_reaches_final_step": int(samples[-1]["step"]) == int(lane.step),
            "saved_sample_count": len(samples),
            "termination_evaluation_sample_count": int(
                sum(bool(sample["termination_evaluation_applied"]) for sample in samples)
            ),
            "reset_telemetry_sample_count": int(
                sum(bool(sample["is_reset_sample"]) for sample in samples)
            ),
            "maximum_saved_step_gap": int(np.max(sample_step_gaps)) if len(sample_step_gaps) else 0,
            "expected_maximum_saved_step_gap": int(args_cli.trajectory_stride),
            "dataset_frames_complete": int(lane.frame_index) == int(lane.action_episode.num_frames),
            "nonfinite_raw_action_values": int(np.count_nonzero(~np.isfinite(raw_actions))),
            "nonfinite_object_position_values": int(np.count_nonzero(~np.isfinite(object_positions))),
            "nonfinite_joint_position_values": int(np.count_nonzero(~np.isfinite(joint_positions))),
            "dataset_control_period_error_s": abs(
                1.0 / max(lane.action_episode.fps, 1.0e-9) - control_dt
            ),
            "maximum_replay_dataset_timestamp_drift_s": max_action_time_error,
        },
    }
    return events, best_achieved, closest_miss, quality


def _fresh_postmortem_diagnostic(
    env,
    object_asset_names: list[str],
    *,
    env_id: int,
    confirm_time_s: float,
) -> Any:
    """Refresh final attribution so it sees the last success update, then serialize generically."""
    diagnostics_fn = getattr(so101_mdp, "benchmark_postmortem_failure_diagnostics", None)
    diagnostics = None
    if callable(diagnostics_fn):
        try:
            kwargs = (
                {"confirm_time_s": confirm_time_s}
                if "confirm_time_s" in inspect.signature(diagnostics_fn).parameters
                else {}
            )
            diagnostics = diagnostics_fn(env.unwrapped, object_asset_names, **kwargs)
        except (AttributeError, IndexError, TypeError, RuntimeError, ValueError):
            diagnostics = None
    if diagnostics is None:
        diagnostics = getattr(env.unwrapped, "_so101_postmortem_failure_diagnostics", None)
    if isinstance(diagnostics, (list, tuple)):
        return diagnostics[env_id] if env_id < len(diagnostics) else None
    if isinstance(diagnostics, dict) and env_id in diagnostics:
        return diagnostics[env_id]
    return diagnostics


def _failure_attribution_from_raw(
    raw: Any,
    *,
    label: dict[str, Any],
    basis: str,
) -> dict[str, Any]:
    raw = _serialize_param_value(raw)
    raw_type = None
    if isinstance(raw, dict):
        raw_type = raw.get("failure_type") or raw.get("primary_failure") or raw.get("primary_failure_type")
    applicable = bool(not label["success"] and raw is not None and raw_type not in {"none", "not_applicable"})
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


def _failure_attribution(
    env,
    *,
    env_id: int,
    object_asset_names: list[str],
    label: dict[str, Any],
    confirm_time_s: float,
    basis: str,
) -> dict[str, Any]:
    raw = _fresh_postmortem_diagnostic(
        env,
        object_asset_names,
        env_id=env_id,
        confirm_time_s=confirm_time_s,
    )
    return _failure_attribution_from_raw(raw, label=label, basis=basis)


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
    final_confirmation_waiver_applied = False
    if (
        final_confirm_time_disabled
        and not lane.final_eval.success
        and _confirmation_waived_success_for_lane(
            env,
            env_id=lane.env_id,
            object_asset_names=object_asset_names,
            success_params=final_success_params,
            failure_params=failure_params,
        )
    ):
        lane.final_eval = TermEval(
            step=lane.step,
            time_s=lane.step * control_dt,
            success=True,
            failure=False,
            timed_out=False,
            reason="success",
        )
        final_confirmation_waiver_applied = True

    if args_cli.save_trajectory:
        _upsert_final_trajectory_sample(
            env,
            lane,
            object_asset_names=object_asset_names,
            control_dt=control_dt,
            # Keep trajectory counters and condition snapshots on the live
            # benchmark rule.  The selected label and explicit override flag
            # carry the final-only confirmation waiver separately.
            success_params=success_params,
            failure_params=failure_params,
            final_scoring_override_applied=final_confirmation_waiver_applied,
        )

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

    event_ledger, best_achieved, closest_miss, outcome_quality = _episode_explanation(
        lane,
        final_scene=final_scene,
        final_label=final_label,
        object_asset_names=object_asset_names,
        contact_channel_catalog=_contact_channel_catalog(env, object_asset_names),
        control_dt=control_dt,
        final_confirm_time_disabled=final_confirm_time_disabled,
        final_confirmation_waiver_applied=final_confirmation_waiver_applied,
        failure_params=failure_params,
    )
    live_confirm_time_s = float(success_params.get("confirm_time_s", 3.0))
    final_failure_attribution = _failure_attribution(
        env,
        env_id=lane.env_id,
        object_asset_names=object_asset_names,
        label=final_label,
        confirm_time_s=live_confirm_time_s,
        basis="final_state_standard_confirmation",
    )
    if lane.first_terminal is None:
        first_terminal_failure_attribution = _failure_attribution_from_raw(
            final_failure_attribution.get("postmortem_raw"),
            label=first_terminal_label,
            basis="no_first_terminal_before_action_stream_end",
        )
    else:
        first_terminal_failure_attribution = _failure_attribution_from_raw(
            lane.first_terminal_postmortem_raw,
            label=first_terminal_label,
            basis="first_terminal_state",
        )
    failure_attribution = (
        final_failure_attribution
        if args_cli.label_source == "final"
        else first_terminal_failure_attribution
    )
    raw_attribution = failure_attribution.get("postmortem_raw")
    outcome_quality["behavioral_attribution_basis"] = failure_attribution.get("basis")
    outcome_quality["behavioral_attribution_applicable"] = bool(
        failure_attribution.get("applicable")
    )
    if isinstance(raw_attribution, dict):
        outcome_quality["behavioral_attribution_confidence"] = raw_attribution.get("confidence")
        outcome_quality["behavioral_attribution_version"] = raw_attribution.get("classification_version")

    record = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _json_now(),
        "provenance": _run_provenance(),
        "dataset": {
            "repo_id": args_cli.repo_id,
            "repo_root": str(args_cli.repo_root) if args_cli.repo_root is not None else None,
            "episode_index": lane.dataset_episode_index,
        },
        "recorded_sim_dataset": (
            {
                "repo_id": args_cli.record_repo_id,
                "repo_root": str(args_cli.record_repo_root),
                "episode_index": lane.recorded_dataset_episode_index,
                "frame_semantics": (
                    "one post-step simulated observation for each replay action (retimed when configured); "
                    "initial/final diagnostic holds are excluded"
                ),
            }
            if lane.recorded_dataset_episode_index is not None
            else None
        ),
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
        "failure_attribution": failure_attribution,
        "first_terminal_failure_attribution": first_terminal_failure_attribution,
        "final_failure_attribution": final_failure_attribution,
        # Convenient alias for analytics; live rule labels above remain unchanged.
        "behavioral_outcome": failure_attribution["postmortem"],
        "event_ledger": event_ledger,
        "best_achieved": best_achieved,
        "closest_miss": closest_miss,
        "outcome_quality": outcome_quality,
        "final_diagnostics": _final_condition_diagnostics(
            env,
            env_id=lane.env_id,
            object_asset_names=object_asset_names,
            success_params=final_success_params,
            failure_params=failure_params,
        ),
        "episode_length": {
            "source_dataset_frames": lane.source_action_num_frames,
            "uniform_retime_scale": (
                lane.action_episode.num_frames / max(lane.source_action_num_frames, 1)
            ),
            "retime_strategy": (
                args_cli.retime_strategy
                if args_cli.retime_reference_repo_root is not None or args_cli.retime_scale is not None
                else None
            ),
            "retime_interpolation": (
                "pchip"
                if args_cli.retime_reference_repo_root is not None or args_cli.retime_scale is not None
                else None
            ),
            "timeout_scale": (
                (
                    float(args_cli.retime_scale)
                    if args_cli.retime_scale is not None
                    else lane.action_episode.num_frames / max(lane.source_action_num_frames, 1)
                )
                if args_cli.retime_reference_repo_root is not None or args_cli.retime_scale is not None
                else 1.0
            ),
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
            "schema_version": SCHEMA_VERSION,
            "object_asset_names": object_asset_names,
            "object_labels": object_pool,
            "action_joint_names": ACTION_JOINT_NAMES,
            "trajectory_stride": args_cli.trajectory_stride if args_cli.save_trajectory else None,
            "includes_grasped_object_made_contact": bool(args_cli.save_trajectory),
            "condition_counter_names": [name for name, _attr in CONDITION_COUNTER_SPECS],
            "grasp_attempt_count_semantics": {
                "grasp_attempt_counts": (
                    "legacy rule-eligible counts; instruction tasks include only the instructed target"
                ),
                "postmortem_attempt_counts_all": (
                    "all active-object jaw-close associations used for behavioral attribution"
                ),
                "postmortem_unassociated_attempt_count": (
                    "jaw-close cycles without a nearby active object"
                ),
            },
            "contact_channel_catalog": _contact_channel_catalog(env, object_asset_names),
            "contact_force_units": "N",
            "contact_impulse_semantics": "force sample multiplied by control_dt; approximate N*s",
            "action_lerobot_semantics": "post-calibration-range-clamp (v1-compatible)",
            "action_lerobot_raw_semantics": "dataset command before any collector clamp",
            "missing_float_state_encoding": "NaN with a companion *_observed/*_available mask",
            "finite_difference_velocity_semantics": (
                "average velocity between saved samples; temporal spacing is recorded in time_s"
            ),
            "condition_trace_semantics": (
                "side-effect-free structured diagnostic snapshot plus explicit counters; details strings are not parsed"
            ),
            "success_signal_semantics": {
                "live_success_confirmed": (
                    "confirmed under the live benchmark rule and its recorded confirmation window"
                ),
                "success": (
                    "selected scoring evaluation; the final sample may be true because of an explicit final-only policy"
                ),
                "final_scoring_override_applied": (
                    "true only on a final sample whose confirmation waiver changed a non-success into success"
                ),
            },
            "reset_sample_semantics": (
                "step 0 captures initial state only; termination_evaluation_applied is false and offline rescorers skip it"
            ),
            "failure_baseline_semantics": (
                "before the settled baseline is recorded, baseline positions fall back to initial-scene positions; "
                "failure_baseline_recorded identifies the transition"
            ),
        },
        "eval_setup": _capture_eval_setup(
            env,
            env_id=lane.env_id,
            control_dt=control_dt,
            physics_dt=physics_dt,
            success_params=success_params,
            failure_params=failure_params,
            final_success_params=final_success_params,
            final_success_confirm_time_disabled=final_confirm_time_disabled,
        ),
        "initial_scene": lane.initial_scene,
        "final_scene": final_scene,
    }
    return _serialize_param_value(record)


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
    retiming_plan = _build_uniform_retiming_plan(args_cli.repo_root)
    episode_specs = load_episode_jsonl(args_cli.episodes_jsonl)
    output_dir = _make_output_dir()
    resume_records = _load_resume_records(output_dir)
    _validate_resume_state(resume_records)
    resume_count = len(resume_records)
    selected_dataset_indices = (
        _parse_episode_indices(
            args_cli.dataset_episode_indices,
            option_name="--dataset_episode_indices",
        )
        if args_cli.dataset_episode_indices is not None
        else None
    )
    if selected_dataset_indices is not None:
        invalid_dataset_indices = [index for index in selected_dataset_indices if index < 0]
        if invalid_dataset_indices:
            raise ValueError(
                f"--dataset_episode_indices contains negative episode indices: {invalid_dataset_indices}."
            )
        if args_cli.benchmark_episode_indices is not None:
            benchmark_indices = _parse_episode_indices(args_cli.benchmark_episode_indices)
            if len(benchmark_indices) != len(selected_dataset_indices):
                raise ValueError(
                    "--dataset_episode_indices and --benchmark_episode_indices must contain the same number "
                    f"of entries, got {len(selected_dataset_indices)} and {len(benchmark_indices)}."
                )
        else:
            benchmark_indices = list(selected_dataset_indices)
        episode_plan = _episode_selection(episode_specs, benchmark_indices)
        dataset_episode_indices = list(selected_dataset_indices)
        run_planned_count = len(episode_plan)
        total_planned_count = run_planned_count
    elif args_cli.benchmark_episode_indices:
        benchmark_indices = _parse_episode_indices(args_cli.benchmark_episode_indices)
        episode_plan = _episode_selection(episode_specs, benchmark_indices)
        run_planned_count = len(episode_plan)
        total_planned_count = run_planned_count
    else:
        requested_benchmark_start = (
            args_cli.dataset_episode_index
            if args_cli.benchmark_episode_index is None
            else args_cli.benchmark_episode_index
        )
        total_planned_count = _planned_count(episode_specs, requested_benchmark_start)
        if resume_count >= total_planned_count:
            raise ValueError(
                "Replay is already complete: "
                f"resume_records={resume_count}, requested_episodes={total_planned_count}."
            )
        benchmark_start = requested_benchmark_start + resume_count
        run_planned_count = total_planned_count - resume_count
        episode_plan, benchmark_indices = _episode_window(
            episode_specs,
            start_index=benchmark_start,
            count=run_planned_count,
        )
    if selected_dataset_indices is None:
        dataset_start = args_cli.dataset_episode_index + resume_count
        dataset_episode_indices = [dataset_start + i for i in range(len(episode_plan))]
    episode_layouts = _load_episode_layouts(episode_plan, benchmark_indices, args_cli.episode_layouts_jsonl)

    dataset_instructions = _load_dataset_episode_instructions(args_cli.repo_root)
    _verify_jsonl_matches_dataset(
        episode_specs=episode_specs,
        episode_plan=episode_plan,
        benchmark_indices=benchmark_indices,
        dataset_episode_indices=dataset_episode_indices,
        dataset_instructions=dataset_instructions,
    )

    video_spans = (
        _load_dataset_video_spans(args_cli.repo_root)
        if args_cli.frame_source == "dataset"
        else {}
    )

    print(f"[INFO]: Loaded {len(episode_specs)} validated JSONL episode(s) from {args_cli.episodes_jsonl}.")
    print(
        "[INFO]: Replay mapping: "
        f"dataset episodes {dataset_episode_indices} -> benchmark rows {benchmark_indices}"
    )
    print(f"[INFO]: Saving outcome artifacts to {output_dir}")
    if resume_count:
        print(
            f"[INFO]: Resuming after {resume_count} completed episode(s); "
            f"next source/benchmark episode is {dataset_episode_indices[0]}."
        )
    if args_cli.frame_source == "dataset":
        print("[INFO]: Saving overhead frames from recorded LeRobot videos; Isaac camera sensors are disabled.")
    if args_cli.no_success_confirm_time:
        print("[INFO]: Final exhausted action streams will be scored with success confirm_time_s=0.0.")
    if retiming_plan is not None:
        print(
            "[INFO]: Smooth trajectory-preserving retiming: "
            f"source_mean={retiming_plan.source_mean_frames:.3f} frames/episode, "
            f"target_mean={retiming_plan.target_mean_frames:.3f}, "
            f"scale={retiming_plan.scale:.9f}, strategy={retiming_plan.strategy}."
        )

    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    np.random.seed(args_cli.seed)

    object_pool_episode_specs = (
        load_episode_jsonl(args_cli.object_pool_episodes_jsonl)
        if args_cli.object_pool_episodes_jsonl is not None
        else episode_plan
    )
    object_pool = _episode_object_pool(object_pool_episode_specs)
    missing_pool_objects = sorted(
        {object_name for episode in episode_plan for object_name in episode.objects} - set(object_pool)
    )
    if missing_pool_objects:
        raise ValueError(
            f"--object_pool_episodes_jsonl is missing selected benchmark object(s): {missing_pool_objects}"
        )
    if args_cli.object_pool_episodes_jsonl is not None:
        print(
            "[INFO]: Using canonical object-pool order from "
            f"{args_cli.object_pool_episodes_jsonl} ({len(object_pool_episode_specs)} episode row(s))."
        )
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
    recorder = None
    recording_camera_sources = None
    if args_cli.record_dataset:
        cameras = _discover_cameras(env)
        if not cameras:
            raise RuntimeError("No simulator cameras were found for --record_dataset.")
        recording_camera_sources = real_compatible_camera_sources(cameras)
        recorder = LeRobotSimDatasetRecorder(
            repo_id=args_cli.record_repo_id,
            dataset_root=args_cli.record_repo_root,
            fps=max(1, round(1.0 / control_dt)),
            cameras=dataset_cameras(cameras, recording_camera_sources),
            streaming_encoding=args_cli.dataset_streaming_encoding,
            vcodec=args_cli.dataset_vcodec,
            encoder_queue_size=args_cli.dataset_encoder_queue_size,
            encoder_threads=None if args_cli.dataset_encoder_threads == 0 else args_cli.dataset_encoder_threads,
            image_writer_processes=args_cli.dataset_image_writer_processes,
            image_writer_threads_per_camera=args_cli.dataset_image_writer_threads_per_camera,
            video_files_size_mb=args_cli.dataset_video_files_size_mb,
        )
        recorder.init_dataset()
        if args_cli.resume and recorder.num_saved_episodes != resume_count:
            raise RuntimeError(
                "Cannot resume: recorder metadata changed during initialization: "
                f"outcomes={resume_count}, recorded_episodes={recorder.num_saved_episodes}."
            )
    else:
        print("[INFO]: Simulated LeRobot dataset recording disabled. Pass --record_dataset to enable it.")

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    hold_action = _initial_robot_action(env)
    hold_action_lerobot = mapper.sim_radians_to_lerobot_positions(hold_action)
    actions[:] = hold_action
    _reset_env(env)

    episodes_path = output_dir / "episodes.jsonl"
    summary_records: list[dict[str, Any]] = list(resume_records)
    collection_complete = False

    try:
        episodes_mode = "a" if args_cli.resume else "w"
        with episodes_path.open(episodes_mode, encoding="utf-8") as episodes_file:
            active_lanes: dict[int, ReplayLane] = {}
            pending_records: dict[int, dict[str, Any]] = {}
            next_offset = 0
            next_write_offset = 0
            completed_count = resume_count
            segment_completed_count = 0
            collection_start_s = time.perf_counter()

            def start_lane(env_id: int) -> None:
                nonlocal next_offset
                lane = _start_replay_lane(
                    env,
                    env_id=env_id,
                    offset=next_offset,
                    episode_plan=episode_plan,
                    benchmark_indices=benchmark_indices,
                    dataset_episode_indices=dataset_episode_indices,
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
                    success_params=success_params,
                    failure_params=failure_params,
                    retiming_plan=retiming_plan,
                )
                active_lanes[env_id] = lane
                if recorder is not None:
                    instruction = str(getattr(env.unwrapped, "_so101_instruction_text", [""])[env_id])
                    recorder.start_episode(task=instruction)
                next_offset += 1

            def flush_ready_records() -> None:
                nonlocal next_write_offset
                while next_write_offset in pending_records:
                    record = pending_records.pop(next_write_offset)
                    episodes_file.write(
                        json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
                    )
                    episodes_file.flush()
                    summary_records.append(record)
                    next_write_offset += 1

            for env_id in range(min(num_parallel_envs, run_planned_count)):
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

                    obs, _rewards, _terminated, _truncated, _info = env.step(actions)
                    for lane in active_lanes.values():
                        lane.step += 1
                        if recorder is not None and lane.last_action_phase == "dataset":
                            assert recording_camera_sources is not None
                            observation_lerobot = mapper.sim_radians_to_lerobot_positions(
                                obs["policy"]["joint_pos_obs"][lane.env_id].clone()
                            )
                            recorder.push_frame(
                                action=lane.last_action_lerobot,
                                observation_state=observation_lerobot,
                                images=recording_images(obs["visual"], recording_camera_sources),
                            )
                    evals = _manual_term_evals(
                        env,
                        steps_by_env_id={env_id: lane.step for env_id, lane in active_lanes.items()},
                        control_dt=control_dt,
                        success_params=success_params,
                        failure_params=failure_params,
                        timeout_scale=(
                            retiming_plan.scale if retiming_plan is not None else 1.0
                        ),
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
                            success_params=success_params,
                            failure_params=failure_params,
                        )

                    if lane.final_eval.done and lane.first_terminal is None:
                        lane.first_terminal = lane.final_eval
                        lane.first_terminal_postmortem_raw = _serialize_param_value(
                            _fresh_postmortem_diagnostic(
                                env,
                                object_asset_names,
                                env_id=lane.env_id,
                                confirm_time_s=float(success_params.get("confirm_time_s", 3.0)),
                            )
                        )
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
                        or (args_cli.stop_on_done and lane.final_eval.done)
                    ):
                        finished_env_ids.append(env_id)

                for env_id in finished_env_ids:
                    lane = active_lanes.pop(env_id)
                    if recorder is not None:
                        recorded_episode_index = recorder.num_saved_episodes
                        instruction = str(
                            getattr(env.unwrapped, "_so101_instruction_text", [""])[env_id]
                        )
                        if recorder.stop_episode(task=instruction):
                            lane.recorded_dataset_episode_index = recorded_episode_index
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
                    segment_completed_count += 1
                    elapsed_s = time.perf_counter() - collection_start_s
                    episodes_per_minute = 60.0 * segment_completed_count / max(elapsed_s, 1.0e-6)
                    remaining_s = (
                        elapsed_s
                        * (total_planned_count - completed_count)
                        / max(segment_completed_count, 1)
                    )
                    expected_completion = datetime.now().astimezone() + timedelta(seconds=remaining_s)
                    print(
                        f"[INFO]: Episode {completed_count}/{total_planned_count} finished on lane {env_id}: "
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
                    if next_offset < run_planned_count and not STOP_REQUESTED:
                        start_lane(env_id)

                flush_ready_records()
                if args_cli.real_time:
                    dt_s = time.perf_counter() - replay_step_start
                    time.sleep(max((control_dt / args_cli.speed) - dt_s, 0.0))

            for offset in sorted(pending_records):
                record = pending_records[offset]
                episodes_file.write(
                    json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
                )
                summary_records.append(record)
            episodes_file.flush()

        successes = sum(1 for record in summary_records if record["label"]["success"])
        failures = len(summary_records) - successes
        source_frame_total = sum(
            int(record["episode_length"]["source_dataset_frames"])
            for record in summary_records
        )
        replay_frame_total = sum(
            int(record["episode_length"]["dataset_frames"])
            for record in summary_records
        )
        failure_counts: dict[str, int] = {}
        behavioral_failure_counts: dict[str, int] = {}
        for record in summary_records:
            reason = record["label"]["failure_reason"]
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
            attribution = record.get("failure_attribution", {})
            postmortem = attribution.get("postmortem") if attribution.get("applicable") else None
            if isinstance(postmortem, dict):
                behavioral_reason = (
                    postmortem.get("failure_type")
                    or postmortem.get("primary_failure")
                    or postmortem.get("primary_failure_type")
                    or "unclassified"
                )
            elif record["label"]["success"]:
                behavioral_reason = "success"
            else:
                behavioral_reason = "unclassified"
            behavioral_failure_counts[str(behavioral_reason)] = (
                behavioral_failure_counts.get(str(behavioral_reason), 0) + 1
            )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _json_now(),
            "provenance": _run_provenance(),
            "episodes_path": str(episodes_path),
            "label_source": args_cli.label_source,
            "planned_episodes": total_planned_count,
            "completed_episodes": len(summary_records),
            "collection_complete": len(summary_records) == total_planned_count,
            "missing_dataset_episode_indices": sorted(
                set(
                    range(
                        args_cli.dataset_episode_index,
                        args_cli.dataset_episode_index + total_planned_count,
                    )
                )
                - {int(record["dataset"]["episode_index"]) for record in summary_records}
            ),
            "successes": successes,
            "failures": failures,
            "success_rate": successes / max(len(summary_records), 1),
            "successful_dataset_episode_indices": [
                int(record["dataset"]["episode_index"])
                for record in summary_records
                if record["label"]["success"]
            ],
            "failed_dataset_episode_indices": [
                int(record["dataset"]["episode_index"])
                for record in summary_records
                if not record["label"]["success"]
            ],
            "failure_reason_counts": failure_counts,
            "behavioral_outcome_counts": behavioral_failure_counts,
            "retiming": {
                "enabled": retiming_plan is not None,
                "requested_scale": retiming_plan.scale if retiming_plan is not None else 1.0,
                "strategy": (
                    retiming_plan.strategy if retiming_plan is not None else None
                ),
                "interpolation": "pchip" if retiming_plan is not None else None,
                "timeout_scale": retiming_plan.scale if retiming_plan is not None else 1.0,
                "reference_repo_root": (
                    retiming_plan.reference_repo_root if retiming_plan is not None else None
                ),
                "reference_target_mean_frames": (
                    retiming_plan.target_mean_frames if retiming_plan is not None else None
                ),
                "source_total_frames": source_frame_total,
                "replay_total_frames": replay_frame_total,
                "source_mean_frames": source_frame_total / max(len(summary_records), 1),
                "replay_mean_frames": replay_frame_total / max(len(summary_records), 1),
                "realized_scale": replay_frame_total / max(source_frame_total, 1),
            },
            "transient_successes": sum(
                1 for record in summary_records if record.get("outcome_quality", {}).get("success_was_transient")
            ),
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args_cli).items()
                if key != "app_launcher"
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(
            "[INFO]: Outcome summary: "
            f"success={successes}/{len(summary_records)} ({100.0 * summary['success_rate']:.1f}%), "
            f"failures={failures}"
        )
        print(f"[INFO]: Wrote {episodes_path} and {output_dir / 'summary.json'}")
        collection_complete = bool(summary["collection_complete"])
    finally:
        try:
            if recorder is not None and recorder.recording:
                recorder.cancel_episode()
        finally:
            try:
                if recorder is not None:
                    recorder.finalize()
            finally:
                env.close()
    return collection_complete


if __name__ == "__main__":
    exit_code = 0
    try:
        if not main():
            exit_code = 130 if STOP_REQUESTED else 1
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
