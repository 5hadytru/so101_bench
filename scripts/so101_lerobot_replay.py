# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay recorded LeRobot dataset actions in the SO-101 Bench simulator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import inspect
import json
import math
from pathlib import Path
import queue
import sys
import threading
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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"Expected an integer >= 1, got {value!r}.")
    return parsed


parser = argparse.ArgumentParser(
    description=(
        "Replay one or more LeRobot dataset episodes in SO-101 Bench by applying the recorded action stream "
        "to the simulated robot."
    )
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="So101Bench-Bin-v0", help="Isaac Lab task name.")
parser.add_argument("--seed", type=int, default=1984, help="Environment seed.")
parser.add_argument(
    "--contact_solver_position_iterations",
    type=_positive_int,
    default=64,
    help=(
        "PhysX solver position iteration count for the bin and object rigid bodies (default: 64). "
        "Lowering it speeds up the simulation but changes contact resolution."
    ),
)
parser.add_argument(
    "--episodes_jsonl",
    type=Path,
    required=True,
    help="Required JSONL file defining benchmark episodes, matching the original teleop/eval run.",
)
parser.add_argument(
    "--episode_layouts_jsonl",
    "--layouts_jsonl",
    type=Path,
    default=None,
    help=(
        "Optional JSONL file with object and bin poses to apply as-is from the original run. Rows are matched "
        "to episodes by trial_id when present; otherwise rows are indexed by benchmark episode index. "
        "Provided layouts are not revalidated."
    ),
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="5hadytru/so101_bench_sim_1",
    help="LeRobot dataset repo id. A local --repo_root is used when provided.",
)
parser.add_argument(
    "--repo_root",
    type=Path,
    default=Path("data/lerobot/so101_bench_follower_teleop"),
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
        "First benchmark JSONL/layout row to reset before replay. Defaults to --dataset_episode_index, which "
        "matches sequential teleop recordings with no skipped/cancelled episodes."
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
    default=1,
    help="Number of consecutive LeRobot dataset episodes to replay.",
)
parser.add_argument(
    "--initial_hold_time_s",
    type=float,
    default=1.5,
    help="Seconds to hold the initial sim joint pose before replaying the first recorded action.",
)
parser.add_argument(
    "--hold_last_action_time_s",
    type=float,
    default=0.0,
    help="Seconds to hold the final recorded action after the action stream is exhausted.",
)
parser.add_argument(
    "--continue_after_done",
    action="store_true",
    default=False,
    help="Continue playing recorded actions after the environment reports done. By default replay stops on done.",
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
parser.add_argument(
    "--inspect_initial_scene",
    action="store_true",
    default=False,
    help="Reset the first selected benchmark episode, print/view initial poses, and exit when the Isaac app closes.",
)
parser.add_argument(
    "--terminal_control_stdin",
    nargs="?",
    const=True,
    default=True,
    type=_str_to_bool,
    help="Accept pause/resume/skip/quit commands typed in the launch terminal.",
)
parser.add_argument(
    "--keyboard_debug",
    action="store_true",
    default=False,
    help="Print every Isaac keyboard press seen by the replay control listener.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import so101_bench.tasks  # noqa: F401
from so101_bench.benchmark import BenchmarkEpisodeSpec, load_episode_jsonl
from so101_bench.layouts import normalize_layout_object_slots
from so101_bench.mdp import benchmark_object_positions, mark_benchmark_robot_start
from so101_bench.tasks.direct.so101_bench.so101_bench_env_cfg import (
    OBJECT_LABELS,
    configure_contact_solver_position_iterations,
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


def _normalize_keyboard_key(key: str) -> str:
    return key.strip().upper().replace("-", "_").replace(" ", "_")


def _matches_keyboard_key(event_name: str, key: str) -> bool:
    event_name = _normalize_keyboard_key(event_name)
    key = _normalize_keyboard_key(key)
    return event_name in {key, f"KEY_{key}"}


def _normalize_terminal_command(command: str) -> str:
    return command.strip().lower().replace("-", "_").replace(" ", "_")


class _ReplayControls:
    """Keyboard and terminal controls for replay."""

    def __init__(self, *, terminal_enabled: bool, debug: bool):
        self.paused = False
        self._events: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._input = None
        self._keyboard = None
        self._keyboard_sub = None
        self._key_press_type = None
        self._terminal_enabled = terminal_enabled
        self._debug = debug
        self._start_isaac_keyboard_listener()
        if terminal_enabled:
            self._start_stdin_listener()

    def _start_isaac_keyboard_listener(self) -> None:
        try:
            import carb.input
            import omni.appwindow

            app_window = omni.appwindow.get_default_app_window()
            if app_window is None:
                print("[WARN]: No Isaac app window found; keyboard controls are unavailable.")
                return

            self._input = carb.input.acquire_input_interface()
            self._keyboard = app_window.get_keyboard()
            self._key_press_type = carb.input.KeyboardEventType.KEY_PRESS
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                self._keyboard,
                self._on_keyboard_event,
            )
            print("[INFO]: Keyboard controls: P pause/resume, N skip episode, Q quit.")
        except Exception as exc:
            print(f"[WARN]: Keyboard controls unavailable: {exc}")

    def _start_stdin_listener(self) -> None:
        if not sys.stdin or not sys.stdin.isatty():
            return

        def _read_stdin():
            while True:
                try:
                    line = sys.stdin.readline()
                except Exception:
                    return
                if line == "":
                    return
                self._maybe_queue_terminal_command(line)

        thread = threading.Thread(target=_read_stdin, daemon=True)
        thread.start()
        print("[INFO]: Terminal controls: pause, resume, skip, quit (type command then Enter).")

    def _maybe_queue_terminal_command(self, line: str) -> None:
        command = _normalize_terminal_command(line)
        aliases = {
            "p": "toggle_pause",
            "pause": "pause",
            "resume": "resume",
            "play": "resume",
            "continue": "resume",
            "n": "skip_episode",
            "next": "skip_episode",
            "skip": "skip_episode",
            "skip_episode": "skip_episode",
            "q": "quit",
            "quit": "quit",
            "exit": "quit",
        }
        event = aliases.get(command)
        if event is None:
            print("[WARN]: Unknown terminal command. Use pause, resume, skip, or quit.")
            return
        self._events.put(event)

    def _on_keyboard_event(self, event, *args, **kwargs):
        event_name = getattr(getattr(event, "input", None), "name", "")
        if event.type == self._key_press_type and self._debug:
            print(f"[DEBUG]: Isaac key press: {event_name!r}")
        if event.type != self._key_press_type:
            return False

        if _matches_keyboard_key(event_name, "P"):
            self._events.put("toggle_pause")
            return True
        if _matches_keyboard_key(event_name, "N"):
            self._events.put("skip_episode")
            return True
        if _matches_keyboard_key(event_name, "Q"):
            self._events.put("quit")
            return True
        return False

    def poll(self) -> tuple[bool, bool]:
        skip_requested = False
        quit_requested = False
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            if event == "pause":
                if not self.paused:
                    self.paused = True
                    print("[INFO]: Replay paused.")
                continue
            if event == "resume":
                if self.paused:
                    self.paused = False
                    print("[INFO]: Replay resumed.")
                continue
            if event == "toggle_pause":
                self.paused = not self.paused
                print(f"[INFO]: Replay {'paused' if self.paused else 'resumed'}.")
                continue
            if event == "skip_episode":
                skip_requested = True
                if self.paused:
                    self.paused = False
                continue
            if event == "quit":
                quit_requested = True
                if self.paused:
                    self.paused = False
                continue
        return skip_requested, quit_requested

    def close(self) -> None:
        if self._input is None or self._keyboard is None or self._keyboard_sub is None:
            return
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None


def _format_lerobot_pose(values: torch.Tensor) -> str:
    numbers = values.detach().cpu().tolist()
    return ", ".join(f"{name}={value:.1f}" for name, value in zip(LEROBOT_JOINT_ORDER, numbers, strict=True))


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

    stacked_actions = torch.stack(actions, dim=0)
    return LeRobotActionEpisode(
        episode_index=episode_index,
        fps=_dataset_fps(dataset),
        action_names=tuple(LEROBOT_JOINT_FEATURE_ORDER),
        actions=stacked_actions,
    )


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
) -> tuple[gym.Env, list[str]]:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.scene.robot.init_state.joint_pos = dict(INITIAL_ROBOT_JOINT_POS)
    object_asset_names = configure_env_cfg_for_object_pool(env_cfg, object_pool)
    configure_contact_solver_position_iterations(env_cfg, args_cli.contact_solver_position_iterations)
    print(
        f"[INFO]: Contact solver position iterations: {args_cli.contact_solver_position_iterations} "
        "(bin and object rigid bodies)."
    )
    env_cfg.events.reset_benchmark_scene.params.update(
        _episode_reset_params(first_episode, first_episode_layout, object_pool, object_asset_names)
    )
    return gym.make(args_cli.task, cfg=env_cfg), object_asset_names


def _print_episode_setup(env) -> None:
    episodes = getattr(env.unwrapped, "so101_bench_episodes", [])
    if not episodes:
        return
    episode = episodes[0]
    active_assets = ", ".join(episode.get("active_asset_names", []))
    active_labels = ", ".join(episode.get("active_labels", []))
    print(
        "[INFO]: Active tabletop object(s): "
        f"{active_assets or 'unknown'} ({active_labels or 'unknown'})"
    )


def _episode_end_reason(env, terminated, truncated, term_log: dict) -> str:
    if bool(term_log.get("Episode_Termination/success", 0.0) > 0.0):
        return "success"

    failure_reasons = getattr(env.unwrapped, "_so101_failure_reasons", None)
    if failure_reasons:
        active_env_ids = torch.nonzero(terminated, as_tuple=False).flatten().tolist()
        for env_id in active_env_ids:
            reason = failure_reasons[env_id]
            if reason != "none":
                return reason

    if bool(term_log.get("Episode_Termination/failure", 0.0) > 0.0):
        return "failure"
    if bool(truncated.any().item()):
        return "time_out"
    return "unknown"


def _begin_robot_control(env, object_asset_names: list[str]) -> None:
    mark_benchmark_robot_start(
        env.unwrapped,
        object_asset_names=object_asset_names,
        bin_name="plastic_bin",
        force_robot_start_time=True,
    )


def _initial_robot_action(env) -> torch.Tensor:
    return torch.tensor(
        [INITIAL_ROBOT_JOINT_POS[joint_name] for joint_name in ACTION_JOINT_NAMES],
        dtype=torch.float32,
        device=env.unwrapped.device,
    )


def _restore_robot_initial_pose(env) -> None:
    robot = env.unwrapped.scene["robot"]
    joint_ids = [robot.joint_names.index(joint_name) for joint_name in ACTION_JOINT_NAMES]
    joint_pos = _initial_robot_action(env).unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
    joint_vel = torch.zeros_like(joint_pos)
    robot.data.default_joint_pos[:, joint_ids] = joint_pos
    robot.data.default_joint_vel[:, joint_ids] = joint_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids)
    robot.set_joint_position_target(joint_pos, joint_ids=joint_ids)
    robot.write_data_to_sim()


def _reset_env(env) -> tuple[dict, dict]:
    obs, info = env.reset()
    _restore_robot_initial_pose(env)
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


def _print_initial_scene(env, object_asset_names: list[str]) -> None:
    unwrapped = env.unwrapped
    print(f"[INFO]: Episode instruction: {getattr(unwrapped, 'so101_bench_instruction', '')}")

    active_mask = getattr(unwrapped, "_so101_active_object_mask", None)
    reset_params = unwrapped.cfg.events.reset_benchmark_scene.params
    object_labels = reset_params.get("object_labels", OBJECT_LABELS)
    object_positions = benchmark_object_positions(unwrapped, object_asset_names)
    for object_id, asset_name in enumerate(object_asset_names):
        label = object_labels[object_id] if object_id < len(object_labels) else asset_name
        pos = object_positions[0, object_id].detach().cpu().tolist()
        active = bool(active_mask[0, object_id].item()) if active_mask is not None else True
        state = "active" if active else "inactive"
        print(
            f"[INFO]: Initial {asset_name} / {label} ({state}): "
            f"x={pos[0]:.5f}, y={pos[1]:.5f}, z={pos[2]:.5f}"
        )

    bin_asset = unwrapped.scene["plastic_bin"]
    bin_pos = bin_asset.data.root_pos_w[0].detach().cpu().tolist()
    print(f"[INFO]: Initial plastic_bin: x={bin_pos[0]:.5f}, y={bin_pos[1]:.5f}, z={bin_pos[2]:.5f}")


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
    invalid_indices = [
        index for index in benchmark_indices if index < 0 or index >= len(episode_specs)
    ]
    if invalid_indices:
        raise ValueError(
            f"Requested benchmark episode indices {invalid_indices}, but "
            f"{args_cli.episodes_jsonl} contains {len(episode_specs)} validated row(s)."
        )
    return [episode_specs[index] for index in benchmark_indices]


def _print_final_score(completed: int, successes: int, skipped: int, exhausted: int) -> None:
    evaluated = completed - skipped
    rate = 100.0 * successes / max(evaluated, 1)
    print(
        "[INFO]: Replay summary: "
        f"completed={completed}, success={successes}/{evaluated} ({rate:.1f}%), "
        f"skipped={skipped}, exhausted_without_done={exhausted}"
    )


def main():
    if args_cli.dataset_episode_index < 0:
        raise ValueError(f"Expected --dataset_episode_index >= 0, got {args_cli.dataset_episode_index}.")
    if args_cli.speed <= 0.0:
        raise ValueError(f"Expected --speed > 0, got {args_cli.speed}.")
    if args_cli.initial_hold_time_s < 0.0:
        raise ValueError(f"Expected --initial_hold_time_s >= 0, got {args_cli.initial_hold_time_s}.")
    if args_cli.hold_last_action_time_s < 0.0:
        raise ValueError(f"Expected --hold_last_action_time_s >= 0, got {args_cli.hold_last_action_time_s}.")

    episode_specs = load_episode_jsonl(args_cli.episodes_jsonl)
    if args_cli.benchmark_episode_indices:
        benchmark_indices = _parse_episode_indices(args_cli.benchmark_episode_indices)
        if args_cli.inspect_initial_scene:
            benchmark_indices = benchmark_indices[:1]
        episode_plan = _episode_selection(episode_specs, benchmark_indices)
        planned_count = len(episode_plan)
    else:
        benchmark_start = (
            args_cli.dataset_episode_index
            if args_cli.benchmark_episode_index is None
            else args_cli.benchmark_episode_index
        )
        planned_count = 1 if args_cli.inspect_initial_scene else args_cli.num_episodes
        episode_plan, benchmark_indices = _episode_window(
            episode_specs,
            start_index=benchmark_start,
            count=planned_count,
        )
    episode_layouts = _load_episode_layouts(episode_plan, benchmark_indices, args_cli.episode_layouts_jsonl)

    print(f"[INFO]: Loaded {len(episode_specs)} validated JSONL episode(s) from {args_cli.episodes_jsonl}.")
    print(
        "[INFO]: Replay mapping: "
        f"dataset episodes {args_cli.dataset_episode_index}"
        f"..{args_cli.dataset_episode_index + planned_count - 1} -> "
        f"benchmark rows {benchmark_indices[0]}..{benchmark_indices[-1]}"
    )

    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    np.random.seed(args_cli.seed)

    object_pool = _episode_object_pool(episode_plan)
    print(f"[INFO]: Pre-spawning {len(object_pool)} benchmark object asset(s): {', '.join(object_pool)}")

    env, object_asset_names = _make_env(object_pool, episode_plan[0], episode_layouts[0])
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    control_dt = float(env.unwrapped.step_dt)
    physics_dt = float(env.unwrapped.cfg.sim.dt)
    render_dt = physics_dt * int(env.unwrapped.cfg.sim.render_interval)
    initial_hold_steps = max(0, math.ceil(args_cli.initial_hold_time_s / control_dt))
    hold_last_steps = max(0, math.ceil(args_cli.hold_last_action_time_s / control_dt))
    print(
        "[INFO]: Timing: "
        f"physics_dt={physics_dt:.6f}s, control_dt={control_dt:.6f}s, render_dt={render_dt:.6f}s"
    )
    if initial_hold_steps > 0:
        print(f"[INFO]: Initial hold: {initial_hold_steps} steps ({initial_hold_steps * control_dt:.3f}s)")
    if hold_last_steps > 0:
        print(f"[INFO]: Final hold: {hold_last_steps} steps ({hold_last_steps * control_dt:.3f}s)")

    if args_cli.inspect_initial_scene:
        _reset_env(env)
        _print_initial_scene(env, object_asset_names)
        print("[INFO]: Inspecting initial scene. Close the Isaac app window to exit; physics is not being stepped.")
        while simulation_app.is_running():
            simulation_app.update()
        env.close()
        return

    mapper = SO101ReplayActionMapper(device=env.unwrapped.device)
    expected_reset_pose = mapper.sim_radians_to_lerobot_positions(_initial_robot_action(env))
    print(f"[INFO]: Expected sim reset pose: {_format_lerobot_pose(expected_reset_pose)}")

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    hold_action = _initial_robot_action(env)
    actions[:] = hold_action

    controls = _ReplayControls(
        terminal_enabled=args_cli.terminal_control_stdin,
        debug=args_cli.keyboard_debug,
    )

    completed = 0
    successes = 0
    skipped = 0
    exhausted = 0
    quit_requested = False

    try:
        for offset, (episode, benchmark_index, episode_layout) in enumerate(
            zip(episode_plan, benchmark_indices, episode_layouts, strict=True)
        ):
            if not simulation_app.is_running() or quit_requested:
                break

            dataset_episode_index = args_cli.dataset_episode_index + offset
            print(
                f"[INFO]: Loading LeRobot dataset episode {dataset_episode_index} "
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

            print(
                f"[INFO]: Resetting benchmark row {benchmark_index} for replay "
                f"{offset + 1}/{planned_count}..."
            )
            _configure_env_for_episode(env, episode, episode_layout, object_pool, object_asset_names)
            _obs, _ = _reset_env(env)
            _print_episode_setup(env)
            print(f"[INFO]: Episode instruction: {getattr(env.unwrapped, 'so101_bench_instruction', '')}")
            print(
                f"[INFO]: Replaying {action_episode.num_frames} frame(s) at dataset_fps={action_episode.fps:.3f}."
            )

            step = 0
            frame_index = 0
            robot_control_started = False
            last_info: dict[str, Any] = {}
            done_seen = False
            episode_skipped = False
            episode_exhausted = False

            while simulation_app.is_running():
                skip_requested, quit_requested = controls.poll()
                if quit_requested:
                    print("[INFO]: Replay quit requested.")
                    break
                if skip_requested:
                    episode_skipped = True
                    skipped += 1
                    print(f"[INFO]: Replay {offset + 1}/{planned_count}: skipped by user.")
                    break

                if controls.paused:
                    env.unwrapped.sim.render()
                    time.sleep(0.02)
                    continue

                replay_step_start = time.perf_counter()
                with torch.inference_mode():
                    if step < initial_hold_steps:
                        actions[:] = hold_action
                    else:
                        if not robot_control_started:
                            _begin_robot_control(env, object_asset_names)
                            robot_control_started = True

                        replay_step = step - initial_hold_steps
                        if replay_step < action_episode.num_frames:
                            action_lerobot = mapper.clamp_lerobot_positions(action_episode.actions[frame_index])
                            actions[:] = mapper.lerobot_positions_to_sim_radians(action_lerobot)
                            frame_index += 1
                        elif replay_step < action_episode.num_frames + hold_last_steps:
                            pass
                        else:
                            episode_exhausted = True
                            exhausted += 1
                            print(
                                f"[INFO]: Replay {offset + 1}/{planned_count}: action stream exhausted "
                                f"after {frame_index} frame(s)."
                            )
                            break

                    _obs, _rewards, terminated, truncated, info = env.step(actions)
                    last_info = info
                    step += 1

                is_done = bool(terminated.any().item() or truncated.any().item())
                if is_done and not done_seen:
                    done_seen = True
                    term_log = last_info.get("log", {})
                    is_success = bool(term_log.get("Episode_Termination/success", 0.0) > 0.0)
                    end_reason = _episode_end_reason(env, terminated, truncated, term_log)
                    successes += int(is_success)
                    episode_duration_s = step * control_dt
                    print(
                        f"[INFO]: Replay {offset + 1}/{planned_count}: success={is_success}, "
                        f"reason={end_reason}, length={episode_duration_s:.2f}s, "
                        f"frames_played={frame_index}/{action_episode.num_frames}"
                    )
                    if not args_cli.continue_after_done:
                        break
                elif is_done and not args_cli.continue_after_done:
                    break

                if args_cli.real_time:
                    dt_s = time.perf_counter() - replay_step_start
                    time.sleep(max((control_dt / args_cli.speed) - dt_s, 0.0))

            if quit_requested:
                break
            if episode_skipped:
                completed += 1
                continue
            if episode_exhausted and not done_seen:
                print(
                    f"[INFO]: Replay {offset + 1}/{planned_count}: no done signal before action exhaustion "
                    f"(sim_steps={step}, frames_played={frame_index})."
                )
            completed += 1

        _print_final_score(completed, successes, skipped, exhausted)
    finally:
        controls.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
