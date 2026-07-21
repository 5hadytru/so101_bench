# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperate SO-101 Bench in sim using a real follower arm or Xbox/gamepad virtual leader."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime
import errno
import html
import inspect
import json
import math
import os
from pathlib import Path
import queue
import random
import re
import sys
import struct
import threading
import time
from typing import Any
import weakref

from isaaclab.app import AppLauncher


DEFAULT_ACTION_VELOCITY_LIMIT_UNITS_PER_S = (110.0, 140.0, 150.0, 125.0, 110.0, 120.0)
ACTION_VELOCITY_LIMIT_JOINT_COUNT = 6
DEBUG_TASKS_PRINT_INTERVAL_S = 5.0
TASK_STATUS_UI_UPDATE_INTERVAL_S = 0.1


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _parse_action_velocity_limits(value: str) -> tuple[float, ...] | None:
    normalized = value.strip().lower()
    if normalized in {"none", "off", "false", "0"}:
        return None

    parts = [part.strip() for part in value.replace(";", ",").split(",")]
    if len(parts) != ACTION_VELOCITY_LIMIT_JOINT_COUNT:
        raise argparse.ArgumentTypeError(
            f"Expected {ACTION_VELOCITY_LIMIT_JOINT_COUNT} comma-separated velocity limits, got {value!r}."
        )

    limits = []
    for part in parts:
        try:
            limit = float(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Expected numeric velocity limits, got {value!r}.") from exc
        if not math.isfinite(limit) or limit <= 0.0:
            raise argparse.ArgumentTypeError(f"Velocity limits must be finite positive numbers, got {value!r}.")
        limits.append(limit)
    return tuple(limits)


parser = argparse.ArgumentParser(
    description="SO-101 Bench sim teleoperation with a real SO-101 follower arm or Xbox/gamepad leader."
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="So101Bench-Bin-v0", help="Isaac Lab task name.")
parser.add_argument("--seed", type=int, default=1984, help="Environment seed.")
parser.add_argument(
    "--num_episodes",
    type=int,
    default=None,
    help=(
        "Optional total JSONL episode cap, counted from the beginning of the file. "
        "If omitted, use every row."
    ),
)
parser.add_argument(
    "--start_episode",
    type=int,
    default=None,
    help=(
        "1-based JSONL episode number to start from. Overrides automatic resume from the existing "
        "LeRobot dataset."
    ),
)
parser.add_argument(
    "--n_skipped",
    type=int,
    default=0,
    help=(
        "Number of JSONL episodes skipped so far without saving to the LeRobot dataset. Added to the "
        "automatic resume offset so recording starts at saved episodes + skipped episodes + 1."
    ),
)
parser.add_argument(
    "--resume_from_dataset",
    nargs="?",
    const=True,
    default=True,
    type=_str_to_bool,
    help=(
        "If --repo_root already contains a valid LeRobot dataset, start at total_episodes + 1. "
        "Accepts either '--resume_from_dataset' or '--resume_from_dataset false'."
    ),
)
parser.add_argument(
    "--episodes_jsonl",
    type=Path,
    required=True,
    help="Required JSONL file defining benchmark episodes, matching scripts/groot_eval.py.",
)
parser.add_argument(
    "--object_pool_episodes_jsonl",
    type=Path,
    default=None,
    help=(
        "Optional full episode JSONL used only to define the pre-spawned object-pool order. "
        "Use this when recording a subset so its Object_N/PhysX actor ordering matches the canonical run."
    ),
)
parser.add_argument(
    "--episode_layouts_jsonl",
    "--layouts_jsonl",
    type=Path,
    default=None,
    help=(
        "Optional JSONL file with object and bin poses to apply as-is. Rows are matched to episodes by "
        "trial_id when present; otherwise they are consumed in order. Provided layouts are not revalidated."
    ),
)
parser.add_argument(
    "--sample_random_valid_spatial_layout",
    action="store_true",
    default=False,
    help=(
        "For next-to, between, and move tasks, sample uniformly from every valid generated layout "
        "instead of the retained top valid layouts."
    ),
)
parser.add_argument(
    "--debug_object_placement",
    action="store_true",
    default=False,
    help=(
        "Save a text/JSONL explanation and one top-down SVG per selected layout beside the layouts JSONL. "
        "Provided layouts are rechecked against the current spatial feasibility rules."
    ),
)
parser.add_argument(
    "--leader",
    "--teleop_source",
    choices=("follower", "xbox"),
    default=os.getenv("SO101_TELEOP_LEADER", "follower").lower(),
    help="Teleoperation source: a real SO-101 follower arm or an Xbox/gamepad virtual joint controller.",
)
parser.add_argument(
    "--xbox",
    dest="leader",
    action="store_const",
    const="xbox",
    help="Shortcut for '--leader xbox'; do not connect to a real SO-101 follower arm.",
)
parser.add_argument(
    "--follower_port",
    type=str,
    default=os.getenv("SO101_FOLLOWER_PORT", "/dev/ttyACM0"),
    help="Serial port for the real SO-101 follower arm that will be hand-guided as the leader.",
)
parser.add_argument(
    "--follower_id",
    type=str,
    default=os.getenv("SO101_FOLLOWER_ID", "follower_arm_1"),
    help="LeRobot id for the real SO-101 follower arm calibration.",
)
parser.add_argument(
    "--xbox_index",
    type=int,
    default=0,
    help="Gamepad index to read when using '--leader xbox'.",
)
parser.add_argument(
    "--xbox_backend",
    choices=("auto", "omni", "linux"),
    default="auto",
    help="Gamepad input backend. 'linux' polls /dev/input/js* directly.",
)
parser.add_argument(
    "--xbox_device",
    type=Path,
    default=None,
    help="Linux joystick device path for '--xbox_backend linux', e.g. /dev/input/js0.",
)
parser.add_argument(
    "--xbox_dead_zone",
    type=float,
    default=0.08,
    help="Analog dead zone for Xbox/gamepad joint control.",
)
parser.add_argument(
    "--xbox_joint_speed",
    type=float,
    default=55.0,
    help="Non-gripper joint speed in LeRobot position units per second for Xbox/gamepad control.",
)
parser.add_argument(
    "--xbox_gripper_speed",
    type=float,
    default=90.0,
    help="Legacy Xbox/gamepad gripper speed; commanded SO-101 jaw is overridden by keyboard Up/Down.",
)
parser.add_argument(
    "--keyboard_gripper_speed",
    type=float,
    default=90.0,
    help="Gripper speed in LeRobot position units per second for keyboard Up/Down control.",
)
parser.add_argument(
    "--xbox_debug",
    action="store_true",
    default=False,
    help="Print every gamepad event seen by the Xbox/gamepad leader.",
)
parser.add_argument(
    "--disable_follower_torque",
    nargs="?",
    const=True,
    default=True,
    type=_str_to_bool,
    help=(
        "Disable torque after connecting so the follower can be moved by hand. "
        "Accepts either '--disable_follower_torque' or '--disable_follower_torque false'."
    ),
)
parser.add_argument(
    "--calibrate_on_connect",
    nargs="?",
    const=True,
    default=True,
    type=_str_to_bool,
    help=(
        "Allow LeRobot to run or write calibration during follower connection if needed. "
        "Accepts either '--calibrate_on_connect' or '--calibrate_on_connect false'."
    ),
)
parser.add_argument(
    "--initial_hold_time_s",
    type=float,
    default=0.0,
    help="Seconds to hold the initial sim joint pose before teleoperation starts.",
)
parser.add_argument(
    "--action_smoothing",
    type=float,
    default=0.0,
    help="Exponential smoothing factor for leader actions in [0, 1). 0 disables smoothing.",
)
parser.add_argument(
    "--action_velocity_limit_units_per_s",
    "--action_velocity_limits",
    type=_parse_action_velocity_limits,
    default=DEFAULT_ACTION_VELOCITY_LIMIT_UNITS_PER_S,
    help=(
        "Comma-separated LeRobot-position velocity limits for "
        "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper. "
        "Use 'none' or 'off' to disable. Default: "
        + ",".join(f"{limit:g}" for limit in DEFAULT_ACTION_VELOCITY_LIMIT_UNITS_PER_S)
    ),
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="5hadytru/so101_bench_sim_2",
    help="LeRobot dataset repo id used for local dataset metadata.",
)
parser.add_argument(
    "--repo_root",
    type=Path,
    default=Path("data/lerobot/so101_bench_sim_2"),
    help="Local root directory for the LeRobot dataset.",
)
parser.add_argument(
    "--task_name",
    type=str,
    default=None,
    help="Fixed LeRobot task string. If omitted, each JSONL episode instruction is saved as the task.",
)
parser.add_argument(
    "--dataset_streaming_encoding",
    nargs="?",
    const=True,
    default=True,
    type=_str_to_bool,
    help=(
        "Encode LeRobot videos while recording instead of after each episode. "
        "Accepts either '--dataset_streaming_encoding' or '--dataset_streaming_encoding false'."
    ),
)
parser.add_argument(
    "--dataset_vcodec",
    type=str,
    default="libsvtav1",
    help=(
        "Video codec for LeRobot recording. The default libsvtav1 writes AV1 videos, matching "
        "so101_bench_real_2 for lerobot-edit-dataset merge compatibility. Use 'auto' for faster local "
        "recording only if you do not plan to merge with that dataset."
    ),
)
parser.add_argument(
    "--dataset_encoder_threads",
    type=int,
    default=2,
    help="Threads per streaming video encoder. Use 0 to let the codec choose.",
)
parser.add_argument(
    "--dataset_encoder_queue_size",
    type=int,
    default=300,
    help="Max queued frames per camera for streaming encoding; larger values reduce drops at higher memory cost.",
)
parser.add_argument(
    "--dataset_image_writer_processes",
    type=int,
    default=0,
    help="Async image writer process count used only when streaming video encoding is disabled.",
)
parser.add_argument(
    "--dataset_image_writer_threads_per_camera",
    type=int,
    default=4,
    help="Async image writer threads per camera used only when streaming video encoding is disabled.",
)
parser.add_argument(
    "--dataset_video_files_size_mb",
    type=int,
    default=200,
    help="LeRobot video file rollover size in MB. The default matches so101_bench_real_2.",
)
parser.add_argument(
    "--no_record",
    action="store_true",
    default=False,
    help="Run teleop without initializing or writing a LeRobot dataset.",
)
parser.add_argument(
    "--auto_record",
    action="store_true",
    default=False,
    help="Automatically start recording each episode when teleoperation begins.",
)
parser.add_argument(
    "--advance_on_stop",
    nargs="?",
    const=True,
    default=True,
    type=_str_to_bool,
    help="Advance to the next JSONL episode after accepting a recording with stop/S.",
)
parser.add_argument(
    "--end_on_success",
    action="store_true",
    default=False,
    help="Let benchmark success terminate teleop episodes. By default success does not end teleop.",
)
parser.add_argument(
    "--end_on_failure",
    action="store_true",
    default=False,
    help="Let benchmark failure terminate teleop episodes. By default failures do not end teleop.",
)
parser.add_argument(
    "--debug_tasks",
    action="store_true",
    default=False,
    help=(
        "Track benchmark success/failure conditions while teleoperating and print detailed statuses "
        "every 5 seconds. Also disable episode timeout auto-end behavior."
    ),
)
parser.add_argument(
    "--task_status_ui",
    "--show_task_status",
    nargs="?",
    const=True,
    default=False,
    type=_str_to_bool,
    help=(
        "Show live success/failure and geometric measurements in the Isaac UI for next-to, between, "
        "and move tasks. Disabled by default so those conditions are not computed unless another option, "
        "such as --debug_tasks or --end_on_success/--end_on_failure, requests them."
    ),
)
parser.add_argument(
    "--save_failed_episodes",
    dest="save_failed_episodes",
    action="store_true",
    default=True,
    help="Save an in-progress recording when the env ends in an enabled failure condition or timeout.",
)
parser.add_argument(
    "--discard_failed_episodes",
    dest="save_failed_episodes",
    action="store_false",
    help="Cancel an in-progress recording when the env ends in failure or timeout.",
)
parser.add_argument(
    "--inspect_initial_scene",
    action="store_true",
    default=False,
    help="Reset the first task, print/view initial poses, and exit only when the Isaac app closes.",
)
parser.add_argument(
    "--render_warmup_frames",
    type=int,
    default=16,
    help=(
        "Number of render() passes after each reset before recomputing observations. RTX dome-light sampling "
        "can be dark for the first few frames after a scene reset; lower this if reset latency matters."
    ),
)
parser.add_argument(
    "--terminal_control_stdin",
    nargs="?",
    const=True,
    default=True,
    type=_str_to_bool,
    help="Accept start/stop/cancel/reset/next commands typed in the launch terminal.",
)
parser.add_argument(
    "--keyboard_debug",
    action="store_true",
    default=False,
    help="Print every Isaac keyboard press/release seen by the teleop control listener.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.sim.utils.stage import get_current_stage

import so101_bench.tasks  # noqa: F401
from so101_bench.benchmark import (
    BenchmarkEpisodeSpec,
    TASK_BETWEEN,
    TASK_BIN,
    TASK_MOVE,
    TASK_NAMED_BIN,
    TASK_NEXT_TO,
    load_episode_jsonl,
)
from so101_bench.layouts import (
    DEFAULT_BIN_FOOTPRINT_HALF_EXTENTS,
    DEFAULT_OBJECT_FOOTPRINT_HALF_EXTENTS,
    generate_episode_layout,
    normalize_layout_object_slots,
)
from so101_bench.mdp import (
    benchmark_failure,
    benchmark_object_positions,
    mark_benchmark_robot_start,
    task_condition_diagnostics,
    task_success,
)
from so101_bench.tasks.direct.so101_bench.so101_bench_env_cfg import (
    BIN_RANDOM_POSES,
    OBJECT_LABELS,
    SO101_BOUNDING_BOX,
    TABLE_BOUNDS,
    TABLE_OBJECT_Z,
    VALID_OBJECT_SPAWN_REGIONS,
    configure_env_cfg_for_object_pool,
)
from so101_bench.utils.lerobot_calibration import (
    LEROBOT_INITIAL_JOINT_POS,
    LEROBOT_JOINT_FEATURE_ORDER,
    LEROBOT_JOINT_ORDER,
)
from so101_bench.utils.lerobot_dataset import (
    LeRobotSimDatasetRecorder,
    SO101CalibrationMapper,
    dataset_cameras as _dataset_cameras,
    real_compatible_camera_sources as _real_compatible_camera_sources,
    recording_images as _recording_images,
)
from pxr import UsdGeom, UsdLux


ACTION_JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
GRIPPER_JOINT_INDEX = LEROBOT_JOINT_ORDER.index("gripper")
BEDROOM_DOME_LIGHT_PATH = "/World/BedroomDomeLight"
BEDROOM_DOME_LIGHT_INTENSITY = 800.0
BEDROOM_DOME_LIGHT_COLOR = (1.0, 0.96, 0.88)
BEDROOM_DOME_LIGHT_TEMPERATURE_K = 6500.0
OBSOLETE_EXTRA_LIGHT_PATHS = ("/World/BedroomKeyLight", "/World/BedroomFillLight")


def _normalize_keyboard_key(key: str) -> str:
    return key.strip().upper().replace("-", "_").replace(" ", "_")


def _matches_keyboard_key(event_name: str, key: str) -> bool:
    event_name = _normalize_keyboard_key(event_name)
    key = _normalize_keyboard_key(key)
    return event_name in {key, f"KEY_{key}"}


def _normalize_terminal_command(command: str) -> str:
    return command.strip().lower().replace("-", "_").replace(" ", "_")


class _TeleopControls:
    """Keyboard and terminal commands for teleop collection."""

    def __init__(self, *, terminal_enabled: bool, debug: bool):
        self._events: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._input = None
        self._keyboard = None
        self._keyboard_sub = None
        self._key_press_type = None
        self._key_release_type = None
        self._gripper_keys_down: set[str] = set()
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
            self._key_release_type = carb.input.KeyboardEventType.KEY_RELEASE
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                self._keyboard,
                self._on_keyboard_event,
            )
            print(
                "[INFO]: Keyboard controls: S start episode, C cancel, R retry, Enter save/next, "
                "N next, Q finish, Up opens gripper, Down closes gripper."
            )
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
        print(
            "[INFO]: Terminal controls: start, stop, save, cancel, retry, next, finish "
            "(type command then Enter; blank Enter saves and advances)."
        )

    def _maybe_queue_terminal_command(self, line: str) -> None:
        command = _normalize_terminal_command(line)
        aliases = {
            "": "save_and_next",
            "s": "start_episode",
            "start": "start_episode",
            "record": "start_episode",
            "stop": "stop_recording",
            "save": "save_and_next",
            "save_next": "save_and_next",
            "save_and_next": "save_and_next",
            "c": "cancel_recording",
            "cancel": "cancel_recording",
            "r": "reset_episode",
            "retry": "reset_episode",
            "reset": "reset_episode",
            "n": "next_episode",
            "next": "next_episode",
            "skip": "next_episode",
            "q": "finish_session",
            "quit": "finish_session",
            "exit": "finish_session",
            "finish": "finish_session",
            "done": "finish_session",
        }
        event = aliases.get(command)
        if event is None:
            print(
                "[WARN]: Unknown terminal command. "
                "Use start, stop, save, cancel, retry, next, or finish."
            )
            return
        self._events.put(event)

    def queue_event(self, event: str) -> None:
        self._events.put(event)

    def _on_keyboard_event(self, event, *args, **kwargs):
        event_name = getattr(getattr(event, "input", None), "name", "")
        is_press = event.type == self._key_press_type
        is_release = event.type == self._key_release_type
        if (is_press or is_release) and self._debug:
            event_type = "press" if is_press else "release"
            print(f"[DEBUG]: Isaac key {event_type}: {event_name!r}")
        if not is_press and not is_release:
            return False

        if _matches_keyboard_key(event_name, "UP"):
            if is_press:
                self._gripper_keys_down.add("UP")
            else:
                self._gripper_keys_down.discard("UP")
            return True
        if _matches_keyboard_key(event_name, "DOWN"):
            if is_press:
                self._gripper_keys_down.add("DOWN")
            else:
                self._gripper_keys_down.discard("DOWN")
            return True

        if not is_press:
            return False

        if _matches_keyboard_key(event_name, "S"):
            self._events.put("start_episode")
            return True
        if _matches_keyboard_key(event_name, "C"):
            self._events.put("cancel_recording")
            return True
        if _matches_keyboard_key(event_name, "R"):
            self._events.put("reset_episode")
            return True
        if (
            _matches_keyboard_key(event_name, "ENTER")
            or _matches_keyboard_key(event_name, "RETURN")
            or _matches_keyboard_key(event_name, "NUMPAD_ENTER")
        ):
            self._events.put("save_and_next")
            return True
        if _matches_keyboard_key(event_name, "N"):
            self._events.put("next_episode")
            return True
        if _matches_keyboard_key(event_name, "Q"):
            self._events.put("finish_session")
            return True
        return False

    def poll(self) -> list[str]:
        events = []
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            events.append(event)
        return events

    def gripper_command(self) -> float:
        up_pressed = "UP" in self._gripper_keys_down
        down_pressed = "DOWN" in self._gripper_keys_down
        if up_pressed == down_pressed:
            return 0.0
        return 1.0 if up_pressed else -1.0

    def close(self) -> None:
        if self._input is None or self._keyboard is None or self._keyboard_sub is None:
            return
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None


_TASK_STATUS_METRIC_RE_TEMPLATE = r"(?:^|[;,]\s*){name}=([^,;\s)]+)"


def _task_status_ui_text(diagnostics) -> str:
    """Format the concise spatial-task status shown in the Isaac UI."""
    if not diagnostics:
        return "Task status: waiting for task state"

    snapshot = diagnostics[0]
    success_conditions = [condition for condition in snapshot.conditions if condition.kind == "success"]
    failed_conditions = [
        condition for condition in snapshot.conditions if condition.kind == "failure" and condition.met
    ]
    if failed_conditions:
        status = "FAILURE (" + ", ".join(condition.name for condition in failed_conditions) + ")"
    elif any(condition.met for condition in success_conditions):
        status = "SUCCESS"
    else:
        status = "IN PROGRESS"

    success_details = success_conditions[0].details if success_conditions else ""

    def metric(name: str) -> str:
        match = re.search(_TASK_STATUS_METRIC_RE_TEMPLATE.format(name=re.escape(name)), success_details)
        return match.group(1) if match is not None else "unavailable"

    lines = [f"Task status: {status}"]
    if snapshot.task_family == TASK_NEXT_TO:
        lines.append(f"Target/referent surface distance: {metric('surface_distance')}")
    elif snapshot.task_family == TASK_BETWEEN:
        lines.append(f"Perpendicular distance: {metric('perpendicular_distance')}")
    elif snapshot.task_family == TASK_MOVE:
        lines.append(f"Distance to boundary: {metric('distance_to_boundary')}")
        lines.append(f"Trajectory straightness (lateral error): {metric('current_lateral_error')}")
    return "\n".join(lines)


class _SimClockRateWindow:
    """Omni UI window showing the instruction, sim speed, and optional task status."""

    def __init__(
        self,
        *,
        control_dt: float,
        show_task_status: bool = False,
        update_interval_s: float = 0.5,
    ):
        self._control_dt = control_dt
        self._show_task_status = show_task_status
        self._update_interval_s = update_interval_s
        self._window = None
        self._instruction_label = None
        self._task_status_label = None
        self._rate_label = None
        self._fps_label = None
        self._sim_time_s = 0.0
        self._last_update_wall_s = time.perf_counter()
        self._last_update_sim_s = 0.0
        self._target_fps = 1.0 / control_dt if control_dt > 0.0 else 0.0
        self._create_window()
        self.update(force=True)

    def _create_window(self) -> None:
        try:
            import omni.ui as ui

            height = 184 if self._show_task_status else 126
            self._window = ui.Window("SO-101 Teleop", width=620, height=height)
            with self._window.frame:
                with ui.VStack(spacing=4):
                    self._instruction_label = ui.Label("Instruction: --", word_wrap=True)
                    self._rate_label = ui.Label("Sim speed: --")
                    self._fps_label = ui.Label("Sim FPS: --")
                    if self._show_task_status:
                        self._task_status_label = ui.Label(
                            "Task status: waiting for episode start",
                            word_wrap=True,
                        )
        except Exception as exc:
            print(f"[WARN]: Teleop status UI unavailable: {exc}")
            self._window = None
            self._instruction_label = None
            self._task_status_label = None
            self._rate_label = None
            self._fps_label = None

    def set_instruction(self, instruction: str) -> None:
        if self._instruction_label is not None:
            self._instruction_label.text = f"Instruction: {instruction}"

    def set_task_status(self, status: str) -> None:
        if self._task_status_label is not None:
            self._task_status_label.text = status

    def reset(self) -> None:
        self._sim_time_s = 0.0
        now = time.perf_counter()
        self._last_update_wall_s = now
        self._last_update_sim_s = 0.0
        if self._task_status_label is not None:
            self._task_status_label.text = "Task status: waiting for episode start"
        self.update(force=True)

    def add_step(self) -> None:
        self._sim_time_s += self._control_dt
        self.update()

    def update(self, *, force: bool = False) -> None:
        if self._rate_label is None or self._fps_label is None:
            return

        now = time.perf_counter()
        wall_dt = now - self._last_update_wall_s
        if not force and wall_dt < self._update_interval_s:
            return

        sim_dt = self._sim_time_s - self._last_update_sim_s
        rate = sim_dt / wall_dt if wall_dt > 1.0e-9 else 0.0
        sim_fps = rate * self._target_fps

        self._rate_label.text = f"Sim speed: {rate:.2f}x wall clock"
        self._fps_label.text = f"Sim FPS: {sim_fps:.1f} / target {self._target_fps:.1f}"
        self._last_update_wall_s = now
        self._last_update_sim_s = self._sim_time_s

    def close(self) -> None:
        if self._window is not None:
            self._window.visible = False
        self._window = None
        self._instruction_label = None
        self._task_status_label = None
        self._rate_label = None
        self._fps_label = None


class SO101FollowerLeader:
    """Read a real SO-101 follower arm as a passive LeRobot-position teleop source."""

    def __init__(
        self,
        *,
        port: str,
        robot_id: str,
        device: str,
        disable_torque: bool,
        calibrate_on_connect: bool,
    ):
        self.port = port
        self.robot_id = robot_id
        self.device = device
        self.disable_torque = disable_torque
        self.calibrate_on_connect = calibrate_on_connect
        self.robot = None

    @staticmethod
    def _import_lerobot_so101():
        try:
            from lerobot.robots.so101_follower import SO101FollowerConfig
        except ImportError:
            try:
                from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
            except ImportError:
                from lerobot.robots.so_follower import SO101FollowerConfig
        from lerobot.robots import make_robot_from_config

        return SO101FollowerConfig, make_robot_from_config

    def connect(self) -> None:
        SO101FollowerConfig, make_robot_from_config = self._import_lerobot_so101()
        signature = inspect.signature(SO101FollowerConfig)
        cfg_kwargs = {"port": self.port}
        if "id" in signature.parameters:
            cfg_kwargs["id"] = self.robot_id
        if "cameras" in signature.parameters:
            cfg_kwargs["cameras"] = {}
        if "disable_torque_on_disconnect" in signature.parameters:
            cfg_kwargs["disable_torque_on_disconnect"] = True
        if "use_degrees" in signature.parameters:
            cfg_kwargs["use_degrees"] = False

        cfg = SO101FollowerConfig(**cfg_kwargs)
        self.robot = make_robot_from_config(cfg)
        connect_signature = inspect.signature(self.robot.connect)
        if "calibrate" in connect_signature.parameters:
            self.robot.connect(calibrate=self.calibrate_on_connect)
        else:
            self.robot.connect()
        print(f"[INFO]: Connected follower arm '{self.robot_id}' on {self.port}.")
        if self.disable_torque:
            self.disable_motor_torque()

    def disable_motor_torque(self) -> None:
        if self.robot is None:
            return
        bus = getattr(self.robot, "bus", None)
        if bus is not None and hasattr(bus, "disable_torque"):
            bus.disable_torque()
            print("[INFO]: Disabled follower torque; support the arm while hand-guiding it.")
            return
        if hasattr(self.robot, "disable_torque"):
            self.robot.disable_torque()
            print("[INFO]: Disabled follower torque; support the arm while hand-guiding it.")
            return
        print("[WARN]: Could not find a torque-disable method on the follower arm.")

    def read_action(self) -> torch.Tensor:
        if self.robot is None:
            raise RuntimeError("Follower arm is not connected.")
        observation = self.robot.get_observation()
        values = []
        for joint_index, key in enumerate(LEROBOT_JOINT_FEATURE_ORDER):
            if joint_index == GRIPPER_JOINT_INDEX:
                values.append(float(LEROBOT_INITIAL_JOINT_POS["gripper"]))
                continue
            if key in observation:
                values.append(float(observation[key]))
                continue
            bare_key = key.removesuffix(".pos")
            if bare_key in observation:
                values.append(float(observation[bare_key]))
                continue
            raise KeyError(f"Follower observation is missing {key!r}; got keys: {list(observation.keys())}")
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def poll_events(self) -> None:
        pass

    def reset(self, action_lerobot: torch.Tensor) -> None:
        pass

    def close(self) -> None:
        if self.robot is None:
            return
        try:
            if hasattr(self.robot, "disconnect"):
                self.robot.disconnect()
        except Exception as exc:
            print(f"[WARN]: Error while disconnecting follower arm: {exc}")
        self.robot = None


class SO101XboxLeader:
    """Read an Xbox/gamepad as a virtual SO-101 follower in LeRobot-position space."""

    def __init__(
        self,
        *,
        mapper: SO101CalibrationMapper,
        initial_action_lerobot: torch.Tensor,
        control_dt: float,
        gamepad_index: int,
        backend: str,
        device_path: Path | None,
        joint_speed: float,
        gripper_speed: float,
        dead_zone: float,
        event_sink: Callable[[str], None] | None,
        debug: bool,
    ):
        self.mapper = mapper
        self.device = mapper.device
        self.control_dt = control_dt
        self.gamepad_index = gamepad_index
        self.backend = backend
        self.device_path = device_path
        self.dead_zone = dead_zone
        self.event_sink = event_sink
        self.debug = debug
        self.joint_speeds = torch.tensor(
            [joint_speed, joint_speed, joint_speed, joint_speed, joint_speed, gripper_speed],
            dtype=torch.float32,
            device=self.device,
        )
        self._action = self._bounded_action(initial_action_lerobot).clone()
        self._reset_action = self._action.clone()
        self._command_raw = np.zeros((2, 6), dtype=np.float32)
        self._button_pressed: dict[Any, bool] = {}
        self._input_mapping = {}
        self._button_event_mapping = {}
        self._linux_trigger_axis_mapping = {
            2: (1, 5),  # LT closes/decreases gripper
            5: (0, 5),  # RT opens/increases gripper
        }
        self._linux_button_axis_mapping = {
            4: (1, 5),  # LB
            5: (0, 5),  # RB
        }
        self._linux_button_event_mapping = {
            0: "toggle_recording",  # A
            1: "cancel_recording",  # B
            3: "reset_episode",  # Y
            6: "next_episode",  # Back/Menu1
        }
        self._linux_reset_button = 2  # X
        self._input = None
        self._gamepad = None
        self._gamepad_sub = None
        self._linux_js_fd: int | None = None
        self._linux_js_path: Path | None = None

    def _bounded_action(self, action_lerobot: torch.Tensor) -> torch.Tensor:
        action_lerobot = action_lerobot.to(device=self.device, dtype=torch.float32)
        return torch.minimum(torch.maximum(action_lerobot, self.mapper.lerobot_mins), self.mapper.lerobot_maxs)

    def connect(self) -> None:
        connected = False
        if self.backend in ("auto", "linux"):
            connected = self._connect_linux_joystick()
            if not connected and self.backend == "linux":
                path = self._linux_candidate_path()
                raise RuntimeError(f"Could not open Linux joystick device {path}.")

        if self.backend in ("auto", "omni"):
            connected = self._connect_omni_gamepad() or connected
            if not connected and self.backend == "omni":
                raise RuntimeError("No Isaac app window found; Xbox/gamepad controls are unavailable.")

        if not connected:
            raise RuntimeError("Could not connect an Xbox/gamepad through Omniverse or /dev/input/js*.")

        print(
            "[INFO]: Xbox controls: left stick pan/lift, right stick elbow/roll, D-pad wrist pitch, "
            "X reset virtual pose. Use keyboard Up/Down for gripper."
        )
        if self.event_sink is not None:
            print("[INFO]: Xbox recording controls: A start/stop, B cancel, Y reset, Menu1 next.")

    def _connect_omni_gamepad(self) -> bool:
        try:
            import carb
            import carb.input
            import omni.appwindow

            carb.settings.get_settings().set_bool("/persistent/app/omniverse/gamepadCameraControl", False)
            app_window = omni.appwindow.get_default_app_window()
            if app_window is None:
                return False

            self._create_gamepad_bindings(carb.input.GamepadInput)
            self._input = carb.input.acquire_input_interface()
            self._gamepad = app_window.get_gamepad(self.gamepad_index)
            self._gamepad_sub = self._input.subscribe_to_gamepad_events(
                self._gamepad,
                lambda event, *args, obj=weakref.proxy(self): obj._on_gamepad_event(event, *args),
            )
            gamepad_name = self._input.get_gamepad_name(self._gamepad)
            print(
                f"[INFO]: Connected Omniverse Xbox/gamepad leader {self.gamepad_index}: "
                f"{gamepad_name or 'unknown device'}."
            )
            return True
        except Exception as exc:
            if self.backend == "omni":
                raise
            print(f"[WARN]: Omniverse gamepad backend unavailable: {exc}")
            return False

    def _linux_candidate_path(self) -> Path:
        if self.device_path is not None:
            return self.device_path
        return Path(f"/dev/input/js{self.gamepad_index}")

    def _connect_linux_joystick(self) -> bool:
        path = self._linux_candidate_path()
        try:
            self._linux_js_fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.EACCES, errno.EPERM):
                print(f"[WARN]: Could not open Linux joystick device {path}: {exc.strerror}.")
                return False
            raise
        self._linux_js_path = path
        print(f"[INFO]: Connected Linux joystick backend: {path}.")
        return True

    def _create_gamepad_bindings(self, gamepad_input) -> None:
        self._input_mapping = {
            gamepad_input.LEFT_STICK_RIGHT: (0, 0),
            gamepad_input.LEFT_STICK_LEFT: (1, 0),
            gamepad_input.LEFT_STICK_UP: (0, 1),
            gamepad_input.LEFT_STICK_DOWN: (1, 1),
            gamepad_input.RIGHT_STICK_UP: (0, 2),
            gamepad_input.RIGHT_STICK_DOWN: (1, 2),
            gamepad_input.DPAD_UP: (0, 3),
            gamepad_input.DPAD_DOWN: (1, 3),
            gamepad_input.RIGHT_STICK_RIGHT: (0, 4),
            gamepad_input.RIGHT_STICK_LEFT: (1, 4),
            gamepad_input.RIGHT_TRIGGER: (0, 5),
            gamepad_input.LEFT_TRIGGER: (1, 5),
            gamepad_input.RIGHT_SHOULDER: (0, 5),
            gamepad_input.LEFT_SHOULDER: (1, 5),
        }
        self._button_event_mapping = {
            gamepad_input.A: "toggle_recording",
            gamepad_input.B: "cancel_recording",
            gamepad_input.Y: "reset_episode",
            gamepad_input.MENU1: "next_episode",
        }
        self._reset_button = gamepad_input.X

    def _on_gamepad_event(self, event, *args, **kwargs):
        cur_val = float(event.value)
        if abs(cur_val) < self.dead_zone:
            cur_val = 0.0
        input_name = getattr(event.input, "name", str(event.input))
        if self.debug:
            print(f"[DEBUG]: Xbox/gamepad event: {input_name}={cur_val:.3f}")

        if event.input in self._input_mapping:
            direction, axis = self._input_mapping[event.input]
            self._command_raw[direction, axis] = max(cur_val, 0.0)

        if self._is_new_button_press(event.input, cur_val):
            if event.input == self._reset_button:
                self._center_on_reset_pose()
                print("[INFO]: Reset Xbox/gamepad virtual pose to the current episode's initial SO-101 pose.")
            elif self.event_sink is not None and event.input in self._button_event_mapping:
                self.event_sink(self._button_event_mapping[event.input])
        return True

    def _poll_linux_joystick(self) -> None:
        if self._linux_js_fd is None:
            return
        while True:
            try:
                payload = os.read(self._linux_js_fd, 8)
            except BlockingIOError:
                return
            except OSError as exc:
                if exc.errno == errno.EAGAIN:
                    return
                raise
            if len(payload) < 8:
                return
            _event_time_ms, value, event_type, number = struct.unpack("IhBB", payload)
            event_type = event_type & ~0x80
            if event_type == 0x01:
                self._on_linux_button(number, value)
            elif event_type == 0x02:
                self._on_linux_axis(number, value)

    def _on_linux_button(self, number: int, value: int) -> None:
        pressed = value > 0
        if self.debug:
            print(f"[DEBUG]: Linux joystick button {number}={int(pressed)}")

        if number in self._linux_button_axis_mapping:
            direction, axis = self._linux_button_axis_mapping[number]
            self._command_raw[direction, axis] = 1.0 if pressed else 0.0

        if not self._is_new_button_press(("linux", number), 1.0 if pressed else 0.0):
            return
        if number == self._linux_reset_button:
            self._center_on_reset_pose()
            print("[INFO]: Reset Xbox/gamepad virtual pose to the current episode's initial SO-101 pose.")
        elif self.event_sink is not None and number in self._linux_button_event_mapping:
            self.event_sink(self._linux_button_event_mapping[number])

    def _on_linux_axis(self, number: int, value: int) -> None:
        normalized = float(value) / 32767.0
        if abs(normalized) < self.dead_zone:
            normalized = 0.0
        normalized = max(-1.0, min(1.0, normalized))
        if self.debug:
            print(f"[DEBUG]: Linux joystick axis {number}={normalized:.3f}")

        if number in self._linux_trigger_axis_mapping:
            direction, axis = self._linux_trigger_axis_mapping[number]
            trigger_value = (normalized + 1.0) * 0.5
            self._command_raw[direction, axis] = 0.0 if trigger_value < self.dead_zone else trigger_value
            return

        if number == 7:
            self._set_axis_command(axis=3, signed_value=-normalized)
            return
        if number == 0:
            self._set_axis_command(axis=0, signed_value=normalized)
            return
        if number == 1:
            self._set_axis_command(axis=1, signed_value=-normalized)
            return
        if number == 3:
            self._set_axis_command(axis=4, signed_value=normalized)
            return
        if number == 4:
            self._set_axis_command(axis=2, signed_value=-normalized)

    def _set_axis_command(self, *, axis: int, signed_value: float) -> None:
        self._command_raw[:, axis] = 0.0
        if signed_value > 0.0:
            self._command_raw[0, axis] = signed_value
        elif signed_value < 0.0:
            self._command_raw[1, axis] = -signed_value

    def _is_new_button_press(self, gamepad_input, value: float) -> bool:
        is_pressed = value > 0.5
        was_pressed = self._button_pressed.get(gamepad_input, False)
        self._button_pressed[gamepad_input] = is_pressed
        return is_pressed and not was_pressed

    @staticmethod
    def _resolve_command_buffer(raw_command: np.ndarray) -> np.ndarray:
        command_sign = raw_command[1, :] > raw_command[0, :]
        command = raw_command.max(axis=0)
        command[command_sign] *= -1.0
        return command

    def _center_on_reset_pose(self) -> None:
        self._action = self._reset_action.clone()
        self._command_raw.fill(0.0)

    def reset(self, action_lerobot: torch.Tensor) -> None:
        self._action = self._bounded_action(action_lerobot).clone()
        self._reset_action = self._action.clone()
        self._command_raw.fill(0.0)

    def read_action(self) -> torch.Tensor:
        self.poll_events()
        command = torch.tensor(
            self._resolve_command_buffer(self._command_raw),
            dtype=torch.float32,
            device=self.device,
        )
        self._action = self._bounded_action(self._action + command * self.joint_speeds * self.control_dt)
        return self._action.clone()

    def poll_events(self) -> None:
        self._poll_linux_joystick()

    def close(self) -> None:
        if self._input is not None and self._gamepad is not None and self._gamepad_sub is not None:
            try:
                self._input.unsubscribe_to_gamepad_events(self._gamepad, self._gamepad_sub)
            except Exception as exc:
                print(f"[WARN]: Error while disconnecting Xbox/gamepad leader: {exc}")
            self._gamepad_sub = None
        if self._linux_js_fd is not None:
            os.close(self._linux_js_fd)
            self._linux_js_fd = None


class KeyboardJawController:
    """Own the SO-101 jaw target while the arm joints follow the selected teleop leader."""

    def __init__(
        self,
        *,
        mapper: SO101CalibrationMapper,
        initial_action_lerobot: torch.Tensor,
        control_dt: float,
        speed: float,
    ):
        self.device = mapper.device
        self.control_dt = control_dt
        self.speed = speed
        self._min = mapper.lerobot_mins[GRIPPER_JOINT_INDEX]
        self._max = mapper.lerobot_maxs[GRIPPER_JOINT_INDEX]
        self._position = self._min.clone()
        self.reset(initial_action_lerobot)

    def reset(self, action_lerobot: torch.Tensor) -> None:
        action_lerobot = action_lerobot.to(device=self.device, dtype=torch.float32)
        gripper_position = action_lerobot[GRIPPER_JOINT_INDEX].detach().clone()
        self._position = torch.minimum(torch.maximum(gripper_position, self._min), self._max)

    def apply(self, action_lerobot: torch.Tensor, command: float) -> torch.Tensor:
        delta = float(command) * self.speed * self.control_dt
        if delta != 0.0:
            self._position = torch.minimum(torch.maximum(self._position + delta, self._min), self._max)
        overridden_action = action_lerobot.clone()
        overridden_action[GRIPPER_JOINT_INDEX] = self._position
        return overridden_action


def _discover_cameras(env) -> dict[str, dict[str, int]]:
    cameras = {}
    for scene_key in env.unwrapped.scene.keys():
        if not scene_key.startswith("camera_"):
            continue
        camera_cfg = getattr(env.unwrapped.scene.cfg, scene_key)
        camera_name = scene_key.replace("camera_", "")
        cameras[camera_name] = {"height": camera_cfg.height, "width": camera_cfg.width}
        print(f"[INFO]: Found camera '{camera_name}' ({camera_cfg.width}x{camera_cfg.height})")
    return cameras


def _instruction(env, override: str | None) -> str:
    if override:
        return override
    return getattr(env.unwrapped, "so101_bench_instruction", "Place each object in the plastic bin.")


def _timestamped_layout_path(episodes_jsonl: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = episodes_jsonl.parent / "layouts" if episodes_jsonl.parent.name == "tasks" else Path("tasks/layouts")
    return output_dir / f"{episodes_jsonl.stem}_layouts_{timestamp}.jsonl"


def _object_placement_debug_dir(layout_path: Path) -> Path:
    return layout_path.parent / f"{layout_path.stem}_object_placement_debug"


def _debug_rotate_xy(point: tuple[float, float], yaw: float) -> tuple[float, float]:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        cos_yaw * point[0] - sin_yaw * point[1],
        sin_yaw * point[0] + cos_yaw * point[1],
    )


def _debug_footprint_vertices(
    center: tuple[float, float],
    half_extents: tuple[float, float],
    center_offset: tuple[float, float],
    yaw: float,
) -> list[tuple[float, float]]:
    offset_x, offset_y = _debug_rotate_xy(center_offset, yaw)
    footprint_center = (center[0] + offset_x, center[1] + offset_y)
    vertices = []
    for local_x, local_y in (
        (-half_extents[0], -half_extents[1]),
        (half_extents[0], -half_extents[1]),
        (half_extents[0], half_extents[1]),
        (-half_extents[0], half_extents[1]),
    ):
        x, y = _debug_rotate_xy((local_x, local_y), yaw)
        vertices.append((footprint_center[0] + x, footprint_center[1] + y))
    return vertices


def _debug_layout_object_vertices(entry: dict[str, Any]) -> list[tuple[float, float]]:
    position = entry["position"]
    half_extents = entry.get("footprint_half_extents", DEFAULT_OBJECT_FOOTPRINT_HALF_EXTENTS)
    center_offset = entry.get("footprint_center_offset", (0.0, 0.0))
    yaw = float(entry.get("yaw", entry.get("rpy", (0.0, 0.0, 0.0))[2]))
    return _debug_footprint_vertices(
        (float(position[0]), float(position[1])),
        (float(half_extents[0]), float(half_extents[1])),
        (float(center_offset[0]), float(center_offset[1])),
        yaw,
    )


def _debug_layout_bin_vertices(layout: dict[str, Any]) -> list[tuple[float, float]]:
    entry = layout["bin"]
    position = entry["position"]
    half_extents = entry.get("footprint_half_extents", DEFAULT_BIN_FOOTPRINT_HALF_EXTENTS)
    center_offset = entry.get("footprint_center_offset", (0.0, 0.0))
    return _debug_footprint_vertices(
        (float(position[0]), float(position[1])),
        (float(half_extents[0]), float(half_extents[1])),
        (float(center_offset[0]), float(center_offset[1])),
        float(entry["rpy"][2]),
    )


def _debug_object_placement_lines(
    layout: dict[str, Any],
    episode: BenchmarkEpisodeSpec,
    task_feasibility: dict[str, Any] | None,
) -> list[str]:
    placement = layout.get("placement", {})
    lines = [
        f"trial_id={layout.get('trial_id')!r}, episode_index={layout.get('episode_index')!r}",
        f"task_family={episode.task_family}",
        f"instruction={episode.instruction}",
        (
            f"sampler={placement.get('layout_selection', 'provided')}, "
            f"valid_attempts={placement.get('valid_attempts', 'unknown')}/"
            f"{placement.get('attempts', 'unknown')}"
        ),
        (
            f"selected_min_object_gap_m={placement.get('min_between_object_surface_distance_m')!r}, "
            f"selected_min_bin_gap_m={placement.get('min_bin_surface_distance_m')!r}"
        ),
        f"rejection_counts={placement.get('rejection_counts', {})}",
    ]
    if episode.task_family in {TASK_BIN, TASK_NAMED_BIN}:
        lines.append("task_feasibility=not applicable for bin placement")
    elif task_feasibility is None:
        lines.append("task_feasibility=not evaluated (placement is task-agnostic)")
    else:
        lines.append("task_feasibility=accepted by current rules")

    if task_feasibility is None:
        return lines
    if episode.task_family == TASK_BETWEEN:
        lines.extend(
            (
                (
                    "between_initial_distance_to_success_region_m="
                    f"{task_feasibility['initial_target_distance_to_success_region_m']:.5f} "
                    "(required >="
                    f"{task_feasibility['required_min_initial_target_distance_to_success_region_m']:.5f})"
                ),
                (
                    "between_referent_segment_robot_gap_m="
                    f"{task_feasibility.get('referent_segment_robot_surface_distance_m')!r}"
                ),
                (
                    f"between_accepted_pose={task_feasibility['feasible_target_position']}, "
                    f"segment_fraction={task_feasibility['feasible_target_segment_fraction']:.4f}"
                ),
            )
        )
    elif episode.task_family == TASK_NEXT_TO:
        lines.append(
            "next_to_initial_target_referent_gap_m="
            f"{task_feasibility['initial_target_referent_surface_distance_m']:.5f}"
        )
        lines.append(f"next_to_accepted_pose={task_feasibility['feasible_target_position']}")
    elif episode.task_family == TASK_MOVE:
        selected_boundary = task_feasibility.get("initial_target_selected_boundary")
        clear_path_m = float(
            task_feasibility.get(
                "initial_target_clear_path_m",
                task_feasibility.get("initial_target_min_boundary_gap_m", 0.0),
            )
        )
        nearest_object_gap = task_feasibility.get("initial_target_min_swept_object_surface_distance_m")
        lines.append(
            f"move_clear_path_m={clear_path_m:.5f} "
            f"(required >={task_feasibility['required_min_clear_path_m']:.5f})"
        )
        lines.append(
            f"move_forward_table_gap_m={task_feasibility['initial_target_forward_table_gap_m']:.5f}, "
            f"nearest_swept_object_gap_m={nearest_object_gap!r}"
        )
        if selected_boundary is not None:
            boundary_id = int(selected_boundary["boundary_id"])
            lines.append(
                f"move_boundary=object_{boundary_id + 1}, "
                f"gap_m={float(selected_boundary['surface_gap_m']):.5f}, "
                f"overlap_m={float(selected_boundary['lateral_overlap_m']):.5f} "
                f"(required >={float(selected_boundary['required_min_lateral_overlap_m']):.5f})"
            )
    return lines


def _debug_object_placement_svg(
    layout: dict[str, Any],
    episode: BenchmarkEpisodeSpec,
    task_feasibility: dict[str, Any] | None,
) -> str:
    object_entries = sorted(layout["objects"], key=lambda entry: int(entry["slot"]))
    object_vertices = [_debug_layout_object_vertices(entry) for entry in object_entries]
    bin_vertices = _debug_layout_bin_vertices(layout)
    table_vertices = [
        (float(TABLE_BOUNDS["x"][0]), float(TABLE_BOUNDS["y"][0])),
        (float(TABLE_BOUNDS["x"][1]), float(TABLE_BOUNDS["y"][0])),
        (float(TABLE_BOUNDS["x"][1]), float(TABLE_BOUNDS["y"][1])),
        (float(TABLE_BOUNDS["x"][0]), float(TABLE_BOUNDS["y"][1])),
    ]
    robot_vertices = [(float(point[0]), float(point[1])) for point in SO101_BOUNDING_BOX]
    all_vertices = [*table_vertices, *robot_vertices, *bin_vertices]
    for vertices in object_vertices:
        all_vertices.extend(vertices)

    ghost_vertices = None
    if task_feasibility is not None and "feasible_target_position" in task_feasibility:
        target_entry = object_entries[int(episode.target_object_id)]
        half_extents = target_entry.get("footprint_half_extents", DEFAULT_OBJECT_FOOTPRINT_HALF_EXTENTS)
        center_offset = target_entry.get("footprint_center_offset", (0.0, 0.0))
        ghost_vertices = _debug_footprint_vertices(
            tuple(float(value) for value in task_feasibility["feasible_target_position"]),
            (float(half_extents[0]), float(half_extents[1])),
            (float(center_offset[0]), float(center_offset[1])),
            float(task_feasibility["feasible_target_yaw"]),
        )
        all_vertices.extend(ghost_vertices)

    min_x = min(point[0] for point in all_vertices) - 0.025
    max_x = max(point[0] for point in all_vertices) + 0.025
    min_y = min(point[1] for point in all_vertices) - 0.025
    max_y = max(point[1] for point in all_vertices) + 0.025
    plot_x, plot_y, plot_width, plot_height = 42.0, 82.0, 700.0, 620.0

    def canvas(point: tuple[float, float]) -> tuple[float, float]:
        x = plot_x + (point[0] - min_x) / (max_x - min_x) * plot_width
        y = plot_y + (max_y - point[1]) / (max_y - min_y) * plot_height
        return x, y

    def points(vertices: list[tuple[float, float]]) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in (canvas(vertex) for vertex in vertices))

    colors = ("#e76f51", "#2a9d8f", "#e9c46a", "#6c63ff")
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="760" viewBox="0 0 1280 760">',
        "<defs><marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" "
        "orient=\"auto\"><polygon points=\"0 0, 10 3.5, 0 7\" fill=\"#9b2226\"/></marker></defs>",
        '<rect width="1280" height="760" fill="#fbf7ef"/>',
        f'<text x="42" y="38" font-size="22" font-family="sans-serif" font-weight="bold">'
        f'{html.escape(str(layout.get("trial_id")))}: {html.escape(episode.instruction)}</text>',
        f'<polygon points="{points(table_vertices)}" fill="#f4efe4" stroke="#61564a" stroke-width="2"/>',
        f'<polygon points="{points(bin_vertices)}" fill="#94d2bd" fill-opacity="0.55" '
        'stroke="#0a9396" stroke-width="2"/>',
        f'<polygon points="{points(robot_vertices)}" fill="#adb5bd" fill-opacity="0.75" '
        'stroke="#495057" stroke-width="2"/>',
    ]
    robot_x, robot_y = canvas(robot_vertices[0])
    svg.append(f'<text x="{robot_x:.1f}" y="{robot_y - 8:.1f}" font-size="13" font-family="sans-serif">robot base</text>')

    if episode.task_family == TASK_BETWEEN:
        ref_a = object_entries[int(episode.referent_object_ids[0])]["position"]
        ref_b = object_entries[int(episode.referent_object_ids[1])]["position"]
        a_x, a_y = canvas((float(ref_a[0]), float(ref_a[1])))
        b_x, b_y = canvas((float(ref_b[0]), float(ref_b[1])))
        svg.append(
            f'<line x1="{a_x:.1f}" y1="{a_y:.1f}" x2="{b_x:.1f}" y2="{b_y:.1f}" '
            'stroke="#9b2226" stroke-width="3" stroke-dasharray="8 5"/>'
        )
    elif episode.task_family == TASK_MOVE and task_feasibility is not None:
        origin = task_feasibility["initial_target_move_ray_origin"]
        axis = int(task_feasibility["axis"])
        sign = float(task_feasibility["sign"])
        gap = float(
            task_feasibility.get(
                "initial_target_clear_path_m",
                task_feasibility.get("initial_target_min_boundary_gap_m", 0.0),
            )
        )
        arrow_end = [float(origin[0]), float(origin[1])]
        arrow_end[axis] += sign * gap
        origin_x, origin_y = canvas((float(origin[0]), float(origin[1])))
        end_x, end_y = canvas((arrow_end[0], arrow_end[1]))
        svg.append(
            f'<line x1="{origin_x:.1f}" y1="{origin_y:.1f}" x2="{end_x:.1f}" y2="{end_y:.1f}" '
            'stroke="#9b2226" stroke-width="4" marker-end="url(#arrow)"/>'
        )

    for index, (entry, vertices) in enumerate(zip(object_entries, object_vertices, strict=True)):
        color = colors[index % len(colors)]
        svg.append(
            f'<polygon points="{points(vertices)}" fill="{color}" fill-opacity="0.55" '
            f'stroke="{color}" stroke-width="2"/>'
        )
        label_x, label_y = canvas((float(entry["position"][0]), float(entry["position"][1])))
        svg.append(
            f'<text x="{label_x + 6:.1f}" y="{label_y - 6:.1f}" font-size="13" '
            f'font-family="sans-serif">{index}: {html.escape(str(entry.get("name", "object")))}</text>'
        )
    if ghost_vertices is not None:
        svg.append(
            f'<polygon points="{points(ghost_vertices)}" fill="none" stroke="#9b2226" '
            'stroke-width="3" stroke-dasharray="7 4"/>'
        )
        ghost_x, ghost_y = canvas(tuple(float(value) for value in task_feasibility["feasible_target_position"]))
        svg.append(
            f'<text x="{ghost_x + 6:.1f}" y="{ghost_y + 18:.1f}" font-size="13" '
            'font-family="sans-serif" fill="#9b2226">accepted target pose</text>'
        )

    for index, line in enumerate(_debug_object_placement_lines(layout, episode, task_feasibility)):
        svg.append(
            f'<text x="780" y="{92 + 24 * index}" font-size="14" font-family="monospace">'
            f"{html.escape(line)}</text>"
        )
    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def _write_object_placement_debug_artifacts(
    episode_plan: list[BenchmarkEpisodeSpec],
    episode_layouts: list[dict],
    layout_path: Path,
) -> Path:
    output_dir = _object_placement_debug_dir(layout_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = []
    debug_records = []
    index_items = []
    for episode, layout in zip(episode_plan, episode_layouts, strict=True):
        # Placement is task-agnostic; per-task feasibility annotations are no longer computed.
        task_feasibility = None
        trial_id = layout.get("trial_id", layout.get("episode_index"))
        svg_name = f"trial_{trial_id}_episode_{layout.get('episode_index', 'unknown')}.svg"
        (output_dir / svg_name).write_text(
            _debug_object_placement_svg(layout, episode, task_feasibility),
            encoding="utf-8",
        )
        lines = _debug_object_placement_lines(layout, episode, task_feasibility)
        summary_lines.extend([*lines, f"visual={svg_name}", ""])
        debug_records.append(
            {
                "trial_id": trial_id,
                "episode_index": layout.get("episode_index"),
                "task_family": episode.task_family,
                "instruction": episode.instruction,
                "placement": layout.get("placement", {}),
                "visual": svg_name,
            }
        )
        index_items.append(
            f'<li><a href="{html.escape(svg_name)}">{html.escape(str(trial_id))}: '
            f'{html.escape(episode.instruction)}</a><br/><img src="{html.escape(svg_name)}" width="960"/></li>'
        )

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    with (output_dir / "summary.jsonl").open("w", encoding="utf-8") as file:
        for record in debug_records:
            file.write(json.dumps(record, separators=(",", ":")) + "\n")
    (output_dir / "index.html").write_text(
        "<!doctype html><html><body><h1>SO-101 object placement debug</h1><ol>"
        + "".join(index_items)
        + "</ol></body></html>\n",
        encoding="utf-8",
    )
    print(f"[INFO]: Saved object-placement debug artifacts: {output_dir}")
    return summary_path


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
    layout_path: Path,
    *,
    episode_start_index: int = 0,
) -> list[dict]:
    available_layouts = _load_layout_jsonl(layout_path)
    requested_trial_ids = [
        _episode_trial_id(episode, episode_start_index + index) for index, episode in enumerate(episode_plan)
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
        required_rows = episode_start_index + len(episode_plan)
        if len(available_layouts) >= required_rows:
            episode_layouts = available_layouts[episode_start_index:required_rows]
        elif len(available_layouts) >= len(episode_plan):
            print(
                "[WARN]: Provided layouts do not cover the requested start offset; "
                "using the first layout row for the first requested episode."
            )
            episode_layouts = available_layouts[: len(episode_plan)]
        else:
            raise ValueError(
                f"{layout_path} contains {len(available_layouts)} layout row(s), "
                f"but {len(episode_plan)} episode(s) were requested."
            )

    normalized_layouts = []
    for episode_index, (episode, layout) in enumerate(zip(episode_plan, episode_layouts, strict=True)):
        layout = normalize_layout_object_slots(
            layout,
            episode.objects,
            episode_index=episode_start_index + episode_index,
        )
        normalized_layouts.append(layout)
    print(f"[INFO]: Loaded provided initial layouts for {len(normalized_layouts)} episode(s): {layout_path}")
    return normalized_layouts


def _generate_and_save_episode_layouts(
    episode_plan: list[BenchmarkEpisodeSpec],
) -> tuple[list[dict], Path]:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    # Layout diversity should not depend on the deterministic simulator/teleop
    # seed. Use one auditable wall-clock seed for the entire generated file.
    layout_seed = time.time_ns()
    layout_rng = random.Random(layout_seed)
    print(f"[INFO]: Generating episode layouts with time-derived seed {layout_seed}.")
    # Shared across the run so object roots spread away from recent episodes
    # (high inter-episode placement variance).
    placement_history: list[tuple[float, float]] = []
    layouts = [
        generate_episode_layout(
            episode,
            episode_index=episode_index,
            rng=layout_rng,
            bin_random_poses=BIN_RANDOM_POSES,
            valid_spawn_regions=VALID_OBJECT_SPAWN_REGIONS,
            table_object_z=TABLE_OBJECT_Z,
            seed=layout_seed,
            generated_at=generated_at,
            robot_bounding_box=SO101_BOUNDING_BOX,
            placement_history=placement_history,
        )
        for episode_index, episode in tqdm(
            enumerate(episode_plan),
            total=len(episode_plan),
            desc="Generating episode layouts",
            unit="episode",
        )
    ]

    layout_path = _timestamped_layout_path(args_cli.episodes_jsonl)
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    with layout_path.open("w", encoding="utf-8") as file:
        for layout in layouts:
            file.write(json.dumps(layout, separators=(",", ":")) + "\n")
    print(f"[INFO]: Saved replayable initial layouts for {len(layouts)} episode(s): {layout_path}")
    return layouts, layout_path


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
    first_episode_layout: dict,
) -> tuple[gym.Env, list[str], dict[str, Any], dict[str, Any]]:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.num_rerenders_on_reset = max(0, int(args_cli.render_warmup_frames))
    object_asset_names = configure_env_cfg_for_object_pool(env_cfg, object_pool)
    success_term_params = dict(env_cfg.terminations.success.params)
    failure_term_params = dict(env_cfg.terminations.failure.params)
    if not args_cli.end_on_success:
        env_cfg.terminations.success = None
    if not args_cli.end_on_failure:
        env_cfg.terminations.failure = None
    if args_cli.debug_tasks:
        env_cfg.terminations.time_out = None
    env_cfg.events.reset_benchmark_scene.params.update(
        _episode_reset_params(first_episode, first_episode_layout, object_pool, object_asset_names)
    )
    print(
        "[INFO]: Episode auto-end terms: "
        f"success={'enabled' if args_cli.end_on_success else 'disabled'}, "
        f"failure={'enabled' if args_cli.end_on_failure else 'disabled'}, "
        f"timeout={'disabled' if args_cli.debug_tasks else 'enabled'}"
    )
    return gym.make(args_cli.task, cfg=env_cfg), object_asset_names, success_term_params, failure_term_params


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


def _format_grasp_attempts(
    unwrapped,
    object_asset_names: list[str],
    env_id: int,
    max_grasp_attempts: int,
    enforce_max_grasp_attempts: bool,
) -> str:
    """Read-only per-object grasp-attempt counts for one env, for terminal debugging only."""
    attempt_counts = getattr(unwrapped, "_so101_grasp_attempt_counts", None)
    if attempt_counts is None:
        return "grasp_attempts: unavailable (episode not reset yet)"

    active_mask = getattr(unwrapped, "_so101_active_object_mask", None)
    target_ids = getattr(unwrapped, "_so101_target_object_ids", None)
    target_id = int(target_ids[env_id].item()) if target_ids is not None else -1
    reset_params = unwrapped.cfg.events.reset_benchmark_scene.params
    object_labels = reset_params.get("object_labels", OBJECT_LABELS)

    parts = []
    for object_id, asset_name in enumerate(object_asset_names):
        if active_mask is not None and not bool(active_mask[env_id, object_id].item()):
            continue
        label = object_labels[object_id] if object_id < len(object_labels) else asset_name
        count = int(attempt_counts[env_id, object_id].item())
        marker = " [target]" if object_id == target_id else ""
        parts.append(f"{asset_name}/{label}{marker}={count}")
    detail = ", ".join(parts) if parts else "no active objects"
    rule = (
        f"failure if any >{max_grasp_attempts}"
        if enforce_max_grasp_attempts
        else f"limit disabled; would fail if any >{max_grasp_attempts}"
    )
    return f"grasp_attempts ({rule}): {detail}"


# A diagnostic line is treated as unchanged if its only differences from the last
# printed line are numeric jitter below these thresholds (e.g. distances wobbling by
# fractions of a millimetre). A change counts as negligible when every number is within
# EITHER tolerance of its counterpart.
_DEBUG_TASKS_NUMBER_RE = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?")
_DEBUG_TASKS_ABS_TOL = 1e-4
_DEBUG_TASKS_REL_TOL = 0.02


def _debug_line_changed(previous: str, current: str) -> bool:
    """Return True if `current` differs meaningfully from `previous`.

    Non-numeric text must match exactly; numeric tokens are compared with tolerance so
    sub-threshold jitter does not force a reprint.
    """
    if previous == current:
        return False
    prev_nums = _DEBUG_TASKS_NUMBER_RE.findall(previous)
    cur_nums = _DEBUG_TASKS_NUMBER_RE.findall(current)
    if len(prev_nums) != len(cur_nums):
        return True
    # Skeletons (text with numbers stripped out) must be identical.
    if _DEBUG_TASKS_NUMBER_RE.sub("", previous) != _DEBUG_TASKS_NUMBER_RE.sub("", current):
        return True
    for prev_tok, cur_tok in zip(prev_nums, cur_nums):
        prev_val = float(prev_tok)
        cur_val = float(cur_tok)
        diff = abs(prev_val - cur_val)
        if diff <= _DEBUG_TASKS_ABS_TOL:
            continue
        if diff <= _DEBUG_TASKS_REL_TOL * max(abs(prev_val), abs(cur_val)):
            continue
        return True
    return False


def _print_task_diagnostics(
    diagnostics,
    episode_label: str,
    unwrapped,
    object_asset_names: list[str],
    max_grasp_attempts: int,
    enforce_max_grasp_attempts: bool,
    prev_lines: dict[int, dict[str, str]],
) -> None:
    for snapshot in diagnostics:
        # Build the current set of diagnostic lines, keyed so we can diff against
        # the previous print for this env. The episode/age header is metadata and is
        # intentionally excluded from the diff so a constantly-changing age does not
        # force everything to reprint.
        grasp_attempts = _format_grasp_attempts(
            unwrapped,
            object_asset_names,
            snapshot.env_id,
            max_grasp_attempts,
            enforce_max_grasp_attempts,
        )
        lines: dict[str, str] = {
            "grasp": f"[DEBUG TASKS]:   {grasp_attempts}"
        }
        for condition in snapshot.conditions:
            key = f"{condition.kind} {condition.name}"
            lines[key] = (
                f"[DEBUG TASKS]:   {condition.kind} {condition.name}: "
                f"met={condition.met}; {condition.details}"
            )

        previous = prev_lines.get(snapshot.env_id, {})
        # Compare against the last *printed* value (the stored baseline) so that a long
        # run of sub-threshold changes can still accumulate into a printed update. The
        # grasp-attempts line uses exact comparison so that *any* count increase prints,
        # even when the relative jitter tolerance would otherwise absorb a +1 at high
        # counts.
        changed = {
            key: text
            for key, text in lines.items()
            if key not in previous
            or (previous[key] != text if key == "grasp" else _debug_line_changed(previous[key], text))
        }
        # Also surface keys that disappeared since the last print.
        removed = [key for key in previous if key not in lines]

        # Persist the printed value for changed lines; keep the old baseline otherwise.
        baseline = {key: (changed[key] if key in changed else previous[key]) for key in lines}
        prev_lines[snapshot.env_id] = baseline

        if not changed and not removed:
            continue

        print(
            f"[DEBUG TASKS]: Episode {episode_label}, env={snapshot.env_id}, "
            f"task={snapshot.task_family}, age={snapshot.episode_age_s:.2f}s"
        )
        for key in lines:
            if key in changed:
                print(changed[key])
        for key in removed:
            print(f"[DEBUG TASKS]:   {key}: (no longer reported)")


def _begin_robot_control(env, object_asset_names: list[str]) -> None:
    mark_benchmark_robot_start(
        env.unwrapped,
        object_asset_names=object_asset_names,
        bin_name="plastic_bin",
        force_robot_start_time=True,
    )


def _write_robot_action_pose(env, joint_action: torch.Tensor) -> None:
    robot = env.unwrapped.scene["robot"]
    joint_ids = [robot.joint_names.index(joint_name) for joint_name in ACTION_JOINT_NAMES]
    joint_pos = joint_action.to(device=env.unwrapped.device, dtype=torch.float32).unsqueeze(0)
    joint_pos = joint_pos.repeat(env.unwrapped.num_envs, 1)
    joint_vel = torch.zeros_like(joint_pos)
    robot.data.default_joint_pos[:, joint_ids] = joint_pos
    robot.data.default_joint_vel[:, joint_ids] = joint_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids)
    robot.set_joint_position_target(joint_pos, joint_ids=joint_ids)
    robot.write_data_to_sim()


def _ensure_scene_lighting() -> None:
    """Repair the global dome light after reset before RTX sensors render."""

    stage = get_current_stage()
    for light_path in OBSOLETE_EXTRA_LIGHT_PATHS:
        prim = stage.GetPrimAtPath(light_path)
        if prim.IsValid():
            stage.RemovePrim(prim.GetPath())

    prim = stage.GetPrimAtPath(BEDROOM_DOME_LIGHT_PATH)
    if prim.IsValid():
        dome_light = UsdLux.DomeLight(prim)
    else:
        dome_light = UsdLux.DomeLight.Define(stage, BEDROOM_DOME_LIGHT_PATH)
        prim = dome_light.GetPrim()

    UsdGeom.Imageable(prim).MakeVisible()
    dome_light.CreateIntensityAttr().Set(BEDROOM_DOME_LIGHT_INTENSITY)
    dome_light.CreateColorAttr().Set(BEDROOM_DOME_LIGHT_COLOR)
    dome_light.CreateEnableColorTemperatureAttr().Set(True)
    dome_light.CreateColorTemperatureAttr().Set(BEDROOM_DOME_LIGHT_TEMPERATURE_K)


def _reset_env(env, robot_action: torch.Tensor | None = None) -> tuple[dict, dict]:
    obs, info = env.reset()
    _ensure_scene_lighting()
    if robot_action is not None:
        _write_robot_action_pose(env, robot_action)
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


def _clamp_lerobot_positions(mapper: SO101CalibrationMapper, values: torch.Tensor) -> torch.Tensor:
    return torch.minimum(torch.maximum(values, mapper.lerobot_mins), mapper.lerobot_maxs)


def _limit_lerobot_velocity(
    target: torch.Tensor,
    previous: torch.Tensor,
    max_speed: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    max_delta = max_speed * float(dt)
    delta = torch.clamp(target - previous, min=-max_delta, max=max_delta)
    return previous + delta


def _format_lerobot_pose(values: torch.Tensor) -> str:
    numbers = values.detach().cpu().tolist()
    return ", ".join(f"{name}={value:.1f}" for name, value in zip(LEROBOT_JOINT_ORDER, numbers, strict=True))


def _format_lerobot_joint_speeds(values: torch.Tensor) -> str:
    numbers = values.detach().cpu().tolist()
    return ", ".join(f"{name}={value:.1f}" for name, value in zip(LEROBOT_JOINT_ORDER, numbers, strict=True))


def _existing_lerobot_episode_count(dataset_root: Path) -> int:
    info_path = dataset_root / "meta" / "info.json"
    episodes_dir = dataset_root / "meta" / "episodes"
    if not info_path.exists() or not any(episodes_dir.glob("*/*.parquet")):
        return 0
    with info_path.open("r", encoding="utf-8") as file:
        info = json.load(file)
    return int(info.get("total_episodes", 0))


def main():
    if not 0.0 <= args_cli.action_smoothing < 1.0:
        raise ValueError(f"--action_smoothing must be in [0, 1), got {args_cli.action_smoothing}.")
    if not 0.0 <= args_cli.xbox_dead_zone < 1.0:
        raise ValueError(f"--xbox_dead_zone must be in [0, 1), got {args_cli.xbox_dead_zone}.")
    if args_cli.xbox_joint_speed < 0.0:
        raise ValueError(f"--xbox_joint_speed must be non-negative, got {args_cli.xbox_joint_speed}.")
    if args_cli.xbox_gripper_speed < 0.0:
        raise ValueError(f"--xbox_gripper_speed must be non-negative, got {args_cli.xbox_gripper_speed}.")
    if args_cli.keyboard_gripper_speed < 0.0:
        raise ValueError(
            f"--keyboard_gripper_speed must be non-negative, got {args_cli.keyboard_gripper_speed}."
        )
    if args_cli.n_skipped < 0:
        raise ValueError(f"--n_skipped must be non-negative, got {args_cli.n_skipped}.")

    episode_specs = load_episode_jsonl(args_cli.episodes_jsonl)
    episode_limit = len(episode_specs) if args_cli.num_episodes is None else args_cli.num_episodes
    if episode_limit < 1:
        raise ValueError(f"Expected at least one episode, got {episode_limit}.")
    if episode_limit > len(episode_specs):
        raise ValueError(
            f"Requested {episode_limit} episode(s), but {args_cli.episodes_jsonl} contains "
            f"{len(episode_specs)} validated row(s)."
        )
    print(f"[INFO]: Loaded {len(episode_specs)} validated JSONL episode(s) from {args_cli.episodes_jsonl}.")

    existing_dataset_episodes = 0
    if not args_cli.no_record:
        existing_dataset_episodes = _existing_lerobot_episode_count(args_cli.repo_root)

    if args_cli.start_episode is not None:
        if args_cli.start_episode < 1:
            raise ValueError(f"--start_episode is 1-based and must be >= 1, got {args_cli.start_episode}.")
        episode_start_index = args_cli.start_episode - 1
        if existing_dataset_episodes and episode_start_index != existing_dataset_episodes:
            print(
                "[WARN]: Existing LeRobot dataset contains "
                f"{existing_dataset_episodes} episode(s), but teleop will start at JSONL episode "
                f"{args_cli.start_episode}. New recordings will append to the dataset."
            )
        if args_cli.n_skipped:
            print("[WARN]: --start_episode was provided, so --n_skipped is ignored.")
    elif args_cli.resume_from_dataset and not args_cli.no_record:
        episode_start_index = existing_dataset_episodes + args_cli.n_skipped
    else:
        episode_start_index = args_cli.n_skipped

    if episode_start_index >= episode_limit:
        print(
            "[INFO]: No remaining episodes to teleoperate: "
            f"start episode {episode_start_index + 1} is beyond requested episode cap {episode_limit}."
        )
        return

    planned_count = 1 if args_cli.inspect_initial_scene else episode_limit - episode_start_index
    episode_end_index = episode_start_index + planned_count
    episode_plan = episode_specs[episode_start_index:episode_end_index]
    episode_count = episode_limit
    print(
        "[INFO]: Teleop episode range: "
        f"{episode_start_index + 1}-{episode_end_index} of {episode_count} "
        f"(dataset already has {existing_dataset_episodes} episode(s), "
        f"n_skipped={args_cli.n_skipped})."
    )

    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    if args_cli.episode_layouts_jsonl is not None:
        layout_path = args_cli.episode_layouts_jsonl
        episode_layouts = _load_episode_layouts(
            episode_plan,
            layout_path,
            episode_start_index=episode_start_index,
        )
    else:
        generated_layouts, layout_path = _generate_and_save_episode_layouts(episode_specs[:episode_end_index])
        episode_layouts = generated_layouts[episode_start_index:episode_end_index]
    if args_cli.debug_object_placement:
        _write_object_placement_debug_artifacts(episode_plan, episode_layouts, layout_path)

    object_pool_episode_specs = (
        load_episode_jsonl(args_cli.object_pool_episodes_jsonl)
        if args_cli.object_pool_episodes_jsonl is not None
        else episode_plan
    )
    object_pool = _episode_object_pool(object_pool_episode_specs)
    if args_cli.object_pool_episodes_jsonl is not None:
        print(
            f"[INFO]: Using canonical object-pool order from {args_cli.object_pool_episodes_jsonl} "
            f"({len(object_pool_episode_specs)} episode row(s))."
        )
    print(f"[INFO]: Pre-spawning {len(object_pool)} benchmark object asset(s): {', '.join(object_pool)}")

    env, object_asset_names, success_term_params, failure_term_params = _make_env(
        object_pool, episode_plan[0], episode_layouts[0]
    )
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    control_dt = float(env.unwrapped.step_dt)
    physics_dt = float(env.unwrapped.cfg.sim.dt)
    render_dt = physics_dt * int(env.unwrapped.cfg.sim.render_interval)
    initial_hold_steps = max(0, math.ceil(args_cli.initial_hold_time_s / control_dt))
    print(
        "[INFO]: Timing: "
        f"physics_dt={physics_dt:.6f}s, control_dt={control_dt:.6f}s, render_dt={render_dt:.6f}s"
    )
    if initial_hold_steps > 0:
        print(f"[INFO]: Initial hold: {initial_hold_steps} steps ({initial_hold_steps * control_dt:.3f}s)")
    if args_cli.debug_tasks:
        print(
            "[INFO]: Task debugging enabled: observer checks run every control step and detailed statuses print "
            f"every {DEBUG_TASKS_PRINT_INTERVAL_S:.1f}s."
        )
    if args_cli.task_status_ui:
        print(
            "[INFO]: Spatial task status UI enabled: next-to, between, and move measurements update "
            f"every {TASK_STATUS_UI_UPDATE_INTERVAL_S:.1f}s."
        )

    cameras = _discover_cameras(env)
    if not cameras:
        raise RuntimeError("No cameras were found. LeRobot dataset recording requires visual observations.")
    camera_sources = _real_compatible_camera_sources(cameras)
    dataset_cameras = _dataset_cameras(cameras, camera_sources)

    sim_speed_ui = _SimClockRateWindow(
        control_dt=control_dt,
        show_task_status=args_cli.task_status_ui,
    )

    if args_cli.inspect_initial_scene:
        _reset_env(env)
        sim_speed_ui.set_instruction(_instruction(env, args_cli.task_name))
        _print_initial_scene(env, object_asset_names)
        print("[INFO]: Inspecting initial scene. Close the Isaac app window to exit; physics is not being stepped.")
        while simulation_app.is_running():
            simulation_app.update()
        sim_speed_ui.close()
        env.close()
        return

    mapper = SO101CalibrationMapper(device=env.unwrapped.device)
    action_velocity_limits_lerobot = None
    if args_cli.action_velocity_limit_units_per_s is not None:
        action_velocity_limits_lerobot = torch.tensor(
            args_cli.action_velocity_limit_units_per_s,
            dtype=torch.float32,
            device=env.unwrapped.device,
        )
        print(
            "[INFO]: Action velocity limiter enabled "
            f"(LeRobot units/s): {_format_lerobot_joint_speeds(action_velocity_limits_lerobot)}"
        )
    else:
        print("[INFO]: Action velocity limiter disabled.")
    controls = _TeleopControls(
        terminal_enabled=args_cli.terminal_control_stdin,
        debug=args_cli.keyboard_debug,
    )
    xbox_initial_action_lerobot = torch.tensor(
        [LEROBOT_INITIAL_JOINT_POS[joint_name] for joint_name in LEROBOT_JOINT_ORDER],
        dtype=torch.float32,
        device=env.unwrapped.device,
    )
    if args_cli.leader == "xbox":
        leader = SO101XboxLeader(
            mapper=mapper,
            initial_action_lerobot=xbox_initial_action_lerobot,
            control_dt=control_dt,
            gamepad_index=args_cli.xbox_index,
            backend=args_cli.xbox_backend,
            device_path=args_cli.xbox_device,
            joint_speed=args_cli.xbox_joint_speed,
            gripper_speed=args_cli.xbox_gripper_speed,
            dead_zone=args_cli.xbox_dead_zone,
            event_sink=controls.queue_event,
            debug=args_cli.xbox_debug,
        )
    else:
        leader = SO101FollowerLeader(
            port=args_cli.follower_port,
            robot_id=args_cli.follower_id,
            device=env.unwrapped.device,
            disable_torque=args_cli.disable_follower_torque,
            calibrate_on_connect=args_cli.calibrate_on_connect,
        )
    leader.connect()
    first_leader_action = _clamp_lerobot_positions(mapper, leader.read_action())
    if args_cli.leader == "xbox":
        print(f"[INFO]: Initial Xbox/gamepad virtual pose: {_format_lerobot_pose(first_leader_action)}")
    else:
        print(f"[INFO]: Current follower pose: {_format_lerobot_pose(first_leader_action)}")
    keyboard_jaw = KeyboardJawController(
        mapper=mapper,
        initial_action_lerobot=first_leader_action,
        control_dt=control_dt,
        speed=args_cli.keyboard_gripper_speed,
    )
    print("[INFO]: Leader gripper/jaw input is ignored; use keyboard Up to open and Down to close.")

    recorder = None
    if not args_cli.no_record:
        recorder = LeRobotSimDatasetRecorder(
            repo_id=args_cli.repo_id,
            dataset_root=args_cli.repo_root,
            fps=max(1, round(1.0 / control_dt)),
            cameras=dataset_cameras,
            streaming_encoding=args_cli.dataset_streaming_encoding,
            vcodec=args_cli.dataset_vcodec,
            encoder_queue_size=args_cli.dataset_encoder_queue_size,
            encoder_threads=None if args_cli.dataset_encoder_threads == 0 else args_cli.dataset_encoder_threads,
            image_writer_processes=args_cli.dataset_image_writer_processes,
            image_writer_threads_per_camera=args_cli.dataset_image_writer_threads_per_camera,
            video_files_size_mb=args_cli.dataset_video_files_size_mb,
        )
        recorder.init_dataset()
    else:
        print("[INFO]: Dataset recording disabled by --no_record.")

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    hold_action_lerobot = first_leader_action.clone()
    hold_action = mapper.lerobot_positions_to_sim_radians(hold_action_lerobot)
    actions[:] = hold_action
    obs: dict | None = None

    episode_index = 0
    step = 0
    saved_episodes = 0
    cancelled_episodes = 0
    skipped_episodes = 0
    robot_control_started = False
    episode_started = False
    smoothed_action_lerobot: torch.Tensor | None = None
    limited_action_lerobot: torch.Tensor | None = (
        hold_action_lerobot.clone() if action_velocity_limits_lerobot is not None else None
    )
    next_task_debug_print_time = time.monotonic() + DEBUG_TASKS_PRINT_INTERVAL_S
    next_task_status_ui_update_time = time.monotonic()
    task_debug_prev_lines: dict[int, dict[str, str]] = {}

    def _current_task() -> str:
        return _instruction(env, args_cli.task_name)

    def _current_episode_label() -> str:
        return f"{episode_start_index + episode_index + 1}/{episode_count}"

    def _sample_leader_start_pose() -> tuple[torch.Tensor, torch.Tensor]:
        action_lerobot = _clamp_lerobot_positions(mapper, leader.read_action())
        action_sim = mapper.lerobot_positions_to_sim_radians(action_lerobot)
        return action_lerobot, action_sim

    def _reset_action_velocity_limiter(action_lerobot: torch.Tensor) -> None:
        nonlocal limited_action_lerobot
        if action_velocity_limits_lerobot is None:
            limited_action_lerobot = None
        else:
            limited_action_lerobot = action_lerobot.clone()

    def _apply_robot_start_pose_from_leader() -> None:
        nonlocal obs, hold_action, hold_action_lerobot
        hold_action_lerobot, hold_action = _sample_leader_start_pose()
        leader.reset(hold_action_lerobot)
        keyboard_jaw.reset(hold_action_lerobot)
        _reset_action_velocity_limiter(hold_action_lerobot)
        actions[:] = hold_action
        _write_robot_action_pose(env, hold_action)
        env.unwrapped.scene.write_data_to_sim()
        env.unwrapped.sim.forward()
        obs = env.unwrapped.observation_manager.compute(update_history=True)
        env.unwrapped.obs_buf = obs

    def _reset_current_episode() -> None:
        nonlocal obs, hold_action, hold_action_lerobot, step, robot_control_started, episode_started
        nonlocal next_task_debug_print_time, next_task_status_ui_update_time, smoothed_action_lerobot
        print(f"[INFO]: Resetting episode {_current_episode_label()}...")
        hold_action_lerobot, hold_action = _sample_leader_start_pose()
        _configure_env_for_episode(
            env,
            episode_plan[episode_index],
            episode_layouts[episode_index],
            object_pool,
            object_asset_names,
        )
        obs, _ = _reset_env(env, hold_action)
        sim_speed_ui.reset()
        sim_speed_ui.set_instruction(_current_task())
        if args_cli.task_status_ui and episode_plan[episode_index].task_family in {TASK_BIN, TASK_NAMED_BIN}:
            sim_speed_ui.set_task_status("Task status: not computed for bin tasks")
        _print_episode_setup(env)
        print(f"[INFO]: Episode instruction: {_current_task()}")
        leader.reset(hold_action_lerobot)
        keyboard_jaw.reset(hold_action_lerobot)
        _reset_action_velocity_limiter(hold_action_lerobot)
        actions[:] = hold_action
        step = 0
        robot_control_started = False
        episode_started = False
        smoothed_action_lerobot = None
        next_task_debug_print_time = time.monotonic() + DEBUG_TASKS_PRINT_INTERVAL_S
        next_task_status_ui_update_time = time.monotonic()
        task_debug_prev_lines.clear()
        print("[INFO]: Episode ready at timestep 0. Click the Isaac window and press S to start.")

    def _advance_episode() -> bool:
        nonlocal episode_index
        episode_index += 1
        if episode_index >= len(episode_plan):
            return False
        _reset_current_episode()
        return True

    def _start_recording() -> None:
        if recorder is None:
            print("[WARN]: Recording requested, but --no_record is enabled.")
            return
        recorder.start_episode(task=_current_task())

    def _start_episode() -> None:
        nonlocal episode_started, next_task_debug_print_time, next_task_status_ui_update_time, step
        if episode_started:
            return
        _apply_robot_start_pose_from_leader()
        step = 0
        episode_started = True
        next_task_debug_print_time = time.monotonic() + DEBUG_TASKS_PRINT_INTERVAL_S
        next_task_status_ui_update_time = time.monotonic()
        sim_speed_ui.reset()
        if args_cli.task_status_ui and episode_plan[episode_index].task_family in {TASK_BIN, TASK_NAMED_BIN}:
            sim_speed_ui.set_task_status("Task status: not computed for bin tasks")
        if recorder is not None:
            recorder.start_episode(task=_current_task())
        print(
            f"[INFO]: Started episode {_current_episode_label()} from timestep 0 "
            f"using current leader pose: {_format_lerobot_pose(hold_action_lerobot)}"
        )

    def _stop_recording(*, advance: bool) -> bool:
        nonlocal saved_episodes
        if recorder is None:
            return True
        saved = recorder.stop_episode(task=_current_task())
        if saved:
            saved_episodes += 1
        if advance and saved:
            return _advance_episode()
        return True

    def _save_and_next_episode() -> bool:
        nonlocal saved_episodes
        if recorder is None:
            return _advance_episode()
        if not recorder.recording:
            print("[WARN]: Save/next requested, but no recording is active.")
            return True
        saved = recorder.stop_episode(task=_current_task())
        if not saved:
            return True
        saved_episodes += 1
        return _advance_episode()

    def _cancel_recording() -> None:
        nonlocal cancelled_episodes
        if recorder is not None and recorder.recording:
            recorder.cancel_episode()
            cancelled_episodes += 1

    def _finish_session() -> bool:
        nonlocal saved_episodes
        if recorder is not None and recorder.recording:
            saved = recorder.stop_episode(task=_current_task())
            saved_episodes += int(saved)
        print("[INFO]: Finish requested; finalizing the LeRobot dataset and stopping teleop.")
        return False

    def _handle_events(events: list[str]) -> bool:
        nonlocal skipped_episodes
        for event in events:
            if event == "toggle_recording":
                if not episode_started:
                    _start_episode()
                elif recorder is not None and recorder.recording:
                    if not _stop_recording(advance=args_cli.advance_on_stop):
                        return False
                else:
                    _start_recording()
            elif event == "start_episode":
                _start_episode()
            elif event == "start_recording":
                _start_episode()
            elif event == "stop_recording":
                if not _stop_recording(advance=args_cli.advance_on_stop):
                    return False
            elif event == "save_and_next":
                if not _save_and_next_episode():
                    return False
            elif event == "cancel_recording":
                _cancel_recording()
            elif event == "reset_episode":
                _cancel_recording()
                _reset_current_episode()
            elif event == "next_episode":
                _cancel_recording()
                skipped_episodes += 1
                if not _advance_episode():
                    return False
            elif event == "finish_session":
                return _finish_session()
        return True

    def _print_summary() -> None:
        print(
            "[INFO]: Teleop summary: "
            f"saved={saved_episodes}, cancelled={cancelled_episodes}, skipped={skipped_episodes}, "
            f"dataset={args_cli.repo_root if recorder is not None else 'disabled'}"
        )

    def _current_task_diagnostics():
        return task_condition_diagnostics(
            env.unwrapped,
            object_asset_names=object_asset_names,
            bin_name=success_term_params["bin_name"],
            table_bounds=success_term_params.get("table_bounds"),
            success_min_episode_time_s=success_term_params.get("min_episode_time_s", 5.0),
            confirm_time_s=success_term_params.get("confirm_time_s", 1.0),
            move_straightness_tolerance=success_term_params.get(
                "move_straightness_tolerance", 0.0508
            ),
            failure_min_episode_time_s=failure_term_params.get("min_episode_time_s", 5.0),
            max_grasp_attempts=failure_term_params.get("max_grasp_attempts", 3),
            enforce_max_grasp_attempts=failure_term_params.get("enforce_max_grasp_attempts", True),
            bin_displacement_limit=failure_term_params.get("bin_displacement_limit", 0.0254),
            non_target_displacement_limit=failure_term_params.get(
                "non_target_displacement_limit", 0.0127
            ),
            boundary_displacement_limit=failure_term_params.get("boundary_displacement_limit", 0.0127),
        )

    def _update_task_observers() -> None:
        nonlocal next_task_debug_print_time, next_task_status_ui_update_time
        show_spatial_status = (
            args_cli.task_status_ui
            and episode_plan[episode_index].task_family not in {TASK_BIN, TASK_NAMED_BIN}
        )
        if not args_cli.debug_tasks and not show_spatial_status:
            return

        unwrapped = env.unwrapped
        if not args_cli.end_on_failure:
            benchmark_failure(unwrapped, **failure_term_params)
        if not args_cli.end_on_success:
            task_success(unwrapped, **success_term_params)
        now = time.monotonic()
        debug_due = args_cli.debug_tasks and now >= next_task_debug_print_time
        status_ui_due = show_spatial_status and now >= next_task_status_ui_update_time
        if not debug_due and not status_ui_due:
            return

        diagnostics = _current_task_diagnostics()
        if status_ui_due:
            while next_task_status_ui_update_time <= now:
                next_task_status_ui_update_time += TASK_STATUS_UI_UPDATE_INTERVAL_S
            sim_speed_ui.set_task_status(_task_status_ui_text(diagnostics))

        if not debug_due:
            return
        while next_task_debug_print_time <= now:
            next_task_debug_print_time += DEBUG_TASKS_PRINT_INTERVAL_S
        max_grasp_attempts = failure_term_params.get("max_grasp_attempts", 3)
        enforce_max_grasp_attempts = failure_term_params.get("enforce_max_grasp_attempts", True)
        _print_task_diagnostics(
            diagnostics,
            _current_episode_label(),
            unwrapped,
            object_asset_names,
            max_grasp_attempts,
            enforce_max_grasp_attempts,
            task_debug_prev_lines,
        )

    _reset_current_episode()

    try:
        while simulation_app.is_running():
            if not _handle_events(controls.poll()):
                break

            if not episode_started:
                if hasattr(leader, "poll_events"):
                    leader.poll_events()
                env.unwrapped.sim.render()
                sim_speed_ui.update()
                time.sleep(0.02)
                continue

            # IsaacLab mutates simulation buffers across steps and resets; inference tensors
            # can poison those buffers and fail later on reset with inplace-update errors.
            with torch.no_grad():
                if step < initial_hold_steps:
                    action_lerobot = hold_action_lerobot
                    actions[:] = hold_action
                else:
                    if not robot_control_started:
                        _begin_robot_control(env, object_asset_names)
                        robot_control_started = True
                        if args_cli.auto_record:
                            _start_recording()

                    raw_action_lerobot = _clamp_lerobot_positions(mapper, leader.read_action())
                    if args_cli.action_smoothing > 0.0:
                        if smoothed_action_lerobot is None:
                            smoothed_action_lerobot = raw_action_lerobot.clone()
                        else:
                            alpha = args_cli.action_smoothing
                            smoothed_action_lerobot = (
                                alpha * smoothed_action_lerobot + (1.0 - alpha) * raw_action_lerobot
                            )
                        action_lerobot = _clamp_lerobot_positions(mapper, smoothed_action_lerobot)
                    else:
                        action_lerobot = raw_action_lerobot
                    desired_action_lerobot = keyboard_jaw.apply(action_lerobot, controls.gripper_command())
                    if action_velocity_limits_lerobot is not None:
                        if limited_action_lerobot is None:
                            limited_action_lerobot = desired_action_lerobot.clone()
                        else:
                            limited_action_lerobot = _limit_lerobot_velocity(
                                desired_action_lerobot,
                                limited_action_lerobot,
                                action_velocity_limits_lerobot,
                                control_dt,
                            )
                        action_lerobot = _clamp_lerobot_positions(mapper, limited_action_lerobot)
                    else:
                        action_lerobot = desired_action_lerobot
                    actions[:] = mapper.lerobot_positions_to_sim_radians(action_lerobot)

                obs, _rewards, terminated, truncated, info = env.step(actions)
                step += 1
                sim_speed_ui.add_step()
                _update_task_observers()

                if recorder is not None and recorder.recording:
                    observation_lerobot = mapper.sim_radians_to_lerobot_positions(
                        obs["policy"]["joint_pos_obs"][0].clone()
                    )
                    recorder.push_frame(
                        action=action_lerobot,
                        observation_state=observation_lerobot,
                        images=_recording_images(obs["visual"], camera_sources),
                    )

                if not _handle_events(controls.poll()):
                    break

                is_done = bool(terminated.any().item() or truncated.any().item())
                if not is_done:
                    continue

                term_log = info.get("log", {})
                is_success = bool(term_log.get("Episode_Termination/success", 0.0) > 0.0)
                end_reason = _episode_end_reason(env, terminated, truncated, term_log)
                print(
                    f"[INFO]: Episode {_current_episode_label()} ended: "
                    f"success={is_success}, reason={end_reason}"
                )

                if recorder is not None and recorder.recording:
                    if is_success or args_cli.save_failed_episodes:
                        saved = recorder.stop_episode(task=_current_task())
                        saved_episodes += int(saved)
                    else:
                        recorder.cancel_episode()
                        cancelled_episodes += 1

                if not _advance_episode():
                    break
    finally:
        _cancel_recording()
        if recorder is not None:
            recorder.finalize()
        controls.close()
        leader.close()
        sim_speed_ui.close()
        env.close()
        _print_summary()


if __name__ == "__main__":
    main()
    simulation_app.close()
