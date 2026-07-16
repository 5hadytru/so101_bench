# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run SO-101 Bench with a remote MolmoAct2 policy server."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import queue
import random
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


parser = argparse.ArgumentParser(description="SO-101 Bench MolmoAct2 remote-policy evaluator.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="So101Bench-Bin-v0", help="Isaac Lab task name.")
parser.add_argument("--seed", type=int, default=1984, help="Environment seed.")
parser.add_argument(
    "--num_episodes",
    type=int,
    default=None,
    help="Optional number of JSONL episodes to evaluate. If omitted, evaluate every row.",
)
parser.add_argument("--policy_host", type=str, default="localhost", help="MolmoAct2 policy server host.")
parser.add_argument("--policy_port", type=int, default=8000, help="MolmoAct2 policy server port.")
parser.add_argument("--action_horizon", type=int, default=30, help="Action steps to execute per server query.")
parser.add_argument(
    "--policy_cameras",
    type=str,
    default="overhead,wrist",
    help=(
        "Comma-separated pair of sim cameras sent to MolmoAct2. The checkpoint was trained on two third-person "
        "views, so '--policy_cameras overhead,overhead' is also useful for testing without the wrist camera."
    ),
)
parser.add_argument(
    "--policy_timeout_s",
    type=float,
    default=120.0,
    help="HTTP timeout for MolmoAct2 health checks and inference requests.",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=10,
    help="MolmoAct2 continuous-flow solver iterations.",
)
parser.add_argument(
    "--max_joint_step_deg",
    type=float,
    default=15.0,
    help="Maximum sim joint-target change per control tick. Use 0 to disable the MolmoAct2 safety clamp.",
)
parser.add_argument(
    "--initial_hold_time_s",
    type=float,
    default=0.5,
    help="Seconds to hold the initial joint pose before querying MolmoAct2.",
)
parser.add_argument(
    "--hold_init",
    action="store_true",
    default=False,
    help="Continuously hold the robot at the initial joint pose without connecting to or querying MolmoAct2.",
)
parser.add_argument(
    "--remote_reset_each_episode",
    dest="remote_reset_each_episode",
    action="store_true",
    default=True,
    help=(
        "Request a remote policy reset before every new episode when the MolmoAct2 policy helper supports it. "
        "Enabled by default to match the evaluator reset contract."
    ),
)
parser.add_argument(
    "--no_remote_reset_each_episode",
    dest="remote_reset_each_episode",
    action="store_false",
    help="Only clear local cached actions between episodes; useful for debugging reset behavior.",
)
parser.add_argument(
    "--lang_instruction",
    type=str,
    default=None,
    help="Fixed language instruction. If omitted, the env-generated instruction for each reset is used.",
)
parser.add_argument(
    "--episodes_jsonl",
    type=Path,
    required=True,
    help=(
        "Required JSONL file defining objects and a supported benchmark instruction for each episode. "
        "Rows are validated against OBJECT_SPLITS before evaluation."
    ),
)
parser.add_argument(
    "--episode_layouts_jsonl",
    "--layouts_jsonl",
    type=Path,
    default=None,
    help=(
        "Optional JSONL file with precomputed object and bin initial poses to apply as-is. "
        "Rows are matched to episodes by trial_id when present; otherwise they are consumed in episode order. "
        "Provided layouts are not revalidated."
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
    "--policy_image_width",
    type=int,
    default=640,
    help="Resize every policy image to this width before sending it to MolmoAct2. Use 0 to disable resizing.",
)
parser.add_argument(
    "--policy_image_height",
    type=int,
    default=480,
    help="Resize every policy image to this height before sending it to MolmoAct2. Use 0 to disable resizing.",
)
parser.add_argument(
    "--inspect_initial_scene",
    action="store_true",
    default=False,
    help=(
        "Reset the task, print/view the initial object poses, and exit only when the Isaac app closes "
        "without stepping physics."
    ),
)
parser.add_argument(
    "--camera_snapshot_key",
    type=str,
    default="P",
    help=(
        "Keyboard key in the Isaac window that saves the current images from all cameras. "
        "Use an empty string to disable."
    ),
)
parser.add_argument(
    "--episode_skip_key",
    type=str,
    default="N",
    help="Keyboard key in the Isaac window that skips to the next episode. Use an empty string to disable.",
)
parser.add_argument(
    "--camera_snapshot_dir",
    type=Path,
    default=Path("logs/molmoact2_eval_camera_snapshots"),
    help="Directory for manual camera snapshots saved during MolmoAct2 evaluation.",
)
parser.add_argument(
    "--camera_snapshot_stdin",
    nargs="?",
    const=True,
    default=True,
    type=_str_to_bool,
    help=(
        "Also accept the snapshot key typed into the launch terminal followed by Enter. "
        "Accepts either '--camera_snapshot_stdin' or '--camera_snapshot_stdin false'."
    ),
)
parser.add_argument(
    "--terminal_control_stdin",
    nargs="?",
    const=True,
    default=True,
    type=_str_to_bool,
    help=(
        "Accept pause/resume/skip commands typed into the launch terminal followed by Enter. "
        "Accepts either '--terminal_control_stdin' or '--terminal_control_stdin false'."
    ),
)
parser.add_argument(
    "--camera_snapshot_debug",
    action="store_true",
    default=False,
    help="Print every Isaac keyboard press seen by the snapshot listener.",
)
parser.add_argument(
    "--record_dataset",
    action="store_true",
    default=False,
    help="Save evaluated actions, states, and camera frames to a local LeRobot dataset.",
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="5hadytru/so101_bench_molmoact2_eval",
    help="LeRobot dataset repo id used for local dataset metadata.",
)
parser.add_argument(
    "--repo_root",
    type=Path,
    default=Path("data/lerobot/so101_bench_molmoact2_eval"),
    help="Local root directory for the optional LeRobot dataset.",
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

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Workaround: the headless camera kit (isaaclab.python.headless.rendering.kit) fails to
# produce the LdrColorSD render var on this setup, crashing TiledCamera annotator.attach()
# with "Unable to write from unknown dtype, kind=f, size=0". The GUI rendering kit renders
# the same cameras fine, so force it even under --headless unless the user overrides
# --experience explicitly. (AppLauncher resolves the bare name against IsaacLab's apps/ dir.)
if args_cli.headless and not getattr(args_cli, "experience", ""):
    args_cli.experience = "isaaclab.python.rendering.kit"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import so101_bench.tasks  # noqa: F401
from so101_bench.benchmark import (
    BenchmarkEpisodeSpec,
    INCH,
    load_episode_jsonl,
)
from so101_bench.layouts import (
    generate_episode_layout,
    normalize_layout_object_slots,
)
from so101_bench.mdp import benchmark_object_positions, mark_benchmark_robot_start
from so101_bench.mdp.terminations import POSTMORTEM_FAILURE_TYPES, POSTMORTEM_NOT_APPLICABLE
from so101_bench.tasks.direct.so101_bench.so101_bench_env_cfg import (
    BIN_RANDOM_POSES,
    OBJECT_LABELS,
    SO101_BOUNDING_BOX,
    TABLE_OBJECT_Z,
    VALID_OBJECT_SPAWN_REGIONS,
    configure_env_cfg_for_object_pool,
)
from so101_bench.utils.lerobot_calibration import (
    LEROBOT_INITIAL_JOINT_POS,
    lerobot_pose_to_sim_joint_pos,
)
from so101_bench.utils.lerobot_dataset import (
    LeRobotSimDatasetRecorder,
    SO101CalibrationMapper,
    dataset_cameras,
    real_compatible_camera_sources,
    recording_images,
)
from so101_bench.utils.molmoact2 import MolmoAct2RemotePolicy

ACTION_JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
INITIAL_ROBOT_JOINT_POS = lerobot_pose_to_sim_joint_pos(LEROBOT_INITIAL_JOINT_POS)


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


def _normalize_keyboard_key(key: str) -> str:
    return key.strip().upper().replace("-", "_").replace(" ", "_")


def _matches_keyboard_key(event_name: str, key: str) -> bool:
    event_name = _normalize_keyboard_key(event_name)
    key = _normalize_keyboard_key(key)
    return bool(key) and event_name in {key, f"KEY_{key}"}


def _normalize_terminal_command(command: str) -> str:
    return command.strip().lower().replace("-", "_").replace(" ", "_")


class _RuntimeControls:
    """Minimal runtime controls for terminal pause/resume, episode skip, and camera snapshots."""

    def __init__(
        self,
        snapshot_key: str,
        *,
        skip_key: str,
        terminal_enabled: bool,
        snapshot_stdin_enabled: bool,
        debug: bool,
    ):
        self.snapshot_key = _normalize_keyboard_key(snapshot_key) if snapshot_key else ""
        self.skip_key = _normalize_keyboard_key(skip_key) if skip_key else ""
        self.paused = False
        self._events: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._input = None
        self._keyboard = None
        self._keyboard_sub = None
        self._key_press_type = None
        self._debug = debug
        self._terminal_enabled = terminal_enabled
        self._snapshot_stdin_enabled = snapshot_stdin_enabled

        if self.snapshot_key or self.skip_key:
            self._start_isaac_keyboard_listener()
        if terminal_enabled or snapshot_stdin_enabled:
            self._start_stdin_listener()

    @staticmethod
    def _is_pause_command(command: str) -> bool:
        return command in {"pause", "pause_eval"}

    @staticmethod
    def _is_resume_command(command: str) -> bool:
        return command in {"resume", "play", "unpause", "continue"}

    @staticmethod
    def _is_toggle_command(command: str) -> bool:
        return command in {"toggle", "toggle_pause"}

    @staticmethod
    def _is_skip_command(command: str) -> bool:
        return command in {"skip", "next", "next_episode", "skip_episode"}

    @staticmethod
    def _is_snapshot_command(command: str) -> bool:
        return command in {"snapshot", "snap", "capture"}

    def _maybe_queue_terminal_command(self, line: str) -> None:
        command = _normalize_terminal_command(line)
        if not command:
            return

        if self._terminal_enabled:
            if self._is_pause_command(command):
                self._events.put("pause")
                return
            if self._is_resume_command(command):
                self._events.put("resume")
                return
            if self._is_toggle_command(command):
                self._events.put("toggle_pause")
                return
            if self._is_skip_command(command):
                self._events.put("skip_episode")
                return

        if self._snapshot_stdin_enabled and (
            self._is_snapshot_command(command)
            or (self.snapshot_key and _matches_keyboard_key(line.strip(), self.snapshot_key))
        ):
            self._events.put("snapshot")
            return

        if self._terminal_enabled or self._snapshot_stdin_enabled:
            print(
                "[WARN]: Unknown terminal command. Use 'pause', 'resume', "
                "'toggle', 'skip', or 'snapshot'."
            )

    def _start_isaac_keyboard_listener(self) -> None:
        try:
            import carb.input
            import omni.appwindow

            app_window = omni.appwindow.get_default_app_window()
            if app_window is None:
                print("[WARN]: No Isaac app window found; Isaac keyboard shortcuts are unavailable.")
                return

            self._input = carb.input.acquire_input_interface()
            self._keyboard = app_window.get_keyboard()
            self._key_press_type = carb.input.KeyboardEventType.KEY_PRESS
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                self._keyboard,
                self._on_keyboard_event,
            )
            if self.snapshot_key:
                print(f"[INFO]: Press '{self.snapshot_key}' in the Isaac window to save all current camera images.")
            if self.skip_key:
                print(f"[INFO]: Press '{self.skip_key}' in the Isaac window to skip to the next episode.")
        except Exception as exc:
            print(f"[WARN]: Isaac keyboard shortcuts unavailable: {exc}")

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
        terminal_parts = []
        if self._terminal_enabled:
            terminal_parts.append(
                "type 'pause' then Enter to pause, 'resume' then Enter to continue, "
                "or 'skip' then Enter to skip to the next episode"
            )
        if self._snapshot_stdin_enabled and self.snapshot_key:
            terminal_parts.append(f"type '{self.snapshot_key}' or 'snapshot' then Enter to save camera images")
        if terminal_parts:
            print(f"[INFO]: Terminal controls: {'; '.join(terminal_parts)}.")

    def _on_keyboard_event(self, event, *args, **kwargs):
        event_name = getattr(getattr(event, "input", None), "name", "")
        if event.type == self._key_press_type and self._debug:
            print(f"[DEBUG]: Isaac key press: {event_name!r}")
        if event.type == self._key_press_type and _matches_keyboard_key(event_name, self.snapshot_key):
            print(f"[INFO]: Camera snapshot key received from Isaac window: {event_name}")
            self._events.put("snapshot")
        if event.type == self._key_press_type and _matches_keyboard_key(event_name, self.skip_key):
            print(f"[INFO]: Episode skip key received from Isaac window: {event_name}")
            self._events.put("skip_episode")
        return True

    def poll(self) -> tuple[int, bool]:
        snapshot_requests = 0
        skip_requested = False
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            if event == "snapshot":
                snapshot_requests += 1
            elif event == "pause":
                if not self.paused:
                    self.paused = True
                    print("[INFO]: MolmoAct2 eval paused. Type 'resume' then Enter to continue.")
            elif event == "resume":
                if self.paused:
                    self.paused = False
                    print("[INFO]: MolmoAct2 eval resumed.")
            elif event == "toggle_pause":
                self.paused = not self.paused
                state = "paused" if self.paused else "resumed"
                print(f"[INFO]: MolmoAct2 eval {state}.")
            elif event == "skip_episode":
                skip_requested = True
                if self.paused:
                    self.paused = False
        return snapshot_requests, skip_requested

    def close(self) -> None:
        if self._input is None or self._keyboard is None or self._keyboard_sub is None:
            return
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None


class _SimClockRateWindow:
    """Omni UI window showing sim-time speed relative to wall time."""

    def __init__(self, *, control_dt: float, update_interval_s: float = 0.5):
        self._control_dt = control_dt
        self._update_interval_s = update_interval_s
        self._window = None
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

            self._window = ui.Window("SO-101 Sim Speed", width=260, height=82)
            with self._window.frame:
                with ui.VStack(spacing=4):
                    self._rate_label = ui.Label("Sim speed: --")
                    self._fps_label = ui.Label("Sim FPS: --")
        except Exception as exc:
            print(f"[WARN]: Sim speed UI unavailable: {exc}")
            self._window = None
            self._rate_label = None
            self._fps_label = None

    def reset(self) -> None:
        self._sim_time_s = 0.0
        now = time.perf_counter()
        self._last_update_wall_s = now
        self._last_update_sim_s = 0.0
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
        self._rate_label = None
        self._fps_label = None


def _write_image(path: Path, rgb: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        png_path = path.with_suffix(".png")
        cv2.imwrite(str(png_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return png_path
    except Exception:
        ppm_path = path.with_suffix(".ppm")
        with ppm_path.open("wb") as file:
            file.write(f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode("ascii"))
            file.write(rgb.tobytes())
        return ppm_path


def _save_camera_snapshot(
    output_dir: Path,
    policy: MolmoAct2RemotePolicy,
    visual_obs: dict,
    cameras: dict[str, dict[str, int]],
    episode_index: int,
    step: int,
    snapshot_index: int,
) -> list[Path]:
    snapshot_dir = output_dir / f"episode_{episode_index:04d}" / f"step_{step:05d}_capture_{snapshot_index:04d}"
    saved_paths = []

    for camera_name in cameras:
        try:
            rgb = policy._camera_frame(visual_obs, camera_name)
        except Exception as exc:
            print(f"[WARN]: Could not save camera '{camera_name}': {exc}")
            continue
        saved_paths.append(_write_image(snapshot_dir / camera_name, rgb))

    if policy.use_overhead_init and policy.overhead_init_image is not None:
        saved_paths.append(_write_image(snapshot_dir / policy.overhead_init_key, policy.overhead_init_image))

    if saved_paths:
        saved = ", ".join(str(path) for path in saved_paths)
        print(f"[INFO]: Saved camera snapshot: {saved}")
    else:
        print("[WARN]: Camera snapshot requested, but no images were saved.")
    return saved_paths


def _instruction(env, override: str | None) -> str:
    if override:
        return override
    return getattr(env.unwrapped, "so101_bench_instruction", "Place each object in the plastic bin.")


def _timestamped_layout_path(episodes_jsonl: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = episodes_jsonl.parent / "layouts" if episodes_jsonl.parent.name == "tasks" else Path("tasks/layouts")
    return output_dir / f"{episodes_jsonl.stem}_layouts_{timestamp}.jsonl"


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
) -> list[dict]:
    available_layouts = _load_layout_jsonl(layout_path)
    requested_trial_ids = [_episode_trial_id(episode, index) for index, episode in enumerate(episode_plan)]
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
        if len(available_layouts) < len(episode_plan):
            raise ValueError(
                f"{layout_path} contains {len(available_layouts)} layout row(s), "
                f"but {len(episode_plan)} episode(s) were requested."
            )
        episode_layouts = available_layouts[: len(episode_plan)]

    normalized_layouts = []
    for episode_index, (episode, layout) in enumerate(zip(episode_plan, episode_layouts, strict=True)):
        layout = normalize_layout_object_slots(layout, episode.objects, episode_index=episode_index)
        normalized_layouts.append(layout)
    episode_layouts = normalized_layouts
    print(f"[INFO]: Loaded provided initial layouts for {len(episode_layouts)} episode(s): {layout_path}")
    return episode_layouts


def _generate_and_save_episode_layouts(
    episode_plan: list[BenchmarkEpisodeSpec],
) -> tuple[list[dict], Path]:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    layout_rng = random.Random(args_cli.seed)
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
            seed=args_cli.seed,
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


def _begin_robot_control(env, policy: MolmoAct2RemotePolicy, obs: dict, object_asset_names: list[str]) -> None:
    mark_benchmark_robot_start(
        env.unwrapped,
        object_asset_names=object_asset_names,
        bin_name="plastic_bin",
        force_robot_start_time=True,
    )
    policy.set_episode_initial_observation(obs["visual"])


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
    # Episode transitions may be requested while action inference is active.
    # Isaac Lab keeps mutable articulation buffers across resets, so ensure reset
    # does not create inference tensors that a later reset cannot update.
    with torch.inference_mode(False):
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
    multi_rigid_body_info = getattr(unwrapped, "_so101_multi_rigid_body_info", {}) or {}
    for object_id, asset_name in enumerate(object_asset_names):
        label = object_labels[object_id] if object_id < len(object_labels) else asset_name
        pos = object_positions[0, object_id].detach().cpu().tolist()
        active = bool(active_mask[0, object_id].item()) if active_mask is not None else True
        state = "active" if active else "inactive"
        print(
            f"[INFO]: Initial {asset_name} / {label} ({state}): "
            f"x={pos[0]:.5f}, y={pos[1]:.5f}, z={pos[2]:.5f}"
        )
        child_positions = []
        for view_info in multi_rigid_body_info.get(asset_name, ()):
            transforms = view_info["view"].get_transforms()
            if not isinstance(transforms, torch.Tensor):
                transforms = torch.as_tensor(transforms)
            child_pos = transforms[0, :3].detach().cpu().tolist()
            child_positions.append(child_pos)
            print(
                f"[INFO]: Initial {asset_name} child {view_info['rel_path']}: "
                f"x={child_pos[0]:.5f}, y={child_pos[1]:.5f}, z={child_pos[2]:.5f}"
            )
        if len(child_positions) > 1:
            separation = math.dist(child_positions[0], child_positions[1])
            print(f"[INFO]: Initial {asset_name} child origin separation: {separation:.5f} m")

    bin_asset = unwrapped.scene["plastic_bin"]
    bin_pos = bin_asset.data.root_pos_w[0].detach().cpu().tolist()
    print(f"[INFO]: Initial plastic_bin: x={bin_pos[0]:.5f}, y={bin_pos[1]:.5f}, z={bin_pos[2]:.5f}")


def main():
    episode_specs = load_episode_jsonl(args_cli.episodes_jsonl)
    requested_count = len(episode_specs) if args_cli.num_episodes is None else args_cli.num_episodes
    planned_count = 1 if args_cli.inspect_initial_scene else requested_count
    if planned_count < 1:
        raise ValueError(f"Expected at least one episode, got {planned_count}.")
    if planned_count > len(episode_specs):
        raise ValueError(
            f"Requested {planned_count} episode(s), but {args_cli.episodes_jsonl} contains "
            f"{len(episode_specs)} validated row(s)."
        )
    episode_plan = episode_specs[:planned_count]
    episode_count = len(episode_plan)
    print(f"[INFO]: Loaded {len(episode_specs)} validated JSONL episode(s) from {args_cli.episodes_jsonl}.")

    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    if args_cli.episode_layouts_jsonl is not None:
        episode_layouts = _load_episode_layouts(episode_plan, args_cli.episode_layouts_jsonl)
    else:
        episode_layouts, _layout_path = _generate_and_save_episode_layouts(episode_plan)

    object_pool = _episode_object_pool(episode_plan)
    print(f"[INFO]: Pre-spawning {len(object_pool)} benchmark object asset(s): {', '.join(object_pool)}")

    env, object_asset_names = _make_env(object_pool, episode_plan[0], episode_layouts[0])
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    control_dt = float(env.unwrapped.step_dt)
    physics_dt = float(env.unwrapped.cfg.sim.dt)
    render_dt = physics_dt * int(env.unwrapped.cfg.sim.render_interval)
    initial_hold_steps = max(0, math.ceil(args_cli.initial_hold_time_s / control_dt))
    print(
        "[INFO]: Timing: "
        f"physics_dt={physics_dt:.6f}s, control_dt={control_dt:.6f}s, "
        f"render_dt={render_dt:.6f}s, action_chunk={args_cli.action_horizon * control_dt:.3f}s"
    )
    if initial_hold_steps > 0:
        print(f"[INFO]: Initial hold: {initial_hold_steps} steps ({initial_hold_steps * control_dt:.3f}s)")
    if args_cli.hold_init:
        print("[INFO]: Hold-init mode enabled: MolmoAct2 policy connection and queries are disabled.")

    cameras = _discover_cameras(env)
    if not cameras:
        raise RuntimeError("No cameras were found. MolmoAct2 inference requires visual observations.")

    if args_cli.inspect_initial_scene:
        _reset_env(env)
        _print_initial_scene(env, object_asset_names)
        print("[INFO]: Inspecting initial scene. Close the Isaac app window to exit; physics is not being stepped.")
        while simulation_app.is_running():
            simulation_app.update()
        env.close()
        return

    recorder = None
    recorder_mapper = None
    recording_camera_sources = None
    if args_cli.record_dataset:
        recording_camera_sources = real_compatible_camera_sources(cameras)
        recorder_mapper = SO101CalibrationMapper(device=env.unwrapped.device)
        recorder = LeRobotSimDatasetRecorder(
            repo_id=args_cli.repo_id,
            dataset_root=args_cli.repo_root,
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
    else:
        print("[INFO]: LeRobot dataset recording disabled. Pass --record_dataset to enable it.")

    policy_cameras = [name.strip() for name in args_cli.policy_cameras.split(",") if name.strip()]
    print(f"[INFO]: MolmoAct2 policy cameras: {policy_cameras}")
    image_size = (
        (args_cli.policy_image_width, args_cli.policy_image_height)
        if args_cli.policy_image_width > 0 and args_cli.policy_image_height > 0
        else None
    )
    if image_size is not None:
        print(f"[INFO]: Resizing policy video frames to {image_size[0]}x{image_size[1]}")

    sim_speed_ui = _SimClockRateWindow(control_dt=control_dt)
    policy = MolmoAct2RemotePolicy(
        device=env.unwrapped.device,
        cameras=cameras,
        host=args_cli.policy_host,
        port=args_cli.policy_port,
        action_horizon=args_cli.action_horizon,
        lang_instruction=args_cli.lang_instruction or "Place each object in the plastic bin.",
        camera_names=policy_cameras,
        image_size=image_size,
        timeout_s=args_cli.policy_timeout_s,
        num_steps=args_cli.num_steps,
        max_joint_step_deg=args_cli.max_joint_step_deg,
    )
    if not args_cli.hold_init:
        policy.connect()

    obs, _ = _reset_env(env)
    sim_speed_ui.reset()
    _print_episode_setup(env)
    policy.set_language_instruction(_instruction(env, args_cli.lang_instruction))
    policy.reset()
    print(f"[INFO]: Episode instruction: {policy.lang_instruction}")

    hold_action = _initial_robot_action(env)
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    actions[:] = hold_action
    if recorder is not None:
        recorder.init_dataset()
        recorder.start_episode(task=_instruction(env, args_cli.task_name))

    step = 0
    episodes = 0
    successes = 0
    skipped = 0
    end_reason_counts: Counter[str] = Counter()
    postmortem_failure_counts: Counter[str] = Counter()
    robot_control_started = False
    snapshot_index = 0
    runtime_controls = _RuntimeControls(
        args_cli.camera_snapshot_key,
        skip_key=args_cli.episode_skip_key,
        terminal_enabled=args_cli.terminal_control_stdin,
        snapshot_stdin_enabled=args_cli.camera_snapshot_stdin,
        debug=args_cli.camera_snapshot_debug,
    )

    def _save_snapshot_requests(request_count: int) -> None:
        nonlocal snapshot_index
        if request_count <= 0:
            return
        snapshot_index += 1
        _save_camera_snapshot(
            args_cli.camera_snapshot_dir,
            policy,
            obs["visual"],
            cameras,
            episodes + 1,
            step,
            snapshot_index,
        )

    def _poll_runtime_controls() -> bool:
        snapshot_requests, skip_requested = runtime_controls.poll()
        _save_snapshot_requests(snapshot_requests)
        return skip_requested

    def _print_final_score() -> None:
        evaluated = episodes - skipped
        rate = 100.0 * successes / max(evaluated, 1)
        print(f"[INFO]: Success Rate: {successes}/{evaluated} ({rate:.1f}%), skipped={skipped}")
        if end_reason_counts:
            breakdown = ", ".join(
                f"{reason}={count}" for reason, count in sorted(end_reason_counts.items())
            )
            print(f"[INFO]: Episode end reasons: {breakdown}")
        failures = evaluated - successes
        if failures > 0:
            ordered_types = [*POSTMORTEM_FAILURE_TYPES, POSTMORTEM_NOT_APPLICABLE]
            parts = []
            for failure_type in ordered_types:
                count = postmortem_failure_counts.get(failure_type, 0)
                if count > 0:
                    parts.append(f"{failure_type}={count} ({100.0 * count / failures:.1f}%)")
            for failure_type, count in sorted(postmortem_failure_counts.items()):
                if failure_type not in ordered_types:
                    parts.append(f"{failure_type}={count} ({100.0 * count / failures:.1f}%)")
            breakdown = ", ".join(parts) if parts else "none"
            print(f"[INFO]: Postmortem failure types ({failures} failed episode(s)): {breakdown}")

    def _cancel_recording() -> None:
        if recorder is not None:
            recorder.cancel_episode()

    def _push_recording_frame() -> None:
        if recorder is None:
            return
        assert recorder_mapper is not None
        assert recording_camera_sources is not None
        recorder.push_frame(
            action=recorder_mapper.sim_radians_to_lerobot_positions(actions[0].clone()),
            observation_state=recorder_mapper.sim_radians_to_lerobot_positions(
                obs["policy"]["joint_pos_obs"][0].clone()
            ),
            images=recording_images(obs["visual"], recording_camera_sources),
        )

    def _save_recording() -> None:
        if recorder is not None:
            recorder.stop_episode(task=_instruction(env, args_cli.task_name))

    def _start_next_episode() -> None:
        nonlocal obs, actions, hold_action, step, robot_control_started
        next_episode_number = episodes + 1
        print(f"[INFO]: Resetting episode {next_episode_number}/{episode_count}...")
        _configure_env_for_episode(
            env,
            episode_plan[episodes],
            episode_layouts[episodes],
            object_pool,
            object_asset_names,
        )
        obs, _ = _reset_env(env)
        sim_speed_ui.reset()
        _print_episode_setup(env)
        policy.set_language_instruction(_instruction(env, args_cli.lang_instruction))
        policy.reset(reset_remote=args_cli.remote_reset_each_episode)
        print(f"[INFO]: Episode instruction: {policy.lang_instruction}")
        hold_action = _initial_robot_action(env)
        actions[:] = hold_action
        step = 0
        robot_control_started = False
        if recorder is not None:
            recorder.start_episode(task=_instruction(env, args_cli.task_name))

    try:
        while simulation_app.is_running():
            skip_requested = _poll_runtime_controls()
            if skip_requested:
                _cancel_recording()
                episodes += 1
                skipped += 1
                print(f"[INFO]: Episode {episodes}/{episode_count}: skipped by control request.")
                if episodes >= episode_count:
                    _print_final_score()
                    break
                _start_next_episode()
                continue

            if runtime_controls.paused:
                env.unwrapped.sim.render()
                sim_speed_ui.update()
                time.sleep(0.02)
                continue

            if step < initial_hold_steps:
                actions[:] = hold_action
            else:
                if args_cli.hold_init:
                    if not robot_control_started:
                        policy.set_episode_initial_observation(obs["visual"])
                        robot_control_started = True
                    actions[:] = hold_action
                else:
                    if not robot_control_started:
                        _begin_robot_control(env, policy, obs, object_asset_names)
                        robot_control_started = True
                    with torch.inference_mode():
                        joint_positions = obs["policy"]["joint_pos_obs"][0].clone()
                        actions[:] = policy.get_action(joint_positions, obs["visual"])

            obs, _rewards, terminated, truncated, info = env.step(actions)
            step += 1
            sim_speed_ui.add_step()
            _push_recording_frame()

            skip_requested = _poll_runtime_controls()
            if skip_requested:
                _cancel_recording()
                episodes += 1
                skipped += 1
                print(f"[INFO]: Episode {episodes}/{episode_count}: skipped by control request.")
                if episodes >= episode_count:
                    _print_final_score()
                    break
                _start_next_episode()
                continue

            is_done = bool(terminated.any().item() or truncated.any().item())
            if not is_done:
                continue

            term_log = info.get("log", {})
            is_success = bool(term_log.get("Episode_Termination/success", 0.0) > 0.0)
            end_reason = _episode_end_reason(env, terminated, truncated, term_log)
            failure_reasons = getattr(env.unwrapped, "_so101_failure_reasons", None)
            live_failure_reason = failure_reasons[0] if failure_reasons else "none"
            episodes += 1
            successes += int(is_success)
            end_reason_counts[end_reason] += 1
            episode_duration_s = step * control_dt
            message = (
                f"[INFO]: Episode {episodes}/{episode_count}: success={is_success}, "
                f"reason={end_reason}, length={episode_duration_s:.2f}s"
            )
            if not is_success:
                # Read the classification stored by benchmark_failure on the terminating
                # step: the env auto-resets inside env.step() and zeros the lift buffer,
                # so recomputing here would always see a cleared episode.
                diagnostics = getattr(env.unwrapped, "_so101_postmortem_failure_diagnostics", None)
                postmortem = diagnostics[0] if diagnostics else None
                if postmortem is not None and postmortem.failure_type != POSTMORTEM_NOT_APPLICABLE:
                    postmortem_failure_counts[postmortem.failure_type] += 1
                    threshold_in = postmortem.lift_threshold_m / INCH
                    message += (
                        f", failure_type={postmortem.failure_type}"
                        f", live_failure_reason={live_failure_reason}"
                        f" [target={postmortem.target_object}, "
                        f"target_lift={postmortem.target_lift_m / INCH:.2f}in, "
                        f"wrong_object={postmortem.lifted_wrong_object}, "
                        f"max_distractor_lift={postmortem.max_non_target_lift_m / INCH:.2f}in, "
                        f"lift_threshold={threshold_in:.2f}in]"
                    )
                elif postmortem is not None:
                    postmortem_failure_counts[postmortem.failure_type] += 1
                    message += f", live_failure_reason={live_failure_reason}"
            print(message)

            # Finalize the recording only after the per-episode result is logged: with
            # --record_dataset the video encode can take minutes, and printing first keeps
            # the success/failure line timely instead of buried behind the encoder output.
            _save_recording()

            if episodes >= episode_count:
                _print_final_score()
                break

            _start_next_episode()
    finally:
        _cancel_recording()
        if recorder is not None:
            recorder.finalize()
        runtime_controls.close()
        sim_speed_ui.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
