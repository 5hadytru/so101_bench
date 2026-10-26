#!/usr/bin/env python3
"""Run a GR00T policy server on the real SO-101 follower, with the sim joint alignment.

Joint state/actions pass through the same canonical `.pos` mapper the sim eval uses, and
`RealSimAlignment` (scripts/so101_sim_alignment.py) maps sim angles to motor ticks.
Cameras are the Innomaker wrist cam and the iPhone overhead via DroidCam.

    /home/truman/env_isaaclab_51/bin/python scripts/so101_real_policy_runner.py \
        --instruction "Place the green cube in the plastic bin."

MMRV episode mode stages benchmark episodes: the overhead view shows the episode's saved layout
(object/bin footprints projected through the sim overhead camera), the instruction and timeout
come from the task row, and outcomes are appended to <results_root>/<policy_label>/results.jsonl:

    ... so101_real_policy_runner.py --policy groot --episodes_jsonl tasks/mmrv_eval.jsonl --episode 0

Keys (in the video window):
    space  start / pause the policy          o / w / i  capture overhead_init / wrist_init / both
    d      dock (pauses, ends the episode)   a          overhead overlay: off/layout/blend/edges/diff
    n      next sim reference frame          [ / ]      overlay opacity
    , / .  previous / next episode           y / f      score success / failure (opens the scoring form)
    p / 1-6  switch policy: next, or 1 groot 2 g05 3 pi05 4 molmoact2 5 argon 6 sol (stops this server, starts
           that one, applies its inputs, resumes at its first unscored episode)

sol = GPT-6.1 Sol (high, your OpenAI key, Codex runtime) and argon = Gemini 4 Argon (Vertex): each + frozen
PyRUA-Lean v9.6 + G0.5 hand-offs (scripts/so101_real_pyrua.py), the
configuration of the Oct 17 sim campaign, through a local Vertex proxy on your GCP project only.
    r      re-save the session's camera/lighting reference (only after a deliberate change)
    v      recording mode on/off: each run -> pause cycle becomes a clip (overhead | wrist, real time) in
           <results_root>/<policy>/recordings/{raw,annotated}/<policy>_trial<NN>_<family>_<split>_clipNN_*.mp4;
           raw is untouched at wall-clock speed; annotated adds the full agent HUD, verdicts and messages.
           On each pause a pop-up asks to save both (Yes) or delete both (No)
    q/Esc  dock, torque off, quit
Type a line in the terminal to change the instruction (free mode only).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path
import queue
import re
import sys
import threading
import time
import types

import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import shutil
import signal
import subprocess
import tkinter as tk
from tkinter import messagebox
from urllib import request as urlrequest
import torch

from so101_real_recording import ClipRecorder


REPO_ROOT = Path(__file__).resolve().parents[1]
HOME_DIR = Path.home()
PACKAGE_DIR = REPO_ROOT / "source" / "so101_bench" / "so101_bench"


def _import_utils():
    # so101_bench/__init__ pulls in Isaac Lab; these utils only need the package namespace.
    for name, path in (("so101_bench", PACKAGE_DIR), ("so101_bench.utils", PACKAGE_DIR / "utils")):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
    from so101_bench import benchmark
    from so101_bench.utils import groot, layout_overlay, lerobot_calibration, molmoact2, real_sim_alignment

    return benchmark, groot, layout_overlay, lerobot_calibration, molmoact2, real_sim_alignment


benchmark_lib, groot_lib, overlay_lib, calibration_lib, molmoact2_lib, alignment_lib = _import_utils()

JOINTS = alignment_lib.JOINT_ORDER
CONTROL_HZ = 30.0
FRAME_SIZE = (640, 480)
# Sim joint speed caps (SO101_VELOCITY_LIMITS_SIM in assets/so101.py), rad/s, in JOINTS order.
VELOCITY_LIMITS_RAD_S = np.array([3.0, 3.3, 3.1, 2.9, 4.2, 3.2])
DOCK_SPEED_DEG_S = 30.0
DOCK_SIM_DEG = np.array(
    [calibration_lib.lerobot_position_to_sim_degrees(joint, calibration_lib.LEROBOT_INITIAL_JOINT_POS[joint]) for joint in JOINTS]
)
SERIAL_RETRIES = 3
# A camera dropout freezes the episode (arm holds, no steps count, the policy's queued actions are kept), as sim
# pauses physics during a query, for as long as it lasts (user, 2026-10-24): no pause, the recording clip keeps
# going, and the episode resumes by itself when frames return.
CAMERA_FREEZE_MAX_S = float("inf")
JAW_FLOOR_DEG = calibration_lib.USD_SIM_JOINT_LIMITS_DEG["gripper"][0]
# Squeeze ramps in as the policy's jaw target goes from open-ish to fully closed (sim deg).
SQUEEZE_RAMP_DEG = (5.0, -10.0)
WATCHDOG_ERROR_DEG = 25.0
WATCHDOG_AFTER_S = 1.0
# Per-policy settings matching their primary_eval runs (run_primary_eval_groot_local.sh, run_primary_eval_pod.sh).
# HTTP policies use molmoact2_eval's client; joint_frame/num_steps as in those runs.
POLICY_DEFAULTS = {
    "groot": {"policy_port": 5555, "action_horizon": 16, "use_overhead_init": True, "use_wrist_init": True},
    "g05": {"policy_port": 8001, "action_horizon": 15, "use_overhead_init": False, "use_wrist_init": False,
            "joint_frame": "lerobot", "num_steps": 10},
    "pi05": {"policy_port": 8002, "action_horizon": 15, "use_overhead_init": True, "use_wrist_init": True,
             "joint_frame": "lerobot", "num_steps": 10},
    "molmoact2": {"policy_port": 8003, "action_horizon": 30, "use_overhead_init": True, "use_wrist_init": True,
                  "joint_frame": "molmoact2", "num_steps": 10},
    # LLM agent: drives the arm itself; its policy_port is the G0.5 server behind robo.vla_run.
    "argon": {"policy_port": 8001, "action_horizon": 15, "use_overhead_init": False, "use_wrist_init": False,
              "joint_frame": "lerobot", "num_steps": 10, "agent": True},
    "sol": {"policy_port": 8001, "action_horizon": 15, "use_overhead_init": False, "use_wrist_init": False,
            "joint_frame": "lerobot", "num_steps": 10, "agent": True},
}
# LLM agents (PyRUA-Lean v9.6 + G0.5 hand-offs): model, runtime and effort exactly as their sim campaigns ran.
# Sol = outputs/primary_eval_v2_sol_pyrua_g05_v9.6 and outputs/mmrv_eval_sim_sol_pyrua_g05_v96_seed0 (codex, high,
# the user's own OpenAI key); Argon = argon_medium_vertex60_v96_20261017 (Vertex proxy, --argon_effort).
AGENTS = {
    "sol": {"name": "GPT-6.1 Sol", "model": "gpt-6.1-sol", "runtime": "codex", "effort": "high",
            "codex_home": HOME_DIR / ".codex-user-api"},
    "argon": {"name": "Gemini 4 Argon", "model": "gemini-4-argon", "runtime": "openai"},
}
# One-click fail_type tags: the reviewer's most-used free-text tags from the sim failure reviews
# (logs/groot_sim_real_correspondence/real_sim_correspondence.jsonl). Any other text can be typed.
FAIL_TYPE_TAGS = (
    "imprecise", "totally confused", "refused to lift", "bad strategy", "grasped other object",
    "refused to undock", "placed on top", "NCE", "moved object", "refused to release the object",
    "not close", "placed between other objects", "grasped class distractor object", "moved boundary",
    "stuck on top", "stuck on bin", "moved slowly", "placed next to other object",
    "grasped color distractor object", "missed bin", "knocked bin", "hovered indefinitely", "move trajectory not straight enough", "timed out",
)
DOCK_START_TOLERANCE_DEG = 5.0
POLICY_ORDER = ("groot", "g05", "pi05", "molmoact2", "argon", "sol")
HOME = Path.home()
CHECKPOINTS = REPO_ROOT / "checkpoints"
# Local server launch recipes (same checkpoints and flags as each policy's primary_eval_v2 run).
SERVER_RECIPES = {
    "groot": {
        "cwd": HOME / "workspace/Isaac-GR00T",
        "cmd": [str(HOME / "workspace/Isaac-GR00T/.venv/bin/python"), "gr00t/eval/run_gr00t_server.py",
                "--model-path", str(CHECKPOINTS / "so101_bench_GR00T_N1.7-3B_official"),
                "--embodiment-tag", "NEW_EMBODIMENT", "--device", "cuda", "--host", "127.0.0.1", "--port", "{port}"],
        "env": {"HF_HOME": str(HOME / ".cache/huggingface"), "HF_MODULES_CACHE": str(HOME / ".cache/huggingface/modules")},
    },
    "g05": {
        "cwd": REPO_ROOT,
        "cmd": [str(HOME / "workspace/GalaxeaVLA/.venv/bin/python"), "scripts/g05_server.py",
                "--repo", str(HOME / "workspace/GalaxeaVLA"),
                "--ckpt_path", str(CHECKPOINTS / "so101_bench_G0.5_step25600/checkpoints/model_state_dict.pt"),
                "--mode", "ar", "--replan_steps", "15", "--host", "127.0.0.1", "--port", "{port}"],
        "env": {"HF_HUB_OFFLINE": "1"},
    },
    "pi05": {
        "cwd": REPO_ROOT,
        "cmd": [str(HOME / "workspace/pi05_server/.venv/bin/python"), "scripts/pi05_server.py",
                "--repo_id", str(CHECKPOINTS / "so101_bench_pi05_step18000"), "--device", "cuda:0",
                "--dtype", "bfloat16", "--num_steps", "10", "--host", "127.0.0.1", "--port", "{port}"],
        "env": {},
    },
    "molmoact2": {
        "cwd": REPO_ROOT,
        "cmd": [str(HOME / "workspace/molmoact2_server/.venv/bin/python"), "scripts/molmoact2_server.py",
                "--repo_id", str(CHECKPOINTS / "MolmoAct2-SO101_step10000"), "--device", "cuda:0",
                "--dtype", "bfloat16", "--num_steps", "10", "--host", "127.0.0.1", "--port", "{port}"],
        "env": {},
    },
}
SERVER_RECIPES["argon"] = SERVER_RECIPES["g05"]  # the agents' hand-offs use the G0.5 server
SERVER_RECIPES["sol"] = SERVER_RECIPES["g05"]
SERVER_START_TIMEOUT_S = 900.0
# Gemini 4 Argon on Vertex (scripts/vertex_argon_proxy.py); list $ per 1M tokens: input, cached input, output
# (pricing revision vertex-global-list-before-promotional-rebate-20261017).
ARGON_PRICING_PER_M = (4.0, 0.2, 20.0)
VERTEX_TOKEN_FILE = HOME / ".config/so101_bench/vertex_proxy_token"
GCLOUD = HOME / ".local/bin/gcloud"
# Session drift limits between policy blocks: global overhead-image shift and median-brightness change.
MAX_CAMERA_SHIFT_PX = 3.0
MAX_BRIGHTNESS_CHANGE = 0.08
DEFAULT_REFERENCE_DIR = REPO_ROOT / "outputs" / "real_robot" / "sim_camera_reference"
OVERLAY_MODES = ("off", "layout", "blend", "edges", "diff")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instruction", default="Place each object in the plastic bin.")
    parser.add_argument("--policy_host", default="127.0.0.1")
    parser.add_argument(
        "--policy",
        choices=sorted(POLICY_DEFAULTS),
        default="groot",
        help="groot: GR00T N1.7 ZMQ server. g05 / pi05 / molmoact2: HTTP servers (scripts/*_server.py).",
    )
    parser.add_argument("--policy_port", type=int, default=None, help="Default per policy: see POLICY_DEFAULTS.")
    parser.add_argument("--action_horizon", type=int, default=None, help="Default per policy (each one's primary_eval run).")
    parser.add_argument("--use_overhead_init", type=_str_to_bool, default=None, help="Default per policy.")
    parser.add_argument("--use_wrist_init", type=_str_to_bool, default=None, help="Default per policy.")
    parser.add_argument("--no_policy", action="store_true", help="Cameras, overlay and docking only.")
    parser.add_argument("--follower_port", default="/dev/ttyACM0")
    parser.add_argument("--robot_id", default="follower_arm_1")
    parser.add_argument("--alignment", type=Path, default=None)
    parser.add_argument("--wrist_camera", default="/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0",
                        help="Stable by-id path: a USB reconnect can renumber /dev/videoN.")
    parser.add_argument("--overhead_camera", default="/dev/video10", help="DroidCam v4l2loopback device.")
    parser.add_argument("--reference_dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument(
        "--gripper_squeeze_deg",
        type=float,
        default=0.0,
        help="Extra closing added to the real jaw target while the policy commands it closed. Off by default: "
        "15 deg with a lowered motor limit crushed the jaws together and cracked a printed jaw (2026-10-13). "
        "Never commands past the USD jaw limit.",
    )
    parser.add_argument(
        "--gripper_min_tick",
        type=int,
        default=None,
        help="Gripper Min_Position_Limit written at startup. Default: the tick of sim's own jaw limit (USD -12.75 deg, "
        "0.75 deg past tip contact at -12 deg) from the measured gripper zero, so the jaws close fully as in sim with "
        "the same light preload. Never set it further: below that the motor crushes the jaws (cracked one 2026-10-14).",
    )
    parser.add_argument("--rotate_every", type=int, default=0,
                        help="Interleave policies: after every N scored episodes, switch to the rotation's policy "
                        "with the fewest scored episodes (all run the same N-episode chunk). 0 = off.")
    parser.add_argument("--rotation", default="molmoact2,groot,pi05,g05",
                        help="Comma-separated policies for --rotate_every; append argon after g05 to include it (it reuses the G0.5 server).")
    parser.add_argument("--recording_mode", action="store_true", help="Record raw wall-clock clips and fully annotated copies (key v; save both on pause).")
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "outputs" / "real_robot" / "runs")
    parser.add_argument("--episodes_jsonl", type=Path, default=None, help="Benchmark task file (e.g. tasks/mmrv_eval.jsonl).")
    parser.add_argument(
        "--layouts_jsonl",
        type=Path,
        default=None,
        help="Layouts for --episodes_jsonl. Default: tasks/layouts/<name>_layouts.jsonl. Re-read on every episode change.",
    )
    parser.add_argument(
        "--episode", type=int, default=None, help="Row index to start at. Default: this policy's first unscored episode."
    )
    parser.add_argument("--results_root", type=Path, default=REPO_ROOT / "outputs" / "real_robot" / "mmrv")
    parser.add_argument("--policy_label", default=None, help="Results subdirectory. Default: --policy.")
    parser.add_argument("--argon_effort", choices=("medium", "high"), default="medium",
                        help="Argon reasoning effort. medium = the Oct 17 sim campaign (77%%, 46/60).")
    parser.add_argument("--gcp_project", default=None, help="Vertex project. Default: gcloud configuration so101-bench.")
    parser.add_argument("--vertex_proxy_port", type=int, default=8211)
    parser.add_argument("--sol_budget_usd", type=float, default=11.0,
                        help="Refuse to start a new Sol episode once Sol's priced real-robot spend reaches this.")
    parser.add_argument("--argon_budget_usd", type=float, default=120.0,
                        help="Refuse to start a new Argon episode once the proxy's recorded spend reaches this.")
    return parser


def _str_to_bool(value: str) -> bool:
    if value.lower() in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value.lower() in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


class LatestFrameCamera:
    """Background reader that keeps only the newest frame; reopens the device if it drops."""

    def __init__(self, name: str, device: str, fourcc: str | None, controls: dict[str, int] | None = None):
        self.name = name
        self.device = device
        self.fourcc = fourcc
        self.controls = controls  # v4l2 controls re-applied on every (re)open: a USB reconnect resets them
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"camera-{name}", daemon=True)
        self._thread.start()

    def _open(self) -> cv2.VideoCapture | None:
        capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not capture.isOpened():
            return None
        if self.controls:
            for key, value in sorted(self.controls.items(), key=lambda kv: kv[0] != "auto_exposure"):
                subprocess.run(["v4l2-ctl", "-d", self.device, "-c", f"{key}={value}"], check=False, capture_output=True)
        if self.fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _run(self) -> None:
        capture = None
        while not self._stop.is_set():
            if capture is None:
                capture = self._open()
                if capture is None:
                    time.sleep(1.0)
                    continue
            ok, frame = capture.read()
            if not ok:
                capture.release()
                capture = None
                time.sleep(0.5)
                continue
            with self._lock:
                self._frame, self._stamp = frame, time.monotonic()
        if capture is not None:
            capture.release()

    def stop(self) -> None:
        # Joining keeps OpenCV from being torn down mid-read at interpreter exit.
        self._stop.set()
        self._thread.join(timeout=3.0)

    def latest_bgr(self, max_age_s: float = 0.5) -> np.ndarray | None:
        with self._lock:
            if self._frame is None or time.monotonic() - self._stamp > max_age_s:
                return None
            return self._frame


class RealArm:
    def __init__(self, port: str, alignment):
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        self.alignment = alignment
        self.bus = FeetechMotorsBus(
            port=port,
            motors={name: Motor(index + 1, "sts3215", MotorNormMode.RANGE_M100_100) for index, name in enumerate(JOINTS)},
        )
        self.bus.connect()
        if mismatches := alignment.calibration_mismatches(self.bus.read_calibration()):
            self.bus.port_handler.closePort()
            raise SystemExit("Motor calibration changed since the sim alignment was measured:\n  " + "\n  ".join(mismatches))

    def set_gripper_min_tick(self, tick: int) -> None:
        """Lower the gripper's EEPROM goal limit (torque must be off) and clamp to it here too.

        LeRobot rewrites its calibration file's range_min (1580) if teleop reconnects with
        calibration, so this is reapplied at every runner start.
        """

        current = self.bus.read("Min_Position_Limit", "gripper", normalize=False)
        if current != tick:
            self.bus.disable_torque("gripper", num_retry=SERIAL_RETRIES)
            self.bus.write("Min_Position_Limit", "gripper", tick, normalize=False, num_retry=SERIAL_RETRIES)
            print(f"[INFO]: Gripper Min_Position_Limit {current} -> {tick}.", flush=True)
        snapshot = self.alignment.motor_calibration["gripper"]
        self.alignment.motor_calibration["gripper"] = replace(snapshot, range_min=tick)

    def sim_deg(self) -> np.ndarray:
        ticks = self.bus.sync_read("Present_Position", normalize=False, num_retry=SERIAL_RETRIES)
        return np.array([self.alignment.ticks_to_sim_degrees(joint, ticks[joint]) for joint in JOINTS])

    def stall_ticks(self) -> float:
        """Measured gripper stall short of its goal (scripts/so101_gripper_test.py t): at P=16 the motor stops
        ~2 deg before a closing goal against hinge friction, where sim's drive gets there."""

        return float(self.alignment.notes.get("gripper_stall_ticks", 0.0))

    def gripper_ticks(self, sim_deg: float) -> int:
        # Aim past the goal by the measured stall, ramping in from 0 deg to the jaw floor, so the jaw lands where
        # sim's would. Open-side goals are unchanged; with an object in hand the extra push is only the stall.
        ramp = float(np.clip(sim_deg / JAW_FLOOR_DEG, 0.0, 1.0))
        ticks = self.alignment.sim_degrees_to_ticks("gripper", sim_deg, clamp=False) - self.stall_ticks() * ramp
        calibration = self.alignment.motor_calibration["gripper"]
        return int(round(min(calibration.range_max, max(calibration.range_min, ticks))))

    def command(self, sim_deg: np.ndarray) -> None:
        goals = {joint: int(round(self.alignment.sim_degrees_to_ticks(joint, float(sim_deg[i])))) for i, joint in enumerate(JOINTS)}
        goals["gripper"] = self.gripper_ticks(float(sim_deg[5]))
        self.bus.sync_write("Goal_Position", goals, normalize=False)

    def enable(self, hold_sim_deg: np.ndarray) -> None:
        self.command(hold_sim_deg)
        self.bus.enable_torque(num_retry=SERIAL_RETRIES)

    def close(self, *, torque_off: bool = True) -> None:
        try:
            if torque_off:
                self.bus.disable_torque(num_retry=SERIAL_RETRIES)
        finally:
            self.bus.port_handler.closePort()


def gripper_min_tick(alignment) -> int:
    """EEPROM floor: sim's jaw limit, less the measured stall (the motor stops that far short of it)."""

    floor = alignment.sim_degrees_to_ticks("gripper", JAW_FLOOR_DEG, clamp=False)
    return math.ceil(floor - float(alignment.notes.get("gripper_stall_ticks", 0.0)))


class DockMotion:
    """Raise through an L pose at the current pan, then fold into the sim start pose."""

    def __init__(self, start: np.ndarray):
        self.waypoints = [DOCK_SIM_DEG.copy()]
        if start[1] > 0.0:
            # Leaning forward over the table: first stand the upper arm up and level any
            # downward-pointing forearm/wrist, keeping pan, roll and gripper.
            lifted = start.copy()
            lifted[1] = 0.0
            lifted[2:4] = np.minimum(start[2:4], 0.0)
            self.waypoints.insert(0, lifted)
        self._segment_start = start
        self._t0 = time.monotonic()

    def target(self) -> np.ndarray | None:
        while self.waypoints:
            goal = self.waypoints[0]
            duration = max(np.abs(goal - self._segment_start).max() / DOCK_SPEED_DEG_S, 1e-3)
            s = (time.monotonic() - self._t0) / duration
            if s < 1.0:
                s = s * s * (3.0 - 2.0 * s)
                return self._segment_start + s * (goal - self._segment_start)
            self._segment_start = self.waypoints.pop(0)
            self._t0 = time.monotonic()
        return None


class EpisodeRecorder:
    def __init__(self, root: Path, instruction: str, inits: dict[str, np.ndarray], episode: dict | None = None):
        slug = re.sub(r"[^a-z0-9]+", "_", instruction.lower()).strip("_")[:48]
        self.dir = root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slug}"
        self.dir.mkdir(parents=True, exist_ok=True)
        for name, rgb in inits.items():
            cv2.imwrite(str(self.dir / f"{name}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writers = {
            camera: cv2.VideoWriter(str(self.dir / f"{camera}.mp4"), fourcc, CONTROL_HZ, FRAME_SIZE)
            for camera in ("overhead", "wrist")
        }
        self.rows: dict[str, list] = {"t": [], "state_sim_deg": [], "command_sim_deg": [], "policy_query": [], "gripper_squeeze_deg": [],
                                      "controller": []}  # LLM-agent episodes: SOL / G0.5 / "SOL thinking" per step
        self.instructions: list[tuple[float, str]] = [(0.0, instruction)]
        self.camera_gaps: list[dict] = []  # {"step", "t", "seconds"}: dropouts the episode was frozen through
        self.episode = episode
        self.t0 = time.monotonic()

    @property
    def steps(self) -> int:
        return len(self.rows["t"])

    def add(self, frames_rgb: dict[str, np.ndarray], state, command, queried: bool, squeeze: float, controller: str = "") -> None:
        for camera, writer in self.writers.items():
            writer.write(cv2.cvtColor(frames_rgb[camera], cv2.COLOR_RGB2BGR))
        self.rows["t"].append(time.monotonic() - self.t0)
        self.rows["state_sim_deg"].append(state)
        self.rows["command_sim_deg"].append(command)
        self.rows["policy_query"].append(queried)
        self.rows["gripper_squeeze_deg"].append(squeeze)
        self.rows["controller"].append(controller or "")

    def close(self, reason: str) -> Path:
        for writer in self.writers.values():
            writer.release()
        np.savez_compressed(self.dir / "trajectory.npz", **{k: np.asarray(v) for k, v in self.rows.items()})
        meta = {"joints": list(JOINTS), "instructions": self.instructions, "end_reason": reason, "steps": self.steps,
                "camera_gaps": self.camera_gaps,
                "episode": self.episode}
        (self.dir / "episode.json").write_text(json.dumps(meta, indent=2) + "\n")
        return self.dir


CONTROLLER_COLORS_BGR = {"G0.5": (60, 160, 255), "thinking": (150, 150, 150), "finished": (120, 200, 120)}


def draw_controller_banner(frame: np.ndarray, label: str) -> None:
    """Top banner naming who drives the arm: the LLM agent (blue), G0.5 (orange), or the agent thinking (grey)."""

    color = (CONTROLLER_COLORS_BGR["G0.5"] if label == "G0.5" else CONTROLLER_COLORS_BGR["thinking"] if "thinking" in label
             else CONTROLLER_COLORS_BGR["finished"] if label == "finished" else (230, 120, 40))
    text = {"G0.5": "G0.5 IN CONTROL", "finished": "AGENT FINISHED - ARM HOLDING"}.get(
        label, label.replace(" thinking", " THINKING - ARM HOLDING") if "thinking" in label else f"{label} IN CONTROL")
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), color, -1)
    cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


def server_is_up(policy: str, host: str, port: int) -> bool:
    if policy == "groot":
        # Own REQ socket with linger 0 (as run_primary_eval_groot_local.sh's ping): PolicyClient.ping
        # re-creates its socket on failure and leaves the old one, which makes context teardown hang.
        import msgpack
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 2000)
        socket.setsockopt(zmq.SNDTIMEO, 2000)
        try:
            socket.connect(f"tcp://{host}:{port}")
            socket.send(msgpack.packb({"endpoint": "ping"}, use_bin_type=True))
            socket.recv()
            return True
        except zmq.error.ZMQError:
            return False
        finally:
            socket.close(0)
            context.term()
    try:
        with urlrequest.urlopen(f"http://{host}:{port}/act", timeout=2.0) as response:
            return json.loads(response.read().decode()).get("status") == "ok"
    except Exception:  # noqa: BLE001 - any failure means "not up yet"
        return False


class PolicyServerManager:
    """Starts/stops one policy server at a time on the local GPU.

    A server that already answers on a policy's port and was not started here (e.g. a
    primary eval's GR00T server) is reused and never stopped.
    """

    def __init__(self, host: str, log_dir: Path):
        self.host = host
        self.log_dir = log_dir
        self.process: subprocess.Popen | None = None
        self.policy: str | None = None

    def stop_strays(self, keep: str, progress=print) -> None:
        """Stop other policies' servers left on this GPU by an earlier runner (a crash or a reused server):
        two models do not fit on the 3090, and the new one would hang loading. Matched by checkpoint path."""

        keep_cmd = SERVER_RECIPES[keep]["cmd"]
        for policy, recipe in SERVER_RECIPES.items():
            if recipe["cmd"] is keep_cmd:
                continue
            # the server script AND its checkpoint, so sim evals or tools that only mention the checkpoint are left alone
            checkpoint = next(part for part in recipe["cmd"] if "checkpoints" in part)
            pattern = f"{re.escape(recipe['cmd'][1])}.*{re.escape(checkpoint)}"
            found = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
            for pid in map(int, found):
                if self.process is not None and pid == self.process.pid:
                    continue
                progress(f"Stopping a leftover {policy} server (pid {pid}) that is holding the GPU...")
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    continue
                for _ in range(60):
                    if subprocess.run(["ps", "-p", str(pid)], capture_output=True).returncode:
                        break
                    time.sleep(0.5)
                else:
                    os.kill(pid, signal.SIGKILL)

    def ensure(self, policy: str, port: int, progress=print) -> None:
        self.stop_strays(policy, progress)
        if server_is_up(policy, self.host, port):
            progress(f"{policy} server already answering on :{port}; reusing it.")
            return
        self.stop(progress)
        recipe = SERVER_RECIPES[policy]
        cmd = [part.replace("{port}", str(port)) for part in recipe["cmd"]]
        missing = [part for part in cmd if part.startswith("/") and "checkpoints" in part and not Path(part).exists()]
        if missing:
            raise RuntimeError(f"{policy} checkpoint missing: {missing[0]}")
        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}  # the shell's HF_TOKEN is stale
        env.update(recipe["env"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = (self.log_dir / f"{policy}_server.log").open("a")
        progress(f"Starting {policy} server on :{port} (log {self.log_dir / f'{policy}_server.log'})...")
        self.process = subprocess.Popen(cmd, cwd=recipe["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        self.policy = policy
        started = time.monotonic()
        while not server_is_up(policy, self.host, port):
            if self.process.poll() is not None:
                self.process = None
                raise RuntimeError(f"{policy} server exited during startup; see its log in {self.log_dir}")
            elapsed = time.monotonic() - started
            if elapsed > SERVER_START_TIMEOUT_S:
                self.stop(progress)
                raise RuntimeError(f"{policy} server did not come up within {SERVER_START_TIMEOUT_S:.0f} s")
            progress(f"Loading {policy} server... {elapsed:.0f} s")
            time.sleep(3.0)
        progress(f"{policy} server ready ({time.monotonic() - started:.0f} s).")

    def stop(self, progress=print) -> None:
        if self.process is None:
            return
        progress(f"Stopping {self.policy} server...")
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.process, self.policy = None, None


class VertexProxy:
    """scripts/vertex_argon_proxy.py on loopback: Argon calls go to Vertex on this GCP project only."""

    def __init__(self, project: str, effort: str, port: int, audit: Path):
        self.project, self.effort, self.port, self.audit = project, effort, port, audit
        self.process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def health(self) -> dict | None:
        try:
            with urlrequest.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2.0) as response:
                return json.loads(response.read().decode())
        except Exception:  # noqa: BLE001
            return None

    def start(self) -> None:
        health = self.health()
        if health is not None:
            if health.get("project") != self.project or health.get("effort") != self.effort:
                raise RuntimeError(f"A different Vertex proxy already listens on :{self.port}: {health}")
            return
        self.audit.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "scripts/vertex_argon_proxy.py"), "--project", self.project,
             "--effort", self.effort, "--key-file", str(VERTEX_TOKEN_FILE), "--audit", str(self.audit),
             "--port", str(self.port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        for _ in range(50):
            if self.health() is not None:
                return
            time.sleep(0.2)
        raise RuntimeError("Vertex proxy did not start")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.process = None

    def spend_usd(self, since: float = 0.0) -> float:
        if not self.audit.exists():
            return 0.0
        rate_in, rate_cached, rate_out = ARGON_PRICING_PER_M
        total = 0.0
        for line in self.audit.read_text().splitlines():
            record = json.loads(line)
            usage = record.get("normalized_usage")
            if not usage or record.get("time", 0.0) < since:
                continue
            prompt = int(usage.get("prompt_tokens") or 0)
            cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            total += ((prompt - cached) * rate_in + cached * rate_cached + int(usage.get("completion_tokens") or 0) * rate_out) / 1e6
        return total


class AgentPolicy:
    """Stands in for a VLA client while an LLM agent drives: it owns the G0.5 session for hand-offs."""

    is_agent = True

    def __init__(self, vla):
        self.vla = vla
        self.action_queue: list = []
        self.overhead_init_image = None
        self.wrist_init_image = None

    def connect(self) -> None:
        pass

    def set_language_instruction(self, instruction: str) -> None:
        pass


def default_gcp_project() -> str:
    return subprocess.run([str(GCLOUD), "--configuration=so101-bench", "config", "get-value", "project"],
                          capture_output=True, text=True, timeout=30).stdout.strip()


class SessionReference:
    """Camera/lighting reference frame for a whole MMRV session (all policy blocks).

    The overhead view is compared by phase correlation (global image shift) and median
    brightness, which tolerate the different objects on the table in each episode.
    """

    def __init__(self, path: Path):
        self.path = path
        self.image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.exists() else None

    def save(self, frame_bgr: np.ndarray) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.rename(self.path.with_name(f"overhead_reference.{time.strftime('%Y%m%d_%H%M%S')}.jpg"))
        cv2.imwrite(str(self.path), frame_bgr)
        self.image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def drift(self, frame_bgr: np.ndarray) -> dict[str, float] | None:
        if self.image is None:
            return None
        gray = cv2.cvtColor(cv2.resize(frame_bgr, (self.image.shape[1], self.image.shape[0])), cv2.COLOR_BGR2GRAY)
        window = cv2.createHanningWindow(gray.shape[::-1], cv2.CV_32F)
        (dx, dy), _ = cv2.phaseCorrelate(np.float32(self.image), np.float32(gray), window)
        brightness = float(np.median(gray)) / max(float(np.median(self.image)), 1.0) - 1.0
        return {"camera_shift_px": float(math.hypot(dx, dy)), "brightness_change": brightness}

    @staticmethod
    def problems(drift: dict[str, float]) -> list[str]:
        problems = []
        if drift["camera_shift_px"] > MAX_CAMERA_SHIFT_PX:
            problems.append(f"overhead camera moved {drift['camera_shift_px']:.1f} px")
        if abs(drift["brightness_change"]) > MAX_BRIGHTNESS_CHANGE:
            problems.append(f"lighting changed {drift['brightness_change'] * 100:+.0f}%")
        return problems


class Runner:
    def __init__(self, args):
        self.args = args
        alignment_path = args.alignment or alignment_lib.default_alignment_path(args.robot_id)
        self.alignment = alignment_lib.RealSimAlignment.load(alignment_path)
        self.instruction = args.instruction
        wrist_controls = REPO_ROOT / "source/so101_bench/so101_bench/assets/real_robot" / f"{args.robot_id}_wrist_camera_controls.json"
        controls = json.loads(wrist_controls.read_text())["controls"] if wrist_controls.exists() else None
        if controls is None:
            print("[WARN]: No locked wrist camera controls (scripts/so101_wrist_white_balance.py); auto white balance drifts.")
        self.wrist_cam = LatestFrameCamera("wrist", args.wrist_camera, "MJPG", controls)
        self.overhead_cam = LatestFrameCamera("overhead", args.overhead_camera, None)
        self.references = sorted(args.reference_dir.glob("overhead_*.jpg")) if args.reference_dir.exists() else []
        self.reference_index = 0
        self.overlay_mode = 0
        self.overlay_alpha = 0.5
        self.projector = overlay_lib.OverheadProjector.from_benchmark_defaults(benchmark_lib)
        self.episodes: list[dict] = []
        self.episode_index = 0
        self.layout: dict | None = None
        self.last_episode: dict | None = None  # {"dir": Path, "row": dict, "steps": int, "timed_out": bool}
        self.pending: dict | None = None  # {"success": bool}: scoring form requested / open
        if args.episodes_jsonl is not None:
            self.episodes = self._balanced_order(
                [json.loads(line) for line in args.episodes_jsonl.read_text().splitlines() if line.strip()])
            self.layouts_path = args.layouts_jsonl or (
                args.episodes_jsonl.parent / "layouts" / f"{args.episodes_jsonl.stem}_layouts.jsonl"
            )
            self.results_path = args.results_root / (args.policy_label or args.policy) / "results.jsonl"
            self.overlay_mode = OVERLAY_MODES.index("layout")
            self._load_episode(args.episode if args.episode is not None else self._first_unscored())
        self.policy = None
        self.switching = False
        self.servers = PolicyServerManager(args.policy_host, args.results_root / "servers")
        self.proxy: VertexProxy | None = None
        self.agent_episode = None  # so101_real_pyrua.RealPyruaEpisode while an LLM agent episode is open
        self._agent_tick = time.monotonic()
        self.session = SessionReference(args.results_root / "overhead_reference.jpg") if args.episodes_jsonl else None
        self.drift: dict[str, float] | None = None
        if not args.no_policy:
            self.servers.ensure(args.policy, args.policy_port)
            self.policy = self._make_policy()
            self.policy.connect()
        self.arm = RealArm(args.follower_port, self.alignment)
        if args.gripper_min_tick is None:
            args.gripper_min_tick = gripper_min_tick(self.alignment)
        self.arm.set_gripper_min_tick(args.gripper_min_tick)
        self.mode = "hold"  # hold | run | dock
        self.goal = self.arm.sim_deg()
        self.dock: DockMotion | None = None
        self.inits: dict[str, np.ndarray] = {}
        self.recorder: EpisodeRecorder | None = None
        self.agent_controller: str | None = None  # set by the LLM-agent bridge: who drives right now
        self.camera_frozen_since: float | None = None
        self.recording_mode = args.recording_mode
        self.clip: ClipRecorder | None = None
        self.saved_clips: list[ClipRecorder] = []
        self.clip_prompts: queue.Queue[ClipRecorder] = queue.Queue()  # stopped clips awaiting save / delete
        if not self.episodes:
            self.message = "Press d to dock, set up the scene, capture inits (i), then space."
        self.last_query_s = 0.0
        self.loop_hz = 0.0
        self.over_since: float | None = None
        self.quit_after_dock = False
        self.lock = threading.RLock()
        self.stop_control = threading.Event()
        self.instruction_lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._read_instructions, daemon=True).start()

    def _make_policy(self):
        args = self.args
        common = {
            "device": "cpu",
            "cameras": {"wrist": {}, "overhead": {}},
            "host": args.policy_host,
            "port": args.policy_port,
            "action_horizon": args.action_horizon,
            "lang_instruction": self.instruction,
            "use_overhead_init": args.use_overhead_init,
            "use_wrist_init": args.use_wrist_init,
            "image_size": FRAME_SIZE,
        }
        if POLICY_DEFAULTS[args.policy].get("agent"):
            return self._make_agent()
        if args.policy == "groot":
            return groot_lib.GR00TRemotePolicy(**common)
        # The MolmoAct2 HTTP client speaks every *_server.py protocol (G0.5, pi0.5, MolmoAct2).
        defaults = POLICY_DEFAULTS[args.policy]
        return molmoact2_lib.MolmoAct2RemotePolicy(
            **common, camera_names=["overhead", "wrist"], joint_frame=defaults["joint_frame"],
            num_steps=defaults["num_steps"], history_ticks=1,
        )

    # ----- MMRV episodes -----
    def _row(self) -> dict | None:
        return self.episodes[self.episode_index] if self.episodes else None

    def _family(self, row: dict) -> str:
        family = benchmark_lib.infer_task_family(row["instruction"])
        return f"bin_{row['n_objects']}" if family == benchmark_lib.TASK_BIN else family

    def _timeout_steps(self) -> int | None:
        row = self._row()
        if row is None:
            return None
        family = benchmark_lib.infer_task_family(row["instruction"])
        return int(round(benchmark_lib.episode_length_s(family, row["n_objects"]) * CONTROL_HZ))

    def _results(self) -> dict[int, dict]:
        """Latest recorded result per trial_id (re-marking appends; the last row wins)."""

        if not self.episodes or not self.results_path.exists():
            return {}
        latest = {}
        for line in self.results_path.read_text().splitlines():
            if line.strip():
                result = json.loads(line)
                latest[result["trial_id"]] = result
        return latest

    def _load_episode(self, index: int) -> None:
        self.episode_index = index % len(self.episodes)
        row = self._row()
        layouts = [json.loads(line) for line in self.layouts_path.read_text().splitlines() if line.strip()]
        self.layout = next((layout for layout in layouts if layout["trial_id"] == row["trial_id"]), None)
        if self.layout is None or [o["name"] for o in self.layout["objects"]] != row["objects"]:
            raise SystemExit(f"{self.layouts_path} has no layout matching trial {row['trial_id']}.")
        self.instruction = row["instruction"]
        if getattr(self, "policy", None) is not None:
            self.policy.set_language_instruction(self.instruction)
        done = self._results().get(row["trial_id"])
        status = f" (already recorded: {'success' if done['success'] else ', '.join(done['fail_type']) or 'failure'})" if done else ""
        self.message = f"Episode {row['trial_id']}: stage the scene on the outlines, dock, i, space.{status}"

    def _balanced_order(self, rows: list[dict]) -> list[dict]:
        """Run order (trial_ids unchanged): the 1st trial of every family x split cell, then every 2nd, then
        every 3rd, so stopping after any whole round (15 / 30 / 45) leaves a balanced set."""

        cells: dict[tuple, list[dict]] = {}
        for row in rows:
            cells.setdefault((row["ood_key"], self._family(row)), []).append(row)
        rounds = max(len(c) for c in cells.values())
        return [c[k] for k in range(rounds) for c in cells.values() if k < len(c)]

    def _next_unscored(self, index: int) -> int:
        """The next unscored episode after `index` in run order (wrapping), so re-runs scattered through the
        order are visited; when everything is scored, just the next episode."""

        done = self._results()
        n = len(self.episodes)
        for step in range(1, n + 1):
            i = (index + step) % n
            if self.episodes[i]["trial_id"] not in done:
                return i
        return (index + 1) % n

    def _first_unscored(self) -> int:
        done = self._results()
        return next((i for i, row in enumerate(self.episodes) if row["trial_id"] not in done), 0)

    def request_switch(self, target: str | None = None) -> None:
        if self.args.no_policy:
            self.message = "Started with --no_policy."
        elif self.switching:
            self.message = "Already switching policies."
        elif self.mode != "hold" or self.recorder is not None or self.pending is not None:
            self.message = "Dock and score the current episode before switching policies."
        else:
            following = target or POLICY_ORDER[(POLICY_ORDER.index(self.args.policy) + 1) % len(POLICY_ORDER)]
            if following == self.args.policy and self.policy is not None:
                self.message = f"Already running {following}."
                return
            self.switching = True
            threading.Thread(target=self._switch_policy, args=(following,), name="policy-switch", daemon=True).start()

    def _switch_policy(self, policy: str) -> None:
        def progress(text: str) -> None:
            self.message = text
            print(f"[SWITCH]: {text}", flush=True)

        previous = self.args.policy
        try:
            with self.lock:
                self.policy = None  # nothing can start while the server is swapped
                self.inits.clear()
            settings = POLICY_DEFAULTS[policy]
            running = self.servers.policy
            if not (running and SERVER_RECIPES[running] is SERVER_RECIPES[policy]
                    and POLICY_DEFAULTS[running]["policy_port"] == settings["policy_port"]):
                self.servers.stop(progress)  # g05 <-> argon share the G0.5 server and keep it loaded
            if policy != "argon" and self.proxy is not None:
                self.proxy.stop()
            self.servers.ensure(policy, settings["policy_port"], progress)
            with self.lock:
                self.args.policy, self.args.policy_label = policy, None
                for key in ("policy_port", "action_horizon", "use_overhead_init", "use_wrist_init"):
                    setattr(self.args, key, settings[key])
                self.policy = self._make_policy()
                self.policy.connect()
                if self.episodes:
                    self.results_path = self.args.results_root / policy / "results.jsonl"
                    self._load_episode(self._first_unscored())
                inits = " + ".join(k for k in ("overhead_init", "wrist_init") if getattr(self.args, f"use_{k}"))
                self.message = (
                    f"Now running {policy} ({self.args.action_horizon} actions/query, inits: {inits or 'none'}). "
                    f"Resuming at trial {self._row()['trial_id'] if self.episodes else '-'}."
                )
        except Exception as exc:  # noqa: BLE001 - surface the failure; the arm keeps holding
            self.message = f"Switch to {policy} failed: {exc}. Press p to try the next policy."
            print(f"[SWITCH]: failed: {exc!r} (was {previous})", flush=True)
            with self.lock:
                self.args.policy = policy  # so the next p moves on rather than retrying the same one
        finally:
            self.switching = False

    def reset_session_reference(self) -> None:
        frame = self.overhead_cam.latest_bgr()
        if self.session is None or frame is None:
            self.message = "No overhead frame (or not in episode mode)."
            return
        self.session.save(frame)
        self.message = f"Saved new camera/lighting reference: {self.session.path}"

    def _change_episode(self, delta: int) -> None:
        if not self.episodes:
            self.message = "No --episodes_jsonl loaded."
        elif self.mode == "run" or self.recorder is not None or self.pending is not None:
            self.message = "Finish the current episode (y / f) before changing episodes."
        else:
            self._load_episode(self.episode_index + delta)

    def mark_result(self, success: bool) -> None:
        """Close the episode and ask the UI thread to open the scoring form."""

        if not self.episodes:
            self.message = "Results are recorded in episode mode (--episodes_jsonl)."
            return
        self._end_agent_episode(success, "human_scored")
        if self.mode == "run":
            self._pause("scoring")
        if self.recorder is not None:
            self._close_recorder("scored")
        if self.last_episode is None:
            self.message = "Nothing to score: run the episode first."
            return
        self.pending = {"success": success}
        self.message = "Scoring form open."

    def save_result(self, success: bool, fail_types: list[str], notes: str, in_bin: list[bool] | None) -> None:
        """Append one row in the sim failure-review format (annotate_groot_sim_real_correspondence.py),
        without grasp attempts, plus the run's metadata."""

        row, episode = self.last_episode["row"], self.last_episode
        result = {
            "trial_id": row["trial_id"],
            "instruction": row["instruction"],
            "ood_key": row["ood_key"],
            "n_objects": row["n_objects"],
            "objects": list(row["objects"]),
            "success": success,
            "clutter": False,
            "fail_type": [] if success else fail_types,
            "notes": [notes] if notes else [],
            "trial_keys": [],
        }
        for field in ("target", "referents", "direction"):
            if row.get(field) is not None:
                result[field] = row[field]
        if in_bin is not None:
            # Four-object bins record per-object outcomes (bin_4 progress), as in the sim review.
            result["objects"] = [
                {"name": name, "attempts": None, "success": placed} for name, placed in zip(row["objects"], in_bin)
            ]
            result["success"] = success = all(in_bin)
            if success:
                result["fail_type"] = []
        result.update(
            task_family=self._family(row),
            timed_out=episode["timed_out"],
            steps=episode["steps"],
            policy=self.args.policy,
            policy_label=self.args.policy_label or self.args.policy,
            action_horizon=self.args.action_horizon,
            use_overhead_init=self.args.use_overhead_init,
            use_wrist_init=self.args.use_wrist_init,
            episodes_jsonl=str(self.args.episodes_jsonl),
            layout=self.layout,
            episode_dir=str(episode["dir"]),
            session_drift=self.drift,
            agent=None if not episode.get("agent") else {
                **episode["agent"], "harness": "v9.6",
                **({"model": AGENTS["sol"]["model"], "effort": AGENTS["sol"]["effort"], "runtime": "codex",
                    "credential": "user"} if self.args.policy == "sol" else
                   {"model": "gemini-4-argon", "effort": self.args.argon_effort,
                    "vertex_project": self.proxy.project if self.proxy else None}),
                "api_usd_list_price": round(self.agent_spend(since=episode["agent"]["started_at"] or 0.0), 4),
            },
            recorded_at=datetime.now().isoformat(timespec="seconds"),
        )
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        meta_path = Path(episode["dir"]) / "episode.json"
        meta = json.loads(meta_path.read_text())
        meta["result"] = {k: result[k] for k in ("success", "fail_type", "notes", "objects", "timed_out")}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        self.refresh_recording_annotations(Path(episode["dir"]))
        self.pending, self.last_episode = None, None
        verdict = "SUCCESS" if success else ", ".join(result["fail_type"]) or "failure"
        if in_bin is not None:
            verdict += f" ({sum(in_bin)}/4 in bin)"
        print(f"[RESULT]: trial {row['trial_id']} {verdict} -> {self.results_path}", flush=True)
        self._load_episode(self._next_unscored(self.episode_index))
        self.message = f"Saved trial {row['trial_id']}: {verdict}. Next: trial {self._row()['trial_id']} -- d to dock, stage it."
        self._maybe_rotate()

    def _scored_count(self, policy: str) -> int:
        path = self.args.results_root / policy / "results.jsonl"
        if not path.exists():
            return 0
        return len({json.loads(line)["trial_id"] for line in path.read_text().splitlines() if line.strip()})

    def _maybe_rotate(self) -> None:
        """--rotate_every N: at each multiple of N scored episodes, hand over to the policy that is furthest behind,
        so slow drift (night lighting, wear) spreads across policies instead of landing on one block."""

        n = self.args.rotate_every
        if not n or not self.episodes:
            return
        rotation = [p.strip() for p in self.args.rotation.split(",") if p.strip()]
        current = self.args.policy
        if self._scored_count(current) % n and self._scored_count(current) < len(self.episodes):
            return
        counts = {p: self._scored_count(p) for p in rotation}
        behind = [p for p in rotation if counts[p] < len(self.episodes)]
        if not behind:
            self.message += "  ROTATION COMPLETE: every policy has all episodes scored."
            return
        order = rotation[rotation.index(current) + 1:] + rotation[: rotation.index(current) + 1] if current in rotation else rotation
        following = min((p for p in order if p in behind), key=lambda p: counts[p])
        if following == current:
            return
        print(f"[ROTATE]: {current} reached {counts.get(current, self._scored_count(current))} scored; switching to "
              f"{following} ({counts[following]} scored)", flush=True)
        self.request_switch(following)
        self.message = f"ROTATING to {following} (every {n} episodes). Wait for 'Now running', then d, stage, inits."

    def cancel_result(self) -> None:
        self.pending = None
        self.message = "Scoring cancelled; press y / f to score this episode."

    def _make_agent(self) -> AgentPolicy:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import so101_real_pyrua as pyrua

        pyrua.check_frozen_harness()
        self.pyrua = pyrua
        if self.args.policy == "sol":
            home = AGENTS["sol"]["codex_home"]
            check = subprocess.run(["codex", "login", "status"], env={**os.environ, "CODEX_HOME": str(home)},
                                   capture_output=True, text=True, timeout=60)
            if check.returncode != 0 or "Logged in" not in (check.stdout + check.stderr):
                raise RuntimeError(f"Codex is not logged in under {home}: {(check.stdout + check.stderr).strip()}")
            # Sol bills the user's own OpenAI key, as in its sim campaign (Jay's key ran out on 2026-10-17).
            os.environ["CODEX_HOME"] = str(home)
            if self.proxy is not None:
                self.proxy.stop()
            return AgentPolicy(pyrua.RealVlaSession(self.args.policy_port))
        project = self.args.gcp_project or default_gcp_project()
        if not project:
            raise RuntimeError("No GCP project: pass --gcp_project")
        if self.proxy is None:
            self.proxy = VertexProxy(project, self.args.argon_effort, self.args.vertex_proxy_port,
                                     self.args.results_root / "argon" / "vertex_proxy.audit.jsonl")
        self.proxy.start()
        # The agent's only key is the loopback proxy token: Argon can reach Gemini only through Vertex.
        os.environ["SO101_ARGON_PROXY_TOKEN"] = VERTEX_TOKEN_FILE.read_text().strip()
        return AgentPolicy(pyrua.RealVlaSession(self.args.policy_port))

    def agent_spend(self, since: float = 0.0) -> float:
        """Priced $ of the current agent's real episodes since `since` (Argon: the Vertex proxy's audit;
        Sol: the tokens each finished episode recorded, priced as scripts/pyrua_campaign.py prices Sol)."""

        if self.args.policy == "argon":
            return self.proxy.spend_usd(since=since) if self.proxy else 0.0
        total = 0.0
        for path in self.args.output_root.glob("*/agent/pyrua_result.json"):
            try:
                if path.stat().st_mtime < since:
                    continue
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if data.get("model") == AGENTS["sol"]["model"]:
                total += self.pyrua.priced_usd((data.get("pyrua") or {}).get("tokens") or {})
        return total

    def agent_budget(self) -> float:
        return self.args.sol_budget_usd if self.args.policy == "sol" else self.args.argon_budget_usd

    @property
    def is_agent(self) -> bool:
        return getattr(self.policy, "is_agent", False)

    def _agent_frames(self):
        """Frames for the agent; through a camera dropout it waits (frozen, no steps) rather than failing."""

        while True:
            with self.lock:
                frames = self._frames()
                if frames is not None and self.camera_frozen_since is not None:
                    frames = self._camera_freeze(frames)
                if frames is not None:
                    return self.arm.sim_deg(), frames
                if self.mode == "run":
                    self._camera_freeze(None)
            episode = self.agent_episode
            if episode is None or episode.arm is None or episode.arm.terminal is not None:
                raise self.pyrua.EpisodeEnded()
            time.sleep(0.05)

    def agent_snapshot(self):
        return self._agent_frames()

    def agent_step(self, target_deg: np.ndarray):
        """One 30 Hz control step for the agent: command, wait for the tick, re-read arm and cameras."""

        episode = self.agent_episode
        while True:
            if episode is None or episode.arm is None or episode.arm.terminal is not None:
                raise self.pyrua.EpisodeEnded()
            if self.mode == "run":
                break
            time.sleep(0.05)  # paused: the agent waits, and the step does not count
        with self.lock:
            max_step = np.degrees(VELOCITY_LIMITS_RAD_S) / CONTROL_HZ
            self.goal = self.goal + np.clip(np.asarray(target_deg) - self.goal, -max_step, max_step)
            sent = self.goal.copy()
            sent[5] = max(sent[5], JAW_FLOOR_DEG)
            self.arm.command(sent)
        self._agent_tick = max(self._agent_tick + 1.0 / CONTROL_HZ, time.monotonic())
        time.sleep(max(0.0, self._agent_tick - time.monotonic()))
        state, frames = self._agent_frames()
        with self.lock:
            if self.recorder is not None:
                self.recorder.add(frames, state, self.goal.copy(), False, 0.0, self.agent_controller or "")
        return state, frames

    def _end_agent_episode(self, success: bool | None, reason: str) -> None:
        if self.agent_episode is not None:
            self.agent_episode.end(success, reason)

    def _read_instructions(self) -> None:
        for line in sys.stdin:
            if line.strip():
                self.instruction_lines.put(line.strip())

    @staticmethod
    def _rgb(frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if (rgb.shape[1], rgb.shape[0]) != FRAME_SIZE:
            rgb = cv2.resize(rgb, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        return rgb

    def _frames(self) -> dict[str, np.ndarray] | None:
        wrist, overhead = self.wrist_cam.latest_bgr(), self.overhead_cam.latest_bgr()
        if wrist is None or overhead is None:
            return None
        return {"wrist": self._rgb(wrist), "overhead": self._rgb(overhead)}

    # ----- actions bound to keys -----
    def capture_init(self, cameras: tuple[str, ...]) -> None:
        if self.episodes:
            # Sim init frames show the arm at rest in the dock before the episode, never mid-motion or mid-episode.
            off = np.abs(self.arm.sim_deg()[:5] - DOCK_SIM_DEG[:5]).max()
            if self.mode != "hold" or self.recorder is not None or off > DOCK_START_TOLERANCE_DEG:
                self.message = ("Inits are captured at the start of an episode, with the arm docked and still: "
                                + ("score or dock this episode first." if self.recorder is not None
                                   else f"press d and wait for 'Docked.' (off by {off:.0f} deg)."))
                return
        frames = self._frames()
        if frames is None:
            self.message = "Cannot capture init: a camera has no signal."
            return
        for camera in cameras:
            self.inits[f"{camera}_init"] = frames[camera].copy()
            if self.policy is not None:
                setattr(self.policy, f"{camera}_init_image", frames[camera].copy())
        self.message = f"Captured {', '.join(f'{c}_init' for c in cameras)}."

    def toggle_run(self) -> None:
        if self.mode == "run":
            self._pause("paused")
            return
        if self.switching:
            self.message = "Policy server is switching; wait until it is ready."
            return
        if self.policy is None:
            self.message = "No policy loaded (started with --no_policy, or a switch failed: press p)."
            return
        # Sim episodes start docked, and the init frames must show the docked arm.
        if self.episodes and self.recorder is None:
            off = np.abs(self.arm.sim_deg()[:5] - DOCK_SIM_DEG[:5]).max()
            if off > DOCK_START_TOLERANCE_DEG:
                self.message = f"Episodes start from the dock pose (off by {off:.0f} deg): press d first."
                return
        missing = [
            key
            for key, used in (("overhead_init", self.args.use_overhead_init), ("wrist_init", self.args.use_wrist_init))
            if used and key not in self.inits
        ]
        if missing:
            self.message = f"Capture {' and '.join(missing)} first (o / w / i)."
            return
        if self._frames() is None:
            self.message = "A camera has no signal."
            return
        if self.pending is not None:
            self.message = "Finish entering the previous result first."
            return
        if self.session is not None and self.recorder is None:
            overhead = self.overhead_cam.latest_bgr()
            if self.session.image is None:
                self.session.save(overhead)
                print(f"[INFO]: Saved the session camera/lighting reference to {self.session.path}", flush=True)
            self.drift = self.session.drift(overhead)
            if problems := SessionReference.problems(self.drift):
                self.message = "Conditions differ from the session reference: " + "; ".join(problems) + \
                    ". Restore them (or press r if the change is deliberate)."
                return
        if self.is_agent and self.recorder is None:
            if self.agent_episode is not None and not self.agent_episode.done.is_set():
                self.message = "The agent is still wrapping up the previous episode (writing its result); try again shortly."
                return
            spent, budget = self.agent_spend(), self.agent_budget()
            if spent >= budget:
                self.message = f"{self.args.policy} budget reached (${spent:.2f} of ${budget:.0f}); raise --{self.args.policy}_budget_usd."
                return
        self.policy.set_language_instruction(self.instruction)
        if self.recorder is None:
            row = self._row()
            self.recorder = EpisodeRecorder(
                self.args.output_root, self.instruction, self.inits,
                episode={"trial_id": row["trial_id"], "episodes_jsonl": str(self.args.episodes_jsonl)} if row else None,
            )
            if hasattr(self.policy, "trace_episode"):
                # Stateful HTTP servers (G0.5's visual memory) start a fresh episode when this changes.
                self.policy.trace_episode = int(time.time())
            if self.is_agent:
                if row is None:
                    self.message = "The LLM agent runs benchmark episodes only (--episodes_jsonl)."
                    self.recorder = None
                    return
                if self.args.policy == "sol":
                    sol = AGENTS["sol"]
                    self.agent_episode = self.pyrua.RealPyruaEpisode(
                        self, row, self.recorder.dir / "agent", model=sol["model"], effort=sol["effort"],
                        runtime="codex", codex_bin=shutil.which("codex") or "codex", vla=self.policy.vla,
                    )
                else:
                    self.agent_episode = self.pyrua.RealPyruaEpisode(
                        self, row, self.recorder.dir / "agent", model="gemini-4-argon", effort=self.args.argon_effort,
                        base_url=self.proxy.base_url, api_key_env="SO101_ARGON_PROXY_TOKEN", vla=self.policy.vla,
                    )
                self.agent_started_at = time.time()
                self.agent_controller = f"{self.agent_episode.agent_label} thinking"
                self._agent_tick = time.monotonic()
                self.agent_episode.start(np.radians(DOCK_SIM_DEG))
        self.mode = "run"
        self.message = "Policy running. space = pause, d = dock."
        if self.recording_mode and self.clip is None:
            self._start_clip()

    def _start_clip(self) -> None:
        row = self._row()
        policy = self.args.policy
        self.clip = ClipRecorder(
            self._clip_dir(), self._clip_stem(), (self.overhead_cam, self.wrist_cam),
            label_fn=lambda: self.agent_controller if policy in AGENTS else policy.upper(),
            metadata={
                "policy": policy, "instruction": self.instruction,
                "trial_id": row["trial_id"] if row else None,
                "task_family": self._family(row) if row else "",
                "episode_dir": str(self.recorder.dir.resolve()) if self.recorder else None,
            },
        )

    def refresh_recording_annotations(self, episode_dir: Path) -> None:
        """Refresh kept clips after the final call ledger or human score is written."""
        with self.lock:
            for clip in self.saved_clips:
                if clip.metadata.get("episode_dir") == str(episode_dir.resolve()):
                    clip.annotate_async()

    def _keep_clip(self, clip: ClipRecorder) -> Path:
        self.saved_clips.append(clip)
        return clip.keep()

    def _clip_dir(self) -> Path:
        if self.episodes:
            return self.results_path.parent / "recordings"
        return self.args.output_root / "recordings"

    def _clip_stem(self) -> str:
        policy = self.args.policy_label or self.args.policy
        row = self._row()
        if row is None:
            slug = re.sub(r"[^a-z0-9]+", "_", self.instruction.lower()).strip("_")[:40]
            return f"{policy}_{slug}"
        return f"{policy}_trial{row['trial_id']:02d}_{self._family(row)}_{row['ood_key']}"

    def toggle_recording_mode(self) -> None:
        self.recording_mode = not self.recording_mode
        if self.recording_mode:
            self.message = "Recording mode ON: raw + annotated clips (asks to save both on pause)."
            if self.mode == "run" and self.clip is None:
                self._start_clip()
        else:
            self.message = "Recording mode OFF" + (" (the current clip still ends at the next pause)." if self.clip else ".")

    def _finish_clip(self) -> None:
        """Called whenever the arm is no longer running: stop the clip and queue the save prompt."""

        if self.clip is not None:
            clip, self.clip = self.clip, None
            clip.stop()
            self.clip_prompts.put(clip)

    def _pause(self, why: str) -> None:
        self.mode = "hold"
        self.goal = self.arm.sim_deg()
        if self.policy is not None:
            self.policy.action_queue.clear()
        self.message = f"Holding ({why})."

    def start_dock(self) -> None:
        self.mode = "dock"
        self.dock = DockMotion(self.arm.sim_deg())
        scoring = self.recorder is not None and self.episodes
        self._end_episode("docked")
        if not scoring and self.pending is None:
            self.message = "Docking... (episode ended; recapture inits for the next one)"

    def _close_recorder(self, reason: str) -> None:
        recorder, self.recorder = self.recorder, None
        self.agent_controller = None
        directory = recorder.close(reason)
        print(f"[INFO]: Episode saved to {directory}", flush=True)
        if self.episodes and recorder.episode is not None:
            timeout = self._timeout_steps()
            terminal = (self.agent_episode.arm.terminal or {}) if self.is_agent and self.agent_episode else None
            self.last_episode = {
                "dir": directory,
                "row": self._row(),
                "steps": recorder.steps,
                "timed_out": (terminal.get("termination_reason") == "time_out" if terminal is not None
                              else timeout is not None and recorder.steps >= timeout),
                "agent": None if terminal is None else {
                    "terminal": dict(terminal), "started_at": getattr(self, "agent_started_at", None),
                    "ended_at": time.time(), "output": str(directory / "agent"),
                },
            }
            if self.pending is None:
                self.message = f"Episode ended ({reason}). Score it: y = success, f = failure."

    def _end_episode(self, reason: str) -> None:
        self._end_agent_episode(None, reason)
        if self.policy is not None:
            self.policy.action_queue.clear()
        if self.recorder is not None:
            self._close_recorder(reason)
        self.inits.clear()
        if self.policy is not None:
            self.policy.overhead_init_image = None
            self.policy.wrist_init_image = None

    # ----- control -----
    def step(self) -> None:
        while not self.instruction_lines.empty():
            line = self.instruction_lines.get()
            if self.episodes:
                self.message = "Episode mode: the instruction comes from the task file."
                continue
            self.instruction = line
            if self.policy is not None:
                self.policy.set_language_instruction(self.instruction)
            if self.recorder is not None:
                self.recorder.instructions.append((time.monotonic() - self.recorder.t0, self.instruction))
            self.message = f"Instruction: {self.instruction}"

        state = self.arm.sim_deg()
        queried = False
        if self.mode == "run" and self.is_agent:
            episode = self.agent_episode
            if episode is not None and episode.arm is not None and episode.arm.terminal is not None:
                reason = episode.arm.terminal.get("termination_reason")
                self._pause(f"Agent episode ended: {reason}")
                if self.recorder is not None:
                    self._close_recorder(reason)
            error = np.abs(self.goal[:5] - state[:5]).max()
            if error > WATCHDOG_ERROR_DEG:
                self.over_since = self.over_since or time.monotonic()
                if time.monotonic() - self.over_since > WATCHDOG_AFTER_S:
                    self._end_agent_episode(False, "watchdog")
                    self._pause(f"WATCHDOG: tracking error {error:.0f} deg, blocked?")
                    self.over_since = None
            else:
                self.over_since = None
            return  # the agent thread commands the arm (agent_step)
        if self.mode == "run":
            frames = self._frames()
            if frames is None or self.camera_frozen_since is not None:
                frames = self._camera_freeze(frames)
            if frames is None:
                pass  # frozen: hold the current goal below, keep the action queue, count no step
            else:
                queried = not self.policy.action_queue
                started = time.perf_counter()
                visual = {f"rgb_{camera}": frame for camera, frame in frames.items()}
                action_rad = self.policy.get_action(torch.tensor(np.radians(state), dtype=torch.float32), visual)
                if queried:
                    self.last_query_s = time.perf_counter() - started
                target = np.degrees(action_rad.numpy().astype(np.float64))
                # Per control step, as in sim (which pauses during a query); a slow query must not widen it.
                max_step = np.degrees(VELOCITY_LIMITS_RAD_S) / CONTROL_HZ
                self.goal = self.goal + np.clip(target - self.goal, -max_step, max_step)
                if self.recorder is not None:
                    self.recorder.add(frames, state, self.goal.copy(), queried, self._squeeze())
                    timeout = self._timeout_steps()
                    if timeout is not None and self.recorder.steps == timeout:
                        # Keep running and recording; the sim limit is flagged, and the form marks it TIMED OUT.
                        print(f"[INFO]: TIMED OUT (sim limit {timeout / CONTROL_HZ:.0f} s); still running.", flush=True)
        elif self.mode == "dock":
            target = self.dock.target()
            if target is None:
                self.mode = "hold"
                self.message = "Docked."
            else:
                self.goal = target
        sent = self.goal.copy()
        # Never past sim's own jaw limit: below it the real motor only bends the printed jaws.
        sent[5] = max(sent[5] - self._squeeze(), JAW_FLOOR_DEG)
        self.arm.command(sent)

        error = np.abs(self.goal[:5] - state[:5]).max()
        if self.mode != "hold" and error > WATCHDOG_ERROR_DEG:
            self.over_since = self.over_since or time.monotonic()
            if time.monotonic() - self.over_since > WATCHDOG_AFTER_S:
                self._pause(f"WATCHDOG: tracking error {error:.0f} deg, blocked?")
                self.over_since = None
        else:
            self.over_since = None

    def _camera_freeze(self, frames):
        """Track a camera dropout. Returns the frames once they are back (ending the freeze), else None."""

        now = time.monotonic()
        if frames is not None:
            gap = now - self.camera_frozen_since
            self.camera_frozen_since = None
            if self.recorder is not None:
                self.recorder.camera_gaps.append({"step": self.recorder.steps, "t": round(now - self.recorder.t0, 2),
                                                  "seconds": round(gap, 2)})
            print(f"[CAMERA]: back after {gap:.1f} s; episode continues where it froze.", flush=True)
            self.message = f"Camera reconnected after {gap:.1f} s; continuing (frozen time not counted)."
            return frames
        if self.camera_frozen_since is None:
            self.camera_frozen_since = now
            missing = [n for n, c in (("wrist", self.wrist_cam), ("overhead", self.overhead_cam)) if c.latest_bgr() is None]
            print(f"[CAMERA]: {'/'.join(missing)} dropped; freezing the episode.", flush=True)
        waited = now - self.camera_frozen_since
        self.message = f"CAMERA RECONNECTING ({waited:.0f} s): episode frozen, arm holding; it resumes by itself."
        if waited > CAMERA_FREEZE_MAX_S:
            self.camera_frozen_since = None
            if self.recorder is not None:
                self.recorder.camera_gaps.append({"step": self.recorder.steps, "t": round(now - self.recorder.t0, 2),
                                                  "seconds": round(waited, 2), "paused": True})
            self._pause(f"camera gone for {waited:.0f} s; fix it, then space resumes this episode")
        return None

    def _squeeze(self) -> float:
        """Real motors push less per degree of jaw error than the sim drive; add error while closing."""

        if self.mode != "run" or self.args.gripper_squeeze_deg <= 0.0:
            return 0.0
        open_deg, closed_deg = SQUEEZE_RAMP_DEG
        ramp = np.clip((open_deg - self.goal[5]) / (open_deg - closed_deg), 0.0, 1.0)
        return float(ramp * self.args.gripper_squeeze_deg)

    # ----- display -----
    def _overhead_overlay(self, live_bgr: np.ndarray) -> np.ndarray:
        mode = OVERLAY_MODES[self.overlay_mode]
        if mode == "layout":
            return overlay_lib.draw_layout(live_bgr, self.layout, self.projector) if self.layout else live_bgr
        if mode == "off" or not self.references:
            return live_bgr
        reference = cv2.imread(str(self.references[self.reference_index % len(self.references)]))
        reference = cv2.resize(reference, (live_bgr.shape[1], live_bgr.shape[0]))
        if mode == "blend":
            return cv2.addWeighted(live_bgr, 1.0 - self.overlay_alpha, reference, self.overlay_alpha, 0.0)
        if mode == "edges":
            edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (5, 5), 0), 60, 150)
            edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
            out = live_bgr.copy()
            out[edges > 0] = (0, 255, 0)
            return out
        return cv2.absdiff(live_bgr, reference)

    def render(self) -> np.ndarray:
        panes = []
        for name, camera in (("overhead", self.overhead_cam), ("wrist", self.wrist_cam)):
            frame = camera.latest_bgr()
            if frame is None:
                pane = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), np.uint8)
                cv2.putText(pane, f"{name}: NO SIGNAL ({camera.device})", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                pane = cv2.resize(frame, FRAME_SIZE) if (frame.shape[1], frame.shape[0]) != FRAME_SIZE else frame.copy()
                if (frame.shape[1], frame.shape[0]) != FRAME_SIZE:
                    cv2.putText(pane, f"source {frame.shape[1]}x{frame.shape[0]} resized", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                if name == "overhead":
                    pane = self._overhead_overlay(pane)
            init = self.inits.get(f"{name}_init")
            if init is not None:
                thumb = cv2.resize(cv2.cvtColor(init, cv2.COLOR_RGB2BGR), (160, 120))
                pane[8:128, -168:-8] = thumb
                cv2.rectangle(pane, (FRAME_SIZE[0] - 169, 7), (FRAME_SIZE[0] - 7, 129), (0, 200, 0), 2)
            label = name + ("" if name != "overhead" else f"  overlay={OVERLAY_MODES[self.overlay_mode]}")
            cv2.putText(pane, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            panes.append(pane)
        status = np.zeros((118, FRAME_SIZE[0] * 2, 3), np.uint8)
        row = self._row()
        if row is not None:
            timeout = self._timeout_steps()
            elapsed = self.recorder.steps if self.recorder is not None else 0
            episode_line = (
                f"[{self.args.policy_label or self.args.policy}] episode {self.episode_index + 1}/{len(self.episodes)}  "
                f"trial {row['trial_id']}  {self._family(row)} / {row['ood_key']}  "
                f"time {elapsed / CONTROL_HZ:5.1f}/{timeout / CONTROL_HZ:.0f} s  recorded {len(self._results())}/{len(self.episodes)}"
            )
        else:
            episode_line = "free mode (no --episodes_jsonl)"
        if self.is_agent and self.args.policy == "sol":
            episode_line += f"  | Sol high, your OpenAI key: ${self.agent_spend():.2f} of ${self.args.sol_budget_usd:.0f}"
        elif self.is_agent and self.proxy is not None:
            episode_line += f"  | Argon {self.args.argon_effort} via Vertex {self.proxy.project}: " \
                f"${self.proxy.spend_usd():.2f} of ${self.args.argon_budget_usd:.0f}"
        timed_out = row is not None and self.recorder is not None and self.recorder.steps >= self._timeout_steps()
        if timed_out:
            status[:] = (0, 0, 160)
            cv2.putText(panes[0], "TIMED OUT (past sim limit)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            cv2.putText(panes[1], "TIMED OUT (past sim limit)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        lines = [
            f"mode={self.mode.upper()}  loop={self.loop_hz:4.1f} Hz  last query={self.last_query_s * 1000:4.0f} ms  "
            f"inits: overhead={'Y' if 'overhead_init' in self.inits else '-'} wrist={'Y' if 'wrist_init' in self.inits else '-'}"
            + (f"  REC {'*' if self.clip else 'armed'}" + (f" {self.clip.frames / CONTROL_HZ:.0f}s" if self.clip else "")
               if self.recording_mode or self.clip else ""),
            episode_line,
            f"instruction: {self.instruction}",
            self.message,
            "space run/pause  o/w/i inits  d dock  a overlay  , . episode  y/f score  p/1-6 policy  r ref  v rec  q quit",
        ]
        for i, line in enumerate(lines):
            cv2.putText(status, line, (10, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
        view = np.hstack(panes)
        if self.is_agent and self.recorder is not None and self.agent_controller:
            draw_controller_banner(view, self.agent_controller)
        return np.vstack([view, status])

    def handle_key(self, key: int) -> bool:
        if key in (ord("q"), 27):
            self.quit_after_dock = True
            self.start_dock()
        elif key == ord(" "):
            self.toggle_run()
        elif key == ord("o"):
            self.capture_init(("overhead",))
        elif key == ord("w"):
            self.capture_init(("wrist",))
        elif key == ord("i"):
            self.capture_init(("overhead", "wrist"))
        elif key == ord("d"):
            self.start_dock()
        elif key == ord("a"):
            self.overlay_mode = (self.overlay_mode + 1) % len(OVERLAY_MODES)
        elif key == ord("n"):
            self.reference_index += 1
        elif key == ord("["):
            self.overlay_alpha = max(0.05, self.overlay_alpha - 0.1)
        elif key == ord("]"):
            self.overlay_alpha = min(0.95, self.overlay_alpha + 0.1)
        elif key == ord(","):
            self._change_episode(-1)
        elif key == ord("."):
            self._change_episode(1)
        elif key == ord("p"):
            self.request_switch()
        elif chr(key) in "123456":
            self.request_switch(POLICY_ORDER[int(chr(key)) - 1])
        elif key == ord("r"):
            self.reset_session_reference()
        elif key == ord("v"):
            self.toggle_recording_mode()
        elif key == ord("y"):
            self.mark_result(True)
        elif key == ord("f"):
            self.mark_result(False)
        return True

    def _control_loop(self) -> None:
        dt = 1.0 / CONTROL_HZ
        next_tick = time.monotonic()
        last = time.monotonic()
        while not self.stop_control.is_set():
            now = time.monotonic()
            try:
                with self.lock:
                    self.step()
                    if self.mode != "run":
                        self._finish_clip()
                    if self.quit_after_dock and self.mode == "hold":
                        self.stop_control.set()
            except Exception as exc:  # noqa: BLE001 - keep the arm holding and surface the error
                print(f"[ERROR]: control step failed: {exc!r}", flush=True)
                with self.lock:
                    if self.mode == "run":
                        self.mode = "hold"
                        if self.policy is not None:
                            self.policy.action_queue.clear()
                    self._finish_clip()
                    self.message = f"ERROR: {exc!r} -- holding last goal."
            self.loop_hz = 0.9 * self.loop_hz + 0.1 / max(now - last, 1e-3)
            last = now
            next_tick += dt
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()

    def run(self) -> None:
        print(__doc__)
        print(f"[INFO]: Dock pose (sim deg): {np.round(DOCK_SIM_DEG, 1).tolist()}")
        root = tk.Tk()
        root.title("SO-101 real runner")
        canvas = tk.Label(root)
        canvas.pack()
        interrupted = {"sigint": False}

        def on_key(event) -> None:
            key = ord(event.char) if event.char else (27 if event.keysym == "Escape" else 255)
            with self.lock:
                self.handle_key(key)

        form = {"open": False}

        def open_scoring_form() -> None:
            with self.lock:
                episode, request = self.last_episode, self.pending
            if episode is None or request is None:
                return
            row = episode["row"]
            four_bin = self._family(row) == "bin_4"
            form["open"] = True
            win = tk.Toplevel(root)
            win.title(f"Score trial {row['trial_id']}")
            win.transient(root)
            header = (
                f"{self.args.policy_label or self.args.policy}  |  trial {row['trial_id']}  |  "
                f"{self._family(row)} / {row['ood_key']}{'  |  TIMED OUT' if episode['timed_out'] else ''}\n"
                f"{row['instruction']}"
            )
            tk.Label(win, text=header, justify="left", font=("Helvetica", 12, "bold")).pack(anchor="w", padx=10, pady=6)
            success = tk.BooleanVar(master=win, value=request["success"])
            in_bin = []
            if four_bin:
                box = tk.LabelFrame(win, text="In the bin at the end (bin_4 progress)")
                box.pack(fill="x", padx=10)
                for name in row["objects"]:
                    var = tk.BooleanVar(master=win, value=request["success"])
                    tk.Checkbutton(box, text=name, variable=var).pack(anchor="w")
                    in_bin.append(var)
            else:
                tk.Checkbutton(win, text="Success", variable=success).pack(anchor="w", padx=10)
            tk.Label(win, text="Fail type (comma-separated; click tags to add/remove):").pack(anchor="w", padx=10, pady=(8, 0))
            fail_entry = tk.Entry(win, width=70)
            fail_entry.pack(fill="x", padx=10)
            if episode["timed_out"] and not request["success"]:
                fail_entry.insert(0, "timed out")
            tags = tk.Frame(win)
            tags.pack(fill="x", padx=10, pady=4)

            def toggle(tag: str) -> None:
                current = [t.strip() for t in fail_entry.get().split(",") if t.strip()]
                current = [t for t in current if t != tag] if tag in current else current + [tag]
                fail_entry.delete(0, "end")
                fail_entry.insert(0, ", ".join(current))

            for index, tag in enumerate(FAIL_TYPE_TAGS):
                tk.Button(tags, text=tag, command=lambda t=tag: toggle(t)).grid(row=index // 4, column=index % 4, sticky="ew")
            tk.Label(win, text="Notes:").pack(anchor="w", padx=10)
            notes_entry = tk.Entry(win, width=70)
            notes_entry.pack(fill="x", padx=10)
            warning = tk.Label(win, text="", fg="red")
            warning.pack(anchor="w", padx=10)

            def close() -> None:
                form["open"] = False
                win.destroy()

            def save(*_) -> None:
                placed = [var.get() for var in in_bin] if four_bin else None
                succeeded = all(placed) if four_bin else success.get()
                fail_types = [t.strip() for t in fail_entry.get().split(",") if t.strip()]
                if not succeeded and not fail_types:
                    warning.configure(text="A failure needs at least one fail type.")
                    return
                with self.lock:
                    self.save_result(succeeded, fail_types, notes_entry.get().strip(), placed)
                close()

            def cancel(*_) -> None:
                with self.lock:
                    self.cancel_result()
                close()

            buttons = tk.Frame(win)
            buttons.pack(fill="x", padx=10, pady=8)
            tk.Button(buttons, text="Save (Enter)", command=save).pack(side="left")
            tk.Button(buttons, text="Cancel (Esc)", command=cancel).pack(side="left", padx=6)
            win.bind("<Return>", save)
            win.bind("<Escape>", cancel)
            win.protocol("WM_DELETE_WINDOW", cancel)
            fail_entry.focus_set()

        def ask_clips() -> None:
            while not self.clip_prompts.empty():
                clip = self.clip_prompts.get()
                if clip.frames == 0:
                    clip.discard()
                    continue
                keep = messagebox.askyesno(
                    "Save recording?", f"Save recording?\n\n{clip.path.name}\n{clip.frames / CONTROL_HZ:.1f} s\n\n"
                    "Yes = save raw + fully annotated videos, No = delete both", parent=root)
                with self.lock:
                    if keep:
                        self.message = f"Raw saved: {self._keep_clip(clip).name} (full annotation builds in background)"
                    else:
                        clip.discard()
                        self.message = "Recording deleted."
                print(f"[INFO]: {self.message}", flush=True)

        def refresh() -> None:
            if interrupted["sigint"] or (self.stop_control.is_set() and not control.is_alive()):
                root.destroy()
                return
            ask_clips()
            if self.pending is not None and not form["open"]:
                open_scoring_form()
            image = cv2.cvtColor(self.render(), cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(image))
            canvas.configure(image=photo)
            canvas.image = photo
            root.after(33, refresh)

        root.bind("<Key>", on_key)
        root.protocol("WM_DELETE_WINDOW", lambda: on_key(type("E", (), {"char": "q", "keysym": "q"})()))
        signal.signal(signal.SIGINT, lambda *_: interrupted.update(sigint=True))

        self.arm.enable(self.goal)
        control = threading.Thread(target=self._control_loop, name="control", daemon=True)
        control.start()
        try:
            root.after(33, refresh)
            root.mainloop()
        finally:
            self.stop_control.set()
            control.join(timeout=5.0)
            with self.lock:
                self._end_episode("quit")
                self._finish_clip()
            while not self.clip_prompts.empty():  # never delete unanswered clips
                print(f"[INFO]: Recording kept (unanswered): {self._keep_clip(self.clip_prompts.get())}", flush=True)
            if interrupted["sigint"]:
                # Dropping a limp arm mid-task is worse than leaving the motors holding their last goal.
                print("\n[WARN]: Ctrl-C: torque LEFT ON, arm holding. Rerun and press q to dock, or power-cycle.", flush=True)
                self.arm.close(torque_off=False)
            else:
                self.arm.close()
                print("[INFO]: Docked, torque off.", flush=True)
            for camera in (self.wrist_cam, self.overhead_cam):
                camera.stop()
            self.servers.stop()
            if self.proxy is not None:
                self.proxy.stop()


def main() -> None:
    args = _parser().parse_args()
    for key, value in POLICY_DEFAULTS[args.policy].items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, value)
    Runner(args).run()


if __name__ == "__main__":
    main()
