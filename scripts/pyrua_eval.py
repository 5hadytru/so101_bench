# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run SO-101 Bench episodes with PyRUA-Lean (DAGroup-PKU) writing Python against the arm.

PyRUA-Lean runs the way it runs on RoboDojo: the benchmark's evaluation loop owns the simulator
and judges the episode, and PyRUA-Lean answers it as a policy (``pyrualean play --robot
so101_bench``, its ``hosts/so101_bench_host.py``).  This script is that evaluation loop.  Per
episode it resets the env exactly as ``gpt_policy_eval.py`` does, holds the start pose for
1.5 s, starts one ``pyrualean play`` process and serves it over a local socket:

*   it sends an observation (overhead and wrist RGB, their calibration, the measured joints, the
    robot base pose, the native-step count and limit) and asks for the next action;
*   an action is absolute joint targets for one 30 Hz control step; "wait" means the model is
    thinking (physics stays paused); an empty answer means the program finished, and the arm
    then holds its last command until the benchmark ends the episode;
*   when the benchmark ends the episode, it reports the verdict and waits for PyRUA-Lean to
    write its ``result.json``.

The model sees only the instruction, the public task rules (on the task card), the colour images
it asks for, camera calibration and robot proprioception.  Codex's shell, exec, web search and
file viewer are switched off, so the agent has only PyRUA-Lean's ``python`` and ``finish`` tools.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO-101 Bench PyRUA-Lean evaluator.")
parser.add_argument("--task", type=str, default="So101Bench-Bin-v0", help="Isaac Lab task name.")
parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
parser.add_argument("--episodes_jsonl", type=Path, required=True, help="Benchmark episode JSONL.")
parser.add_argument("--episode_layouts_jsonl", type=Path, required=True, help="Layouts matched by trial_id.")
parser.add_argument("--episode_indices", type=str, default=None, help="Comma-separated row indices to run.")
parser.add_argument("--output_dir", type=Path, required=True, help="New directory for this run's outputs.")
parser.add_argument("--pyrua_python", type=Path, default=Path.home() / "workspace" / "pyrua_venv" / "bin" / "python")
parser.add_argument("--codex_bin", type=str, default="codex", help="Codex CLI executable.")
parser.add_argument("--model", type=str, default="gpt-6.1-sol")
parser.add_argument("--runtime", choices=("codex", "openai"), default="codex", help="codex: Codex CLI; openai: PyRUA-Lean's chat-completions loop (Gemini).")
parser.add_argument("--base_url", type=str, default="https://generativelanguage.googleapis.com/v1beta/openai", help="Chat-completions endpoint (--runtime openai).")
parser.add_argument("--api_key_env", type=str, default="GEMINI_API_KEY", help="Environment variable holding the key (--runtime openai).")
parser.add_argument("--effort", type=str, default="high", help="Reasoning effort (PyRUA-Lean's RoboDojo run: high).")
parser.add_argument("--max_turns", type=int, default=100, help="Code cells per episode (RoboDojo profile: 100).")
parser.add_argument("--budget_s", type=float, default=5400.0, help="Wall-clock limit per episode (RoboDojo: 5400).")
parser.add_argument("--images", type=str, default="on-demand", help="PyRUA-Lean image policy (RoboDojo: on-demand).")
parser.add_argument("--feedback", type=str, default="pure", help="PyRUA-Lean feedback (RoboDojo: pure).")
parser.add_argument("--vla_handoff", action="store_true", help="Expose robo.vla_run: G0.5 drives, Sol watches and takes over as a last resort.")
parser.add_argument("--vla_port", type=int, default=8001, help="G0.5 server port (scripts/g05_server.py).")
parser.add_argument("--harness_version", type=str, default="v9.6", choices=("v9", "v9.1", "v9.2", "v9.3", "v9.4", "v9.5", "v9.6"),
                    help="Hand-off harness: v9 (takeover opens + returns to the start pose) or v9.1 (takeover in place, "
                         "slow first 3 s; move-task gap guidance) v9.2 (v9.1 + non-bin hover >= 2 in above the measured target top, next-to/between guidance), v9.3 (no bin takeover while G0.5 holds an object lifted), v9.4 (MOVE gap target 0.5-1 cm), v9.5 (NEXT-TO and MOVE gap targets 1-2 cm without contact), or v9.6 (v9.5 + one-second slow takeover recovery).")
parser.add_argument("--sam3_checkpoint", type=Path, default=Path.home() / "workspace" / "sam3" / "sam3.pt")
parser.add_argument("--baseline_root", type=Path, default=Path.home() / "workspace" / "PyRUA-Lean-so101-baseline",
                    help="Pinned original SO-101 port, used with --plain_profile original.")
parser.add_argument("--plain_profile", choices=("matched-v9.3", "matched-v9.4", "matched-v9.5", "matched-v9.6", "original"), default="matched-v9.6",
                    help="No-G0.5 arm: tuned v9.6 minus G0.5 (default), frozen v9.3/v9.4/v9.5, or historical original port.")
parser.add_argument("--matched_root", type=Path, default=None,
                    help="Override the frozen matched runtime; by default selected from --harness_version.")
parser.add_argument("--initial_hold_time_s", type=float, default=1.5)
parser.add_argument(
    "--scripted_check",
    type=Path,
    default=None,
    help="Run this script (in the PyRUA-Lean venv, same bridge arguments) instead of pyrualean play: no model.",
)
parser.add_argument(
    "--dry_run",
    action="store_true",
    default=False,
    help="Reset the first episode, print the observation the policy would get, and exit (no model, no SAM3).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.vla_handoff and args_cli.plain_profile.startswith("matched-"):
    args_cli.harness_version = args_cli.plain_profile.removeprefix("matched-")
MATCHED_RUNTIME = args_cli.harness_version in ("v9.3", "v9.4", "v9.5", "v9.6") and (
    args_cli.vla_handoff or args_cli.plain_profile.startswith("matched-"))
if args_cli.matched_root is None:
    matched_tag = args_cli.harness_version.replace(".", "") if MATCHED_RUNTIME else "v93"
    args_cli.matched_root = Path.home()/"workspace"/f"PyRUA-Lean-so101-{matched_tag}-matched"
PYRUA_SOURCE_ROOT = (args_cli.matched_root if MATCHED_RUNTIME else
                     args_cli.baseline_root if not args_cli.vla_handoff else None)
NO_G05_LABEL = f"{args_cli.harness_version}-no-g05" if MATCHED_RUNTIME else "plain-original"
args_cli.enable_cameras = True
# The headless camera kit fails to produce LdrColorSD on this machine; see groot_eval.py.
if args_cli.headless and not getattr(args_cli, "experience", ""):
    args_cli.experience = "isaaclab.python.rendering.kit"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from multiprocessing.connection import Client  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import so101_bench.tasks  # noqa: E402,F401
from so101_bench.benchmark import TASK_BIN, BenchmarkEpisodeSpec, load_episode_jsonl  # noqa: E402
from so101_bench.utils.task_guidance import task_rules_and_guidance, REVISION as GUIDANCE_REVISION
from eval_efficiency import efficiency_metrics
from so101_bench.mdp import mark_benchmark_robot_start  # noqa: E402
from so101_bench.tasks.direct.so101_bench.so101_bench_env_cfg import configure_env_cfg_for_object_pool  # noqa: E402
from so101_bench.utils import so101_kinematics as kin  # noqa: E402
from so101_bench.utils.llm_harness import (  # noqa: E402
    CAMERAS,
    INITIAL_ROBOT_JOINT_POS,
    PUBLIC_TASK_RULES,
    NonTargetPoseTracker,
    SimArm,
    load_layouts,
    object_pool,
    reset_env,
    reset_params,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
AUTHKEY = b"so101-bench"
# Codex keeps only PyRUA-Lean's MCP tools: no shell (it could read the layout files), no exec,
# no web search, no image viewer, no plugins or memories.
CODEX_EXTRA_ARGS = [
    "-c", "features.shell_tool=false",
    "-c", "features.unified_exec=false",
    "-c", "features.view_image=false",
    "-c", "features.apps=false",
    "-c", "features.plugins=false",
    "-c", "features.memories=false",
    "-c", 'web_search="disabled"',
]  # fmt: skip


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def observation(arm: SimArm, instruction: str, rules: str) -> dict:
    images = arm.images()
    q = arm.joints()
    return {
        "vision": {name: {"color": images[name]} for name in CAMERAS},
        "state": {
            "joint_pos": [float(v) for v in q],
            "base_pose": arm.world_from_base().tolist(),
            "cameras": arm.calibration(images),
            "status": status(arm),
        },
        "instruction": instruction,
        "task_rules": rules,
    }


VLA_SESSION: "VlaSession | None" = None
#: object asset slot i holds POOL[i] (reset_params); used only for the diagnostic pose report
POOL: list[str] = []


class VlaSession:
    """The G0.5 server behind robo.vla_run (molmoact2_eval.py's G0.5 settings)."""

    def __init__(self, env, port: int) -> None:
        from so101_bench.utils.molmoact2 import MolmoAct2RemotePolicy

        self.policy = MolmoAct2RemotePolicy(
            device=env.unwrapped.device, cameras=["overhead", "wrist"], host="127.0.0.1", port=port, action_horizon=15,
            lang_instruction="", camera_names=["overhead", "wrist"], image_size=None, timeout_s=120.0,
            num_steps=10, max_joint_step_deg=15.0, use_overhead_init=False, overhead_init_camera="overhead",
            use_wrist_init=False, wrist_init_camera="wrist", joint_frame="lerobot", history_ticks=1,
            prediction_trace_dir=None,
        )
        self.policy.connect()
        self.handoffs = 0
        self.arm = None  # the episode the policy last drove, and whether the planner moved since
        self.resumable = False
        self.watch = None  # robot-signal state of the current stint (_Watch)
        self.strip = None  # frames of the last stint, for the planner's look


#: G0.5's own jaw command, as a pad gap: below CLOSE_CMD_GAP_M it is commanding a close, above
#: OPEN_CMD_GAP_M an open.  These robot signals decide only WHEN the planner looks (a pause with
#: physics frozen); they never end G0.5's turn, and they never judge a grasp or a placement: the
#: planner does, from the cameras.
CLOSE_CMD_GAP_M = 0.012
OPEN_CMD_GAP_M = 0.030
LOOK_EVERY_S = 5.0  # a routine look at least this often
STILL_S, STILL_TCP_M, STILL_JAW_M = 2.0, 0.015, 0.005  # "arm_still": barely moved for 2 s
LAG_RAD, LAG_S = 0.2, 0.5  # "arm_lagging": joints this far behind G0.5's command for 0.5 s (jammed)
RELEASE_LOOK_S = 0.5  # "jaw_reopened": the look comes this long after the reopen (G0.5 keeps driving)
#: spatial tasks: the planner takes the arm the moment it sees the target lifted, so G0.5 must not
#: carry it off unseen.  Object-independent motion triggers (no jaw signals): a look whenever the
#: gripper has risen RISE_M above its lowest point since the last look ("rose_2cm") or travelled
#: SIDE_M sideways from where it was at the last look ("moved_3cm"); none in G0.5's first QUIET_S
#: (motion keeps counting, so a lift in that time triggers the look right at QUIET_S)
RISE_M, SIDE_M, QUIET_S = 0.02, 0.03, 3.0
STRIP_EVERY_S, STRIP_FRAMES = 1.0, 6  # the look's strip: one frame a second, the last six


def _tcp(arm: SimArm) -> np.ndarray:
    return np.asarray(kin.tool_pose(arm.world_from_base(), arm.joints()[:5]).tcp, dtype=float)


def _strip(frames: list[tuple[float, dict[str, np.ndarray]]], now_s: float) -> np.ndarray:
    """Overhead (top row) and wrist (bottom row) frames since the planner last looked, oldest left."""

    import cv2

    if len(frames) > STRIP_FRAMES:
        keep = np.linspace(0, len(frames) - 1, STRIP_FRAMES).round().astype(int)
        frames = [frames[i] for i in keep]
    cols = []
    for t, images in frames:
        tiles = [cv2.resize(images[name], (320, 240), interpolation=cv2.INTER_AREA) for name in ("overhead", "wrist")]
        col = np.concatenate(tiles, axis=0)
        cv2.putText(col, f"t-{now_s - t:.1f}s" if now_s - t > 0.05 else "now", (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2, cv2.LINE_AA)
        cols.append(col)
    return np.ascontiguousarray(np.concatenate(cols, axis=1))


class _Watch:
    """Robot-signal state of one G0.5 stint (kept across pauses: a resumed stint continues it)."""

    def __init__(self) -> None:
        self.widest, self.gap_hist, self.closed = None, [], 0
        self.was_open, self.grip, self.grip_gap, self.close_z, self.lift_fired = False, False, None, None, False
        self.stint_steps = 0  # G0.5 steps since this stint (fresh start) began
        self.open_n, self.release_in, self.lag_n = 0, 0, 0
        self.tcp_hist, self.jaw_hist = [], []


def run_vla(arm: SimArm, instruction: str, request: dict, session: "VlaSession | None") -> dict:
    """Let G0.5 drive until the planner should look (``mode="run"``); physics freezes meanwhile.

    The pause triggers (robot signals only, ``paused_because``): ``jaw_reopened`` (0.5 s after
    G0.5 reopened its jaw from a close: bin tasks only), ``rose_2cm`` / ``moved_3cm`` (spatial
    tasks only: the gripper rose 2 cm above its lowest point, or moved 3 cm sideways, since the
    last look; none in G0.5's first 3 s); ``arm_still`` (barely moved for 2 s), ``arm_lagging``
    (joints 0.2 rad behind the command for 0.5 s: pushing against something), ``check`` (5 s
    since the last look), ``deadline`` (``max_seconds`` from the planner's API: the takeover clock).
    A paused stint resumes exactly where it stopped (same policy memory, same action chunk) unless
    the planner moved the arm meanwhile; then G0.5 starts fresh.  With ``neutral_roll`` a fresh
    start first turns only the wrist roll back to its start-of-episode angle.  The reply says how
    long G0.5 drove and why it paused; never whether anything was grasped or placed.
    """

    mode = request.get("mode")
    if session is None:
        return {"mode": mode, "error": "no VLA server in this run"}
    if mode != "run":
        return {"mode": mode, "error": f"unknown hand-off mode {mode!r}"}
    policy = session.policy
    resume = session.arm is arm and session.resumable
    if not resume:
        # a new episode, or the planner moved the arm since G0.5 last drove: start G0.5 fresh.
        # Otherwise it continues where it paused, with its visual memory and the rest of its chunk.
        session.handoffs += 1
        policy.set_language_instruction(instruction)
        policy.reset()
        policy.trace_episode = session.handoffs  # a new id: the server starts a fresh episode memory
        session.watch = _Watch()
    w = session.watch
    spatial = arm.episode.task_family != TASK_BIN
    limit = max(1, int(round(float(request.get("max_seconds", LOOK_EVERY_S)) / arm.control_dt)))
    look_every = int(round(LOOK_EVERY_S / arm.control_dt))
    still_n, lag_need = int(round(STILL_S / arm.control_dt)), int(round(LAG_S / arm.control_dt))
    strip_every = int(round(STRIP_EVERY_S / arm.control_dt))
    steps, trigger, close_seen = 0, None, False
    frames = [(arm.elapsed_s(), arm.images())]
    start_tcp = _tcp(arm)  # spatial motion triggers: where the gripper was at the last look
    low_z = float(start_tcp[2])
    quiet = int(round(QUIET_S / arm.control_dt))
    if request.get("neutral_roll") and not resume:
        target = arm.action[0].clone()
        target[:5] = torch.as_tensor(arm.joints()[:5], dtype=torch.float32, device=target.device)
        roll0 = float(arm.home_q[4])
        for _ in range(60):
            now = float(target[4])
            if abs(roll0 - now) < 0.01:
                break
            target[4] = now + max(-0.06, min(0.06, roll0 - now))
            arm.action[0] = target
            steps += 1
            if arm.step():
                break
    while arm.terminal is None:
        with torch.inference_mode():
            joints = arm.obs["policy"]["joint_pos_obs"][0].clone()
            arm.action[0] = policy.get_action(joints, arm.obs["visual"])
        steps += 1
        if arm.step():
            break
        w.stint_steps += 1
        if steps % strip_every == 0:
            frames.append((arm.elapsed_s(), arm.images()))
        q = arm.joints()
        cmd_gap = float(kin.opening_for_jaw_angle(float(arm.action[0, 5])))
        tcp = _tcp(arm)
        # a close, relative to G0.5's own commands (thick objects are never commanded below their
        # width): under 12 mm, or 15 mm under its widest command and steady (1 mm over 5 steps),
        # held 10 steps, after the jaw had been opened
        w.was_open = w.was_open or cmd_gap > OPEN_CMD_GAP_M
        w.widest = cmd_gap if w.widest is None else max(w.widest, cmd_gap)
        w.gap_hist = (w.gap_hist + [cmd_gap])[-5:]
        settled = len(w.gap_hist) == 5 and max(w.gap_hist) - min(w.gap_hist) < 0.001 and cmd_gap <= w.widest - 0.015
        w.closed = w.closed + 1 if (cmd_gap < CLOSE_CMD_GAP_M or settled) else 0
        if not w.grip and w.was_open and w.closed >= 10:
            w.grip, w.grip_gap, w.close_z, w.lift_fired, w.open_n = True, cmd_gap, float(tcp[2]), False, 0
            close_seen = True
        if spatial:
            low_z = min(low_z, float(tcp[2]))
            if w.stint_steps >= quiet:
                if float(tcp[2]) - low_z >= RISE_M:
                    trigger = trigger or "rose_2cm"
                elif float(np.linalg.norm(tcp[:2] - start_tcp[:2])) >= SIDE_M:
                    trigger = trigger or "moved_3cm"
        if w.grip:
            w.grip_gap = min(w.grip_gap, cmd_gap)
            w.close_z = min(w.close_z, float(tcp[2]))
            w.open_n = w.open_n + 1 if cmd_gap >= w.grip_gap + 0.015 else 0
            if w.open_n >= 3:  # G0.5 reopened: an attempt ended, or a drop
                w.grip, w.release_in = False, int(round(RELEASE_LOOK_S / arm.control_dt))
        elif w.release_in > 0:
            w.release_in -= 1
            if w.release_in == 0 and not spatial:  # bin tasks only: spatial looks use motion alone
                trigger = trigger or "jaw_reopened"
        w.tcp_hist = (w.tcp_hist + [tcp])[-still_n:]
        w.jaw_hist = (w.jaw_hist + [cmd_gap])[-still_n:]
        if (len(w.tcp_hist) == still_n and float(np.max(np.ptp(np.asarray(w.tcp_hist), axis=0))) < STILL_TCP_M
                and float(np.ptp(w.jaw_hist)) < STILL_JAW_M):
            trigger = trigger or "arm_still"
            w.tcp_hist, w.jaw_hist = [], []
        lag = float(np.max(np.abs(arm.action[0, :5].detach().cpu().numpy() - q[:5])))
        w.lag_n = w.lag_n + 1 if lag > LAG_RAD else 0
        if w.lag_n >= lag_need:
            trigger, w.lag_n = trigger or "arm_lagging", 0
        if trigger is None and steps >= limit:
            trigger = "deadline" if limit < look_every else "check"
        if trigger is not None:
            break
    if not frames or frames[-1][0] != arm.elapsed_s():
        frames.append((arm.elapsed_s(), arm.terminal_frame if arm.terminal is not None else arm.images()))
    session.strip = _strip(frames, arm.elapsed_s())
    session.arm, session.resumable = arm, True
    return {
        "mode": mode, "steps_used": steps, "seconds": round(steps * arm.control_dt, 2),
        "ended": arm.terminal is not None, "paused_because": trigger or ("ended" if arm.terminal is not None else "check"),
        # internal, for the API's checks of the planner's verdicts (never shown to it as such):
        # whether G0.5 commanded a close during this stretch, and whether its jaw is closed now
        "close_seen": close_seen, "gripping": w.grip,
        # internal: the jaw command G0.5 left, so the planner's next motions keep it (popped by the host)
        "jaw_command": float(arm.action[0, 5]),
    }


def status(arm: SimArm) -> dict:
    return {"step": int(arm.step_id), "step_limit": int(round(arm.time_limit_s / arm.control_dt))}


def connect(port: int, proc: subprocess.Popen, timeout_s: float = 900.0):
    deadline = time.monotonic() + timeout_s
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"pyrualean exited ({proc.returncode}) before accepting the connection")
        try:
            return Client(("127.0.0.1", port), authkey=AUTHKEY)
        except (ConnectionRefusedError, FileNotFoundError):
            if time.monotonic() > deadline:
                raise TimeoutError("pyrualean did not listen in time") from None
            time.sleep(1.0)


def run_episode(env, episode: BenchmarkEpisodeSpec, obs: dict, output: Path, label: str) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    arm = SimArm(env, episode, output, obs)
    instruction = getattr(env.unwrapped, "so101_bench_instruction", episode.instruction)
    rules = PUBLIC_TASK_RULES[episode.task_family]
    if args_cli.harness_version == 'v9.6' and MATCHED_RUNTIME:
        rules = task_rules_and_guidance(episode.task_family, PUBLIC_TASK_RULES)
    (output/'task_prompt.txt').write_text(rules)
    for _ in range(max(0, math.ceil(args_cli.initial_hold_time_s / arm.control_dt))):
        if arm.step():
            break
    mark_benchmark_robot_start(
        env.unwrapped,
        object_asset_names=env.unwrapped.cfg.events.reset_benchmark_scene.params["object_asset_names"],
        bin_name="plastic_bin",
        force_robot_start_time=True,
    )
    arm.home_q = arm.joints()[:5].copy()  # start pose: the neutral roll of a fresh G0.5 start
    if episode.task_family != TASK_BIN:
        # diagnostic only (user, 2026-10-15): orientation changes of non-target objects, never scored
        u = env.unwrapped
        asset_names = u.cfg.events.reset_benchmark_scene.params["object_asset_names"]
        active = u._so101_active_object_mask[0].tolist()
        target = int(u._so101_target_object_ids[0])
        others = [i for i, on in enumerate(active) if on and i != target]
        arm.pose_tracker = NonTargetPoseTracker(env.unwrapped, asset_names, POOL, others)
    port = _free_port()
    trial = (episode.metadata or {}).get("trial_id")
    cmd = [
        str(args_cli.pyrua_python), "-m", "pyrualean.play",
        "--robot", "so101_bench", "--arm", "cells",
        "--task", label, "--seed", str(trial if trial is not None else 0),
        "--suite", args_cli.episodes_jsonl.stem,
        "--bridge-port", str(port), "--sam3-checkpoint", str(args_cli.sam3_checkpoint),
        "--out", str((output / "pyrua").resolve()), "--cuda-device", "0",
        "--model", args_cli.model, "--reasoning", args_cli.effort,
        "--runtime", args_cli.runtime, "--codex-bin", args_cli.codex_bin,
        "--max-turns", str(args_cli.max_turns), "--budget-s", str(args_cli.budget_s),
        "--images", args_cli.images, "--feedback", args_cli.feedback,
        "--primitives", "full" if args_cli.vla_handoff else "no-vla",
    ]  # fmt: skip
    if args_cli.runtime == "openai":
        # PyRUA-Lean's own chat-completions loop: tools run in-process (no shell at all). Effort goes in
        # the request body; no max_tokens, since thinking models count reasoning against it.
        cmd += [
            "--base-url", args_cli.base_url, "--api-key-env", args_cli.api_key_env, "--openai-max-tokens", "0",
            "--openai-extra-body", json.dumps({"reasoning_effort": args_cli.effort}),
        ]  # fmt: skip
    if args_cli.scripted_check is not None:
        cmd = [str(args_cli.pyrua_python), str(args_cli.scripted_check), "--bridge-port", str(port), "--out", str((output / "pyrua").resolve())]
    env_vars = dict(os.environ)
    env_vars.update(
        SO101_BENCH_ROOT=str(REPO_ROOT),
        SO101_TASK_RULES=rules,
        SAM3_CHECKPOINT_PATH=str(args_cli.sam3_checkpoint),
        # Native simulation can outlast Codex's default MCP tool timeout (~60 s).
        # Give both arms of the comparison the same execution allowance.
        PYRUALEAN_CODEX_EXTRA_ARGS=json.dumps(
            CODEX_EXTRA_ARGS + ["-c", "mcp_servers.pyrualean.tool_timeout_sec=900"]
        ),
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        SO101_HARNESS_VERSION=args_cli.harness_version if args_cli.vla_handoff or MATCHED_RUNTIME else "v9",
        SO101_NO_G05_ABLATION="1" if MATCHED_RUNTIME and not args_cli.vla_handoff else "0",
    )
    if PYRUA_SOURCE_ROOT is not None:
        source = PYRUA_SOURCE_ROOT / "src"
        if not (source / "pyrualean" / "so101_bench.py").is_file():
            raise FileNotFoundError(f"Pinned PyRUA runtime missing: {source}")
        env_vars["PYTHONPATH"] = str(source) + os.pathsep + env_vars.get("PYTHONPATH", "")
    (output / "pyrua_cmd.json").write_text(json.dumps(cmd, indent=1))
    log = (output / "pyrua.log").open("w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env_vars, cwd=str(output), start_new_session=True)
    started = time.monotonic()
    sim_wall_baseline, step_baseline = arm.sim_loop_wall_s, arm.step_id
    control_snapshot = None
    def capture_control_timing():
        return (time.monotonic()-started, arm.sim_loop_wall_s-sim_wall_baseline,
                (arm.step_id-step_baseline)*arm.control_dt)
    finished_program = False
    stop = "native_terminal"
    try:
        conn = connect(port, proc)
        conn.send({"op": "obs", "obs": observation(arm, instruction, rules), "status": status(arm)})
        conn.recv()
        while arm.terminal is None:
            if time.monotonic() - started > args_cli.budget_s + 600:
                stop = "wall_time_budget"
                break
            if finished_program:
                arm.step()
                continue
            conn.send({"op": "get_action"})
            reply = conn.recv().get("action")
            if reply == []:
                finished_program = True
                control_snapshot = capture_control_timing()
                stop = "program_finished"
                print(f"[INFO]: {label}: program finished at native step {arm.step_id}; holding.", flush=True)
                continue
            if not reply or reply[0].get("pyrualean") == "wait":
                if proc.poll() is not None:
                    stop = "policy_exited"
                    break
                continue
            if reply[0].get("pyrualean") == "vla":
                result = run_vla(arm, instruction, reply[0], VLA_SESSION)
                if arm.terminal is not None:
                    break
                obs_out = observation(arm, instruction, rules)
                if VLA_SESSION is not None and VLA_SESSION.strip is not None:
                    # what happened while G0.5 drove (recent.png): the planner's look at the pause
                    obs_out["vision"]["recent"] = {"color": VLA_SESSION.strip}
                    VLA_SESSION.strip = None
                conn.send({"op": "obs", "obs": obs_out, "status": {**status(arm), "vla": result}})
                conn.recv()
                continue
            targets = reply[0]["joint_targets"]
            arm.action[0] = torch.as_tensor(targets, dtype=torch.float32, device=arm.action.device)
            if VLA_SESSION is not None:
                VLA_SESSION.resumable = False  # the planner took control: the policy restarts next time
            if arm.step():
                break
            conn.send({"op": "obs", "obs": observation(arm, instruction, rules), "status": status(arm)})
            conn.recv()
        if control_snapshot is None:
            control_snapshot = capture_control_timing()
        if arm.terminal is None:
            # The policy stopped answering: the arm freezes on its last command until the end.
            while not arm.step():
                pass
        verdict = {"success": bool(arm.terminal.get("success")), "reason": arm.terminal.get("termination_reason")}
        try:
            conn.send({"op": "end", "verdict": verdict})
            conn.recv()
        except (OSError, EOFError):
            pass
        try:
            proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            proc.terminate()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.close()
        arm.close()
    pyrua = {}
    result_path = output / "pyrua" / "result.json"
    if result_path.is_file():
        data = json.loads(result_path.read_text())
        generation = data.get("generation") or {}
        pyrua = {
            key: data.get(key)
            for key in ("status", "reason", "finish", "tool_turns", "stateful_calls", "env_steps", "policy_wall_s", "notes")
        }
        pyrua["tokens"] = {k: generation.get(k) for k in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")}
        warnings = generation.get("warnings") or []
        pyrua["http_429"] = sum("HTTP 429" in str(w) for w in warnings)
        pyrua["provider_errors"] = generation.get("errors")
        pyrua["shell_commands"] = generation.get("shell_commands")
        pyrua["sandbox_flags"] = generation.get("sandbox_flags")
    result = {
        **arm.terminal,
        "episode": label,
        "trial_id": trial,
        "instruction": instruction,
        "task_family": episode.task_family,
        "objects": list(episode.objects),
        "agent_stop": stop,
        "wall_time_s": round(time.monotonic() - started, 1),
        "model": args_cli.model,
        "effort": args_cli.effort,
        "harness": (args_cli.harness_version if args_cli.vla_handoff else NO_G05_LABEL),
        "pyrua": pyrua,
    }
    tokens = pyrua.get('tokens') or {}
    inp, cached, out = (tokens.get(k) or 0 for k in ('input_tokens','cached_input_tokens','output_tokens'))
    estimated_usd = ((inp-cached)*2 + cached*.1 + out*10)/1e6
    if control_snapshot is not None:
        result['efficiency'] = efficiency_metrics(*control_snapshot, estimated_usd)
        result['efficiency']['pricing_note'] = 'Estimate from aggregate Codex tokens; separate cache writes and per-request long-context/tier adjustments are unavailable.'
    (output / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


from primary_eval_exclusions import EXCLUDED_TRIAL_IDS, primary_episode_excluded
from so101_bench.benchmark import episode_is_executable

def main() -> None:
    specs = load_episode_jsonl(args_cli.episodes_jsonl)
    indices = [int(v) for v in args_cli.episode_indices.split(",")] if args_cli.episode_indices else list(range(len(specs)))
    indices = [i for i in indices if not primary_episode_excluded(specs[i]) and episode_is_executable(specs[i])]
    if not indices:
        raise ValueError("No eligible episodes remain after exclusions")
    plan = [specs[i] for i in indices]
    layouts = load_layouts(plan, args_cli.episode_layouts_jsonl)
    pool = object_pool(plan)
    POOL[:] = pool
    args_cli.output_dir.mkdir(parents=True, exist_ok=args_cli.dry_run)
    (args_cli.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **{k: str(v) for k, v in vars(args_cli).items()},
                "episode_indices": indices,
                "codex_extra_args": CODEX_EXTRA_ARGS,
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "runtime_provenance": {
                    "guidance_revision": GUIDANCE_REVISION if args_cli.harness_version == 'v9.6' and MATCHED_RUNTIME else None,
                    "guidance_sha256": hashlib.sha256(Path(task_rules_and_guidance.__code__.co_filename).read_bytes()).hexdigest() if args_cli.harness_version == 'v9.6' and MATCHED_RUNTIME else None,
                    "pyrua_source_root": str(PYRUA_SOURCE_ROOT) if PYRUA_SOURCE_ROOT else "installed/live",
                    "profile": args_cli.harness_version if args_cli.vla_handoff else NO_G05_LABEL,
                    "baseline_revision": "b0e4fe56124769ba7bcea41fb84bd25669f76ce0" if not args_cli.vla_handoff and NO_G05_LABEL == "plain-original" else None,
                    "matched_manifest_sha256": hashlib.sha256((args_cli.matched_root / "matched_profile.json").read_bytes()).hexdigest() if MATCHED_RUNTIME else None,
                    "credential_label": os.environ.get("SO101_CREDENTIAL_LABEL") or ("jay" if os.environ.get("CODEX_HOME", "").endswith(".codex-jay-api") else "other"),
                },
            },
            indent=2,
        )
    )
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed
    env_cfg.scene.robot.init_state.joint_pos = dict(INITIAL_ROBOT_JOINT_POS)
    env_cfg.scene.camera_wrist.update_latest_camera_pose = True
    env_cfg.scene.camera_overhead.update_latest_camera_pose = True
    asset_names = configure_env_cfg_for_object_pool(env_cfg, pool)
    env_cfg.events.reset_benchmark_scene.params.update(reset_params(plan[0], layouts[0], pool, asset_names))
    env = gym.make(args_cli.task, cfg=env_cfg)
    if args_cli.vla_handoff:
        global VLA_SESSION
        VLA_SESSION = VlaSession(env, args_cli.vla_port)
    results = []
    try:
        for position, (index, episode, layout) in enumerate(zip(indices, plan, layouts, strict=True)):
            params = reset_params(episode, layout, pool, asset_names)
            env.unwrapped.cfg.events.reset_benchmark_scene.params.update(params)
            env.unwrapped.event_manager.get_term_cfg("reset_benchmark_scene").params.update(params)
            obs = reset_env(env)
            label = f"ep{index:03d}"
            if args_cli.dry_run:
                arm = SimArm(env, episode, args_cli.output_dir, obs)
                for _ in range(45):
                    arm.step()
                packet = observation(arm, episode.instruction, PUBLIC_TASK_RULES[episode.task_family])
                packet["vision"] = {k: list(v["color"].shape) for k, v in packet["vision"].items()}
                print("[DRY]", json.dumps(packet), flush=True)
                arm.close()
                return
            print(f"[INFO]: {label} ({position + 1}/{len(plan)}): {episode.instruction} | {list(episode.objects)}", flush=True)
            result = run_episode(env, episode, obs, args_cli.output_dir / label, label)
            results.append(result)
            print(
                f"[INFO]: {label}: success={result.get('success')} reason={result.get('termination_reason')} "
                f"cells={result['pyrua'].get('tool_turns')} sim_s={result.get('episode_length_s')} "
                f"wall_s={result.get('wall_time_s')} stop={result.get('agent_stop')}",
                flush=True,
            )
            with (args_cli.output_dir / "results.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(result, default=str) + "\n")
    finally:
        env.close()
    print(f"[INFO]: Finished {len(results)} episode(s): {sum(1 for r in results if r.get('success'))} success(es).", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
