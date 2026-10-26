"""PyRUA-Lean (+ G0.5 hand-offs) on the real SO-101: the bridge side of scripts/pyrua_eval.py.

``pyrualean play --robot so101_bench`` talks to an evaluation loop over a local socket (obs /
get_action / end). In sim that loop is pyrua_eval.py; here it is :class:`RealPyruaEpisode`, driven
by so101_real_policy_runner.py. The observation builder and the whole G0.5 hand-off (``run_vla``,
its pause triggers and look strips) are not copied: their definitions are loaded verbatim from
pyrua_eval.py at import (``_load_sim_bridge``), so real and sim run identical hand-off code. Only
the arm, the cameras, the calibration and the episode end are real:

* joints/images come from the real arm and cameras (640x480 RGB, as in sim);
* calibration is the sim's: overhead intrinsics/extrinsics from the benchmark camera defaults
  (the real iPhone is aligned to them), the wrist camera through the arm's forward kinematics,
  and the robot base at the sim spawn pose;
* a "wait" holds the arm and does not count toward the step limit (sim pauses physics);
* the episode ends at the sim step limit, or when the human scores it (y / f) or docks.
"""

from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from so101_real_recording import AgentMessageRecorder

from so101_bench.benchmark import TASK_BIN, DomainRandomizationSpec, episode_length_s
from so101_bench.utils import so101_kinematics as kin


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_BRIDGE = REPO_ROOT / "scripts" / "pyrua_eval.py"
CAMERAS = ("overhead", "wrist")
AUTHKEY = b"so101-bench"
CONTROL_DT = 1.0 / 30.0
INITIAL_HOLD_S = 1.5
# The configuration of the Argon sim campaign this mirrors
# (outputs/llm_harness_h2h/argon_medium_vertex60_v96_20261017/*/run_config.json).
HARNESS_VERSION = "v9.6"
MATCHED_ROOT = Path.home() / "workspace" / "PyRUA-Lean-so101-v96-matched"
MATCHED_MANIFEST_SHA256 = "8b0230aa63989de5afdd3b4bff0de6c547ddb1c1ee0df27392cf8a571d0d078e"
PYRUA_PYTHON = Path.home() / "workspace" / "pyrua_venv" / "bin" / "python"
SAM3_CHECKPOINT = Path.home() / "workspace" / "sam3" / "sam3.pt"
TASK_RULES_FILE = REPO_ROOT / "source/so101_bench/so101_bench/assets/real_robot/pyrua_v96_task_rules.json"
MAX_TURNS, BUDGET_S, IMAGES, FEEDBACK = 100, 5400.0, "on-demand", "pure"
# Sim spawn pose of the robot base in the environment frame (assets/so101.py SO101_CFG.init_state).
BASE_POS = (0.05209, 0.18061, -0.03102)
BASE_YAW_DEG = 6.195

# Definitions taken verbatim from the sim bridge.
_SIM_NAMES = (
    "observation", "status", "CLOSE_CMD_GAP_M", "OPEN_CMD_GAP_M", "LOOK_EVERY_S", "STILL_S", "LAG_RAD",
    "RELEASE_LOOK_S", "RISE_M", "STRIP_EVERY_S", "_tcp", "_strip", "_Watch", "run_vla", "CODEX_EXTRA_ARGS",
)


def _load_sim_bridge() -> dict:
    tree = ast.parse(SIM_BRIDGE.read_text())
    namespace = {"np": np, "torch": torch, "kin": kin, "TASK_BIN": TASK_BIN, "CAMERAS": CAMERAS,
                 "SimArm": object, "VlaSession": object, "json": json, "math": math, "time": time}
    found = set()
    for node in tree.body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for target in node.targets for t in ([target] if isinstance(target, ast.Name) else
                     target.elts if isinstance(target, ast.Tuple) else []) if isinstance(t, ast.Name)]
        if set(names) & set(_SIM_NAMES):
            exec(compile(ast.Module([node], type_ignores=[]), str(SIM_BRIDGE), "exec"), namespace)
            found |= set(names)
    if missing := set(_SIM_NAMES) - found:
        raise RuntimeError(f"pyrua_eval.py no longer defines {sorted(missing)}")
    return namespace


SIM = _load_sim_bridge()
observation, status, run_vla = SIM["observation"], SIM["status"], SIM["run_vla"]
#: Sol's Codex CLI flags, exactly as pyrua_eval.py passes them (plus its 900 s MCP tool timeout).
CODEX_EXTRA_ARGS = SIM["CODEX_EXTRA_ARGS"] + ["-c", "mcp_servers.pyrualean.tool_timeout_sec=900"]
#: $ per 1M tokens (uncached input, cached input, output incl. reasoning), as scripts/pyrua_campaign.py prices Sol.
SOL_PRICES = (2.00, 0.10, 10.00)


def priced_usd(tokens: dict, prices=SOL_PRICES) -> float:
    inp, cached, out = (int(tokens.get(k) or 0) for k in ("input_tokens", "cached_input_tokens", "output_tokens"))
    u, c, o = prices
    return (max(0, inp - cached) * u + cached * c + out * o) / 1e6


def _rpy_matrix(rpy_deg) -> np.ndarray:
    rx, ry, rz = (math.radians(v) for v in rpy_deg)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


_OPENGL_TO_ROS = np.diag([1.0, -1.0, -1.0])


def _intrinsic(hfov_deg: float, width: int = 640, height: int = 480) -> np.ndarray:
    focal = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])


def world_from_base() -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = _rpy_matrix((0.0, 0.0, BASE_YAW_DEG))
    result[:3, 3] = BASE_POS
    return result


def calibration(joints_rad: np.ndarray) -> dict:
    """The sim's camera calibration (canonical appearance) for these joint angles."""

    spec = DomainRandomizationSpec()
    overhead = np.eye(4)
    overhead[:3, :3] = _rpy_matrix((spec.overhead_camera_roll_deg, spec.overhead_camera_pitch_deg,
                                    spec.overhead_camera_yaw_deg)) @ _OPENGL_TO_ROS
    overhead[:3, 3] = (spec.overhead_camera_position_x_m, spec.overhead_camera_position_y_m,
                       spec.overhead_camera_position_z_m)
    mount = np.eye(4)
    mount[:3, :3] = _rpy_matrix((spec.wrist_camera_roll_deg, spec.wrist_camera_pitch_deg,
                                 spec.wrist_camera_yaw_deg)) @ _OPENGL_TO_ROS
    mount[:3, 3] = (spec.wrist_camera_position_x_m, spec.wrist_camera_position_y_m, spec.wrist_camera_position_z_m)
    wrist = world_from_base() @ kin.gripper_pose_in_base(joints_rad[:5]) @ mount
    result = {}
    for name, transform, fov in (("overhead", overhead, spec.overhead_camera_horizontal_fov_deg),
                                 ("wrist", wrist, spec.wrist_camera_horizontal_fov_deg)):
        result[name] = {
            "intrinsic": _intrinsic(fov).round(6).tolist(),
            "resolution": [640, 480],
            "camera_to_environment": transform.round(6).tolist(),
            "distortion_model": "pinhole",
            "frame": "environment",
            "optical_axes": "x_right_y_down_z_forward",
        }
    return result


def task_rules(task_family: str) -> str:
    data = json.loads(TASK_RULES_FILE.read_text())
    return data["rules"][task_family]


def check_frozen_harness() -> None:
    import hashlib

    manifest = MATCHED_ROOT / "matched_profile.json"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != MATCHED_MANIFEST_SHA256:
        raise RuntimeError(f"{manifest} differs from the frozen v9.6 runtime the sim campaigns (Sol, Argon) used")
    guidance = REPO_ROOT / "source/so101_bench/so101_bench/utils/task_guidance.py"
    if hashlib.sha256(guidance.read_bytes()).hexdigest() != json.loads(TASK_RULES_FILE.read_text())["guidance_sha256"]:
        raise RuntimeError("task_guidance.py changed since the real task rules were frozen")


class RealVlaSession:
    """pyrua_eval.VlaSession for the real arm: the same G0.5 client and settings, on CPU tensors."""

    def __init__(self, port: int) -> None:
        from so101_bench.utils.molmoact2 import MolmoAct2RemotePolicy

        self.policy = MolmoAct2RemotePolicy(
            device="cpu", cameras=["overhead", "wrist"], host="127.0.0.1", port=port, action_horizon=15,
            lang_instruction="", camera_names=["overhead", "wrist"], image_size=None, timeout_s=120.0,
            num_steps=10, max_joint_step_deg=15.0, use_overhead_init=False, overhead_init_camera="overhead",
            use_wrist_init=False, wrist_init_camera="wrist", joint_frame="lerobot", history_ticks=1,
            prediction_trace_dir=None,
        )
        self.policy.connect()
        self.handoffs = 0
        self.arm = None
        self.resumable = False
        self.watch = None
        self.strip = None


class EpisodeEnded(Exception):
    pass


class RealPyruaArm:
    """The SimArm interface pyrua_eval's observation() and run_vla() use, on the real robot.

    ``step()`` hands this control step's joint targets to the runner's 30 Hz loop and blocks until
    it has been commanded and the arm/cameras re-read.
    """

    def __init__(self, runner, episode, initial_rad: np.ndarray):
        self.runner = runner
        self.episode = episode
        self.control_dt = CONTROL_DT
        self.time_limit_s = episode_length_s(episode.task_family, len(episode.objects))
        self.action = torch.as_tensor(initial_rad, dtype=torch.float32).reshape(1, 6).clone()
        self.step_id = 0
        self.sim_loop_wall_s = 0.0
        self.terminal: dict | None = None
        self.terminal_frame = None
        self.home_q = None
        self._refresh(*runner.agent_snapshot())

    def _refresh(self, joints_deg: np.ndarray, frames_rgb: dict) -> None:
        self.q = np.radians(np.asarray(joints_deg, dtype=float))
        self.frames = frames_rgb
        self.obs = {
            "policy": {"joint_pos_obs": torch.as_tensor(self.q, dtype=torch.float32).reshape(1, 6)},
            "visual": {f"rgb_{name}": frame[None] for name, frame in frames_rgb.items()},
        }

    def joints(self) -> np.ndarray:
        return self.q.copy()

    def world_from_base(self) -> np.ndarray:
        return world_from_base()

    def images(self) -> dict:
        return {name: self.frames[name] for name in CAMERAS}

    def calibration(self, images) -> dict:
        return calibration(self.q)

    def elapsed_s(self) -> float:
        return self.step_id * self.control_dt

    def end(self, success: bool | None, reason: str) -> None:
        if self.terminal is None:
            self.terminal_frame = self.images()
            self.terminal = {"success": bool(success), "termination_reason": reason,
                             "episode_length_s": self.step_id * self.control_dt, "agent_control_steps": self.step_id}

    def step(self) -> bool:
        """One control step with the current action; True once the episode has ended."""

        if self.terminal is not None:
            return True
        started = time.monotonic()
        try:
            snapshot = self.runner.agent_step(np.degrees(self.action[0].numpy().astype(np.float64)))
        except EpisodeEnded:
            return True
        finally:
            self.sim_loop_wall_s += time.monotonic() - started
        self.step_id += 1
        self._refresh(*snapshot)
        if self.step_id >= int(round(self.time_limit_s / self.control_dt)):
            self.end(False, "time_out")
        return self.terminal is not None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class RealPyruaEpisode:
    """One episode: start pyrualean play, serve it like pyrua_eval.run_episode, report its result."""

    def __init__(self, runner, row: dict, output: Path, *, model: str, effort: str, vla: RealVlaSession,
                 runtime: str = "openai", base_url: str | None = None, api_key_env: str | None = None,
                 codex_bin: str = "codex"):
        from so101_bench.benchmark import episode_spec_from_json

        self.runner, self.row, self.output, self.vla = runner, row, output, vla
        self.episode = episode_spec_from_json(row)
        self.model, self.effort, self.base_url, self.api_key_env = model, effort, base_url, api_key_env
        # runtime "openai": PyRUA-Lean's chat-completions loop (Argon via the Vertex proxy);
        # "codex": the Codex CLI with the caller's CODEX_HOME credential (Sol), as pyrua_eval runs it.
        self.runtime, self.codex_bin = runtime, codex_bin
        # who drives, shown on the runner's view and recording clips and logged per step
        self.agent_label = "SOL" if "sol" in model else "ARGON" if "argon" in model else model.upper()
        self.arm: RealPyruaArm | None = None
        self.proc: subprocess.Popen | None = None
        self.stop_reason = None
        self.finished_program = False
        self.done = threading.Event()
        self.error: str | None = None
        self.result: dict | None = None

    def start(self, initial_rad: np.ndarray) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        self.arm = RealPyruaArm(self.runner, self.episode, initial_rad)
        self.messages = AgentMessageRecorder(self.output)
        self.messages.start()
        threading.Thread(target=self._run, name="pyrua-episode", daemon=True).start()

    def end(self, success: bool | None, reason: str) -> None:
        if self.arm is not None:
            self.arm.end(success, reason)

    def _recv(self, conn):
        """An operator ending an episode must also interrupt a stalled bridge RPC."""
        while not conn.poll(0.1):
            if self.arm.terminal is not None:
                raise EpisodeEnded()
        return conn.recv()

    def _finish_process(self) -> None:
        """Let the host flush its ledger, without waiting minutes for Codex to stop."""
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            self.proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        # Stop only this host's Codex planner first. Leaving the Python host
        # alive lets its normal finally block write calls, usage and results.
        children = subprocess.run(
            ["ps", "--ppid", str(self.proc.pid), "-o", "pid=,args="], capture_output=True, text=True,
        ).stdout.splitlines()
        for child in children:
            parts = child.strip().split(maxsplit=1)
            if len(parts) == 2 and "codex exec " in parts[1]:
                try:
                    os.kill(int(parts[0]), signal.SIGTERM)
                except ProcessLookupError:
                    pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Argon or an unresponsive host: stop the owned process group,
            # which was started in its own session and excludes the runner.
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
            except ProcessLookupError:
                return
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=3)

    def _command(self, port: int, rules: str) -> tuple[list[str], dict]:
        label = f"ep{int(self.row['trial_id']):03d}"
        cmd = [
            str(PYRUA_PYTHON), "-m", "pyrualean.play", "--robot", "so101_bench", "--arm", "cells",
            # same suite label as the sim campaigns (it appears in the prompt): their task file was primary_eval_v2.jsonl
            "--task", label, "--seed", str(self.row["trial_id"]), "--suite", "primary_eval_v2",
            "--bridge-port", str(port), "--sam3-checkpoint", str(SAM3_CHECKPOINT),
            "--out", str((self.output / "pyrua").resolve()), "--cuda-device", "0",
            "--model", self.model, "--reasoning", self.effort, "--runtime", self.runtime, "--codex-bin", self.codex_bin,
            "--max-turns", str(MAX_TURNS), "--budget-s", str(BUDGET_S), "--images", IMAGES, "--feedback", FEEDBACK,
            "--primitives", "full",
        ]  # fmt: skip
        if self.runtime == "openai":
            cmd += [
                "--base-url", self.base_url, "--api-key-env", self.api_key_env, "--openai-max-tokens", "0",
                "--openai-extra-body", json.dumps({"reasoning_effort": self.effort}),
            ]  # fmt: skip
        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        env.update(
            SO101_BENCH_ROOT=str(REPO_ROOT), SO101_TASK_RULES=rules, SAM3_CHECKPOINT_PATH=str(SAM3_CHECKPOINT),
            CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            SO101_HARNESS_VERSION=HARNESS_VERSION, SO101_NO_G05_ABLATION="0",
            PYTHONPATH=str(MATCHED_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        )
        if self.runtime == "codex":
            env["PYRUALEAN_CODEX_EXTRA_ARGS"] = json.dumps(CODEX_EXTRA_ARGS)
        return cmd, env

    def _run(self) -> None:
        from multiprocessing.connection import Client

        arm, instruction = self.arm, self.row["instruction"]
        rules = task_rules(self.episode.task_family)
        (self.output / "task_prompt.txt").write_text(rules)
        started = time.monotonic()
        try:
            # pyrua_eval holds the start pose for 1.5 s before the agent starts, counted as steps
            for _ in range(math.ceil(INITIAL_HOLD_S / CONTROL_DT)):
                if arm.step():
                    break
            arm.home_q = arm.joints()[:5].copy()
            port = _free_port()
            cmd, env = self._command(port, rules)
            (self.output / "pyrua_cmd.json").write_text(json.dumps(cmd, indent=1))
            log = (self.output / "pyrua.log").open("w")
            self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(self.output),
                                         start_new_session=True)
            conn = self._connect(port)
            conn.send({"op": "obs", "obs": observation(arm, instruction, rules), "status": status(arm)})
            self._recv(conn)
            stop = "native_terminal"
            while arm.terminal is None:
                if time.monotonic() - started > BUDGET_S + 600:
                    stop = "wall_time_budget"
                    arm.end(False, "wall_time_budget")
                    break
                if self.finished_program:
                    arm.step()  # the arm holds its last command until the episode ends, as in sim
                    continue
                conn.send({"op": "get_action"})
                reply = self._recv(conn).get("action")
                if reply == []:
                    self.finished_program, stop = True, "program_finished"
                    self.runner.agent_controller = "finished"
                    self.runner.message = f"{self.agent_label} finished its program; the arm holds. Score it: y / f."
                    continue
                if not reply or reply[0].get("pyrualean") == "wait":
                    self.runner.agent_controller = f"{self.agent_label} thinking"
                    if self.proc.poll() is not None:
                        stop = "policy_exited"
                        arm.end(False, "policy_exited")
                        break
                    time.sleep(0.01)
                    continue
                if reply[0].get("pyrualean") == "vla":
                    self.runner.agent_controller = "G0.5"
                    result = run_vla(arm, instruction, reply[0], self.vla)
                    self.runner.agent_controller = f"{self.agent_label} thinking"
                    if arm.terminal is not None:
                        break
                    obs_out = observation(arm, instruction, rules)
                    if self.vla.strip is not None:
                        obs_out["vision"]["recent"] = {"color": self.vla.strip}
                        self.vla.strip = None
                    conn.send({"op": "obs", "obs": obs_out, "status": {**status(arm), "vla": result}})
                    self._recv(conn)
                    continue
                self.runner.agent_controller = self.agent_label
                arm.action[0] = torch.as_tensor(reply[0]["joint_targets"], dtype=torch.float32)
                self.vla.resumable = False  # the planner took control: G0.5 restarts next time
                if arm.step():
                    break
                conn.send({"op": "obs", "obs": observation(arm, instruction, rules), "status": status(arm)})
                self._recv(conn)
            self.stop_reason = stop
            try:
                conn.send({"op": "end", "verdict": {"success": bool(arm.terminal.get("success")),
                                                    "reason": arm.terminal.get("termination_reason")}})
                if conn.poll(2.0):
                    conn.recv()
            except (OSError, EOFError):
                pass
        except EpisodeEnded:
            self.stop_reason = (arm.terminal or {}).get("termination_reason", "episode_ended")
        except Exception as exc:  # noqa: BLE001 - surfaced to the runner; the arm holds
            self.error = repr(exc)
            arm.end(False, "bridge_error")
        finally:
            self._finish_process()
            self.result = self._summary(started)
            (self.output / "pyrua_result.json").write_text(json.dumps(self.result, indent=2, default=str))
            self.messages.stop()
            self.done.set()
            self.runner.refresh_recording_annotations(self.output.parent)

    def _connect(self, port: int, timeout_s: float = 900.0):
        from multiprocessing.connection import Client

        deadline = time.monotonic() + timeout_s
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"pyrualean exited ({self.proc.returncode}) before accepting the connection")
            if self.arm.terminal is not None:
                raise EpisodeEnded()
            try:
                return Client(("127.0.0.1", port), authkey=AUTHKEY)
            except (ConnectionRefusedError, FileNotFoundError):
                if time.monotonic() > deadline:
                    raise TimeoutError("pyrualean did not listen in time") from None
                time.sleep(1.0)

    def _summary(self, started: float) -> dict:
        pyrua = {}
        result_path = self.output / "pyrua" / "result.json"
        if result_path.is_file():
            data = json.loads(result_path.read_text())
            generation = data.get("generation") or {}
            pyrua = {k: data.get(k) for k in ("status", "reason", "finish", "tool_turns", "stateful_calls",
                                                "env_steps", "policy_wall_s", "notes")}
            pyrua["tokens"] = {k: generation.get(k) for k in ("input_tokens", "cached_input_tokens", "output_tokens",
                                                               "reasoning_output_tokens")}
            pyrua["provider_errors"] = generation.get("errors")
        return {
            **(self.arm.terminal or {}),
            "agent_stop": self.stop_reason,
            "bridge_error": self.error,
            "wall_time_s": round(time.monotonic() - started, 1),
            "model": self.model, "effort": self.effort, "harness": HARNESS_VERSION,
            "g05_handoffs": self.vla.handoffs if self.vla else None,
            "pyrua": pyrua,
        }
