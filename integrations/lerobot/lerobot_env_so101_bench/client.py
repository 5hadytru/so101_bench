"""Start the Isaac Lab simulator server and talk to it (LeRobot side, Python >= 3.12)."""

from __future__ import annotations

import atexit
import logging
import os
import socket
import subprocess
import time
from pathlib import Path

import msgpack
import numpy as np
import zmq

logger = logging.getLogger(__name__)
SERVER_SCRIPT = Path(__file__).with_name("sim_server.py")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SimClient:
    """One Isaac Sim process per LeRobot process, shared by every episode's gym.Env.

    Isaac Sim allows one SimulationApp per process and takes ~1 min to start, so the server is
    started once and reconfigured per episode; it is shut down when the LeRobot process exits.

    lerobot-eval builds the envs before it loads the policy, and loading a large policy can briefly
    need most of a 32 GB machine's RAM (pi0.5 peaks at ~19 GB). So the constructor only starts the
    server long enough to read the episode list (`info`) and stops it again; the simulator proper
    starts on the first request, after the policy is loaded.
    """

    def __init__(self, *, so101_bench_root: Path, isaaclab_sh: Path, isaac_venv: Path | None,
                 episodes_jsonl: Path, layouts_jsonl: Path, trial_ids: list[int] | None, init_frames: bool,
                 max_joint_step_deg: float, initial_hold_time_s: float, seed: int, port: int, log_path: Path,
                 startup_timeout_s: float = 600.0, step_timeout_s: float = 300.0):
        self.port = port or _free_port()
        command = [str(isaaclab_sh), "-p", str(SERVER_SCRIPT), "--headless",
                   "--episodes_jsonl", str(episodes_jsonl), "--episode_layouts_jsonl", str(layouts_jsonl),
                   "--port", str(self.port), "--seed", str(seed),
                   "--max_joint_step_deg", str(max_joint_step_deg), "--initial_hold_time_s", str(initial_hold_time_s)]
        if trial_ids:
            command += ["--trial_ids", ",".join(map(str, trial_ids))]
        if init_frames:
            command.append("--init_frames")
        # isaaclab.sh picks the interpreter from the active environment; point it at the Isaac venv
        # and drop LeRobot's (Python 3.12) environment.
        env = {k: v for k, v in os.environ.items() if k not in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONHOME",
                                                                  "PYTHONPATH", "VIRTUAL_ENV")}
        if isaac_venv is not None:
            env["VIRTUAL_ENV"] = str(isaac_venv)
            env["PATH"] = f"{isaac_venv / 'bin'}:{env.get('PATH', '')}"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._command, self._env, self._cwd = command, env, so101_bench_root
        self._startup_timeout_s, self._step_timeout_s = startup_timeout_s, step_timeout_s
        self.log_path = log_path
        self.process = None
        atexit.register(self.close)
        self.info = self._start()
        logger.info("SO-101 Bench episodes: %d; stopping the simulator until the first episode", len(self.info["episodes"]))
        self.close()

    def _start(self) -> dict:
        """Start the server and return its `info` reply once it answers."""

        self._log = open(self.log_path, "ab")
        logger.info("Starting SO-101 Bench simulator (log: %s)", self.log_path)
        self.process = subprocess.Popen(self._command, cwd=self._cwd, env=self._env, stdout=self._log,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        self.socket = zmq.Context.instance().socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://127.0.0.1:{self.port}")
        deadline = time.monotonic() + self._startup_timeout_s
        self.socket.setsockopt(zmq.RCVTIMEO, 2000)
        self.socket.send(msgpack.packb({"cmd": "info"}, use_bin_type=True))
        while True:
            try:
                info = msgpack.unpackb(self.socket.recv(), raw=False)
                break
            except zmq.Again:
                if self.process.poll() is not None:
                    raise RuntimeError(f"SO-101 Bench simulator exited ({self.process.returncode}); see {self.log_path}")
                if time.monotonic() > deadline:
                    self.close()
                    raise TimeoutError(f"SO-101 Bench simulator did not start in {self._startup_timeout_s:.0f} s; "
                                       f"see {self.log_path}")
        self.socket.setsockopt(zmq.RCVTIMEO, int(self._step_timeout_s * 1000))
        logger.info("SO-101 Bench simulator ready")
        return info

    def request(self, payload: dict) -> dict:
        if self.process is None:
            self._start()
        elif self.process.poll() is not None:
            raise RuntimeError(f"SO-101 Bench simulator exited ({self.process.returncode}); see {self.log_path}")
        self.socket.send(msgpack.packb(payload, use_bin_type=True))
        try:
            reply = msgpack.unpackb(self.socket.recv(), raw=False)
        except zmq.Again as exc:
            raise TimeoutError(f"SO-101 Bench simulator stopped responding; see {self.log_path}") from exc
        if "error" in reply:
            raise RuntimeError(f"SO-101 Bench simulator: {reply['error']} (see {self.log_path})")
        return reply

    def decode_images(self, reply: dict) -> dict[str, np.ndarray]:
        shapes = {**self.info["cameras"], **{f"{k}_init": v for k, v in self.info["cameras"].items()}}
        return {name: np.frombuffer(data, dtype=np.uint8).reshape(*shapes[name], 3)
                for name, data in reply["images"].items()}

    def close(self) -> None:
        if getattr(self, "process", None) is None:
            return
        if self.process.poll() is not None:
            self.socket.close()
            self.process = None
            self._log.close()
            return
        try:
            self.socket.setsockopt(zmq.RCVTIMEO, 10000)
            self.socket.send(msgpack.packb({"cmd": "close"}, use_bin_type=True))
            self.socket.recv()
            self.process.wait(timeout=60)
        except Exception:
            pass
        if self.process.poll() is None:
            os.killpg(self.process.pid, 15)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, 9)
        self.socket.close()
        self.process = None
        self._log.close()
