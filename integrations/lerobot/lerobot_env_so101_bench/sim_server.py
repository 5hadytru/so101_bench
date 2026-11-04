"""SO-101 Bench simulator server for the LeRobot plugin. Runs inside Isaac Lab (Python 3.11).

LeRobot needs Python >= 3.12 and Isaac Sim 5.1 is Python 3.11, so the gym.Env in
``lerobot_env_so101_bench.env`` drives this process over a local ZMQ REQ/REP socket
(msgpack). The plugin starts it; to run it by hand:

    ~/IsaacLab/isaaclab.sh -p integrations/lerobot/lerobot_env_so101_bench/sim_server.py --headless \
        --episodes_jsonl tasks/val_set.jsonl --episode_layouts_jsonl tasks/layouts/val_set_layouts.jsonl \
        --port 5600

The episode loop matches the leaderboard evaluator (scripts/molmoact2_eval.py): one Isaac env with
every object of the episode set pre-spawned, reconfigured per episode; a 1.5 s hold of the initial
pose before the first observation is returned; joint targets clamped to 15 deg per control tick.
State and actions are in LeRobot SO-101 follower units (body joints in [-100, 100], gripper in
[0, 100]), the convention of the so101_follower robot and of LeRobot SO-101 datasets.

Requests -> replies:
    {"cmd": "info"}                    -> {"episodes": [...], "fps", "cameras": {name: [h, w]}}
    {"cmd": "reset", "trial_id": int}  -> observation
    {"cmd": "step", "action": [6]}     -> observation + reward, terminated, truncated, is_success, end_reason
    {"cmd": "close"}                   -> {}
An observation is {"state": [6], "images": {name: bytes}}: HWC uint8 RGB at the camera's [h, w],
plus "<camera>_init" frames (the cameras at robot start) when --init_frames is set.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO-101 Bench simulator server for lerobot-eval.")
parser.add_argument("--episodes_jsonl", type=Path, required=True)
parser.add_argument("--episode_layouts_jsonl", type=Path, required=True)
parser.add_argument("--trial_ids", type=str, default="", help="Comma-separated subset of trial ids; default all.")
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--task", type=str, default="So101Bench-Bin-v0", help="Isaac Lab task name.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--initial_hold_time_s", type=float, default=1.5)
parser.add_argument("--max_joint_step_deg", type=float, default=15.0, help="0 disables the clamp.")
parser.add_argument("--init_frames", action="store_true", help="Also return <camera>_init frames.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# The headless camera kit fails to produce LdrColorSD on our setups; the GUI rendering kit renders
# the same cameras fine (same workaround as the leaderboard evaluators).
if args_cli.headless and not getattr(args_cli, "experience", ""):
    args_cli.experience = "isaaclab.python.rendering.kit"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import msgpack
import numpy as np
import torch
import zmq

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import so101_bench.tasks  # noqa: F401
from so101_bench.benchmark import BenchmarkEpisodeSpec, episode_is_executable, episode_length_s, load_episode_jsonl
from so101_bench.layouts import normalize_layout_object_slots
from so101_bench.mdp import mark_benchmark_robot_start
from so101_bench.tasks.direct.so101_bench.so101_bench_env_cfg import configure_env_cfg_for_object_pool
from so101_bench.utils.lerobot_calibration import LEROBOT_INITIAL_JOINT_POS, lerobot_pose_to_sim_joint_pos
from so101_bench.utils.molmoact2 import SO101LeRobotUnitsMapper

ACTION_JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
INITIAL_ROBOT_JOINT_POS = lerobot_pose_to_sim_joint_pos(LEROBOT_INITIAL_JOINT_POS)


def _load_layouts(episodes: list[BenchmarkEpisodeSpec], path: Path) -> list[dict]:
    by_trial = {}
    for line in path.read_text().splitlines():
        if line.strip():
            layout = json.loads(line)
            by_trial[layout["trial_id"]] = layout
    layouts = []
    for episode in episodes:
        layout = by_trial[episode.metadata["trial_id"]]
        if layout.get("instruction", episode.instruction) != episode.instruction:
            raise ValueError(f"Layout/task mismatch for trial {episode.metadata['trial_id']}")
        layouts.append(normalize_layout_object_slots(layout, episode.objects))
    return layouts


def _object_pool(episodes: list[BenchmarkEpisodeSpec]) -> list[str]:
    pool = []
    for episode in episodes:
        for name in episode.objects:
            if name not in pool:
                pool.append(name)
    return pool


def _reset_params(episode, layout, pool: list[str], asset_names: list[str]) -> dict:
    index = {name: i for i, name in enumerate(pool)}
    payload = episode.reset_payload()
    local_to_pool = [index[name] for name in episode.objects]
    payload["active_object_ids"] = local_to_pool
    payload["target_object_id"] = local_to_pool[episode.target_object_id]
    payload["referent_object_ids"] = [local_to_pool[i] for i in episode.referent_object_ids]
    pool_layout = dict(layout)
    pool_layout["objects"] = []
    for entry in layout["objects"]:
        entry = dict(entry)
        slot = index[str(entry.get("name") or episode.objects[int(entry["slot"])])]
        entry["slot"], entry["asset_name"] = slot, f"object_{slot + 1}"
        pool_layout["objects"].append(entry)
    return {
        "object_asset_names": asset_names,
        "object_labels": pool,
        "task_family": episode.task_family,
        "object_count_range": (len(episode.objects), len(episode.objects)),
        "active_object_selection": "fixed",
        "fixed_active_object_ids": tuple(payload["active_object_ids"]),
        "shuffle_object_labels": False,
        "force_bin_all_objects_instruction": False,
        "episode_spec": payload,
        "episode_layout": pool_layout,
    }


class Sim:
    def __init__(self):
        episodes = [e for e in load_episode_jsonl(args_cli.episodes_jsonl) if episode_is_executable(e)]
        if args_cli.trial_ids:
            wanted = {int(t) for t in args_cli.trial_ids.split(",")}
            episodes = [e for e in episodes if e.metadata["trial_id"] in wanted]
        self.episodes = {e.metadata["trial_id"]: e for e in episodes}
        self.layouts = dict(zip(self.episodes, _load_layouts(episodes, args_cli.episode_layouts_jsonl)))
        self.pool = _object_pool(episodes)

        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.seed = args_cli.seed
        env_cfg.scene.robot.init_state.joint_pos = dict(INITIAL_ROBOT_JOINT_POS)
        self.asset_names = configure_env_cfg_for_object_pool(env_cfg, self.pool)
        first = episodes[0]
        env_cfg.events.reset_benchmark_scene.params.update(
            _reset_params(first, self.layouts[first.metadata["trial_id"]], self.pool, self.asset_names))
        self.env = gym.make(args_cli.task, cfg=env_cfg)
        unwrapped = self.env.unwrapped
        self.control_dt = float(unwrapped.step_dt)
        self.hold_steps = max(0, math.ceil(args_cli.initial_hold_time_s / self.control_dt))
        self.cameras = {
            key.removeprefix("camera_"): [getattr(unwrapped.scene.cfg, key).height, getattr(unwrapped.scene.cfg, key).width]
            for key in unwrapped.scene.keys() if key.startswith("camera_")
        }
        self.mapper = SO101LeRobotUnitsMapper(device=unwrapped.device)
        self.max_step_rad = math.radians(args_cli.max_joint_step_deg)
        self.hold = torch.tensor([INITIAL_ROBOT_JOINT_POS[j] for j in ACTION_JOINT_NAMES],
                                 dtype=torch.float32, device=unwrapped.device)
        self.obs = None
        self.init_images: dict[str, bytes] = {}
        self.done = True
        self.last_reply: dict = {}
        self._trial = -1

    def info(self) -> dict:
        episodes = []
        for trial_id, e in self.episodes.items():
            episodes.append({
                "trial_id": trial_id, "instruction": e.instruction, "split": e.metadata.get("ood_key", ""),
                "family": e.task_family if e.task_family != "bin" else f"bin_{len(e.objects)}",
                "objects": list(e.objects), "appearance_profile": e.metadata.get("appearance_profile", ""),
                # The env's own timeout (from robot start, plus any success-confirmation extension)
                # ends the episode; LeRobot's step cap only needs to stay above it.
                "max_steps": math.ceil((episode_length_s(e.task_family, len(e.objects)) + 10.0) / self.control_dt),
            })
        return {"episodes": episodes, "fps": round(1.0 / self.control_dt), "cameras": self.cameras,
                "init_frames": args_cli.init_frames}

    def _images(self) -> dict[str, bytes]:
        visual = self.obs["visual"]
        images = {}
        for name in self.cameras:
            image = visual[f"rgb_{name}"][0]
            if image.dtype != torch.uint8:
                image = (image.clamp(0, 1) * 255 if image.max() <= 1.0 else image.clamp(0, 255)).to(torch.uint8)
            images[name] = np.ascontiguousarray(image[..., :3].cpu().numpy()).tobytes()
        return images

    def _observation(self) -> dict:
        state = self.mapper.sim_radians_to_model_positions(self.obs["policy"]["joint_pos_obs"][0])
        return {"state": state.cpu().numpy().astype(np.float32).tolist(), "images": {**self._images(), **self.init_images}}

    def reset(self, trial_id: int) -> dict:
        episode = self.episodes[trial_id]
        unwrapped = self.env.unwrapped
        params = _reset_params(episode, self.layouts[trial_id], self.pool, self.asset_names)
        unwrapped.cfg.events.reset_benchmark_scene.params.update(params)
        unwrapped.event_manager.get_term_cfg("reset_benchmark_scene").params.update(params)
        with torch.inference_mode(False):
            self.env.reset()
            robot = unwrapped.scene["robot"]
            joint_ids = [robot.joint_names.index(j) for j in ACTION_JOINT_NAMES]
            joint_pos = self.hold.unsqueeze(0)
            robot.data.default_joint_pos[:, joint_ids] = joint_pos
            robot.data.default_joint_vel[:, joint_ids] = torch.zeros_like(joint_pos)
            robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), joint_ids=joint_ids)
            robot.set_joint_position_target(joint_pos, joint_ids=joint_ids)
            robot.write_data_to_sim()
            unwrapped.scene.write_data_to_sim()
            unwrapped.sim.forward()
            if unwrapped.sim.has_rtx_sensors():
                for _ in range(getattr(unwrapped.cfg, "num_rerenders_on_reset", 0)):
                    unwrapped.sim.render()
            self.obs = unwrapped.observation_manager.compute(update_history=True)
            unwrapped.obs_buf = self.obs
        for _ in range(self.hold_steps):
            self.obs, *_ = self.env.step(self.hold.unsqueeze(0))
        mark_benchmark_robot_start(unwrapped, object_asset_names=self.asset_names, bin_name="plastic_bin",
                                   force_robot_start_time=True)
        self.init_images = {}
        if args_cli.init_frames:
            self.init_images = {f"{name}_init": image for name, image in self._images().items()}
        self.done = False
        return self._observation()

    def step(self, action: list[float]) -> dict:
        if self.done:
            return self.last_reply
        unwrapped = self.env.unwrapped
        current = self.obs["policy"]["joint_pos_obs"][0]
        target = self.mapper.model_positions_to_sim_radians(
            torch.as_tensor(action, dtype=torch.float32, device=unwrapped.device))
        delta = target - current
        biggest = torch.max(torch.abs(delta))
        if self.max_step_rad > 0 and biggest > self.max_step_rad:
            target = current + delta * (self.max_step_rad / biggest)
        reply = self._observation()  # returned at episode end: env.step() auto-resets the scene
        obs, reward, terminated, truncated, info = self.env.step(target.unsqueeze(0))
        terminated, truncated = bool(terminated.any()), bool(truncated.any())
        if not (terminated or truncated):
            self.obs = obs
            return {**self._observation(), "reward": float(reward[0]), "terminated": False, "truncated": False,
                    "is_success": False, "end_reason": ""}
        log = info.get("log", {})
        is_success = bool(log.get("Episode_Termination/success", 0.0) > 0.0)
        reasons = getattr(unwrapped, "_so101_failure_reasons", None) or ["none"]
        end_reason = "success" if is_success else (reasons[0] if reasons[0] != "none" else
                                                    "time_out" if truncated else "failure")
        self.done = True
        self.last_reply = {**reply, "reward": float(is_success), "terminated": terminated, "truncated": truncated,
                           "is_success": is_success, "end_reason": end_reason}
        print(f"[INFO]: trial {self._trial}: success={is_success}, reason={end_reason}", flush=True)
        return self.last_reply


def main():
    sim = Sim()
    socket = zmq.Context.instance().socket(zmq.REP)
    socket.bind(f"tcp://127.0.0.1:{args_cli.port}")
    print(f"[INFO]: SO-101 Bench sim server ready on port {args_cli.port} ({len(sim.episodes)} episodes)", flush=True)
    while simulation_app.is_running():
        request = msgpack.unpackb(socket.recv(), raw=False)
        cmd = request["cmd"]
        try:
            if cmd == "info":
                reply = sim.info()
            elif cmd == "reset":
                sim._trial = int(request["trial_id"])
                reply = sim.reset(sim._trial)
            elif cmd == "step":
                # Not under inference_mode (as in the leaderboard evaluator): env.step() resets the
                # scene at episode end, and buffers written there must stay writable for reset().
                reply = sim.step(request["action"])
            elif cmd == "close":
                socket.send(msgpack.packb({}, use_bin_type=True))
                break
            else:
                raise ValueError(f"unknown command {cmd!r}")
        except Exception as exc:  # report to the client instead of hanging it
            import traceback
            traceback.print_exc()
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        socket.send(msgpack.packb(reply, use_bin_type=True))
    sim.env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
