"""gym.Env wrappers: one per SO-101 Bench episode, all sharing one simulator process."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from lerobot.envs.utils import freeze_after_episode_end

from .client import SimClient

# LeRobot SO-101 follower units: body joints in [-100, 100], gripper in [0, 100].
ACTION_LOW = np.array([-100, -100, -100, -100, -100, 0], dtype=np.float32)
ACTION_HIGH = np.array([100, 100, 100, 100, 100, 100], dtype=np.float32)


class SO101BenchEnv(gym.Env):
    """One fixed SO-101 Bench episode (objects, layout, instruction, appearance)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, client: SimClient, episode: dict, cameras: list[str]):
        super().__init__()
        self.client, self.episode, self.cameras = client, episode, cameras
        self.trial_id = int(episode["trial_id"])
        self.task = f"{episode['split']}/{episode['family']}/{self.trial_id}"
        self.task_description = episode["instruction"]
        self._max_episode_steps = int(episode["max_steps"])
        self.metadata = {**self.metadata, "render_fps": client.info["fps"]}
        shapes = client.info["cameras"]
        self.observation_space = spaces.Dict({
            "pixels": spaces.Dict({
                name: spaces.Box(0, 255, (*shapes[name.removesuffix("_init")], 3), dtype=np.uint8) for name in cameras
            }),
            "agent_pos": spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
        })
        self.action_space = spaces.Box(ACTION_LOW, ACTION_HIGH, dtype=np.float32)
        self._last_frame: np.ndarray | None = None

    def _format(self, reply: dict) -> dict:
        images = self.client.decode_images(reply)
        self._last_frame = images[self.cameras[0]]
        return {"pixels": {name: images[name] for name in self.cameras},
                "agent_pos": np.asarray(reply["state"], dtype=np.float32)}

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        reply = self.client.request({"cmd": "reset", "trial_id": self.trial_id})
        return self._format(reply), {"is_success": False}

    def step(self, action: np.ndarray):
        reply = self.client.request({"cmd": "step", "action": np.asarray(action, dtype=np.float32).tolist()})
        info = {"is_success": bool(reply["is_success"]), "end_reason": reply["end_reason"], "trial_id": self.trial_id}
        return self._format(reply), float(reply["reward"]), bool(reply["terminated"]), bool(reply["truncated"]), info

    def render(self) -> np.ndarray | None:
        return self._last_frame

    def close(self) -> None:
        # lerobot-eval closes each task's env after its episodes; the shared simulator stays up
        # until the process exits (SimClient registers its own shutdown).
        pass


def create_so101_bench_envs(client: SimClient, cameras: list[str]) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """{"<split>/<family>": {trial_id: VectorEnv}}: one suite per task-family x object-split cell.

    Each VectorEnv holds a single env: episodes share one simulator, so they run one at a time.
    FreezeAfterEpisodeEnd absorbs Gymnasium's autoreset, which would otherwise rebuild the scene.
    """

    out: dict[str, dict[int, gym.vector.VectorEnv]] = defaultdict(dict)
    for episode in client.info["episodes"]:
        suite = f"{episode['split']}/{episode['family']}"
        out[suite][int(episode["trial_id"])] = gym.vector.SyncVectorEnv(
            [freeze_after_episode_end(lambda e=episode: SO101BenchEnv(client, e, cameras))],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
    return dict(out)


def episode_summary(info: dict[str, Any]) -> dict[str, int]:
    """Episode counts per suite, for logging."""

    counts: dict[str, int] = defaultdict(int)
    for episode in info["episodes"]:
        counts[f"{episode['split']}/{episode['family']}"] += 1
    return dict(counts)
