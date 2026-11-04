"""``--env.type=so101_bench``: the SO-101 Bench config for lerobot-eval."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

logger = logging.getLogger(__name__)

# Named episode sets in the SO-101 Bench repo: (tasks, layouts).
EPISODE_SETS = {
    "val_set": ("tasks/val_set.jsonl", "tasks/layouts/val_set_layouts.jsonl"),
    "mmrv_eval": ("tasks/mmrv_eval.jsonl", "tasks/layouts/mmrv_eval_layouts.jsonl"),
}
IMAGE_SHAPE = (480, 640, 3)


def _parse_ids(spec: str | None) -> list[int] | None:
    """"0-9,20" -> [0, ..., 9, 20]."""

    if not spec:
        return None
    ids = []
    for part in str(spec).split(","):
        start, _, end = part.strip().partition("-")
        ids += list(range(int(start), int(end or start) + 1))
    return ids


@EnvConfig.register_subclass("so101_bench")
@dataclass
class SO101BenchEnvConfig(EnvConfig):
    """SO-101 Bench (Isaac Sim 5.1) through a local simulator process.

    `task` names an episode set ("val_set", the public validation set, or "mmrv_eval") or is a
    path to a tasks JSONL; `layouts` is then its layouts JSONL. Suites are task-family x
    object-split cells ("seen/bin_1", ...); task ids are the episodes' trial ids. Run one episode
    per task (`--eval.n_episodes=1`): each task is one fixed scene.
    """

    task: str = "val_set"
    layouts: str | None = None
    episodes: str | None = None  # subset of trial ids, e.g. "0-9,20"
    fps: int = 30
    # Camera names become observation.images.<name>; "overhead_init" / "wrist_init" are the
    # cameras' frames at robot start, held fixed for the episode.
    cameras: str = "overhead,wrist"
    so101_bench_root: str = field(default_factory=lambda: os.environ.get("SO101_BENCH_ROOT", str(Path.home() / "so101_bench")))
    isaaclab_sh: str = field(default_factory=lambda: os.environ.get("ISAACLAB_SH", str(Path.home() / "IsaacLab/isaaclab.sh")))
    isaac_venv: str | None = field(default_factory=lambda: os.environ.get("SO101_BENCH_ISAAC_VENV"))
    port: int = 0  # 0 picks a free port
    seed: int = 0
    max_joint_step_deg: float = 15.0
    initial_hold_time_s: float = 1.5
    sim_log: str | None = None  # default: outputs/lerobot_sim_server.log in so101_bench_root
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        }
    )
    features_map: dict[str, str] = field(default_factory=lambda: {ACTION: ACTION, "agent_pos": OBS_STATE})

    def __post_init__(self):
        for camera in self.camera_list:
            self.features[f"pixels/{camera}"] = PolicyFeature(type=FeatureType.VISUAL, shape=IMAGE_SHAPE)
            self.features_map.setdefault(f"pixels/{camera}", f"{OBS_IMAGES}.{camera}")

    @property
    def camera_list(self) -> list[str]:
        return [c.strip() for c in self.cameras.split(",") if c.strip()]

    @property
    def gym_kwargs(self) -> dict:
        return {}

    def episode_files(self) -> tuple[Path, Path]:
        root = Path(self.so101_bench_root).expanduser()
        if self.task in EPISODE_SETS:
            tasks, layouts = EPISODE_SETS[self.task]
            return root / tasks, root / (self.layouts or layouts)
        if not self.layouts:
            raise ValueError("A custom tasks JSONL needs --env.layouts=<layouts JSONL>.")
        return Path(self.task).expanduser().resolve(), Path(self.layouts).expanduser().resolve()

    def create_envs(self, n_envs: int, use_async_envs: bool = False):
        from .client import SimClient
        from .env import create_so101_bench_envs, episode_summary

        if n_envs != 1:
            logger.warning("SO-101 Bench runs episodes one at a time in a single simulator; "
                           "ignoring batch_size=%d.", n_envs)
        root = Path(self.so101_bench_root).expanduser()
        tasks, layouts = self.episode_files()
        unknown = [c for c in self.camera_list if c.removesuffix("_init") not in ("overhead", "wrist")]
        if unknown:
            raise ValueError(f"Unknown SO-101 Bench cameras {unknown}; use overhead, wrist, overhead_init, wrist_init.")
        client = SimClient(
            so101_bench_root=root,
            isaaclab_sh=Path(self.isaaclab_sh).expanduser(),
            isaac_venv=Path(self.isaac_venv).expanduser() if self.isaac_venv else None,
            episodes_jsonl=tasks,
            layouts_jsonl=layouts,
            trial_ids=_parse_ids(self.episodes),
            init_frames=any(c.endswith("_init") for c in self.camera_list),
            max_joint_step_deg=self.max_joint_step_deg,
            initial_hold_time_s=self.initial_hold_time_s,
            seed=self.seed,
            port=self.port,
            log_path=Path(self.sim_log) if self.sim_log else root / "outputs/lerobot_sim_server.log",
        )
        logger.info("SO-101 Bench suites: %s", episode_summary(client.info))
        return create_so101_bench_envs(client, self.camera_list)
