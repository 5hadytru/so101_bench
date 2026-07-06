"""LeRobot dataset recording helpers for SO-101 Bench.

This module bridges the Isaac Lab observation/action space and the LeRobot
dataset format used by the real SO-101 recordings, so that GR00T / MolmoAct2
rollouts and teleoperated episodes can be written into a LeRobot dataset that is
merge-compatible with ``5hadytru/so101_bench_real_sim_1``.

It exposes four symbols consumed by the eval / teleop scripts:

- :class:`SO101CalibrationMapper` -- sim USD joint radians <-> calibrated LeRobot
  ``.pos`` units (identical to the private mappers in ``groot.py`` and the replay
  scripts; kept here so recording code has a single import site).
- :func:`real_compatible_camera_sources` -- select the sim cameras that match the
  real-robot rig and assign each a LeRobot dataset video key.
- :func:`dataset_cameras` -- turn those sources into the camera spec passed to the
  recorder (and, in turn, into LeRobot ``observation.images.*`` video features).
- :func:`recording_images` -- pull the current RGB frame for each source out of the
  Isaac Lab visual observation as a ``uint8`` ``HxWx3`` array keyed by dataset key.
- :class:`LeRobotSimDatasetRecorder` -- a thin lifecycle wrapper around
  ``lerobot.datasets.LeRobotDataset`` (create/append, per-episode buffering,
  save/cancel, finalize).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from so101_bench.utils.lerobot_calibration import (
    LEROBOT_JOINT_FEATURE_ORDER,
    LEROBOT_JOINT_ORDER,
    REAL_SO101_CALIBRATION,
    SIM_LIMIT_MARGIN_DEG,
    STS3215_CENTER_POSITION,
    STS3215_DEGREES_PER_TICK,
    USD_SIM_JOINT_LIMITS_DEG,
    lerobot_position_bounds,
)

# Sim cameras that exist on the real SO-101 rig, in the order they should be
# written, mapped to the LeRobot dataset video key (``observation.images.<key>``).
# The overhead key is verified against the shipped dataset
# (``observation.images.overhead``); edit this table if your target real dataset
# names the wrist stream differently (e.g. ``wrist`` -> ``front``).
REAL_COMPATIBLE_CAMERA_KEYS: dict[str, str] = {
    "overhead": "overhead",
    "wrist": "wrist",
}


class SO101CalibrationMapper:
    """Convert between SO-101 USD joint radians and calibrated LeRobot ``.pos`` units."""

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
        sim_values = sim_values.to(device=self.device, dtype=torch.float32)
        mapped_deg = sim_values * 180.0 / torch.pi
        mapped_deg = torch.minimum(torch.maximum(mapped_deg, self.usd_mins_deg), self.usd_maxs_deg)

        motor_positions = mapped_deg / STS3215_DEGREES_PER_TICK + STS3215_CENTER_POSITION
        body_normalized = (motor_positions - self.calibration_mins) / (self.calibration_maxs - self.calibration_mins)
        body_positions = body_normalized * 200.0 - 100.0

        gripper_normalized = (mapped_deg - self.usd_mins_deg) / (self.usd_maxs_deg - self.usd_mins_deg)
        gripper_positions = gripper_normalized * 100.0

        lerobot_positions = torch.where(self.is_gripper, gripper_positions, body_positions)
        return self.clamp_lerobot_positions(lerobot_positions)

    def lerobot_positions_to_sim_radians(self, lerobot_positions: torch.Tensor) -> torch.Tensor:
        lerobot_positions = lerobot_positions.to(device=self.device, dtype=torch.float32)
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
class CameraSource:
    """One camera to record: its sim name, the LeRobot dataset key, and frame size."""

    name: str  # sim camera name; the visual observation key is ``rgb_{name}``
    key: str  # dataset video key -> ``observation.images.{key}``
    height: int
    width: int


def real_compatible_camera_sources(cameras: dict[str, dict[str, int]]) -> list[CameraSource]:
    """Select the discovered sim cameras that match the real SO-101 rig.

    ``cameras`` is the ``{name: {"height": H, "width": W}}`` mapping produced by the
    eval/teleop ``_discover_cameras`` helper. Sim-only cameras that do not appear in
    :data:`REAL_COMPATIBLE_CAMERA_KEYS` are dropped so the recorded dataset stays
    merge-compatible with the real recordings.
    """

    sources: list[CameraSource] = []
    for name, key in REAL_COMPATIBLE_CAMERA_KEYS.items():
        spec = cameras.get(name)
        if spec is None:
            continue
        sources.append(CameraSource(name=name, key=key, height=int(spec["height"]), width=int(spec["width"])))
    if not sources:
        raise ValueError(
            "No real-compatible cameras were found. Expected one of "
            f"{sorted(REAL_COMPATIBLE_CAMERA_KEYS)}, got {sorted(cameras)}."
        )
    return sources


def dataset_cameras(
    cameras: dict[str, dict[str, int]],
    sources: list[CameraSource] | None = None,
) -> list[CameraSource]:
    """Return the camera sources used to build LeRobot ``observation.images.*`` features."""

    if sources is None:
        sources = real_compatible_camera_sources(cameras)
    return list(sources)


def _to_uint8_rgb(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Coerce a sim RGB frame (tensor or array, CHW/HWC, float or int) to ``HxWx3`` uint8."""

    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)

    if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Expected an RGB image tensor/array, got shape {array.shape}.")

    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"Expected image channel dimension of 3 or 4, got shape {array.shape}.")

    if np.issubdtype(array.dtype, np.floating):
        if array.max(initial=0.0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0)
    return np.ascontiguousarray(array.astype(np.uint8))


def _visual_frame(visual_obs: dict, camera: str, env_index: int = 0) -> np.ndarray:
    key = f"rgb_{camera}"
    if key not in visual_obs:
        raise KeyError(f"Expected visual observation key {key!r}; got: {list(visual_obs.keys())}")
    image = visual_obs[key]
    if isinstance(image, torch.Tensor):
        if image.ndim >= 4:
            image = image[env_index]
    else:
        image = np.asarray(image)
        if image.ndim >= 4:
            image = image[env_index]
    return _to_uint8_rgb(image)


def recording_images(visual_obs: dict, sources: list[CameraSource]) -> dict[str, np.ndarray]:
    """Return ``{dataset_key: HxWx3 uint8}`` frames for each recorded camera source."""

    return {source.key: _visual_frame(visual_obs, source.name) for source in sources}


def _joint_feature() -> dict[str, Any]:
    return {
        "dtype": "float32",
        "shape": (len(LEROBOT_JOINT_FEATURE_ORDER),),
        "names": list(LEROBOT_JOINT_FEATURE_ORDER),
    }


def _as_feature_vector(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.shape != (len(LEROBOT_JOINT_FEATURE_ORDER),):
        raise ValueError(
            f"Expected a {len(LEROBOT_JOINT_FEATURE_ORDER)}-DOF joint vector, got shape {array.shape}."
        )
    return array


class LeRobotSimDatasetRecorder:
    """Buffer and write SO-101 sim rollouts as a LeRobot dataset.

    Lifecycle: construct -> :meth:`init_dataset` (create fresh, or append to an
    existing dataset at ``dataset_root``) -> per episode
    :meth:`start_episode` / N x :meth:`push_frame` / :meth:`stop_episode` (or
    :meth:`cancel_episode`) -> :meth:`finalize` at shutdown.
    """

    def __init__(
        self,
        *,
        repo_id: str,
        dataset_root: str | Path,
        fps: int,
        cameras: list[CameraSource],
        streaming_encoding: bool = True,
        vcodec: str = "libsvtav1",
        encoder_queue_size: int = 300,
        encoder_threads: int | None = None,
        image_writer_processes: int = 0,
        image_writer_threads_per_camera: int = 4,
        video_files_size_mb: int = 200,
    ):
        self.repo_id = repo_id
        self.dataset_root = Path(dataset_root)
        self.fps = int(fps)
        self.cameras = list(cameras)
        self.streaming_encoding = streaming_encoding
        self.vcodec = vcodec
        self.encoder_queue_size = encoder_queue_size
        self.encoder_threads = encoder_threads
        self.image_writer_processes = image_writer_processes
        self.image_writer_threads_per_camera = image_writer_threads_per_camera
        self.video_files_size_mb = video_files_size_mb

        self.dataset: Any = None
        self._recording = False
        self._current_task: str | None = None

    @property
    def recording(self) -> bool:
        return self._recording

    def _features(self) -> dict[str, Any]:
        features: dict[str, Any] = {
            "action": _joint_feature(),
            "observation.state": _joint_feature(),
        }
        for source in self.cameras:
            features[f"observation.images.{source.key}"] = {
                "dtype": "video",
                "shape": (source.height, source.width, 3),
                "names": ["height", "width", "channels"],
            }
        return features

    def init_dataset(self) -> None:
        """Create the dataset (or reopen an existing one at ``dataset_root``)."""

        if self.dataset is not None:
            return
        try:
            from lerobot.datasets import LeRobotDataset
        except ImportError:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

        meta_dir = self.dataset_root / "meta"
        image_writer_threads = self.image_writer_threads_per_camera * max(1, len(self.cameras))
        if meta_dir.exists():
            self.dataset = LeRobotDataset(self.repo_id, root=self.dataset_root)
            start_image_writer = getattr(self.dataset, "start_image_writer", None)
            if callable(start_image_writer):
                start_image_writer(
                    num_processes=self.image_writer_processes,
                    num_threads=image_writer_threads,
                )
            return

        create_kwargs = {
            "repo_id": self.repo_id,
            "fps": self.fps,
            "features": self._features(),
            "root": self.dataset_root,
            "robot_type": "so101",
            "use_videos": True,
            "image_writer_processes": self.image_writer_processes,
            "image_writer_threads": image_writer_threads,
            "batch_encoding_size": 1,
        }
        # Only forward kwargs the installed LeRobot understands (its ``create``
        # signature has shifted across releases).
        supported = set(inspect.signature(LeRobotDataset.create).parameters)
        create_kwargs = {key: value for key, value in create_kwargs.items() if key in supported}
        self.dataset = LeRobotDataset.create(**create_kwargs)

    def start_episode(self, task: str) -> None:
        if self.dataset is None:
            self.init_dataset()
        # Drop any buffer left over from a cancelled episode so the next episode
        # index is reused rather than skipped.
        self.dataset.episode_buffer = None
        self._current_task = task
        self._recording = True

    def push_frame(
        self,
        *,
        action: torch.Tensor | np.ndarray,
        observation_state: torch.Tensor | np.ndarray,
        images: dict[str, np.ndarray],
    ) -> None:
        if not self._recording or self.dataset is None:
            raise RuntimeError("push_frame() called before start_episode().")
        frame: dict[str, Any] = {
            "action": _as_feature_vector(action),
            "observation.state": _as_feature_vector(observation_state),
            "task": self._current_task or "",
        }
        for source in self.cameras:
            if source.key not in images:
                raise KeyError(
                    f"Missing recorded image for camera {source.key!r}; got keys {sorted(images)}."
                )
            frame[f"observation.images.{source.key}"] = np.ascontiguousarray(images[source.key], dtype=np.uint8)
        self.dataset.add_frame(frame)

    def _buffered_frame_count(self) -> int:
        buffer = getattr(self.dataset, "episode_buffer", None)
        if not buffer:
            return 0
        return int(buffer.get("size", 0))

    def stop_episode(self, task: str | None = None) -> int:
        """Save the buffered episode. Returns 1 if an episode was written, else 0."""

        if not self._recording or self.dataset is None:
            return 0
        self._recording = False
        if task is not None:
            self._current_task = task
        if self._buffered_frame_count() <= 0:
            self.cancel_episode()
            return 0
        self.dataset.save_episode()
        return 1

    def cancel_episode(self) -> None:
        self._recording = False
        if self.dataset is None:
            return
        # Nothing was buffered (empty episode); LeRobot's clear_episode_buffer
        # assumes a live buffer, so only invoke it when one exists.
        if getattr(self.dataset, "episode_buffer", None) is None:
            return
        clear = getattr(self.dataset, "clear_episode_buffer", None)
        if callable(clear):
            clear()
        else:
            self.dataset.episode_buffer = None

    def finalize(self) -> None:
        if self.dataset is None:
            return
        if self._recording:
            self.stop_episode()
        stop_image_writer = getattr(self.dataset, "stop_image_writer", None)
        if callable(stop_image_writer):
            stop_image_writer()
        finalize = getattr(self.dataset, "finalize", None)
        if callable(finalize):
            finalize()
