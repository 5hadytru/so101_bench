"""LeRobot dataset recording helpers for SO-101 sim episodes."""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

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

OVERHEAD_CAMERA_NAME = "overhead"
REAL_COMPAT_CAMERA_SOURCES = {
    "front": ("wrist", "front"),
    "overhead": (OVERHEAD_CAMERA_NAME,),
}


def _encode_video_frames_in_clean_subprocess(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = None,
    overwrite: bool = False,
    preset: int | None = None,
    encoder_threads: int | None = None,
) -> None:
    """Encode outside Isaac so PyAV/SVT never runs in or forks the Kit process."""

    worker_path = Path(__file__).with_name("lerobot_video_encode_worker.py")
    request = {
        "imgs_dir": str(imgs_dir),
        "video_path": str(video_path),
        "fps": int(fps),
        "vcodec": vcodec,
        "pix_fmt": pix_fmt,
        "g": g,
        "crf": crf,
        "fast_decode": fast_decode,
        "overwrite": overwrite,
        "preset": preset,
        "encoder_threads": encoder_threads,
    }
    print(
        f"[INFO]: Encoding {Path(imgs_dir).name} to {Path(video_path).name} "
        f"in a clean {vcodec} subprocess...",
        flush=True,
    )
    process = subprocess.Popen(
        [sys.executable, str(worker_path), json.dumps(request)],
        start_new_session=True,
    )
    try:
        return_code = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    if return_code != 0:
        raise RuntimeError(
            f"Video encoder subprocess exited with status {return_code}: {video_path}"
        )


def _encode_video_worker_in_clean_subprocess(
    video_key: str,
    episode_index: int,
    root,
    fps: int,
    vcodec: str = "libsvtav1",
    encoder_threads: int | None = None,
):
    """Drop-in for LeRobot's `_encode_video_worker` that never forks the Kit process.

    LeRobot runs its worker inside a ProcessPoolExecutor; on Linux those workers fork
    Isaac's already-threaded CUDA/Kit process and can deadlock. We keep the parallel
    fan-out (see `_ThreadedProcessPoolShim`) but do the encode in a fresh interpreter.
    """

    import tempfile

    from lerobot.datasets.utils import DEFAULT_IMAGE_PATH

    root = Path(root)
    temp_path = Path(tempfile.mkdtemp(dir=root)) / f"{video_key}_{episode_index:03d}.mp4"
    fpath = DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=episode_index, frame_index=0)
    img_dir = (root / fpath).parent
    _encode_video_frames_in_clean_subprocess(
        img_dir,
        temp_path,
        fps,
        vcodec=vcodec,
        overwrite=True,
        encoder_threads=encoder_threads,
    )
    shutil.rmtree(img_dir)
    return temp_path


class _ThreadedProcessPoolShim:
    """Namespace that makes LeRobot's `concurrent.futures.ProcessPoolExecutor` a thread pool.

    Each thread only waits on a `subprocess.Popen`, so cameras encode concurrently
    without forking Isaac.
    """

    class futures:  # noqa: N801 - mirrors `concurrent.futures` attribute access
        ProcessPoolExecutor = concurrent.futures.ThreadPoolExecutor
        as_completed = staticmethod(concurrent.futures.as_completed)


class SO101CalibrationMapper:
    """Convert between SO-101 USD radians and calibrated LeRobot `.pos` space."""

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

    def sim_radians_to_lerobot_positions(self, sim_values: torch.Tensor) -> torch.Tensor:
        mapped_deg = sim_values * 180.0 / torch.pi
        mapped_deg = torch.minimum(torch.maximum(mapped_deg, self.usd_mins_deg), self.usd_maxs_deg)

        motor_positions = mapped_deg / STS3215_DEGREES_PER_TICK + STS3215_CENTER_POSITION
        body_normalized = (motor_positions - self.calibration_mins) / (
            self.calibration_maxs - self.calibration_mins
        )
        body_positions = body_normalized * 200.0 - 100.0

        gripper_normalized = (mapped_deg - self.usd_mins_deg) / (self.usd_maxs_deg - self.usd_mins_deg)
        gripper_positions = gripper_normalized * 100.0

        lerobot_positions = torch.where(self.is_gripper, gripper_positions, body_positions)
        return torch.minimum(torch.maximum(lerobot_positions, self.lerobot_mins), self.lerobot_maxs)

    def lerobot_positions_to_sim_radians(self, lerobot_positions: torch.Tensor) -> torch.Tensor:
        bounded_positions = torch.minimum(torch.maximum(lerobot_positions, self.lerobot_mins), self.lerobot_maxs)
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


class LeRobotSimDatasetRecorder:
    """Small LeRobot dataset writer for SO-101 sim episodes."""

    def __init__(
        self,
        *,
        repo_id: str,
        dataset_root: Path,
        fps: int,
        cameras: dict[str, dict[str, int]],
        streaming_encoding: bool,
        vcodec: str,
        encoder_queue_size: int,
        encoder_threads: int | None,
        parallel_encoding: bool = True,
        image_writer_processes: int,
        image_writer_threads_per_camera: int,
        video_files_size_mb: int,
        preloaded_metadata=None,
        timing_callback: Callable[[str, float], None] | None = None,
    ):
        self.repo_id = repo_id
        self.dataset_root = dataset_root
        self.fps = fps
        self.cameras = cameras
        self.streaming_encoding = streaming_encoding
        self.vcodec = vcodec
        self.encoder_queue_size = encoder_queue_size
        self.encoder_threads = encoder_threads
        self.parallel_encoding = parallel_encoding
        self.image_writer_processes = image_writer_processes
        self.image_writer_threads = max(0, image_writer_threads_per_camera) * len(cameras)
        self.video_files_size_mb = video_files_size_mb
        self.preloaded_metadata = preloaded_metadata
        self.timing_callback = timing_callback
        self.dataset = None
        self.recording = False
        self._frame_count = 0
        self._task: str | None = None
        self._resolved_vcodec: str | None = None
        self.num_saved_episodes = 0

    def _features(self) -> dict[str, dict[str, Any]]:
        joint_names = list(LEROBOT_JOINT_FEATURE_ORDER)
        features: dict[str, dict[str, Any]] = {
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": joint_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": joint_names,
            },
        }
        for camera_name, camera in self.cameras.items():
            features[f"observation.images.{camera_name}"] = {
                "dtype": "video",
                "shape": (camera["height"], camera["width"], 3),
                "names": ["height", "width", "channels"],
            }
        return features

    def _metadata_ready(self) -> bool:
        tasks_path = self.dataset_root / "meta" / "tasks.parquet"
        episodes_dir = self.dataset_root / "meta" / "episodes"
        return tasks_path.exists() and any(episodes_dir.glob("*/*.parquet"))

    def _clear_incomplete_empty_root(self) -> bool:
        if not self.dataset_root.exists():
            return False

        allowed_partial_files = {self.dataset_root / "meta" / "info.json"}
        files = [path for path in self.dataset_root.rglob("*") if path.is_file()]
        unexpected_files = [path for path in files if path not in allowed_partial_files]
        if unexpected_files:
            paths = ", ".join(str(path) for path in unexpected_files[:5])
            suffix = "" if len(unexpected_files) <= 5 else ", ..."
            raise ValueError(
                f"LeRobot dataset root {self.dataset_root} exists but is missing complete metadata. "
                f"Found non-empty dataset files ({paths}{suffix}); use a new --repo_root or repair the dataset."
            )

        shutil.rmtree(self.dataset_root)
        return True

    def _resolve_fast_vcodec(self) -> str:
        if self._resolved_vcodec is not None:
            return self._resolved_vcodec
        if self.vcodec != "auto":
            self._resolved_vcodec = self.vcodec
            return self._resolved_vcodec
        try:
            from lerobot.datasets.video_utils import HW_ENCODERS, detect_available_hw_encoders

            available = set(detect_available_hw_encoders())
            for encoder in HW_ENCODERS:
                if encoder in available:
                    self._resolved_vcodec = encoder
                    return self._resolved_vcodec
        except Exception as exc:  # pragma: no cover - codec probing depends on host ffmpeg
            print(f"[WARN]: Could not probe hardware video encoders ({exc}); using h264.")
        self._resolved_vcodec = "h264"
        return self._resolved_vcodec

    def _dataset_kwargs(self) -> dict[str, Any]:
        return {
            # Avoid LeRobot's automatic torchcodec probe.  Inside Isaac/Kit the
            # probe can deadlock against already-loaded CUDA/multimedia
            # libraries while reopening a dataset.  Recording does not decode
            # historical video; pyav is only a safe fallback if reads occur.
            "video_backend": "pyav",
            "batch_encoding_size": 1,
            "vcodec": self._resolve_fast_vcodec(),
            "streaming_encoding": self.streaming_encoding,
            "encoder_queue_maxsize": self.encoder_queue_size,
            "encoder_threads": self.encoder_threads,
        }

    def _configure_recording_helpers(self) -> None:
        if self.dataset is None:
            return
        if self.video_files_size_mb > 0 and hasattr(self.dataset.meta, "update_chunk_settings"):
            self.dataset.meta.update_chunk_settings(video_files_size_in_mb=self.video_files_size_mb)
        if not self.streaming_encoding and self.image_writer_threads > 0:
            self.dataset.start_image_writer(
                num_processes=self.image_writer_processes,
                num_threads=self.image_writer_threads,
            )

    def _print_recording_config(self) -> None:
        mode = "streaming video" if self.streaming_encoding else "async image writer"
        print(
            "[INFO]: LeRobot recording mode: "
            f"{mode}, vcodec={self._resolve_fast_vcodec()}, "
            f"encoder_threads={self.encoder_threads}, parallel_encoding={self.parallel_encoding}, "
            f"queue={self.encoder_queue_size}, "
            f"video_rollover_mb={self.video_files_size_mb}"
        )

    def _sync_saved_episode_count(self) -> None:
        if self.dataset is None:
            self.num_saved_episodes = 0
            return
        meta = getattr(self.dataset, "meta", None)
        self.num_saved_episodes = int(getattr(meta, "total_episodes", 0) or 0)

    def _open_append_only_dataset(self):
        """Resume recording without loading historical frame parquet into HF Datasets."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.datasets.video_utils import StreamingVideoEncoder

        dataset = LeRobotDataset.__new__(LeRobotDataset)
        torch.utils.data.Dataset.__init__(dataset)
        dataset.repo_id = self.repo_id
        dataset.root = self.dataset_root
        dataset.revision = None
        dataset.tolerance_s = 1e-4
        dataset.image_writer = None
        dataset.batch_encoding_size = 1
        dataset.episodes_since_last_encoding = 0
        dataset.vcodec = self._resolve_fast_vcodec()
        dataset._encoder_threads = self.encoder_threads
        dataset.meta = self.preloaded_metadata
        if dataset.meta is None:
            dataset.meta = LeRobotDatasetMetadata(self.repo_id, self.dataset_root)
        dataset.episodes = None
        dataset.hf_dataset = None
        dataset.image_transforms = None
        dataset.delta_timestamps = None
        dataset.delta_indices = None
        dataset._absolute_to_relative_idx = None
        dataset.video_backend = "pyav"
        dataset.writer = None
        dataset.latest_episode = None
        dataset._current_file_start_frame = None
        dataset._lazy_loading = True
        dataset._recorded_frames = dataset.meta.total_frames
        dataset._writer_closed_for_reading = False
        dataset.episode_buffer = dataset.create_episode_buffer()
        if self.streaming_encoding and dataset.meta.video_keys:
            dataset._streaming_encoder = StreamingVideoEncoder(
                fps=dataset.meta.fps,
                vcodec=dataset.vcodec,
                pix_fmt="yuv420p",
                g=2,
                crf=30,
                preset=None,
                queue_maxsize=self.encoder_queue_size,
                encoder_threads=self.encoder_threads,
            )
        else:
            dataset._streaming_encoder = None
        return dataset

    def init_dataset(self) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if self.dataset_root.exists() and self._metadata_ready():
            print(f"[INFO]: Opening existing LeRobot metadata at {self.dataset_root}...", flush=True)
            self.dataset = self._open_append_only_dataset()
            print("[INFO]: Existing LeRobot metadata loaded; validating recording schema...", flush=True)
            self._validate_existing_features()
            self._configure_recording_helpers()
            self._sync_saved_episode_count()
            print(f"[INFO]: Existing LeRobot dataset initialized: {self.dataset.root}", flush=True)
            self._print_recording_config()
            return

        if self._clear_incomplete_empty_root():
            print(f"[INFO]: Removed incomplete empty LeRobot dataset root: {self.dataset_root}")

        print(f"[INFO]: Creating new LeRobot dataset at {self.dataset_root}...", flush=True)
        self.dataset = LeRobotDataset.create(
            self.repo_id,
            fps=self.fps,
            features=self._features(),
            root=self.dataset_root,
            robot_type="so101_follower",
            image_writer_processes=0,
            image_writer_threads=0,
            **self._dataset_kwargs(),
        )
        self._configure_recording_helpers()
        self._sync_saved_episode_count()
        print(f"[INFO]: New LeRobot dataset initialized: {self.dataset.root}")
        self._print_recording_config()

    def _validate_existing_features(self) -> None:
        from lerobot.datasets.utils import DEFAULT_FEATURES

        meta = getattr(self.dataset, "meta", None)
        existing_features = getattr(meta, "features", None)
        if existing_features is None:
            return

        def _without_video_info(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
            return {
                key: {field: value for field, value in feature.items() if field != "info"}
                for key, feature in features.items()
            }

        def _canonical_features(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
            return json.loads(json.dumps(_without_video_info(features)))

        expected_features = {**self._features(), **DEFAULT_FEATURES}
        expected_keys = set(expected_features)
        existing_keys = set(existing_features)
        missing = sorted(expected_keys - existing_keys)
        extra = sorted(existing_keys - expected_keys)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if extra:
                details.append(f"extra: {', '.join(extra)}")
            raise ValueError(
                f"Existing LeRobot dataset at {self.dataset_root} has incompatible feature keys "
                f"({'; '.join(details)}). Use a new --repo_root or migrate the dataset schema."
            )

        expected_base = _canonical_features(expected_features)
        existing_base = _canonical_features(existing_features)
        if existing_base != expected_base:
            raise ValueError(
                f"Existing LeRobot dataset at {self.dataset_root} has incompatible feature metadata. "
                "Use a new --repo_root or migrate the dataset schema."
            )

        if self._resolve_fast_vcodec() == "libsvtav1":
            incompatible_video_codecs = []
            for key in expected_features:
                if expected_features[key]["dtype"] != "video":
                    continue
                codec = existing_features[key].get("info", {}).get("video.codec")
                if codec is not None and codec != "av1":
                    incompatible_video_codecs.append(f"{key}={codec}")
            if incompatible_video_codecs:
                raise ValueError(
                    f"Existing LeRobot dataset at {self.dataset_root} has video codec(s) incompatible "
                    f"with so101_bench_real_2 merge compatibility: {', '.join(incompatible_video_codecs)}. "
                    "Use a new --repo_root or re-record with --dataset_vcodec libsvtav1."
                )

    def start_episode(self, *, task: str) -> None:
        if self.recording:
            return
        if self.dataset is None:
            raise RuntimeError("Dataset has not been initialized.")
        if self.dataset.episode_buffer is not None and self.dataset.episode_buffer.get("size", 0) > 0:
            self.dataset.clear_episode_buffer()
        self._frame_count = 0
        self._task = task
        self.recording = True
        print("[INFO]: Started recording current episode.")

    def push_frame(
        self,
        *,
        action: torch.Tensor,
        observation_state: torch.Tensor,
        images: dict[str, np.ndarray],
    ) -> None:
        if not self.recording:
            return
        if self.dataset is None:
            raise RuntimeError("Dataset has not been initialized.")
        if self._task is None:
            raise RuntimeError("Recording task was not set.")

        started = time.perf_counter() if self.timing_callback is not None else 0.0
        dataset_frame = {
            "action": action.detach().cpu().numpy().astype(np.float32, copy=True),
            "observation.state": observation_state.detach().cpu().numpy().astype(np.float32, copy=True),
            "task": self._task,
        }
        if self.timing_callback is not None:
            self.timing_callback("recording.frame.action_state_to_cpu", time.perf_counter() - started)
        for camera_name, image in images.items():
            dataset_frame[f"observation.images.{camera_name}"] = image
        started = time.perf_counter() if self.timing_callback is not None else 0.0
        self.dataset.add_frame(dataset_frame)
        if self.timing_callback is not None:
            self.timing_callback("recording.frame.dataset_add_and_encoder_enqueue", time.perf_counter() - started)
        self._frame_count += 1

    def stop_episode(self, *, task: str) -> bool:
        if not self.recording:
            return False
        if self.dataset is None:
            raise RuntimeError("Dataset has not been initialized.")
        if self._frame_count == 0:
            print("[WARN]: Recording contained no frames; nothing was saved.")
            self.recording = False
            self._task = None
            return False

        if task != self._task:
            print(f"[WARN]: Recording task changed while saving; using task captured at start: {self._task}")
        start_save = time.perf_counter()
        print(f"[INFO]: Finalizing {self._frame_count} frame(s) to LeRobot dataset...")
        if self.streaming_encoding:
            self.dataset.save_episode()
        else:
            # LeRobot normally uses ProcessPoolExecutor here. Its Linux workers
            # fork Isaac's already-threaded CUDA/Kit process and can deadlock
            # permanently. Keep the per-camera fan-out but dispatch it from threads
            # that each drive a clean child process.
            import lerobot.datasets.lerobot_dataset as lerobot_dataset_module

            original_encode_video_frames = lerobot_dataset_module.encode_video_frames
            original_encode_video_worker = lerobot_dataset_module._encode_video_worker
            original_concurrent = lerobot_dataset_module.concurrent
            lerobot_dataset_module.encode_video_frames = _encode_video_frames_in_clean_subprocess
            lerobot_dataset_module._encode_video_worker = _encode_video_worker_in_clean_subprocess
            lerobot_dataset_module.concurrent = _ThreadedProcessPoolShim
            try:
                self.dataset.save_episode(parallel_encoding=self.parallel_encoding)
            finally:
                lerobot_dataset_module.encode_video_frames = original_encode_video_frames
                lerobot_dataset_module._encode_video_worker = original_encode_video_worker
                lerobot_dataset_module.concurrent = original_concurrent
        elapsed = time.perf_counter() - start_save
        self.num_saved_episodes += 1
        self._frame_count = 0
        self._task = None
        self.recording = False
        print(f"[INFO]: Saved LeRobot episode {self.num_saved_episodes} in {elapsed:.2f}s.")
        return True

    def cancel_episode(self) -> None:
        if not self.recording and self._frame_count == 0:
            return
        if self.dataset is not None and self.dataset.episode_buffer is not None:
            self.dataset.clear_episode_buffer()
        self._frame_count = 0
        self._task = None
        self.recording = False
        print("[INFO]: Cancelled current recording.")

    def streaming_encoder_stats(self) -> dict[str, dict[str, int]]:
        """Return non-blocking queue/drop telemetry for optional evaluation profiling."""

        if self.dataset is None:
            return {"queue_depths": {}, "dropped_frames": {}}
        encoder = getattr(self.dataset, "_streaming_encoder", None)
        if encoder is None:
            return {"queue_depths": {}, "dropped_frames": {}}
        frame_queues = getattr(encoder, "_frame_queues", {})
        dropped_by_camera = getattr(encoder, "_dropped_frames", {})
        queue_depths = {
            str(camera): int(frame_queue.qsize())
            for camera, frame_queue in frame_queues.items()
        }
        dropped_frames = {
            str(camera): int(dropped_by_camera.get(camera, 0))
            for camera in frame_queues.keys() | dropped_by_camera.keys()
        }
        return {"queue_depths": queue_depths, "dropped_frames": dropped_frames}

    def finalize(self) -> None:
        if self.dataset is None:
            return
        if hasattr(self.dataset, "stop_image_writer"):
            self.dataset.stop_image_writer()
        if hasattr(self.dataset, "finalize"):
            self.dataset.finalize()


def real_compatible_camera_sources(cameras: dict[str, dict[str, int]]) -> dict[str, str]:
    """Map sim camera names to the camera feature names used by so101_bench_real_2."""

    camera_sources = {}
    for dataset_name, source_candidates in REAL_COMPAT_CAMERA_SOURCES.items():
        for source_name in source_candidates:
            if source_name in cameras:
                camera_sources[dataset_name] = source_name
                break
        else:
            raise KeyError(
                f"Expected one of {source_candidates} for dataset camera {dataset_name!r}; "
                f"found cameras: {list(cameras)}"
            )

    for dataset_name, source_name in camera_sources.items():
        if dataset_name != source_name:
            print(f"[INFO]: Recording sim camera '{source_name}' as LeRobot camera '{dataset_name}'.")
    return camera_sources


def dataset_cameras(
    cameras: dict[str, dict[str, int]],
    camera_sources: dict[str, str],
) -> dict[str, dict[str, int]]:
    return {dataset_name: cameras[source_name] for dataset_name, source_name in camera_sources.items()}


def _to_uint8_rgb(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"Expected image channel dimension of 3 or 4, got shape {array.shape}.")
    if np.issubdtype(array.dtype, np.floating):
        if array.max(initial=0.0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0)
    return np.ascontiguousarray(array.astype(np.uint8))


def _visual_buffer(visual_obs: dict, camera_name: str) -> np.ndarray:
    key = f"rgb_{camera_name}"
    if key not in visual_obs:
        raise KeyError(f"Expected visual observation key {key!r}; got: {list(visual_obs.keys())}")
    image = visual_obs[key]
    if isinstance(image, torch.Tensor) and image.ndim >= 4:
        image = image[0]
    elif not isinstance(image, torch.Tensor):
        image = np.asarray(image)
        if image.ndim >= 4:
            image = image[0]
    return _to_uint8_rgb(image)


def recording_images(
    visual_obs: dict,
    camera_sources: dict[str, str],
) -> dict[str, np.ndarray]:
    return {
        dataset_name: _visual_buffer(visual_obs, source_name)
        for dataset_name, source_name in camera_sources.items()
    }
