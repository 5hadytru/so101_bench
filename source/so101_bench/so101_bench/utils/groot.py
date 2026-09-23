"""GR00T remote-policy helpers for the SO-101 Isaac Lab environment.

The joint mapping and message shape follow NVIDIA's public SO-101 workshop.
This module is intentionally LeRobot-light: it only needs NumPy, Torch, msgpack,
and ZeroMQ to talk to a running GR00T policy server.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, is_dataclass
from enum import Enum
import io
import time
from typing import Any, Callable

import msgpack
import numpy as np
import torch
import zmq

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


def _to_json_serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {key: _to_json_serializable(value) for key, value in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, dict):
        return {key: _to_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class MsgSerializer:
    """Msgpack serializer compatible with the GR00T policy server helper."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes, raw=False)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        if is_dataclass(obj):
            return _to_json_serializable(obj)
        return obj


class PolicyClient:
    """Small ZeroMQ request client for a GR00T policy server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str | None = None,
        timing_callback: Callable[[str, float], None] | None = None,
        counter_callback: Callable[[str, float], None] | None = None,
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.timing_callback = timing_callback
        self.counter_callback = counter_callback
        self._init_socket()

    def _record_timing(self, name: str, elapsed_s: float) -> None:
        if self.timing_callback is not None:
            self.timing_callback(name, elapsed_s)

    def _increment(self, name: str, value: float = 1.0) -> None:
        if self.counter_callback is not None:
            self.counter_callback(name, value)

    def _init_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        total_started = time.perf_counter()
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        if self.api_token:
            request["api_token"] = self.api_token

        try:
            started = time.perf_counter()
            request_bytes = MsgSerializer.to_bytes(request)
            self._record_timing(f"policy.rpc.{endpoint}.serialize", time.perf_counter() - started)
            self._increment(f"policy.rpc.{endpoint}.request_bytes", len(request_bytes))

            started = time.perf_counter()
            self.socket.send(request_bytes)
            self._record_timing(f"policy.rpc.{endpoint}.send", time.perf_counter() - started)

            started = time.perf_counter()
            message = self.socket.recv()
            self._record_timing(f"policy.rpc.{endpoint}.receive_wait", time.perf_counter() - started)
            self._increment(f"policy.rpc.{endpoint}.response_bytes", len(message))

            started = time.perf_counter()
            result = MsgSerializer.from_bytes(message)
            self._record_timing(f"policy.rpc.{endpoint}.deserialize", time.perf_counter() - started)
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            return result
        finally:
            self._record_timing(f"policy.rpc.{endpoint}.total", time.perf_counter() - total_started)

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()
            return False

    def reset(self, options: dict[str, Any] | None = None):
        return self.call_endpoint("reset", {"options": options})

    def get_action(self, observation: dict) -> tuple[dict, dict]:
        result = self.call_endpoint("get_action", {"observation": observation, "options": None}, requires_input=True)
        if isinstance(result, tuple):
            return result
        if isinstance(result, list) and len(result) == 2:
            return result[0], result[1]
        return result, {}


class SO101JointMapper:
    """Convert between SO-101 USD radians and calibrated LeRobot/GR00T `.pos` space."""

    joint_order = LEROBOT_JOINT_FEATURE_ORDER

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

    def raw_action_tensor(self, real_action: dict[str, float]) -> torch.Tensor:
        return torch.tensor([real_action[joint] for joint in self.joint_order], dtype=torch.float32, device=self.device)

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

    def sim_radians_to_raw_degrees(self, sim_values: torch.Tensor) -> torch.Tensor:
        return self.sim_radians_to_lerobot_positions(sim_values)

    def raw_degrees_to_sim_radians(self, raw_values: torch.Tensor) -> torch.Tensor:
        return self.lerobot_positions_to_sim_radians(raw_values)


class GR00TRemotePolicy:
    """Chunked-action GR00T remote policy wrapper for the SO-101 env."""

    def __init__(
        self,
        device: str,
        cameras: dict[str, dict[str, int]],
        host: str = "localhost",
        port: int = 5555,
        action_horizon: int = 16,
        lang_instruction: str = "Place each object in the plastic bin.",
        rename_map: dict[str, str] | None = None,
        use_overhead_init: bool = False,
        overhead_init_camera: str = "overhead",
        overhead_init_key: str = "overhead_init",
        image_size: tuple[int, int] | None = (640, 480),
        timing_callback: Callable[[str, float], None] | None = None,
        counter_callback: Callable[[str, float], None] | None = None,
    ):
        self.mapper = SO101JointMapper(device=device)
        self.cameras = cameras
        self.host = host
        self.port = port
        self.action_horizon = action_horizon
        self.lang_instruction = lang_instruction
        self.rename_map = rename_map or {}
        self.use_overhead_init = use_overhead_init
        self.overhead_init_camera = overhead_init_camera
        self.overhead_init_key = overhead_init_key
        self.image_size = image_size
        self.timing_callback = timing_callback
        self.counter_callback = counter_callback
        self.overhead_init_image: np.ndarray | None = None
        self.action_queue: deque[dict[str, float]] = deque()
        self.client: PolicyClient | None = None

    def _record_timing(self, name: str, elapsed_s: float) -> None:
        if self.timing_callback is not None:
            self.timing_callback(name, elapsed_s)

    def _increment(self, name: str, value: float = 1.0) -> None:
        if self.counter_callback is not None:
            self.counter_callback(name, value)

    def _timed_call(self, name: str, function: Callable[..., Any], *args, **kwargs):
        if self.timing_callback is None:
            return function(*args, **kwargs)
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            self._record_timing(name, time.perf_counter() - started)

    def connect(self):
        print(f"[INFO]: Connecting to GR00T policy server at {self.host}:{self.port}...")
        self.client = PolicyClient(
            host=self.host,
            port=self.port,
            timing_callback=self.timing_callback,
            counter_callback=self.counter_callback,
        )
        if not self.client.ping():
            raise RuntimeError("Cannot connect to GR00T policy server.")
        print("[INFO]: Policy server connected.")

    def reset(self, initial_visual_obs: dict | None = None, *, reset_remote: bool = True):
        if reset_remote and self.client is not None:
            self.client.reset()
        self.action_queue.clear()
        self.overhead_init_image = None
        if initial_visual_obs is not None:
            self.set_episode_initial_observation(initial_visual_obs)

    def set_language_instruction(self, instruction: str):
        self.lang_instruction = instruction
        self.action_queue.clear()

    def set_episode_initial_observation(self, visual_obs: dict):
        """Capture the fixed settled overhead-init frame used by WM-conditioned policies."""

        if not self.use_overhead_init:
            return
        self.overhead_init_image = self._timed_call(
            "policy.input.overhead_init_capture",
            lambda: self._camera_frame(visual_obs, self.overhead_init_camera).copy(),
        )
        self.action_queue.clear()

    @staticmethod
    def _add_batch_time_dims(obs: dict) -> dict:
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                obs[key] = val[np.newaxis, ...]
            elif isinstance(val, dict):
                obs[key] = GR00TRemotePolicy._add_batch_time_dims(val)
            else:
                obs[key] = [val]
        return obs

    @staticmethod
    def _to_uint8_rgb(image: torch.Tensor | np.ndarray) -> np.ndarray:
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

    @staticmethod
    def _resize_rgb(image: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
        if size is None:
            return image

        width, height = size
        if image.shape[1] == width and image.shape[0] == height:
            return image

        try:
            import cv2

            resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            return np.ascontiguousarray(resized)
        except Exception:
            pass

        try:
            from PIL import Image

            resized = Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR)
            return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))
        except Exception:
            pass

        y_idx = np.linspace(0, image.shape[0] - 1, height).astype(np.int64)
        x_idx = np.linspace(0, image.shape[1] - 1, width).astype(np.int64)
        return np.ascontiguousarray(image[y_idx][:, x_idx])

    def _camera_frame(self, visual_obs: dict, camera: str, env_index: int = 0) -> np.ndarray:
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
        uint8_image = self._timed_call(
            f"policy.input.camera.{camera}.to_cpu_uint8",
            self._to_uint8_rgb,
            image,
        )
        return self._timed_call(
            f"policy.input.camera.{camera}.resize",
            self._resize_rgb,
            uint8_image,
            self.image_size,
        )

    def _sim_obs_to_groot_inputs(self, joint_positions: torch.Tensor, visual_obs: dict) -> dict:
        mapped_state = self._timed_call(
            "policy.input.state_mapping",
            self.mapper.sim_radians_to_raw_degrees,
            joint_positions,
        )
        state = self._timed_call(
            "policy.input.state_to_cpu",
            lambda: mapped_state.cpu().numpy().astype(np.float32),
        )

        model_obs: dict[str, Any] = {
            "video": {},
            "state": {
                "single_arm": state[:5],
                "gripper": state[5:6],
            },
            "language": {
                "annotation.human.task_description": self.lang_instruction,
            },
        }

        for camera in self.cameras:
            img = self._timed_call(
                f"policy.input.camera.{camera}",
                self._camera_frame,
                visual_obs,
                camera,
            )
            policy_camera_name = self.rename_map.get(camera, camera)
            model_obs["video"][policy_camera_name] = img

        if self.use_overhead_init:
            if self.overhead_init_image is None:
                raise RuntimeError(
                    "use_overhead_init=True, but no initial overhead frame has been captured. "
                    "Call set_episode_initial_observation(obs['visual']) after the episode settle hold "
                    "and before the first policy query."
                )
            model_obs["video"][self.overhead_init_key] = self._timed_call(
                "policy.input.overhead_init_copy",
                self.overhead_init_image.copy,
            )

        model_obs = self._timed_call("policy.input.batch_dimensions", self._add_batch_time_dims, model_obs)
        model_obs = self._timed_call("policy.input.batch_dimensions", self._add_batch_time_dims, model_obs)
        return model_obs

    def _decode_action_chunk(self, action_chunk: dict) -> list[dict[str, float]]:
        single_arm = action_chunk.get("single_arm", action_chunk.get("action.single_arm"))
        gripper = action_chunk.get("gripper", action_chunk.get("action.gripper"))
        if single_arm is None or gripper is None:
            raise KeyError(f"Expected GR00T action keys 'single_arm' and 'gripper', got: {list(action_chunk.keys())}")

        single_arm = np.asarray(single_arm)
        gripper = np.asarray(gripper)
        if single_arm.ndim == 1:
            single_arm = single_arm[np.newaxis, np.newaxis, ...]
        if single_arm.ndim == 2:
            single_arm = single_arm[np.newaxis, ...]
        if gripper.ndim == 0:
            gripper = gripper.reshape(1, 1, 1)
        if gripper.ndim == 1:
            gripper = gripper[np.newaxis, :, np.newaxis]
        if gripper.ndim == 2:
            if gripper.shape[0] == single_arm.shape[0] and gripper.shape[1] == single_arm.shape[1]:
                gripper = gripper[..., np.newaxis]
            else:
                gripper = gripper[np.newaxis, ...]
        horizon = min(single_arm.shape[1], self.action_horizon)
        actions: list[dict[str, float]] = []
        for step in range(horizon):
            full = np.concatenate([single_arm[0][step], gripper[0][step]], axis=0)
            if not np.isfinite(full).all():
                raise ValueError(f"GR00T returned a non-finite action at chunk step {step}: {full.tolist()}")
            actions.append({joint: float(full[idx]) for idx, joint in enumerate(self.mapper.joint_order)})
        return actions

    def get_action(self, joint_positions: torch.Tensor, visual_obs: dict) -> torch.Tensor:
        if self.client is None:
            raise RuntimeError("Call connect() before requesting actions.")

        total_started = time.perf_counter() if self.timing_callback is not None else 0.0
        try:
            if not self.action_queue:
                self._increment("policy.action_cache.misses")
                query_started = time.perf_counter() if self.timing_callback is not None else 0.0
                model_input = self._timed_call(
                    "policy.input.prepare_total",
                    self._sim_obs_to_groot_inputs,
                    joint_positions,
                    visual_obs,
                )
                action_chunk, info = self.client.get_action(model_input)
                server_timing = info.get("_timing", {}).get("server", {}) if isinstance(info, dict) else {}
                if isinstance(server_timing, dict):
                    if isinstance(server_timing.get("decode_s"), (int, float)):
                        self._record_timing("policy.server.request_decode_reported", server_timing["decode_s"])
                    if isinstance(server_timing.get("handler_s"), (int, float)):
                        self._record_timing("policy.server.handler_reported", server_timing["handler_s"])
                decoded_actions = self._timed_call(
                    "policy.action.decode_chunk",
                    self._decode_action_chunk,
                    action_chunk,
                )
                self.action_queue.extend(decoded_actions)
                self._increment("policy.action_chunks")
                self._increment("policy.actions_received", len(decoded_actions))
                if self.timing_callback is not None:
                    self._record_timing("policy.query.total", time.perf_counter() - query_started)
            else:
                self._increment("policy.action_cache.hits")

            def map_next_action() -> torch.Tensor:
                action_dict = self.action_queue.popleft()
                raw_tensor = self.mapper.raw_action_tensor(action_dict)
                return self.mapper.raw_degrees_to_sim_radians(raw_tensor)

            return self._timed_call("policy.action.map", map_next_action)
        finally:
            if self.timing_callback is not None:
                self._record_timing("policy.get_action.total", time.perf_counter() - total_started)
