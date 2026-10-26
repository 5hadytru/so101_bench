"""MolmoAct2 remote-policy helpers for the SO-101 Isaac Lab environment."""

from __future__ import annotations

import base64
from collections import deque
import io
import json
from typing import Any
from urllib import error, request

import numpy as np
import torch

from so101_bench.utils.lerobot_calibration import (
    LEROBOT_JOINT_ORDER,
    REAL_SO101_CALIBRATION,
    SIM_LIMIT_MARGIN_DEG,
    STS3215_CENTER_POSITION,
    STS3215_DEGREES_PER_TICK,
    USD_SIM_JOINT_LIMITS_DEG,
)

MODEL_JOINT_SIGNS = (1.0, -1.0, 1.0, 1.0, 1.0, 1.0)
MODEL_JOINT_OFFSETS_DEG = (0.0, 90.0, 90.0, 0.0, 0.0, 0.0)


class SO101MolmoAct2Mapper:
    """Convert SO-101 USD radians to the released MolmoAct2 SO-100/101 frame."""

    joint_order = LEROBOT_JOINT_ORDER

    def __init__(self, device: str):
        self.device = device
        self.is_gripper = torch.tensor(
            [joint_name == "gripper" for joint_name in self.joint_order],
            device=device,
        )
        self.model_signs = torch.tensor(MODEL_JOINT_SIGNS, dtype=torch.float32, device=device)
        self.model_offsets_deg = torch.tensor(MODEL_JOINT_OFFSETS_DEG, dtype=torch.float32, device=device)
        self.usd_mins_deg = torch.tensor(
            [USD_SIM_JOINT_LIMITS_DEG[name][0] for name in self.joint_order],
            dtype=torch.float32,
            device=device,
        )
        self.usd_maxs_deg = torch.tensor(
            [USD_SIM_JOINT_LIMITS_DEG[name][1] for name in self.joint_order],
            dtype=torch.float32,
            device=device,
        )
        self.lerobot_degree_offsets = torch.tensor(
            [
                (
                    STS3215_CENTER_POSITION
                    - 0.5
                    * (
                        REAL_SO101_CALIBRATION[name].range_min
                        + REAL_SO101_CALIBRATION[name].range_max
                    )
                )
                * STS3215_DEGREES_PER_TICK
                for name in self.joint_order
            ],
            dtype=torch.float32,
            device=device,
        )

    def sim_radians_to_model_positions(self, sim_values: torch.Tensor) -> torch.Tensor:
        """Map sim joints into the public SO-100/101 v2.1 dataset convention."""

        sim_deg = sim_values * 180.0 / torch.pi
        body_positions = sim_deg + self.lerobot_degree_offsets
        gripper_positions = 100.0 * (sim_deg - self.usd_mins_deg) / (self.usd_maxs_deg - self.usd_mins_deg)
        arm_positions = torch.where(self.is_gripper, gripper_positions, body_positions)
        return arm_positions * self.model_signs + self.model_offsets_deg

    def model_positions_to_sim_radians(self, model_values: torch.Tensor) -> torch.Tensor:
        """Map model-frame absolute targets back into bounded sim joint radians."""

        arm_positions = (model_values - self.model_offsets_deg) * self.model_signs
        body_degrees = arm_positions - self.lerobot_degree_offsets
        gripper_degrees = self.usd_mins_deg + (arm_positions / 100.0) * (
            self.usd_maxs_deg - self.usd_mins_deg
        )
        sim_deg = torch.where(self.is_gripper, gripper_degrees, body_degrees)
        sim_deg = torch.minimum(
            torch.maximum(sim_deg, self.usd_mins_deg + SIM_LIMIT_MARGIN_DEG),
            self.usd_maxs_deg - SIM_LIMIT_MARGIN_DEG,
        )
        return sim_deg * torch.pi / 180.0


class SO101LeRobotUnitsMapper:
    """Sim radians <-> the dataset's native LeRobot `.pos` units (for pi0.5 and other LeRobot policies).

    Same interface as SO101MolmoAct2Mapper; wraps the GR00T client's calibrated mapper, which
    the so101_bench_sim_WM dataset was recorded with.
    """

    joint_order = LEROBOT_JOINT_ORDER

    def __init__(self, device: str):
        from so101_bench.utils.groot import SO101JointMapper

        self.device = device
        self._mapper = SO101JointMapper(device=device)

    def sim_radians_to_model_positions(self, sim_values: torch.Tensor) -> torch.Tensor:
        return self._mapper.sim_radians_to_lerobot_positions(sim_values)

    def model_positions_to_sim_radians(self, model_values: torch.Tensor) -> torch.Tensor:
        return self._mapper.lerobot_positions_to_sim_radians(model_values)


JOINT_FRAMES = {"molmoact2": SO101MolmoAct2Mapper, "lerobot": SO101LeRobotUnitsMapper}


class MolmoAct2RemotePolicy:
    """Chunked-action HTTP client for ``scripts/molmoact2_server.py``."""

    def __init__(
        self,
        device: str,
        cameras: dict[str, dict[str, int]],
        host: str = "localhost",
        port: int = 8000,
        action_horizon: int = 30,
        lang_instruction: str = "Place each object in the plastic bin.",
        camera_names: list[str] | None = None,
        image_size: tuple[int, int] | None = None,
        timeout_s: float = 120.0,
        num_steps: int = 10,
        max_joint_step_deg: float = 15.0,
        use_overhead_init: bool = False,
        overhead_init_camera: str = "overhead",
        overhead_init_key: str = "overhead_init",
        use_wrist_init: bool = False,
        wrist_init_camera: str = "wrist",
        wrist_init_key: str = "wrist_init",
        joint_frame: str = "molmoact2",
        history_ticks: int = 1,
        prediction_trace_dir: str | None = None,
    ):
        # "molmoact2": the released MolmoAct2 SO-100/101 frame (MolmoAct2 fine-tunes too);
        # "lerobot": the dataset's native LeRobot units (pi0.5 and other LeRobot policies).
        if joint_frame not in JOINT_FRAMES:
            raise ValueError(f"joint_frame must be one of {sorted(JOINT_FRAMES)}, got {joint_frame!r}")
        self.mapper = JOINT_FRAMES[joint_frame](device=device)
        self.cameras = cameras
        self.host = host
        self.port = port
        self.action_horizon = action_horizon
        self.lang_instruction = lang_instruction
        self.camera_names = camera_names or ["overhead", "wrist"]
        self.image_size = image_size
        self.timeout_s = timeout_s
        self.num_steps = num_steps
        self.max_joint_step_rad = max_joint_step_deg * torch.pi / 180.0
        self.action_queue: deque[torch.Tensor] = deque()
        # History-conditioned servers (flux3_server.py) receive the last ``history_ticks`` control ticks
        # with every request: measured state, the command issued before each tick, and the camera frames
        # of the oldest and newest tick. The first tick of an episode fills the window with itself, and
        # its "previous command" is the measured state, as in flux-action's ObservationHistory.
        self.history_ticks = history_ticks
        self._history: deque[dict] = deque(maxlen=max(1, history_ticks))
        self._last_command: np.ndarray | None = None
        # With prediction_trace_dir set, each request names where the server should save the video it
        # predicts (flux3_server.py --save_predictions); the eval loop sets the episode/frame per tick.
        self.prediction_trace_dir = prediction_trace_dir
        self.trace_episode = 0
        self.trace_frame = 0

        if self.history_ticks < 1:
            raise ValueError(f"history_ticks must be at least 1, got {self.history_ticks}.")
        if self.action_horizon < 1:
            raise ValueError(f"action_horizon must be at least 1, got {self.action_horizon}.")
        if self.timeout_s <= 0.0:
            raise ValueError(f"timeout_s must be positive, got {self.timeout_s}.")
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be at least 1, got {self.num_steps}.")
        if max_joint_step_deg < 0.0:
            raise ValueError(f"max_joint_step_deg must be non-negative, got {max_joint_step_deg}.")

        # WM-conditioned fine-tunes (so101_bench_sim_WM) also see the settled frames captured when
        # robot control starts, appended after the live cameras: overhead_init, then wrist_init.
        self.use_overhead_init = use_overhead_init
        self.overhead_init_camera = overhead_init_camera
        self.overhead_init_key = overhead_init_key
        self.use_wrist_init = use_wrist_init
        self.wrist_init_camera = wrist_init_camera
        self.wrist_init_key = wrist_init_key
        self.overhead_init_image: np.ndarray | None = None
        self.wrist_init_image: np.ndarray | None = None

        if not self.camera_names:
            raise ValueError("MolmoAct2 needs at least one policy camera.")
        init_cameras = [overhead_init_camera] * use_overhead_init + [wrist_init_camera] * use_wrist_init
        missing_cameras = [name for name in [*self.camera_names, *init_cameras] if name not in self.cameras]
        if missing_cameras:
            raise ValueError(f"Unknown MolmoAct2 policy camera(s) {missing_cameras}; found: {list(self.cameras)}")

    @property
    def image_keys(self) -> list[str]:
        """Image order sent to the server; it must match the checkpoint's training camera order."""
        keys = list(self.camera_names)
        if self.use_overhead_init:
            keys.append(self.overhead_init_key)
        if self.use_wrist_init:
            keys.append(self.wrist_init_key)
        return keys

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/act"

    def connect(self) -> None:
        print(f"[INFO]: Connecting to MolmoAct2 policy server at {self.url}...")
        try:
            with request.urlopen(self.url, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot connect to MolmoAct2 policy server at {self.url}: {exc}") from exc
        if payload.get("status") != "ok":
            raise RuntimeError(f"MolmoAct2 policy server health check failed: {payload}")
        server_history = int(payload.get("history_ticks", 1))
        if server_history != self.history_ticks:
            raise RuntimeError(
                f"Policy server conditions on {server_history} history ticks, but this client sends {self.history_ticks}."
            )
        expected_images = payload.get("num_images")
        if expected_images is not None and int(expected_images) != len(self.image_keys):
            raise RuntimeError(
                f"MolmoAct2 server checkpoint expects {expected_images} images "
                f"({payload.get('camera_keys')}), but this client sends {self.image_keys}."
            )
        print(
            f"[INFO]: MolmoAct2 policy server connected ({payload.get('repo_id', 'unknown checkpoint')}, "
            f"norm_tag={payload.get('norm_tag')}, images={self.image_keys})."
        )

    def reset(self, initial_visual_obs: dict | None = None, *, reset_remote: bool = True) -> None:
        del initial_visual_obs, reset_remote
        self._clear_episode_state()

    def set_language_instruction(self, instruction: str) -> None:
        self.lang_instruction = instruction
        self._clear_episode_state()

    def set_episode_initial_observation(self, visual_obs: dict) -> None:
        """Capture the fixed settled camera frames used by WM-conditioned checkpoints."""

        if self.use_overhead_init:
            self.overhead_init_image = self._camera_frame(visual_obs, self.overhead_init_camera).copy()
        if self.use_wrist_init:
            self.wrist_init_image = self._camera_frame(visual_obs, self.wrist_init_camera).copy()
        self._clear_episode_state()

    def _clear_episode_state(self) -> None:
        self.action_queue.clear()
        self._history.clear()
        self._last_command = None

    def _record_tick(self, joint_positions: torch.Tensor, visual_obs: dict) -> None:
        state = self.mapper.sim_radians_to_model_positions(joint_positions).detach().cpu().numpy().astype(np.float32)
        tick = {
            "state": state,
            "command": state if self._last_command is None else self._last_command,
            "frames": [self._camera_frame(visual_obs, name) for name in self.camera_names],
        }
        for _ in range(self.history_ticks if not self._history else 1):
            self._history.append(tick)

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

        from PIL import Image

        resized = Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR)
        return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))

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
        return self._resize_rgb(self._to_uint8_rgb(image), self.image_size)

    @staticmethod
    def _encode_png(image: np.ndarray) -> str:
        from PIL import Image

        output = io.BytesIO()
        Image.fromarray(image).save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")

    def _policy_images(self, visual_obs: dict) -> list[np.ndarray]:
        images = [self._camera_frame(visual_obs, name) for name in self.camera_names]
        if self.use_overhead_init:
            if self.overhead_init_image is None:
                raise RuntimeError("overhead_init frame requested before set_episode_initial_observation().")
            images.append(self.overhead_init_image)
        if self.use_wrist_init:
            if self.wrist_init_image is None:
                raise RuntimeError("wrist_init frame requested before set_episode_initial_observation().")
            images.append(self.wrist_init_image)
        return images

    def _query_server(self, joint_positions: torch.Tensor, visual_obs: dict) -> np.ndarray:
        model_state = self.mapper.sim_radians_to_model_positions(joint_positions).detach().cpu().numpy()
        payload = {
            "images_png_base64": [self._encode_png(image) for image in self._policy_images(visual_obs)],
            "camera_names": self.image_keys,
            "instruction": self.lang_instruction,
            "state": model_state.astype(np.float32).tolist(),
            "num_steps": self.num_steps,
            # Stateful servers (internw0_server.py) reset their per-episode memory when this changes.
            "episode": self.trace_episode,
        }
        if self.history_ticks > 1:
            payload["snapshot_images_png_base64"] = [self._encode_png(image) for image in self._history[0]["frames"]]
            payload["state_history"] = [tick["state"].tolist() for tick in self._history]
            payload["command_history"] = [np.asarray(tick["command"], dtype=np.float32).tolist() for tick in self._history]
        if self.prediction_trace_dir:
            payload["trace"] = {"dir": self.prediction_trace_dir, "episode": self.trace_episode, "frame": self.trace_frame}
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        http_request = request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        try:
            with request.urlopen(http_request, timeout=self.timeout_s) as response:
                result = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"MolmoAct2 server returned HTTP {exc.code}: {details}") from exc
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"MolmoAct2 server request failed: {exc}") from exc

        if "error" in result:
            raise RuntimeError(f"MolmoAct2 server inference failed: {result['error']}")
        actions = np.asarray(result.get("actions"), dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[1] != len(self.mapper.joint_order):
            raise RuntimeError(f"Expected MolmoAct2 actions with shape (T, 6), got {actions.shape}.")
        print(f"[INFO]: MolmoAct2 inference: {float(result.get('dt_ms', 0.0)):.1f} ms, chunk={actions.shape}")
        return actions

    def _limit_joint_step(self, current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.max_joint_step_rad <= 0.0:
            return target
        delta = target - current
        biggest = torch.max(torch.abs(delta))
        if biggest <= self.max_joint_step_rad:
            return target
        return current + delta * (self.max_joint_step_rad / biggest)

    def get_action(self, joint_positions: torch.Tensor, visual_obs: dict) -> torch.Tensor:
        if self.history_ticks > 1:
            self._record_tick(joint_positions, visual_obs)
        if not self.action_queue:
            action_chunk = self._query_server(joint_positions, visual_obs)
            horizon = min(action_chunk.shape[0], self.action_horizon)
            self.action_queue.extend(
                torch.as_tensor(row, dtype=torch.float32, device=self.mapper.device)
                for row in action_chunk[:horizon]
            )

        model_action = self.action_queue.popleft()
        if self.history_ticks > 1:
            self._last_command = model_action.detach().cpu().numpy().astype(np.float32)
        sim_action = self.mapper.model_positions_to_sim_radians(model_action)
        return self._limit_joint_step(joint_positions, sim_action)
