"""MolmoAct2 remote-policy helper for the SO-101 Isaac Lab environment.

This mirrors :class:`so101_bench.utils.groot.GR00TRemotePolicy`, but talks to the
HTTP MolmoAct2 inference server in ``scripts/molmoact2_server.py`` instead of a
ZeroMQ GR00T server, and speaks the MolmoAct2 SO-100/101 joint frame instead of
LeRobot ``.pos`` units.

Wire protocol (see ``scripts/molmoact2_server.py``):

    GET  /act  -> health check
    POST /act  -> {"images_png_base64": [png_b64, png_b64],
                   "instruction": str,
                   "state": [6 floats],
                   "num_steps": int}
              <- {"actions": [[6 floats], ...], "dt_ms": float}

Joint-frame note
----------------
The server normalizes ``state`` / ``actions`` internally via
``norm_tag="so100_so101_molmoact2"``, so the client only has to speak the model's
*un-normalized* per-joint frame. That frame is the Feetech present-position angle in
degrees (0-360, servo centre ~180 deg), which the server warmup vector
``[0, 180, 180, 60, 0, 0]`` is consistent with. We reach it through the repo's own
calibration bridge (sim radians -> USD sim degrees -> calibrated LeRobot ``.pos`` ->
motor ticks -> degrees), reusing the same real-robot calibration the GR00T mapper
uses. This mapping is not defined anywhere else in the repo; if a future MolmoAct2
checkpoint expects a different joint frame, adjust
:meth:`_sim_radians_to_model_degrees` / :meth:`_model_degrees_to_sim_radians`.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from collections import deque

import numpy as np
import torch

from so101_bench.utils.lerobot_calibration import (
    LEROBOT_JOINT_ORDER,
    STS3215_DEGREES_PER_TICK,
    lerobot_position_to_motor_position,
    lerobot_position_to_sim_degrees,
    motor_position_to_lerobot_position,
    sim_degrees_to_lerobot_position,
)


class MolmoAct2RemotePolicy:
    """Chunked-action MolmoAct2 HTTP policy wrapper for the SO-101 env."""

    def __init__(
        self,
        device: str,
        cameras: dict[str, dict[str, int]],
        host: str = "localhost",
        port: int = 8000,
        action_horizon: int = 30,
        lang_instruction: str = "Place each object in the plastic bin.",
        camera_names: list[str] | None = None,
        image_size: tuple[int, int] | None = (640, 480),
        timeout_s: float = 120.0,
        num_steps: int = 10,
        max_joint_step_deg: float = 15.0,
    ):
        self.device = device
        self.cameras = cameras
        self.host = host
        self.port = port
        self.action_horizon = action_horizon
        self.lang_instruction = lang_instruction
        self.camera_names = list(camera_names) if camera_names else list(cameras)
        if len(self.camera_names) != 2:
            raise ValueError(
                "MolmoAct2-SO100_101 expects exactly two camera views, got "
                f"{self.camera_names!r}. Pass --policy_cameras as a comma-separated pair."
            )
        self.image_size = image_size
        self.timeout_s = timeout_s
        self.num_steps = num_steps
        self.max_joint_step_deg = max_joint_step_deg

        # Overhead-init (working-memory) conditioning is a GR00T-only feature; the
        # eval script still probes these attributes, so expose them as inert here.
        self.use_overhead_init = False
        self.overhead_init_camera = "overhead"
        self.overhead_init_key = "overhead_init"
        self.overhead_init_image: np.ndarray | None = None

        self.action_queue: deque[np.ndarray] = deque()
        self._last_command_deg: np.ndarray | None = None
        self._connected = False

    # ------------------------------------------------------------------ connection

    @property
    def _base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def connect(self) -> None:
        print(f"[INFO]: Connecting to MolmoAct2 policy server at {self.host}:{self.port}...")
        request = urllib.request.Request(f"{self._base_url}/act", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot reach MolmoAct2 policy server at {self._base_url}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise RuntimeError(f"Unexpected MolmoAct2 health response: {payload!r}")
        self._connected = True
        print(f"[INFO]: Policy server connected (norm_tag={payload.get('norm_tag')}).")

    def _post_act(self, state_deg: np.ndarray, images_png_base64: list[str]) -> np.ndarray:
        body = json.dumps(
            {
                "images_png_base64": images_png_base64,
                "instruction": self.lang_instruction,
                "state": [float(value) for value in state_deg],
                "num_steps": int(self.num_steps),
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/act",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"MolmoAct2 inference request failed ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"MolmoAct2 inference request failed: {exc}") from exc
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(f"MolmoAct2 server error: {payload['error']}")
        actions = np.asarray(payload["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 6:
            raise RuntimeError(f"Expected an action chunk with shape (T, 6), got {actions.shape}.")
        return actions

    # --------------------------------------------------------------------- episode

    def reset(self, initial_visual_obs: dict | None = None, *, reset_remote: bool = True) -> None:
        del reset_remote  # the MolmoAct2 server is stateless per request
        self.action_queue.clear()
        self._last_command_deg = None
        self.overhead_init_image = None
        if initial_visual_obs is not None:
            self.set_episode_initial_observation(initial_visual_obs)

    def set_language_instruction(self, instruction: str) -> None:
        self.lang_instruction = instruction
        self.action_queue.clear()

    def set_episode_initial_observation(self, visual_obs: dict) -> None:
        """No-op for MolmoAct2 (no working-memory overhead-init conditioning)."""

        if not self.use_overhead_init:
            return
        self.overhead_init_image = self._camera_frame(visual_obs, self.overhead_init_camera).copy()

    # ------------------------------------------------------------------ image prep

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

    def _resize_rgb(self, image: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
        if size is None:
            return image
        width, height = size
        if image.shape[1] == width and image.shape[0] == height:
            return image
        try:
            import cv2

            return np.ascontiguousarray(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
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
        return self._resize_rgb(self._to_uint8_rgb(image), self.image_size)

    @staticmethod
    def _encode_png_base64(image: np.ndarray) -> str:
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    # --------------------------------------------------------------- joint mapping

    @staticmethod
    def _sim_radians_to_model_degrees(sim_radians: np.ndarray) -> np.ndarray:
        degrees = np.empty(6, dtype=np.float32)
        for index, joint in enumerate(LEROBOT_JOINT_ORDER):
            sim_deg = float(sim_radians[index]) * 180.0 / np.pi
            lerobot_pos = sim_degrees_to_lerobot_position(joint, sim_deg)
            motor_position = lerobot_position_to_motor_position(joint, lerobot_pos)
            degrees[index] = motor_position * STS3215_DEGREES_PER_TICK
        return degrees

    @staticmethod
    def _model_degrees_to_sim_radians(model_degrees: np.ndarray) -> np.ndarray:
        radians = np.empty(6, dtype=np.float32)
        for index, joint in enumerate(LEROBOT_JOINT_ORDER):
            motor_position = float(model_degrees[index]) / STS3215_DEGREES_PER_TICK
            lerobot_pos = motor_position_to_lerobot_position(joint, motor_position)
            sim_deg = lerobot_position_to_sim_degrees(joint, lerobot_pos)
            radians[index] = sim_deg * np.pi / 180.0
        return radians

    # -------------------------------------------------------------------- stepping

    def _build_state(self, joint_positions: torch.Tensor) -> np.ndarray:
        sim_radians = joint_positions.detach().cpu().numpy().astype(np.float32).reshape(-1)
        if sim_radians.shape != (6,):
            raise ValueError(f"Expected a 6-DOF joint position vector, got shape {sim_radians.shape}.")
        return self._sim_radians_to_model_degrees(sim_radians)

    def _query_server(self, joint_positions: torch.Tensor, visual_obs: dict) -> None:
        state_deg = self._build_state(joint_positions)
        images_png_base64 = [
            self._encode_png_base64(self._camera_frame(visual_obs, camera)) for camera in self.camera_names
        ]
        action_chunk = self._post_act(state_deg, images_png_base64)
        horizon = min(action_chunk.shape[0], self.action_horizon)
        for step in range(horizon):
            row = action_chunk[step]
            if not np.isfinite(row).all():
                raise ValueError(f"MolmoAct2 returned a non-finite action at chunk step {step}: {row.tolist()}")
            self.action_queue.append(row.astype(np.float32))

    def _clamp_step(self, command_deg: np.ndarray) -> np.ndarray:
        """Limit the per-tick change in each joint target (degrees)."""

        if self._last_command_deg is None or self.max_joint_step_deg <= 0.0:
            self._last_command_deg = command_deg.copy()
            return command_deg
        delta = np.clip(
            command_deg - self._last_command_deg,
            -self.max_joint_step_deg,
            self.max_joint_step_deg,
        )
        clamped = self._last_command_deg + delta
        self._last_command_deg = clamped.copy()
        return clamped

    def get_action(self, joint_positions: torch.Tensor, visual_obs: dict) -> torch.Tensor:
        if not self._connected:
            raise RuntimeError("Call connect() before requesting actions.")
        if self._last_command_deg is None:
            # Seed the step clamp from the current pose so the first command cannot
            # jump more than max_joint_step_deg away from where the arm actually is.
            self._last_command_deg = self._build_state(joint_positions)
        if not self.action_queue:
            self._query_server(joint_positions, visual_obs)
        command_deg = self._clamp_step(self.action_queue.popleft())
        sim_radians = self._model_degrees_to_sim_radians(command_deg)
        return torch.from_numpy(sim_radians).to(device=self.device, dtype=torch.float32)
