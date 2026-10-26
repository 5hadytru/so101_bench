#!/usr/bin/env python3
"""Serve a LeRobot pi0.5 checkpoint over the same small HTTP API as molmoact2_server.py.

Wire protocol (shared with scripts/molmoact2_server.py, so molmoact2_eval.py and the
RunPod monitor can drive either model):

    GET  /act -> health check {status, repo_id, norm_tag, num_images, camera_keys}
    POST /act -> {
        "images_png_base64": [str, ...],
        "camera_names": ["overhead", "wrist", "overhead_init", "wrist_init"],
        "instruction": str,
        "state": [6 floats, LeRobot units: body joints in [-100, 100], gripper in [0, 100]],
        "num_steps": int,   # flow-matching denoising steps
    }

The response is ``{"actions": [[6 floats], ...], "dt_ms": float}`` in LeRobot units, one row
per step of the checkpoint's action chunk. Images are matched to the checkpoint's
``observation.images.<name>`` features by name, not position.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

IMAGE_PREFIX = "observation.images."

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pi05.so101.server")


def _decode_png(encoded: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))


class Policy:
    def __init__(self, *, checkpoint: str, device: str, default_num_steps: int):
        log.info("Loading pi0.5 checkpoint %s", checkpoint)
        self.policy = PI05Policy.from_pretrained(checkpoint)
        self.policy.to(device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": device}},
        )
        self.image_features = [k for k in self.policy.config.image_features]
        self.camera_keys = [k[len(IMAGE_PREFIX):] if k.startswith(IMAGE_PREFIX) else k for k in self.image_features]
        self.checkpoint = checkpoint
        self.device = device
        self.default_num_steps = default_num_steps
        log.info("Cameras %s, chunk_size %d", self.camera_keys, self.policy.config.chunk_size)

    @torch.inference_mode()
    def predict(self, *, images: dict[str, np.ndarray], instruction: str, state: Any, num_steps: int | None) -> np.ndarray:
        missing = sorted(set(self.camera_keys) - set(images))
        extra = sorted(set(images) - set(self.camera_keys))
        if missing or extra:
            raise ValueError(f"camera mismatch: checkpoint expects {self.camera_keys}; missing {missing}, unexpected {extra}")
        state_array = np.asarray(state, dtype=np.float32).reshape(1, -1)
        batch: dict[str, Any] = {
            "observation.state": torch.from_numpy(state_array),
            "task": [instruction],
        }
        for name, image in images.items():
            tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)
            batch[IMAGE_PREFIX + name] = tensor.unsqueeze(0)
        self.policy.config.num_inference_steps = int(num_steps or self.default_num_steps)
        batch = self.preprocessor(batch)
        chunk = self.policy.predict_action_chunk(batch)
        if chunk.ndim == 2:
            chunk = chunk.unsqueeze(0)
        actions = torch.stack([self.postprocessor(chunk[:, i, :]) for i in range(chunk.shape[1])], dim=1)[0]
        return actions.float().cpu().numpy()


def build_app(policy: Policy) -> FastAPI:
    app = FastAPI(title="pi0.5 SO-101 inference server", version="0.1.0")

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "repo_id": policy.checkpoint,
                "norm_tag": "pi05",
                "num_images": len(policy.camera_keys),
                "camera_keys": policy.camera_keys,
                "device": policy.device,
            }
        )

    @app.post("/act")
    async def act(http_request: Request) -> JSONResponse:
        try:
            payload = await http_request.json()
            encoded = payload["images_png_base64"]
            names = payload.get("camera_names") or policy.camera_keys
            if len(names) != len(encoded):
                raise ValueError(f"{len(encoded)} images but {len(names)} camera_names")
            images = {name: _decode_png(data) for name, data in zip(names, encoded)}
            instruction = str(payload["instruction"])
            state = payload["state"]
            num_steps = int(payload.get("num_steps", policy.default_num_steps))
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": f"invalid request: {exc}"}, status_code=400)
        start = time.perf_counter()
        try:
            actions = policy.predict(images=images, instruction=instruction, state=state, num_steps=num_steps)
        except Exception as exc:  # noqa: BLE001
            log.exception("pi0.5 inference failed")
            return JSONResponse({"error": f"inference failed: {exc}"}, status_code=500)
        return JSONResponse({"actions": actions.tolist(), "dt_ms": (time.perf_counter() - start) * 1000.0})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="pi0.5 SO-101 HTTP inference server.")
    parser.add_argument("--repo_id", required=True, help="Exported checkpoint directory (pretrained_model layout).")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_steps", type=int, default=10, help="Flow-matching denoising steps.")
    parser.add_argument("--dtype", default="bfloat16", help="Accepted for CLI parity with molmoact2_server.py.")
    parser.add_argument("--no_warmup", action="store_true")
    args = parser.parse_args()
    if not Path(args.repo_id).is_dir():
        raise SystemExit(f"--repo_id must be a local export directory: {args.repo_id}")
    policy = Policy(checkpoint=args.repo_id, device=args.device, default_num_steps=args.num_steps)
    if not args.no_warmup:
        blank = {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in policy.camera_keys}
        start = time.perf_counter()
        out = policy.predict(images=blank, instruction="warmup", state=[0.0] * 6, num_steps=None)
        log.info("Warmup: chunk %s in %.1f ms", out.shape, (time.perf_counter() - start) * 1000.0)

    import uvicorn

    uvicorn.run(build_app(policy), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
