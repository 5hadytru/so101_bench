#!/usr/bin/env python3
"""Serve a LeRobot LaWAM checkpoint (lerobot.policies.lawam) over the HTTP API of molmoact2_server.py /
pi05_server.py, so molmoact2_eval.py and the validation monitors drive it like the other policies.

The checkpoint's own preprocessor and postprocessor (saved by lerobot-train next to the weights) do all
input handling, exactly as in training: resize to 256x256, Qwen3-VL chat inputs, MIN_MAX normalization
with so101_bench_sim's statistics, and clip/unnormalize on the way out. LaWAM takes no robot state
(flow_use_state false); the state is accepted and ignored. The SO-101 cameras keep the pretrain roles
under the names the run was trained with (--camera_map, default overhead -> image, wrist -> image2).

    GET  /act -> {status, repo_id, norm_tag, num_images, camera_keys, history_ticks: 1, chunk_size, n_action_steps}
    POST /act -> {"images_png_base64": [overhead, wrist], "camera_names": ["overhead", "wrist"],
                  "instruction": str, "state": [6 floats, LeRobot units]}
    -> {"actions": [[6 floats] x action_horizon], "dt_ms": float}   (LeRobot units; the client executes a prefix)
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
from lerobot.policies.lawam.modeling_lawam import LaWAMPolicy

IMAGE_PREFIX = "observation.images."

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("lawam.so101.server")


def _decode_png(encoded: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))


class Policy:
    def __init__(self, *, checkpoint: str, device: str, camera_map: dict[str, str], dump_inputs_dir: str | None):
        log.info("Loading LaWAM checkpoint %s", checkpoint)
        self.policy = LaWAMPolicy.from_pretrained(checkpoint)
        self.policy.to(device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": device}},
        )
        self.camera_map = camera_map
        features = set(self.policy.config.image_features)
        targets = {IMAGE_PREFIX + v for v in camera_map.values()}
        if targets != features:
            raise SystemExit(f"--camera_map targets {sorted(targets)} != checkpoint image features {sorted(features)}")
        self.camera_keys = list(camera_map)
        self.checkpoint = checkpoint
        self.device = device
        self.dump_dir = Path(dump_inputs_dir) if dump_inputs_dir else None
        self.dumped: set = set()
        cfg = self.policy.config
        log.info("Cameras %s, action_horizon %d, n_action_steps %d, primary %s, wrist %s, lam %s", camera_map,
                 cfg.action_horizon, cfg.n_action_steps, cfg.primary_image_features, cfg.wrist_image_features,
                 cfg.lam_image_feature)

    @torch.inference_mode()
    def predict(self, *, images: dict[str, np.ndarray], instruction: str, episode: Any = None) -> np.ndarray:
        missing = sorted(set(self.camera_keys) - set(images))
        if missing:
            raise ValueError(f"camera mismatch: server expects {self.camera_keys}; missing {missing}")
        batch: dict[str, Any] = {"task": [instruction]}
        for name in self.camera_keys:
            tensor = torch.from_numpy(np.ascontiguousarray(images[name])).permute(2, 0, 1).float().div_(255.0)
            batch[IMAGE_PREFIX + self.camera_map[name]] = tensor.unsqueeze(0)
        if self.dump_dir is not None and episode != "warmup" and episode not in self.dumped:
            self.dumped.add(episode)  # the raw frames this episode's first decision was made from
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            for name in self.camera_keys:
                Image.fromarray(images[name]).save(self.dump_dir / f"ep{episode}_{name}.png")
        batch = self.preprocessor(batch)
        chunk = self.policy.predict_action_chunk(batch)
        if chunk.ndim == 2:
            chunk = chunk.unsqueeze(0)
        actions = torch.stack([self.postprocessor(chunk[:, i, :]) for i in range(chunk.shape[1])], dim=1)[0]
        return actions.float().cpu().numpy()


def build_app(policy: Policy) -> FastAPI:
    app = FastAPI(title="LaWAM SO-101 inference server", version="0.1.0")
    cfg = policy.policy.config

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "repo_id": policy.checkpoint, "norm_tag": "lawam",
                             "num_images": len(policy.camera_keys), "camera_keys": policy.camera_keys,
                             "history_ticks": 1, "chunk_size": cfg.action_horizon,
                             "n_action_steps": cfg.n_action_steps, "device": policy.device})

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
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": f"invalid request: {exc}"}, status_code=400)
        start = time.perf_counter()
        try:
            actions = policy.predict(images=images, instruction=instruction, episode=payload.get("episode"))
        except ValueError as exc:
            return JSONResponse({"error": f"invalid request: {exc}"}, status_code=400)
        except Exception as exc:  # noqa: BLE001
            log.exception("LaWAM inference failed")
            return JSONResponse({"error": f"inference failed: {exc}"}, status_code=500)
        return JSONResponse({"actions": actions.tolist(), "dt_ms": (time.perf_counter() - start) * 1000.0})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", required=True, help="Checkpoint pretrained_model directory.")
    parser.add_argument("--camera_map", default="overhead=image,wrist=image2",
                        help="client camera name = checkpoint observation.images.<key>, comma-separated.")
    parser.add_argument("--dump_inputs_dir", default=None, help="Save each episode's first raw camera frames.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_warmup", action="store_true")
    args = parser.parse_args()
    if not Path(args.repo_id).is_dir():
        raise SystemExit(f"--repo_id must be a local checkpoint directory: {args.repo_id}")
    camera_map = dict(pair.split("=", 1) for pair in args.camera_map.split(","))
    policy = Policy(checkpoint=args.repo_id, device=args.device, camera_map=camera_map, dump_inputs_dir=args.dump_inputs_dir)
    if not args.no_warmup:
        blank = {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in policy.camera_keys}
        start = time.perf_counter()
        out = policy.predict(images=blank, instruction="warmup", episode="warmup")
        log.info("Warmup: chunk %s in %.1f ms", out.shape, (time.perf_counter() - start) * 1000.0)

    import uvicorn

    uvicorn.run(build_app(policy), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
