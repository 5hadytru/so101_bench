#!/usr/bin/env python3
"""Serve the released MolmoAct2 SO-100/101 checkpoint over a small HTTP API.

Wire protocol:

    GET  /act -> health check
    POST /act -> {
        "images_png_base64": [str, str],
        "instruction": str,
        "state": [float, float, float, float, float, float],
        "num_steps": int,
    }

The response is ``{"actions": [[...], ...], "dt_ms": float}``.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import nullcontext
import io
import logging
import time
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ID = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"
DEFAULT_NUM_STEPS = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("molmoact2.so101.server")


def _decode_png(encoded: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def _load_processor(local_dir: str):
    try:
        return AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        log.warning(
            "Retrying processor load without legacy extra_special_tokens metadata "
            "for compatibility with this transformers version."
        )
        return AutoProcessor.from_pretrained(
            local_dir,
            trust_remote_code=True,
            extra_special_tokens={},
        )


class Policy:
    """Own the loaded checkpoint and serialize inference calls through FastAPI."""

    def __init__(
        self,
        *,
        repo_id: str,
        device: str,
        dtype: torch.dtype,
        enable_cuda_graph: bool,
        default_num_steps: int,
    ):
        local_dir = snapshot_download(repo_id=repo_id)
        log.info("Resolved checkpoint snapshot: %s", local_dir)
        log.info("Loading processor")
        self.processor = _load_processor(local_dir)
        log.info("Loading model (dtype=%s, device=%s)", dtype, device)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                local_dir,
                trust_remote_code=True,
                dtype=dtype,
            )
            .to(device)
            .eval()
        )
        self.repo_id = repo_id
        self.device = device
        self.dtype = dtype
        self.enable_cuda_graph = enable_cuda_graph
        self.default_num_steps = default_num_steps

    def _autocast(self):
        if self.dtype == torch.float32 or not self.device.startswith("cuda"):
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.dtype)

    @torch.inference_mode()
    def predict(
        self,
        *,
        images: list[Image.Image],
        instruction: str,
        state: Any,
        num_steps: int | None,
    ) -> np.ndarray:
        if len(images) != 2:
            raise ValueError(f"MolmoAct2-SO100_101 expects two RGB images, got {len(images)}.")
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_array.shape != (6,):
            raise ValueError(f"MolmoAct2-SO100_101 state must have shape (6,), got {state_array.shape}.")

        with self._autocast():
            output = self.model.predict_action(
                processor=self.processor,
                images=images,
                task=instruction,
                state=state_array,
                norm_tag=NORM_TAG,
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=num_steps or self.default_num_steps,
                normalize_language=True,
                enable_cuda_graph=self.enable_cuda_graph,
            )
        raw_actions = output.actions
        if torch.is_tensor(raw_actions):
            raw_actions = raw_actions.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions = np.asarray(raw_actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[1] != 6:
            raise RuntimeError(f"Expected action chunk with shape (T, 6), got {actions.shape}.")
        return actions


def build_app(policy: Policy) -> FastAPI:
    app = FastAPI(title="MolmoAct2 SO-101 inference server", version="0.1.0")

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "repo_id": policy.repo_id,
                "norm_tag": NORM_TAG,
                "device": policy.device,
                "dtype": str(policy.dtype),
            }
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/act")
    async def act(http_request: Request) -> JSONResponse:
        try:
            payload = await http_request.json()
            images = [_decode_png(encoded) for encoded in payload["images_png_base64"]]
            instruction = str(payload["instruction"])
            state = payload["state"]
            num_steps = int(payload.get("num_steps", policy.default_num_steps))
        except Exception as exc:
            return JSONResponse({"error": f"invalid request: {exc}"}, status_code=400)

        start = time.perf_counter()
        try:
            actions = policy.predict(
                images=images,
                instruction=instruction,
                state=state,
                num_steps=num_steps,
            )
        except Exception as exc:
            log.exception("MolmoAct2 inference failed")
            return JSONResponse({"error": f"inference failed: {exc}"}, status_code=500)
        dt_ms = (time.perf_counter() - start) * 1000.0
        return JSONResponse({"actions": actions.tolist(), "dt_ms": dt_ms})

    return app


def warmup(policy: Policy) -> None:
    log.info("Warming up model (cuda_graph=%s)", policy.enable_cuda_graph)
    image = Image.new("RGB", (640, 480))
    state = np.asarray([0.0, 180.0, 180.0, 60.0, 0.0, 0.0], dtype=np.float32)
    start = time.perf_counter()
    policy.predict(images=[image, image], instruction="warmup", state=state, num_steps=None)
    log.info("Warmup completed in %.1f ms", (time.perf_counter() - start) * 1000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MolmoAct2 SO-100/101 HTTP inference server.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument("--repo_id", default=REPO_ID, help="Hugging Face checkpoint repository or local directory.")
    parser.add_argument("--device", default="cuda:0", help="Torch device.")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS, help="Continuous-flow solver iterations.")
    parser.add_argument("--cuda_graph", action="store_true", help="Enable CUDA graph capture for faster inference.")
    parser.add_argument("--no_warmup", action="store_true", help="Skip the startup inference pass.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_steps < 1:
        raise ValueError(f"--num_steps must be at least 1, got {args.num_steps}.")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    policy = Policy(
        repo_id=args.repo_id,
        device=args.device,
        dtype=dtype,
        enable_cuda_graph=args.cuda_graph,
        default_num_steps=args.num_steps,
    )
    if not args.no_warmup:
        warmup(policy)

    import uvicorn

    log.info("Listening on %s:%d", args.host, args.port)
    uvicorn.run(build_app(policy), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
