#!/usr/bin/env python3
"""Serve a FLUX 3 Action SO-101 policy over the HTTP API of molmoact2_server.py / pi05_server.py.

Runs in the black-forest-labs/flux-action venv. Two checkpoint kinds:

  --export DIR    a standalone flux-action export (export-checkpoint --dtype bfloat16); training
                  exports carry no sampler, so the SO-101 recipe preset below is applied.
  --package DIR   the released black-forest-labs/flux-3-action-so101 package (legacy-degree frame).

FLUX conditions on observation history (the SO-101 recipe: 8 control ticks of measured state and
issued commands, camera snapshots at the oldest and newest tick). The server is stateless: the
client (MolmoAct2RemotePolicy with history_ticks > 1) sends the window with every request.

    GET  /act -> {status, repo_id, norm_tag, num_images, camera_keys, history_ticks, chunk_size, n_action_steps}
    POST /act -> {
        "images_png_base64": [str, ...],            # newest tick, one per camera_names entry
        "snapshot_images_png_base64": [str, ...],   # oldest tick of the window, same order
        "camera_names": ["overhead", "wrist"],
        "instruction": str,
        "state_history": [[6 floats] x history_ticks],    # oldest -> newest measured state
        "command_history": [[6 floats] x history_ticks],  # absolute command issued before each tick
        "trace": {"dir": str, "episode": int, "frame": int},  # optional; see --save_predictions
    }

With ``--save_predictions`` the server also decodes the video FLUX denoises jointly with the actions
(normally discarded) and writes it, for requests that carry ``trace``, to
``<dir>/predicted/ep<episode>_f<frame>.mp4``: the predicted scene|wrist canvas (256x512) for the frames
that follow the request's observation. Predicted frame j lines up with recorded rollout frame
``frame + j`` of molmoact2_eval.py's LeRobot dataset; compose_flux3_prediction_videos.py stitches them.

The response is ``{"actions": [[6 floats] x chunk_size], "dt_ms": float}``: absolute commands in the
checkpoint's units, the predicted deltas integrated onto the newest issued command.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

from einops import rearrange

from flux_action.inference import sampling
from flux_action.inference.so101 import load_policy
from flux_action.policy import FluxActionPolicy
from flux_action.processing import packing

# The SO-101 recipe's sampler (flux-3-action-so101 config; docs/so101-lora.md evaluate command).
SO101_INFERENCE_SETTINGS = {
    "sampler": "euler",
    "num_inference_steps": 4,
    "sampler_shift": 6.93,
    "guidance_scale": 3.0,
    "inference_seed": 42,
}

class _VideoCapture:
    """Keep the last sampler output: the joint video tokens the policy computes and then drops."""

    def __init__(self):
        self.last: dict[str, torch.Tensor] | None = None
        self._euler = sampling.euler
        self._unipc = sampling.cosmos_unipc_order2

        def euler(*args, **kwargs):
            out = self._euler(*args, **kwargs)
            self.last = out
            return out

        def unipc(*args, **kwargs):
            out = self._unipc(*args, **kwargs)
            self.last = out
            return out

        sampling.euler = euler
        sampling.cosmos_unipc_order2 = unipc


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("flux3.so101.server")


def _decode_png(encoded: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))


def _local_package(package: Path, vae: Path, text_encoder: Path) -> Path:
    """Copy of the released package whose config points at local encoders instead of Hub ids."""
    local = package.parent / f"{package.name}_local_encoders"
    local.mkdir(exist_ok=True)
    for item in package.iterdir():
        target = local / item.name
        if item.name == "config.json" or target.exists():
            continue
        target.symlink_to(item.resolve())
    config = json.loads((package / "config.json").read_text())
    config.update(video_vae_id=str(vae), text_encoder_id=str(text_encoder))
    (local / "config.json").write_text(json.dumps(config, indent=2))
    return local


class Policy:
    def __init__(self, args: argparse.Namespace):
        self.device = torch.device(args.device)
        if args.package:
            source = _local_package(Path(args.package), Path(args.vae), Path(args.text_encoder))
            log.info("Loading released SO-101 package %s", args.package)
            self.policy = load_policy(source)
            self.checkpoint = args.package
        else:
            log.info("Loading flux-action export %s", args.export)
            self.policy = FluxActionPolicy.from_pretrained(args.export)
            self.checkpoint = args.export
            for name, value in SO101_INFERENCE_SETTINGS.items():
                if getattr(self.policy.config, name, None) is None or name == "inference_seed":
                    setattr(self.policy.config, name, value)
            self.policy.config.validate_inference()
        self.policy.to(self.device).eval()
        cfg = self.policy.config
        if cfg.inference_profile != "history":
            raise SystemExit("flux3_server.py serves history-profile (SO-101) checkpoints only")
        # images.scene <- overhead, images.wrist <- wrist (camera_keys are in layout order).
        self.camera_keys = [key.removeprefix("images.") for key in cfg.camera_keys]
        self.client_to_key = dict(zip(args.client_cameras.split(","), cfg.camera_keys, strict=True))
        self.history_ticks = cfg.n_obs_steps
        self.capture = _VideoCapture() if args.save_predictions else None
        log.info(
            "Cameras %s <- client %s; history %d ticks, %d snapshots; chunk %d, executes %d; sampler %s x%d, cfg %.1f",
            list(cfg.camera_keys), list(self.client_to_key), cfg.n_obs_steps, cfg.history_snapshots,
            cfg.chunk_size, cfg.n_action_steps, cfg.sampler, cfg.num_inference_steps, cfg.guidance_scale,
        )

    @torch.inference_mode()
    def predict(self, payload: dict[str, Any]) -> np.ndarray:
        cfg = self.policy.config
        names = payload["camera_names"]
        current = [_decode_png(x) for x in payload["images_png_base64"]]
        oldest = [_decode_png(x) for x in payload["snapshot_images_png_base64"]]
        if sorted(names) != sorted(self.client_to_key) or len(current) != len(names) or len(oldest) != len(names):
            raise ValueError(f"expected images for cameras {list(self.client_to_key)}, got {names}")
        states = np.asarray(payload["state_history"], dtype=np.float32)
        commands = np.asarray(payload["command_history"], dtype=np.float32)
        n = self.history_ticks
        if states.shape != (n, cfg.action_dim) or commands.shape != (n, cfg.action_dim):
            raise ValueError(f"state_history/command_history must be ({n}, {cfg.action_dim})")
        batch: dict[str, Any] = {
            "state": torch.from_numpy(states)[None].to(self.device),
            "command_history": torch.from_numpy(commands)[None].to(self.device),
            "task": [str(payload["instruction"])],
        }
        for name, now, old in zip(names, current, oldest):
            # Only the snapshot ticks are encoded (index 0 and n-1 for the SO-101 recipe); the ticks in
            # between repeat the newest frame. Frames stay uint8: flux-action divides by 255 on the CPU
            # itself (GPU normalization changes the BF16 VAE inputs); it wants them on the state's device.
            frames = [old] + [now] * (n - 1)
            stack = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
            batch[self.client_to_key[name]] = stack[None].to(self.device)
        chunk = self.policy.predict_action_chunk(batch)
        trace = payload.get("trace")
        if self.capture is not None and trace:
            self._save_prediction(trace)
        return chunk[0].float().cpu().numpy()

    @torch.inference_mode()
    def _save_prediction(self, trace: dict[str, Any]) -> None:
        cfg = self.policy.config
        out = self.capture.last
        if out is None or "x_video" not in out:
            return
        n_pred = packing.latent_frames(cfg.chunk_size)
        lh, lw = cfg.latent_hw
        tokens = out["x_video"][0]  # (n_pred * lh * lw, 96), the prc_vid "(t h w) c" layout
        latents = rearrange(tokens, "(t h w) c -> 1 c t h w", t=n_pred, h=lh, w=lw)
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            pixels = self.policy.video_vae.decode(latents.to(self.device, torch.bfloat16))
        frames = ((pixels[0].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)  # (3, T, H, W)
        frames = frames.permute(1, 2, 3, 0).cpu().numpy()
        path = Path(trace["dir"]) / "predicted" / f"ep{int(trace['episode']):03d}_f{int(trace['frame']):05d}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_mp4(path, frames, fps=int(round(cfg.fps)))
        log.info("Saved predicted video %s (%d frames, decode+write %.0f ms)", path.name, len(frames),
                 (time.perf_counter() - start) * 1000.0)


def build_app(policy: Policy) -> FastAPI:
    app = FastAPI(title="FLUX 3 Action SO-101 inference server", version="0.1.0")
    cfg = policy.policy.config

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "repo_id": policy.checkpoint,
                "norm_tag": "flux3",
                "num_images": len(policy.client_to_key),
                "camera_keys": list(policy.client_to_key),
                "history_ticks": policy.history_ticks,
                "chunk_size": cfg.chunk_size,
                "n_action_steps": cfg.n_action_steps,
                "device": str(policy.device),
            }
        )

    @app.post("/act")
    async def act(http_request: Request) -> JSONResponse:
        try:
            payload = await http_request.json()
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": f"invalid request: {exc}"}, status_code=400)
        start = time.perf_counter()
        try:
            actions = policy.predict(payload)
        except (KeyError, ValueError) as exc:
            return JSONResponse({"error": f"invalid request: {exc}"}, status_code=400)
        except Exception as exc:  # noqa: BLE001
            log.exception("FLUX inference failed")
            return JSONResponse({"error": f"inference failed: {exc}"}, status_code=500)
        return JSONResponse({"actions": actions.tolist(), "dt_ms": (time.perf_counter() - start) * 1000.0})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--export", help="flux-action export directory (config.json, model.safetensors, manifest.json)")
    source.add_argument("--package", help="released flux-3-action-so101 package directory")
    parser.add_argument("--vae", default="/root/flux3/weights/base/video_vae.safetensors")
    parser.add_argument("--text_encoder", default="/root/flux3/weights/base/text_encoder")
    parser.add_argument("--client_cameras", default="overhead,wrist", help="Client camera names in the checkpoint's layout order.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_warmup", action="store_true")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Decode and save FLUX's predicted video for requests that carry 'trace'.")
    args = parser.parse_args()
    policy = Policy(args)
    if not args.no_warmup:
        cfg = policy.policy.config
        blank = base64.b64encode(_png(np.zeros((480, 640, 3), np.uint8))).decode()
        zeros = [[0.0] * cfg.action_dim] * policy.history_ticks
        payload = {
            "camera_names": list(policy.client_to_key),
            "images_png_base64": [blank] * len(policy.client_to_key),
            "snapshot_images_png_base64": [blank] * len(policy.client_to_key),
            "instruction": "warmup",
            "state_history": zeros,
            "command_history": zeros,
        }
        start = time.perf_counter()
        out = policy.predict(payload)
        log.info("Warmup: chunk %s in %.1f ms (peak %.1f GiB)", out.shape, (time.perf_counter() - start) * 1000.0,
                 torch.cuda.max_memory_allocated(policy.device) / 2**30)

    import uvicorn

    uvicorn.run(build_app(policy), host=args.host, port=args.port, log_level="info")


def _write_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    import av

    with av.open(str(path), "w") as container:
        try:
            stream = container.add_stream("libx264", rate=fps)
            stream.options = {"crf": "23", "preset": "veryfast"}
        except Exception:  # noqa: BLE001 (PyAV builds without x264)
            stream = container.add_stream("mpeg4", rate=fps)
            stream.bit_rate = 4_000_000
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _png(image: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


if __name__ == "__main__":
    main()
