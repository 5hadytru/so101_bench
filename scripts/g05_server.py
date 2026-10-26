#!/usr/bin/env python3
"""HTTP policy server for G0.5 (GalaxeaVLA) checkpoints, speaking scripts/molmoact2_eval.py's protocol.

Builds the policy and processor with GalaxeaVLA's own serve_policy_mem.py helpers (config from the
checkpoint's run dir, dataset stats, bf16 weights), so preprocessing matches training exactly. The client's
`overhead` camera goes to the model's `exterior` slot and `wrist` to `wrist_right` (configs/data/so101_bench.yaml).
State and returned actions are the dataset's native LeRobot units (`--joint_frame lerobot` on the client).

Visual memory (obs_size > 1): the server keeps every request frame of the current episode and feeds the frames
`k * frames_per_stride` requests back (k = 0..obs_size-1, clamped to the episode start), where
frames_per_stride = obs_stride_second * fps / replan_steps. That reproduces training's frame spacing exactly when
replan_steps divides obs_stride_second * fps (e.g. 15 or 30 at 30 Hz with a 1 s stride).

POST /act  {"images_png_base64": [...], "camera_names": ["overhead","wrist"], "instruction": str,
            "state": [6 floats], "episode": int}  ->  {"actions": [[6] * T], "dt_ms": float}
GET  /act  health: {"status": "ok", "history_ticks": 1, "num_images": 2, ...}
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

log = logging.getLogger("g05_server")
CLIENT_TO_MODEL_CAMERA = {"overhead": "exterior", "wrist": "wrist_right"}


def decode_chw(encoded: str) -> np.ndarray:
    rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))  # [C, H, W] uint8, raw resolution


class G05Policy:
    def __init__(self, args: argparse.Namespace):
        repo = Path(args.repo).resolve()
        sys.path.insert(0, str(repo / "src"))
        sys.path.insert(0, str(repo / "scripts"))
        import os

        os.chdir(repo)  # configs reference repo-relative checkpoints/ and configs/ paths
        import serve_policy_mem as spm
        from g05.models.g05.inferencer import PolicyInferencer

        self.spm = spm
        # GalaxeaVLA's serve setup builds the full fp32 model on the CPU (13.5 GB) before loading weights, which
        # stalled the 31 GB workstation (2026-10-06). Its meta-device path memory-maps the weights onto the GPU.
        import functools
        spm.load_model_from_checkpoint = functools.partial(spm.load_model_from_checkpoint, use_meta_device=True)
        overrides = [
            f"model.model_arch.discrete_action={str(args.mode == 'ar').lower()}",
            f"model.model_arch.continuous_action={str(args.mode == 'flow').lower()}",
            "model.model_weights_to_bf16=true",
            f"model.use_torch_compile={str(args.compile).lower()}",
            "model.model_arch.attn_implementation=sdpa",
            *args.overrides,
        ]
        if args.task_config:
            cfg = spm.load_config_from_task_yaml(args.task_config, args.ckpt_path, overrides)
        else:
            cfg = spm.load_config_from_run_dir(spm.find_run_dir(args.ckpt_path), args.ckpt_path, overrides)
        if "embodiment_datasets" in cfg.data:
            spm.filter_embodiment(cfg, args.embodiment)
        self.policy, self.processor = spm.setup(cfg, device=args.device)
        torch.cuda.empty_cache()  # drop the fp32 staging blocks; Isaac Sim shares this 24 GB GPU
        self.inferencer = PolicyInferencer(self.policy, self.processor, device=args.device)
        p = spm.resolve_processor(self.processor, {"embodiment_type": args.embodiment})
        self.num_obs_steps = int(p.num_obs_steps)
        self.cam_keys = [m["key"] for m in p.shape_meta["images"]]
        self.state_key = p.shape_meta["state"][0]["key"]
        stride_s = float(cfg.data.get("obs_stride_second", 0.0) or 0.0)
        self.frames_per_stride = 1
        if self.num_obs_steps > 1:
            exact = stride_s * args.fps / args.replan_steps
            self.frames_per_stride = max(1, round(exact))
            if abs(exact - self.frames_per_stride) > 1e-6:
                log.warning("obs stride %.2fs = %.2f replans; rounding to %d (frame spacing differs from training)",
                            stride_s, exact, self.frames_per_stride)
        self.fps, self.embodiment, self.mode = args.fps, args.embodiment, args.mode
        self.episode, self.history = None, []
        self.checkpoint = args.ckpt_path
        log.info("G0.5 %s policy: cameras=%s obs_size=%d frames_per_stride=%d replan=%d", args.mode, self.cam_keys,
                 self.num_obs_steps, self.frames_per_stride, args.replan_steps)

    @torch.inference_mode()
    def predict(self, payload: dict) -> np.ndarray:
        episode = payload.get("episode")
        if episode != self.episode:
            self.episode, self.history = episode, []
        frames = {CLIENT_TO_MODEL_CAMERA[name]: decode_chw(img)
                  for name, img in zip(payload["camera_names"], payload["images_png_base64"], strict=True)}
        state = np.asarray(payload["state"], dtype=np.float32)
        self.history.append((frames, state))
        # Frames k * frames_per_stride requests back, oldest first, clamped to the episode's first frame.
        idx = [max(0, len(self.history) - 1 - k * self.frames_per_stride) for k in reversed(range(self.num_obs_steps))]
        # Camera slots the client does not have (released checkpoints also expect wrist_left) get black frames,
        # as GalaxeaVLA's own SO-100 client sends.
        blank = np.zeros_like(next(iter(frames.values())))
        frame_buffers = {cam: [torch.from_numpy(self.history[i][0].get(cam, blank)) for i in idx] for cam in self.cam_keys}
        state_buffers = {self.state_key: [torch.from_numpy(self.history[i][1]) for i in idx]}
        raw_obs = {"images": {cam: frames.get(cam, blank) for cam in self.cam_keys}, "state": {self.state_key: state},
                   "task": str(payload["instruction"]), "frequency": float(self.fps), "embodiment_type": self.embodiment}
        obs = self.spm.build_obs_dict(raw_obs, self.processor, frame_buffers=frame_buffers, state_buffers=state_buffers)
        action = self.inferencer.infer([obs])[0]
        chunk = action[self.state_key]
        chunk = chunk[0] if chunk.ndim == 3 else chunk
        return np.asarray(chunk.float().cpu().numpy() if torch.is_tensor(chunk) else chunk, dtype=np.float32)


def build_app(policy: G05Policy) -> FastAPI:
    app = FastAPI()

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "repo_id": policy.checkpoint, "norm_tag": "g05", "num_images": 2,
                             "camera_keys": ["overhead", "wrist"], "history_ticks": 1, "obs_size": policy.num_obs_steps,
                             "mode": policy.mode})

    @app.post("/act")
    async def act(request: Request) -> JSONResponse:
        payload = await request.json()
        t0 = time.perf_counter()
        try:
            actions = policy.predict(payload)
        except Exception as exc:  # noqa: BLE001
            log.exception("inference failed")
            return JSONResponse({"error": repr(exc)})
        return JSONResponse({"actions": actions.tolist(), "dt_ms": (time.perf_counter() - t0) * 1000.0})

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="/root/g05/GalaxeaVLA")
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--task_config", default=None, help="task YAML when the checkpoint has no run dir")
    ap.add_argument("--embodiment", default="so100")
    ap.add_argument("--mode", choices=["ar", "flow"], default="ar", help="AR action tokens (paper default) or flow head")
    ap.add_argument("--replan_steps", type=int, default=15, help="actions the client executes per request")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("overrides", nargs="*", help="extra Hydra overrides")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import uvicorn

    uvicorn.run(build_app(G05Policy(args)), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
