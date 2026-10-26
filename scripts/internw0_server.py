#!/usr/bin/env python3
"""Serve an InternW0-delta SO-101 checkpoint over the HTTP API of molmoact2_server.py / flux3_server.py.

Runs in the InternW0-Delta venv (repo root on sys.path). The model, processor and normalization come
from the training run's own Hydra config (task=so101 plus the run's overrides, saved as
``train_config.yaml`` by run_train.sh), so inference matches training: the same 2-camera vertical
canvas (overhead above wrist, each 192x256, 384x256 total, bilinear+antialias, [-1, 1]), the same
80-D merger and z-score statistics, and the same visual memory (episode-first anchor frame + the frame
32 steps back).

InternW0 inference is stateful per episode (memory, action history), so the server keeps one
RobotWinPolicySession and resets it when the request's ``episode`` changes. Replanning every 16
actions (``--replan_steps``): with the 32-step memory offset, the recent frame is then exactly the
observation two replans back, so the chunk-at-a-time client feeds memory the frames training used.

    GET  /act -> {status, repo_id, num_images, camera_keys, history_ticks: 1, chunk_size, n_action_steps}
    POST /act -> {"images_png_base64": [overhead, wrist], "camera_names": ["overhead", "wrist"],
                  "instruction": str, "state": [6 floats, LeRobot units], "episode": int}
    -> {"actions": [[6 floats] x replan_steps], "dt_ms": float}
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("internw0.so101.server")


def _decode_png(encoded: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))


def _camera_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).float().div(255.0)


def build_vertical_canvas(overhead: np.ndarray, wrist: np.ndarray, camera_hw: tuple[int, int]) -> torch.Tensor:
    """Training's `concat_multi_camera: vertical`: per-camera resize to camera_hw, stack, scale to [-1, 1]."""
    cams = [
        transforms_F.resize(_camera_tensor(img), size=list(camera_hw),
                            interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True)
        for img in (overhead, wrist)
    ]
    return torch.cat(cams, dim=-2).mul(2.0).sub(1.0)


class Policy:
    def __init__(self, args: argparse.Namespace):
        repo = Path(args.repo).resolve()
        sys.path[:0] = [str(repo), str(repo / "src")]
        # Model ids like Wan-AI/Wan2.2-TI2V-5B resolve under the repo's checkpoints/ as in training.
        import os

        os.chdir(repo)
        from omegaconf import OmegaConf

        from eval.robotwin.policy import RobotWinPolicySession, build_robotwin_runtime_and_session_config

        cfg = OmegaConf.load(args.train_config)
        shape_meta = cfg.data.train.processor.shape_meta
        if [str(m.key) for m in shape_meta.images] != ["overhead", "wrist"]:
            raise SystemExit(f"expected overhead, wrist cameras; got {[m.key for m in shape_meta.images]}")
        if str(cfg.data.train.concat_multi_camera) != "vertical":
            raise SystemExit("expected the vertical two-camera canvas")
        self.camera_hw = tuple(int(x) for x in shape_meta.images[0].shape[-2:])
        self.memory = bool(cfg.model.memory.enabled)
        # The repo's own evaluation contract (configs/sim_robodojo.yaml), with SO-101's control rate and
        # 16-step replanning; training-time image augmentation is off at inference.
        cfg.EVALUATION = OmegaConf.create({
            "action_horizon": int(cfg.data.train.num_frames) - 1, "replan_steps": args.replan_steps,
            "num_inference_steps": args.num_inference_steps, "action_hz": args.action_hz, "sigma_shift": None,
            "rand_device": "cpu", "tiled": False, "timing_enabled": False, "device": args.device,
            "dataset_stats_path": args.stats,
        })
        # Training-speed settings that do not change the weights: no MoT compilation at inference.
        cfg.model.mot_compile_mode = "off"
        cfg.model.mot_compile_gradient_checkpointing = False
        cfg.data.train.image_color_jitter.enabled = False
        cfg.data.train.image_color_jitter.apply_to_vlm = False
        self.runtime, self.session_cfg = build_robotwin_runtime_and_session_config({
            "resolved_config": cfg, "ckpt_setting": args.checkpoint, "dataset_stats_path": args.stats,
            "seed": args.seed, "device": args.device, "mixed_precision": str(cfg.mixed_precision),
        })
        camera_hw = self.camera_hw

        dump_dir = Path(args.dump_inputs_dir) if args.dump_inputs_dir else None
        dumped: set = set()

        class SO101Session(RobotWinPolicySession):
            def _build_robotwin_image_tensor(self, observation):
                obs = observation["observation"]
                canvas = build_vertical_canvas(obs["overhead"]["rgb"], obs["wrist"]["rgb"], camera_hw)
                episode = observation.get("episode")
                if dump_dir is not None and episode not in dumped:
                    dumped.add(episode)  # the model's own input tiles for this episode's first decision
                    dump_dir.mkdir(parents=True, exist_ok=True)
                    tiles = ((canvas[0] + 1.0) * 127.5).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                    h = camera_hw[0]
                    Image.fromarray(tiles[:h]).save(dump_dir / f"ep{episode}_overhead_model_input.png")
                    Image.fromarray(tiles[h:]).save(dump_dir / f"ep{episode}_wrist_model_input.png")
                if tuple(canvas.shape[-2:]) != (self.video_height, self.video_width):
                    raise RuntimeError(f"canvas {tuple(canvas.shape)} vs video {(self.video_height, self.video_width)}")
                return canvas.to(device=self.model.device, dtype=self.model.torch_dtype)

            def _build_robotwin_vlm_images(self, observation):
                obs = observation["observation"]
                return torch.stack([torch.from_numpy(np.ascontiguousarray(obs[k]["rgb"])).permute(2, 0, 1)
                                    for k in ("overhead", "wrist")]).to(device=self.model.device)

        self.session_cls = SO101Session
        self.replan_steps = self.session_cfg.replan_steps
        self.num_inference_steps = self.session_cfg.num_inference_steps
        self.action_horizon = self.session_cfg.action_horizon
        self.session = None
        self.episode = None
        self.checkpoint = args.checkpoint
        log.info("InternW0 SO-101 policy: anchor+recent frame memory, persistent_kv=%s camera=%s replan=%d denoise=%d", self.memory, camera_hw,
                 self.replan_steps, self.num_inference_steps)

    def _new_session(self):
        return self.session_cls(self.runtime, self.session_cfg)

    @torch.inference_mode()
    def predict(self, payload: dict[str, Any]) -> np.ndarray:
        names = payload["camera_names"]
        images = dict(zip(names, (_decode_png(x) for x in payload["images_png_base64"]), strict=True))
        state = np.asarray(payload["state"], dtype=np.float32)
        episode = payload.get("episode")
        if self.session is None or episode != self.episode:
            self.session = self._new_session()  # new episode: fresh memory and action history
            self.episode = episode
        # The client executed the whole previous chunk: acknowledge it. Skipped ticks carry no frame;
        # memory needs only replan frames and the frame 32 steps back, itself a replan frame at 16.
        while self.session.pending_model_actions:
            self.session.update_obs(None)
        observation = {
            "observation": {"overhead": {"rgb": images["overhead"]}, "wrist": {"rgb": images["wrist"]}},
            "joint_action": {"vector": state},
            "episode": episode,
        }
        chunk = self.session.get_action({"observation": observation, "instruction": str(payload["instruction"])})
        return np.asarray(chunk, dtype=np.float32)


def build_app(policy: Policy) -> FastAPI:
    app = FastAPI(title="InternW0-delta SO-101 inference server", version="0.1.0")

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "repo_id": policy.checkpoint, "norm_tag": "internw0",
                             "num_images": 2, "camera_keys": ["overhead", "wrist"], "history_ticks": 1,
                             "chunk_size": policy.action_horizon, "n_action_steps": policy.replan_steps,
                             "memory": policy.memory})

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
            log.exception("InternW0 inference failed")
            return JSONResponse({"error": f"inference failed: {exc}"}, status_code=500)
        return JSONResponse({"actions": actions.tolist(), "dt_ms": (time.perf_counter() - start) * 1000.0})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="/root/internw0/InternW0-Delta")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_config", required=True, help="the run's resolved train_config.yaml")
    parser.add_argument("--stats", required=True, help="the run's normalization stats JSON")
    parser.add_argument("--replan_steps", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=10, help="Repo eval default.")
    parser.add_argument("--action_hz", type=float, default=30.0, help="SO-101 Bench control rate.")
    parser.add_argument("--seed", type=int, default=0, help="Repo eval default (configs/sim_robodojo.yaml).")
    parser.add_argument("--dump_inputs_dir", default=None,
                        help="Save the preprocessed overhead/wrist model-input tiles of each episode's first decision.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    policy = Policy(args)
    import uvicorn

    uvicorn.run(build_app(policy), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
