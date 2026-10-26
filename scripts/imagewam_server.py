#!/usr/bin/env python3
"""Serve an ImageWAM (FLUX.2 klein-base-4B) SO-101 checkpoint over the HTTP API of molmoact2_server.py /
internw0_server.py.

Runs in the ImageWAM venv. The model, processor (SO101ToCanonicalEE16 + z-score) and canvas come from the
training run's resolved config.yaml, so inference matches training:
- Canvas: each camera uint8 -> [0, 1] -> Resize(240x320, antialias) as the processor's transforms, then the
  dataset's concat (compact_288x256: overhead 192x256 on top, wrist 96x128 bottom-left, a zero third camera
  bottom-right; or the vertical two-tile canvas), the dataset's final resize/center-crop, and [-1, 1].
- Text: the instruction rendered as training's cache did (scripts/flux2/precompute_flux2_qwen3_embeds.py:
  DEFAULT_PROMPT, Qwen3 chat template without thinking, padded to the run's qwen_context_len, the FLUX.2
  Qwen3 output layers stacked), passed to infer_action as context, so train and inference share one text
  path. Encoded once per instruction.
- State: the processor's action/state transform and normalizer (masked dims zeroed), as upstream's
  RoboTwin deploy policy.

ImageWAM is memoryless (current canvas + state + text), so there is no per-episode state. Each request
predicts the run's full chunk (num_frames - 1 actions) and returns the first --replan_steps.

    GET  /act -> {status, repo_id, num_images, camera_keys, history_ticks: 1, chunk_size, n_action_steps}
    POST /act -> {"images_png_base64": [overhead, wrist], "camera_names": ["overhead", "wrist"],
                  "instruction": str, "state": [6 floats, LeRobot units], "episode": int}
    -> {"actions": [[6 floats] x replan_steps], "dt_ms": float}
"""

from __future__ import annotations

import argparse
import base64
import gc
import inspect
import io
import json
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
log = logging.getLogger("imagewam.so101.server")


def _decode_png(encoded: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))


class Policy:
    def __init__(self, args: argparse.Namespace):
        repo = Path(args.repo).resolve()
        sys.path[:0] = [str(repo / "src"), str(repo / "third_party" / "flux2" / "src"), str(repo / "third_party" / "flux2")]
        from einops import rearrange
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from imagewam.datasets.dataset_utils import CenterCrop, ResizeSmallestSideAspectPreserving
        from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT, RobotVideoDataset
        from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        from imagewam.models.backbones.flux2_imports import ensure_flux2_importable

        self._rearrange = rearrange
        cfg = OmegaConf.load(args.train_config)
        data = cfg.data.train
        shape_meta = data.processor.shape_meta
        if [str(m.key) for m in shape_meta.images] != ["overhead", "wrist"]:
            raise SystemExit(f"expected overhead, wrist cameras; got {[m.key for m in shape_meta.images]}")
        self.camera_hw = [int(x) for x in shape_meta.images[0].shape[-2:]]
        self.concat = str(data.concat_multi_camera)
        self.num_output_cameras = int(data.processor.num_output_cameras)
        self.layout_sizes = RobotVideoDataset._robotwin_camera_sizes(str(data.robotwin_camera_layout))
        video_h, video_w = (int(x) for x in data.video_size)
        self.final_resize = ResizeSmallestSideAspectPreserving(args={"img_w": video_w, "img_h": video_h})
        self.final_crop = CenterCrop(args={"img_w": video_w, "img_h": video_h})
        self.action_horizon = int(data.num_frames) - 1
        self.replan_steps = min(args.replan_steps, self.action_horizon)
        self.num_inference_steps = args.num_inference_steps
        self.seed = args.seed
        self.prompt_template = DEFAULT_PROMPT
        self.context_len = int(data.qwen_context_len)

        dtype = torch.bfloat16 if str(cfg.mixed_precision) == "bf16" else torch.float32
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        model_cfg.update(flux2_src_path=str(repo / "third_party" / "flux2"), flux2_model_path=args.flux2_model_path,
                         ae_model_path=args.ae_model_path, qwen3_model_spec=args.qwen3_path, load_text_encoder=False)
        target = model_cfg["_target_"].rsplit(".", 1)
        target_fn = getattr(__import__(target[0], fromlist=[target[1]]), target[1])
        allowed = set(inspect.signature(target_fn).parameters) | {"_target_"}
        model_cfg = {k: v for k, v in model_cfg.items() if k in allowed}
        self.model = instantiate(OmegaConf.create(model_cfg), model_dtype=dtype, device=args.device)
        self.model.load_checkpoint(args.checkpoint)
        self.model = self.model.to(args.device).eval()

        self.processor = instantiate(data.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(args.stats))
        self.state_key = shape_meta.state[0].key
        self.action_key = shape_meta.action[0].key

        ensure_flux2_importable(str(repo / "third_party" / "flux2"))
        from flux2.text_encoder import OUTPUT_LAYERS_QWEN3

        self.output_layers = OUTPUT_LAYERS_QWEN3
        self.tokenizer = AutoTokenizer.from_pretrained(args.qwen3_path)
        self.text_model = AutoModelForCausalLM.from_pretrained(args.qwen3_path, torch_dtype=dtype).to(args.device).eval()
        self.text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.device = args.device
        if args.precompute_instructions:
            self._precompute_text(args.precompute_instructions)
        self.checkpoint = args.checkpoint
        self.dump_dir = Path(args.dump_inputs_dir) if args.dump_inputs_dir else None
        self.dumped: set = set()
        log.info("ImageWAM SO-101 policy: canvas=%s %s cameras=%d chunk=%d replan=%d denoise=%d text_len=%d", self.concat,
                 data.video_size, self.num_output_cameras, self.action_horizon, self.replan_steps,
                 self.num_inference_steps, self.context_len)

    @torch.inference_mode()
    def _precompute_text(self, jsonl: str) -> None:
        """Encode every instruction in an episodes JSONL, then free the Qwen3 text model.

        The embeddings come from the same GPU forward pass as on-demand encoding; they are kept in
        CPU memory and copied to the GPU per request (a device copy leaves bf16 values unchanged).
        This frees ~8 GB of VRAM so the server and Isaac fit on one 24 GB card.
        """
        instructions = sorted({json.loads(line)["instruction"].strip()
                               for line in Path(jsonl).read_text().splitlines() if line.strip()})
        for instruction in instructions:
            hidden, mask = self._encode_text(instruction)
            prompt = self.prompt_template.format(task=instruction)
            self.text_cache[prompt] = (hidden.cpu(), mask.cpu())
        del self.text_model
        self.text_model = None
        gc.collect()
        torch.cuda.empty_cache()
        log.info("Precomputed %d instruction embeddings from %s; text model freed", len(instructions), jsonl)

    def _encode_text(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = self.prompt_template.format(task=instruction.strip())
        if prompt in self.text_cache:
            hidden, mask = self.text_cache[prompt]
            return hidden.to(self.device), mask.to(self.device)
        if self.text_model is None:
            raise RuntimeError(f"instruction not in --precompute_instructions: {instruction!r}")
        if prompt not in self.text_cache:
            rendered = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                          add_generation_prompt=True, enable_thinking=False)
            enc = self.tokenizer([rendered], return_tensors="pt", padding="max_length", truncation=True,
                                 max_length=self.context_len)
            out = self.text_model(input_ids=enc.input_ids.to(self.device), attention_mask=enc.attention_mask.to(self.device),
                                  output_hidden_states=True, use_cache=False)
            hidden = torch.stack([out.hidden_states[k] for k in self.output_layers], dim=1)
            hidden = self._rearrange(hidden, "b c l d -> b l (c d)").to(torch.bfloat16)
            self.text_cache[prompt] = (hidden, enc.attention_mask.to(self.device).bool())
        return self.text_cache[prompt]

    def _canvas(self, images: dict[str, np.ndarray]) -> torch.Tensor:
        cams = []
        for key in ("overhead", "wrist"):
            x = torch.from_numpy(np.ascontiguousarray(images[key])).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W] uint8
            for trans in self.processor.val_transforms:  # ToTensor, Resize(240x320), as training
                x = trans(x)
            if list(x.shape[-2:]) != self.camera_hw:
                raise RuntimeError(f"camera tile {tuple(x.shape)} vs {self.camera_hw}")
            cams.append(x)
        while len(cams) < self.num_output_cameras:
            cams.append(torch.zeros_like(cams[0]))  # the processor's zero-padded camera
        if self.concat == "robotwin":
            top, left, right = (transforms_F.resize(c, size=s, interpolation=transforms_F.InterpolationMode.BILINEAR,
                                                    antialias=True) for c, s in zip(cams, self.layout_sizes))
            video = torch.cat([top, torch.cat([left, right], dim=-1)], dim=-2)
        elif self.concat == "vertical":
            video = torch.cat(cams, dim=-2)
        elif self.concat == "horizontal":
            video = torch.cat(cams, dim=-1)
        else:
            raise RuntimeError(f"unsupported concat {self.concat}")
        video = self.final_crop(self.final_resize(video))
        return video.mul(2.0).sub(1.0)  # Normalize(mean 0.5, std 0.5)

    @torch.inference_mode()
    def predict(self, payload: dict[str, Any]) -> np.ndarray:
        names = payload["camera_names"]
        images = dict(zip(names, (_decode_png(x) for x in payload["images_png_base64"]), strict=True))
        canvas = self._canvas(images)
        episode = payload.get("episode")
        if self.dump_dir is not None and episode not in self.dumped:
            self.dumped.add(episode)  # the model's own input canvas at this episode's first decision
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            img = ((canvas[0] + 1.0) * 127.5).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
            Image.fromarray(img).save(self.dump_dir / f"ep{episode}_model_input.png")
        state = torch.as_tensor(np.asarray(payload["state"], dtype=np.float32)).unsqueeze(0)
        batch = self.processor.action_state_transform({"state": {self.state_key: state}})
        proprio = self.processor.normalizer.forward(batch)["state"][self.state_key]
        context, context_mask = self._encode_text(str(payload["instruction"]))
        pred = self.model.infer_action(
            prompt=None, context=context, context_mask=context_mask,
            input_image=canvas.to(device=self.model.device, dtype=self.model.torch_dtype),
            action_horizon=self.action_horizon, proprio=proprio.to(self.model.device),
            num_inference_steps=self.num_inference_steps, seed=self.seed, rand_device="cpu",
        )
        action = pred["action"]
        if action.ndim == 2:
            action = action.unsqueeze(0)
        denorm = self.processor.normalizer.normalizers["action"][self.action_key].backward(action.float().cpu())
        out = {"action": {self.action_key: denorm}}
        for trans in reversed(self.processor.action_state_transforms or []):
            out = trans.backward(out)  # 16D canonical -> 6D SO-101 joints
        chunk = out["action"][self.action_key][0].numpy()
        return chunk[: self.replan_steps].astype(np.float32)


def build_app(policy: Policy) -> FastAPI:
    app = FastAPI(title="ImageWAM SO-101 inference server", version="0.1.0")

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "repo_id": policy.checkpoint, "norm_tag": "imagewam",
                             "num_images": 2, "camera_keys": ["overhead", "wrist"], "history_ticks": 1,
                             "chunk_size": policy.action_horizon, "n_action_steps": policy.replan_steps})

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
            log.exception("ImageWAM inference failed")
            return JSONResponse({"error": f"inference failed: {exc}"}, status_code=500)
        return JSONResponse({"actions": actions.tolist(), "dt_ms": (time.perf_counter() - start) * 1000.0})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="/root/imagewam/ImageWAM")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_config", required=True, help="the run's resolved config.yaml")
    parser.add_argument("--stats", required=True, help="the run's dataset_stats.json")
    parser.add_argument("--flux2_model_path", default="/root/imagewam/checkpoints/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors")
    parser.add_argument("--ae_model_path", default="/root/imagewam/checkpoints/flux2/FLUX.2-dev/ae.safetensors")
    parser.add_argument("--qwen3_path", default="/root/imagewam/checkpoints/Qwen3-4B")
    parser.add_argument("--replan_steps", type=int, default=16, help="Execute half the 32-step chunk.")
    parser.add_argument("--num_inference_steps", type=int, default=10, help="Upstream sim-eval default.")
    parser.add_argument("--seed", type=int, default=0, help="Upstream deploy default.")
    parser.add_argument("--dump_inputs_dir", default=None, help="Save each episode's first model-input canvas.")
    parser.add_argument("--precompute_instructions", default=None,
                        help="Episodes JSONL: encode all its instructions at startup, then free the text model.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    policy = Policy(args)
    import uvicorn

    uvicorn.run(build_app(policy), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
