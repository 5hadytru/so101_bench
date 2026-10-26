#!/usr/bin/env python3
"""Metric geometry for the VFM planner: MoGe-3 (ViT-L, Ruicheng/moge-3-vitl) behind a tiny HTTP
endpoint, run in its own venv (~/workspace/vfm_server/.venv) so the frozen PyRUA runtime is untouched.

POST /depth  {"image_png_b64": "...", "fov_x_deg": 63.0}  ->  {"depth_npy_b64": "..."}
             float32, H x W, metres along the camera's z axis (NaN where MoGe-3 masks the pixel out)

The camera's true horizontal field of view is passed in (MoGe-3 then needs no focal estimate); the
planner still fits scale and offset per image against the known table plane.  Model choice
(2026-10-24): MoGe-3 (released 2026-08-18, arXiv 2607.17967; MoGe-2 base + self-guided sparse 3D
refiner for fine detail, state of the art on the standard zero-shot geometry benchmarks); MoGe-2 was
already the most accurate current monocular model with a per-image scale fit (arXiv 2610.27199).
Semantics (instance masks) come from SAM 3 inside the planner's process.

    ~/workspace/vfm_server/.venv/bin/python scripts/vfm_depth_server.py --port 8010
"""

from __future__ import annotations

import argparse
import base64
import io

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

MODEL = "Ruicheng/moge-3-vitl"


class Req(BaseModel):
    image_png_b64: str
    fov_x_deg: float | None = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    from moge.model.v3 import MoGeModel

    model = MoGeModel.from_pretrained(MODEL).eval()
    state = {"device": args.device}
    try:
        model.to(args.device)
    except RuntimeError as exc:  # no room on the GPU: start on the CPU
        print("GPU load failed, CPU mode:", exc, flush=True)
        state["device"] = "cpu"
    app = FastAPI()
    served = []  # provenance: how every request was run

    @app.get("/health")
    def health():
        return {"ok": True, "model": MODEL, "device": state["device"], "served": served[-20:]}

    def run(img, fov, device, level, refine):
        x = torch.tensor(img / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.inference_mode():
            return model.infer(x, fov_x=fov, resolution_level=level, refine_steps=refine, use_fp16=device != "cpu")

    @app.post("/depth")
    def depth(req: Req):
        img = np.asarray(Image.open(io.BytesIO(base64.b64decode(req.image_png_b64))).convert("RGB"))
        out, mode = None, None
        # MoGe-3 on the GPU (full, then a lower resolution); last resort: the CPU without the sparse
        # refiner (FlexGEMM/Triton is GPU-only), i.e. MoGe-3's base network
        attempts = [(state["device"], 9, 3), (state["device"], 6, 3)] if state["device"] != "cpu" else []
        attempts.append(("cpu", 9, 0))
        for device, level, refine in attempts:
            try:
                if device == "cpu" and next(model.parameters()).device.type != "cpu":
                    model.to("cpu")
                    state["device"] = "cpu"
                out = run(img, req.fov_x_deg, device, level, refine)
                mode = f"{device} level {level} refine {refine}"
                break
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                print("depth attempt failed:", device, level, refine, str(exc)[:120], flush=True)
                torch.cuda.empty_cache()
        if out is None:
            raise RuntimeError("MoGe-3 failed on every device")
        served.append(mode)
        print("served:", mode, flush=True)
        d = out["depth"].float().cpu().numpy()
        if "mask" in out:
            d = np.where(out["mask"].cpu().numpy(), d, np.nan)
        buf = io.BytesIO()
        np.save(buf, d.astype(np.float32))
        return {"depth_npy_b64": base64.b64encode(buf.getvalue()).decode(), "mode": mode}

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
