#!/usr/bin/env python3
"""Export a LeRobot pi0.5 checkpoint for local validation.

  export_checkpoint.py <run>/checkpoints/<step> <out_dir>

Copies pretrained_model_ema (the EMA weights; falls back to pretrained_model) with its
config, train_config and processor pipelines/stats. model.safetensors is cast to bf16
(the server runs in bf16); processor statistics are left untouched. EXPORT_OK is
written last, so a partially written export is never pulled.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main(ckpt: str, out: str) -> None:
    ckpt_dir, out_dir = Path(ckpt).resolve(), Path(out)
    src = ckpt_dir / "pretrained_model_ema"
    if not src.is_dir():
        src = ckpt_dir / "pretrained_model"
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    weights = tmp / "model.safetensors"
    if not weights.is_file():
        raise SystemExit(f"{src} has no model.safetensors (sharded checkpoint formats are not supported here)")
    tensors = load_file(weights)
    cast = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v).contiguous() for k, v in tensors.items()}
    size_gb = sum(v.numel() * v.element_size() for v in cast.values()) / 1e9
    save_file(cast, weights.with_suffix(".tmp"), metadata={"format": "pt"})
    weights.with_suffix(".tmp").replace(weights)
    (tmp / "export_source.json").write_text(json.dumps(
        {"source": str(src), "step": int(ckpt_dir.name), "ema": src.name.endswith("_ema"),
         "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "weights_gb": round(size_gb, 2)},
        indent=2))
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.replace(out_dir)
    (out_dir / "EXPORT_OK").touch()
    print(f"EXPORTED {out_dir} from {src.name} ({size_gb:.2f} GB bf16)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
