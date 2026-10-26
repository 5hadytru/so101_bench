#!/usr/bin/env python3
"""Export a G0.5 training checkpoint (model + optimizer + scheduler, ~19 GB) to a weights-only bf16 policy bundle
(~6.8 GB) in the layout scripts/g05_server.py loads: <out>/.hydra/config.yaml, <out>/dataset_stats.json,
<out>/action_tokenizer.pt, <out>/checkpoints/model_state_dict.pt. The server casts weights to bf16 anyway
(model_weights_to_bf16=true), so only the parameters it keeps in fp32 lose precision.
  export_checkpoint.py RUN_DIR STEP OUT_DIR
"""
import shutil, sys
from pathlib import Path
import torch

run, step, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
src = run / "checkpoints" / f"step_{step}.pt"
state = torch.load(src, map_location="cpu", weights_only=False, mmap=True)["model_state_dict"]
weights = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous() for k, v in state.items()}
(out / "checkpoints").mkdir(parents=True, exist_ok=True)
(out / ".hydra").mkdir(exist_ok=True)
torch.save({"model_state_dict": weights, "step": step}, out / "checkpoints" / "model_state_dict.pt.tmp")
(out / "checkpoints" / "model_state_dict.pt.tmp").rename(out / "checkpoints" / "model_state_dict.pt")
for f in (".hydra/config.yaml", ".hydra/overrides.yaml", "dataset_stats.json", "action_tokenizer.pt"):
    if (run / f).exists():
        shutil.copy2(run / f, out / f)
# Inline model.tokenizer and model_arch.AT_CONFIG (saved as `${tokenizer}` / `${model.tokenizer.vq_config}`): GalaxeaVLA's loader patches
# model.tokenizer.vq_config.ckpt_dir through that reference when the bundle carries action_tokenizer.pt,
# which replaces the node and drops tokenizer._target_.
from omegaconf import OmegaConf
cfg = OmegaConf.load(out / ".hydra/config.yaml")
cfg.model.tokenizer = OmegaConf.create(OmegaConf.to_container(cfg.tokenizer, resolve=False))
cfg.model.model_arch.AT_CONFIG = OmegaConf.create(OmegaConf.to_container(cfg.tokenizer.vq_config, resolve=False))
OmegaConf.save(cfg, out / ".hydra/config.yaml")
(out / "EXPORT_OK").write_text(f"{src}\n")
print(f"exported {src} -> {out} ({sum(v.numel() for v in weights.values()) / 1e9:.2f}B params)")
