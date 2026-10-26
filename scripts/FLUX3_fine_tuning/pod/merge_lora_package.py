#!/usr/bin/env python3
"""Merge a LeRobot FLUX 3 LoRA checkpoint into a complete SO-101 package for flux3_server.py --package.

Run in the LeRobot venv. The adapter (``pretrained_model`` or ``pretrained_model_ema``) is applied to
its base package, merged with PEFT's merge_and_unload (adapters folded into the projections, the
trained action heads taken from modules_to_save), and the merged DiT is written as the package's
``model.safetensors`` (bf16, LeRobot's ``dit.*`` keys). Config and processors come from the base
package; the adapter's processors must match it (same normalization). Both comparison arms then run
through the same standalone inference code.

  merge_lora_package.py ADAPTER_DIR BASE_PACKAGE OUT_PACKAGE
"""

from __future__ import annotations

import filecmp
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import save_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.flux3 import Flux3Policy

VAE = "/root/flux3/weights/base/video_vae.safetensors"
TEXT_ENCODER = "/root/flux3/weights/base/text_encoder"


def main() -> None:
    adapter, base, out = (Path(a) for a in sys.argv[1:4])
    # Normalization state must match; the processor JSONs differ only by the training rename_map, which
    # the standalone loader rejects, so the base package's JSONs are kept.
    for name in (p.name for p in base.glob("policy_*processor_step_*.safetensors")):
        if not filecmp.cmp(adapter / name, base / name, shallow=False):
            raise SystemExit(f"{name} differs between adapter and base package")
    config = PreTrainedConfig.from_pretrained(str(base))
    config.video_vae_id, config.text_encoder_id = VAE, TEXT_ENCODER
    policy = Flux3Policy.from_pretrained(str(base), config=config)
    model = PeftModel.from_pretrained(policy, str(adapter))
    merged = model.merge_and_unload()
    state = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in merged.state_dict().items() if k.startswith("dit.")}
    reference = Flux3Policy.from_pretrained(str(base), config=config).state_dict()
    missing = sorted(set(k for k in reference if k.startswith("dit.")) - set(state))
    extra = sorted(set(state) - set(reference))
    if missing or extra:
        raise SystemExit(f"merged keys differ from the base: missing {missing[:5]}, extra {extra[:5]}")
    changed = sum(not torch.equal(state[k], reference[k].to(torch.bfloat16)) for k in state)
    print(f"{len(state)} tensors, {changed} changed by the adapter")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for item in base.iterdir():
        if item.name != "model.safetensors" and item.resolve().is_file():
            shutil.copy2(item.resolve(), out / item.name)
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt", "merged_adapter": str(adapter)})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
