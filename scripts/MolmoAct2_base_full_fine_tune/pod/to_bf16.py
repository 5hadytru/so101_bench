#!/usr/bin/env python3
"""Rewrite an HF MolmoAct2 export's float32 safetensors shards as bfloat16, in place.

The inference server loads with dtype=bfloat16 anyway, so this only halves the
transfer and disk size (about 21 GB -> 11 GB).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main(export_dir: str) -> None:
    root = Path(export_dir)
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    total = 0
    for shard in sorted(set(index["weight_map"].values())):
        path = root / shard
        tensors = load_file(path)
        converted = {
            name: (t.to(torch.bfloat16) if t.is_floating_point() and t.dtype == torch.float32 else t).contiguous()
            for name, t in tensors.items()
        }
        total += sum(t.numel() * t.element_size() for t in converted.values())
        save_file(converted, path.with_suffix(".tmp"), metadata={"format": "pt"})
        path.with_suffix(".tmp").replace(path)
    index.setdefault("metadata", {})["total_size"] = total
    index_path.write_text(json.dumps(index, indent=2))
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    for key in ("torch_dtype", "dtype"):
        if key in config:
            config[key] = "bfloat16"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"bf16 export: {total / 1e9:.2f} GB in {root}")


if __name__ == "__main__":
    main(sys.argv[1])
