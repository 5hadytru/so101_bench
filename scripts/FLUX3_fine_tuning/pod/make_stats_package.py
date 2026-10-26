#!/usr/bin/env python3
"""Copy the released SO-101 package with its saved normalization replaced by so101_bench_sim's own.

The package's processors carry one community dataset's quantiles (VoicAndrei/so100_kitchen, a
legacy-degree frame). The SO-101 checkpoint itself was trained with per-dataset normalization, and
LeRobot's docs say to "prepare a separate base with matching processors" for different statistics.
This writes that base: the weights are symlinked, and the three processor state files and
dataset_statistics.json hold the flux-action index statistics (joint-delta actions with the
gripper absolute, measured state). These are the same quantiles the standalone FFT arm normalizes
with, so both arms of the LoRA/FFT comparison see identical targets.

  make_stats_package.py SRC_PACKAGE INDEX_STATISTICS_JSON DST_PACKAGE
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    src, stats_path, dst = (Path(a) for a in sys.argv[1:4])
    stats = json.loads(stats_path.read_text())
    assert stats["action"]["parameterization"] == "joint_delta" and stats["action"]["absolute_dims"] == [5]
    quantiles = {
        f"{stream}.{q}": torch.tensor(stats[stream][q], dtype=torch.float32)
        for stream in ("action", "state")
        for q in ("q01", "q99")
    }
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for item in src.iterdir():
        if item.name.endswith(".safetensors") and item.name.startswith("policy_"):
            original = load_file(str(item))
            replaced = {}
            for key, value in original.items():
                assert key in quantiles and value.shape == quantiles[key].shape, (item.name, key)
                replaced[key] = quantiles[key].to(value.dtype)
            save_file(replaced, str(dst / item.name))
            print(f"{item.name}: replaced {sorted(replaced)}")
        elif item.name == "dataset_statistics.json":
            payload = json.loads(item.read_text())
            for stream in ("action", "state"):
                for q in ("q01", "q99"):
                    payload[stream][q] = stats[stream][q]
            (dst / item.name).write_text(json.dumps(payload, indent=2) + "\n")
            print("dataset_statistics.json: replaced")
        else:
            (dst / item.name).symlink_to(item.resolve())
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
