#!/usr/bin/env python3
"""Compute the 16D normalization statistics of so101_bench_sim once (the trainer would redo it every launch).

Builds the train dataset of task=so101_flux2_klein_4b_ee16 with pretrained_norm_stats unset, which makes
RobotVideoDataset compute the stats through the processor's action/state transforms (SO101ToCanonicalEE16)
and write dataset_stats.json into the work dir; the file is then copied to OUT.

  compute_stats.py OUT [hydra overrides...]
"""

import shutil
import sys
import tempfile
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from imagewam.utils import misc
from imagewam.utils.config_resolvers import register_default_resolvers

REPO = Path(__file__).resolve().parents[1] / "ImageWAM"

register_default_resolvers()
out = Path(sys.argv[1])
with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
    cfg = compose(config_name="train", overrides=["task=so101_flux2_klein_4b_ee16", "data.train.pretrained_norm_stats=null", *sys.argv[2:]])
work = Path(tempfile.mkdtemp(prefix="so101_stats_"))
misc.register_work_dir(str(work))
ds = instantiate(cfg.data.train)
print(f"train samples: {len(ds)}")
shutil.copy(work / "dataset_stats.json", out)
print(f"wrote {out}")
