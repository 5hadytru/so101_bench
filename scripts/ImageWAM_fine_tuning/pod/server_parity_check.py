#!/usr/bin/env python3
"""Train/inference input parity for scripts/imagewam_server.py (CPU only; no ImageWAM model is loaded).

For a few training samples, compares what the trainer feeds the model (RobotVideoDataset with augmentation
off: canvas video[:, 0], cached Qwen3 context and mask, proprio[0]) with what the server builds from the
same raw uint8 frames, instruction and state (Policy._canvas, Policy._encode_text, the processor's state
path). Identical inputs mean the server reproduces training exactly up to the model call.

  server_parity_check.py RUN_CONFIG.yaml STATS.json [--samples 3]
"""

import argparse
import sys
import tempfile
import types

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

sys.path.insert(0, "/root/imagewam/pod")
import imagewam_server as srv  # noqa: E402
from imagewam.datasets.dataset_utils import CenterCrop, ResizeSmallestSideAspectPreserving  # noqa: E402
from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT, RobotVideoDataset  # noqa: E402
from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from imagewam.utils import misc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("stats")
ap.add_argument("--samples", type=int, default=3)
ap.add_argument("--qwen3", default="/root/imagewam/checkpoints/Qwen3-4B")
a = ap.parse_args()
torch.set_num_threads(8)

cfg = OmegaConf.load(a.config)
data = OmegaConf.to_container(cfg.data.train, resolve=True)
data.update(is_training_set=False, video_augmentation=None, pretrained_norm_stats=a.stats)
misc.register_work_dir(tempfile.mkdtemp())
ds = instantiate(OmegaConf.create(data))

# A server Policy without the image-editing model: the same preprocessing attributes as Policy.__init__.
from einops import rearrange  # noqa: E402
from flux2.text_encoder import OUTPUT_LAYERS_QWEN3  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

p = srv.Policy.__new__(srv.Policy)
d = cfg.data.train
p._rearrange = rearrange
p.camera_hw = [int(x) for x in d.processor.shape_meta.images[0].shape[-2:]]
p.concat = str(d.concat_multi_camera)
p.num_output_cameras = int(d.processor.num_output_cameras)
p.layout_sizes = RobotVideoDataset._robotwin_camera_sizes(str(d.robotwin_camera_layout))
vh, vw = (int(x) for x in d.video_size)
p.final_resize = ResizeSmallestSideAspectPreserving(args={"img_w": vw, "img_h": vh})
p.final_crop = CenterCrop(args={"img_w": vw, "img_h": vh})
p.prompt_template = DEFAULT_PROMPT
p.context_len = int(d.qwen_context_len)
p.processor = instantiate(d.processor).eval()
p.processor.set_normalizer_from_stats(load_dataset_stats_from_json(a.stats))
p.output_layers = OUTPUT_LAYERS_QWEN3
p.tokenizer = AutoTokenizer.from_pretrained(a.qwen3)
p.text_model = AutoModelForCausalLM.from_pretrained(a.qwen3, torch_dtype=torch.bfloat16).eval()
p.text_cache = {}
p.device = "cpu"
state_key = d.processor.shape_meta.state[0].key

rng = np.random.default_rng(0)
base = ds.lerobot_dataset
for idx in rng.choice(len(ds), a.samples, replace=False).tolist():
    s = ds[idx]
    proc = base.processor
    base.processor = None  # raw sample: uint8 images [2 endpoint frames, 3, H, W], raw 6D state, task text
    raw = base[idx]
    base.processor = proc
    images = {k: raw["images"][k][0].permute(1, 2, 0).numpy() for k in ("overhead", "wrist")}
    with torch.inference_mode():
        canvas = p._canvas(images)[0]
        ctx, mask = p._encode_text(str(raw["task"]))
        batch = p.processor.action_state_transform({"state": {state_key: raw["state"]["default"][:1].float()}})
        proprio = p.processor.normalizer.forward(batch)["state"][state_key][0]
    train_canvas = s["video"][:, 0].float()
    print(f"idx {idx} task={raw['task']!r}")
    print(f"  canvas  max|diff| {(canvas - train_canvas).abs().max():.2e}  mean {(canvas - train_canvas).abs().mean():.2e}  shape {tuple(canvas.shape)} vs {tuple(train_canvas.shape)}")
    c, t = ctx[0].float(), s["text_hidden_states"].float()
    m = s["text_attention_mask"].bool()
    rel = ((c - t)[m].norm() / t[m].norm()).item()
    cos = torch.nn.functional.cosine_similarity(c[m], t[m], dim=-1)
    print(f"  context max|diff| {(c - t).abs().max():.2e} (max|ref| {t.abs().max():.2e})  rel L2 over valid tokens {rel:.2e}  "
          f"min token cos {cos.min():.6f}  valid tokens {int(m.sum())}  mask equal {bool((mask[0] == m).all())}")
    print(f"  proprio max|diff| {(proprio - s['proprio'][0].float()).abs().max():.2e}")
