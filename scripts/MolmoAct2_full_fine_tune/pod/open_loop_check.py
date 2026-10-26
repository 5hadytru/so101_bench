#!/usr/bin/env python3
"""Open-loop sanity check of an exported (HF-format) MolmoAct2 checkpoint on dataset frames.

Loads the export the same way scripts/molmoact2_server.py does, feeds the four sim_WM
cameras + state + task for random frames, and reports the mean absolute error between
the predicted 30-step chunk and the recorded actions (checkpoint joint frame, degrees).
This is a smoke test of the export and of the camera/state/tag plumbing, not a
success-rate estimate.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from PIL import Image

CAMERAS = [
    "observation.images.overhead",
    "observation.images.wrist",
    "observation.images.overhead_init",
    "observation.images.wrist_init",
]


def to_pil(chw: torch.Tensor) -> Image.Image:
    array = (chw.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir")
    parser.add_argument("--repo_id", default="5hadytru/so101_bench_sim_WM_molmoframe")
    parser.add_argument("--root", default="/workspace/datasets/lerobot/5hadytru/so101_bench_sim_WM_molmoframe")
    parser.add_argument("--norm_tag", default="so101_bench_sim_wm")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="Optional JSON file for the summary.")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from transformers import AutoModelForImageTextToText, AutoProcessor

    fps = 30
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        delta_timestamps={"action": [i / fps for i in range(args.horizon)]},
        video_backend="pyav",
    )
    processor = AutoProcessor.from_pretrained(args.export_dir, trust_remote_code=True)
    model = (
        AutoModelForImageTextToText.from_pretrained(args.export_dir, trust_remote_code=True, dtype=torch.bfloat16)
        .to(args.device)
        .eval()
    )

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=args.samples, replace=False)
    errors, first_step_errors, times = [], [], []
    for idx in indices:
        item = dataset[int(idx)]
        images = [to_pil(item[key]) for key in CAMERAS]
        state = item["observation.state"].numpy().astype(np.float32)
        target = item["action"].numpy()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model.predict_action(
                processor=processor,
                images=images,
                task=item["task"],
                state=state,
                norm_tag=args.norm_tag,
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=args.num_steps,
                normalize_language=True,
                enable_cuda_graph=False,
            )
        times.append(time.perf_counter() - start)
        actions = output.actions
        actions = actions.detach().float().cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
        actions = actions.reshape(-1, actions.shape[-1])[: args.horizon]
        n = min(len(actions), len(target))
        errors.append(np.abs(actions[:n] - target[:n]).mean(axis=0))
        first_step_errors.append(np.abs(actions[0] - target[0]))
    summary = {
        "export_dir": args.export_dir,
        "samples": int(args.samples),
        "chunk_shape": list(actions.shape),
        "mae_per_joint_deg": np.mean(errors, axis=0).round(3).tolist(),
        "mae_deg": float(np.mean(errors)),
        "first_step_mae_deg": float(np.mean(first_step_errors)),
        "latency_s_median": float(np.median(times)),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
