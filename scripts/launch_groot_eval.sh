#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

/home/truman/IsaacLab/isaaclab.sh -p scripts/groot_eval.py \
    --task So101Bench-Bin-v0 \
    --episodes_jsonl tasks/real_gr00t_WM_combined.jsonl \
    --episode_layouts_jsonl tasks/layouts/real_gr00t_WM_combined_layouts.jsonl \
    --seed 0 \
    --record_dataset \
    --repo_root data/lerobot/groot_n16_v7_merge_1 \
    --action_horizon 8 \
    --headless \
    2>&1 | tee "logs/groot_eval_$(date +%Y%m%d_%H%M%S).log"
