#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

/home/truman/IsaacLab/isaaclab.sh -p scripts/molmoact2_eval.py \
    --task So101Bench-Bin-v0 \
    --episodes_jsonl tasks/real_gr00t_WM_combined.jsonl \
    --episode_layouts_jsonl tasks/layouts/real_gr00t_WM_combined_layouts.jsonl \
    --seed 0 \
    --record_dataset \
    --repo_root data/lerobot/molmoact2_real_sim_1_ah30 \
    --action_horizon 30 \
    --headless \
    2>&1 | tee "logs/molmoact2_eval_$(date +%Y%m%d_%H%M%S).log"
