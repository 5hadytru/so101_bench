#!/usr/bin/env bash
# LoRA arm of the 2026-10-01 LoRA-vs-FFT comparison (equal ~3 h wall clock on 4x A100).
# LeRobot main examples/flux3/lora.json, unchanged except: micro-batch 2 x 4 GPUs x accumulation 16 =
# 128 global (as the FFT arm), gradient checkpointing off (6.40 windows/s in the sweep), no held-out
# split, and the base package with so101_bench_sim's own normalization (make_stats_package.py), so
# both arms train on identical targets. STEPS counts microsteps (LeRobot): 540 updates = 8640.
set -euo pipefail
ROOT=/root/flux3
UPDATES=${UPDATES:-540}
ACCUM=16
OUT=${OUT:-${ROOT}/runs/ab_lora}
MICROSTEPS=$((UPDATES * ACCUM))
cd "${ROOT}/lerobot"
python3 - "${OUT}.lora.json" "${ACCUM}" <<'EOF'
import json, sys
cfg = json.load(open("examples/flux3/lora.json"))
cfg["accelerator"]["gradient_accumulation"]["steps"] = int(sys.argv[2])
json.dump(cfg, open(sys.argv[1], "w"), indent=2)
EOF
export HF_HUB_OFFLINE=1 HF_HOME=${ROOT}/.hf-home PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
unset HF_TOKEN
exec .venv/bin/accelerate launch --num_processes=4 --multi_gpu --mixed_precision=bf16 -m lerobot.scripts.lerobot_train \
    --config_path="${OUT}.lora.json" \
    --policy.path="${ROOT}/weights/so101_ourstats" --policy.device=cuda \
    --policy.video_vae_id="${ROOT}/weights/base/video_vae.safetensors" \
    --policy.text_encoder_id="${ROOT}/weights/base/text_encoder" \
    --policy.gradient_checkpointing=false \
    --dataset.repo_id=5hadytru/so101_bench_sim --dataset.root="${ROOT}/datasets/so101_bench_sim" \
    --dataset.eval_split=0 --eval_steps=0 \
    '--rename_map={"observation.images.overhead":"observation.images.scene"}' \
    --batch_size=2 --steps="${MICROSTEPS}" --save_freq="${MICROSTEPS}" --log_freq=160 \
    --num_workers=4 --wandb.enable=false --output_dir="${OUT}"
