#!/usr/bin/env bash
# Launch a LaWAM fine-tune (LeRobot adapter) on so101_bench_sim from jialei02/lawam-pretrain-lerobot on this
# 4xA100 pod. Every knob is an environment variable so the sweep (bench.py) and the full run share one path:
#   MICRO (per-GPU batch), ACCUM (gradient accumulation), UPDATES (optimizer steps; LeRobot counts
#   microsteps = UPDATES x ACCUM), LR, WARMUP (updates), DECAY (updates; cosine end, default UPDATES),
#   AH (action_horizon = predicted chunk; the LAM target frame is the chunk's last), NAS (n_action_steps),
#   SAVE_UPDATES, WORKERS, CLIP (clip_normalized_actions), WANDB (0|1), RUN_NAME, RESUME (1 = resume RUN_NAME).
# LeRobot steps the LR schedule every micro-batch (accelerate step_scheduler_with_optimizer=False), so
# WARMUP/DECAY, given in optimizer updates, are passed as micro-batches (x ACCUM).
# Cameras keep the pretrain roles: overhead -> observation.images.image (primary, LAM), wrist -> image2.
# Official recipe defaults (LIBERO SFT / LeRobot docs): LR 1e-4 cosine to 5e-7, 1500 warmup, AdamW
# (0.9, 0.95) wd 1e-8, clip 1.0, bf16, no image augmentation, no robot-state input (flow_use_state false).
set -euo pipefail
ROOT=${ROOT:-/root/lawam}
MICRO=${MICRO:-16}
ACCUM=${ACCUM:-2}
UPDATES=${UPDATES:-100000}
LR=${LR:-1e-4}
WARMUP=${WARMUP:-1500}
DECAY=${DECAY:-${UPDATES}}
AH=${AH:-36}
NAS=${NAS:-${AH}}
SAVE_UPDATES=${SAVE_UPDATES:-1000}
WORKERS=${WORKERS:-8}
CLIP=${CLIP:-true}
WANDB=${WANDB:-0}
RUN_NAME=${RUN_NAME:-lawam-dev}
RUN=${ROOT}/runs/${RUN_NAME}
export HF_HOME=${ROOT}/.hf-home HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
[[ "${WANDB}" == 1 ]] && export WANDB_API_KEY=$(cat "${ROOT}/secrets/wandb_api_key")
unset HF_TOKEN
cd "${ROOT}/lerobot"
mkdir -p "${ROOT}/runs"
echo "$(date -u +%FT%TZ) [run_train] ${RUN_NAME}: micro=${MICRO} accum=${ACCUM} global=$((MICRO * ACCUM * 4)) updates=${UPDATES} lr=${LR} warmup=${WARMUP} decay=${DECAY} ah=${AH} nas=${NAS} workers=${WORKERS} clip=${CLIP} resume=${RESUME:-0}"
launch=(.venv/bin/accelerate launch --num_processes=4 --multi_gpu --mixed_precision=bf16 -m lerobot.scripts.lerobot_train)
if [[ "${RESUME:-0}" == 1 ]]; then
    exec "${launch[@]}" --config_path="${RUN}/checkpoints/last/pretrained_model/train_config.json" --resume=true
fi
exec "${launch[@]}" \
    --policy.path="${ROOT}/weights/lawam-pretrain-lerobot" --policy.device=cuda --policy.push_to_hub=false \
    --policy.action_horizon="${AH}" --policy.n_action_steps="${NAS}" \
    --policy.clip_normalized_actions="${CLIP}" \
    --policy.optimizer_lr="${LR}" --policy.scheduler_warmup_steps=$((WARMUP * ACCUM)) --policy.scheduler_decay_steps=$((DECAY * ACCUM)) \
    --dataset.repo_id=5hadytru/so101_bench_sim --dataset.root="${ROOT}/data/so101_bench_sim" \
    --dataset.image_transforms.enable=false \
    '--rename_map={"observation.images.overhead":"observation.images.image","observation.images.wrist":"observation.images.image2"}' \
    --accelerator.mixed_precision=bf16 --accelerator.gradient_accumulation.steps="${ACCUM}" \
    --batch_size="${MICRO}" --steps=$((UPDATES * ACCUM)) --save_freq=$((SAVE_UPDATES * ACCUM)) --log_freq=$((10 * ACCUM)) \
    --num_workers="${WORKERS}" --eval_steps=0 --output_dir="${RUN}" --job_name="${RUN_NAME}" \
    --wandb.enable=$([[ "${WANDB}" == 1 ]] && echo true || echo false) --wandb.project=finetune-lawam-so101 --wandb.disable_artifact=true \
    "$@"
