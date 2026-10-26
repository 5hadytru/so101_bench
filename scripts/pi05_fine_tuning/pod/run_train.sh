#!/usr/bin/env bash
# pi0.5 (lerobot/pi05_base) full fine-tune on so101_bench_sim_WM (4 cameras, LeRobot units),
# following openpi's only pi0.5 sim recipe (pi05_libero), scaled to the time budget:
# global batch 256, AdamW(0.9, 0.95, wd 1e-10, clip 1.0), 10k-of-30k warmup to 5e-5 then constant
# (LeRobot scales the warmup by STEPS/30000), EMA 0.999, 1-second action chunks (30 steps at 30 Hz).
# Every knob is an env var so the smoke test, the sweep and the real run share one command.
# RESUME=1 resumes from $OUTPUT_DIR/checkpoints/last.
set -euo pipefail

WORK=/workspace/pi05-so101
# shellcheck disable=SC1091
source /root/lerobot-venv/bin/activate

export HF_HOME=$WORK/hf_home
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # base model + tokenizer are pre-downloaded
HF_TOKEN=$(cat "$WORK/secrets/hf_token")     # gated PaliGemma tokenizer
export HF_TOKEN
export WANDB_DIR=$WORK/wandb
export WANDB_MODE=${WANDB_MODE:-online}
if [ -z "${WANDB_API_KEY:-}" ]; then
    WANDB_API_KEY=$(cat "$WORK/secrets/wandb_api_key")
    export WANDB_API_KEY
fi
export TMPDIR=$WORK/tmp
export TRITON_CACHE_DIR=$WORK/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$WORK/cache/inductor
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$WANDB_DIR"

NPROC=${NPROC:-4}
RUN_NAME=${RUN_NAME:-pi05-so101-sim-wm}
OUTPUT_DIR=${OUTPUT_DIR:-$WORK/runs/$RUN_NAME}
LAUNCH=(torchrun --standalone --nproc-per-node="$NPROC" "$(which lerobot-train)")

if [ "${RESUME:-0}" = "1" ]; then
    exec "${LAUNCH[@]}" \
      --config_path="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json" \
      --resume=true \
      ${EXTRA_ARGS:-}
fi

exec "${LAUNCH[@]}" \
  --dataset.repo_id=5hadytru/so101_bench_sim_WM \
  --dataset.root=/workspace/datasets/lerobot/5hadytru/so101_bench_sim_WM \
  --dataset.video_backend="${VIDEO_BACKEND:-pyav}" \
  --dataset.image_transforms.enable="${IMAGE_AUG:-true}" \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.gradient_checkpointing="${GRAD_CKPT:-true}" \
  --policy.compile_model="${COMPILE:-false}" \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.chunk_size="${CHUNK:-30}" \
  --policy.n_action_steps="${N_ACTION_STEPS:-15}" \
  --policy.optimizer_lr="${LR:-5e-5}" \
  --policy.optimizer_weight_decay="${WD:-1e-10}" \
  --policy.scheduler_warmup_steps="${WARMUP:-10000}" \
  --policy.scheduler_decay_steps="${DECAY_STEPS:-30000}" \
  --policy.scheduler_decay_lr="${DECAY_LR:-5e-5}" \
  --ema.enable="${EMA:-true}" \
  --ema.decay=0.999 \
  --batch_size="${DEVICE_BS:-8}" \
  --accelerator.gradient_accumulation.steps="${ACCUM:-1}" \
  --parallelism.dp_shard="${DP_SHARD:-1}" \
  --num_workers="${NUM_WORKERS:-8}" \
  --prefetch_factor="${PREFETCH:-4}" \
  --steps="${STEPS:?set STEPS}" \
  --save_freq="${SAVE_FREQ:-1000}" \
  --log_freq="${LOG_FREQ:-20}" \
  --seed=1000 \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$RUN_NAME" \
  --wandb.enable="${WANDB_ENABLE:-true}" \
  --wandb.project=finetune-pi05-so101 \
  --wandb.entity=5hadytru \
  --wandb.disable_artifact=true \
  ${EXTRA_ARGS:-}
