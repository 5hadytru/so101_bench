#!/usr/bin/env bash
# MolmoAct2-SO100_101 full fine-tune on so101_bench_sim_WM (4-camera, checkpoint joint frame).
# Every knob is an env var so the smoke test, the benchmarks and the real run share one command.
set -euo pipefail

WORK=/workspace/molmoact2-so101
REPO=/workspace/molmoact2
# shellcheck disable=SC1091
source /root/molmoact2-venv/bin/activate

export LEROBOT_DATA_ROOT=/workspace/datasets/lerobot
export MOLMO_DATA_DIR=$WORK/molmo_data      # required by the loader even with no VLM data
export HF_HOME=$WORK/hf_home
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # everything is pre-downloaded; never stall on the Hub mid-run
export LEROBOT_VIDEO_BACKEND=${VIDEO_BACKEND:-pyav}
export TMPDIR=$WORK/tmp
export TRITON_CACHE_DIR=$WORK/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$WORK/cache/inductor
export WANDB_DIR=$WORK/wandb
export WANDB_PROJECT=${WANDB_PROJECT:-finetune-molmoact2-so101}
export WANDB_ENTITY=${WANDB_ENTITY:-5hadytru}
export WANDB_MODE=${WANDB_MODE:-online}
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$WORK/secrets/wandb_api_key" ]; then
    WANDB_API_KEY=$(cat "$WORK/secrets/wandb_api_key")
    export WANDB_API_KEY
fi
export PYTHONPATH="$REPO/experiments:$REPO/experiments/lerobot/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
mkdir -p "$MOLMO_DATA_DIR" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$WANDB_DIR"

NPROC=${NPROC:-4}
RUN_NAME=${RUN_NAME:-molmoact2-so101-sim-wm}
SAVE_FOLDER=${SAVE_FOLDER:-$WORK/checkpoints/$RUN_NAME}

cd "$REPO/experiments"
exec torchrun --standalone --nproc-per-node="$NPROC" \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2-SO100_101 \
  so101_bench_sim_wm \
  --run_name="$RUN_NAME" \
  --wandb.name="$RUN_NAME" \
  --max_duration="${MAX_DURATION:?set MAX_DURATION}" \
  --device_batch_size="${DEVICE_BS:-2}" \
  --global_batch_size="${GLOBAL_BS:-256}" \
  --num_workers="${NUM_WORKERS:-6}" --pin_memory=true \
  --prefetch_factor="${PREFETCH:-4}" \
  --data.timeout=900 \
  --save_interval="${SAVE_INTERVAL:-1000}" \
  --save_num_checkpoints_to_keep="${KEEP:-3}" \
  --save_interval_ephemeral="${SAVE_EPHEMERAL:-null}" \
  --save_folder="$SAVE_FOLDER" \
  --log_interval="${LOG_INTERVAL:-10}" \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=true \
  --ft_action_expert=true \
  --ft_embedding=lm_head \
  --lora_enable=false \
  --llm_learning_rate="${LLM_LR:-2e-5}" \
  --vit_learning_rate="${VIT_LR:-1e-5}" \
  --connector_learning_rate="${CONNECTOR_LR:-1e-5}" \
  --action_expert_learning_rate="${AE_LR:-1e-4}" \
  ${EXTRA_ARGS:-}
