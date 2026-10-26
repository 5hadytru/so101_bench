#!/usr/bin/env bash
# MolmoAct2 fine-tune on so101_bench_sim_WM from MODEL_ID (default allenai/MolmoAct2, the general
# multi-embodiment checkpoint; the 2026-09-28 run started from allenai/MolmoAct2-SO100_101)
# (4 cameras, checkpoint joint frame). FT_MODE selects the official recipe's parameter set
# (allenai/molmoact2 experiments/README.md @ 66b87e6):
#   lora     --ft_vlm=true --ft_action_expert=true --ft_embedding=lm_head --lora_enable=true
#            --lora_rank=64, all four LRs 5e-5 (LoRA adapters on LLM + ViT linears; connector,
#            LM head and action expert trained as full parameters)
#   ae_only  --ft_vlm=false --ft_action_expert=true --ft_embedding=none --lora_enable=false,
#            action-expert LR 5e-5 (VLM, ViT, connector, embeddings, LM head frozen)
#   full     the full_fine_tune run's flags, for reference measurements only
# Every other knob is an env var so the smoke test, the benchmarks and the real run share one command.
set -euo pipefail

WORK=${WORK:-/root/molmoact2-so101}
REPO=${REPO:-/root/molmoact2}
# shellcheck disable=SC1091
source /root/molmoact2-venv/bin/activate

export LEROBOT_DATA_ROOT=${LEROBOT_DATA_ROOT:-/root/datasets/lerobot}
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
# K>1 flow timesteps expand every layer's VLM KV states K times inside the action expert;
# the allocator otherwise strands ~8 GB in fragments (seen at the first OOM).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$MOLMO_DATA_DIR" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$WANDB_DIR"

FT_MODE=${FT_MODE:?set FT_MODE to lora, ae_only or full}
case "$FT_MODE" in
    lora)
        MODE_ARGS=(--ft_vlm=true --ft_action_expert=true --ft_embedding=lm_head
                   --lora_enable=true --lora_rank="${LORA_RANK:-64}"
                   --llm_learning_rate="${LLM_LR:-5e-5}" --vit_learning_rate="${VIT_LR:-5e-5}"
                   --connector_learning_rate="${CONNECTOR_LR:-5e-5}"
                   --action_expert_learning_rate="${AE_LR:-5e-5}") ;;
    ae_only)
        MODE_ARGS=(--ft_vlm=false --ft_action_expert=true --ft_embedding=none
                   --lora_enable=false --action_expert_learning_rate="${AE_LR:-5e-5}") ;;
    full)
        MODE_ARGS=(--ft_vlm=true --ft_action_expert=true --ft_embedding=lm_head --lora_enable=false
                   --llm_learning_rate="${LLM_LR:-1e-5}" --vit_learning_rate="${VIT_LR:-5e-6}"
                   --connector_learning_rate="${CONNECTOR_LR:-5e-6}"
                   --action_expert_learning_rate="${AE_LR:-5e-5}") ;;
    *) echo "unknown FT_MODE=$FT_MODE" >&2; exit 2 ;;
esac

NPROC=${NPROC:-4}
RUN_NAME=${RUN_NAME:-molmoact2-so101-sim-wm-$FT_MODE}
SAVE_FOLDER=${SAVE_FOLDER:-$WORK/checkpoints/$RUN_NAME}

cd "$REPO/experiments"
exec torchrun --standalone --nproc-per-node="$NPROC" \
  launch_scripts/train_lerobot.py \
  "${MODEL_ID:-allenai/MolmoAct2}" \
  so101_bench_sim_wm \
  --run_name="$RUN_NAME" \
  --wandb.name="$RUN_NAME" \
  --max_duration="${MAX_DURATION:?set MAX_DURATION}" \
  --device_batch_size="${DEVICE_BS:-2}" \
  --global_batch_size="${GLOBAL_BS:-64}" \
  --num_workers="${NUM_WORKERS:-8}" --pin_memory=true \
  --prefetch_factor="${PREFETCH:-4}" \
  --data.timeout=900 \
  --save_interval="${SAVE_INTERVAL:-1000}" \
  --save_num_checkpoints_to_keep="${KEEP:-3}" \
  --save_interval_ephemeral="${SAVE_EPHEMERAL:-null}" \
  --save_folder="$SAVE_FOLDER" \
  --log_interval="${LOG_INTERVAL:-10}" \
  --packing="${PACKING:-false}" \
  --dynamic_seq_len=true \
  --num_flow_timesteps="${FLOW_K:-8}" \
  --action_format="${ACTION_FORMAT:-continuous}" \
  "${MODE_ARGS[@]}" \
  ${EXTRA_ARGS:-}
