#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=/workspace/Isaac-GR00T
WORK_DIR=/workspace/gr00t-so101-n1d7
DATASET_DIR="$REPO_DIR/examples/SO101/so101_bench_sim_WM_v2.1"
CONFIG_PATH="$REPO_DIR/examples/SO101/so101_config.py"

export XDG_CACHE_HOME="$WORK_DIR/cache"
export HF_HOME="$WORK_DIR/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
# The locked environment has hf-xet, but not the optional hf_transfer package.
export HF_HUB_ENABLE_HF_TRANSFER=0
export UV_CACHE_DIR="$WORK_DIR/cache/uv"
export UV_PYTHON_INSTALL_DIR="$WORK_DIR/python"
export TORCH_HOME="$WORK_DIR/cache/torch"
export TORCH_EXTENSIONS_DIR="$WORK_DIR/cache/torch_extensions"
export TRITON_CACHE_DIR="$WORK_DIR/cache/triton"
export CUDA_CACHE_PATH="$WORK_DIR/cache/nv/ComputeCache"
export WANDB_DIR="$WORK_DIR/wandb"
export TMPDIR="$WORK_DIR/tmp"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" \
    "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$TORCH_HOME" \
    "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" \
    "$WORK_DIR/cache/nv" "$WANDB_DIR" "$TMPDIR" "$WORK_DIR/checkpoints"

# Reuse an existing Hugging Face login while storing downloads under /workspace.
if [[ -z "${HF_TOKEN:-}" && ! -f "$HF_HOME/token" && -r "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN
    HF_TOKEN="$(< "$HOME/.cache/huggingface/token")"
fi
if [[ -z "${HF_TOKEN:-}" && ! -s "$HF_HOME/token" ]]; then
    echo "Hugging Face login required for gated nvidia/Cosmos-Reason2-2B." >&2
    echo "Run: HF_HOME=$HF_HOME $REPO_DIR/.venv/bin/hf auth login" >&2
    exit 1
fi

PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-160}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_GPUS=4
GLOBAL_BATCH_SIZE="$((PER_GPU_BATCH_SIZE * NUM_GPUS))"
OUTPUT_DIR="${SO101_OUTPUT_DIR:-$WORK_DIR/checkpoints}"
if [[ "$OUTPUT_DIR" != /workspace/* ]]; then
    echo "SO101_OUTPUT_DIR must be under /workspace: $OUTPUT_DIR" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"
SAVE_ONLY_MODEL_ARGS=()
if [[ "${SO101_SAVE_ONLY_MODEL:-0}" == 1 ]]; then
    SAVE_ONLY_MODEL_ARGS+=(--save-only-model)
fi
RESUME_ARGS=()
if [[ "${SO101_RESUME:-0}" == 1 ]]; then
    RESUME_ARGS+=(--resume-from-checkpoint)
fi

if [[ ! -f "$DATASET_DIR/meta/modality.json" || ! -f "$DATASET_DIR/meta/info.json" ]]; then
    echo "Dataset metadata missing from $DATASET_DIR" >&2
    exit 1
fi

cd "$REPO_DIR"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS="$NUM_GPUS" \
GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}" \
SAVE_STEPS="${SAVE_STEPS:-2000}" \
MAX_STEPS="${MAX_STEPS:-51000}" \
USE_WANDB="${USE_WANDB:-1}" \
uv run bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path "$DATASET_DIR" \
    --modality-config-path "$CONFIG_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --output-dir "$OUTPUT_DIR" \
    "${SAVE_ONLY_MODEL_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    -- --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
