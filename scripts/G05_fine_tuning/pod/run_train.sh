#!/usr/bin/env bash
# Launch G0.5 fine-tuning on so101_bench_sim (task=so101_bench) on this 4xA100 pod. Every knob is an env var
# so the sweep (bench.py) and the full run share one path:
#   MICRO (per-GPU batch), ACCUM, MAX_STEPS, OBS (history frames: 6 = g05-base memory, 1 = single frame),
#   CKPT_VISION (ViT activation checkpointing true/false), COMPILE (torch.compile true/false),
#   OPT8BIT (bitsandbytes 8-bit AdamW true/false),
#   SAVE_EVERY, WANDB (0|1), RUN_NAME, OUT_ROOT, NGPU
set -euo pipefail
ROOT=${ROOT:-/root/g05}
REPO=${ROOT}/GalaxeaVLA
MICRO=${MICRO:-16}; ACCUM=${ACCUM:-2}; MAX_STEPS=${MAX_STEPS:-100000}; OBS=${OBS:-6}
CKPT_VISION=${CKPT_VISION:-true}; COMPILE=${COMPILE:-false}; OPT8BIT=${OPT8BIT:-false}; SAVE_EVERY=${SAVE_EVERY:-1000}
WANDB=${WANDB:-0}; RUN_NAME=${RUN_NAME:-so101-dev}; OUT_ROOT=${OUT_ROOT:-${ROOT}/runs}; NGPU=${NGPU:-4}
export HF_HOME=${ROOT}/.hf-home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SO101_DATA_ROOT=${ROOT}/data/so101_bench_sim SO101_STATS=${ROOT}/stats/so101_bench_stats.json
export G05_OUTPUT_DIR=${OUT_ROOT} EXP_NAME=${RUN_NAME} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VIRTUAL_ENV=${REPO}/.venv PATH=${REPO}/.venv/bin:${PATH} PYTHONPATH=${REPO}/src:${PYTHONPATH:-}
if [[ "${WANDB}" == 1 ]]; then export WANDB_API_KEY=$(cat "${ROOT}/secrets/wandb_api_key"); MODE=online; else MODE=disabled; fi
unset HF_TOKEN
mkdir -p "$(dirname "${SO101_STATS}")"
cp "${ROOT}/configs/data/so101_bench.yaml" "${REPO}/configs/data/so101_bench.yaml"
cp "${ROOT}/configs/task/so101_bench.yaml" "${REPO}/configs/task/so101_bench.yaml"
cd "${REPO}"
echo "$(date -u +%FT%TZ) [run_train] ${RUN_NAME}: micro=${MICRO} accum=${ACCUM} global=$((MICRO * ACCUM * NGPU)) obs=${OBS} ckpt_vision=${CKPT_VISION} compile=${COMPILE} opt8bit=${OPT8BIT} max_steps=${MAX_STEPS}"
exec bash scripts/run/finetune.sh "${NGPU}" so101_bench \
    model.batch_size="${MICRO}" model.grad_accumulation_steps="${ACCUM}" model.max_epochs=null model.max_steps="${MAX_STEPS}" \
    data.obs_size="${OBS}" model.model_arch.checkpoint_vision="${CKPT_VISION}" model.use_torch_compile="${COMPILE}" model.use_8bit_optimizer="${OPT8BIT}" \
    checkpointing_steps="${SAVE_EVERY}" eval_steps=100000000 \
    logger.mode="${MODE}" logger.workspace=5hadytru logger.project=finetune-g05-so101 "$@"
