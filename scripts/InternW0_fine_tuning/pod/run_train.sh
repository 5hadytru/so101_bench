#!/usr/bin/env bash
# Launch InternW0-delta post-training on so101_bench_sim (task=so101) on this 4xA100 pod.
# Every knob is an environment variable so the sweep (bench.py) and the full run share one path:
#   MICRO (per-GPU batch), ACCUM (gradient accumulation), MAX_STEPS (optimizer steps; sets the cosine
#   length), SAVE_EVERY, ZERO (1|2, DeepSpeed stage), INFRA (none|online|compile), CKPT_STRIDE
#   (MoT activation-checkpoint layer stride), WANDB (0|1), RUN_NAME, OUT_ROOT, RESUME (step dir or "").
set -euo pipefail
ROOT=${ROOT:-/root/internw0}
REPO=${ROOT}/InternW0-Delta
MICRO=${MICRO:-8}
ACCUM=${ACCUM:-4}
MAX_STEPS=${MAX_STEPS:-100000}
SAVE_EVERY=${SAVE_EVERY:-500}
ZERO=${ZERO:-1}
INFRA=${INFRA:-none}
CKPT_STRIDE=${CKPT_STRIDE:-1}
WANDB=${WANDB:-0}
RUN_NAME=${RUN_NAME:-so101-dev}
OUT_ROOT=${OUT_ROOT:-${ROOT}/runs}
export WAM_DATA_ROOT=${ROOT}/data WAM_CACHE_ROOT=${ROOT}/cache WAM_CHECKPOINT_ROOT=${REPO}/checkpoints
export SO101_DATA_ROOT=${ROOT}/data/so101_bench_sim_v21 SO101_NORM_STATS=${ROOT}/cache/stats/so101.json
export WAM_PRETRAIN_CHECKPOINT=${REPO}/checkpoints/pretrain.pt
export WAM_OUTPUT_ROOT=${OUT_ROOT} RUN_ID=${RUN_NAME} NPROC_PER_NODE=4
export WAM_ACCELERATE_CONFIG=scripts/accelerate_configs/accelerate_zero${ZERO}_ds.yaml
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VIRTUAL_ENV=${REPO}/.venv PATH=${REPO}/.venv/bin:${PATH}
export WANDB_MODE=$([[ "${WANDB}" == 1 ]] && echo online || echo disabled)
[[ "${WANDB}" == 1 ]] && export WANDB_API_KEY=$(cat "${ROOT}/secrets/wandb_api_key")
unset HF_TOKEN
cd "${REPO}"
args=(
    "batch_size=${MICRO}" "gradient_accumulation_steps=${ACCUM}" "max_steps=${MAX_STEPS}" "num_epochs=1000"
    "save_every=${SAVE_EVERY}" "model.mot_checkpoint_layer_stride=${CKPT_STRIDE}"
    "wandb.enabled=$([[ "${WANDB}" == 1 ]] && echo true || echo false)" "wandb.mode=${WANDB_MODE}"
    "wandb.workspace=5hadytru" "wandb.project=finetune-internw0-so101" "wandb.name=${RUN_NAME}"
)
[[ "${INFRA}" != none ]] && args+=("+infra=${INFRA}")
[[ -n "${RESUME:-}" ]] && args+=("resume=${RESUME}")
echo "$(date -u +%FT%TZ) [run_train] ${RUN_NAME}: micro=${MICRO} accum=${ACCUM} global=$((MICRO * ACCUM * 4)) zero=${ZERO} infra=${INFRA} stride=${CKPT_STRIDE} max_steps=${MAX_STEPS}"
exec bash run.sh so101 "${args[@]}" "$@"
