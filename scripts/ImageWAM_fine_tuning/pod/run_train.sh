#!/usr/bin/env bash
# Launch the ImageWAM so101_bench_sim fine-tune (task=so101_flux2_klein_4b_ee16) on this 4xA100 pod.
# Every knob is an environment variable so the sweep (bench.py) and the full run share one path:
#   MICRO (per-GPU batch), ACCUM (gradient accumulation), MAX_STEPS (optimizer steps; sets the cosine length),
#   LR, WARMUP (steps; empty = the trainer's 5%), SAVE_EVERY, ZERO (1|2), GRAD_CKPT (true|false: MoT mixed-
#   attention activation checkpointing), CANVAS (pretrain = compact 288x256 three-slot | tall = 384x256 two
#   tiles), CHUNK (32|16 actions), LORA (0|1: the repo's r16 LoRA on the FLUX.2 editing transformer, action
#   expert still full), WORKERS, WANDB (0|1), RUN_NAME, OUT_ROOT, RESUME (state dir; empty = start from the
#   InternData-A1-EE weights).
set -euo pipefail
ROOT=${ROOT:-/root/imagewam}
REPO=${ROOT}/ImageWAM
MICRO=${MICRO:-8}
ACCUM=${ACCUM:-4}
MAX_STEPS=${MAX_STEPS:-100000}
LR=${LR:-2.5e-5}
WARMUP=${WARMUP:-}
SAVE_EVERY=${SAVE_EVERY:-1000}
ZERO=${ZERO:-1}
GRAD_CKPT=${GRAD_CKPT:-true}
CANVAS=${CANVAS:-pretrain}
CHUNK=${CHUNK:-32}
LORA=${LORA:-0}
WORKERS=${WORKERS:-16}
WANDB=${WANDB:-0}
RUN_NAME=${RUN_NAME:-so101-dev}
OUT_ROOT=${OUT_ROOT:-${ROOT}/runs}
CK=${ROOT}/checkpoints
export SO101_DATA_ROOT=${ROOT}/data/so101_bench_sim_v21 SO101_NORM_STATS=${ROOT}/cache/so101_ee16_dataset_stats.json
export SO101_QWEN_CACHE=${ROOT}/cache/flux2_qwen3_cache_4b
export TASK=so101_flux2_klein_4b_ee16 FLUX2_SRC=${REPO}/third_party/flux2 ZERO_STAGE=${ZERO} RUN_ID=${RUN_NAME}
# Wide rich console: the trainer logs through rich, which wraps at 80 columns when piped.
export COLUMNS=400
export HF_HOME=${ROOT}/.hf-home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VIRTUAL_ENV=${REPO}/.venv PATH=${REPO}/.venv/bin:${PATH}
export WANDB_MODE=$([[ "${WANDB}" == 1 ]] && echo online || echo disabled)
[[ "${WANDB}" == 1 ]] && export WANDB_API_KEY=$(cat "${ROOT}/secrets/wandb_api_key")
unset HF_TOKEN
cd "${REPO}"

args=(
    "output_dir=${OUT_ROOT}/${RUN_NAME}"
    "model.flux2_model_path=${CK}/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors"
    "model.ae_model_path=${CK}/flux2/FLUX.2-dev/ae.safetensors"
    "model.qwen3_model_spec=${CK}/Qwen3-4B"
    "model.mot_checkpoint_mixed_attn=${GRAD_CKPT}"
    "batch_size=${MICRO}" "gradient_accumulation_steps=${ACCUM}" "max_steps=${MAX_STEPS}"
    "learning_rate=${LR}" "save_every=${SAVE_EVERY}" "num_workers=${WORKERS}"
    "data.train.num_frames=$((CHUNK + 1))"
    "wandb.enabled=$([[ "${WANDB}" == 1 ]] && echo true || echo false)" "wandb.mode=${WANDB_MODE}"
    "wandb.name=${RUN_NAME}"
)
[[ -n "${WARMUP}" ]] && args+=("warmup_steps=${WARMUP}")
[[ "${LORA}" == 1 ]] && args+=("model.flux2_lora_config.enabled=true")
case "${CANVAS}" in
    pretrain) ;;
    tall) args+=("data.train.concat_multi_camera=vertical" "data.train.video_size=[384,256]" "data.train.processor.num_output_cameras=2") ;;
    *) echo "CANVAS must be pretrain or tall, got ${CANVAS}" >&2; exit 2 ;;
esac
args+=("resume=${RESUME:-${CK}/imagewam_interndata_a1_ee/model.pt}")

echo "$(date -u +%FT%TZ) [run_train] ${RUN_NAME}: micro=${MICRO} accum=${ACCUM} global=$((MICRO * ACCUM * 4)) lr=${LR} zero=${ZERO} grad_ckpt=${GRAD_CKPT} canvas=${CANVAS} chunk=${CHUNK} lora=${LORA} max_steps=${MAX_STEPS} resume=${RESUME:-pretrain}"
exec bash scripts/flux2/train_flux2_klein_imagewam.sh 4 "${args[@]}" "$@"
