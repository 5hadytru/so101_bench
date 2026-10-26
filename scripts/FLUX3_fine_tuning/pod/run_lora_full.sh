#!/usr/bin/env bash
# The FLUX 3 Action full run (2026-10-01): LeRobot main examples/flux3/lora.json on the SO-101 package
# re-based on so101_bench_sim's own normalization (make_stats_package.py). Changes from the preset:
# micro-batch 2 x 4 GPUs x accumulation 16 = 128 global, gradient checkpointing off (6.40 windows/s,
# ~54 GB in the sweep), no held-out split. 6280 updates = the 36 h budget at the run's measured
# 20.6 s/update (LeRobot counts microsteps: 6280 x 16 = 100480). The 2026-10-01 run launched with 6480
# (the sweep's 20.0 s/update) and was retargeted to 6280 at its first checkpoint
# (retarget_steps_at_checkpoint.sh); the LR is constant, so only the stopping point changed and
# UPDATES=6280 reproduces it. Checkpoints every 500 updates (8000 microsteps).
# Supervised: on a non-zero exit it resumes from the last checkpoint (up to MAX_RESTARTS times).
set -uo pipefail
ROOT=/root/flux3
RUN=${RUN:-${ROOT}/runs/flux3-so101-lora-b128}
UPDATES=${UPDATES:-6280}
ACCUM=16
SAVE_UPDATES=${SAVE_UPDATES:-500}
MAX_RESTARTS=${MAX_RESTARTS:-5}
MICROSTEPS=$((UPDATES * ACCUM))
SAVE_MICROSTEPS=$((SAVE_UPDATES * ACCUM))
export HF_HUB_OFFLINE=1 HF_HOME=${ROOT}/.hf-home PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat ${ROOT}/secrets/wandb_api_key)
unset HF_TOKEN
cd "${ROOT}/lerobot"
mkdir -p "$(dirname "${RUN}")"
log() { echo "$(date -u +%FT%TZ) [supervisor] $*" | tee -a "${RUN}.supervisor.log"; }

if [[ ! -d "${RUN}" ]]; then
    python3 - "${RUN}.lora.json" "${ACCUM}" <<'EOF'
import json, sys
cfg = json.load(open("examples/flux3/lora.json"))
cfg["accelerator"]["gradient_accumulation"]["steps"] = int(sys.argv[2])
json.dump(cfg, open(sys.argv[1], "w"), indent=2)
EOF
    log "fresh launch: ${UPDATES} updates (${MICROSTEPS} microsteps), save every ${SAVE_MICROSTEPS} microsteps"
    .venv/bin/accelerate launch --num_processes=4 --multi_gpu --mixed_precision=bf16 -m lerobot.scripts.lerobot_train \
        --config_path="${RUN}.lora.json" \
        --policy.path="${ROOT}/weights/so101_ourstats" --policy.device=cuda \
        --policy.video_vae_id="${ROOT}/weights/base/video_vae.safetensors" \
        --policy.text_encoder_id="${ROOT}/weights/base/text_encoder" \
        --policy.gradient_checkpointing=false \
        --dataset.repo_id=5hadytru/so101_bench_sim --dataset.root="${ROOT}/datasets/so101_bench_sim" \
        --dataset.eval_split=0 --eval_steps=0 \
        '--rename_map={"observation.images.overhead":"observation.images.scene"}' \
        --batch_size=2 --steps="${MICROSTEPS}" --save_freq="${SAVE_MICROSTEPS}" --log_freq=160 \
        --num_workers=4 --output_dir="${RUN}" --job_name=flux3-so101-lora-b128 \
        --wandb.enable=true --wandb.project=finetune-flux3-so101 --wandb.disable_artifact=true \
        >> "${RUN}.train.log" 2>&1
    status=$?
else
    status=1
fi
restarts=0
while ((status != 0)); do
    last="${RUN}/checkpoints/last/pretrained_model/train_config.json"
    if [[ ! -f "${last}" ]]; then log "exit ${status} with no checkpoint to resume from; giving up"; exit "${status}"; fi
    if ((restarts >= MAX_RESTARTS)); then log "exit ${status}; restart budget (${MAX_RESTARTS}) used up"; exit "${status}"; fi
    restarts=$((restarts + 1))
    log "exit ${status}; resuming from $(readlink -f "${RUN}/checkpoints/last") (restart ${restarts}/${MAX_RESTARTS})"
    sleep 30
    .venv/bin/accelerate launch --num_processes=4 --multi_gpu --mixed_precision=bf16 -m lerobot.scripts.lerobot_train \
        --config_path="${last}" --resume=true >> "${RUN}.train.log" 2>&1
    status=$?
done
log "training finished"
touch "${RUN}.COMPLETE"
