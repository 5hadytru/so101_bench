#!/usr/bin/env bash
# Supervisor for the full G0.5 so101_bench run: pod/run_train.sh with the sweep's winner (micro 16 x accum 2,
# 8-bit AdamW, 6-frame memory). On a crash it relaunches from the newest training checkpoint (model + 8-bit
# optimizer + scheduler state). Run inside tmux:  bash /root/g05/pod/run_full.sh
set -uo pipefail
ROOT=${ROOT:-/root/g05}
export MICRO=${MICRO:-16} ACCUM=${ACCUM:-2} OPT8BIT=${OPT8BIT:-true} OBS=${OBS:-6} CKPT_VISION=${CKPT_VISION:-true}
# 36 h at the sweep's 5.01 s/step, minus ~3 min startup and ~15 s per save (every 1000 steps).
export MAX_STEPS=${MAX_STEPS:-25600} SAVE_EVERY=${SAVE_EVERY:-1000}
export WANDB=${WANDB:-1} RUN_NAME=${RUN_NAME:-so101-full-m16-8bit} OUT_ROOT=${OUT_ROOT:-${ROOT}/runs}
RUN_DIR=${OUT_ROOT}/so101_bench/${RUN_NAME}
LOG_DIR=${ROOT}/logs/${RUN_NAME}; mkdir -p "${LOG_DIR}"
log() { echo "$(date -u +%FT%TZ) [supervisor] $*" | tee -a "${LOG_DIR}/supervisor.log"; }
for attempt in $(seq 0 5); do
    latest=$(ls "${RUN_DIR}"/checkpoints/step_*.pt 2>/dev/null | sort -V | tail -1)
    extra=(); [[ -n "${latest}" ]] && extra=("resume_ckpt=${latest}")
    log "attempt ${attempt}: ${latest:-fresh from g05-base}"
    bash "${ROOT}/pod/run_train.sh" "${extra[@]}" > "${LOG_DIR}/train_attempt${attempt}.log" 2>&1
    status=$?
    if grep -aq "Max step ${MAX_STEPS} reached" "${LOG_DIR}/train_attempt${attempt}.log"; then log "TRAINING_DONE (exit ${status})"; exit 0; fi
    log "exit ${status} before max_steps; restarting in 60 s"
    sleep 60
done
log "giving up after 5 restarts"; exit 1
