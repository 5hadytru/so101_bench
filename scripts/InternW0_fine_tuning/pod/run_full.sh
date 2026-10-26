#!/usr/bin/env bash
# Supervisor for the full InternW0-delta so101 run: launches pod/run_train.sh with the sweep's winning
# settings and, if training dies before MAX_STEPS, relaunches it from the run's own optimizer state
# (checkpoints/state/latest) so the cosine schedule and data order continue. Run inside tmux.
#   bash /root/internw0/pod/run_full.sh
set -uo pipefail
ROOT=${ROOT:-/root/internw0}
export MICRO=${MICRO:-16} ACCUM=${ACCUM:-2} ZERO=${ZERO:-1} INFRA=${INFRA:-compile} CKPT_STRIDE=${CKPT_STRIDE:-1}
# 36 h at the sweep's 7.0 s/step + ~50 s per save every 500 steps + ~5 min compile at launch.
export MAX_STEPS=${MAX_STEPS:-18200} SAVE_EVERY=${SAVE_EVERY:-500}
export WANDB=${WANDB:-1} RUN_NAME=${RUN_NAME:-so101-full-c16} OUT_ROOT=${OUT_ROOT:-${ROOT}/runs}
RUN_DIR=${OUT_ROOT}/so101/${RUN_NAME}
LOG_DIR=${ROOT}/logs/${RUN_NAME}
MAX_RESTARTS=${MAX_RESTARTS:-5}
mkdir -p "${LOG_DIR}"
log() { echo "$(date -u +%FT%TZ) [supervisor] $*" | tee -a "${LOG_DIR}/supervisor.log"; }

for attempt in $(seq 0 "${MAX_RESTARTS}"); do
    if [[ -f "${RUN_DIR}/checkpoints/state/latest/trainer_state.json" ]]; then
        export RESUME=${RUN_DIR}
    else
        unset RESUME   # task config default: the released pretrain.pt
    fi
    log "attempt ${attempt}: RESUME=${RESUME:-pretrain.pt}"
    bash "${ROOT}/pod/run_train.sh" > "${LOG_DIR}/train_attempt${attempt}.log" 2>&1
    status=$?
    if grep -q "\[done\] training finished" "${LOG_DIR}/train_attempt${attempt}.log"; then
        log "TRAINING_DONE (exit ${status})"
        exit 0
    fi
    log "training exited ${status} before finishing; cleaning GPUs and restarting in 60 s"
    pkill -f "scripts/train.py" ; sleep 60
done
log "giving up after ${MAX_RESTARTS} restarts"
exit 1
