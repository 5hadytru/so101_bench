#!/usr/bin/env bash
# Supervisor for the full run (option B, 2026-10-08): micro 32 x accum 2 x 4 GPUs = 256 (the official LIBERO
# SFT batch shape), LR 1e-4 cosine to 5e-7 over the whole run, 1500 warmup updates, 36-step (1.2 s) chunk,
# 49,000 updates = the 36 h budget at the sweep's 2.62 s/update. Checkpoints every 1000 updates (LeRobot
# names them by micro-batch: update U -> checkpoints/<2U, 6 digits>). On a non-zero exit it resumes from
# checkpoints/last, up to 5 times. Run in tmux.
set -uo pipefail
ROOT=${ROOT:-/root/lawam}
export RUN_NAME=${RUN_NAME:-lawam-so101-b256-ah36}
export MICRO=32 ACCUM=2 UPDATES=${UPDATES:-49000} LR=1e-4 WARMUP=1500 AH=36 SAVE_UPDATES=1000 WORKERS=8 CLIP=true WANDB=1
RUN=${ROOT}/runs/${RUN_NAME}
mkdir -p "${ROOT}/logs/${RUN_NAME}"
log() { echo "$(date -u +%FT%TZ) [supervisor] $*" | tee -a "${ROOT}/logs/${RUN_NAME}/supervisor.log"; }
for attempt in 0 1 2 3 4 5; do
    resume=0
    [[ -f "${RUN}/checkpoints/last/pretrained_model/train_config.json" ]] && resume=1
    logf=${ROOT}/logs/${RUN_NAME}/attempt${attempt}.log
    log "attempt ${attempt} resume=${resume} log=${logf}"
    if [[ ${resume} == 0 && -d "${RUN}" ]]; then
        log "run dir exists without a checkpoint; moving it aside"; mv "${RUN}" "${RUN}.failed_$(date +%s)"
    fi
    RESUME=${resume} bash "${ROOT}/pod/run_train.sh" > "${logf}" 2>&1
    rc=$?
    if grep -aq "End of training" "${logf}"; then log "TRAINING_DONE (exit ${rc})"; touch "${RUN}.COMPLETE"; exit 0; fi
    log "exit ${rc} before the end of training; restarting in 60 s"
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
    sleep 60
done
log "GAVE UP"
exit 1
