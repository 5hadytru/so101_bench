#!/usr/bin/env bash
# Supervisor for the full run (option A, 2026-10-07): micro 8 x accum 4 x 4 GPUs = 128, LR 2.5e-5 cosine,
# pretrain canvas, 32-action chunk, 33,000 steps (~36 h at 3.87 s/step). Relaunches from the latest
# state/step_* on a crash, up to 5 times. Run in tmux.
set -uo pipefail
ROOT=${ROOT:-/root/imagewam}
RUN_NAME=${RUN_NAME:-imagewam-so101-a-b128}
RUN_DIR=${ROOT}/runs/${RUN_NAME}
export MICRO=8 ACCUM=4 MAX_STEPS=${MAX_STEPS:-33000} LR=2.5e-5 SAVE_EVERY=1000 CANVAS=pretrain CHUNK=32 WANDB=1 RUN_NAME
mkdir -p "${ROOT}/logs/${RUN_NAME}"
for attempt in 0 1 2 3 4 5; do
    latest=$(ls -d "${RUN_DIR}"/checkpoints/state/step_* 2>/dev/null | sort | tail -1)
    log=${ROOT}/logs/${RUN_NAME}/attempt${attempt}.log
    echo "$(date -u +%FT%TZ) [supervisor] attempt ${attempt} resume=${latest:-pretrain} log=${log}"
    RESUME=${latest} bash "${ROOT}/pod/run_train.sh" > "${log}" 2>&1
    rc=$?
    if grep -q "max_steps reached" "${log}"; then echo "$(date -u +%FT%TZ) [supervisor] DONE"; exit 0; fi
    echo "$(date -u +%FT%TZ) [supervisor] exit ${rc}; restarting in 60 s"
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
    sleep 60
done
echo "$(date -u +%FT%TZ) [supervisor] GAVE UP"
exit 1
