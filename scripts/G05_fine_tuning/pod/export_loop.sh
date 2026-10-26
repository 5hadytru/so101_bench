#!/usr/bin/env bash
# Training-pod side of local G0.5 validation. For each scheduled step: once its training checkpoint is saved,
# export a weights-only bf16 bundle (pod/export_checkpoint.py) to export/<run>-stepN. The workstation pulls
# exports and touches PULLED; pulled exports are deleted here. Also prunes training checkpoints, keeping the
# newest KEEP for crash resume (a save is 19 GB; the 400 GB disk holds ~15).
set -uo pipefail
ROOT=${ROOT:-/root/g05}; RUN_NAME=${RUN_NAME:-so101-full-m16-8bit}; KEEP=${KEEP:-2}
RUN_DIR=${ROOT}/runs/so101_bench/${RUN_NAME}; EXPORTS=${ROOT}/export
SCHEDULE=${SCHEDULE:-"2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 25600"}
PY=${ROOT}/GalaxeaVLA/.venv/bin/python
log() { echo "$(date -u +%FT%TZ) [export] $*" | tee -a "${ROOT}/logs/${RUN_NAME}/export_loop.log"; }
mkdir -p "${EXPORTS}" "${ROOT}/logs/${RUN_NAME}"
saved() { grep -aq "step $1 checkpoint saved\|Max step $1 reached" "${ROOT}"/logs/"${RUN_NAME}"/train_attempt*.log 2>/dev/null; }
while true; do
    for s in ${SCHEDULE}; do
        out="${EXPORTS}/${RUN_NAME}-step${s}"
        if [[ -f "${out}/PULLED" && -f "${out}/checkpoints/model_state_dict.pt" ]]; then rm -f "${out}/checkpoints/model_state_dict.pt"; log "freed pulled step ${s}"; fi
        [[ -f "${out}/EXPORT_OK" || -f "${out}/PULLED" ]] && continue
        [[ -f "${RUN_DIR}/checkpoints/step_${s}.pt" ]] && saved "${s}" || continue
        sleep 30
        if nice -n 10 "${PY}" "${ROOT}/pod/export_checkpoint.py" "${RUN_DIR}" "${s}" "${out}" >> "${ROOT}/logs/${RUN_NAME}/export_loop.detail.log" 2>&1; then log "exported step ${s}"; else log "export FAILED step ${s}"; rm -rf "${out}"; fi
    done
    mapfile -t ckpts < <(ls "${RUN_DIR}"/checkpoints/step_*.pt 2>/dev/null | sort -V)
    if (( ${#ckpts[@]} > KEEP )); then
        for c in "${ckpts[@]:0:${#ckpts[@]}-KEEP}"; do
            s=$(basename "$c" .pt); s=${s#step_}
            if [[ " ${SCHEDULE} " == *" ${s} "* && ! -f "${EXPORTS}/${RUN_NAME}-step${s}/EXPORT_OK" ]]; then continue; fi
            rm -f "$c"; log "pruned training checkpoint step ${s}"
        done
    fi
    sleep 120
done
