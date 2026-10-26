#!/usr/bin/env bash
# Training-pod side of FLUX LoRA validation: when a scheduled checkpoint is complete, merge its raw
# (pretrained_model) and EMA (pretrained_model_ema) adapters into full SO-101 packages for
# scripts/flux3_server.py --package. Runs on the CPU at low priority next to training. An export is
# deleted after a validation pod has pulled it (it writes PULLED).
#   export/flux3-lora-step<U>-<raw|ema>/{model.safetensors,...,EXPORT_OK}
set -uo pipefail
ROOT=/root/flux3
RUN=${RUN:-${ROOT}/runs/flux3-so101-lora-b128}
EXPORTS=${EXPORTS:-${ROOT}/export}
ACCUM=16
SCHEDULE=${SCHEDULE:-"1000 2000 3000 4000 5000 6000 6280"}
export HF_HUB_OFFLINE=1 HF_HOME=${ROOT}/.hf-home CUDA_VISIBLE_DEVICES=
mkdir -p "${EXPORTS}"
log() { echo "$(date -u +%FT%TZ) [export] $*" | tee -a "${ROOT}/logs/export_loop.log"; }

checkpoint_ready() {  # a checkpoint is complete once its training state is written and has settled
    local dir="$1"
    [[ -f "${dir}/training_state/training_step.json" && -f "${dir}/pretrained_model_ema/adapter_model.safetensors" ]] || return 1
    (( $(date +%s) - $(stat -c %Y "${dir}/training_state/training_step.json") > 60 ))
}

while true; do
    all_done=1
    for updates in ${SCHEDULE}; do
        ckpt="${RUN}/checkpoints/$(printf '%06d' $((updates * ACCUM)))"
        for variant in raw ema; do
            out="${EXPORTS}/flux3-lora-step${updates}-${variant}"
            if [[ -f "${out}/PULLED" ]]; then   # the validation pod has its copy: keep only the markers
                if [[ -f "${out}/model.safetensors" ]]; then
                    find "${out}" -mindepth 1 ! -name PULLED ! -name EXPORT_OK ! -name export_source.json -delete
                    log "freed pulled ${out##*/}"
                fi
                continue
            fi
            [[ -f "${out}/EXPORT_OK" ]] && continue
            all_done=0
            checkpoint_ready "${ckpt}" || continue
            adapter="${ckpt}/pretrained_model"; [[ "${variant}" == ema ]] && adapter="${ckpt}/pretrained_model_ema"
            log "merging step ${updates} ${variant}"
            if nice -n 10 "${ROOT}/lerobot/.venv/bin/python" "${ROOT}/pod/merge_lora_package.py" "${adapter}" \
                "${ROOT}/weights/so101_ourstats" "${out}" >> "${ROOT}/logs/export_loop.merge.log" 2>&1; then
                printf '{"step": %d, "variant": "%s", "adapter": "%s"}\n' "${updates}" "${variant}" "${adapter}" > "${out}/export_source.json"
                touch "${out}/EXPORT_OK"
                log "exported ${out##*/}"
            else
                log "merge FAILED for step ${updates} ${variant}; see export_loop.merge.log"
                rm -rf "${out}"
            fi
        done
    done
    if ((all_done)) && [[ -f "${RUN}.COMPLETE" ]]; then log "all scheduled exports done"; exit 0; fi
    sleep 120
done
