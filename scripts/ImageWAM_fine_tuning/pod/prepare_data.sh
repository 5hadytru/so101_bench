#!/usr/bin/env bash
# so101_bench_sim for ImageWAM: install the SO-101 configs and the SO101ToCanonicalEE16 transform into the
# repo, convert LeRobot v3.0 -> v2.1 (the repo's v2 loader; same converter as the InternW0 run), compute
# normalization statistics (16D, from this dataset) and the FLUX.2 Qwen3 text-embedding cache.
# Markers in ${ROOT}/.done make reruns skip finished steps.
set -euo pipefail
ROOT=${ROOT:-/root/imagewam}
REPO=${ROOT}/ImageWAM
PY=${REPO}/.venv/bin/python
export SO101_DATA_ROOT=${ROOT}/data/so101_bench_sim_v21 SO101_NORM_STATS=${ROOT}/cache/so101_ee16_dataset_stats.json
export SO101_QWEN_CACHE=${ROOT}/cache/flux2_qwen3_cache_4b
export FLUX2_SRC=${REPO}/third_party/flux2
export HF_HOME=${ROOT}/.hf-home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=${REPO}/src:${FLUX2_SRC}/src:${FLUX2_SRC}
unset HF_TOKEN
cd "${REPO}"
is_done() { [[ -f "${ROOT}/.done/$1" ]]; }
log() { echo "$(date -u +%FT%TZ) [prepare] $*"; }

cp "${ROOT}/configs/data/so101_ee16.yaml" configs/data/so101_ee16.yaml
cp "${ROOT}/configs/task/so101_flux2_klein_4b_ee16.yaml" configs/task/so101_flux2_klein_4b_ee16.yaml
cp "${ROOT}/src_patch/so101.py" src/imagewam/datasets/lerobot/transforms/so101.py

if ! is_done convert; then
    log "convert v3.0 -> v2.1"
    rm -rf "${SO101_DATA_ROOT}"
    "${PY}" "${ROOT}/pod/convert_v30_to_v21.py" "${ROOT}/data/so101_bench_sim" "${SO101_DATA_ROOT}" --workers "${CONVERT_WORKERS:-48}"
    touch "${ROOT}/.done/convert"
fi
if ! is_done stats; then
    log "normalization statistics -> ${SO101_NORM_STATS}"
    mkdir -p "$(dirname "${SO101_NORM_STATS}")"
    SO101_NORM_STATS= "${PY}" "${ROOT}/pod/compute_stats.py" "${SO101_NORM_STATS}"
    touch "${ROOT}/.done/stats"
fi
if ! is_done text_cache; then
    log "Qwen3-4B text-embedding cache -> ${SO101_QWEN_CACHE}"
    "${REPO}/.venv/bin/torchrun" --standalone --nproc_per_node=4 scripts/flux2/precompute_flux2_qwen3_embeds.py \
        task=so101_flux2_klein_4b_ee16 \
        qwen_cache_batch_size=16 qwen_cache_save_workers=4 qwen_cache_overwrite=false \
        model.variant=klein-base-4b \
        model.flux2_src_path="${FLUX2_SRC}" \
        model.flux2_model_path="${ROOT}/checkpoints/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors" \
        model.ae_model_path="${ROOT}/checkpoints/flux2/FLUX.2-dev/ae.safetensors" \
        model.qwen3_model_spec="${ROOT}/checkpoints/Qwen3-4B" \
        flux2_qwen3_model_spec="${ROOT}/checkpoints/Qwen3-4B"
    touch "${ROOT}/.done/text_cache"
fi
log "data ready: ${SO101_DATA_ROOT}, stats ${SO101_NORM_STATS}, text cache ${SO101_QWEN_CACHE}"
