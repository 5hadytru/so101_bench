#!/usr/bin/env bash
# so101_bench_sim for InternW0-delta post-training: LeRobot v3.0 -> v2.1 (the vendored loader's format),
# install the SO-101 data/task configs into the repo, compute normalization statistics and the text
# embedding cache. Markers in ${ROOT}/.done make reruns skip finished steps.
set -euo pipefail
ROOT=${ROOT:-/root/internw0}
REPO=${ROOT}/InternW0-Delta
PY=${REPO}/.venv/bin/python
export WAM_DATA_ROOT=${ROOT}/data WAM_CACHE_ROOT=${ROOT}/cache WAM_CHECKPOINT_ROOT=${REPO}/checkpoints
export SO101_DATA_ROOT=${ROOT}/data/so101_bench_sim_v21 SO101_NORM_STATS=${ROOT}/cache/stats/so101.json
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=${REPO}:${REPO}/src
unset HF_TOKEN
cd "${REPO}"
is_done() { [[ -f "${ROOT}/.done/$1" ]]; }
log() { echo "$(date -u +%FT%TZ) [prepare] $*"; }

cp "${ROOT}/configs/data/so101.yaml" configs/data/so101.yaml
cp "${ROOT}/configs/task/so101.yaml" configs/task/so101.yaml

if ! is_done convert; then
    log "convert v3.0 -> v2.1"
    rm -rf "${SO101_DATA_ROOT}"
    "${PY}" "${ROOT}/pod/convert_v30_to_v21.py" "${ROOT}/data/so101_bench_sim" "${SO101_DATA_ROOT}" --workers 112
    touch "${ROOT}/.done/convert"
fi
if ! is_done stats; then
    log "normalization statistics"
    mkdir -p "$(dirname "${SO101_NORM_STATS}")"
    "${PY}" tools/posttrain_stats.py task=so101 +precompute.stats_path="${SO101_NORM_STATS}"
    touch "${ROOT}/.done/stats"
fi
if ! is_done text_cache; then
    log "text embedding cache"
    "${PY}" tools/text_cache.py task=so101
    touch "${ROOT}/.done/text_cache"
fi
log "data ready: ${SO101_DATA_ROOT}, stats ${SO101_NORM_STATS}"
