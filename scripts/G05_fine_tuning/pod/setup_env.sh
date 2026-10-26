#!/usr/bin/env bash
# G0.5 (GalaxeaVLA) training environment on a 4xA100 pod: pinned repo + uv env (Python 3.10, torch 2.7.1
# cu128, flash-attn-4, flash-linear-attention), the G05 checkpoints (g05-base, g05-so101, action tokenizer,
# Qwen3.5 processor) and so101_bench_sim (LeRobot v3.0, no init frames). Markers in ${ROOT}/.done.
set -euo pipefail
ROOT=${ROOT:-/root/g05}
REPO_REV=89f2322
G05_REV=e312be81e90c56a55bcb26b57429bd39a335b449
DATASET_ID=5hadytru/so101_bench_sim
DATASET_REV=34233754992fbd77e42373bad44fd42ddfb43f14
export UV_CACHE_DIR=${ROOT}/.uv-cache HF_HOME=${ROOT}/.hf-home HF_HUB_DISABLE_XET=1
unset HF_TOKEN
mkdir -p "${ROOT}/.done" "${ROOT}/logs" "${ROOT}/data"
cd "${ROOT}"
is_done() { [[ -f "${ROOT}/.done/$1" ]]; }
log() { echo "$(date -u +%FT%TZ) [setup] $*"; }
command -v ffmpeg >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null; }
if ! is_done env; then
    log "repo ${REPO_REV} + uv env"
    [[ -d GalaxeaVLA ]] || git clone -q https://github.com/OpenGalaxea/GalaxeaVLA.git
    git -C GalaxeaVLA checkout -q "${REPO_REV}"
    (cd GalaxeaVLA && uv sync --index-strategy unsafe-best-match)
    # A100 (sm80): FA4's backward needs sm90+, so the ViT falls back to FlashAttention-2 (patch + wheel).
    git -C GalaxeaVLA apply "${ROOT}/patches/galaxeavla_89f2322.patch"
    uv pip install --python GalaxeaVLA/.venv/bin/python \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
    GalaxeaVLA/.venv/bin/python -c "import torch, g05; print('torch', torch.__version__, torch.cuda.is_available())"
    touch .done/env
fi
HF=${ROOT}/GalaxeaVLA/.venv/bin/hf
[[ -x ${HF} ]] || HF="${ROOT}/GalaxeaVLA/.venv/bin/huggingface-cli"
download_models() {
    is_done models && return 0
    log "G05 checkpoints @ ${G05_REV:0:7}"
    # One --include with several patterns: this hf CLI keeps only the last of repeated --include flags.
    "${HF}" download OpenGalaxea/G05 --revision "${G05_REV}" \
        --include 'g05-base/*' 'g05-so101/*' 'action_tokenizer.pt' 'qwen3_5_2b_base_processor/*' 'licenses/*' \
        --local-dir "${ROOT}/GalaxeaVLA/checkpoints"
    touch "${ROOT}/.done/models"
}
download_dataset() {
    is_done dataset && return 0
    log "dataset ${DATASET_ID}@${DATASET_REV:0:7}"
    "${HF}" download "${DATASET_ID}" --repo-type dataset --revision "${DATASET_REV}" \
        --local-dir "${ROOT}/data/so101_bench_sim" --max-workers 16
    touch "${ROOT}/.done/dataset"
}
download_models > logs/setup_models.log 2>&1 & p1=$!
if [[ -n "${SKIP_DATASET:-}" ]]; then touch .done/dataset; fi
download_dataset > logs/setup_dataset.log 2>&1 & p2=$!
wait $p1 || { log "models FAILED"; exit 1; }
wait $p2 || { log "dataset FAILED"; exit 1; }
du -sh GalaxeaVLA/checkpoints/* data/* 2>/dev/null
log "setup complete"
