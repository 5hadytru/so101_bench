#!/usr/bin/env bash
# ImageWAM (FLUX.2 klein-base-4B) training environment on a 4xA100 pod. Everything on the container disk
# under ${ROOT}. Steps leave markers in ${ROOT}/.done so a rerun skips finished work.
set -euo pipefail
ROOT=${ROOT:-/root/imagewam}
REPO_REV=5d4a341ed20a95cdb08f0293f3d44778b9a9e05a                                  # yuyangalin/ImageWAM, 2026-07-30 (main)
FLUX2_SRC_REV=50fe5162777813d869182b139e83b10743caef15  # pinned by the ImageWAM README
DATASET_ID=5hadytru/so101_bench_sim
DATASET_REV=34233754992fbd77e42373bad44fd42ddfb43f14
PRETRAIN_REV=51a71ed9596ec3b3bae52a08f201fa6125a99814  # yuyangalin/ImageWAM-FLUX.2-4B-InternData-A1-EE
KLEIN_REV=a3b4f4849157f664bdbc776fd7453c2783562f4d     # black-forest-labs/FLUX.2-klein-base-4B
FLUXDEV_REV=26afe3a78bb242c0a8bb181dcc8937bb16e5c66c   # black-forest-labs/FLUX.2-dev (ae.safetensors only)
QWEN3_REV=1cfa9a7208912126459214e8b04321603b3df60c     # Qwen/Qwen3-4B (text-cache precompute only)
export UV_CACHE_DIR=${ROOT}/.uv-cache HF_HOME=${ROOT}/.hf-home
unset HF_TOKEN
mkdir -p "${ROOT}/.done" "${ROOT}/logs" "${ROOT}/data" "${ROOT}/checkpoints"
cd "${ROOT}"
is_done() { [[ -f "${ROOT}/.done/$1" ]]; }
log() { echo "$(date -u +%FT%TZ) [setup] $*"; }

if ! is_done env; then
    log "system packages (ffmpeg libs for torchcodec)"
    command -v uv >/dev/null || pip install -q uv
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg tmux rsync >/dev/null
    log "repo ${REPO_REV} + flux2 ${FLUX2_SRC_REV:0:7} + venv"
    [[ -d ImageWAM ]] || git clone https://github.com/yuyangalin/ImageWAM.git
    git -C ImageWAM checkout -q "${REPO_REV}"
    [[ -d ImageWAM/third_party/flux2 ]] || git clone https://github.com/black-forest-labs/flux2 ImageWAM/third_party/flux2
    git -C ImageWAM/third_party/flux2 checkout -q "${FLUX2_SRC_REV}"
    # The README says `uv sync --extra shared`, but the repo's uv.lock predates the `shared` extra (and pins
    # transformers 4.49), so install from pyproject's pins instead: the cu118 torch stack the lock and
    # pyproject name, the package with its shared extra, then the FLUX-compatible transformers.
    uv venv --python 3.11 ImageWAM/.venv
    uv pip install --python ImageWAM/.venv/bin/python torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu118
    (cd ImageWAM && uv pip install --python .venv/bin/python -e '.[shared]')
    # torchcodec from PyPI (CPU build; decoding runs in CPU dataloader workers): the cu118 wheel needs the
    # CUDA 11 NPP libraries, which this CUDA 12 image lacks.
    uv pip install --python ImageWAM/.venv/bin/python "transformers==4.56.1"
    uv pip install --python ImageWAM/.venv/bin/python --reinstall-package torchcodec --index-url https://pypi.org/simple "torchcodec==0.5"
    ImageWAM/.venv/bin/python -c "import torch, transformers, torchcodec; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), transformers.__version__)"
    touch .done/env
fi
PY=${ROOT}/ImageWAM/.venv/bin/python
HF=${ROOT}/ImageWAM/.venv/bin/huggingface-cli
CK=${ROOT}/checkpoints

download_models() {
    is_done models && return 0
    log "FLUX.2 klein-base-4B + FLUX.2 AE + Qwen3-4B + InternData-A1-EE pretrain"
    "${HF}" download black-forest-labs/FLUX.2-klein-base-4B flux-2-klein-base-4b.safetensors --revision "${KLEIN_REV}" \
        --local-dir "${CK}/flux2/FLUX.2-klein-base-4B"
    "${HF}" download black-forest-labs/FLUX.2-dev ae.safetensors --revision "${FLUXDEV_REV}" --local-dir "${CK}/flux2/FLUX.2-dev"
    "${HF}" download Qwen/Qwen3-4B --revision "${QWEN3_REV}" --local-dir "${CK}/Qwen3-4B"
    "${HF}" download yuyangalin/ImageWAM-FLUX.2-4B-InternData-A1-EE --revision "${PRETRAIN_REV}" \
        --local-dir "${CK}/imagewam_interndata_a1_ee"
    touch "${ROOT}/.done/models"
}
download_dataset() {
    is_done dataset && return 0
    log "dataset ${DATASET_ID}@${DATASET_REV:0:7}"
    "${HF}" download "${DATASET_ID}" --repo-type dataset --revision "${DATASET_REV}" \
        --local-dir "${ROOT}/data/so101_bench_sim" --max-workers 16
    touch "${ROOT}/.done/dataset"
}
download_models > logs/setup_models.log 2>&1 &
p1=$!
download_dataset > logs/setup_dataset.log 2>&1 &
p2=$!
wait $p1 || { log "models FAILED"; exit 1; }
wait $p2 || { log "dataset FAILED"; exit 1; }
du -sh "${CK}"/* "${ROOT}/data"/* 2>/dev/null
log "setup complete"
