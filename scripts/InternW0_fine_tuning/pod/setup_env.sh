#!/usr/bin/env bash
# InternW0-Delta training environment on a 4xA100 pod. Everything on the container disk under ${ROOT}.
# Steps leave markers in ${ROOT}/.done so a rerun skips finished work.
set -euo pipefail
ROOT=${ROOT:-/root/internw0}
REPO_REV=90801baa3bdc7829c1e4989edfe3d0fb2410ea6a
DATASET_ID=5hadytru/so101_bench_sim
DATASET_REV=34233754992fbd77e42373bad44fd42ddfb43f14
# Model revisions used by the 2026-10-01 run (sha256 of the large files in records/model_manifest.json).
WAN22_REV=921dbaf3f1674a56f47e83fb80a34bac8a8f203e
WAN21_REV=37ec512624d61f7aa208f7ea8140a131f93afc9a
BASE_REV=ca534d4f3c9c7131205b4ee37e965b3ef21ffb5f
RYNN_REV=36069aa841fc58f500d307316cbfdf379777ce8a
export UV_CACHE_DIR=${ROOT}/.uv-cache HF_HOME=${ROOT}/.hf-home
unset HF_TOKEN
mkdir -p "${ROOT}/.done" "${ROOT}/logs" "${ROOT}/data"
cd "${ROOT}"
is_done() { [[ -f "${ROOT}/.done/$1" ]]; }
log() { echo "$(date -u +%FT%TZ) [setup] $*"; }

if ! is_done env; then
    log "repo ${REPO_REV:0:7} + venv"
    [[ -d InternW0-Delta ]] || git clone https://github.com/InternRobotics/InternW0-Delta.git
    git -C InternW0-Delta checkout -q "${REPO_REV}"
    uv venv --python 3.11 InternW0-Delta/.venv
    uv pip install --python InternW0-Delta/.venv/bin/python torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
    (cd InternW0-Delta && uv pip install --python .venv/bin/python -e '.[train,modelscope]')
    # Not in the repo's extras, but wandb.enabled=true (the full run) imports it.
    uv pip install --python InternW0-Delta/.venv/bin/python wandb==0.30.0
    InternW0-Delta/.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
    touch .done/env
fi
PY=${ROOT}/InternW0-Delta/.venv/bin/python
HF=${ROOT}/InternW0-Delta/.venv/bin/hf
CK=${ROOT}/InternW0-Delta/checkpoints

download_models() {
    is_done models && return 0
    log "backbones + base policy"
    "${HF}" download Wan-AI/Wan2.2-TI2V-5B --revision "${WAN22_REV}" --include 'diffusion_pytorch_model*.safetensors' --include '*.json' --local-dir "${CK}/Wan-AI/Wan2.2-TI2V-5B"
    # Not on Hugging Face (2026-10-01); DiffSynth-Studio publishes it on ModelScope (master; the files are
    # checked against their sha256 in records/model_manifest.json).
    "${PY}" -c "from modelscope import snapshot_download; snapshot_download('DiffSynth-Studio/Wan-Series-Converted-Safetensors', allow_patterns=['models_t5_umt5-xxl-enc-bf16.safetensors', 'Wan2.2_VAE.safetensors'], local_dir='${CK}/DiffSynth-Studio/Wan-Series-Converted-Safetensors')"
    "${HF}" download Wan-AI/Wan2.1-T2V-1.3B --revision "${WAN21_REV}" --include 'google/umt5-xxl/*' --local-dir "${CK}/Wan-AI/Wan2.1-T2V-1.3B"
    "${HF}" download InternRobotics/InternW0-Delta-Base --revision "${BASE_REV}" --local-dir "${CK}/InternW0-Delta-Base"
    ln -sfn "${CK}/InternW0-Delta-Base/pretrain.pt" "${CK}/pretrain.pt"
    # On Hugging Face (2026-10-01) although the repo docs point to ModelScope, where it is missing.
    "${HF}" download Alibaba-DAMO-Academy/RynnBrain1.1-2B --revision "${RYNN_REV}" --local-dir "${CK}/RynnBrain1.1-2B"
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
