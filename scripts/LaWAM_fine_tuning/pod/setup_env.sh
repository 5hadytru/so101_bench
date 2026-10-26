#!/usr/bin/env bash
# LaWAM (LeRobot adapter, lerobot.policies.lawam) training environment on a 4xA100 pod. Everything on the
# container disk under ${ROOT}. Steps leave markers in ${ROOT}/.done so a rerun skips finished work.
set -euo pipefail
ROOT=${ROOT:-/root/lawam}
LEROBOT_REV=ca69a2068462a37f7cdcb74180927a2f863d2bf7      # huggingface/lerobot main, 2026-10-07 (LaWAM added c73cec5)
PRETRAIN_REV=bc4509057b933af38a4d7f231803e56dbad0d8c7     # jialei02/lawam-pretrain-lerobot
QWEN_REV=89644892e4d85e24eaac8bacfd4f463576704203         # Qwen/Qwen3-VL-2B-Instruct
DATASET_ID=5hadytru/so101_bench_sim
DATASET_REV=34233754992fbd77e42373bad44fd42ddfb43f14
export UV_CACHE_DIR=${ROOT}/.uv-cache HF_HOME=${ROOT}/.hf-home
unset HF_TOKEN
mkdir -p "${ROOT}/.done" "${ROOT}/logs" "${ROOT}/data" "${ROOT}/weights"
cd "${ROOT}"
is_done() { [[ -f "${ROOT}/.done/$1" ]]; }
log() { echo "$(date -u +%FT%TZ) [setup] $*"; }

if ! is_done env; then
    log "system packages + lerobot ${LEROBOT_REV:0:7} venv"
    command -v uv >/dev/null || pip install -q uv
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg tmux rsync >/dev/null
    [[ -d lerobot ]] || git clone https://github.com/huggingface/lerobot.git
    git -C lerobot checkout -q "${LEROBOT_REV}"
    uv venv --python 3.12 lerobot/.venv
    (cd lerobot && uv pip install --python .venv/bin/python -e '.[lawam,training]')
    lerobot/.venv/bin/python -c "import torch, transformers, lerobot.policies.lawam as m; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), transformers.__version__)"
    touch .done/env
fi
HF=${ROOT}/lerobot/.venv/bin/hf

download_models() {
    is_done models && return 0
    log "lawam-pretrain-lerobot + Qwen3-VL-2B-Instruct"
    "${HF}" download jialei02/lawam-pretrain-lerobot --revision "${PRETRAIN_REV}" --local-dir "${ROOT}/weights/lawam-pretrain-lerobot"
    # The policy config keeps the portable id Qwen/Qwen3-VL-2B-Instruct; prefetch it into HF_HOME at the pin.
    "${HF}" download Qwen/Qwen3-VL-2B-Instruct --revision "${QWEN_REV}"
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
du -sh "${ROOT}/weights"/* "${ROOT}/data"/* "${HF_HOME}" 2>/dev/null
log "setup complete"
