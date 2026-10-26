#!/usr/bin/env bash
# Build the FLUX 3 Action environments on a 4xA100 training pod and fetch weights + data.
# Everything lives on the container disk under ${ROOT} (/workspace on RunPod secure cloud is MooseFS).
#   flux-action (standalone full fine-tune): ${ROOT}/flux-action/.venv  (torch 2.10 cu128, NATTEN)
#   LeRobot main (LoRA):                     ${ROOT}/lerobot/.venv      (torch 2.11 cu128, NATTEN)
# Each step writes a marker in ${ROOT}/.done/ so a rerun skips finished work.
set -euo pipefail

ROOT="${ROOT:-/root/flux3}"
FLUX_ACTION_REV="e2dd1d8dbc5977b54315d61f7548c63c043d6d4f"
LEROBOT_REV="e0d50211ef236143ae867228662b7dfaba554f02"
BASE_REV="62878e2925e59b7a89ec14463ce89932624c490d"
SO101_REV="c9e13b2aca6a0a472b3ea03fd90cff32d3e85849"
DATASET_ID="5hadytru/so101_bench_sim"
DATASET_REV="34233754992fbd77e42373bad44fd42ddfb43f14"
export UV_CACHE_DIR="${ROOT}/.uv-cache"
export HF_HOME="${ROOT}/.hf-home"
unset HF_TOKEN

mkdir -p "${ROOT}/.done" "${ROOT}/weights" "${ROOT}/datasets" "${ROOT}/logs"
cd "${ROOT}"
done_mark() { touch "${ROOT}/.done/$1"; }
is_done() { [[ -f "${ROOT}/.done/$1" ]]; }
log() { echo "$(date -u +%FT%TZ) [setup] $*"; }

hf_cli() { "${ROOT}/flux-action/.venv/bin/hf" "$@"; }

if ! is_done flux_action_env; then
    log "flux-action ${FLUX_ACTION_REV}"
    [[ -d flux-action ]] || git clone https://github.com/black-forest-labs/flux-action.git
    git -C flux-action checkout -q "${FLUX_ACTION_REV}"
    (cd flux-action && uv sync --locked --extra encoders --extra data --extra serve)
    uv pip install --python flux-action/.venv/bin/python 'natten==0.21.6+torch2100cu128' -f https://whl.natten.org/
    flux-action/.venv/bin/python -c "import torch, natten; print('flux-action torch', torch.__version__, 'natten', natten.__version__, torch.cuda.get_device_capability())"
    done_mark flux_action_env
fi

download_weights() {
    is_done weights && return 0
    log "weights: base@${BASE_REV:0:7}, so101@${SO101_REV:0:7}"
    hf_cli download black-forest-labs/flux-3-action-base --revision "${BASE_REV}" \
        --include 'flux-3-action-base.safetensors' --include 'video_vae.safetensors' --include 'text_encoder/*' \
        --local-dir "${ROOT}/weights/base"
    hf_cli download black-forest-labs/flux-3-action-so101 --revision "${SO101_REV}" \
        --local-dir "${ROOT}/weights/so101"
    # The standalone trainer loads trunk weights without LeRobot's "dit." key prefix; the SO-101
    # package's action/action_cond heads then load as-is (modality "action").
    "${ROOT}/flux-action/.venv/bin/python" - <<EOF
from safetensors.torch import load_file, save_file
sd = load_file("${ROOT}/weights/so101/model.safetensors")
assert all(k.startswith("dit.") for k in sd), "unexpected key prefix"
save_file({k.removeprefix("dit."): v for k, v in sd.items()}, "${ROOT}/weights/flux-3-action-so101-trunk.safetensors", metadata={"format": "pt", "source": "black-forest-labs/flux-3-action-so101@${SO101_REV}"})
print("stripped", len(sd), "tensors")
EOF
    done_mark weights
}

download_dataset() {
    is_done dataset && return 0
    log "dataset ${DATASET_ID}@${DATASET_REV:0:7}"
    hf_cli download "${DATASET_ID}" --repo-type dataset --revision "${DATASET_REV}" \
        --local-dir "${ROOT}/datasets/so101_bench_sim" --max-workers 16
    done_mark dataset
}

build_lerobot() {
    is_done lerobot_env && return 0
    log "lerobot ${LEROBOT_REV}"
    [[ -d lerobot ]] || git clone https://github.com/huggingface/lerobot.git
    git -C lerobot checkout -q "${LEROBOT_REV}"
    (cd lerobot && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[training,flux3,peft,diffusion]')
    local tv
    tv=$(lerobot/.venv/bin/python -c "import torch; print(torch.__version__.split('+')[0].replace('.', '')[:4])")
    uv pip install --python lerobot/.venv/bin/python "natten==0.21.6+torch${tv}0cu128" -f https://whl.natten.org/ \
        || uv pip install --python lerobot/.venv/bin/python "natten==0.21.6+torch${tv}cu128" -f https://whl.natten.org/
    lerobot/.venv/bin/python -c "import torch, natten, peft; print('lerobot torch', torch.__version__, 'natten', natten.__version__)"
    done_mark lerobot_env
}

# Downloads and the LeRobot build are independent; run them side by side.
download_weights > "${ROOT}/logs/setup_weights.log" 2>&1 &
p_w=$!
download_dataset > "${ROOT}/logs/setup_dataset.log" 2>&1 &
p_d=$!
build_lerobot > "${ROOT}/logs/setup_lerobot.log" 2>&1 &
p_l=$!
rc=0
wait "${p_w}" || { log "weights FAILED"; rc=1; }
wait "${p_d}" || { log "dataset FAILED"; rc=1; }
wait "${p_l}" || { log "lerobot FAILED"; rc=1; }
((rc == 0)) || exit 1

if ! is_done index; then
    log "index-lerobot"
    (cd flux-action && .venv/bin/flux-action index-lerobot --source-root "${ROOT}/datasets/so101_bench_sim" \
        --output-dir "${ROOT}/index/so101_bench_sim" \
        --camera scene=observation.images.overhead --camera wrist=observation.images.wrist \
        --dataset-id "${DATASET_ID}" --revision "${DATASET_REV}" --chunk-size 42 --val-episodes 10)
    done_mark index
fi
if ! is_done stats_package; then
    # The LoRA run's base: the SO-101 package with so101_bench_sim's own normalization
    # (byte-identical to records/lora_run/so101_ourstats/).
    log "so101_ourstats package"
    HF_HUB_OFFLINE=1 lerobot/.venv/bin/python "${ROOT}/pod/make_stats_package.py" "${ROOT}/weights/so101" \
        "${ROOT}/index/so101_bench_sim/statistics.json" "${ROOT}/weights/so101_ourstats"
    done_mark stats_package
fi
log "setup complete"
