#!/usr/bin/env bash
# Download the pinned dataset + base checkpoint and build the checkpoint-joint-frame copy.
# Run after setup_env.sh. Secrets (hf_token, wandb_api_key) must already be in $WORK/secrets.
set -euo pipefail

WORK=${WORK:-/root/molmoact2-so101}
DATA_ROOT=${LEROBOT_DATA_ROOT:-/root/datasets/lerobot}
DATASET_REV=${DATASET_REV:-b569ad81ee60ffe3580bc0f1df513bde1bb36e8c}
MODEL_ID=${MODEL_ID:-allenai/MolmoAct2}
MODEL_REV=${MODEL_REV:-e432d85f6e039edca44afb93c262f3084ab72a9c}
# shellcheck disable=SC1091
source /root/molmoact2-venv/bin/activate
export HF_HOME=$WORK/hf_home
export HF_HUB_ENABLE_HF_TRANSFER=0
HF_TOKEN=$(cat "$WORK/secrets/hf_token")
export HF_TOKEN
SRC=$DATA_ROOT/5hadytru/so101_bench_sim_WM
DST=$DATA_ROOT/5hadytru/so101_bench_sim_WM_molmoframe

hf download 5hadytru/so101_bench_sim_WM --repo-type dataset --revision "$DATASET_REV" \
    --local-dir "$SRC" --max-workers 16
hf download "$MODEL_ID" --revision "$MODEL_REV"
# OLMo's build_tokenizer() loads the Qwen3-4B tokenizer by name. Name the files explicitly: an
# --include download on 2026-09-30 fetched only vocab.json/merges.txt, which builds a tokenizer
# without Qwen's special tokens (len 151644, <|im_start|> split into 6 pieces, +13 tokens per
# example); the first launch of the run trained on that for 350 steps and was discarded.
hf download Qwen/Qwen3-4B tokenizer.json tokenizer_config.json vocab.json merges.txt config.json generation_config.json
python - <<'PY'
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
assert len(t) == 151669 and t.encode("<|im_start|>", add_special_tokens=False) == [151644], len(t)
print("Qwen3-4B tokenizer ok", len(t))
PY
# The trainer loads the repo by name with HF_HUB_OFFLINE=1, which resolves refs/main; a
# --revision download does not write it.
MODEL_CACHE="$HF_HOME/hub/models--${MODEL_ID//\//--}"
mkdir -p "$MODEL_CACHE/refs"
printf '%s' "$MODEL_REV" > "$MODEL_CACHE/refs/main"

# Every file against the Hub's sha256 (writes the manifest the supervisor spot-checks).
python "$WORK/verify_dataset.py" --root "$SRC" --revision "$DATASET_REV" --sha --parquet \
    --manifest "$WORK/dataset_manifest.json"
python "$WORK/make_molmoframe_dataset.py" --src "$SRC" --dst "$DST"
echo "PREPARE_DATA_OK"
