#!/usr/bin/env bash
# Export a sharded training checkpoint (model_and_optim) to a self-contained bf16 HF
# checkpoint that scripts/molmoact2_server.py can load (--repo_id <dir>).
#   export_checkpoint.sh <run_dir>/stepN [out_dir]
# Writes <out_dir>/EXPORT_OK last, so a partially written export is never pulled.
set -euo pipefail

STEP_DIR=$(realpath "$1")
WORK=${WORK:-/root/molmoact2-so101}
OUT=${2:-$WORK/export/$(basename "$(dirname "$STEP_DIR")")-$(basename "$STEP_DIR")}
REPO=${REPO:-/root/molmoact2}

# shellcheck disable=SC1091
source /root/molmoact2-venv/bin/activate
export PYTHONPATH="$REPO/experiments:$REPO/experiments/lerobot/src:${PYTHONPATH:-}"
export HF_HOME=$WORK/hf_home HF_HUB_OFFLINE=1 MOLMO_DATA_DIR=$WORK/molmo_data
# Keep the export off the training GPUs.
export CUDA_VISIBLE_DEVICES=""

rm -rf "$OUT.tmp"
cd "$REPO/experiments"
nice -n 10 python -m olmo.hf_model.convert_molmoact2_to_hf "$STEP_DIR" "$OUT.tmp"
nice -n 10 python "$WORK/to_bf16.py" "$OUT.tmp"
echo "{\"source\": \"$STEP_DIR\", \"exported_at\": \"$(date -u +%FT%TZ)\"}" > "$OUT.tmp/export_source.json"
rm -rf "$OUT"
mv "$OUT.tmp" "$OUT"
touch "$OUT/EXPORT_OK"
echo "EXPORTED $OUT"
