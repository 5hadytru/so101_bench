#!/usr/bin/env bash
# Measure what the full run's checkpoint/export cycle costs on this pod: train 3 updates with the
# bench's fft config, write the resumable step-3 checkpoint, export the bf16 ema_0p10 profile.
# The export then feeds the validation-pod test (flux3_server.py --export).
set -euo pipefail
ROOT=/root/flux3
OUT=${OUT:-${ROOT}/ckpt_test}
EXPORT=${EXPORT:-${ROOT}/export/ckpt-test-step3}
BENCH_CONFIG=${BENCH_CONFIG:-${ROOT}/bench/fft_mb4_ema1/bench_config.json}
export HF_HUB_OFFLINE=1 HF_HOME=${ROOT}/.hf-home PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
unset HF_TOKEN
rm -rf "${OUT}" "${EXPORT}"
mkdir -p "${OUT}" "$(dirname "${EXPORT}")"
python3 - "${BENCH_CONFIG}" "${OUT}" <<'EOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg.update(output_dir=sys.argv[2], steps=3, checkpoint_every=3)
json.dump(cfg, open(f"{sys.argv[2]}/config.json", "w"), indent=2)
EOF
cd "${ROOT}/flux-action"
start=$(date +%s)
.venv/bin/torchrun --standalone --nproc_per_node=4 -m flux_action.cli train --config "${OUT}/config.json" > "${OUT}/train.log" 2>&1
echo "train+checkpoint wall: $(( $(date +%s) - start )) s"
grep -a '"event": "checkpoint"' "${OUT}/metrics.jsonl"
du -sh "${OUT}"/step-3
start=$(date +%s)
.venv/bin/flux-action export-checkpoint --checkpoint "${OUT}/step-3" --output "${EXPORT}" --profile ema_0p10 --dtype bfloat16 > "${OUT}/export.log" 2>&1
echo "export wall: $(( $(date +%s) - start )) s"
du -sh "${EXPORT}"; ls -la "${EXPORT}"
df -h / | tail -1
