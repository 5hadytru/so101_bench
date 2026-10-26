#!/usr/bin/env bash
# One-off (2026-10-01): the full run measured 20.6 s/update instead of the sweep's 20.0, so 6480 updates
# would overrun the 36 h budget. With a constant LR only the stopping point changes: once the first
# checkpoint (500 updates = 8000 microsteps) is complete, stop training, set the checkpoint's
# train_config steps to NEW_UPDATES x 16 microsteps, and restart the supervisor, which resumes from it.
set -euo pipefail
ROOT=/root/flux3
RUN=${ROOT}/runs/flux3-so101-lora-b128
NEW_UPDATES=${NEW_UPDATES:-6280}
CKPT=${RUN}/checkpoints/008000
log() { echo "$(date -u +%FT%TZ) [retarget] $*" | tee -a "${RUN}.supervisor.log"; }
until [[ -f "${CKPT}/training_state/training_step.json" ]] && (( $(date +%s) - $(stat -c %Y "${CKPT}/training_state/training_step.json") > 60 )); do
    sleep 30
done
[[ "$(readlink -f "${RUN}/checkpoints/last")" == "$(readlink -f "${CKPT}")" ]] || { log "last does not point at ${CKPT}; aborting"; exit 1; }
log "stopping training to retarget steps to ${NEW_UPDATES} updates"
tmux kill-session -t train
sleep 10
pkill -INT -f "[l]erobot_train" || true
for _ in $(seq 60); do [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && break; sleep 5; done
for f in "${CKPT}/pretrained_model/train_config.json" "${CKPT}/pretrained_model_ema/train_config.json"; do
    [[ -f "${f}" ]] || continue
    python3 - "${f}" "$((NEW_UPDATES * 16))" <<'EOF'
import json, sys
path, steps = sys.argv[1], int(sys.argv[2])
cfg = json.load(open(path))
old = cfg["steps"]
cfg["steps"] = steps
json.dump(cfg, open(path, "w"), indent=4)
print(f"{path}: steps {old} -> {steps}")
EOF
done
cd "${ROOT}" && tmux new -d -s train "bash pod/run_lora_full.sh"
log "restarted supervisor (resumes from ${CKPT})"
