#!/usr/bin/env bash
# Training-pod disk rotation for the full run: every 10 min, delete weight checkpoints that are neither the
# latest nor scheduled for validation, and scheduled ones the validation pod has pulled (<file>.PULLED).
# 33 x 8.5 GB of weights would not fit next to the data and two 60 GB states on the 400 GB disk.
set -uo pipefail
RUN=${RUN:-/root/imagewam/runs/imagewam-so101-a-b128}
EVERY=${EVERY:-3000}
FINAL=${FINAL:-33000}
while true; do
    latest=$(ls "${RUN}"/checkpoints/weights/step_*.pt 2>/dev/null | sort | tail -1)
    for w in "${RUN}"/checkpoints/weights/step_*.pt; do
        [[ -f "$w" && "$w" != "$latest" ]] || continue
        step=$((10#$(basename "$w" .pt | tr -dc 0-9)))
        if (( step % EVERY != 0 && step != FINAL )) || [[ -f "$w.PULLED" ]]; then
            rm -f "$w" "$w.PULLED" && echo "$(date -u +%FT%TZ) pruned $(basename "$w")"
        fi
    done
    sleep 600
done
