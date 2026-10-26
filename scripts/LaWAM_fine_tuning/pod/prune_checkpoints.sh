#!/usr/bin/env bash
# Training-pod disk rotation for the full run: LeRobot keeps every checkpoint (15.3 GB each: 6.7 GB
# pretrained_model + 8.6 GB training_state), which would fill the 400 GB disk. Every 10 min, for every
# checkpoint except checkpoints/last: delete it unless it is scheduled for validation (update multiple of
# EVERY, or the final); for scheduled ones drop training_state at once and delete the rest once the
# validation pod has pulled it (<ckpt>/PULLED).
set -uo pipefail
RUN=${RUN:-/root/lawam/runs/lawam-so101-b256-ah36}
ACCUM=${ACCUM:-2}
EVERY=${EVERY:-5000}
FINAL=${FINAL:-49000}
while true; do
    last=$(readlink -f "${RUN}/checkpoints/last" 2>/dev/null)
    for c in "${RUN}"/checkpoints/[0-9]*; do
        [[ -d "$c" && "$(readlink -f "$c")" != "$last" ]] || continue
        update=$(( 10#$(basename "$c") / ACCUM ))
        if (( update % EVERY != 0 && update != FINAL )) || [[ -f "$c/PULLED" ]]; then
            rm -rf "$c" && echo "$(date -u +%FT%TZ) pruned $(basename "$c") (update ${update})"
        elif [[ -d "$c/training_state" ]]; then
            rm -rf "$c/training_state" && echo "$(date -u +%FT%TZ) dropped training_state of $(basename "$c") (update ${update})"
        fi
    done
    sleep 600
done
