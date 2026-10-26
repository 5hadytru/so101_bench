#!/usr/bin/env bash
# Bring the run back after a pod restart. Everything lives on the container disk (/root); if it
# was wiped, rebuild the venv + data (setup_env.sh, prepare_data.sh; ~15 min, needs these scripts
# and secrets re-copied from the workstation) and restore the newest checkpoint backup from
# /workspace/molmoact2-so101-backup. Then (re)start the supervisor, which verifies the newest
# checkpoint and resumes from it.
set -euo pipefail
WORK=/root/molmoact2-so101
RUN_NAME=$(python3 -c "import json;print(json.load(open('$WORK/supervisor_config.json'))['RUN_NAME'])")

if [ ! -x /root/molmoact2-venv/bin/python ]; then
    bash "$WORK/setup_env.sh"
fi
if [ ! -d /root/datasets/lerobot/5hadytru/so101_bench_sim_WM_molmoframe ]; then
    bash "$WORK/prepare_data.sh"
fi
/root/molmoact2-venv/bin/python -c "import torch, olmo, lerobot; assert torch.cuda.device_count() == 4, torch.cuda.device_count(); print('venv ok', torch.__version__)"

RUN_DIR=$WORK/checkpoints/$RUN_NAME
if ! ls -d "$RUN_DIR"/step* >/dev/null 2>&1; then
    BACKUP=$(ls -d /workspace/molmoact2-so101-backup/"$RUN_NAME"/step* 2>/dev/null | grep -v '\.tmp$' | tail -1 || true)
    if [ -n "$BACKUP" ] && [ -f "$BACKUP/BACKUP_OK" ]; then
        echo "restoring $BACKUP"
        mkdir -p "$RUN_DIR"
        cp -r "$BACKUP" "$RUN_DIR/"
        rm -f "$RUN_DIR/$(basename "$BACKUP")/BACKUP_OK"
        cp "$(dirname "$BACKUP")/config.yaml" "$RUN_DIR/config.yaml"
    fi
fi

if tmux has-session -t supervisor 2>/dev/null; then
    echo "supervisor session already running"
else
    tmux new-session -d -s supervisor "/usr/bin/python3 $WORK/supervise.py >> $WORK/logs/supervisor_stdout.log 2>&1"
    echo "supervisor started"
fi
