#!/usr/bin/env bash
# Bring the fine-tune back after a pod stop/start. The container disk (/root) is wiped
# on restart, so restore the venv from its /workspace backup (or rebuild it), then
# (re)start the supervisor, which resumes from the newest verified checkpoint.
set -euo pipefail
WORK=/workspace/molmoact2-so101

if [ ! -x /root/molmoact2-venv/bin/python ]; then
    if [ -f "$WORK/venv_backup.tar" ]; then
        echo "restoring venv from $WORK/venv_backup.tar"
        tar -C /root -xf "$WORK/venv_backup.tar"
    else
        echo "no venv backup; rebuilding"
        bash "$WORK/setup_env.sh"
    fi
fi
/root/molmoact2-venv/bin/python -c "import torch, olmo, lerobot; assert torch.cuda.device_count() == 4, torch.cuda.device_count(); print('venv ok', torch.__version__)"

if tmux has-session -t supervisor 2>/dev/null; then
    echo "supervisor session already running"
else
    tmux new-session -d -s supervisor "/usr/bin/python3 $WORK/supervise.py >> $WORK/logs/supervisor_stdout.log 2>&1"
    echo "supervisor started"
fi
