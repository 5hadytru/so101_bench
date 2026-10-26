#!/usr/bin/env bash
# Best-effort copy of one training checkpoint to /workspace (RunPod MooseFS network volume), so a
# container reset (which wipes /root) does not lose the run. Keeps only the newest backup.
#   backup_checkpoint.sh <run_dir>/stepN <backup_run_dir>
# restore: copy <backup_run_dir>/stepN back to /root/molmoact2-so101/checkpoints/<RUN_NAME>/ and
# start the supervisor (it verifies the checkpoint is fully readable before resuming).
set -euo pipefail
SRC=$(realpath "$1")
DEST_ROOT=$2
NAME=$(basename "$SRC")
mkdir -p "$DEST_ROOT"
rm -rf "$DEST_ROOT/$NAME.tmp"
nice -n 10 ionice -c3 cp -r "$SRC" "$DEST_ROOT/$NAME.tmp"
mv "$DEST_ROOT/$NAME.tmp" "$DEST_ROOT/$NAME"
touch "$DEST_ROOT/$NAME/BACKUP_OK"
for old in "$DEST_ROOT"/step*; do
    [ "$old" = "$DEST_ROOT/$NAME" ] && continue
    rm -rf "$old"
done
# The trainer resumes from <save_folder>/stepN and reads <save_folder>/config.yaml.
cp "$(dirname "$SRC")/config.yaml" "$DEST_ROOT/config.yaml"
cp /root/molmoact2-so101/supervisor_config.json "$DEST_ROOT/" 2>/dev/null || true
echo "BACKED_UP $SRC -> $DEST_ROOT/$NAME"
du -sh "$DEST_ROOT/$NAME"
