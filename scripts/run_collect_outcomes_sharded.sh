#!/usr/bin/env bash
#
# Run so101_lerobot_collect_outcomes.py in parallel over disjoint dataset slices.
#
# collect_outcomes can replay multiple native Isaac Lab environments in one process. This
# wrapper remains useful for process-level or hybrid sharding: it sizes the number of
# processes from live RAM/VRAM headroom, splits episodes evenly, launches one process per
# shard, and waits. The RAM/VRAM estimates assume NATIVE_ENVS=1.
#
# Usage:
#   scripts/run_collect_outcomes_sharded.sh [extra args passed through to the python script]
#
# Common overrides (environment variables):
#   SHARDS=4                 # force shard count instead of auto-sizing
#   NATIVE_ENVS=4            # Isaac Lab replay lanes within each shard process
#   FRAME_SOURCE=none        # none|dataset|sim   (default: none = no rendering, fastest)
#   REPO_ROOT=data/lerobot/so101_bench_sim_1_v3.0
#   PER_PROC_RAM_GB=5        # estimated RAM per process (tune if you see swapping/OOM)
#   PER_PROC_VRAM_GB=3       # estimated VRAM per process
#   RAM_RESERVE_GB=2         # RAM left free for the OS / desktop
#   MAX_SHARDS=8             # hard cap regardless of headroom
#   OUTPUT_ROOT=outputs/outcomes_sharded_<timestamp>
#   DRY_RUN=1                # print the plan and the commands, but launch nothing
#
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

# Force unbuffered stdout/stderr so per-episode [INFO] lines stream to the shard logs in
# real time. Without this, Python fully-buffers stdout when redirected to a file, so logs
# appear to "freeze" even though the run is progressing (npz/state files keep landing).
export PYTHONUNBUFFERED=1

ISAACLAB="${ISAACLAB:-$HOME/IsaacLab/isaaclab.sh}"
# Python with pandas (the isaaclab venv) for the frame-balanced shard planner.
PLAN_PYTHON="${PLAN_PYTHON:-$HOME/env_isaaclab_51/bin/python}"
SCRIPT="scripts/so101_lerobot_collect_outcomes.py"
FRAME_SOURCE="${FRAME_SOURCE:-none}"
NATIVE_ENVS="${NATIVE_ENVS:-1}"
REPO_ROOT="${REPO_ROOT:-data/lerobot/so101_bench_sim_1_v3.0}"
PER_PROC_RAM_GB="${PER_PROC_RAM_GB:-5}"
PER_PROC_VRAM_GB="${PER_PROC_VRAM_GB:-3}"
RAM_RESERVE_GB="${RAM_RESERVE_GB:-2}"
# Measured GPU knee for the no-render path on the 3090: throughput plateaus at 2 shards
# (1->2 = +24%, 2->3 = +1%; GPU saturates, ~1.25x is the ceiling). Capping here keeps the
# bare invocation optimal. Raise MAX_SHARDS / set SHARDS explicitly for other hardware or
# the rendering (frame_source=sim) path.
MAX_SHARDS="${MAX_SHARDS:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/outcomes_sharded_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

# --- total episodes from the LeRobot dataset meta ---------------------------------------
INFO_JSON="$REPO_ROOT/meta/info.json"
if [[ ! -f "$INFO_JSON" ]]; then
  echo "ERROR: dataset meta not found: $INFO_JSON (set REPO_ROOT)" >&2
  exit 1
fi
TOTAL_EPISODES="$(grep -o '"total_episodes"[[:space:]]*:[[:space:]]*[0-9]*' "$INFO_JSON" | grep -o '[0-9]*$')"
if [[ -z "${TOTAL_EPISODES:-}" || "$TOTAL_EPISODES" -le 0 ]]; then
  echo "ERROR: could not read total_episodes from $INFO_JSON" >&2
  exit 1
fi

# --- auto-size shard count from live headroom -------------------------------------------
if [[ -z "${SHARDS:-}" ]]; then
  AVAIL_RAM_GB="$(free -g | awk '/^Mem:/ {print $7}')"            # "available" column
  USABLE_RAM_GB=$(( AVAIL_RAM_GB - RAM_RESERVE_GB ))
  (( USABLE_RAM_GB < 0 )) && USABLE_RAM_GB=0
  SHARDS_RAM=$(( USABLE_RAM_GB / PER_PROC_RAM_GB ))

  if command -v nvidia-smi >/dev/null 2>&1; then
    FREE_VRAM_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    SHARDS_VRAM=$(( FREE_VRAM_MB / (PER_PROC_VRAM_GB * 1024) ))
  else
    SHARDS_VRAM=$MAX_SHARDS
  fi

  NCPU="$(nproc)"
  SHARDS=$MAX_SHARDS
  for cap in "$SHARDS_RAM" "$SHARDS_VRAM" "$NCPU" "$TOTAL_EPISODES"; do
    (( cap < SHARDS )) && SHARDS=$cap
  done
  (( SHARDS < 1 )) && SHARDS=1
  echo "[plan] auto-sized shards=$SHARDS  (ram-limit=$SHARDS_RAM vram-limit=$SHARDS_VRAM cpu=$NCPU episodes=$TOTAL_EPISODES)"
else
  (( SHARDS > TOTAL_EPISODES )) && SHARDS=$TOTAL_EPISODES
  echo "[plan] using forced shards=$SHARDS"
fi

# --- repo_root / frame_source flags ------------------------------------------------------
EXTRA_FLAGS=(--frame_source "$FRAME_SOURCE" --repo_root "$REPO_ROOT" --num_envs "$NATIVE_ENVS")
[[ "$FRAME_SOURCE" == "sim" ]] && echo "[plan] WARNING: --frame_source sim renders cameras; RAM/VRAM cost per shard is much higher than the estimates above."

[[ "$DRY_RUN" != "1" ]] && mkdir -p "$OUTPUT_ROOT"
echo "[plan] dataset=$REPO_ROOT total_episodes=$TOTAL_EPISODES frame_source=$FRAME_SOURCE native_envs=$NATIVE_ENVS output_root=$OUTPUT_ROOT"

# --- frame-balanced contiguous boundaries and launch ------------------------------------
# Episodes vary widely in length (frame count) and the dataset is grouped (short episodes
# first, long ones later), so an equal-COUNT split leaves some shards doing far more work
# and the whole job waits on that straggler. shard_plan.py partitions into contiguous
# chunks with near-equal total frames instead, so shards finish together.
PLAN="$("$PLAN_PYTHON" scripts/shard_plan.py --repo_root "$REPO_ROOT" --shards "$SHARDS" --summary)"
STARTS=(); COUNTS=()
while read -r s c _; do [[ -n "$s" ]] && { STARTS+=("$s"); COUNTS+=("$c"); }; done <<<"$PLAN"

PIDS=()
LOGS=()
trap 'echo "[abort] killing shards..."; kill "${PIDS[@]}" 2>/dev/null || true' INT TERM

for i in "${!STARTS[@]}"; do
  START="${STARTS[$i]}"
  COUNT="${COUNTS[$i]}"
  OUT="$OUTPUT_ROOT/shard_$(printf '%02d' "$i")"
  LOG="$OUTPUT_ROOT/shard_$(printf '%02d' "$i").log"
  CMD=("$ISAACLAB" -p "$SCRIPT" --headless
       "${EXTRA_FLAGS[@]}"
       --dataset_episode_index "$START" --num_episodes "$COUNT"
       --output_dir "$OUT" "$@")
  echo "[shard $i] episodes [$START, $((START + COUNT)))  -> $OUT"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '          %q ' "${CMD[@]}"; echo
    continue
  fi
  "${CMD[@]}" >"$LOG" 2>&1 &
  PIDS+=("$!")
  LOGS+=("$LOG")
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] nothing launched."
  exit 0
fi

echo "[run] launched ${#PIDS[@]} shard(s); logs in $OUTPUT_ROOT/shard_*.log"
FAIL=0
for idx in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$idx]}"; then
    echo "[fail] shard $idx exited non-zero; see ${LOGS[$idx]}" >&2
    FAIL=1
  fi
done

if (( FAIL )); then
  echo "[done] one or more shards FAILED. Outputs (partial) under $OUTPUT_ROOT" >&2
  exit 1
fi
echo "[done] all shards completed. Per-shard outputs under $OUTPUT_ROOT/shard_*/"
echo "       (each shard wrote its own episodes.jsonl/summary.json; merge them as a separate step.)"
