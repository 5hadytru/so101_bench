#!/usr/bin/env bash
#
# Speed test for sharded so101_lerobot_collect_outcomes.py.
#
# Runs the SAME fixed batch of episodes at 1, 2, 3, 4 shards *sequentially* (one config at a
# time, so configs don't contend) and reports wall time + throughput for each, plus speedup
# vs the 1-shard baseline. Answers "is sharding actually faster, and where's the knee?".
#
# Two metrics are reported per config:
#   * wall epi/min   - EPISODES / total wall time, INCLUDING the ~1-2 min Isaac Sim startup
#                      each process pays. This is what a real run actually experiences.
#   * steady epi/min - derived from the .npz completion timestamps (first->last), EXCLUDING
#                      startup. This is the raw steady-state throughput once warmed up.
# If steady scales with shards but wall doesn't, startup is dominating your batch size.
#
# Usage:
#   scripts/speed_test_collect_outcomes.sh
#
# Overrides (env vars):
#   EPISODES=8                 # fixed batch size used for every config (>= max shard count)
#   SHARD_COUNTS="1 2 3 4"     # configurations to sweep, in order
#   NATIVE_ENVS=1              # Isaac Lab replay lanes per shard process
#   FRAME_SOURCE=none          # none|dataset|sim (default none = no rendering)
#   REPO_ROOT=data/lerobot/so101_bench_sim_1_v3.0
#   OUTPUT_ROOT=outputs/speedtest_<timestamp>
#   DRY_RUN=1                  # print the plan only
#
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
export PYTHONUNBUFFERED=1

ISAACLAB="${ISAACLAB:-$HOME/IsaacLab/isaaclab.sh}"
# Python with pandas (the isaaclab venv) for the frame-balanced shard planner.
PLAN_PYTHON="${PLAN_PYTHON:-$HOME/env_isaaclab_51/bin/python}"
SCRIPT="scripts/so101_lerobot_collect_outcomes.py"
EPISODES="${EPISODES:-16}"
SHARD_COUNTS="${SHARD_COUNTS:-1 2 3 4}"
NATIVE_ENVS="${NATIVE_ENVS:-1}"
FRAME_SOURCE="${FRAME_SOURCE:-none}"
REPO_ROOT="${REPO_ROOT:-data/lerobot/so101_bench_sim_1_v3.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/speedtest_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

# --- sanity: dataset present and big enough ---------------------------------------------
INFO_JSON="$REPO_ROOT/meta/info.json"
[[ -f "$INFO_JSON" ]] || { echo "ERROR: dataset meta not found: $INFO_JSON (set REPO_ROOT)" >&2; exit 1; }
TOTAL_EPISODES="$(grep -o '"total_episodes"[[:space:]]*:[[:space:]]*[0-9]*' "$INFO_JSON" | grep -o '[0-9]*$')"
if (( EPISODES > TOTAL_EPISODES )); then
  echo "ERROR: EPISODES=$EPISODES exceeds dataset total_episodes=$TOTAL_EPISODES" >&2; exit 1
fi

# --- warn if the GPU is already busy (results would be polluted) ------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  BUSY="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
  UTIL="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if (( BUSY > 0 )); then
    echo "[warn] $BUSY compute process(es) already on the GPU (util=${UTIL}%). Results will be"
    echo "       contaminated by that workload. Stop it first, or set DRY_RUN=1 to just preview."
  fi
fi

echo "[plan] EPISODES=$EPISODES per config; sweeping shards: $SHARD_COUNTS; native_envs=$NATIVE_ENVS; frame_source=$FRAME_SOURCE"
echo "[plan] output_root=$OUTPUT_ROOT"
[[ "$DRY_RUN" != "1" ]] && mkdir -p "$OUTPUT_ROOT"

RESULTS="$OUTPUT_ROOT/results.tsv"
[[ "$DRY_RUN" != "1" ]] && printf 'shards\tepisodes\twall_s\twall_epi_min\tsteady_epi_min\n' > "$RESULTS"

PIDS=()
trap 'echo "[abort] killing children..."; kill "${PIDS[@]}" 2>/dev/null || true; exit 130' INT TERM

run_config() {
  local shards="$1"
  (( shards > EPISODES )) && { echo "[skip] shards=$shards > EPISODES=$EPISODES"; return; }
  local cfg_dir="$OUTPUT_ROOT/s${shards}"
  echo "----------------------------------------------------------------------"
  echo "[config] shards=$shards  episodes=$EPISODES"

  # Frame-balanced contiguous boundaries so a config's wall time isn't gated by one
  # unlucky shard that happens to get the long episodes (equal COUNT != equal WORK).
  local plan
  plan="$("$PLAN_PYTHON" scripts/shard_plan.py --repo_root "$REPO_ROOT" \
            --shards "$shards" --start 0 --episodes "$EPISODES" --summary)"
  local STARTS=() COUNTS=()
  while read -r s c _; do [[ -n "$s" ]] && { STARTS+=("$s"); COUNTS+=("$c"); }; done <<<"$plan"

  if [[ "$DRY_RUN" == "1" ]]; then
    for i in "${!STARTS[@]}"; do
      echo "    shard $i: episodes [${STARTS[$i]},$((STARTS[$i]+COUNTS[$i])))  (${COUNTS[$i]} eps)"
    done
    return
  fi

  mkdir -p "$cfg_dir"
  PIDS=()
  local t0 t1
  t0="$(date +%s.%N)"
  for i in "${!STARTS[@]}"; do
    "$ISAACLAB" -p "$SCRIPT" --headless \
      --frame_source "$FRAME_SOURCE" --repo_root "$REPO_ROOT" --num_envs "$NATIVE_ENVS" \
      --dataset_episode_index "${STARTS[$i]}" --num_episodes "${COUNTS[$i]}" \
      --output_dir "$cfg_dir/shard_$(printf '%02d' "$i")" \
      > "$cfg_dir/shard_$(printf '%02d' "$i").log" 2>&1 &
    PIDS+=("$!")
  done

  local rc=0
  for p in "${PIDS[@]}"; do wait "$p" || rc=1; done
  t1="$(date +%s.%N)"
  PIDS=()

  if (( rc != 0 )); then
    echo "[config] shards=$shards FAILED (see $cfg_dir/shard_*.log)"
    printf '%s\t%s\tFAILED\tFAILED\tFAILED\n' "$shards" "$EPISODES" >> "$RESULTS"
    return
  fi

  local wall steady_span n_npz steady_rate wall_rate
  wall="$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')"
  # steady-state from .npz completion timestamps across all shards in this config
  mapfile -t MTIMES < <(find "$cfg_dir" -name '*.npz' -printf '%T@\n' 2>/dev/null | sort -n)
  n_npz="${#MTIMES[@]}"
  if (( n_npz >= 2 )); then
    steady_span="$(awk -v a="${MTIMES[0]}" -v b="${MTIMES[$((n_npz-1))]}" 'BEGIN{printf "%.3f", b-a}')"
    steady_rate="$(awk -v n="$n_npz" -v s="$steady_span" 'BEGIN{ if(s>0) printf "%.2f",(n-1)/s*60; else print "inf"}')"
  else
    steady_rate="n/a"
  fi
  wall_rate="$(awk -v n="$EPISODES" -v w="$wall" 'BEGIN{ if(w>0) printf "%.2f", n/w*60; else print "inf"}')"

  printf '%s\t%s\t%s\t%s\t%s\n' "$shards" "$EPISODES" "$wall" "$wall_rate" "$steady_rate" >> "$RESULTS"
  echo "[config] shards=$shards  wall=${wall}s  wall=${wall_rate} epi/min  steady=${steady_rate} epi/min"
}

for s in $SHARD_COUNTS; do run_config "$s"; done

if [[ "$DRY_RUN" == "1" ]]; then echo "[dry-run] nothing launched."; exit 0; fi

echo "======================================================================"
echo "Speed test summary  (EPISODES=$EPISODES per config, frame_source=$FRAME_SOURCE)"
echo "======================================================================"
# pretty table + speedup vs first (baseline) row
awk -F'\t' '
  NR==1 { printf "%-7s %-9s %-9s %-14s %-15s %-10s\n", $1,$2,"wall_s","wall_epi/min","steady_epi/min","speedup"; next }
  { rows[NR]=$0; if(base=="" && $3!="FAILED"){ base=$4 } }
  END{
    for(i=2;i<=NR;i++){
      n=split(rows[i],c,"\t");
      sp=(base!="" && c[4]!="FAILED")? sprintf("%.2fx", c[4]/base) : "-";
      printf "%-7s %-9s %-9s %-14s %-15s %-10s\n", c[1],c[2],c[3],c[4],c[5],sp;
    }
  }' "$RESULTS"
echo
echo "Raw results: $RESULTS"
echo "Interpretation: if wall_epi/min keeps rising with shards, sharding genuinely helps and"
echo "the knee is where it stops rising. If it plateaus, that's your optimal shard count."
