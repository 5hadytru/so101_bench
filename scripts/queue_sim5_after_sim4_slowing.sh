#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

sim4_source="data/lerobot/so101_bench_sim_4/meta/info.json"
sim4_summary="outputs/so101_bench_sim_4_slowed_1.7x_phase_governed_resimulation/summary.json"
sim4_collector_unit="so101-bench-sim4-slowed-1.7x-phase-governed.service"
sim5_collector_unit="so101-bench-sim5-slowed-1.7x-phase-governed.service"
sim5_source="data/lerobot/so101_bench_sim_5/meta/info.json"
sim5_episodes="tasks/teleop_3.jsonl"
sim5_layouts="tasks/layouts/teleop_3_layouts_20260720_141740.jsonl"
poll_seconds=60

for required_path in \
    "$sim4_source" \
    "$sim5_source" \
    "$sim5_episodes" \
    "$sim5_layouts" \
    "scripts/launch_sim4_slowed_resimulation.sh"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Missing required queue input: $required_path" >&2
        exit 1
    fi
done

sim4_total="$(jq -r '.total_episodes' "$sim4_source")"
sim5_total="$(jq -r '.total_episodes' "$sim5_source")"
sim5_task_rows="$(wc -l < "$sim5_episodes")"
sim5_layout_rows="$(wc -l < "$sim5_layouts")"
if [[ "$sim5_task_rows" != "$sim5_total" || "$sim5_layout_rows" != "$sim5_total" ]]; then
    echo "Sim5 queue inputs do not match the source dataset episode count:" >&2
    echo "  dataset episodes: $sim5_total" >&2
    echo "  task rows:        $sim5_task_rows ($sim5_episodes)" >&2
    echo "  layout rows:      $sim5_layout_rows ($sim5_layouts)" >&2
    exit 1
fi

if systemctl --user is-active --quiet "$sim5_collector_unit" 2>/dev/null; then
    echo "Sim5 slowing is already running as $sim5_collector_unit." >&2
    exit 1
fi

echo "[QUEUE]: Sim5 slowing is queued behind complete sim4 processing."
echo "[QUEUE]: Sim4 target episodes: $sim4_total"
echo "[QUEUE]: Sim5 target episodes: $sim5_total"
echo "[QUEUE]: Poll interval: ${poll_seconds}s"

last_status=""
while true; do
    completed=0
    complete=false
    if [[ -f "$sim4_summary" ]]; then
        completed="$(jq -r '.completed_episodes // 0' "$sim4_summary")"
        complete="$(jq -r '.collection_complete // false' "$sim4_summary")"
    fi

    if [[ "$complete" == true && "$completed" == "$sim4_total" ]]; then
        echo "[QUEUE]: Sim4 completed all $sim4_total source episodes at $(date --iso-8601=seconds)."
        echo "[QUEUE]: Launching sim5 with the same slowing/controller/recording settings."
        SO101_SIM_VERSION=5 bash scripts/launch_sim4_slowed_resimulation.sh
        echo "[QUEUE]: Sim5 launch command completed successfully."
        exit 0
    fi

    if systemctl --user is-active --quiet "$sim4_collector_unit" 2>/dev/null; then
        current_status="sim4 running; completed summary=$completed/$sim4_total"
    else
        current_status="sim4 not running and not complete; waiting for resume ($completed/$sim4_total)"
    fi
    if [[ "$current_status" != "$last_status" ]]; then
        echo "[QUEUE]: $current_status"
        last_status="$current_status"
    fi
    sleep "$poll_seconds"
done
