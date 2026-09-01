#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

sim_version="${1:-}"
case "$sim_version" in
    4|5) ;;
    *) echo "Usage: $0 {4|5}" >&2; exit 2 ;;
esac

stem="so101_bench_sim_${sim_version}_rerecord_round_1"
source_dataset="data/lerobot/${stem}"
tasks_jsonl="tasks/${stem}_retained.jsonl"
layouts_jsonl="tasks/layouts/${stem}_retained_layouts.jsonl"
manifest_jsonl="tasks/${stem}_retained_manifest.jsonl"
slowed_dataset="data/lerobot/${stem}_slowed_1.7x_phase_governed"
outcomes_dir="outputs/${stem}_slowed_1.7x_phase_governed_resimulation"
log_path="outputs/${stem}_slowing.log"

/home/truman/env_isaaclab_51/bin/python scripts/prepare_rerecord_slowing_inputs.py \
    "$sim_version" --overwrite

total_episodes="$(jq -r '.total_episodes' "$source_dataset/meta/info.json")"
for path in "$tasks_jsonl" "$layouts_jsonl" "$manifest_jsonl"; do
    [[ "$(wc -l < "$path")" == "$total_episodes" ]] || {
        echo "Row count mismatch: $path versus $total_episodes source episodes" >&2
        exit 1
    }
done

dataset_exists=false
outcomes_exist=false
[[ -e "$slowed_dataset" ]] && dataset_exists=true
[[ -e "$outcomes_dir/episodes.jsonl" ]] && outcomes_exist=true
if [[ "$dataset_exists" != "$outcomes_exist" ]]; then
    echo "Cannot resume: slowed dataset and outcomes must both exist or both be absent." >&2
    echo "  slowed dataset: $dataset_exists ($slowed_dataset)" >&2
    echo "  outcomes:       $outcomes_exist ($outcomes_dir/episodes.jsonl)" >&2
    exit 1
fi
if [[ "$dataset_exists" == true ]]; then
    recorded="$(jq -r '.total_episodes' "$slowed_dataset/meta/info.json")"
    processed="$(wc -l < "$outcomes_dir/episodes.jsonl")"
    saved_outcomes="$(jq -s '[.[] | select(.recorded_sim_dataset != null)] | length' "$outcomes_dir/episodes.jsonl")"
    if [[ "$recorded" != "$saved_outcomes" ]]; then
        echo "Cannot resume a torn write: dataset=$recorded but saved outcomes=$saved_outcomes." >&2
        exit 1
    fi
    if (( processed >= total_episodes )); then
        echo "[INFO]: All $total_episodes source episodes are already processed."
        echo "[INFO]: Saved $recorded slowed episode(s); quarantined $((processed - recorded))."
        exit 0
    fi
    echo "[INFO]: Resuming after $processed/$total_episodes processed episode(s)."
fi

echo "[INFO]: Slowing ${total_episodes} retained sim_${sim_version} rerecord episode(s)."
echo "[INFO]: Log: $log_path"

exec /home/truman/IsaacLab/isaaclab.sh -p scripts/so101_lerobot_collect_outcomes.py \
    --headless \
    --task So101Bench-Bin-v0 \
    --num_envs 1 \
    --episodes_jsonl "$tasks_jsonl" \
    --episode_layouts_jsonl "$layouts_jsonl" \
    --repo_id "local/${stem}" \
    --repo_root "$source_dataset" \
    --dataset_episode_index 0 \
    --num_episodes "$total_episodes" \
    --resume \
    --retime_scale 1.7 \
    --retime_strategy phase_governed \
    --retime_recovery_integral_gain_s 1.50 \
    --retime_hard_pause_entry_ratio 0.95 \
    --retime_minimum_advancing_phase_rate 0.10 \
    --retime_tracking_limit_policy skip \
    --frame_source none \
    --no_success_confirm_time \
    --label_source final \
    --save_trajectory false \
    --output_dir "$outcomes_dir" \
    --record_dataset \
    --record_repo_id "local/${stem}_slowed_1.7x_phase_governed" \
    --record_repo_root "$slowed_dataset" \
    --dataset_streaming_encoding true \
    --dataset_vcodec libsvtav1 \
    --dataset_encoder_threads 2 \
    --dataset_encoder_queue_size 300 \
    --dataset_video_files_size_mb 200 \
    2>&1 | tee "$log_path"
