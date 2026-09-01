#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

sim_version="${1:-}"
case "$sim_version" in
    4)
        episodes_jsonl="tasks/so101_bench_sim_4_rerecord_round_1.jsonl"
        layouts_jsonl="tasks/layouts/so101_bench_sim_4_rerecord_round_1_layouts.jsonl"
        manifest_jsonl="tasks/so101_bench_sim_4_rerecord_round_1_manifest.jsonl"
        object_pool_jsonl="tasks/so101_bench_sim_4.jsonl"
        ;;
    5)
        episodes_jsonl="tasks/so101_bench_sim_5_rerecord_round_1.jsonl"
        layouts_jsonl="tasks/layouts/so101_bench_sim_5_rerecord_round_1_layouts.jsonl"
        manifest_jsonl="tasks/so101_bench_sim_5_rerecord_round_1_manifest.jsonl"
        object_pool_jsonl="tasks/teleop_3.jsonl"
        ;;
    *)
        echo "Usage: $0 {4|5}" >&2
        exit 2
        ;;
esac

python_bin="/home/truman/env_isaaclab_51/bin/python"
hf_lerobot_home="${SO101_HF_LEROBOT_HOME:-/home/truman/workspace}"
follower_port="${SO101_FOLLOWER_PORT:-/dev/ttyACM0}"
follower_id="${SO101_FOLLOWER_ID:-follower_arm_1}"
repo_id="${SO101_RERECORD_REPO_NAMESPACE:-5hadytru}/so101_bench_sim_${sim_version}_rerecord_round_1"
dataset_root="data/lerobot/so101_bench_sim_${sim_version}_rerecord_round_1"
skipped_episodes_jsonl="outputs/so101_bench_sim_${sim_version}_rerecord_round_1_skipped.jsonl"

for required_path in \
    "$python_bin" \
    "$hf_lerobot_home/calibration/robots/so_follower/follower_arm_1.json" \
    "$episodes_jsonl" \
    "$layouts_jsonl" \
    "$manifest_jsonl" \
    "$object_pool_jsonl"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Missing required input: $required_path" >&2
        exit 1
    fi
done
if [[ ! -e "$follower_port" ]]; then
    echo "Follower serial port does not exist: $follower_port" >&2
    echo "Set SO101_FOLLOWER_PORT if the device name changed." >&2
    exit 1
fi

episode_count="$(wc -l < "$episodes_jsonl")"
layout_count="$(wc -l < "$layouts_jsonl")"
manifest_count="$(wc -l < "$manifest_jsonl")"
if [[ ! "$episode_count" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid task count: $episode_count ($episodes_jsonl)" >&2
    exit 1
fi
if [[ "$episode_count" != "$layout_count" || "$episode_count" != "$manifest_count" ]]; then
    echo "Round-1 task/layout/manifest counts differ:" >&2
    echo "  tasks=$episode_count layouts=$layout_count manifest=$manifest_count" >&2
    exit 1
fi

echo "[INFO]: Starting sim_${sim_version} round-1 re-recording"
printf '  episodes=%s\n' "$episode_count"
printf '  tasks=%q\n' "$episodes_jsonl"
printf '  layouts=%q\n' "$layouts_jsonl"
printf '  manifest=%q\n' "$manifest_jsonl"
printf '  canonical_object_pool=%q\n' "$object_pool_jsonl"
printf '  output_dataset=%q\n' "$dataset_root"
printf '  skipped_episodes=%q\n' "$skipped_episodes_jsonl"
printf '  hf_lerobot_home=%q\n' "$hf_lerobot_home"
printf '  follower_port=%q\n' "$follower_port"

export HF_LEROBOT_HOME="$hf_lerobot_home"
export HF_LEROBOT_CALIBRATION="$hf_lerobot_home/calibration"

exec "$python_bin" scripts/so101_follower_teleop.py \
    --task So101Bench-Bin-v0 \
    --follower_port "$follower_port" \
    --follower_id "$follower_id" \
    --episodes_jsonl "$episodes_jsonl" \
    --object_pool_episodes_jsonl "$object_pool_jsonl" \
    --episode_layouts_jsonl "$layouts_jsonl" \
    --num_episodes "$episode_count" \
    --repo_id "$repo_id" \
    --repo_root "$dataset_root" \
    --skipped_episodes_jsonl "$skipped_episodes_jsonl" \
    --disable_follower_torque true
