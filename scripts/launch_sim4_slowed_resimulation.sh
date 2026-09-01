#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="/home/truman/env_isaaclab_51/bin/python"
sim_version="${SO101_SIM_VERSION:-4}"
case "$sim_version" in
    4)
        episodes_jsonl="tasks/so101_bench_sim_4.jsonl"
        episode_layouts_jsonl="tasks/layouts/so101_bench_sim_4_layouts.jsonl"
        ;;
    5)
        episodes_jsonl="tasks/teleop_3.jsonl"
        episode_layouts_jsonl="tasks/layouts/teleop_3_layouts_20260720_141740.jsonl"
        ;;
    *)
        echo "Unsupported SO101_SIM_VERSION=$sim_version; expected 4 or 5." >&2
        exit 1
        ;;
esac
source_dataset="data/lerobot/so101_bench_sim_${sim_version}"
slowed_dataset="data/lerobot/so101_bench_sim_${sim_version}_slowed_1.7x_phase_governed"
outcomes_dir="outputs/so101_bench_sim_${sim_version}_slowed_1.7x_phase_governed_resimulation"
collector_pid_file="outputs/sim${sim_version}_slowed_1.7x_phase_governed_collector.pid"
collector_log="outputs/sim${sim_version}_slowed_1.7x_phase_governed_collector.log"
collector_unit="so101-bench-sim${sim_version}-slowed-1.7x-phase-governed.service"
retime_scale="1.7"
controller_transition_from_sha256="3b5b790817d7d09fa59edd0aaf8ac94be45beb19859f08380585484d49553c58"
controller_transition_to_sha256="a63c1e6c3869688329bd21108b8bd9ad742836b4f48d954ed9940eab995592af"
controller_module="source/so101_bench/so101_bench/utils/phase_retiming.py"

for required_path in \
    "$source_dataset/meta/info.json" \
    "$episodes_jsonl" \
    "$episode_layouts_jsonl"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Missing required input: $required_path" >&2
        exit 1
    fi
done

current_controller_sha256="$(sha256sum "$controller_module" | awk '{print $1}')"
if [[ "$current_controller_sha256" != "$controller_transition_to_sha256" ]]; then
    echo "The phase-governor module changed after this resume migration was pinned." >&2
    echo "  expected: $controller_transition_to_sha256" >&2
    echo "  current:  $current_controller_sha256 ($controller_module)" >&2
    echo "Review the controller change and update the pinned migration deliberately." >&2
    exit 1
fi

total_episodes="$(jq -r '.total_episodes' "$source_dataset/meta/info.json")"
if [[ ! "$total_episodes" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid total_episodes in $source_dataset/meta/info.json: $total_episodes" >&2
    exit 1
fi

echo "[INFO]: Sim${sim_version} slowed resimulation launch configuration:"
printf '  repo_root=%q\n' "$repo_root"
printf '  python_bin=%q\n' "$python_bin"
printf '  source_dataset=%q\n' "$source_dataset"
printf '  episodes_jsonl=%q\n' "$episodes_jsonl"
printf '  episode_layouts_jsonl=%q\n' "$episode_layouts_jsonl"
printf '  slowed_dataset=%q\n' "$slowed_dataset"
printf '  outcomes_dir=%q\n' "$outcomes_dir"
printf '  collector_pid_file=%q\n' "$collector_pid_file"
printf '  collector_log=%q\n' "$collector_log"
printf '  collector_unit=%q\n' "$collector_unit"
printf '  retime_scale=%q\n' "$retime_scale"
printf '  retime_recovery_integral_gain_s=%q\n' "1.50"
printf '  retime_hard_pause_entry_ratio=%q\n' "0.95"
printf '  retime_minimum_advancing_phase_rate=%q\n' "0.10"
printf '  controller_transition=%q\n' \
    "$controller_transition_from_sha256->$controller_transition_to_sha256"
printf '  total_episodes=%q\n' "$total_episodes"
printf '  task=%q\n' "So101Bench-Bin-v0"
printf '  num_envs=%q\n' "1"
printf '  retime_strategy=%q\n' "phase_governed"
printf '  retime_tracking_limit_policy=%q\n' "skip"
printf '  frame_source=%q\n' "none"
printf '  record_dataset=%q\n' "true"
printf '  dataset_streaming_encoding=%q\n' "true"
printf '  dataset_vcodec=%q\n' "libsvtav1"
printf '  dataset_encoder_threads=%q\n' "2"
printf '  dataset_encoder_queue_size=%q\n' "300"
printf '  dataset_video_files_size_mb=%q\n' "200"

if [[ -e "$collector_pid_file" ]]; then
    existing_pid="$(<"$collector_pid_file")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "A slowed collector is already running with PID $existing_pid." >&2
        exit 1
    fi
    rm -f "$collector_pid_file"
fi
if systemctl --user is-active --quiet "$collector_unit" 2>/dev/null; then
    existing_pid="$(systemctl --user show --property=MainPID --value "$collector_unit")"
    echo "A slowed collector is already running as $collector_unit (PID $existing_pid)." >&2
    exit 1
fi

dataset_exists=false
outcomes_exist=false
[[ -e "$slowed_dataset" ]] && dataset_exists=true
[[ -e "$outcomes_dir/episodes.jsonl" ]] && outcomes_exist=true
if [[ "$dataset_exists" != "$outcomes_exist" ]]; then
    echo "Cannot resume: slowed dataset and outcomes must either both exist or both be absent." >&2
    echo "  slowed dataset exists: $dataset_exists ($slowed_dataset)" >&2
    echo "  outcomes exist:     $outcomes_exist ($outcomes_dir/episodes.jsonl)" >&2
    exit 1
fi

collection_complete=false
if [[ "$dataset_exists" == true ]]; then
    recorded_episodes="$(jq -r '.total_episodes' "$slowed_dataset/meta/info.json")"
    processed_episodes="$(wc -l < "$outcomes_dir/episodes.jsonl")"
    saved_outcome_episodes="$(
        jq -s '[.[] | select(.recorded_sim_dataset != null)] | length' \
            "$outcomes_dir/episodes.jsonl"
    )"
    if [[ "$recorded_episodes" != "$saved_outcome_episodes" ]]; then
        echo "Cannot resume a torn saved-episode write: recorded dataset and saved outcome counts differ." >&2
        echo "  recorded episodes: $recorded_episodes ($slowed_dataset)" >&2
        echo "  saved outcomes:    $saved_outcome_episodes ($outcomes_dir/episodes.jsonl)" >&2
        echo "  processed outcomes:$processed_episodes (includes quarantined episodes)" >&2
        echo "Move both partial roots aside and relaunch to restart cleanly." >&2
        exit 1
    fi
    if [[ "$processed_episodes" == "$total_episodes" ]]; then
        collection_complete=true
        echo "All $total_episodes source episodes have been processed."
        echo "Saved $recorded_episodes training episode(s); quarantined $((processed_episodes - recorded_episodes))."
    else
        echo "Resuming after $processed_episodes processed source episode(s):"
        echo "  saved training episodes: $recorded_episodes"
        echo "  quarantined episodes:    $((processed_episodes - recorded_episodes))"
    fi
fi

if [[ "$collection_complete" == false ]]; then
    # This is a multi-hour job. Prefer a transient user service so it lives in a
    # separate cgroup from the invoking terminal/command runner. A plain '&' (or
    # even setsid under some runners) still receives their execution-timeout
    # SIGINT. SimulationApp turns SIGINT into SystemExit(0), making the
    # interruption look like a successful one-episode run.
    mkdir -p "$(dirname "$collector_log")"
    {
        printf '\n[INFO]: Starting detached collector at %s\n' "$(date --iso-8601=seconds)"
        printf '[INFO]: Target episodes: %s\n' "$total_episodes"
    } >> "$collector_log"
    collector_command=(
        "$python_bin" scripts/so101_lerobot_collect_outcomes.py
        --headless
        --task So101Bench-Bin-v0
        --num_envs 1
        --episodes_jsonl "$episodes_jsonl"
        --episode_layouts_jsonl "$episode_layouts_jsonl"
        --repo_id "local/so101_bench_sim_${sim_version}"
        --repo_root "$source_dataset"
        --dataset_episode_index 0
        --num_episodes "$total_episodes"
        --resume
        --resume_retime_controller_transition \
            "$controller_transition_from_sha256" \
            "$controller_transition_to_sha256"
        --ignore_sigint
        --retime_scale "$retime_scale"
        --retime_strategy phase_governed
        --retime_recovery_integral_gain_s 1.50
        --retime_hard_pause_entry_ratio 0.95
        --retime_minimum_advancing_phase_rate 0.10
        --retime_tracking_limit_policy skip
        --frame_source none
        --no_success_confirm_time
        --label_source final
        --save_trajectory false
        --output_dir "$outcomes_dir"
        --record_dataset
        --record_repo_id "local/so101_bench_sim_${sim_version}_slowed_1.7x_phase_governed"
        --record_repo_root "$slowed_dataset"
        --dataset_streaming_encoding true
        --dataset_vcodec libsvtav1
        --dataset_encoder_threads 2
        --dataset_encoder_queue_size 300
        --dataset_video_files_size_mb 200
    )

    launch_mode="nohup/setsid fallback"
    if systemctl --user is-system-running >/dev/null 2>&1; then
        systemctl --user reset-failed "$collector_unit" >/dev/null 2>&1 || true
        systemd-run \
            --user \
            --quiet \
            --collect \
            --unit "$collector_unit" \
            --property Type=exec \
            --property "StandardOutput=append:$repo_root/$collector_log" \
            --property "StandardError=append:$repo_root/$collector_log" \
            --working-directory "$repo_root" \
            --setenv "PATH=$PATH" \
            --setenv "PYTHONPATH=${PYTHONPATH:-}" \
            --setenv "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" \
            --setenv "PYTHONUNBUFFERED=1" \
            "${collector_command[@]}"
        launch_mode="systemd user service $collector_unit"
        collector_pid=""
        for _ in {1..20}; do
            collector_pid="$(systemctl --user show --property=MainPID --value "$collector_unit" 2>/dev/null || true)"
            [[ "$collector_pid" =~ ^[1-9][0-9]*$ ]] && break
            sleep 0.1
        done
        if [[ ! "$collector_pid" =~ ^[1-9][0-9]*$ ]]; then
            echo "Collector service started without a valid MainPID: $collector_unit" >&2
            systemctl --user status --no-pager "$collector_unit" >&2 || true
            exit 1
        fi
    else
        PYTHONUNBUFFERED=1 nohup setsid "${collector_command[@]}" </dev/null >> "$collector_log" 2>&1 &
        collector_pid=$!
        disown "$collector_pid" 2>/dev/null || true
    fi
    printf '%s\n' "$collector_pid" > "$collector_pid_file"
    sleep 2
    if ! kill -0 "$collector_pid" 2>/dev/null; then
        rm -f "$collector_pid_file"
        echo "Collector failed during startup ($launch_mode)." >&2
        echo "Last 40 log lines from $collector_log:" >&2
        tail -n 40 "$collector_log" >&2
        exit 1
    fi

    echo "Started detached slowed collector with PID $collector_pid via $launch_mode."
    echo "Progress log: $collector_log"
    if [[ "$sim_version" == 4 ]]; then
        echo "Graceful stop: bash scripts/stop_sim4_slowed_resimulation.sh"
    else
        echo "Graceful stop: SO101_SIM_VERSION=$sim_version bash scripts/stop_sim4_slowed_resimulation.sh"
    fi
    echo "Relaunching this script after a stop resumes from completed episodes."
fi

echo "Slowed dataset target: $slowed_dataset"
echo "Per-episode outcomes: $outcomes_dir/episodes.jsonl"
echo "Success/failure summary and episode-index lists: $outcomes_dir/summary.json"
