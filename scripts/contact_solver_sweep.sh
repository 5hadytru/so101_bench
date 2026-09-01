#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# Do not inherit RunPod-oriented cache paths such as /workspace/hf_cache on a
# local workstation. All Hugging Face libraries used by LeRobot/Datasets share
# this explicitly writable cache root.
user_cache_root="${XDG_CACHE_HOME:-${HOME}/.cache}"
local_hf_home="${CONTACT_SOLVER_HF_HOME:-${user_cache_root}/huggingface}"
export HF_HOME="${local_hf_home}"
export HF_MODULES_CACHE="${local_hf_home}/modules"
export HF_DATASETS_CACHE="${local_hf_home}/datasets"
export HUGGINGFACE_HUB_CACHE="${local_hf_home}/hub"
mkdir -p "${HF_MODULES_CACHE}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

dataset="data/lerobot/so101_bench_sim_6_slowed_1.7x_phase_governed"
tasks="tasks/so101_bench_sim_6_slowed_1.7x_phase_governed.jsonl"
layouts="tasks/layouts/so101_bench_sim_6_slowed_1.7x_phase_governed_layouts.jsonl"
# Sixteen representative episodes: four per task family. The bin group includes
# single- and four-object scenes, the spatial groups vary target geometry, and
# the move group covers left, right, forward, and backward once each.
episode_indices="${CONTACT_SOLVER_SWEEP_EPISODES:-80,142,371,445,128,283,229,502,91,155,596,626,334,734,708,764}"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
sweep_root="${CONTACT_SOLVER_SWEEP_ROOT:-outputs/contact_solver_position_iterations_sweep/run_${run_stamp}}"
read -r -a position_iteration_values <<< "${CONTACT_SOLVER_POSITION_ITERS:-8 16 32 64}"
episode_indices_without_commas="${episode_indices//,/}"
expected_episode_count=$((1 + ${#episode_indices} - ${#episode_indices_without_commas}))

printf '[INFO]: Sweep output root: %s\n' "${sweep_root}"

for position_iters in "${position_iteration_values[@]}"; do
    run_dir="${sweep_root}/position_iters_${position_iters}"
    output_dir="${run_dir}/outcomes"
    record_repo_root="${run_dir}/lerobot_dataset"
    if [[ -e "${run_dir}" ]]; then
        echo "[ERROR]: Refusing to append to existing sweep output: ${run_dir}" >&2
        echo "[ERROR]: Set CONTACT_SOLVER_SWEEP_ROOT to a new directory." >&2
        exit 1
    fi

    ~/IsaacLab/isaaclab.sh -p scripts/so101_lerobot_collect_outcomes.py \
        --headless \
        --task So101Bench-Mixed-v0 \
        --num_envs 1 \
        --seed 1984 \
        --contact_solver_position_iterations "${position_iters}" \
        --episodes_jsonl "${tasks}" \
        --episode_layouts_jsonl "${layouts}" \
        --repo_id local/so101_bench_sim_6_slowed_1.7x_phase_governed \
        --repo_root "${dataset}" \
        --dataset_episode_indices "${episode_indices}" \
        --benchmark_episode_indices "${episode_indices}" \
        --frame_source none \
        --initial_hold_time_s 0.5 \
        --stop_on_done \
        --label_source first_terminal \
        --save_trajectory true \
        --trajectory_stride 1 \
        --record_dataset \
        --record_repo_id "local/so101_bench_contact_solver_pos_iters_${position_iters}" \
        --record_repo_root "${record_repo_root}" \
        --dataset_streaming_encoding true \
        --dataset_vcodec libsvtav1 \
        --dataset_encoder_threads 2 \
        --dataset_encoder_queue_size 300 \
        --dataset_image_writer_processes 0 \
        --dataset_image_writer_threads_per_camera 4 \
        --dataset_video_files_size_mb 200 \
        --output_dir "${output_dir}"

    completed_episode_count="$(wc -l < "${output_dir}/episodes.jsonl")"
    if [[ "${completed_episode_count}" -ne "${expected_episode_count}" ]]; then
        echo "[ERROR]: position_iters=${position_iters} produced ${completed_episode_count}/${expected_episode_count} outcomes; refusing to rescore." >&2
        exit 1
    fi

    recorded_episode_count="$(jq -r '.total_episodes' "${record_repo_root}/meta/info.json")"
    if [[ "${recorded_episode_count}" -ne "${expected_episode_count}" ]]; then
        echo "[ERROR]: position_iters=${position_iters} saved ${recorded_episode_count}/${expected_episode_count} LeRobot episodes; refusing to rescore." >&2
        exit 1
    fi

    ~/env_isaaclab_51/bin/python scripts/so101_rescore_outcomes.py \
        --outcomes_dir "${output_dir}"
done

printf 'position_iters\tsuccesses\tfailures\tsuccess_rate\n'
for position_iters in "${position_iteration_values[@]}"; do
    jq -r --arg iters "${position_iters}" \
        '[$iters, .successes, .failures, .success_rate] | @tsv' \
        "${sweep_root}/position_iters_${position_iters}/outcomes/summary_rescored.json"
done
