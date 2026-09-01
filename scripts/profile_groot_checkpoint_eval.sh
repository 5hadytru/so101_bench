#!/usr/bin/env bash
set -euo pipefail

# Run a short, representative evaluation profile. The underlying evaluator
# records its normal workload; layout loading/generation is intentionally not
# timed. Override RECORD_DATASET=0 for a matched recording-off comparison.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "[error] Set MODEL_PATH to the local GR00T checkpoint directory to profile." >&2
    exit 1
fi

export PROFILE_EVAL=1
export PROFILE_CUDA_SYNC="${PROFILE_CUDA_SYNC:-0}"
export NUM_EPISODES="${NUM_EPISODES:-5}"
export RECORD_DATASET="${RECORD_DATASET:-1}"
export CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-$(basename "${MODEL_PATH}")}"
export TASKS_JSONL="${TASKS_JSONL:-${ROOT_DIR}/tasks/real_gr00t_WM_all_filtered.jsonl}"
DEFAULT_LAYOUTS_JSONL="${ROOT_DIR}/tasks/layouts/real_gr00t_WM_all_filtered_layouts_20260717_215034.jsonl"
export LAYOUTS_JSONL="${LAYOUTS_JSONL:-${DEFAULT_LAYOUTS_JSONL}}"

if [[ -z "${RUN_DIR:-}" ]]; then
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    recording_label="recording_on"
    case "${RECORD_DATASET,,}" in
        0|false|no|off) recording_label="recording_off" ;;
    esac
    export RUN_DIR="${ROOT_DIR}/outputs/groot_eval_profiles/${CHECKPOINT_LABEL}/${recording_label}_${timestamp}"
fi

exec bash "${ROOT_DIR}/scripts/test_groot_v7_50k_checkpoint.sh"
