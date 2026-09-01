#!/usr/bin/env bash
set -euo pipefail

# Evaluate the published v7 50k run's checkpoint with the same inference and
# validation settings used by monitor_runpod_groot_checkpoints.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/checkpoints/so101_GR00T_N1.6-3B_WM_v7_50k/checkpoint-52000}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-$(basename "${MODEL_PATH}")}"
ACTION_HORIZON="${ACTION_HORIZON:-8}"
NUM_EPISODES="${NUM_EPISODES:-}"
PROFILE_EVAL="${PROFILE_EVAL:-0}"
PROFILE_CUDA_SYNC="${PROFILE_CUDA_SYNC:-0}"
RECORD_DATASET="${RECORD_DATASET:-1}"
TASKS_JSONL="${TASKS_JSONL:-${ROOT_DIR}/tasks/real_gr00t_val.jsonl}"
LAYOUTS_JSONL="${LAYOUTS_JSONL:-${ROOT_DIR}/tasks/layouts/real_gr00t_val_layouts_20260718_155446.jsonl}"
OBJECT_POOL_EPISODES_JSONL="${OBJECT_POOL_EPISODES_JSONL:-}"

ISAACLAB_SH="${ISAACLAB_SH:-/home/truman/IsaacLab/isaaclab.sh}"
ISAACLAB_VENV="${ISAACLAB_VENV:-${HOME}/env_isaaclab_51}"
GROOT_WORKDIR="${GROOT_WORKDIR:-${HOME}/workspace/Isaac-GR00T}"
GROOT_PYTHON="${GROOT_PYTHON:-${GROOT_WORKDIR}/.venv/bin/python}"
SERVER_HF_HOME="${SERVER_HF_HOME:-${HOME}/.cache/huggingface}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-5555}"
SERVER_STARTUP_TIMEOUT_S="${SERVER_STARTUP_TIMEOUT_S:-180}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/outputs/groot_v7_50k_validation/run_${timestamp}}"
SERVER_LOG="${RUN_DIR}/server.log"
EVAL_LOG="${RUN_DIR}/eval.log"
SERVER_PID=""
PROFILE_ENABLED=0
case "${PROFILE_EVAL,,}" in
    1|true|yes|on) PROFILE_ENABLED=1 ;;
esac

for required_file in \
    "${MODEL_PATH}/config.json" \
    "${MODEL_PATH}/processor_config.json" \
    "${TASKS_JSONL}" \
    "${LAYOUTS_JSONL}" \
    "${ISAACLAB_SH}" \
    "${GROOT_PYTHON}" \
    "${ROOT_DIR}/scripts/select_groot_profile_tasks.py"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "[error] Missing required file: ${required_file}" >&2
        exit 1
    fi
done
if [[ -n "${OBJECT_POOL_EPISODES_JSONL}" && ! -f "${OBJECT_POOL_EPISODES_JSONL}" ]]; then
    echo "[error] Missing object-pool episode file: ${OBJECT_POOL_EPISODES_JSONL}" >&2
    exit 1
fi

mkdir -p "${RUN_DIR}" "${SERVER_HF_HOME}"

episode_args=()
object_pool_args=()
if [[ -n "${OBJECT_POOL_EPISODES_JSONL}" ]]; then
    object_pool_args=(--object_pool_episodes_jsonl "${OBJECT_POOL_EPISODES_JSONL}")
fi
EVAL_TASKS_JSONL="${TASKS_JSONL}"
if [[ -n "${NUM_EPISODES}" ]]; then
    if [[ ! "${NUM_EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[error] NUM_EPISODES must be a positive integer; got ${NUM_EPISODES}." >&2
        exit 1
    fi
    episode_args=(--num_episodes "${NUM_EPISODES}")
    task_row_count="$(awk 'NF { count += 1 } END { print count + 0 }' "${TASKS_JSONL}")"
    if ((NUM_EPISODES > task_row_count)); then
        echo "[error] ${TASKS_JSONL} has fewer than ${NUM_EPISODES} non-empty task rows." >&2
        exit 1
    fi
    if ((PROFILE_ENABLED)); then
        if [[ -z "${OBJECT_POOL_EPISODES_JSONL}" ]]; then
            object_pool_args=(--object_pool_episodes_jsonl "${TASKS_JSONL}")
        fi
        EVAL_TASKS_JSONL="${RUN_DIR}/profile_tasks_stratified_${NUM_EPISODES}.jsonl"
        "${ISAACLAB_VENV}/bin/python" scripts/select_groot_profile_tasks.py \
            "${TASKS_JSONL}" "${EVAL_TASKS_JSONL}" --count "${NUM_EPISODES}"
    else
        EVAL_TASKS_JSONL="${RUN_DIR}/tasks_first_${NUM_EPISODES}.jsonl"
        awk -v count="${NUM_EPISODES}" 'NF { print; seen += 1; if (seen == count) exit }' \
            "${TASKS_JSONL}" >"${EVAL_TASKS_JSONL}"
    fi
fi

profile_args=()
if ((PROFILE_ENABLED)); then
    profile_args=(
        --profile_output_json "${RUN_DIR}/eval_profile.json"
        --profile_episodes_jsonl "${RUN_DIR}/eval_profile_episodes.jsonl"
    )
    case "${PROFILE_CUDA_SYNC,,}" in
        1|true|yes|on) profile_args+=(--profile_cuda_sync) ;;
    esac
fi

recording_args=()
case "${RECORD_DATASET,,}" in
    1|true|yes|on)
        recording_args=(
            --record_dataset
            --repo_id "5hadytru/so101_bench_groot_eval_${CHECKPOINT_LABEL}_ah${ACTION_HORIZON}"
            --repo_root "${RUN_DIR}/lerobot_dataset"
        )
        ;;
esac

ping_server() {
    "${ISAACLAB_VENV}/bin/python" - "${POLICY_HOST}" "${POLICY_PORT}" <<'PY'
import sys
import msgpack
import zmq

host, port = sys.argv[1], int(sys.argv[2])
context = zmq.Context.instance()
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, 2000)
socket.setsockopt(zmq.SNDTIMEO, 2000)
socket.connect(f"tcp://{host}:{port}")
try:
    socket.send(msgpack.packb({"endpoint": "ping"}, use_bin_type=True))
    socket.recv()
except Exception:
    raise SystemExit(1)
finally:
    socket.close(0)
PY
}

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ping_server; then
    echo "[error] A GR00T server is already responding on ${POLICY_HOST}:${POLICY_PORT}." >&2
    exit 1
fi

echo "[server] Starting ${MODEL_PATH} on ${POLICY_HOST}:${POLICY_PORT}"
(
    cd "${GROOT_WORKDIR}"
    exec env \
        HF_HOME="${SERVER_HF_HOME}" \
        HF_MODULES_CACHE="${SERVER_HF_HOME}/modules" \
        "${GROOT_PYTHON}" gr00t/eval/run_gr00t_server.py \
        --model-path "${MODEL_PATH}" \
        --embodiment-tag NEW_EMBODIMENT \
        --device cuda \
        --host "${POLICY_HOST}" \
        --port "${POLICY_PORT}"
) >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"

deadline=$((SECONDS + SERVER_STARTUP_TIMEOUT_S))
until ping_server; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[error] GR00T server exited before becoming ready." >&2
        tail -n 100 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "[error] Timed out waiting for the GR00T server." >&2
        tail -n 100 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    sleep 2
done

echo "[eval] Writing results to ${RUN_DIR}"
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME \
    VIRTUAL_ENV="${ISAACLAB_VENV}" \
    PATH="${ISAACLAB_VENV}/bin:${PATH}" \
    "${ISAACLAB_SH}" -p scripts/groot_eval.py \
    --task So101Bench-Bin-v0 \
    --episodes_jsonl "${EVAL_TASKS_JSONL}" \
    --episode_layouts_jsonl "${LAYOUTS_JSONL}" \
    "${object_pool_args[@]}" \
    "${episode_args[@]}" \
    --seed 0 \
    --policy_host "${POLICY_HOST}" \
    --policy_port "${POLICY_PORT}" \
    --action_horizon "${ACTION_HORIZON}" \
    --use_overhead_init true \
    "${recording_args[@]}" \
    --episode_diagnostics_jsonl "${RUN_DIR}/episode_diagnostics.jsonl" \
    "${profile_args[@]}" \
    --camera_snapshot_key '' \
    --episode_skip_key '' \
    --camera_snapshot_stdin false \
    --terminal_control_stdin false \
    --headless 2>&1 | tee "${EVAL_LOG}"

stop_server

"${ISAACLAB_VENV}/bin/python" scripts/compute_groot_success_rates.py \
    "${EVAL_LOG}" --episodes-jsonl "${EVAL_TASKS_JSONL}" --format text >"${RUN_DIR}/success_rates.txt"
"${ISAACLAB_VENV}/bin/python" scripts/compute_groot_success_rates.py \
    "${EVAL_LOG}" --episodes-jsonl "${EVAL_TASKS_JSONL}" --format json >"${RUN_DIR}/success_rates.json"
"${ISAACLAB_VENV}/bin/python" scripts/write_groot_episode_results.py \
    "${EVAL_LOG}" \
    --episodes-jsonl "${EVAL_TASKS_JSONL}" \
    --output-json "${RUN_DIR}/episode_results.json" \
    --output-tsv "${RUN_DIR}/episode_results.tsv" \
    --diagnostics-jsonl "${RUN_DIR}/episode_diagnostics.jsonl" \
    --checkpoint "${CHECKPOINT_LABEL}"

if [[ -f "${RUN_DIR}/eval_profile.json" ]]; then
    "${ISAACLAB_VENV}/bin/python" scripts/report_groot_eval_profile.py \
        "${RUN_DIR}/eval_profile.json" >"${RUN_DIR}/eval_profile.txt"
fi

{
    printf 'Checkpoint: %s\n' "${CHECKPOINT_LABEL}"
    printf 'Model path: %s\n' "${MODEL_PATH}"
    printf 'Action horizon: %s\n\n' "${ACTION_HORIZON}"
    cat "${RUN_DIR}/success_rates.txt"
    printf '\nDetailed per-episode diagnostics: %s\n' "${RUN_DIR}/episode_diagnostics.jsonl"
    if [[ -f "${RUN_DIR}/eval_profile.txt" ]]; then
        printf 'Evaluation performance profile: %s\n' "${RUN_DIR}/eval_profile.txt"
    fi
} >"${RUN_DIR}/summary.txt"

echo "[done] Evaluation summary: ${RUN_DIR}/summary.txt"
