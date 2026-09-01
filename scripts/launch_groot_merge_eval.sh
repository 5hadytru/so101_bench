#!/usr/bin/env bash
set -euo pipefail

# Evaluate each merged GR00T checkpoint on a deterministic, reduced subset of
# tasks/real_gr00t_WM_combined.jsonl.
#
# The default subset has one single-object grasp/bin episode for each object,
# plus one episode per non-single-object task family for each OOD key.
#
# Server-side dependencies live in NVIDIA's Isaac-GR00T repo, not this repo.
# The server command mirrors:
#   python gr00t/eval/run_gr00t_server.py \
#     --model-path <merged checkpoint> \
#     --embodiment-tag NEW_EMBODIMENT \
#     --device cuda \
#     --host 127.0.0.1 \
#     --port 5555
#
# By default this uses ~/workspace/Isaac-GR00T/.venv/bin/python from
# ~/workspace/Isaac-GR00T. Override GROOT_WORKDIR/GROOT_PYTHON if needed.

CALLER_DIR="$(pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_FOR_TOOLS="${PYTHON_FOR_TOOLS:-${HOME}/env_isaaclab_51/bin/python}"
ISAACLAB_SH="${ISAACLAB_SH:-/home/truman/IsaacLab/isaaclab.sh}"

SOURCE_JSONL="${SOURCE_JSONL:-tasks/real_gr00t_WM_combined.jsonl}"
SOURCE_LAYOUTS_JSONL="${SOURCE_LAYOUTS_JSONL:-tasks/layouts/real_gr00t_WM_combined_layouts.jsonl}"
SUBSET_SEED="${SUBSET_SEED:-0}"
SUBSET_JSONL="${SUBSET_JSONL:-tasks/real_gr00t_WM_combined_reduced_object_task_ood.jsonl}"
DEFAULT_SUBSET_LAYOUTS_JSONL="tasks/layouts/real_gr00t_WM_combined_reduced_object_task_ood_layouts.jsonl"
DEFAULT_SUBSET_SUMMARY_JSON="tasks/layouts/real_gr00t_WM_combined_reduced_object_task_ood_summary.json"
SUBSET_LAYOUTS_JSONL="${SUBSET_LAYOUTS_JSONL:-${DEFAULT_SUBSET_LAYOUTS_JSONL}}"
SUBSET_SUMMARY_JSON="${SUBSET_SUMMARY_JSON:-${DEFAULT_SUBSET_SUMMARY_JSON}}"

COEFFICIENTS=(${COEFFICIENTS:-0.6 0.7 0.8})
CHECKPOINT_TEMPLATE="${CHECKPOINT_TEMPLATE:-checkpoints/so101_GR00T_N1.6-3B_WM_v7_50k_merge_alpha_{alpha_tag}}"

DEFAULT_GROOT_WORKDIR="${HOME}/workspace/Isaac-GR00T"
DEFAULT_GROOT_PYTHON="${DEFAULT_GROOT_WORKDIR}/.venv/bin/python"
GROOT_PYTHON="${GROOT_PYTHON:-${DEFAULT_GROOT_PYTHON}}"
GROOT_REPO="${GROOT_REPO:-}"
GROOT_WORKDIR="${GROOT_WORKDIR:-${GROOT_REPO:-${DEFAULT_GROOT_WORKDIR}}}"
GROOT_SERVER_SCRIPT="${GROOT_SERVER_SCRIPT:-gr00t/eval/run_gr00t_server.py}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
SERVER_DEVICE="${SERVER_DEVICE:-cuda}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-5555}"
SERVER_STARTUP_TIMEOUT_S="${SERVER_STARTUP_TIMEOUT_S:-180}"

TASK="${TASK:-So101Bench-Bin-v0}"
SEED="${SEED:-0}"
ACTION_HORIZON="${ACTION_HORIZON:-8}"
HEADLESS="${HEADLESS:-1}"
RECORD_DATASET="${RECORD_DATASET:-0}"
DATASET_ROOT="${DATASET_ROOT:-data/lerobot/groot_merge_eval_reduced}"
LOG_DIR="${LOG_DIR:-logs/groot_merge_eval_reduced}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p \
    "$(dirname "${SUBSET_JSONL}")" \
    "$(dirname "${SUBSET_LAYOUTS_JSONL}")" \
    "$(dirname "${SUBSET_SUMMARY_JSON}")" \
    "${LOG_DIR}"

"${PYTHON_FOR_TOOLS}" - \
    "${SOURCE_JSONL}" \
    "${SOURCE_LAYOUTS_JSONL}" \
    "${SUBSET_JSONL}" \
    "${SUBSET_LAYOUTS_JSONL}" \
    "${SUBSET_SUMMARY_JSON}" \
    "${SUBSET_SEED}" <<'PY'
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any


source_jsonl = Path(sys.argv[1])
source_layouts_jsonl = Path(sys.argv[2])
subset_jsonl = Path(sys.argv[3])
subset_layouts_jsonl = Path(sys.argv[4])
summary_json = Path(sys.argv[5])
seed = int(sys.argv[6])

OOD_ORDER = ("seen", "unseen_unseen_class")
TASK_ORDER = ("between", "bin_4obj", "move", "next_to")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on {path}:{line_number}, got {type(row).__name__}")
            rows.append(row)
    return rows


def infer_task(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).lower()
    if row.get("direction") is not None or instruction.startswith("move "):
        return "move"
    if "between" in instruction:
        return "between"
    if "next to" in instruction:
        return "next_to"
    if "plastic bin" in instruction:
        n_objects = row.get("n_objects")
        if n_objects in (1, 4):
            return f"bin_{n_objects}obj"
        return "bin"
    return "unknown"


episodes = load_jsonl(source_jsonl)
layouts = load_jsonl(source_layouts_jsonl)
layouts_by_trial_id = {str(layout["trial_id"]): layout for layout in layouts if "trial_id" in layout}

buckets: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
for index, episode in enumerate(episodes):
    key = (str(episode.get("ood_key", "unknown")), infer_task(episode))
    buckets[key].append((index, episode))

rng = random.Random(seed)
selected: list[tuple[int, dict[str, Any]]] = []
single_object_counts: dict[str, int] = {}
task_ood_counts: dict[tuple[str, str], int] = {}

single_object_buckets: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
for index, episode in enumerate(episodes):
    if infer_task(episode) != "bin_1obj":
        continue
    if str(episode.get("ood_key", "unknown")) not in OOD_ORDER:
        continue
    objects = episode.get("objects", [])
    if len(objects) != 1:
        raise ValueError(f"Expected one object for bin_1obj row at source index {index}, got {objects!r}")
    single_object_buckets[str(objects[0])].append((index, episode))

for object_name in sorted(single_object_buckets):
    rows = single_object_buckets[object_name]
    sampled_index, sampled_episode = rng.choice(rows)
    selected.append((sampled_index, sampled_episode))
    single_object_counts[object_name] = 1

if not single_object_counts:
    raise ValueError(f"No single-object grasp/bin episodes found in {source_jsonl}.")

for ood_key in OOD_ORDER:
    for task in TASK_ORDER:
        key = (ood_key, task)
        rows = buckets.get(key, [])
        if not rows:
            raise ValueError(
                f"Bucket {key} has no row(s). Adjust TASK_ORDER/OOD_ORDER or the source task set."
            )
        selected.append(rng.choice(rows))
        task_ood_counts[key] = 1

selected_episodes = [episode for _, episode in selected]
selected_layouts = []
for episode in selected_episodes:
    trial_id = str(episode.get("trial_id"))
    try:
        selected_layouts.append(layouts_by_trial_id[trial_id])
    except KeyError as exc:
        raise ValueError(f"{source_layouts_jsonl} is missing a layout row for trial_id={trial_id}") from exc

subset_jsonl.write_text(
    "".join(json.dumps(episode, sort_keys=False) + "\n" for episode in selected_episodes),
    encoding="utf-8",
)
subset_layouts_jsonl.write_text(
    "".join(json.dumps(layout, sort_keys=False) + "\n" for layout in selected_layouts),
    encoding="utf-8",
)

summary = {
    "source_jsonl": str(source_jsonl),
    "source_layouts_jsonl": str(source_layouts_jsonl),
    "subset_jsonl": str(subset_jsonl),
    "subset_layouts_jsonl": str(subset_layouts_jsonl),
    "seed": seed,
    "total_episodes": len(selected_episodes),
    "single_object_grasp_episodes": [
        {"object": object_name, "count": count}
        for object_name, count in sorted(single_object_counts.items())
    ],
    "task_ood_episodes": [
        {"ood_key": ood_key, "task": task, "count": task_ood_counts[(ood_key, task)]}
        for ood_key in OOD_ORDER
        for task in TASK_ORDER
    ],
}
summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

print(f"[subset] wrote {subset_jsonl} ({len(selected_episodes)} episodes)")
print(f"[subset] wrote {subset_layouts_jsonl}")
print(f"[subset] wrote {summary_json}")
PY

alpha_tag() {
    local alpha="$1"
    alpha="${alpha//-/m}"
    alpha="${alpha//./p}"
    printf '%s' "${alpha}"
}

checkpoint_for_alpha() {
    local alpha="$1"
    local tag
    tag="$(alpha_tag "${alpha}")"
    local path="${CHECKPOINT_TEMPLATE//\{alpha_tag\}/${tag}}"
    path="${path//\{alpha\}/${alpha}}"
    printf '%s' "${path}"
}

server_script_path() {
    if [[ "${GROOT_SERVER_SCRIPT}" = /* ]]; then
        printf '%s' "${GROOT_SERVER_SCRIPT}"
    elif [[ -n "${GROOT_WORKDIR}" ]]; then
        printf '%s/%s' "${GROOT_WORKDIR}" "${GROOT_SERVER_SCRIPT}"
    elif [[ -f "${ROOT_DIR}/${GROOT_SERVER_SCRIPT}" ]]; then
        printf '%s/%s' "${ROOT_DIR}" "${GROOT_SERVER_SCRIPT}"
    elif [[ -f "${CALLER_DIR}/${GROOT_SERVER_SCRIPT}" ]]; then
        printf '%s/%s' "${CALLER_DIR}" "${GROOT_SERVER_SCRIPT}"
    else
        printf '%s' "${GROOT_SERVER_SCRIPT}"
    fi
}

validate_server_script() {
    local script_path
    script_path="$(server_script_path)"
    if [[ ! -f "${script_path}" ]]; then
        cat >&2 <<EOF
[error] Could not find ${GROOT_SERVER_SCRIPT}.
Set GROOT_WORKDIR to the directory where your GR00T command works, for example:

  GROOT_WORKDIR=/path/to/Isaac-GR00T bash scripts/launch_groot_merge_eval.sh

If you still need the base checkpoint used to create the merges:

  huggingface-cli download nvidia/GR00T-N1.6-3B \\
    --local-dir ~/so101_bench/checkpoints/GR00T-N1.6-3B
EOF
        exit 1
    fi
}

server_workdir() {
    if [[ -n "${GROOT_WORKDIR}" ]]; then
        printf '%s' "${GROOT_WORKDIR}"
    elif [[ -f "${ROOT_DIR}/${GROOT_SERVER_SCRIPT}" ]]; then
        printf '%s' "${ROOT_DIR}"
    elif [[ -f "${CALLER_DIR}/${GROOT_SERVER_SCRIPT}" ]]; then
        printf '%s' "${CALLER_DIR}"
    else
        printf '%s' "${ROOT_DIR}"
    fi
}

ping_server() {
    "${PYTHON_FOR_TOOLS}" - "${POLICY_HOST}" "${POLICY_PORT}" <<'PY' >/dev/null 2>&1
import sys

import msgpack
import zmq


host = sys.argv[1]
port = int(sys.argv[2])
context = zmq.Context.instance()
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, 2000)
socket.setsockopt(zmq.SNDTIMEO, 2000)
socket.connect(f"tcp://{host}:{port}")
try:
    socket.send(msgpack.packb({"endpoint": "ping"}, use_bin_type=True))
    socket.recv()
except Exception:
    sys.exit(1)
finally:
    socket.close(0)
PY
}

SERVER_PID=""

stop_server() {
    if [[ -n "${SERVER_PID}" ]]; then
        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            kill "${SERVER_PID}" 2>/dev/null || true
            wait "${SERVER_PID}" 2>/dev/null || true
        fi
        SERVER_PID=""
    fi
}

trap stop_server EXIT

start_server() {
    local checkpoint_dir="$1"
    local server_log="$2"
    local checkpoint_abs
    checkpoint_abs="$(realpath "${checkpoint_dir}")"

    stop_server

    if ping_server; then
        echo "[error] A GR00T server is already responding on ${POLICY_HOST}:${POLICY_PORT}." >&2
        echo "[error] Stop it first or set POLICY_PORT to a free port." >&2
        exit 1
    fi

    echo "[server] starting ${checkpoint_abs}"
    if [[ "${GROOT_SERVER_SCRIPT}" = /* ]]; then
        "${GROOT_PYTHON}" "${GROOT_SERVER_SCRIPT}" \
            --model-path "${checkpoint_abs}" \
            --embodiment-tag "${EMBODIMENT_TAG}" \
            --device "${SERVER_DEVICE}" \
            --host "${POLICY_HOST}" \
            --port "${POLICY_PORT}" >"${server_log}" 2>&1 &
    else
        (
            cd "$(server_workdir)"
            "${GROOT_PYTHON}" "${GROOT_SERVER_SCRIPT}" \
                --model-path "${checkpoint_abs}" \
                --embodiment-tag "${EMBODIMENT_TAG}" \
                --device "${SERVER_DEVICE}" \
                --host "${POLICY_HOST}" \
                --port "${POLICY_PORT}"
        ) >"${server_log}" 2>&1 &
    fi
    SERVER_PID="$!"
}

wait_for_server() {
    local server_log="$1"
    local deadline=$((SECONDS + SERVER_STARTUP_TIMEOUT_S))
    until ping_server; do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[error] GR00T server exited before becoming ready. Last server log lines:" >&2
            tail -n 80 "${server_log}" >&2 || true
            exit 1
        fi
        if (( SECONDS >= deadline )); then
            echo "[error] Timed out waiting for GR00T server on ${POLICY_HOST}:${POLICY_PORT}." >&2
            echo "[error] Last server log lines:" >&2
            tail -n 80 "${server_log}" >&2 || true
            exit 1
        fi
        sleep 2
    done
    echo "[server] ready on ${POLICY_HOST}:${POLICY_PORT}"
}

run_eval() {
    local alpha="$1"
    local checkpoint_dir="$2"
    local tag
    tag="$(alpha_tag "${alpha}")"

    local timestamp
    timestamp="$(date +%Y%m%d_%H%M%S)"
    local server_log="${LOG_DIR}/groot_server_merge_alpha_${tag}_${timestamp}.log"
    local eval_log="${LOG_DIR}/groot_eval_merge_alpha_${tag}_${timestamp}.log"
    local rates_json="${LOG_DIR}/groot_eval_merge_alpha_${tag}_${timestamp}_rates.json"

    start_server "${checkpoint_dir}" "${server_log}"
    wait_for_server "${server_log}"

    local eval_args=(
        -p scripts/groot_eval.py
        --task "${TASK}"
        --episodes_jsonl "${SUBSET_JSONL}"
        --episode_layouts_jsonl "${SUBSET_LAYOUTS_JSONL}"
        --seed "${SEED}"
        --policy_host "${POLICY_HOST}"
        --policy_port "${POLICY_PORT}"
        --action_horizon "${ACTION_HORIZON}"
        --use_overhead_init true
    )
    if [[ "${HEADLESS}" == "1" ]]; then
        eval_args+=(--headless)
    fi
    if [[ "${RECORD_DATASET}" == "1" ]]; then
        eval_args+=(--record_dataset --repo_root "${DATASET_ROOT}/alpha_${tag}")
    fi

    echo "[eval] alpha=${alpha} checkpoint=${checkpoint_dir}"
    "${ISAACLAB_SH}" "${eval_args[@]}" 2>&1 | tee "${eval_log}"

    if [[ -f scripts/compute_groot_success_rates.py ]]; then
        "${PYTHON_FOR_TOOLS}" scripts/compute_groot_success_rates.py \
            "${eval_log}" \
            --episodes-jsonl "${SUBSET_JSONL}" \
            --format json >"${rates_json}"
        echo "[eval] wrote ${rates_json}"
    fi

    stop_server
}

echo "[config] subset=${SUBSET_JSONL}"
echo "[config] layouts=${SUBSET_LAYOUTS_JSONL}"
echo "[config] coefficients=${COEFFICIENTS[*]}"
echo "[config] log_dir=${LOG_DIR}"

for alpha in "${COEFFICIENTS[@]}"; do
    checkpoint_dir="$(checkpoint_for_alpha "${alpha}")"
    if [[ ! -d "${checkpoint_dir}" ]]; then
        echo "[error] Missing merged checkpoint for alpha=${alpha}: ${checkpoint_dir}" >&2
        echo "[error] Generate it with: ${PYTHON_FOR_TOOLS} scripts/merge_gr00t.py --coefficients ${alpha}" >&2
        exit 1
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry-run] would evaluate alpha=${alpha}: ${checkpoint_dir}"
        continue
    fi
    validate_server_script
    run_eval "${alpha}" "${checkpoint_dir}"
done

echo "[done] completed GR00T merge coefficient evaluation"
