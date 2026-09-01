#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-${ROOT_DIR}/outputs/groot_sim_real_correspondence/checkpoint-20000/run_20260809T062956Z}"
REDO_DECISIONS_JSONL="${REDO_DECISIONS_JSONL:-${ROOT_DIR}/logs/groot_sim_real_correspondence/real_sim_correspondence_redo_review.jsonl}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/checkpoints/so101_GR00T_N1.6-3B_WM_runpod_best_8gpu_b768/checkpoint-20000}"
ACTION_HORIZON="${ACTION_HORIZON:-8}"
PYTHON_FOR_TOOLS="${PYTHON_FOR_TOOLS:-${HOME}/env_isaaclab_51/bin/python}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/outputs/groot_sim_real_correspondence/checkpoint-20000/redos/run_${timestamp}}"
TASKS_JSONL="${RUN_DIR}/tasks.jsonl"
LAYOUTS_JSONL="${RUN_DIR}/layouts.jsonl"
REDO_MANIFEST_JSONL="${RUN_DIR}/redo_manifest.jsonl"
REDO_SUMMARY_JSON="${RUN_DIR}/redo_selection_summary.json"

for required_file in \
    "${SOURCE_RUN_DIR}/tasks.jsonl" \
    "${SOURCE_RUN_DIR}/layouts.jsonl" \
    "${REDO_DECISIONS_JSONL}" \
    "${MODEL_PATH}/config.json" \
    "${MODEL_PATH}/processor_config.json" \
    "${PYTHON_FOR_TOOLS}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "[error] Missing required file: ${required_file}" >&2
        exit 1
    fi
done

if [[ -e "${RUN_DIR}" ]]; then
    echo "[error] Refusing to reuse existing redo run directory: ${RUN_DIR}" >&2
    exit 1
fi
mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/launcher.log") 2>&1

"${PYTHON_FOR_TOOLS}" - \
    "${SOURCE_RUN_DIR}/tasks.jsonl" \
    "${SOURCE_RUN_DIR}/layouts.jsonl" \
    "${REDO_DECISIONS_JSONL}" \
    "${TASKS_JSONL}" \
    "${LAYOUTS_JSONL}" \
    "${REDO_MANIFEST_JSONL}" \
    "${REDO_SUMMARY_JSON}" <<'PY'
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def task_family(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).lower()
    if instruction.startswith("move "):
        return "move"
    if " between " in instruction:
        return "between"
    if " next to " in instruction:
        return "next_to"
    if "plastic bin" in instruction:
        return f"bin_{int(row.get('n_objects', 0))}obj"
    return "unknown"


source_tasks_path, source_layouts_path, decisions_path, tasks_path, layouts_path, manifest_path, summary_path = (
    Path(value) for value in sys.argv[1:]
)
tasks = load_jsonl(source_tasks_path)
layouts = load_jsonl(source_layouts_path)
decisions = load_jsonl(decisions_path)
if not tasks or len(tasks) != len(layouts) or len(tasks) != len(decisions):
    raise ValueError(
        "Expected equal nonzero task/layout/redo-decision counts; got "
        f"tasks={len(tasks)}, layouts={len(layouts)}, decisions={len(decisions)}"
    )

selected_indices: list[int] = []
for index, (task, layout, decision) in enumerate(zip(tasks, layouts, decisions, strict=True)):
    expected_episode = index + 1
    if decision.get("episode_index") != index or decision.get("episode") != expected_episode:
        raise ValueError(f"Redo decisions are not contiguous at source episode index {index}")
    trial_id = task.get("trial_id")
    if layout.get("trial_id") != trial_id or decision.get("trial_id") != trial_id:
        raise ValueError(f"Task/layout/redo trial mismatch at source episode index {index}")
    if not isinstance(decision.get("redo"), bool):
        raise ValueError(f"Missing Boolean redo decision at source episode index {index}")
    if decision["redo"]:
        selected_indices.append(index)

if not selected_indices:
    raise ValueError("No episodes were marked redo=True")

selected_tasks = [tasks[index] for index in selected_indices]
selected_layouts = [layouts[index] for index in selected_indices]
manifest = [
    {
        "redo_episode": redo_index + 1,
        "redo_episode_index": redo_index,
        "original_episode": source_index + 1,
        "original_episode_index": source_index,
        "trial_id": tasks[source_index].get("trial_id"),
        "instruction": tasks[source_index].get("instruction", ""),
        "ood_key": tasks[source_index].get("ood_key", "unknown"),
    }
    for redo_index, source_index in enumerate(selected_indices)
]
write_jsonl(tasks_path, selected_tasks)
write_jsonl(layouts_path, selected_layouts)
write_jsonl(manifest_path, manifest)

summary = {
    "source_run_dir": str(source_tasks_path.parent),
    "source_tasks_jsonl": str(source_tasks_path),
    "source_layouts_jsonl": str(source_layouts_path),
    "redo_decisions_jsonl": str(decisions_path),
    "source_episode_count": len(tasks),
    "redo_episode_count": len(selected_indices),
    "selected_original_episode_indices": selected_indices,
    "task_counts": dict(sorted(Counter(task_family(row) for row in selected_tasks).items())),
    "ood_counts": dict(sorted(Counter(str(row.get("ood_key", "unknown")) for row in selected_tasks).items())),
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"[redo subset] selected {len(selected_indices)}/{len(tasks)} episodes")
print(f"[redo subset] tasks: {tasks_path}")
print(f"[redo subset] layouts: {layouts_path}")
print(f"[redo subset] manifest: {manifest_path}")
PY

echo "[redo run] run_dir=${RUN_DIR}"
echo "[redo run] checkpoint=${MODEL_PATH}"
echo "[redo run] action_horizon=${ACTION_HORIZON}"

case "${PREPARE_ONLY,,}" in
    1|true|yes|on)
        echo "[redo run] Preparation-only validation complete; evaluation was not launched."
        exit 0
        ;;
esac

printf '%s\n' "$$" >"${RUN_DIR}/launcher.pid"
printf '%s\n' "${RUN_DIR}" >"${ROOT_DIR}/outputs/groot_sim_real_correspondence/checkpoint-20000/redos/latest_run.txt"

export MODEL_PATH
export CHECKPOINT_LABEL="checkpoint-20000-redos"
export ACTION_HORIZON
export RUN_DIR
export TASKS_JSONL
export LAYOUTS_JSONL
export OBJECT_POOL_EPISODES_JSONL="${SOURCE_RUN_DIR}/tasks.jsonl"
export RECORD_DATASET=1

exec bash "${ROOT_DIR}/scripts/test_groot_v7_50k_checkpoint.sh"
