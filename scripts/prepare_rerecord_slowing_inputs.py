#!/usr/bin/env python3
"""Filter skipped teleop rows out of rerecord tasks/layouts before slowing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_version", choices=("4", "5"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    stem = f"so101_bench_sim_{args.sim_version}_rerecord_round_1"
    tasks_path = Path("tasks") / f"{stem}.jsonl"
    layouts_path = Path("tasks/layouts") / f"{stem}_layouts.jsonl"
    manifest_path = Path("tasks") / f"{stem}_manifest.jsonl"
    skips_path = Path("outputs") / f"{stem}_skipped.jsonl"
    dataset_info_path = Path("data/lerobot") / stem / "meta/info.json"
    output_tasks = Path("tasks") / f"{stem}_retained.jsonl"
    output_layouts = Path("tasks/layouts") / f"{stem}_retained_layouts.jsonl"
    output_manifest = Path("tasks") / f"{stem}_retained_manifest.jsonl"

    outputs = (output_tasks, output_layouts, output_manifest)
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(f"Retained output already exists; pass --overwrite: {outputs}")

    tasks = read_jsonl(tasks_path)
    layouts = read_jsonl(layouts_path)
    manifest = read_jsonl(manifest_path)
    skips = read_jsonl(skips_path) if skips_path.exists() else []
    if not (len(tasks) == len(layouts) == len(manifest)):
        raise ValueError(
            f"Input row counts differ: tasks={len(tasks)}, layouts={len(layouts)}, manifest={len(manifest)}"
        )

    skip_indices = [int(row["task_row_index"]) for row in skips]
    if len(skip_indices) != len(set(skip_indices)) or skip_indices != sorted(skip_indices):
        raise ValueError(f"{skips_path} must contain unique, ordered task_row_index values")
    if any(index < 0 or index >= len(tasks) for index in skip_indices):
        raise ValueError(f"{skips_path} contains an out-of-range task_row_index")
    skip_set = set(skip_indices)
    retained_indices = [index for index in range(len(tasks)) if index not in skip_set]

    info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    recorded_count = int(info["total_episodes"])
    if recorded_count != len(retained_indices):
        raise ValueError(
            f"Dataset has {recorded_count} episodes, but {len(tasks)} tasks minus "
            f"{len(skip_indices)} skips gives {len(retained_indices)} retained rows"
        )

    retained_tasks = [tasks[index] for index in retained_indices]
    retained_layouts = [layouts[index] for index in retained_indices]
    retained_manifest = []
    for dataset_episode_index, task_row_index in enumerate(retained_indices):
        task = retained_tasks[dataset_episode_index]
        layout = retained_layouts[dataset_episode_index]
        row = dict(manifest[task_row_index])
        if task.get("instruction") != layout.get("instruction") or task.get("instruction") != row.get("instruction"):
            raise ValueError(f"Instruction mismatch at original rerecord task row {task_row_index}")
        row["rerecord_task_row_index"] = task_row_index
        row["rerecord_dataset_episode_index"] = dataset_episode_index
        retained_manifest.append(row)

    write_jsonl(output_tasks, retained_tasks)
    write_jsonl(output_layouts, retained_layouts)
    write_jsonl(output_manifest, retained_manifest)
    print(
        f"Prepared sim_{args.sim_version}: {len(retained_indices)} retained, "
        f"{len(skip_indices)} permanently deleted/skipped."
    )
    print(f"  tasks:    {output_tasks}")
    print(f"  layouts:  {output_layouts}")
    print(f"  manifest: {output_manifest}")


if __name__ == "__main__":
    main()
