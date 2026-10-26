#!/usr/bin/env python3
"""Compact aligned SO-101 task/layout JSONL trial IDs to their row order."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PAIRS = (
    (
        ROOT / "tasks" / "teleop_3.jsonl",
        ROOT / "tasks" / "layouts" / "teleop_3_layouts_20260720_141740.jsonl",
    ),
    (
        ROOT / "tasks" / "real_gr00t_WM_all_filtered.jsonl",
        ROOT / "tasks" / "layouts" / "real_gr00t_WM_all_filtered_layouts_20260717_215034.jsonl",
    ),
    (
        ROOT / "tasks" / "real_gr00t_val.jsonl",
        ROOT / "tasks" / "layouts" / "real_gr00t_val_layouts_20260718_155446.jsonl",
    ),
)
BACKUP_SUFFIX = ".before_contiguous_trial_ids_20260725"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def validate_pair(tasks: list[dict], layouts: list[dict], task_path: Path) -> None:
    if len(tasks) != len(layouts):
        raise ValueError(f"{task_path}: task/layout counts differ")
    for row_index, (task, layout) in enumerate(zip(tasks, layouts, strict=True)):
        task_names = task.get("objects")
        layout_names = [entry.get("name") for entry in sorted(layout.get("objects", []), key=lambda entry: entry["slot"])]
        if task.get("instruction") != layout.get("instruction") or task_names != layout_names:
            raise ValueError(f"{task_path}: task/layout content mismatch at row {row_index}")


def main() -> None:
    staged: list[tuple[Path, Path, list[dict], list[dict]]] = []
    for task_path, layout_path in PAIRS:
        tasks, layouts = read_jsonl(task_path), read_jsonl(layout_path)
        validate_pair(tasks, layouts, task_path)
        for trial_id, (task, layout) in enumerate(zip(tasks, layouts, strict=True)):
            task["trial_id"] = trial_id
            layout["trial_id"] = trial_id
            if "episode_index" in layout:
                layout["episode_index"] = trial_id
        validate_pair(tasks, layouts, task_path)
        staged.append((task_path, layout_path, tasks, layouts))

    paths = [path for task_path, layout_path in PAIRS for path in (task_path, layout_path)]
    for path in paths:
        backup = path.with_name(path.name + BACKUP_SUFFIX)
        if backup.exists():
            raise FileExistsError(f"Refusing to overwrite existing backup: {backup}")
    for path in paths:
        shutil.copy2(path, path.with_name(path.name + BACKUP_SUFFIX))
    for task_path, layout_path, tasks, layouts in staged:
        write_jsonl(task_path, tasks)
        write_jsonl(layout_path, layouts)
        print(f"Compacted {len(tasks)} rows: {task_path} and {layout_path}")


if __name__ == "__main__":
    main()
