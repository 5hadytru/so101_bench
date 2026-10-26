#!/usr/bin/env python3
"""Remove aligned task/layout episodes by object or exact instruction."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as jsonl_file:
        return [json.loads(line) for line in jsonl_file if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as output_file:
        temporary_path = Path(output_file.name)
        for row in rows:
            output_file.write(json.dumps(row) + "\n")
    temporary_path.replace(path)


def layout_object_names(layout_row: dict[str, Any]) -> list[str]:
    return [str(object_row["name"]) for object_row in layout_row.get("objects", [])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-jsonl", type=Path, required=True)
    parser.add_argument("--episode-layouts-jsonl", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--object", dest="object_name")
    selection.add_argument("--instruction", dest="instruction")
    parser.add_argument(
        "--remove-last-half",
        action="store_true",
        help="For an exact instruction match, remove its latter half instead of every match.",
    )
    parser.add_argument("--backup-suffix", default=".bak", help="Suffix for immutable pre-filter backups.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_rows = load_jsonl(args.episodes_jsonl)
    layout_rows = load_jsonl(args.episode_layouts_jsonl)
    if len(task_rows) != len(layout_rows):
        raise ValueError(f"Task/layout row counts differ: {len(task_rows)} vs {len(layout_rows)}")

    if args.remove_last_half and not args.instruction:
        raise ValueError("--remove-last-half requires --instruction")

    matching_indices = [
        index
        for index, task in enumerate(task_rows)
        if (args.instruction is not None and task.get("instruction") == args.instruction)
        or (args.instruction is None and args.object_name in task.get("objects", []))
    ]
    if args.remove_last_half:
        if len(matching_indices) % 2:
            raise ValueError(f"Expected an even number of instruction matches, found {len(matching_indices)}")
        remove_indices = set(matching_indices[len(matching_indices) // 2 :])
    else:
        remove_indices = set(matching_indices)

    kept_tasks: list[dict[str, Any]] = []
    kept_layouts: list[dict[str, Any]] = []
    removed_trial_ids: list[int] = []
    for row_number, (task, layout) in enumerate(zip(task_rows, layout_rows), start=1):
        task_objects = [str(name) for name in task.get("objects", [])]
        layout_objects = layout_object_names(layout)
        if task_objects != layout_objects:
            raise ValueError(
                f"Row {row_number} has mismatched objects: task={task_objects}, layout={layout_objects}"
            )
        if task.get("trial_id") != layout.get("trial_id"):
            raise ValueError(f"Row {row_number} has mismatched trial IDs")
        if row_number - 1 in remove_indices:
            removed_trial_ids.append(int(task["trial_id"]))
            continue
        kept_tasks.append(task)
        kept_layouts.append(layout)

    if not removed_trial_ids:
        print(f"No rows containing {args.object_name!r}; files unchanged.")
        return

    for trial_id, (task, layout) in enumerate(zip(kept_tasks, kept_layouts)):
        task["trial_id"] = trial_id
        layout["trial_id"] = trial_id
        if "episode_index" in layout:
            layout["episode_index"] = trial_id

    for path in (args.episodes_jsonl, args.episode_layouts_jsonl):
        backup = path.with_name(path.name + args.backup_suffix)
        if backup.exists():
            raise FileExistsError(f"Refusing to overwrite existing backup: {backup}")
        shutil.copy2(path, backup)

    write_jsonl(args.episodes_jsonl, kept_tasks)
    write_jsonl(args.episode_layouts_jsonl, kept_layouts)
    print(
        f"Removed {len(removed_trial_ids)} aligned rows; "
        f"kept {len(kept_tasks)} aligned task/layout rows."
    )
    print(f"Removed original trial IDs: {removed_trial_ids}")


if __name__ == "__main__":
    main()
