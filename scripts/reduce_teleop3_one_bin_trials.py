#!/usr/bin/env python3
"""Reduce teleop_3 generic one-object bin trials and keep layouts aligned."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


DEFAULT_TASKS = Path("tasks/teleop_3.jsonl")
DEFAULT_LAYOUTS = Path("tasks/layouts/teleop_3_layouts_20260720_141740.jsonl")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)


def atomic_write(path: Path, value: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def is_one_object_bin(row: dict[str, Any]) -> bool:
    return (
        row.get("n_objects") == 1
        and "target" not in row
        and "direction" not in row
        and not row.get("referents")
        and row.get("instruction") == "Place each object in the plastic bin"
    )


def evenly_spaced(values: list[int], count: int) -> list[int]:
    if len(values) < count:
        raise ValueError(f"Cannot retain {count} values from a block of {len(values)}")
    if count == 1:
        return [values[len(values) // 2]]
    positions = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    if len(set(positions)) != count:
        raise RuntimeError(f"Even selection produced duplicate positions: {positions}")
    return [values[position] for position in positions]


def ranges(values: list[int]) -> str:
    chunks = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        chunks.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    chunks.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--layouts", type=Path, default=DEFAULT_LAYOUTS)
    parser.add_argument("--per-object", type=int, default=15)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    task_path = args.tasks if args.tasks.is_absolute() else root / args.tasks
    layout_path = args.layouts if args.layouts.is_absolute() else root / args.layouts
    tasks = read_jsonl(task_path)
    layouts = read_jsonl(layout_path)
    if len(tasks) != len(layouts):
        raise ValueError(f"Task/layout counts differ: {len(tasks)} != {len(layouts)}")
    layouts_by_trial = {str(row["trial_id"]): row for row in layouts}
    if len(layouts_by_trial) != len(layouts):
        raise ValueError("Layout trial IDs are not unique")
    for row in tasks:
        layout = layouts_by_trial.get(str(row["trial_id"]))
        if layout is None:
            raise KeyError(f"Missing layout for trial_id={row['trial_id']}")
        names = [entry["name"] for entry in sorted(layout["objects"], key=lambda entry: entry["slot"])]
        if names != row["objects"] or layout["instruction"] != row["instruction"]:
            raise ValueError(f"Task/layout mismatch for trial_id={row['trial_id']}")

    by_object: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(tasks):
        if is_one_object_bin(row):
            by_object[str(row["objects"][0])].append(index)
    if not by_object:
        raise ValueError("No generic one-object bin trials found")
    if all(len(indices) == args.per_object for indices in by_object.values()):
        raise RuntimeError(f"The file already has {args.per_object} one-object bin trials per object")

    retained_one_bin = {
        index
        for indices in by_object.values()
        for index in evenly_spaced(indices, args.per_object)
    }
    deleted = [
        index for index, row in enumerate(tasks) if is_one_object_bin(row) and index not in retained_one_bin
    ]
    deleted_set = set(deleted)
    new_tasks = []
    new_layouts = []
    manifest = []
    for old_index, task in enumerate(tasks):
        if old_index in deleted_set:
            continue
        old_trial_id = task["trial_id"]
        layout = layouts_by_trial[str(old_trial_id)]
        new_index = len(new_tasks)
        task = dict(task)
        layout = dict(layout)
        task["trial_id"] = new_index
        layout["trial_id"] = new_index
        layout["episode_index"] = new_index
        new_tasks.append(task)
        new_layouts.append(layout)
        manifest.append(
            {
                "schema_version": 1,
                "new_episode_index": new_index,
                "source_episode_index": old_index,
                "source_trial_id": old_trial_id,
                "object": task["objects"][0] if is_one_object_bin(task) else None,
                "one_object_bin": is_one_object_bin(task),
            }
        )

    counts = Counter(row["objects"][0] for row in new_tasks if is_one_object_bin(row))
    if set(counts.values()) != {args.per_object}:
        raise AssertionError(f"Unexpected retained one-object counts: {counts}")
    for index, (task, layout) in enumerate(zip(new_tasks, new_layouts, strict=True)):
        names = [entry["name"] for entry in sorted(layout["objects"], key=lambda entry: entry["slot"])]
        if task["trial_id"] != index or layout["trial_id"] != index or layout["episode_index"] != index:
            raise AssertionError(f"Non-contiguous IDs at new episode {index}")
        if names != task["objects"] or layout["instruction"] != task["instruction"]:
            raise AssertionError(f"Task/layout mismatch at new episode {index}")

    print(f"One-object bin trials: {sum(len(v) for v in by_object.values())} -> {sum(counts.values())}")
    print(f"Per-object counts: {min(counts.values())}-{max(counts.values())} across {len(counts)} objects")
    print(f"Total task/layout rows: {len(tasks)} -> {len(new_tasks)}; deleting {len(deleted)}")
    print(f"Deleted 1-based editor episodes: {ranges([index + 1 for index in deleted])}")
    if not args.apply:
        print("Dry run only; pass --apply to write changes.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_backup = task_path.with_name(f"{task_path.stem}.pre_one_bin_15_{timestamp}.jsonl")
    layout_backup = layout_path.with_name(f"{layout_path.stem}.pre_one_bin_15_{timestamp}.jsonl")
    manifest_path = task_path.with_name(f"{task_path.stem}_one_bin_15_index_manifest_{timestamp}.jsonl")
    deleted_path = task_path.with_name(f"{task_path.stem}_one_bin_15_deleted_{timestamp}.jsonl")
    shutil.copy2(task_path, task_backup)
    shutil.copy2(layout_path, layout_backup)
    atomic_write(task_path, jsonl_text(new_tasks))
    atomic_write(layout_path, jsonl_text(new_layouts))
    atomic_write(manifest_path, jsonl_text(manifest))
    atomic_write(
        deleted_path,
        jsonl_text(
            [
                {
                    "schema_version": 1,
                    "source_episode_index": index,
                    "source_editor_episode_number": index + 1,
                    "source_trial_id": tasks[index]["trial_id"],
                    "object": tasks[index]["objects"][0],
                }
                for index in deleted
            ]
        ),
    )
    print(f"Updated tasks: {task_path}")
    print(f"Updated layouts: {layout_path}")
    print(f"Task backup: {task_backup}")
    print(f"Layout backup: {layout_backup}")
    print(f"Index manifest: {manifest_path}")
    print(f"Deleted-row manifest: {deleted_path}")


if __name__ == "__main__":
    main()
