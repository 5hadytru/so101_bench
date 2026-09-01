#!/usr/bin/env python3
"""Remove selected episodes from task/layout JSONLs and preserve an index manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def _parse_indices(value: str) -> list[int]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("--delete_episode_indices must contain at least one integer")
    try:
        indices = [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("--delete_episode_indices must be comma-separated integers") from exc
    if any(index < 0 for index in indices):
        raise ValueError(f"Episode indices cannot be negative: {indices}")
    if len(indices) != len(set(indices)):
        raise ValueError(f"Episode indices contain duplicates: {indices}")
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--layouts", type=Path, required=True)
    parser.add_argument("--delete_episode_indices", type=str, required=True)
    parser.add_argument("--output_tasks", type=Path, required=True)
    parser.add_argument("--output_layouts", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument(
        "--preserve_source_ids",
        action="store_true",
        help="Keep source trial_id/episode_index values instead of matching LeRobot's compacted indices.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = (args.output_tasks, args.output_layouts, args.output_manifest)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output path(s) already exist; pass --overwrite: {existing}")

    tasks = _read_jsonl(args.tasks)
    layouts = _read_jsonl(args.layouts)
    layouts_by_trial_id: dict[str, dict[str, Any]] = {}
    for layout in layouts:
        if "trial_id" not in layout:
            raise ValueError(f"{args.layouts} contains a row without trial_id")
        key = str(layout["trial_id"])
        if key in layouts_by_trial_id:
            raise ValueError(f"{args.layouts} contains duplicate trial_id={layout['trial_id']!r}")
        layouts_by_trial_id[key] = layout

    deleted = _parse_indices(args.delete_episode_indices)
    out_of_range = [index for index in deleted if index >= len(tasks)]
    if out_of_range:
        raise IndexError(f"Deleted task episode indices are out of range: {out_of_range}")
    deleted_set = set(deleted)

    output_tasks: list[dict[str, Any]] = []
    output_layouts: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for source_index, source_task in enumerate(tasks):
        if source_index in deleted_set:
            continue
        source_trial_id = source_task.get("trial_id", source_index)
        source_layout = layouts_by_trial_id.get(str(source_trial_id))
        if source_layout is None:
            raise KeyError(f"No layout found for task row {source_index}, trial_id={source_trial_id!r}")
        if source_layout.get("instruction") != source_task.get("instruction"):
            raise ValueError(f"Task/layout instruction mismatch for source episode {source_index}")
        layout_objects = [entry.get("name") for entry in sorted(source_layout.get("objects", []), key=lambda x: x["slot"])]
        if layout_objects != list(source_task.get("objects", [])):
            raise ValueError(f"Task/layout object mismatch for source episode {source_index}")

        output_index = len(output_tasks)
        task = dict(source_task)
        layout = dict(source_layout)
        if not args.preserve_source_ids:
            task["trial_id"] = output_index
            layout["trial_id"] = output_index
            layout["episode_index"] = output_index
        output_tasks.append(task)
        output_layouts.append(layout)
        manifest.append(
            {
                "schema_version": 1,
                "pruned_dataset_episode_index": output_index,
                "source_dataset_episode_index": source_index,
                "source_trial_id": source_trial_id,
                "pruned_trial_id": task.get("trial_id", output_index),
                "instruction": source_task.get("instruction"),
            }
        )

    _write_jsonl(args.output_tasks, output_tasks)
    _write_jsonl(args.output_layouts, output_layouts)
    _write_jsonl(args.output_manifest, manifest)
    print(f"Prepared {len(output_tasks)} retained episode(s); removed {len(deleted)}.")
    print(f"  tasks:    {args.output_tasks}")
    print(f"  layouts:  {args.output_layouts}")
    print(f"  manifest: {args.output_manifest}")
    print(f"  deleted source episode IDs: {','.join(map(str, deleted))}")


if __name__ == "__main__":
    main()
