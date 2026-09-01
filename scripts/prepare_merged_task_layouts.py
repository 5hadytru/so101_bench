#!/usr/bin/env python3
"""Append task/layout JSONLs and emit a merged episode provenance manifest."""

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


def _layouts_by_trial_id(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for layout in _read_jsonl(path):
        if "trial_id" not in layout:
            raise ValueError(f"{path} contains a row without trial_id")
        key = str(layout["trial_id"])
        if key in result:
            raise ValueError(f"{path} contains duplicate trial_id={layout['trial_id']!r}")
        result[key] = layout
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_tasks", type=Path, required=True)
    parser.add_argument("--base_layouts", type=Path, required=True)
    parser.add_argument("--base_manifest", type=Path, required=True)
    parser.add_argument("--append_tasks", type=Path, required=True)
    parser.add_argument("--append_layouts", type=Path, required=True)
    parser.add_argument("--append_manifest", type=Path, required=True)
    parser.add_argument("--output_tasks", type=Path, required=True)
    parser.add_argument("--output_layouts", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = (args.output_tasks, args.output_layouts, args.output_manifest)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output path(s) already exist; pass --overwrite: {existing}")

    components = (
        (
            "retained",
            _read_jsonl(args.base_tasks),
            _layouts_by_trial_id(args.base_layouts),
            _read_jsonl(args.base_manifest),
        ),
        (
            "redo",
            _read_jsonl(args.append_tasks),
            _layouts_by_trial_id(args.append_layouts),
            _read_jsonl(args.append_manifest),
        ),
    )

    output_tasks: list[dict[str, Any]] = []
    output_layouts: list[dict[str, Any]] = []
    output_manifest: list[dict[str, Any]] = []
    for source_kind, tasks, layouts, manifest in components:
        if len(tasks) != len(manifest):
            raise ValueError(f"{source_kind} tasks/manifest row counts differ: {len(tasks)} != {len(manifest)}")
        for component_index, (source_task, source_mapping) in enumerate(zip(tasks, manifest, strict=True)):
            source_trial_id = source_task.get("trial_id", component_index)
            source_layout = layouts.get(str(source_trial_id))
            if source_layout is None:
                raise KeyError(f"No {source_kind} layout found for trial_id={source_trial_id!r}")
            if source_task.get("instruction") != source_layout.get("instruction"):
                raise ValueError(f"{source_kind} task/layout instruction mismatch at row {component_index}")
            layout_objects = [
                entry.get("name") for entry in sorted(source_layout.get("objects", []), key=lambda item: item["slot"])
            ]
            if list(source_task.get("objects", [])) != layout_objects:
                raise ValueError(f"{source_kind} task/layout object mismatch at row {component_index}")

            merged_index = len(output_tasks)
            task = dict(source_task)
            task["trial_id"] = merged_index
            layout = dict(source_layout)
            layout["trial_id"] = merged_index
            layout["episode_index"] = merged_index
            output_tasks.append(task)
            output_layouts.append(layout)
            output_manifest.append(
                {
                    "schema_version": 1,
                    "merged_dataset_episode_index": merged_index,
                    "source_kind": source_kind,
                    "source_component_episode_index": component_index,
                    "source_dataset_episode_index": source_mapping["source_dataset_episode_index"],
                    "source_trial_id": source_mapping.get("source_trial_id"),
                    "instruction": source_task.get("instruction"),
                }
            )

    _write_jsonl(args.output_tasks, output_tasks)
    _write_jsonl(args.output_layouts, output_layouts)
    _write_jsonl(args.output_manifest, output_manifest)
    print(f"Prepared {len(output_tasks)} merged episode row(s).")
    print(f"  tasks:    {args.output_tasks}")
    print(f"  layouts:  {args.output_layouts}")
    print(f"  manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
