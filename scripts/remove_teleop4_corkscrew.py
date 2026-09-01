#!/usr/bin/env python3
"""Remove tan-corkscrew bin targets and replace every retained appearance.

The task and layout files are kept aligned.  Layouts for unaffected rows are
preserved (including hand edits); rows receiving a replacement object are
resampled because object footprints differ.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import shutil
from typing import Any

from replace_removed_bin_objects import (
    _env_cfg_literals,
    _load_pure_python_modules,
    layout_object_names,
    read_jsonl,
    write_jsonl_atomic,
)


RETIRED = "tan corkscrew"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-jsonl", type=Path, default=Path("tasks/teleop_4.jsonl"))
    parser.add_argument(
        "--episode-layouts-jsonl",
        type=Path,
        default=Path("tasks/layouts/teleop_4_layouts_20260831_201454.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def colorless(name: str, colors: set[str]) -> str:
    words = name.split()
    return " ".join(words[1:]) if words and words[0] in colors else name


def family(row: dict[str, Any]) -> str:
    if row.get("task_family") == "named_bin":
        return "named_bin"
    if "direction" in row:
        return "move"
    if len(row.get("referents", [])) == 1:
        return "next_to"
    if len(row.get("referents", [])) == 2:
        return "between"
    return "bin"


def should_drop(row: dict[str, Any]) -> bool:
    return row["objects"] == [RETIRED] or (
        row.get("task_family") == "named_bin" and row.get("target") == RETIRED
    )


def label(name: str, *, prefer_full: bool, objects: list[str], colors: set[str]) -> str:
    short = colorless(name, colors)
    ambiguous = any(other != name and colorless(other, colors) == short for other in objects)
    return name if prefer_full or ambiguous else short


def rewritten_instruction(
    old: dict[str, Any], updated: dict[str, Any], replacement: str, colors: set[str]
) -> str:
    task_family = family(updated)
    if task_family == "bin":
        return old["instruction"]

    old_named = [old["target"], *old.get("referents", [])]
    new_named = [updated["target"], *updated.get("referents", [])]
    labels: list[str] = []
    for old_name, new_name in zip(old_named, new_named, strict=True):
        prefer_full = old_name in old["instruction"]
        labels.append(
            label(new_name, prefer_full=prefer_full, objects=updated["objects"], colors=colors)
        )

    if task_family == "named_bin":
        return f"Place the {labels[0]} in the plastic bin"
    if task_family == "next_to":
        return f"Place the {labels[0]} next to the {labels[1]}"
    if task_family == "between":
        return f"Place the {labels[0]} between the {labels[1]} and the {labels[2]}"
    direction = {
        "forward": "forwards",
        "backward": "backwards",
    }.get(updated["direction"], updated["direction"])
    return f"Move the {labels[0]} {direction}"


def replace_row(
    row: dict[str, Any], replacement: str, benchmark: Any, colors: set[str], source: str
) -> dict[str, Any]:
    updated = deepcopy(row)
    updated["objects"] = [replacement if name == RETIRED else name for name in row["objects"]]
    if updated.get("anchor") == RETIRED:
        updated["anchor"] = replacement
    if updated.get("target") == RETIRED:
        updated["target"] = replacement
    if "referents" in updated:
        updated["referents"] = [
            replacement if name == RETIRED else name for name in updated["referents"]
        ]
    updated["instruction"] = rewritten_instruction(row, updated, replacement, colors)
    benchmark.episode_spec_from_json(updated, source=source)
    canonical = benchmark.row_with_canonical_instruction_metadata(updated, source=source)
    for key in ("target", "referents", "direction"):
        if canonical.get(key) != updated.get(key):
            raise ValueError(
                f"{source}: canonical {key} {canonical.get(key)!r} != {updated.get(key)!r}"
            )
    return updated


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    tasks_path = args.episodes_jsonl if args.episodes_jsonl.is_absolute() else root / args.episodes_jsonl
    layouts_path = (
        args.episode_layouts_jsonl
        if args.episode_layouts_jsonl.is_absolute()
        else root / args.episode_layouts_jsonl
    )
    benchmark, layouts_module = _load_pure_python_modules()
    constants = _env_cfg_literals()
    colors = set(benchmark.COLORS)
    tasks = read_jsonl(tasks_path)
    layouts = read_jsonl(layouts_path)
    if len(tasks) != len(layouts):
        raise ValueError(f"Task/layout row counts differ: {len(tasks)} vs {len(layouts)}")
    for index, (task, layout) in enumerate(zip(tasks, layouts, strict=True)):
        if task["trial_id"] != layout["trial_id"] or task["objects"] != layout_object_names(layout):
            raise ValueError(f"Task/layout mismatch at row {index}")

    pool = [name for name in benchmark.OBJECT_SPLITS["seen"] if name != RETIRED]
    rng = random.Random(args.seed)
    layout_rng = random.Random(args.seed + 1)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bin_random_poses = tuple(
        (translation, tuple(math.radians(angle) for angle in rpy_deg))
        for translation, rpy_deg in constants["BIN_RANDOM_POSES_RPY_DEG"]
    )
    replacement_counts: Counter[str] = Counter()
    new_tasks: list[dict[str, Any]] = []
    new_layouts: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for old_index, (task, old_layout) in enumerate(zip(tasks, layouts, strict=True)):
        if should_drop(task):
            manifest.append({"action": "drop", "old_index": old_index, "old_trial_id": task["trial_id"]})
            continue

        new_index = len(new_tasks)
        updated = deepcopy(task)
        layout = deepcopy(old_layout)
        replacement = None
        if RETIRED in task["objects"]:
            candidates = [name for name in pool if name not in task["objects"]]
            rng.shuffle(candidates)
            candidates.sort(key=lambda name: replacement_counts[name])
            for candidate in candidates:
                try:
                    candidate_row = replace_row(
                        task,
                        candidate,
                        benchmark,
                        colors,
                        source=f"replacement for old row {old_index}",
                    )
                except (ValueError, KeyError):
                    continue
                updated = candidate_row
                replacement = candidate
                break
            if replacement is None:
                raise RuntimeError(f"No unambiguous replacement for old row {old_index}")
            replacement_counts[replacement] += 1
            episode = benchmark.episode_spec_from_json(
                updated, source=f"replacement for old row {old_index}"
            )
            layout = layouts_module.generate_episode_layout(
                episode,
                episode_index=new_index,
                rng=layout_rng,
                bin_random_poses=bin_random_poses,
                valid_spawn_regions=constants["VALID_OBJECT_SPAWN_REGIONS"],
                table_object_z=constants["TABLE_TOP_Z"] + 0.001,
                seed=args.seed + 1,
                generated_at=generated_at,
                robot_bounding_box=constants["SO101_BOUNDING_BOX"],
            )
            manifest.append(
                {
                    "action": "replace",
                    "old_index": old_index,
                    "new_index": new_index,
                    "old_trial_id": task["trial_id"],
                    "replacement": replacement,
                }
            )

        updated["trial_id"] = new_index
        layout["trial_id"] = new_index
        layout["episode_index"] = new_index
        layout["instruction"] = updated["instruction"]
        new_tasks.append(updated)
        new_layouts.append(layout)

    for index, (task, layout) in enumerate(zip(new_tasks, new_layouts, strict=True)):
        if task["trial_id"] != index or layout["trial_id"] != index or layout["episode_index"] != index:
            raise AssertionError(f"Non-contiguous IDs at output row {index}")
        if RETIRED in task["objects"] or RETIRED in layout_object_names(layout):
            raise AssertionError(f"Retired object remains at output row {index}")
        if task["objects"] != layout_object_names(layout) or task["instruction"] != layout["instruction"]:
            raise AssertionError(f"Task/layout mismatch at output row {index}")
        benchmark.episode_spec_from_json(task, source=f"output row {index}")

    dropped = sum(entry["action"] == "drop" for entry in manifest)
    replaced = sum(entry["action"] == "replace" for entry in manifest)
    print(f"Rows: {len(tasks)} -> {len(new_tasks)}; dropped={dropped}; replaced={replaced}")
    print(f"Replacement counts: {dict(sorted(replacement_counts.items()))}")
    if not args.apply:
        print("Dry run only; pass --apply to write the validated changes.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_backup = tasks_path.with_name(f"{tasks_path.stem}.pre_corkscrew_removal_{timestamp}.jsonl")
    layout_backup = layouts_path.with_name(
        f"{layouts_path.stem}.pre_corkscrew_removal_{timestamp}.jsonl"
    )
    manifest_path = tasks_path.with_name(f"{tasks_path.stem}_corkscrew_removal_{timestamp}.jsonl")
    shutil.copy2(tasks_path, task_backup)
    shutil.copy2(layouts_path, layout_backup)
    write_jsonl_atomic(tasks_path, new_tasks)
    write_jsonl_atomic(layouts_path, new_layouts)
    write_jsonl_atomic(manifest_path, manifest)
    print(f"Updated tasks: {tasks_path}")
    print(f"Updated layouts: {layouts_path}")
    print(f"Task backup: {task_backup}")
    print(f"Layout backup: {layout_backup}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
