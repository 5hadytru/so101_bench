#!/usr/bin/env python3
"""Replace green-shoes distractors in half of green-clip named-bin episodes."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import math
from pathlib import Path
import random
import shutil

from rebalance_teleop3_black_shoes import (
    BLACK_SHOES,
    DEFAULT_LAYOUTS,
    DEFAULT_TASKS,
    FORBIDDEN,
    atomic_write,
    colorless,
    evenly_spaced_kept,
    footprint_metadata,
    is_named_bin,
    jsonl_text,
    load_geometry,
    read_jsonl,
    replacement_is_safe,
    repo_root,
    ranges,
    target_color_is_omitted,
)


GREEN_CLIP = "green clip"
GREEN_SHOES = "green shoes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--layouts", type=Path, default=DEFAULT_LAYOUTS)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    tasks_path = args.tasks if args.tasks.is_absolute() else root / args.tasks
    layouts_path = args.layouts if args.layouts.is_absolute() else root / args.layouts
    tasks = read_jsonl(tasks_path)
    layout_rows = read_jsonl(layouts_path)
    if len(tasks) != len(layout_rows):
        raise ValueError(f"Task/layout row counts differ: {len(tasks)} != {len(layout_rows)}")
    benchmark, layouts = load_geometry(root)
    layouts_by_trial = {str(row["trial_id"]): row for row in layout_rows}
    if len(layouts_by_trial) != len(layout_rows):
        raise ValueError("Layout trial_id values are not unique")
    for row in tasks:
        layout = layouts_by_trial.get(str(row["trial_id"]))
        if layout is None:
            raise KeyError(f"Missing layout for trial_id={row['trial_id']}")
        names = [entry["name"] for entry in sorted(layout["objects"], key=lambda item: item["slot"])]
        if names != row["objects"] or layout["instruction"] != row["instruction"]:
            raise ValueError(f"Task/layout mismatch at trial_id={row['trial_id']}")
    already_done = [
        row
        for row in layout_rows
        if "green" in str(row.get("placement", {}).get("layout_selection", ""))
        and "shoes_rebalanced" in str(row.get("placement", {}).get("layout_selection", ""))
    ]
    if already_done:
        raise RuntimeError(
            f"These files already contain {len(already_done)} green-shoes-rebalanced rows; refusing a second pass."
        )

    affected = [
        index
        for index, row in enumerate(tasks)
        if is_named_bin(row) and row.get("target") == GREEN_CLIP and GREEN_SHOES in row["objects"]
    ]
    replace_count = math.ceil(len(affected) * 0.5)
    keep = evenly_spaced_kept(affected, len(affected) - replace_count)
    to_modify = [index for index in affected if index not in keep]
    all_objects = [
        name
        for name in benchmark.OBJECT_SPLITS["seen"]
        if name not in {BLACK_SHOES, GREEN_SHOES, FORBIDDEN}
    ]
    named_appearances = Counter()
    for row in tasks:
        if is_named_bin(row):
            named_appearances.update(row["objects"])
    replacement_counts: Counter[str] = Counter()
    rng = random.Random(args.seed)
    manifest = []
    edited_at = datetime.now().astimezone().isoformat(timespec="seconds")
    for episode_index in to_modify:
        task = tasks[episode_index]
        layout = layouts_by_trial[str(task["trial_id"])]
        replaced_slot = task["objects"].index(GREEN_SHOES)
        replaced_entry = next(entry for entry in layout["objects"] if int(entry["slot"]) == replaced_slot)
        candidates = [name for name in all_objects if name not in task["objects"]]
        if target_color_is_omitted(task, benchmark.COLORS):
            candidates = [
                name
                for name in candidates
                if colorless(name, benchmark.COLORS) != colorless(task["target"], benchmark.COLORS)
            ]
        rng.shuffle(candidates)
        candidates.sort(key=lambda name: (named_appearances[name], replacement_counts[name]))
        chosen = None
        distances = None
        for candidate in candidates:
            safe, object_distance, bin_distance = replacement_is_safe(
                layout, replaced_entry, candidate, layouts, replaced_name=GREEN_SHOES
            )
            if safe:
                chosen = candidate
                distances = (object_distance, bin_distance)
                break
        if chosen is None or distances is None:
            raise RuntimeError(f"No safe replacement for green shoes in episode {episode_index}")

        task["objects"][replaced_slot] = chosen
        replaced_entry["name"] = chosen
        replaced_entry.update(footprint_metadata(chosen, layouts))
        placement = layout.setdefault("placement", {})
        old_object = placement.get("min_object_surface_distance_m")
        old_between = placement.get("min_between_object_surface_distance_m")
        old_bin = placement.get("min_bin_surface_distance_m")
        placement["layout_selection"] = (
            "black_and_green_shoes_rebalanced"
            if placement.get("layout_selection") == "black_shoes_rebalanced"
            else "green_shoes_rebalanced"
        )
        placement["min_object_surface_distance_m"] = min(
            value for value in (old_object, distances[0]) if value is not None
        )
        placement["min_between_object_surface_distance_m"] = min(
            value for value in (old_between, distances[0]) if value is not None
        )
        placement["min_bin_surface_distance_m"] = min(
            value for value in (old_bin, distances[1]) if value is not None
        )
        layout["rebalanced_at"] = edited_at
        named_appearances[GREEN_SHOES] -= 1
        named_appearances[chosen] += 1
        replacement_counts[chosen] += 1
        benchmark.episode_spec_from_json(task, source=f"modified task episode {episode_index}")
        manifest.append(
            {
                "schema_version": 1,
                "episode_index": episode_index,
                "editor_episode_number": episode_index + 1,
                "trial_id": task["trial_id"],
                "slot": replaced_slot,
                "target": GREEN_CLIP,
                "old_object": GREEN_SHOES,
                "new_object": chosen,
            }
        )

    remaining = sum(
        is_named_bin(row) and row.get("target") == GREEN_CLIP and GREEN_SHOES in row["objects"]
        for row in tasks
    )
    print(f"Green-clip named-bin episodes with green shoes: {len(affected)} -> {remaining}")
    print(f"Modified {len(to_modify)} rows ({100 * len(to_modify) / len(affected):.1f}%).")
    print(f"Replacement counts: {dict(sorted(replacement_counts.items()))}")
    print(f"Modified 1-based editor episodes: {ranges([index + 1 for index in to_modify])}")
    print(f"Modified 0-based rows/trial IDs: {ranges(to_modify)}")
    if not args.apply:
        print("Dry run only; pass --apply to write the validated changes.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_backup = tasks_path.with_name(f"{tasks_path.stem}.pre_green_shoes_rebalance_{timestamp}.jsonl")
    layout_backup = layouts_path.with_name(f"{layouts_path.stem}.pre_green_shoes_rebalance_{timestamp}.jsonl")
    manifest_path = tasks_path.with_name(f"{tasks_path.stem}_green_shoes_rebalance_{timestamp}.jsonl")
    shutil.copy2(tasks_path, task_backup)
    shutil.copy2(layouts_path, layout_backup)
    atomic_write(tasks_path, jsonl_text(tasks))
    atomic_write(layouts_path, jsonl_text(layout_rows))
    atomic_write(manifest_path, jsonl_text(manifest))
    print(f"Updated tasks: {tasks_path}")
    print(f"Updated layouts: {layouts_path}")
    print(f"Task backup: {task_backup}")
    print(f"Layout backup: {layout_backup}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
