#!/usr/bin/env python3
"""Reduce black-shoes distractors in teleop_3 named-bin tasks.

The task and layout files are changed together. Replacement objects retain the
black-shoes root pose, but receive their own exact footprint metadata. A
replacement is accepted only when its raster footprint preserves the recorded
object/bin clearances and its world AABB fits inside the old black-shoes AABB.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from types import ModuleType
from typing import Any

import numpy as np


BLACK_SHOES = "black shoes"
FORBIDDEN = "brown stuffed animal"
DEFAULT_TASKS = Path("tasks/teleop_3.jsonl")
DEFAULT_LAYOUTS = Path("tasks/layouts/teleop_3_layouts_20260720_141740.jsonl")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_geometry(root: Path) -> tuple[ModuleType, ModuleType]:
    package = ModuleType("so101_bench")
    package.__path__ = [str(root / "source" / "so101_bench" / "so101_bench")]
    sys.modules["so101_bench"] = package
    benchmark = load_module(
        "so101_bench.benchmark",
        root / "source" / "so101_bench" / "so101_bench" / "benchmark.py",
    )
    layouts = load_module(
        "so101_bench.layouts",
        root / "source" / "so101_bench" / "so101_bench" / "layouts.py",
    )
    return benchmark, layouts


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)


def atomic_write(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def is_named_bin(row: dict[str, Any]) -> bool:
    return "target" in row and "direction" not in row and not row.get("referents")


def colorless(name: str, colors: set[str]) -> str:
    words = name.split()
    return " ".join(words[1:]) if words and words[0] in colors else name


def target_color_is_omitted(row: dict[str, Any], colors: set[str]) -> bool:
    target = str(row["target"])
    shortened = colorless(target, colors)
    return shortened != target and f"the {shortened} in the plastic bin" in row["instruction"]


def world_footprint(entry: dict[str, Any], name: str, layouts: ModuleType) -> Any:
    position = entry["position"]
    yaw = float(entry.get("yaw", entry.get("rpy", [0.0, 0.0, 0.0])[2]))
    return layouts._world_footprint(
        layouts._object_local_corners(name),
        (float(position[0]), float(position[1])),
        yaw,
    )


def contained_aabb(inner: tuple[float, ...], outer: tuple[float, ...], tolerance: float = 1e-9) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def replacement_is_safe(
    layout: dict[str, Any],
    replaced_entry: dict[str, Any],
    candidate: str,
    layouts: ModuleType,
    replaced_name: str = BLACK_SHOES,
) -> tuple[bool, float, float]:
    candidate_fp = world_footprint(replaced_entry, candidate, layouts)
    replaced_fp = world_footprint(replaced_entry, replaced_name, layouts)
    if not contained_aabb(candidate_fp.total, replaced_fp.total):
        return False, -math.inf, -math.inf

    object_threshold = float(
        layout.get("placement", {}).get("required_min_object_surface_distance_m")
        or layouts.MIN_OBJECT_SURFACE_DISTANCE_M
    )
    bin_threshold = float(
        layout.get("placement", {}).get("required_min_bin_surface_distance_m")
        or layouts.MIN_BIN_SURFACE_DISTANCE_M
    )
    other_distances = []
    for entry in layout["objects"]:
        if entry is replaced_entry:
            continue
        other_fp = world_footprint(entry, str(entry["name"]), layouts)
        distance = layouts._pieces_surface_distance(
            layouts._footprint_pieces(candidate_fp), layouts._footprint_pieces(other_fp)
        )
        other_distances.append(distance)
        if distance + 1e-9 < object_threshold:
            return False, distance, -math.inf

    bin_entry = layout["bin"]
    half_extents = tuple(float(value) for value in bin_entry["footprint_half_extents"])
    bin_fp = layouts._world_footprint(
        layouts._rectangle_local_corners(half_extents),
        (float(bin_entry["position"][0]), float(bin_entry["position"][1])),
        float(bin_entry["rpy"][2]),
    )
    bin_distance = layouts._pieces_surface_distance(
        layouts._footprint_pieces(candidate_fp), layouts._footprint_pieces(bin_fp)
    )
    if bin_distance + 1e-9 < bin_threshold:
        return False, min(other_distances, default=math.inf), bin_distance
    return True, min(other_distances, default=math.inf), bin_distance


def footprint_metadata(name: str, layouts: ModuleType) -> dict[str, Any]:
    boxes = layouts._object_footprint_boxes(name)
    x0, y0 = min(box[0] for box in boxes), min(box[1] for box in boxes)
    x1, y1 = max(box[2] for box in boxes), max(box[3] for box in boxes)
    return {
        "footprint_boxes": [list(box) for box in boxes],
        "footprint_half_extents": [0.5 * (x1 - x0), 0.5 * (y1 - y0)],
        "footprint_center_offset": [0.5 * (x0 + x1), 0.5 * (y0 + y1)],
    }


def evenly_spaced_kept(values: list[int], keep_count: int) -> set[int]:
    if keep_count <= 0:
        return set()
    if keep_count >= len(values):
        return set(values)
    positions = {round(index * (len(values) - 1) / (keep_count - 1)) for index in range(keep_count)}
    return {values[position] for position in positions}


def ranges(values: list[int]) -> str:
    if not values:
        return "<none>"
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
    parser.add_argument("--replace-fraction", type=float, default=0.9)
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
    already_rebalanced = [
        row
        for row in layout_rows
        if "black" in str(row.get("placement", {}).get("layout_selection", ""))
        and "shoes_rebalanced" in str(row.get("placement", {}).get("layout_selection", ""))
    ]
    if already_rebalanced:
        raise RuntimeError(
            f"These files already contain {len(already_rebalanced)} black-shoes-rebalanced layout rows; "
            "refusing to apply the reduction a second time."
        )

    affected = [
        index for index, row in enumerate(tasks) if is_named_bin(row) and BLACK_SHOES in row["objects"]
    ]
    replace_count = round(len(affected) * args.replace_fraction)
    keep = evenly_spaced_kept(affected, len(affected) - replace_count)
    to_modify = [index for index in affected if index not in keep]

    all_objects = [
        name
        for name in benchmark.OBJECT_SPLITS["seen"]
        if name not in {BLACK_SHOES, FORBIDDEN}
    ]
    named_appearances = Counter()
    for row in tasks:
        if is_named_bin(row):
            named_appearances.update(row["objects"])
    replacement_counts: Counter[str] = Counter()
    rng = random.Random(args.seed)
    manifest = []
    for episode_index in to_modify:
        task = tasks[episode_index]
        layout = layouts_by_trial[str(task["trial_id"])]
        shoe_slot = task["objects"].index(BLACK_SHOES)
        shoe_entry = next(entry for entry in layout["objects"] if int(entry["slot"]) == shoe_slot)
        candidates = [name for name in all_objects if name not in task["objects"]]
        if target_color_is_omitted(task, benchmark.COLORS):
            candidates = [
                name for name in candidates if colorless(name, benchmark.COLORS) != colorless(task["target"], benchmark.COLORS)
            ]
        rng.shuffle(candidates)
        candidates.sort(key=lambda name: (named_appearances[name], replacement_counts[name]))
        chosen = None
        chosen_distances = None
        for candidate in candidates:
            safe, object_distance, bin_distance = replacement_is_safe(layout, shoe_entry, candidate, layouts)
            if safe:
                chosen = candidate
                chosen_distances = (object_distance, bin_distance)
                break
        if chosen is None or chosen_distances is None:
            raise RuntimeError(f"No safe replacement for black shoes in episode {episode_index}")

        task["objects"][shoe_slot] = chosen
        shoe_entry["name"] = chosen
        shoe_entry.update(footprint_metadata(chosen, layouts))
        placement = layout.setdefault("placement", {})
        old_object_distance = placement.get("min_object_surface_distance_m")
        old_between_distance = placement.get("min_between_object_surface_distance_m")
        old_bin_distance = placement.get("min_bin_surface_distance_m")
        placement["layout_selection"] = "black_shoes_rebalanced"
        placement["min_object_surface_distance_m"] = min(
            value for value in (old_object_distance, chosen_distances[0]) if value is not None
        )
        placement["min_between_object_surface_distance_m"] = min(
            value for value in (old_between_distance, chosen_distances[0]) if value is not None
        )
        placement["min_bin_surface_distance_m"] = min(
            value for value in (old_bin_distance, chosen_distances[1]) if value is not None
        )
        layout["rebalanced_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        named_appearances[BLACK_SHOES] -= 1
        named_appearances[chosen] += 1
        replacement_counts[chosen] += 1
        benchmark.episode_spec_from_json(task, source=f"modified task episode {episode_index}")
        manifest.append(
            {
                "schema_version": 1,
                "episode_index": episode_index,
                "editor_episode_number": episode_index + 1,
                "trial_id": task["trial_id"],
                "slot": shoe_slot,
                "old_object": BLACK_SHOES,
                "new_object": chosen,
            }
        )

    remaining = sum(is_named_bin(row) and BLACK_SHOES in row["objects"] for row in tasks)
    expected_remaining = len(affected) - replace_count
    if remaining != expected_remaining:
        raise AssertionError(f"Expected {expected_remaining} remaining black-shoes rows, got {remaining}")
    print(f"Named-bin black-shoes appearances: {len(affected)} -> {remaining}")
    print(f"Modified {len(to_modify)} rows ({100 * len(to_modify) / len(affected):.1f}%).")
    print(f"Replacement counts: {dict(sorted(replacement_counts.items()))}")
    print(f"Modified 1-based editor episodes: {ranges([value + 1 for value in to_modify])}")
    print(f"Modified 0-based rows/trial IDs: {ranges(to_modify)}")
    if not args.apply:
        print("Dry run only; pass --apply to write the validated changes.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_backup = tasks_path.with_name(f"{tasks_path.stem}.pre_black_shoes_rebalance_{timestamp}.jsonl")
    layout_backup = layouts_path.with_name(f"{layouts_path.stem}.pre_black_shoes_rebalance_{timestamp}.jsonl")
    manifest_path = tasks_path.with_name(f"{tasks_path.stem}_black_shoes_rebalance_{timestamp}.jsonl")
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
