#!/usr/bin/env python3
"""Prune teleop_3 four-bin/named-bin rows while preserving its recorded prefix."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


DEFAULT_TASKS = Path("tasks/teleop_3.jsonl")
DEFAULT_LAYOUTS = Path("tasks/layouts/teleop_3_layouts_20260720_141740.jsonl")


def load_generator(root: Path) -> Any:
    path = root / "scripts" / "create_teleop_3_tasks.py"
    spec = importlib.util.spec_from_file_location("teleop3_generator_for_prune", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--layouts", type=Path, default=DEFAULT_LAYOUTS)
    parser.add_argument("--keep-four-bin", type=int, default=304)
    parser.add_argument("--keep-named-bin", type=int, default=400)
    parser.add_argument("--recorded-prefix", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260721)
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
        raise ValueError(f"Task/layout row counts differ: {len(tasks)} != {len(layouts)}")
    generator = load_generator(root)
    benchmark = generator.load_benchmark(root)

    layouts_by_trial = {str(row["trial_id"]): row for row in layouts}
    if len(layouts_by_trial) != len(layouts):
        raise ValueError("Layout trial IDs are not unique")
    for index, task in enumerate(tasks):
        layout = layouts_by_trial.get(str(task["trial_id"]))
        if layout is None:
            raise KeyError(f"Missing layout for task row {index}, trial_id={task['trial_id']}")
        names = [entry["name"] for entry in sorted(layout["objects"], key=lambda entry: entry["slot"])]
        if names != task["objects"] or layout["instruction"] != task["instruction"]:
            raise ValueError(f"Task/layout mismatch at row {index}")

    four = [index for index, row in enumerate(tasks) if generator.family(row) == "bin_4obj"]
    named = [index for index, row in enumerate(tasks) if generator.family(row) == "named_bin"]
    if len(four) < args.keep_four_bin or len(named) < args.keep_named_bin:
        raise ValueError(
            f"Cannot retain four={args.keep_four_bin}, named={args.keep_named_bin} from "
            f"four={len(four)}, named={len(named)}"
        )
    if len(four) == args.keep_four_bin and len(named) == args.keep_named_bin:
        raise RuntimeError("The requested bin-family counts are already present")

    four_set, named_set = set(four), set(named)
    candidates = four + named
    candidate_set = set(candidates)
    graspable = sorted(
        {
            name
            for index in candidates
            for name in (tasks[index]["objects"] if index in four_set else [tasks[index]["target"]])
            if name != generator.SUPPORT_ONLY_OBJECT
        }
    )
    fixed_grasps: Counter[str] = Counter()
    for index, row in enumerate(tasks):
        if index in candidate_set:
            continue
        task_family = generator.family(row)
        if task_family == "bin_1obj":
            fixed_grasps[row["objects"][0]] += 1
        else:
            fixed_grasps[row["target"]] += 1

    matrix = []
    lower_constraints = []
    upper_constraints = []
    matrix.append([int(index in four_set) for index in candidates])
    lower_constraints.append(args.keep_four_bin)
    upper_constraints.append(args.keep_four_bin)
    matrix.append([int(index in named_set) for index in candidates])
    lower_constraints.append(args.keep_named_bin)
    upper_constraints.append(args.keep_named_bin)

    total_grasps = (
        sum(fixed_grasps.values()) + 4 * args.keep_four_bin + args.keep_named_bin
    )
    quotient, remainder = divmod(total_grasps, len(graspable))
    if remainder and quotient + 1 - quotient > 1:
        raise AssertionError("Unexpected grasp quota calculation")
    for object_name in graspable:
        matrix.append(
            [
                int(object_name in tasks[index]["objects"])
                if index in four_set
                else int(tasks[index]["target"] == object_name)
                for index in candidates
            ]
        )
        lower_constraints.append(quotient - fixed_grasps[object_name])
        upper_constraints.append(quotient + int(remainder > 0) - fixed_grasps[object_name])

    rng = np.random.default_rng(args.seed)
    objective = rng.random(len(candidates))
    for variable, source_index in enumerate(candidates):
        layout = layouts_by_trial[str(tasks[source_index]["trial_id"])]
        if layout.get("edited_at") or layout.get("placement", {}).get("layout_selection") == "hand_edited":
            objective[variable] -= 1000.0
    variable_lower = np.zeros(len(candidates))
    variable_upper = np.ones(len(candidates))
    for variable, source_index in enumerate(candidates):
        if source_index < args.recorded_prefix:
            variable_lower[variable] = 1.0
            variable_upper[variable] = 1.0

    result = milp(
        objective,
        integrality=np.ones(len(candidates)),
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(
            np.asarray(matrix, dtype=float),
            np.asarray(lower_constraints, dtype=float),
            np.asarray(upper_constraints, dtype=float),
        ),
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Could not find a balanced retained subset: {result.message}")
    retained_candidates = {
        source_index
        for source_index, value in zip(candidates, np.rint(result.x).astype(int), strict=True)
        if value
    }
    deleted = sorted(candidate_set - retained_candidates)
    deleted_set = set(deleted)

    new_tasks = []
    new_layouts = []
    manifest = []
    final_grasps: Counter[str] = Counter()
    for source_index, source_task in enumerate(tasks):
        if source_index in deleted_set:
            continue
        source_layout = layouts_by_trial[str(source_task["trial_id"])]
        new_index = len(new_tasks)
        task = dict(source_task)
        layout = dict(source_layout)
        task["trial_id"] = new_index
        layout["trial_id"] = new_index
        layout["episode_index"] = new_index
        new_tasks.append(task)
        new_layouts.append(layout)
        task_family = generator.family(task)
        if task_family == "bin_4obj":
            final_grasps.update(task["objects"])
        elif task_family == "bin_1obj":
            final_grasps[task["objects"][0]] += 1
        else:
            final_grasps[task["target"]] += 1
        manifest.append(
            {
                "schema_version": 1,
                "new_episode_index": new_index,
                "source_episode_index": source_index,
                "source_trial_id": source_task["trial_id"],
                "task_family": task_family,
                "recorded_prefix": source_index < args.recorded_prefix,
            }
        )

    if new_tasks[: args.recorded_prefix] != tasks[: args.recorded_prefix]:
        raise AssertionError("Recorded task prefix changed")
    if new_layouts[: args.recorded_prefix] != layouts[: args.recorded_prefix]:
        raise AssertionError("Recorded layout prefix changed")
    family_counts = Counter(generator.family(row) for row in new_tasks)
    expected = {
        "bin_1obj": 330,
        "bin_4obj": args.keep_four_bin,
        "named_bin": args.keep_named_bin,
        "next_to": 100,
        "between": 100,
        "move": 150,
    }
    if family_counts != Counter(expected):
        raise AssertionError(f"Unexpected final family counts: {family_counts}")
    if max(final_grasps.values()) - min(final_grasps.values()) > 1:
        raise AssertionError(f"Final grasps are imbalanced: {final_grasps}")
    for index, (task, layout) in enumerate(zip(new_tasks, new_layouts, strict=True)):
        names = [entry["name"] for entry in sorted(layout["objects"], key=lambda entry: entry["slot"])]
        if task["trial_id"] != index or layout["trial_id"] != index or layout["episode_index"] != index:
            raise AssertionError(f"Non-contiguous IDs at row {index}")
        if names != task["objects"] or layout["instruction"] != task["instruction"]:
            raise AssertionError(f"Task/layout mismatch at row {index}")
        benchmark.episode_spec_from_json(task, source=f"retained task row {index}")

    edited_candidates = [
        index
        for index in candidates
        if layouts_by_trial[str(tasks[index]["trial_id"])].get("edited_at")
        or layouts_by_trial[str(tasks[index]["trial_id"])].get("placement", {}).get("layout_selection")
        == "hand_edited"
    ]
    print(f"Rows: {len(tasks)} -> {len(new_tasks)}; removed {len(deleted)} bin rows")
    print(f"Family counts: {dict(family_counts)}")
    print(f"Grasp balance: {min(final_grasps.values())}-{max(final_grasps.values())}")
    print(
        f"Recorded prefix preserved: rows 0-{args.recorded_prefix - 1}; "
        f"retained edited candidate layouts {sum(index in retained_candidates for index in edited_candidates)}/"
        f"{len(edited_candidates)}"
    )
    if not args.apply:
        print("Dry run only; pass --apply to write changes.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_backup = task_path.with_name(f"{task_path.stem}.pre_bin_count_prune_{timestamp}.jsonl")
    layout_backup = layout_path.with_name(f"{layout_path.stem}.pre_bin_count_prune_{timestamp}.jsonl")
    manifest_path = task_path.with_name(f"{task_path.stem}_bin_count_prune_index_{timestamp}.jsonl")
    deleted_path = task_path.with_name(f"{task_path.stem}_bin_count_prune_deleted_{timestamp}.jsonl")
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
                    "task_family": generator.family(tasks[index]),
                    "target": tasks[index].get("target"),
                    "objects": tasks[index]["objects"],
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
