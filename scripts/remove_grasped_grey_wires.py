#!/usr/bin/env python3
"""Replace future teleop grey-wire grasps and remove them from eval plans.

The script operates on the task/layout pairs currently used for teleop 3 and
the GR00T validation suite.  It leaves grey wires alone when they are only a
distractor or a spatial referent: an episode is selected only if every object
is grasped (the all-object bin task) or its canonical target is grey wires.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
import types
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "source" / "so101_bench" / "so101_bench"

TELEOP_TASKS = ROOT / "tasks" / "teleop_3.jsonl"
TELEOP_LAYOUTS = ROOT / "tasks" / "layouts" / "teleop_3_layouts_20260720_141740.jsonl"
WM_TASKS = ROOT / "tasks" / "real_gr00t_WM_all_filtered.jsonl"
WM_LAYOUTS = ROOT / "tasks" / "layouts" / "real_gr00t_WM_all_filtered_layouts_20260717_215034.jsonl"
VAL_TASKS = ROOT / "tasks" / "real_gr00t_val.jsonl"
VAL_LAYOUTS = ROOT / "tasks" / "layouts" / "real_gr00t_val_layouts_20260718_155446.jsonl"

# The teleop log says "Resetting episode 1022/1377".  That is a one-based
# ordinal, while Python rows are zero-based, so the first editable row is 1022.
CURRENT_TELEOP_ORDINAL = 1022
GREY_WIRES = "grey wires"
REPLACEMENT_RNG_SEED = 20260725
BACKUP_SUFFIX = ".before_grasped_grey_wires_20260725"

# These are copied from so101_bench_env_cfg.py so this utility can use the
# pure-Python layout sampler without importing Isaac Sim.
BIN_RANDOM_POSES_RPY_DEG = (
    ((-0.12917, -0.16276, 0.02), (0.0, 0.0, -66.023)),
    ((-0.15984, -0.06498, 0.02), (0.0, 0.0, -90.0)),
    ((-0.15984, 0.02407, 0.02), (0.0, 0.0, -90.0)),
    ((0.37129, 0.03595, 0.02), (0.0, 0.0, -90.0)),
    ((0.37129, -0.07184, 0.02), (0.0, 0.0, -90.0)),
    ((0.32627, -0.17514, 0.02), (0.0, 0.0, -120.541)),
)
VALID_OBJECT_SPAWN_REGIONS = [
    [(-.11, .15, 0), (-.0512, .04475, 0), (.01521, -.099, 0), (.25, -.099, 0), (.25, .15, 0), (.185, .15, 0), (.147, .07, 0), (.145, .05146, 0), (.141, -.004, 0), (.01, -.0005, 0), (-.047, .15, 0)],
    [(-.02, -.008, 0), (-.02, -.099, 0), (.25, -.099, 0), (.25, .15, 0), (.19, .15, 0), (.165, .09, 0), (.14491, .053, 0), (.137, -.006, 0)],
    [(-.02, -.008, 0), (-.02, -.099, 0), (.25, -.099, 0), (.25, .15, 0), (.19, .15, 0), (.165, .09, 0), (.14491, .053, 0), (.137, -.006, 0)],
    [(-.11, .15, 0), (-.1022, -.084, 0), (.211568, -.099, 0), (.211568, .008515, 0), (.211568, .15, 0), (.185, .15, 0), (.155, .08, 0), (.155, .08, 0), (.137, -.01, 0), (.015, -.01, 0), (-.0085, .05435, 0), (-.043, .128006, 0), (-.043, .15, 0)],
    [(-.11, .15, 0), (-.1022, -.084, 0), (.211568, -.099, 0), (.211568, .008515, 0), (.211568, .15, 0), (.185, .15, 0), (.155, .08, 0), (.155, .08, 0), (.137, -.01, 0), (.015, -.01, 0), (-.0085, .05435, 0), (-.043, .128006, 0), (-.043, .15, 0)],
    [(-.11, .15, 0), (-.1022, -.084, 0), (.186416, -.099, 0), (.25, .008515, 0), (.25, .15, 0), (.185, .15, 0), (.155, .08, 0), (.155, .08, 0), (.137, -.01, 0), (.015, -.01, 0), (-.0085, .05435, 0), (-.043, .128006, 0), (-.043, .15, 0)],
]
SO101_BOUNDING_BOX = [(.012, .241, 0), (.0113, .1267, 0), (.0437, .1262, 0), (.0431, .0182, 0), (.0939, .0179, 0), (.0939, .1262, 0), (.1256, .126, 0), (.1263, .2403, 0)]


def _load_pure_python_modules():
    """Load benchmark/layout helpers without package-level Isaac registration."""

    package = types.ModuleType("so101_bench")
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules.setdefault("so101_bench", package)
    benchmark = importlib.import_module("so101_bench.benchmark")
    layouts = importlib.import_module("so101_bench.layouts")
    return benchmark, layouts


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def object_names(layout: dict[str, Any]) -> list[str]:
    return [entry["name"] for entry in sorted(layout["objects"], key=lambda entry: entry["slot"])]


def validate_pair(tasks: list[dict[str, Any]], layouts: list[dict[str, Any]], *, label: str, benchmark: Any) -> None:
    if len(tasks) != len(layouts):
        raise ValueError(f"{label}: task/layout count mismatch: {len(tasks)} != {len(layouts)}")
    layout_by_trial: dict[str, dict[str, Any]] = {}
    for layout in layouts:
        trial_id = layout.get("trial_id")
        if trial_id is None or str(trial_id) in layout_by_trial:
            raise ValueError(f"{label}: layouts have a missing or duplicate trial_id")
        layout_by_trial[str(trial_id)] = layout
    for index, task in enumerate(tasks):
        benchmark.episode_spec_from_json(task, source=f"{label} task row {index}")
        layout = layout_by_trial.get(str(task.get("trial_id")))
        if layout is None:
            raise ValueError(f"{label}: no layout for task trial_id={task.get('trial_id')!r}")
        if task["instruction"] != layout.get("instruction") or task["objects"] != object_names(layout):
            raise ValueError(f"{label}: task/layout mismatch at task row {index}")


def grey_wires_are_grasped(row: dict[str, Any], benchmark: Any) -> bool:
    if GREY_WIRES not in row["objects"]:
        return False
    episode = benchmark.episode_spec_from_json(row, source=f"trial {row.get('trial_id')}")
    return episode.task_family == benchmark.TASK_BIN or episode.objects[episode.target_object_id] == GREY_WIRES


def replacement_candidates(benchmark: Any) -> list[str]:
    # Exclude objects the teleop-3 generator deliberately never treats as normal
    # grasp targets.  Every remaining candidate is a seen object and has a layout
    # footprint asset.
    excluded = {GREY_WIRES, "brown stuffed animal", "black shoes"}
    return sorted(name for name in benchmark.OBJECT_SPLITS["seen"] if name not in excluded)


def replace_future_teleop_rows(
    tasks: list[dict[str, Any]], layouts: list[dict[str, Any]], benchmark: Any, layouts_module: Any
) -> list[tuple[int, int, str]]:
    candidates = replacement_candidates(benchmark)
    rng = random.Random(REPLACEMENT_RNG_SEED)
    layout_rng = random.Random(REPLACEMENT_RNG_SEED + 1)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    layouts_by_trial = {str(layout["trial_id"]): layout for layout in layouts}
    changed: list[tuple[int, int, str]] = []

    import math

    bin_random_poses = tuple(
        (translation, tuple(math.radians(angle) for angle in rpy_deg))
        for translation, rpy_deg in BIN_RANDOM_POSES_RPY_DEG
    )
    for row_index, task in enumerate(tasks):
        if row_index < CURRENT_TELEOP_ORDINAL or not grey_wires_are_grasped(task, benchmark):
            continue
        if task.get("target") == GREY_WIRES:
            raise ValueError(f"Future teleop target episode unexpectedly selected: trial {task.get('trial_id')}")
        choices = [name for name in candidates if name not in task["objects"]]
        if not choices:
            raise RuntimeError(f"No unambiguous replacement candidate for trial {task.get('trial_id')}")
        replacement = rng.choice(choices)
        updated_task = deepcopy(task)
        updated_task["objects"] = [replacement if name == GREY_WIRES else name for name in task["objects"]]
        benchmark.episode_spec_from_json(updated_task, source=f"updated teleop trial {task['trial_id']}")
        task.clear()
        task.update(updated_task)

        episode = benchmark.episode_spec_from_json(task, source=f"updated teleop trial {task['trial_id']}")
        regenerated = layouts_module.generate_episode_layout(
            episode,
            episode_index=row_index,
            rng=layout_rng,
            bin_random_poses=bin_random_poses,
            valid_spawn_regions=VALID_OBJECT_SPAWN_REGIONS,
            table_object_z=0.001,
            seed=REPLACEMENT_RNG_SEED + 1,
            generated_at=generated_at,
            robot_bounding_box=SO101_BOUNDING_BOX,
        )
        regenerated["trial_id"] = task["trial_id"]
        # Preserve the original ordinal, which differs from trial_id after the
        # historic teleop pruning already present in this file.
        regenerated["episode_index"] = layouts_by_trial[str(task["trial_id"])].get("episode_index", row_index)
        layouts_by_trial[str(task["trial_id"])].clear()
        layouts_by_trial[str(task["trial_id"])].update(regenerated)
        changed.append((row_index, int(task["trial_id"]), replacement))
    return changed


def remove_eval_rows(tasks: list[dict[str, Any]], layouts: list[dict[str, Any]], benchmark: Any) -> list[int]:
    layouts_by_trial = {str(layout["trial_id"]): layout for layout in layouts}
    removed = [task["trial_id"] for task in tasks if grey_wires_are_grasped(task, benchmark)]
    removed_keys = {str(trial_id) for trial_id in removed}
    kept_tasks = [deepcopy(task) for task in tasks if str(task["trial_id"]) not in removed_keys]
    kept_layouts = [deepcopy(layouts_by_trial[str(task["trial_id"])]) for task in tasks if str(task["trial_id"]) not in removed_keys]
    for compact_id, (task, layout) in enumerate(zip(kept_tasks, kept_layouts, strict=True)):
        task["trial_id"] = compact_id
        layout["trial_id"] = compact_id
        if "episode_index" in layout:
            layout["episode_index"] = compact_id
    tasks[:] = kept_tasks
    layouts[:] = kept_layouts
    return [int(trial_id) for trial_id in removed]


def backup(paths: list[Path]) -> None:
    for path in paths:
        destination = path.with_name(path.name + BACKUP_SUFFIX)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing backup: {destination}")
        shutil.copy2(path, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate and report changes without writing files.")
    args = parser.parse_args()
    benchmark, layouts_module = _load_pure_python_modules()

    teleop_tasks, teleop_layouts = read_jsonl(TELEOP_TASKS), read_jsonl(TELEOP_LAYOUTS)
    wm_tasks, wm_layouts = read_jsonl(WM_TASKS), read_jsonl(WM_LAYOUTS)
    val_tasks, val_layouts = read_jsonl(VAL_TASKS), read_jsonl(VAL_LAYOUTS)
    validate_pair(teleop_tasks, teleop_layouts, label="teleop", benchmark=benchmark)
    validate_pair(wm_tasks, wm_layouts, label="WM", benchmark=benchmark)
    validate_pair(val_tasks, val_layouts, label="val", benchmark=benchmark)

    teleop_changes = replace_future_teleop_rows(teleop_tasks, teleop_layouts, benchmark, layouts_module)
    wm_removed = remove_eval_rows(wm_tasks, wm_layouts, benchmark)
    val_removed = remove_eval_rows(val_tasks, val_layouts, benchmark)
    validate_pair(teleop_tasks, teleop_layouts, label="updated teleop", benchmark=benchmark)
    validate_pair(wm_tasks, wm_layouts, label="updated WM", benchmark=benchmark)
    validate_pair(val_tasks, val_layouts, label="updated val", benchmark=benchmark)

    print(f"Teleop replacements ({len(teleop_changes)}): {teleop_changes}")
    print(f"WM removed source trial IDs ({len(wm_removed)}): {wm_removed}")
    print(f"Val removed source trial IDs ({len(val_removed)}): {val_removed}")
    print(f"Resulting eval row counts: WM={len(wm_tasks)}, val={len(val_tasks)}")
    if args.dry_run:
        return
    paths = [TELEOP_TASKS, TELEOP_LAYOUTS, WM_TASKS, WM_LAYOUTS, VAL_TASKS, VAL_LAYOUTS]
    backup(paths)
    for path, rows in (
        (TELEOP_TASKS, teleop_tasks),
        (TELEOP_LAYOUTS, teleop_layouts),
        (WM_TASKS, wm_tasks),
        (WM_LAYOUTS, wm_layouts),
        (VAL_TASKS, val_tasks),
        (VAL_LAYOUTS, val_layouts),
    ):
        write_jsonl_atomic(path, rows)
    print("Wrote all task/layout pairs and immutable pre-change backups.")


if __name__ == "__main__":
    main()
