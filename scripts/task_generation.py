#!/usr/bin/env python3
"""Shared helpers for generating SO-101 Bench task and layout files.

Used by the episode-set generators (scripts/create_val_set.py, scripts/create_mmrv_eval.py, ...).
Nothing here imports Isaac Lab: placement constants are read from the env config source.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "source/so101_bench"
SPLITS = ("seen", "unseen_seen_class", "unseen_unseen_class")
DIRECTIONS = ("left", "right", "forward", "backward")


def env_cfg_constants() -> dict[str, Any]:
    """Read placement constants without importing the Isaac Lab environment."""

    path = PACKAGE_DIR / "so101_bench/tasks/direct/so101_bench/so101_bench_env_cfg.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {
        "TABLE_TOP_Z", "BIN_RANDOM_POSES_RPY_DEG", "TABLE_OBJECT_Z",
        "BIN_RANDOM_POSES", "VALID_OBJECT_SPAWN_REGIONS", "SO101_BOUNDING_BOX",
    }
    namespace: dict[str, Any] = {"math": math}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in wanted for target in node.targets
        ):
            exec(compile(ast.Module([node], type_ignores=[]), str(path), "exec"), namespace)
    required = ("TABLE_OBJECT_Z", "BIN_RANDOM_POSES", "VALID_OBJECT_SPAWN_REGIONS", "SO101_BOUNDING_BOX")
    missing = [name for name in required if name not in namespace]
    if missing:
        raise RuntimeError(f"Missing placement constants {missing} in {path}")
    return {name: namespace[name] for name in required}


def load_modules() -> tuple[ModuleType, ModuleType]:
    package = ModuleType("so101_bench")
    package.__path__ = [str(PACKAGE_DIR / "so101_bench")]
    sys.modules["so101_bench"] = package
    import so101_bench.benchmark as benchmark
    import so101_bench.layouts as layouts

    return benchmark, layouts

def scene_label(name: str, objects: list[str], colors: set[str]) -> str:
    """Omit a leading color only when its class is unique in this scene."""

    if name not in objects:
        raise ValueError(f"Named object {name!r} is absent from the scene")

    def without_color(object_name: str) -> str:
        words = object_name.split()
        return " ".join(words[1:]) if words and words[0] in colors else object_name

    short = without_color(name)
    if short != name and sum(without_color(other) == short for other in objects) == 1:
        return short
    return name


def instruction_for_row(row: dict[str, Any], family: str, colors: set[str]) -> str:
    objects = row["objects"]

    def label(name: str) -> str:
        return scene_label(name, objects, colors)

    if family in ("bin_1obj", "bin_4obj", "bin"):
        return "Place each object in the plastic bin"
    target = label(row["target"])
    if family == "named_bin":
        return f"Place the {target} in the plastic bin"
    if family == "next_to":
        return f"Place the {target} next to the {label(row['referents'][0])}"
    if family == "between":
        first, second = row["referents"]
        return f"Place the {target} between the {label(first)} and the {label(second)}"
    if family == "move":
        direction = {"forward": "forwards", "backward": "backwards"}.get(row["direction"], row["direction"])
        return f"Move the {target} {direction}"
    raise ValueError(f"Unknown task family: {family}")

def cell_key(benchmark: ModuleType, row: dict[str, Any]) -> tuple[str, str]:
    family = benchmark.infer_task_family(row["instruction"])
    return row["ood_key"], f"{family}_{row['n_objects']}" if family == "bin" else family


def order_rows(rows: list[dict[str, Any]], benchmark: ModuleType) -> list[dict[str, Any]]:
    """Group by split, family, and target, leaving four-object bins in place."""

    split_order = {split: index for index, split in enumerate(SPLITS)}
    family_order = {family: index for index, family in enumerate(("bin_1", "bin_4", "next_to", "between", "move"))}

    def sort_key(row: dict[str, Any]) -> tuple[int, int, str, int]:
        split, family = cell_key(benchmark, row)
        if family == "bin_4":
            return split_order[split], family_order[family], "", 0
        target = row["objects"][0] if family == "bin_1" else row["target"]
        profile_order = 0 if row["appearance_profile"] == "canonical" else 1
        return split_order[split], family_order[family], target, profile_order

    ordered = sorted(rows, key=sort_key)
    for trial_id, row in enumerate(ordered):
        row["trial_id"] = trial_id
    return ordered
