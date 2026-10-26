#!/usr/bin/env python3
"""Substitute retired objects in aligned task/layout JSONLs and resample layouts.

Bin episodes must hold exactly one or four objects, so dropping a retired object
from a four-object row is not a legal edit.  This restores the row's object count
by swapping in a replacement drawn from the objects already used elsewhere in the
same task file, then regenerates that episode's layout with the shared sampler.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import types
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "source" / "so101_bench" / "so101_bench"
ENV_CFG = PACKAGE_ROOT / "tasks" / "direct" / "so101_bench" / "so101_bench_env_cfg.py"


def _load_pure_python_modules():
    """Load benchmark/layout helpers without package-level Isaac registration."""

    package = types.ModuleType("so101_bench")
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules.setdefault("so101_bench", package)
    return importlib.import_module("so101_bench.benchmark"), importlib.import_module("so101_bench.layouts")


def _env_cfg_literals() -> dict[str, Any]:
    """Read the placement constants out of the env cfg without importing Isaac."""

    import ast

    wanted = {
        "TABLE_TOP_Z",
        "BIN_RANDOM_POSES_RPY_DEG",
        "VALID_OBJECT_SPAWN_REGIONS",
        "SO101_BOUNDING_BOX",
    }
    found: dict[str, Any] = {}
    for node in ast.parse(ENV_CFG.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            found[target.id] = ast.literal_eval(node.value)
    missing = wanted - found.keys()
    if missing:
        raise RuntimeError(f"{ENV_CFG}: could not read {sorted(missing)}")
    return found


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
            stream.write(json.dumps(row) + "\n")
    temporary.replace(path)


def layout_object_names(layout: dict[str, Any]) -> list[str]:
    return [entry["name"] for entry in sorted(layout["objects"], key=lambda entry: entry["slot"])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-jsonl", type=Path, required=True)
    parser.add_argument("--episode-layouts-jsonl", type=Path, required=True)
    parser.add_argument(
        "--retired-object",
        action="append",
        default=[],
        required=True,
        help="Object name to swap out; repeat for each retired object.",
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--backup-suffix", default=".bak")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark, layouts_module = _load_pure_python_modules()
    constants = _env_cfg_literals()

    tasks = read_jsonl(args.episodes_jsonl)
    layouts = read_jsonl(args.episode_layouts_jsonl)
    if len(tasks) != len(layouts):
        raise ValueError(f"Task/layout row counts differ: {len(tasks)} vs {len(layouts)}")

    retired = set(args.retired_object)
    # Only objects this file already uses are candidates: they are known to have
    # move-footprint assets and to match the set's difficulty mix.
    pool = sorted(
        {name for task in tasks for name in task["objects"] if name not in retired}
    )
    if not pool:
        raise RuntimeError("No replacement candidates available.")

    rng = random.Random(args.seed)
    layout_rng = random.Random(args.seed + 1)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bin_random_poses = tuple(
        (translation, tuple(math.radians(angle) for angle in rpy_deg))
        for translation, rpy_deg in constants["BIN_RANDOM_POSES_RPY_DEG"]
    )

    changed: list[tuple[int, list[tuple[str, str]]]] = []
    for row_index, (task, layout) in enumerate(zip(tasks, layouts)):
        if task["trial_id"] != layout["trial_id"]:
            raise ValueError(f"Row {row_index}: mismatched trial IDs")
        if task["objects"] != layout_object_names(layout):
            raise ValueError(f"Row {row_index}: task/layout objects differ")
        if not retired.intersection(task["objects"]):
            continue

        swaps: list[tuple[str, str]] = []
        objects = list(task["objects"])
        for slot, name in enumerate(objects):
            if name not in retired:
                continue
            choices = [candidate for candidate in pool if candidate not in objects]
            if not choices:
                raise RuntimeError(f"No replacement candidate for trial {task['trial_id']}")
            replacement = rng.choice(choices)
            objects[slot] = replacement
            swaps.append((name, replacement))

        updated = deepcopy(task)
        updated["objects"] = objects
        updated["n_objects"] = len(objects)
        episode = benchmark.episode_spec_from_json(
            updated, source=f"{args.episodes_jsonl}: trial {updated['trial_id']}"
        )
        regenerated = layouts_module.generate_episode_layout(
            episode,
            episode_index=row_index,
            rng=layout_rng,
            bin_random_poses=bin_random_poses,
            valid_spawn_regions=constants["VALID_OBJECT_SPAWN_REGIONS"],
            table_object_z=constants["TABLE_TOP_Z"] + 0.001,
            seed=args.seed + 1,
            generated_at=generated_at,
            robot_bounding_box=constants["SO101_BOUNDING_BOX"],
        )
        regenerated["trial_id"] = layout["trial_id"]
        if "episode_index" in layout:
            regenerated["episode_index"] = layout["episode_index"]

        task.clear()
        task.update(updated)
        layout.clear()
        layout.update(regenerated)
        changed.append((int(updated["trial_id"]), swaps))

    if not changed:
        print("No rows contained a retired object; files unchanged.")
        return

    for trial_id, swaps in changed:
        detail = ", ".join(f"{old!r} -> {new!r}" for old, new in swaps)
        print(f"trial {trial_id}: {detail}")

    if args.dry_run:
        print(f"Dry run: {len(changed)} row(s) would change.")
        return

    for path in (args.episodes_jsonl, args.episode_layouts_jsonl):
        backup = path.with_name(path.name + args.backup_suffix)
        if backup.exists():
            raise FileExistsError(f"Refusing to overwrite existing backup: {backup}")
        backup.write_bytes(path.read_bytes())

    write_jsonl_atomic(args.episodes_jsonl, tasks)
    write_jsonl_atomic(args.episode_layouts_jsonl, layouts)
    print(f"Rewrote {len(changed)} row(s) and their layouts.")


if __name__ == "__main__":
    main()
