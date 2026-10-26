#!/usr/bin/env python3
"""Generate the public SO-101 Bench validation set: a small, balanced episode set for development.

Use it to select checkpoints and tune settings; the leaderboard is computed on the private
primary evaluation, which shares the tasks, splits and scoring but uses held-out objects,
backgrounds and robot colours.

* 3 episodes in each of the 15 task-family x object-split cells (45 total), canonical appearance.
* Only public objects: the public part of each split (no held-out unseen-class objects).
* Spatial targets (next_to/between/move) are distinct within a split; one-object bins prefer
  objects that are not spatial targets; four-object-bin scenes never repeat an object pair;
  move directions are balanced.
* Layouts come from the benchmark's footprint-aware sampler.

    python scripts/create_val_set.py           # write tasks/val_set.jsonl + tasks/layouts/val_set_layouts.jsonl
    python scripts/create_val_set.py --check   # validate the saved files
"""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
import random

from task_generation import (
    ROOT, SPLITS, DIRECTIONS, env_cfg_constants, instruction_for_row, load_modules, order_rows, cell_key,
)

SEED = 20261026
TASKS = ROOT / "tasks/val_set.jsonl"
LAYOUTS = ROOT / "tasks/layouts/val_set_layouts.jsonl"
GENERATED_AT = "2026-10-26T00:00:00+00:00"
EPISODES_PER_CELL = 3
FAMILIES = ("bin_1", "bin_4", "next_to", "between", "move")
SPATIAL = ("next_to", "between", "move")


def public_objects(benchmark):
    """Public objects of each split, in catalog order."""

    public = {split: list(benchmark.OBJECT_SPLITS[split]) for split in SPLITS}
    public["unseen_unseen_class"] = [n for n in benchmark.PUBLIC_UNSEEN_UNSEEN_CLASS
                                     if n in benchmark.OBJECT_SPLITS["unseen_unseen_class"]]
    return public


def bin4_scenes(names, uses, rng):
    """EPISODES_PER_CELL four-object scenes with no repeated object pair, least-used objects first."""

    for _ in range(10000):
        local_uses, used_pairs, scenes = Counter(uses), set(), []
        for _scene in range(EPISODES_PER_CELL):
            scene = []
            for name in sorted(names, key=lambda n: (local_uses[n], rng.random())):
                if len(scene) == 4:
                    break
                if not any(tuple(sorted((name, o))) in used_pairs for o in scene):
                    scene.append(name)
            if len(scene) != 4:
                break
            used_pairs |= {tuple(sorted(p)) for p in combinations(scene, 2)}
            local_uses.update(scene)
            rng.shuffle(scene)
            scenes.append(scene)
        else:
            return scenes
    raise RuntimeError("Could not build pair-unique four-object bins")


def make_rows(benchmark, objects, attempt=0):
    rows = []
    direction_index = 0
    for split in SPLITS:
        names = objects[split]
        eligible = [n for n in names if "ALL" in benchmark.OBJECT_METADATA[n]["tasks"]]
        rng = random.Random(f"{SEED}:{split}:{attempt}")
        in_scene = Counter()
        spatial_targets = rng.sample(eligible, len(SPATIAL) * EPISODES_PER_CELL)
        for index, target in enumerate(spatial_targets):
            family = SPATIAL[index // EPISODES_PER_CELL]
            companions = sorted((n for n in eligible if n != target), key=lambda n: (in_scene[n], rng.random()))[:3]
            row = dict(ood_key=split, appearance_profile="canonical", repeat_index=0,
                       objects=[target, *companions], n_objects=4, target=target)
            rng.shuffle(row["objects"])
            if family == "move":
                row["direction"] = DIRECTIONS[direction_index % len(DIRECTIONS)]
                direction_index += 1
            else:
                row["referents"] = companions[: 1 if family == "next_to" else 2]
            row["instruction"] = instruction_for_row(row, family, set(benchmark.COLORS))
            rows.append(row)
            in_scene.update(row["objects"])
        # Prefer objects that were not spatial targets (the public unseen-class split is small).
        for name in sorted(names, key=lambda n: (n in spatial_targets, rng.random()))[:EPISODES_PER_CELL]:
            rows.append(dict(ood_key=split, appearance_profile="canonical", repeat_index=0, objects=[name],
                             n_objects=1, instruction="Place each object in the plastic bin"))
            in_scene[name] += 1
        for scene in bin4_scenes(names, in_scene, rng):
            rows.append(dict(ood_key=split, appearance_profile="canonical", repeat_index=0, objects=scene,
                             n_objects=4, anchor=scene[0], instruction="Place each object in the plastic bin"))
            in_scene.update(scene)
    return order_rows(rows, benchmark)


def validate(rows, layout_rows, benchmark, objects):
    counts, directions, specs = Counter(), Counter(), []
    for index, row in enumerate(rows):
        assert row["trial_id"] == index
        specs.append(benchmark.episode_spec_from_json(row))
        split, family = cell_key(benchmark, row)
        assert set(row["objects"]) <= set(objects[split]), (index, row["objects"])
        assert len(row["objects"]) == len(set(row["objects"]))
        normalized = benchmark.row_with_canonical_instruction_metadata(row)
        for field in ("target", "referents", "direction"):
            assert normalized.get(field) == row.get(field), (index, field)
        counts[split, family] += 1
        if family == "move":
            directions[row["direction"]] += 1
    assert set(counts) == {(s, f) for s in SPLITS for f in FAMILIES}
    assert set(counts.values()) == {EPISODES_PER_CELL}
    for split in SPLITS:
        spatial = [r["target"] for r in rows if r["ood_key"] == split and "target" in r]
        assert len(spatial) == len(set(spatial)), f"{split}: repeated spatial target"
        bin4 = [r["objects"] for r in rows if cell_key(benchmark, r) == (split, "bin_4")]
        pairs = [tuple(sorted(p)) for scene in bin4 for p in combinations(scene, 2)]
        assert len(pairs) == len(set(pairs)), f"{split}: repeated bin companion pair"
    assert max(directions.values()) - min(directions.values()) <= 1
    benchmark.validate_move_episode_footprints(specs)
    if layout_rows is not None:
        assert len(layout_rows) == len(rows)
        for index, (row, layout) in enumerate(zip(rows, layout_rows)):
            assert layout["trial_id"] == layout["episode_index"] == index
            assert layout["instruction"] == row["instruction"]
            assert [obj["name"] for obj in layout["objects"]] == row["objects"]
    seconds = sum(benchmark.episode_length_s(s.task_family, len(s.objects)) for s in specs)
    return dict(episodes=len(rows), sim_seconds_at_timeout=seconds,
                cells={f"{s}/{f}": n for (s, f), n in sorted(counts.items())}, move_directions=dict(directions))


def make_layouts(rows, benchmark, layouts):
    constants = env_cfg_constants()
    bin_poses, regions = constants["BIN_RANDOM_POSES"], constants["VALID_OBJECT_SPAWN_REGIONS"]
    layout_rows, history, pose_use = [], [], Counter()
    for index, row in enumerate(rows):
        _, family = cell_key(benchmark, row)
        kwargs = dict(bin_random_poses=bin_poses, valid_spawn_regions=regions)
        pose_index = None
        if family == "bin_4":
            rng = random.Random(f"{SEED}:binpose:{index}")
            pose_index = min(range(len(bin_poses)), key=lambda p: (pose_use[p], rng.random()))
            kwargs = dict(bin_random_poses=(bin_poses[pose_index],), valid_spawn_regions=[regions[pose_index]])
        layout = layouts.generate_episode_layout(
            benchmark.episode_spec_from_json(row), episode_index=index,
            rng=random.Random(f"{SEED}:layout:{index}"),
            table_object_z=constants["TABLE_OBJECT_Z"], seed=SEED, generated_at=GENERATED_AT,
            robot_bounding_box=constants["SO101_BOUNDING_BOX"], placement_history=history, **kwargs,
        )
        if pose_index is not None:
            layout["bin"]["pose_index"] = pose_index
            layout["placement"]["valid_spawn_region_index"] = pose_index
            pose_use[pose_index] += 1
        layout_rows.append(layout)
    return layout_rows


def encode(rows):
    return "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Validate the saved task and layout files.")
    args = parser.parse_args()
    benchmark, layouts = load_modules()
    objects = public_objects(benchmark)
    if args.check:
        rows = [json.loads(line) for line in TASKS.read_text().splitlines()]
        layout_rows = [json.loads(line) for line in LAYOUTS.read_text().splitlines()]
    else:
        if TASKS.exists() or LAYOUTS.exists():
            raise FileExistsError("val_set files exist; use --check (delete them to regenerate)")
        for attempt in range(100):
            rows = make_rows(benchmark, objects, attempt)
            try:
                validate(rows, None, benchmark, objects)
                break
            except (AssertionError, FileNotFoundError, KeyError, ValueError) as exc:
                print(f"[plan attempt {attempt}] rejected: {exc}")
        else:
            raise RuntimeError("No valid plan found")
        layout_rows = make_layouts(rows, benchmark, layouts)
        TASKS.write_text(encode(rows))
        LAYOUTS.write_text(encode(layout_rows))
    print(json.dumps(validate(rows, layout_rows, benchmark, objects), indent=2))


if __name__ == "__main__":
    main()
