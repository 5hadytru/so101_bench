#!/usr/bin/env python3
"""Generate mmrv_eval: the minimal sim/real correspondence (MMRV) task set.

Every episode uses only objects that exist physically next to the real SO-101, with the
canonical appearance (no domain randomization), so each one can be staged in real with
matched object/bin poses. Design:

* 3 episodes in each of the 15 task-family x object-split cells (45 total). Three is the
  minimum per cell for which every owned object is a manipulated object at least once
  (the seen split has 18 objects and 24 target slots; two per cell gives only 16).
* Spatial targets (next_to/between/move) are distinct within a split; one-object and
  four-object bins cover the remaining objects first, then the least-used ones.
* Four-object-bin scenes never repeat an object pair; companions/referents come from
  the same split and the least-used objects; move directions are balanced.
* Layouts come from the same footprint-aware sampler as primary_eval_v2, as a starting
  point for hand editing with scripts/edit_layouts.py.

    ~/env_isaaclab_51/bin/python scripts/create_mmrv_eval.py           # write tasks + layouts
    ~/env_isaaclab_51/bin/python scripts/create_mmrv_eval.py --check   # validate saved files
"""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
import random

from task_generation import (
    ROOT, SPLITS, DIRECTIONS, env_cfg_constants, instruction_for_row, load_modules, order_rows, cell_key,
)

SEED = 20261014
TASKS = ROOT / "tasks/mmrv_eval.jsonl"
LAYOUTS = ROOT / "tasks/layouts/mmrv_eval_layouts.jsonl"
GENERATED_AT = "2026-10-14T00:00:00+00:00"
EPISODES_PER_CELL = 3
FAMILIES = ("bin_1", "bin_4", "next_to", "between", "move")
SPATIAL = ("next_to", "between", "move")
# Physical objects available next to the real robot (2026-10-14).
REAL_OBJECTS = {
    "seen": "white pen, black pen, altoids container, blue pliers, green clip, pink eraser, yellow wires, "
    "red tape, black tape, cardboard box, flower pot, yellow toy car, "
    "grey toy car, green shoes, blue bowl, blue scissors",
    "unseen_seen_class": "blue clip, blue tape, yellow tape, blue screwdriver, white bowl, black wires, brown wires, "
    "orange toy car, blue pen, red pen, white shoes",
    "unseen_unseen_class": "blue highlighter, purple toothbrush, blue controller, action figure, razor, silver tongs, "
    "playing cards, toy fire truck, toy monster truck, toy dinosaur",  # retired via scripts/replace_mmrv_object.py: candy bar 2026-10-16, yellow screwdriver 2026-10-19, cooking spoon 2026-10-21
}
REAL_OBJECTS = {split: sorted(names.split(", ")) for split, names in REAL_OBJECTS.items()}
# Hand edits the user chose over the generator's coverage goals (validate allows exactly these).
USER_OVERRIDES = {
    "never_manipulated": {"grey toy car"},  # 2026-10-20: trial 7 target grey -> yellow toy car
    "repeated_spatial_target": {"yellow toy car"},  # trials 7 and 14
}


def _least_used(candidates, uses, rng, k):
    """k distinct candidates with the fewest uses so far, ties broken randomly."""

    keyed = sorted(candidates, key=lambda name: (uses[name], rng.random()))
    return keyed[:k]


def bin4_scenes(names, must_cover, uses, rng):
    """EPISODES_PER_CELL pair-unique scenes containing every object in must_cover."""

    for _ in range(10000):
        cover = list(must_cover)
        rng.shuffle(cover)
        # Spread the must-cover objects evenly, then fill each scene with the least-used
        # objects that form no pair already used in an earlier scene.
        seeds = [cover[i::EPISODES_PER_CELL] for i in range(EPISODES_PER_CELL)]
        local_uses, used_pairs, scenes = Counter(uses), set(), []
        for scene in seeds:
            scene = list(scene)
            for name in sorted(names, key=lambda n: (local_uses[n], rng.random())):
                if len(scene) == 4:
                    break
                if name not in scene and not any(tuple(sorted((name, o))) in used_pairs for o in scene):
                    scene.append(name)
            pairs = {tuple(sorted(p)) for p in combinations(scene, 2)}
            if len(scene) != 4 or pairs & used_pairs:
                break
            used_pairs |= pairs
            local_uses.update(scene)
            rng.shuffle(scene)
            scenes.append(scene)
        else:
            return scenes
    raise RuntimeError("Could not build pair-unique four-object bins")


def make_rows(benchmark, attempt=0):
    rows = []
    direction_index = 0
    for split in SPLITS:
        names = REAL_OBJECTS[split]
        eligible = [n for n in names if "ALL" in benchmark.OBJECT_METADATA[n]["tasks"]]
        rng = random.Random(f"{SEED}:{split}:{attempt}")
        targets = Counter()  # times each object is a manipulated object
        in_scene = Counter()  # times each object is on the table at all

        spatial_targets = eligible.copy()
        rng.shuffle(spatial_targets)
        spatial_targets = spatial_targets[: len(SPATIAL) * EPISODES_PER_CELL]
        for index, target in enumerate(spatial_targets):
            family = SPATIAL[index // EPISODES_PER_CELL]
            companions = _least_used([n for n in eligible if n != target], in_scene, rng, 3)
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
            targets[target] += 1
            in_scene.update(row["objects"])

        uncovered = [n for n in names if not targets[n]]
        rng.shuffle(uncovered)
        bin1 = uncovered[:EPISODES_PER_CELL]
        bin1 += _least_used([n for n in names if n not in bin1], targets, rng, EPISODES_PER_CELL - len(bin1))
        for name in bin1:
            rows.append(dict(ood_key=split, appearance_profile="canonical", repeat_index=0, objects=[name],
                             n_objects=1, instruction="Place each object in the plastic bin"))
            targets[name] += 1
            in_scene[name] += 1

        for scene in bin4_scenes(names, [n for n in names if not targets[n]], targets, rng):
            rows.append(dict(ood_key=split, appearance_profile="canonical", repeat_index=0, objects=scene,
                             n_objects=4, anchor=scene[0], instruction="Place each object in the plastic bin"))
            targets.update(scene)
            in_scene.update(scene)
    return order_rows(rows, benchmark)


def validate(rows, layout_rows, benchmark):
    counts, targets, directions = Counter(), defaultdict(Counter), Counter()
    specs = []
    for index, row in enumerate(rows):
        assert row["trial_id"] == index
        specs.append(benchmark.episode_spec_from_json(row))
        split, family = cell_key(benchmark, row)
        assert set(row["objects"]) <= set(REAL_OBJECTS[split]), (index, row["objects"])
        assert len(row["objects"]) == len(set(row["objects"]))
        assert row["appearance_profile"] == "canonical" and "domain_randomization" not in row
        normalized = benchmark.row_with_canonical_instruction_metadata(row)
        for field in ("target", "referents", "direction"):
            assert normalized.get(field) == row.get(field), (index, field)
        counts[split, family] += 1
        targets[split].update(row["objects"] if family.startswith("bin_") else [row["target"]])
        if family == "move":
            directions[row["direction"]] += 1
    assert set(counts) == {(s, f) for s in SPLITS for f in FAMILIES}
    assert set(counts.values()) == {EPISODES_PER_CELL}
    for split in SPLITS:
        missing = set(REAL_OBJECTS[split]) - set(targets[split]) - USER_OVERRIDES["never_manipulated"]
        assert not missing, f"{split}: object never manipulated: {sorted(missing)}"
        spatial = [r["target"] for r in rows if r["ood_key"] == split and "target" in r]
        repeated = {n for n in spatial if spatial.count(n) > 1} - USER_OVERRIDES["repeated_spatial_target"]
        assert not repeated, f"{split}: repeated spatial target: {sorted(repeated)}"
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
    return dict(
        episodes=len(rows),
        sim_seconds_at_timeout=seconds,
        cells={f"{s}/{f}": n for (s, f), n in sorted(counts.items())},
        manipulations_per_object={s: dict(sorted(c.items())) for s, c in targets.items()},
        move_directions=dict(directions),
    )


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
    for split, names in REAL_OBJECTS.items():
        assert set(names) <= set(benchmark.OBJECT_SPLITS[split]), f"{split}: object not in this split"
    if args.check:
        rows = [json.loads(line) for line in TASKS.read_text().splitlines()]
        layout_rows = [json.loads(line) for line in LAYOUTS.read_text().splitlines()]
    else:
        if TASKS.exists() or LAYOUTS.exists():
            raise FileExistsError("mmrv_eval files exist (layouts may be hand-edited); use --check")
        for attempt in range(100):
            rows = make_rows(benchmark, attempt)
            try:
                validate(rows, None, benchmark)
                break
            except (AssertionError, FileNotFoundError, KeyError, ValueError) as exc:
                print(f"[plan attempt {attempt}] rejected: {exc}")
        else:
            raise RuntimeError("No valid plan found")
        layout_rows = make_layouts(rows, benchmark, layouts)
        TASKS.write_text(encode(rows))
        LAYOUTS.write_text(encode(layout_rows))
    print(json.dumps(validate(rows, layout_rows, benchmark), indent=2))


if __name__ == "__main__":
    main()
