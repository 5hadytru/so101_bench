#!/usr/bin/env python3
"""Generate the public SO-101 Bench validation set: a small, balanced episode set for development.

Use it to select checkpoints and tune settings; the leaderboard is computed on the private
primary evaluation, which shares the tasks, splits and scoring but uses held-out objects,
backgrounds and robot colours.

* 135 episodes over the 15 task-family x object-split cells: 10 per one-object bin, next_to,
  between and move cell, 5 per four-object bin cell (those run 75 s). The public unseen-class split
  has 10 objects, so each of them is the target once per family; more episodes would only repeat
  targets.
* Only public objects: the public part of each split (no held-out unseen-class objects).
* Half the episodes of every cell use the canonical scene, half a varied one: a training
  background (2-18) and a non-orange training robot colour, with no background repeated in a cell
  and no background/colour pair repeated anywhere. The held-out ROBUST visuals are not used.
* Targets are distinct within a cell and, where a split has enough objects, across its spatial
  families; one-object bins prefer objects that are not spatial targets; four-object-bin scenes
  never repeat an object pair; move directions are balanced in every cell.
* Layouts come from the benchmark's footprint-aware sampler; the released ones were then hand-tuned
  with scripts/edit_layouts.py, so regenerating replaces them (use --check to validate edits).

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
CELL_SIZES = {"bin_1": 10, "bin_4": 5, "next_to": 10, "between": 10, "move": 10}
FAMILIES = tuple(CELL_SIZES)
SPATIAL = ("next_to", "between", "move")
VARIED_BACKGROUNDS = tuple(float(i) for i in range(2, 19))
# The training palette (mdp.resets.ROBOT_COLORS) without the canonical orange.
VARIED_ROBOT_COLORS = {
    "red": (0.64, 0.012, 0.03),
    "turquoise": (0.15, 0.60, 0.69),
    "yellow": (0.84, 0.69, 0.07),
    "white": (0.95, 0.95, 0.95),
    "black": (0.08, 0.08, 0.08),
}


def public_objects(benchmark):
    """Public objects of each split, in catalog order."""

    public = {split: list(benchmark.OBJECT_SPLITS[split]) for split in SPLITS}
    public["unseen_unseen_class"] = [n for n in benchmark.PUBLIC_UNSEEN_UNSEEN_CLASS
                                     if n in benchmark.OBJECT_SPLITS["unseen_unseen_class"]]
    return public


def bin4_scenes(names, uses, rng):
    """Four-object scenes with no repeated object pair, least-used objects first."""

    for _ in range(10000):
        local_uses, used_pairs, scenes = Counter(uses), set(), []
        for _scene in range(CELL_SIZES["bin_4"]):
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
        in_scene, as_target = Counter(), Counter()
        for family in SPATIAL:
            targets = sorted(eligible, key=lambda n: (as_target[n], rng.random()))[:CELL_SIZES[family]]
            as_target.update(targets)
            for target in targets:
                companions = sorted((n for n in eligible if n != target),
                                    key=lambda n: (in_scene[n], rng.random()))[:3]
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
        for name in sorted(names, key=lambda n: (as_target[n], rng.random()))[:CELL_SIZES["bin_1"]]:
            rows.append(dict(ood_key=split, appearance_profile="canonical", repeat_index=0, objects=[name],
                             n_objects=1, instruction="Place each object in the plastic bin"))
            in_scene[name] += 1
        for scene in bin4_scenes(names, in_scene, rng):
            rows.append(dict(ood_key=split, appearance_profile="canonical", repeat_index=0, objects=scene,
                             n_objects=4, anchor=scene[0], instruction="Place each object in the plastic bin"))
            in_scene.update(scene)
    rows = order_rows(rows, benchmark)
    assign_appearances(rows, benchmark, attempt)
    return rows


def assign_appearances(rows, benchmark, attempt):
    """Make half of every cell varied; odd cells alternate which profile gets the extra episode.

    Varied rows walk the background and colour cycles together; 17 backgrounds and 5 colours are
    coprime, so the first 85 varied rows all get distinct pairs and a cell's backgrounds differ.
    """

    rng = random.Random(f"{SEED}:appearance:{attempt}")
    backgrounds, colors = list(VARIED_BACKGROUNDS), list(VARIED_ROBOT_COLORS.values())
    rng.shuffle(backgrounds)
    rng.shuffle(colors)
    cells = defaultdict(list)
    for row in rows:
        cells[cell_key(benchmark, row)].append(row)
    varied_index = 0
    for index, (cell, cell_rows) in enumerate(sorted(cells.items())):
        picks = sorted(range(len(cell_rows)), key=lambda _: rng.random())
        n_varied = (len(cell_rows) + index % 2) // 2
        for position in sorted(picks[:n_varied]):
            row = cell_rows[position]
            color = colors[varied_index % len(colors)]
            row["appearance_profile"] = "varied"
            row["domain_randomization"] = dict(
                background_mesh_index=backgrounds[varied_index % len(backgrounds)],
                robot_color_r=color[0], robot_color_g=color[1], robot_color_b=color[2])
            varied_index += 1


def validate(rows, layout_rows, benchmark, objects):
    counts, directions, specs = Counter(), Counter(), []
    cell_directions, profiles, cell_backgrounds = defaultdict(Counter), defaultdict(Counter), defaultdict(list)
    visual_pairs = []
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
        profiles[split, family][row["appearance_profile"]] += 1
        if row["appearance_profile"] == "varied":
            dr = row["domain_randomization"]
            color = (dr["robot_color_r"], dr["robot_color_g"], dr["robot_color_b"])
            assert dr["background_mesh_index"] in VARIED_BACKGROUNDS, index
            assert color in VARIED_ROBOT_COLORS.values(), index
            cell_backgrounds[split, family].append(dr["background_mesh_index"])
            visual_pairs.append((dr["background_mesh_index"], color))
        else:
            assert row["appearance_profile"] == "canonical" and "domain_randomization" not in row, index
        if family == "move":
            directions[row["direction"]] += 1
            cell_directions[split][row["direction"]] += 1
    assert set(counts) == {(s, f) for s in SPLITS for f in FAMILIES}
    assert all(n == CELL_SIZES[f] for (_, f), n in counts.items())
    for cell, profile in profiles.items():
        assert abs(profile["canonical"] - profile["varied"]) <= 1, cell
    for cell, backgrounds in cell_backgrounds.items():
        assert len(backgrounds) == len(set(backgrounds)), f"{cell}: repeated background"
    assert len(visual_pairs) == len(set(visual_pairs)), "repeated background/colour pair"
    for split in SPLITS:
        eligible = [n for n in objects[split] if "ALL" in benchmark.OBJECT_METADATA[n]["tasks"]]
        for family in SPATIAL:
            targets = [r["target"] for r in rows if cell_key(benchmark, r) == (split, family)]
            assert len(targets) == len(set(targets)), f"{split}/{family}: repeated target"
        spatial = [r["target"] for r in rows if r["ood_key"] == split and "target" in r]
        if len(eligible) >= len(spatial):
            assert len(spatial) == len(set(spatial)), f"{split}: repeated spatial target"
        moves = cell_directions[split]
        assert max(moves.values()) - min(moves.values()) <= 1, f"{split}: unbalanced move directions"
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
    scenes = Counter(name for row in rows for name in row["objects"])
    return dict(episodes=len(rows), sim_seconds_at_timeout=seconds,
                profiles=dict(Counter(row["appearance_profile"] for row in rows)),
                cells={f"{s}/{f}": n for (s, f), n in sorted(counts.items())}, move_directions=dict(directions),
                scenes_per_object={split: [min(scenes[n] for n in objects[split]), max(scenes[n] for n in objects[split])]
                                   for split in SPLITS})


def make_layouts(rows, benchmark, layouts):
    constants = env_cfg_constants()
    bin_poses, regions = constants["BIN_RANDOM_POSES"], constants["VALID_OBJECT_SPAWN_REGIONS"]
    layout_rows, history, pose_use = [], [], Counter()
    for index, row in enumerate(rows):
        _, family = cell_key(benchmark, row)
        candidates = [(None, dict(bin_random_poses=bin_poses, valid_spawn_regions=regions))]
        if family == "bin_4":
            # Least-used bin pose first; fall back to the next when the scene does not fit its region.
            rng = random.Random(f"{SEED}:binpose:{index}")
            order = sorted(range(len(bin_poses)), key=lambda p: (pose_use[p], rng.random()))
            candidates = [(p, dict(bin_random_poses=(bin_poses[p],), valid_spawn_regions=[regions[p]]))
                          for p in order]
        for candidate, (pose_index, kwargs) in enumerate(candidates):
            try:
                layout = layouts.generate_episode_layout(
                    benchmark.episode_spec_from_json(row), episode_index=index,
                    rng=random.Random(f"{SEED}:layout:{index}"),
                    table_object_z=constants["TABLE_OBJECT_Z"], seed=SEED, generated_at=GENERATED_AT,
                    robot_bounding_box=constants["SO101_BOUNDING_BOX"], placement_history=history, **kwargs,
                )
                break
            except layouts.LayoutGenerationError:
                if candidate == len(candidates) - 1:
                    raise
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
