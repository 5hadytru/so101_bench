#!/usr/bin/env python3
"""Generate the teleop_2 benchmark episode JSONL file.

This is an additive catch-up dataset for the first sim teleop set.  It derives
which seen objects were previously grasped from ``tasks/teleop_1_edited.jsonl``
and assigns larger quotas to the remaining active seen objects.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import ModuleType
from typing import Any


TASK_BIN = "bin"
TASK_NEXT_TO = "next_to"
TASK_BETWEEN = "between"
TASK_MOVE = "move"

COLORS = {
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
}
DIRECTIONS = ("left", "right", "forwards", "backwards")


@dataclass(frozen=True)
class Quota:
    bin_1obj: int
    next_to: int
    between: int
    move: int
    bin_4obj: int


COVERED_QUOTA = Quota(bin_1obj=0, next_to=2, between=6, move=6, bin_4obj=4)
UNCOVERED_QUOTA = Quota(bin_1obj=5, next_to=6, between=10, move=10, bin_4obj=6)
TASK_SORT_ORDER = {
    "bin_1obj": 0,
    "bin_4obj": 1,
    TASK_NEXT_TO: 2,
    TASK_BETWEEN: 3,
    TASK_MOVE: 4,
}


@dataclass(frozen=True)
class EpisodePlan:
    task_family: str
    objects: tuple[str, ...]
    target: str | None = None
    referents: tuple[str, ...] = ()
    direction: str | None = None
    requested_distractor: bool = False
    has_distractor: bool = False
    anchor: str | None = None

    def instruction_objects(self) -> tuple[str, ...]:
        if self.task_family == TASK_NEXT_TO:
            assert self.target is not None and len(self.referents) == 1
            return (self.target, self.referents[0])
        if self.task_family == TASK_BETWEEN:
            assert self.target is not None and len(self.referents) == 2
            return (self.target, self.referents[0], self.referents[1])
        if self.task_family == TASK_MOVE:
            assert self.target is not None
            return (self.target,)
        return ()

    def sort_family(self) -> str:
        if self.task_family == TASK_BIN:
            return "bin_1obj" if len(self.objects) == 1 else "bin_4obj"
        return self.task_family


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_benchmark(repo_root: Path) -> ModuleType:
    benchmark_path = repo_root / "source" / "so101_bench" / "so101_bench" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("so101_bench_benchmark_standalone", benchmark_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark module from {benchmark_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical_direction(direction: str) -> str:
    return {"forwards": "forward", "backwards": "backward"}.get(direction, direction)


def _shuffle_tuple(items: tuple[str, ...], rng: random.Random) -> tuple[str, ...]:
    shuffled = list(items)
    rng.shuffle(shuffled)
    return tuple(shuffled)


def _colorless_label(object_name: str) -> str:
    words = object_name.split()
    if words and words[0] in COLORS:
        return " ".join(words[1:])
    return object_name


def _color_label(object_name: str) -> str | None:
    words = object_name.split()
    if words and words[0] in COLORS:
        return words[0]
    return None


def _can_omit_color(object_name: str, episode_objects: tuple[str, ...]) -> bool:
    colorless = _colorless_label(object_name)
    if colorless == object_name or not colorless:
        return False
    for other_name in episode_objects:
        if other_name == object_name:
            continue
        if other_name == colorless or _colorless_label(other_name) == colorless:
            return False
    return True


def _is_distractor_for(candidate: str, named_objects: tuple[str, ...]) -> bool:
    candidate_class = _colorless_label(candidate)
    candidate_color = _color_label(candidate)
    for object_name in named_objects:
        if candidate == object_name:
            continue
        if candidate_class == _colorless_label(object_name):
            return True
        if candidate_color is not None and candidate_color == _color_label(object_name):
            return True
    return False


def _choose_balanced(
    candidates: tuple[str, ...],
    count: int,
    appearances: Counter[str],
    rng: random.Random,
) -> tuple[str, ...]:
    if len(candidates) < count:
        raise ValueError(f"Need {count} candidate(s), got {len(candidates)}: {candidates}.")
    available = list(candidates)
    chosen: list[str] = []
    for _ in range(count):
        min_appearances = min(appearances[name] for name in available)
        tied = [name for name in available if appearances[name] == min_appearances]
        selected = rng.choice(tied)
        available.remove(selected)
        chosen.append(selected)
        appearances[selected] += 1
    return tuple(chosen)


def _choose_fillers(
    seen_objects: tuple[str, ...],
    base_objects: tuple[str, ...],
    *,
    count: int,
    named_objects: tuple[str, ...],
    prefer_distractor: bool,
    support_appearances: Counter[str],
    rng: random.Random,
) -> tuple[tuple[str, ...], bool]:
    chosen: list[str] = []
    excluded = set(base_objects)
    has_distractor = False

    if prefer_distractor:
        distractor_candidates = tuple(
            object_name
            for object_name in seen_objects
            if object_name not in excluded and _is_distractor_for(object_name, named_objects)
        )
        if distractor_candidates:
            distractor = _choose_balanced(distractor_candidates, 1, support_appearances, rng)[0]
            chosen.append(distractor)
            excluded.add(distractor)
            has_distractor = True

    remaining = count - len(chosen)
    if remaining:
        candidates = tuple(object_name for object_name in seen_objects if object_name not in excluded)
        chosen.extend(_choose_balanced(candidates, remaining, support_appearances, rng))

    if not has_distractor:
        has_distractor = any(_is_distractor_for(object_name, named_objects) for object_name in chosen)
    return tuple(chosen), has_distractor


def _direction_sequence(count: int, rng: random.Random) -> tuple[str, ...]:
    directions = list(DIRECTIONS) * (count // len(DIRECTIONS))
    directions.extend(rng.sample(DIRECTIONS, count % len(DIRECTIONS)))
    rng.shuffle(directions)
    return tuple(directions)


def _infer_task_family(row: dict[str, Any]) -> str:
    if row.get("n_objects") == 1 and "target" not in row:
        return "bin_1obj"
    if row.get("n_objects") == 4 and "target" not in row:
        return "bin_4obj"
    if "direction" in row:
        return TASK_MOVE
    if len(row.get("referents", [])) == 1:
        return TASK_NEXT_TO
    if len(row.get("referents", [])) == 2:
        return TASK_BETWEEN
    return "unknown"


def _covered_objects(previous_path: Path, seen_objects: tuple[str, ...]) -> set[str]:
    seen_set = set(seen_objects)
    covered: set[str] = set()
    with previous_path.open("r", encoding="utf-8") as jsonl_file:
        for line_no, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            task_family = _infer_task_family(row)
            if task_family == "bin_1obj":
                object_name = row["objects"][0]
            elif task_family in {TASK_NEXT_TO, TASK_BETWEEN, TASK_MOVE}:
                object_name = row["target"]
            else:
                continue
            if object_name in seen_set:
                covered.add(object_name)
            else:
                print(f"[WARN]: Ignoring non-seen covered object {object_name!r} on {previous_path}:{line_no}.")
    return covered


def _make_bin_4_plan(
    seen_objects: tuple[str, ...],
    anchor: str,
    support_appearances: Counter[str],
    rng: random.Random,
) -> EpisodePlan:
    supports = _choose_fillers(
        seen_objects,
        (anchor,),
        count=3,
        named_objects=(),
        prefer_distractor=False,
        support_appearances=support_appearances,
        rng=rng,
    )[0]
    return EpisodePlan(
        TASK_BIN,
        _shuffle_tuple((anchor, *supports), rng),
        anchor=anchor,
    )


def _make_next_to_plan(
    seen_objects: tuple[str, ...],
    target: str,
    referent_appearances: Counter[str],
    support_appearances: Counter[str],
    rng: random.Random,
    distractor_rate: float,
) -> EpisodePlan:
    referents = _choose_balanced(
        tuple(object_name for object_name in seen_objects if object_name != target),
        1,
        referent_appearances,
        rng,
    )
    named_objects = (target, referents[0])
    requested_distractor = rng.random() < distractor_rate
    fillers, has_distractor = _choose_fillers(
        seen_objects,
        named_objects,
        count=2,
        named_objects=named_objects,
        prefer_distractor=requested_distractor,
        support_appearances=support_appearances,
        rng=rng,
    )
    return EpisodePlan(
        TASK_NEXT_TO,
        _shuffle_tuple((target, *referents, *fillers), rng),
        target=target,
        referents=referents,
        requested_distractor=requested_distractor,
        has_distractor=has_distractor,
    )


def _make_between_plan(
    seen_objects: tuple[str, ...],
    target: str,
    referent_appearances: Counter[str],
    support_appearances: Counter[str],
    rng: random.Random,
    distractor_rate: float,
) -> EpisodePlan:
    referents = _choose_balanced(
        tuple(object_name for object_name in seen_objects if object_name != target),
        2,
        referent_appearances,
        rng,
    )
    named_objects = (target, *referents)
    requested_distractor = rng.random() < distractor_rate
    fillers, has_distractor = _choose_fillers(
        seen_objects,
        named_objects,
        count=1,
        named_objects=named_objects,
        prefer_distractor=requested_distractor,
        support_appearances=support_appearances,
        rng=rng,
    )
    return EpisodePlan(
        TASK_BETWEEN,
        _shuffle_tuple((target, *referents, *fillers), rng),
        target=target,
        referents=referents,
        requested_distractor=requested_distractor,
        has_distractor=has_distractor,
    )


def _make_move_plan(
    seen_objects: tuple[str, ...],
    target: str,
    direction: str,
    support_appearances: Counter[str],
    rng: random.Random,
    distractor_rate: float,
) -> EpisodePlan:
    named_objects = (target,)
    requested_distractor = rng.random() < distractor_rate
    fillers, has_distractor = _choose_fillers(
        seen_objects,
        named_objects,
        count=3,
        named_objects=named_objects,
        prefer_distractor=requested_distractor,
        support_appearances=support_appearances,
        rng=rng,
    )
    return EpisodePlan(
        TASK_MOVE,
        _shuffle_tuple((target, *fillers), rng),
        target=target,
        direction=direction,
        requested_distractor=requested_distractor,
        has_distractor=has_distractor,
    )


def _make_plans(
    seen_objects: tuple[str, ...],
    covered_objects: set[str],
    rng: random.Random,
    distractor_rate: float,
) -> list[EpisodePlan]:
    plans: list[EpisodePlan] = []
    next_to_referent_appearances: Counter[str] = Counter()
    between_referent_appearances: Counter[str] = Counter()
    support_appearances: Counter[str] = Counter()
    bin_4_support_appearances: Counter[str] = Counter()

    for target in seen_objects:
        quota = COVERED_QUOTA if target in covered_objects else UNCOVERED_QUOTA

        for _ in range(quota.bin_1obj):
            plans.append(EpisodePlan(TASK_BIN, (target,), anchor=target))

        for _ in range(quota.bin_4obj):
            plans.append(_make_bin_4_plan(seen_objects, target, bin_4_support_appearances, rng))

        for _ in range(quota.next_to):
            plans.append(
                _make_next_to_plan(
                    seen_objects,
                    target,
                    next_to_referent_appearances,
                    support_appearances,
                    rng,
                    distractor_rate,
                )
            )

        for _ in range(quota.between):
            plans.append(
                _make_between_plan(
                    seen_objects,
                    target,
                    between_referent_appearances,
                    support_appearances,
                    rng,
                    distractor_rate,
                )
            )

        for direction in _direction_sequence(quota.move, rng):
            plans.append(
                _make_move_plan(
                    seen_objects,
                    target,
                    direction,
                    support_appearances,
                    rng,
                    distractor_rate,
                )
            )

    object_order = {object_name: index for index, object_name in enumerate(seen_objects)}

    def sort_key(plan: EpisodePlan) -> tuple[int, int]:
        if plan.task_family == TASK_BIN:
            targetish = plan.anchor if plan.anchor is not None else plan.objects[0]
        else:
            assert plan.target is not None
            targetish = plan.target
        return (TASK_SORT_ORDER[plan.sort_family()], object_order[targetish])

    return sorted(plans, key=sort_key)


def _instruction(plan: EpisodePlan, labels: tuple[str, ...]) -> str:
    if plan.task_family == TASK_BIN:
        return "Place each object in the plastic bin"
    if plan.task_family == TASK_NEXT_TO:
        return f"Place the {labels[0]} next to the {labels[1]}"
    if plan.task_family == TASK_BETWEEN:
        return f"Place the {labels[0]} between the {labels[1]} and the {labels[2]}"
    if plan.task_family == TASK_MOVE:
        assert plan.direction is not None
        return f"Move the {labels[0]} {plan.direction}"
    raise ValueError(f"Unknown task family: {plan.task_family}")


def _rows_for_plans(
    plans: list[EpisodePlan],
    rng: random.Random,
    color_omission_rate: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    omission_stats: Counter[str] = Counter()
    for trial_id, plan in enumerate(plans):
        labels: list[str] = []
        for object_name in plan.instruction_objects():
            if _can_omit_color(object_name, plan.objects):
                omission_stats["eligible"] += 1
                if rng.random() < color_omission_rate:
                    labels.append(_colorless_label(object_name))
                    omission_stats["omitted"] += 1
                    continue
            labels.append(object_name)

        row: dict[str, Any] = {
            "objects": list(plan.objects),
            "ood_key": "seen",
            "trial_id": trial_id,
            "n_objects": len(plan.objects),
            "instruction": _instruction(plan, tuple(labels)),
        }
        if plan.target is not None:
            row["target"] = plan.target
        if plan.referents:
            row["referents"] = list(plan.referents)
        if plan.direction is not None:
            row["direction"] = _canonical_direction(plan.direction)
        rows.append(row)
    return rows, omission_stats


def _validate_rows(rows: list[dict[str, Any]], benchmark: ModuleType) -> None:
    episodes = [benchmark.episode_spec_from_json(row, source=f"generated row {row_index}") for row_index, row in enumerate(rows)]
    benchmark.validate_move_episode_footprints(episodes)


def _plan_task_counts(plans: list[EpisodePlan]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for plan in plans:
        if plan.task_family == TASK_BIN:
            counts["bin_1obj" if len(plan.objects) == 1 else "bin_4obj"] += 1
        else:
            counts[plan.task_family] += 1
    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tasks/teleop_2.jsonl"),
        help="JSONL file to write. Defaults to tasks/teleop_2.jsonl.",
    )
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path("tasks/teleop_1_edited.jsonl"),
        help="Previous sim task JSONL used to derive previously grasped seen objects.",
    )
    parser.add_argument("--seed", type=int, default=202, help="Deterministic seed for generation.")
    parser.add_argument(
        "--color-omission-rate",
        type=float,
        default=0.5,
        help="Independent chance to omit an eligible color mention.",
    )
    parser.add_argument(
        "--distractor-rate",
        type=float,
        default=0.35,
        help="Chance that a non-bin episode requests a color/class distractor filler.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing the JSONL file.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 <= args.color_omission_rate <= 1.0:
        raise ValueError("--color-omission-rate must be between 0 and 1.")
    if not 0.0 <= args.distractor_rate <= 1.0:
        raise ValueError("--distractor-rate must be between 0 and 1.")

    repo_root = _repo_root()
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    previous_path = args.previous if args.previous.is_absolute() else repo_root / args.previous
    benchmark = _load_benchmark(repo_root)
    seen_objects = tuple(benchmark.OBJECT_SPLITS["seen"])
    rng = random.Random(args.seed)

    covered_objects = _covered_objects(previous_path, seen_objects)
    uncovered_objects = tuple(object_name for object_name in seen_objects if object_name not in covered_objects)
    plans = _make_plans(seen_objects, covered_objects, rng, args.distractor_rate)
    rows, omission_stats = _rows_for_plans(plans, rng, args.color_omission_rate)
    _validate_rows(rows, benchmark)

    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as jsonl_file:
            for row in rows:
                jsonl_file.write(json.dumps(row) + "\n")

    task_counts = _plan_task_counts(plans)
    non_bin = [plan for plan in plans if plan.task_family != TASK_BIN]
    requested_distractors = sum(plan.requested_distractor for plan in non_bin)
    actual_distractors = sum(plan.has_distractor for plan in non_bin)
    destination = output_path if not args.dry_run else f"{output_path} (dry run)"
    print(f"Prepared {len(rows)} episodes for {destination}.")
    print(f"Previously covered seen objects ({len(covered_objects)}): {', '.join(sorted(covered_objects))}")
    print(f"Newly covered seen objects ({len(uncovered_objects)}): {', '.join(uncovered_objects)}")
    print("Task counts: " + ", ".join(f"{task}={task_counts[task]}" for task in ("bin_1obj", "bin_4obj", "next_to", "between", "move")))
    print(
        "Non-bin distractor fillers: "
        f"{actual_distractors}/{len(non_bin)} present ({actual_distractors / len(non_bin):.1%}); "
        f"{requested_distractors} explicitly requested."
    )
    eligible_mentions = omission_stats["eligible"]
    omitted_mentions = omission_stats["omitted"]
    print(
        "Safe color omissions: "
        f"{omitted_mentions}/{eligible_mentions} eligible mentions "
        f"({omitted_mentions / eligible_mentions:.1%})."
    )


if __name__ == "__main__":
    main()
