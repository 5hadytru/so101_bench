#!/usr/bin/env python3
"""Generate the teleop_1 benchmark episode JSONL file.

The generator reads the active ``OBJECT_SPLITS["seen"]`` entries from
``benchmark.py``. Objects that are commented out there are therefore excluded.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
class EpisodePlan:
    task_family: str
    objects: tuple[str, ...]
    target: str | None = None
    referents: tuple[str, ...] = ()
    direction: str | None = None

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


def _cyclic_support_groups(
    seen_objects: tuple[str, ...],
    target: str,
    *,
    group_count: int,
    group_size: int,
    rng: random.Random,
) -> list[tuple[str, ...]]:
    if len(seen_objects) <= group_size:
        raise ValueError(
            f"Need at least {group_size + 1} seen objects to sample {group_size} non-target objects per episode."
        )

    target_index = seen_objects.index(target)
    offset_count = len(seen_objects) - 1
    support_count = group_count * group_size
    offsets = [(index % offset_count) + 1 for index in range(support_count)]
    groups = [
        tuple(
            seen_objects[(target_index + offset) % len(seen_objects)]
            for offset in offsets[start : start + group_size]
        )
        for start in range(0, support_count, group_size)
    ]
    rng.shuffle(groups)
    return groups


def _choose_balanced(
    candidates: tuple[str, ...],
    count: int,
    appearances: Counter[str],
    rng: random.Random,
) -> tuple[str, ...]:
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


def _make_plans(seen_objects: tuple[str, ...], rng: random.Random) -> list[EpisodePlan]:
    plans: list[EpisodePlan] = []
    next_to_referent_appearances: Counter[str] = Counter()
    between_referent_appearances: Counter[str] = Counter()

    for target in seen_objects:
        for _ in range(5):
            plans.append(EpisodePlan(TASK_BIN, (target,)))

        for support_group in _cyclic_support_groups(
            seen_objects,
            target,
            group_count=4,
            group_size=3,
            rng=rng,
        ):
            plans.append(EpisodePlan(TASK_BIN, _shuffle_tuple((target, *support_group), rng)))

        support_groups = _cyclic_support_groups(
            seen_objects,
            target,
            group_count=16,
            group_size=3,
            rng=rng,
        )

        for support_group in support_groups[:6]:
            referent = _choose_balanced(support_group, 1, next_to_referent_appearances, rng)
            plans.append(
                EpisodePlan(
                    TASK_NEXT_TO,
                    _shuffle_tuple((target, *support_group), rng),
                    target=target,
                    referents=referent,
                )
            )

        for support_group in support_groups[6:12]:
            referents = _choose_balanced(support_group, 2, between_referent_appearances, rng)
            plans.append(
                EpisodePlan(
                    TASK_BETWEEN,
                    _shuffle_tuple((target, *support_group), rng),
                    target=target,
                    referents=referents,
                )
            )

        for direction, support_group in zip(DIRECTIONS, support_groups[12:], strict=True):
            plans.append(
                EpisodePlan(
                    TASK_MOVE,
                    _shuffle_tuple((target, *support_group), rng),
                    target=target,
                    direction=direction,
                )
            )

    return plans


def _colorless_label(object_name: str) -> str:
    words = object_name.split()
    if words and words[0] in COLORS:
        return " ".join(words[1:])
    return object_name


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


def _omitted_color_positions(
    plans: list[EpisodePlan],
    rng: random.Random,
    omission_rate: float,
) -> tuple[set[tuple[int, int]], dict[str, dict[str, int]]]:
    eligible_by_object: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for plan_index, plan in enumerate(plans):
        for mention_index, object_name in enumerate(plan.instruction_objects()):
            if _can_omit_color(object_name, plan.objects):
                eligible_by_object[object_name].append((plan_index, mention_index))

    omitted_positions: set[tuple[int, int]] = set()
    stats: dict[str, dict[str, int]] = {}
    for object_name, positions in eligible_by_object.items():
        shuffled_positions = list(positions)
        rng.shuffle(shuffled_positions)
        omit_count = int(len(shuffled_positions) * omission_rate + 0.5)
        omitted_positions.update(shuffled_positions[:omit_count])
        stats[object_name] = {"eligible": len(shuffled_positions), "omitted": omit_count}

    return omitted_positions, stats


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
    omitted_positions: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial_id, plan in enumerate(plans):
        labels = tuple(
            _colorless_label(object_name) if (trial_id, mention_index) in omitted_positions else object_name
            for mention_index, object_name in enumerate(plan.instruction_objects())
        )
        row: dict[str, Any] = {
            "objects": list(plan.objects),
            "ood_key": "seen",
            "trial_id": trial_id,
            "n_objects": len(plan.objects),
            "instruction": _instruction(plan, labels),
        }
        if plan.target is not None:
            row["target"] = plan.target
        if plan.referents:
            row["referents"] = list(plan.referents)
        if plan.direction is not None:
            row["direction"] = _canonical_direction(plan.direction)
        rows.append(row)
    return rows


def _validate_rows(rows: list[dict[str, Any]], benchmark: ModuleType) -> None:
    for row_index, row in enumerate(rows):
        benchmark.episode_spec_from_json(row, source=f"generated row {row_index}")


def _support_appearances(plans: list[EpisodePlan], task_families: set[str]) -> Counter[str]:
    appearances: Counter[str] = Counter()
    for plan in plans:
        if plan.task_family not in task_families or plan.target is None:
            continue
        appearances.update(object_name for object_name in plan.objects if object_name != plan.target)
    return appearances


def _bin_object_appearances(plans: list[EpisodePlan]) -> Counter[str]:
    appearances: Counter[str] = Counter()
    for plan in plans:
        if plan.task_family == TASK_BIN and len(plan.objects) == 4:
            appearances.update(plan.objects)
    return appearances


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tasks/teleop_1.jsonl"),
        help="JSONL file to write. Defaults to tasks/teleop_1.jsonl.",
    )
    parser.add_argument("--seed", type=int, default=101, help="Deterministic seed for shuffling choices.")
    parser.add_argument(
        "--color-omission-rate",
        type=float,
        default=0.5,
        help="Fraction of safe, color-bearing instruction mentions to shorten.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing the JSONL file.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 <= args.color_omission_rate <= 1.0:
        raise ValueError("--color-omission-rate must be between 0 and 1.")

    repo_root = _repo_root()
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    benchmark = _load_benchmark(repo_root)
    seen_objects = tuple(benchmark.OBJECT_SPLITS["seen"])
    rng = random.Random(args.seed)

    plans = _make_plans(seen_objects, rng)
    omitted_positions, omission_stats = _omitted_color_positions(plans, rng, args.color_omission_rate)
    rows = _rows_for_plans(plans, omitted_positions)
    _validate_rows(rows, benchmark)

    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as jsonl_file:
            for row in rows:
                jsonl_file.write(json.dumps(row) + "\n")

    support_counts = _support_appearances(plans, {TASK_NEXT_TO, TASK_BETWEEN, TASK_MOVE})
    bin_counts = _bin_object_appearances(plans)
    eligible_mentions = sum(stats["eligible"] for stats in omission_stats.values())
    omitted_mentions = sum(stats["omitted"] for stats in omission_stats.values())
    destination = output_path if not args.dry_run else f"{output_path} (dry run)"
    print(f"Prepared {len(rows)} episodes for {destination}.")
    print(
        "Non-target appearances in next-to/between/move episodes: "
        f"min={min(support_counts.values())}, max={max(support_counts.values())}."
    )
    print(
        "Four-object bin object appearances: "
        f"min={min(bin_counts.values())}, max={max(bin_counts.values())}."
    )
    print(
        "Safe color omissions: "
        f"{omitted_mentions}/{eligible_mentions} eligible mentions "
        f"({omitted_mentions / eligible_mentions:.1%})."
    )


if __name__ == "__main__":
    main()
