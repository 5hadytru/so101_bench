#!/usr/bin/env python3
"""Generate ``tasks/teleop_4.jsonl`` for seen objects absent from sim-6.

Every target object receives six one-object bin episodes, eight generic
four-object bin episodes, six named-bin episodes, eight next-to episodes, ten
between episodes, and eight directional move episodes.  Generic four-object
bin companions are independently drawn from the active legacy/new seen pools
with probabilities 0.1/0.9.  Other four-object scenes draw uniformly from the
complete active seen split.
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


PER_OBJECT_COUNTS = {
    "bin_1obj": 6,
    "bin_4obj": 8,
    "named_bin": 6,
    "next_to": 8,
    "between": 10,
    "move": 8,
}
DIRECTION_WORDS = ("left", "right", "forwards", "backwards")
LEGACY_FOUR_BIN_PROBABILITY = 0.1
EXPLICIT_COLOR_FRACTION = 0.2
EXPECTED_NEW_TARGETS = 31


@dataclass(frozen=True)
class Plan:
    family: str
    objects: tuple[str, ...]
    owner: str
    target: str | None = None
    referents: tuple[str, ...] = ()
    direction: str | None = None

    def named_objects(self) -> tuple[str, ...]:
        if self.family in {"named_bin", "move"}:
            assert self.target is not None
            return (self.target,)
        if self.family in {"next_to", "between"}:
            assert self.target is not None
            return (self.target, *self.referents)
        return ()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_benchmark(root: Path) -> ModuleType:
    path = root / "source/so101_bench/so101_bench/benchmark.py"
    spec = importlib.util.spec_from_file_location("so101_bench_benchmark_teleop4", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shuffled(values: tuple[str, ...], rng: random.Random) -> tuple[str, ...]:
    result = list(values)
    rng.shuffle(result)
    return tuple(result)


def colorless(name: str, colors: set[str]) -> str:
    words = name.split()
    return " ".join(words[1:]) if words and words[0] in colors else name


def can_omit_color(name: str, objects: tuple[str, ...], colors: set[str]) -> bool:
    shortened = colorless(name, colors)
    return shortened != name and all(
        other == name or colorless(other, colors) != shortened for other in objects
    )


def signature(plan: Plan) -> tuple[Any, ...]:
    referents = tuple(sorted(plan.referents)) if plan.family == "between" else plan.referents
    return (
        plan.family,
        tuple(sorted(plan.objects)),
        plan.target,
        referents,
        plan.direction,
    )


def random_seen_companions(
    target: str,
    seen_objects: tuple[str, ...],
    rng: random.Random,
) -> tuple[str, str, str]:
    candidates = [name for name in seen_objects if name != target]
    return tuple(rng.sample(candidates, 3))  # type: ignore[return-value]


def weighted_four_bin_companions(
    target: str,
    legacy_objects: tuple[str, ...],
    new_objects: tuple[str, ...],
    rng: random.Random,
) -> tuple[str, str, str]:
    chosen: list[str] = []
    for _ in range(3):
        pool = legacy_objects if rng.random() < LEGACY_FOUR_BIN_PROBABILITY else new_objects
        candidates = [name for name in pool if name != target and name not in chosen]
        if not candidates:
            candidates = [
                name
                for name in (*legacy_objects, *new_objects)
                if name != target and name not in chosen
            ]
        chosen.append(rng.choice(candidates))
    return tuple(chosen)  # type: ignore[return-value]


def add_unique_random_plan(
    plans: list[Plan],
    signatures: set[tuple[Any, ...]],
    make_plan: Any,
) -> None:
    for _ in range(10_000):
        plan = make_plan()
        key = signature(plan)
        if key not in signatures:
            plans.append(plan)
            signatures.add(key)
            return
    raise RuntimeError("Could not sample a unique task configuration after 10,000 attempts")


def make_plans(
    seen_objects: tuple[str, ...],
    legacy_objects: tuple[str, ...],
    new_objects: tuple[str, ...],
    rng: random.Random,
) -> list[Plan]:
    plans: list[Plan] = []
    signatures: set[tuple[Any, ...]] = set()

    for target in new_objects:
        for _ in range(PER_OBJECT_COUNTS["bin_1obj"]):
            plans.append(Plan("bin_1obj", (target,), owner=target))

        for _ in range(PER_OBJECT_COUNTS["bin_4obj"]):
            def make_four_bin(target: str = target) -> Plan:
                companions = weighted_four_bin_companions(
                    target, legacy_objects, new_objects, rng
                )
                return Plan(
                    "bin_4obj",
                    shuffled((target, *companions), rng),
                    owner=target,
                )

            add_unique_random_plan(plans, signatures, make_four_bin)

        for _ in range(PER_OBJECT_COUNTS["named_bin"]):
            def make_named_bin(target: str = target) -> Plan:
                companions = random_seen_companions(target, seen_objects, rng)
                return Plan(
                    "named_bin",
                    shuffled((target, *companions), rng),
                    owner=target,
                    target=target,
                )

            add_unique_random_plan(plans, signatures, make_named_bin)

        for _ in range(PER_OBJECT_COUNTS["next_to"]):
            def make_next_to(target: str = target) -> Plan:
                companions = random_seen_companions(target, seen_objects, rng)
                return Plan(
                    "next_to",
                    shuffled((target, *companions), rng),
                    owner=target,
                    target=target,
                    referents=(companions[0],),
                )

            add_unique_random_plan(plans, signatures, make_next_to)

        for _ in range(PER_OBJECT_COUNTS["between"]):
            def make_between(target: str = target) -> Plan:
                companions = random_seen_companions(target, seen_objects, rng)
                return Plan(
                    "between",
                    shuffled((target, *companions), rng),
                    owner=target,
                    target=target,
                    referents=companions[:2],
                )

            add_unique_random_plan(plans, signatures, make_between)

        directions = list(DIRECTION_WORDS) * 2
        rng.shuffle(directions)
        for direction in directions:
            def make_move(target: str = target, direction: str = direction) -> Plan:
                companions = random_seen_companions(target, seen_objects, rng)
                return Plan(
                    "move",
                    shuffled((target, *companions), rng),
                    owner=target,
                    target=target,
                    direction=direction,
                )

            add_unique_random_plan(plans, signatures, make_move)

    return plans


def choose_full_color_mentions(
    plans: list[Plan],
    colors: set[str],
    rng: random.Random,
) -> tuple[set[tuple[int, int]], dict[str, int]]:
    by_object: dict[str, list[tuple[int, int]]] = defaultdict(list)
    forced_by_object: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for plan_index, plan in enumerate(plans):
        for mention_index, name in enumerate(plan.named_objects()):
            if colorless(name, colors) == name:
                continue
            position = (plan_index, mention_index)
            by_object[name].append(position)
            if not can_omit_color(name, plan.objects, colors):
                forced_by_object[name].append(position)

    full_color: set[tuple[int, int]] = set()
    for name, positions in by_object.items():
        forced = forced_by_object[name]
        full_color.update(forced)
        desired = max(len(forced), round(EXPLICIT_COLOR_FRACTION * len(positions)))
        optional = [position for position in positions if position not in full_color]
        full_color.update(rng.sample(optional, desired - len(forced)))

    total = sum(len(positions) for positions in by_object.values())
    forced_total = sum(len(positions) for positions in forced_by_object.values())
    return full_color, {
        "colored_mentions": total,
        "full_color_mentions": len(full_color),
        "forced_full_color_mentions": forced_total,
    }


def plan_to_row(
    plan: Plan,
    trial_id: int,
    plan_index: int,
    full_color_mentions: set[tuple[int, int]],
    colors: set[str],
) -> dict[str, Any]:
    labels = [
        name if (plan_index, mention_index) in full_color_mentions else colorless(name, colors)
        for mention_index, name in enumerate(plan.named_objects())
    ]
    if plan.family in {"bin_1obj", "bin_4obj"}:
        instruction = "Place each object in the plastic bin"
    elif plan.family == "named_bin":
        instruction = f"Place the {labels[0]} in the plastic bin"
    elif plan.family == "next_to":
        instruction = f"Place the {labels[0]} next to the {labels[1]}"
    elif plan.family == "between":
        instruction = f"Place the {labels[0]} between the {labels[1]} and the {labels[2]}"
    else:
        instruction = f"Move the {labels[0]} {plan.direction}"

    row: dict[str, Any] = {
        "objects": list(plan.objects),
        "ood_key": "seen",
        "trial_id": trial_id,
        "n_objects": len(plan.objects),
        "instruction": instruction,
    }
    if plan.family == "bin_4obj":
        row["anchor"] = plan.owner
    if plan.target is not None:
        row["target"] = plan.target
    if plan.family == "named_bin":
        row["task_family"] = "named_bin"
    if plan.referents:
        row["referents"] = list(plan.referents)
    if plan.direction is not None:
        row["direction"] = {
            "forwards": "forward",
            "backwards": "backward",
        }.get(plan.direction, plan.direction)
    return row


def family(row: dict[str, Any]) -> str:
    if row.get("task_family") == "named_bin":
        return "named_bin"
    if "direction" in row:
        return "move"
    if len(row.get("referents", [])) == 1:
        return "next_to"
    if len(row.get("referents", [])) == 2:
        return "between"
    return "bin_1obj" if row["n_objects"] == 1 else "bin_4obj"


def validate(
    rows: list[dict[str, Any]],
    plans: list[Plan],
    benchmark: ModuleType,
    seen_objects: tuple[str, ...],
    legacy_objects: tuple[str, ...],
    new_objects: tuple[str, ...],
    color_stats: dict[str, int],
) -> dict[str, Any]:
    expected_rows = len(new_objects) * sum(PER_OBJECT_COUNTS.values())
    if len(rows) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, got {len(rows)}")
    if [row["trial_id"] for row in rows] != list(range(len(rows))):
        raise AssertionError("trial_id values are not contiguous")

    actual_families = Counter(family(row) for row in rows)
    expected_families = Counter(
        {name: count * len(new_objects) for name, count in PER_OBJECT_COUNTS.items()}
    )
    if actual_families != expected_families:
        raise AssertionError(f"Task family mismatch: {actual_families} != {expected_families}")

    owner_families: dict[str, Counter[str]] = defaultdict(Counter)
    direction_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for plan, row in zip(plans, rows, strict=True):
        owner_families[plan.owner][plan.family] += 1
        if plan.direction is not None:
            direction_counts[plan.owner][plan.direction] += 1
        if len(set(row["objects"])) != len(row["objects"]):
            raise AssertionError(f"Duplicate object in row {row['trial_id']}")
        if not set(row["objects"]) <= set(seen_objects):
            raise AssertionError(f"Non-seen object in row {row['trial_id']}")
        benchmark.episode_spec_from_json(row, source=f"generated row {row['trial_id']}")
        canonical = benchmark.row_with_canonical_instruction_metadata(
            row, source=f"generated row {row['trial_id']}"
        )
        for key in ("target", "referents", "direction"):
            if canonical.get(key) != row.get(key):
                raise AssertionError(
                    f"Canonical {key} mismatch in row {row['trial_id']}: "
                    f"{canonical.get(key)!r} != {row.get(key)!r}"
                )

    expected_owner_counts = Counter(PER_OBJECT_COUNTS)
    for name in new_objects:
        if owner_families[name] != expected_owner_counts:
            raise AssertionError(
                f"Per-object task counts differ for {name}: "
                f"{owner_families[name]} != {expected_owner_counts}"
            )
        if direction_counts[name] != Counter({direction: 2 for direction in DIRECTION_WORDS}):
            raise AssertionError(f"Move directions differ for {name}: {direction_counts[name]}")

    episode_specs = [
        benchmark.episode_spec_from_json(row, source=f"generated row {index}")
        for index, row in enumerate(rows)
    ]
    benchmark.validate_move_episode_footprints(episode_specs)

    legacy_set = set(legacy_objects)
    four_bin_companions = [
        name
        for plan in plans
        if plan.family == "bin_4obj"
        for name in plan.objects
        if name != plan.owner
    ]
    legacy_companions = sum(name in legacy_set for name in four_bin_companions)
    four_bin_rate = legacy_companions / len(four_bin_companions)
    return {
        "rows": len(rows),
        "targets": len(new_objects),
        "task_counts": dict(actual_families),
        "four_bin_companion_selections": len(four_bin_companions),
        "four_bin_legacy_companions": legacy_companions,
        "four_bin_legacy_fraction": four_bin_rate,
        **color_stats,
        "full_color_fraction": (
            color_stats["full_color_mentions"] / color_stats["colored_mentions"]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tasks/teleop_4.jsonl"))
    parser.add_argument(
        "--existing-tasks",
        type=Path,
        default=Path("tasks/so101_bench_sim_6_slowed_1.7x_phase_governed.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    benchmark = load_benchmark(root)
    seen_objects = tuple(benchmark.OBJECT_SPLITS["seen"])
    existing_path = args.existing_tasks if args.existing_tasks.is_absolute() else root / args.existing_tasks
    existing_objects = {
        name for row in read_jsonl(existing_path) for name in row["objects"]
    }
    legacy_objects = tuple(name for name in seen_objects if name in existing_objects)
    new_objects = tuple(name for name in seen_objects if name not in existing_objects)
    if len(new_objects) != EXPECTED_NEW_TARGETS:
        raise ValueError(
            f"Expected {EXPECTED_NEW_TARGETS} new seen targets, found {len(new_objects)}: "
            f"{list(new_objects)}"
        )
    invalid_targets = [
        name
        for name in new_objects
        if "ALL" not in benchmark.OBJECT_SPLITS["seen"][name]["tasks"]
    ]
    if invalid_targets:
        raise ValueError(f"New targets do not support all requested task families: {invalid_targets}")

    rng = random.Random(args.seed)
    plans = make_plans(seen_objects, legacy_objects, new_objects, rng)
    colors = set(benchmark.COLORS)
    full_color_mentions, color_stats = choose_full_color_mentions(plans, colors, rng)
    rows = [
        plan_to_row(plan, index, index, full_color_mentions, colors)
        for index, plan in enumerate(plans)
    ]
    stats = validate(
        rows,
        plans,
        benchmark,
        seen_objects,
        legacy_objects,
        new_objects,
        color_stats,
    )

    output = args.output if args.output.is_absolute() else root / args.output
    if not args.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
    print(json.dumps({"output": str(output), "seed": args.seed, **stats}, indent=2))


if __name__ == "__main__":
    main()
