#!/usr/bin/env python3
"""Generate the balanced, distractor-rich ``tasks/teleop_3.jsonl`` set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import ModuleType
from typing import Any

import networkx as nx


COUNTS = {"bin_1obj": 330, "bin_4obj": 304, "named_bin": 400, "next_to": 100, "between": 100, "move": 150}
COLORS = {"black", "blue", "brown", "gray", "green", "grey", "orange", "pink", "purple", "red", "silver", "white", "yellow"}
DIRECTIONS = ("left", "right", "forwards", "backwards")
FAMILY_ORDER = {"bin_1obj": 0, "bin_4obj": 1, "named_bin": 2, "next_to": 3, "between": 4, "move": 5}
FORBIDDEN_OBJECT = "brown stuffed animal"
SUPPORT_ONLY_OBJECT = "black shoes"
BIN_ONLY_TARGET = "green shoes"


@dataclass(frozen=True)
class Plan:
    family: str
    objects: tuple[str, ...]
    target: str
    referents: tuple[str, ...] = ()
    direction: str | None = None
    anchor: str | None = None

    def named_objects(self) -> tuple[str, ...]:
        if self.family in {"named_bin", "move"}:
            return (self.target,)
        if self.family in {"next_to", "between"}:
            return (self.target, *self.referents)
        return ()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_benchmark(root: Path) -> ModuleType:
    path = root / "source" / "so101_bench" / "so101_bench" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("so101_bench_benchmark_teleop3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def colorless(name: str) -> str:
    words = name.split()
    return " ".join(words[1:]) if words and words[0] in COLORS else name


def color(name: str) -> str | None:
    first = name.split()[0]
    return first if first in COLORS else None


def is_distractor(candidate: str, named: tuple[str, ...]) -> bool:
    return any(
        candidate != other
        and (colorless(candidate) == colorless(other) or (color(candidate) is not None and color(candidate) == color(other)))
        for other in named
    )


def can_omit_color(name: str, objects: tuple[str, ...]) -> bool:
    shortened = colorless(name)
    return shortened != name and all(
        other == name or (other != shortened and colorless(other) != shortened) for other in objects
    )


def family(row: dict[str, Any]) -> str:
    if "direction" in row:
        return "move"
    if len(row.get("referents", [])) == 1:
        return "next_to"
    if len(row.get("referents", [])) == 2:
        return "between"
    if "target" in row:
        return "named_bin"
    return "bin_1obj" if row["n_objects"] == 1 else "bin_4obj"


def semantic_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    task = family(row)
    referents = tuple(row.get("referents", ()))
    if task == "between":
        referents = tuple(sorted(referents))
    return (task, tuple(sorted(row["objects"])), row.get("target"), referents, row.get("direction"))


def prior_signatures(paths: tuple[Path, ...]) -> set[tuple[Any, ...]]:
    result: set[tuple[Any, ...]] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result.add(semantic_signature(json.loads(line)))
    return result


def balanced_quota(items: tuple[str, ...], total: int, rng: random.Random) -> Counter[str]:
    quotient, remainder = divmod(total, len(items))
    result = Counter({item: quotient for item in items})
    for item in rng.sample(list(items), remainder):
        result[item] += 1
    return result


def target_sequence(quota: Counter[str], object_order: dict[str, int]) -> list[str]:
    return [name for name in sorted(quota, key=object_order.__getitem__) for _ in range(quota[name])]


def choose_least(
    candidates: list[str], count: int, appearances: Counter[str], rng: random.Random
) -> tuple[str, ...]:
    chosen: list[str] = []
    available = list(candidates)
    for _ in range(count):
        minimum = min(appearances[name] for name in available)
        # A small slack keeps appearances near-uniform while avoiding repeated
        # layouts when the strict minimum candidates have already been used.
        tied = [name for name in available if appearances[name] <= minimum + 3]
        name = rng.choice(tied)
        chosen.append(name)
        available.remove(name)
        appearances[name] += 1
    return tuple(chosen)


def make_four_bin_plans(
    graspable: tuple[str, ...],
    anchor_quota: Counter[str],
    desired: Counter[str],
    old_signatures: set[tuple[Any, ...]],
    rng: random.Random,
) -> list[Plan]:
    """Make unique four-object sets with exactly balanced grasp appearances."""

    anchors = target_sequence(anchor_quota, {name: i for i, name in enumerate(graspable)})
    support_needed = Counter({name: desired[name] - anchor_quota[name] for name in graspable})
    if min(support_needed.values()) < 0 or sum(support_needed.values()) != 3 * len(anchors):
        raise RuntimeError(f"Invalid four-bin support quotas: {support_needed}")

    for attempt in range(100):
        attempt_rng = random.Random(rng.randrange(2**63) + attempt)
        graph = nx.DiGraph()
        source, sink = "source", "sink"
        graph.add_node(source, demand=-3 * len(anchors))
        graph.add_node(sink, demand=3 * len(anchors))
        for index, anchor in enumerate(anchors):
            episode_node = f"episode:{index}"
            graph.add_node(episode_node, demand=0)
            graph.add_edge(source, episode_node, capacity=3, weight=0)
            for name in graspable:
                if name != anchor:
                    graph.add_edge(episode_node, f"object:{name}", capacity=1, weight=attempt_rng.randrange(1000))
        for name in graspable:
            object_node = f"object:{name}"
            graph.add_node(object_node, demand=0)
            graph.add_edge(object_node, sink, capacity=support_needed[name], weight=0)
        _cost, flow = nx.network_simplex(graph)
        supports = [
            [name for name in graspable if flow[f"episode:{index}"].get(f"object:{name}", 0)]
            for index in range(len(anchors))
        ]

        def object_set(index: int) -> tuple[str, ...]:
            return tuple(sorted((anchors[index], *supports[index])))

        def is_old(values: tuple[str, ...]) -> bool:
            dummy = {"n_objects": 4, "objects": list(values), "instruction": "Place each object in the plastic bin"}
            return semantic_signature(dummy) in old_signatures

        # Flow balances appearances exactly. Pairwise swaps retain those totals
        # while removing duplicate or previously used four-object sets.
        for _ in range(20_000):
            sets = [object_set(index) for index in range(len(anchors))]
            counts = Counter(sets)
            bad = [
                index
                for index, values in enumerate(sets)
                if is_old(values) or (counts[values] > 1 and index != sets.index(values))
            ]
            if not bad:
                plans: list[Plan] = []
                for index, anchor in enumerate(anchors):
                    objects = [anchor, *supports[index]]
                    attempt_rng.shuffle(objects)
                    plans.append(Plan("bin_4obj", tuple(objects), target=anchor, anchor=anchor))
                return plans
            first = attempt_rng.choice(bad)
            second = attempt_rng.randrange(len(anchors))
            if first == second:
                continue
            a = attempt_rng.choice(supports[first])
            b = attempt_rng.choice(supports[second])
            if a == b or b == anchors[first] or a == anchors[second]:
                continue
            if b in supports[first] or a in supports[second]:
                continue
            old_first, old_second = object_set(first), object_set(second)
            supports[first][supports[first].index(a)] = b
            supports[second][supports[second].index(b)] = a
            new_first, new_second = object_set(first), object_set(second)
            other_sets = set(sets)
            other_sets.discard(old_first)
            other_sets.discard(old_second)
            if is_old(new_first) or is_old(new_second) or new_first in other_sets or new_second in other_sets or new_first == new_second:
                supports[first][supports[first].index(b)] = a
                supports[second][supports[second].index(a)] = b
    raise RuntimeError("Could not construct balanced, novel four-object bin plans")


def choose_referents(
    target: str,
    count: int,
    all_objects: tuple[str, ...],
    appearances: Counter[str],
    rng: random.Random,
) -> tuple[str, ...]:
    candidates = [name for name in all_objects if name != target]
    return choose_least(candidates, count, appearances, rng)


def add_fillers(
    plan: Plan,
    all_objects: tuple[str, ...],
    slots: int,
    appearances: Counter[str],
    want_distractor: bool,
    rng: random.Random,
    diversity_mode: bool = False,
) -> Plan:
    base = (plan.target, *plan.referents)
    excluded = set(base)
    named = plan.named_objects()
    candidates = [name for name in all_objects if name not in excluded]
    fillers: list[str] = []

    distractors = [name for name in candidates if is_distractor(name, named)]
    nondistractors = [name for name in candidates if not is_distractor(name, named)]
    if want_distractor and distractors:
        if diversity_mode:
            name = rng.choice(distractors)
            fillers.append(name)
            appearances[name] += 1
        else:
            fillers.extend(choose_least(distractors, 1, appearances, rng))
    elif not want_distractor and len(nondistractors) >= slots:
        candidates = nondistractors

    excluded.update(fillers)
    remaining_candidates = [name for name in candidates if name not in excluded]
    remaining = slots - len(fillers)
    if diversity_mode:
        selected = rng.sample(remaining_candidates, remaining)
        fillers.extend(selected)
        appearances.update(selected)
    else:
        fillers.extend(choose_least(remaining_candidates, remaining, appearances, rng))
    objects = list((*base, *fillers))
    # Target identity is explicit metadata, so named-bin targets may occupy any
    # scene slot just like targets in the other instruction-following families.
    rng.shuffle(objects)
    return replace(plan, objects=tuple(objects))


def make_plans(
    all_objects: tuple[str, ...],
    graspable: tuple[str, ...],
    spatial_targets: tuple[str, ...],
    old_signatures: set[tuple[Any, ...]],
    rng: random.Random,
) -> list[Plan]:
    object_order = {name: index for index, name in enumerate(graspable)}
    bin1_quota = balanced_quota(graspable, COUNTS["bin_1obj"], rng)
    anchor_quota = balanced_quota(graspable, COUNTS["bin_4obj"], rng)
    next_quota = balanced_quota(spatial_targets, COUNTS["next_to"], rng)
    between_quota = balanced_quota(spatial_targets, COUNTS["between"], rng)
    move_quota = balanced_quota(spatial_targets, COUNTS["move"], rng)

    # Balance total grasp events, compensating green shoes with named-bin work
    # because it is intentionally absent from all spatial target quotas.
    spatial_grasps = next_quota + between_quota + move_quota
    four_bin_grasps = balanced_quota(graspable, COUNTS["bin_4obj"] * 4, random.Random(30303))
    provisional = bin1_quota + four_bin_grasps + spatial_grasps
    desired_total = balanced_quota(graspable, sum(COUNTS.values()) + 3 * COUNTS["bin_4obj"], rng)
    named_quota = Counter({name: desired_total[name] - provisional[name] for name in graspable})
    if min(named_quota.values()) < 0 or sum(named_quota.values()) != COUNTS["named_bin"]:
        raise RuntimeError(f"Invalid named-bin quota solution: {named_quota}")

    four_bin = make_four_bin_plans(graspable, anchor_quota, four_bin_grasps, old_signatures, rng)

    plans: list[Plan] = []
    for target in graspable:
        plans.extend(Plan("bin_1obj", (target,), target=target) for _ in range(bin1_quota[target]))
    plans.extend(four_bin)
    for target in graspable:
        plans.extend(Plan("named_bin", (target,), target=target) for _ in range(named_quota[target]))

    referent_appearances: Counter[str] = Counter()
    for target in target_sequence(next_quota, object_order):
        refs = choose_referents(target, 1, all_objects, referent_appearances, rng)
        plans.append(Plan("next_to", (target, *refs), target=target, referents=refs))
    for target in target_sequence(between_quota, object_order):
        refs = choose_referents(target, 2, all_objects, referent_appearances, rng)
        plans.append(Plan("between", (target, *refs), target=target, referents=refs))
    direction_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for target in target_sequence(move_quota, object_order):
        minimum = min(direction_counts[target][direction] for direction in DIRECTIONS)
        direction = rng.choice([direction for direction in DIRECTIONS if direction_counts[target][direction] == minimum])
        direction_counts[target][direction] += 1
        plans.append(Plan("move", (target,), target=target, direction=direction))

    base_appearances: Counter[str] = Counter()
    for plan in plans:
        base_appearances.update(plan.objects)

    # Nine tenths of each named family request a deliberate class/color
    # distractor. Some colorless, unique-class targets have no possible
    # distractor, and the remaining tenth supplies the requested clean scenes.
    distractor_flags: dict[str, list[bool]] = {}
    for task in ("named_bin", "next_to", "between", "move"):
        count = sum(plan.family == task for plan in plans)
        flags = [True] * round(0.9 * count) + [False] * (count - round(0.9 * count))
        rng.shuffle(flags)
        distractor_flags[task] = flags
    family_flag_index: Counter[str] = Counter()
    filled: list[Plan] = []
    for plan in plans:
        slots = {"named_bin": 3, "next_to": 2, "between": 1, "move": 3}.get(plan.family, 0)
        if not slots:
            filled.append(plan)
            continue
        want_distractor = distractor_flags[plan.family][family_flag_index[plan.family]]
        candidate_plan: Plan | None = None
        # Retry filler choices until the semantic configuration is new.
        for attempt in range(500):
            trial_counts = base_appearances.copy()
            candidate = add_fillers(
                plan,
                all_objects,
                slots,
                trial_counts,
                want_distractor,
                rng,
                diversity_mode=attempt >= 100,
            )
            dummy = plan_to_row(candidate, 0, {})
            signature = semantic_signature(dummy)
            if signature not in old_signatures and all(semantic_signature(plan_to_row(existing, 0, {})) != signature for existing in filled):
                candidate_plan = candidate
                base_appearances = trial_counts
                break
        if candidate_plan is None:
            raise RuntimeError(f"Could not find a novel configuration for {plan}")
        filled.append(candidate_plan)
        family_flag_index[plan.family] += 1

    return sorted(filled, key=lambda plan: (object_order[plan.anchor or plan.target], FAMILY_ORDER[plan.family]))


def plan_to_row(plan: Plan, trial_id: int, omitted: dict[tuple[int, int], bool]) -> dict[str, Any]:
    labels = []
    for mention_index, name in enumerate(plan.named_objects()):
        labels.append(colorless(name) if omitted.get((trial_id, mention_index), False) else name)
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
    if plan.family in {"named_bin", "next_to", "between", "move"}:
        row["target"] = plan.target
    if plan.family == "named_bin":
        row["task_family"] = "named_bin"
    if plan.referents:
        row["referents"] = list(plan.referents)
    if plan.direction is not None:
        row["direction"] = {"forwards": "forward", "backwards": "backward"}.get(plan.direction, plan.direction)
    return row


def omission_map(plans: list[Plan], rng: random.Random) -> tuple[dict[tuple[int, int], bool], Counter[str]]:
    eligible: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for plan_index, plan in enumerate(plans):
        for mention_index, name in enumerate(plan.named_objects()):
            if can_omit_color(name, plan.objects):
                eligible[name].append((plan_index, mention_index))
    omitted: dict[tuple[int, int], bool] = {}
    stats: Counter[str] = Counter()
    for name, positions in eligible.items():
        rng.shuffle(positions)
        count = round(len(positions) * 0.5)
        for position in positions[:count]:
            omitted[position] = True
        stats["eligible"] += len(positions)
        stats["omitted"] += count
    return omitted, stats


def validate(
    rows: list[dict[str, Any]], benchmark: ModuleType, all_objects: tuple[str, ...], old: set[tuple[Any, ...]]
) -> dict[str, Any]:
    actual_counts = Counter(family(row) for row in rows)
    if actual_counts != Counter(COUNTS):
        raise AssertionError(f"Task counts differ: {actual_counts}")
    if any(FORBIDDEN_OBJECT in row["objects"] for row in rows):
        raise AssertionError(f"{FORBIDDEN_OBJECT} appears")
    if any(row.get("target") == SUPPORT_ONLY_OBJECT for row in rows):
        raise AssertionError(f"{SUPPORT_ONLY_OBJECT} is targeted")
    if any(SUPPORT_ONLY_OBJECT in row["objects"] and family(row) == "bin_4obj" for row in rows):
        raise AssertionError(f"{SUPPORT_ONLY_OBJECT} would be grasped by a generic bin instruction")
    if any(row.get("target") == BIN_ONLY_TARGET and family(row) not in {"named_bin"} for row in rows):
        raise AssertionError(f"{BIN_ONLY_TARGET} is targeted outside a named bin task")

    grasp_counts: Counter[str] = Counter()
    appearance_counts: Counter[str] = Counter()
    distractors: Counter[str] = Counter()
    named_counts: Counter[str] = Counter()
    seen_signatures: set[tuple[Any, ...]] = set()
    unavoidable_overlap = 0
    for index, row in enumerate(rows):
        benchmark.episode_spec_from_json(row, source=f"generated row {index}")
        appearance_counts.update(row["objects"])
        task = family(row)
        if task == "bin_4obj":
            grasp_counts.update(row["objects"])
        elif task == "bin_1obj":
            grasp_counts[row["objects"][0]] += 1
        else:
            grasp_counts[row["target"]] += 1
        named = tuple(([row["target"]] if "target" in row else []) + row.get("referents", []))
        if named:
            named_counts[task] += 1
            fillers = set(row["objects"]) - set(named)
            distractors[task] += any(is_distractor(name, named) for name in fillers)
        signature = semantic_signature(row)
        if task == "bin_1obj":
            unavoidable_overlap += signature in old
        else:
            if signature in old:
                raise AssertionError(f"Avoidable prior-task overlap at row {index}: {row}")
            if signature in seen_signatures:
                raise AssertionError(f"Duplicate generated configuration at row {index}: {row}")
        seen_signatures.add(signature)
    benchmark.validate_move_episode_footprints(
        [benchmark.episode_spec_from_json(row, source=f"generated row {index}") for index, row in enumerate(rows)]
    )
    if max(grasp_counts.values()) - min(grasp_counts.values()) > 1:
        raise AssertionError(f"Grasp counts are not balanced: {grasp_counts}")
    return {
        "task_counts": actual_counts,
        "grasp_counts": grasp_counts,
        "appearance_counts": appearance_counts,
        "distractors": distractors,
        "named_counts": named_counts,
        "unavoidable_overlap": unavoidable_overlap,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tasks/teleop_3.jsonl"))
    parser.add_argument(
        "--bin-eval-output",
        type=Path,
        default=Path("tasks/teleop_3_bin_1obj_named_bin.jsonl"),
        help="Filtered, compactly renumbered 1-object/named-bin evaluation task file.",
    )
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    benchmark = load_benchmark(root)
    all_objects = tuple(
        name for name in benchmark.OBJECT_SPLITS["seen"] if name != FORBIDDEN_OBJECT
    )
    graspable = tuple(name for name in all_objects if name != SUPPORT_ONLY_OBJECT)
    spatial_targets = tuple(name for name in graspable if name != BIN_ONLY_TARGET)
    old = prior_signatures((root / "tasks/teleop_1.jsonl", root / "tasks/teleop_2.jsonl"))
    rng = random.Random(args.seed)
    plans = make_plans(all_objects, graspable, spatial_targets, old, rng)
    omitted, omission_stats = omission_map(plans, rng)
    rows = [plan_to_row(plan, index, omitted) for index, plan in enumerate(plans)]
    stats = validate(rows, benchmark, all_objects, old)
    bin_eval_rows = [dict(row) for row in rows if family(row) in {"bin_1obj", "named_bin"}]
    for trial_id, row in enumerate(bin_eval_rows):
        row["trial_id"] = trial_id
        benchmark.episode_spec_from_json(row, source=f"bin eval row {trial_id}")

    output = args.output if args.output.is_absolute() else root / args.output
    bin_eval_output = args.bin_eval_output if args.bin_eval_output.is_absolute() else root / args.bin_eval_output
    if not args.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        bin_eval_output.parent.mkdir(parents=True, exist_ok=True)
        bin_eval_output.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in bin_eval_rows),
            encoding="utf-8",
        )
    destination = f"{output} (dry run)" if args.dry_run else str(output)
    print(f"Prepared {len(rows)} episodes for {destination}")
    print("Task counts:", dict(stats["task_counts"]))
    print(f"Grasp counts: min={min(stats['grasp_counts'].values())}, max={max(stats['grasp_counts'].values())}")
    print(f"All appearances: min={min(stats['appearance_counts'].values())}, max={max(stats['appearance_counts'].values())}")
    print(
        "Distractor episodes:",
        ", ".join(
            f"{task}={stats['distractors'][task]}/{stats['named_counts'][task]}"
            for task in ("named_bin", "next_to", "between", "move")
        ),
    )
    print(f"Safe color omissions: {omission_stats['omitted']}/{omission_stats['eligible']}")
    print(f"Prior overlap: {stats['unavoidable_overlap']} one-object bin rows; 0 avoidable rows")
    print(f"Bin eval subset: {len(bin_eval_rows)} episodes at {bin_eval_output}")


if __name__ == "__main__":
    main()
