#!/usr/bin/env python3
"""Compute real-world success rates on the GR00T sim-eval subset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Counts:
    successes: int = 0
    total: int = 0

    @property
    def rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.successes / self.total

    def add(self, success: bool) -> None:
        self.successes += int(success)
        self.total += 1


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on {path}:{line_number}, got {type(row).__name__}")
            rows.append(row)
    return rows


def infer_task(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).lower()
    if row.get("direction") is not None or instruction.startswith("move "):
        return "move"
    if "between" in instruction:
        return "between"
    if "next to" in instruction:
        return "next_to"
    if "plastic bin" in instruction:
        if "each object" not in instruction:
            return "named_bin"
        n_objects = row.get("n_objects")
        if n_objects in (1, 4):
            return f"bin_{n_objects}obj"
        return "bin"
    return "unknown"


def sim_object_names(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(name) for name in row.get("objects", []))


def real_bin_object_names(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(obj["name"]) for obj in row.get("objects", []))


def real_bin_success(row: dict[str, Any]) -> bool:
    object_results = row.get("objects", [])
    if not isinstance(object_results, list) or not object_results:
        raise ValueError(f"Expected non-empty object result list for bin row: {row}")
    return all(isinstance(obj, dict) and obj.get("success") is True for obj in object_results)


def match_subset_rows(
    subset_rows: list[dict[str, Any]],
    real_if_rows: list[dict[str, Any]],
    real_bin_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if_index: dict[tuple[tuple[str, ...], str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in real_if_rows:
        if_index[(sim_object_names(row), str(row.get("instruction", "")), str(row.get("ood_key", "unknown")))].append(row)

    bin_index: dict[tuple[str, int, tuple[str, ...]], deque[dict[str, Any]]] = defaultdict(deque)
    for row in real_bin_rows:
        bin_index[(str(row.get("ood_key", "unknown")), int(row.get("n_objects", 0)), real_bin_object_names(row))].append(row)

    matched_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for index, subset_row in enumerate(subset_rows, start=1):
        task = infer_task(subset_row)
        ood_key = str(subset_row.get("ood_key", "unknown"))

        if task.startswith("bin"):
            key = (ood_key, int(subset_row.get("n_objects", 0)), sim_object_names(subset_row))
            bucket = bin_index.get(key)
            if not bucket:
                missing.append(f"subset row {index}: bin key={key}")
                continue
            real_row = bucket.popleft()
            success = real_bin_success(real_row)
        else:
            key = (sim_object_names(subset_row), str(subset_row.get("instruction", "")), ood_key)
            bucket = if_index.get(key)
            if not bucket:
                missing.append(f"subset row {index}: IF key={key}")
                continue
            real_row = bucket.popleft()
            success = real_row.get("success") is True

        matched_rows.append(
            {
                "task": task,
                "ood_key": ood_key,
                "success": success,
                "subset_trial_id": subset_row.get("trial_id"),
                "real_trial_id": real_row.get("trial_id"),
            }
        )

    if missing:
        sample = "\n".join(missing[:20])
        raise ValueError(f"Could not match {len(missing)} subset row(s) to real-world results:\n{sample}")

    return matched_rows


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[str, ...], Counts]:
    counts: dict[tuple[str, ...], Counts] = defaultdict(Counts)
    for row in rows:
        key = tuple(str(row[k]) for k in keys)
        counts[key].add(bool(row["success"]))
    return dict(sorted(counts.items()))


def format_table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return f"{title}\n(no rows)"
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    lines = [title, "  ".join(header.ljust(width) for header, width in zip(headers, widths))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend("  ".join(value.ljust(width) for value, width in zip(row, widths)) for row in rows)
    return "\n".join(lines)


def count_rows(counts: dict[tuple[str, ...], Counts]) -> list[list[str]]:
    rows: list[list[str]] = []
    for key, value in counts.items():
        rows.append([*key, str(value.successes), str(value.total), f"{100.0 * value.rate:.1f}%"])
    return rows


def print_text(rows: list[dict[str, Any]]) -> None:
    overall = Counts()
    for row in rows:
        overall.add(bool(row["success"]))

    print(f"overall: {overall.successes}/{overall.total} ({100.0 * overall.rate:.1f}%)")
    print()
    print(
        format_table(
            "by task and ood_key",
            ["task", "ood_key", "successes", "total", "success_rate"],
            count_rows(aggregate(rows, ("task", "ood_key"))),
        )
    )
    print()
    print(
        format_table(
            "by task",
            ["task", "successes", "total", "success_rate"],
            count_rows(aggregate(rows, ("task",))),
        )
    )
    print()
    print(
        format_table(
            "by ood_key",
            ["ood_key", "successes", "total", "success_rate"],
            count_rows(aggregate(rows, ("ood_key",))),
        )
    )


def print_json(rows: list[dict[str, Any]]) -> None:
    def encode_counts(counts: dict[tuple[str, ...], Counts], key_names: tuple[str, ...]) -> list[dict[str, Any]]:
        encoded: list[dict[str, Any]] = []
        for key, value in counts.items():
            encoded.append(
                {
                    **dict(zip(key_names, key)),
                    "successes": value.successes,
                    "total": value.total,
                    "success_rate": value.rate,
                }
            )
        return encoded

    overall = Counts()
    for row in rows:
        overall.add(bool(row["success"]))

    print(
        json.dumps(
            {
                "overall": {
                    "successes": overall.successes,
                    "total": overall.total,
                    "success_rate": overall.rate,
                },
                "by_task_ood_key": encode_counts(
                    aggregate(rows, ("task", "ood_key")),
                    ("task", "ood_key"),
                ),
                "by_task": encode_counts(aggregate(rows, ("task",)), ("task",)),
                "by_ood_key": encode_counts(aggregate(rows, ("ood_key",)), ("ood_key",)),
            },
            indent=2,
        )
    )


def print_csv(rows: list[dict[str, Any]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=["task", "ood_key", "successes", "total", "success_rate"])
    writer.writeheader()
    for (task, ood_key), value in aggregate(rows, ("task", "ood_key")).items():
        writer.writerow(
            {
                "task": task,
                "ood_key": ood_key,
                "successes": value.successes,
                "total": value.total,
                "success_rate": f"{value.rate:.6f}",
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute real-world policy success rates on the subset defined by "
            "tasks/real_gr00t_WM_combined.jsonl."
        )
    )
    parser.add_argument(
        "--subset-jsonl",
        type=Path,
        default=Path("tasks/real_gr00t_WM_combined.jsonl"),
        help="Subset task JSONL to evaluate. Defaults to tasks/real_gr00t_WM_combined.jsonl.",
    )
    parser.add_argument(
        "--if-results-jsonl",
        type=Path,
        default=Path("logs/real_world_results/results_gr00t_IF_WM.jsonl"),
        help="Real-world instruction-following results JSONL.",
    )
    parser.add_argument(
        "--bin-results-jsonl",
        type=Path,
        default=Path("logs/real_world_results/results_gr00t_grasp_WM.jsonl"),
        help="Real-world bin/grasp results JSONL.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "csv"),
        default="text",
        help="Output format. CSV includes the task+ood_key table only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset_rows = load_jsonl(args.subset_jsonl)
    real_if_rows = load_jsonl(args.if_results_jsonl)
    real_bin_rows = load_jsonl(args.bin_results_jsonl)
    rows = match_subset_rows(subset_rows, real_if_rows, real_bin_rows)

    if args.format == "json":
        print_json(rows)
    elif args.format == "csv":
        print_csv(rows)
    else:
        print(f"subset_jsonl: {args.subset_jsonl}")
        print(f"if_results_jsonl: {args.if_results_jsonl}")
        print(f"bin_results_jsonl: {args.bin_results_jsonl}")
        print_text(rows)


if __name__ == "__main__":
    main()
