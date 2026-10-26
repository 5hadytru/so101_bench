#!/usr/bin/env python3
"""Compute task- and OOD-key-wise success rates from a GR00T eval log."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOADED_EPISODES_RE = re.compile(r"Loaded \d+ validated JSONL episode\(s\) from (?P<path>.+)\.")
EPISODE_RESULT_RE = re.compile(
    r"Episode (?P<episode>\d+)/(?P<total>\d+): success=(?P<success>True|False)\b"
)


@dataclass
class EpisodeResult:
    episode: int
    total: int
    success: bool


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


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


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


def parse_results(log_path: Path) -> tuple[list[EpisodeResult], Path | None]:
    results: list[EpisodeResult] = []
    episodes_jsonl: Path | None = None

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = strip_ansi(raw_line).replace("\r", "")

            loaded_match = LOADED_EPISODES_RE.search(line)
            if loaded_match is not None:
                episodes_jsonl = Path(loaded_match.group("path").strip())

            result_match = EPISODE_RESULT_RE.search(line)
            if result_match is None:
                continue
            results.append(
                EpisodeResult(
                    episode=int(result_match.group("episode")),
                    total=int(result_match.group("total")),
                    success=result_match.group("success") == "True",
                )
            )

    return results, episodes_jsonl


def resolve_episodes_jsonl(log_path: Path, parsed_path: Path | None, override_path: Path | None) -> Path:
    if override_path is not None:
        candidate = override_path
    elif parsed_path is not None:
        candidate = parsed_path
    else:
        raise ValueError("Could not find the loaded episodes JSONL path in the log; pass --episodes-jsonl.")

    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(Path.cwd() / candidate)
        candidates.append(log_path.parent / candidate)
        candidates.append(log_path.parent.parent / candidate)

    for path in candidates:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(f"Could not find episodes JSONL path: {candidate}")


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


def make_rows(results: list[EpisodeResult], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        spec_index = result.episode - 1
        if spec_index < 0 or spec_index >= len(specs):
            raise ValueError(
                f"Log contains episode {result.episode}, but the task JSONL has {len(specs)} row(s)."
            )
        spec = specs[spec_index]
        rows.append(
            {
                "task": infer_task(spec),
                "ood_key": str(spec.get("ood_key", "unknown")),
                "success": result.success,
            }
        )
    return rows


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
        description="Compute success rates by benchmark task type and OOD key from a GR00T eval log."
    )
    parser.add_argument("log_path", type=Path, help="Path to the GR00T eval log.")
    parser.add_argument(
        "--episodes-jsonl",
        type=Path,
        default=None,
        help="Task JSONL used for the run. Defaults to the path parsed from the log.",
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
    results, parsed_episodes_jsonl = parse_results(args.log_path)
    if not results:
        raise ValueError(f"Found no episode result lines in {args.log_path}.")

    episodes_jsonl = resolve_episodes_jsonl(args.log_path, parsed_episodes_jsonl, args.episodes_jsonl)
    specs = load_jsonl(episodes_jsonl)
    rows = make_rows(results, specs)

    expected_total = results[-1].total
    if len(results) != expected_total:
        print(
            f"[WARN] Parsed {len(results)} result line(s), but the log's final episode total is {expected_total}.",
            file=sys.stderr,
        )
    if len(results) > len(specs):
        raise ValueError(f"Parsed {len(results)} result line(s), but {episodes_jsonl} has only {len(specs)} row(s).")

    if args.format == "json":
        print_json(rows)
    elif args.format == "csv":
        print_csv(rows)
    else:
        print(f"log: {args.log_path}")
        print(f"episodes_jsonl: {episodes_jsonl}")
        print_text(rows)


if __name__ == "__main__":
    main()
