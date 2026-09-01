#!/usr/bin/env python3
"""Select a deterministic task-family-stratified subset for performance profiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROFILE_BUCKETS = ("bin_1obj", "bin_4obj", "move", "next_to", "between")
PREFERRED_OOD = {
    "bin_1obj": "seen",
    "bin_4obj": "unseen_seen_class",
    "move": "unseen_unseen_class",
    "next_to": "seen",
    "between": "unseen_seen_class",
}


def task_bucket(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).lower()
    if instruction.startswith("move "):
        return "move"
    if "next to" in instruction:
        return "next_to"
    if "between" in instruction:
        return "between"
    if "plastic bin" in instruction and "each object" in instruction:
        object_count = int(row.get("n_objects", len(row.get("objects", []))))
        if object_count == 1:
            return "bin_1obj"
        if object_count == 4:
            return "bin_4obj"
    return "other"


def select_profile_indices(rows: list[dict[str, Any]], count: int) -> list[int]:
    if count < 1:
        raise ValueError(f"count must be positive, got {count}.")
    if count > len(rows):
        raise ValueError(f"Requested {count} rows from a file containing only {len(rows)} rows.")

    selected: list[int] = []
    for bucket in PROFILE_BUCKETS:
        if len(selected) >= count:
            break
        candidates = [index for index, row in enumerate(rows) if task_bucket(row) == bucket]
        if not candidates:
            continue
        preferred = [
            index
            for index in candidates
            if str(rows[index].get("ood_key", "")) == PREFERRED_OOD[bucket]
        ]
        pool = preferred or candidates
        selected.append(pool[len(pool) // 2])

    remaining_count = count - len(selected)
    if remaining_count > 0:
        selected_set = set(selected)
        remaining = [index for index in range(len(rows)) if index not in selected_set]
        for sample in range(remaining_count):
            position = round((sample + 1) * (len(remaining) - 1) / (remaining_count + 1))
            selected.append(remaining[position])

    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()

    source_lines = [line for line in args.input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in source_lines]
    indices = select_profile_indices(rows, args.count)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl.exists() and args.output_jsonl.stat().st_size > 0:
        raise FileExistsError(f"Refusing to overwrite non-empty profile task subset: {args.output_jsonl}")
    args.output_jsonl.write_text("".join(f"{source_lines[index]}\n" for index in indices), encoding="utf-8")

    print(f"[profile] Selected {len(indices)} representative episode(s):")
    for output_episode, source_index in enumerate(indices, start=1):
        row = rows[source_index]
        print(
            f"[profile]   {output_episode}: source_row={source_index + 1}, "
            f"trial_id={row.get('trial_id', 'none')}, task={task_bucket(row)}, "
            f"ood_key={row.get('ood_key', 'unknown')}"
        )


if __name__ == "__main__":
    main()
