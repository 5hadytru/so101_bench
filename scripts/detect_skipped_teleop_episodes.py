# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Detect which teleop_1.jsonl episodes were skipped (or re-recorded) during teleoperation.

The teleoperator works through the benchmark episode JSONL in order, but may skip some
planned episodes, re-record others (two takes of the same instruction back to back), and
stop before finishing the final object. This leaves the LeRobot dataset episode indices
out of sync with the benchmark JSONL rows, which breaks the dataset<->benchmark row
mapping that ``so101_lerobot_collect_outcomes.py`` relies on.

This script aligns the two sequences purely by their per-episode *instruction* text:

  * teleop JSONL instructions  -> the planned order (trusted)
  * LeRobot dataset `tasks`    -> what was actually recorded, in episode order

It then reports skipped JSONL rows, re-recorded dataset episodes, the trailing cutoff,
and a dataset-episode -> benchmark-row mapping ready to paste into
``--benchmark_episode_indices``.

Layout-file instructions are intentionally ignored; only the dataset and JSONL are used.
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path


def _load_teleop_instructions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(
                {
                    "row_index": line_no,
                    "trial_id": row.get("trial_id", line_no),
                    "instruction": row["instruction"],
                    "objects": row.get("objects", []),
                }
            )
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows


def _load_dataset_instructions(repo_root: Path) -> list[dict]:
    import pyarrow.parquet as pq

    meta_root = repo_root / "meta" / "episodes"
    parquet_paths = sorted(meta_root.glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No episode metadata parquet files found under {meta_root}")

    episodes: list[dict] = []
    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path, columns=["episode_index", "tasks"])
        data = table.to_pydict()
        for episode_index, tasks in zip(data["episode_index"], data["tasks"]):
            tasks = list(tasks) if tasks is not None else []
            instruction = tasks[0] if tasks else ""
            episodes.append({"episode_index": int(episode_index), "instruction": str(instruction)})

    episodes.sort(key=lambda row: row["episode_index"])
    if not episodes:
        raise ValueError(f"No dataset episodes found under {meta_root}.")
    return episodes


def _group_consecutive(indices: list[int]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    for index in indices:
        if groups and index == groups[-1][1] + 1:
            groups[-1] = (groups[-1][0], index)
        else:
            groups.append((index, index))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes_jsonl", type=Path, default=Path("tasks/teleop_1.jsonl"))
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path("data/lerobot/so101_bench_sim_1_v3.0"),
        help="Local LeRobot dataset root containing meta/episodes/*.parquet.",
    )
    parser.add_argument(
        "--generic_instruction",
        type=str,
        default="Place each object in the plastic bin",
        help="Instruction expected to never be skipped; used only for a sanity warning.",
    )
    args = parser.parse_args()

    teleop = _load_teleop_instructions(args.episodes_jsonl)
    dataset = _load_dataset_instructions(args.repo_root)
    teleop_instr = [row["instruction"] for row in teleop]
    dataset_instr = [row["instruction"] for row in dataset]

    print(f"teleop rows:      {len(teleop)}  ({args.episodes_jsonl})")
    print(f"dataset episodes: {len(dataset)}  ({args.repo_root})")
    print()

    matcher = difflib.SequenceMatcher(a=teleop_instr, b=dataset_instr, autojunk=False)
    opcodes = matcher.get_opcodes()

    skipped_rows: list[int] = []  # teleop row indices with no dataset match
    rerecorded: list[tuple[int, int]] = []  # (dataset_episode_index, mirrored teleop row)
    # dataset episode index -> teleop row index it should reset to
    dataset_to_teleop_row: dict[int, int] = {}

    last_matched_teleop_row = -1
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for offset in range(i2 - i1):
                dataset_to_teleop_row[dataset[j1 + offset]["episode_index"]] = i1 + offset
            last_matched_teleop_row = i2 - 1
        elif tag == "delete":
            skipped_rows.extend(range(i1, i2))
        elif tag == "insert":
            for j in range(j1, j2):
                dataset_to_teleop_row[dataset[j]["episode_index"]] = last_matched_teleop_row
                rerecorded.append((dataset[j]["episode_index"], last_matched_teleop_row))
        elif tag == "replace":
            skipped_rows.extend(range(i1, i2))
            for j in range(j1, j2):
                dataset_to_teleop_row[dataset[j]["episode_index"]] = last_matched_teleop_row
                rerecorded.append((dataset[j]["episode_index"], last_matched_teleop_row))

    # Separate the trailing cutoff (final object never finished) from interior skips: any
    # skipped rows beyond the last teleop row that was actually matched are the cutoff.
    cutoff_rows = sorted(r for r in skipped_rows if r > last_matched_teleop_row)
    interior_skips = sorted(r for r in skipped_rows if r <= last_matched_teleop_row)

    print("=" * 88)
    print(f"SKIPPED EPISODES (interior): {len(interior_skips)} episode(s) in "
          f"{len(_group_consecutive(interior_skips))} contiguous group(s)")
    print("=" * 88)
    for start, end in _group_consecutive(interior_skips):
        if start == end:
            label = f"row {start}"
        else:
            label = f"rows {start}-{end} ({end - start + 1} in a row)"
        print(f"  [{label}]")
        for row in range(start, end + 1):
            entry = teleop[row]
            print(f"      trial_id={entry['trial_id']:<4} {entry['instruction']!r}")

    skipped_generic = [r for r in interior_skips if teleop[r]["instruction"] == args.generic_instruction]
    if skipped_generic:
        print(f"\n  [WARN] {len(skipped_generic)} skipped row(s) use the generic instruction "
              f"{args.generic_instruction!r} (expected none).")

    print()
    print("=" * 88)
    print(f"RE-RECORDED EPISODES: {len(rerecorded)} extra dataset take(s) of an already-recorded row")
    print("=" * 88)
    for dataset_episode_index, teleop_row in rerecorded:
        instruction = dataset[next(i for i, d in enumerate(dataset) if d["episode_index"] == dataset_episode_index)]["instruction"]
        mirror = f"teleop row {teleop_row} (trial_id={teleop[teleop_row]['trial_id']})" if teleop_row >= 0 else "n/a"
        print(f"  dataset ep {dataset_episode_index:<4} -> duplicate of {mirror}: {instruction!r}")

    print()
    print("=" * 88)
    print(f"TRAILING CUTOFF (final object not finished): {len(cutoff_rows)} unrecorded row(s)")
    print("=" * 88)
    if cutoff_rows:
        print(f"  teleop rows {cutoff_rows[0]}-{cutoff_rows[-1]} "
              f"(trial_id {teleop[cutoff_rows[0]]['trial_id']}-{teleop[cutoff_rows[-1]]['trial_id']})")
        for row in cutoff_rows:
            entry = teleop[row]
            print(f"      trial_id={entry['trial_id']:<4} {entry['instruction']!r}")

    print()
    print("=" * 88)
    print("DATASET -> BENCHMARK ROW MAPPING")
    print("=" * 88)
    ordered = [dataset_to_teleop_row[d["episode_index"]] for d in dataset]
    contiguous = ordered == list(range(ordered[0], ordered[0] + len(ordered)))
    print(f"  {len(ordered)} dataset episodes map to {len(set(ordered))} distinct benchmark rows.")
    print(f"  mapping is {'contiguous' if contiguous else 'NON-contiguous (skips/re-records present)'}.")
    print()
    print("  Paste into so101_lerobot_collect_outcomes.py --benchmark_episode_indices:")
    print()
    print("  --benchmark_episode_indices " + ",".join(str(teleop[r]["trial_id"]) for r in ordered))


if __name__ == "__main__":
    main()
