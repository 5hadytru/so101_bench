#!/usr/bin/env python3
"""Build a teleop task/layout subset from failed collect_outcomes records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_episode_indices(value: str) -> list[int]:
    """Parse a comma-separated explicit dataset episode selection."""

    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("--episode_indices must contain at least one integer")
    try:
        indices = [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"--episode_indices must be comma-separated integers, got {value!r}") from exc
    if any(index < 0 for index in indices):
        raise ValueError(f"--episode_indices cannot contain negative values: {indices}")
    if len(indices) != len(set(indices)):
        raise ValueError(f"--episode_indices contains duplicate values: {indices}")
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--layouts", type=Path, required=True)
    parser.add_argument("--output_tasks", type=Path, required=True)
    parser.add_argument("--output_layouts", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument(
        "--episode_indices",
        type=str,
        default=None,
        help=(
            "Optional comma-separated dataset episode indices to select, preserving the supplied order. "
            "Unlike the default behavior, explicitly selected episodes need not have failed in --outcomes."
        ),
    )
    parser.add_argument(
        "--review_file",
        type=Path,
        default=None,
        help=(
            "Optional completed trajectory-review JSONL. Select every row whose decision is "
            "'rerecord' and carry its source/slowed mapping and review reason into the manifest."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = (args.output_tasks, args.output_layouts, args.output_manifest)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output path(s) already exist; pass --overwrite: {existing}")

    outcome_rows = _read_jsonl(args.outcomes)
    task_rows = _read_jsonl(args.tasks)
    layout_rows = _read_jsonl(args.layouts)
    layouts_by_trial_id = {str(row["trial_id"]): row for row in layout_rows}
    if len(layouts_by_trial_id) != len(layout_rows):
        raise ValueError(f"{args.layouts} contains missing or duplicate trial_id values")

    if args.review_file is not None and args.episode_indices is not None:
        raise ValueError("--review_file and --episode_indices are mutually exclusive")

    review_by_episode_index: dict[int, dict[str, Any]] = {}
    if args.review_file is not None:
        review_rows = _read_jsonl(args.review_file)
        for row in review_rows:
            source_index = int(row["source_dataset_episode_index"])
            if source_index in review_by_episode_index:
                raise ValueError(
                    f"{args.review_file} contains duplicate source episode {source_index}"
                )
            decision = str(row.get("decision"))
            if decision not in {"keep", "rerecord"}:
                raise ValueError(
                    f"{args.review_file} source episode {source_index} has invalid decision {decision!r}"
                )
            review_by_episode_index[source_index] = row

        outcomes_by_episode_index = {
            int(row["dataset"]["episode_index"]): row for row in outcome_rows
        }
        if len(outcomes_by_episode_index) != len(outcome_rows):
            raise ValueError(f"{args.outcomes} contains duplicate dataset episode indices")
        outcome_indices = set(outcomes_by_episode_index)
        review_indices = set(review_by_episode_index)
        missing_reviews = sorted(outcome_indices - review_indices)
        extra_reviews = sorted(review_indices - outcome_indices)
        if missing_reviews or extra_reviews:
            raise ValueError(
                "Review/outcome source episode sets differ: "
                f"missing reviews={missing_reviews[:20]}, extra reviews={extra_reviews[:20]}"
            )
        requested_indices = sorted(
            source_index
            for source_index, row in review_by_episode_index.items()
            if row["decision"] == "rerecord"
        )
        selected_outcomes = [outcomes_by_episode_index[index] for index in requested_indices]
        selection_mode = "trajectory_review_rerecord"
    elif args.episode_indices is None:
        selected_outcomes = [row for row in outcome_rows if not bool(row.get("label", {}).get("success"))]
        selected_outcomes.sort(key=lambda row: int(row["dataset"]["episode_index"]))
        selection_mode = "failed_outcomes"
    else:
        requested_indices = _parse_episode_indices(args.episode_indices)
        outcomes_by_episode_index: dict[int, dict[str, Any]] = {}
        for row in outcome_rows:
            episode_index = int(row["dataset"]["episode_index"])
            if episode_index in outcomes_by_episode_index:
                raise ValueError(f"{args.outcomes} contains duplicate dataset episode {episode_index}")
            outcomes_by_episode_index[episode_index] = row
        missing = [index for index in requested_indices if index not in outcomes_by_episode_index]
        if missing:
            raise KeyError(f"{args.outcomes} is missing requested dataset episode(s): {missing}")
        selected_outcomes = [outcomes_by_episode_index[index] for index in requested_indices]
        selection_mode = "explicit_episode_indices"
    source_hashes = {
        "outcomes_jsonl": _sha256(args.outcomes),
        "tasks_jsonl": _sha256(args.tasks),
        "layouts_jsonl": _sha256(args.layouts),
    }
    if args.review_file is not None:
        source_hashes["trajectory_review_jsonl"] = _sha256(args.review_file)
    selected_tasks: list[dict[str, Any]] = []
    selected_layouts: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for patch_index, outcome in enumerate(selected_outcomes):
        source_episode_index = int(outcome["dataset"]["episode_index"])
        benchmark = outcome["benchmark"]
        source_benchmark_index = int(benchmark["episode_index"])
        if source_benchmark_index < 0 or source_benchmark_index >= len(task_rows):
            raise IndexError(f"Benchmark row {source_benchmark_index} is outside {args.tasks}")
        task = task_rows[source_benchmark_index]
        trial_id = task.get("trial_id", source_benchmark_index)
        layout = layouts_by_trial_id.get(str(trial_id))
        if layout is None:
            raise KeyError(f"No layout found for trial_id={trial_id!r}")

        expected_objects = list(benchmark["objects"])
        layout_objects = [entry.get("name") for entry in layout.get("objects", [])]
        if task.get("instruction") != benchmark.get("instruction"):
            raise ValueError(f"Instruction mismatch for source episode {source_episode_index}")
        if list(task.get("objects", [])) != expected_objects or layout_objects != expected_objects:
            raise ValueError(f"Object mismatch for source episode {source_episode_index}")
        if layout.get("instruction") != task.get("instruction"):
            raise ValueError(f"Task/layout instruction mismatch for trial_id={trial_id!r}")

        selected_tasks.append(task)
        selected_layouts.append(layout)
        postmortem = outcome.get("final_failure_attribution", {}).get("postmortem") or {}
        manifest.append(
            {
                "schema_version": 1,
                "patch_episode_index": patch_index,
                "redo_dataset_episode_index": patch_index,
                "subset_row_index": patch_index,
                "source_dataset_episode_index": source_episode_index,
                "source_benchmark_episode_index": source_benchmark_index,
                "source_trial_id": trial_id,
                "task_family": benchmark.get("task_family"),
                "instruction": benchmark.get("instruction"),
                "objects": expected_objects,
                "selection_mode": selection_mode,
                "source_outcome_success": bool(outcome.get("label", {}).get("success")),
                "original_failure_reason": outcome.get("label", {}).get("failure_reason"),
                "original_behavioral_failure_type": postmortem.get("failure_type"),
                "review_decision_source": (
                    review_by_episode_index[source_episode_index].get("decision_source")
                    if review_by_episode_index
                    else None
                ),
                "review_reason": (
                    review_by_episode_index[source_episode_index].get("reason")
                    if review_by_episode_index
                    else None
                ),
                "replaced_slowed_dataset_episode_index": (
                    review_by_episode_index[source_episode_index].get(
                        "slowed_dataset_episode_index"
                    )
                    if review_by_episode_index
                    else None
                ),
                "source_sha256": source_hashes,
            }
        )

    if not selected_outcomes:
        raise ValueError(f"No episodes selected from {args.outcomes}")
    source_ids = [row["source_dataset_episode_index"] for row in manifest]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Failed outcome set contains duplicate source dataset episode indices")

    _write_jsonl(args.output_tasks, selected_tasks)
    _write_jsonl(args.output_layouts, selected_layouts)
    _write_jsonl(args.output_manifest, manifest)
    print(f"Prepared {len(selected_outcomes)} episode(s) for rerecording ({selection_mode}).")
    print(f"  tasks:    {args.output_tasks} (sha256={_sha256(args.output_tasks)})")
    print(f"  layouts:  {args.output_layouts} (sha256={_sha256(args.output_layouts)})")
    print(f"  manifest: {args.output_manifest} (sha256={_sha256(args.output_manifest)})")
    print(f"  source episode IDs: {','.join(map(str, source_ids))}")


if __name__ == "__main__":
    main()
