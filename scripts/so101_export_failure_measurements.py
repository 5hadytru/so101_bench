#!/usr/bin/env python3
"""Export numeric, per-episode diagnostics from collect_outcomes JSONL output.

The output intentionally retains classifier codes but excludes the natural-language
postmortem rationale.  It is designed to be opened with ``column -t`` or a
spreadsheet and contains one row for every episode whose final label is a failure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "episode_index",
    "task_family",
    "target_objects",
    "referents_or_direction",
    "final_step",
    "final_time_s",
    "final_failure_code",
    "postmortem_code",
    "raw_live_failure_codes",
    "goal_metric",
    "goal_value",
    "goal_threshold",
    "goal_margin",
    "goal_instant",
    "max_goal_hold_steps",
    "required_goal_hold_steps",
    "best_goal_margin",
    "best_goal_step",
    "target_lift_m",
    "target_displacement_m",
    "target_transport_m",
    "target_grasp_attempts",
    "target_acquired",
    "target_manipulated",
    "between_perpendicular_m",
    "between_center_fraction",
    "between_center_fraction_min",
    "between_center_fraction_max",
    "move_directional_progress_m",
    "move_lateral_error_m",
    "move_boundary_gap_m",
    "move_has_boundary",
    "move_overshot",
    "bin_objects_inside_final",
    "bin_objects_required",
    "bin_objects_ever_inside",
    "bin_objects_never_inside",
)


def _labels_for_ids(scene: dict[str, Any], object_ids: list[int]) -> str:
    by_slot = {item["slot"]: item["label"] for item in scene.get("objects", [])}
    return "|".join(by_slot.get(item_id, f"slot_{item_id}") for item_id in object_ids)


def _names_for_benchmark_ids(benchmark: dict[str, Any], object_ids: list[int]) -> str:
    objects = benchmark.get("objects", [])
    return "|".join(objects[item_id] for item_id in object_ids)


def _value(data: dict[str, Any], key: str) -> Any:
    value = data.get(key)
    return "" if value is None else value


def _row(record: dict[str, Any]) -> dict[str, Any]:
    benchmark = record["benchmark"]
    postmortem = record["failure_attribution"]["postmortem"]
    evidence = postmortem.get("evidence", {})
    metric = evidence.get("current_goal_metric", {})
    final_eval = record["final_eval"].get("eval") or {}
    task_family = benchmark["task_family"]
    target_ids = [benchmark["target_object_id"]]
    scene = record.get("final_scene", {})

    if task_family == "bin":
        target_objects = "|".join(benchmark.get("objects", []))
        refs_or_direction = ""
    elif task_family == "named_bin":
        target_objects = _names_for_benchmark_ids(benchmark, target_ids)
        refs_or_direction = ""
    else:
        target_objects = _names_for_benchmark_ids(benchmark, target_ids)
        referent_ids = benchmark.get("referent_object_ids", [])
        refs_or_direction = (
            _names_for_benchmark_ids(benchmark, referent_ids)
            if referent_ids
            else str(benchmark.get("direction") or "")
        )

    ever_inside_ids = evidence.get("ever_inside_bin_object_ids", [])
    ever_inside = _labels_for_ids(scene, ever_inside_ids)
    ever_inside_names = set(ever_inside.split("|")) if ever_inside else set()
    return {
        "episode_index": record["dataset"]["episode_index"],
        "task_family": task_family,
        "target_objects": target_objects,
        "referents_or_direction": refs_or_direction,
        "final_step": _value(final_eval, "step"),
        "final_time_s": _value(final_eval, "time_s"),
        "final_failure_code": record["label"]["failure_reason"],
        "postmortem_code": postmortem["failure_type"],
        "raw_live_failure_codes": "|".join(evidence.get("raw_live_failure_reasons", [])),
        "goal_metric": _value(metric, "name"),
        "goal_value": _value(metric, "value"),
        "goal_threshold": _value(metric, "threshold"),
        "goal_margin": _value(metric, "margin"),
        "goal_instant": _value(metric, "instant"),
        "max_goal_hold_steps": postmortem.get("max_goal_hold_steps", ""),
        "required_goal_hold_steps": evidence.get("goal_required_confirmation_steps", ""),
        "best_goal_margin": evidence.get("best_goal_margin", ""),
        "best_goal_step": evidence.get("best_goal_step", ""),
        "target_lift_m": postmortem.get("target_lift_m", ""),
        "target_displacement_m": postmortem.get("target_max_displacement_m", ""),
        "target_transport_m": postmortem.get("target_associated_transport_m", ""),
        "target_grasp_attempts": postmortem.get("target_attempt_count", ""),
        "target_acquired": postmortem.get("target_acquired", ""),
        "target_manipulated": postmortem.get("target_manipulated", ""),
        "between_perpendicular_m": _value(metric, "perpendicular_distance_m"),
        "between_center_fraction": _value(metric, "center_fraction"),
        "between_center_fraction_min": _value(metric, "center_fraction_min"),
        "between_center_fraction_max": _value(metric, "center_fraction_max"),
        "move_directional_progress_m": _value(metric, "directional_progress_m"),
        "move_lateral_error_m": _value(metric, "lateral_error_m"),
        "move_boundary_gap_m": _value(metric, "boundary_gap_m"),
        "move_has_boundary": _value(metric, "has_boundary"),
        "move_overshot": _value(metric, "overshot"),
        "bin_objects_inside_final": _value(metric, "value") if task_family in {"bin", "named_bin"} else "",
        "bin_objects_required": _value(metric, "threshold") if task_family in {"bin", "named_bin"} else "",
        "bin_objects_ever_inside": ever_inside if task_family in {"bin", "named_bin"} else "",
        "bin_objects_never_inside": (
            "|".join(name for name in target_objects.split("|") if name and name not in ever_inside_names)
            if task_family in {"bin", "named_bin"}
            else ""
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outcomes_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to failure_measurements.tsv within outcomes_dir.",
    )
    args = parser.parse_args()
    episodes_path = args.outcomes_dir / "episodes.jsonl"
    output_path = args.output or args.outcomes_dir / "failure_measurements.tsv"

    with episodes_path.open() as episodes_file:
        records = [json.loads(line) for line in episodes_file if line.strip()]
    failures = [record for record in records if not record["label"]["success"]]

    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(_row(record) for record in failures)
    print(f"Wrote {len(failures)} failure rows to {output_path}")


if __name__ == "__main__":
    main()
