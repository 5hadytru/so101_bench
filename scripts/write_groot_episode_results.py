#!/usr/bin/env python3
"""Write complete per-episode GR00T evaluation outcomes in JSON and TSV formats.

``scripts/groot_eval.py`` writes one ``Episode N/T: success=...`` line for
each trial.  This utility joins those lines with the source task JSONL so a
checkpoint evaluation has an auditable list of successes and failures, not
only aggregate rates.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from compute_groot_success_rates import (
    infer_task,
    load_jsonl,
    parse_results,
    resolve_episodes_jsonl,
    strip_ansi,
)


EPISODE_DETAIL_RE = re.compile(
    r"Episode (?P<episode>\d+)/(?P<total>\d+): success=(?P<success>True|False)(?P<tail>.*)"
)


def match_text(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1).strip() if match is not None else None


def parse_details(log_path: Path) -> dict[int, dict[str, Any]]:
    """Parse termination/postmortem diagnostics printed by groot_eval.py."""

    details: dict[int, dict[str, Any]] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = strip_ansi(raw_line).replace("\r", "")
            match = EPISODE_DETAIL_RE.search(line)
            if match is None:
                continue
            episode = int(match.group("episode"))
            tail = match.group("tail")
            if episode in details:
                raise ValueError(f"Found more than one outcome line for episode {episode} in {log_path}.")

            length_text = match_text(r", length=([0-9.]+)s", tail)
            details[episode] = {
                "termination_reason": match_text(r", reason=([^,]+)", tail),
                "episode_length_s": float(length_text) if length_text is not None else None,
                "failure_type": match_text(r", failure_type=([^,]+)", tail),
                "live_failure_reason": match_text(r", live_failure_reason=([^,\s\[]+)", tail),
                "target_object": match_text(r"\[target=([^,\]]+)", tail),
                "target_lift_in": None,
                "lifted_wrong_object": match_text(r", wrong_object=([^,\]]+)", tail),
                "max_distractor_lift_in": None,
                "lift_threshold_in": None,
            }
            for log_key, output_key in (
                ("target_lift", "target_lift_in"),
                ("max_distractor_lift", "max_distractor_lift_in"),
                ("lift_threshold", "lift_threshold_in"),
            ):
                value = match_text(rf", {log_key}=([0-9.]+)in", tail)
                if value is not None:
                    details[episode][output_key] = float(value)
    return details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join a GR00T evaluation log with its task JSONL and write per-episode results."
    )
    parser.add_argument("log_path", type=Path, help="Evaluation log produced by scripts/groot_eval.py.")
    parser.add_argument("--episodes-jsonl", type=Path, required=True, help="Task JSONL used for the evaluation.")
    parser.add_argument("--output-json", type=Path, required=True, help="Destination JSON outcome file.")
    parser.add_argument("--output-tsv", type=Path, required=True, help="Destination human-readable TSV outcome file.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint label stored in the output.")
    parser.add_argument(
        "--diagnostics-jsonl",
        type=Path,
        default=None,
        help="Optional detailed per-episode diagnostics JSONL emitted by scripts/groot_eval.py.",
    )
    return parser.parse_args()


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def load_detailed_diagnostics(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            episode = int(row["episode"])
            if episode in rows:
                raise ValueError(f"Duplicate detailed diagnostic for episode {episode} in {path}:{line_number}.")
            rows[episode] = row
    return rows


def main() -> None:
    args = parse_args()
    results, parsed_episodes_jsonl = parse_results(args.log_path)
    episodes_jsonl = resolve_episodes_jsonl(args.log_path, parsed_episodes_jsonl, args.episodes_jsonl)
    specs = load_jsonl(episodes_jsonl)
    details_by_episode = parse_details(args.log_path)
    detailed_diagnostics = load_detailed_diagnostics(args.diagnostics_jsonl)

    if not results:
        raise ValueError(f"Found no episode result lines in {args.log_path}.")

    expected_numbers = list(range(1, len(specs) + 1))
    result_numbers = [result.episode for result in results]
    if result_numbers != expected_numbers:
        raise ValueError(
            "Evaluation did not produce exactly one ordered result for every requested episode: "
            f"expected {expected_numbers!r}, found {result_numbers!r}."
        )

    declared_totals = {result.total for result in results}
    if declared_totals != {len(specs)}:
        raise ValueError(
            f"Evaluation log declares total(s) {sorted(declared_totals)!r}, but {episodes_jsonl} has {len(specs)} rows."
        )

    episode_rows: list[dict[str, Any]] = []
    for result, spec in zip(results, specs, strict=True):
        diagnostics = details_by_episode.get(result.episode, {})
        detailed = detailed_diagnostics.get(result.episode)
        postmortem = (detailed or {}).get("postmortem") or {}
        episode_rows.append(
            {
                "episode": result.episode,
                "success": result.success,
                "outcome": "success" if result.success else "failure",
                "trial_id": spec.get("trial_id"),
                "task": infer_task(spec),
                "ood_key": str(spec.get("ood_key", "unknown")),
                "instruction": str(spec.get("instruction", "")),
                "objects": spec.get("objects", []),
                "n_objects": spec.get("n_objects"),
                "direction": spec.get("direction"),
                "termination_reason": diagnostics.get("termination_reason"),
                "episode_length_s": diagnostics.get("episode_length_s"),
                "failure_type": diagnostics.get("failure_type"),
                "live_failure_reason": diagnostics.get("live_failure_reason"),
                "target_object": diagnostics.get("target_object"),
                "target_lift_in": diagnostics.get("target_lift_in"),
                "lifted_wrong_object": diagnostics.get("lifted_wrong_object"),
                "max_distractor_lift_in": diagnostics.get("max_distractor_lift_in"),
                "lift_threshold_in": diagnostics.get("lift_threshold_in"),
                "classification_version": postmortem.get("classification_version"),
                "legacy_failure_type": postmortem.get("legacy_failure_type"),
                "diagnostic_confidence": postmortem.get("confidence"),
                "secondary_failure_types": postmortem.get("secondary_failure_types", []),
                "diagnostic_rationale": postmortem.get("rationale"),
                "target_attempt_count": postmortem.get("target_attempt_count"),
                "wrong_object_attempt_count": postmortem.get("wrong_object_attempt_count"),
                "unassociated_attempt_count": postmortem.get("unassociated_attempt_count"),
                "target_acquired": postmortem.get("target_acquired"),
                "wrong_object_acquired": postmortem.get("wrong_object_acquired"),
                "target_manipulated": postmortem.get("target_manipulated"),
                "target_max_displacement_m": postmortem.get("target_max_displacement_m"),
                "target_associated_transport_m": postmortem.get("target_associated_transport_m"),
                "target_post_acquisition_transport_m": postmortem.get(
                    "target_post_acquisition_transport_m"
                ),
                "target_drop_count": postmortem.get("target_drop_count"),
                "target_release_count": postmortem.get("target_release_count"),
                "goal_ever_reached": postmortem.get("goal_ever_reached"),
                "goal_ever_confirmed": postmortem.get("goal_ever_confirmed"),
                "max_goal_hold_steps": postmortem.get("max_goal_hold_steps"),
                "final_goal_met": postmortem.get("final_goal_met"),
                "attempts": postmortem.get("attempts", []),
                "evidence": postmortem.get("evidence", {}),
                "diagnostics": detailed,
            }
        )

    successes = sum(row["success"] for row in episode_rows)
    payload = {
        "checkpoint": args.checkpoint,
        "log_path": str(args.log_path.resolve()),
        "episodes_jsonl": str(episodes_jsonl),
        "diagnostics_jsonl": str(args.diagnostics_jsonl.resolve()) if args.diagnostics_jsonl else None,
        "total_episodes": len(episode_rows),
        "successes": successes,
        "failures": len(episode_rows) - successes,
        "success_rate": successes / len(episode_rows),
        "episodes": episode_rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "episode",
        "outcome",
        "trial_id",
        "task",
        "ood_key",
        "instruction",
        "objects",
        "n_objects",
        "direction",
        "termination_reason",
        "episode_length_s",
        "failure_type",
        "live_failure_reason",
        "target_object",
        "target_lift_in",
        "lifted_wrong_object",
        "max_distractor_lift_in",
        "lift_threshold_in",
        "classification_version",
        "legacy_failure_type",
        "diagnostic_confidence",
        "secondary_failure_types",
        "diagnostic_rationale",
        "target_attempt_count",
        "wrong_object_attempt_count",
        "unassociated_attempt_count",
        "target_acquired",
        "wrong_object_acquired",
        "target_manipulated",
        "target_max_displacement_m",
        "target_associated_transport_m",
        "target_post_acquisition_transport_m",
        "target_drop_count",
        "target_release_count",
        "goal_ever_reached",
        "goal_ever_confirmed",
        "max_goal_hold_steps",
        "final_goal_met",
        "attempts",
        "evidence",
        "diagnostics",
    ]
    with args.output_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in episode_rows:
            writer.writerow(
                {
                    "episode": row["episode"],
                    "outcome": row["outcome"],
                    "trial_id": row["trial_id"],
                    "task": row["task"],
                    "ood_key": row["ood_key"],
                    "instruction": row["instruction"],
                    "objects": json_cell(row["objects"]),
                    "n_objects": row["n_objects"],
                    "direction": row["direction"],
                    "termination_reason": row["termination_reason"],
                    "episode_length_s": row["episode_length_s"],
                    "failure_type": row["failure_type"],
                    "live_failure_reason": row["live_failure_reason"],
                    "target_object": row["target_object"],
                    "target_lift_in": row["target_lift_in"],
                    "lifted_wrong_object": row["lifted_wrong_object"],
                    "max_distractor_lift_in": row["max_distractor_lift_in"],
                    "lift_threshold_in": row["lift_threshold_in"],
                    "classification_version": row["classification_version"],
                    "legacy_failure_type": row["legacy_failure_type"],
                    "diagnostic_confidence": row["diagnostic_confidence"],
                    "secondary_failure_types": json_cell(row["secondary_failure_types"]),
                    "diagnostic_rationale": row["diagnostic_rationale"],
                    "target_attempt_count": row["target_attempt_count"],
                    "wrong_object_attempt_count": row["wrong_object_attempt_count"],
                    "unassociated_attempt_count": row["unassociated_attempt_count"],
                    "target_acquired": row["target_acquired"],
                    "wrong_object_acquired": row["wrong_object_acquired"],
                    "target_manipulated": row["target_manipulated"],
                    "target_max_displacement_m": row["target_max_displacement_m"],
                    "target_associated_transport_m": row["target_associated_transport_m"],
                    "target_post_acquisition_transport_m": row["target_post_acquisition_transport_m"],
                    "target_drop_count": row["target_drop_count"],
                    "target_release_count": row["target_release_count"],
                    "goal_ever_reached": row["goal_ever_reached"],
                    "goal_ever_confirmed": row["goal_ever_confirmed"],
                    "max_goal_hold_steps": row["max_goal_hold_steps"],
                    "final_goal_met": row["final_goal_met"],
                    "attempts": json_cell(row["attempts"]),
                    "evidence": json_cell(row["evidence"]),
                    "diagnostics": json_cell(row["diagnostics"]),
                }
            )

    print(
        f"Wrote {args.output_json} and {args.output_tsv}: "
        f"{successes}/{len(episode_rows)} successful ({100.0 * payload['success_rate']:.1f}%)."
    )


if __name__ == "__main__":
    main()
