#!/usr/bin/env python3
"""Bucket final simulated non-bin failures by OOD split and failure stage."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "logs/groot_sim_real_correspondence/real_sim_correspondence.jsonl"
DEFAULT_TASKS = (
    ROOT
    / "outputs/groot_sim_real_correspondence/checkpoint-20000/run_20260809T062956Z/tasks.jsonl"
)
DEFAULT_DIAGNOSTICS = (
    ROOT
    / "outputs/groot_sim_real_correspondence/checkpoint-20000/run_20260809T062956Z/episode_diagnostics.jsonl"
)
DEFAULT_OUTPUT = ROOT / "outputs/groot_sim_real_correspondence/final_correspondence_analysis"

OOD_ORDER = ["seen", "unseen_seen_class", "unseen_unseen_class"]
TASK_ORDER = ["next_to", "between", "move"]
BUCKET_ORDER = ["semantic", "grasping", "placement"]
BUCKET_COLORS = {"semantic": "#e45756", "grasping": "#72b7b2", "placement": "#f2cf5b"}

# Every normalized token observed on a failed non-bin sim episode is mapped
# explicitly. Generic "imprecise", "bad strategy", lift failures, and release
# failures follow the user's annotation definitions rather than older plotting
# heuristics.
FAILURE_MAP: dict[str, dict[str, str]] = {
    "nce": {
        "bucket": "semantic",
        "reason": "The established annotation convention treats NCE as a semantic/object-selection error; the acronym is left unexpanded.",
    },
    "totally confused": {
        "bucket": "semantic",
        "reason": "The behavior shows no coherent understanding of the requested object relation or motion.",
    },
    "totally confused (hovered above)": {
        "bucket": "semantic",
        "reason": "Hovering without a coherent attempt indicates failure to select/execute the task semantics.",
    },
    "grasped other object": {
        "bucket": "semantic",
        "reason": "The policy selected the wrong target object, so the error precedes grasp execution quality.",
    },
    "grasped class distractor object": {
        "bucket": "semantic",
        "reason": "The policy confused the requested target with a same-class distractor.",
    },
    "grasped color distractor object": {
        "bucket": "semantic",
        "reason": "The policy confused the requested target with a same-color distractor.",
    },
    "placed between other objects": {
        "bucket": "semantic",
        "reason": "The policy used the wrong referent objects for the between relation.",
    },
    "placed next to other object": {
        "bucket": "semantic",
        "reason": "The policy used the wrong referent for the next-to relation.",
    },
    "placed next to class distractor object": {
        "bucket": "semantic",
        "reason": "The policy semantically confused the requested referent with a class distractor.",
    },
    "placed next to color distractor object": {
        "bucket": "semantic",
        "reason": "The policy semantically confused the requested referent with a color distractor.",
    },
    "imprecise": {
        "bucket": "grasping",
        "reason": "Per the annotation protocol, generic imprecise denotes an imprecise failed grasp.",
    },
    "bad strategy": {
        "bucket": "grasping",
        "reason": "Per the requested grouping, bad manipulation/grasp strategy is a grasping failure.",
    },
    "refused to lift": {
        "bucket": "grasping",
        "reason": "The object was not successfully lifted after the grasp attempt.",
    },
    "refused to undock": {
        "bucket": "grasping",
        "reason": "The gripper did not complete the post-grasp undocking/lift phase.",
    },
    "failed to undock": {
        "bucket": "grasping",
        "reason": "The grasp sequence failed during post-contact undocking.",
    },
    "refused to release the object": {
        "bucket": "grasping",
        "reason": "Release is part of grasp/manipulation execution and explicitly belongs in grasping.",
    },
    "refused to release": {
        "bucket": "grasping",
        "reason": "Release is part of grasp/manipulation execution and explicitly belongs in grasping.",
    },
    "refused to drop": {
        "bucket": "grasping",
        "reason": "The gripper failed to release/drop the held object.",
    },
    "stuck on bin": {
        "bucket": "grasping",
        "reason": "The manipulation became mechanically stuck on the bin rather than completing grasp/release execution.",
    },
    "making contact": {
        "bucket": "grasping",
        "reason": "The policy reached/contacted the object but did not establish a successful grasp.",
    },
    "moved object": {
        "bucket": "placement",
        "reason": "Per the requested grouping, it manipulated the target but failed the required transport/placement.",
    },
    "moved objects": {
        "bucket": "placement",
        "reason": "Manipulation/transport disturbed objects and failed to produce the required final placement.",
    },
    "moved boundary": {
        "bucket": "placement",
        "reason": "Transport ended at or violated the spatial boundary rather than the required destination.",
    },
    "moved slowly": {
        "bucket": "placement",
        "reason": "The target was manipulated but transport magnitude/timing was insufficient to reach the required placement.",
    },
    "placed on top": {
        "bucket": "placement",
        "reason": "The correct manipulation reached the destination area but produced the wrong precise spatial relation.",
    },
    "stuck on top": {
        "bucket": "placement",
        "reason": "The object was transported but remained on top of an object instead of reaching the required placement.",
    },
    "not close": {
        "bucket": "placement",
        "reason": "The object was transported but its final position was outside the required proximity tolerance.",
    },
    "rolled": {
        "bucket": "placement",
        "reason": "The object was manipulated but rolled away from the required final pose.",
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def task_name(task: dict[str, Any]) -> str:
    instruction = str(task.get("instruction", "")).lower()
    if instruction.startswith("place each object in the plastic bin"):
        return "bin"
    if instruction.startswith("move "):
        return "move"
    if "between" in instruction:
        return "between"
    if "next to" in instruction:
        return "next_to"
    raise ValueError(f"Unknown task instruction: {task}")


def normalized_tokens(row: dict[str, Any]) -> list[str]:
    raw = row.get("fail_type") or []
    if isinstance(raw, str):
        raw = [raw]
    return [
        token
        for value in raw
        for token in (part.strip().lower().rstrip(".") for part in str(value).split(","))
        if token
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def primary_bucket(buckets: set[str]) -> str | None:
    # Placement is defined as successful semantics + successful grasp followed
    # by failed transport/precision. Therefore an observed semantic or grasping
    # error takes precedence over a downstream placement symptom.
    return next((bucket for bucket in BUCKET_ORDER if bucket in buckets), None)


def build_assignments(
    tasks: list[dict[str, Any]],
    results: list[dict[str, Any]],
    diagnostics: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    if len(tasks) != len(results):
        raise ValueError(f"Task/result length mismatch: {len(tasks)} != {len(results)}")
    assignments: list[dict[str, Any]] = []
    unknown: set[str] = set()
    for index, (task, result) in enumerate(zip(tasks, results)):
        task_type = task_name(task)
        if task_type == "bin" or result.get("success") is True:
            continue
        tokens = normalized_tokens(result)
        mappings: list[dict[str, str]] = []
        diagnostic_evidence: dict[str, Any] | None = None
        if tokens:
            for token in tokens:
                mapping = FAILURE_MAP.get(token)
                if mapping is None:
                    unknown.add(token)
                    continue
                mappings.append({"fail_type": token, **mapping})
            buckets = {mapping["bucket"] for mapping in mappings}
            bucket = primary_bucket(buckets)
            classification_source = "human_fail_type"
        else:
            diagnostic = diagnostics.get(index + 1, {})
            postmortem = diagnostic.get("postmortem") or {}
            if postmortem.get("legacy_failure_type") != "placement":
                raise ValueError(
                    f"Failed non-bin episode {index + 1} has no fail_type and no placement diagnostic"
                )
            bucket = "placement"
            buckets = {"placement"}
            classification_source = "saved_diagnostics"
            diagnostic_evidence = {
                "termination_reason": diagnostic.get("termination_reason"),
                "live_failure_reason": diagnostic.get("live_failure_reason"),
                "postmortem_failure_type": postmortem.get("failure_type"),
                "postmortem_legacy_failure_type": postmortem.get("legacy_failure_type"),
                "goal_ever_reached": postmortem.get("goal_ever_reached"),
                "max_goal_hold_steps": postmortem.get("max_goal_hold_steps"),
                "rationale": postmortem.get("rationale"),
            }
        assignments.append(
            {
                "episode_index": index,
                "episode": index + 1,
                "trial_id": task.get("trial_id"),
                "task": task_type,
                "ood_key": task.get("ood_key"),
                "instruction": task.get("instruction"),
                "objects": task.get("objects"),
                "original_fail_type": result.get("fail_type") or [],
                "normalized_fail_types": tokens,
                "mapped_fail_types": mappings,
                "buckets_present": [name for name in BUCKET_ORDER if name in buckets],
                "primary_bucket": bucket,
                "primary_bucket_rule": "semantic > grasping > placement",
                "classification_source": classification_source,
                "human_fail_type_missing": not tokens,
                "diagnostic_evidence": diagnostic_evidence,
            }
        )
    return assignments, sorted(unknown)


def summarize(
    tasks: list[dict[str, Any]], results: list[dict[str, Any]], assignments: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    def one(task_filter: str | None, ood_filter: str | None) -> dict[str, Any]:
        eligible = [
            result
            for task, result in zip(tasks, results)
            if task_name(task) != "bin"
            and (task_filter is None or task_name(task) == task_filter)
            and (ood_filter is None or task.get("ood_key") == ood_filter)
        ]
        failures = [
            row
            for row in assignments
            if (task_filter is None or row["task"] == task_filter)
            and (ood_filter is None or row["ood_key"] == ood_filter)
        ]
        counts = Counter(row["primary_bucket"] for row in failures)
        summary: dict[str, Any] = {
            "task": task_filter or "all_non_bin",
            "ood_key": ood_filter or "all_ood",
            "non_bin_episodes": len(eligible),
            "successes": sum(row.get("success") is True for row in eligible),
            "failures": len(failures),
            "diagnostics_derived_failures": sum(
                row["classification_source"] == "saved_diagnostics" for row in failures
            ),
            "multi_bucket_failures": sum(len(row["buckets_present"]) > 1 for row in failures),
        }
        for bucket in BUCKET_ORDER:
            summary[f"{bucket}_failures"] = counts[bucket]
            summary[f"{bucket}_share_of_all_failures"] = counts[bucket] / len(failures)
        return summary

    return {
        "by_task_ood": [one(task, ood) for task in TASK_ORDER for ood in OOD_ORDER],
        "by_ood": [one(None, ood) for ood in OOD_ORDER],
        "by_task": [one(task, None) for task in TASK_ORDER],
        "overall": [one(None, None)],
    }


def save_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def save_plot(path: Path, summaries: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(15, 5.3), sharey=True, constrained_layout=True)
    positions = np.arange(len(OOD_ORDER))
    for axis, task in zip(axes, TASK_ORDER):
        task_rows = [row for row in summaries if row["task"] == task]
        bottom = np.zeros(len(task_rows))
        for bucket in BUCKET_ORDER:
            values = np.array([row[f"{bucket}_share_of_all_failures"] for row in task_rows])
            axis.bar(
                positions,
                values,
                bottom=bottom,
                label=bucket,
                color=BUCKET_COLORS[bucket],
            )
            for position, value, base in zip(positions, values, bottom):
                if value >= 0.06:
                    axis.text(
                        position,
                        base + value / 2,
                        f"{100*value:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )
            bottom += values
        axis.set_xticks(
            positions,
            [f"{row['ood_key']}\n(n={row['failures']})" for row in task_rows],
            rotation=17,
            ha="right",
        )
        axis.set_title(task.replace("_", " "))
        axis.set_ylim(0, 1)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].set_yticks(np.linspace(0, 1, 6), [f"{value:.0f}%" for value in np.linspace(0, 100, 6)])
    axes[0].set_ylabel("Share of failed episodes")
    axes[-1].legend(title="Primary bucket", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("Sim non-bin failure composition by task and OOD split")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_markdown(
    summaries: dict[str, list[dict[str, Any]]], token_counts: Counter[str]
) -> str:
    lines = [
        "# Sim non-bin failure buckets by task and OOD split",
        "",
        "Only `next_to`, `between`, and `move` episodes are included. Bin tasks are excluded. Each failed episode receives one primary bucket using `semantic > grasping > placement`: placement requires that task semantics and grasp execution succeeded, so an upstream semantic or grasping label takes precedence when labels span stages.",
        "",
        "| Task | OOD split | Episodes | Failures | Semantic | Grasping | Placement | Mixed-stage labels |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries["by_task_ood"]:
        lines.append(
            f"| `{row['task']}` | `{row['ood_key']}` | {row['non_bin_episodes']} | {row['failures']} "
            f"| {row['semantic_failures']} ({100*row['semantic_share_of_all_failures']:.1f}%) "
            f"| {row['grasping_failures']} ({100*row['grasping_share_of_all_failures']:.1f}%) "
            f"| {row['placement_failures']} ({100*row['placement_share_of_all_failures']:.1f}%) "
            f"| {row['multi_bucket_failures']} |"
        )
    lines.extend(
        [
            "",
            "Percentages use all failures in that task×OOD cell. The single blank human annotation is classified from its already-saved diagnostic record, so all 351 failures are assigned.",
            "",
            "## Pooled by OOD split",
            "",
            "| OOD split | Failures | Semantic | Grasping | Placement |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summaries["by_ood"]:
        lines.append(
            f"| `{row['ood_key']}` | {row['failures']} "
            f"| {row['semantic_failures']} ({100*row['semantic_share_of_all_failures']:.1f}%) "
            f"| {row['grasping_failures']} ({100*row['grasping_share_of_all_failures']:.1f}%) "
            f"| {row['placement_failures']} ({100*row['placement_share_of_all_failures']:.1f}%) |"
        )
    lines.extend(
        [
            "",
            "## Mapping decisions",
            "",
            "| Normalized fail type | Occurrences | Bucket | Decision |",
            "|---|---:|---|---|",
        ]
    )
    for token in sorted(token_counts, key=lambda value: (BUCKET_ORDER.index(FAILURE_MAP[value]["bucket"]), value)):
        mapping = FAILURE_MAP[token]
        lines.append(f"| `{token}` | {token_counts[token]} | {mapping['bucket']} | {mapping['reason']} |")
    lines.extend(
        [
            "",
            "One failed seen `between` episode (episode 503, trial 519) has an empty human `fail_type`. Its existing diagnostics say `success_confirmation_breached` / `timeout_during_confirmation`, `goal_ever_reached=true`, and legacy failure type `placement`; it is therefore classified as placement without modifying the source JSONL.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_jsonl(args.tasks)
    results = load_jsonl(args.results)
    diagnostics = {int(row["episode"]): row for row in load_jsonl(args.diagnostics)}
    assignments, unknown = build_assignments(tasks, results, diagnostics)
    if unknown:
        raise ValueError(f"Unmapped non-bin failure labels: {unknown}")
    summaries = summarize(tasks, results, assignments)
    token_counts = Counter(
        mapping["fail_type"] for row in assignments for mapping in row["mapped_fail_types"]
    )
    observed_tokens = set(token_counts)
    unused_mapping = sorted(set(FAILURE_MAP) - observed_tokens)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "sim_nonbin_failure_assignments.jsonl", assignments)
    save_csv(args.output_dir / "sim_nonbin_failure_buckets_by_task_ood.csv", summaries["by_task_ood"])
    save_csv(args.output_dir / "sim_nonbin_failure_buckets_by_ood.csv", summaries["by_ood"])
    (args.output_dir / "sim_nonbin_failure_buckets.md").write_text(
        build_markdown(summaries, token_counts), encoding="utf-8"
    )
    (args.output_dir / "sim_nonbin_failure_taxonomy.json").write_text(
        json.dumps(
            {
                "scope": "failed sim next_to, between, and move episodes only",
                "normalization": "split commas, strip whitespace, lowercase, strip trailing periods",
                "primary_bucket_priority": BUCKET_ORDER,
                "primary_bucket_rationale": (
                    "Placement requires successful semantics and grasping; upstream semantic/grasping errors take precedence."
                ),
                "mapping": FAILURE_MAP,
                "observed_tokens": sorted(observed_tokens),
                "unused_mapping_tokens": unused_mapping,
                "human_fail_type_missing_count": sum(row["human_fail_type_missing"] for row in assignments),
                "summary_by_task_ood": summaries["by_task_ood"],
                "summary_by_ood": summaries["by_ood"],
                "summary_by_task": summaries["by_task"],
                "overall": summaries["overall"][0],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    save_plot(args.output_dir / "sim_nonbin_failure_buckets_by_task_ood.png", summaries["by_task_ood"])
    print(f"non-bin failed episodes: {len(assignments)}")
    for row in summaries["by_task_ood"]:
        print(
            f"{row['task']}/{row['ood_key']}: semantic={row['semantic_failures']} grasping={row['grasping_failures']} "
            f"placement={row['placement_failures']} diagnostics_derived={row['diagnostics_derived_failures']} "
            f"total={row['failures']}"
        )
    print(f"wrote: {args.output_dir / 'sim_nonbin_failure_buckets.md'}")


if __name__ == "__main__":
    main()
