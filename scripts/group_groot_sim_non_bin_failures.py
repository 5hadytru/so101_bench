#!/usr/bin/env python3
"""Group final sim non-bin failures into semantic, grasping, and placement buckets."""

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
DEFAULT_DIAGNOSTICS = (
    ROOT
    / "outputs/groot_sim_real_correspondence/checkpoint-20000/run_20260809T062956Z/episode_diagnostics.jsonl"
)
DEFAULT_OUTPUT = ROOT / "outputs/groot_sim_real_correspondence/final_correspondence_analysis"

# The rationale is deliberately label-specific so the grouping can be audited.
FAIL_TYPE_TAXONOMY: dict[str, tuple[str, str]] = {
    # Semantic: wrong task/entity/referent interpretation or no coherent task strategy.
    "nce": ("semantic", "Annotation shorthand for a semantic/entity-selection error."),
    "totally confused": ("semantic", "The policy exhibited no coherent interpretation of the requested task."),
    "totally confused (hovered above)": (
        "semantic",
        "Hovering without a coherent manipulation is treated as failure to interpret/execute the requested semantics.",
    ),
    "grasped other object": ("semantic", "The policy selected and grasped the wrong tabletop object."),
    "grasped class distractor object": (
        "semantic",
        "The policy confused the target with a same-class distractor.",
    ),
    "grasped color distractor object": (
        "semantic",
        "The policy confused the target with a color distractor.",
    ),
    "placed between other objects": (
        "semantic",
        "The target was related to the wrong pair of referent objects.",
    ),
    "placed next to other object": ("semantic", "The target was placed beside the wrong referent object."),
    "placed next to class distractor object": (
        "semantic",
        "The policy used a same-class distractor as the referent.",
    ),
    "placed next to color distractor object": (
        "semantic",
        "The policy used a color distractor as the referent.",
    ),
    "bad strategy": (
        "semantic",
        "The chosen high-level manipulation strategy was incompatible with the instruction, rather than a local grasp/placement miss.",
    ),

    # Grasping: acquisition, lift, retention, undocking, or release failed.
    "imprecise": (
        "grasping",
        "Per the annotation protocol, generic 'imprecise' denotes an imprecise failed grasp.",
    ),
    "refused to lift": ("grasping", "The object was not successfully lifted after the grasp interaction."),
    "refused to undock": (
        "grasping",
        "The grasp sequence did not complete because the gripper would not retract/undock.",
    ),
    "failed to undock": (
        "grasping",
        "The grasp sequence failed during gripper retraction/undocking.",
    ),
    "refused to release the object": ("grasping", "Release is part of successful gripper execution."),
    "refused to release": ("grasping", "Release is part of successful gripper execution."),
    "refused to drop": ("grasping", "The gripper failed to release/drop the transported object."),
    "stuck on bin": (
        "grasping",
        "The manipulator became stuck on the bin/fixture before completing the grasp sequence.",
    ),
    "making contact": (
        "grasping",
        "The gripper contacted the target but did not convert contact into a successful grasp.",
    ),

    # Placement: correct target semantics and grasp, but transport/final geometry failed.
    "placed on top": (
        "placement",
        "The target was transported to the referent region but ended on top rather than in the required relation.",
    ),
    "moved object": (
        "placement",
        "Per the requested convention, moving an object during execution is a transport/placement failure.",
    ),
    "moved objects": (
        "placement",
        "Per the requested convention, moving objects during execution is a transport/placement failure.",
    ),
    "not close": (
        "placement",
        "The correct target was transported but did not finish within the required spatial tolerance.",
    ),
    "moved boundary": (
        "placement",
        "The move task failed at its transport endpoint/boundary rather than object semantics or acquisition.",
    ),
    "stuck on top": (
        "placement",
        "The target reached the destination region but remained on top instead of settling correctly.",
    ),
    "moved slowly": (
        "placement",
        "The target was transported in the intended manipulation phase but insufficiently to satisfy the goal.",
    ),
    "rolled": (
        "placement",
        "The target became unstable/rolled during final transport or placement.",
    ),
}

BUCKET_ORDER = ("semantic", "grasping", "placement")
TASK_ORDER = ("next_to", "between", "move")
OOD_ORDER = ("seen", "unseen_seen_class", "unseen_unseen_class")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def infer_task(row: dict[str, Any]) -> str:
    objects = row.get("objects") or []
    if objects and isinstance(objects[0], dict):
        return "bin"
    instruction = str(row.get("instruction", "")).lower()
    if "plastic bin" in instruction:
        return "bin"
    if instruction.startswith("move "):
        return "move"
    if "between" in instruction:
        return "between"
    if "next to" in instruction:
        return "next_to"
    raise ValueError(f"Could not infer task: {row}")


def normalize_fail_types(row: dict[str, Any]) -> list[str]:
    raw = row.get("fail_type") or []
    if isinstance(raw, str):
        raw = [raw]
    return [
        part.strip().lower().rstrip(".")
        for value in raw
        for part in str(value).split(",")
        if part.strip()
    ]


def choose_bucket(categories: set[str]) -> str:
    """Use the earliest causal-stage error when labels span multiple stages."""
    return next(bucket for bucket in BUCKET_ORDER if bucket in categories)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_jsonl(args.results)
    diagnostics = {int(row["episode"]): row for row in load_jsonl(args.diagnostics)}
    classified: list[dict[str, Any]] = []
    observed_tokens: Counter[str] = Counter()
    token_tasks: dict[str, Counter[str]] = defaultdict(Counter)
    token_oods: dict[str, Counter[str]] = defaultdict(Counter)
    unknown_tokens: set[str] = set()

    for index, row in enumerate(results):
        task = infer_task(row)
        if task == "bin" or row.get("success") is True:
            continue
        tokens = normalize_fail_types(row)
        derived_from_diagnostics = False
        if tokens:
            categories = {FAIL_TYPE_TAXONOMY[token][0] for token in tokens if token in FAIL_TYPE_TAXONOMY}
            unknown_tokens.update(token for token in tokens if token not in FAIL_TYPE_TAXONOMY)
            if not categories:
                continue
            bucket = choose_bucket(categories)
            rationale = (
                f"Observed categories={sorted(categories)}; selected {bucket} using causal-stage precedence "
                "semantic > grasping > placement."
            )
        else:
            diagnostic = diagnostics.get(index + 1, {})
            postmortem = diagnostic.get("postmortem") or {}
            if postmortem.get("legacy_failure_type") != "placement":
                raise ValueError(f"Blank fail_type at row {index} lacks placement diagnostics: {postmortem}")
            bucket = "placement"
            categories = {"placement"}
            derived_from_diagnostics = True
            rationale = (
                "Human fail_type is blank; saved diagnostics classify timeout_during_confirmation/"
                "success_confirmation_breached after transiently reaching the goal, so this is placement."
            )

        ood = str(row.get("ood_key"))
        for token in tokens:
            observed_tokens[token] += 1
            token_tasks[token][task] += 1
            token_oods[token][ood] += 1
        classified.append(
            {
                "source_index": index,
                "episode": index + 1,
                "trial_id": row.get("trial_id"),
                "task": task,
                "ood_key": ood,
                "instruction": row.get("instruction"),
                "fail_type": row.get("fail_type") or [],
                "normalized_fail_types": tokens,
                "observed_buckets": sorted(categories),
                "failure_bucket": bucket,
                "derived_from_diagnostics": derived_from_diagnostics,
                "bucket_rationale": rationale,
            }
        )

    if unknown_tokens:
        raise ValueError(f"Unmapped non-bin fail_type tokens: {sorted(unknown_tokens)}")
    if len(classified) != 351:
        raise ValueError(f"Expected 351 failed non-bin rows, found {len(classified)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = args.output_dir / "sim_non_bin_failure_buckets.jsonl"
    with episode_path.open("w", encoding="utf-8") as stream:
        for row in classified:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    taxonomy_rows: list[dict[str, Any]] = []
    for token in sorted(observed_tokens, key=lambda value: (BUCKET_ORDER.index(FAIL_TYPE_TAXONOMY[value][0]), value)):
        bucket, rationale = FAIL_TYPE_TAXONOMY[token]
        taxonomy_rows.append(
            {
                "fail_type": token,
                "bucket": bucket,
                "rationale": rationale,
                "occurrences": observed_tokens[token],
                "by_task": dict(token_tasks[token]),
                "by_ood": dict(token_oods[token]),
            }
        )

    summary: dict[str, Any] = {
        "source": str(args.results.resolve()),
        "scope": "Failed next_to, between, and move episodes only; all plastic-bin tasks excluded.",
        "failed_episode_count": len(classified),
        "non_empty_fail_type_episode_count": sum(not row["derived_from_diagnostics"] for row in classified),
        "blank_fail_type_episode_count": sum(row["derived_from_diagnostics"] for row in classified),
        "multi_label_rule": (
            "When labels span buckets, assign the episode to the earliest causal stage: "
            "semantic > grasping > placement."
        ),
        "taxonomy": taxonomy_rows,
        "blank_resolution": [row for row in classified if row["derived_from_diagnostics"]],
    }

    def grouped_counts(key: str, order: tuple[str, ...]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for value in order:
            rows = [row for row in classified if row[key] == value]
            counts = Counter(row["failure_bucket"] for row in rows)
            output.append(
                {
                    key: value,
                    "failures": len(rows),
                    **{f"{bucket}_count": counts[bucket] for bucket in BUCKET_ORDER},
                    **{
                        f"{bucket}_fraction_of_failures": counts[bucket] / len(rows) if rows else None
                        for bucket in BUCKET_ORDER
                    },
                }
            )
        return output

    summary["by_ood"] = grouped_counts("ood_key", OOD_ORDER)
    summary["by_task"] = grouped_counts("task", TASK_ORDER)
    overall_counts = Counter(row["failure_bucket"] for row in classified)
    summary["overall"] = {
        "failures": len(classified),
        **{f"{bucket}_count": overall_counts[bucket] for bucket in BUCKET_ORDER},
        **{
            f"{bucket}_fraction_of_failures": overall_counts[bucket] / len(classified)
            for bucket in BUCKET_ORDER
        },
    }
    taxonomy_path = args.output_dir / "sim_non_bin_failure_taxonomy.json"
    taxonomy_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = args.output_dir / "sim_non_bin_failure_buckets_by_ood.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "ood_key",
            "failures",
            *[f"{bucket}_count" for bucket in BUCKET_ORDER],
            *[f"{bucket}_fraction_of_failures" for bucket in BUCKET_ORDER],
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary["by_ood"])

    report_lines = [
        "# Sim non-bin failure taxonomy",
        "",
        "Scope: failed `next_to`, `between`, and `move` episodes. Every plastic-bin episode is excluded.",
        "",
        "For multi-label rows, the episode bucket is the earliest causal stage: semantic > grasping > placement.",
        "",
        "## Bucket counts by OOD split",
        "",
        "| OOD split | Failures | Semantic | Grasping | Placement |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["by_ood"]:
        report_lines.append(
            f"| {row['ood_key']} | {row['failures']} | "
            f"{row['semantic_count']} ({100*row['semantic_fraction_of_failures']:.1f}%) | "
            f"{row['grasping_count']} ({100*row['grasping_fraction_of_failures']:.1f}%) | "
            f"{row['placement_count']} ({100*row['placement_fraction_of_failures']:.1f}%) |"
        )
    report_lines.extend(
        [
            "",
            "## Every observed fail_type decision",
            "",
            "| fail_type | Bucket | Occurrences | Decision rationale |",
            "|---|---|---:|---|",
        ]
    )
    for row in taxonomy_rows:
        report_lines.append(
            f"| `{row['fail_type']}` | {row['bucket']} | {row['occurrences']} | {row['rationale']} |"
        )
    report_lines.extend(
        [
            "",
            "## Blank annotation",
            "",
            "Episode 503 (source index 502, trial 519) has no human `fail_type`. Its saved diagnostics say "
            "`timeout_during_confirmation` / `success_confirmation_breached`: the target reached the between-goal "
            "transiently but did not hold it. It is included as placement and marked `derived_from_diagnostics=true`; "
            "the source JSONL is not modified.",
            "",
        ]
    )
    report_path = args.output_dir / "sim_non_bin_failure_taxonomy.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    plot_path = args.output_dir / "sim_non_bin_failure_buckets_by_ood.png"
    positions = np.arange(len(summary["by_ood"]))
    bottom = np.zeros(len(positions))
    colors = {"semantic": "#e45756", "grasping": "#72b7b2", "placement": "#f2cf5b"}
    fig, axis = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    for bucket in BUCKET_ORDER:
        values = np.array([row[f"{bucket}_fraction_of_failures"] for row in summary["by_ood"]])
        bars = axis.bar(positions, values, bottom=bottom, color=colors[bucket], label=bucket)
        for bar, value, base in zip(bars, values, bottom):
            if value >= 0.035:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{100*value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
        bottom += values
    axis.set_xticks(
        positions,
        [f"{row['ood_key']}\n(n={row['failures']})" for row in summary["by_ood"]],
    )
    axis.set_ylim(0, 1)
    axis.set_yticks(np.linspace(0, 1, 6), [f"{value:.0f}%" for value in np.linspace(0, 100, 6)])
    axis.set_ylabel("Fraction of failed non-bin episodes")
    axis.set_title("Sim non-bin failure modes by OOD split")
    axis.legend(loc="upper left")
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    print(f"classified failed non-bin episodes: {len(classified)}")
    print(f"overall buckets: {dict(overall_counts)}")
    print(f"wrote: {taxonomy_path}")
    print(f"wrote: {episode_path}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {report_path}")
    print(f"wrote: {plot_path}")


if __name__ == "__main__":
    main()
