#!/usr/bin/env python3
"""Bucket manually annotated sim failures for non-bin correspondence tasks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "logs/groot_sim_real_correspondence/real_sim_correspondence.jsonl"
DEFAULT_OUTPUT = (
    ROOT / "outputs/groot_sim_real_correspondence/final_correspondence_analysis/sim_failure_buckets"
)

TASK_ORDER = ["next_to", "between", "move"]
OOD_ORDER = ["seen", "unseen_seen_class", "unseen_unseen_class"]
BUCKET_ORDER = ["semantic", "grasping", "placement"]
BUCKET_PRIORITY = ["semantic", "grasping", "placement"]
BUCKET_COLORS = {"semantic": "#e45756", "grasping": "#72b7b2", "placement": "#f2cf5b"}

# Every observed non-bin token is enumerated. The reasons are written into the
# machine-readable rules and the generated report.
RULES: dict[str, dict[str, str]] = {
    "grasped class distractor object": {
        "bucket": "semantic",
        "reason": "The policy executed a grasp on a class distractor, so target-object grounding was wrong.",
    },
    "grasped color distractor object": {
        "bucket": "semantic",
        "reason": "The policy executed a grasp on a color distractor, so target-object grounding was wrong.",
    },
    "grasped other object": {
        "bucket": "semantic",
        "reason": "The policy grasped a non-target object; the failure is object selection rather than grasp mechanics.",
    },
    "placed next to class distractor object": {
        "bucket": "semantic",
        "reason": "The target was related to the wrong class referent, indicating incorrect relational grounding.",
    },
    "placed next to color distractor object": {
        "bucket": "semantic",
        "reason": "The target was related to the wrong color referent, indicating incorrect relational grounding.",
    },
    "placed next to other object": {
        "bucket": "semantic",
        "reason": "The next-to relation used the wrong referent object.",
    },
    "placed between other objects": {
        "bucket": "semantic",
        "reason": "The between relation used the wrong referent objects.",
    },
    "bad strategy": {
        "bucket": "grasping",
        "reason": "Explicit user rule: bad strategy is a grasping failure.",
    },
    "imprecise": {
        "bucket": "grasping",
        "reason": "Explicit user rule: imprecise denotes an imprecise failed grasp.",
    },
    "refused to lift": {
        "bucket": "grasping",
        "reason": "Explicit user rule: failure to lift is part of the grasp/lift chain.",
    },
    "refused to release": {
        "bucket": "grasping",
        "reason": "Explicit user rule: failure to release is a grasping-system failure.",
    },
    "refused to release the object": {
        "bucket": "grasping",
        "reason": "Explicit user rule: failure to release is a grasping-system failure.",
    },
    "refused to drop": {
        "bucket": "grasping",
        "reason": "Dropping is the release phase of grasping; the object was not released.",
    },
    "stuck on bin": {
        "bucket": "grasping",
        "reason": "The manipulation/gripper became stuck on the bin before a valid transport-and-place completion.",
    },
    "failed to undock": {
        "bucket": "grasping",
        "reason": "Unlike the explicitly ignored 'refused to undock', this records a mechanical undocking failure in the grasp/release chain.",
    },
    "making contact": {
        "bucket": "grasping",
        "reason": "The gripper contacted the target but did not secure a successful grasp.",
    },
    "moved object": {
        "bucket": "placement",
        "reason": "Explicit user rule: moved object is a transport/placement failure.",
    },
    "moved objects": {
        "bucket": "placement",
        "reason": "Plural form of moved object; transport occurred but the final object configuration was invalid.",
    },
    "nce": {
        "bucket": "placement",
        "reason": "Explicit user rule: NCE is a placement failure.",
    },
    "not close": {
        "bucket": "placement",
        "reason": "Explicit user rule: the target was transported but not placed close enough.",
    },
    "placed on top": {
        "bucket": "placement",
        "reason": "Explicit user rule: the target was grasped and transported, but placed in the wrong relation.",
    },
    "stuck on top": {
        "bucket": "placement",
        "reason": "The target reached the placement area but remained on top rather than achieving the requested relation.",
    },
    "moved boundary": {
        "bucket": "placement",
        "reason": "Movement occurred, but the final displacement/boundary outcome was invalid; treat this as transport/placement.",
    },
    "moved slowly": {
        "bucket": "placement",
        "reason": "The correct movement began but did not transport the target far enough before timeout.",
    },
    "rolled": {
        "bucket": "placement",
        "reason": "The target moved, but rolling produced an invalid final pose/relation.",
    },
    "refused to undock": {
        "bucket": "ignored",
        "reason": "Explicit user instruction: ignore refused to undock.",
    },
    "totally confused": {
        "bucket": "ignored",
        "reason": "Explicit user instruction: ignore totally confused.",
    },
    "totally confused (hovered above)": {
        "bucket": "ignored",
        "reason": "Variant of totally confused; ignored consistently with the explicit instruction.",
    },
    "totally confused after going ood": {
        "bucket": "ignored",
        "reason": "Variant of totally confused; ignored consistently with the explicit instruction.",
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def infer_task(row: dict[str, Any]) -> str | None:
    instruction = str(row.get("instruction", "")).lower()
    if row.get("direction") is not None or instruction.startswith("move "):
        return "move"
    if "between" in instruction:
        return "between"
    if "next to" in instruction:
        return "next_to"
    return None


def normalized_tokens(row: dict[str, Any]) -> list[str]:
    raw = row.get("fail_type") or []
    if isinstance(raw, str):
        raw = [raw]
    tokens: list[str] = []
    for value in raw:
        tokens.extend(part.strip().lower().rstrip(".") for part in str(value).split(",") if part.strip())
    return tokens


def classify_failure(row: dict[str, Any], episode_index: int) -> dict[str, Any]:
    task = infer_task(row)
    if task is None:
        raise ValueError("classify_failure called on a bin or unknown task")
    tokens = normalized_tokens(row)
    unknown = sorted({token for token in tokens if token not in RULES})
    if unknown:
        raise ValueError(f"Unmapped token(s) at episode index {episode_index}: {unknown}")

    ignored = [token for token in tokens if RULES[token]["bucket"] == "ignored"]
    used = [token for token in tokens if RULES[token]["bucket"] != "ignored"]
    candidates = sorted({RULES[token]["bucket"] for token in used}, key=BUCKET_PRIORITY.index)
    if candidates:
        final_bucket = candidates[0]
        status = "classified"
        resolution = (
            f"single bucket: {final_bucket}"
            if len(candidates) == 1
            else f"multiple buckets {candidates}; selected {final_bucket} by semantic > grasping > placement"
        )
    elif tokens:
        final_bucket = None
        status = "ignored"
        resolution = "all labels were explicitly ignored"
    else:
        final_bucket = None
        status = "unclassified"
        resolution = "failure has a blank fail_type"

    return {
        "episode_index": episode_index,
        "episode": episode_index + 1,
        "trial_id": row.get("trial_id"),
        "task": task,
        "ood_key": row.get("ood_key"),
        "instruction": row.get("instruction"),
        "objects": row.get("objects"),
        "raw_fail_type": row.get("fail_type") or [],
        "normalized_tokens": tokens,
        "used_tokens": used,
        "ignored_tokens": ignored,
        "candidate_buckets": candidates,
        "bucket": final_bucket,
        "status": status,
        "resolution": resolution,
    }


def aggregate(classified: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in classified:
        groups[tuple(str(row[key]) for key in keys)].append(row)

    output: list[dict[str, Any]] = []
    for key, rows in groups.items():
        counts = Counter(row["bucket"] for row in rows if row["bucket"] is not None)
        status = Counter(row["status"] for row in rows)
        denominator = sum(counts.values())
        output.append(
            {
                **dict(zip(keys, key)),
                "total_failures": len(rows),
                "classified_failures": denominator,
                "ignored_failures": status["ignored"],
                "unclassified_failures": status["unclassified"],
                **{f"{bucket}_count": counts[bucket] for bucket in BUCKET_ORDER},
                **{
                    f"{bucket}_share_of_classified": counts[bucket] / denominator if denominator else None
                    for bucket in BUCKET_ORDER
                },
            }
        )

    def sort_key(row: dict[str, Any]) -> tuple[int, ...]:
        values: list[int] = []
        if "task" in row:
            values.append(TASK_ORDER.index(row["task"]))
        if "ood_key" in row:
            values.append(OOD_ORDER.index(row["ood_key"]))
        return tuple(values)

    return sorted(output, key=sort_key)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def markdown_counts(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> str:
    first_headers = ["Task" if key == "task" else "OOD split" for key in keys]
    lines = [
        "| " + " | ".join(first_headers + ["All failures", "Classified", "Semantic", "Grasping", "Placement", "Ignored", "Blank"]) + " |",
        "|" + "---|" * len(first_headers) + "---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        labels = [str(row[key]) for key in keys]
        cells = [
            str(row["total_failures"]),
            str(row["classified_failures"]),
            f"{row['semantic_count']} ({pct(row['semantic_share_of_classified'])})",
            f"{row['grasping_count']} ({pct(row['grasping_share_of_classified'])})",
            f"{row['placement_count']} ({pct(row['placement_share_of_classified'])})",
            str(row["ignored_failures"]),
            str(row["unclassified_failures"]),
        ]
        lines.append("| " + " | ".join(labels + cells) + " |")
    return "\n".join(lines)


def save_plot(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [f"{row['task']}\n{row['ood_key']}" for row in rows]
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    fig, axis = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    for bucket in BUCKET_ORDER:
        values = np.asarray([row[f"{bucket}_share_of_classified"] or 0 for row in rows])
        axis.bar(x, values, bottom=bottom, color=BUCKET_COLORS[bucket], label=bucket)
        bottom += values
    for index, row in enumerate(rows):
        axis.text(index, 1.02, f"n={row['classified_failures']}\nignored={row['ignored_failures']}", ha="center", va="bottom", fontsize=8)
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylim(0, 1.16)
    axis.set_yticks(np.linspace(0, 1, 6), [f"{value:.0f}%" for value in np.linspace(0, 100, 6)])
    axis.set_ylabel("Share of classified failures")
    axis.set_title("Sim non-bin failure buckets by task and OOD")
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_report(
    input_path: Path,
    classified: list[dict[str, Any]],
    by_task_ood: list[dict[str, Any]],
    by_ood: list[dict[str, Any]],
    token_counts: Counter[str],
) -> str:
    ignored = sum(row["status"] == "ignored" for row in classified)
    blank = sum(row["status"] == "unclassified" for row in classified)
    conflicts = sum(len(row["candidate_buckets"]) > 1 for row in classified)
    rule_lines = ["| Normalized fail type | Bucket | Non-bin occurrences | Rationale |", "|---|---|---:|---|"]
    for token, rule in sorted(RULES.items(), key=lambda item: (BUCKET_ORDER.index(item[1]["bucket"]) if item[1]["bucket"] in BUCKET_ORDER else 3, item[0])):
        if token_counts[token] == 0:
            continue
        rule_lines.append(f"| `{token}` | {rule['bucket']} | {token_counts[token]} | {rule['reason']} |")
    rules_table = "\n".join(rule_lines)
    return f"""# Sim non-bin failure buckets

Source: `{input_path}`

This analysis includes failed next-to, between, and move episodes only. It found {len(classified)} failures: {len(classified)-ignored-blank} classified into semantic/grasping/placement, {ignored} excluded because every label was explicitly ignored, and {blank} with an empty `fail_type`.

Percentages below use **classified failures as the denominator**. Ignored and blank rows remain visible in separate columns.

## By task and OOD

{markdown_counts(by_task_ood, ('task', 'ood_key'))}

## Pooled by OOD

{markdown_counts(by_ood, ('ood_key',))}

## Classification logic

- Semantic: the policy selected the wrong target/referent or otherwise executed the relation using the wrong objects.
- Grasping: the grasp, lift, handling, or release chain failed. Per the requested rules, `bad strategy` and generic `imprecise` are here.
- Placement: the policy understood the task and achieved enough manipulation to attempt transport/placement, but the final displacement or relation was wrong. Per the requested rules, `moved object`, `nce`, `not close`, and `placed on top` are here.
- Ignored: `refused to undock` and all `totally confused` variants.
- Multi-label resolution: semantic > grasping > placement. Placement requires both correct semantics and a successful grasp; grasping requires correct semantics. This affected {conflicts} multi-bucket rows. Every row preserves its candidate buckets and final resolution in `classified_failure_rows.jsonl`.

## Every fail-type decision

{rules_table}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    failures = [
        classify_failure(row, index)
        for index, row in enumerate(rows)
        if infer_task(row) in TASK_ORDER and row.get("success") is not True
    ]
    if not failures:
        raise ValueError("Found no failed non-bin rows")
    token_counts = Counter(token for row in failures for token in row["normalized_tokens"])
    observed_tokens = set(token_counts)
    missing_rules = observed_tokens - set(RULES)
    if missing_rules:
        raise ValueError(f"Observed fail types lack rules: {sorted(missing_rules)}")

    by_task_ood = aggregate(failures, ("task", "ood_key"))
    by_ood = aggregate(failures, ("ood_key",))
    by_task = aggregate(failures, ("task",))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "classified_failure_rows.jsonl", failures)
    write_csv(args.output_dir / "failure_buckets_by_task_ood.csv", by_task_ood)
    write_csv(args.output_dir / "failure_buckets_by_ood.csv", by_ood)
    write_csv(args.output_dir / "failure_buckets_by_task.csv", by_task)
    (args.output_dir / "bucket_rules.json").write_text(
        json.dumps(
            {
                "bucket_priority": BUCKET_PRIORITY,
                "rules": RULES,
                "percentage_denominator": "classified failures only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        build_report(args.input.resolve(), failures, by_task_ood, by_ood, token_counts), encoding="utf-8"
    )
    save_plot(by_task_ood, args.output_dir / "failure_buckets_by_task_ood.png")
    print(f"non-bin failures: {len(failures)}")
    print(f"classified: {sum(row['bucket'] is not None for row in failures)}")
    print(f"ignored: {sum(row['status'] == 'ignored' for row in failures)}")
    print(f"blank/unclassified: {sum(row['status'] == 'unclassified' for row in failures)}")
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
