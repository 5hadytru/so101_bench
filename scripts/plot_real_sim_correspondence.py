#!/usr/bin/env python3
"""Plot real-vs-sim correspondence for GR00T subset evaluation results."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPISODE_RESULT_RE = re.compile(
    r"Episode (?P<episode>\d+)/(?P<total>\d+): success=(?P<success>True|False), reason=(?P<reason>[^,]+)"
)

TASK_ORDER = ["bin_1obj", "bin_4obj", "named_bin", "next_to", "between", "move"]
OOD_ORDER = ["seen", "unseen_seen_class", "unseen_unseen_class"]
FAILURE_ORDER = ["semantic", "grasping", "placement"]

TASK_COLORS = {
    "bin_1obj": "#4c78a8",
    "bin_4obj": "#f58518",
    "named_bin": "#59a14f",
    "next_to": "#54a24b",
    "between": "#b279a2",
    "move": "#e45756",
}
DOMAIN_COLORS = {"sim": "#4c78a8", "real": "#f58518"}
FAILURE_COLORS = {"semantic": "#e45756", "grasping": "#72b7b2", "placement": "#f2cf5b"}
OOD_MARKERS = {"seen": "o", "unseen_seen_class": "s", "unseen_unseen_class": "^"}

REAL_FAILURE_MAP = {
    "NCE": "semantic",
    "NSE": "semantic",
    "TCR": "semantic",
    "grasped class distractor object": "semantic",
    "grasped color distractor object": "semantic",
    "grasped other object": "semantic",
    "moved boundary": "semantic",
    "moved object": "semantic",
    "moved objects": "semantic",
    "moved past boundary": "semantic",
    "occluded": "semantic",
    "placed next to class distractor object": "semantic",
    "placed next to color distractor object": "semantic",
    "placed next to other object": "semantic",
    "bad strategy": "semantic",
    "camera stuck on bin": "grasping",
    "docked with object in grip": "grasping",
    "grabbed air": "grasping",
    "grabbed air indefinitely": "grasping",
    "refused to dock": "grasping",
    "refused to drop": "grasping",
    "refused to lift": "grasping",
    "refused to undock": "grasping",
    "repeatedly reached and retracted": "grasping",
    "throttled gripper": "grasping",
    "imprecise": "placement",
    "imprecise by 1cm": "placement",
    "imprecise by 2cm": "placement",
    "imprecise by 3cm": "placement",
    "knocked bin": "placement",
    "missed bin": "placement",
    "moved bin": "placement",
    "not close": "placement",
    "not close enough": "placement",
    "not fully in bin": "placement",
    "placed on top": "placement",
    "stuck on top": "placement",
}


@dataclass
class SimResult:
    episode: int
    total: int
    success: bool
    reason: str
    failure_type: str | None


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


def parse_sim_log(path: Path) -> dict[int, SimResult]:
    results: dict[int, SimResult] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            result_match = EPISODE_RESULT_RE.search(line)
            if result_match is None:
                continue
            failure_match = re.search(r"failure_type=([^,\] ]+)", line)
            episode = int(result_match.group("episode"))
            results[episode] = SimResult(
                episode=episode,
                total=int(result_match.group("total")),
                success=result_match.group("success") == "True",
                reason=result_match.group("reason"),
                failure_type=failure_match.group(1) if failure_match is not None else None,
            )
    return results


def real_bin_success(row: dict[str, Any]) -> bool:
    object_results = row.get("objects", [])
    if not isinstance(object_results, list) or not object_results:
        raise ValueError(f"Expected non-empty object result list for bin row: {row}")
    return all(isinstance(obj, dict) and obj.get("success") is True for obj in object_results)


def build_real_indices(
    real_if_rows: list[dict[str, Any]],
    real_bin_rows: list[dict[str, Any]],
) -> tuple[dict[tuple[tuple[str, ...], str, str], deque[dict[str, Any]]], dict[tuple[str, int, tuple[str, ...]], deque[dict[str, Any]]]]:
    if_index: dict[tuple[tuple[str, ...], str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in real_if_rows:
        if_index[(sim_object_names(row), str(row.get("instruction", "")), str(row.get("ood_key", "unknown")))].append(row)

    bin_index: dict[tuple[str, int, tuple[str, ...]], deque[dict[str, Any]]] = defaultdict(deque)
    for row in real_bin_rows:
        bin_index[(str(row.get("ood_key", "unknown")), int(row.get("n_objects", 0)), real_bin_object_names(row))].append(row)

    return if_index, bin_index


def classify_sim_failure(result: SimResult, task: str) -> str | None:
    if result.success:
        return None
    if result.failure_type == "semantic":
        return "semantic"
    if result.failure_type == "placement":
        return "placement"
    if result.failure_type in {"failed_grasp", "grasping"}:
        return "grasping"
    if result.reason in {"non_target_moved", "wrong_object"}:
        return "semantic"
    if result.reason in {"bin_displaced", "success_confirmation_breached"}:
        return "placement"
    if result.reason == "time_out" and (task.startswith("bin") or task == "named_bin"):
        return "grasping"
    if result.reason == "time_out":
        return "placement"
    raise ValueError(f"Unknown sim failure type: failure_type={result.failure_type}, reason={result.reason}")


def classify_real_failure(row: dict[str, Any], task: str, success: bool) -> str | None:
    if success:
        return None

    categories: set[str] = set()
    for fail_type in row.get("fail_type") or []:
        if fail_type not in REAL_FAILURE_MAP:
            raise ValueError(f"Unmapped real fail_type={fail_type!r} in row: {row}")
        categories.add(REAL_FAILURE_MAP[fail_type])

    if not categories:
        return "grasping" if task.startswith("bin") or task == "named_bin" else "placement"

    for category in FAILURE_ORDER:
        if category in categories:
            return category
    raise ValueError(f"Could not choose real failure category for row: {row}")


def build_records(
    subset_rows: list[dict[str, Any]],
    sim_results: dict[int, SimResult],
    real_if_rows: list[dict[str, Any]],
    real_bin_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if_index, bin_index = build_real_indices(real_if_rows, real_bin_rows)
    records: list[dict[str, Any]] = []
    missing_real: list[str] = []

    for subset_index, subset_row in enumerate(subset_rows, start=1):
        sim_result = sim_results.get(subset_index)
        if sim_result is None:
            continue

        task = infer_task(subset_row)
        ood_key = str(subset_row.get("ood_key", "unknown"))
        if task.startswith("bin"):
            key = (ood_key, int(subset_row.get("n_objects", 0)), sim_object_names(subset_row))
            bucket = bin_index.get(key)
            if not bucket:
                missing_real.append(f"subset row {subset_index}: bin key={key}")
                continue
            real_row = bucket.popleft()
            real_success = real_bin_success(real_row)
        else:
            key = (sim_object_names(subset_row), str(subset_row.get("instruction", "")), ood_key)
            bucket = if_index.get(key)
            if not bucket:
                missing_real.append(f"subset row {subset_index}: IF key={key}")
                continue
            real_row = bucket.popleft()
            real_success = real_row.get("success") is True

        records.append(
            {
                "subset_index": subset_index,
                "task": task,
                "ood_key": ood_key,
                "sim_success": sim_result.success,
                "real_success": real_success,
                "sim_failure_type": classify_sim_failure(sim_result, task),
                "real_failure_type": classify_real_failure(real_row, task, real_success),
            }
        )

    if missing_real:
        sample = "\n".join(missing_real[:20])
        raise ValueError(f"Could not match {len(missing_real)} subset row(s) to real-world results:\n{sample}")

    return records


def group_sort_key(group: tuple[str, str]) -> tuple[int, int, str, str]:
    task, ood_key = group
    task_index = TASK_ORDER.index(task) if task in TASK_ORDER else len(TASK_ORDER)
    ood_index = OOD_ORDER.index(ood_key) if ood_key in OOD_ORDER else len(OOD_ORDER)
    return task_index, ood_index, task, ood_key


def grouped_success_rates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Counts]] = defaultdict(lambda: {"sim": Counts(), "real": Counts()})
    for record in records:
        key = (record["task"], record["ood_key"])
        groups[key]["sim"].add(bool(record["sim_success"]))
        groups[key]["real"].add(bool(record["real_success"]))

    rows: list[dict[str, Any]] = []
    for task, ood_key in sorted(groups, key=group_sort_key):
        sim_counts = groups[(task, ood_key)]["sim"]
        real_counts = groups[(task, ood_key)]["real"]
        rows.append(
            {
                "task": task,
                "ood_key": ood_key,
                "label": f"{task}\n{ood_key}",
                "sim_successes": sim_counts.successes,
                "sim_total": sim_counts.total,
                "sim_rate": sim_counts.rate,
                "real_successes": real_counts.successes,
                "real_total": real_counts.total,
                "real_rate": real_counts.rate,
            }
        )
    return rows


def pct_axis(axis: Any) -> None:
    axis.set_xlim(0, 1)
    axis.set_xticks(np.linspace(0, 1, 6))
    axis.set_xticklabels([f"{int(x * 100)}%" for x in np.linspace(0, 1, 6)])


def save_success_scatter(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for row in rows:
        ax.scatter(
            row["sim_rate"],
            row["real_rate"],
            s=50 + 4 * row["sim_total"],
            color=TASK_COLORS.get(row["task"], "#79706e"),
            marker=OOD_MARKERS.get(row["ood_key"], "o"),
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
        )
        label = f"{row['task'].replace('bin_', 'b')}/{row['ood_key'].replace('unseen_', 'u_')}"
        ax.annotate(label, (row["sim_rate"], row["real_rate"]), xytext=(5, 4), textcoords="offset points", fontsize=8)

    ax.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1)
    pct_axis(ax)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(x * 100)}%" for x in np.linspace(0, 1, 6)])
    ax.set_xlabel("Sim success rate")
    ax.set_ylabel("Real success rate")
    ax.set_title("Real-Sim Success Rate Correspondence")
    ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.8)

    task_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", label=task)
        for task, color in TASK_COLORS.items()
    ]
    marker_handles = [
        plt.Line2D([0], [0], marker=marker, color="#333333", linestyle="none", label=ood_key)
        for ood_key, marker in OOD_MARKERS.items()
    ]
    first_legend = ax.legend(handles=task_handles, title="Task", loc="lower right", fontsize=8)
    ax.add_artist(first_legend)
    ax.legend(handles=marker_handles, title="OOD key", loc="upper left", fontsize=8)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_success_bars(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [row["label"] for row in rows]
    y = np.arange(len(rows))
    height = 0.38

    fig_height = max(7, 0.45 * len(rows))
    fig, ax = plt.subplots(figsize=(11, fig_height), constrained_layout=True)
    ax.barh(y - height / 2, [row["sim_rate"] for row in rows], height, label="Sim", color=DOMAIN_COLORS["sim"])
    ax.barh(y + height / 2, [row["real_rate"] for row in rows], height, label="Real", color=DOMAIN_COLORS["real"])

    for idx, row in enumerate(rows):
        ax.text(row["sim_rate"] + 0.01, idx - height / 2, f"{100 * row['sim_rate']:.0f}%", va="center", fontsize=8)
        ax.text(row["real_rate"] + 0.01, idx + height / 2, f"{100 * row['real_rate']:.0f}%", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    pct_axis(ax)
    ax.set_xlabel("Success rate")
    ax.set_title("Real vs. Sim Success Rates by Task and OOD Regime")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.legend(loc="lower right")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def failure_counts(records: list[dict[str, Any]], domain: str, group_key: str) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {category: 0 for category in FAILURE_ORDER})
    success_key = f"{domain}_success"
    failure_key = f"{domain}_failure_type"
    for record in records:
        if record[success_key]:
            continue
        category = record[failure_key]
        if category not in FAILURE_ORDER:
            raise ValueError(f"Unexpected failure category {category!r}")
        counts[str(record[group_key])][category] += 1
    return counts


def save_failure_mix_by_task(records: list[dict[str, Any]], output_path: Path) -> None:
    tasks = [task for task in TASK_ORDER if any(record["task"] == task for record in records)]
    sim_counts = failure_counts(records, "sim", "task")
    real_counts = failure_counts(records, "real", "task")

    labels: list[str] = []
    domains: list[str] = []
    data: list[list[int]] = []
    for task in tasks:
        for domain, counts in (("sim", sim_counts), ("real", real_counts)):
            labels.append(f"{task}\n{domain}")
            domains.append(domain)
            data.append([counts[task][category] for category in FAILURE_ORDER])

    totals = np.array([sum(row) for row in data], dtype=float)
    proportions = np.divide(np.array(data, dtype=float), totals[:, None], out=np.zeros((len(data), len(FAILURE_ORDER))), where=totals[:, None] > 0)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    bottom = np.zeros(len(data))
    x = np.arange(len(data))
    for idx, category in enumerate(FAILURE_ORDER):
        ax.bar(x, proportions[:, idx], bottom=bottom, color=FAILURE_COLORS[category], label=category)
        bottom += proportions[:, idx]

    for idx, total in enumerate(totals):
        ax.text(idx, 1.02, f"n={int(total)}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
    ax.set_ylabel("Share of failed episodes")
    ax.set_title("Failure Type Mix by Task")
    ax.legend(loc="upper right")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)

    for idx, domain in enumerate(domains):
        ax.get_xticklabels()[idx].set_color(DOMAIN_COLORS[domain])

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_failure_mix_by_ood(records: list[dict[str, Any]], output_path: Path) -> None:
    ood_keys = [ood_key for ood_key in OOD_ORDER if any(record["ood_key"] == ood_key for record in records)]
    sim_counts = failure_counts(records, "sim", "ood_key")
    real_counts = failure_counts(records, "real", "ood_key")

    labels: list[str] = []
    domains: list[str] = []
    data: list[list[int]] = []
    for ood_key in ood_keys:
        for domain, counts in (("sim", sim_counts), ("real", real_counts)):
            labels.append(f"{ood_key}\n{domain}")
            domains.append(domain)
            data.append([counts[ood_key][category] for category in FAILURE_ORDER])

    totals = np.array([sum(row) for row in data], dtype=float)
    proportions = np.divide(
        np.array(data, dtype=float),
        totals[:, None],
        out=np.zeros((len(data), len(FAILURE_ORDER))),
        where=totals[:, None] > 0,
    )

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    bottom = np.zeros(len(data))
    x = np.arange(len(data))
    for idx, category in enumerate(FAILURE_ORDER):
        ax.bar(x, proportions[:, idx], bottom=bottom, color=FAILURE_COLORS[category], label=category)
        bottom += proportions[:, idx]

    for idx, total in enumerate(totals):
        ax.text(idx, 1.02, f"n={int(total)}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
    ax.set_ylabel("Share of failed episodes")
    ax.set_title("Failure Type Mix by OOD Key")
    ax.legend(loc="upper right")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)

    for idx, domain in enumerate(domains):
        ax.get_xticklabels()[idx].set_color(DOMAIN_COLORS[domain])

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_outcome_confusion(records: list[dict[str, Any]], output_path: Path) -> None:
    matrix = np.zeros((2, 2), dtype=int)
    for record in records:
        sim_idx = 0 if record["sim_success"] else 1
        real_idx = 0 if record["real_success"] else 1
        matrix[real_idx, sim_idx] += 1

    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Sim success", "Sim failure"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Real success", "Real failure"])
    ax.set_title("Episode-Level Outcome Correspondence")

    for row in range(2):
        for col in range(2):
            value = matrix[row, col]
            ax.text(col, row, str(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "#222222", fontsize=14)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_failure_correspondence(records: list[dict[str, Any]], output_path: Path) -> None:
    both_failed = [record for record in records if not record["sim_success"] and not record["real_success"]]
    matrix = np.zeros((len(FAILURE_ORDER), len(FAILURE_ORDER)), dtype=int)
    for record in both_failed:
        row = FAILURE_ORDER.index(record["real_failure_type"])
        col = FAILURE_ORDER.index(record["sim_failure_type"])
        matrix[row, col] += 1

    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Purples")
    ax.set_xticks(range(len(FAILURE_ORDER)))
    ax.set_xticklabels([f"Sim {category}" for category in FAILURE_ORDER], rotation=35, ha="right")
    ax.set_yticks(range(len(FAILURE_ORDER)))
    ax.set_yticklabels([f"Real {category}" for category in FAILURE_ORDER])
    ax.set_title("Failure-Type Correspondence When Both Fail")

    for row in range(len(FAILURE_ORDER)):
        for col in range(len(FAILURE_ORDER)):
            value = matrix[row, col]
            ax.text(col, row, str(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "#222222", fontsize=12)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot real-vs-sim correspondence for GR00T eval results.")
    parser.add_argument("--sim-log", type=Path, default=Path("logs/groot_eval_20260603_031213.log"))
    parser.add_argument("--subset-jsonl", type=Path, default=Path("tasks/real_gr00t_WM_combined.jsonl"))
    parser.add_argument("--if-results-jsonl", type=Path, default=Path("logs/real_world_results/results_gr00t_IF_WM.jsonl"))
    parser.add_argument("--bin-results-jsonl", type=Path, default=Path("logs/real_world_results/results_gr00t_grasp_WM.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/real_sim_correspondence"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset_rows = load_jsonl(args.subset_jsonl)
    sim_results = parse_sim_log(args.sim_log)
    real_if_rows = load_jsonl(args.if_results_jsonl)
    real_bin_rows = load_jsonl(args.bin_results_jsonl)

    if not sim_results:
        raise ValueError(f"Found no episode result lines in {args.sim_log}")

    records = build_records(subset_rows, sim_results, real_if_rows, real_bin_rows)
    if len(records) != len(subset_rows):
        print(
            f"[WARN] Matched {len(records)} row(s), but {args.subset_jsonl} has {len(subset_rows)} row(s). "
            "Plots use rows with sim results only."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    success_rows = grouped_success_rates(records)
    outputs = {
        "success_scatter": args.output_dir / "success_rate_correspondence.png",
        "success_bars": args.output_dir / "success_rates_by_task_ood.png",
        "failure_mix": args.output_dir / "failure_type_mix_by_task.png",
        "failure_mix_ood": args.output_dir / "failure_type_mix_by_ood.png",
        "outcome_confusion": args.output_dir / "outcome_correspondence.png",
        "failure_correspondence": args.output_dir / "failure_type_correspondence.png",
    }

    save_success_scatter(success_rows, outputs["success_scatter"])
    save_success_bars(success_rows, outputs["success_bars"])
    save_failure_mix_by_task(records, outputs["failure_mix"])
    save_failure_mix_by_ood(records, outputs["failure_mix_ood"])
    save_outcome_confusion(records, outputs["outcome_confusion"])
    save_failure_correspondence(records, outputs["failure_correspondence"])

    sim_successes = sum(1 for record in records if record["sim_success"])
    real_successes = sum(1 for record in records if record["real_success"])
    both_success = sum(1 for record in records if record["sim_success"] and record["real_success"])
    both_fail = sum(1 for record in records if not record["sim_success"] and not record["real_success"])

    print(f"matched_rows: {len(records)}")
    print(f"sim_success: {sim_successes}/{len(records)} ({100 * sim_successes / len(records):.1f}%)")
    print(f"real_success: {real_successes}/{len(records)} ({100 * real_successes / len(records):.1f}%)")
    print(f"both_success: {both_success}")
    print(f"both_fail: {both_fail}")
    print("wrote:")
    for path in outputs.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
