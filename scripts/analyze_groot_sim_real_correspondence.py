#!/usr/bin/env python3
"""Build and analyze the final paired GR00T real/sim correspondence benchmark.

The real and simulated evaluations used different fine-tuned policies.  The
statistics produced here therefore measure paired *policy-domain system*
correspondence, not simulator fidelity for a policy whose weights are held
fixed across domains.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SIM = ROOT / "logs/groot_sim_real_correspondence/real_sim_correspondence.jsonl"
DEFAULT_PRE_REDO = (
    ROOT
    / "logs/groot_sim_real_correspondence/real_sim_correspondence.pre_redo_merge_20260811T065815Z.jsonl"
)
DEFAULT_REDO_REVIEW = ROOT / "logs/groot_sim_real_correspondence/real_sim_correspondence_redo_review.jsonl"
DEFAULT_REDO_SCORES = ROOT / "logs/groot_sim_real_correspondence/real_sim_correspondence_redos.jsonl"
DEFAULT_REDO_MANIFEST = (
    ROOT
    / "outputs/groot_sim_real_correspondence/checkpoint-20000/redos/run_20260811T015034Z/redo_manifest.jsonl"
)
DEFAULT_TASKS = (
    ROOT
    / "outputs/groot_sim_real_correspondence/checkpoint-20000/run_20260809T062956Z/tasks.jsonl"
)
DEFAULT_REAL_IF = ROOT / "logs/real_world_results/results_gr00t_IF_WM.jsonl"
DEFAULT_REAL_BIN = ROOT / "logs/real_world_results/results_gr00t_grasp_WM.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/groot_sim_real_correspondence/final_correspondence_analysis"

TASK_ORDER = ["bin_1obj", "bin_4obj", "next_to", "between", "move"]
OOD_ORDER = ["seen", "unseen_seen_class", "unseen_unseen_class"]
TASK_LABELS = {
    "bin_1obj": "1-object bin",
    "bin_4obj": "4-object bin",
    "next_to": "next-to",
    "between": "between",
    "move": "move",
}
OOD_LABELS = {
    "seen": "seen",
    "unseen_seen_class": "unseen object / seen class",
    "unseen_unseen_class": "unseen object / unseen class",
}
TASK_COLORS = {
    "bin_1obj": "#4c78a8",
    "bin_4obj": "#f58518",
    "next_to": "#54a24b",
    "between": "#b279a2",
    "move": "#e45756",
}
OOD_MARKERS = {"seen": "o", "unseen_seen_class": "s", "unseen_unseen_class": "^"}

FAILURE_PRIORITY = ["semantic", "grasping", "placement", "unknown"]
FAILURE_TOKENS: dict[str, set[str]] = {
    "semantic": {
        "nce",
        "nse",
        "tcr",
        "grasped other object",
        "grasped class distractor object",
        "grasped color distractor object",
        "placed between other objects",
        "placed next to other object",
        "placed next to class distractor object",
        "placed next to color distractor object",
        "occluded",
        "totally confused",
        "totally confused after going ood",
        "totally confused (hovered above)",
    },
    "grasping": {
        "refused to lift",
        "refused to undock",
        "failed to undock",
        "refused to dock",
        "refused to drop",
        "refused to release",
        "refused to release the object",
        "grabbed air",
        "grabbed air indefinitely",
        "repeatedly reached and retracted",
        "camera stuck on bin",
        "throttled gripper",
        "docked with object in grip",
        "stuck on bin",
        "dropped pen",
        "missed completely",
        "missed",
        "making contact",
        "imprecise",
        "bad strategy",
    },
    "placement": {
        "imprecise by 1cm",
        "imprecise by 2cm",
        "imprecise by 3cm",
        "placed on top",
        "stuck on top",
        "not close",
        "not close enough",
        "missed bin",
        "knocked bin",
        "ran into bin",
        "rolled",
        "moved object",
        "moved objects",
        "moved boundary",
        "moved past boundary",
        "moved slowly",
        "moved very slowly",
    },
}

# Sensitivity matching the older plotting script.  The primary mapping above
# follows the annotation protocol's use of generic "imprecise" for an
# imprecise grasp; distance-qualified "imprecise by Ncm" remains placement.
LEGACY_IMPRECISE_AS_PLACEMENT_TOKENS = {category: set(tokens) for category, tokens in FAILURE_TOKENS.items()}
LEGACY_IMPRECISE_AS_PLACEMENT_TOKENS["grasping"].remove("imprecise")
LEGACY_IMPRECISE_AS_PLACEMENT_TOKENS["placement"].add("imprecise")


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


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_float(value: float | np.floating[Any] | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def infer_task(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).lower()
    if row.get("direction") is not None or instruction.startswith("move "):
        return "move"
    if "between" in instruction:
        return "between"
    if "next to" in instruction:
        return "next_to"
    if "plastic bin" in instruction:
        n_objects = int(row.get("n_objects", 0))
        if n_objects in (1, 4):
            return f"bin_{n_objects}obj"
    raise ValueError(f"Could not infer task from row: {row}")


def object_names(row: dict[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for item in row.get("objects", []):
        if isinstance(item, dict):
            names.append(str(item["name"]))
        else:
            names.append(str(item))
    return tuple(names)


def task_success(row: dict[str, Any], task: str) -> bool:
    if task == "bin_4obj":
        objects = row.get("objects", [])
        return len(objects) == 4 and all(isinstance(item, dict) and item.get("success") is True for item in objects)
    if task == "bin_1obj" and row.get("success") is None:
        objects = row.get("objects", [])
        return len(objects) == 1 and isinstance(objects[0], dict) and objects[0].get("success") is True
    return row.get("success") is True


def object_progress(row: dict[str, Any], task: str) -> tuple[int, int] | None:
    if task != "bin_4obj":
        return None
    objects = row.get("objects", [])
    if len(objects) != 4 or not all(isinstance(item, dict) for item in objects):
        raise ValueError(f"Expected four object result dicts: {row}")
    return sum(item.get("success") is True for item in objects), 4


def object_attempts(row: dict[str, Any], task: str) -> tuple[int, int] | None:
    if task != "bin_4obj":
        return None
    objects = row.get("objects", [])
    attempted = sum(item.get("attempts") is not None for item in objects if isinstance(item, dict))
    return attempted, len(objects)


def verify_score_context(task_row: dict[str, Any], score_row: dict[str, Any], index: int, label: str) -> None:
    comparisons = {
        "trial_id": (task_row.get("trial_id"), score_row.get("trial_id")),
        "objects": (object_names(task_row), object_names(score_row)),
        "ood_key": (task_row.get("ood_key"), score_row.get("ood_key")),
        "n_objects": (int(task_row.get("n_objects", 0)), int(score_row.get("n_objects", 0))),
    }
    # The real-world-format 4-bin score schema stores per-object results and
    # intentionally omits the instruction; every other score row retains it.
    if "instruction" in score_row or infer_task(task_row) != "bin_4obj":
        comparisons["instruction"] = (task_row.get("instruction"), score_row.get("instruction"))
    for optional in ("target", "referents", "direction"):
        if optional in task_row or optional in score_row:
            comparisons[optional] = (task_row.get(optional), score_row.get(optional))
    for key, (expected, found) in comparisons.items():
        if expected != found:
            raise ValueError(f"{label} row {index + 1} disagrees with task row on {key}: {found!r} != {expected!r}")


def build_real_indices(
    real_if: list[dict[str, Any]], real_bin: list[dict[str, Any]]
) -> tuple[
    dict[tuple[tuple[str, ...], str, str], deque[tuple[int, dict[str, Any]]]],
    dict[tuple[str, int, tuple[str, ...], bool], deque[tuple[int, dict[str, Any]]]],
]:
    if_index: dict[tuple[tuple[str, ...], str, str], deque[tuple[int, dict[str, Any]]]] = defaultdict(deque)
    for index, row in enumerate(real_if):
        key = (object_names(row), str(row.get("instruction", "")), str(row.get("ood_key", "unknown")))
        if_index[key].append((index, row))

    bin_index: dict[tuple[str, int, tuple[str, ...], bool], deque[tuple[int, dict[str, Any]]]] = defaultdict(deque)
    for index, row in enumerate(real_bin):
        key = (
            str(row.get("ood_key", "unknown")),
            int(row.get("n_objects", 0)),
            object_names(row),
            bool(row.get("clutter", False)),
        )
        bin_index[key].append((index, row))
    return if_index, bin_index


def match_real_rows(
    task_rows: list[dict[str, Any]], real_if: list[dict[str, Any]], real_bin: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if_index, bin_index = build_real_indices(real_if, real_bin)
    matched: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []

    for pair_index, task_row in enumerate(task_rows):
        task = infer_task(task_row)
        ood = str(task_row.get("ood_key", "unknown"))
        if task.startswith("bin_"):
            key = (
                ood,
                int(task_row.get("n_objects", 0)),
                object_names(task_row),
                bool(task_row.get("clutter", False)),
            )
            bucket = bin_index.get(key)
            source_name = "results_gr00t_grasp_WM.jsonl"
            if not bucket:
                raise ValueError(f"No real bin match for pair {pair_index}: {key}")
        else:
            key = (object_names(task_row), str(task_row.get("instruction", "")), ood)
            bucket = if_index.get(key)
            source_name = "results_gr00t_IF_WM.jsonl"
            if not bucket:
                raise ValueError(f"No real instruction-following match for pair {pair_index}: {key}")

        source_index, real_row = bucket.popleft()
        if object_names(real_row) != object_names(task_row):
            raise AssertionError("Ordered object-name match failed")
        if str(real_row.get("ood_key")) != ood:
            raise AssertionError("OOD match failed")

        if not task.startswith("bin_"):
            if real_row.get("instruction") != task_row.get("instruction"):
                raise AssertionError("Instruction match failed")
            real_referents = list(real_row.get("referents") or [])
            if real_referents:
                if task_row.get("target") != real_referents[0]:
                    raise ValueError(f"Target mismatch at pair {pair_index}")
                if task != "move" and list(task_row.get("referents") or []) != real_referents[1:]:
                    raise ValueError(f"Anchor/referent mismatch at pair {pair_index}")

        matched.append(real_row)
        provenance.append(
            {
                "pair_index": pair_index,
                "pair_episode": pair_index + 1,
                "task_trial_id": task_row.get("trial_id"),
                "real_source_file": source_name,
                "real_source_index": source_index,
                "real_source_line": source_index + 1,
                "real_trial_id": real_row.get("trial_id"),
            }
        )

    return matched, provenance


def failure_tokens(row: dict[str, Any]) -> list[str]:
    raw = row.get("fail_type") or []
    if isinstance(raw, str):
        raw = [raw]
    tokens: list[str] = []
    for value in raw:
        tokens.extend(part.strip().lower().rstrip(".") for part in str(value).split(",") if part.strip())
    return tokens


def failure_category_set(
    row: dict[str, Any], success: bool, taxonomy: dict[str, set[str]] = FAILURE_TOKENS
) -> set[str]:
    if success:
        return {"success"}
    tokens = failure_tokens(row)
    if not tokens:
        return {"unknown"}
    categories: set[str] = set()
    for token in tokens:
        matched = [category for category, known in taxonomy.items() if token in known]
        categories.add(matched[0] if matched else "unknown")
    return categories


def primary_category(categories: set[str]) -> str:
    if "success" in categories:
        return "success"
    return next(category for category in FAILURE_PRIORITY if category in categories)


def build_pairs(
    tasks: list[dict[str, Any]],
    sim: list[dict[str, Any]],
    real: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
    redo_indices: set[int],
) -> list[dict[str, Any]]:
    if not (len(tasks) == len(sim) == len(real) == len(provenance)):
        raise ValueError(f"Length mismatch: tasks={len(tasks)}, sim={len(sim)}, real={len(real)}")

    pairs: list[dict[str, Any]] = []
    for index, (task_row, sim_row, real_row, source) in enumerate(zip(tasks, sim, real, provenance)):
        verify_score_context(task_row, sim_row, index, "sim")
        task = infer_task(task_row)
        sim_success = task_success(sim_row, task)
        real_success = task_success(real_row, task)
        sim_progress = object_progress(sim_row, task)
        real_progress = object_progress(real_row, task)
        sim_attempted = object_attempts(sim_row, task)
        real_attempted = object_attempts(real_row, task)
        pairs.append(
            {
                **source,
                "task": task,
                "ood_key": str(task_row.get("ood_key")),
                "ordered_objects": list(object_names(task_row)),
                "target": task_row.get("target"),
                "anchors": list(task_row.get("referents") or []),
                "direction": task_row.get("direction"),
                "instruction": task_row.get("instruction"),
                "redo_applied": index in redo_indices,
                "real_success": real_success,
                "real_progress_successes": real_progress[0] if real_progress else None,
                "real_progress_total": real_progress[1] if real_progress else None,
                "real_attempted_objects": real_attempted[0] if real_attempted else None,
                "real_fail_type": real_row.get("fail_type") or [],
                "real_notes": real_row.get("notes") or [],
                "real_failure_categories": sorted(failure_category_set(real_row, real_success)),
                "sim_success": sim_success,
                "sim_progress_successes": sim_progress[0] if sim_progress else None,
                "sim_progress_total": sim_progress[1] if sim_progress else None,
                "sim_attempted_objects": sim_attempted[0] if sim_attempted else None,
                "sim_fail_type": sim_row.get("fail_type") or [],
                "sim_notes": sim_row.get("notes") or [],
                "sim_failure_categories": sorted(failure_category_set(sim_row, sim_success)),
            }
        )
    return pairs


def contingency_metrics(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    real = np.array([bool(row["real_success"]) for row in pairs], dtype=bool)
    sim = np.array([bool(row["sim_success"]) for row in pairs], dtype=bool)
    n11 = int(np.sum(real & sim))
    n10 = int(np.sum(real & ~sim))
    n01 = int(np.sum(~real & sim))
    n00 = int(np.sum(~real & ~sim))
    n = len(pairs)
    real_rate = float(np.mean(real)) if n else float("nan")
    sim_rate = float(np.mean(sim)) if n else float("nan")
    observed = (n11 + n00) / n if n else float("nan")
    expected = ((n11 + n10) * (n11 + n01) + (n01 + n00) * (n10 + n00)) / (n * n) if n else float("nan")
    kappa = safe_div(observed - expected, 1.0 - expected)
    mcc_denominator = math.sqrt((n11 + n01) * (n11 + n10) * (n00 + n01) * (n00 + n10))
    mcc = safe_div(n11 * n00 - n01 * n10, mcc_denominator)
    discordant = n10 + n01
    mcnemar_p = stats.binomtest(min(n10, n01), discordant, 0.5, alternative="two-sided").pvalue if discordant else 1.0
    sensitivity = safe_div(n11, n11 + n10)
    specificity = safe_div(n00, n00 + n01)
    balanced_accuracy = (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None
    return {
        "n": n,
        "both_success_n11": n11,
        "real_success_sim_failure_n10": n10,
        "real_failure_sim_success_n01": n01,
        "both_failure_n00": n00,
        "sim_successes": int(sim.sum()),
        "sim_success_rate": sim_rate,
        "real_successes": int(real.sum()),
        "real_success_rate": real_rate,
        "sim_minus_real_rate": sim_rate - real_rate,
        "raw_agreement": observed,
        "positive_agreement": safe_div(2 * n11, 2 * n11 + n10 + n01),
        "negative_agreement": safe_div(2 * n00, 2 * n00 + n10 + n01),
        "cohen_kappa": kappa,
        "mcc_phi": mcc,
        "sensitivity_sim_success_given_real_success": sensitivity,
        "specificity_sim_failure_given_real_failure": specificity,
        "positive_predictive_value": safe_div(n11, n11 + n01),
        "negative_predictive_value": safe_div(n00, n00 + n10),
        "balanced_accuracy": balanced_accuracy,
        "mcnemar_exact_two_sided_p": float(mcnemar_p),
    }


def metric_arrays(n11: np.ndarray, n10: np.ndarray, n01: np.ndarray, n00: np.ndarray) -> dict[str, np.ndarray]:
    total = n11 + n10 + n01 + n00
    agreement = (n11 + n00) / total
    sim_rate = (n11 + n01) / total
    real_rate = (n11 + n10) / total
    expected = ((n11 + n10) * (n11 + n01) + (n01 + n00) * (n10 + n00)) / total**2
    kappa = np.divide(agreement - expected, 1 - expected, out=np.full_like(agreement, np.nan), where=(1 - expected) != 0)
    denominator = np.sqrt((n11 + n01) * (n11 + n10) * (n00 + n01) * (n00 + n10))
    mcc = np.divide(n11 * n00 - n01 * n10, denominator, out=np.full_like(agreement, np.nan), where=denominator != 0)
    return {
        "sim_minus_real_rate": sim_rate - real_rate,
        "raw_agreement": agreement,
        "cohen_kappa": kappa,
        "mcc_phi": mcc,
    }


def percentile_ci(values: np.ndarray) -> list[float | None]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return [None, None]
    low, high = np.percentile(finite, [2.5, 97.5])
    return [float(low), float(high)]


def bootstrap_binary(
    pairs: Sequence[dict[str, Any]], strata: Sequence[str], replicates: int, rng: np.random.Generator
) -> dict[str, list[float | None]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        groups[stratum].append(index)
    real = np.array([bool(row["real_success"]) for row in pairs], dtype=bool)
    sim = np.array([bool(row["sim_success"]) for row in pairs], dtype=bool)
    n11 = np.zeros(replicates)
    n10 = np.zeros(replicates)
    n01 = np.zeros(replicates)
    n00 = np.zeros(replicates)
    for indices in groups.values():
        local_real = real[indices]
        local_sim = sim[indices]
        sampled = rng.integers(0, len(indices), size=(replicates, len(indices)))
        br = local_real[sampled]
        bs = local_sim[sampled]
        n11 += np.sum(br & bs, axis=1)
        n10 += np.sum(br & ~bs, axis=1)
        n01 += np.sum(~br & bs, axis=1)
        n00 += np.sum(~br & ~bs, axis=1)
    return {name: percentile_ci(values) for name, values in metric_arrays(n11, n10, n01, n00).items()}


def semantic_cluster_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Exact task configuration; only the repeated 1-bin trials share this key."""
    return (
        row["task"],
        row["ood_key"],
        row["instruction"],
        tuple(row["ordered_objects"]),
        row["target"],
        tuple(row["anchors"]),
        row["direction"],
    )


def target_object_cluster_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Broader sensitivity cluster for policies repeatedly acting on one target."""
    if row["task"] == "bin_1obj":
        target: Any = row["ordered_objects"][0]
    elif row["task"] == "bin_4obj":
        # There is no single target; preserve the full tabletop object set.
        target = tuple(sorted(row["ordered_objects"]))
    else:
        target = row["target"]
    return row["task"], row["ood_key"], target


def cluster_bootstrap_binary(
    pairs: Sequence[dict[str, Any]],
    strata: Sequence[str],
    replicates: int,
    rng: np.random.Generator,
    cluster_key: Any = semantic_cluster_key,
) -> dict[str, list[float | None]]:
    """Resample exact semantic configurations jointly within each stratum."""
    by_stratum: dict[str, dict[tuple[Any, ...], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row, stratum in zip(pairs, strata):
        by_stratum[stratum][cluster_key(row)].append(row)

    n11 = np.zeros(replicates)
    n10 = np.zeros(replicates)
    n01 = np.zeros(replicates)
    n00 = np.zeros(replicates)
    for clusters in by_stratum.values():
        summaries = []
        for rows in clusters.values():
            metric = contingency_metrics(rows)
            summaries.append(
                [
                    metric["both_success_n11"],
                    metric["real_success_sim_failure_n10"],
                    metric["real_failure_sim_success_n01"],
                    metric["both_failure_n00"],
                ]
            )
        cluster_counts = np.asarray(summaries, dtype=float)
        sampled = rng.integers(0, len(cluster_counts), size=(replicates, len(cluster_counts)))
        totals = np.sum(cluster_counts[sampled], axis=1)
        n11 += totals[:, 0]
        n10 += totals[:, 1]
        n01 += totals[:, 2]
        n00 += totals[:, 3]
    return {name: percentile_ci(values) for name, values in metric_arrays(n11, n10, n01, n00).items()}


def paired_gap_ci(pairs: Sequence[dict[str, Any]], replicates: int, rng: np.random.Generator) -> list[float | None]:
    differences = np.array(
        [int(bool(row["sim_success"])) - int(bool(row["real_success"])) for row in pairs], dtype=float
    )
    sampled = rng.integers(0, len(differences), size=(replicates, len(differences)))
    return percentile_ci(np.mean(differences[sampled], axis=1))


def group_rows(
    pairs: Sequence[dict[str, Any]], key_names: Sequence[str], replicates: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        groups[tuple(str(row[key]) for key in key_names)].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in groups.items():
        metrics = contingency_metrics(rows)
        metrics["sim_minus_real_rate_ci95"] = paired_gap_ci(rows, replicates, rng)
        metrics["semantic_cluster_bootstrap_ci95"] = cluster_bootstrap_binary(
            rows, ["all"] * len(rows), replicates, rng
        )
        output.append({**dict(zip(key_names, key)), **metrics})

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        key: list[Any] = []
        if "task" in row:
            key.append(TASK_ORDER.index(row["task"]))
        if "ood_key" in row:
            key.append(OOD_ORDER.index(row["ood_key"]))
        return tuple(key)

    return sorted(output, key=sort_key)


def rowwise_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    y_centered = y - np.mean(y, axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1))
    return np.divide(numerator, denominator, out=np.full(len(x), np.nan), where=denominator != 0)


def weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    mean_x = np.average(x, weights=weights)
    mean_y = np.average(y, weights=weights)
    covariance = np.average((x - mean_x) * (y - mean_y), weights=weights)
    variance_x = np.average((x - mean_x) ** 2, weights=weights)
    variance_y = np.average((y - mean_y) ** 2, weights=weights)
    return float(covariance / math.sqrt(variance_x * variance_y))


def condition_correspondence(
    condition_rows: list[dict[str, Any]], pairs: Sequence[dict[str, Any]], replicates: int, rng: np.random.Generator
) -> dict[str, Any]:
    sim_rates = np.array([row["sim_success_rate"] for row in condition_rows])
    real_rates = np.array([row["real_success_rate"] for row in condition_rows])
    weights = np.array([row["n"] for row in condition_rows], dtype=float)
    pearson = stats.pearsonr(sim_rates, real_rates)
    spearman = stats.spearmanr(sim_rates, real_rates)

    by_condition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        by_condition[(row["task"], row["ood_key"])].append(row)
    bootstrap_sim = np.zeros((replicates, len(condition_rows)))
    bootstrap_real = np.zeros_like(bootstrap_sim)
    for column, condition in enumerate(condition_rows):
        rows = by_condition[(condition["task"], condition["ood_key"])]
        sim = np.array([bool(row["sim_success"]) for row in rows])
        real = np.array([bool(row["real_success"]) for row in rows])
        sampled = rng.integers(0, len(rows), size=(replicates, len(rows)))
        bootstrap_sim[:, column] = np.mean(sim[sampled], axis=1)
        bootstrap_real[:, column] = np.mean(real[sampled], axis=1)
    bootstrap_pearson = rowwise_correlation(bootstrap_sim, bootstrap_real)
    bootstrap_spearman = rowwise_correlation(
        stats.rankdata(bootstrap_sim, axis=1), stats.rankdata(bootstrap_real, axis=1)
    )

    error = sim_rates - real_rates
    return {
        "n_conditions": len(condition_rows),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "pearson_bootstrap_ci95": percentile_ci(bootstrap_pearson),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "spearman_bootstrap_ci95": percentile_ci(bootstrap_spearman),
        "weighted_pearson_r": weighted_correlation(sim_rates, real_rates, weights),
        "macro_mean_bias_sim_minus_real": float(np.mean(error)),
        "macro_mae": float(np.mean(np.abs(error))),
        "macro_rmse": float(np.sqrt(np.mean(error**2))),
        "episode_weighted_mean_bias_sim_minus_real": float(np.average(error, weights=weights)),
        "episode_weighted_mae": float(np.average(np.abs(error), weights=weights)),
        "episode_weighted_rmse": float(np.sqrt(np.average(error**2, weights=weights))),
    }


def stratified_permutation_test(
    condition_rows: list[dict[str, Any]], observed: dict[str, Any], replicates: int, rng: np.random.Generator
) -> dict[str, Any]:
    null_n11 = np.zeros(replicates, dtype=int)
    for row in condition_rows:
        total = int(row["n"])
        real_successes = int(row["real_successes"])
        sim_successes = int(row["sim_successes"])
        null_n11 += rng.hypergeometric(real_successes, total - real_successes, sim_successes, size=replicates)
    return summarize_permutation_null(
        null_n11,
        observed,
        replicates,
        "Sim labels are exchangeable within each task×OOD cell, with both domains' cell margins fixed.",
    )


def summarize_permutation_null(
    null_n11: np.ndarray, observed: dict[str, Any], replicates: int, null_hypothesis: str
) -> dict[str, Any]:
    n = int(observed["n"])
    real_successes = int(observed["real_successes"])
    sim_successes = int(observed["sim_successes"])
    null_n10 = real_successes - null_n11
    null_n01 = sim_successes - null_n11
    null_n00 = n - null_n11 - null_n10 - null_n01
    null_metrics = metric_arrays(
        null_n11.astype(float), null_n10.astype(float), null_n01.astype(float), null_n00.astype(float)
    )
    null_agreement = null_metrics["raw_agreement"]
    null_mcc = null_metrics["mcc_phi"]
    observed_agreement = float(observed["raw_agreement"])
    observed_mcc = float(observed["mcc_phi"])
    return {
        "null_hypothesis": null_hypothesis,
        "replicates": replicates,
        "observed_agreement": observed_agreement,
        "null_mean_agreement": float(np.mean(null_agreement)),
        "excess_agreement_over_stratified_null": observed_agreement - float(np.mean(null_agreement)),
        "agreement_one_sided_p": float((1 + np.sum(null_agreement >= observed_agreement)) / (replicates + 1)),
        "observed_mcc": observed_mcc,
        "null_mean_mcc": float(np.nanmean(null_mcc)),
        "mcc_one_sided_p": float((1 + np.sum(null_mcc >= observed_mcc)) / (replicates + 1)),
    }


def semantic_cluster_block_permutation_test(
    pairs: Sequence[dict[str, Any]], observed: dict[str, Any], replicates: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Shuffle whole repeated-trial blocks, never individual rows inside a block."""
    strata: dict[tuple[str, str], dict[tuple[Any, ...], list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in pairs:
        strata[(row["task"], row["ood_key"])][semantic_cluster_key(row)].append(row)

    null_n11 = np.zeros(replicates, dtype=int)
    for clusters in strata.values():
        by_size: dict[int, list[list[dict[str, Any]]]] = defaultdict(list)
        for rows in clusters.values():
            by_size[len(rows)].append(rows)
        for cluster_size, equally_sized_clusters in by_size.items():
            real_vectors = np.asarray(
                [[bool(row["real_success"]) for row in rows] for rows in equally_sized_clusters], dtype=bool
            )
            sim_vectors = np.asarray(
                [[bool(row["sim_success"]) for row in rows] for rows in equally_sized_clusters], dtype=bool
            )
            cluster_count = len(equally_sized_clusters)
            if cluster_size == 1:
                real_successes = int(real_vectors.sum())
                sim_successes = int(sim_vectors.sum())
                null_n11 += rng.hypergeometric(
                    real_successes, cluster_count - real_successes, sim_successes, size=replicates
                )
            elif cluster_count == 1:
                null_n11 += int(np.sum(real_vectors & sim_vectors))
            else:
                # Independent random keys give one uniform permutation per row.
                permutations = np.argsort(rng.random((replicates, cluster_count)), axis=1)
                permuted_sim = sim_vectors[permutations]
                null_n11 += np.sum(real_vectors[None, :, :] & permuted_sim, axis=(1, 2))

    return summarize_permutation_null(
        null_n11,
        observed,
        replicates,
        "Whole exact-semantic-configuration sim blocks are exchangeable within task×OOD and block-size groups; repeated rows stay together.",
    )


def quadratic_weighted_kappa(real: np.ndarray, sim: np.ndarray, levels: int) -> float | None:
    matrix = np.zeros((levels, levels), dtype=float)
    for r, s in zip(real, sim):
        matrix[int(r), int(s)] += 1
    weights = np.fromfunction(lambda i, j: ((i - j) / (levels - 1)) ** 2, (levels, levels))
    expected = np.outer(matrix.sum(axis=1), matrix.sum(axis=0)) / matrix.sum()
    observed_disagreement = float(np.sum(weights * matrix) / matrix.sum())
    expected_disagreement = float(np.sum(weights * expected) / expected.sum())
    return safe_div(expected_disagreement - observed_disagreement, expected_disagreement)


def four_bin_analysis(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in pairs if row["task"] == "bin_4obj"]
    real_progress = np.array([int(row["real_progress_successes"]) for row in rows])
    sim_progress = np.array([int(row["sim_progress_successes"]) for row in rows])
    spearman = stats.spearmanr(sim_progress, real_progress)
    progress_confusion = np.zeros((5, 5), dtype=int)
    for r, s in zip(real_progress, sim_progress):
        progress_confusion[r, s] += 1
    return {
        "episodes": len(rows),
        "sim_task_successes": int(np.sum(sim_progress == 4)),
        "sim_task_success_rate": float(np.mean(sim_progress == 4)),
        "real_task_successes": int(np.sum(real_progress == 4)),
        "real_task_success_rate": float(np.mean(real_progress == 4)),
        "sim_completed_objects": int(sim_progress.sum()),
        "real_completed_objects": int(real_progress.sum()),
        "total_objects": int(4 * len(rows)),
        "sim_progress_rate": float(np.mean(sim_progress / 4)),
        "real_progress_rate": float(np.mean(real_progress / 4)),
        "mean_progress_difference_sim_minus_real_objects": float(np.mean(sim_progress - real_progress)),
        "progress_mae_objects": float(np.mean(np.abs(sim_progress - real_progress))),
        "progress_rmse_objects": float(np.sqrt(np.mean((sim_progress - real_progress) ** 2))),
        "progress_exact_agreement": float(np.mean(sim_progress == real_progress)),
        "progress_spearman_rho": json_float(spearman.statistic),
        "progress_spearman_p": json_float(spearman.pvalue),
        "progress_quadratic_weighted_kappa": json_float(quadratic_weighted_kappa(real_progress, sim_progress, 5)),
        "progress_confusion_real_rows_sim_columns": progress_confusion.tolist(),
        "sim_attempted_object_slots": int(sum(int(row["sim_attempted_objects"]) for row in rows)),
        "real_attempted_object_slots": int(sum(int(row["real_attempted_objects"]) for row in rows)),
    }


def multiclass_kappa(matrix: np.ndarray) -> float | None:
    total = matrix.sum()
    observed = np.trace(matrix) / total
    expected = float(np.sum(matrix.sum(axis=1) * matrix.sum(axis=0)) / total**2)
    return safe_div(observed - expected, 1 - expected)


def cramers_v(matrix: np.ndarray) -> float | None:
    if matrix.sum() == 0:
        return None
    expected = np.outer(matrix.sum(axis=1), matrix.sum(axis=0)) / matrix.sum()
    mask = expected > 0
    chi_square = float(np.sum((matrix[mask] - expected[mask]) ** 2 / expected[mask]))
    denominator = matrix.sum() * min(matrix.shape[0] - 1, matrix.shape[1] - 1)
    return safe_div(math.sqrt(chi_square), math.sqrt(denominator))


def normalized_mutual_information(matrix: np.ndarray) -> float | None:
    probabilities = matrix / matrix.sum()
    row_p = probabilities.sum(axis=1)
    col_p = probabilities.sum(axis=0)
    expected = np.outer(row_p, col_p)
    mask = probabilities > 0
    mutual_information = float(np.sum(probabilities[mask] * np.log(probabilities[mask] / expected[mask])))
    row_entropy = float(-np.sum(row_p[row_p > 0] * np.log(row_p[row_p > 0])))
    col_entropy = float(-np.sum(col_p[col_p > 0] * np.log(col_p[col_p > 0])))
    return safe_div(mutual_information, math.sqrt(row_entropy * col_entropy))


def apply_failure_taxonomy(
    pairs: Sequence[dict[str, Any]], taxonomy: dict[str, set[str]]
) -> list[dict[str, Any]]:
    remapped: list[dict[str, Any]] = []
    for row in pairs:
        copy = dict(row)
        for domain in ("sim", "real"):
            copy[f"{domain}_failure_categories"] = sorted(
                failure_category_set(
                    {"fail_type": row[f"{domain}_fail_type"]}, bool(row[f"{domain}_success"]), taxonomy
                )
            )
        remapped.append(copy)
    return remapped


def failure_analysis(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    categories = ["success", "semantic", "grasping", "placement", "unknown"]
    matrix = np.zeros((len(categories), len(categories)), dtype=int)
    token_counts = {"sim": defaultdict(int), "real": defaultdict(int)}
    blank_failures = {"sim": 0, "real": 0}
    for row in pairs:
        real_primary = primary_category(set(row["real_failure_categories"]))
        sim_primary = primary_category(set(row["sim_failure_categories"]))
        matrix[categories.index(real_primary), categories.index(sim_primary)] += 1
        for domain in ("sim", "real"):
            tokens = failure_tokens({"fail_type": row[f"{domain}_fail_type"]})
            if not row[f"{domain}_success"] and not tokens:
                blank_failures[domain] += 1
            for token in tokens:
                token_counts[domain][token] += 1

    both_failed = [row for row in pairs if not row["sim_success"] and not row["real_success"]]
    exact_sets = sum(set(row["sim_failure_categories"]) == set(row["real_failure_categories"]) for row in both_failed)
    overlaps = sum(bool(set(row["sim_failure_categories"]) & set(row["real_failure_categories"])) for row in both_failed)
    jaccards = []
    for row in both_failed:
        sim_set = set(row["sim_failure_categories"])
        real_set = set(row["real_failure_categories"])
        jaccards.append(len(sim_set & real_set) / len(sim_set | real_set))

    failure_categories = ["semantic", "grasping", "placement", "unknown"]
    failure_matrix = matrix[1:, 1:]
    sim_failed_counts = failure_matrix.sum(axis=0)
    real_failed_counts = failure_matrix.sum(axis=1)
    # These marginals are among joint failures; separately compute all-failure mixes.
    all_failure_mix: dict[str, dict[str, int]] = {"sim": defaultdict(int), "real": defaultdict(int)}
    for row in pairs:
        for domain in ("sim", "real"):
            if not row[f"{domain}_success"]:
                all_failure_mix[domain][primary_category(set(row[f"{domain}_failure_categories"]))] += 1
    sim_mix = np.array([all_failure_mix["sim"][category] for category in failure_categories], dtype=float)
    real_mix = np.array([all_failure_mix["real"][category] for category in failure_categories], dtype=float)
    sim_probability = sim_mix / sim_mix.sum()
    real_probability = real_mix / real_mix.sum()
    midpoint = (sim_probability + real_probability) / 2

    def kl_bits(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    per_category: list[dict[str, Any]] = []
    for index, category in enumerate(failure_categories):
        tp = int(failure_matrix[index, index])
        predicted = int(sim_failed_counts[index])
        actual = int(real_failed_counts[index])
        precision = safe_div(tp, predicted)
        recall = safe_div(tp, actual)
        f1 = safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
        per_category.append({"category": category, "precision": precision, "recall": recall, "f1": f1})

    known_tokens = set().union(*FAILURE_TOKENS.values())
    return {
        "status": "exploratory; free-form labels were mapped post hoc and several choices are judgment calls",
        "categories": categories,
        "all_episode_primary_confusion_real_rows_sim_columns": matrix.tolist(),
        "all_episode_primary_agreement": float(np.trace(matrix) / matrix.sum()),
        "all_episode_primary_kappa": json_float(multiclass_kappa(matrix)),
        "all_episode_cramers_v": json_float(cramers_v(matrix)),
        "all_episode_normalized_mutual_information": json_float(normalized_mutual_information(matrix)),
        "both_failed_n": len(both_failed),
        "both_failed_categories": failure_categories,
        "both_failed_primary_confusion_real_rows_sim_columns": failure_matrix.tolist(),
        "both_failed_primary_agreement": float(np.trace(failure_matrix) / failure_matrix.sum()),
        "both_failed_primary_kappa": json_float(multiclass_kappa(failure_matrix)),
        "both_failed_cramers_v": json_float(cramers_v(failure_matrix)),
        "both_failed_exact_category_set_matches": exact_sets,
        "both_failed_exact_category_set_rate": exact_sets / len(both_failed),
        "both_failed_any_category_overlap": overlaps,
        "both_failed_any_category_overlap_rate": overlaps / len(both_failed),
        "both_failed_mean_jaccard": float(np.mean(jaccards)),
        "per_category_sim_as_predictor_of_real_among_joint_failures": per_category,
        "all_sim_failure_primary_mix": dict(all_failure_mix["sim"]),
        "all_real_failure_primary_mix": dict(all_failure_mix["real"]),
        "failure_mix_total_variation_distance": float(0.5 * np.sum(np.abs(sim_probability - real_probability))),
        "failure_mix_jensen_shannon_divergence_bits": float(
            0.5 * kl_bits(sim_probability, midpoint) + 0.5 * kl_bits(real_probability, midpoint)
        ),
        "blank_failure_labels": blank_failures,
        "unmapped_tokens": {
            domain: {token: count for token, count in sorted(counts.items()) if token not in known_tokens}
            for domain, counts in token_counts.items()
        },
    }


def add_object_level_metrics(
    analysis: dict[str, Any], pairs: Sequence[dict[str, Any]], sim_rows: Sequence[dict[str, Any]], real_rows: Sequence[dict[str, Any]]
) -> None:
    object_pairs: list[dict[str, Any]] = []
    for pair in pairs:
        if pair["task"] != "bin_4obj":
            continue
        index = int(pair["pair_index"])
        sim_objects = sim_rows[index]["objects"]
        real_objects = real_rows[index]["objects"]
        if object_names(sim_rows[index]) != object_names(real_rows[index]):
            raise AssertionError(f"4-bin object order mismatch at pair {index}")
        for object_index, (sim_object, real_object) in enumerate(zip(sim_objects, real_objects)):
            object_pairs.append(
                {
                    "real_success": real_object.get("success") is True,
                    "sim_success": sim_object.get("success") is True,
                    "object_name": sim_object["name"],
                    "pair_index": index,
                    "object_index": object_index,
                }
            )
    analysis["object_level_completed_indicator"] = contingency_metrics(object_pairs)


def sensitivity_analysis(
    pairs: Sequence[dict[str, Any]],
    tasks: Sequence[dict[str, Any]],
    pre_redo: Sequence[dict[str, Any]],
    real_rows: Sequence[dict[str, Any]],
    provenance: Sequence[dict[str, Any]],
    redo_indices: set[int],
) -> dict[str, Any]:
    pre_pairs = build_pairs(list(tasks), list(pre_redo), list(real_rows), list(provenance), set())
    clean_pairs = [row for row in pairs if int(row["pair_index"]) not in redo_indices]
    selected_final = [row for row in pairs if int(row["pair_index"]) in redo_indices]
    selected_original = [row for row in pre_pairs if int(row["pair_index"]) in redo_indices]
    transitions = defaultdict(int)
    for before, after in zip(selected_original, selected_final):
        transitions[f"{int(before['sim_success'])}->{int(after['sim_success'])}"] += 1
    by_task: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        final_task = [row for row in pairs if row["task"] == task]
        original_task = [row for row in pre_pairs if row["task"] == task]
        clean_task = [row for row in clean_pairs if row["task"] == task]
        by_task.append(
            {
                "task": task,
                "n": len(final_task),
                "original_sim_successes": sum(row["sim_success"] for row in original_task),
                "original_sim_success_rate": sum(row["sim_success"] for row in original_task) / len(original_task),
                "final_sim_successes": sum(row["sim_success"] for row in final_task),
                "final_sim_success_rate": sum(row["sim_success"] for row in final_task) / len(final_task),
                "exclude_flagged_n": len(clean_task),
                "exclude_flagged_sim_success_rate": (
                    sum(row["sim_success"] for row in clean_task) / len(clean_task) if clean_task else None
                ),
            }
        )
    return {
        "final_replaced": contingency_metrics(pairs),
        "original_pre_redo": contingency_metrics(pre_pairs),
        "exclude_all_redo_flagged": contingency_metrics(clean_pairs),
        "redo_flagged_n": len(redo_indices),
        "redo_selected_original_sim_successes": sum(row["sim_success"] for row in selected_original),
        "redo_selected_final_sim_successes": sum(row["sim_success"] for row in selected_final),
        "redo_outcome_transitions_original_to_final": dict(transitions),
        "by_task": by_task,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            for key, value in encoded.items():
                if isinstance(value, list):
                    encoded[key] = json.dumps(value, separators=(",", ":"))
            writer.writerow(encoded)


def save_condition_scatter(rows: list[dict[str, Any]], metrics: dict[str, Any], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 7.2), constrained_layout=True)
    for row in rows:
        axis.scatter(
            row["real_success_rate"],
            row["sim_success_rate"],
            s=35 + row["n"] * 1.8,
            color=TASK_COLORS[row["task"]],
            marker=OOD_MARKERS[row["ood_key"]],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
        )
    axis.plot([0, 1], [0, 1], "--", color="#666666", linewidth=1)
    ticks = np.linspace(0, 1, 6)
    axis.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), xlabel="Matched real success rate", ylabel="Sim success rate")
    axis.set_xticks(ticks, [f"{100*x:.0f}%" for x in ticks])
    axis.set_yticks(ticks, [f"{100*x:.0f}%" for x in ticks])
    axis.grid(True, color="#dddddd", linewidth=0.7)
    axis.set_title(
        "Task × OOD condition correspondence\n"
        f"Pearson r={metrics['pearson_r']:.3f}; Spearman ρ={metrics['spearman_rho']:.3f}; "
        f"weighted MAE={100*metrics['episode_weighted_mae']:.1f} pp"
    )
    task_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=TASK_COLORS[task], markeredgecolor="white", label=TASK_LABELS[task])
        for task in TASK_ORDER
    ]
    ood_handles = [
        plt.Line2D([0], [0], marker=OOD_MARKERS[ood], linestyle="none", color="#333333", label=OOD_LABELS[ood])
        for ood in OOD_ORDER
    ]
    first = axis.legend(handles=task_handles, title="Task", loc="lower right", fontsize=8)
    axis.add_artist(first)
    axis.legend(handles=ood_handles, title="OOD split", loc="upper left", fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_rate_bars(rows: list[dict[str, Any]], label_key: str, path: Path, title: str) -> None:
    labels = [TASK_LABELS[row[label_key]] if label_key == "task" else OOD_LABELS[row[label_key]] for row in rows]
    positions = np.arange(len(rows))
    width = 0.36
    fig, axis = plt.subplots(figsize=(max(7.5, len(rows) * 1.5), 5.2), constrained_layout=True)
    axis.bar(positions - width / 2, [row["real_success_rate"] for row in rows], width, label="Real-only policy in real", color="#f58518")
    axis.bar(positions + width / 2, [row["sim_success_rate"] for row in rows], width, label="Sim-only policy in sim", color="#4c78a8")
    for position, row in zip(positions, rows):
        axis.text(position - width / 2, row["real_success_rate"] + 0.018, f"{100*row['real_success_rate']:.1f}%", ha="center", fontsize=8)
        axis.text(position + width / 2, row["sim_success_rate"] + 0.018, f"{100*row['sim_success_rate']:.1f}%", ha="center", fontsize=8)
    axis.set_xticks(positions, labels, rotation=15 if label_key == "ood_key" else 0, ha="right" if label_key == "ood_key" else "center")
    axis.set_ylim(0, 1.05)
    axis.set_yticks(np.linspace(0, 1, 6), [f"{x:.0f}%" for x in np.linspace(0, 100, 6)])
    axis.set_ylabel("Task success rate")
    axis.set_title(title)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_confusion(matrix: np.ndarray, xlabels: Sequence[str], ylabels: Sequence[str], title: str, path: Path, cmap: str) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    image = axis.imshow(matrix, cmap=cmap)
    axis.set_xticks(range(len(xlabels)), xlabels, rotation=25, ha="right")
    axis.set_yticks(range(len(ylabels)), ylabels)
    axis.set_title(title)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > threshold else "#222222")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def percent(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def interval_percent(interval: Sequence[float | None]) -> str:
    if interval[0] is None:
        return "—"
    return f"[{100*interval[0]:.2f}, {100*interval[1]:.2f}] pp"


def markdown_rate_table(rows: Sequence[dict[str, Any]], key: str) -> str:
    label_header = "Task" if key == "task" else "OOD split"
    lines = [
        f"| {label_header} | Sim | Matched real | Sim − real (95% semantic-cluster CI) | Agreement | MCC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        label = TASK_LABELS[row[key]] if key == "task" else OOD_LABELS[row[key]]
        gap = 100 * row["sim_minus_real_rate"]
        cluster_interval = row["semantic_cluster_bootstrap_ci95"]["sim_minus_real_rate"]
        lines.append(
            f"| {label} | {row['sim_successes']}/{row['n']} ({percent(row['sim_success_rate'])}) "
            f"| {row['real_successes']}/{row['n']} ({percent(row['real_success_rate'])}) "
            f"| {gap:+.2f} pp {interval_percent(cluster_interval)} "
            f"| {percent(row['raw_agreement'])} | {row['mcc_phi']:.3f} |"
        )
    return "\n".join(lines)


def build_report(
    metrics: dict[str, Any], task_rows: list[dict[str, Any]], ood_rows: list[dict[str, Any]], hashes: dict[str, Any]
) -> str:
    overall = metrics["overall"]
    condition = metrics["condition_level"]
    progress = metrics["four_bin"]
    permutation = metrics["stratified_permutation"]
    block_permutation = metrics["semantic_cluster_block_permutation"]
    sensitivity = metrics["redo_sensitivity"]
    matching = metrics["matching"]
    ci = overall["bootstrap_ci95"]
    cluster_ci = overall["semantic_cluster_bootstrap_ci95"]
    target_cluster_ci = overall["target_object_cluster_bootstrap_ci95"]
    sensitivity_lines = [
        "| Task | Pre-redo sim | Final sim | Change | Excluding flagged |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sensitivity["by_task"]:
        sensitivity_lines.append(
            f"| {TASK_LABELS[row['task']]} | {row['original_sim_successes']}/{row['n']} "
            f"({percent(row['original_sim_success_rate'])}) | {row['final_sim_successes']}/{row['n']} "
            f"({percent(row['final_sim_success_rate'])}) | "
            f"{100*(row['final_sim_success_rate']-row['original_sim_success_rate']):+.2f} pp | "
            f"{row['exclude_flagged_n']} rows, {percent(row['exclude_flagged_sim_success_rate'])} |"
        )
    sensitivity_table = "\n".join(sensitivity_lines)
    return f"""# Final GR00T paired real–sim correspondence analysis

Generated from {overall['n']} rows matched exactly on recorded task/semantic keys and canonical task ordering, with intended corresponding sim layouts. The final sim file includes the 76 manifest-indexed physics-QC redos.

## Headline outcome

The sim-only policy succeeded on **{overall['sim_successes']}/{overall['n']} ({percent(overall['sim_success_rate'])})** versus **{overall['real_successes']}/{overall['n']} ({percent(overall['real_success_rate'])})** for the real-only policy in the matched real trials. The paired gap is **{100*overall['sim_minus_real_rate']:+.2f} percentage points** (task×OOD-stratified semantic-configuration cluster bootstrap 95% CI {interval_percent(cluster_ci['sim_minus_real_rate'])}; broader target-object cluster sensitivity CI {interval_percent(target_cluster_ci['sim_minus_real_rate'])}; episode-pair bootstrap CI {interval_percent(ci['sim_minus_real_rate'])}).

The exact episode outcomes agree **{percent(overall['raw_agreement'])}** of the time (episode-pair bootstrap 95% CI [{100*ci['raw_agreement'][0]:.2f}%, {100*ci['raw_agreement'][1]:.2f}%]); MCC is **{overall['mcc_phi']:.3f}** (95% CI [{ci['mcc_phi'][0]:.3f}, {ci['mcc_phi'][1]:.3f}]) and Cohen's κ is **{overall['cohen_kappa']:.3f}**. The paired table is:

| | Sim success | Sim failure |
|---|---:|---:|
| Real success | {overall['both_success_n11']} | {overall['real_success_sim_failure_n10']} |
| Real failure | {overall['real_failure_sim_success_n01']} | {overall['both_failure_n00']} |

Positive agreement is {percent(overall['positive_agreement'])}, negative agreement is {percent(overall['negative_agreement'])}, P(sim success | real success) is {percent(overall['sensitivity_sim_success_given_real_success'])}, and P(sim failure | real failure) is {percent(overall['specificity_sim_failure_given_real_failure'])}.

These episode-level numbers are secondary: the rows match task semantics and object sets, but the real and simulated object positions/poses are not the same. A pose-sensitive stochastic policy is not expected to reproduce the same binary outcome across those physically different starts. Condition-level success rates and task×OOD failure composition are the more relevant correspondence targets.

McNemar's exact two-sided p-value is {overall['mcnemar_exact_two_sided_p']:.3g}; this is an episode-level descriptive test whose nominal calibration ignores repeated-object/configuration clustering and the fixed curated cohort. It reflects the directional marginal gap, not simulator causality.

## Cohort construction and selection

The two raw real result files contain {matching['raw_real_source_rows']} rows. The canonical filtered task file retains {overall['n']} and excludes {matching['excluded_real_rows']}. Matched included real success is {overall['real_successes']}/{overall['n']} ({percent(overall['real_success_rate'])}), whereas the excluded real rows are {matching['excluded_real_successes']}/{matching['excluded_real_rows']} ({percent(matching['excluded_real_success_rate'])}). Selection therefore materially changes the evaluated difficulty distribution.

These results apply to this **curated correspondence cohort**, not automatically to the full real evaluation distribution. The filtered task file is the operational inclusion list; a paper should state every substantive filtering criterion alongside the exact source-line manifest emitted here.

## By task

{markdown_rate_table(task_rows, 'task')}

The 1-object and 4-object bin episodes are deliberately separate tasks. A 4-object episode counts as task success only when all four objects succeeded.

## By OOD split

{markdown_rate_table(ood_rows, 'ood_key')}

## Relative difficulty correspondence

Across the 15 task×OOD cells, Pearson r is **{condition['pearson_r']:.3f}** (bootstrap 95% CI [{condition['pearson_bootstrap_ci95'][0]:.3f}, {condition['pearson_bootstrap_ci95'][1]:.3f}]) and Spearman ρ is **{condition['spearman_rho']:.3f}** (CI [{condition['spearman_bootstrap_ci95'][0]:.3f}, {condition['spearman_bootstrap_ci95'][1]:.3f}]). The episode-weighted Pearson r is {condition['weighted_pearson_r']:.3f}; weighted MAE is {100*condition['episode_weighted_mae']:.2f} pp and weighted RMSE is {100*condition['episode_weighted_rmse']:.2f} pp.

This is an ecological, small-N condition-level association and can be driven by between-task difficulty; it is not an episode-level correlation or an absolute-calibration metric. The weighted r is secondary because large cells and task composition dominate it.

This is the strongest correspondence finding: the two policy-domain systems preserve much of the aggregate task/OOD difficulty ordering even though exact episode agreement is only moderate and sim is systematically lower. Observed agreement is {100*block_permutation['excess_agreement_over_stratified_null']:.2f} pp above a task×OOD-stratified independence null that shuffles whole repeated-trial semantic blocks (one-sided descriptive p={block_permutation['agreement_one_sided_p']:.3g}). The row-shuffle sensitivity is {100*permutation['excess_agreement_over_stratified_null']:.2f} pp (p={permutation['agreement_one_sided_p']:.3g}). Neither permutation p-value is causal; target-object dependence and the fixed curated cohort limit nominal calibration.

## Four-object bin progress

- All-four task success: sim {progress['sim_task_successes']}/{progress['episodes']} ({percent(progress['sim_task_success_rate'])}); real {progress['real_task_successes']}/{progress['episodes']} ({percent(progress['real_task_success_rate'])}).
- Object progress: sim {progress['sim_completed_objects']}/{progress['total_objects']} ({percent(progress['sim_progress_rate'])}); real {progress['real_completed_objects']}/{progress['total_objects']} ({percent(progress['real_progress_rate'])}).
- Progress-level Spearman ρ={progress['progress_spearman_rho']:.3f}; quadratic weighted κ={progress['progress_quadratic_weighted_kappa']:.3f}; mean absolute progress error={progress['progress_mae_objects']:.3f} objects per episode.
- Per-object completed/not-completed MCC={progress['object_level_completed_indicator']['mcc_phi']:.3f} and κ={progress['object_level_completed_indicator']['cohen_kappa']:.3f}. A null attempt/success is retained as “not attempted”; it contributes zero completed objects but is not relabeled as a grasp failure.

## Redo sensitivity

The post-hoc physics review flagged {sensitivity['redo_flagged_n']}/{overall['n']} episodes. The selected set changed from {sensitivity['redo_selected_original_sim_successes']}/{sensitivity['redo_flagged_n']} successes before rerunning to {sensitivity['redo_selected_final_sim_successes']}/{sensitivity['redo_flagged_n']} after rerunning. Overall sim success changed from {sensitivity['original_pre_redo']['sim_successes']}/{overall['n']} ({percent(sensitivity['original_pre_redo']['sim_success_rate'])}) to {overall['sim_successes']}/{overall['n']} ({percent(overall['sim_success_rate'])}). On the {sensitivity['exclude_all_redo_flagged']['n']}-episode subset excluding every flagged row, sim success is {percent(sensitivity['exclude_all_redo_flagged']['sim_success_rate'])}, real success is {percent(sensitivity['exclude_all_redo_flagged']['real_success_rate'])}, and MCC is {sensitivity['exclude_all_redo_flagged']['mcc_phi']:.3f}.

{sensitivity_table}

Because episodes were selected for redo after video review, the final replacement analysis is primary but the pre-redo and exclude-flagged results must remain disclosed.

The flagged set was disproportionately successful under the original score (57/76), and each was replaced by one new stochastic rollout (30/76). The change can combine corrected physics, policy stochasticity, regression to the mean, and outcome-aware selection. The final score is best described as a **QC-adjudicated benchmark outcome**, not an unbiased estimate from repeat trials.

## Failure modes

Episode-level failure-category correspondence is not emphasized because real and sim episodes use different object poses and positions. A later, authoritative sim-only analysis groups failed next-to, between, and move episodes by task and OOD using the explicitly supplied semantic/grasping/placement rules. See [`sim_failure_buckets/report.md`](sim_failure_buckets/report.md), with every row-level decision in [`sim_failure_buckets/classified_failure_rows.jsonl`](sim_failure_buckets/classified_failure_rows.jsonl).

The legacy paired-taxonomy values remain in `metrics.json` only for provenance and should not be used as the primary failure analysis.

## Literature-to-metric rationale

| Work | Relevant design/metric | Use here |
|---|---|---|
| [SIMPLER (CoRL 2024; PMLR 2025)](https://proceedings.mlr.press/v270/li25c.html) | Same-policy sim/real evaluation; Pearson correlation plus MMRV for policy rankings | Report condition-level correlation, but pair it with bias/MAE and do not use MMRV with one policy per domain |
| [PolaRiS (2025)](https://arxiv.org/abs/2512.16881) and [REALM (RA-L 2026)](https://arxiv.org/abs/2512.19562) | Normalized task progress, Pearson, MMRV, perturbation stratification | Keep task×OOD strata and report 4-bin object progress separately from all-four success |
| [SimFoundry (2026)](https://arxiv.org/abs/2606.28276) | Subtask scores expose correspondence/bottlenecks hidden by binary task success | Use per-object 4-bin completion and progress confusion |
| [RoboWorld (2026)](https://arxiv.org/abs/2607.01060) | Pearson, Spearman, trial-count sensitivity, progress-aware rubrics | Report both linear and rank correspondence; use progress rather than binary success alone |
| [SureSim (2025)](https://arxiv.org/abs/2510.04354) | Same-policy paired real/sim bias correction and finite-sample inference; repeated nearby sim initializations | Motivates the stochastic-rollout/noise-ceiling caveat; its guarantees are not applicable to these different policies |
| [STAR-Gen (2025)](https://arxiv.org/abs/2503.01238) | Separates semantic, visual, and behavioral shifts | Report task and OOD strata explicitly instead of one pooled rate |

## What this experiment can and cannot demonstrate

Most recent real2sim evaluation work—including [SIMPLER](https://proceedings.mlr.press/v270/li25c.html), [REALM](https://arxiv.org/abs/2512.19562), [SimFoundry](https://arxiv.org/abs/2606.28276), and [RoboWorld](https://arxiv.org/abs/2607.01060)—holds policy weights fixed across real and simulation. [PolaRiS](https://arxiv.org/abs/2512.16881) is an adapted-policy exception: it shallow-fine-tunes policies for 1,000 sim co-training steps before sim evaluation. Here, by contrast, the two sides are independently fine-tuned real-only and sim-only policies. Training regime/policy and domain are therefore perfectly confounded to a much greater degree.

The defensible claim is: **condition-level behavioral correspondence between independently trained real-only and sim-only GR00T policies on matched task specifications in an intended digital twin**. The data do not isolate simulator fidelity, prove that sim predicts deployment of this exact policy in reality, or identify whether a mismatch comes from rendering, physics, initial-state/reset fidelity, scoring/annotation, training data, or policy stochasticity. Because there are no repeated real trials, they also cannot estimate the real-domain test–retest/noise ceiling. [SureSim](https://arxiv.org/abs/2510.04354) is relevant to that sampling problem, but its same-policy paired inference guarantees do not transfer to this confounded design.

The manual success/failure and free-form failure labels were produced by a domain-aware reviewer without blinded duplicate annotation or an inter-rater reliability estimate. That is a measurement limitation, particularly for the failure taxonomy.

MMRV is intentionally not reported: it ranks multiple policies by their real-vs-sim performance and is undefined/trivial with only one policy per domain. Pearson/Spearman here are condition-level summaries, and are reported alongside absolute bias, MAE, episode agreement, paired tests, and progress.

## Reproducibility and uncertainty

- Reproduce all tables, paired files, statistics, and plots with `~/env_isaaclab_51/bin/python scripts/analyze_groot_sim_real_correspondence.py`.
- Matching uses the canonical 906-row task order. IF tasks match exact ordered objects + instruction + OOD, with target/referents checked when present. Six real Move rows lack recorded referents; their exact ordered objects + instruction + OOD keys are unique and the instruction names the target. Bin tasks match OOD + object count + ordered object names + clutter using FIFO source order for repeated 1-bin trials. `trial_id` is never used as a cross-domain join key. This proves recorded semantic-row correspondence, not independently measured equality of every real and sim initial pose.
- Primary intervals jointly resample exact semantic configurations within task×OOD cells ({metrics['configuration']['bootstrap_replicates']} replicates); this clusters the eight repeated 1-bin trials for each object. `metrics.json` also retains episode-pair intervals and a more conservative sensitivity that clusters all trials sharing the same target object within task×OOD. These describe sampling from a conceptual matched task distribution; all rates for this fixed curated benchmark are exact.
- The primary permutation sensitivity shuffles exact semantic-configuration blocks within task×OOD and equal-block-size groups, preserving repeated 1-bin rows and both domains' margins ({metrics['configuration']['permutation_replicates']} randomizations). A row-shuffle result is retained as a secondary sensitivity.
- Repeated objects across otherwise distinct configurations still make full episode-level independence imperfect; this is why the target-object-level cluster sensitivity is included.
- Task/OOD subgroup intervals are descriptive and are not multiplicity-adjusted; avoid turning individual subgroup intervals into a family of confirmatory significance claims.
- Input hashes and exact source-line provenance are in `input_hashes.json`, `matched_real_world_manifest.jsonl`, and `paired_results.jsonl`.

Merged sim SHA-256: `{hashes['final_sim_results']['sha256']}`
Pre-redo SHA-256: `{hashes['pre_redo_sim_results']['sha256']}`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-results", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--pre-redo-sim-results", type=Path, default=DEFAULT_PRE_REDO)
    parser.add_argument("--redo-review", type=Path, default=DEFAULT_REDO_REVIEW)
    parser.add_argument("--redo-scores", type=Path, default=DEFAULT_REDO_SCORES)
    parser.add_argument("--redo-manifest", type=Path, default=DEFAULT_REDO_MANIFEST)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--real-if-results", type=Path, default=DEFAULT_REAL_IF)
    parser.add_argument("--real-bin-results", type=Path, default=DEFAULT_REAL_BIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--permutation-replicates", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = {
        "final_sim_results": args.sim_results,
        "pre_redo_sim_results": args.pre_redo_sim_results,
        "redo_review": args.redo_review,
        "redo_scores": args.redo_scores,
        "redo_manifest": args.redo_manifest,
        "canonical_tasks": args.tasks,
        "real_if_results": args.real_if_results,
        "real_bin_results": args.real_bin_results,
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")

    sim_rows = load_jsonl(args.sim_results)
    pre_redo_rows = load_jsonl(args.pre_redo_sim_results)
    review_rows = load_jsonl(args.redo_review)
    redo_scores = load_jsonl(args.redo_scores)
    manifest = load_jsonl(args.redo_manifest)
    tasks = load_jsonl(args.tasks)
    real_if = load_jsonl(args.real_if_results)
    real_bin = load_jsonl(args.real_bin_results)
    if not len(tasks) == len(sim_rows) == len(pre_redo_rows) == len(review_rows) == 906:
        raise ValueError(
            f"Expected 906 canonical/final/pre-redo/review rows; got "
            f"{len(tasks)}/{len(sim_rows)}/{len(pre_redo_rows)}/{len(review_rows)}"
        )
    if len(manifest) != len(redo_scores):
        raise ValueError(f"Redo manifest/scores length mismatch: {len(manifest)} != {len(redo_scores)}")

    redo_indices = {int(row["original_episode_index"]) for row in manifest}
    review_indices = {index for index, row in enumerate(review_rows) if row.get("redo") is True}
    if redo_indices != review_indices or len(redo_indices) != 76:
        raise ValueError("Redo manifest does not exactly match the 76 redo=true review rows")
    for redo_index, (manifest_row, redo_row) in enumerate(zip(manifest, redo_scores)):
        original_index = int(manifest_row["original_episode_index"])
        verify_score_context(tasks[original_index], redo_row, original_index, f"redo score {redo_index}")
        if sim_rows[original_index] != redo_row:
            raise ValueError(f"Final sim row {original_index} is not the scored redo row {redo_index}")

    matched_real, provenance = match_real_rows(tasks, real_if, real_bin)
    pairs = build_pairs(tasks, sim_rows, matched_real, provenance, redo_indices)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hashes = {
        name: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in inputs.items()
    }
    write_jsonl(args.output_dir / "matched_real_world_results.jsonl", matched_real)
    write_jsonl(args.output_dir / "matched_real_world_manifest.jsonl", provenance)
    write_jsonl(args.output_dir / "paired_results.jsonl", pairs)
    write_json(args.output_dir / "input_hashes.json", hashes)

    rng = np.random.default_rng(args.seed)
    strata = [f"{row['task']}::{row['ood_key']}" for row in pairs]
    overall = contingency_metrics(pairs)
    overall["bootstrap_ci95"] = bootstrap_binary(pairs, strata, args.bootstrap_replicates, rng)
    overall["semantic_cluster_bootstrap_ci95"] = cluster_bootstrap_binary(
        pairs, strata, args.bootstrap_replicates, rng
    )
    overall["target_object_cluster_bootstrap_ci95"] = cluster_bootstrap_binary(
        pairs, strata, args.bootstrap_replicates, rng, target_object_cluster_key
    )
    by_task = group_rows(pairs, ["task"], args.bootstrap_replicates, rng)
    by_ood = group_rows(pairs, ["ood_key"], args.bootstrap_replicates, rng)
    by_condition = group_rows(pairs, ["task", "ood_key"], args.bootstrap_replicates, rng)
    condition = condition_correspondence(by_condition, pairs, args.bootstrap_replicates, rng)
    permutation = stratified_permutation_test(by_condition, overall, args.permutation_replicates, rng)
    block_permutation = semantic_cluster_block_permutation_test(
        pairs, overall, args.permutation_replicates, rng
    )
    four_bin = four_bin_analysis(pairs)
    add_object_level_metrics(four_bin, pairs, sim_rows, matched_real)
    failures = failure_analysis(pairs)
    legacy_failure_pairs = apply_failure_taxonomy(pairs, LEGACY_IMPRECISE_AS_PLACEMENT_TOKENS)
    legacy_failures = failure_analysis(legacy_failure_pairs)
    sensitivity = sensitivity_analysis(pairs, tasks, pre_redo_rows, matched_real, provenance, redo_indices)
    used_if_indices = {
        int(row["real_source_index"])
        for row in provenance
        if row["real_source_file"] == "results_gr00t_IF_WM.jsonl"
    }
    used_bin_indices = {
        int(row["real_source_index"])
        for row in provenance
        if row["real_source_file"] == "results_gr00t_grasp_WM.jsonl"
    }
    excluded_if = [row for index, row in enumerate(real_if) if index not in used_if_indices]
    excluded_bin = [row for index, row in enumerate(real_bin) if index not in used_bin_indices]
    excluded_if_successes = sum(row.get("success") is True for row in excluded_if)
    excluded_bin_successes = sum(
        bool(row.get("objects"))
        and all(isinstance(item, dict) and item.get("success") is True for item in row["objects"])
        for row in excluded_bin
    )
    excluded_total = len(excluded_if) + len(excluded_bin)
    excluded_successes = excluded_if_successes + excluded_bin_successes

    metrics = {
        "configuration": {
            "seed": args.seed,
            "bootstrap_replicates": args.bootstrap_replicates,
            "permutation_replicates": args.permutation_replicates,
            "bootstrap_unit": "matched episode pair, resampled within task×OOD",
        },
        "matching": {
            "paired_rows": len(pairs),
            "real_if_source_rows": len(real_if),
            "real_bin_source_rows": len(real_bin),
            "matched_if_rows": sum(not row["task"].startswith("bin_") for row in pairs),
            "matched_bin_rows": sum(row["task"].startswith("bin_") for row in pairs),
            "excluded_real_if_rows": len(real_if) - sum(not row["task"].startswith("bin_") for row in pairs),
            "excluded_real_bin_rows": len(real_bin) - sum(row["task"].startswith("bin_") for row in pairs),
            "raw_real_source_rows": len(real_if) + len(real_bin),
            "excluded_real_rows": excluded_total,
            "excluded_real_successes": excluded_successes,
            "excluded_real_success_rate": excluded_successes / excluded_total,
            "excluded_if_successes": excluded_if_successes,
            "excluded_bin_successes": excluded_bin_successes,
            "exact_semantic_configuration_clusters": len({semantic_cluster_key(row) for row in pairs}),
            "target_object_clusters_within_task_ood": len({target_object_cluster_key(row) for row in pairs}),
        },
        "overall": overall,
        "by_task": by_task,
        "by_ood": by_ood,
        "by_task_ood": by_condition,
        "condition_level": condition,
        "stratified_permutation": permutation,
        "semantic_cluster_block_permutation": block_permutation,
        "four_bin": four_bin,
        "failure_modes_exploratory": failures,
        "failure_modes_taxonomy_sensitivity_imprecise_as_placement": legacy_failures,
        "redo_sensitivity": sensitivity,
    }
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(
        args.output_dir / "failure_taxonomy.json",
        {
            "normalization": "split on commas; strip whitespace; lowercase; strip trailing periods",
            "primary_category_priority": FAILURE_PRIORITY,
            "token_to_category": {
                token: category for category, tokens in FAILURE_TOKENS.items() for token in sorted(tokens)
            },
            "generic_imprecise_rationale": (
                "Primary: grasping, following the stated protocol that failed grasps were subclassified as imprecise. "
                "Distance-qualified imprecise-by-Ncm labels remain placement."
            ),
            "legacy_sensitivity_token_to_category": {
                token: category
                for category, tokens in LEGACY_IMPRECISE_AS_PLACEMENT_TOKENS.items()
                for token in sorted(tokens)
            },
            "unmapped_category": "unknown",
            "warning": "Post-hoc exploratory taxonomy; ambiguous assignments must be disclosed.",
        },
    )

    fields = [
        "task",
        "ood_key",
        "n",
        "sim_successes",
        "sim_success_rate",
        "real_successes",
        "real_success_rate",
        "sim_minus_real_rate",
        "sim_minus_real_rate_ci95",
        "semantic_cluster_bootstrap_ci95",
        "raw_agreement",
        "cohen_kappa",
        "mcc_phi",
        "mcnemar_exact_two_sided_p",
    ]
    write_csv(args.output_dir / "success_rates_by_task.csv", by_task, fields)
    write_csv(args.output_dir / "success_rates_by_ood.csv", by_ood, fields)
    write_csv(args.output_dir / "success_rates_by_task_ood.csv", by_condition, fields)

    save_condition_scatter(by_condition, condition, args.output_dir / "condition_correspondence.png")
    save_rate_bars(by_task, "task", args.output_dir / "success_rates_by_task.png", "Matched real vs sim task success")
    save_rate_bars(by_ood, "ood_key", args.output_dir / "success_rates_by_ood.png", "Matched real vs sim success by OOD split")
    outcome_matrix = np.array(
        [
            [overall["both_success_n11"], overall["real_success_sim_failure_n10"]],
            [overall["real_failure_sim_success_n01"], overall["both_failure_n00"]],
        ]
    )
    save_confusion(
        outcome_matrix,
        ["Sim success", "Sim failure"],
        ["Real success", "Real failure"],
        "Paired task outcomes",
        args.output_dir / "outcome_confusion.png",
        "Blues",
    )
    save_confusion(
        np.array(four_bin["progress_confusion_real_rows_sim_columns"]),
        [f"Sim {value}/4" for value in range(5)],
        [f"Real {value}/4" for value in range(5)],
        "4-object bin progress",
        args.output_dir / "four_bin_progress_confusion.png",
        "Greens",
    )
    save_confusion(
        np.array(failures["both_failed_primary_confusion_real_rows_sim_columns"]),
        [f"Sim {value}" for value in FAILURE_PRIORITY],
        [f"Real {value}" for value in FAILURE_PRIORITY],
        "Exploratory failure category correspondence",
        args.output_dir / "failure_category_confusion.png",
        "Purples",
    )
    save_confusion(
        np.array(failures["all_episode_primary_confusion_real_rows_sim_columns"]),
        ["Sim success", "Sim semantic", "Sim grasping", "Sim placement", "Sim unknown"],
        ["Real success", "Real semantic", "Real grasping", "Real placement", "Real unknown"],
        "All-episode outcome/failure correspondence",
        args.output_dir / "failure_category_all_episode_confusion.png",
        "Purples",
    )

    report = build_report(metrics, by_task, by_ood, hashes)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"paired rows: {len(pairs)}")
    print(f"sim success: {overall['sim_successes']}/{overall['n']} ({100*overall['sim_success_rate']:.2f}%)")
    print(f"real success: {overall['real_successes']}/{overall['n']} ({100*overall['real_success_rate']:.2f}%)")
    print(f"agreement={100*overall['raw_agreement']:.2f}% MCC={overall['mcc_phi']:.3f} kappa={overall['cohen_kappa']:.3f}")
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
