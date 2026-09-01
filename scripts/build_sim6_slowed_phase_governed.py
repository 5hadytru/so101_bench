#!/usr/bin/env python3
"""Build the reviewed sim_4 + sim_5 phase-governed slowed dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from lerobot.datasets.dataset_tools import merge_datasets, split_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset


OUTPUT_REPO_ID = "local/so101_bench_sim_6_slowed_1.7x_phase_governed"
OUTPUT_ROOT = Path("data/lerobot/so101_bench_sim_6_slowed_1.7x_phase_governed")
BUILD_ROOT = Path("data/lerobot/.build_so101_bench_sim_6_slowed_1.7x_phase_governed")
OUTPUT_TASKS = Path("tasks/so101_bench_sim_6_slowed_1.7x_phase_governed.jsonl")
OUTPUT_LAYOUTS = Path("tasks/layouts/so101_bench_sim_6_slowed_1.7x_phase_governed_layouts.jsonl")
OUTPUT_MANIFEST = Path("tasks/so101_bench_sim_6_slowed_1.7x_phase_governed_manifest.jsonl")
OUTPUT_SUMMARY = Path("outputs/so101_bench_sim_6_slowed_1.7x_phase_governed_build_summary.json")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def layouts_by_trial(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {str(row["trial_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{path} contains duplicate trial_id values")
    return result


def reviewed_component(sim_version: int, *, patch: bool) -> dict[str, Any]:
    if patch:
        source_stem = f"so101_bench_sim_{sim_version}_rerecord_round_1"
        slowed_stem = f"{source_stem}_slowed_1.7x_phase_governed"
        review_path = Path("outputs") / f"{slowed_stem}_resimulation/trajectory_review_delete_only.jsonl"
        tasks_path = Path("tasks") / f"{source_stem}_retained.jsonl"
        layouts_path = Path("tasks/layouts") / f"{source_stem}_retained_layouts.jsonl"
        patch_manifest_path = Path("tasks") / f"{source_stem}_retained_manifest.jsonl"
        patch_manifest = {
            int(row["rerecord_dataset_episode_index"]): row for row in read_jsonl(patch_manifest_path)
        }
        component_name = f"sim_{sim_version}_rerecorded"
    else:
        source_stem = f"so101_bench_sim_{sim_version}"
        slowed_stem = f"{source_stem}_slowed_1.7x_phase_governed"
        review_path = Path("outputs") / f"{slowed_stem}_resimulation/trajectory_review.jsonl"
        if sim_version == 4:
            tasks_path = Path("tasks/so101_bench_sim_4.jsonl")
            layouts_path = Path("tasks/layouts/so101_bench_sim_4_layouts.jsonl")
        else:
            tasks_path = Path("tasks/teleop_3.jsonl")
            layouts_path = Path("tasks/layouts/teleop_3_layouts_20260720_141740.jsonl")
        patch_manifest = None
        patch_manifest_path = None
        component_name = f"sim_{sim_version}_original"

    slowed_root = Path("data/lerobot") / slowed_stem
    review = read_jsonl(review_path)
    expected_source_count = 63 if patch and sim_version == 4 else 47 if patch else 885 if sim_version == 4 else 1377
    if len(review) != expected_source_count:
        raise ValueError(f"{review_path} has {len(review)} rows; expected {expected_source_count}")
    if len({int(row["source_dataset_episode_index"]) for row in review}) != len(review):
        raise ValueError(f"{review_path} contains duplicate source episode indices")
    undecided = [row for row in review if row.get("decision") not in {"keep", "rerecord", "delete"}]
    if undecided:
        raise ValueError(f"{review_path} contains invalid/unreviewed decisions")

    tasks = read_jsonl(tasks_path)
    layouts = layouts_by_trial(layouts_path)
    selected = []
    for row in review:
        if row["decision"] != "keep":
            continue
        slowed_index = row.get("slowed_dataset_episode_index")
        if slowed_index is None:
            raise ValueError(f"Keep decision lacks slowed episode mapping in {review_path}")
        source_index = int(row["source_dataset_episode_index"])
        source_mapping = patch_manifest[source_index] if patch_manifest is not None else None
        original_index = (
            int(source_mapping["source_dataset_episode_index"])
            if source_mapping is not None
            else source_index
        )
        task = tasks[source_index]
        trial_id = task.get("trial_id", source_index)
        layout = layouts[str(trial_id)]
        if task.get("instruction") != row.get("instruction") or task.get("instruction") != layout.get("instruction"):
            raise ValueError(f"Instruction mismatch in {component_name}, source episode {source_index}")
        selected.append(
            {
                "slowed_dataset_episode_index": int(slowed_index),
                "component_source_episode_index": source_index,
                "original_dataset_episode_index": original_index,
                "task": task,
                "layout": layout,
                "review": row,
                "source_mapping": source_mapping,
            }
        )

    selected.sort(key=lambda row: row["slowed_dataset_episode_index"])
    selected_indices = [row["slowed_dataset_episode_index"] for row in selected]
    if selected_indices != sorted(set(selected_indices)):
        raise ValueError(f"Duplicate or unordered slowed mappings in {review_path}")
    dataset_info = json.loads((slowed_root / "meta/info.json").read_text(encoding="utf-8"))
    if any(index < 0 or index >= int(dataset_info["total_episodes"]) for index in selected_indices):
        raise ValueError(f"Selected episode index is outside {slowed_root}")
    return {
        "name": component_name,
        "slowed_root": slowed_root,
        "review_path": review_path,
        "tasks_path": tasks_path,
        "layouts_path": layouts_path,
        "patch_manifest_path": patch_manifest_path,
        "selected": selected,
        "source_total": expected_source_count,
    }


def main() -> None:
    protected_outputs = (OUTPUT_ROOT, BUILD_ROOT, OUTPUT_TASKS, OUTPUT_LAYOUTS, OUTPUT_MANIFEST, OUTPUT_SUMMARY)
    existing = [path for path in protected_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing build output(s): {existing}")

    components = [
        reviewed_component(4, patch=False),
        reviewed_component(4, patch=True),
        reviewed_component(5, patch=False),
        reviewed_component(5, patch=True),
    ]
    expected_counts = [800, 60, 1324, 44]
    actual_counts = [len(component["selected"]) for component in components]
    if actual_counts != expected_counts:
        raise ValueError(f"Reviewed component counts changed: {actual_counts} != {expected_counts}")

    BUILD_ROOT.mkdir(parents=True)
    filtered_datasets = []
    output_tasks = []
    output_layouts = []
    output_manifest = []
    for component in components:
        print(f"[BUILD]: Filtering {component['name']}: {len(component['selected'])} episode(s)", flush=True)
        source = LeRobotDataset(
            f"local/{component['name']}_slowed",
            root=component["slowed_root"],
            video_backend="pyav",
        )
        split_root = BUILD_ROOT / component["name"]
        filtered = split_dataset(
            source,
            {"selected": [row["slowed_dataset_episode_index"] for row in component["selected"]]},
            output_dir=split_root,
        )["selected"]
        filtered_datasets.append(filtered)

        for component_output_index, row in enumerate(component["selected"]):
            merged_index = len(output_tasks)
            task = dict(row["task"])
            task["trial_id"] = merged_index
            layout = dict(row["layout"])
            layout["trial_id"] = merged_index
            layout["episode_index"] = merged_index
            output_tasks.append(task)
            output_layouts.append(layout)
            output_manifest.append(
                {
                    "schema_version": 1,
                    "merged_dataset_episode_index": merged_index,
                    "source_component": component["name"],
                    "source_component_output_episode_index": component_output_index,
                    "source_slowed_root": str(component["slowed_root"]),
                    "source_slowed_dataset_episode_index": row["slowed_dataset_episode_index"],
                    "source_component_episode_index": row["component_source_episode_index"],
                    "source_original_dataset_episode_index": row["original_dataset_episode_index"],
                    "instruction": task.get("instruction"),
                    "review_decision": row["review"]["decision"],
                }
            )

    print(f"[BUILD]: Merging {sum(actual_counts)} episodes into {OUTPUT_ROOT}", flush=True)
    merged = merge_datasets(filtered_datasets, output_repo_id=OUTPUT_REPO_ID, output_dir=OUTPUT_ROOT)
    if merged.meta.total_episodes != sum(actual_counts):
        raise ValueError(f"Merged episode count is {merged.meta.total_episodes}, expected {sum(actual_counts)}")

    write_jsonl(OUTPUT_TASKS, output_tasks)
    write_jsonl(OUTPUT_LAYOUTS, output_layouts)
    write_jsonl(OUTPUT_MANIFEST, output_manifest)
    summary = {
        "schema_version": 1,
        "output_repo_id": OUTPUT_REPO_ID,
        "output_root": str(OUTPUT_ROOT),
        "total_episodes": merged.meta.total_episodes,
        "total_frames": merged.meta.total_frames,
        "component_counts": dict(zip((component["name"] for component in components), actual_counts, strict=True)),
        "deleted_original_benchmark_episodes": 2262 - merged.meta.total_episodes,
        "source_review_sha256": {
            component["name"]: sha256(component["review_path"]) for component in components
        },
        "tasks_sha256": sha256(OUTPUT_TASKS),
        "layouts_sha256": sha256(OUTPUT_LAYOUTS),
        "manifest_sha256": sha256(OUTPUT_MANIFEST),
    }
    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
