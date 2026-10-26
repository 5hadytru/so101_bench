#!/usr/bin/env python3
"""Merge finalized sim6 and sim7 datasets into the SO-101 Bench training set."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rename_lerobot_camera_keys import migrate as rename_camera_keys


ROOTS = (
    Path("data/lerobot/so101_bench_sim_6_slowed_1.7x_phase_governed_domain_randomized"),
    Path("data/lerobot/so101_bench_sim_7_slowed_1.7x_phase_governed"),
)
TASKS = (
    Path("tasks/so101_bench_sim_6_slowed_1.7x_phase_governed_domain_randomized.jsonl"),
    Path("tasks/so101_bench_sim_7_slowed_1.7x_phase_governed.jsonl"),
)
LAYOUTS = (
    Path("tasks/layouts/so101_bench_sim_6_slowed_1.7x_phase_governed_domain_randomized_layouts.jsonl"),
    Path("tasks/layouts/so101_bench_sim_7_slowed_1.7x_phase_governed_layouts.jsonl"),
)
MANIFESTS = (
    Path("tasks/so101_bench_sim_6_slowed_1.7x_phase_governed_domain_randomized_manifest.jsonl"),
    Path("tasks/so101_bench_sim_7_slowed_1.7x_phase_governed_manifest.jsonl"),
)
OUTPUT_ROOT = Path("data/lerobot/so101_bench_sim")
OUTPUT_TASKS = Path("tasks/so101_bench_sim.jsonl")
OUTPUT_LAYOUTS = Path("tasks/layouts/so101_bench_sim_layouts.jsonl")
OUTPUT_MANIFEST = Path("tasks/so101_bench_sim_manifest.jsonl")
OUTPUT_SUMMARY = Path("outputs/so101_bench_sim_build_summary.json")
BUILD_ROOT = Path("data/lerobot/.build_so101_bench_sim")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def install() -> None:
    merged = BUILD_ROOT / "merged"
    staged = {
        OUTPUT_TASKS: BUILD_ROOT / OUTPUT_TASKS.name,
        OUTPUT_LAYOUTS: BUILD_ROOT / OUTPUT_LAYOUTS.name,
        OUTPUT_MANIFEST: BUILD_ROOT / OUTPUT_MANIFEST.name,
        OUTPUT_SUMMARY: BUILD_ROOT / OUTPUT_SUMMARY.name,
    }
    missing = [path for path in [merged / "meta/info.json", *staged.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete staged build: {missing}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if OUTPUT_ROOT.exists():
        OUTPUT_ROOT.replace(OUTPUT_ROOT.with_name(f"{OUTPUT_ROOT.name}_pre_merge_{stamp}"))
    merged.replace(OUTPUT_ROOT)
    for destination, source in staged.items():
        if destination.exists():
            destination.replace(destination.with_name(f"{destination.name}.pre_merge_{stamp}"))
        source.replace(destination)
    shutil.rmtree(BUILD_ROOT)
    print(f"Installed {OUTPUT_ROOT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--install-existing", action="store_true")
    args = parser.parse_args()
    if args.install_existing:
        install()
        return
    if BUILD_ROOT.exists():
        raise FileExistsError(f"Staging directory exists: {BUILD_ROOT}")

    datasets = [LeRobotDataset(f"local/sim_{i}", root=root, video_backend="pyav") for i, root in enumerate(ROOTS)]
    expected_counts = [dataset.meta.total_episodes for dataset in datasets]
    expected_frames = sum(dataset.meta.total_frames for dataset in datasets)
    if expected_counts != [2228, 1401]:
        raise ValueError(f"Unexpected component counts: {expected_counts}")
    if any(dataset.meta.features != datasets[0].meta.features for dataset in datasets[1:]):
        raise ValueError("Component feature schemas differ")

    task_groups, layout_groups, manifest_groups = map(
        lambda paths: [read_jsonl(path) for path in paths], (TASKS, LAYOUTS, MANIFESTS)
    )
    for index, count in enumerate(expected_counts):
        lengths = (len(task_groups[index]), len(layout_groups[index]), len(manifest_groups[index]))
        if lengths != (count, count, count):
            raise ValueError(f"Component {index} metadata counts {lengths} != dataset count {count}")

    merged = merge_datasets(datasets, output_repo_id="local/so101_bench_sim", output_dir=BUILD_ROOT / "merged")
    if merged.meta.total_episodes != sum(expected_counts) or merged.meta.total_frames != expected_frames:
        raise ValueError(
            f"Merged totals are episodes={merged.meta.total_episodes}, frames={merged.meta.total_frames}; "
            f"expected {sum(expected_counts)}, {expected_frames}"
        )

    # The historical sim6/sim7 components use ``front``.  Keep those source
    # datasets immutable, but publish the canonical training dataset with the
    # current ``wrist`` camera name.
    camera_rename_backup = BUILD_ROOT / "camera_rename_backup"
    rename_camera_keys(BUILD_ROOT / "merged", camera_rename_backup)
    shutil.rmtree(camera_rename_backup)

    final_tasks: list[dict[str, Any]] = []
    final_layouts: list[dict[str, Any]] = []
    final_manifest: list[dict[str, Any]] = []
    for component, (root, tasks, layouts, manifests) in enumerate(
        zip(ROOTS, task_groups, layout_groups, manifest_groups, strict=True)
    ):
        for source_index, (task_raw, layout_raw, source_manifest) in enumerate(
            zip(tasks, layouts, manifests, strict=True)
        ):
            output_index = len(final_tasks)
            task, layout = dict(task_raw), dict(layout_raw)
            task["trial_id"] = output_index
            layout["trial_id"] = output_index
            layout["episode_index"] = output_index
            if task.get("instruction") != layout.get("instruction"):
                raise ValueError(f"Task/layout mismatch at output episode {output_index}")
            final_tasks.append(task)
            final_layouts.append(layout)
            final_manifest.append({
                "schema_version": 1,
                "dataset_episode_index": output_index,
                "source_component": "sim6_domain_randomized" if component == 0 else "sim7_slowed",
                "source_repo_root": str(root),
                "source_dataset_episode_index": source_index,
                "source_manifest": source_manifest,
            })

    write_jsonl(BUILD_ROOT / OUTPUT_TASKS.name, final_tasks)
    write_jsonl(BUILD_ROOT / OUTPUT_LAYOUTS.name, final_layouts)
    write_jsonl(BUILD_ROOT / OUTPUT_MANIFEST.name, final_manifest)
    summary = {
        "schema_version": 1,
        "output_repo_id": "local/so101_bench_sim",
        "output_root": str(OUTPUT_ROOT),
        "total_episodes": merged.meta.total_episodes,
        "total_frames": merged.meta.total_frames,
        "component_counts": dict(zip((str(path) for path in ROOTS), expected_counts, strict=True)),
    }
    (BUILD_ROOT / OUTPUT_SUMMARY.name).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.apply:
        install()
    else:
        print(f"Validated staged build: {BUILD_ROOT}")


if __name__ == "__main__":
    main()
