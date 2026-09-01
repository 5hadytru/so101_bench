#!/usr/bin/env python
"""Repair LeRobot v3 dataset-level stats from per-episode metadata."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_lerobot_on_path() -> None:
    lerobot_src = Path(os.environ.get("LEROBOT_SRC", "/home/truman/lerobot/src"))
    if lerobot_src.exists():
        sys.path.insert(0, str(lerobot_src))


def _normalise_stat(value, feature_key: str, stat_key: str, feature_dtype: str):
    value = np.asarray(value)
    if feature_dtype in {"image", "video"} and stat_key != "count":
        if value.dtype == object:
            scalars = []
            for item in value:
                while isinstance(item, np.ndarray):
                    item = item.flatten()[0]
                scalars.append(item)
            value = np.asarray(scalars, dtype=np.float64)
        if value.shape == (3,):
            value = value.reshape(3, 1, 1)
    return value


def _episode_stats_from_row(row, features: dict) -> dict[str, dict[str, np.ndarray]]:
    stats: dict[str, dict[str, np.ndarray]] = {}
    for column, value in row.items():
        if not column.startswith("stats/"):
            continue
        feature_key, stat_key = column.removeprefix("stats/").rsplit("/", 1)
        if feature_key not in features:
            continue
        feature_dtype = features[feature_key]["dtype"]
        stats.setdefault(feature_key, {})[stat_key] = _normalise_stat(
            value, feature_key, stat_key, feature_dtype
        )
    return stats


def main() -> None:
    _ensure_lerobot_on_path()
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.utils import load_info, write_stats

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default=str(_repo_root() / "data/lerobot/so101_bench_sim_2"),
        help="Path to the local LeRobot dataset root.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    info = load_info(dataset_root)
    episodes_dir = dataset_root / "meta/episodes"
    parquet_paths = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No episode metadata parquet files found under {episodes_dir}")

    all_stats = []
    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path)
        for _, row in df.sort_values("episode_index").iterrows():
            all_stats.append(_episode_stats_from_row(row, info["features"]))

    if not all_stats:
        raise RuntimeError(f"No per-episode stats found under {episodes_dir}")

    stats = aggregate_stats(all_stats)
    stats = {key: value for key, value in stats.items() if key in info["features"]}
    write_stats(stats, dataset_root)
    print(f"Wrote {dataset_root / 'meta/stats.json'} from {len(all_stats)} episodes")


if __name__ == "__main__":
    main()
