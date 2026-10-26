#!/usr/bin/env python3
"""Rewrite so101_bench_sim_WM's joint values into the released MolmoAct2 SO-100/101 frame.

The dataset stores LeRobot range units (body joints in [-100, 100] over the real
robot's calibrated tick range, gripper in [0, 100]). allenai/MolmoAct2-SO100_101 was
trained on calibrated degrees with shoulder_lift negated and +90 deg offsets on
shoulder_lift and elbow_flex (see SO101MolmoAct2Mapper in
source/so101_bench/so101_bench/utils/molmoact2.py). Converting keeps the fine-tune in
the checkpoint's joint convention and lets the same client mapper serve both the
zero-shot and fine-tuned checkpoints.

Per joint: model = scale * lerobot + offset, with
    scale = sign * (range_max - range_min) / 200 * 360 / 4095   (gripper: 1)
Stats in meta/stats.json and meta/episodes are transformed exactly (min/max and the
quantile pairs swap where the scale is negative). Videos are symlinked, not copied.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# REAL_SO101_CALIBRATION (range_min, range_max) from so101_bench/utils/lerobot_calibration.py
RANGES = {
    "shoulder_pan": (735, 3443),
    "shoulder_lift": (874, 3230),
    "elbow_flex": (862, 3077),
    "wrist_flex": (887, 3178),
    "wrist_roll": (187, 4022),
}
SIGNS = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
OFFSETS = np.array([0.0, 90.0, 90.0, 0.0, 0.0, 0.0])
SCALE = SIGNS * np.array(
    [1.0 if j == "gripper" else (RANGES[j][1] - RANGES[j][0]) / 200.0 * 360.0 / 4095.0 for j in JOINTS]
)
KEYS = ("action", "observation.state")
PAIRED = {"min": "max", "q01": "q99", "q10": "q90"}


def transform(values: np.ndarray) -> np.ndarray:
    return values * SCALE + OFFSETS


def transform_stats(stats: dict) -> dict:
    """Exact affine transform of LeRobot per-dimension stats."""
    out = dict(stats)
    neg = SCALE < 0
    for lo, hi in PAIRED.items():
        if lo in stats and hi in stats:
            a, b = np.asarray(stats[lo], float), np.asarray(stats[hi], float)
            new_lo = np.where(neg, transform(b), transform(a))
            new_hi = np.where(neg, transform(a), transform(b))
            out[lo], out[hi] = new_lo.tolist(), new_hi.tolist()
    for key in ("mean", "q50"):
        if key in stats:
            out[key] = transform(np.asarray(stats[key], float)).tolist()
    if "std" in stats:
        out["std"] = (np.asarray(stats["std"], float) * np.abs(SCALE)).tolist()
    return out


def rewrite_data_file(src: Path, dst: Path) -> int:
    table = pq.read_table(src)
    for key in KEYS:
        idx = table.schema.get_field_index(key)
        column = table.column(key).combine_chunks()
        flat = column.values.to_numpy(zero_copy_only=False).reshape(len(column), -1)
        new = transform(flat.astype(np.float64)).astype(np.float32)
        list_type = column.type
        if pa.types.is_fixed_size_list(list_type):
            arr = pa.FixedSizeListArray.from_arrays(pa.array(new.reshape(-1), pa.float32()), list_type.list_size)
        else:
            arr = pa.ListArray.from_arrays(
                pa.array(np.arange(0, new.size + 1, new.shape[1], dtype=np.int32)),
                pa.array(new.reshape(-1), pa.float32()),
            )
        table = table.set_column(idx, table.schema.field(idx), arr.cast(list_type))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)
    return table.num_rows


def rewrite_episodes_file(src: Path, dst: Path) -> None:
    import pandas as pd

    df = pd.read_parquet(src)
    for key in KEYS:
        prefix = f"stats/{key}/"
        stat_names = [c[len(prefix):] for c in df.columns if c.startswith(prefix)]
        rows = []
        for _, row in df.iterrows():
            stats = {name: np.asarray(row[prefix + name], float) for name in stat_names}
            rows.append(transform_stats({k: v for k, v in stats.items() if k != "count"}))
        for name in stat_names:
            if name == "count":
                continue
            df[prefix + name] = [np.asarray(r[name], dtype=np.float64) for r in rows]
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    args = parser.parse_args()
    src, dst = args.src.resolve(), args.dst.resolve()
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    (dst / "videos").symlink_to(src / "videos", target_is_directory=True)

    rows = 0
    for path in sorted((src / "data").rglob("*.parquet")):
        rows += rewrite_data_file(path, dst / path.relative_to(src))

    (dst / "meta").mkdir()
    for path in (src / "meta").rglob("*"):
        rel = path.relative_to(src)
        if path.is_dir() or rel.parts[:2] == ("meta", "episodes") or rel.name == "stats.json":
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst / rel)
    for path in sorted((src / "meta" / "episodes").rglob("*.parquet")):
        rewrite_episodes_file(path, dst / path.relative_to(src))

    stats = json.loads((src / "meta" / "stats.json").read_text())
    for key in KEYS:
        stats[key] = transform_stats(stats[key])
    (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))
    (dst / "meta" / "molmoact2_frame.json").write_text(
        json.dumps(
            {
                "source": str(src),
                "transform": "model = scale * lerobot + offset",
                "joints": JOINTS,
                "scale": SCALE.tolist(),
                "offset": OFFSETS.tolist(),
            },
            indent=2,
        )
    )
    print(f"rewrote {rows} rows; scale={np.round(SCALE, 6).tolist()} offset={OFFSETS.tolist()}")
    for key in KEYS:
        print(key, "q01", np.round(stats[key]["q01"], 2).tolist(), "q99", np.round(stats[key]["q99"], 2).tolist())


if __name__ == "__main__":
    main()
