#!/usr/bin/env python3
"""Compute frame-balanced *contiguous* shard boundaries for the LeRobot dataset.

collect_outcomes only accepts a contiguous slice per process (--dataset_episode_index +
--num_episodes), and episodes vary widely in length (frame count), so a naive equal-COUNT
split leaves some shards doing far more work than others. This partitions a contiguous
episode range into `shards` contiguous chunks that minimize the maximum total frame count
per chunk (classic "split array into k contiguous parts minimizing the largest sum", solved
exactly with DP), so every shard finishes at roughly the same wall time.

Prints one line per shard:  <start> <count> <frames>
(extra shards beyond what the range can fill are omitted.)

Usage:
  shard_plan.py --repo_root <dir> --shards N [--start S] [--episodes E]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path


def _episode_lengths(repo_root: Path) -> list[int]:
    import pandas as pd  # provided by the isaaclab env

    files = sorted(glob.glob(str(repo_root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No episodes parquet under {repo_root/'meta'/'episodes'}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cols = [c for c in df.columns if c.lower() == "length"] or [
        c for c in df.columns if "length" in c.lower() or c.lower() in ("num_frames", "frames")
    ]
    if not cols:
        raise KeyError(f"No length-like column in episodes parquet; columns={list(df.columns)}")
    return [int(x) for x in df[cols[0]].to_numpy().tolist()]


def balanced_contiguous(lengths: list[int], k: int) -> list[tuple[int, int]]:
    """Return [(local_start, count), ...] splitting `lengths` into <=k contiguous chunks
    minimizing the maximum chunk sum. Uses DP over exactly j=min(k, n) parts."""
    n = len(lengths)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    if k == 1:
        return [(0, n)]

    prefix = [0] * (n + 1)
    for i, v in enumerate(lengths):
        prefix[i + 1] = prefix[i] + v

    def seg(a: int, b: int) -> int:  # sum of lengths[a:b]
        return prefix[b] - prefix[a]

    INF = float("inf")
    # dp[j][i] = min achievable max-sum splitting first i episodes into j parts
    dp = [[INF] * (n + 1) for _ in range(k + 1)]
    cut = [[0] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0
    for j in range(1, k + 1):
        for i in range(j, n + 1):
            for p in range(j - 1, i):
                cand = max(dp[j - 1][p], seg(p, i))
                if cand < dp[j][i]:
                    dp[j][i] = cand
                    cut[j][i] = p
    # backtrack boundaries
    bounds = []
    i = n
    for j in range(k, 0, -1):
        p = cut[j][i]
        bounds.append((p, i - p))
        i = p
    bounds.reverse()
    return bounds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=Path, required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--start", type=int, default=0, help="First dataset episode index of the range.")
    ap.add_argument("--episodes", type=int, default=None, help="Range size; default = to end of dataset.")
    ap.add_argument("--summary", action="store_true", help="Print imbalance summary to stderr.")
    args = ap.parse_args()

    lengths_all = _episode_lengths(args.repo_root)
    total = len(lengths_all)
    start = args.start
    end = total if args.episodes is None else min(start + args.episodes, total)
    if start < 0 or start >= end:
        print(f"Empty range start={start} end={end} (dataset has {total})", file=sys.stderr)
        return 1

    rng = lengths_all[start:end]
    parts = balanced_contiguous(rng, args.shards)

    sums = []
    for local_start, count in parts:
        frames = sum(rng[local_start:local_start + count])
        sums.append(frames)
        print(f"{start + local_start} {count} {frames}")

    if args.summary and sums:
        mx, mn = max(sums), min(sums)
        ideal = sum(sums) / len(sums)
        print(
            f"[shard_plan] {len(parts)} shard(s) over episodes [{start},{end}): "
            f"frames per shard min={mn} max={mx} ideal={ideal:.0f} "
            f"imbalance(max/ideal)={mx/ideal:.2f}x",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
