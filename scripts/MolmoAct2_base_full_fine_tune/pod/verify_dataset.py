#!/usr/bin/env python3
"""Verify a downloaded LeRobot v3.0 dataset before (and during) training.

Checks, each selectable:
  --sha      every file's size and SHA-256 against the Hugging Face manifest
             (LFS sha256, or the git blob sha1 for small files). Writes a local
             manifest so later runs can re-check without the network.
  --decode   fully decodes every video file and compares its frame count with the
             frames the episode metadata assigns to it.
  --parquet  reads every data parquet file: row counts, episode/index continuity,
             finite action/state values.
  --sample N re-hashes N random files against the local manifest (cheap periodic check).

Exit status is non-zero if anything fails; failures are listed at the end.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

CHUNK = 8 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def remote_manifest(repo_id: str, revision: str) -> dict[str, dict]:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id, revision=revision, files_metadata=True)
    manifest = {}
    for s in info.siblings:
        entry = {"size": s.size}
        if s.lfs is not None:
            entry["sha256"] = s.lfs.sha256
        else:
            entry["blob_sha1"] = s.blob_id
        manifest[s.rfilename] = entry
    return manifest


def check_one(root: Path, rel: str, expected: dict) -> tuple[str, str | None, dict]:
    path = root / rel
    if not path.is_file():
        return rel, "missing", {}
    size = path.stat().st_size
    if expected.get("size") is not None and size != expected["size"]:
        return rel, f"size {size} != {expected['size']}", {}
    if "sha256" in expected:
        digest = sha256_file(path)
        if digest != expected["sha256"]:
            return rel, "sha256 mismatch", {}
        return rel, None, {"size": size, "sha256": digest}
    digest = git_blob_sha1(path)
    if digest != expected.get("blob_sha1"):
        return rel, "blob sha1 mismatch", {}
    return rel, None, {"size": size, "sha256": sha256_file(path)}


def run_sha(root: Path, repo_id: str, revision: str, manifest_out: Path, workers: int) -> list[str]:
    expected = remote_manifest(repo_id, revision)
    failures, local = [], {}
    t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(check_one, root, rel, exp) for rel, exp in expected.items()]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            rel, err, entry = fut.result()
            if err:
                failures.append(f"{rel}: {err}")
            else:
                local[rel] = entry
            if i % 100 == 0:
                print(f"[sha] {i}/{len(futures)} files, {time.time() - t0:.0f}s", flush=True)
    manifest_out.write_text(json.dumps({"repo_id": repo_id, "revision": revision, "files": local}, indent=1))
    print(f"[sha] {len(local)}/{len(expected)} ok in {time.time() - t0:.0f}s; manifest -> {manifest_out}")
    return failures


def run_sample(root: Path, manifest_path: Path, n: int) -> list[str]:
    files = json.loads(manifest_path.read_text())["files"]
    picks = random.sample(sorted(files), min(n, len(files)))
    failures = []
    for rel in picks:
        path = root / rel
        if not path.is_file():
            failures.append(f"{rel}: missing")
        elif sha256_file(path) != files[rel]["sha256"]:
            failures.append(f"{rel}: sha256 mismatch")
    print(f"[sample] {len(picks) - len(failures)}/{len(picks)} random files match the manifest")
    return failures


def expected_video_frames(root: Path) -> dict[str, int]:
    import pandas as pd

    info = json.loads((root / "meta/info.json").read_text())
    fps = info["fps"]
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    episodes = pd.concat(
        [pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet"))],
        ignore_index=True,
    )
    frames: dict[str, int] = {}
    for key in video_keys:
        for (chunk, file), group in episodes.groupby(
            [f"videos/{key}/chunk_index", f"videos/{key}/file_index"]
        ):
            rel = info["video_path"].format(video_key=key, chunk_index=int(chunk), file_index=int(file))
            span = (group[f"videos/{key}/to_timestamp"] - group[f"videos/{key}/from_timestamp"]) * fps
            if not ((span.round() - group["length"]).abs() <= 1).all():
                frames[rel] = -1  # timestamp span disagrees with episode length
            else:
                frames[rel] = int(group["length"].sum())
    return frames


def decode_count(path: str) -> tuple[str, int, str | None]:
    import av

    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            n = sum(1 for _ in container.decode(stream))
        return path, n, None
    except Exception as exc:  # noqa: BLE001 - report every decode failure
        return path, -1, f"{type(exc).__name__}: {exc}"


def run_decode(root: Path, workers: int) -> list[str]:
    expected = expected_video_frames(root)
    failures = []
    t0 = time.time()
    with cf.ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(decode_count, str(root / rel)): rel for rel in expected}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            rel = futures[fut]
            _, n, err = fut.result()
            if err:
                failures.append(f"{rel}: decode error {err}")
            elif expected[rel] < 0:
                failures.append(f"{rel}: episode timestamps disagree with episode lengths")
            elif n != expected[rel]:
                failures.append(f"{rel}: decoded {n} frames, metadata expects {expected[rel]}")
            if i % 50 == 0:
                print(f"[decode] {i}/{len(futures)} files, {time.time() - t0:.0f}s", flush=True)
    print(f"[decode] {len(expected) - len(failures)}/{len(expected)} video files ok in {time.time() - t0:.0f}s")
    return failures


def run_parquet(root: Path) -> list[str]:
    import numpy as np
    import pandas as pd

    info = json.loads((root / "meta/info.json").read_text())
    failures = []
    frames = []
    for path in sorted((root / "data").rglob("*.parquet")):
        df = pd.read_parquet(path)
        for key in ("action", "observation.state"):
            values = np.stack(df[key].to_numpy())
            if not np.isfinite(values).all():
                failures.append(f"{path.relative_to(root)}: non-finite {key}")
        frames.append(df[["episode_index", "frame_index", "index"]])
    frames = pd.concat(frames, ignore_index=True).sort_values("index")
    if len(frames) != info["total_frames"]:
        failures.append(f"data rows {len(frames)} != total_frames {info['total_frames']}")
    if not (frames["index"].to_numpy() == np.arange(len(frames))).all():
        failures.append("global index is not contiguous")
    episodes = frames.groupby("episode_index")["frame_index"].agg(["min", "max", "count"])
    if len(episodes) != info["total_episodes"]:
        failures.append(f"episodes {len(episodes)} != total_episodes {info['total_episodes']}")
    bad = episodes[(episodes["min"] != 0) | (episodes["max"] != episodes["count"] - 1)]
    if len(bad):
        failures.append(f"{len(bad)} episodes have non-contiguous frame_index")
    print(f"[parquet] {len(frames)} rows, {len(episodes)} episodes checked")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo_id", default="5hadytru/so101_bench_sim_WM")
    parser.add_argument("--revision", default="b569ad81ee60ffe3580bc0f1df513bde1bb36e8c")
    parser.add_argument("--manifest", type=Path, default=None, help="Local sha256 manifest (written by --sha).")
    parser.add_argument("--sha", action="store_true")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--parquet", action="store_true")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    args = parser.parse_args()
    manifest = args.manifest or args.root.parent / f"{args.root.name}.manifest.json"

    failures: list[str] = []
    if args.sha:
        failures += run_sha(args.root, args.repo_id, args.revision, manifest, min(args.workers, 32))
    if args.sample:
        failures += run_sample(args.root, manifest, args.sample)
    if args.parquet:
        failures += run_parquet(args.root)
    if args.decode:
        failures += run_decode(args.root, args.workers)

    if failures:
        print(f"FAILED ({len(failures)}):")
        for line in failures:
            print("  " + line)
        return 1
    print("DATASET_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
