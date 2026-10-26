#!/usr/bin/env python3
"""Validate that every LeRobot video decodes to the frames declared by metadata.

This catches a failure mode that a normal MP4 probe misses: an MP4 header can
advertise the expected length while the encoded bitstream stops early.  That
causes loaders such as torchcodec to fail only when they request one of the
metadata-declared but undecodable frames.

Both LeRobot v2.1 (one video per episode) and v3.0 (chunked video files with
per-episode timestamp spans) are supported.  The validator fully decodes each
stream with ffmpeg and compares that count against the LeRobot episode length.

Examples:
  python scripts/validate_dataset_frames.py data/lerobot/so101_bench_sim_4_WM \
      --report outputs/sim_4_wm_video_validation.json

  python scripts/validate_dataset_frames.py data/lerobot/so101_bench_sim_4_WM_v3.0 \
      --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable


@dataclass(frozen=True)
class VideoJob:
    path: Path
    video_key: str
    expected_frames: int
    episode_indices: tuple[int, ...]


@dataclass
class VideoResult:
    path: str
    video_key: str
    expected_frames: int
    episode_indices: list[int]
    decoded_frames: int | None = None
    header_frames: int | None = None
    codec: str | None = None
    width: int | None = None
    height: int | None = None
    status: str = "unknown"
    errors: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fully decode LeRobot videos and compare decoded frames against episode metadata."
    )
    parser.add_argument("dataset_root", type=Path, help="LeRobot dataset root (containing meta/ and videos/).")
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Validate one video feature key; repeat for multiple keys. Default: every video feature.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Concurrent ffmpeg decoders (default: 4).")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Validate only the first N video files after sorting; intended for smoke tests.",
    )
    parser.add_argument("--report", type=Path, default=None, help="Write a detailed JSON report to this path.")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Retained for CLI clarity; validation always reports every bad file instead of stopping at the first one.",
    )
    return parser.parse_args()


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name!r} is required but was not found in PATH.")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required LeRobot metadata file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(value).__name__}.")
    return value


def video_keys(info: dict[str, Any], requested: list[str] | None) -> list[str]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("meta/info.json is missing its features object.")
    available = sorted(key for key, feature in features.items() if isinstance(feature, dict) and feature.get("dtype") == "video")
    if not available:
        raise ValueError("The dataset metadata declares no video features.")
    if requested is None:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Requested --camera key(s) are not video features: {', '.join(unknown)}. Available: {available}")
    return list(dict.fromkeys(requested))


def load_v21_jobs(root: Path, info: dict[str, Any], keys: list[str]) -> tuple[list[VideoJob], list[str]]:
    episodes_path = root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"LeRobot v2.1 dataset is missing {episodes_path}.")
    chunks_size = int(info.get("chunks_size", 1000))
    template = info.get("video_path")
    if not isinstance(template, str):
        template = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    jobs: list[VideoJob] = []
    metadata_errors: list[str] = []
    total_frames = 0
    episode_count = 0
    with episodes_path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            if not raw_line.strip():
                continue
            try:
                episode = json.loads(raw_line)
                episode_index = int(episode["episode_index"])
                length = int(episode["length"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                metadata_errors.append(f"{episodes_path}:{line_number}: invalid episode metadata ({exc})")
                continue
            if length <= 0:
                metadata_errors.append(f"{episodes_path}:{line_number}: episode {episode_index} has non-positive length {length}")
                continue
            episode_count += 1
            total_frames += length
            for key in keys:
                try:
                    relative = template.format(
                        video_key=key,
                        episode_index=episode_index,
                        episode_chunk=episode_index // chunks_size,
                        chunk_index=episode_index // chunks_size,
                    )
                except KeyError as exc:
                    metadata_errors.append(f"Could not expand meta/info.json video_path {template!r}: missing {exc}")
                    continue
                jobs.append(VideoJob(root / relative, key, length, (episode_index,)))

    declared_episodes = int(info.get("total_episodes", episode_count))
    declared_frames = int(info.get("total_frames", total_frames))
    if declared_episodes != episode_count:
        metadata_errors.append(f"meta/info.json total_episodes={declared_episodes}, but episodes.jsonl has {episode_count} rows")
    if declared_frames != total_frames:
        metadata_errors.append(f"meta/info.json total_frames={declared_frames}, but episode lengths sum to {total_frames}")
    return jobs, metadata_errors


def load_v30_jobs(root: Path, info: dict[str, Any], keys: list[str]) -> tuple[list[VideoJob], list[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("LeRobot v3.0 metadata requires pyarrow. Install it in the validation environment.") from exc

    episode_files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"LeRobot v3.0 dataset is missing episode parquet metadata beneath {root / 'meta/episodes'}.")
    template = info.get("video_path")
    if not isinstance(template, str):
        template = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

    grouped: dict[tuple[str, Path], tuple[int, list[int]]] = {}
    metadata_errors: list[str] = []
    total_frames = 0
    seen_episodes: set[int] = set()
    for episode_file in episode_files:
        table = pq.read_table(episode_file)
        rows = table.to_pylist()
        for row_number, row in enumerate(rows, start=1):
            try:
                episode_index = int(row["episode_index"])
                length = int(row["length"])
            except (KeyError, TypeError, ValueError) as exc:
                metadata_errors.append(f"{episode_file}:{row_number}: invalid episode metadata ({exc})")
                continue
            if episode_index in seen_episodes:
                metadata_errors.append(f"Duplicate episode_index={episode_index} in v3 episode metadata")
            seen_episodes.add(episode_index)
            if length <= 0:
                metadata_errors.append(f"{episode_file}:{row_number}: episode {episode_index} has non-positive length {length}")
                continue
            total_frames += length
            for key in keys:
                prefix = f"videos/{key}/"
                try:
                    chunk_index = int(row[f"{prefix}chunk_index"])
                    file_index = int(row[f"{prefix}file_index"])
                except (KeyError, TypeError, ValueError) as exc:
                    metadata_errors.append(
                        f"{episode_file}:{row_number}: missing/invalid video span for {key!r} ({exc})"
                    )
                    continue
                try:
                    relative = template.format(video_key=key, chunk_index=chunk_index, file_index=file_index)
                except KeyError as exc:
                    metadata_errors.append(f"Could not expand meta/info.json video_path {template!r}: missing {exc}")
                    continue
                path = root / relative
                group_key = (key, path)
                expected, episodes = grouped.get(group_key, (0, []))
                grouped[group_key] = (expected + length, [*episodes, episode_index])

    declared_episodes = int(info.get("total_episodes", len(seen_episodes)))
    declared_frames = int(info.get("total_frames", total_frames))
    if declared_episodes != len(seen_episodes):
        metadata_errors.append(f"meta/info.json total_episodes={declared_episodes}, but episode parquet metadata has {len(seen_episodes)} rows")
    if declared_frames != total_frames:
        metadata_errors.append(f"meta/info.json total_frames={declared_frames}, but episode lengths sum to {total_frames}")
    jobs = [VideoJob(path, key, expected, tuple(episodes)) for (key, path), (expected, episodes) in grouped.items()]
    return jobs, metadata_errors


def probe_header(path: Path) -> tuple[int | None, str | None, int | None, int | None, list[str]]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,nb_frames", "-of", "json", str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    errors = [line for line in completed.stderr.splitlines() if line.strip()]
    if completed.returncode != 0:
        return None, None, None, None, [f"ffprobe exited {completed.returncode}", *errors]
    try:
        streams = json.loads(completed.stdout).get("streams", [])
        stream = streams[0]
    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
        return None, None, None, None, [f"ffprobe returned no readable video stream ({exc})", *errors]
    value = stream.get("nb_frames")
    try:
        header_frames = int(value) if value not in (None, "N/A") else None
    except (TypeError, ValueError):
        header_frames = None
    return header_frames, stream.get("codec_name"), stream.get("width"), stream.get("height"), errors


def decode_job(job: VideoJob) -> VideoResult:
    result = VideoResult(
        path=str(job.path),
        video_key=job.video_key,
        expected_frames=job.expected_frames,
        episode_indices=list(job.episode_indices),
    )
    if not job.path.is_file():
        result.status = "missing"
        result.errors.append("Video file is missing.")
        return result

    header_frames, codec, width, height, probe_errors = probe_header(job.path)
    result.header_frames = header_frames
    result.codec = codec
    result.width = width
    result.height = height
    result.errors.extend(probe_errors)

    # -f null forces every frame through the decoder. The final frame=N progress
    # line is the actual decodable count, unlike MP4 nb_frames header metadata.
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(job.path),
        "-map", "0:v:0", "-an", "-sn", "-dn", "-f", "null", "-", "-progress", "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    decoded_frames: int | None = None
    for line in completed.stdout.splitlines():
        if line.startswith("frame="):
            try:
                decoded_frames = int(line.split("=", 1)[1])
            except ValueError:
                pass
    result.decoded_frames = decoded_frames
    decode_errors = [line for line in completed.stderr.splitlines() if line.strip()]
    if completed.returncode != 0:
        result.status = "decode_error"
        result.errors.extend([f"ffmpeg exited {completed.returncode}", *decode_errors])
    elif decoded_frames is None:
        result.status = "decode_error"
        result.errors.extend(["ffmpeg returned no decoded frame count.", *decode_errors])
    elif decoded_frames != job.expected_frames:
        result.status = "frame_count_mismatch"
        result.errors.extend(decode_errors)
    elif header_frames is not None and header_frames != job.expected_frames:
        # The stream is decodable, but this still indicates metadata/container
        # disagreement that should not silently pass a dataset validation.
        result.status = "header_count_mismatch"
        result.errors.extend(decode_errors)
    else:
        result.status = "ok"
        result.errors.extend(decode_errors)
    return result


def serialize_results(results: Iterable[VideoResult]) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    require_executable("ffmpeg")
    require_executable("ffprobe")

    root = args.dataset_root.resolve()
    info = read_json(root / "meta" / "info.json")
    version = str(info.get("codebase_version", ""))
    keys = video_keys(info, args.camera)
    if version.startswith("v2"):
        jobs, metadata_errors = load_v21_jobs(root, info, keys)
    elif version.startswith("v3"):
        jobs, metadata_errors = load_v30_jobs(root, info, keys)
    else:
        raise ValueError(f"Unsupported or missing LeRobot codebase_version {version!r}; expected v2.x or v3.x.")

    jobs.sort(key=lambda job: (job.video_key, str(job.path)))
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        raise ValueError("No videos selected for validation.")

    print(f"[dataset] {root}")
    print(f"[format] {version}; cameras={', '.join(keys)}; videos={len(jobs)}; workers={args.workers}")
    if args.limit is not None:
        print(f"[limit] validating first {len(jobs)} video file(s) only")

    results: list[VideoResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(decode_job, job): job for job in jobs}
        for completed_count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # Defensive: one video must not hide later failures.
                result = VideoResult(
                    path=str(job.path), video_key=job.video_key, expected_frames=job.expected_frames,
                    episode_indices=list(job.episode_indices), status="validator_error", errors=[repr(exc)],
                )
            results.append(result)
            if result.status != "ok":
                print(
                    f"[BAD] {result.status}: {result.path} "
                    f"expected={result.expected_frames} decoded={result.decoded_frames} "
                    f"header={result.header_frames} episodes={result.episode_indices}",
                    file=sys.stderr,
                )
            elif completed_count % 50 == 0 or completed_count == len(jobs):
                print(f"[progress] decoded {completed_count}/{len(jobs)} video files")

    results.sort(key=lambda result: (result.video_key, result.path))
    counts = Counter(result.status for result in results)
    bad_results = [result for result in results if result.status != "ok"]
    report = {
        "dataset_root": str(root),
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "codebase_version": version,
        "video_keys": keys,
        "metadata_errors": metadata_errors,
        "summary": {
            "video_files_checked": len(results),
            "ok": counts["ok"],
            "bad": len(bad_results),
            "by_status": dict(sorted(counts.items())),
        },
        "bad_videos": serialize_results(bad_results),
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[report] wrote {args.report}")

    print(
        f"[summary] {counts['ok']}/{len(results)} video files fully decoded with the metadata-declared frame count; "
        f"bad={len(bad_results)}, metadata_errors={len(metadata_errors)}"
    )
    if metadata_errors:
        for error in metadata_errors:
            print(f"[METADATA] {error}", file=sys.stderr)
    if bad_results:
        print("[result] FAILED: re-export or remove the listed complete episodes; re-encoding a truncated stream cannot restore frames.", file=sys.stderr)
        return 1
    if metadata_errors:
        print("[result] FAILED: metadata is inconsistent.", file=sys.stderr)
        return 1
    print("[result] PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2)
