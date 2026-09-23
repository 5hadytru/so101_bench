#!/usr/bin/env python3
"""Create SO-101 Bench WM datasets with a static overhead_init stream.

The output dataset gets a new LeRobot video feature named
``observation.images.overhead_init``. For each episode, every frame in that
stream is the first frame from ``observation.images.overhead`` for the episode.

Both LeRobot v2.1 per-episode videos and v3.0 consolidated videos are supported.
Existing source data/videos are hardlinked into the destination when possible.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


OVERHEAD_KEY = "observation.images.overhead"
OVERHEAD_INIT_KEY = "observation.images.overhead_init"
QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Source LeRobot dataset root.")
    parser.add_argument("--dest", type=Path, required=True, help="Destination LeRobot dataset root.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel encoder workers.")
    parser.add_argument("--overwrite", action="store_true", help="Delete destination before recreating it.")
    parser.add_argument("--codec", type=str, default="libaom-av1", help="PyAV encoder for new MP4s.")
    parser.add_argument("--crf", type=int, default=40, help="Encoder CRF for new MP4s.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=4) + "\n")
    tmp_path.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    tmp_path.replace(path)


def hardlink_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_dataset_tree(source: Path, dest: Path, overwrite: bool) -> None:
    if dest.exists() and overwrite:
        shutil.rmtree(dest)
    if dest.exists():
        print(f"[INFO] Reusing existing destination: {dest}", flush=True)
        return
    print(f"[INFO] Hardlinking {source} -> {dest}", flush=True)
    shutil.copytree(
        source,
        dest,
        copy_function=hardlink_or_copy,
        ignore=shutil.ignore_patterns(".cache"),
    )


def validate_source(source: Path) -> dict[str, Any]:
    info_path = source / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing metadata: {info_path}")
    info = read_json(info_path)
    version = info.get("codebase_version")
    if version not in {"v2.1", "v3.0"}:
        raise ValueError(f"Unsupported codebase_version {version!r}; expected v2.1 or v3.0.")
    features = info.get("features", {})
    if features.get(OVERHEAD_KEY, {}).get("dtype") != "video":
        raise ValueError(f"Source dataset does not have video feature {OVERHEAD_KEY!r}.")
    if OVERHEAD_INIT_KEY in features:
        raise ValueError(f"Source dataset already has {OVERHEAD_INIT_KEY!r}.")
    return info


def add_overhead_init_feature(info: dict[str, Any]) -> dict[str, Any]:
    new_info = copy.deepcopy(info)
    features = new_info["features"]
    ordered_features: dict[str, Any] = {}
    for key, value in features.items():
        ordered_features[key] = value
        if key == OVERHEAD_KEY:
            ordered_features[OVERHEAD_INIT_KEY] = copy.deepcopy(value)
    new_info["features"] = ordered_features

    if new_info.get("codebase_version") == "v2.1":
        video_count = sum(1 for feature in ordered_features.values() if feature.get("dtype") == "video")
        new_info["total_videos"] = int(new_info["total_episodes"]) * video_count
    return new_info


def v2_video_path(root: Path, info: dict[str, Any], video_key: str, episode_index: int) -> Path:
    chunk_index = episode_index // int(info.get("chunks_size", 1000))
    return root / info["video_path"].format(
        chunk_index=chunk_index,
        video_key=video_key,
        episode_index=episode_index,
    )


def v3_video_path(root: Path, info: dict[str, Any], video_key: str, chunk_index: int, file_index: int) -> Path:
    return root / info["video_path"].format(
        video_key=video_key,
        chunk_index=chunk_index,
        file_index=file_index,
    )


def probe_frame_count(path: Path) -> int | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return int(value) if value.isdigit() else None


def first_frame_rgb(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            return frame.to_ndarray(format="rgb24")
    raise ValueError(f"No video frames found in {path}")


def downsample_for_lerobot_stats(frame_rgb: np.ndarray) -> np.ndarray:
    img_chw = frame_rgb.transpose(2, 0, 1)
    _, height, width = img_chw.shape
    if max(width, height) >= 300:
        factor = int(width / 150) if width > height else int(height / 150)
        img_chw = img_chw[:, ::factor, ::factor]
    return img_chw.transpose(1, 2, 0).reshape(-1, 3).astype(np.float64) / 255.0


def stat_shape(values: np.ndarray) -> list[list[list[float]]]:
    return values.reshape(3, 1, 1).tolist()


def frame_stats(frame_rgb: np.ndarray, episode_length: int) -> dict[str, Any]:
    flat = downsample_for_lerobot_stats(frame_rgb)
    stats: dict[str, Any] = {
        "min": stat_shape(flat.min(axis=0)),
        "max": stat_shape(flat.max(axis=0)),
        "mean": stat_shape(flat.mean(axis=0)),
        "std": stat_shape(flat.std(axis=0)),
        "count": [int(flat.shape[0] * episode_length)],
    }
    for name, quantile in QUANTILES.items():
        stats[name] = stat_shape(np.quantile(flat, quantile, axis=0))
    return stats


def stats_array(stats: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(stats[key], dtype=np.float64)


def aggregate_video_stats(stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    counts = np.asarray([stats["count"] for stats in stats_list], dtype=np.float64)
    means = np.stack([stats_array(stats, "mean") for stats in stats_list])
    variances = np.stack([stats_array(stats, "std") ** 2 for stats in stats_list])
    total_count = counts.sum(axis=0)

    expanded_counts = counts
    while expanded_counts.ndim < means.ndim:
        expanded_counts = np.expand_dims(expanded_counts, axis=-1)

    total_mean = (means * expanded_counts).sum(axis=0) / total_count
    delta_means = means - total_mean
    total_variance = ((variances + delta_means**2) * expanded_counts).sum(axis=0) / total_count

    aggregated: dict[str, Any] = {
        "min": np.min(np.stack([stats_array(stats, "min") for stats in stats_list]), axis=0).tolist(),
        "max": np.max(np.stack([stats_array(stats, "max") for stats in stats_list]), axis=0).tolist(),
        "mean": total_mean.tolist(),
        "std": np.sqrt(total_variance).tolist(),
        "count": total_count.astype(np.int64).tolist(),
    }
    for key in QUANTILES:
        quantiles = np.stack([stats_array(stats, key) for stats in stats_list])
        aggregated[key] = ((quantiles * expanded_counts).sum(axis=0) / total_count).tolist()
    return aggregated


def encode_static_video(
    frame_segments: list[tuple[np.ndarray, int]],
    dest_video: Path,
    fps: int,
    codec: str,
    crf: int,
) -> None:
    if not frame_segments:
        raise ValueError(f"No frame segments to encode for {dest_video}.")
    dest_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_video = dest_video.with_suffix(".tmp.mp4")
    if tmp_video.exists():
        tmp_video.unlink()

    first_frame = frame_segments[0][0]
    height, width = first_frame.shape[:2]
    container = av.open(str(tmp_video), "w")
    stream = container.add_stream(
        codec,
        rate=fps,
        options={"cpu-used": "8", "crf": str(crf), "threads": "1"},
    )
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, fps)

    frame_index = 0
    try:
        for frame_rgb, repeats in frame_segments:
            if frame_rgb.shape[:2] != (height, width):
                raise ValueError(f"Frame shape changed while encoding {dest_video}.")
            for _ in range(repeats):
                frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                frame.pts = frame_index
                frame.time_base = Fraction(1, fps)
                for packet in stream.encode(frame):
                    container.mux(packet)
                frame_index += 1
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    tmp_video.replace(dest_video)


def process_v2_episode(
    source: Path,
    dest: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
    codec: str,
    crf: int,
) -> tuple[int, dict[str, Any], bool]:
    episode_index = int(episode["episode_index"])
    length = int(episode["length"])
    fps = int(info["fps"])
    src_video = v2_video_path(source, info, OVERHEAD_KEY, episode_index)
    dst_video = v2_video_path(dest, info, OVERHEAD_INIT_KEY, episode_index)
    if not src_video.exists():
        raise FileNotFoundError(f"Missing source overhead video: {src_video}")

    frame = first_frame_rgb(src_video)
    stats = frame_stats(frame, length)

    if probe_frame_count(dst_video) == length:
        return episode_index, stats, False
    encode_static_video([(frame, length)], dst_video, fps, codec, crf)
    actual_frames = probe_frame_count(dst_video)
    if actual_frames != length:
        raise RuntimeError(f"{dst_video} has {actual_frames} frame(s), expected {length}.")
    return episode_index, stats, True


def add_global_stats(dest: Path, wm_stats: dict[int, dict[str, Any]]) -> None:
    stats = read_json(dest / "meta" / "stats.json")
    ordered_stats: dict[str, Any] = {}
    global_overhead_init_stats = aggregate_video_stats([wm_stats[index] for index in sorted(wm_stats)])
    for key, value in stats.items():
        ordered_stats[key] = value
        if key == OVERHEAD_KEY:
            ordered_stats[OVERHEAD_INIT_KEY] = global_overhead_init_stats
    if OVERHEAD_INIT_KEY not in ordered_stats:
        ordered_stats[OVERHEAD_INIT_KEY] = global_overhead_init_stats
    write_json(dest / "meta" / "stats.json", ordered_stats)


def create_v2_wm(source: Path, dest: Path, info: dict[str, Any], workers: int, codec: str, crf: int) -> None:
    episodes = read_jsonl(source / "meta" / "episodes.jsonl")
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError("episodes.jsonl row count does not match info.json total_episodes.")

    print(f"[INFO] v2.1: encoding {len(episodes)} per-episode overhead_init videos", flush=True)
    wm_stats: dict[int, dict[str, Any]] = {}
    encoded = 0
    reused = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(process_v2_episode, source, dest, info, episode, codec, crf)
            for episode in episodes
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            episode_index, stats, did_encode = future.result()
            wm_stats[episode_index] = stats
            encoded += int(did_encode)
            reused += int(not did_encode)
            if done == 1 or done % 25 == 0 or done == len(futures):
                print(f"[INFO] v2.1: {done}/{len(futures)} episodes (encoded={encoded}, reused={reused})", flush=True)

    write_json(dest / "meta" / "info.json", add_overhead_init_feature(info))
    add_global_stats(dest, wm_stats)

    rows = read_jsonl(dest / "meta" / "episodes_stats.jsonl")
    for row in rows:
        row["stats"][OVERHEAD_INIT_KEY] = wm_stats[int(row["episode_index"])]
    write_jsonl(dest / "meta" / "episodes_stats.jsonl", rows)


def load_v3_episode_files(source: Path) -> list[tuple[Path, list[dict[str, Any]]]]:
    episode_files: list[tuple[Path, list[dict[str, Any]]]] = []
    paths = sorted((source / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No v3.0 episode metadata parquet files found in {source / 'meta' / 'episodes'}.")
    for path in paths:
        table = pq.read_table(path)
        episode_files.append((path, table.to_pylist()))
    return episode_files


def capture_v3_first_frames(src_video: Path, records: list[dict[str, Any]], fps: int) -> dict[int, np.ndarray]:
    target_frames = {
        round(float(record[f"videos/{OVERHEAD_KEY}/from_timestamp"]) * fps): int(record["episode_index"])
        for record in records
    }
    captured: dict[int, np.ndarray] = {}
    with av.open(str(src_video)) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            episode_index = target_frames.get(frame_index)
            if episode_index is not None:
                captured[episode_index] = frame.to_ndarray(format="rgb24")
                if len(captured) == len(target_frames):
                    break
    if len(captured) != len(target_frames):
        missing = sorted(set(target_frames.values()) - set(captured))
        raise RuntimeError(f"Could not capture first frames for episodes {missing[:10]} from {src_video}.")
    return captured


def process_v3_video_file(
    source: Path,
    dest: Path,
    info: dict[str, Any],
    chunk_index: int,
    file_index: int,
    records: list[dict[str, Any]],
    codec: str,
    crf: int,
) -> tuple[tuple[int, int], dict[int, dict[str, Any]], bool]:
    fps = int(info["fps"])
    src_video = v3_video_path(source, info, OVERHEAD_KEY, chunk_index, file_index)
    dst_video = v3_video_path(dest, info, OVERHEAD_INIT_KEY, chunk_index, file_index)
    if not src_video.exists():
        raise FileNotFoundError(f"Missing source overhead video: {src_video}")

    sorted_records = sorted(records, key=lambda rec: float(rec[f"videos/{OVERHEAD_KEY}/from_timestamp"]))
    frames = capture_v3_first_frames(src_video, sorted_records, fps)
    wm_stats = {
        int(record["episode_index"]): frame_stats(frames[int(record["episode_index"])], int(record["length"]))
        for record in sorted_records
    }

    expected_frames = sum(int(record["length"]) for record in sorted_records)
    if probe_frame_count(dst_video) == expected_frames:
        return (chunk_index, file_index), wm_stats, False

    segments = [(frames[int(record["episode_index"])], int(record["length"])) for record in sorted_records]
    encode_static_video(segments, dst_video, fps, codec, crf)
    actual_frames = probe_frame_count(dst_video)
    if actual_frames != expected_frames:
        raise RuntimeError(f"{dst_video} has {actual_frames} frame(s), expected {expected_frames}.")
    return (chunk_index, file_index), wm_stats, True


def update_v3_episode_metadata(
    dest: Path,
    episode_files: list[tuple[Path, list[dict[str, Any]]]],
    wm_stats: dict[int, dict[str, Any]],
    fps: int,
) -> None:
    # The new overhead_init stream contains only frames from retained episodes.
    # It may therefore be shorter than the source overhead shard when the source
    # dataset has pruned episodes whose video spans remain in that shard. Build
    # fresh contiguous spans for overhead_init instead of copying overhead spans.
    all_records = [record for _, records in episode_files for record in records]
    by_source_video: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in all_records:
        key = (
            int(record[f"videos/{OVERHEAD_KEY}/chunk_index"]),
            int(record[f"videos/{OVERHEAD_KEY}/file_index"]),
        )
        by_source_video[key].append(record)

    init_spans: dict[int, tuple[int, int, float, float]] = {}
    for (chunk_index, file_index), records in by_source_video.items():
        offset_frames = 0
        for record in sorted(records, key=lambda rec: float(rec[f"videos/{OVERHEAD_KEY}/from_timestamp"])):
            episode_index = int(record["episode_index"])
            length = int(record["length"])
            init_spans[episode_index] = (
                chunk_index,
                file_index,
                offset_frames / fps,
                (offset_frames + length) / fps,
            )
            offset_frames += length

    for source_meta_path, records in episode_files:
        dest_meta_path = dest / source_meta_path.relative_to(source_meta_path.parents[3])
        updated_records = []
        for record in records:
            episode_index = int(record["episode_index"])
            record = dict(record)
            chunk_index, file_index, from_timestamp, to_timestamp = init_spans[episode_index]
            record[f"videos/{OVERHEAD_INIT_KEY}/chunk_index"] = chunk_index
            record[f"videos/{OVERHEAD_INIT_KEY}/file_index"] = file_index
            record[f"videos/{OVERHEAD_INIT_KEY}/from_timestamp"] = from_timestamp
            record[f"videos/{OVERHEAD_INIT_KEY}/to_timestamp"] = to_timestamp
            for stat_key, value in wm_stats[episode_index].items():
                record[f"stats/{OVERHEAD_INIT_KEY}/{stat_key}"] = value
            updated_records.append(record)
        table = pa.Table.from_pylist(updated_records)
        dest_meta_path.parent.mkdir(parents=True, exist_ok=True)
        # copy_dataset_tree hardlinks source files. Write-and-replace so this
        # metadata update never mutates the source parquet through that link.
        tmp_meta_path = dest_meta_path.with_suffix(dest_meta_path.suffix + ".tmp")
        pq.write_table(table, tmp_meta_path)
        tmp_meta_path.replace(dest_meta_path)


def create_v3_wm(source: Path, dest: Path, info: dict[str, Any], workers: int, codec: str, crf: int) -> None:
    episode_files = load_v3_episode_files(source)
    all_records = [record for _, records in episode_files for record in records]
    if len(all_records) != int(info["total_episodes"]):
        raise ValueError("v3.0 episode metadata row count does not match info.json total_episodes.")

    by_video_file: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in all_records:
        key = (
            int(record[f"videos/{OVERHEAD_KEY}/chunk_index"]),
            int(record[f"videos/{OVERHEAD_KEY}/file_index"]),
        )
        by_video_file[key].append(record)

    print(f"[INFO] v3.0: encoding {len(by_video_file)} consolidated overhead_init videos", flush=True)
    wm_stats: dict[int, dict[str, Any]] = {}
    encoded = 0
    reused = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                process_v3_video_file,
                source,
                dest,
                info,
                chunk_index,
                file_index,
                records,
                codec,
                crf,
            )
            for (chunk_index, file_index), records in sorted(by_video_file.items())
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            _, stats_for_file, did_encode = future.result()
            wm_stats.update(stats_for_file)
            encoded += int(did_encode)
            reused += int(not did_encode)
            print(f"[INFO] v3.0: {done}/{len(futures)} video files (encoded={encoded}, reused={reused})", flush=True)

    write_json(dest / "meta" / "info.json", add_overhead_init_feature(info))
    add_global_stats(dest, wm_stats)
    update_v3_episode_metadata(dest, episode_files, wm_stats, int(info["fps"]))


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    dest = args.dest.resolve()
    info = validate_source(source)

    copy_dataset_tree(source, dest, args.overwrite)
    version = info["codebase_version"]
    if version == "v2.1":
        create_v2_wm(source, dest, info, args.workers, args.codec, args.crf)
    elif version == "v3.0":
        create_v3_wm(source, dest, info, args.workers, args.codec, args.crf)
    else:
        raise AssertionError(f"Unexpected version {version!r}")

    print(f"[INFO] Done: {dest}", flush=True)


if __name__ == "__main__":
    main()
