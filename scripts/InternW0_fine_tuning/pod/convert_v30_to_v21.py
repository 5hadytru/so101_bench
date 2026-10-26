#!/usr/bin/env python3
"""Convert a LeRobot v3.0 dataset (consolidated parquet/mp4 files) to the v2.1 per-episode layout.

InternW0-delta's post-training loader is a vendored LeRobot v2.1 reader (meta/episodes.jsonl, one
parquet and one mp4 per episode and camera). so101_bench_sim is published as v3.0 only.

Per episode: the parquet rows are copied unchanged; each camera's frames are cut from the shared v3
mp4 by the episode's [from_timestamp, to_timestamp) and re-encoded to H.264 (CRF 18, yuv420p) with
timestamps restarting at 0. Stream copy is not used: an episode boundary need not fall on a keyframe.
Every episode's decoded frame count is checked against its length. Metadata: info.json (v2.1 paths),
tasks.jsonl, episodes.jsonl, episodes_stats.jsonl (from the v3 per-episode stats columns), stats.json.

  convert_v30_to_v21.py SRC_V30 DST_V21 [--workers N]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK = 1000
V21_DATA = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def read_episodes(src: Path) -> list[dict]:
    rows = []
    for f in sorted((src / "meta" / "episodes").glob("*/*.parquet")):
        rows.extend(pq.read_table(f).to_pylist())
    rows.sort(key=lambda r: r["episode_index"])
    return rows


def convert_video(src: Path, info: dict, ep: dict, key: str, dst: Path) -> tuple[int, str, int]:
    fps = info["fps"]
    path = src / info["video_path"].format(
        video_key=key, chunk_index=ep[f"videos/{key}/chunk_index"], file_index=ep[f"videos/{key}/file_index"]
    )
    start, length = float(ep[f"videos/{key}/from_timestamp"]), int(ep["length"])
    out = dst / V21_VIDEO.format(episode_chunk=ep["episode_index"] // CHUNK, video_key=key, episode_index=ep["episode_index"])
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int((start - 1.0) / stream.time_base)), stream=stream, backward=True)
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < start - 0.5 / fps:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == length:
                break
    if len(frames) != length:
        raise RuntimeError(f"episode {ep['episode_index']} {key}: decoded {len(frames)} of {length} frames")
    h, w = frames[0].shape[:2]
    with av.open(str(out), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.time_base = Fraction(1, fps)
        stream.options = {"crf": "18", "preset": "veryfast", "g": "2"}
        for i, rgb in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return ep["episode_index"], key, length


def write_parquets(src: Path, info: dict, episodes: list[dict], dst: Path) -> None:
    by_file: dict[tuple[int, int], list[dict]] = {}
    for ep in episodes:
        by_file.setdefault((ep["data/chunk_index"], ep["data/file_index"]), []).append(ep)
    for (chunk, file), eps in sorted(by_file.items()):
        table = pq.read_table(src / info["data_path"].format(chunk_index=chunk, file_index=file))
        ep_col = table.column("episode_index").to_numpy()
        for ep in eps:
            rows = table.filter(pa.array(ep_col == ep["episode_index"]))
            if rows.num_rows != ep["length"]:
                raise RuntimeError(f"episode {ep['episode_index']}: {rows.num_rows} rows, expected {ep['length']}")
            out = dst / V21_DATA.format(episode_chunk=ep["episode_index"] // CHUNK, episode_index=ep["episode_index"])
            out.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(rows, out)


def episode_stats(ep: dict) -> dict:
    stats: dict[str, dict] = {}
    for col, value in ep.items():
        if not col.startswith("stats/"):
            continue
        _, feature, stat = col.split("/", 2)
        stats.setdefault(feature, {})[stat] = np.asarray(value).tolist()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--workers", type=int, default=96)
    args = parser.parse_args()
    src, dst = args.src, args.dst
    info = json.loads((src / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0", info["codebase_version"]
    episodes = read_episodes(src)
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    tasks = pq.read_table(src / "meta" / "tasks.parquet").to_pandas()
    (dst / "meta").mkdir(parents=True, exist_ok=True)

    print(f"{len(episodes)} episodes, cameras {video_keys}", flush=True)
    write_parquets(src, info, episodes, dst)
    print("parquets written", flush=True)

    done = 0
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(convert_video, src, info, ep, key, dst) for ep in episodes for key in video_keys]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 500 == 0:
                print(f"videos {done}/{len(futures)}", flush=True)

    task_rows = [{"task_index": int(i), "task": str(t)} for t, i in zip(tasks.index, tasks["task_index"])]
    task_rows.sort(key=lambda r: r["task_index"])
    with (dst / "meta" / "tasks.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in task_rows)
    with (dst / "meta" / "episodes.jsonl").open("w") as f:
        f.writelines(json.dumps({"episode_index": e["episode_index"], "tasks": list(e["tasks"]), "length": e["length"]}) + "\n" for e in episodes)
    with (dst / "meta" / "episodes_stats.jsonl").open("w") as f:
        f.writelines(json.dumps({"episode_index": e["episode_index"], "stats": episode_stats(e)}) + "\n" for e in episodes)
    shutil.copy2(src / "meta" / "stats.json", dst / "meta" / "stats.json")
    v21 = {k: v for k, v in info.items() if k not in ("data_files_size_in_mb", "video_files_size_in_mb")}
    v21.update(
        codebase_version="v2.1", data_path=V21_DATA, video_path=V21_VIDEO, chunks_size=CHUNK,
        total_chunks=math.ceil(len(episodes) / CHUNK), total_videos=len(episodes) * len(video_keys),
    )
    for key in video_keys:
        v21["features"][key].setdefault("info", {})["video.codec"] = "h264"
    (dst / "meta" / "info.json").write_text(json.dumps(v21, indent=4))
    print(f"wrote v2.1 dataset to {dst}", flush=True)


if __name__ == "__main__":
    main()
