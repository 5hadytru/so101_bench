#!/usr/bin/env python3
"""Review rollout episodes and initialize the so101_bench_sim_5 seed dataset.

The tool decodes the recorded videos directly; it does not launch Isaac Lab or
replay actions. Decisions are saved after every edit and are safe to resume.
Once every episode is decided, optional finalization materializes a compact
LeRobot dataset and moves the corresponding task rows to the start of
tasks/teleop_3.jsonl.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import textwrap
import time
import tkinter as tk
from typing import Any, Iterator

import av
import cv2
import numpy as np
from PIL import Image, ImageTk
import pyarrow.parquet as pq


DEFAULT_DATASET = Path("data/lerobot/groot_n16_v7_50k_teleop3_bins_ah8")
DEFAULT_SELECTION = Path("data/lerobot/so101_bench_sim_5_teleop3_bins_seed_selection.json")
DEFAULT_TARGET_DATASET = Path("data/lerobot/so101_bench_sim_5")
DEFAULT_TASK_FILE = Path("tasks/teleop_3.jsonl")
DEFAULT_EVAL_TASK_FILE = Path("tasks/teleop_3_bin_1obj_named_bin.jsonl")
DEFAULT_FILTER_MANIFEST = Path("tasks/teleop_3_bin_1obj_only_filter_manifest.jsonl")
SOURCE_REPO_ID = "local/groot_n16_v7_50k_teleop3_bins_ah8"
TARGET_REPO_ID = "5hadytru/so101_bench_sim_5"
VIDEO_KEYS = ("observation.images.overhead", "observation.images.front")
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
WINDOW_NAME = "LeRobot episode selector - so101_bench_sim_5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--selection-file", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--target-dataset-root", type=Path, default=DEFAULT_TARGET_DATASET)
    parser.add_argument("--task-file", type=Path, default=DEFAULT_TASK_FILE)
    parser.add_argument("--eval-task-file", type=Path, default=DEFAULT_EVAL_TASK_FILE)
    parser.add_argument("--filter-manifest", type=Path, default=DEFAULT_FILTER_MANIFEST)
    parser.add_argument(
        "--speed",
        type=float,
        default=4.0,
        help="Initial playback multiplier. The nearest supported speed is used (default: 4).",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=None,
        help="Episode index to open first. By default, resume at the first undecided episode.",
    )
    parser.add_argument(
        "--view-width",
        type=int,
        default=640,
        help="Displayed width of each camera view (default: 640).",
    )
    parser.add_argument(
        "--overhead-only",
        action="store_true",
        help="Display only the overhead camera instead of overhead and front side by side.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate metadata and decode the first frame of the selected starting episode, then exit.",
    )
    parser.add_argument(
        "--finalize-on-complete",
        action="store_true",
        help="After the last decision, build the seed dataset and reorder teleop_3.jsonl.",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Skip the GUI and finalize an already-complete saved review.",
    )
    parser.add_argument(
        "--dry-run-finalize",
        action="store_true",
        help="Validate finalization and print its effects without changing the task or dataset.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_dataset(dataset_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing episode metadata: {episodes_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    columns = ["episode_index", "tasks", "length"]
    for video_key in VIDEO_KEYS:
        columns.extend(
            [
                f"videos/{video_key}/chunk_index",
                f"videos/{video_key}/file_index",
                f"videos/{video_key}/from_timestamp",
                f"videos/{video_key}/to_timestamp",
            ]
        )
    episodes = pq.read_table(episodes_path, columns=columns).to_pylist()
    episodes.sort(key=lambda row: int(row["episode_index"]))

    expected = list(range(len(episodes)))
    actual = [int(row["episode_index"]) for row in episodes]
    if actual != expected:
        raise ValueError("This selector currently requires contiguous episode indices starting at zero.")
    if int(info.get("total_episodes", -1)) != len(episodes):
        raise ValueError(
            f"info.json reports {info.get('total_episodes')} episodes, but metadata contains {len(episodes)}"
        )
    return info, episodes


def empty_selection(dataset_root: Path, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_dataset": str(dataset_root),
        "target_dataset": "so101_bench_sim_5",
        "total_source_episodes": len(episodes),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "decisions": {},
    }


def load_selection(
    path: Path, dataset_root: Path, episodes: list[dict[str, Any]]
) -> dict[str, Any]:
    if not path.exists():
        return empty_selection(dataset_root, episodes)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("decisions"), dict):
        raise ValueError(f"Unsupported selection file schema: {path}")
    if int(value.get("total_source_episodes", -1)) != len(episodes):
        raise ValueError(f"Selection file does not describe this {len(episodes)}-episode dataset: {path}")
    recorded_source = Path(str(value.get("source_dataset", ""))).resolve()
    if recorded_source != dataset_root.resolve():
        raise ValueError(
            f"Selection file belongs to {recorded_source}, not {dataset_root.resolve()}: {path}"
        )
    for key, row in value["decisions"].items():
        index = int(key)
        if index < 0 or index >= len(episodes):
            raise ValueError(f"Selection file contains out-of-range episode {index}")
        if row.get("decision") not in ("selected", "rejected"):
            raise ValueError(f"Invalid decision for episode {index}: {row}")
    return value


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)


def instruction(row: dict[str, Any]) -> str:
    tasks = row.get("tasks") or []
    return str(tasks[0]) if tasks else "<missing instruction>"


def selected_manifest_path(selection_path: Path) -> Path:
    return selection_path.with_name(f"{selection_path.stem}_selected.jsonl")


def selected_indices_path(selection_path: Path) -> Path:
    return selection_path.with_name(f"{selection_path.stem}_selected_episode_indices.txt")


def save_selection(
    path: Path,
    selection: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> None:
    selection["updated_at"] = utc_now()
    atomic_write_text(path, json.dumps(selection, indent=2, sort_keys=True) + "\n")

    selected_rows: list[dict[str, Any]] = []
    selected_indices: list[int] = []
    for row in episodes:
        index = int(row["episode_index"])
        decision = selection["decisions"].get(str(index))
        if decision is None or decision["decision"] != "selected":
            continue
        selected_indices.append(index)
        selected_rows.append(
            {
                "schema_version": 1,
                "target_dataset": "so101_bench_sim_5",
                "source_dataset": selection["source_dataset"],
                "source_episode_index": index,
                "instruction": instruction(row),
                "length": int(row["length"]),
                "selected_at": decision["decided_at"],
            }
        )
    jsonl = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selected_rows)
    atomic_write_text(selected_manifest_path(path), jsonl)
    indices = ",".join(str(index) for index in selected_indices)
    atomic_write_text(selected_indices_path(path), indices + ("\n" if indices else ""))


def video_path(dataset_root: Path, info: dict[str, Any], row: dict[str, Any], video_key: str) -> Path:
    template = str(info["video_path"])
    relative = template.format(
        video_key=video_key,
        chunk_index=int(row[f"videos/{video_key}/chunk_index"]),
        file_index=int(row[f"videos/{video_key}/file_index"]),
    )
    path = dataset_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing video shard: {path}")
    return path


@dataclass
class EpisodeDecoder:
    """Forward-only, exact-timestamp PyAV decoder for one episode and camera."""

    path: Path
    start_s: float
    length: int
    fps: float

    def __post_init__(self) -> None:
        self.container: av.container.InputContainer | None = None
        self.frames: Iterator[av.VideoFrame] | None = None
        self.relative_index = -1
        self.image: np.ndarray | None = None
        self.restart()

    def restart(self) -> None:
        self.close()
        self.container = av.open(str(self.path))
        stream = self.container.streams.video[0]
        seek_pts = max(0, int(self.start_s / float(stream.time_base)))
        self.container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
        tolerance = 0.5 / self.fps

        def episode_frames() -> Iterator[av.VideoFrame]:
            assert self.container is not None
            accepted = 0
            for frame in self.container.decode(stream):
                timestamp = frame.time
                if timestamp is None:
                    timestamp = float(frame.pts * stream.time_base)
                if timestamp < self.start_s - tolerance:
                    continue
                yield frame
                accepted += 1
                if accepted >= self.length:
                    return

        self.frames = episode_frames()
        self.relative_index = -1
        self.image = None

    def get(self, target_index: int) -> np.ndarray:
        if target_index < self.relative_index:
            self.restart()
        assert self.frames is not None
        while self.relative_index < target_index:
            try:
                frame = next(self.frames)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Video ended early at episode frame {self.relative_index + 1}/{self.length}: {self.path}"
                ) from exc
            self.relative_index += 1
            if self.relative_index == target_index:
                self.image = frame.to_ndarray(format="bgr24")
        assert self.image is not None
        return self.image

    def close(self) -> None:
        if self.container is not None:
            self.container.close()
        self.container = None
        self.frames = None


def resize_view(image: np.ndarray, width: int) -> np.ndarray:
    height = round(image.shape[0] * width / image.shape[1])
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def put_text(image: np.ndarray, value: str, x: int, y: int, scale: float = 0.62) -> None:
    cv2.putText(image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 2, cv2.LINE_AA)


class TkDisplay:
    """Small Tk display wrapper, used because the environment ships headless OpenCV."""

    def __init__(self, title: str):
        self.root = tk.Tk()
        self.root.title(title)
        self.label = tk.Label(self.root, borderwidth=0)
        self.label.pack()
        self.pending_key = ""
        self.closed = False
        self.photo: ImageTk.PhotoImage | None = None
        self.root.bind("<KeyPress>", self._on_key)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.focus_force()

    def _on_key(self, event: tk.Event) -> None:
        if event.keysym == "Escape":
            self.pending_key = "q"
        elif event.keysym == "space":
            self.pending_key = " "
        elif event.char:
            self.pending_key = event.char.lower()

    def _on_close(self) -> None:
        self.closed = True
        self.pending_key = "q"

    def show_and_poll(self, bgr_image: np.ndarray, delay_ms: int) -> str:
        if self.closed:
            return "q"
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb_image))
        self.label.configure(image=self.photo)
        try:
            self.root.update_idletasks()
            self.root.update()
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
                self.root.update()
        except tk.TclError:
            return "q"
        key = self.pending_key
        self.pending_key = ""
        return key

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def compose_display(
    images: list[np.ndarray],
    row: dict[str, Any],
    current_position: int,
    total: int,
    frame_index: int,
    speed: float,
    paused: bool,
    ended: bool,
    decisions: dict[str, Any],
    view_width: int,
) -> np.ndarray:
    views = [resize_view(image, view_width) for image in images]
    camera_canvas = np.hstack(views)
    panel_height = 150
    canvas = np.zeros((camera_canvas.shape[0] + panel_height, camera_canvas.shape[1], 3), dtype=np.uint8)
    canvas[panel_height:] = camera_canvas

    episode_index = int(row["episode_index"])
    current_decision = decisions.get(str(episode_index), {}).get("decision", "undecided")
    selected = sum(item["decision"] == "selected" for item in decisions.values())
    rejected = sum(item["decision"] == "rejected" for item in decisions.values())
    state = "END - choose, replay, or navigate" if ended else ("PAUSED" if paused else "PLAYING")
    put_text(
        canvas,
        f"Episode {episode_index} ({current_position + 1}/{total})  frame {frame_index + 1}/{row['length']}  "
        f"{speed:g}x  {state}",
        12,
        25,
    )
    put_text(
        canvas,
        f"Decision: {current_decision.upper()}   selected={selected} rejected={rejected} "
        f"undecided={total - selected - rejected}",
        12,
        52,
    )
    wrapped = textwrap.wrap(instruction(row), width=max(50, camera_canvas.shape[1] // 13))[:2]
    for line_number, line in enumerate(wrapped):
        put_text(canvas, line, 12, 79 + line_number * 24, scale=0.58)
    put_text(
        canvas,
        "K/Y keep  X/N reject  U clear  R replay  SPACE pause  [/] speed  A/D prev/next  J next-undecided  Q quit",
        12,
        139,
        scale=0.46,
    )
    return canvas


def nearest_speed_index(value: float) -> int:
    if value <= 0:
        raise ValueError("--speed must be positive")
    return min(range(len(SPEEDS)), key=lambda index: abs(SPEEDS[index] - value))


def next_undecided(start: int, decisions: dict[str, Any], total: int) -> int | None:
    for offset in range(total):
        candidate = (start + offset) % total
        if str(candidate) not in decisions:
            return candidate
    return None


def make_decoders(
    dataset_root: Path,
    info: dict[str, Any],
    row: dict[str, Any],
    video_keys: tuple[str, ...],
) -> list[EpisodeDecoder]:
    fps = float(info["fps"])
    result = []
    for video_key in video_keys:
        result.append(
            EpisodeDecoder(
                path=video_path(dataset_root, info, row, video_key),
                start_s=float(row[f"videos/{video_key}/from_timestamp"]),
                length=int(row["length"]),
                fps=fps,
            )
        )
    return result


def close_decoders(decoders: list[EpisodeDecoder]) -> None:
    for decoder in decoders:
        decoder.close()


def is_bin_eval_task(row: dict[str, Any]) -> bool:
    """Match create_teleop_3_tasks.py's bin_1obj/named_bin filter."""
    if "direction" in row or row.get("referents"):
        return False
    return "target" in row or int(row.get("n_objects", 0)) == 1


def task_semantics(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "trial_id"}


def selected_episode_indices(selection: dict[str, Any], total: int) -> list[int]:
    return [
        index
        for index in range(total)
        if selection["decisions"].get(str(index), {}).get("decision") == "selected"
    ]


def prepare_task_reorder(
    task_file: Path,
    eval_task_file: Path,
    filter_manifest_path: Path,
    selected_eval_indices: list[int],
    episode_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    tasks = read_jsonl(task_file)
    eval_tasks = read_jsonl(eval_task_file)
    filter_manifest = read_jsonl(filter_manifest_path)
    if len(eval_tasks) != episode_count or len(filter_manifest) != episode_count:
        raise ValueError(
            f"Expected {episode_count} eval task/manifest rows, got "
            f"{len(eval_tasks)} and {len(filter_manifest)}"
        )

    bin_eval_original_indices = [index for index, row in enumerate(tasks) if is_bin_eval_task(row)]
    if len(bin_eval_original_indices) != 1000:
        raise ValueError(
            f"Expected teleop_3.jsonl to contain 1000 1-object/named-bin rows; "
            f"found {len(bin_eval_original_indices)}. It may already have been reordered."
        )

    eval_to_task: dict[int, int] = {}
    for expected_eval_index, manifest_row in enumerate(filter_manifest):
        eval_index = int(manifest_row["pruned_dataset_episode_index"])
        if eval_index != expected_eval_index:
            raise ValueError("Filter manifest is not contiguous and ordered by pruned episode index")
        source_bin_index = int(manifest_row["source_dataset_episode_index"])
        if source_bin_index < 0 or source_bin_index >= len(bin_eval_original_indices):
            raise IndexError(f"Filter manifest source index is out of range: {source_bin_index}")
        original_task_index = bin_eval_original_indices[source_bin_index]
        if task_semantics(eval_tasks[eval_index]) != task_semantics(tasks[original_task_index]):
            raise ValueError(
                f"Eval task {eval_index} does not match teleop_3 task row {original_task_index}"
            )
        eval_to_task[eval_index] = original_task_index

    selected_task_indices = [eval_to_task[index] for index in selected_eval_indices]
    if len(selected_task_indices) != len(set(selected_task_indices)):
        raise ValueError("Multiple selected episodes mapped to the same teleop_3 task row")
    selected_set = set(selected_task_indices)
    old_indices = selected_task_indices + [index for index in range(len(tasks)) if index not in selected_set]
    reordered: list[dict[str, Any]] = []
    reorder_manifest: list[dict[str, Any]] = []
    eval_by_task = {task_index: eval_index for eval_index, task_index in eval_to_task.items()}
    for new_index, old_index in enumerate(old_indices):
        row = dict(tasks[old_index])
        row["trial_id"] = new_index
        reordered.append(row)
        reorder_manifest.append(
            {
                "schema_version": 1,
                "new_trial_id": new_index,
                "source_trial_id": old_index,
                "selected_seed_episode": old_index in selected_set,
                "source_eval_episode_index": eval_by_task.get(old_index),
            }
        )
    return reordered, reorder_manifest, selected_task_indices


def materialize_selected_dataset(
    source_root: Path,
    target_root: Path,
    selected_indices: list[int],
    total: int,
    provenance: dict[str, Any],
) -> None:
    if target_root.exists():
        raise FileExistsError(
            f"Target dataset already exists: {target_root}. Refusing to overwrite it."
        )
    target_root.parent.mkdir(parents=True, exist_ok=True)
    build_root = target_root.with_name(f".{target_root.name}.build-{os.getpid()}-{time.time_ns()}")
    try:
        if len(selected_indices) == total:
            shutil.copytree(source_root, build_root)
        else:
            from lerobot.datasets.dataset_tools import delete_episodes
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            rejected = sorted(set(range(total)) - set(selected_indices))
            source_dataset = LeRobotDataset(SOURCE_REPO_ID, root=source_root)
            delete_episodes(
                source_dataset,
                rejected,
                output_dir=build_root,
                repo_id=TARGET_REPO_ID,
            )
        built_info = json.loads((build_root / "meta" / "info.json").read_text(encoding="utf-8"))
        if int(built_info.get("total_episodes", -1)) != len(selected_indices):
            raise RuntimeError(
                f"Built dataset has {built_info.get('total_episodes')} episodes, "
                f"expected {len(selected_indices)}"
            )
        atomic_write_text(
            build_root / "meta" / "seed_selection_provenance.json",
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        )
        os.replace(build_root, target_root)
    except BaseException:
        if build_root.exists():
            print(f"Incomplete build retained for inspection: {build_root}")
        raise


def finalize(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    selection_path = args.selection_file.resolve()
    target_root = args.target_dataset_root.resolve()
    task_file = args.task_file.resolve()
    eval_task_file = args.eval_task_file.resolve()
    filter_manifest = args.filter_manifest.resolve()
    info, episodes = read_dataset(dataset_root)
    selection = load_selection(selection_path, dataset_root, episodes)
    if selection.get("finalization", {}).get("status") == "complete":
        print(f"This review was already finalized at {selection['finalization']['completed_at']}.")
        return
    undecided = [index for index in range(len(episodes)) if str(index) not in selection["decisions"]]
    if undecided:
        raise ValueError(
            f"Cannot finalize: {len(undecided)} episode(s) remain undecided "
            f"(first: {undecided[:10]})."
        )
    selected = selected_episode_indices(selection, len(episodes))
    if not selected:
        raise ValueError("Cannot initialize a LeRobot dataset with zero selected episodes")

    reordered, reorder_manifest, selected_task_indices = prepare_task_reorder(
        task_file, eval_task_file, filter_manifest, selected, len(episodes)
    )
    selected_frames = sum(int(episodes[index]["length"]) for index in selected)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_backup = task_file.with_name(f"{task_file.stem}.pre_seed_reorder_{timestamp}{task_file.suffix}")
    reorder_manifest_path = task_file.with_name(
        f"{task_file.stem}_seed_reorder_manifest_{timestamp}.jsonl"
    )
    provenance = {
        "schema_version": 1,
        "created_at": utc_now(),
        "source_dataset": str(dataset_root),
        "source_total_episodes": len(episodes),
        "source_total_frames": int(info["total_frames"]),
        "selection_file": str(selection_path),
        "selected_source_episode_indices": selected,
        "selected_source_task_indices": selected_task_indices,
        "target_repo_id": TARGET_REPO_ID,
        "target_total_episodes": len(selected),
        "target_total_frames": selected_frames,
    }
    print(
        f"Finalization validated: {len(selected)}/{len(episodes)} selected, "
        f"{selected_frames} frames."
    )
    print(f"Task rows moving to prefix: {len(selected_task_indices)}")
    print(f"Target dataset: {target_root}")
    if args.dry_run_finalize:
        print("Dry run only; no files changed.")
        return

    materialize_selected_dataset(dataset_root, target_root, selected, len(episodes), provenance)
    shutil.copy2(task_file, task_backup)
    atomic_write_text(task_file, jsonl_text(reordered))
    atomic_write_text(reorder_manifest_path, jsonl_text(reorder_manifest))
    selection["finalization"] = {
        "status": "complete",
        "completed_at": utc_now(),
        "target_dataset": str(target_root),
        "target_repo_id": TARGET_REPO_ID,
        "selected_episodes": len(selected),
        "selected_frames": selected_frames,
        "task_file": str(task_file),
        "task_backup": str(task_backup),
        "task_reorder_manifest": str(reorder_manifest_path),
    }
    save_selection(selection_path, selection, episodes)
    print(f"Initialized {target_root} with {len(selected)} selected episodes.")
    print(f"Moved their task rows to the beginning of {task_file}.")
    print(f"Original task file backup: {task_backup}")


def run(args: argparse.Namespace) -> bool:
    dataset_root = args.dataset_root.resolve()
    info, episodes = read_dataset(dataset_root)
    selection = load_selection(args.selection_file, dataset_root, episodes)
    decisions = selection["decisions"]
    video_keys = VIDEO_KEYS[:1] if args.overhead_only else VIDEO_KEYS
    speed_index = nearest_speed_index(args.speed)

    if args.start_episode is not None:
        if args.start_episode < 0 or args.start_episode >= len(episodes):
            raise IndexError(f"--start-episode must be in [0, {len(episodes) - 1}]")
        position = args.start_episode
    else:
        position = next_undecided(0, decisions, len(episodes)) or 0

    if args.verify_only:
        decoders = make_decoders(dataset_root, info, episodes[position], video_keys)
        try:
            shapes = [decoder.get(0).shape for decoder in decoders]
        finally:
            close_decoders(decoders)
        print(f"Validated {len(episodes)} episodes; decoded episode {position} first-frame shapes: {shapes}")
        return False

    save_selection(args.selection_file, selection, episodes)
    print(f"Loaded {len(episodes)} episodes; starting at episode {position}.")
    print(f"Selections: {args.selection_file}")
    print(f"Selected manifest: {selected_manifest_path(args.selection_file)}")
    print(f"Selected indices: {selected_indices_path(args.selection_file)}")

    display_window = TkDisplay(WINDOW_NAME)
    decoders: list[EpisodeDecoder] = []
    try:
        while True:
            row = episodes[position]
            close_decoders(decoders)
            decoders = make_decoders(dataset_root, info, row, video_keys)
            source_cursor = 0.0
            paused = False
            ended = False
            navigate_to: int | None = None

            while navigate_to is None:
                target_frame = min(int(source_cursor), int(row["length"]) - 1)
                images = [decoder.get(target_frame) for decoder in decoders]
                display = compose_display(
                    images=images,
                    row=row,
                    current_position=position,
                    total=len(episodes),
                    frame_index=target_frame,
                    speed=SPEEDS[speed_index],
                    paused=paused,
                    ended=ended,
                    decisions=decisions,
                    view_width=args.view_width,
                )
                key_char = display_window.show_and_poll(display, max(1, round(1000 / float(info["fps"]))))

                if key_char == "q":
                    return len(decisions) == len(episodes)
                if key_char in ("k", "y"):
                    decisions[str(position)] = {"decision": "selected", "decided_at": utc_now()}
                    save_selection(args.selection_file, selection, episodes)
                    if len(decisions) == len(episodes):
                        print("All episodes decided.")
                        return True
                    navigate_to = (position + 1) % len(episodes)
                    continue
                if key_char in ("x", "n"):
                    decisions[str(position)] = {"decision": "rejected", "decided_at": utc_now()}
                    save_selection(args.selection_file, selection, episodes)
                    if len(decisions) == len(episodes):
                        print("All episodes decided.")
                        return True
                    navigate_to = (position + 1) % len(episodes)
                    continue
                if key_char == "u":
                    decisions.pop(str(position), None)
                    save_selection(args.selection_file, selection, episodes)
                elif key_char == "r":
                    for decoder in decoders:
                        decoder.restart()
                    source_cursor = 0.0
                    paused = False
                    ended = False
                elif key_char == " ":
                    paused = not paused
                elif key_char in ("]", "+", "=",
                ):
                    speed_index = min(speed_index + 1, len(SPEEDS) - 1)
                elif key_char in ("[", "-", "_"):
                    speed_index = max(speed_index - 1, 0)
                elif key_char in ("a", "h"):
                    navigate_to = (position - 1) % len(episodes)
                elif key_char in ("d", "l"):
                    navigate_to = (position + 1) % len(episodes)
                elif key_char == "j":
                    candidate = next_undecided(position + 1, decisions, len(episodes))
                    if candidate is not None:
                        navigate_to = candidate

                if not paused and not ended:
                    source_cursor += SPEEDS[speed_index]
                    if source_cursor >= int(row["length"]) - 1:
                        source_cursor = float(int(row["length"]) - 1)
                        ended = True

            position = navigate_to
    finally:
        close_decoders(decoders)
        display_window.close()


def main() -> None:
    args = parse_args()
    if args.finalize or args.dry_run_finalize:
        finalize(args)
        return
    completed = run(args)
    if completed and args.finalize_on_complete:
        finalize(args)
    elif completed:
        print(f"Review complete. Finalize with:\n  {Path(__file__).name} --finalize")


if __name__ == "__main__":
    main()
