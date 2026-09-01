#!/home/truman/env_isaaclab_51/bin/python
"""View matching original and slowed LeRobot episodes side by side.

The original starts on the left and the slowed resimulation starts on the
right. Both are played at the requested viewing speed. When the original ends
first, its last frame remains visible while the slowed episode finishes.
Interactive mode keeps one window alive across episodes and accepts keyboard
navigation without launching a new player process. Optional review mode writes
durable keep/re-record decisions and automatically includes quarantined source
episodes in the re-record set.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq


DEFAULT_ORIGINAL_ROOT = Path("data/lerobot/so101_bench_sim_4")
DEFAULT_SLOWED_ROOT = Path(
    "data/lerobot/so101_bench_sim_4_slowed_1.7x_phase_governed"
)
DEFAULT_SLOWED_OUTCOMES = Path(
    "outputs/so101_bench_sim_4_slowed_1.7x_phase_governed_resimulation/episodes.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively review matching original/slowed LeRobot episodes "
            "side by side in one persistent keyboard-controlled window."
        )
    )
    parser.add_argument(
        "episode",
        type=int,
        nargs="?",
        default=0,
        help="Zero-based dataset episode index (default: 0).",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Interactive mode now always keeps "
            "one live window and accepts keyboard episode controls."
        ),
    )
    parser.add_argument(
        "--camera",
        choices=("overhead", "front"),
        default="overhead",
        help="Camera to compare (default: overhead).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=2.0,
        help="Viewing-speed multiplier applied to both videos (default: 2.0).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height of each side of the comparison window (default: 480).",
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        default=DEFAULT_ORIGINAL_ROOT,
        help=f"Original LeRobot dataset root (default: {DEFAULT_ORIGINAL_ROOT}).",
    )
    parser.add_argument(
        "--slowed-root",
        type=Path,
        default=DEFAULT_SLOWED_ROOT,
        help=f"Slowed LeRobot dataset root (default: {DEFAULT_SLOWED_ROOT}).",
    )
    parser.add_argument(
        "--slowed-outcomes",
        type=Path,
        default=DEFAULT_SLOWED_OUTCOMES,
        help=(
            "Outcome JSONL used to resolve video spans while slowed metadata is being rewritten "
            f"(default: {DEFAULT_SLOWED_OUTCOMES})."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write one H.264 comparison MP4 instead of opening the interactive viewer.",
    )
    parser.add_argument(
        "--review-file",
        type=Path,
        default=None,
        help=(
            "Enable curation and persist keep/re-record decisions to this JSONL. "
            "Quarantined source episodes are automatically marked re-record."
        ),
    )
    parser.add_argument(
        "--resume-review",
        action="store_true",
        help="Start at the first source episode without a review decision.",
    )
    parser.add_argument(
        "--delete-only-review",
        action="store_true",
        help=(
            "Treat X/Delete and automatic quarantines as permanent deletion rather than re-recording. "
            "Requires --review-file."
        ),
    )
    parser.add_argument(
        "--final-frame-only",
        action="store_true",
        help=(
            "Show only the original and slowed final frames. Press F in the "
            "viewer to toggle between final frames and video playback."
        ),
    )
    return parser.parse_args()


def load_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata does not exist: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def read_episode_row(
    root: Path,
    episode_index: int,
    *,
    columns: list[str],
    attempts: int = 4,
) -> dict[str, Any]:
    """Read an episode row from a stable snapshot of possibly-live metadata."""

    metadata_paths = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not metadata_paths:
        raise FileNotFoundError(f"No episode metadata parquet files found under {root}")

    last_error: Exception | None = None
    for attempt in range(attempts):
        for metadata_path in metadata_paths:
            try:
                # The active recorder rewrites its parquet metadata. Copying first
                # prevents a valid read from being invalidated halfway through.
                with tempfile.NamedTemporaryFile(suffix=".parquet") as snapshot:
                    with metadata_path.open("rb") as source:
                        shutil.copyfileobj(source, snapshot)
                    snapshot.flush()
                    table = pq.read_table(snapshot.name, columns=columns)
                indices = table.column("episode_index").to_pylist()
                try:
                    row_index = indices.index(episode_index)
                except ValueError:
                    continue
                return table.slice(row_index, 1).to_pylist()[0]
            except Exception as exc:  # A live parquet rewrite can be briefly incomplete.
                last_error = exc
        if attempt + 1 < attempts:
            time.sleep(0.25)

    if last_error is not None:
        raise RuntimeError(
            f"Could not read stable episode metadata from {root} after {attempts} attempts"
        ) from last_error
    raise IndexError(
        f"Episode {episode_index} is not available in {root}. "
        "For an active slowed run, choose an episode that has finished recording."
    )


def video_span(root: Path, episode_index: int, camera: str) -> dict[str, Any]:
    info = load_info(root)
    video_key = f"observation.images.{camera}"
    if video_key not in info.get("features", {}):
        available = sorted(
            key for key in info.get("features", {}) if key.startswith("observation.images.")
        )
        raise KeyError(f"{root} has no {video_key!r}; available video keys: {available}")

    prefix = f"videos/{video_key}"
    columns = [
        "episode_index",
        "tasks",
        "length",
        f"{prefix}/chunk_index",
        f"{prefix}/file_index",
        f"{prefix}/from_timestamp",
        f"{prefix}/to_timestamp",
    ]
    row = read_episode_row(root, episode_index, columns=columns)
    chunk_index = int(row[f"{prefix}/chunk_index"])
    file_index = int(row[f"{prefix}/file_index"])
    video_path_template = str(
        info.get(
            "video_path",
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        )
    )
    video_path = root / video_path_template.format(
        video_key=video_key,
        chunk_index=chunk_index,
        file_index=file_index,
    )
    if not video_path.is_file():
        raise FileNotFoundError(f"Episode video does not exist: {video_path}")

    start = float(row[f"{prefix}/from_timestamp"])
    end = float(row[f"{prefix}/to_timestamp"])
    if end <= start:
        raise ValueError(f"Invalid video span for episode {episode_index} in {root}: {start}..{end}")
    return {
        "path": video_path,
        "start": start,
        "duration": end - start,
        "frames": int(row["length"]),
        "fps": float(info.get("fps", 30.0)),
        "tasks": list(row.get("tasks") or []),
    }


def probe_video_frames(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams or streams[0].get("nb_frames") in (None, "N/A"):
        raise RuntimeError(f"ffprobe did not report a frame count for {path}")
    return int(streams[0]["nb_frames"])


def probe_video_dimensions(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe did not report dimensions for {path}")
    width = int(streams[0]["width"])
    height = int(streams[0]["height"])
    if width < 1 or height < 1:
        raise RuntimeError(f"Invalid video dimensions for {path}: {width}x{height}")
    return width, height


@lru_cache(maxsize=8)
def load_outcome_records(outcomes_path: Path) -> tuple[dict[str, Any], ...]:
    if not outcomes_path.is_file():
        return ()
    return tuple(
        json.loads(line)
        for line in outcomes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def load_slowed_outcome(
    outcomes_path: Path,
    source_episode_index: int,
) -> dict[str, Any] | None:
    """Resolve a source episode to its admitted slowed-dataset episode."""

    if not outcomes_path.is_file():
        return None
    records = load_outcome_records(outcomes_path)
    requested = next(
        (
            record
            for record in records
            if int(record["dataset"]["episode_index"]) == source_episode_index
        ),
        None,
    )
    if requested is None:
        raise IndexError(
            f"Source episode {source_episode_index} has not finished processing; "
            f"{outcomes_path} currently contains {len(records)} completed outcome(s)."
        )
    recorded = requested.get("recorded_sim_dataset")
    if recorded is None:
        reason = (
            requested.get("outcome_quality", {}).get(
                "training_data_quarantine_reason"
            )
            or requested.get("label", {}).get("failure_reason")
            or "unknown"
        )
        raise IndexError(
            f"Source episode {source_episode_index} was quarantined and has no "
            f"slowed training video (reason: {reason})."
        )
    return requested


def video_span_from_outcomes(
    root: Path,
    outcomes_path: Path,
    episode_index: int,
    camera: str,
) -> dict[str, Any]:
    """Reconstruct spans when a live recorder is rewriting episode parquet."""

    if not outcomes_path.is_file():
        raise FileNotFoundError(f"Slowed outcome JSONL does not exist: {outcomes_path}")
    records = list(load_outcome_records(outcomes_path))
    records = [
        record
        for record in records
        if record.get("recorded_sim_dataset") is not None
    ]
    records.sort(
        key=lambda record: int(record["recorded_sim_dataset"]["episode_index"])
    )
    requested = next(
        (
            record
            for record in records
            if int(record["dataset"]["episode_index"]) == episode_index
        ),
        None,
    )
    if requested is None:
        raise IndexError(
            f"Episode {episode_index} has not finished recording; "
            f"{outcomes_path} currently contains {len(records)} completed episode(s)."
        )

    info = load_info(root)
    fps = float(info.get("fps", 30.0))
    video_key = f"observation.images.{camera}"
    video_paths = sorted((root / "videos" / video_key).glob("chunk-*/*.mp4"))
    if not video_paths:
        raise FileNotFoundError(f"No {video_key} MP4 files found under {root}")

    record_cursor = 0
    for video_path in video_paths:
        video_frames = probe_video_frames(video_path)
        frames_in_file = 0
        while record_cursor < len(records):
            record = records[record_cursor]
            record_frames = int(record["episode_length"]["dataset_frames"])
            if frames_in_file + record_frames > video_frames:
                break
            record_episode = int(record["dataset"]["episode_index"])
            if record_episode == episode_index:
                tasks = [str(record.get("benchmark", {}).get("instruction", ""))]
                return {
                    "path": video_path,
                    "start": frames_in_file / fps,
                    "duration": record_frames / fps,
                    "frames": record_frames,
                    "fps": fps,
                    "tasks": [task for task in tasks if task],
                }
            frames_in_file += record_frames
            record_cursor += 1
            if frames_in_file == video_frames:
                break

        # Every finalized file must end at an episode boundary. The active last
        # file may also contain an unfinished episode not present in outcomes.
        if frames_in_file != video_frames and video_path != video_paths[-1]:
            raise RuntimeError(
                f"Could not align completed episode lengths with {video_path}: "
                f"outcomes account for {frames_in_file}/{video_frames} frames."
            )

    raise RuntimeError(
        f"Could not map completed episode {episode_index} into the slowed video files."
    )


def ffmpeg_comparison_command(
    original: dict[str, Any],
    slowed: dict[str, Any],
    *,
    speed: float,
    height: int,
) -> list[str]:
    output_fps = max(float(original["fps"]), float(slowed["fps"])) * speed
    filter_graph = (
        f"[0:v]setpts=(PTS-STARTPTS)/{speed:.12g},"
        f"scale=-2:{height},setsar=1,"
        "drawbox=x=0:y=0:w=iw:h=42:color=black@0.65:t=fill,"
        "drawtext=text='ORIGINAL':x=16:y=10:fontsize=24:fontcolor=white[left];"
        f"[1:v]setpts=(PTS-STARTPTS)/{speed:.12g},"
        f"scale=-2:{height},setsar=1,"
        "drawbox=x=0:y=0:w=iw:h=42:color=black@0.65:t=fill,"
        "drawtext=text='SLOWED':x=16:y=10:fontsize=24:fontcolor=white[right];"
        "[left][right]hstack=inputs=2:shortest=0[v]"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-ss",
        f"{original['start']:.9f}",
        "-t",
        f"{original['duration']:.9f}",
        "-i",
        str(original["path"]),
        "-ss",
        f"{slowed['start']:.9f}",
        "-t",
        f"{slowed['duration']:.9f}",
        "-i",
        str(slowed["path"]),
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
        "-an",
        "-r",
        f"{output_fps:.12g}",
        "-vsync",
        "vfr",
    ]


def comparison_spans(
    args: argparse.Namespace,
    episode_index: int,
) -> tuple[dict[str, Any], dict[str, Any], str, int, str]:
    original = video_span(args.original_root, episode_index, args.camera)
    outcome = load_slowed_outcome(args.slowed_outcomes, episode_index)
    slowed_episode_index = (
        int(outcome["recorded_sim_dataset"]["episode_index"])
        if outcome is not None
        else episode_index
    )
    try:
        slowed = video_span(
            args.slowed_root,
            slowed_episode_index,
            args.camera,
        )
    except (IndexError, RuntimeError):
        print(
            "Slowed episode parquet is busy; reconstructing the span from live outcomes...",
            file=sys.stderr,
        )
        slowed = video_span_from_outcomes(
            args.slowed_root,
            args.slowed_outcomes,
            episode_index,
            args.camera,
        )
    instruction = (
        slowed["tasks"][0]
        if slowed["tasks"]
        else original["tasks"][0]
        if original["tasks"]
        else "<unknown>"
    )
    if outcome is None:
        outcome_text = "OUTCOME UNKNOWN"
    elif bool(outcome.get("label", {}).get("success")):
        outcome_text = "SUCCESS"
    else:
        failure_reason = str(
            outcome.get("label", {}).get("failure_reason") or "unknown"
        )
        outcome_text = f"FAILURE ({failure_reason})"
    return original, slowed, instruction, slowed_episode_index, outcome_text


def _outcome_instruction(outcome: dict[str, Any]) -> str:
    return str(outcome.get("benchmark", {}).get("instruction") or "<unknown>")


def _quarantine_reason(outcome: dict[str, Any]) -> str | None:
    if outcome.get("recorded_sim_dataset") is not None:
        return None
    return str(
        outcome.get("outcome_quality", {}).get("training_data_quarantine_reason")
        or outcome.get("label", {}).get("failure_reason")
        or "quarantined"
    )


class ReviewStore:
    """Durable source-to-slowed curation decisions with atomic rewrites."""

    VALID_DECISIONS = {"keep", "rerecord", "delete"}

    def __init__(
        self,
        path: Path,
        *,
        original_root: Path,
        slowed_root: Path,
        outcomes_path: Path,
        delete_only: bool = False,
    ) -> None:
        self.path = path
        self.original_root = original_root
        self.slowed_root = slowed_root
        self.outcomes_path = outcomes_path
        self.delete_only = delete_only
        outcomes = load_outcome_records(outcomes_path)
        self.outcomes: dict[int, dict[str, Any]] = {}
        for outcome in outcomes:
            source_index = int(outcome["dataset"]["episode_index"])
            if source_index in self.outcomes:
                raise ValueError(
                    f"{outcomes_path} contains duplicate source episode {source_index}"
                )
            self.outcomes[source_index] = outcome
        if not self.outcomes:
            raise ValueError(f"No completed outcomes found in {outcomes_path}")

        self.decisions: dict[int, dict[str, Any]] = {}
        if path.exists():
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                record = json.loads(line)
                source_index = int(record["source_dataset_episode_index"])
                decision = str(record.get("decision"))
                if decision not in self.VALID_DECISIONS:
                    raise ValueError(
                        f"{path}:{line_number}: invalid decision {decision!r}"
                    )
                if source_index in self.decisions:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate source episode {source_index}"
                    )
                self.decisions[source_index] = record

        changed = False
        for source_index, outcome in self.outcomes.items():
            reason = _quarantine_reason(outcome)
            if reason is None:
                continue
            existing = self.decisions.get(source_index)
            if (
                existing is not None
                and existing.get("decision") == ("delete" if self.delete_only else "rerecord")
                and existing.get("decision_source") == "automatic_quarantine"
            ):
                continue
            self.decisions[source_index] = self._record(
                source_index,
                "delete" if self.delete_only else "rerecord",
                decision_source="automatic_quarantine",
                reason=reason,
            )
            changed = True
        if changed or not path.exists():
            self.save()

    @property
    def source_indices(self) -> list[int]:
        return sorted(self.outcomes)

    def _record(
        self,
        source_index: int,
        decision: str,
        *,
        decision_source: str,
        reason: str,
    ) -> dict[str, Any]:
        outcome = self.outcomes[source_index]
        recorded = outcome.get("recorded_sim_dataset")
        return {
            "schema_version": 1,
            "source_dataset_episode_index": source_index,
            "slowed_dataset_episode_index": (
                int(recorded["episode_index"]) if recorded is not None else None
            ),
            "decision": decision,
            "decision_source": decision_source,
            "reason": reason,
            "instruction": _outcome_instruction(outcome),
            "slowed_outcome_success": bool(outcome.get("label", {}).get("success")),
            "original_root": str(self.original_root),
            "slowed_root": str(self.slowed_root),
            "slowed_outcomes": str(self.outcomes_path),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def set_manual(self, source_index: int, decision: str) -> None:
        if decision not in self.VALID_DECISIONS:
            raise ValueError(f"Invalid review decision: {decision}")
        if source_index not in self.outcomes:
            raise IndexError(f"No completed outcome for source episode {source_index}")
        quarantine = _quarantine_reason(self.outcomes[source_index])
        if quarantine is not None and decision not in {"rerecord", "delete"}:
            raise ValueError(
                f"Source episode {source_index} was quarantined and must be re-recorded"
            )
        self.decisions[source_index] = self._record(
            source_index,
            decision,
            decision_source="manual_keyboard",
            reason=(
                "reviewer_keep"
                if decision == "keep"
                else "reviewer_delete"
                if decision == "delete"
                else "reviewer_rerecord"
            ),
        )
        self.save()

    def clear_manual(self, source_index: int) -> None:
        record = self.decisions.get(source_index)
        if record is None:
            return
        if record.get("decision_source") == "automatic_quarantine":
            raise ValueError("Automatic quarantine decisions cannot be cleared")
        del self.decisions[source_index]
        self.save()

    def decision(self, source_index: int) -> str:
        record = self.decisions.get(source_index)
        return str(record["decision"]).upper() if record is not None else "UNREVIEWED"

    def next_undecided(self, source_index: int, *, include_current: bool = False) -> int | None:
        for candidate in self.source_indices:
            if candidate < source_index or (candidate == source_index and not include_current):
                continue
            if candidate not in self.decisions:
                return candidate
        for candidate in self.source_indices:
            if candidate >= source_index:
                break
            if candidate not in self.decisions:
                return candidate
        return None

    def summary(self) -> str:
        keeps = sum(row["decision"] == "keep" for row in self.decisions.values())
        redos = sum(row["decision"] == "rerecord" for row in self.decisions.values())
        deletes = sum(row["decision"] == "delete" for row in self.decisions.values())
        automatic = sum(
            row.get("decision_source") == "automatic_quarantine"
            for row in self.decisions.values()
        )
        undecided = len(self.outcomes) - len(self.decisions)
        return (
            f"KEEP {keeps} | DELETE {deletes} | RE-RECORD {redos} "
            f"({automatic} auto-quarantine) | "
            f"UNREVIEWED {undecided}"
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for source_index in sorted(self.decisions):
                stream.write(
                    json.dumps(
                        self.decisions[source_index],
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            stream.flush()
        temporary.replace(self.path)


def export_episode(args: argparse.Namespace, episode_index: int) -> int:
    (
        original,
        slowed,
        instruction,
        _slowed_episode_index,
        outcome_text,
    ) = comparison_spans(args, episode_index)
    print(f"Episode {episode_index}: {instruction}")
    print(f"Slowed outcome: {outcome_text}")
    print(
        f"Original: {original['frames']} frames, {original['duration']:.2f}s | "
        f"Slowed: {slowed['frames']} frames, {slowed['duration']:.2f}s | "
        f"Viewing at {args.speed:g}x"
    )

    command = ffmpeg_comparison_command(
        original,
        slowed,
        speed=args.speed,
        height=args.height,
    )
    assert args.output is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            *command,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(args.output),
        ],
        check=True,
    )
    print(f"Saved comparison: {args.output}")
    return 0


class SpanReader:
    """Incrementally decode one episode span while retaining its last frame."""

    def __init__(self, span: dict[str, Any], *, height: int):
        self.span = span
        self.frames = int(span["frames"])
        self.fps = float(span["fps"])
        source_width, source_height = probe_video_dimensions(span["path"])
        self.height = int(height)
        self.width = max(
            2,
            int(round(source_width * self.height / source_height)),
        )
        # yuv420-backed encoders require even dimensions; keeping the raw BGR
        # output even also makes side-by-side layout deterministic.
        self.width += self.width % 2
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{float(span['start']):.9f}",
                "-t",
                f"{float(span['duration']):.9f}",
                "-i",
                str(span["path"]),
                "-vf",
                f"scale={self.width}:{self.height}",
                "-an",
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert self.process.stdout is not None
        self.frame_bytes = self.width * self.height * 3
        self.decoded_index = -1
        self.last_frame: np.ndarray | None = None

    @property
    def duration(self) -> float:
        return self.frames / max(self.fps, 1.0e-9)

    def frame_at(self, media_time_s: float) -> np.ndarray:
        target = min(
            self.frames - 1,
            max(0, int(media_time_s * self.fps)),
        )
        while self.decoded_index < target:
            raw = self.process.stdout.read(self.frame_bytes)
            if len(raw) != self.frame_bytes:
                if self.last_frame is None:
                    raise RuntimeError(
                        f"Could not decode episode frame from {self.span['path']}"
                    )
                break
            self.last_frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                self.height,
                self.width,
                3,
            )
            self.decoded_index += 1
        assert self.last_frame is not None
        return self.last_frame

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
        if self.process.stdout is not None:
            self.process.stdout.close()
        try:
            self.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


def decode_final_frame(span: dict[str, Any], *, height: int) -> np.ndarray:
    """Decode the last recorded frame from a short window ending at the span boundary."""

    source_width, source_height = probe_video_dimensions(span["path"])
    width = max(2, int(round(source_width * height / source_height)))
    width += width % 2
    frame_bytes = width * height * 3
    duration = float(span["duration"])
    decode_duration = min(1.0, duration)
    seek_timestamp = float(span["start"]) + duration - decode_duration
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{seek_timestamp:.9f}",
            "-i",
            str(span["path"]),
            "-t",
            f"{decode_duration:.9f}",
            "-vf",
            f"scale={width}:{height}",
            "-an",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    decoded_frames, remainder = divmod(len(result.stdout), frame_bytes)
    if decoded_frames < 1 or remainder:
        raise RuntimeError(
            f"Could not decode final episode frame from {span['path']}: "
            f"expected whole {frame_bytes}-byte frames, got {len(result.stdout)} bytes"
        )
    return np.frombuffer(result.stdout[-frame_bytes:], dtype=np.uint8).reshape(
        height, width, 3
    )


def _missing_slowed_frame(*, height: int, message: str) -> np.ndarray:
    import cv2

    width = max(2, int(round(height * 4 / 3)))
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "NO SLOWED VIDEO",
        (24, max(70, height // 2 - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (80, 170, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        message[:52],
        (24, max(105, height // 2 + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return frame


def _fit_height(frame: np.ndarray, height: int) -> np.ndarray:
    import cv2

    width = max(2, int(round(frame.shape[1] * height / frame.shape[0])))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _labeled_frame(
    frame: np.ndarray,
    *,
    label: str,
    height: int,
) -> np.ndarray:
    import cv2

    fitted = _fit_height(frame, height)
    cv2.rectangle(fitted, (0, 0), (fitted.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        fitted,
        label,
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return fitted


def _status_canvas(
    *,
    height: int,
    message: str,
) -> np.ndarray:
    import cv2

    canvas = np.zeros((height + 76, max(960, height * 2), 3), dtype=np.uint8)
    lines = [message[i : i + 105] for i in range(0, len(message), 105)] or [""]
    for line_index, line in enumerate(lines[:8]):
        cv2.putText(
            canvas,
            line,
            (22, 54 + 34 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
    return canvas


def interactive_viewer(args: argparse.Namespace) -> int:
    """Keep one window alive while keyboard commands switch episode spans."""

    import cv2
    from PIL import Image, ImageTk
    import tkinter as tk

    review = (
        ReviewStore(
            args.review_file,
            original_root=args.original_root,
            slowed_root=args.slowed_root,
            outcomes_path=args.slowed_outcomes,
            delete_only=args.delete_only_review,
        )
        if args.review_file is not None
        else None
    )
    outcomes_by_source = {
        int(row["dataset"]["episode_index"]): row
        for row in load_outcome_records(args.slowed_outcomes)
    }

    window_name = "SO-101 original vs slowed curation"
    root = tk.Tk()
    root.title(window_name)
    image_label = tk.Label(root, background="black")
    image_label.pack(fill=tk.BOTH, expand=True)
    pending_keys: list[tuple[str, str]] = []
    window_open = True

    def queue_key(event: tk.Event) -> None:
        pending_keys.append((str(event.keysym), str(event.char)))

    def close_window() -> None:
        nonlocal window_open
        window_open = False

    root.bind("<KeyPress>", queue_key)
    root.protocol("WM_DELETE_WINDOW", close_window)
    root.focus_force()

    episode_index = args.episode
    if review is not None and args.resume_review:
        resumable = review.next_undecided(episode_index, include_current=True)
        if resumable is not None:
            episode_index = resumable
    speed = float(args.speed)
    final_frame_only = bool(args.final_frame_only)
    original_reader: SpanReader | None = None
    slowed_reader: SpanReader | None = None
    original_span: dict[str, Any] | None = None
    slowed_span: dict[str, Any] | None = None
    original_final: np.ndarray | None = None
    slowed_final: np.ndarray | None = None
    instruction = ""
    slowed_episode_index: int | None = None
    outcome_text = "OUTCOME UNKNOWN"
    load_error: str | None = None
    playing = True
    media_time_s = 0.0
    wall_anchor = time.perf_counter()

    def close_readers() -> None:
        nonlocal original_reader, slowed_reader
        if original_reader is not None:
            original_reader.close()
        if slowed_reader is not None:
            slowed_reader.close()
        original_reader = None
        slowed_reader = None

    def load_episode() -> None:
        nonlocal instruction, slowed_episode_index, outcome_text, load_error
        nonlocal playing, media_time_s, wall_anchor
        nonlocal original_reader, slowed_reader
        nonlocal original_span, slowed_span, original_final, slowed_final
        close_readers()
        load_error = None
        original_span = None
        slowed_span = None
        original_final = None
        slowed_final = None
        instruction = ""
        slowed_episode_index = None
        outcome_text = "OUTCOME UNKNOWN"
        try:
            outcome = outcomes_by_source.get(episode_index)
            if outcome is not None and outcome.get("recorded_sim_dataset") is None:
                original_span = video_span(args.original_root, episode_index, args.camera)
                reason = _quarantine_reason(outcome) or "quarantined"
                instruction = _outcome_instruction(outcome)
                outcome_text = f"QUARANTINED ({reason})"
            else:
                (
                    original_span,
                    slowed_span,
                    instruction,
                    slowed_episode_index,
                    outcome_text,
                ) = comparison_spans(args, episode_index)
            assert original_span is not None
            if final_frame_only:
                original_final = decode_final_frame(original_span, height=args.height)
                slowed_final = (
                    decode_final_frame(slowed_span, height=args.height)
                    if slowed_span is not None
                    else _missing_slowed_frame(height=args.height, message=outcome_text)
                )
            else:
                original_reader = SpanReader(original_span, height=args.height)
                if slowed_span is not None:
                    slowed_reader = SpanReader(slowed_span, height=args.height)
            slowed_label = (
                str(slowed_episode_index) if slowed_episode_index is not None else "none"
            )
            print(f"Source episode {episode_index} -> slowed episode {slowed_label}: {instruction}")
            if slowed_span is not None:
                print(
                    f"Original {original_span['frames']} frames | "
                    f"slowed {slowed_span['frames']} frames | viewing at {speed:g}x"
                )
            print(f"Slowed outcome: {outcome_text}")
            if review is not None:
                print(f"Review: {review.decision(episode_index)} | {review.summary()}")
                root.title(f"{window_name} — {review.summary()}")
        except (FileNotFoundError, IndexError, KeyError, RuntimeError, ValueError) as exc:
            close_readers()
            load_error = str(exc)
            print(f"Episode {episode_index}: {load_error}", file=sys.stderr)
        media_time_s = 0.0
        wall_anchor = time.perf_counter()
        playing = load_error is None and not final_frame_only

    def reload_episode() -> None:
        load_episode()

    reload_episode()
    print(
        f"Controls: K=keep, X/Delete={'delete' if args.delete_only_review else 're-record'}, "
        "U=clear, J=next unreviewed, "
        "N/Right=next, P/Left=previous, F=toggle final frames/video, "
        "R=replay, Space=pause, +/-=speed, Q/Esc=quit"
    )

    def move_relative(offset: int) -> None:
        nonlocal episode_index
        if review is None:
            episode_index = max(0, episode_index + offset)
            return
        indices = review.source_indices
        try:
            position = indices.index(episode_index)
        except ValueError:
            position = 0
        episode_index = indices[min(len(indices) - 1, max(0, position + offset))]

    def advance_to_undecided() -> bool:
        nonlocal episode_index
        if review is None:
            return False
        candidate = review.next_undecided(episode_index)
        if candidate is None:
            print(f"Review complete: {review.summary()}")
            return False
        episode_index = candidate
        return True

    try:
        while window_open:
            now = time.perf_counter()
            if playing:
                current_media_time = media_time_s + (now - wall_anchor) * speed
            else:
                current_media_time = media_time_s

            decision_text = review.decision(episode_index) if review is not None else ""
            if load_error is not None:
                canvas = _status_canvas(
                    height=args.height,
                    message=(
                        f"Source episode {episode_index}: {load_error}   "
                        f"{decision_text}  N/Right=next  P/Left=previous  R=retry  Q=quit"
                    ),
                )
            else:
                if final_frame_only:
                    assert original_final is not None and slowed_final is not None
                    current_media_time = 0.0
                    end_time = 0.0
                    left_raw = original_final
                    right_raw = slowed_final
                else:
                    assert original_reader is not None
                    end_time = max(
                        original_reader.duration,
                        slowed_reader.duration if slowed_reader is not None else 0.0,
                    )
                    if current_media_time >= end_time:
                        current_media_time = end_time
                        media_time_s = end_time
                        playing = False
                    left_raw = original_reader.frame_at(current_media_time)
                    right_raw = (
                        slowed_reader.frame_at(current_media_time)
                        if slowed_reader is not None
                        else _missing_slowed_frame(height=args.height, message=outcome_text)
                    )
                left = _labeled_frame(
                    left_raw,
                    label=f"ORIGINAL · source {episode_index}",
                    height=args.height,
                )
                right = _labeled_frame(
                    right_raw,
                    label=(
                        f"SLOWED · saved {slowed_episode_index if slowed_episode_index is not None else 'none'} · "
                        f"{outcome_text}"
                    ),
                    height=args.height,
                )
                video = np.hstack((left, right))
                status = np.zeros((104, video.shape[1], 3), dtype=np.uint8)
                state = "FINAL FRAMES" if final_frame_only else "PLAYING" if playing else "PAUSED/ENDED"
                cv2.putText(
                    status,
                    (
                        f"{state}  {speed:g}x  t={current_media_time:.1f}/"
                        f"{end_time:.1f}s  REVIEW={decision_text or 'OFF'}  {outcome_text}"
                    ),
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    status,
                    instruction[:120],
                    (12, 57),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    status,
                    f"K keep | X/Delete {'delete' if args.delete_only_review else 're-record'} | "
                    "U clear | J next unreviewed | N/P navigate | F final/video | Q quit",
                    (12, 86),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (180, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
                canvas = np.vstack((video, status))

            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            image_label.configure(image=photo)
            image_label.image = photo
            root.update_idletasks()
            root.update()
            if not pending_keys:
                time.sleep(0.01)
                continue
            keysym, character = pending_keys.pop(0)
            lowered = character.lower()

            if keysym == "Escape" or lowered == "q":
                if review is not None:
                    print(f"Saved review: {review.summary()} -> {review.path}")
                return 0
            if keysym == "Right" or lowered in ("n", "d"):
                move_relative(1)
                reload_episode()
                continue
            if keysym == "Left" or lowered in ("p", "a"):
                move_relative(-1)
                reload_episode()
                continue
            if lowered == "j" and review is not None:
                if advance_to_undecided():
                    reload_episode()
                continue
            if (lowered == "k" or lowered == "x" or keysym == "Delete") and review is not None:
                decision = (
                    "keep"
                    if lowered == "k"
                    else "delete"
                    if args.delete_only_review
                    else "rerecord"
                )
                try:
                    review.set_manual(episode_index, decision)
                    print(f"Marked source episode {episode_index}: {decision.upper()}")
                    print(review.summary())
                except (IndexError, ValueError) as exc:
                    print(f"Cannot mark episode {episode_index}: {exc}", file=sys.stderr)
                    continue
                if advance_to_undecided():
                    reload_episode()
                continue
            if lowered == "u" and review is not None:
                try:
                    review.clear_manual(episode_index)
                    print(f"Cleared source episode {episode_index} review decision")
                    reload_episode()
                except ValueError as exc:
                    print(f"Cannot clear episode {episode_index}: {exc}", file=sys.stderr)
                continue
            if lowered == "f":
                final_frame_only = not final_frame_only
                reload_episode()
                continue
            if lowered == "r":
                reload_episode()
                continue
            if keysym == "space" and load_error is None and not final_frame_only:
                if playing:
                    media_time_s = current_media_time
                    playing = False
                else:
                    wall_anchor = time.perf_counter()
                    playing = True
                continue
            if character in ("+", "=", "]"):
                media_time_s = current_media_time
                speed = min(64.0, speed * 1.25)
                wall_anchor = time.perf_counter()
                continue
            if character in ("-", "_", "["):
                media_time_s = current_media_time
                speed = max(0.05, speed / 1.25)
                wall_anchor = time.perf_counter()
                continue
        return 0
    finally:
        close_readers()
        try:
            root.destroy()
        except tk.TclError:
            pass


def main() -> int:
    args = parse_args()
    if args.episode < 0:
        raise ValueError(f"Expected a non-negative episode index, got {args.episode}.")
    if args.speed <= 0.0:
        raise ValueError(f"Expected --speed > 0, got {args.speed}.")
    if args.height < 2:
        raise ValueError(f"Expected --height >= 2, got {args.height}.")
    if args.resume_review and args.review_file is None:
        raise ValueError("--resume-review requires --review-file.")
    if args.delete_only_review and args.review_file is None:
        raise ValueError("--delete-only-review requires --review-file.")
    if args.output is not None and args.review_file is not None:
        raise ValueError("--review-file is available only in the interactive viewer.")
    required_executables = {"ffmpeg", "ffprobe"}
    for executable in sorted(required_executables):
        if shutil.which(executable) is None:
            raise FileNotFoundError(f"Required executable is not on PATH: {executable}")

    if args.output is not None:
        return export_episode(args, args.episode)
    return interactive_viewer(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, IndexError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
