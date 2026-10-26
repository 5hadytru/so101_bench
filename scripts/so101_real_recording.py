"""Wall-clock camera clips and full PyRUA annotations for the real runner.

The raw MP4 is never drawn on. Its JSON sidecar records controller changes and
clock anchors; the agent's call ledger supplies wall-clock verdict/takeover times.
Keeping a clip starts a background encode. When the episode finishes writing its
ledger, and again after human scoring, the runner refreshes the annotated copy.
Both copies have identical frame counts and retain thinking/camera pauses.
Rebuild a kept clip (without hardware) with:

    python scripts/so101_real_recording.py path/to/raw/clip.recording.json
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import tempfile
import textwrap
import threading
import time

import cv2
import numpy as np

from annotate_handoff_videos import COLORS, FONT, VERDICT_COLORS, verdict_text


FPS = 30.0
FRAME_SIZE = (640, 480)
_ENCODE_LOCK = threading.Lock()  # one encoder at a time, alongside the robot/policy


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a live writer may not have finished its final line yet
    return rows


class AgentMessageRecorder:
    """Timestamp public Codex messages as they arrive, including during thinking.

    The frozen harness writes Codex stdout directly to disk without timestamps.
    Poll that stream without modifying the harness, preserving partial JSON lines.
    """

    def __init__(self, agent_dir: Path):
        self.source = agent_dir / "pyrua" / "codex_events.jsonl"
        self.path = agent_dir / "annotation_messages.jsonl"
        self._offset = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="agent-captions", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _poll(self, output) -> None:
        if not self.source.exists():
            return
        with self.source.open("rb") as source:
            source.seek(self._offset)
            while True:
                line = source.readline()
                if not line.endswith(b"\n"):
                    break
                self._offset = source.tell()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                item = event.get("item") or {}
                if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                    message = " ".join((item.get("text") or "").split())
                    if message:
                        output.write(json.dumps({"at": time.time(), "text": message}) + "\n")
                        output.flush()

    def _run(self) -> None:
        try:
            with self.path.open("w") as output:
                while not self._stop.is_set():
                    self._poll(output)
                    self._stop.wait(0.05)
                self._poll(output)
        except Exception as exc:  # recording must never interrupt arm control
            print(f"[RECORDING]: caption log failed: {exc}", flush=True)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


def annotation_overlays(meta: dict) -> list[dict]:
    """Build overlays on the clip clock, never the accelerated control-step clock."""
    duration = meta["frames"] / meta["fps"]
    anchor = meta["started_at"]
    agent_dir = (Path(meta["episode_dir"]) / "agent"
                 if meta.get("episode_dir") and meta["policy"] in ("sol", "argon") else None)
    calls = read_jsonl(agent_dir / "pyrua" / "calls.jsonl") if agent_dir else []
    messages = read_jsonl(agent_dir / "annotation_messages.jsonl") if agent_dir else []
    overlays = []

    def add(text, start, end, x, y, size, color="0x000000"):
        start, end = max(0.0, start), min(duration, end)
        if text and end > start:
            overlays.append(dict(text=text, start=start, end=end, x=x, y=y, size=size, color=color))

    # Controller samples reflect the actual driver, even while an RPC stalls or
    # cameras reconnect. Verdicts add the Sol approach/placement/takeover phase.
    changes = [(row["t"], "controller", row["label"]) for row in meta["controllers"]]
    verdicts = [c for c in calls if c.get("tool") == "sol_verdict"]
    takeover_label = "SOL TAKEOVER (SLOW)" if any(c.get("tool") == "takeover_slow" for c in calls) else "SOL TAKEOVER"
    for call in verdicts:
        kw = call.get("kwargs") or {}
        decision = kw.get("decision")
        phase = (takeover_label if decision == "takeover" else "SOL" if decision == "restart_policy"
                 else "SOL APPROACH" if decision == "first_handoff"
                 else "SOL PLACING" if kw.get("verdict") == "lifted" else None)
        if phase:
            changes.append((call["started_at"] - anchor, "phase", phase))
    changes.sort(key=lambda row: row[0])
    controller, phase = meta["policy"].upper(), "SOL"
    for i, (at, kind, value) in enumerate(changes):
        if kind == "phase":
            phase = value
        else:
            controller = value or "HOLD"
        end = changes[i + 1][0] if i + 1 < len(changes) else duration
        label = phase if controller == "SOL" else controller
        if "thinking" in label:
            label = label.replace(" thinking", " THINKING - ARM HOLDING")
        if label == "finished":
            label = "AGENT FINISHED - ARM HOLDING"
        color = COLORS.get(label, "0x6b7280" if "HOLD" in label or "THINKING" in label else "0x3b82f6")
        add(label, at, end, "12", "8", 24, color)

    fails, in_bin = {}, 0
    bin_task = "bin" in meta.get("task_family", "")
    for i, call in enumerate(verdicts):
        kw = call.get("kwargs") or {}
        at = call["started_at"] - anchor
        following = verdicts[i + 1]["started_at"] - anchor if i + 1 < len(verdicts) else duration
        color = "0xb91c1c" if kw.get("decision") == "takeover" else VERDICT_COLORS.get(kw.get("verdict"), "0x2563eb")
        text = verdict_text(kw).replace("Sol: None", "Sol: monitoring G0.5")
        if meta["policy"] != "sol":
            text = text.replace("Sol", meta["policy"].capitalize())
        add(text, at, min(at + 1.5, following), "12", "83", 22, color)
        if kw.get("verdict") == "failed_attempt" and kw.get("object"):
            fails[kw["object"]] = int(kw.get("attempts") or 0)
        if kw.get("in_bin") is not None:
            in_bin = int(kw["in_bin"])
        tally = [f"{meta['policy'].capitalize()}-reported in bin {in_bin}"] if bin_task else []
        if fails:
            tally.append("failed: " + ", ".join(f"{name} x{n}" for name, n in fails.items()))
        add(" | ".join(tally), at, following, "12", "h-30", 18)

    for i, message in enumerate(messages):
        at = message["at"] - anchor
        following = messages[i + 1]["at"] - anchor if i + 1 < len(messages) else duration
        # Keep the whole message in the sidecar; paginate long captions rather
        # than silently dropping everything beyond two lines.
        lines = textwrap.wrap(f"{meta['policy'].capitalize()}: {message['text']}", width=105)
        pages = [lines[j:j + 2] for j in range(0, len(lines), 2)]
        page_s = max(3.0, len(message["text"].split()) / 3.0 / max(1, len(pages)))
        for j, page in enumerate(pages):
            for k, line in enumerate(page):
                add(line, at + j * page_s, min(following, at + (j + 1) * page_s),
                    "12", f"h-{90 - 25 * k}", 18, "0x1e3a8a")

    verdict = "UNSCORED"
    if meta.get("episode_dir"):
        path = Path(meta["episode_dir"]) / "episode.json"
        episode = json.loads(path.read_text()) if path.exists() else {}
        result = episode.get("result")
        if result:
            verdict = "SUCCESS" if result.get("success") else "FAIL"
            if not result.get("success") and result.get("fail_type"):
                verdict += " (" + ", ".join(result["fail_type"]) + ")"
            if result.get("timed_out"):
                verdict += " (TIMED OUT)"
            objects = result.get("objects") or []
            if objects and all(isinstance(obj, dict) for obj in objects):
                verdict += f" | in bin {sum(bool(obj.get('success')) for obj in objects)}/{len(objects)}"
        elif episode.get("end_reason"):
            verdict += f" ({episode['end_reason']})"
    title = f"{meta['policy'].upper()} | trial {meta.get('trial_id', '-')}"
    if agent_dir:
        title += " | PyRUA v9.6"
    title += f" | FINAL {verdict}"
    add(title, 0, duration, "w-tw-12", "14", 16)
    add(meta["instruction"], 0, duration, "12", "51", 18)
    return overlays


def annotate_recording(sidecar: Path) -> Path:
    """Encode an annotated sibling; leave raw bytes and duration untouched."""
    with _ENCODE_LOCK:
        meta = json.loads(sidecar.read_text())
        target = Path(meta["annotated_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".building.mp4")
        with tempfile.TemporaryDirectory(prefix="so101-annotation-") as scratch:
            filters = ["drawbox=x=0:y=0:w=iw:h=44:color=black@0.55:t=fill"]
            for i, overlay in enumerate(annotation_overlays(meta)):
                text_path = Path(scratch) / f"text{i}.txt"
                text_path.write_text(overlay["text"])
                filters.append(
                    f"drawtext=fontfile={FONT}:textfile={text_path}:expansion=none:"
                    f"x={overlay['x']}:y={overlay['y']}:fontsize={overlay['size']}:fontcolor=white:"
                    f"box=1:boxcolor={overlay['color']}@0.85:boxborderw=5:"
                    f"enable='gte(t,{overlay['start']:.6f})*lt(t,{overlay['end']:.6f})'"
                )
            script = Path(scratch) / "overlays.filter"
            script.write_text(",".join(filters))
            try:
                subprocess.run([
                    "ffmpeg", "-v", "error", "-y", "-threads", "2", "-i", meta["raw_path"],
                    "-filter_script:v", str(script), "-c:v", "libx264", "-threads", "2", "-crf", "23",
                    "-preset", "veryfast", "-an", "-movflags", "+faststart", str(temporary),
                ], check=True, capture_output=True, text=True)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return target


class ClipRecorder:
    """Capture unmodified overhead | wrist at wall-clock speed on its own thread."""

    def __init__(self, directory: Path, stem: str, cameras, label_fn=None, *, metadata=None):
        directory = directory.resolve()
        raw = directory / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        # Include partial and old-format files; never overwrite an earlier clip.
        numbers = []
        for parent in (directory, raw):
            for path in parent.glob(f"{stem}_clip*.mp4"):
                suffix = path.name[len(stem) + 5:].split("_", 1)[0]
                if suffix.isdigit():
                    numbers.append(int(suffix))
        clip = max(numbers, default=0) + 1
        name = f"{stem}_clip{clip:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        self.path = raw / name
        self.partial = self.path.with_suffix(".partial.mp4")
        self.annotated_path = directory / "annotated" / name
        self.sidecar = self.path.with_suffix(".recording.json")
        self.cameras, self.label_fn = cameras, label_fn
        self.frames = 0
        self.metadata = dict(metadata or {})
        self.controllers = []
        self.error = None
        self.writer = cv2.VideoWriter(str(self.partial), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                      (FRAME_SIZE[0] * 2, FRAME_SIZE[1]))
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {self.partial}")
        self._stop = threading.Event()
        self._annotation_lock = threading.Lock()
        self._annotation_dirty = False
        self._annotation_running = False
        self.t0 = time.monotonic()
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, name="clip", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        black = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), np.uint8)
        previous = np.hstack((black, black))
        try:
            while not self._stop.is_set():
                # Missed deadlines duplicate the previous image rather than
                # compressing a disk/camera stall out of the wall-clock video.
                due = int((time.monotonic() - self.t0) * FPS)
                while self.frames < due:
                    self.writer.write(previous)
                    self.frames += 1
                panes = []
                for camera in self.cameras:
                    frame = camera.latest_bgr()
                    if frame is None:
                        frame = black
                    elif (frame.shape[1], frame.shape[0]) != FRAME_SIZE:
                        frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
                    panes.append(frame)
                frame = np.hstack(panes)
                label = self.label_fn() if self.label_fn else self.metadata.get("policy", "HOLD").upper()
                if not self.controllers or label != self.controllers[-1]["label"]:
                    self.controllers.append({"t": self.frames / FPS, "label": label})
                self.writer.write(frame)  # absolutely no text, banner, layout or init thumbnails
                previous = frame
                self.frames += 1
                self._stop.wait(max(0.0, self.t0 + self.frames / FPS - time.monotonic()))
            target_frames = math.ceil((self.stopped_at - self.t0) * FPS)
            while self.frames < target_frames:
                self.writer.write(previous)
                self.frames += 1
        except Exception as exc:
            self.error = repr(exc)
            print(f"[RECORDING]: capture failed: {exc}", flush=True)
        finally:
            self.writer.release()

    def stop(self) -> None:
        self.stopped_at = time.monotonic()
        self._stop.set()
        self._thread.join()  # never release the writer while its thread is writing

    def keep(self) -> Path:
        self.partial.replace(self.path)
        meta = dict(self.metadata, raw_path=str(self.path), annotated_path=str(self.annotated_path),
                    fps=FPS, frames=self.frames, started_at=self.started_at,
                    wall_duration_s=self.stopped_at - self.t0, controllers=self.controllers, capture_error=self.error)
        self.sidecar.write_text(json.dumps(meta, indent=2) + "\n")
        self.annotate_async()
        return self.path

    def annotate_async(self) -> None:
        if not self.sidecar.exists():
            return
        with self._annotation_lock:
            self._annotation_dirty = True
            if self._annotation_running:
                return
            self._annotation_running = True
        # Non-daemon: on normal shutdown finish encodes for saved clips.
        threading.Thread(target=self._annotate, name="clip-annotation", daemon=False).start()

    def _annotate(self) -> None:
        while True:
            with self._annotation_lock:
                if not self._annotation_dirty:
                    self._annotation_running = False
                    return
                self._annotation_dirty = False
            try:
                target = annotate_recording(self.sidecar)
                print(f"[RECORDING]: annotated video saved: {target}", flush=True)
            except Exception as exc:
                detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
                print(f"[RECORDING]: annotation failed (raw kept): {detail}\n"
                      f"Retry: {Path(__file__)} {self.sidecar}", flush=True)

    def discard(self) -> None:
        self.partial.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecars", nargs="+", type=Path)
    for sidecar in parser.parse_args().sidecars:
        print(annotate_recording(sidecar))
