"""Hardware-free regression checks for wall-clock clips and PyRUA annotations."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from so101_real_recording import AgentMessageRecorder, ClipRecorder, annotate_recording, annotation_overlays


class RealRecordingTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.agent = self.root / "episode" / "agent"
        (self.agent / "pyrua").mkdir(parents=True)

    def write_calls(self, rows):
        (self.agent / "pyrua" / "calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def metadata(self):
        return dict(policy="sol", instruction="Put Bob's cube: 50% [blue], in the bin.",
                    episode_dir=str(self.agent.parent), task_family="bin_4", trial_id=7,
                    frames=300, fps=30, started_at=1000,
                    controllers=[{"t": 0, "label": "SOL thinking"}, {"t": 4, "label": "G0.5"},
                                 {"t": 8, "label": "SOL"}])

    def test_annotations_use_wall_time_through_query_stalls_and_resumed_clips(self):
        # These calls use a handful of control steps but seconds of wall time.
        # They must not be placed at env_steps / 30 (the sim annotator's clock).
        self.write_calls([
            dict(tool="sol_verdict", started_at=999, kwargs={"verdict": "failed_attempt", "object": "cube",
                 "attempts": 2, "in_bin": 1, "decision": "continue"}),
            dict(tool="vla_run", started_at=1004, elapsed_s=4, env_steps=2, stateful=True),
            dict(tool="sol_verdict", started_at=1007.5, kwargs={"verdict": "stuck", "object": "cube",
                 "decision": "takeover", "in_bin": 1, "stuck_s": 6, "reason": "stuck"}),
        ])
        (self.agent / "annotation_messages.jsonl").write_text(json.dumps({"at": 1006, "text": "I will take over."}) + "\n")
        overlays = annotation_overlays(self.metadata())
        caption = next(o for o in overlays if "I will take over" in o["text"])
        self.assertEqual(caption["start"], 6)
        flash = next(o for o in overlays if "STUCK" in o["text"])
        self.assertEqual(flash["start"], 7.5)
        takeover = next(o for o in overlays if o["text"] == "SOL TAKEOVER")
        self.assertEqual(takeover["start"], 8)
        hud = next(o for o in overlays if "cube x2" in o["text"])
        self.assertEqual(hud["start"], 0)  # tally from before a resumed clip survives
        self.assertTrue(all(0 <= o["start"] < o["end"] <= 10 for o in overlays))

    def test_caption_log_preserves_partial_lines_and_ignores_nonpublic_events(self):
        recorder = AgentMessageRecorder(self.agent)
        message = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "Now  lift."}}).encode()
        recorder.source.write_bytes(message[:30])
        output = io.StringIO()
        recorder._poll(output)
        self.assertEqual(recorder._offset, 0)
        with recorder.source.open("ab") as handle:
            handle.write(message[30:] + b"\n")
            handle.write(b'{"type":"item.completed","item":{"type":"reasoning","text":"private"}}\n')
        with patch("so101_real_recording.time.time", return_value=1234):
            recorder._poll(output)
            recorder._poll(output)
        self.assertEqual(output.getvalue(), '{"at": 1234, "text": "Now lift."}\n')

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_raw_pixels_speed_and_full_encode_then_score_refresh(self):
        class Camera:
            def __init__(self, color, stall=0):
                self.image = np.full((480, 640, 3), color, dtype=np.uint8)
                self.stall = stall

            def latest_bgr(self):
                if self.stall:
                    time.sleep(self.stall)
                    self.stall = 0
                return self.image

        metadata = self.metadata()
        # Avoid unrelated clock anchors from the overlay unit test.
        for key in ("frames", "fps", "started_at", "controllers"):
            metadata.pop(key)
        clip = ClipRecorder(self.root / "recordings", "sol_trial07_bin_4_id", (Camera(80, .15), Camera(140)),
                            label_fn=lambda: "G0.5", metadata=metadata)
        time.sleep(.42)
        clip.stop()
        with patch.object(clip, "annotate_async"):
            clip.keep()
        meta = json.loads(clip.sidecar.read_text())
        self.assertLess(abs(clip.frames / 30 - meta["wall_duration_s"]), 1 / 30 + .02)
        raw_hash = hashlib.sha256(clip.path.read_bytes()).digest()
        capture = cv2.VideoCapture(str(clip.path))
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), clip.frames)
        capture.set(cv2.CAP_PROP_POS_FRAMES, clip.frames - 1)
        ok, raw_frame = capture.read()
        capture.release()
        self.assertTrue(ok)
        self.assertLess(raw_frame[:100, :640].std(), 3)  # no burned-in banner or text
        self.assertLess(np.abs(raw_frame[:100, :640].astype(float) - 80).mean(), 5)
        self.assertLess(np.abs(raw_frame[:, 640:].astype(float) - 140).mean(), 5)
        now = meta["started_at"]
        self.write_calls([dict(tool="sol_verdict", started_at=now, kwargs={"verdict": "placed", "in_bin": 1})])
        (self.agent / "annotation_messages.jsonl").write_text(json.dumps({"at": now, "text": "It's placed: 100%."}) + "\n")
        target = annotate_recording(clip.sidecar)
        annotated = cv2.VideoCapture(str(target))
        self.assertEqual(int(annotated.get(cv2.CAP_PROP_FRAME_COUNT)), clip.frames)
        self.assertEqual(annotated.get(cv2.CAP_PROP_FPS), 30)
        annotated.set(cv2.CAP_PROP_POS_FRAMES, clip.frames - 1)
        ok, annotated_frame = annotated.read()
        annotated.release()
        self.assertTrue(ok)
        self.assertGreater(annotated_frame[:110].std(), raw_frame[:110].std() + 10)
        (self.agent.parent / "episode.json").write_text(json.dumps({"result": {"success": True}}))
        self.assertTrue(any("FINAL SUCCESS" in o["text"] for o in annotation_overlays(meta)))
        annotate_recording(clip.sidecar)
        self.assertEqual(hashlib.sha256(clip.path.read_bytes()).digest(), raw_hash)

    def test_discard_and_clip_numbering(self):
        class Camera:
            def latest_bgr(self):
                return None  # a camera dropout is still a real-time black frame

        directory = self.root / "recordings"
        directory.mkdir()
        (directory / "sol_clip03_old.mp4").touch()
        clip = ClipRecorder(directory, "sol", (Camera(), Camera()), metadata={"policy": "sol"})
        clip.stop()
        self.assertIn("_clip04_", clip.path.name)
        clip.discard()
        self.assertFalse(clip.partial.exists())
        self.assertFalse(clip.path.exists())
        self.assertFalse(clip.annotated_path.exists())
        self.assertFalse(clip.sidecar.exists())


if __name__ == "__main__":
    unittest.main()
