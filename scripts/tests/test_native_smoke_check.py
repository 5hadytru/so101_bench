import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_smoke_check import validate


class NativeSmokeCheck(unittest.TestCase):
    def test_masked_startup_exception_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                validate(folder)

    def test_requires_motion_both_cameras_and_video(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            proof = dict(ok=True, native_steps=46,
                         rgb_shapes=dict(overhead=[480, 640, 3], wrist=[480, 640, 3]))
            (p/'smoke.json').write_text(json.dumps(proof))
            with self.assertRaises(FileNotFoundError):
                validate(p)
            (p/'rollout_overhead_wrist.mp4').write_bytes(b'camera')
            self.assertEqual(validate(p), proof)
            proof['native_steps'] = 45
            (p/'smoke.json').write_text(json.dumps(proof))
            with self.assertRaises(AssertionError):
                validate(p)


if __name__ == '__main__':
    unittest.main()
