#!/usr/bin/env python3
"""Lock the wrist camera's white balance (auto WB drifts orange over the black table).

Hold a sheet of plain white paper ~15 cm in front of the wrist camera, filling the frame, under the
lighting you run episodes in. The script turns auto white balance off, sweeps the colour temperature,
picks the one where the paper comes out most neutral (R ~= B), and saves the camera controls to
assets/real_robot/<robot_id>_wrist_camera_controls.json. so101_real_policy_runner.py applies that
file every time it opens the wrist camera (start and USB reconnects). Quit the runner first: the
camera can only stream to one program.

    /home/truman/env_isaaclab_51/bin/python scripts/so101_wrist_white_balance.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "source" / "so101_bench" / "so101_bench" / "assets" / "real_robot"
# The settings the camera keeps in its own flash (it restores them after a USB reconnect) and that MMRV round 1
# ran with. They put the black table at sim's wrist brightness (median ~41 vs sim 40; factory defaults give
# ~115, washed-out grey). Gamma/contrast cannot separate the black jaws from the black table (their contrast
# stays ~6-10 grey levels at any setting, 2026-10-20 sweep). White balance is added by the sweep.
BASE_CONTROLS = {
    "brightness": 5, "contrast": 32, "saturation": 64, "hue": 0, "gamma": 100, "gain": 0, "sharpness": 3,
    "backlight_compensation": 0, "power_line_frequency": 2, "auto_exposure": 3, "exposure_dynamic_framerate": 1,
}


def controls_path(robot_id: str) -> Path:
    return ASSET_DIR / f"{robot_id}_wrist_camera_controls.json"


def apply_controls(device: str, controls: dict[str, int]) -> None:
    # auto_exposure first: exposure-related controls are rejected while it is in the wrong mode.
    ordered = sorted(controls.items(), key=lambda kv: kv[0] != "auto_exposure")
    for key, value in ordered:
        subprocess.run(["v4l2-ctl", "-d", device, "-c", f"{key}={value}"], check=False, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--robot_id", default="follower_arm_1")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not capture.isOpened() or not capture.read()[0]:
        sys.exit(f"Cannot read {args.device} (is the runner still open?).")
    input("Hold white paper ~15 cm in front of the wrist camera, filling the frame, then press Enter...")
    apply_controls(args.device, {**BASE_CONTROLS, "white_balance_automatic": 0})

    def mean_rgb() -> np.ndarray:
        for _ in range(12):  # let exposure and the new setting settle; drop buffered frames
            capture.read()
        ok, frame = capture.read()
        centre = frame[120:360, 160:480].reshape(-1, 3).astype(float)
        return centre.mean(axis=0)[::-1]  # RGB

    results = []
    for temperature in range(2800, 6501, 200):
        subprocess.run(["v4l2-ctl", "-d", args.device, "-c", f"white_balance_temperature={temperature}"], check=False)
        time.sleep(0.2)
        r, g, b = mean_rgb()
        results.append((abs(np.log(r / b)), temperature, r, g, b))
        print(f"  {temperature} K  R {r:5.1f}  G {g:5.1f}  B {b:5.1f}  R/B {r / b:4.2f}")
    _, best, r, g, b = min(results)
    capture.release()
    controls = {**BASE_CONTROLS, "white_balance_automatic": 0, "white_balance_temperature": best}
    apply_controls(args.device, controls)
    path = controls_path(args.robot_id)
    path.write_text(json.dumps({"device_hint": args.device, "controls": controls,
                                "measured": {"rgb_on_white": [round(r, 1), round(g, 1), round(b, 1)]},
                                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2) + "\n")
    print(f"Locked white balance at {best} K (R/B {r / b:.2f} on the paper). Saved {path}.")


if __name__ == "__main__":
    main()
