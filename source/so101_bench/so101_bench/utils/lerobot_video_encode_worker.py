"""Clean-process video encoder used by SO-101 Isaac teleoperation."""

from __future__ import annotations

import json
import sys

from lerobot.datasets.video_utils import encode_video_frames


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Expected one JSON encoding request argument.")
    request = json.loads(sys.argv[1])
    encode_video_frames(**request)


if __name__ == "__main__":
    main()
