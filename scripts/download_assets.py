#!/usr/bin/env python3
"""Download the SO-101 Bench USD assets (robot, bin, room scans, backgrounds and objects).

The assets are too large for git; they live in the Hugging Face dataset
huggingface.co/datasets/5hadytru/so101_bench_assets and unpack into
source/so101_bench/so101_bench/assets/usd/.

    python scripts/download_assets.py            # download, verify and unpack the public bundle
    python scripts/download_assets.py --force    # re-download even if already present
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "source/so101_bench/so101_bench/assets"
REPO = "https://huggingface.co/datasets/5hadytru/so101_bench_assets/resolve/main"
BUNDLE = "so101_bench_assets_public_v2.tar.gz"
SHA256 = "672a19e7de218579af50c99c8e53c9fe4757bb927008e2b5cf8c806e198773b3"
MARKER = ASSETS / "usd/.so101_bench_assets_public_v2"


def download(url: str, path: Path) -> None:
    with urllib.request.urlopen(url) as response, path.open("wb") as out:
        total, done = int(response.headers.get("Content-Length", 0)), 0
        while chunk := response.read(1 << 22):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Download again even if the assets are present.")
    args = parser.parse_args()
    if MARKER.exists() and not args.force:
        print(f"Assets already installed ({MARKER.parent}). Use --force to re-download.")
        return
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / BUNDLE
        print(f"Downloading {BUNDLE} (about 2 GB) ...")
        download(f"{REPO}/{BUNDLE}", bundle)
        digest = hashlib.sha256()
        with bundle.open("rb") as f:
            while chunk := f.read(1 << 22):
                digest.update(chunk)
        if digest.hexdigest() != SHA256:
            raise SystemExit(f"Checksum mismatch for {BUNDLE}: {digest.hexdigest()}")
        print(f"Unpacking into {ASSETS} ...")
        with tarfile.open(bundle) as tar:
            members = [m for m in tar.getmembers() if not (m.name.startswith("/") or ".." in Path(m.name).parts)]
            tar.extractall(ASSETS, members=members)
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(BUNDLE + "\n")
    print("Done.")


if __name__ == "__main__":
    main()
