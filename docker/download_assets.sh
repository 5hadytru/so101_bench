#!/bin/bash
# Download and extract the SO-101 Bench USD assets (~430 MB) from Hugging Face.
# Idempotent: does nothing if the assets are already in place.
set -e

ASSET_DIR="${SO101_ASSET_DIR:-/workspace/so101_bench/source/so101_bench/so101_bench/assets}"
MARKER="$ASSET_DIR/usd/room_scan.usdc"
HF_REPO="5hadytru/so101_bench_assets"
TARBALL="so101_bench_usd_assets.tar.gz"

if [ -f "$MARKER" ]; then
  echo "[assets] USD assets already present ($MARKER) — skipping download."
  exit 0
fi

echo "[assets] Downloading USD assets (~430 MB) from ${HF_REPO} ..."
TMP="$(mktemp -d)"
huggingface-cli download "$HF_REPO" "$TARBALL" \
  --repo-type dataset --local-dir "$TMP"

echo "[assets] Extracting into ${ASSET_DIR} ..."
mkdir -p "$ASSET_DIR"
tar -xzf "$TMP/$TARBALL" -C "$ASSET_DIR"
rm -rf "$TMP"

if [ -f "$MARKER" ]; then
  echo "[assets] Done."
else
  echo "[assets] ERROR: extraction did not produce $MARKER" >&2
  exit 1
fi
