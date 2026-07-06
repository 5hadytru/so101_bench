#!/bin/bash
# Convenience launcher for the SO-101 Bench sim/eval container.
#
#   ./docker/run.sh                 # interactive shell in the container
#   ./docker/run.sh <command...>    # run a command, e.g. an eval
#
# --network=host lets the eval reach a GR00T policy server on localhost:5555
# (run the server natively or via docker/gr00t-server). GUI needs a local X server.
set -e

IMAGE="${SO101_IMAGE:-so101-bench:latest}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Allow the container's root user to talk to the host X server (for the GUI viewer).
xhost +local:root >/dev/null 2>&1 || true

# Persist Omniverse/Isaac caches + HF downloads across runs.
mkdir -p ~/docker/isaac-sim/cache/{kit,ov,pip,glcache,computecache} \
         ~/docker/isaac-sim/{logs,data,documents} ~/.cache/huggingface

docker run --name so101_bench -it --rm --gpus all \
  --network=host \
  -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
  -e DISPLAY \
  -e "HF_TOKEN=${HF_TOKEN:-}" \
  -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
  -v ~/docker/isaac-sim/cache/kit:/workspace/isaaclab/_isaac_sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/.cache/huggingface:/root/.cache/huggingface:rw \
  -v "$REPO_ROOT/tasks:/workspace/so101_bench/tasks:rw" \
  -v "$REPO_ROOT/data:/workspace/so101_bench/data:rw" \
  "$IMAGE" "$@"
