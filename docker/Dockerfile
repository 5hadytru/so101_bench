# syntax=docker/dockerfile:1
#
# SO-101 Bench — Isaac Lab simulation + GR00T evaluation container.
#
# Reference: NVIDIA Sim-to-Real-SO-101-Workshop (docker/sim/Dockerfile).
# The base image bundles Isaac Sim + Isaac Lab at /workspace/isaaclab.
#
# Build (from repo root):
#   docker build -t so101-bench:latest -f docker/Dockerfile .
# Optionally bake the ~430 MB USD assets into the image:
#   docker build --build-arg DOWNLOAD_ASSETS=true -t so101-bench:latest -f docker/Dockerfile .

ARG ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2
FROM ${ISAAC_LAB_IMAGE}

# Isaac Lab lives at /workspace/isaaclab in the base image.
ENV ISAACLAB_PATH=/workspace/isaaclab
ENV PYTHON=/workspace/isaaclab/_isaac_sim/python.sh
ENV PATH="/workspace/isaaclab/_isaac_sim/kit/python/bin:$PATH"
# hf_transfer = faster asset/model downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# ---------------------------------------------------------------------------
# System deps: ffmpeg (LeRobot dataset video encode), X11 libs (GUI viewer).
# ---------------------------------------------------------------------------
RUN curl --proto "=https" --tlsv1.2 -sSf -L -o /tmp/ffmpeg.tar.xz \
      https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-lgpl-shared-7.1.tar.xz && \
    tar -xf /tmp/ffmpeg.tar.xz -C /usr/local --strip-components=1 && \
    ldconfig && rm /tmp/ffmpeg.tar.xz

# X11 libs are only needed for the GUI viewer and are already present in the
# Isaac Lab base image; keep this best-effort so a build behind a restricted apt
# mirror (or IPv6-only DNS) still succeeds. git/curl are also present in the base.
RUN (apt-get update && apt-get install -y --no-install-recommends \
      libx11-6 libxcursor1 libxrandr2 libxi6 libxinerama1 \
      libxkbcommon0 libxkbcommon-x11-0 git curl ca-certificates \
      && rm -rf /var/lib/apt/lists/*) \
    || echo "[docker] WARNING: apt unavailable; relying on base-image X11/git/curl."

# ---------------------------------------------------------------------------
# Version constraints. CRITICAL: nothing may upgrade numpy past 2.0 or swap
# torch — Isaac Sim's compiled synthetic-data / camera bindings are built
# against numpy 1.26 and break with `unknown dtype, kind=f, size=0` otherwise.
# (This mirrors the numpy 2.x regression we hit installing lerobot bare.)
# ---------------------------------------------------------------------------
RUN printf '%s\n' \
      "numpy==1.26.4" \
      "torch==2.7.0" \
      "torchvision==0.22.0" \
      "packaging==23.0" \
      "lxml==4.9.4" \
      "imageio==2.37.0" \
      > /tmp/constraints.txt

# ---------------------------------------------------------------------------
# LeRobot (needed for `--record_dataset`). Install --no-deps, then add only
# the extra deps under the constraints so numpy/torch stay pinned.
# ---------------------------------------------------------------------------
WORKDIR /workspace
RUN git clone https://github.com/huggingface/lerobot.git && \
    cd lerobot && git checkout e670ac5daf9b76 && \
    $PYTHON -m pip install --no-deps -e .

RUN $PYTHON -m pip install -c /tmp/constraints.txt \
      "datasets>=4.0.0,<4.2.0" \
      "diffusers>=0.27.2,<0.36.0" \
      "huggingface-hub[hf-transfer,cli]>=0.34.2,<0.36.0" \
      "accelerate>=1.10.0,<2.0.0" \
      "cmake>=3.29.0.1,<4.2.0" \
      "av>=15.0.0,<16.0.0" \
      "jsonlines>=4.0.0,<5.0.0" \
      "pynput>=1.7.7,<1.9.0" \
      "pyserial>=3.5,<4.0" \
      "wandb>=0.20.0,<0.22.0" \
      "torchcodec>=0.2.1,<0.6.0" \
      "draccus==0.10.0" \
      "deepdiff>=7.0.1,<9.0.0" \
      "feetech-servo-sdk>=1.0.0,<2.0.0"

# so101_bench extension runtime deps (also declared in setup.py)
RUN $PYTHON -m pip install -c /tmp/constraints.txt psutil msgpack pyzmq

# ---------------------------------------------------------------------------
# Copy the repo and install the extension (editable).
# ---------------------------------------------------------------------------
WORKDIR /workspace/so101_bench
COPY . /workspace/so101_bench
RUN $PYTHON -m pip install -c /tmp/constraints.txt -e source/so101_bench

# Optionally bake the USD assets into the image (else fetched at first run).
ARG DOWNLOAD_ASSETS=false
RUN chmod +x docker/entrypoint.sh docker/download_assets.sh && \
    if [ "$DOWNLOAD_ASSETS" = "true" ]; then docker/download_assets.sh; fi

# Route bytecode cache to a writable dir.
RUN mkdir -p /tmp/pycache && chmod 1777 /tmp/pycache
ENV PYTHONPYCACHEPREFIX=/tmp/pycache

ENTRYPOINT ["/workspace/so101_bench/docker/entrypoint.sh"]
CMD ["bash"]
