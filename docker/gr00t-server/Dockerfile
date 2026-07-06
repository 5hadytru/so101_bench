# syntax=docker/dockerfile:1
#
# GR00T-N1.6 policy server (reference).
#
# Serves the fine-tuned checkpoint over TCP :5555 for scripts/groot_eval.py.
# This is a SEPARATE environment from the sim container — its torch/transformers
# pins conflict with Isaac Lab's, which is exactly why it must be its own image.
#
# Encodes the two fixes we found bringing the server up on a Blackwell GPU:
#   * torch 2.7.1 +cu128  -> Blackwell sm_120 kernels (cu126 tops out at sm_90)
#   * transformers 4.51.3 -> the checkpoint's Eagle3-VL processor imports
#                            VideoInput from transformers.image_utils (moved in 4.52+)
#
# The ~10 GB checkpoint is NOT baked in — mount it at runtime.
#
# Build:  docker build -t gr00t-server:latest -f docker/gr00t-server/Dockerfile .
# Run:    docker run --rm -it --gpus all --network=host \
#           -v ~/workspace/so101_GR00T_N1.6-3B_WM_v7_50k:/models \
#           gr00t-server:latest \
#           --model-path /models/checkpoint-52000 --device cuda --host 0.0.0.0 --port 5555

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip git ffmpeg libgl1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /workspace
# Pin your Isaac-GR00T commit for reproducibility if desired.
RUN git clone https://github.com/NVIDIA/Isaac-GR00T.git
WORKDIR /workspace/Isaac-GR00T
RUN python -m pip install --upgrade pip && \
    python -m pip install -e .

# cu128 torch for Blackwell, then the transformers the checkpoint's processor needs.
# (Order matters: install these AFTER the repo so they win over its pins.)
RUN python -m pip install --force-reinstall \
      torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
RUN python -m pip install "transformers==4.51.3"

EXPOSE 5555
ENTRYPOINT ["python", "gr00t/eval/run_gr00t_server.py"]
CMD ["--help"]
