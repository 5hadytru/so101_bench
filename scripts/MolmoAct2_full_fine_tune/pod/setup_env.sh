#!/usr/bin/env bash
# Build the MolmoAct2 training environment on a RunPod pod.
# The venv lives on the container disk (fast, rebuildable); the repo, caches,
# dataset and checkpoints live under /workspace.
set -euo pipefail

MOLMOACT2_COMMIT="${MOLMOACT2_COMMIT:-66b87e64efd99dfd103241418113955cf64dfa9c}"
REPO=/workspace/molmoact2
VENV=/root/molmoact2-venv
TORCH_INDEX=https://download.pytorch.org/whl/cu128

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    screen fio sysstat iotop jq rsync pkg-config \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev \
    libavdevice-dev libavfilter-dev

if [ ! -d "$REPO/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/allenai/molmoact2 "$REPO"
fi
cd "$REPO"
GIT_LFS_SKIP_SMUDGE=1 git fetch origin "$MOLMOACT2_COMMIT"
GIT_LFS_SKIP_SMUDGE=1 git checkout -q "$MOLMOACT2_COMMIT"

rm -rf "$VENV"
uv venv --python /usr/bin/python3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export UV_LINK_MODE=copy

# Same runtime stack as experiments/Dockerfile (torch 2.10 cu128, torchao 0.16, torchcodec 0.10).
uv pip install --index-url "$TORCH_INDEX" torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 torchao==0.16.0
uv pip install torchcodec==0.10.0
cd "$REPO/experiments"
uv pip install -e ".[all]" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match
uv pip install -e "./lerobot[async]" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match
# Re-pin after LeRobot resolves its own dependencies (what the Dockerfile does).
# Only the torch packages are reinstalled; a blanket --reinstall also pulls the newest
# numpy/pillow/fsspec from the PyTorch index and breaks the ai2-molmo/lerobot pins.
uv pip install --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
    --reinstall-package torchao \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 torchao==0.16.0
uv pip install --reinstall-package torchcodec --no-deps torchcodec==0.10.0
uv pip install "numpy>=2.0.0,<2.3.0" "pillow>=9.2.0,<12.0" "fsspec[http]>=2023.1.0,<=2026.2.0" \
    "protobuf>=6.33.5,<7" debugpy

python - <<'EOF'
import torch, torchcodec, transformers, lerobot, olmo
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
print("torchcodec", torchcodec.__version__, "transformers", transformers.__version__)
print("lerobot", lerobot.__file__)
print("olmo", olmo.__file__)
EOF
uv pip freeze > /workspace/molmoact2-so101/requirements-frozen.txt
echo "SETUP_ENV_OK"
