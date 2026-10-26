#!/usr/bin/env bash
# Build the pi0.5 (LeRobot upstream) training environment on the RunPod pod.
# venv on the container disk (/root), repo + caches + checkpoints under /workspace.
set -euo pipefail

LEROBOT_COMMIT="${LEROBOT_COMMIT:?pin the huggingface/lerobot commit}"
REPO=/workspace/lerobot
VENV=/root/lerobot-venv
TORCH_INDEX=https://download.pytorch.org/whl/cu128

if [ ! -d "$REPO/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/huggingface/lerobot "$REPO"
fi
cd "$REPO"
GIT_LFS_SKIP_SMUDGE=1 git fetch origin "$LEROBOT_COMMIT"
GIT_LFS_SKIP_SMUDGE=1 git checkout -q "$LEROBOT_COMMIT"

rm -rf "$VENV"
uv venv --python /usr/bin/python3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export UV_LINK_MODE=copy
uv pip install --index-url "$TORCH_INDEX" torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
uv pip install -e ".[pi,training,dataset,diffusion]" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match
uv pip install wandb debugpy
# lerobot resolves torchcodec 0.11 (built for torch 2.11); 0.10 matches torch 2.10.
uv pip install --index-url "$TORCH_INDEX" --reinstall-package torchcodec --no-deps torchcodec==0.10.0

python - <<'EOF'
import torch, transformers, lerobot
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: F401
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
print("transformers", transformers.__version__, "lerobot", lerobot.__file__)
EOF
uv pip check || true
uv pip freeze > /workspace/pi05-so101/requirements-frozen.txt
echo "SETUP_ENV_OK"
