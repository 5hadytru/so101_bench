#!/usr/bin/env bash
# Local (RTX 3090) G0.5 inference env for validation: the pinned GalaxeaVLA repo + uv env, the shared action
# tokenizer and Qwen3.5 processor (gated HF repo; uses the local HF login), plus fastapi/uvicorn for
# scripts/g05_server.py. Checkpoints come from the training pod, not from here.
set -euo pipefail
export UV_CACHE_DIR=${HOME}/.cache/uv
# egl-probe (via robomimic) declares cmake_minimum_required < 3.5, which CMake 4 rejects.
export CMAKE_POLICY_VERSION_MINIMUM=3.5
unset HF_TOKEN
cd "${HOME}/workspace"
[[ -d GalaxeaVLA ]] || git clone -q https://github.com/OpenGalaxea/GalaxeaVLA.git
git -C GalaxeaVLA checkout -q 89f2322
(cd GalaxeaVLA && uv sync --index-strategy unsafe-best-match)
git -C GalaxeaVLA apply "${HOME}/so101_bench/scripts/G05_fine_tuning/patches/galaxeavla_89f2322.patch"  # RTX 3090 (sm86): FA2, not FA4
uv pip install -q --python GalaxeaVLA/.venv/bin/python fastapi 'uvicorn[standard]' \
    'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl'
GalaxeaVLA/.venv/bin/hf download OpenGalaxea/G05 --revision e312be81e90c56a55bcb26b57429bd39a335b449 \
    --include 'action_tokenizer.pt' 'qwen3_5_2b_base_processor/*' 'g05-base/.hydra/*' 'g05-base/dataset_stats.json' \
    --local-dir GalaxeaVLA/checkpoints
GalaxeaVLA/.venv/bin/python -c "import torch, g05, fastapi; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
echo G05_LOCAL_SETUP_OK
