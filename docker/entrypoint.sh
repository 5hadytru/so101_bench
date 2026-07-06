#!/bin/bash
# Entrypoint for the SO-101 Bench sim/eval container.
# Sets up the Isaac Sim Python environment, ensures the extension is installed,
# fetches USD assets if missing, then execs the requested command.
set -e

ISAAC_SIM=/workspace/isaaclab/_isaac_sim
export CARB_APP_PATH=$ISAAC_SIM/kit
export ISAAC_PATH=$ISAAC_SIM
export EXP_PATH=$ISAAC_SIM/apps
source "${ISAAC_SIM}/setup_python_env.sh"

# Make `python` resolve to Isaac Sim's interpreter (so scripts can call `python`
# directly, matching `isaaclab.sh -p`).
cat > /usr/local/bin/python << 'WRAPPER'
#!/bin/bash
exec /workspace/isaaclab/_isaac_sim/python.sh "$@"
WRAPPER
chmod +x /usr/local/bin/python

# Re-install the extension in case /workspace/so101_bench/source is bind-mounted
# from the host (editable install path must resolve). Quiet + best-effort.
python -m pip install -q -e /workspace/so101_bench/source/so101_bench >/dev/null 2>&1 || true

# Fetch USD meshes if they aren't present (skips if already baked/mounted).
/workspace/so101_bench/docker/download_assets.sh || \
  echo "[entrypoint] WARNING: USD assets missing and download failed. " \
       "Set HF_TOKEN or mount the assets before running the sim."

exec "$@"
