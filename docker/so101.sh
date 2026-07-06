#!/bin/bash
# Short entry point for the SO-101 Bench container.
#
#   ./docker/so101.sh build            # build the sim image
#   ./docker/so101.sh test             # smoke test: boot Isaac Sim + list envs
#   ./docker/so101.sh eval [args...]   # run a GR00T eval (server must be up on :5555)
#   ./docker/so101.sh shell            # interactive shell in the container
#   ./docker/so101.sh run <cmd...>     # run an arbitrary command in the container
#
# Env overrides for `eval`: TASK, EPISODES, NUM_EPISODES, REPO_ROOT.
set -e

IMAGE="${SO101_IMAGE:-so101-bench:latest}"
REPO_ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT_DIR"

# Shared `docker run` invocation (GPU, host net for the :5555 server, caches, mounts).
_run() {
  xhost +local:root >/dev/null 2>&1 || true
  mkdir -p ~/.cache/huggingface "$REPO_ROOT_DIR/data"

  # Reuse local USD meshes if present so the container skips the ~430 MB download.
  local ASSETS_MOUNT=()
  local LOCAL_USD="$REPO_ROOT_DIR/source/so101_bench/so101_bench/assets/usd"
  if [ -f "$LOCAL_USD/room_scan.usdc" ]; then
    ASSETS_MOUNT=(-v "$LOCAL_USD:/workspace/so101_bench/source/so101_bench/so101_bench/assets/usd:ro")
  fi

  docker run --rm -it --gpus all --network=host \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e "HF_TOKEN=${HF_TOKEN:-}" \
    -e DISPLAY -e "NVIDIA_DRIVER_CAPABILITIES=all" -e "NVIDIA_VISIBLE_DEVICES=all" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
    -v ~/.cache/huggingface:/root/.cache/huggingface:rw \
    "${ASSETS_MOUNT[@]}" \
    -v "$REPO_ROOT_DIR/tasks:/workspace/so101_bench/tasks:rw" \
    -v "$REPO_ROOT_DIR/data:/workspace/so101_bench/data:rw" \
    "$IMAGE" "$@"
}

ISAACLAB="/workspace/isaaclab/isaaclab.sh -p"

case "${1:-}" in
  build)
    shift
    exec docker build --network=host -t "$IMAGE" -f docker/Dockerfile "$@" .
    ;;
  test)
    _run $ISAACLAB scripts/list_envs.py
    ;;
  eval)
    shift
    # Windowed by default (Isaac Sim GUI). Set HEADLESS=1 for batch/no-window.
    HEADLESS_FLAG=""
    [ -n "${HEADLESS:-}" ] && HEADLESS_FLAG="--headless"
    _run $ISAACLAB scripts/groot_eval.py \
      --task "${TASK:-So101Bench-Bin-v0}" \
      --episodes_jsonl "${EPISODES:-tasks/custom_bin.jsonl}" \
      --policy_host localhost --policy_port 5555 \
      --action_horizon 16 --use_overhead_init true \
      --num_episodes "${NUM_EPISODES:-5}" \
      --record_dataset --repo_root "${REPO_ROOT:-data/lerobot/docker_eval}" \
      $HEADLESS_FLAG "$@"
    ;;
  shell|"")
    _run bash
    ;;
  run)
    shift
    _run "$@"
    ;;
  *)
    _run $ISAACLAB "$@"
    ;;
esac
