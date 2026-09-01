#!/usr/bin/env bash
# Download and evaluate selected completed checkpoints from a live RunPod GR00T job.
#
# Each remote checkpoint is copied into one script-owned, *flat* local model
# directory, started behind a local GR00T inference server, and evaluated on
# the fixed real_gr00t_val task/layout pair.  The disposable evaluation cache
# holds one model at once.  Separately, the best completed models are retained
# in script-owned checkpoint directories.  A completed.json marker makes
# completed evaluations idempotent across restarts. A skipped.json marker in a
# checkpoint result directory suppresses evaluation without claiming success.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

REMOTE_HOST="${REMOTE_HOST:-root@38.128.232.45}"
REMOTE_PORT="${REMOTE_PORT:-42544}"
REMOTE_IDENTITY="${REMOTE_IDENTITY:-${HOME}/.ssh/id_ed25519}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-/workspace/so101_GR00T-N1.6-3B_WM_sim_v1}"
RUN_LABEL="${RUN_LABEL:-}"
if [[ -n "${RUN_LABEL}" && ! "${RUN_LABEL}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "[error] RUN_LABEL may contain only letters, numbers, dots, underscores, and hyphens." >&2
    exit 2
fi
RUN_SUFFIX="${RUN_LABEL:+_${RUN_LABEL}}"

# This directory is an intentionally disposable cache. It contains the files
# *inside* one remote checkpoint (config.json, processor_config.json, model
# shards, etc.), so it is suitable directly for --model-path.  Do not point it
# at an existing manually managed checkpoint unless that directory has first
# been made script-owned (see assert_cache_is_safe).
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-${ROOT_DIR}/checkpoints/so101_GR00T_N1.6-3B_WM_runpod_eval_active}"
BEST_CHECKPOINTS_DIR="${BEST_CHECKPOINTS_DIR:-${ROOT_DIR}/checkpoints/so101_GR00T_N1.6-3B_WM_runpod_best${RUN_SUFFIX}}"
BEST_CHECKPOINT_COUNT="${BEST_CHECKPOINT_COUNT:-3}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/outputs/groot_runpod_checkpoint_validation${RUN_SUFFIX}}"

TASKS_JSONL="${TASKS_JSONL:-${ROOT_DIR}/tasks/real_gr00t_val.jsonl}"
LAYOUTS_JSONL="${LAYOUTS_JSONL:-${ROOT_DIR}/tasks/layouts/real_gr00t_val_layouts_20260718_155446.jsonl}"
TASK="${TASK:-So101Bench-Bin-v0}"
SEED="${SEED:-0}"
ACTION_HORIZON="${ACTION_HORIZON:-8}"

ISAACLAB_SH="${ISAACLAB_SH:-/home/truman/IsaacLab/isaaclab.sh}"
PYTHON_FOR_TOOLS="${PYTHON_FOR_TOOLS:-${HOME}/env_isaaclab_51/bin/python}"
GROOT_WORKDIR="${GROOT_WORKDIR:-${HOME}/workspace/Isaac-GR00T}"
GROOT_PYTHON="${GROOT_PYTHON:-${GROOT_WORKDIR}/.venv/bin/python}"
GROOT_SERVER_SCRIPT="${GROOT_SERVER_SCRIPT:-gr00t/eval/run_gr00t_server.py}"
SERVER_HF_HOME="${SERVER_HF_HOME:-${HOME}/.cache/huggingface}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
SERVER_DEVICE="${SERVER_DEVICE:-cuda}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-5555}"
SERVER_STARTUP_TIMEOUT_S="${SERVER_STARTUP_TIMEOUT_S:-180}"

# A checkpoint must have an unchanged file manifest across this interval before
# it is downloaded. This avoids copying a checkpoint while Trainer is writing it.
CHECKPOINT_STABILITY_S="${CHECKPOINT_STABILITY_S:-30}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-300}"
ONLY_CHECKPOINT="${ONLY_CHECKPOINT:-}"
# Evaluating all 31 validation episodes takes about two hours, while this
# eight-GPU run emits a checkpoint roughly every 90 minutes. Evaluate every
# 2,000 training steps by default so the monitor does not fall behind the
# remote save_total_limit window. An explicit --checkpoint bypasses this rule.
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-2000}"

ONCE=0
DRY_RUN=0
SERVER_PID=""

usage() {
    cat <<'EOF'
Usage: bash scripts/monitor_runpod_groot_checkpoints.sh [options]

Continuously evaluates stable, unvalidated checkpoints in REMOTE_RUN_DIR whose
step is divisible by EVAL_EVERY_STEPS (default: 2000).

Options:
  --once                 Scan and evaluate currently available checkpoints once, then exit.
  --poll-interval SEC    Seconds between scans in monitor mode (default: 300).
  --stability-wait SEC   Unchanged-manifest wait before downloading a checkpoint (default: 30).
  --every-steps STEPS    Evaluate checkpoints divisible by STEPS (default: 2000).
  --checkpoint NAME      Only consider this checkpoint (for example, checkpoint-28000).
                         This overrides --every-steps for the named checkpoint.
  --dry-run              Print checkpoint decisions without downloading or evaluating.
  -h, --help             Show this help.

Useful environment overrides:
  REMOTE_HOST, REMOTE_PORT, REMOTE_IDENTITY, REMOTE_RUN_DIR, RUN_LABEL,
  LOCAL_MODEL_DIR, BEST_CHECKPOINTS_DIR, BEST_CHECKPOINT_COUNT, RESULT_ROOT,
  EVAL_EVERY_STEPS, POLICY_PORT, GROOT_WORKDIR, GROOT_PYTHON, SERVER_HF_HOME.

After every successful evaluation, the best three checkpoints by overall
success rate are retained under:
  checkpoints/so101_GR00T_N1.6-3B_WM_runpod_best/checkpoint-N/

Ties are broken by successes and then by the newer checkpoint step. On
restart, missing top checkpoints are backfilled from REMOTE_RUN_DIR.

Outputs for checkpoint-N are written under:
  outputs/groot_runpod_checkpoint_validation/checkpoint-N/
    completed.json                 idempotency marker and file locations
    run_TIMESTAMP/eval.log         raw evaluator output
    run_TIMESTAMP/success_rates.*  aggregate success rates
    run_TIMESTAMP/episode_results.{json,tsv}  every success/failure and task
    run_TIMESTAMP/lerobot_dataset/ recorded evaluation episodes
EOF
}

while (($#)); do
    case "$1" in
        --once)
            ONCE=1
            ;;
        --poll-interval)
            POLL_INTERVAL_S="${2:?--poll-interval requires seconds}"
            shift
            ;;
        --stability-wait)
            CHECKPOINT_STABILITY_S="${2:?--stability-wait requires seconds}"
            shift
            ;;
        --every-steps)
            EVAL_EVERY_STEPS="${2:?--every-steps requires a positive step interval}"
            shift
            ;;
        --checkpoint)
            ONLY_CHECKPOINT="${2:?--checkpoint requires checkpoint-N}"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[error] Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ -n "${ONLY_CHECKPOINT}" && ! "${ONLY_CHECKPOINT}" =~ ^checkpoint-[0-9]+$ ]]; then
    echo "[error] --checkpoint must have the form checkpoint-N; got ${ONLY_CHECKPOINT@Q}." >&2
    exit 2
fi

for numeric_var in POLL_INTERVAL_S CHECKPOINT_STABILITY_S SERVER_STARTUP_TIMEOUT_S POLICY_PORT BEST_CHECKPOINT_COUNT EVAL_EVERY_STEPS; do
    if ! [[ "${!numeric_var}" =~ ^[0-9]+$ ]]; then
        echo "[error] ${numeric_var} must be a non-negative integer; got ${!numeric_var@Q}." >&2
        exit 2
    fi
done
if ((EVAL_EVERY_STEPS == 0)); then
    echo "[error] EVAL_EVERY_STEPS must be greater than zero." >&2
    exit 2
fi

require_file() {
    local path="$1"
    local description="$2"
    if [[ ! -f "${path}" ]]; then
        echo "[error] Missing ${description}: ${path}" >&2
        exit 1
    fi
}

require_file "${TASKS_JSONL}" "task JSONL"
require_file "${LAYOUTS_JSONL}" "layout JSONL"
require_file "${PYTHON_FOR_TOOLS}" "Isaac-Lab Python"
require_file "${ISAACLAB_SH}" "Isaac Lab launcher"
require_file "${GROOT_PYTHON}" "GR00T Python"
require_file "${GROOT_WORKDIR}/${GROOT_SERVER_SCRIPT}" "GR00T server script"
require_file "${ROOT_DIR}/scripts/groot_eval.py" "GR00T evaluator"
require_file "${ROOT_DIR}/scripts/compute_groot_success_rates.py" "success-rate utility"
require_file "${ROOT_DIR}/scripts/write_groot_episode_results.py" "episode-outcome utility"
require_file "${REMOTE_IDENTITY}" "SSH identity file"

if ! command -v ssh >/dev/null; then
    echo "[error] ssh is required but was not found in PATH." >&2
    exit 1
fi
if ! command -v rsync >/dev/null; then
    echo "[error] rsync is required locally. Install it with: sudo apt-get install rsync" >&2
    exit 1
fi

SSH_BASE=(
    ssh
    -p "${REMOTE_PORT}"
    -i "${REMOTE_IDENTITY}"
    -o BatchMode=yes
    -o ConnectTimeout=20
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
)
RSYNC_RSH="ssh -p ${REMOTE_PORT} -i ${REMOTE_IDENTITY} -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=3"

# Training-resume state is not needed by the inference server. In an
# eight-GPU DeepSpeed checkpoint, global_step*/ can contain tens of gigabytes
# of per-rank optimizer state, so exclude it from both active and retained
# checkpoint downloads.
CHECKPOINT_RSYNC_FILTERS=(
    --exclude='/global_step*/'
    --exclude='/optimizer.pt'
    --exclude='/scheduler.pt'
    --exclude='/rng_state*.pth'
    --exclude='/trainer_state.json'
    --exclude='/training_args.bin'
    --exclude='/zero_to_fp32.py'
    --exclude='/latest'
)

quote_remote() {
    # The RunPod shell is bash. printf %q keeps configured paths as one safe shell word.
    printf '%q' "$1"
}

remote_command() {
    "${SSH_BASE[@]}" "${REMOTE_HOST}" "$1"
}

remote_rsync_status=0
remote_command "command -v rsync >/dev/null 2>&1 || exit 127" || remote_rsync_status=$?
if ((remote_rsync_status == 127)); then
    cat >&2 <<EOF
[error] rsync is required on the remote host (${REMOTE_HOST}) but was not found.
Install it there before starting the monitor, for example:
  apt-get update && apt-get install -y rsync
EOF
    exit 1
elif ((remote_rsync_status != 0)); then
    echo "[error] Could not connect to ${REMOTE_HOST} to verify remote rsync (ssh exit ${remote_rsync_status})." >&2
    exit 1
fi

list_remote_checkpoints() {
    local remote_dir_quoted
    remote_dir_quoted="$(quote_remote "${REMOTE_RUN_DIR}")"
    remote_command "find ${remote_dir_quoted} -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' | LC_ALL=C sort -V"
}

remote_manifest() {
    local checkpoint="$1"
    local remote_checkpoint_quoted
    remote_checkpoint_quoted="$(quote_remote "${REMOTE_RUN_DIR}/${checkpoint}")"
    remote_command "set -euo pipefail; cd ${remote_checkpoint_quoted}; test -f trainer_state.json; test -f config.json; test -f processor_config.json; test -f model.safetensors.index.json -o -f model.safetensors; find . -type f ! -path './global_step*/*' ! -name 'optimizer.pt' ! -name 'scheduler.pt' ! -name 'rng_state*.pth' ! -name 'trainer_state.json' ! -name 'training_args.bin' ! -name 'zero_to_fp32.py' ! -name 'latest' -printf '%P\\t%s\\t%T@\\n' | LC_ALL=C sort | sha256sum | awk '{print \$1}'"
}

checkpoint_is_stable() {
    local checkpoint="$1"
    local first_manifest second_manifest
    if ! first_manifest="$(remote_manifest "${checkpoint}")"; then
        echo "[wait] ${checkpoint} is not a complete GR00T checkpoint yet." >&2
        return 1
    fi
    if ((CHECKPOINT_STABILITY_S > 0)); then
        sleep "${CHECKPOINT_STABILITY_S}"
    fi
    if ! second_manifest="$(remote_manifest "${checkpoint}")"; then
        echo "[wait] ${checkpoint} disappeared or is incomplete after stability wait." >&2
        return 1
    fi
    if [[ "${first_manifest}" != "${second_manifest}" ]]; then
        echo "[wait] ${checkpoint} is still changing; defer until the next scan." >&2
        return 1
    fi
    printf '%s' "${second_manifest}"
}

assert_cache_is_safe() {
    local cache_parent allowed_parent cache_real
    cache_parent="$(dirname "${LOCAL_MODEL_DIR}")"
    allowed_parent="${ROOT_DIR}/checkpoints"
    cache_real="$(realpath -m "${LOCAL_MODEL_DIR}")"

    if [[ "${cache_real}" != "${allowed_parent}/"* ]]; then
        echo "[error] LOCAL_MODEL_DIR must be below ${allowed_parent}; got ${cache_real}" >&2
        exit 1
    fi
    if [[ -e "${LOCAL_MODEL_DIR}" && ! -f "${LOCAL_MODEL_DIR}/.runpod_groot_eval_cache" ]]; then
        cat >&2 <<EOF
[error] Refusing to overwrite a non-empty, manually managed LOCAL_MODEL_DIR:
  ${LOCAL_MODEL_DIR}

Use the default script-owned cache, or choose a new empty directory below
${allowed_parent}. The monitor only uses rsync --delete inside directories
marked .runpod_groot_eval_cache.
EOF
        exit 1
    fi
    mkdir -p "${cache_parent}" "${LOCAL_MODEL_DIR}"
    printf 'This directory is an automatically replaced RunPod GR00T checkpoint cache.\n' \
        >"${LOCAL_MODEL_DIR}/.runpod_groot_eval_cache"
}

local_checkpoint_is_complete() {
    checkpoint_dir_is_complete "${LOCAL_MODEL_DIR}"
}

checkpoint_dir_is_complete() {
    local checkpoint_dir="$1"
    local model_glob
    [[ -f "${checkpoint_dir}/config.json" ]] || return 1
    [[ -f "${checkpoint_dir}/processor_config.json" ]] || return 1
    model_glob="$(compgen -G "${checkpoint_dir}/model*.safetensors" || true)"
    [[ -n "${model_glob}" ]]
}

validate_safetensors_checkpoint() {
    local checkpoint_dir="$1"
    "${GROOT_PYTHON}" - "${checkpoint_dir}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from safetensors import safe_open

root = Path(sys.argv[1])
index_path = root / "model.safetensors.index.json"
if index_path.is_file():
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    if not shard_names:
        raise SystemExit(f"Safetensors index has no weight-map entries: {index_path}")
    shard_paths = [root / name for name in shard_names]
else:
    shard_paths = sorted(root.glob("model*.safetensors"))

if not shard_paths:
    raise SystemExit(f"No model safetensors shards found in {root}")

for shard in shard_paths:
    if not shard.is_file():
        raise SystemExit(f"Safetensors index references a missing shard: {shard}")
    try:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
    except Exception as exc:
        raise SystemExit(f"Invalid safetensors shard {shard}: {exc}") from exc
    if not keys:
        raise SystemExit(f"Safetensors shard contains no tensors: {shard}")

print(f"[download] validated {len(shard_paths)} safetensors shard(s) in {root}")
PY
}

download_checkpoint() {
    local checkpoint="$1"
    local manifest="$2"
    local remote_checkpoint="${REMOTE_RUN_DIR}/${checkpoint}/"

    assert_cache_is_safe
    echo "[download] ${REMOTE_HOST}:${remote_checkpoint} -> ${LOCAL_MODEL_DIR}/"
    # This cache previously contains a *different* checkpoint whose shard
    # names and sizes can match the new one. --append/--append-verify assumes
    # an existing destination is a prefix of the source and can therefore
    # preserve stale or overlong shards. Use checksum quick-checking so rsync
    # compares same-sized shards and reconstructs/truncates them normally.
    if ! rsync -a --checksum --delete --delete-excluded --partial \
        --filter='protect /.runpod_groot_eval_cache' \
        "${CHECKPOINT_RSYNC_FILTERS[@]}" \
        --no-owner --no-group --no-perms \
        -e "${RSYNC_RSH}" \
        "${REMOTE_HOST}:${remote_checkpoint}" "${LOCAL_MODEL_DIR}/"; then
        echo "[error] rsync failed while downloading ${checkpoint}." >&2
        return 1
    fi
    if ! local_checkpoint_is_complete; then
        echo "[error] Downloaded ${checkpoint}, but the local cache is not a complete GR00T model." >&2
        return 1
    fi
    if ! validate_safetensors_checkpoint "${LOCAL_MODEL_DIR}"; then
        echo "[error] Downloaded ${checkpoint}, but its local safetensors files are invalid." >&2
        return 1
    fi
    printf '%s\n' "${manifest}" >"${LOCAL_MODEL_DIR}/.remote_manifest_sha256"
    printf '%s\n' "${checkpoint}" >"${LOCAL_MODEL_DIR}/.remote_checkpoint"
}

assert_best_checkpoints_dir_is_safe() {
    local allowed_parent best_real
    allowed_parent="${ROOT_DIR}/checkpoints"
    best_real="$(realpath -m "${BEST_CHECKPOINTS_DIR}")"
    if [[ "${best_real}" != "${allowed_parent}/"* ]]; then
        echo "[error] BEST_CHECKPOINTS_DIR must be below ${allowed_parent}; got ${best_real}" >&2
        return 1
    fi
    if [[ -e "${BEST_CHECKPOINTS_DIR}" &&
          ! -f "${BEST_CHECKPOINTS_DIR}/.runpod_groot_best_checkpoints" ]]; then
        echo "[error] Refusing to manage unmarked BEST_CHECKPOINTS_DIR: ${BEST_CHECKPOINTS_DIR}" >&2
        return 1
    fi
    mkdir -p "$(dirname "${BEST_CHECKPOINTS_DIR}")" "${BEST_CHECKPOINTS_DIR}"
    printf 'This directory contains automatically retained RunPod GR00T checkpoints.\n' \
        >"${BEST_CHECKPOINTS_DIR}/.runpod_groot_best_checkpoints"
}

ranked_best_checkpoints() {
    "${PYTHON_FOR_TOOLS}" - "${RESULT_ROOT}" "${BEST_CHECKPOINT_COUNT}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

result_root = Path(sys.argv[1])
limit = int(sys.argv[2])
ranked = []
invalid_results = False
for marker in result_root.glob("checkpoint-*/completed.json"):
    try:
        completed = json.loads(marker.read_text(encoding="utf-8"))
        checkpoint = str(completed["checkpoint"])
        match = re.fullmatch(r"checkpoint-(\d+)", checkpoint)
        if match is None:
            continue
        rates_path = Path(completed["success_rates_json"])
        rates = json.loads(rates_path.read_text(encoding="utf-8"))
        overall = rates["overall"]
        successes = int(overall["successes"])
        total = int(overall["total"])
        success_rate = float(overall["success_rate"])
        manifest = str(completed["remote_manifest_sha256"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[retain] invalid completed result {marker}: {exc}", file=sys.stderr)
        invalid_results = True
        continue
    ranked.append((success_rate, successes, int(match.group(1)), checkpoint, manifest, total))

if invalid_results:
    raise SystemExit(1)

ranked.sort(reverse=True)
for success_rate, successes, _, checkpoint, manifest, total in ranked[:limit]:
    print(f"{checkpoint}\t{manifest}\t{success_rate:.17g}\t{successes}\t{total}")
PY
}

install_retained_checkpoint_from_cache() {
    local checkpoint="$1"
    local manifest="$2"
    local destination="${BEST_CHECKPOINTS_DIR}/${checkpoint}"
    local temporary="${BEST_CHECKPOINTS_DIR}/.${checkpoint}.tmp.$$"

    if [[ ! -f "${LOCAL_MODEL_DIR}/.remote_manifest_sha256" ]] ||
       [[ "$(<"${LOCAL_MODEL_DIR}/.remote_manifest_sha256")" != "${manifest}" ]]; then
        echo "[error] Active cache manifest does not match completed result for ${checkpoint}." >&2
        return 1
    fi
    echo "[retain] copying ${checkpoint} from the active cache -> ${destination}"
    mkdir -p "${temporary}"
    cp -a --reflink=auto "${LOCAL_MODEL_DIR}/." "${temporary}/"
    rm -f "${temporary}/.runpod_groot_eval_cache"
    printf '%s\n' "${checkpoint}" >"${temporary}/.remote_checkpoint"
    printf '%s\n' "${manifest}" >"${temporary}/.remote_manifest_sha256"
    printf 'This is an automatically retained RunPod GR00T checkpoint.\n' \
        >"${temporary}/.runpod_groot_retained_checkpoint"
    mv "${temporary}" "${destination}"
}

download_retained_checkpoint() {
    local checkpoint="$1"
    local manifest="$2"
    local destination="${BEST_CHECKPOINTS_DIR}/${checkpoint}"
    local temporary="${BEST_CHECKPOINTS_DIR}/.${checkpoint}.tmp.$$"
    local remote_checkpoint="${REMOTE_RUN_DIR}/${checkpoint}/"
    local before_manifest after_manifest

    if ! before_manifest="$(remote_manifest "${checkpoint}")" ||
       [[ "${before_manifest}" != "${manifest}" ]]; then
        echo "[error] Remote ${checkpoint} is unavailable, incomplete, or changed since evaluation." >&2
        return 1
    fi
    echo "[retain] backfilling ${REMOTE_HOST}:${remote_checkpoint} -> ${destination}"
    mkdir -p "${temporary}"
    if ! rsync -a --checksum --delete --delete-excluded --partial \
        "${CHECKPOINT_RSYNC_FILTERS[@]}" \
        --no-owner --no-group --no-perms \
        -e "${RSYNC_RSH}" \
        "${REMOTE_HOST}:${remote_checkpoint}" "${temporary}/"; then
        rm -rf -- "${temporary}"
        return 1
    fi
    if ! checkpoint_dir_is_complete "${temporary}"; then
        echo "[error] Backfilled ${checkpoint} is not a complete GR00T model." >&2
        rm -rf -- "${temporary}"
        return 1
    fi
    if ! validate_safetensors_checkpoint "${temporary}"; then
        echo "[error] Backfilled ${checkpoint} has invalid safetensors files." >&2
        rm -rf -- "${temporary}"
        return 1
    fi
    if ! after_manifest="$(remote_manifest "${checkpoint}")" ||
       [[ "${after_manifest}" != "${manifest}" ]]; then
        echo "[error] Remote ${checkpoint} changed while it was being backfilled." >&2
        rm -rf -- "${temporary}"
        return 1
    fi
    printf '%s\n' "${checkpoint}" >"${temporary}/.remote_checkpoint"
    printf '%s\n' "${manifest}" >"${temporary}/.remote_manifest_sha256"
    printf 'This is an automatically retained RunPod GR00T checkpoint.\n' \
        >"${temporary}/.runpod_groot_retained_checkpoint"
    mv "${temporary}" "${destination}"
}

reconcile_best_checkpoints() {
    local rows=() desired=() checkpoint manifest rate successes total
    local destination active_checkpoint="" retained ranking_output
    local retention_failed=0
    ((BEST_CHECKPOINT_COUNT > 0)) || return 0
    assert_best_checkpoints_dir_is_safe || return 1
    if ! ranking_output="$(ranked_best_checkpoints)"; then
        echo "[error] Refusing to change retained checkpoints because result ranking failed." >&2
        return 1
    fi
    [[ -n "${ranking_output}" ]] || return 0
    mapfile -t rows <<<"${ranking_output}"

    if [[ -f "${LOCAL_MODEL_DIR}/.remote_checkpoint" ]]; then
        active_checkpoint="$(<"${LOCAL_MODEL_DIR}/.remote_checkpoint")"
    fi
    for row in "${rows[@]}"; do
        IFS=$'\t' read -r checkpoint manifest rate successes total <<<"${row}"
        desired+=("${checkpoint}")
        destination="${BEST_CHECKPOINTS_DIR}/${checkpoint}"
        if [[ -d "${destination}" ]]; then
            if [[ ! -f "${destination}/.runpod_groot_retained_checkpoint" ]]; then
                echo "[error] Refusing to replace unmarked retained path: ${destination}" >&2
                return 1
            fi
            if checkpoint_dir_is_complete "${destination}" &&
               [[ -f "${destination}/.remote_manifest_sha256" ]] &&
               [[ "$(<"${destination}/.remote_manifest_sha256")" == "${manifest}" ]]; then
                continue
            fi
            echo "[retain] replacing incomplete or stale managed path: ${destination}"
            rm -rf -- "${destination}"
        fi
        if ((DRY_RUN)); then
            echo "[dry-run] would retain ${checkpoint} (${successes}/${total}, rate=${rate})."
        elif [[ "${active_checkpoint}" == "${checkpoint}" ]] &&
             checkpoint_dir_is_complete "${LOCAL_MODEL_DIR}"; then
            install_retained_checkpoint_from_cache "${checkpoint}" "${manifest}" || return 1
        elif ! download_retained_checkpoint "${checkpoint}" "${manifest}"; then
            echo "[error] Could not backfill top-ranked ${checkpoint}; it will be retried next scan." >&2
            retention_failed=1
        fi
    done

    # Keep the previous lower-ranked model until every intended replacement is
    # safely materialized, so a transient network failure never leaves fewer
    # than BEST_CHECKPOINT_COUNT retained models.
    ((retention_failed == 0)) || return 1
    for retained in "${BEST_CHECKPOINTS_DIR}"/checkpoint-*; do
        [[ -d "${retained}" ]] || continue
        checkpoint="$(basename "${retained}")"
        if [[ " ${desired[*]} " == *" ${checkpoint} "* ]]; then
            continue
        fi
        if [[ ! -f "${retained}/.runpod_groot_retained_checkpoint" ]]; then
            echo "[error] Refusing to prune unmarked path: ${retained}" >&2
            return 1
        fi
        if ((DRY_RUN)); then
            echo "[dry-run] would prune retained ${checkpoint}."
        else
            echo "[retain] pruning ${checkpoint}; it is no longer in the top ${BEST_CHECKPOINT_COUNT}."
            rm -rf -- "${retained}"
        fi
    done
}

ping_server() {
    "${PYTHON_FOR_TOOLS}" - "${POLICY_HOST}" "${POLICY_PORT}" <<'PY' >/dev/null 2>&1
import sys

import msgpack
import zmq

host = sys.argv[1]
port = int(sys.argv[2])
context = zmq.Context.instance()
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, 2000)
socket.setsockopt(zmq.SNDTIMEO, 2000)
socket.connect(f"tcp://{host}:{port}")
try:
    socket.send(msgpack.packb({"endpoint": "ping"}, use_bin_type=True))
    socket.recv()
except Exception:
    sys.exit(1)
finally:
    socket.close(0)
PY
}

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}

trap stop_server EXIT
trap 'stop_server; exit 130' INT
trap 'stop_server; exit 143' TERM

start_server() {
    local server_log="$1"
    if ping_server; then
        echo "[error] A GR00T server is already responding on ${POLICY_HOST}:${POLICY_PORT}." >&2
        echo "[error] Stop it or set POLICY_PORT to a free local port before starting the monitor." >&2
        return 1
    fi

    echo "[server] starting ${LOCAL_MODEL_DIR} on ${POLICY_HOST}:${POLICY_PORT}"
    mkdir -p "${SERVER_HF_HOME}"
    if [[ ! -w "${SERVER_HF_HOME}" ]]; then
        echo "[error] Local Hugging Face cache is not writable: ${SERVER_HF_HOME}" >&2
        return 1
    fi
    (
        cd "${GROOT_WORKDIR}"
        # Do not inherit a RunPod-oriented HF_HOME such as /workspace/hf_cache.
        # Transformers needs a writable cache for imported model code, but that
        # cache belongs to this local server and is unrelated to the checkpoint
        # source directory on the remote host.
        exec env \
            HF_HOME="${SERVER_HF_HOME}" \
            HF_MODULES_CACHE="${SERVER_HF_HOME}/modules" \
            "${GROOT_PYTHON}" "${GROOT_SERVER_SCRIPT}" \
            --model-path "${LOCAL_MODEL_DIR}" \
            --embodiment-tag "${EMBODIMENT_TAG}" \
            --device "${SERVER_DEVICE}" \
            --host "${POLICY_HOST}" \
            --port "${POLICY_PORT}"
    ) >"${server_log}" 2>&1 &
    SERVER_PID="$!"
}

wait_for_server() {
    local server_log="$1"
    local deadline=$((SECONDS + SERVER_STARTUP_TIMEOUT_S))
    until ping_server; do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[error] GR00T server exited before becoming ready. Last server log lines:" >&2
            tail -n 100 "${server_log}" >&2 || true
            return 1
        fi
        if ((SECONDS >= deadline)); then
            echo "[error] Timed out waiting for GR00T server on ${POLICY_HOST}:${POLICY_PORT}." >&2
            tail -n 100 "${server_log}" >&2 || true
            return 1
        fi
        sleep 2
    done
    echo "[server] ready"
}

write_completed_marker() {
    local result_dir="$1"
    local run_dir="$2"
    local checkpoint="$3"
    local manifest="$4"
    local marker="${result_dir}/completed.json"
    "${PYTHON_FOR_TOOLS}" - "${marker}" "${checkpoint}" "${manifest}" "${run_dir}" \
        "${REMOTE_HOST}" "${REMOTE_RUN_DIR}" "${LOCAL_MODEL_DIR}" "${TASKS_JSONL}" "${LAYOUTS_JSONL}" \
        "${ACTION_HORIZON}" <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

(
    marker,
    checkpoint,
    manifest,
    run_dir,
    remote_host,
    remote_run_dir,
    local_model_dir,
    tasks_jsonl,
    layouts_jsonl,
    action_horizon,
) = sys.argv[1:]
run = Path(run_dir)
payload = {
    "checkpoint": checkpoint,
    "remote_host": remote_host,
    "remote_run_dir": remote_run_dir,
    "remote_manifest_sha256": manifest,
    "local_model_dir_used": local_model_dir,
    "tasks_jsonl": tasks_jsonl,
    "layouts_jsonl": layouts_jsonl,
    "action_horizon": int(action_horizon),
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_dir": str(run),
    "eval_log": str(run / "eval.log"),
    "server_log": str(run / "server.log"),
    "success_rates_text": str(run / "success_rates.txt"),
    "success_rates_json": str(run / "success_rates.json"),
    "episode_results_json": str(run / "episode_results.json"),
    "episode_results_tsv": str(run / "episode_results.tsv"),
    "episode_diagnostics_jsonl": str(run / "episode_diagnostics.jsonl"),
    "lerobot_dataset": str(run / "lerobot_dataset"),
}
marker_path = Path(marker)
temporary = marker_path.with_name(f".{marker_path.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(marker_path)
PY
}

validate_recorded_dataset() {
    local dataset_root="$1"
    local expected_episodes="$2"
    "${PYTHON_FOR_TOOLS}" - "${dataset_root}" "${expected_episodes}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected_episodes = int(sys.argv[2])
info_path = root / "meta" / "info.json"
tasks_path = root / "meta" / "tasks.parquet"
if not info_path.is_file() or not tasks_path.is_file():
    raise SystemExit(f"Recorded LeRobot dataset is incomplete: expected {info_path} and {tasks_path}.")
info = json.loads(info_path.read_text(encoding="utf-8"))
total_episodes = int(info.get("total_episodes", -1))
total_frames = int(info.get("total_frames", 0))
if total_episodes != expected_episodes:
    raise SystemExit(
        f"Recorded LeRobot dataset has {total_episodes} episodes; expected {expected_episodes}."
    )
if total_frames <= 0:
    raise SystemExit("Recorded LeRobot dataset has no frames.")
if not any(root.glob("data/**/*.parquet")):
    raise SystemExit("Recorded LeRobot dataset has no frame parquet files.")
if not any(root.glob("videos/**/*.mp4")):
    raise SystemExit("Recorded LeRobot dataset has no MP4 video files.")
print(f"[dataset] validated {root}: {total_episodes} episodes, {total_frames} frames")
PY
}

run_validation() {
    local checkpoint="$1"
    local manifest="$2"
    local result_dir="${RESULT_ROOT}/${checkpoint}"
    local timestamp run_dir server_log eval_log rates_text rates_json outcomes_json outcomes_tsv diagnostics_jsonl summary
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    run_dir="${result_dir}/run_${timestamp}"
    server_log="${run_dir}/server.log"
    eval_log="${run_dir}/eval.log"
    rates_text="${run_dir}/success_rates.txt"
    rates_json="${run_dir}/success_rates.json"
    outcomes_json="${run_dir}/episode_results.json"
    outcomes_tsv="${run_dir}/episode_results.tsv"
    diagnostics_jsonl="${run_dir}/episode_diagnostics.jsonl"
    summary="${run_dir}/summary.txt"
    mkdir -p "${run_dir}"

    printf '%s\n' "${manifest}" >"${run_dir}/remote_manifest_sha256"
    cp "${TASKS_JSONL}" "${run_dir}/tasks.jsonl"
    cp "${LAYOUTS_JSONL}" "${run_dir}/layouts.jsonl"

    if ! start_server "${server_log}" || ! wait_for_server "${server_log}"; then
        stop_server
        return 1
    fi

    echo "[eval] ${checkpoint}: ${TASKS_JSONL} ($(wc -l <"${TASKS_JSONL}") episodes)"
    local eval_status=0
    local isaaclab_venv
    isaaclab_venv="$(dirname "$(dirname "${PYTHON_FOR_TOOLS}")")"
    set +e
    # isaaclab.sh prioritizes CONDA_PREFIX over VIRTUAL_ENV. Clear inherited
    # Conda state and explicitly provide the Isaac-Lab venv, so a monitor
    # launched from a "base" shell still imports isaaclab correctly.
    env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME \
        VIRTUAL_ENV="${isaaclab_venv}" \
        PATH="${isaaclab_venv}/bin:${PATH}" \
        "${ISAACLAB_SH}" -p scripts/groot_eval.py \
        --task "${TASK}" \
        --episodes_jsonl "${TASKS_JSONL}" \
        --episode_layouts_jsonl "${LAYOUTS_JSONL}" \
        --seed "${SEED}" \
        --policy_host "${POLICY_HOST}" \
        --policy_port "${POLICY_PORT}" \
        --action_horizon "${ACTION_HORIZON}" \
        --use_overhead_init true \
        --record_dataset \
        --repo_id "5hadytru/so101_bench_groot_eval_${checkpoint}" \
        --repo_root "${run_dir}/lerobot_dataset" \
        --episode_diagnostics_jsonl "${diagnostics_jsonl}" \
        --camera_snapshot_key '' \
        --episode_skip_key '' \
        --camera_snapshot_stdin false \
        --terminal_control_stdin false \
        --headless 2>&1 | tee "${eval_log}"
    eval_status=${PIPESTATUS[0]}
    set -e
    stop_server

    if ((eval_status != 0)); then
        echo "[error] ${checkpoint} evaluator exited ${eval_status}; retaining failed attempt at ${run_dir}." >&2
        return "${eval_status}"
    fi

    if ! "${PYTHON_FOR_TOOLS}" scripts/compute_groot_success_rates.py \
        "${eval_log}" --episodes-jsonl "${TASKS_JSONL}" --format text >"${rates_text}"; then
        echo "[error] ${checkpoint} has no complete success-rate results; treating the run as incomplete." >&2
        return 1
    fi
    if ! "${PYTHON_FOR_TOOLS}" scripts/compute_groot_success_rates.py \
        "${eval_log}" --episodes-jsonl "${TASKS_JSONL}" --format json >"${rates_json}"; then
        echo "[error] ${checkpoint} could not produce valid JSON success rates; treating the run as incomplete." >&2
        return 1
    fi
    if ! "${PYTHON_FOR_TOOLS}" scripts/write_groot_episode_results.py \
        "${eval_log}" \
        --episodes-jsonl "${TASKS_JSONL}" \
        --output-json "${outcomes_json}" \
        --output-tsv "${outcomes_tsv}" \
        --diagnostics-jsonl "${diagnostics_jsonl}" \
        --checkpoint "${checkpoint}"; then
        echo "[error] ${checkpoint} has no complete per-episode results; treating the run as incomplete." >&2
        return 1
    fi
    if ! validate_recorded_dataset "${run_dir}/lerobot_dataset" "$(wc -l <"${TASKS_JSONL}")"; then
        echo "[error] ${checkpoint} recorded an incomplete dataset; treating the run as incomplete." >&2
        return 1
    fi

    {
        printf 'Checkpoint: %s\n' "${checkpoint}"
        printf 'Remote source: %s:%s/%s\n' "${REMOTE_HOST}" "${REMOTE_RUN_DIR}" "${checkpoint}"
        printf 'Remote manifest SHA-256: %s\n' "${manifest}"
        printf 'Recorded LeRobot dataset: %s\n\n' "${run_dir}/lerobot_dataset"
        cat "${rates_text}"
        printf '\nPer-episode outcomes: %s\n' "${outcomes_tsv}"
        printf 'Detailed per-episode diagnostics: %s\n' "${diagnostics_jsonl}"
    } >"${summary}"
    if ! write_completed_marker "${result_dir}" "${run_dir}" "${checkpoint}" "${manifest}"; then
        echo "[error] Could not atomically mark ${checkpoint} complete." >&2
        return 1
    fi
    if ! reconcile_best_checkpoints; then
        echo "[error] ${checkpoint} completed, but best-checkpoint retention needs attention and will retry." >&2
    fi
    echo "[done] ${checkpoint}; see ${summary}"
}

checkpoint_already_completed() {
    [[ -f "${RESULT_ROOT}/$1/completed.json" ]]
}

checkpoint_is_skipped() {
    [[ -f "${RESULT_ROOT}/$1/skipped.json" ]]
}

scan_once() {
    local checkpoints=() checkpoint checkpoint_step manifest
    if ! reconcile_best_checkpoints; then
        echo "[error] Could not fully reconcile retained best checkpoints; continuing evaluation scan." >&2
    fi
    mapfile -t checkpoints < <(list_remote_checkpoints)
    if ((${#checkpoints[@]} == 0)); then
        echo "[scan] No checkpoint-* directories found in ${REMOTE_RUN_DIR}."
        return 0
    fi

    for checkpoint in "${checkpoints[@]}"; do
        if ! [[ "${checkpoint}" =~ ^checkpoint-[0-9]+$ ]]; then
            echo "[scan] Ignoring non-checkpoint directory: ${checkpoint}"
            continue
        fi
        if [[ -n "${ONLY_CHECKPOINT}" && "${checkpoint}" != "${ONLY_CHECKPOINT}" ]]; then
            continue
        fi
        checkpoint_step="${checkpoint#checkpoint-}"
        if [[ -z "${ONLY_CHECKPOINT}" ]] && ((checkpoint_step % EVAL_EVERY_STEPS != 0)); then
            echo "[scan] Ignoring ${checkpoint}; step is not divisible by ${EVAL_EVERY_STEPS}."
            continue
        fi
        if checkpoint_already_completed "${checkpoint}"; then
            echo "[scan] ${checkpoint} already completed."
            continue
        fi
        if checkpoint_is_skipped "${checkpoint}"; then
            echo "[scan] ${checkpoint} explicitly skipped by ${RESULT_ROOT}/${checkpoint}/skipped.json."
            continue
        fi
        echo "[scan] Found unvalidated ${checkpoint}."
        if [[ "${DRY_RUN}" == "1" ]]; then
            echo "[dry-run] would wait for stability, download, and evaluate ${checkpoint}."
            continue
        fi
        if ! manifest="$(checkpoint_is_stable "${checkpoint}")"; then
            continue
        fi
        if ! download_checkpoint "${checkpoint}" "${manifest}"; then
            echo "[error] Could not download ${checkpoint}; it will be retried on the next scan." >&2
            continue
        fi
        if ! run_validation "${checkpoint}" "${manifest}"; then
            echo "[error] Validation failed for ${checkpoint}; no completed marker was written, so it will retry later." >&2
        fi
    done
}

mkdir -p "${RESULT_ROOT}"
echo "[config] remote=${REMOTE_HOST}:${REMOTE_RUN_DIR}"
if [[ -n "${RUN_LABEL}" ]]; then
    echo "[config] run label=${RUN_LABEL}"
fi
echo "[config] local one-checkpoint cache=${LOCAL_MODEL_DIR}"
echo "[config] retained best ${BEST_CHECKPOINT_COUNT}=${BEST_CHECKPOINTS_DIR}"
echo "[config] results=${RESULT_ROOT}"
echo "[config] task/layout=${TASKS_JSONL} / ${LAYOUTS_JSONL}"
echo "[config] local Hugging Face cache=${SERVER_HF_HOME}"
if [[ -n "${ONLY_CHECKPOINT}" ]]; then
    echo "[config] checkpoint target=${ONLY_CHECKPOINT}"
fi
echo "[config] action horizon=${ACTION_HORIZON}"
if [[ -z "${ONLY_CHECKPOINT}" ]]; then
    echo "[config] evaluate every ${EVAL_EVERY_STEPS} steps"
else
    echo "[config] step interval bypassed by explicit checkpoint target"
fi

while true; do
    if ! scan_once; then
        echo "[error] Checkpoint scan failed (for example, temporary RunPod connectivity loss)." >&2
    fi
    if ((ONCE)); then
        break
    fi
    echo "[monitor] Sleeping ${POLL_INTERVAL_S}s before next scan."
    sleep "${POLL_INTERVAL_S}"
done
