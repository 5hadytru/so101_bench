#!/usr/bin/env bash
# Local (3090) G0.5 inference test through our eval client: AR vs flow latency and open-loop error on the
# 48 training query frames, for one checkpoint. Takes the shared local-GPU validation lock so it never overlaps
# a monitor's Isaac validation.
#   CKPT=... [TASK_CONFIG=...] LABEL=... bash scripts/G05_fine_tuning/local_server_test.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
CKPT=${CKPT:?}; LABEL=${LABEL:?}; TASK_CONFIG=${TASK_CONFIG:-}
OUT=outputs/g05_local_tests/${LABEL}; mkdir -p "${OUT}"
exec {fd}>>logs/.local_gpu_validation.lock
echo "$(date) waiting for the local GPU lock"; flock "${fd}"; echo "$(date) lock acquired"
G05=${HOME}/workspace/GalaxeaVLA
for mode in ar flow; do
    args=(--repo "${G05}" --ckpt_path "${CKPT}" --mode "${mode}" --replan_steps 15 --port 8010)
    [[ -n "${TASK_CONFIG}" ]] && args+=(--task_config "${TASK_CONFIG}")
    systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=0 env -u HF_TOKEN HF_HUB_OFFLINE=1 "${G05}/.venv/bin/python" scripts/g05_server.py "${args[@]}" > "${OUT}/server_${mode}.log" 2>&1 &
    pid=$!
    for _ in $(seq 120); do curl -sf --max-time 5 localhost:8010/act >/dev/null && break; kill -0 ${pid} 2>/dev/null || break; sleep 5; done
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > "${OUT}/gpu_${mode}.txt"
    ~/env_isaaclab_51/bin/python scripts/open_loop_server_check.py outputs/open_loop_checks/wm_query_frames.npz \
        --joint_frame lerobot --port 8010 --label "${LABEL}_${mode}" --out "${OUT}/open_loop_${mode}.json" > "${OUT}/check_${mode}.log" 2>&1
    grep -o "inference: [0-9.]* ms" "${OUT}/check_${mode}.log" | awk '{print $2}' | sort -n | awk '{a[NR]=$1} END{if(NR) print "'${mode}' latency ms: n="NR" median="a[int((NR+1)/2)]" p90="a[int(NR*0.9)]}' | tee -a "${OUT}/summary.txt"
    kill ${pid}; wait ${pid} 2>/dev/null
done
echo "$(date) LOCAL_TEST_DONE"
