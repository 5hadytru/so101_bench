#!/usr/bin/env python3
"""Throughput / memory sweep for run_train.sh.

Each config runs as a real 4-GPU training job; after `--warmup` logged steps the wall
time between step log lines is measured for `--measure` steps, then the job is killed
(before its end-of-run checkpoint). Results append to bench_results.jsonl.

  bench.py "name=ae_bs16 FT_MODE=ae_only DEVICE_BS=16" "name=lora_bs16 FT_MODE=lora DEVICE_BS=16 FLOW_K=1" ...

FT_MODE, FLOW_K (default 8), EXTRA_ARGS (e.g. --activation_checkpointing=false) pass through to run_train.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

WORK = Path(os.environ.get("WORK", "/root/molmoact2-so101"))
STEP_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+).*\[step=(\d+)/")
PEAK_RE = re.compile(r"System/Peak GPU Memory \(MB\)=([\d,]+)")
LOSS_RE = re.compile(r"train/action_flow_loss=([\d.]+)")


def gpu_util() -> list[int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout
    return [int(x) for x in out.split()]


def run_config(spec: dict[str, str], warmup: int, measure: int, timeout_s: int) -> dict:
    name = spec.pop("name")
    env = dict(os.environ)
    env.update(
        NPROC="4", RUN_NAME=f"bench_{name}", SAVE_FOLDER=str(WORK / "bench" / name),
        MAX_DURATION="100000", SAVE_INTERVAL="100000", KEEP="1", WANDB_MODE="disabled",
        LOG_INTERVAL="1",
    )
    env.setdefault("GLOBAL_BS", str(int(spec.get("DEVICE_BS", "2")) * 4))
    env.update(spec)
    if "GLOBAL_BS" not in spec:
        env["GLOBAL_BS"] = str(int(env.get("DEVICE_BS", "2")) * 4)
    log_path = WORK / "logs" / f"bench_{name}.log"
    subprocess.run(["rm", "-rf", str(WORK / "bench" / name)])
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [str(WORK / "run_train.sh")], env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    t0 = time.time()
    steps: dict[int, float] = {}
    peak, losses, utils, error = 0, [], [], None
    try:
        while time.time() - t0 < timeout_s:
            time.sleep(5)
            text = log_path.read_text(errors="replace")
            for line in text.splitlines():
                m = STEP_RE.match(line)
                if m:
                    steps[int(m.group(2))] = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()
            peak = max([peak] + [int(v.replace(",", "")) for v in PEAK_RE.findall(text)])
            losses = [float(v) for v in LOSS_RE.findall(text)]
            if steps and max(steps) > warmup:
                utils.append(gpu_util())
            if "OutOfMemoryError" in text or "CUDA out of memory" in text:
                error = "OOM"
                break
            if proc.poll() is not None:
                error = f"exited rc={proc.returncode}"
                break
            if steps and max(steps) >= warmup + measure:
                break
        else:
            error = "timeout"
    finally:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
                proc.wait(timeout=60)
                break
            except (ProcessLookupError, subprocess.TimeoutExpired):
                continue
        subprocess.run("pkill -9 -f '[l]aunch_scripts/train_lerobot.py' || true", shell=True)
        time.sleep(10)
        subprocess.run(["rm", "-rf", str(WORK / "bench" / name)])

    result = {"name": name, **spec, "global_bs": env["GLOBAL_BS"], "error": error, "peak_mem_mb": peak}
    window = [s for s in sorted(steps) if s > warmup]
    if len(window) >= 3:
        span = steps[window[-1]] - steps[window[0]]
        s_per_step = span / (window[-1] - window[0])
        result.update(
            s_per_step=round(s_per_step, 3),
            samples_per_s=round(int(env["GLOBAL_BS"]) / s_per_step, 2),
            measured_steps=window[-1] - window[0],
            mean_gpu_util=round(sum(sum(u) / len(u) for u in utils) / max(1, len(utils)), 1),
            last_loss=losses[-1] if losses else None,
        )
    with open(WORK / "bench_results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", help="Space-separated KEY=VALUE lists, one config per arg.")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--measure", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    for config in args.configs:
        spec = dict(item.split("=", 1) for item in shlex.split(config))
        run_config(spec, args.warmup, args.measure, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
