#!/usr/bin/env python3
"""Throughput / memory sweep for the pi0.5 run_train.sh.

Each config runs as a real 4-GPU training job with --log_freq=1 and no checkpoints.
After `--warmup` logged steps, it measures wall time per step (from log timestamps)
and the mean `updt_s` (compute) / `data_s` (waiting on the loader) that LeRobot logs,
plus peak GPU memory from nvidia-smi, then kills the job. Results append to
bench_results.jsonl.

  bench.py "name=bs8 DEVICE_BS=8 NUM_WORKERS=8" "name=bs16 DEVICE_BS=16" ...
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

WORK = Path("/workspace/pi05-so101")
TS_RE = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
KV_RE = re.compile(r"([a-z_/]+):([0-9.eE+-]+K?)(?=\s|$)")


def parse_line(line: str) -> tuple[int, float, dict[str, float]] | None:
    """Parse a LeRobot MetricsTracker log line (step:5 smpl:.. loss:.. data_s:.. updt_s:.. mem_gb:..)."""
    if " step:" not in line or "loss:" not in line:
        return None
    ts = TS_RE.search(line)
    kv = dict(KV_RE.findall(line))
    if not ts or "step" not in kv or kv["step"].endswith("K"):
        return None
    vals = {k: float(v) for k, v in kv.items() if not v.endswith("K")}
    return int(vals["step"]), datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S").timestamp(), vals


def gpu_mem_util() -> tuple[int, float]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.split("\n")
    rows = [list(map(float, r.split(","))) for r in out if r.strip()]
    return int(max(r[0] for r in rows)), sum(r[1] for r in rows) / len(rows)


def run_config(spec: dict[str, str], warmup: int, measure: int, timeout_s: int) -> dict:
    name = spec.pop("name")
    env = dict(os.environ)
    env.update(
        NPROC="4", RUN_NAME=f"bench_{name}", OUTPUT_DIR=str(WORK / "bench" / name),
        STEPS="100000", SAVE_FREQ="1000000", LOG_FREQ="1", WANDB_ENABLE="false", WANDB_MODE="disabled",
    )
    env.update(spec)
    # DDP and FSDP (dp_shard) both keep a data-parallel world of 4 processes.
    global_bs = int(env.get("DEVICE_BS", "8")) * 4 * int(env.get("ACCUM", "1"))
    log_path = WORK / "logs" / f"bench_{name}.log"
    subprocess.run(["rm", "-rf", str(WORK / "bench" / name)])
    with open(log_path, "w") as log:
        proc = subprocess.Popen([str(WORK / "run_train.sh")], env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    t0 = time.time()
    rows: dict[int, tuple[float, float, float, float, float]] = {}
    peak_mem, utils, error = 0, [], None
    try:
        while time.time() - t0 < timeout_s:
            time.sleep(4)
            text = log_path.read_text(errors="replace")
            for line in text.splitlines():
                parsed = parse_line(line)
                if parsed:
                    step, ts, vals = parsed
                    rows[step] = (ts, vals.get("loss", float("nan")), vals.get("updt_s", float("nan")),
                                  vals.get("data_s", float("nan")), vals.get("mem_gb", float("nan")))
            mem, util = gpu_mem_util()
            peak_mem = max(peak_mem, mem)
            if rows and max(rows) > warmup:
                utils.append(util)
            if "OutOfMemoryError" in text or "CUDA out of memory" in text:
                error = "OOM"
                break
            if proc.poll() is not None:
                error = f"exited rc={proc.returncode}"
                break
            if rows and max(rows) >= warmup + measure:
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
        subprocess.run("pkill -9 -f '[l]erobot-train' || true", shell=True)
        time.sleep(10)
        subprocess.run(["rm", "-rf", str(WORK / "bench" / name)])

    result = {"name": name, **spec, "global_bs": global_bs, "error": error, "peak_mem_mb": peak_mem}
    window = [s for s in sorted(rows) if s > warmup]
    if len(window) >= 3:
        span = rows[window[-1]][0] - rows[window[0]][0]
        steps = window[-1] - window[0]
        s_per_step = span / steps if span > 0 else float("nan")
        result.update(
            s_per_step=round(s_per_step, 3),
            samples_per_s=round(global_bs / s_per_step, 2) if span > 0 else None,
            mean_updt_s=round(sum(rows[s][2] for s in window) / len(window), 3),
            mean_data_s=round(sum(rows[s][3] for s in window) / len(window), 3),
            torch_mem_gb=max(rows[s][4] for s in window),
            measured_steps=steps,
            mean_gpu_util=round(sum(utils) / max(1, len(utils)), 1),
            last_loss=rows[window[-1]][1],
        )
    with open(WORK / "bench_results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measure", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    for config in args.configs:
        spec = dict(item.split("=", 1) for item in shlex.split(config))
        run_config(spec, args.warmup, args.measure, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
