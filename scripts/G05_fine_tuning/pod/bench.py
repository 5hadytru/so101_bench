#!/usr/bin/env python3
"""Throughput / VRAM sweep for G0.5 fine-tuning on so101_bench_sim (4x A100, global batch 128).

Each point launches pod/run_train.sh (a real run: g05-base weights, augmentation on), times optimizer
steps after a warmup from the trainer's tqdm "Step N" lines, samples nvidia-smi for peak memory, and kills
the run before its first checkpoint. Results go to records/bench_results.jsonl.

  bench.py list | bench.py run [NAME ...]
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", "/root/g05"))
RESULTS = ROOT / "records" / "bench_results.jsonl"
GPUS, GLOBAL = 4, 128
WARMUP, MEASURED = int(os.environ.get("WARMUP_STEPS", "4")), int(os.environ.get("MEASURED_STEPS", "10"))
TIMEOUT_S = int(os.environ.get("POINT_TIMEOUT_S", "2700"))
STEP_RE = re.compile(r"Step (\d+)")


def point(name, micro, **env):
    return {"name": name, "micro": micro, **env}


SWEEP = [
    point("m16_obs6", 16),                         # repo so100 recipe batch, g05-base memory on
    point("m8_obs6", 8),
    point("m32_obs6", 32),
    point("m16_obs1", 16, OBS="1"),                # memory off: the cost of the 6-frame history
    point("m16_obs6_noViTckpt", 16, CKPT_VISION="false"),
    point("m16_obs6_compile", 16, COMPILE="true"),  # fails: dynamo cannot trace the action tokenizer
    # Round 2 (micro 16 OOMs at 80 GB; micro 8 peaks at 79.6 GB): memory headroom.
    point("m8_obs1", 8, OBS="1"),
    point("m4_obs6", 4),
    point("m8_obs6_8bit", 8, OPT8BIT="true"),
    point("m16_obs6_8bit", 16, OPT8BIT="true"),
]


def load_results() -> list[dict]:
    return [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()] if RESULTS.exists() else []


def best_base() -> dict:
    ok = [r for r in load_results() if r["status"] == "ok"]
    if not ok:
        raise RuntimeError("no successful base point yet")
    best = max(ok, key=lambda r: r["samples_per_s"])
    return {"micro": best["micro"], **best["env"]}


class GpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.util, self.record, self.stop = [0] * GPUS, [], False, threading.Event()

    def run(self):
        while not self.stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.strip().splitlines()
            utils = []
            for i, line in enumerate(out[:GPUS]):
                mem, util = (int(x) for x in line.split(","))
                self.peak[i] = max(self.peak[i], mem)
                utils.append(util)
            if self.record and utils:
                self.util.append(sum(utils) / len(utils))
            self.stop.wait(2.0)


def kill_all(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        if proc.poll() is not None:
            break
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            pass
    for _ in range(30):
        pids = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.split()
        if not pids:
            return
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
        time.sleep(2)


def run_point(p: dict) -> dict:
    p = dict(p)
    if p["micro"] is None:
        base = best_base()
        p["micro"] = base.pop("micro")
        for k, v in base.items():
            p.setdefault(k, v)
    env_over = {k: v for k, v in p.items() if k not in ("name", "micro")}
    accum = GLOBAL // (GPUS * p["micro"])
    out = ROOT / "bench" / p["name"]
    subprocess.run(["rm", "-rf", str(out)])
    out.mkdir(parents=True)
    env = {**os.environ, "MICRO": str(p["micro"]), "ACCUM": str(accum), "MAX_STEPS": "100000", "SAVE_EVERY": "100000",
           "RUN_NAME": p["name"], "OUT_ROOT": str(out), "WANDB": "0", "SAVE_EVERY": "100000", **{k: str(v) for k, v in env_over.items()}}
    log_path = out / "stdout.log"
    result = {"name": p["name"], "micro": p["micro"], "accum": accum, "global_batch": p["micro"] * accum * GPUS,
              "env": env_over, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    sampler = GpuSampler()
    sampler.start()
    t0 = time.time()
    with log_path.open("w") as log:
        proc = subprocess.Popen(["bash", str(ROOT / "pod" / "run_train.sh"), "logger.log_steps=1"], env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    marks: dict[int, float] = {}
    status, detail = "timeout", ""
    try:
        while time.time() - t0 < TIMEOUT_S:
            time.sleep(3)
            text = log_path.read_text(errors="replace")
            if "CUDA out of memory" in text or "OutOfMemoryError" in text:
                status, detail = "oom", "\n".join(text.splitlines()[-30:])
                break
            for m in STEP_RE.finditer(text):
                marks.setdefault(int(m.group(1)), time.time())
            if len(marks) > WARMUP:
                sampler.record = True
            if len(marks) >= WARMUP + MEASURED:
                status = "ok"
                break
            if proc.poll() is not None:
                status, detail = "error", "\n".join(text.splitlines()[-40:])
                break
    finally:
        kill_all(proc)
        sampler.stop.set()
        sampler.join()
    steps = sorted(marks)
    result.update(status=status, detail=detail, peak_nvidia_smi_mib=sampler.peak,
                  mean_gpu_util=round(sum(sampler.util) / len(sampler.util), 1) if sampler.util else None,
                  first_step_s=round(marks[steps[0]] - t0, 1) if steps else None)
    if status == "ok":
        window = steps[WARMUP:]
        dt = marks[window[-1]] - marks[window[0]]
        n = window[-1] - window[0]
        result.update(s_per_step=round(dt / n, 2), samples_per_s=round(result["global_batch"] * n / dt, 2))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"[bench] {p['name']}: {result['status']} {result.get('samples_per_s')} samples/s peak {sampler.peak}", flush=True)
    return result


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("list", "run"):
        sys.exit(__doc__)
    if sys.argv[1] == "list":
        for p in SWEEP:
            print(json.dumps(p))
        return
    names, done = set(sys.argv[2:]), {r["name"] for r in load_results()}
    for p in SWEEP:
        if (names and p["name"] not in names) or (not names and p["name"] in done):
            continue
        try:
            run_point(p)
        except Exception as exc:  # noqa: BLE001
            print(f"[bench] {p['name']} failed to launch: {exc!r}", flush=True)
            with RESULTS.open("a") as f:
                f.write(json.dumps({"name": p["name"], "micro": p["micro"], "status": "launch_error", "detail": repr(exc)}) + "\n")


if __name__ == "__main__":
    main()
