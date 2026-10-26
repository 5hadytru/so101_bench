#!/usr/bin/env python3
"""Throughput / VRAM sweep for the LaWAM so101_bench_sim fine-tune (LeRobot adapter, 4x A100).

Each point launches pod/run_train.sh (a real run from jialei02/lawam-pretrain-lerobot), reads LeRobot's own
training log lines (every 10 optimizer updates: step_s, data_s, updt_s, smp/s = global samples per second,
mem_gb), skips WARMUP_LOGS lines, averages the next MEASURED_LOGS, samples nvidia-smi for peak memory and
utilization, and kills the run before its first checkpoint. Results go to records/bench_results.jsonl.

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

ROOT = Path(os.environ.get("ROOT", "/root/lawam"))
RESULTS = ROOT / "records" / "bench_results.jsonl"
GPUS = 4
WARMUP_LOGS, MEASURED_LOGS = int(os.environ.get("WARMUP_LOGS", "2")), int(os.environ.get("MEASURED_LOGS", "4"))
TIMEOUT_S = int(os.environ.get("POINT_TIMEOUT_S", "2400"))
LOG_RE = re.compile(r"step:\S+ smpl:\S+ .*")
FIELD_RE = re.compile(r"(step_s|data_s|updt_s|prep_s|smp/s|mem_gb|loss):([0-9.]+)")


def point(name, micro, accum, **env):
    return {"name": name, "micro": micro, "accum": accum, **env}


# Defaults: AH=36 (1.2 s at 30 fps, the official RoboTwin chunk), WORKERS=8.
SWEEP = [
    point("m16_a2", 16, 2),           # global 128
    point("m32_a1", 32, 1),           # global 128
    point("m8_a4", 8, 4),             # global 128
    point("m64_a1", 64, 1),           # global 256
    point("m32_a2", 32, 2),           # global 256: the official LIBERO SFT shape (32 per GPU, 256 global)
    point("m32_a1_w16", 32, 1, WORKERS="16"),
    point("m32_a1_ah24", 32, 1, AH="24"),
]


def load_results() -> list[dict]:
    return [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()] if RESULTS.exists() else []


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
    env_over = {k: v for k, v in p.items() if k not in ("name", "micro", "accum")}
    out = ROOT / "bench" / p["name"]
    subprocess.run(["rm", "-rf", str(out), str(ROOT / "runs" / f"bench-{p['name']}")])
    out.mkdir(parents=True)
    env = {**os.environ, "MICRO": str(p["micro"]), "ACCUM": str(p["accum"]), "UPDATES": "100000",
           "SAVE_UPDATES": "100000", "RUN_NAME": f"bench-{p['name']}", "WANDB": "0", **env_over}
    log_path = out / "stdout.log"
    result = {"name": p["name"], "micro": p["micro"], "accum": p["accum"], "global_batch": p["micro"] * p["accum"] * GPUS,
              "env": env_over, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    sampler = GpuSampler()
    sampler.start()
    t0 = time.time()
    with log_path.open("w") as log:
        proc = subprocess.Popen(["bash", str(ROOT / "pod" / "run_train.sh")], env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    status, detail, lines = "timeout", "", []
    try:
        while time.time() - t0 < TIMEOUT_S:
            time.sleep(5)
            text = log_path.read_text(errors="replace")
            if "CUDA out of memory" in text or "OutOfMemoryError" in text:
                status, detail = "oom", "\n".join(text.splitlines()[-30:])
                break
            lines = LOG_RE.findall(text)
            if len(lines) > WARMUP_LOGS:
                sampler.record = True
            if len(lines) >= WARMUP_LOGS + MEASURED_LOGS:
                status = "ok"
                break
            if proc.poll() is not None:
                status, detail = "error", "\n".join(text.splitlines()[-40:])
                break
    finally:
        kill_all(proc)
        sampler.stop.set()
        sampler.join()
    result.update(status=status, detail=detail, peak_nvidia_smi_mib=sampler.peak,
                  mean_gpu_util=round(sum(sampler.util) / len(sampler.util), 1) if sampler.util else None,
                  log_lines=lines[-(MEASURED_LOGS + 1):])
    if status == "ok":
        measured = [dict((k, float(v)) for k, v in FIELD_RE.findall(l)) for l in lines[WARMUP_LOGS:WARMUP_LOGS + MEASURED_LOGS]]
        mean = lambda k: round(sum(m[k] for m in measured if k in m) / len(measured), 4)
        sps = mean("smp/s")
        result.update(samples_per_s=sps, s_per_update=round(result["global_batch"] / sps, 2) if sps else None,
                      step_s=mean("step_s"), data_s=mean("data_s"), updt_s=mean("updt_s"), mem_gb=mean("mem_gb"))
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
