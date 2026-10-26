#!/usr/bin/env python3
"""Compare SO101 loader worker counts from the same step-4000 checkpoint."""

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/workspace/gr00t-so101-n1d7")
LAUNCHER = Path("/workspace/Isaac-GR00T/examples/SO101/finetune.sh")
CHECKPOINT = ROOT / "checkpoints/checkpoint-4000"
STEP_RE = re.compile(rb"(?<!\d)(\d+)/51000\s*\[")


def gpu_free():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0 and not result.stdout.strip()


def training_groups(root_pid, output_dir):
    groups = {root_pid}
    target = str(output_dir).encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
            if b"--output_dir" in args:
                index = args.index(b"--output_dir")
                if index + 1 < len(args) and args[index + 1] == target:
                    groups.add(os.getpgid(int(entry.name)))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return groups


def stop_group(proc, output_dir):
    groups = training_groups(proc.pid, output_dir)
    for pgid in groups:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    for pgid in groups | training_groups(proc.pid, output_dir):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Training processes stuck after SIGKILL")


def run_one(workers, duration=1800):
    bench_dir = ROOT / "bench" / f"w{workers}"
    bench_dir.mkdir(parents=True, exist_ok=True)
    link = bench_dir / CHECKPOINT.name
    if not link.exists():
        link.symlink_to(Path("../../checkpoints/checkpoint-4000"))
    elif not link.is_symlink() or link.resolve() != CHECKPOINT:
        raise RuntimeError(f"Unexpected checkpoint path: {link}")
    log = bench_dir / "train.log"
    env = os.environ.copy()
    env.update({
        "SO101_OUTPUT_DIR": str(bench_dir), "SO101_RESUME": "1",
        "DATALOADER_NUM_WORKERS": str(workers), "SAVE_STEPS": "1000000",
        "MAX_STEPS": "51000", "USE_WANDB": "0", "SO101_SAVE_ONLY_MODEL": "0",
        "PER_GPU_BATCH_SIZE": "160", "GRADIENT_ACCUMULATION_STEPS": "1",
        "NO_ALBUMENTATIONS_UPDATE": "1",
    })
    with log.open("w") as output:
        proc = subprocess.Popen(["bash", str(LAUNCHER)], stdout=output,
                                stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        print(f"started workers={workers} pid={proc.pid} log={log}", flush=True)
        start = time.monotonic()
        while time.monotonic() - start < duration:
            if proc.poll() is not None:
                stop_group(proc, bench_dir)
                raise RuntimeError(f"workers={workers} exited early ({proc.returncode}); see {log}")
            time.sleep(10)
        stop_group(proc, bench_dir)
    content = log.read_bytes()
    steps = [int(s) for s in STEP_RE.findall(content)]
    if not steps or max(steps) <= 4010:
        raise RuntimeError(f"workers={workers} made too little progress; see {log}")
    if any(marker in content for marker in (b"Traceback", b"No such device or address", b"SIGBUS")):
        raise RuntimeError(f"workers={workers} had read/training errors; see {log}")
    print(f"completed workers={workers} max_step={max(steps)}", flush=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if gpu_free():
            return log
        time.sleep(10)
    raise RuntimeError(f"GPU compute processes remained after workers={workers}")


def main():
    if not CHECKPOINT.is_dir():
        raise RuntimeError(f"Missing checkpoint: {CHECKPOINT}")
    if not gpu_free():
        raise RuntimeError("GPUs are occupied before benchmarking")
    duration = int(os.environ.get("SO101_BENCH_SECONDS", "1800"))
    logs = [run_one(workers, duration) for workers in (1, 2, 3)]
    subprocess.run([sys.executable, str(ROOT / "bench_stats.py"),
                    *map(str, logs)], check=True)


if __name__ == "__main__":
    main()
