#!/usr/bin/env python3
"""Resume SO101 training after storage failures, with checkpoint verification."""

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


WORK = Path("/workspace/gr00t-so101-n1d7")
CHECKPOINTS = WORK / "checkpoints"
LAUNCHER = Path("/workspace/Isaac-GR00T/examples/SO101/finetune.sh")
SCANNER = Path("/root/scan_dataset_reads.py")
# 2026-09-26: budget cut from 51000 to 20000 steps (user); cosine LR now anneals to 0 at 20000.
TARGET = 20000
STEP_RE = re.compile(rb"(?<!\d)(\d+)/%d\s*\[" % TARGET)
STORAGE_ERRORS = (
    "no such device or address", "sigbus", "input/output error",
    "stale file handle", "transport endpoint is not connected", "failed to read",
)
FATAL_ERRORS = ("out of memory", "cuda error", "cuda out of memory")


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def record(message):
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line, flush=True)
    with (WORK / "supervisor.log").open("a") as out:
        out.write(line + "\n")


def latest_checkpoints():
    candidates = []
    for path in CHECKPOINTS.glob("checkpoint-*"):
        if path.is_dir() and re.fullmatch(r"checkpoint-\d+", path.name):
            candidates.append((int(path.name.split("-")[-1]), path))
    return sorted(candidates, reverse=True)


def checkpoint_complete(path):
    """A crash mid-save leaves readable but partial checkpoints; require the files a resume needs."""
    shards = [d for d in path.glob("global_step*") if d.is_dir() and any(d.iterdir())]
    missing = [name for name in ("trainer_state.json", "latest", "config.json") if not (path / name).is_file()]
    if missing or not shards:
        record(f"checkpoint incomplete {path.name}: missing={missing} global_step_dirs={len(shards)}")
        return False
    return True


def verify_checkpoint(path):
    if not checkpoint_complete(path):
        return False
    results = WORK / f"scan_{path.name}_{stamp()}.jsonl"
    log = WORK / f"scan_{path.name}_{stamp()}.log"
    with log.open("w") as out:
        proc = subprocess.Popen(
            ["/usr/bin/python3", str(SCANNER), str(path),
             "--results", str(results), "--workers", "4", "--timeout", "600"],
            stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            proc.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            stop_group(proc)
            record(f"checkpoint scan timed out: {path} ({log})")
            return False
    last = log.read_text(errors="replace").splitlines()[-1:] or [""]
    good = proc.returncode == 0 and last[0].startswith("DONE ") and results.stat().st_size == 0
    record(f"checkpoint verification {path.name}: {'PASS' if good else 'FAIL'} ({log})")
    return good


def valid_latest_checkpoint():
    while True:
        candidates = latest_checkpoints()
        if not candidates:
            record("STOP: no readable checkpoint remains")
            return None
        _, path = candidates[0]
        if verify_checkpoint(path):
            return path
        quarantine = CHECKPOINTS / "_unreadable"
        quarantine.mkdir(exist_ok=True)
        dest = quarantine / f"{path.name}_{stamp()}"
        shutil.move(str(path), str(dest))
        record(f"quarantined unreadable checkpoint {path} -> {dest}")


def gpu_free_within(seconds=600):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and not result.stdout.strip():
            return True
        time.sleep(10)
    record("STOP: GPU compute processes remained after 10 minutes")
    return False


def training_groups(root_pid, output_dir=None):
    groups = {root_pid}
    target = str(output_dir).encode() if output_dir is not None else None
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
            if target is not None and b"--output_dir" in args:
                index = args.index(b"--output_dir")
                if index + 1 < len(args) and args[index + 1] == target:
                    groups.add(os.getpgid(int(entry.name)))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    # Include child sessions such as the scanner's per-file dd processes.
    table = subprocess.run(["ps", "-eo", "pid=,ppid=,pgid="],
                           capture_output=True, text=True, timeout=30)
    children = {}
    for line in table.stdout.splitlines():
        try:
            pid, parent, pgid = map(int, line.split())
        except ValueError:
            continue
        children.setdefault(parent, []).append((pid, pgid))
    pending = [root_pid]
    while pending:
        for pid, pgid in children.get(pending.pop(), ()):
            groups.add(pgid)
            pending.append(pid)
    return groups


def stop_group(proc, output_dir=None):
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
        record(f"process group {proc.pid} still has uninterruptible processes")


def run_attempt():
    name = f"train_{stamp()}.log"
    log = WORK / name
    link = WORK / "train.log"
    tmp_link = WORK / ".train.log.next"
    if link.exists() and not link.is_symlink():
        link.rename(WORK / f"train_before_supervisor_{stamp()}.log")
    tmp_link.unlink(missing_ok=True)
    tmp_link.symlink_to(name)
    tmp_link.replace(link)
    env = os.environ.copy()
    env.update({
        "SO101_RESUME": "1", "SAVE_STEPS": "1000", "MAX_STEPS": str(TARGET),
        "PER_GPU_BATCH_SIZE": "160", "GRADIENT_ACCUMULATION_STEPS": "1",
        "USE_WANDB": "1", "NO_ALBUMENTATIONS_UPDATE": "1",
    })
    with log.open("w") as out:
        proc = subprocess.Popen(
            ["bash", str(LAUNCHER)],
            stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True,
        )
        record(f"attempt pid={proc.pid}, workers={env.get('DATALOADER_NUM_WORKERS', '4')}, log={log}")
        step = 0
        baseline_step = None
        made_progress = False
        last_advance = time.monotonic()
        start = last_advance
        position = 0
        pending = b""
        while proc.poll() is None:
            with log.open("rb") as source:
                source.seek(position)
                data = source.read()
                position = source.tell()
            matches = STEP_RE.findall(pending + data)
            pending = (pending + data)[-128:]
            if matches:
                observed = max(map(int, matches))
                if baseline_step is None:
                    baseline_step = observed
                    step = observed
                elif observed > step:
                    step = observed
                    made_progress = True
                    last_advance = time.monotonic()
            grace = 3600 if not made_progress else 1200
            age = time.monotonic() - (start if not made_progress else last_advance)
            if age > grace:
                reason = f"watchdog: no step progress for {int(age)}s at step {step}"
                record(reason)
                stop_group(proc, CHECKPOINTS)
                return step, reason, log
            time.sleep(10)
        with log.open("rb") as source:
            source.seek(position)
            data = source.read()
        matches = STEP_RE.findall(pending + data)
        if matches:
            step = max(step, *map(int, matches))
        reason = f"exit={proc.returncode} at step {step}"
        if proc.returncode != 0:
            stop_group(proc, CHECKPOINTS)
        record(reason)
        return step, reason, log


def main():
    if not SCANNER.is_file():
        record(f"STOP: checkpoint scanner missing: {SCANNER}")
        return 1
    if not os.environ.get("DATALOADER_NUM_WORKERS"):
        record("STOP: set DATALOADER_NUM_WORKERS to the benchmark winner")
        return 1
    restart_times = []
    prior_failed_step = None
    while True:
        checkpoint = valid_latest_checkpoint()
        if checkpoint is None or not gpu_free_within():
            return 1
        step, reason, log = run_attempt()
        if step >= TARGET and reason.startswith("exit=0"):
            record(f"COMPLETE: reached step {TARGET}")
            return 0
        content = log.read_text(errors="replace").lower()
        if any(error in content for error in FATAL_ERRORS):
            record(f"STOP: non-storage training error at step {step}: {reason}")
            return 1
        storage_related = any(error in content for error in STORAGE_ERRORS)
        if not storage_related and not reason.startswith("watchdog"):
            record(f"STOP: unexpected non-storage failure at step {step}: {reason}")
            return 1
        if prior_failed_step == step:
            record(f"STOP: same failure recurred twice without progress at step {step}")
            return 1
        prior_failed_step = step
        now = time.monotonic()
        restart_times = [when for when in restart_times if now - when < 6 * 3600]
        if len(restart_times) >= 5:
            record("STOP: restart budget exhausted (5 in 6h)")
            return 1
        restart_times.append(now)
        record(f"RESTART {len(restart_times)}/5 in 6h after {reason}; backoff 120s")
        time.sleep(120)


if __name__ == "__main__":
    sys.exit(main())
