#!/usr/bin/env python3
"""Keep the MolmoAct2 (base checkpoint) so101_bench full fine-tune running, and export checkpoints.

Adapted from scripts/MolmoAct2_full_fine_tune/pod/supervise.py for a pod whose /workspace is
a MooseFS network mount: everything lives on the container disk (/root), and each exported
checkpoint is also copied (best effort, one kept) to /workspace so a container reset does not
lose the run.

Loop: preflight (GPUs free, disk space, dataset spot-check, newest checkpoint fully
readable) -> launch run_train.sh (the trainer auto-resumes from the newest stepN) ->
watch step progress -> on a crash or a stall, restart with backoff.

Every regular (non-ephemeral) checkpoint is exported in the background to a bf16 HF
checkpoint under export/, which the local monitor pulls and marks PULLED. Exports
are pruned oldest-first once pulled, keeping EXPORT_KEEP.

Stops (status.json state=STOPPED) instead of looping when the same failure repeats
without progress, the restart budget is exhausted, or a preflight check fails; the
hourly check-up then debugs it.

Settings come from supervisor_config.json (re-read before every launch). Stdlib only,
so it runs under /usr/bin/python3 independently of the training venv.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WORK = Path("/root/molmoact2-so101")
# This pod has no network volume (its /workspace was MooseFS, avoided), so backups are off unless set.
BACKUP = Path(os.environ["BACKUP_DIR"]) if os.environ.get("BACKUP_DIR") else None
CONFIG = WORK / "supervisor_config.json"
STATUS = WORK / "status.json"
LOG = WORK / "supervisor.log"
EXPORTS = WORK / "export"
DATASET = Path("/root/datasets/lerobot/5hadytru/so101_bench_sim_WM")
VENV_PY = Path("/root/molmoact2-venv/bin/python")

STEP_RE = re.compile(r"\[step=(\d+)/(\d+)")
SAVED_RE = re.compile(r"Checkpoint saved to (\S+/step(\d+))\s*$")
STALL_S = 20 * 60          # no new step for this long -> restart
STARTUP_GRACE_S = 45 * 60  # first step after launch (model load, FSDP, loaders)
RESTARTS_PER_6H = 6
BACKOFF_S = 120
MIN_FREE_GB = int(os.environ.get("MIN_FREE_GB", "60"))  # room for the next checkpoint plus an export


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record(message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    with LOG.open("a") as out:
        out.write(line + "\n")


def load_config() -> dict:
    cfg = json.loads(CONFIG.read_text())
    for key in ("RUN_NAME", "MAX_DURATION", "SAVE_INTERVAL"):
        if key not in cfg:
            raise KeyError(f"{CONFIG} is missing {key}")
    return cfg


class Status:
    def __init__(self) -> None:
        self.data = json.loads(STATUS.read_text()) if STATUS.is_file() else {}
        self.data.setdefault("restarts", [])

    def update(self, **kwargs) -> None:
        self.data.update(kwargs, updated_at=now(), supervisor_pid=os.getpid())
        tmp = STATUS.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(STATUS)


def run_dir(cfg: dict) -> Path:
    return WORK / "checkpoints" / cfg["RUN_NAME"]


def checkpoints(cfg: dict) -> list[tuple[int, Path]]:
    found = []
    for path in run_dir(cfg).glob("step*"):
        m = re.fullmatch(r"step(\d+)", path.name)
        if m and path.is_dir():
            found.append((int(m.group(1)), path))
    return sorted(found, reverse=True)


def read_all(path: Path, timeout_s: int = 1800) -> bool:
    """Read every byte of a checkpoint (catches lost chunks / I/O errors) under a timeout."""
    proc = subprocess.run(
        ["timeout", str(timeout_s), "bash", "-c", f"find {str(path)!r} -type f -print0 | xargs -0 -P 16 -n 8 cat > /dev/null"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        record(f"read check failed for {path}: rc={proc.returncode} {proc.stderr.strip()[-400:]}")
    return proc.returncode == 0


def checkpoint_ok(path: Path) -> bool:
    required = [path / "config.yaml", path / "model_and_optim" / ".metadata"]
    missing = [str(p.relative_to(path)) for p in required if not p.is_file()]
    if missing or not any((path / "train").glob("*")):
        record(f"checkpoint {path.name} incomplete: missing={missing}")
        return False
    return read_all(path)


def newest_valid_checkpoint(cfg: dict) -> Path | None:
    while True:
        found = checkpoints(cfg)
        if not found:
            return None
        _, path = found[0]
        if checkpoint_ok(path):
            record(f"checkpoint {path.name} verified readable")
            return path
        quarantine = run_dir(cfg) / "_unreadable"
        quarantine.mkdir(exist_ok=True)
        dest = quarantine / f"{path.name}_{int(time.time())}"
        shutil.move(str(path), str(dest))
        record(f"QUARANTINED unreadable checkpoint {path} -> {dest}")


def gpus_free(wait_s: int = 600) -> bool:
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60)
        if out.returncode == 0 and not out.stdout.strip():
            return True
        time.sleep(10)
    return False


def free_gb() -> float:
    return shutil.disk_usage(WORK).free / 1e9


def _exports_oldest_first() -> list[Path]:
    def step_of(path: Path) -> int:
        m = re.search(r"step(\d+)$", path.name)
        return int(m.group(1)) if m else -1

    return sorted((p for p in EXPORTS.glob("*-step*") if p.is_dir() and not p.name.endswith(".tmp")), key=step_of)


def prune_exports(keep: int, force: bool = False) -> None:
    """Delete the oldest pulled exports beyond `keep`.

    Unpulled exports are deleted (oldest first, always keeping the newest two) only when
    forced or when /workspace drops below MIN_FREE_GB, so a stalled local monitor can
    never fill the disk the training checkpoints need.
    """
    exports = _exports_oldest_first()
    for path in exports[: max(0, len(exports) - keep)]:
        pulled = (path / "PULLED").is_file()
        if pulled or force:
            shutil.rmtree(path, ignore_errors=True)
            record(f"pruned export {path.name}{'' if pulled else ' (unpulled, forced)'}")
    for path in _exports_oldest_first()[:-2]:
        if free_gb() >= MIN_FREE_GB:
            break
        shutil.rmtree(path, ignore_errors=True)
        record(f"pruned export {path.name} (low disk: {free_gb():.0f} GB free)")


def dataset_spot_check() -> bool:
    proc = subprocess.run(
        [str(VENV_PY), str(WORK / "verify_dataset.py"), "--root", str(DATASET),
         "--manifest", str(WORK / "dataset_manifest.json"), "--sample", "48", "--parquet"],
        capture_output=True, text=True, timeout=1800,
    )
    ok = proc.returncode == 0 and "DATASET_OK" in proc.stdout
    record(f"dataset spot-check {'PASS' if ok else 'FAIL'}: {proc.stdout.strip().splitlines()[-1:] or proc.stderr[-300:]}")
    return ok


def preflight(cfg: dict, status: Status) -> str | None:
    if not VENV_PY.is_file():
        return f"training venv missing ({VENV_PY}); run bootstrap_after_restart.sh"
    if not gpus_free():
        return "GPU compute processes still running after 10 min"
    if free_gb() < MIN_FREE_GB:
        prune_exports(int(cfg.get("EXPORT_KEEP", 6)))
    if free_gb() < MIN_FREE_GB:
        prune_exports(2, force=True)
    if free_gb() < MIN_FREE_GB:
        return f"only {free_gb():.0f} GB free on {WORK}"
    if not dataset_spot_check():
        return "dataset spot-check failed (see supervisor.log); re-download the bad files from HF"
    ckpt = newest_valid_checkpoint(cfg)
    status.update(resume_from=str(ckpt) if ckpt else None)
    return None


class Exporter:
    """One background export at a time; queue regular checkpoints as they are saved."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.queue: list[Path] = []
        self.proc: subprocess.Popen | None = None
        self.current: Path | None = None
        self.backup: subprocess.Popen | None = None

    def enqueue(self, step_dir: Path) -> None:
        name = f"{self.cfg['RUN_NAME']}-{step_dir.name}"
        if (EXPORTS / name / "EXPORT_OK").is_file() or step_dir in self.queue:
            return
        self.queue.append(step_dir)

    def poll(self) -> None:
        if self.proc is not None and self.proc.poll() is not None:
            ok = self.proc.returncode == 0
            record(f"export {self.current.name}: {'OK' if ok else f'FAILED rc={self.proc.returncode}'}")
            if ok:
                if BACKUP is not None:
                    self.start_backup(self.current)
            self.proc, self.current = None, None
            prune_exports(int(self.cfg.get("EXPORT_KEEP", 6)))
        if self.proc is None and self.queue:
            self.current = self.queue.pop(0)
            if not self.current.is_dir():
                record(f"export skipped, {self.current} no longer exists")
                self.current = None
                return
            log = (WORK / "logs" / f"export_{self.current.name}.log").open("w")
            self.proc = subprocess.Popen(
                ["bash", str(WORK / "export_checkpoint.sh"), str(self.current),
                 str(EXPORTS / f"{self.cfg['RUN_NAME']}-{self.current.name}")],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            record(f"exporting {self.current.name}")

    def start_backup(self, step_dir: Path) -> None:
        """Copy the exported checkpoint to /workspace in the background (best effort, newest kept)."""
        if self.backup is not None and self.backup.poll() is None:
            record(f"backup of {step_dir.name} skipped: previous backup still running")
            return
        log = (WORK / "logs" / f"backup_{step_dir.name}.log").open("w")
        self.backup = subprocess.Popen(
            ["bash", str(WORK / "backup_checkpoint.sh"), str(step_dir), str(BACKUP / self.cfg["RUN_NAME"])],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        record(f"backing up {step_dir.name} to {BACKUP}")

    def drain(self, timeout_s: int = 1800) -> None:
        deadline = time.monotonic() + timeout_s
        while (self.proc is not None or self.queue) and time.monotonic() < deadline:
            self.poll()
            time.sleep(5)


def kill_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=60)
    # torchrun children share the session; sweep any stragglers still holding GPUs.
    subprocess.run(["pkill", "-9", "-s", str(proc.pid)], capture_output=True)


def run_attempt(cfg: dict, status: Status, exporter: Exporter) -> tuple[int, str]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = WORK / "logs" / f"train_{stamp}.log"
    link = WORK / "train.log"
    link.unlink(missing_ok=True)
    link.symlink_to(log_path)
    env = dict(os.environ)
    env.update({k: str(v) for k, v in cfg.items() if k.isupper()})
    env["SAVE_FOLDER"] = str(run_dir(cfg))
    target = int(cfg["MAX_DURATION"])
    export_every = int(cfg.get("EXPORT_EVERY", cfg["SAVE_INTERVAL"]))
    with log_path.open("w") as out:
        proc = subprocess.Popen([str(WORK / "run_train.sh")], stdout=out, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
    record(f"LAUNCH pid={proc.pid} log={log_path.name} cfg={json.dumps(cfg, sort_keys=True)}")
    status.update(state="RUNNING", train_pid=proc.pid, train_log=str(log_path), attempt_started_at=now())

    step, launched, last_advance, position = -1, time.monotonic(), None, 0
    buffer = ""
    while proc.poll() is None:
        with log_path.open("r", errors="replace") as source:
            source.seek(position)
            chunk = source.read()
            position = source.tell()
        lines = (buffer + chunk).split("\n")
        buffer = lines.pop()
        for line in lines:
            if (m := STEP_RE.search(line)) is not None:
                observed = int(m.group(1))
                if observed > step:
                    step, last_advance = observed, time.monotonic()
                    status.update(step=step, target=target, last_step_at=now())
            if (m := SAVED_RE.search(line)) is not None:
                saved_step = int(m.group(2))
                status.update(last_checkpoint=m.group(1), last_checkpoint_at=now())
                if saved_step % export_every == 0 or saved_step >= target:
                    exporter.enqueue(Path(m.group(1)))
        exporter.poll()
        age = time.monotonic() - (last_advance or launched)
        limit = STALL_S if last_advance else STARTUP_GRACE_S
        if age > limit:
            reason = f"watchdog: no step progress for {int(age)}s (step {step})"
            record(reason)
            kill_tree(proc)
            return step, reason
        time.sleep(15)
    rc = proc.returncode
    tail = log_path.read_text(errors="replace")[-20000:]
    for line in tail.splitlines():
        if (m := STEP_RE.search(line)) is not None:
            step = max(step, int(m.group(1)))
    if rc != 0:
        kill_tree(proc)
        errors = [l for l in tail.splitlines() if "Error" in l or "CRITICAL" in l][-3:]
        reason = f"exit={rc} at step {step}: {' | '.join(e.strip()[:300] for e in errors)}"
    else:
        reason = f"exit=0 at step {step}"
    record(reason)
    return step, reason


def main() -> int:
    (WORK / "logs").mkdir(exist_ok=True)
    EXPORTS.mkdir(exist_ok=True)
    status = Status()
    record(f"supervisor start pid={os.getpid()}")
    last_failure_step = None
    while True:
        cfg = load_config()
        exporter = Exporter(cfg)
        target = int(cfg["MAX_DURATION"])
        # Export any checkpoint on the export cadence that was saved but never exported
        # (e.g. the job died while its export was queued).
        export_every = int(cfg.get("EXPORT_EVERY", cfg["SAVE_INTERVAL"]))
        for step_no, path in checkpoints(cfg):
            if step_no % export_every == 0:
                exporter.enqueue(path)
        problem = preflight(cfg, status)
        if problem:
            record(f"STOP: preflight failed: {problem}")
            status.update(state="STOPPED", stop_reason=f"preflight: {problem}")
            return 1
        step, reason = run_attempt(cfg, status, exporter)
        status.update(last_exit=reason, last_exit_at=now())
        if reason.startswith("exit=0") and step >= target:
            record(f"COMPLETE at step {step}; finishing exports")
            for step_no, path in checkpoints(cfg):
                if step_no == step:
                    exporter.enqueue(path)
            exporter.drain()
            status.update(state="COMPLETE")
            return 0
        exporter.drain(timeout_s=900)
        if last_failure_step is not None and step <= last_failure_step:
            record(f"STOP: failed again without progress past step {last_failure_step}: {reason}")
            status.update(state="STOPPED", stop_reason=f"repeat failure without progress: {reason}")
            return 1
        last_failure_step = step
        window = [t for t in status.data["restarts"] if time.time() - t < 6 * 3600]
        if len(window) >= RESTARTS_PER_6H:
            record("STOP: restart budget exhausted")
            status.update(state="STOPPED", stop_reason=f"restart budget exhausted; last: {reason}")
            return 1
        window.append(time.time())
        status.update(state="RESTARTING", restarts=window)
        record(f"RESTART {len(window)}/{RESTARTS_PER_6H} in 6h after: {reason}; backoff {BACKOFF_S}s")
        time.sleep(BACKOFF_S)


if __name__ == "__main__":
    sys.exit(main())
