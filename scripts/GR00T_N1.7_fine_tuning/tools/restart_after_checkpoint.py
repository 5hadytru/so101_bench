#!/usr/bin/env python3
"""Controlled restart once checkpoint-6000 is complete, so the relaunch picks up the save_steps=500 fix.

Waits for checkpoint-6000 (trainer_state.json + latest + non-empty global_step6000/) and for training to
move past it, then stops the supervisor and every training process (matched via /proc cmdline, never
this process), waits for free GPUs and relaunches the supervisor in screen with 4 loader workers.
"""
import os, signal, subprocess, time
from pathlib import Path
from datetime import datetime, timezone
W = Path("/workspace/gr00t-so101-n1d7"); CK = W / "checkpoints/checkpoint-6000"
def log(msg):
    line = f"{datetime.now(timezone.utc).isoformat()} restart_after_ckpt: {msg}"
    print(line, flush=True)
    with (W / "supervisor.log").open("a") as f: f.write(line + "\n")
def complete():
    gs = CK / "global_step6000"
    return (CK / "trainer_state.json").is_file() and (CK / "latest").is_file() and gs.is_dir() and len(list(gs.iterdir())) >= 5
def matching(markers):
    out = []
    for e in Path("/proc").iterdir():
        if not e.name.isdigit() or int(e.name) in (os.getpid(), os.getppid()): continue
        try: a = (e / "cmdline").read_bytes()
        except OSError: continue
        if any(m in a for m in markers): out.append(int(e.name))
    return out
def gpu_free():
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True)
    return r.returncode == 0 and not r.stdout.strip()
while not complete():
    time.sleep(60)
time.sleep(120)  # let DeepSpeed finish flushing and training move past the save
# Verify checkpoint-6000 end to end while the current run keeps training; only stop it if the
# checkpoint is fully readable, so a bad save can never cost progress.
res = W / "scan_checkpoint-6000_pre_relaunch.jsonl"
scan = subprocess.run(["/usr/bin/python3", "/root/scan_dataset_reads.py", str(CK), "--results", str(res),
                       "--workers", "4", "--timeout", "600"], capture_output=True, text=True, timeout=5400)
last = (scan.stdout.strip().splitlines() or [""])[-1]
if scan.returncode != 0 or not last.startswith("DONE ") or res.stat().st_size != 0:
    log(f"checkpoint-6000 FAILED pre-relaunch verification ({last}); NOT restarting, current run continues")
    raise SystemExit(1)
log(f"checkpoint-6000 verified readable ({last}); stopping supervisor + training for relaunch with save_steps fix")
subprocess.run(["screen", "-S", "gr00t_so101_supervisor", "-X", "quit"])
for p in matching([b"supervise_finetune"]):
    try: os.kill(p, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(5)
for sig in (signal.SIGTERM, signal.SIGKILL):
    for p in matching([b"gr00t/experiment/launch_finetune.py", b"examples/finetune.sh", b"examples/SO101/finetune.sh"]):
        try: os.killpg(os.getpgid(p), sig)
        except (ProcessLookupError, PermissionError): pass
    time.sleep(30)
for _ in range(60):
    if gpu_free(): break
    time.sleep(10)
else:
    log("STOP: GPUs not free after 10 min; NOT relaunching"); raise SystemExit(1)
subprocess.run(["screen", "-dmS", "gr00t_so101_supervisor", "bash", "-lc",
                "DATALOADER_NUM_WORKERS=4 /workspace/gr00t-so101-n1d7/supervise_finetune.sh >> /workspace/gr00t-so101-n1d7/supervisor_stdout.log 2>&1"])
log("relaunched supervisor (resume from checkpoint-6000, save_steps 500 via ResumeStepIntervalsCallback)")
