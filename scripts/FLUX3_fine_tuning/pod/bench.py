#!/usr/bin/env python3
"""Throughput / VRAM sweep for FLUX 3 Action fine-tuning on so101_bench_sim (4x A100).

Each point launches a real training run, lets it warm up, times a fixed number of logged updates,
then kills it before the final-step checkpoint (a full-FT checkpoint is ~110 GB). Results are
appended to records/bench_results.jsonl, one line per point (OOM and errors included).

  fft  : standalone flux-action full fine-tune (configs/so101/train.json, FSDP2 over 4 GPUs),
         initialized from the SO-101 policy with its "dit." prefix stripped.
  lora : LeRobot main examples/flux3/lora.json on the SO-101 package (accelerate DDP, 4 GPUs).

Every point keeps global batch 128 (per-GPU micro-batch x 4 GPUs x accumulation).

  bench.py list                 # print the sweep
  bench.py run [NAME ...]       # run all points not yet in the results file (or just NAME ...)
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

ROOT = Path(os.environ.get("ROOT", "/root/flux3"))
FLUX = ROOT / "flux-action"
LEROBOT = ROOT / "lerobot"
RESULTS = ROOT / "records" / "bench_results.jsonl"
RUNS = ROOT / "bench"
GPUS = 4
GLOBAL_BATCH = 128
WARMUP_UPDATES = int(os.environ.get("WARMUP_UPDATES", "3"))
MEASURED_UPDATES = int(os.environ.get("MEASURED_UPDATES", "8"))
TIMEOUT_S = int(os.environ.get("POINT_TIMEOUT_S", "2400"))

WEIGHTS = ROOT / "weights"
VAE = WEIGHTS / "base" / "video_vae.safetensors"
TEXT_ENCODER = WEIGHTS / "base" / "text_encoder"


def fft(name: str, micro: int, **over) -> dict:
    return {"name": name, "mode": "fft", "micro": micro, **over}


def lora(name: str, micro: int, **over) -> dict:
    return {"name": name, "mode": "lora", "micro": micro, **over}


SWEEP = [
    # fft_mb8_torch: OOM in update 2 (Adam moments + two fp32 EMA shadows arrive after update 1).
    fft("fft_mb8_torch", 8),
    fft("fft_mb4_torch", 4),
    fft("fft_mb2_torch", 2),
    fft("fft_mb8_ema1_vae4", 8, ema_sigma_rels=[0.10], vae_batch_windows=4),
    fft("fft_mb4_ema1", 4, ema_sigma_rels=[0.10]),
    fft("fft_mb4_cudnn", 4, attn_mode="cudnn"),
    # compile_model=True: upstream training bug ("accessing tensor output of CUDAGraphs that has been
    # overwritten") in the reduce-overhead VAE encoder; the encoders are not the bottleneck anyway.
    fft("fft_best_compile", None, compile_model=True),
    fft("fft_best_noreshard", None, reshard_after_forward=False),
    fft("fft_mb4_ema0", 4, ema_sigma_rels=[]),
    lora("lora_mb8", 8),
    lora("lora_mb4", 4),
    lora("lora_mb2", 2),
    lora("lora_mb12", 12),
    lora("lora_mb4_nogc", 4, gradient_checkpointing=False),
    lora("lora_best_w8", None, num_workers=8),
    lora("lora_mb2_nogc", 2, gradient_checkpointing=False),
]


# ------------------------------------------------------------------ results helpers
def load_results() -> list[dict]:
    if not RESULTS.exists():
        return []
    return [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]


def best_micro(mode: str, prefix: str) -> tuple[int, dict]:
    """Fastest successful point of ``mode`` among names starting with ``prefix``: (micro, its overrides)."""
    ok = [r for r in load_results() if r["mode"] == mode and r["status"] == "ok" and r["name"].startswith(prefix)]
    if not ok:
        raise RuntimeError(f"no successful {mode} point yet to base a 'best' point on")
    best = max(ok, key=lambda r: r["windows_per_s"])
    return best["micro"], best.get("overrides", {})


# ------------------------------------------------------------------ GPU memory sampler
class GpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = [0] * GPUS
        self.util = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip().splitlines()
                utils = []
                for i, line in enumerate(out[:GPUS]):
                    mem, util = (int(x) for x in line.split(","))
                    self.peak[i] = max(self.peak[i], mem)
                    utils.append(util)
                if self.record:
                    self.util.append(sum(utils) / len(utils))
            except Exception:
                pass
            self.stop.wait(2.0)

    record = False


# ------------------------------------------------------------------ launchers
def fft_command(point: dict, out: Path) -> tuple[list[str], Path, Path]:
    cfg = json.loads((FLUX / "configs/so101/train.json").read_text())
    micro = point["micro"]
    cfg.update(
        source_root=str(ROOT / "datasets/so101_bench_sim"),
        index_dir=str(ROOT / "index/so101_bench_sim"),
        output_dir=str(out),
        steps=100_000,  # never reached: the bench kills the run before any checkpoint
        checkpoint_every=100_000,
        log_every=1,
        resume="none",
        windows_per_rank=micro,
        grad_accumulation=GLOBAL_BATCH // (GPUS * micro),
        shard_size=GPUS,
        num_workers=point.get("num_workers", 8),
        frozen_steps=0,
        trunk_warmup_steps=1,
        heads_warmup_steps=1,
        cooldown_start=None,
        cooldown_steps=0,
        visits_per_epoch=8,
    )
    for key in ("activation_checkpointing", "reshard_after_forward", "ema_sigma_rels"):
        if key in point:
            cfg[key] = point[key]
    cfg["policy"].update(
        trunk_weights=str(WEIGHTS / "flux-3-action-so101-trunk.safetensors"),
        video_vae_id=str(VAE),
        text_encoder_id=str(TEXT_ENCODER),
    )
    for key in ("attn_mode", "compile_model", "vae_batch_windows"):
        if key in point:
            cfg["policy"][key] = point[key]
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "bench_config.json"
    config_path.write_text(json.dumps(cfg, indent=2))
    cmd = [
        str(FLUX / ".venv/bin/torchrun"), "--standalone", f"--nproc_per_node={GPUS}",
        "-m", "flux_action.cli", "train", "--config", str(config_path),
    ]
    return cmd, FLUX, out / "metrics.jsonl"


def lora_command(point: dict, out: Path) -> tuple[list[str], Path, Path]:
    micro = point["micro"]
    accum = GLOBAL_BATCH // (GPUS * micro)
    cfg = json.loads((LEROBOT / "examples/flux3/lora.json").read_text())
    cfg["accelerator"]["gradient_accumulation"]["steps"] = accum
    config_path = out.parent / f"{out.name}_lora.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=2))
    cmd = [
        str(LEROBOT / ".venv/bin/accelerate"), "launch", f"--num_processes={GPUS}", "--multi_gpu",
        "--mixed_precision=bf16", "-m", "lerobot.scripts.lerobot_train",
        f"--config_path={config_path}",
        f"--policy.path={WEIGHTS / 'so101'}", "--policy.device=cuda",
        f"--policy.video_vae_id={VAE}", f"--policy.text_encoder_id={TEXT_ENCODER}",
        f"--policy.gradient_checkpointing={str(point.get('gradient_checkpointing', True)).lower()}",
        "--dataset.repo_id=5hadytru/so101_bench_sim",
        f"--dataset.root={ROOT / 'datasets/so101_bench_sim'}",
        "--dataset.eval_split=0",
        '--rename_map={"observation.images.overhead":"observation.images.scene"}',
        f"--batch_size={micro}", "--steps=1000000", "--save_freq=1000000", "--eval_steps=0",
        "--log_freq=4", f"--num_workers={point.get('num_workers', 4)}",
        # LeRobot refuses an existing output_dir, and the bench keeps stdout.log in ``out``.
        "--wandb.enable=false", f"--output_dir={out / 'train'}",
    ]
    return cmd, LEROBOT, out / "train.log"


LEROBOT_STEP = re.compile(r"step:(\d+(?:\.\d+)?[KM]?)\b")


# ------------------------------------------------------------------ one point
def run_point(point: dict) -> dict:
    point = dict(point)
    if point["micro"] is None:  # a "best" point reuses the fastest micro-batch of its mode so far
        prefix = "fft_mb" if point["mode"] == "fft" else "lora_mb"
        micro, overrides = best_micro(point["mode"], prefix)
        point["micro"] = micro
        for k, v in overrides.items():
            point.setdefault(k, v)
    out = RUNS / point["name"]
    if out.exists():
        subprocess.run(["rm", "-rf", str(out)], check=True)
    if point["mode"] == "fft":
        cmd, cwd, progress = fft_command(point, out)
    else:
        cmd, cwd, progress = lora_command(point, out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "stdout.log"
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "HF_HOME": str(ROOT / ".hf-home"),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled",
    }
    env.pop("HF_TOKEN", None)
    overrides = {k: v for k, v in point.items() if k not in ("name", "mode", "micro")}
    result = {
        "name": point["name"], "mode": point["mode"], "micro": point["micro"],
        "accum": GLOBAL_BATCH // (GPUS * point["micro"]), "global_batch": GLOBAL_BATCH,
        "overrides": overrides, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(f"[bench] {point['name']}: {' '.join(cmd)}", flush=True)
    sampler = GpuSampler()
    sampler.start()
    started = time.time()
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    marks: list[tuple[float, int]] = []  # (time, update index) for each logged update
    status, detail = "timeout", ""
    try:
        while time.time() - started < TIMEOUT_S:
            time.sleep(5)
            text = log_path.read_text(errors="replace")
            if "CUDA out of memory" in text or "OutOfMemoryError" in text:
                status, detail = "oom", _tail(text)
                break
            if point["mode"] == "fft":
                records = [json.loads(l) for l in progress.read_text().splitlines()] if progress.exists() else []
                records = [r for r in records if "step_time" in r]
                marks = [(r["time"], r["step"]) for r in records]
                result["fft_records"] = records
            else:
                seen = {m[1] for m in marks}
                for m in LEROBOT_STEP.finditer(text):
                    step = _parse_k(m.group(1))
                    if step not in seen:
                        marks.append((time.time(), step))
                        seen.add(step)
            updates = _updates(point, marks)
            if updates >= WARMUP_UPDATES + 1:
                sampler.record = True
            if updates >= WARMUP_UPDATES + MEASURED_UPDATES:
                status = "ok"
                break
            if proc.poll() is not None:
                status, detail = "error", _tail(text)
                break
    finally:
        _kill(proc)
        sampler.stop.set()
        sampler.join()
    result.update(status=status, detail=detail, load_s=None, peak_nvidia_smi_mib=sampler.peak,
                  mean_gpu_util=round(sum(sampler.util) / len(sampler.util), 1) if sampler.util else None)
    if marks:
        result["load_s"] = round(marks[0][0] - started, 1)
    if status == "ok":
        result.update(_throughput(point, marks))
    result.pop("fft_records", None) if point["mode"] == "lora" else None
    if point["mode"] == "fft" and "fft_records" in result:
        recs = result.pop("fft_records")[WARMUP_UPDATES:]
        if recs:
            result["fft_step_time_s"] = round(sum(r["step_time"] for r in recs) / len(recs), 2)
            result["fft_data_wait_s"] = round(sum(r["data_wait"] for r in recs) / len(recs), 3)
            result["fft_peak_mem_gb"] = max(r.get("peak_mem_gb", 0) for r in recs)
            result["fft_loss"] = [round(r["loss"], 4) for r in recs]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as stream:
        stream.write(json.dumps(result) + "\n")
    print(f"[bench] {point['name']}: {json.dumps({k: result.get(k) for k in ('status', 'windows_per_s', 's_per_update', 'peak_nvidia_smi_mib', 'fft_peak_mem_gb')})}", flush=True)
    return result


def _parse_k(s: str) -> int:
    mult = {"K": 1_000, "M": 1_000_000}.get(s[-1], 1)
    return int(float(s[:-1] if mult > 1 else s) * mult)


def _updates(point: dict, marks: list[tuple[float, int]]) -> int:
    if point["mode"] == "fft":
        return len(marks)
    # LeRobot counts microsteps; with log_freq 4 each line is 4 microsteps.
    accum = GLOBAL_BATCH // (GPUS * point["micro"])
    return max((m[1] for m in marks), default=0) // accum


def _throughput(point: dict, marks: list[tuple[float, int]]) -> dict:
    if point["mode"] == "fft":
        window = marks[WARMUP_UPDATES:]
        (t0, s0), (t1, s1) = window[0], window[-1]
        updates = s1 - s0
    else:
        accum = GLOBAL_BATCH // (GPUS * point["micro"])
        start_micro = WARMUP_UPDATES * accum
        window = [m for m in marks if m[1] >= start_micro]
        (t0, s0), (t1, s1) = window[0], window[-1]
        updates = (s1 - s0) / accum
    if updates <= 0:
        return {}
    s_per_update = (t1 - t0) / updates
    return {"s_per_update": round(s_per_update, 2), "windows_per_s": round(GLOBAL_BATCH / s_per_update, 3)}


def _tail(text: str, n: int = 40) -> str:
    return "\n".join(text.splitlines()[-n:])


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    for sig, wait in ((signal.SIGINT, 20), (signal.SIGTERM, 20), (signal.SIGKILL, 10)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(wait)
            break
        except subprocess.TimeoutExpired:
            continue
    # Stragglers (dataloader workers) can outlive the group leader; make sure the GPUs are free.
    for _ in range(30):
        busy = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.split()
        if not busy:
            return
        for pid in busy:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
        time.sleep(2)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("list", "run"):
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "list":
        for p in SWEEP:
            print(json.dumps(p))
        return
    names = set(sys.argv[2:])
    done = {r["name"] for r in load_results()}
    for point in SWEEP:
        if names and point["name"] not in names:
            continue
        if not names and point["name"] in done:
            continue
        try:
            run_point(point)
        except Exception as exc:  # a broken point must not stop the sweep
            print(f"[bench] {point['name']} failed to launch: {exc!r}", flush=True)
            with RESULTS.open("a") as stream:
                stream.write(json.dumps({"name": point["name"], "mode": point["mode"], "micro": point["micro"],
                                         "status": "launch_error", "detail": repr(exc)}) + "\n")


if __name__ == "__main__":
    main()
