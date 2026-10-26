# GR00T N1.7 fine-tune on so101_bench_sim_WM_v2.1

Everything needed to rerun the SO-101 Bench GR00T N1.7 fine-tune, except the
upstream Isaac-GR00T repo and the dataset. Those are pinned below, not copied.

## Pins

| What | Value |
|---|---|
| Isaac-GR00T | https://github.com/NVIDIA/Isaac-GR00T @ `51d4c89f72fda44cbf77285c6a8114b52676b8a1`, plus `patches/isaac-gr00t_51d4c89.patch` |
| Base model | `nvidia/GR00T-N1.7-3B` (loads gated `nvidia/Cosmos-Reason2-2B`; needs an HF login with access) |
| Dataset | `so101_bench_sim_WM_v2.1` (source: this repo's `data/lerobot/so101_bench_sim_WM_v2.1`, 3629 episodes, 18153 files, ~50 GB) |
| Embodiment tag | `NEW_EMBODIMENT` with `configs/so101_config.py` |
| Hardware | 4x A100-SXM4-80GB (RunPod), driver 580.126.20 |
| Software | Python 3.12.3, torch 2.9.0+cu128, transformers 4.57.3, deepspeed 0.17.6 (`environment/`) |

## Layout

- `finetune_so101.sh` is the SO-101 launcher. It lives at `examples/SO101/finetune.sh` inside the GR00T checkout. It sets the paths and caches under `/workspace`, the batch size, and the resume/save knobs, then calls upstream `examples/finetune.sh`.
- `configs/so101_config.py` is the modality config: video `overhead, wrist, overhead_init, wrist_init`; state and action `single_arm, gripper`; 16-step action chunk with relative arm and absolute gripper; language `annotation.human.task_description`.
- `patches/isaac-gr00t_51d4c89.patch` holds the two upstream changes:
  1. `examples/finetune.sh`: `--save_total_limit` reads `SAVE_TOTAL_LIMIT` (default 3).
  2. `gr00t/experiment/experiment.py`: `ResumeStepIntervalsCallback`. transformers 4.57 reloads `trainer_state.json` on resume *after* applying the arguments, so a resumed run silently keeps the checkpoint's `save_steps`. This callback reapplies the launch's intervals in `on_train_begin`.
- `supervisor/` is the resume-on-failure wrapper used for the whole run:
  - restarts after storage crashes and step stalls (20 min without a step; 60 min before the first step),
  - checks a checkpoint is complete and fully readable before resuming, and moves unreadable ones to `checkpoints/_unreadable/`,
  - allows 5 restarts per 6 h.
  Its `TARGET` (20000) and `SAVE_STEPS` (1000) are set inside `supervise_finetune.py`.
- `tools/`:
  - `scan_dataset_reads.py` reads every file with per-file timeouts, to verify the dataset or a checkpoint on flaky storage.
  - `benchmark_workers.py` and `bench_stats.py` run the data-loader worker benchmark.
  - `restart_after_checkpoint.py` is the one-off helper that verified checkpoint-6000 and relaunched the run on the new settings.
- `resolved_config/` is the `experiment_cfg/` GR00T wrote at the final launch (`conf.yaml` shows `max_steps: 20000`, `save_steps: 1000`, `save_total_limit: 3`).
- `dataset_meta/` has GR00T's generated `meta/stats.json` and `meta/relative_stats.json` for the dataset. These are deterministic caches of the parquet data, and GR00T regenerates them if missing. `stats.json` differs from the LeRobot original that ships with the dataset.
- `environment/` has the frozen `pip freeze` of the training venv and a versions summary.
- `records/` has the final `trainer_state.json` (loss and LR history), the supervisor log (`supervisor_log.txt`), and the worker-benchmark log (`worker_benchmark.txt`).

## Rerun from scratch

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T /workspace/Isaac-GR00T
cd /workspace/Isaac-GR00T && git checkout 51d4c89f72fda44cbf77285c6a8114b52676b8a1
git apply /path/to/GR00T_N1.7_fine_tuning/patches/isaac-gr00t_51d4c89.patch
uv sync   # uses the repo's uv.lock; environment/requirements-frozen.txt is the reference
mkdir -p examples/SO101
cp /path/to/GR00T_N1.7_fine_tuning/finetune_so101.sh examples/SO101/finetune.sh
cp /path/to/GR00T_N1.7_fine_tuning/configs/so101_config.py examples/SO101/
# dataset -> examples/SO101/so101_bench_sim_WM_v2.1 (verify with tools/scan_dataset_reads.py)
HF_HOME=/workspace/gr00t-so101-n1d7/cache/huggingface .venv/bin/hf auth login   # Cosmos-Reason2-2B is gated

# Clean single-schedule run (what the final run approximates):
DATALOADER_NUM_WORKERS=4 MAX_STEPS=20000 SAVE_STEPS=1000 SAVE_TOTAL_LIMIT=3 \
  bash examples/SO101/finetune.sh
```

For unattended runs on a network volume, use the supervisor instead. It sets `SO101_RESUME=1` and the steps and saves itself:
```bash
cp -r /path/to/GR00T_N1.7_fine_tuning/supervisor/* /workspace/gr00t-so101-n1d7/
cp /path/to/GR00T_N1.7_fine_tuning/tools/scan_dataset_reads.py /root/   # the supervisor calls it from container disk
screen -dmS gr00t_so101_supervisor bash -lc \
  "DATALOADER_NUM_WORKERS=4 /workspace/gr00t-so101-n1d7/supervise_finetune.sh >> /workspace/gr00t-so101-n1d7/supervisor_stdout.log 2>&1"
```
A fresh run needs at least one checkpoint for the supervisor's resume. Either start the first leg with `finetune_so101.sh` directly (no `SO101_RESUME`), or remove `SO101_RESUME` from the supervisor env for the first attempt.

Hyperparameters are fixed in `finetune_so101.sh` and upstream `examples/finetune.sh`, and the resolved values are in `resolved_config/conf.yaml`:
- global batch 640 (160 per GPU x 4), gradient accumulation 1,
- LR 1e-4 cosine with 5% warmup, weight decay 1e-5,
- color jitter (brightness 0.3, contrast 0.4, saturation 0.5, hue 0.08),
- shard size 1024, episode sampling rate 0.1, seed 42,
- only the projector, the diffusion model and VLLN are trained (LLM and vision tower frozen).

## What the actual run did (for exact reproduction)

The trained checkpoint did **not** come from one clean schedule:

| Leg | Steps | max_steps (LR schedule) | save_steps / kept | Notes |
|---|---|---|---|---|
| 1 | 0 → 5485 | 51000 | 2000 / 5 | Started 2026-09-25 06:10 UTC. Crashed at 5485 when the RunPod `/workspace` MooseFS mount degraded (`No such device or address` on a video read, then SIGBUS). Lost everything after checkpoint-4000. |
| 2 | 4000 → 6000 | 51000 | 2000 / 5 (500 was requested, but the resume kept 2000; that's the bug the patch fixes) | Resumed 2026-09-26 00:42 UTC under the supervisor after a pod restart, 21 dataset files re-uploaded, and full read scans. |
| 3 | 6000 → 20000 | **20000** | 1000 / 3 | Relaunched 05:44 UTC from checkpoint-6000 (verified before stopping and before resuming). Budget cut to 20k; the cosine curve switched from the 51k schedule (LR 9.9e-5 at step 6000) to the 20k one (about 8.4e-5), with no second warmup, then annealed to 0 at 20000. |

Legs 2 and 3 ran with `ignore_data_skip=True` (GR00T default), so the data order after each resume is not a replay of the uninterrupted stream. A clean `MAX_STEPS=20000` rerun will differ slightly from the published checkpoint: a single 20k cosine from step 0 (1000-step warmup rather than 2550), and a different data order.

## Throughput and cost notes

- Data-loader worker benchmark: 30 min trials from checkpoint-4000, mean s/step over the final 10 min (`records/worker_benchmark.txt`):

  | Workers per GPU | s/step |
  |---|---|
  | 1 | 23.0 |
  | 2 | 13.6 |
  | 3 | 9.9 |
  | 4 | 7.3 |
  | 6 | 7.2 |

  Training is data-bound (video decode). 4 workers is the knee, and 8 was not needed.
- Steady speed: about 8.0 s/step including 1000-step saves. A checkpoint is about 34 GB (including DeepSpeed ZeRO optimizer state) and takes about 5 min to save.
- Disk: with `save_total_limit` 3, 4 checkpoints exist briefly during each save (136 GB), plus the dataset (47 GB) and HF cache (27 GB).
- Storage: `/workspace` on RunPod is a network FUSE mount that can stall or lose chunks. Keep the supervisor on, and run `tools/scan_dataset_reads.py` on the dataset before long runs.

## Evaluation

Checkpoints were validated while training ran with `scripts/monitor_runpod_groot_checkpoints.sh`: 39-episode `tasks/real_gr00t_val_v2.jsonl`, action horizon 16, only multiples of 2000. Results are under `outputs/groot_runpod_checkpoint_validation_GR00T_N1.7-3B/`.
