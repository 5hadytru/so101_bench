# pi0.5 fine-tune on so101_bench_sim_WM

Full fine-tune of `lerobot/pi05_base` on the SO-101 Bench simulated teleop dataset (3,629
episodes, 1,996,120 frames) with the same four cameras as the GR00T N1.7 and MolmoAct2 runs:
`overhead`, `wrist`, and the settled frames captured at robot start, `overhead_init` and
`wrist_init`. Same pod and the same 36 h budget as MolmoAct2.

## Pins

| What | Value |
|---|---|
| LeRobot | https://github.com/huggingface/lerobot @ `eb7a54c6ff4983fc86075442bf4fa61503e7b5d1` (unmodified) |
| Base model | `lerobot/pi05_base` @ `b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba` (4.14B trainable params); tokenizer from gated `google/paligemma-3b-pt-224` |
| Dataset | `5hadytru/so101_bench_sim_WM` @ `b569ad81ee60ffe3580bc0f1df513bde1bb36e8c` (LeRobot v3.0, native LeRobot joint units; no frame conversion) |
| Hardware | 4x A100-SXM4-80GB (NVLink), RunPod volume disk at `/workspace` |
| Software | Python 3.12.3, torch 2.10.0+cu128, transformers 5.5.4, torchcodec 0.10.0, diffusers 0.39.0 (`environment/`) |

## Recipe and where it comes from

OpenPI has exactly one pi0.5 simulation fine-tune config, `pi05_libero`
(`openpi/src/openpi/training/config.py`, checked at commit `215abfb`): full fine-tune from
`pi05_base`, **30,000 steps at global batch 256 (7.68M samples)**, AdamW(0.9, 0.95, eps 1e-8,
weight decay 1e-10, clip 1.0), 10k-step warmup to 5e-5 then constant (cosine with
`decay_lr = peak`), EMA 0.999, quantile normalization, 10-step (1 s at 10 Hz) action chunks,
`discrete_state_input=False`. Its LIBERO eval executes 5 of the 10 steps (`replan_steps=5`,
`examples/libero/main.py`); the DROID eval executes 8 of 15 (`open_loop_horizon=8`).
LeRobot's own LIBERO recipe (`docs/source/pi05.mdx`) is 30,000 steps at batch 64 on one GPU from
`pi05_libero_base`, lr 2.5e-5 cosine to 2.5e-6, mean/std normalization, 50-step chunks executing 10.

This run follows `pi05_libero`, scaled to the budget:

| Setting | Value | Note |
|---|---|---|
| Trainable | everything (`freeze_vision_encoder=false`, `train_expert_only=false`) | as pi05_libero |
| Global batch | 256 = 32/GPU x 4 GPUs (DDP) x 2 grad accumulation | as pi05_libero |
| LR schedule | warmup to 5e-5, then constant | WARMUP=10000, DECAY_STEPS=30000, DECAY_LR=5e-5; LeRobot auto-scales both by STEPS/30000, so the warmup is the first third of the run, as in pi05_libero |
| Optimizer | AdamW(0.9, 0.95), eps 1e-8, wd 1e-10, clip 1.0 | as pi05_libero (LeRobot's preset wd is 0.01) |
| EMA | 0.999 | as pi05_libero; exports use the EMA weights |
| Normalization | quantile (q01/q99 from the dataset's meta/stats.json) | as pi05_libero |
| Action chunk | 30 steps = 1 s at 30 Hz; eval executes 15 (0.5 s) | openpi's 1 s chunks, replan every half chunk |
| Steps | 18,000 LeRobot steps = 9,000 optimizer updates = 2.30M samples (~1.15 epochs) | budget-limited (pi05_libero: 7.68M) |
| Images | 224x224 resize-with-pad, 4 cameras, LeRobot image augmentation on | 4 cameras for parity with the other two models |
| State | discretized into the prompt (LeRobot default) | deliberate deviation: pi05_libero drops state, but LIBERO uses delta end-effector actions while this benchmark uses absolute joints, and the other two models get state |

W&B: `5hadytru/finetune-pi05-so101`, run `g260uy3k`. Launched 2026-09-29 13:47 UTC and **completed
2026-09-30 23:21 UTC** (18,000 micro-steps in 33.5 h, 6.6 s/micro-step, no restarts; `records/`).

## Throughput sweep (`records/bench_results.jsonl`)

4 GPUs, 8 loader workers/GPU, PyAV decode, EMA on, gradient checkpointing on unless noted:

| Per-GPU batch | Samples/s | Peak GPU memory |
|---|---|---|
| 4 (DDP) | 12.0 | 69 GB |
| 8 (DDP) | 15.1 | 69 GB |
| 12 (DDP) | 16.6 | 69 GB |
| 16 (DDP) | 16.9 | 69 GB |
| **32 (DDP)** | **18.8** | **76 GB** (rank 0 holds the EMA shadow; a checkpoint save at this size also passed) |
| 8, no gradient checkpointing | OOM | 81 GB |
| 8, FSDP (`dp_shard=4`) | not supported | LeRobot's PI05 policy declares no FSDP wrap units |

GPU-bound throughout (util 97-98 %); the loader waited < 10 ms per step, so worker count and video
backend don't matter. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set.

## LeRobot facts that shaped the setup

- `--steps`, `--save_freq` and the LR scheduler count **micro-batches**, not optimizer updates (with
  accumulation 2, one step = 128 samples). EMA updates once per optimizer update.
- LeRobot never deletes checkpoints (each ~48 GB: weights, optimizer, EMA); the supervisor keeps the
  newest 2. `--wandb.disable_artifact=true` stops each checkpoint being uploaded to W&B.
- Needs `lerobot[pi,training,dataset,diffusion]`; LeRobot resolves torchcodec 0.11 (built for torch
  2.11), so torchcodec is pinned to 0.10.
- Resume = `--config_path=<run>/checkpoints/last/pretrained_model/train_config.json --resume=true`;
  it keeps the saved step count.

## Layout (`pod/`)

- `setup_env.sh`: venv at `/root/lerobot-venv` with the pins above.
- `run_train.sh`: the one launch command (all knobs are env vars; `RESUME=1` resumes).
- `bench.py`: the sweep; parses LeRobot's `step:/loss:/data_s:/updt_s:/mem_gb:` log fields.
- `supervise.py` + `supervisor_config.json`: preflight (GPUs free, disk, dataset SHA spot-check,
  newest checkpoint complete and fully readable, quarantine otherwise), launch/resume, 20 min stall
  watchdog, 6 restarts per 6 h, checkpoint pruning (keep 2), background exports every 1,500 steps.
- `export_checkpoint.py`: `pretrained_model_ema` -> bf16 `model.safetensors` (8.3 GB) + processor
  configs/stats, `EXPORT_OK` last.
- `verify_dataset.py`: same dataset checks as the MolmoAct2 run.
- `bootstrap_after_restart.sh`: restore the venv if the container disk was wiped, start the supervisor.

## Local validation

- `scripts/pi05_server.py` (venv `~/workspace/pi05_server/.venv`: the same LeRobot commit + torch 2.10
  cu128 + FastAPI) serves an export with the same HTTP protocol as `molmoact2_server.py`; images are
  matched to `observation.images.<name>` by name. ~270 ms per 30x6 chunk on the RTX 3090.
- `scripts/molmoact2_eval.py --joint_frame lerobot --action_horizon 15 --use_overhead_init true
  --use_wrist_init true` (the `lerobot` frame is the dataset's native units via GR00T's
  `SO101JointMapper`).
- `scripts/monitor_runpod_pi05_checkpoints.sh`, started by `start_local_monitor.sh` (tmux
  `pi05_monitor`): 39-episode `tasks/real_gr00t_val_v2.jsonl`, every 1,500 steps, best 3 kept in
  `checkpoints/so101_pi05-SO101_WM_runpod_best/`; results in
  `outputs/pi05_runpod_checkpoint_validation_pi05-SO101/stepN/`.
- Gate before monitoring (2026-09-29, step-1500 export): open-loop on dataset frames returned (30, 6)
  chunks in LeRobot units with the first action within 2-10 units of the recorded state; two closed-loop
  Isaac episodes ran cleanly.

`TRANSITION_RUNBOOK.md` records how the pod was handed over from MolmoAct2 and the decisions made.

## Results (39-episode `real_gr00t_val_v2`, fixed layouts, EMA weights, execute 15 of 30)

| Micro-step | 1500 | 3000 | 4500 | 6000 | 7500 | 9000 | 10500 | 12000 | 13500 | 15000 | 16500 | 18000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Samples seen | 0.19M | 0.38M | 0.58M | 0.77M | 0.96M | 1.15M | 1.34M | 1.54M | 1.73M | 1.92M | 2.11M | 2.30M |
| Successes / 39 | 0 | 0 | 0 | 7 | 12 | 11 | 15 | **18** | 17 | 16 | 15 | **16 (final, official)** |

- **Official checkpoint: step18000**, the final checkpoint within the 36 h budget. This is the same
  rule as GR00T (step 16000) and MolmoAct2 (step 10000). It is 7.8 GB, EMA, bf16, on Hugging Face at
  [`5hadytru/so101_bench_pi05`](https://huggingface.co/5hadytru/so101_bench_pi05) (commit `de712e6`, verified by SHA-256;
  the local copy was deleted 2026-10-06). All other pi0.5 checkpoints were deleted
  (user, 2026-09-30). The best validated checkpoint, step12000 (18/39), is within noise of the
  15-18/39 plateau and was not used.
- **The curve:** it jumped from 0 to 7/39 between 0.58M and 0.77M samples, then plateaued around
  15-18/39 (40-45%) from 1.34M samples on. GR00T N1.7's official checkpoint scored 19/39.
- **Pod:** tnp7b21hxe5po7 was deleted at 23:28 UTC on 2026-09-30, at the user's request, after the
  step16500 and step18000 exports were pulled and manifest-verified locally and the records were
  copied. The last two validations run from those local copies.

### This run is under-trained

**The loss had not plateaued.** Means over the 30 log lines before each step, from
`records/logs/train_20260929_134710.log`:

| Micro-step | 1500 | 3000 | 6000 | 9000 | 12000 | 15000 | 16500 | 18000 (final) |
|---|---|---|---|---|---|---|---|---|
| Loss | 0.0914 | 0.0651 | 0.0515 | 0.0434 | 0.0377 | 0.0337 | 0.0311 | 0.0305 |

- It fell another 10% over the last 3,000 micro-steps, at a constant LR of 5e-5 (the warmup covered
  the first third of the run, and the recipe has no decay).
- The run ended mid-descent, not at convergence.

**The official sim recipe trains about 3.3x longer at the same batch.**
- OpenPI's pi05_libero is 30,000 optimizer steps at global batch 256, which is 7.68M samples.
- This run did 9,000 optimizer updates (18,000 LeRobot micro-steps with accumulation 2) at the same
  global batch of 256, which is 2.30M samples, ~1.15 epochs of our 2.0M frames.
- LeRobot's own LIBERO recipe is also 30,000 steps, at batch 64.

Under-training is imposed by the benchmark's equal-budget rule (36 h on 4x A100), not by a recipe
choice. At ~19 samples/s on this hardware, pi05_libero's 7.68M samples would take ~113 h.
