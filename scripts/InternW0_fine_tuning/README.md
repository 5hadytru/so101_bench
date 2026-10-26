# InternW0-Δ full fine-tune on so101_bench_sim

InternW0-Δ (InternRobotics) is a world-action model: the Wan2.2-TI2V-5B video expert plus a 1.2B action expert, conditioned through the frozen RynnBrain1.1-2B VLM. It is post-trained in full from `InternW0-Delta-Base` on `5hadytru/so101_bench_sim`, using overhead and wrist cameras with no init frames.

Hardware:
- **Training:** RunPod 4x A100-SXM4-80GB (driver 580.159.04), 300 GB container disk. MooseFS `/workspace` is not used.
- **Validation:** one A40.

## Pins

Models and data are in `records/model_manifest.json`, with sha256 for every file that isn't on Hugging Face.

| What | Value |
|---|---|
| InternW0-Delta repo | `90801baa3bdc7829c1e4989edfe3d0fb2410ea6a`. Python 3.11, torch 2.10.0 cu128, DeepSpeed 0.19.2, transformers 5.13.0, wandb 0.30.0 |
| Base policy | `InternRobotics/InternW0-Delta-Base` @ `ca534d4`, `pretrain.pt` |
| Backbones | `Wan-AI/Wan2.2-TI2V-5B` @ `921dbaf`; UMT5 tokenizer from `Wan-AI/Wan2.1-T2V-1.3B` @ `37ec512`; `Alibaba-DAMO-Academy/RynnBrain1.1-2B` @ `36069aa`. The converted T5 encoder and Wan2.2 VAE come from ModelScope `DiffSynth-Studio/Wan-Series-Converted-Safetensors` (not on Hugging Face) |
| Dataset | `5hadytru/so101_bench_sim` @ `3423375`: 3,629 episodes, 1,996,120 frames, converted to the LeRobot v2.1 layout that the repo's loader reads |

The frozen environments are in `records/environment/`: the training venv, the A40 server venv (the same core packages plus fastapi and uvicorn), and `versions.txt`.

## Layout

- `configs/data/so101.yaml`, `configs/task/so101.yaml`: the SO-101 data and task configs. `prepare_data.sh` copies them into the repo's `configs/`. They are the only change to the repo.
- `pod/setup_env.sh`: pinned repo, venv (including wandb), models and dataset.
- `pod/convert_v30_to_v21.py`: converts LeRobot v3.0 to v2.1, writing per-episode parquet and H.264 video at CRF 18 with a keyframe every 2 frames.
- `pod/prepare_data.sh`: copies the configs, converts the data, computes normalization stats (`tools/posttrain_stats.py`) and builds the text-embedding cache (`tools/text_cache.py`).
- `pod/run_train.sh`: one launch. Every knob is an environment variable: `MICRO ACCUM MAX_STEPS SAVE_EVERY ZERO INFRA CKPT_STRIDE WANDB RUN_NAME OUT_ROOT RESUME`.
- `pod/run_full.sh`: the supervisor for the full run. It relaunches from `checkpoints/state/latest` on a crash, up to 5 times.
- `pod/bench.py`: the throughput sweep.
- `check_wandb_metrics.py`, `HOURLY_CHECKUP.md`, `ship_validation_results.sh`: run monitoring and copying results to the workstation.
- `records/`:
  - `bench_results.jsonl`
  - `resolved_config/`: the full run's and the 10-step test's Hydra configs, `dataset_stats.json`, and the normalization stats
  - `environment/`
  - `logs/`: setup, prepare, bench, checkpoint test, supervisor

## Throughput sweep (2026-10-01, global batch 128)

Each point is a real training run (memory on, augmentation on), timed over 10 optimizer steps after a 4-step warmup.

| Point | Micro x accum | Options | Samples/s | s/step | Peak (nvidia-smi) |
|---|---|---|---:|---:|---:|
| b16_z1 | 16 x 2 | ZeRO-1 | OOM | | 81.1 GB |
| b8_z1 | 8 x 4 | ZeRO-1 | 16.69 | 7.67 | 71.4 GB |
| b4_z1 | 4 x 8 | ZeRO-1 | 13.71 | 9.33 | 70.0 GB |
| b16_z2 | 16 x 2 | ZeRO-2 | 16.00 | 8.00 | 80.9 GB |
| best_stride2 | 8 x 4 | activation-checkpoint stride 2 | 17.45 | 7.33 | 81.0 GB |
| best_online | 8 x 4 | stride 2, `+infra=online` | 17.45 | 7.33 | 80.9 GB |
| best_compile | 8 x 4 | stride 2, `+infra=compile` | 16.69 | 7.67 | 58.1 GB |
| **c16_compile** | **16 x 2** | **stride 1, `+infra=compile`, ZeRO-1** | **18.28** | **7.00** | **61.8 GB** |

`+infra=compile` compiles the mixture-of-transformers with `reduce-overhead` and uses compiled gradient checkpointing. Startup takes ~5 min, versus ~1.5 min without it.

Checkpoint cost: 12 GB of weights plus an 89 GB DeepSpeed state, saved in ~50 s. Pulling one checkpoint pod-to-pod to the A40 took 172 s for 12.4 GB.

## The full run: `so101-full-c16` (2026-10-01)

W&B: `5hadytru/finetune-internw0-so101/12stqtva`. Launched 18:57 UTC. The first launch, at 18:50, crash-looped because wandb was missing; `setup_env.sh` now installs it.

**Chosen or swept**

| Setting | Value | Why |
|---|---|---|
| Batch | micro 16 x accum 2 x 4 GPUs = 128 | Swept. 128 matches the other runs and the repo's RobotWin recipe (16 x 2). |
| DeepSpeed | ZeRO-1 | Swept; ZeRO-2 was slower. |
| Infra profile | `compile` | Swept; it frees the memory micro 16 needs. |
| Activation-checkpoint stride | 1 | Stride 2 was only tested at micro 8, where it peaked at 81 GB. |
| `max_steps` | 18,200 (~1.17 epochs) | Fills 36 h at ~7.0–7.1 s/step, plus 50 s per save and 5 min of compilation. |
| Save interval | every 500 steps, 2 rotating weight slots, one optimizer state | Disk budget. |
| Cameras | overhead above wrist, each 192x256, on the model's fixed 384x256 canvas | The vertical two-camera layout is the repo's LIBERO layout. |
| Action/state | 6-D absolute joint targets, z-score normalized with our stats, no delta dims. Arm goes in 80-D slots 0–4, gripper in slot 16. | RoboDojo joint-space convention. |
| Color jitter | brightness/contrast/saturation 0.3, gamma 0.15, also applied to VLM images | Raised from RobotWin's 0.2/0.2/0.25/0.1; not swept. |
| Memory | anchor frame (episode start) + recent frame (32 steps back); persistent KV memory off | The repo's training validator rejects `model.memory.enabled=true`. |

**Repo defaults kept**

| Setting | Value |
|---|---|
| Optimizer | AdamW, betas 0.9/0.95, weight decay 0.01, clip 1.0, bf16 |
| Learning rate | 5e-5 peak; linear warmup over 5% of steps (910); cosine decay to 1% of peak |
| Window | 33 frames with an action/video ratio of 4 → 32 actions (1.07 s at 30 Hz); stride 1 |
| Loss weights | action 1.0, video 0.5, future-delta 0.5, semantic alignment 0.1 |
| Flow shifts | video 6.0, action 1.0 |
| Frozen / trained | VLM frozen; text embeddings precomputed; the 5.0B video expert and 1.21B action expert trained in full |
| Seed | 42 |
| Sampler | length-bucketed |

The exact resolved configuration is `records/resolved_config/full_run_config.yaml`.

### Validation

`scripts/validation_pod/monitor_internw0.sh` runs on the A40 as two tmux loops:

- **`MODE=pull`** copies each scheduled weight slot (plus `config.yaml` and `dataset_stats.json`) as soon as the weights manifest lists it. Slots rotate about every 2 h.
- **`MODE=validate`** serves each copy with `scripts/internw0_server.py`, then runs `scripts/validation_pod/run_internw0_validation.sh`.

**Server** (`scripts/internw0_server.py`):
- It rebuilds the model from the checkpoint's own config and stats, with color jitter and compile turned off.
- Per request: 10 denoising steps, seed 0.
- It keeps one session per episode, which holds that episode's anchor and recent frames.
- It replans every 16 of the 32 predicted actions.
- `--dump_inputs_dir` saves the exact overhead and wrist tiles the model sees.

**Client:**
- 39-episode `real_gr00t_val_v2`, fixed layouts, seed 0
- `--action_horizon 16 --history_ticks 1`, LeRobot joint frame, no init frames
- LeRobot dataset recorded

One validation takes about 3.5 h. The schedule is steps 2000, 4000, …, 16000 and 18200. Step 500 was started as an end-to-end check and then aborted on purpose (0/3, partial). Step 18000 is skipped because it's only 200 steps before the final checkpoint. Results are copied to `outputs/internw0_runpod_checkpoint_validation/`.

### Reproduce

1. On a 4x A100-80GB pod:
   - Copy this directory to `/root/internw0`, with `pod/` and `configs/` at the top level.
   - Run `bash pod/setup_env.sh`, then `bash pod/prepare_data.sh`.
   - Put the W&B key in `secrets/wandb_api_key`.
   - Start the run: `tmux new -s iw_train 'bash pod/run_full.sh'`.
2. On the A40:
   - Run `scripts/validation_pod/setup_isaac.sh` and `setup_internw0_server.sh`.
   - Set up the pod-to-pod key `/root/.ssh/val_to_train`.
   - Start `MODE=pull` and `MODE=validate` of `monitor_internw0.sh`, setting `REMOTE`/`REMOTE_PORT` to the training pod.

The validation task files (`tasks/real_gr00t_val_v2.jsonl`, `primary_eval.jsonl`) contain held-out objects, so they are kept out of the public repo.

### Results (39-episode `real_gr00t_val_v2`)

| Step | 2000 | 4000 | 6000 | 8000 | 10000 | 12000 | 14000 | **16000** | 18200 (final) |
|---|---|---|---|---|---|---|---|---|---|
| Success | 3/39 | 14/39 | 12/39 | 16/39 | 17/39 | 19/39 | 21/39 | **22/39 (56.4%)** | 20/39 |

- **Official checkpoint (user decision, 2026-10-06): step 18200, the final checkpoint** (20/39). Step 16000 scored 22/39 and is kept as a val-selected alternative (it ties GR00T N1.7's 22/39).
  - Step 18200 by task: move 6/9, named bin 6/9, next to 4/9, between 3/9, 4-object bin 1/3; step 16000: between 6/9, named bin 6/9, move 5/9, next to 4/9, 4-object bin 1/3.
  - Unseen-unseen objects: 5/13 at step 18200, 6/13 at step 16000.
- **Where the weights are.** Hugging Face [`5hadytru/so101_bench_InternW0-Delta`](https://huggingface.co/5hadytru/so101_bench_InternW0-Delta) (commit `8316ccc`): the official step 18200 (user decision, 2026-10-06) at the root, step 16000 in `step016000/`, plus training logs. Verified by SHA-256; the local copies were deleted 2026-10-06.
- **Run end.** Training finished at 2026-10-03 02:20 CT. All pods have been deleted.
