# FLUX 3 Action fine-tuning on so101_bench_sim

FLUX 3 Action (Black Forest Labs, 7B video-action DiT) fine-tuned on
`5hadytru/so101_bench_sim` (no init frames: overhead + wrist only). Two paths were benchmarked on
a 4x A100-SXM4-80GB RunPod pod (NVLink, 128 vCPU, 250 GB container disk; `/workspace` is MooseFS
and unused):

- **FFT**: standalone `black-forest-labs/flux-action` full fine-tune (`configs/so101/train.json`,
  FSDP2 over 4 GPUs), initialized from the released SO-101 policy with its `dit.` key prefix
  stripped, so its `action`/`action_cond` heads load as-is.
- **LoRA**: LeRobot `main` `examples/flux3/lora.json` (rank 32 on every attention/MLP projection,
  heads trained in full) on the released SO-101 package.

## Pins

| What | Value |
|---|---|
| flux-action | `e2dd1d8dbc5977b54315d61f7548c63c043d6d4f` (torch 2.10 cu128, NATTEN 0.21.6) |
| LeRobot | `e0d50211ef236143ae867228662b7dfaba554f02` (torch 2.11 cu128, NATTEN 0.21.6) |
| `black-forest-labs/flux-3-action-so101` | `c9e13b2aca6a0a472b3ea03fd90cff32d3e85849` (weights identical to main) |
| `black-forest-labs/flux-3-action-base` | `62878e2925e59b7a89ec14463ce89932624c490d` (VAE, Qwen3-VL-4B text encoder, trunk) |
| Dataset | `5hadytru/so101_bench_sim` @ `34233754992fbd77e42373bad44fd42ddfb43f14` (3,629 episodes; index: 3,619 train / 10 val, 1.84M valid window starts) |

## Layout (`pod/`)

- `setup_env.sh`: builds both venvs under `/root/flux3`, downloads weights and the dataset, writes
  the stripped SO-101 trunk and the `index-lerobot` index (`--chunk-size 42 --val-episodes 10`,
  `scene=observation.images.overhead`, `wrist=observation.images.wrist`).
- `bench.py`: the throughput/VRAM sweep. Each point is a real training run, timed over 8 updates
  after 3 warmup updates, then killed before any checkpoint. Results go to
  `/root/flux3/records/bench_results.jsonl`.
- `ckpt_export_test.sh`: 3 FFT updates, then the resumable checkpoint and a bf16 `ema_0p10` export, all
  timed.

## Throughput sweep (2026-10-01, global batch 128)

| Point | Windows/s | s/update | Peak memory | Notes |
|---|---:|---:|---|---|
| FFT micro 8, 2 EMAs (recipe default) | OOM | | | Adam moments + two fp32 EMA shadows arrive after update 1 |
| FFT micro 4, 2 EMAs | 4.09 | 31.3 | 76.5 GB allocated | |
| FFT micro 2, 2 EMAs | 4.05 | 31.6 | 74.8 GB | activations are a small share; state dominates |
| FFT micro 4, `attn_mode=cudnn` | 4.06 | 31.6 | 76.5 GB | no gain |
| FFT micro 4, 1 EMA (`ema_sigma_rels=[0.10]`) | 4.70 | 27.2 | 70.1 GB | each fp32 power-EMA update costs ~4 s |
| FFT micro 8, 1 EMA, `vae_batch_windows=4` | 4.72 | 27.2 | 74.1 GB | |
| FFT micro 4, no EMA | 4.93 | 26.0 | 63.6 GB | |
| FFT `compile_model=true` | error | | | upstream: "accessing tensor output of CUDAGraphs that has been overwritten" in the VAE |
| FFT `reshard_after_forward=false` | OOM | | | |
| LoRA micro 8 (preset, grad ckpt) | 5.69 | 22.5 | ~44 GB (nvidia-smi) | |
| LoRA micro 4 (grad ckpt) | 5.53 | 23.1 | ~38 GB | |
| LoRA micro 2 (grad ckpt) | 5.12 | 25.0 | ~35 GB | |
| LoRA micro 12 (grad ckpt) | 5.69 | 16.9 | ~51 GB | global batch 96 (128 not divisible by 48); rate corrected |
| LoRA micro 4, no grad ckpt | 6.83 | 18.8 | ~79 GB | too close to the limit |
| LoRA micro 4, no grad ckpt, 8 workers | 6.83 | 18.8 | ~79 GB | not data-bound |
| LoRA micro 2, no grad ckpt | 6.40 | 20.0 | ~54 GB | |

FFT data wait was ~0 throughout (8 PyAV workers per rank); GPU utilization was 80–95%. Model load
takes ~110 s per launch.

## Checkpoint and export (FFT micro 4, 1 EMA)

- Resumable `step-N` (model + Adam + one EMA): **104 GB**, written in 42 s.
- `export-checkpoint --profile ema_0p10 --dtype bfloat16`: **13.9 GB** in 70 s.
- Pod-to-pod pull to the A40 validation pod: 193 s.
- The 250 GB container disk (117 GB used by venvs, weights and data) holds only **one** resumable
  checkpoint.

## The full run: `flux3-so101-lora-b128` (LoRA, 2026-10-01)

W&B: `5hadytru/finetune-flux3-so101/9ixare4e`. Pod: 4x A100-SXM4-80GB (driver 580.159.04), launched
02:52 UTC.

| Setting | Value | Source |
|---|---|---|
| Recipe | LeRobot `examples/flux3/lora.json`: rank 32 / alpha 32 on every attention/MLP projection, action heads trained in full | preset |
| Base package | `flux-3-action-so101` with so101_bench_sim's own normalization (`so101_ourstats`, built by `make_stats_package.py` from the index statistics) | decided: the released processors carry `VoicAndrei/so100_kitchen` quantiles |
| Batch | micro 2 x 4 GPUs x accumulation 16 = 128 | swept (micro 2 without grad checkpointing: 6.40 windows/s, ~54 GB) |
| Gradient checkpointing | off | swept |
| Optimizer | AdamW, LR 1e-4 constant (no warmup or decay), betas 0.9/0.999, eps 1e-8, weight decay 0, clip 1.0 | preset |
| Length | 6280 updates = 100,480 LeRobot microsteps (~0.40 epochs of 1.996M frames) | fills 36 h at the measured 20.6 s/update |
| Checkpoints | every 500 updates (8000 microsteps); raw adapters plus EMA (decay 0.999) | preset EMA |
| Data | `5hadytru/so101_bench_sim` @ `3423375`, overhead renamed to the model's `scene` camera plus wrist, no init frames, no held-out split | decided |
| Workers / seed | 4 / 42 | preset |

Exact config: `records/lora_run/resolved_config/` (`train_config.json`, `lora.json`,
`adapter_config.json`, `config.json`). Normalization package: `records/lora_run/so101_ourstats/`.
Environment: `records/lora_run/environment/` (frozen LeRobot and flux-action venvs, A40 server venv,
versions). Logs: `records/lora_run/logs/` (setup, supervisor, export loop).

History: launched for 6480 updates at the sweep's 20.0 s/update. The run measured 20.6 s/update, so
`retarget_steps_at_checkpoint.sh` stopped it at the first checkpoint (update 500, 05:48 UTC) and
resumed with 6280 updates. With a constant LR only the stopping point changed, so `run_lora_full.sh`
(now defaulting to 6280) reproduces the run.

### Validation

- **Export.** `pod/export_loop.sh` runs on the training pod's CPU. It merges the raw and EMA adapters at each scheduled update into full packages (`merge_lora_package.py`) and deletes each export once it has been pulled.
- **Pull and validate.** Each A40 runs `scripts/validation_pod/monitor_flux3_lora.sh` (`VARIANT=raw` on one A40, `VARIANT=ema` on the other). It pulls the export and serves it with `scripts/flux3_server.py --package ... --save_predictions` (stateless, 8-tick history). It then runs `scripts/validation_pod/run_flux3_validation.sh`:
  - 39-episode `real_gr00t_val_v2` with fixed layouts, seed 0,
  - action horizon 32 of the 42 predicted,
  - `--history_ticks 8`, LeRobot joint frame, no init frames,
  - LeRobot dataset recorded,
  - predicted videos composed side by side with the rollout (`scripts/compose_flux3_prediction_videos.py`).
- **Schedule.** Updates 1000, 2000, 3000, 4000, 5000 and 6280. Update 6000 was skipped at the user's request.
- **Results.** They are copied to `outputs/flux3_runpod_checkpoint_validation_LoRA_{raw,ema}/` by `ship_validation_results.sh`.

| Checkpoint | Raw | EMA |
|---|---|---|
| zero-shot (released SO-101 package) | 0/39 | — |
| update 1000 | 1/39 | 1/39 |
| update 2000 | 4/39 | 1/39 |
| update 3000 | **5/39** | 1/39 |
| update 4000 | 2/39 | **4/39** |
| update 5000 | 3/39 | 2/39 |
| update 6280 (final) | 3/39 | 2/39 |

The run finished at 2026-10-02 15:13 UTC, at 0.40 epochs. The best checkpoints are raw update 3000 (12.8%) and EMA update 4000 (10.3%). **Official checkpoint (user decision, 2026-10-06): raw update 3000** (LeRobot checkpoint `048000` = update 3000; its `pretrained_model` adapter merges with `pod/merge_lora_package.py`).

- **Where the successes are.** Almost all of them are `named_bin`. The final checkpoints scored 0 on `move`, `next_to` and `bin_4obj`. Raw got 1/9 on `between`.
- **Where the files are.** Hugging Face [`5hadytru/so101_bench_flux-3-action-so101`](https://huggingface.co/5hadytru/so101_bench_flux-3-action-so101) (commit `a74b588`): the official raw update-3000 adapter at the root, and every checkpoint of the run (raw + EMA adapters, training state) under `run/checkpoints/`. Verified by SHA-256; the local copies were deleted 2026-10-06.

### Reproduce

1. On a 4x A100-80GB pod, run `ROOT=/root/flux3 bash pod/setup_env.sh`. It builds the pinned venvs, the weights, the dataset, the index and the `so101_ourstats` package. Put the W&B key in `/root/flux3/secrets/wandb_api_key`.
2. Start training: `tmux new -s train 'bash pod/run_lora_full.sh'`. Then start `bash pod/export_loop.sh` in a second tmux session.
3. On each validation A40:
   - run `scripts/validation_pod/setup_isaac.sh` and `setup_flux3_server.sh`;
   - start `VARIANT=raw|ema bash scripts/validation_pod/monitor_flux3_lora.sh`;
   - set the pod-to-pod key `/root/.ssh/val_to_train` and `REMOTE`/`REMOTE_PORT`.

The validation task file (`tasks/real_gr00t_val_v2.jsonl`) contains held-out objects, so it is kept out of the public repo.
