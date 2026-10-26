# MolmoAct2-SO100_101 full fine-tune (`full_fine_tune`) on so101_bench_sim_WM

Full fine-tune of `allenai/MolmoAct2-SO100_101` on the SO-101 Bench simulated
teleop dataset (3,629 episodes, 1,996,120 frames), with the same four cameras the
GR00T N1.7 run used: `overhead`, `wrist`, and the settled frames captured when robot
control starts, `overhead_init` and `wrist_init`.

## Pins

| What | Value |
|---|---|
| MolmoAct2 | https://github.com/allenai/molmoact2 @ `66b87e64efd99dfd103241418113955cf64dfa9c`, plus `patches/molmoact2_66b87e6.patch` (registers the `so101_bench_sim_wm` mixture) |
| Base model | `allenai/MolmoAct2-SO100_101` @ `152569fe57914d97be91055800035f54e250d009` |
| Dataset | `5hadytru/so101_bench_sim_WM` @ `b569ad81ee60ffe3580bc0f1df513bde1bb36e8c` (LeRobot v3.0; the local copy under `data/lerobot/` is v2.1, which the vendored LeRobot cannot read) |
| Hardware | 4x A100-SXM4-80GB (NVLink), 192 vCPU, 944 GB RAM, RunPod volume disk at `/workspace` (local XFS block device) |
| Software | Python 3.12.3, torch 2.10.0+cu128, torchcodec 0.10.0, transformers 5.17.0 (`environment/`) |

## Joint frame

The dataset stores LeRobot range units (body joints in [-100, 100] over the real
robot's calibrated tick range, gripper in [0, 100]). The released checkpoint was
trained on calibrated degrees with `shoulder_lift` negated and +90 deg on
`shoulder_lift` and `elbow_flex` (the frame `SO101MolmoAct2Mapper` in
`source/so101_bench/so101_bench/utils/molmoact2.py` already produces for zero-shot
eval). `pod/make_molmoframe_dataset.py` rewrites `action` and `observation.state`
(and `meta/stats.json` plus the per-episode stats) into that frame, as a sibling
dataset `so101_bench_sim_WM_molmoframe` whose `videos/` is a symlink:

    model = scale * lerobot + offset
    scale  = [1.190330, -1.035604, 0.973626, 1.007033, 1.685714, 1.0]
    offset = [0, 90, 90, 0, 0, 0]

Checked against the existing mapper chain (`SO101JointMapper` ->
`SO101MolmoAct2Mapper`) on 58k sampled rows: max error 0.001 deg. Keeping the
checkpoint's convention keeps joint directions consistent with its pretraining, and
lets one client mapper serve zero-shot and fine-tuned checkpoints.

## Mixture (`patches/`)

`so101_bench_sim_wm`: `action` / `observation.state`, cameras in this fixed order:
`observation.images.overhead`, `observation.images.wrist`,
`observation.images.overhead_init`, `observation.images.wrist_init`;
`normalize_gripper=True`, setup text "single so100/so101 robotic arm in molmoact2",
control text "absolute joint pose" (both identical to the released SO-100/101 tag),
30-step action chunks (the checkpoint's horizon; evaluation executes the full chunk, 30, since step 7000 — steps 1000-6000 were validated at 16; 16 vs 30 ablations at steps 4000 and 6000 were within noise).

## Layout (`pod/`)

- `setup_env.sh` builds the venv at `/root/molmoact2-venv` (container disk) with the
  upstream Dockerfile's stack. Pitfalls it handles: a blanket `--reinstall` of torch
  pulls the newest numpy/pillow/fsspec from the PyTorch index (breaks the ai2-molmo
  and lerobot pins); `torchaudio` must be pinned to 2.10.0 or transformers fails to
  import; the launcher imports `debugpy` unconditionally.
- `verify_dataset.py`: `--sha` (every file vs the HF manifest, writes a local manifest),
  `--decode` (fully decodes all 616 videos and checks frame counts against the episode
  metadata), `--parquet` (row counts, contiguity, finite values), `--sample N`
  (re-hash N random files; used by the supervisor before every launch).
- `make_molmoframe_dataset.py`: the joint-frame conversion above.
- `run_train.sh`: the single launch command (every knob is an env var). Sets
  `HF_HUB_OFFLINE=1`, `LEROBOT_DATA_ROOT`, `MOLMO_DATA_DIR` (required even with no VLM
  data) and reads the W&B key from `/workspace/molmoact2-so101/secrets/wandb_api_key`.
- `bench.py`: the throughput/memory sweep (results below).
- `supervise.py`: resume-on-failure wrapper (see "Failure handling").
- `export_checkpoint.sh` + `to_bf16.py`: sharded training checkpoint -> self-contained
  bf16 HF checkpoint (~11 GB) for `scripts/molmoact2_server.py`.
- `open_loop_check.py`: loads an export like the server does and reports open-loop
  action error on random dataset frames (a plumbing check, not a success rate).
- `bootstrap_after_restart.sh`: after a pod stop/start, restores the venv from
  `/workspace/molmoact2-so101/venv_backup.tar` and restarts the supervisor.

## Throughput sweep

4 GPUs, 8 loader workers per GPU, PyAV decode, mean over 12-20 steps after 8 warmup
steps. Training is GPU-bound (GPU util 85-99%, CPUs mostly idle), so loader workers
and the video backend do not matter at these rates. Keyframes are every other frame
in these AV1 files, so a random frame costs about two decoded frames per camera.

| Per-GPU batch | Activation checkpointing | Samples/s (4 GPUs) | Peak GPU memory |
|---|---|---|---|
| 2 | on | 5.74 | 24.9 GB |
| 4 | on | 7.84 | 24.8 GB |
| 8 | on | 9.58 | 29.1 GB |
| **16** | **on** | **10.92** | **39.8 GB** |
| 32 | on | OOM at step 3 (single 24.5 GiB backward allocation, 43.9 GiB reserved-unallocated) | 61.1 GB at step 2 |
| 4 | off | 10.54 | 62.9 GB |
| 8 | off | OOM | - |

Chosen: 16 per GPU with activation checkpointing (fastest safe setting; 32 fails on
a batch-size-proportional backward allocation, likely full-vocabulary logits), 8
loader workers per GPU, 2 gradient-accumulation steps for a global batch of 128.
The live run measures about 11.6 s/step.

## Run settings (`pod/supervisor_config.json`)

| Setting | Value | Why |
|---|---|---|
| Start checkpoint | `allenai/MolmoAct2-SO100_101` | closest released embodiment |
| Trainable | everything (`--ft_vlm=true --ft_action_expert=true --ft_embedding=lm_head`) | upstream recipe for the released SO-100/101 checkpoint; the camera setup changes |
| Global batch | 128 (16 x 4 GPUs x 2 accumulation) | MolmoAct2's pretraining batch (theirs was packed, so effectively larger); 64 was catastrophic for GR00T. A 256 run (m3oeom9w) was aborted at step ~110 in favour of this |
| Learning rates | action expert 7e-5, LLM 1.4e-5, ViT 7e-6, connector 7e-6 | upstream 5e-5 / 1e-5 / 5e-6 / 5e-6 at batch 64, times sqrt(128/64) |
| Schedule | 200-step linear warmup, cosine to 10% of peak at `max_duration` (upstream, fixed in `train_lerobot.py`) | |
| Steps | 10,000 (1.28M samples, ~0.64 epochs of frames) | 36 h budget: 10,000 x ~11.6 s = ~32 h plus saves, slack for one restart |
| Checkpoints | ephemeral every 200 steps (~40 min, newest kept), regular every 500 steps (last 3 kept), each ~59 GB and ~20 s to write | crash loses at most ~40 min; saves cost <1.5% of wall time |
| Exports | every 1000 steps -> bf16 HF export (~11 GB, ~1 min, CPU only) | what the local monitor validates |
| Other | `--packing=false --dynamic_seq_len=true`, `--img_aug full` (upstream default), `norm_mode q01_q99`, 30-step chunks, PyAV decode | upstream custom-dataset recipe |

W&B: `5hadytru/finetune-molmoact2-so101`, run `krwz749d` (aborted batch-256 run: `m3oeom9w`).

## Failure handling

`pod/supervise.py` runs in tmux `supervisor` under `/usr/bin/python3` (stdlib only).
Before every (re)launch it checks: GPUs free, >=120 GB free on `/workspace` (pruning
exports if needed), a 48-file SHA-256 spot-check of the dataset plus a full parquet
scan, and that the newest `stepN` is complete (`config.yaml`,
`model_and_optim/.metadata`, `train/`) and fully readable (every byte read under a
timeout). Unreadable checkpoints move to `checkpoints/<run>/_unreadable/` and it falls
back to the previous one; the trainer's `allow_resume` then resumes from the newest
remaining `stepN` (partially written saves are `stepN-tmp` until renamed).
While training it restarts on a crash or on 20 min without a new step (45 min grace
at startup), with 120 s backoff and at most 6 restarts per 6 h. It stops
(`status.json` state `STOPPED`) if a failure repeats without progress, so the hourly
check-up can debug it. It also exports every 1000-step checkpoint in the background
and prunes exports once the local monitor has pulled them (`PULLED` marker), or
oldest-first if `/workspace` runs low.

Compared with the GR00T run's failure (a MooseFS FUSE mount losing chunks mid-run):
this pod's `/workspace` is a local XFS block device, the dataset was verified end to
end before training (SHA-256 of all 627 files vs HF, full decode of all 616 videos
with frame counts matched to the metadata, parquet scan), and every relaunch
re-verifies a sample. `HF_HUB_OFFLINE=1` keeps the run from depending on the Hub.

After a pod stop/start (container disk wiped): `bash /workspace/molmoact2-so101/bootstrap_after_restart.sh`.

## Local validation

`scripts/monitor_runpod_molmoact2_checkpoints.sh` (started by
`start_local_monitor.sh` in tmux `molmoact2_monitor`, log `logs/molmoact2_monitor.log`)
polls every 10 min, pulls each 1000-step export (~11 GB, ~11 min at ~17 MB/s),
validates the safetensors, marks it `PULLED` on the pod, serves it with
`scripts/molmoact2_server.py` (venv `~/workspace/molmoact2_server/.venv`: torch 2.10
cu128, transformers 5.17, FastAPI; the server reads the norm tag and camera order from
the export's `norm_stats.json`), and runs `scripts/molmoact2_eval.py` on the 39-episode
`tasks/real_gr00t_val_v2.jsonl` with its fixed layouts, action horizon 30 (16 before step 7000), cameras
`overhead,wrist` plus `--use_overhead_init --use_wrist_init`. Newest unvalidated export
first, so it never falls behind the run. The best 3 exports by success rate are kept in
`checkpoints/so101_MolmoAct2-SO101_WM_runpod_best/` (active cache + 3 retained = ~44 GB;
it refuses to download with <40 GB free). Results:
`outputs/molmoact2_runpod_checkpoint_validation_MolmoAct2-SO101_full_fine_tune/stepN/`.
Inference on the 3090: ~0.57 s per 30x6 chunk with four 640x480 images (bf16, 10 flow
steps), ~11.7 GB of GPU memory.

`checkup.sh` prints a one-shot status of the pod and the local monitor.

## Results (39-episode `real_gr00t_val_v2`, fixed layouts)

| Step | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 | 9000 | 10000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Execution horizon | 16 | 16 | 16 | 16 | 16 | 16 | 30 | 30 | 30 | 30 |
| Successes / 39 | 6 | 2 | 4 | 7 | (stopped at 3/31) | 6 | 7 | 5 | 7 | **5 (final, official)** |

- Horizon ablations on the same checkpoints: step 4000 16 -> 7/39, 30 -> 8/39; step 6000 16 -> 6/39,
  30 -> 6/39 (`outputs/molmoact2_runpod_checkpoint_validation_MolmoAct2-SO101_full_fine_tune_ah30/`). The default was
  switched to 30 (MolmoAct2's full chunk and its LeRobot synchronous-deployment default) from step 7000.
- Paired 4000-vs-6000 comparison: 4 both-success, 2 gained, 3 lost; no improvement beyond noise
  (±~8 pt on 39 episodes).
- Official checkpoint (user decision, 2026-09-29): step10000, the final checkpoint of the run. It was
  uploaded to Hugging Face by the user, and the local copy
  (`checkpoints/so101_MolmoAct2-SO101_WM_full_fine_tune/step10000`) was removed on 2026-09-30. All
  other MolmoAct2 checkpoints were deleted. This matches GR00T's rule of using its
  final-within-budget checkpoint (step 16000, ~35 h).
- The run is **under-trained**: the loss was still falling at the end, and the official recipes train
  ~5x longer (see "This run is under-trained" below).
- Per-step results: `outputs/molmoact2_runpod_checkpoint_validation_MolmoAct2-SO101_full_fine_tune/stepN/`.

## Why it trails GR00T N1.7 here (analysis, 2026-09-29)

Integration bugs were checked and ruled out: the joint-frame conversion matches the existing mapper to
<0.001 deg; camera order (training `camera_keys` order, positional at inference) and init-frame timing
(first recorded frame after the 1.5 s hold, both in replay and eval) match; horizon 16 vs 30 makes no
difference; in failed episodes the arm moves purposefully across its range (failures are mostly
"no manipulation attempt" = no successful grasp/lift), and it reaches goals in some episodes.

Most likely cause, budget: under the equal 36 h budget, the full fine-tune of 5.4B parameters at
378 px saw 1.28M samples (~0.64 epochs); GR00T N1.7, trained with its backbone frozen (NVIDIA's own
recipe: every official sim example, e.g. `examples/LIBERO/README.md` at Isaac-GR00T `51d4c89`, uses
`examples/finetune.sh` with the `tune_llm=False`, `tune_visual=False` defaults, 20k steps x batch 640),
got through ~12.8M. MolmoAct2's published results used 50k-100k updates on 32 H100s (paper section
4.3.1: LIBERO 50k at batch 64, SO-100/101 100k at batch 64), roughly 10-20x this compute. RoboDojo,
an independent benchmark, also reports MolmoAct2 at ~9 %.

### This run is under-trained

**The loss had not plateaued.** The training flow-matching loss was still falling steeply when the
budget ran out. Means over the preceding 200 steps, from W&B `krwz749d`:

| Step | 1000 | 3000 | 6000 | 9000 | 10000 (final) |
|---|---|---|---|---|---|
| `train/action_flow_loss` | 0.0443 | 0.0367 | 0.0278 | 0.0230 | 0.0195 |

It dropped another 15% in the last 1,000 steps alone. The run ended mid-descent, not at convergence.

**The official recipes train about 5x longer at a comparable per-step batch.**
- MolmoAct2's sim fine-tune (paper section 4.3.1, LIBERO) is **50,000 steps** at global batch 64,
  and every fine-tuning recipe in `experiments/README.md` at `66b87e6` uses the same budget:
  custom full, LoRA and action-expert-only, and DROID. This run did **10,000 steps**, 5x fewer.
- Those recipes pack 2,100-token sequences (`--packing=true --seq_len=2100`). Our examples are ~894
  tokens, so each packed sequence holds about two of them, and 64 packed sequences is roughly the
  same number of examples per step as our unpacked batch of 128.
- In samples, that's about 6.4M for the recipe versus 1.28M here (~0.64 epochs of our 2.0M frames).
  The SO-100/101 checkpoint itself was trained for 100k updates at batch 64.

Under-training is imposed by the benchmark's equal-budget rule (36 h on 4x A100), not by a recipe
choice. At ~11 samples/s on this hardware, 50k steps x 128 examples would take ~160 h.

Open question: success plateaued at ~15 % from step 4000 to 10000 while training loss kept falling
(0.04 -> 0.02), which pure under-training would not predict. Suggested diagnostic: open-loop action
error of the final MolmoAct2 and GR00T checkpoints on the same dataset frames. Similar open-loop fit but
worse closed-loop success -> a closed-loop/generalization gap; clearly worse open-loop fit ->
under-training (the budget explanation).

## Deviation from the paper's fine-tuning recipe (found 2026-09-29, after the run)

This run followed the custom-dataset command in the repo's `experiments/README.md`, which leaves two
settings at their launcher defaults. The paper (arXiv 2605.02881v2, Sec. 4.3.1) says every MolmoAct2
fine-tune instead uses:
- **8 sampled flow times per action chunk** (`--num_flow_timesteps=8`; this run used the default 1).
  Table 12 (LIBERO, from MolmoAct2-Pretrain): K=1 94.15% vs K=8 95.90% average.
- **Co-training of the discrete autoregressive action loss with the flow-matching expert**
  (`--action_format=both`; this run used the default `continuous`). Table 13: removing discrete
  co-training gives a similar average (96.95% vs 97.20%) but shifts per-suite results.
The paper's fine-tunes also use sequence packing (throughput only) and global batch 64; the SO-100/101
checkpoint itself was trained for 100K updates at batch 64 (6.4M packed sequences) on 32 H100s.
Table 13 also compares parameter sets at equal steps on LIBERO: full fine-tuning 97.20%, LoRA 96.25%
(-2.8 on Long), action expert only 93.05% ("the clearest failure mode").
