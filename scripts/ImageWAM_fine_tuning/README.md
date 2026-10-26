# ImageWAM fine-tune on so101_bench_sim

ImageWAM ([paper](https://arxiv.org/abs/2606.19531), [code](https://github.com/yuyangalin/ImageWAM)) is a world-action model built on an image-editing model.
- **Backbone:** the FLUX.2 klein-base-4B editing transformer does one forward pass, and its attention KV conditions a ~642M flow-matching action DiT. Training also keeps an editing loss on the chunk's last frame (λ_video 0.5, λ_action 1.0).
- **Start point:** the released `yuyangalin/ImageWAM-FLUX.2-4B-InternData-A1-EE` weights, pretrained for 60k steps on InternData-A1 (Franka, Lift2, split ALOHA, Genie-1; no SO-101).
- **Inputs:** overhead and wrist cameras, with no history and no init frames.

Hardware:
- **Training:** RunPod 4x A100-SXM4-80GB (driver 580.126.16) with a 400 GB container disk and a ~54-core CPU quota.
- **Validation:** planned for the local 3090.

## Pins

| What | Value |
|---|---|
| ImageWAM repo | `5d4a341ed20a95cdb08f0293f3d44778b9a9e05a`. Python 3.11, torch 2.7.1 cu118, transformers 4.56.1, torchcodec 0.5 (CPU, PyPI). Full freeze in `records/requirements-frozen.txt` |
| FLUX.2 source | `black-forest-labs/flux2` @ `50fe516` (README pin) |
| Weights | FLUX.2-klein-base-4B @ `a3b4f48`; FLUX.2-dev `ae.safetensors` @ `26afe3a`; Qwen/Qwen3-4B @ `1cfa9a7` (text cache only); InternData-A1-EE pretrain @ `51a71ed` |
| Dataset | `5hadytru/so101_bench_sim` @ `3423375`: 3,629 episodes, 1,996,120 frames, 30 fps, 1,495 task prompts. Converted to LeRobot v2.1 |

## SO-101 adaptation

These files are copied into the repo by `pod/prepare_data.sh`; they are the only changes to the repo.

- **`src_patch/so101.py`:** `SO101ToCanonicalEE16`. The 6 joint values go into the pretrain's 16-dim action/state layout:
  - The 5 arm joints go in slots 0–4 and the gripper in slot 7, where the pretrain kept its gripper.
  - The other 10 dims are masked: the normalizer zeroes them and the action loss ignores them.
  - This mirrors upstream's `RobotWin14ToCanonicalEE16`. Joints are not converted to end-effector poses.
- **`configs/data/so101_ee16.yaml`:**
  - Normalization is z-score from this dataset.
  - The default canvas is the pretrain one: compact 288×256, with overhead in the 192×256 top slot, wrist at 96×128 bottom-left, and a black third slot.
  - Video augmentation is upstream's.
  - The chunk is 32 actions at native 30 fps, and the editing target is the chunk's last frame.
- **`configs/task/so101_flux2_klein_4b_ee16.yaml`:** the official RoboTwin ee16 fine-tune's optimizer and schedule (AdamW 0.9/0.95, wd 1e-2, cosine with 5% warmup, bf16, ZeRO-1). Weights are loaded with `resume=<model.pt>`, which loads weights only (0 missing / 0 unexpected MoT keys).

## Pod scripts

- **`pod/setup_env.sh`:** the venv, repo, weights and dataset. The repo's `uv.lock` predates its `shared` extra, so the script installs from `pyproject`'s pins instead. The cu118 torchcodec wheel needs CUDA 11 NPP libraries, so it uses the PyPI CPU build.
- **`pod/prepare_data.sh`:** converts v3.0 → v2.1 (InternW0's converter, ~17 min with 48 workers), computes stats (`pod/compute_stats.py`) and builds the Qwen3 text cache.
- **`pod/run_train.sh`:** one launch, with every knob as an environment variable: `MICRO ACCUM MAX_STEPS LR WARMUP SAVE_EVERY ZERO GRAD_CKPT CANVAS CHUNK LORA WORKERS WANDB RUN_NAME OUT_ROOT RESUME`.
- **`pod/bench.py`:** the throughput sweep.
- **`rent_pod_retry.sh`:** retries renting the 4xA100 SXM pod. On 2026-10-07 none was free at first; one appeared within 4 min.

## Throughput sweep (2026-10-07)

The sweep covers 4 GPUs at global batch 128 unless noted. Each point is a real run from the pretrain weights with augmentation on. Times come from the second pass, which polled every 0.5 s over 20 optimizer steps; the first pass polled every 3 s over 8 steps.

| Point | Micro x accum | Options | Samples/s | s/step | Peak (nvidia-smi) |
|---|---|---|---:|---:|---:|
| p_m4 | 4 x 8 | | 30.00 | 4.27 | 45.9 GB |
| **p_m8** | **8 x 4** | **full FT, pretrain canvas, chunk 32** | **33.06** | **3.87** | **59.0 GB** |
| m16_z1, m32_z1 | 16 x 2, 32 x 1 | | OOM | | |
| best_nockpt (pass 1) | 8 x 4 | `mot_checkpoint_mixed_attn=false` | 33.18 | 3.86 | 59.7 GB (no effect) |
| best_z2 (pass 1) | 8 x 4 | ZeRO-2 | 29.86 | 4.29 | 58.6 GB |
| best_chunk16 (pass 1) | 8 x 4 | 16-action chunk | 33.18 | 3.86 | 59.0 GB (chunk length costs nothing) |
| p_tall_m8 | 8 x 4 | 384x256 two-tile canvas | 25.58 | 5.00 | 65.8 GB |
| p_official_m12 | 12 x 8 | the official per-GPU batch, global 384 | 35.05 | 10.96 | 72.3 GB |
| p_lora_m8 | 8 x 4 | the repo's r16 LoRA on FLUX.2, action expert full | 37.96 | 3.37 | 36.9 GB |
| p_lora_m16 | 16 x 2 | LoRA | 39.83 | 3.21 | 59.2 GB |

The data loader is never the bottleneck: data time per micro-step is about 2 ms. For full fine-tuning, each micro-step at micro 8 takes 0.32 s forward and 0.76 s backward.

Checkpoints: each save writes 8.5 GB of weights (`weights/step_*.pt`) plus a 60 GB resumable DeepSpeed state (`state/step_*`, only the latest kept), in about 38 s.

**LoRA caveat:** with `flux2_lora_config.enabled=true`, loading the full-FT pretrain checkpoint reports 400 missing and 80 unexpected MoT keys. The 80 wrapped Linear layers keep plain FLUX.2 weights instead of the InternData-pretrained ones, so a LoRA run needs a key remap first.
