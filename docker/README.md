# SO-101 Bench — Docker

Containerized Isaac Lab simulation + GR00T evaluation for SO-101 Bench.
Adapted from NVIDIA's [Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop)
`docker/sim` image, with the dependency fixes this repo required baked in.

## Prerequisites

- NVIDIA GPU + recent driver (Blackwell/RTX PRO 6000 supported)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Docker (with `docker compose`)
- NGC access to pull `nvcr.io/nvidia/isaac-lab` (`docker login nvcr.io`)

> **Version note:** the base image `nvcr.io/nvidia/isaac-lab:2.3.2` must bundle
> **Isaac Sim 5.1** — this repo's code uses `isaacsim.core.prims` and is validated
> against 5.1. Override with `--build-arg ISAAC_LAB_IMAGE=...` if you need a different tag.

---

## Quick commands (`docker/so101.sh`)

One short script for the common actions:

```bash
./docker/so101.sh build          # build the sim image
./docker/so101.sh test           # smoke test: boot Isaac Sim + list envs
./docker/so101.sh eval           # run a GR00T eval (server must be up on :5555)
./docker/so101.sh shell          # interactive shell in the container
./docker/so101.sh run <cmd...>   # arbitrary command in the container
```

`eval` is overridable via env vars, e.g.:

```bash
EPISODES=tasks/red_tape.jsonl NUM_EPISODES=1 REPO_ROOT=data/lerobot/red_tape \
  ./docker/so101.sh eval
```

Everything below is the underlying long-form equivalents.

---

## Two images

| Image | What it is | Env |
|-------|-----------|-----|
| **`so101-bench`** (`docker/Dockerfile`) | Isaac Lab sim + eval client (this repo). Runs smoke tests, teleop, replay, scoring, and `groot_eval.py`. | Isaac Lab / numpy 1.26.4 |
| **`gr00t-server`** (`docker/gr00t-server/Dockerfile`) | GR00T-N1.6 policy server on `:5555`. *Reference* — needs the checkpoint mounted. | cu128 torch / transformers 4.51.3 |

They are **separate** because their torch/transformers pins conflict — the whole reason
bringing this up natively was fiddly.

---

## Quick start (full demo via compose)

```bash
# Point at the directory that contains checkpoint-52000/
export GR00T_MODEL=~/workspace/so101_GR00T_N1.6-3B_WM_v7_50k
export HF_TOKEN=hf_...        # optional, for gated asset/model downloads

docker compose -f docker/compose.yaml up --build
```

This builds both images, starts the policy server, waits for it to load, then runs a
5-episode seen-object bin eval and records a LeRobot dataset into `./data`.

---

## Sim image only

### Build

```bash
docker build -t so101-bench:latest -f docker/Dockerfile .
# self-contained (bakes the ~430 MB USD assets into the image):
docker build --build-arg DOWNLOAD_ASSETS=true -t so101-bench:latest -f docker/Dockerfile .
```

If you did not bake assets, they are downloaded from Hugging Face on first run
(`docker/download_assets.sh`, idempotent).

### Run

```bash
./docker/run.sh                       # interactive shell
./docker/run.sh /workspace/isaaclab/isaaclab.sh -p scripts/list_envs.py
```

The eval (needs the server up on `:5555` — native or the `gr00t-server` image):

```bash
./docker/run.sh /workspace/isaaclab/isaaclab.sh -p scripts/groot_eval.py \
  --task So101Bench-Bin-v0 --episodes_jsonl tasks/custom_bin.jsonl \
  --policy_host localhost --policy_port 5555 \
  --action_horizon 16 --use_overhead_init true \
  --num_episodes 5 --record_dataset --repo_root data/lerobot/seen_bin_ah16 --headless
```

`run.sh` uses `--network=host` (reach the server on localhost), `--gpus all`, mounts the
Isaac/HF caches for fast re-runs, and bind-mounts `tasks/` and `data/` so your task files
and recorded datasets live on the host. Drop `--headless` for the GUI viewer (the script
already runs `xhost +local:root`).

---

## GR00T server image

```bash
docker build -t gr00t-server:latest -f docker/gr00t-server/Dockerfile .
docker run --rm -it --gpus all --network=host \
  -v ~/workspace/so101_GR00T_N1.6-3B_WM_v7_50k:/models:ro \
  gr00t-server:latest \
  --model-path /models/checkpoint-52000 --device cuda --host 0.0.0.0 --port 5555
```

Wait for `Server is ready and listening on tcp://0.0.0.0:5555`.

---

## What's baked in (and why)

These reproduce the fixes documented in `../SETUP_FIXES.md`:

- **`numpy==1.26.4` pinned** — numpy ≥2 breaks Isaac Sim's synthetic-data/camera
  bindings (`unknown dtype, kind=f, size=0`).
- **lerobot installed `--no-deps`** under a constraints file — a bare install drags
  numpy past 2.0.
- **ffmpeg** — required for LeRobot video (`--record_dataset`).
- **GR00T server: cu128 torch + transformers 4.51.3** — Blackwell `sm_120` support and
  the checkpoint's Eagle3-VL `VideoInput` import.

## Notes / caveats

- The sim image follows a **proven** NVIDIA workshop pattern; the `gr00t-server` image is
  a **reference** — adjust to your exact Isaac-GR00T checkout / model layout.
- Building pulls a multi-GB Isaac Lab base image; first build is slow.
- GUI viewer needs a local X server (Linux). Headless works anywhere with a GPU.
- The checkpoint is mounted, never baked (it's ~10 GB and machine-specific).
