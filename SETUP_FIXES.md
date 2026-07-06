# SO-101 Bench — Setup Fixes & Session Worklog

This document records the fixes applied to make the repository run from a fresh
clone against **Isaac Lab 5.1 / Isaac Sim 5.1**, plus the verification steps run to
confirm the environment works end-to-end.

**Branch:** `fix/portable-paths`
**Date:** 2026-07-05

---

## TL;DR

The repo crashed on first launch. Four root causes were fixed:

1. A version-drifted import (`isaaclab.sim.views`) that no longer exists in Isaac Lab 5.1.
2. A broken/mislocated import + hardcoded `/home/truman/...` path in an asset script.
3. **Missing object footprint JSONs** — episode loading validates against them, and none were present.
4. **README commands that referenced non-existent files** and omitted a required `cd`.

After the fixes, all smoke tests pass: extension imports, `zero_agent` runs, all 10
Gym envs register, and episode loading + layout generation work.

Bringing up a **live GR00T policy evaluation** then surfaced a second, separate set of
issues — this time in the **Isaac-GR00T server environment** (`~/.venv`) and the GPU
stack, not in this repo. Those are documented in
[GR00T inference bring-up](#gr00t-inference-bring-up-environment-fixes) below.

---

## Fixes applied

### 1. `source/so101_bench/so101_bench/mdp/resets.py` — import crash

**Symptom**

```
ModuleNotFoundError: No module named 'isaaclab.sim.views'
```

**Cause**

`isaaclab.sim.views.XformPrimView` was never part of mainline Isaac Lab —
`XformPrimView` is the old (pre-4.2) Isaac Sim name. In Isaac Sim 5.1 the equivalent
vectorized class is `isaacsim.core.prims.XFormPrim`, which exposes the same
`set_world_poses` / `get_world_poses` / `.count` API this file uses.

**Fix**

```diff
-from isaaclab.sim.views import XformPrimView
+from isaacsim.core.prims import XFormPrim as XformPrimView
```

Low risk: the `XformPrimView` branches only run when
`env._so101_multi_rigid_body_info` is populated, and that attribute is **read** (in
`resets.py`, `groot_eval.py`, `molmoact2_eval.py`) but **never written** anywhere in
the repo — so those branches are inert as shipped and the import just needs to
resolve to a valid class.

> **Latent note:** `resets.py:702` calls `asset.device`, but `XFormPrim` only has a
> private `._device`. Harmless today (dead path); if the multi-rigid-body path is
> ever wired up, guard it with `getattr(asset, "device", asset._device)`.

### 2. `source/so101_bench/so101_bench/assets/make_usd_deformable.py` — bad imports + hardcoded path

- Removed a **dead, mislocated import**:
  `from isaaclab.sim.utils.prims import bind_physics_material` — there is no
  `prims.py` module (it lives in `isaaclab.sim.utils.utils`), and the symbol was
  never used in the script.
- Replaced the hardcoded `/home/truman/...` `USD_PATH` with a `__file__`-relative
  default plus an optional CLI override:

```python
import sys
from pathlib import Path
...
DEFAULT_USD_PATH = Path(__file__).resolve().parent / "usd" / "objects" / "brown_stuffed_animal.usdc"
USD_PATH = str(Path(sys.argv[1]).resolve()) if len(sys.argv) > 1 else str(DEFAULT_USD_PATH)
```

The other two imports in this file (`define_deformable_body_properties` from
`isaaclab.sim.schemas`, and `open_stage`/`get_current_stage`/`update_stage` from
`isaaclab.sim.utils.stage`) were verified valid and left unchanged.

### 3. Missing object footprint JSONs

**Symptom**

```
ValueError: Missing generated move-task footprint metadata for 'action figure':
.../assets/objects/action_figure.json. Run scripts/generate_object_move_footprints.py
```

**Cause**

`assets/objects/` was **empty** (zero JSONs, none tracked in git), even though the
README claims they are committed. Every episode load failed validation
(`validate_move_episode_footprints`).

**Fix**

Generated all footprints (pure `pxr` + numpy, no simulator boot required):

```bash
~/IsaacLab/isaaclab.sh -p scripts/generate_object_move_footprints.py
```

Produced **50 JSONs** (one per USD object), which were committed so the repo is
self-contained. The ~430 MB USD meshes remain gitignored and download-separately.

### 4. README corrections

| Issue | Fix |
|-------|-----|
| GR00T **server** command used a relative script path with no `cd` | Added `cd ~/Isaac-GR00T` + note it runs in that repo's own env |
| GR00T **eval** command referenced non-existent `tasks/layouts/real_gr00t_WM_combined_layouts.jsonl` | Removed the flag; documented that omitting it auto-samples and **saves** a timestamped layout file, with a snippet to re-pass it |
| `--episode_layouts_jsonl` flag doc | Clarified the omit-to-auto-sample-and-save behavior |
| Teleop + MolmoAct2 examples used non-shipped `tasks/real_gr00t_WM_seen_bin_1obj.jsonl` | Swapped to the shipped `tasks/real_gr00t_WM_combined.jsonl` |

Left unchanged (correct as written): the `WM_v7_50k/checkpoint-52000` checkpoint name
(machine-specific symlink issue, not a doc error) and the `tasks/teleop_1.*` replay
references (inherently user-recorded during teleop).

---

## GR00T inference bring-up (environment fixes)

Running an actual policy rollout needs the **Isaac-GR00T** server (`~/Isaac-GR00T`,
in its own `~/.venv`) talking to the `groot_eval.py` client (Isaac Lab env). Getting
one eval step to run surfaced four more issues, all environmental (not code in this
repo). They are listed in the order they appeared.

### A. GR00T server — transformers too new (`VideoInput` ImportError)

**Symptom**

```
ImportError: cannot import name 'VideoInput' from 'transformers.image_utils'
```

**Cause** — Isaac-GR00T's `pyproject.toml` pins `transformers==4.53.0`. In 4.52+,
`VideoInput` moved from `transformers.image_utils` to `transformers.video_utils`, but
the Eagle3-VL processor bundled with the checkpoint (`processing_eagle3_vl.py`, loaded
via `trust_remote_code`) still imports it from `image_utils`. Isaac-GR00T's own
sim-eval setup scripts actually pin the older `transformers==4.51.3`.

**Fix** — downgrade in the GR00T venv:

```bash
/home/zeux/.venv/bin/pip install "transformers==4.51.3"
```

### B. GR00T server — PyTorch didn't support the Blackwell GPU (`sm_120`)

**Symptom** (warning at model load; would fail at first inference)

```
NVIDIA RTX PRO 6000 Blackwell ... CUDA capability sm_120 is not compatible with the
current PyTorch installation. The current PyTorch install supports ... sm_50 ... sm_90.
```

**Cause** — the installed torch was `2.7.1+cu126`, whose arch list stops at `sm_90`.
The RTX PRO 6000 is Blackwell = `sm_120`. A plain `pip install` was a **no-op**
because the version string `2.7.1` already matched — pip ignores the `+cuXXX` build
tag, so it must be uninstalled first.

**Fix** — force the cu128 build:

```bash
/home/zeux/.venv/bin/pip uninstall -y torch torchvision
/home/zeux/.venv/bin/pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

Verify (want `2.7.1+cu128` and `sm_120` in the list):

```bash
/home/zeux/.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list())"
# 2.7.1+cu128  ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120','compute_120']
```

After this, the server restarts with **no `sm_120` warning**. (Note: the `nvidia-*-cu12`
libs may resolve back to 12.6 due to other pins in the venv; torch's own kernels carry
sm_120 regardless, so only revisit if inference throws a cuBLAS/cuDNN error.)

### C. Evaluator — `lerobot` missing in the Isaac Lab env

**Symptom**

```
ModuleNotFoundError: No module named 'lerobot'
```

`--record_dataset` requires `lerobot` in the **Isaac Lab** env (`~/IsaacLab/_isaac_sim`),
which is separate from the GR00T venv that already had it.

**Fix**

```bash
~/IsaacLab/isaaclab.sh -p -m pip install "lerobot==0.4.3"
```

### D. Evaluator — lerobot dragged numpy to 2.x, breaking Isaac Sim cameras

**Symptom** (during camera / render-product init in `sim.reset()`)

```
tiled_camera.py:226 annotator.attach(...) →
omni/syntheticdata/.../SyntheticData.py dep_attrib_data.set(dep_data)
TypeError: Unable to write from unknown dtype, kind=f, size=0
```

**Cause** — installing `lerobot` (step C) upgraded **numpy to 2.4.6** in the Isaac Lab
env. Isaac Sim's synthetic-data C++ bindings were built against numpy 1.x; the 2.x ABI
break produces the `unknown dtype` error. `pip check` confirmed the whole Isaac stack
requires `numpy<2` (tightest: `numba 0.59.1` → `numpy<1.27,>=1.22`, and
`cmeel-boost` → `numpy~=1.26.0`).

**Fix** — pin numpy back:

```bash
~/IsaacLab/isaaclab.sh -p -m pip install "numpy==1.26.4"
```

> **Lesson:** install `lerobot` into the Isaac Lab env **with a numpy guard** so it
> can't cross 2.0 in the first place:
> `~/IsaacLab/isaaclab.sh -p -m pip install "lerobot==0.4.3" "numpy<2"`
> (or `--no-deps` and add only the missing pure-Python deps).

### E. Recorder — corrupt/incomplete dataset directory blocks `--record_dataset`

**Symptom**

```
FileNotFoundError: ...does not contain any parquet file:
  data/lerobot/groot_n16_real_sim_1_ah16/meta/episodes
→ falls back to the Hub → 404 Not Found: 5hadytru/so101_bench_groot_eval
```

**Cause** — the recorder decides create-vs-reopen purely by whether `<repo_root>/meta`
exists (`utils/lerobot_dataset.py:310`, no force-recreate option):

```python
if meta_dir.exists():
    self.dataset = LeRobotDataset(self.repo_id, root=self.dataset_root)   # LOAD
else:
    self.dataset = LeRobotDataset.create(...)                            # CREATE
```

An earlier `--record_dataset` run crashed mid-recording, leaving a **half-written**
dataset: `info.json` says `total_episodes: 0`, `meta/episodes/` was never created, yet
orphan `episode-000000/` frames + a stray `data` parquet remain (~357 MB). `meta/`
exists → the recorder takes the LOAD path → LeRobot can't find `meta/episodes/*.parquet`
locally → falls back to the Hub → 404.

**Fix** — remove the corrupt dir (safe; 0 committed episodes) and re-run, or point at a
fresh `--repo_root`:

```bash
rm -rf data/lerobot/groot_n16_real_sim_1_ah16      # or use a new --repo_root
```

### Result — end-to-end run confirmed ✅

With A–E applied, the **full pipeline runs**: the evaluator boots the sim, initializes
cameras (numpy fix), connects to the policy, and — the milestone — executes real
**GR00T inference on the Blackwell GPU with no `sm_120` / cuBLAS error** (the cu128 fix
held), drives the arm, and **records mp4 video** into a LeRobot dataset. Confirmed on a
`So101Bench-Bin-v0` run over `tasks/custom_bin.jsonl`:

```
[INFO]: Policy server connected.
[INFO]: Active tabletop object(s): object_1 (black pen)
[INFO]: Episode 1/5: success=False, reason=time_out, length=25.00s, live_failure_reason=none
[INFO]: Resetting episode 2/5...
```

So the **setup is fully working**. Note `success=False, reason=time_out` on episode 1 —
the policy moved but didn't sink the object in time. That's a *policy/tuning* outcome,
not a setup bug, and is expected to be lower than the paper here because (a) the served
checkpoint is the **v6/55k** symlink, not v7, and (b) sim-vs-real camera framing differs
from what the policy trained on.

The two env stacks:

| Environment | Path | Key pins after fixes |
|-------------|------|----------------------|
| GR00T server | `~/.venv` | `transformers==4.51.3`, `torch==2.7.1+cu128` |
| Isaac Lab / evaluator | `~/IsaacLab/_isaac_sim` | `numpy==1.26.4`, `lerobot==0.4.3` |

---

## Verification (smoke tests)

| Test | Command | Result |
|------|---------|--------|
| Extension imports | — | ✅ resolve after `resets.py` fix |
| Debug agent | `zero_agent.py --task So101Bench-Bin-v0 --enable_cameras --headless` | ✅ scene + SO-101 + cameras load, physics steps |
| Env registration | `list_envs.py` | ✅ all 10 `So101Bench-*` tasks registered |
| Footprints | `generate_object_move_footprints.py` | ✅ 50/50 generated, no errors |
| Episode load + layout | `groot_eval.py --inspect_initial_scene` | ✅ 281 episodes validated, layout generated, scene built |
| **Full GR00T rollout** | `groot_eval.py ... --record_dataset` (server up) | ✅ GPU inference on Blackwell, arm driven, mp4 recorded; episodes score `success/time_out` |

**Registered environments (10):**
`So101Bench-Between-v0`, `So101Bench-Bin-Object1-v0` … `-Object4-v0`,
`So101Bench-Bin-SingleObject-v0`, `So101Bench-Bin-v0`, `So101Bench-Mixed-v0`,
`So101Bench-Move-v0`, `So101Bench-NextTo-v0`.

---

## Environment gotchas discovered

- **`list_envs.py` table doesn't appear when stdout is redirected** to a file/pipe —
  Isaac Sim's carb logger reroutes `print()`. Run it interactively to see the table.
  (Env registration was confirmed by writing the registry to a file directly.)
- **`timeout`/`Ctrl-C`-killed Isaac runs leak the grandchild Kit process** — the wrapper
  (`isaaclab.sh`) dies but the Python/Kit process survives and holds the Kit KVDB
  lock + GPU memory, slowing the next boot. These **accumulate**: over a session of
  crashed/killed runs, a dozen+ zombie `groot_eval` Kit processes piled up. Periodically
  check `ps -eo pid,args | grep groot_eval | grep -v grep` and kill leftovers **by
  explicit PID** (pattern-based `pkill` can be blocked and is risky). The GR00T *server*
  process (`run_gr00t_server.py`) is separate — don't kill it during cleanup.
- **GR00T checkpoint symlink mismatch (this machine):**
  `~/workspace/so101_GR00T_N1.6-3B_WM_v7_50k/checkpoint-52000` →
  `~/workspace/so101_GR00T-N1.6-3B_WM_v6_55k/checkpoint-55000`. Despite the
  `v7_50k / checkpoint-52000` name, the weights served are **v6 / 55k steps**.
  Loads and runs fine, but results reflect v6 unless the symlink is repointed.
- **`pip install pkg==X` is a no-op if version `X` is already present**, even when the
  CUDA build tag differs (`+cu126` vs `+cu128`). Uninstall first, or use
  `--force-reinstall`, to actually switch builds.
- **Installing ML packages into the Isaac Lab env is risky** — they pull numpy/torch
  and can silently break Isaac Sim's compiled bindings. Always constrain
  (`"numpy<2"`) or use `--no-deps`. Isaac Sim 5.1 needs `numpy 1.26.x`.

---

## Running a full GR00T evaluation

**Prerequisites (one-time, per the [bring-up fixes](#gr00t-inference-bring-up-environment-fixes) above):**
- GR00T venv: `transformers==4.51.3`, `torch==2.7.1+cu128` (for Blackwell `sm_120`).
- Isaac Lab env: `numpy==1.26.4`, `lerobot==0.4.3` (only if using `--record_dataset`).

1. **Start the policy server** (Isaac-GR00T repo, its own env). Runs long; wait for
   `Server is ready and listening on tcp://127.0.0.1:5555`:

   ```bash
   cd ~/Isaac-GR00T
   python gr00t/eval/run_gr00t_server.py \
     --model-path ~/workspace/so101_GR00T_N1.6-3B_WM_v7_50k/checkpoint-52000/ \
     --device cuda --host 127.0.0.1 --port 5555
   ```

2. **Run the evaluator** (this repo, Isaac Lab env). Recommended first pass: drop
   `--record_dataset` and add `--num_episodes 1` to isolate the sim→policy→action loop
   before adding dataset I/O:

   ```bash
   ~/IsaacLab/isaaclab.sh -p scripts/groot_eval.py \
     --task So101Bench-Bin-v0 \
     --episodes_jsonl tasks/real_gr00t_WM_combined.jsonl \
     --policy_host localhost --policy_port 5555 \
     --action_horizon 16 --use_overhead_init true \
     --record_dataset --repo_root data/lerobot/groot_n16_real_sim_1_ah16 \
     --headless
   ```

> **Two environments, two GPUs-worth of care.** The server (`~/.venv`) and the
> evaluator (`~/IsaacLab/_isaac_sim`) are **separate** Python environments — a fix in
> one does not apply to the other. Most of the bring-up pain came from fixing a
> package in the wrong env.

### Running with a custom object

A quick single-object pick-and-place is just a small task file (`tasks/custom_bin.jsonl`
was added as an example). One JSON object per line; object names must match
`OBJECT_SPLITS` (spaces intact):

```json
{"objects": ["blue scissors"], "instruction": "Place each object in the plastic bin"}
{"objects": ["blue scissors", "grey wires", "red tape"], "instruction": "Place each object in the plastic bin"}
```

```bash
~/IsaacLab/isaaclab.sh -p scripts/groot_eval.py --task So101Bench-Bin-v0 \
  --episodes_jsonl tasks/custom_bin.jsonl --policy_host localhost --policy_port 5555 \
  --action_horizon 16 --use_overhead_init true --num_episodes 3 \
  --record_dataset --repo_root data/lerobot/custom_bin_ah16
```

---

## Isaac Lab launcher reference (`isaaclab.sh`)

`~/IsaacLab/isaaclab.sh` is the entry point for everything in the Isaac Lab env. Key flags:

| Flag | Purpose |
|------|---------|
| `-p, --python <script>` | Run a script with Isaac Sim's Python (what all the eval/teleop commands use). |
| `-s, --sim` | Launch the bare Isaac Sim **GUI** (`isaac-sim.sh`), no script. |
| `-i, --install [LIB]` | Install Isaac Lab extensions / learning frameworks. |
| `-t, --test` | Run pytest. |
| `-f, --format` | Run pre-commit format/lint. |
| `-n, --new` | Scaffold a new external project / task from template. |
| `-c, --conda` / `-u, --uv [NAME]` | Create the Isaac Lab conda/uv env. |

**Headless vs GUI (watch the robot grasp):** add `--headless` to run with no window
(fastest, for batch recording). **Omit `--headless`** to open the Isaac Sim viewer and
watch the rollout live:

```bash
# Headless (batch/record):
~/IsaacLab/isaaclab.sh -p scripts/groot_eval.py ... --headless

# GUI (watch the arm grasp — drop --headless):
~/IsaacLab/isaaclab.sh -p scripts/groot_eval.py \
  --task So101Bench-Bin-v0 --episodes_jsonl tasks/custom_bin.jsonl \
  --policy_host localhost --policy_port 5555 \
  --action_horizon 16 --use_overhead_init true --num_episodes 6 \
  --record_dataset --repo_root data/lerobot/seen_unseen_grasp_ah16
```

GUI keys during a `groot_eval` run: `P` snapshot all cameras, `N` skip to next episode.

---

## Docker containerization

A `docker/` setup was added so the demo runs without the native env dance, modeled on
NVIDIA's [Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop)
`docker/sim` image. That reference **independently confirmed** the fixes above — it also
pins `numpy`, installs lerobot `--no-deps`, and bundles ffmpeg.

```
docker/
├── Dockerfile              # sim + eval image (base: nvcr.io/nvidia/isaac-lab:2.3.2)
├── entrypoint.sh           # Isaac env setup, extension install, asset fetch
├── download_assets.sh      # idempotent HF USD-asset download (~430 MB)
├── run.sh                  # launcher: --gpus all, --network=host, X11, caches
├── compose.yaml            # full demo: gr00t-server + sim wired together
├── gr00t-server/Dockerfile # GR00T policy server (reference)
└── README.md               # build / run / eval instructions
```

**Two images, because the pins conflict** (the same reason native bring-up was fiddly):

| Image | Encodes | Env |
|-------|---------|-----|
| `so101-bench` (sim/eval) | `numpy==1.26.4`, lerobot `--no-deps` + constraints, ffmpeg | Isaac Lab |
| `gr00t-server` (policy) | `torch==2.7.1+cu128` (Blackwell sm_120), `transformers==4.51.3` | CUDA 12.8 |

Run the full demo:

```bash
export GR00T_MODEL=~/workspace/so101_GR00T_N1.6-3B_WM_v7_50k
docker compose -f docker/compose.yaml up --build
```

**Built and tested (sim image):** the `so101-bench` image was built (28 GB) and run:
numpy stays pinned at `1.26.4`, GPU passthrough works (RTX PRO 6000 Blackwell seen
inside the container), Isaac Sim boots and registers all 10 envs, and the asset download
works from HF. `.dockerignore` excludes `data/`, mp4s, and USD meshes so the build
context stays small and reproducible.

### Build + runtime fixes found by actually running it

- **apt mirror is IPv6-only in this build sandbox** (`Temporary failure resolving
  archive.ubuntu.com`; Docker DNS is IPv4 `8.8.8.8`). The X11 libs are already in the
  Isaac Lab base image, so the apt step was made **best-effort** (`|| echo WARNING`).
  pypi/github worked fine, so the real payload installed.
- **`accelerate` (+ diffusers/wandb/pynput/cmake) were missing** — I trimmed the workshop
  dep list too far, and lerobot imports `accelerate` at import time
  (`ModuleNotFoundError` at `recorder.init_dataset()`). Restored the full workshop set;
  verified `LeRobotDataset.create()` works in the image.
- **Container was re-downloading the 430 MB assets** even though they exist locally —
  `so101.sh` now bind-mounts the host `assets/usd` read-only (the entrypoint then skips
  the download). Falls back to HF download if the local copy is absent.
- **Isaac Sim GUI window needs more than `-e DISPLAY`** — added the X socket mount
  (`-v /tmp/.X11-unix:/tmp/.X11-unix`) and `NVIDIA_DRIVER_CAPABILITIES=all` (exposes the
  GPU's graphics/display libs, not just compute). `so101.sh eval` is now windowed by
  default (`HEADLESS=1` to disable).

### Convenience scripts

- **`docker/so101.sh`** — one dispatcher: `build` / `test` / `eval` / `shell` / `run`.
  Handles `--gpus all`, `--network=host` (reach the `:5555` server), X11, caches, and the
  local-asset mount. `eval` overridable via `TASK` / `EPISODES` / `NUM_EPISODES` /
  `REPO_ROOT` / `HEADLESS` env vars.
- **`docker/pick.sh`** — interactive picker: lists all 49 objects (grouped by split),
  you choose object(s) + task family, it writes `tasks/picker.jsonl` and launches the
  eval. `DRY_RUN=1` to preview, `NATIVE=1` to run outside Docker.

**Caveats:** the sim image is built + tested; the `gr00t-server` image is a *reference*
(clones `NVIDIA/Isaac-GR00T`, applies our pins) and hasn't been built end-to-end. The
base `isaac-lab:2.3.2` must bundle Isaac Sim 5.1 (overridable via `ISAAC_LAB_IMAGE`).
The ~10 GB checkpoint is mounted, never baked.

---

## Object set reference

49 registered objects (all have USD + footprint present, so all are runnable). GR00T
was **fine-tuned only on the 23 "seen" objects** — the other 26 are generalization
tests. Defined in `benchmark.py` → `OBJECT_SPLITS`.

- **seen (23)** — reliable pick-and-place: black/silver glasses, white/black pen,
  altoids container, blue pliers, green clip, pink eraser, yellow/grey wires,
  black/yellow screwdriver, red/black tape, cardboard box, flower pot, cooking spoon,
  yellow/grey toy car, green/black shoes (multi-body), blue bowl, blue scissors.
- **unseen / seen-class (13)** — novel instance of a trained class: white glasses,
  blue clip, blue/yellow tape, blue screwdriver, pink/white bowl, black/brown wires,
  orange toy car, blue/red pen, white shoes.
- **unseen / unseen-class (13)** — entirely novel: blue highlighter, purple toothbrush,
  blue controller, action figure, razor, silver tongs, playing cards, candy bar,
  toy fire/monster truck, toy dinosaur, sponge, yellow flashlight.
- Commented out (no USD, reserved): brown/white stuffed animal (deformable),
  orange glasses, blue headband, baby doll.

---

## Files changed

```
source/so101_bench/so101_bench/mdp/resets.py                 # import fix
source/so101_bench/so101_bench/assets/make_usd_deformable.py # imports + portable path
source/so101_bench/so101_bench/assets/objects/*.json         # 50 generated footprints
README.md                                                    # command/reality fixes
tasks/custom_bin.jsonl                                       # example bin task file (seen objects)
tasks/custom_mixed.jsonl                                     # example: all 4 task families (4 objects each)
tasks/red_tape.jsonl                                         # example: single red-tape bin episode
docker/Dockerfile, entrypoint.sh, download_assets.sh         # sim image (built & tested, 28 GB)
docker/gr00t-server/Dockerfile, compose.yaml                 # policy server (reference) + full-demo compose
docker/run.sh, so101.sh, pick.sh                             # launcher, dispatcher, interactive object picker
docker/README.md                                             # docker build/run/eval docs
.dockerignore                                                # excludes data/, mp4s, USD meshes from build context
tasks/custom_mixed.jsonl, red_tape.jsonl, picker.jsonl       # example / generated task files
SETUP_FIXES.md                                               # this worklog
```

The GR00T bring-up fixes (A–E) are **environment changes** (pip installs in `~/.venv`
and the Isaac Lab env), not edits to this repo — they must be reapplied on any machine.

## Open / follow-up items

- Commit the README fix (and this file) onto `fix/portable-paths`.
- Two still-untracked util files not part of these fixes:
  `source/so101_bench/so101_bench/utils/{lerobot_dataset.py, molmoact2.py}`
  (`molmoact2.py` is needed by the MolmoAct2 eval workflow).
- ✅ **End-to-end run verified** — full rollout executes on the Blackwell GPU and records
  video. Remaining is *policy quality*, not setup: early episodes score `time_out`.
- **Improve success rate (tuning, not setup):** repoint the checkpoint symlink to the
  intended **v7** model, and/or reconcile sim camera framing with the real rig
  (`INNOMAKER_WRIST_CAMERA_*` in `so101_bench_env_cfg.py`, compare via
  `scripts/view_wrist_camera.py`). Consider a longer `episode_length_s` if runs
  `time_out` before the policy finishes.
- **Recorder hardening (optional):** the create-vs-load check (fix E) trusts a bare
  `meta/` dir. A crashed run leaves a `meta/` with `total_episodes: 0` that then blocks
  the next run. Could detect an incomplete dataset and recreate instead of failing.
- ✅ **Docker sim image built + tested** — builds, GPU passthrough, Isaac Sim boot, env
  registration, asset provisioning, and `LeRobotDataset.create()` all verified in-container.
- **Docker: build the `gr00t-server` image** — still only a reference; hasn't been built
  against a real Isaac-GR00T checkout.
- **Docker GUI window** — the X11 wiring is in place but window *rendering* wasn't
  verifiable from this session (no display/TTY). Confirm `./docker/so101.sh eval` paints a
  window on the machine with the physical display.
- Optional: repoint the GR00T checkpoint symlink to the intended v7 model (currently
  serves v6 / 55k).
- Optional: update `make_usd_deformable.py` further to take a proper `--usd_path`
  argparse flag instead of positional `sys.argv[1]`.
