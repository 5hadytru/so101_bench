# SO-101 Bench for LeRobot

`lerobot_env_so101_bench` makes [SO-101 Bench](https://so101-bench.github.io/) a LeRobot evaluation
environment: `lerobot-eval --env.type=so101_bench`. LeRobot discovers installed `lerobot_env_*`
packages automatically, so no LeRobot changes are needed.

State and actions use the LeRobot SO-101 follower units (body joints in [-100, 100], gripper in
[0, 100]) and the cameras are `overhead` and `wrist` (480x640 RGB), so a policy trained on SO-101 Bench
data, or any LeRobot SO-101 dataset with the same two views, runs without adapters.

## Install

SO-101 Bench runs in Isaac Sim 5.1 (Python 3.11) and LeRobot needs Python >= 3.12, so they live in two
environments; the plugin starts the simulator as a subprocess and talks to it over a local socket.

1. Install SO-101 Bench and Isaac Lab 2.3 as in the [main README](../../README.md) (the simulator
   environment).
2. In your LeRobot environment (Python >= 3.12):

```bash
pip install -e integrations/lerobot          # from the so101_bench checkout; needs lerobot >= 0.6.1
export SO101_BENCH_ROOT=/path/to/so101_bench
export ISAACLAB_SH=/path/to/IsaacLab/isaaclab.sh
export SO101_BENCH_ISAAC_VENV=/path/to/isaac/venv   # if Isaac Lab runs from a venv
```

## Evaluate

The public validation set (`tasks/val_set.jsonl`, 135 episodes) is the default episode set:

```bash
lerobot-eval --env.type=so101_bench --policy.path=<your policy> --eval.n_episodes=1
python -m lerobot_env_so101_bench.score outputs/eval/<run>/eval_info.json
```

Each task is one fixed episode (objects, layout, instruction, appearance), so use
`--eval.n_episodes=1`. Episodes run one at a time in one simulator; `--eval.batch_size` is ignored.

The score is the mean success over the 15 task-family x object-split cells, the suites of the run
(`seen/bin_1`, ..., `unseen_unseen_class/move`); `lerobot-eval`'s overall `pc_success` averages
episodes instead, so use the scorer above for numbers comparable with the leaderboard.

Options (`--env.<name>`):

| Option | Default | Meaning |
|---|---|---|
| `task` | `val_set` | Episode set (`val_set`, `mmrv_eval`) or a tasks JSONL path (then also set `layouts`) |
| `episodes` | all | Trial-id subset, e.g. `0-9,20` |
| `cameras` | `overhead,wrist` | Policy cameras; add `overhead_init,wrist_init` for the frames at robot start |
| `max_joint_step_deg` | 15 | Per-tick joint-target clamp (leaderboard setting) |
| `initial_hold_time_s` | 1.5 | Initial-pose hold before the first observation (leaderboard setting) |
| `sim_log` | `outputs/lerobot_sim_server.log` | Simulator log |

The SO-101 Bench pi0.5 checkpoint, for example (it uses the start-of-episode frames, and its config
needs a LeRobot newer than 0.6.1, i.e. a recent `main`):

```bash
lerobot-eval --env.type=so101_bench --env.cameras=overhead,wrist,overhead_init,wrist_init \
    --policy.path=5hadytru/so101_bench_pi05 --eval.n_episodes=1
```

## How it works

- `sim_server.py` runs inside Isaac Lab and reproduces the leaderboard evaluator's episode loop: every
  object of the episode set is pre-spawned once and the scene is reconfigured per episode, the robot
  holds its initial pose for 1.5 s, joint targets are clamped to 15 deg per tick, and the
  benchmark's own success and timeout terminations end the episode.
- `client.py` starts that server (one per LeRobot process) and speaks msgpack over ZMQ. `lerobot-eval`
  builds the envs before it loads the policy, and a large policy can briefly need most of a 32 GB
  machine's RAM while loading (pi0.5 peaks at ~19 GB), so the client starts the simulator once to read
  the episode list, stops it, and starts it again for the first episode, after the policy is loaded.
- `env.py` exposes each episode as a `gym.Env` returning `pixels` and `agent_pos` and
  `info["is_success"]`; `config.py` registers `--env.type=so101_bench`.

`tests/fake_sim_server.py` speaks the same protocol without Isaac Sim, for testing the LeRobot side:

```bash
FAKE_PYTHON=$(which python) lerobot-eval --env.type=so101_bench \
    --env.isaaclab_sh=integrations/lerobot/tests/fake_isaaclab.sh \
    --policy.type=act --policy.device=cpu --eval.n_episodes=1 --output_dir=/tmp/so101_fake_eval
```
