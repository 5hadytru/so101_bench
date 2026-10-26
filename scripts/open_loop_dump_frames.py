#!/usr/bin/env python3
"""Dump query frames from so101_bench_sim_WM training episodes for open-loop policy checks.

For each sampled (episode, frame t): the overhead/wrist frames at t and at t-7 (the oldest tick of an
8-tick history window), the episode's first frames (init cameras), the 8 states and previous commands
for ticks t-7..t, and the recorded 30-action chunk from t. LeRobot units throughout.
  open_loop_dump_frames.py OUT.npz [--episodes N] [--per_episode K]
"""
import argparse, numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO, REV = "5hadytru/so101_bench_sim_WM", "b569ad81ee60ffe3580bc0f1df513bde1bb36e8c"
CAMS = ["observation.images.overhead", "observation.images.wrist", "observation.images.overhead_init", "observation.images.wrist_init"]

def u8(x):  # CHW float [0,1] -> HWC uint8
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--per_episode", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    eps = sorted(rng.choice(3629, a.episodes, replace=False).tolist())
    ds = LeRobotDataset(REPO, revision=REV, episodes=eps, video_backend="pyav")
    rows = []
    for e in eps:
        idx = [i for i in range(len(ds.hf_dataset)) if int(ds.hf_dataset[i]["episode_index"]) == e]
        start, n = idx[0], len(idx)
        for t in sorted(rng.choice(np.arange(8, n - 30), a.per_episode, replace=False).tolist()):
            cur, old, first = ds[start + t], ds[start + t - 7], ds[start]
            states = np.stack([ds.hf_dataset[start + k]["observation.state"] for k in range(t - 7, t + 1)]).astype(np.float32)
            prev_cmds = np.stack([ds.hf_dataset[start + k - 1]["action"] for k in range(t - 7, t + 1)]).astype(np.float32)
            chunk = np.stack([ds.hf_dataset[start + t + k]["action"] for k in range(30)]).astype(np.float32)
            rows.append(dict(episode=e, t=t, task=cur["task"], overhead=u8(cur[CAMS[0]]), wrist=u8(cur[CAMS[1]]),
                             overhead_old=u8(old[CAMS[0]]), wrist_old=u8(old[CAMS[1]]),
                             overhead_init=u8(first[CAMS[2]]), wrist_init=u8(first[CAMS[3]]),
                             states=states, prev_cmds=prev_cmds, chunk=chunk))
            print(e, t, cur["task"], flush=True)
    np.savez_compressed(a.out, **{k: np.array([r[k] for r in rows], dtype=object if k == "task" else None) for k in rows[0]})
    print("wrote", len(rows), "query frames to", a.out)

if __name__ == "__main__":
    main()
