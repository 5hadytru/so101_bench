#!/usr/bin/env python3
"""Open-loop check of a running policy server through the exact eval-time client path.

Feeds training-episode query frames (scripts/open_loop_dump_frames.py) to a server via
MolmoAct2RemotePolicy (same joint mapper, camera order, init frames and history payload as
molmoact2_eval.py), maps the returned chunk back to LeRobot units, and compares it with the
recorded demo actions. On training frames a correct pipeline should beat "hold the current state"
by a wide margin; a large error points at plumbing (units, normalization, cameras, history), not
at closed-loop behavior. Also reports the joint-unit round trip of the demo actions through the
client mapper (should be ~0).

  open_loop_server_check.py FRAMES.npz --joint_frame molmoact2|lerobot [--init] [--history_ticks 8] --out OUT.json
Needs no Isaac Sim: the so101_bench package __init__ (which imports Isaac Lab) is bypassed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "source" / "so101_bench" / "so101_bench"
package = types.ModuleType("so101_bench")
package.__path__ = [str(SRC)]
sys.modules["so101_bench"] = package
molmoact2 = importlib.import_module("so101_bench.utils.molmoact2")
groot = importlib.import_module("so101_bench.utils.groot")

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames")
    ap.add_argument("--joint_frame", required=True)
    ap.add_argument("--init", action="store_true", help="send overhead_init/wrist_init (sim_WM checkpoints)")
    ap.add_argument("--history_ticks", type=int, default=1)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = np.load(a.frames, allow_pickle=True)
    lr = groot.SO101JointMapper(device="cpu")
    cams = {"overhead": {}, "wrist": {}}
    policy = molmoact2.MolmoAct2RemotePolicy(
        device="cpu", cameras=cams, port=a.port, action_horizon=30, joint_frame=a.joint_frame,
        use_overhead_init=a.init, use_wrist_init=a.init, history_ticks=a.history_ticks)
    policy.connect()
    to_sim = lambda x: lr.lerobot_positions_to_sim_radians(torch.as_tensor(x, dtype=torch.float32))
    to_lr = lambda s: lr.sim_radians_to_lerobot_positions(s).numpy()

    errs, holds, trips, grip = [], [], [], []
    for i in range(len(d["chunk"])):
        gt, states = d["chunk"][i], d["states"][i]
        policy.set_language_instruction(str(d["task"][i]))
        if a.init:
            policy.set_episode_initial_observation({"rgb_overhead": d["overhead_init"][i], "rgb_wrist": d["wrist_init"][i]})
        else:
            policy._clear_episode_state()
        current = {"rgb_overhead": d["overhead"][i], "rgb_wrist": d["wrist"][i]}
        if a.history_ticks > 1:
            old = {"rgb_overhead": d["overhead_old"][i], "rgb_wrist": d["wrist_old"][i]}
            for k in range(8 - a.history_ticks, 8):
                policy._last_command = policy.mapper.sim_radians_to_model_positions(to_sim(d["prev_cmds"][i][k])).numpy()
                policy._record_tick(to_sim(states[k]), old if k == 8 - a.history_ticks else current)
        chunk = policy._query_server(to_sim(states[-1]), current)
        pred = np.stack([to_lr(policy.mapper.model_positions_to_sim_radians(torch.as_tensor(r))) for r in chunk])
        h = min(len(pred), len(gt))
        errs.append(np.abs(pred[:h] - gt[:h]).mean(0))
        holds.append(np.abs(states[-1][None] - gt[:h]).mean(0))
        trip = np.stack([to_lr(policy.mapper.model_positions_to_sim_radians(policy.mapper.sim_radians_to_model_positions(to_sim(r)))) for r in gt])
        trips.append(np.abs(trip - gt).max(0))
        g_change = gt[h - 1, 5] - states[-1][5]
        if abs(g_change) > 15:
            grip.append(bool(np.sign(pred[h - 1, 5] - states[-1][5]) == np.sign(g_change)))
    e, hb, tr = np.mean(errs, 0), np.mean(holds, 0), np.max(trips, 0)
    out = {"label": a.label, "n": len(errs), "chunk_len": int(h),
           "mae_per_joint": dict(zip(JOINTS, e.round(2).tolist())), "hold_state_mae": dict(zip(JOINTS, hb.round(2).tolist())),
           "arm_mae": round(float(e[:5].mean()), 2), "arm_hold_mae": round(float(hb[:5].mean()), 2),
           "gripper_mae": round(float(e[5]), 2), "gripper_hold_mae": round(float(hb[5]), 2),
           "gripper_direction_agreement": f"{sum(grip)}/{len(grip)}",
           "roundtrip_max_abs_err": dict(zip(JOINTS, tr.round(3).tolist()))}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
