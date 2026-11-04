"""Protocol stand-in for sim_server.py (no Isaac): fixed episodes, random frames, success after 20 steps."""

import argparse

import msgpack
import numpy as np
import zmq

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--init_frames", action="store_true")
args, _ = parser.parse_known_args()

H, W = 480, 640
EPISODES = [
    {"trial_id": 0, "instruction": "Place each object in the plastic bin", "split": "seen", "family": "bin_1",
     "objects": ["black tape"], "appearance_profile": "canonical", "max_steps": 40},
    {"trial_id": 1, "instruction": "Move the pliers left", "split": "seen", "family": "move",
     "objects": ["blue pliers"], "appearance_profile": "varied", "max_steps": 40},
]
rng = np.random.default_rng(0)


def observation():
    images = {name: rng.integers(0, 255, (H, W, 3), dtype=np.uint8).tobytes() for name in ("overhead", "wrist")}
    if args.init_frames:
        images |= {f"{k}_init": v for k, v in list(images.items())}
    return {"state": [0.0] * 6, "images": images}


socket = zmq.Context.instance().socket(zmq.REP)
socket.bind(f"tcp://127.0.0.1:{args.port}")
steps, trial = 0, None
while True:
    request = msgpack.unpackb(socket.recv(), raw=False)
    cmd = request["cmd"]
    if cmd == "info":
        reply = {"episodes": EPISODES, "fps": 30, "cameras": {"overhead": [H, W], "wrist": [H, W]},
                 "init_frames": args.init_frames}
    elif cmd == "reset":
        steps, trial = 0, request["trial_id"]
        reply = observation()
    elif cmd == "step":
        assert len(request["action"]) == 6
        steps += 1
        done = steps >= 20
        reply = {**observation(), "reward": float(done and trial == 0), "terminated": done, "truncated": False,
                 "is_success": done and trial == 0, "end_reason": "success" if done and trial == 0 else ""}
    elif cmd == "close":
        socket.send(msgpack.packb({}, use_bin_type=True))
        break
    socket.send(msgpack.packb(reply, use_bin_type=True))
