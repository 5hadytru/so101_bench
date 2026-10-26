#!/usr/bin/env python3
"""Read every regular dataset file and report storage read failures.

Each read runs in its own process so a stuck FUSE read cannot block the scan.
The result file contains failures only; a completed scan prints the total checked.
"""

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(path for path in args.dataset.rglob("*") if path.is_file())
    random.Random(1701).shuffle(paths)
    if args.max_files:
        paths = paths[: args.max_files]
    args.results.parent.mkdir(parents=True, exist_ok=True)
    active = {}
    next_index = 0
    completed = 0
    failures = 0
    last_report = time.monotonic()
    with args.results.open("w") as results:
        while next_index < len(paths) or active:
            while next_index < len(paths) and len(active) < args.workers:
                path = paths[next_index]
                next_index += 1
                process = subprocess.Popen(
                    ["dd", f"if={path}", "of=/dev/null", "bs=1048576", "status=none"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                active[process.pid] = (process, path, time.monotonic())
            now = time.monotonic()
            for pid, (process, path, started) in list(active.items()):
                returncode = process.poll()
                timed_out = returncode is None and now - started > args.timeout
                if returncode is None and not timed_out:
                    continue
                if timed_out:
                    os.killpg(pid, 9)
                else:
                    stderr = process.communicate()[1].decode(errors="replace").strip()
                completed += 1
                if timed_out or returncode:
                    failures += 1
                    record = {
                        "path": str(path.relative_to(args.dataset)),
                        "error": "timeout" if timed_out else stderr,
                        "seconds": round(now - started, 2),
                    }
                    results.write(json.dumps(record) + "\n")
                    results.flush()
                    print("FAILED", json.dumps(record), flush=True)
                del active[pid]
            if now - last_report >= 15:
                print(f"checked={completed}/{len(paths)} failures={failures} active={len(active)}", flush=True)
                last_report = now
            time.sleep(0.1)
    print(f"DONE checked={completed} failures={failures}", flush=True)


if __name__ == "__main__":
    main()
