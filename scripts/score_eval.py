#!/usr/bin/env python3
"""Score an SO-101 Bench evaluation run the way the leaderboard does.

Reads one or more evaluator logs (the ``Episode i/n: success=...`` lines written by
scripts/molmoact2_eval.py, scripts/groot_eval.py and scripts/pyrua_eval.py) together with the
task file the run used, and reports:

* success per task-family x object-split cell (15 cells);
* the headline score: the mean of the cell success rates, with a 95% bootstrap CI that
  resamples episodes within each cell;
* four-object bin progress (objects binned / objects) when the log reports it.

    python scripts/score_eval.py --tasks tasks/val_set.jsonl --log outputs/val/eval.log
    python scripts/score_eval.py --tasks tasks/val_set.jsonl --log a.log --log b.log --json score.json

Episode i of a log is row i of the task file (the order the evaluator ran them in).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re

from task_generation import SPLITS, cell_key, load_modules

LINE = re.compile(r"Episode (\d+)/(\d+): success=(True|False), reason=([^,]+), length=([\d.]+)s"
                  r"(?:.*?objects_in_bin=(\d+)/(\d+))?")
FAMILIES = ("bin_1", "bin_4", "next_to", "between", "move")


def cell_mean(cells: dict[tuple[str, str], list[bool]]) -> float:
    return sum(sum(v) / len(v) for v in cells.values()) / len(cells)


def bootstrap(cells: dict[tuple[str, str], list[bool]], reps: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(cell_mean({c: [rng.choice(v) for _ in v] for c, v in cells.items()}) for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", required=True, type=Path, help="Task JSONL the run evaluated (in run order).")
    parser.add_argument("--log", required=True, type=Path, action="append",
                        help="Evaluator log; repeat for chunked runs (each log indexes the same task file).")
    parser.add_argument("--json", type=Path, help="Optional path for a JSON copy of the scores.")
    args = parser.parse_args()

    benchmark, _ = load_modules()
    rows = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    episodes: dict[int, dict] = {}
    for log in args.log:
        for m in LINE.finditer(log.read_text(errors="replace")):
            k = int(m.group(1)) - 1
            if k >= len(rows):
                continue
            split, family = cell_key(benchmark, rows[k])
            episodes[k] = dict(split=split, task=family, success=m.group(3) == "True",
                               progress=int(m.group(6)) / int(m.group(7)) if m.group(6) else None)
    if not episodes:
        raise SystemExit("No 'Episode i/n: success=...' lines found in the log(s).")

    cells: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for e in episodes.values():
        cells[e["split"], e["task"]].append(e["success"])
    score = cell_mean(cells)
    low, high = bootstrap(cells)
    progress = [e["progress"] for e in episodes.values() if e["task"] == "bin_4" and e["progress"] is not None]

    print(f"Episodes scored: {len(episodes)}/{len(rows)}   cells covered: {len(cells)}/15")
    print(f"Score (mean of cell success): {100 * score:.1f}  [95% CI {100 * low:.1f}, {100 * high:.1f}]")
    if progress:
        print(f"Four-object bin progress: {100 * sum(progress) / len(progress):.1f}% of objects binned")
    print()
    print(f"{'':22s}" + "".join(f"{f:>10s}" for f in FAMILIES))
    for split in SPLITS:
        cols = []
        for family in FAMILIES:
            v = cells.get((split, family))
            cols.append(f"{100 * sum(v) / len(v):5.0f}% {sum(v)}/{len(v)}".rjust(10) if v else f"{'--':>10s}")
        print(f"{split:22s}" + "".join(cols))
    if len(cells) < 15:
        print("\nWarning: not every cell has episodes yet; the score is over the covered cells only.")
    if args.json:
        args.json.write_text(json.dumps(dict(
            episodes=len(episodes), total=len(rows), score=score, ci95=[low, high],
            bin4_progress=sum(progress) / len(progress) if progress else None,
            cells={f"{s}/{f}": dict(success=sum(v), n=len(v)) for (s, f), v in sorted(cells.items())},
        ), indent=2) + "\n")


if __name__ == "__main__":
    main()
