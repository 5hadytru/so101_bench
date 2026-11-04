"""Score a lerobot-eval run the way the SO-101 Bench leaderboard does.

lerobot-eval's overall ``pc_success`` averages episodes; the SO-101 Bench score is the mean of the
task-family x object-split cell success rates (its suites), with a 95% bootstrap CI that resamples
episodes within each cell (as scripts/score_eval.py does).

    python -m lerobot_env_so101_bench.score outputs/eval/<run>/eval_info.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SPLITS = ("seen", "unseen_seen_class", "unseen_unseen_class")
FAMILIES = ("bin_1", "bin_4", "next_to", "between", "move")


def cell_mean(cells: dict[str, list[bool]]) -> float:
    return sum(sum(v) / len(v) for v in cells.values()) / len(cells)


def bootstrap(cells: dict[str, list[bool]], reps: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(cell_mean({c: [rng.choice(v) for _ in v] for c, v in cells.items()}) for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def cells_from_eval_info(info: dict) -> dict[str, list[bool]]:
    cells: dict[str, list[bool]] = defaultdict(list)
    for task in info["per_task"]:
        cells[task["task_group"]].extend(bool(s) for s in task["metrics"]["successes"])
    return dict(cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_info", type=Path, help="eval_info.json written by lerobot-eval.")
    parser.add_argument("--json", type=Path, help="Optional path for a JSON copy of the scores.")
    args = parser.parse_args()

    cells = cells_from_eval_info(json.loads(args.eval_info.read_text()))
    score = cell_mean(cells)
    low, high = bootstrap(cells)
    episodes = sum(len(v) for v in cells.values())
    print(f"Episodes scored: {episodes}   cells covered: {len(cells)}/15")
    print(f"Score (mean of cell success): {100 * score:.1f}  [95% CI {100 * low:.1f}, {100 * high:.1f}]")
    print()
    print(f"{'':22s}" + "".join(f"{f:>10s}" for f in FAMILIES))
    for split in SPLITS:
        cols = []
        for family in FAMILIES:
            v = cells.get(f"{split}/{family}")
            cols.append(f"{100 * sum(v) / len(v):5.0f}% {sum(v)}/{len(v)}".rjust(10) if v else f"{'--':>10s}")
        print(f"{split:22s}" + "".join(cols))
    if len(cells) < 15:
        print("\nWarning: not every cell has episodes; the score is over the covered cells only.")
    if args.json:
        args.json.write_text(json.dumps(dict(
            episodes=episodes, score=score, ci95=[low, high],
            cells={c: dict(success=sum(v), n=len(v)) for c, v in sorted(cells.items())},
        ), indent=2) + "\n")


if __name__ == "__main__":
    main()
