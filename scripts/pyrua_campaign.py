#!/usr/bin/env python3
"""Run PyRUA-Lean over a whole episode file, unattended: shards, relaunches, spend cap.

Rows run in a stratified order (task x OOD split interleaved in proportion), so any prefix of the
campaign is a balanced sample and two models run with the same order can be compared at any time.
Each round splits the rows that are not done yet over ``--workers`` ``pyrua_eval.py`` processes
(output ``<out>/run_r<K>_w<J>``). A row is done once some ``run*/results.jsonl`` holds a result for
it that is not an infrastructure failure; infra-failed rows go back into the next round.

The campaign stops when every row is done, when the priced spend reaches ``--spend_cap``, or when a
worker log shows the provider's budget is exhausted (OpenAI project spend limit, Gemini quota):
then it writes ``<out>/STOPPED`` with the reason and exits, and a later rerun resumes.

    python3 scripts/pyrua_campaign.py --out outputs/pyrua_primary_eval_v2_sol --workers 2 \
        --spend_cap 70 --prices sol -- --runtime codex --model gpt-6.1-sol --effort high --codex_bin ...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from primary_eval_exclusions import EXCLUDED_TRIAL_IDS, excluded  # noqa: E402
ISAAC_PY = Path(os.environ.get("ISAAC_PY", Path.home() / "env_isaaclab_51" / "bin" / "python"))
#: $ per million tokens: uncached input, cached input, output (output includes reasoning).
PRICES = {"sol": (2.00, 0.10, 10.00), "argon": (2.00, 0.10, 10.00)}
#: OpenAI's hard budget errors only: Gemini words its per-minute rate limits as "quota" too, and the
#: OpenAI-compatible loop waits those out.
BUDGET_MARKERS = ("project_spend_limit_exceeded", "insufficient_quota", "billing_hard_limit")
INFRA_STATUSES = ("boot_error", "error", "planner_error")


def log(message: str) -> None:
    print(f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z} {message}", flush=True)


def task_type(row: dict) -> str:
    i = row["instruction"]
    if "bin" in i:
        return f"bin_{row['n_objects']}"
    if "between" in i:
        return "between"
    if "next to" in i:
        return "next_to"
    return "move"


def stratified_order(rows: list[dict]) -> list[int]:
    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        cells[(task_type(row), row["ood_key"])].append(index)
    keyed = []
    for cell, members in sorted(cells.items()):
        for rank, index in enumerate(members):
            keyed.append(((rank + 0.5) / len(members), cell, index))
    # rows removed from primary_eval_v2 are dropped after ordering, so every other row keeps its place
    return [index for _, _, index in sorted(keyed) if index not in EXCLUDED_TRIAL_IDS]


def shard_rows(rows: list[dict], j: int, n: int) -> set[int]:
    """Rows of shard j of n: dealt round-robin inside each task x split cell, so shards stay balanced."""

    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        cells[(task_type(row), row["ood_key"])].append(index)
    offset, mine = 0, set()
    for _, members in sorted(cells.items()):
        mine.update(m for k, m in enumerate(members) if (k + offset) % n == j)
        offset += len(members)
    return mine - EXCLUDED_TRIAL_IDS  # dealt as before; removed rows dropped


def infra_failure(result: dict) -> bool:
    if result.get("success") is None or result.get("termination_reason") == "runtime_error":
        return True
    pyrua = result.get("pyrua") or {}
    # an episode that ran out of wall time while waiting out provider rate limits was not the model's doing
    return pyrua.get("status") in INFRA_STATUSES or (pyrua.get("status") == "timeout" and (pyrua.get("http_429") or 0) > 0)


def results(out: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    for path in sorted(out.glob("run*/results.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if not infra_failure(row) and not excluded(row["episode"]):
                    done[row["episode"]] = row
    return done


def episode_cost(row: dict, prices: tuple[float, float, float]) -> float:
    t = (row.get("pyrua") or {}).get("tokens") or {}
    inp, cached, out = (t.get(k) or 0 for k in ("input_tokens", "cached_input_tokens", "output_tokens"))
    u, c, o = prices
    return (max(0, inp - cached) * u + cached * c + out * o) / 1e6


def spent(out: Path, prices) -> float:
    """Priced spend of every recorded episode, infra failures included (they were billed too)."""

    total = 0.0
    for path in out.glob("run*/results.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                total += episode_cost(json.loads(line), prices)
    return total


def budget_exhausted(run_dirs: list[Path]) -> str | None:
    for run_dir in run_dirs:
        for path in list(run_dir.glob("ep*/pyrua.log")) + list(run_dir.glob("ep*/pyrua/*.jsonl")):
            try:
                text = path.read_text(errors="replace")[-200_000:]
            except OSError:
                continue
            for marker in BUDGET_MARKERS:
                if marker in text:
                    return f"{marker} in {path.relative_to(run_dir.parent)}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes_jsonl", type=Path, default=ROOT / "tasks" / "primary_eval_v2.jsonl")
    parser.add_argument("--layouts_jsonl", type=Path, required=True)
    parser.add_argument("--rows", type=str, default=None, help="Comma-separated row indices (default: all).")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard", type=str, default="0/1", help="j/N: this machine runs every N-th row of the order, from j.")
    parser.add_argument("--prices", choices=sorted(PRICES), required=True)
    parser.add_argument("--spend_cap", type=float, required=True, help="Stop launching once priced spend reaches this $.")
    parser.add_argument("--max_rounds", type=int, default=8)
    parser.add_argument("--fallback_codex_home", type=Path, default=None,
                        help="Codex home with another API key: when the current key's budget is exhausted, switch to it once.")
    parser.add_argument("eval_args", nargs=argparse.REMAINDER, help="After --: extra pyrua_eval.py arguments.")
    args = parser.parse_args()
    extra = [a for a in args.eval_args if a != "--"]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tasks = out / "primary_eval_v2.jsonl"
    layouts = out / "primary_eval_v2_layouts.jsonl"
    if not tasks.exists():
        shutil.copy(args.episodes_jsonl, tasks)
        shutil.copy(args.layouts_jsonl, layouts)
    rows = [json.loads(line) for line in tasks.read_text().splitlines() if line.strip()]
    wanted = set(int(v) for v in args.rows.split(",")) if args.rows else set(range(len(rows)))
    j, n = (int(v) for v in args.shard.split("/"))
    order = [i for i in stratified_order(rows) if i in wanted and i in shard_rows(rows, j, n)]
    (out / "order.json").write_text(json.dumps(order))
    prices = PRICES[args.prices]
    (out / "STOPPED").unlink(missing_ok=True)
    for _ in range(args.max_rounds):
        done = results(out)
        remaining = [i for i in order if f"ep{i:03d}" not in done]
        cost = spent(out, prices)
        log(f"{len(done)}/{len(order)} done, priced spend ${cost:.2f} (cap ${args.spend_cap:.2f})")
        if not remaining:
            (out / "ALL_DONE").write_text(datetime.now().astimezone().isoformat(timespec="seconds") + "\n")
            log("all rows done")
            return
        if cost >= args.spend_cap:
            (out / "STOPPED").write_text(f"spend cap: ${cost:.2f} >= ${args.spend_cap:.2f}\n")
            log(f"STOPPED at the spend cap with {len(remaining)} rows left")
            return
        index = 1 + len({p.name.split("_w")[0] for p in out.glob("run_r*")})
        shards = [remaining[j :: args.workers] for j in range(args.workers)]
        procs, run_dirs = [], []
        for j, shard in enumerate(s for s in shards if s):
            run_dir = out / f"run_r{index}_w{j}"
            run_dirs.append(run_dir)
            cmd = [str(ISAAC_PY), str(ROOT / "scripts" / "pyrua_eval.py"), "--headless", "--seed", "0",
                   "--episodes_jsonl", str(tasks), "--episode_layouts_jsonl", str(layouts),
                   "--episode_indices", ",".join(map(str, shard)), "--output_dir", str(run_dir), *extra]
            logf = open(out / f"{run_dir.name}.eval.log", "w")
            procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True))
            log(f"round {index} worker {j}: {len(shard)} rows -> {run_dir.name}")
            time.sleep(90)  # stagger Isaac start-up
        stop_reason = None
        while any(p.poll() is None for p in procs):
            time.sleep(60)
            reason = budget_exhausted(run_dirs)
            cost = spent(out, prices)
            if reason or cost >= args.spend_cap:
                stop_reason = reason or f"spend cap: ${cost:.2f} >= ${args.spend_cap:.2f}"
                for p in procs:
                    if p.poll() is None:
                        os.killpg(p.pid, 15)
                time.sleep(30)
                for p in procs:
                    if p.poll() is None:
                        os.killpg(p.pid, 9)
        if stop_reason and args.fallback_codex_home and not stop_reason.startswith("spend cap") \
                and os.environ.get("CODEX_HOME") != str(args.fallback_codex_home):
            # the first key's budget is gone: rerun the interrupted rows on the fallback key
            old = os.environ.get("CODEX_HOME")
            os.environ["CODEX_HOME"] = str(args.fallback_codex_home)
            (out / "SWITCHED_KEY").write_text(f"{datetime.now().astimezone().isoformat(timespec='seconds')} "
                                              f"{old} -> {args.fallback_codex_home}: {stop_reason}\n")
            log(f"budget exhausted ({stop_reason}); switched CODEX_HOME to {args.fallback_codex_home}")
            continue
        if stop_reason:
            (out / "STOPPED").write_text(stop_reason + "\n")
            log(f"STOPPED: {stop_reason}")
            return
        log(f"round {index} finished: exit codes {[p.returncode for p in procs]}")
    log("gave up after max_rounds")


if __name__ == "__main__":
    sys.exit(main())
