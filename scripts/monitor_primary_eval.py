#!/usr/bin/env python3
"""Live success-rate monitor for a primary-eval output dir.

Reads chunks/*/run_*/episode_diagnostics.jsonl (later runs override earlier
ones for the same trial_id), and prints success by task family x OOD split.
Bin tasks with 4 objects also report mean progress = objects_in_bin / 4.
"""
import glob, json, sys
from collections import defaultdict

root = sys.argv[1] if len(sys.argv) > 1 else \
    "outputs/primary_eval_v2_GR00T_N1.7-3B_step16000"
OOD = ["seen", "unseen_seen_class", "unseen_unseen_class"]

eps = {}
for p in sorted(glob.glob(f"{root}/chunks/*/run_*/episode_diagnostics.jsonl")):
    for line in open(p):
        d = json.loads(line)
        m = d["episode_spec"]["metadata"]
        eps[m["trial_id"]] = d  # sorted by run timestamp -> latest wins

rows = defaultdict(list)
for d in eps.values():
    s, m = d["episode_spec"], d["episode_spec"]["metadata"]
    fam = s.get("task_family", "?")
    n = m.get("n_objects", 1)
    key = f"{fam}-{n}obj" if fam == "bin" else fam
    prog = d["objects_in_bin"] / d["bin_object_count"] if fam == "bin" and d.get("bin_object_count") else None
    rows[(key, m["ood_key"])].append((bool(d["success"]), prog))

def cell(v):
    if not v:
        return "-"
    k, n = sum(s for s, _ in v), len(v)
    out = f"{k}/{n} {100*k/n:5.1f}%"
    pr = [p for _, p in v if p is not None]
    if pr and len(v) and "bin-4obj" in cell.key:
        out += f"  prog {100*sum(pr)/len(pr):5.1f}%"
    return out

keys = sorted({k for k, _ in rows})
w = 34
print(f"{'task':<14}" + "".join(f"{o:<{w}}" for o in OOD) + f"{'ALL':<{w}}")
for k in keys + ["ALL"]:
    line = f"{k:<14}"
    for o in OOD + [None]:
        v = [x for (kk, oo), vs in rows.items() if (k == "ALL" or kk == k) and (o is None or oo == o) for x in vs]
        cell.key = k
        line += f"{cell(v):<{w}}"
    print(line)
print(f"\nepisodes scored: {len(eps)} (dedup by trial_id)")
