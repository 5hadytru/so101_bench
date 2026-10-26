#!/usr/bin/env python3
"""Per-step wall time from tqdm lines 'N/51000 [H:MM:SS<...' in each bench train.log."""
import re, sys, statistics
pat = re.compile(r"(\d+)/51000 \[(?:(\d+):)?(\d+):(\d+)<")
for log in sys.argv[1:]:
    text = open(log, errors="replace").read()
    pts = {}
    for m in pat.finditer(text):
        step = int(m.group(1)); h = int(m.group(2) or 0)
        pts[step] = h * 3600 + int(m.group(3)) * 60 + int(m.group(4))
    steps = sorted(pts)
    if len(steps) < 3:
        print(f"{log}: {len(steps)} steps logged"); continue
    first, last = steps[0], steps[-1]
    t_end = pts[last]
    steady = [s for s in steps if pts[s] >= t_end - 600]  # final 10 minutes
    dts = [pts[b] - pts[a] for a, b in zip(steady, steady[1:]) if b == a + 1]
    rate = (steady[-1] - steady[0]) / max(1, pts[steady[-1]] - pts[steady[0]])
    waits = text.count("Wait for shard")
    errs = len(re.findall(r"Traceback|Error|SIGBUS|No such device", text))
    print(f"{log}: steps {first}->{last} ({last-first} done) | final-10min: {len(steady)} steps, "
          f"{1/rate if rate else float('nan'):.2f} s/step mean, median dt {statistics.median(dts) if dts else float('nan'):.1f}s "
          f"| shard waits={waits} errors={errs}")
