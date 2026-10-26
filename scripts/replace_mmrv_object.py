#!/usr/bin/env python3
"""Replace one object everywhere in the MMRV set, keeping the hand-edited layouts.

For each row that uses ``--old``, pick a replacement from the same split's owned objects (not
already in that scene; least used across the set first, then closest in size to the old object;
bin_4 scenes must not repeat a companion pair), put it where the old object stood (same root and
yaw), and if that is not clear, search nearby poses (yaw steps, then small shifts). Clear means
the generator's rules: >= the scene's object spacing (1.5 in floor) from every other object,
>= 0.5 in from the bin, inside the bin pose's spawn region, not overlapping the robot. Rewrites the
instruction/metadata (e.g. a between referent), then re-validates the whole set with
create_mmrv_eval.validate. Originals are backed up next to the files.

    ~/env_isaaclab_51/bin/python scripts/replace_mmrv_object.py --old "candy bar"
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import create_mmrv_eval as M  # noqa: E402
from task_generation import cell_key  # noqa: E402

INCH = 0.0254


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    benchmark, layouts = M.load_modules()
    L = layouts
    constants = M.env_cfg_constants()
    regions, bin_poses = constants["VALID_OBJECT_SPAWN_REGIONS"], constants["BIN_RANDOM_POSES"]
    robot_poly = L._xy_polygon(constants["SO101_BOUNDING_BOX"])
    robot_aabb = L._piece_aabb(robot_poly)

    rows = [json.loads(line) for line in M.TASKS.read_text().splitlines()]
    lays = [json.loads(line) for line in M.LAYOUTS.read_text().splitlines()]
    pool = {s: [n for n in names if n != args.old] for s, names in M.REAL_OBJECTS.items()}
    M.REAL_OBJECTS = pool  # the old object leaves the owned set (validate checks against it)

    uses = Counter(n for r in rows for n in r["objects"] if n != args.old)
    size = lambda name: L._boxes_extent(L._object_footprint_boxes(name))  # noqa: E731
    old_size = size(args.old)

    def bin4_pairs(split, skip):
        out = set()
        for i, r in enumerate(rows):
            if i != skip and cell_key(benchmark, r) == (split, "bin_4"):
                out |= {tuple(sorted(p)) for p in combinations(r["objects"], 2)}
        return out

    # where objects may stand: the area the (hand-edited) layouts already use, across the whole set
    # (the generator's spawn-region and robot polygons do not line up with the edited layout frame)
    allc = [L._world_footprint(L._object_local_corners(o["name"]), tuple(o["position"][:2]), o["yaw"]).total
            for lay_ in lays for o in lay_["objects"]]
    area = (min(t[0] for t in allc), min(t[1] for t in allc), max(t[2] for t in allc), max(t[3] for t in allc))

    def clear(lay, slot_index, name, center, yaw, spacing, robot_ok):
        fp = L._world_footprint(L._object_local_corners(name), center, yaw)
        b = lay["bin"]
        bin_fp = L._world_footprint(L._rectangle_local_corners(tuple(b["footprint_half_extents"])),
                                    (b["position"][0], b["position"][1]), b["rpy"][2])
        if not L._footprints_clear(fp, bin_fp, L.MIN_BIN_SURFACE_DISTANCE_M):
            return False
        t = fp.total
        if t[0] < area[0] or t[1] < area[1] or t[2] > area[2] or t[3] > area[3]:
            return False
        if not robot_ok and L._footprint_overlaps_polygon(fp, robot_poly, robot_aabb):
            return False
        for j, o in enumerate(lay["objects"]):
            if j == slot_index:
                continue
            ofp = L._world_footprint(L._object_local_corners(o["name"]), tuple(o["position"][:2]), o["yaw"])
            if not L._footprints_clear(fp, ofp, spacing):
                return False
        return True

    def scene_spacing(lay, slot_index):
        fps = [L._world_footprint(L._object_local_corners(o["name"]), tuple(o["position"][:2]), o["yaw"])
               for j, o in enumerate(lay["objects"]) if j != slot_index]
        gaps = [L._pieces_surface_distance(L._footprint_pieces(a), L._footprint_pieces(b)) for a, b in combinations(fps, 2)]
        return max(L.MIN_OBJECT_SURFACE_DISTANCE_M, min(min(gaps) if gaps else 9.0, 2.5 * INCH) - 1e-4)

    report = []
    for i, row in enumerate(rows):
        if args.old not in row["objects"]:
            continue
        split, family = cell_key(benchmark, row)
        lay = lays[i]
        slot = next(k for k, o in enumerate(lay["objects"]) if o["name"] == args.old)
        start = (lay["objects"][slot]["position"][0], lay["objects"][slot]["position"][1])
        yaw0 = lay["objects"][slot]["yaw"]
        spacing = scene_spacing(lay, slot)
        old_fp = L._world_footprint(L._object_local_corners(args.old), start, yaw0)
        robot_ok = L._footprint_overlaps_polygon(old_fp, robot_poly, robot_aabb)  # no worse than the old object
        cands = [n for n in pool[split] if n not in row["objects"]]
        if family == "bin_4":
            used = bin4_pairs(split, i)
            others = [n for n in row["objects"] if n != args.old]
            cands = [n for n in cands if not any(tuple(sorted((n, o))) in used for o in others)]
        if family not in ("bin_1", "bin_4") and row.get("target") == args.old:
            spatial_targets = {r.get("target") for r in rows if r["ood_key"] == split}
            cands = [n for n in cands if n not in spatial_targets]
        cands.sort(key=lambda n: (uses[n], abs(size(n) - old_size), n))
        chosen = None
        for name in cands:
            poses = [(start, yaw0 + math.radians(d)) for d in (0, 90, -90, 180, 45, -45, 135, -135)]
            for r_m in (0.01, 0.02, 0.03, 0.04):
                for a in range(0, 360, 30):
                    c = (start[0] + r_m * math.cos(math.radians(a)), start[1] + r_m * math.sin(math.radians(a)))
                    poses += [(c, yaw0 + math.radians(d)) for d in (0, 90, 180, -90)]
            for c, y in poses:
                if clear(lay, slot, name, c, y, spacing, robot_ok):
                    chosen = (name, c, y)
                    break
            if chosen:
                break
        if chosen is None:
            raise SystemExit(f"row {i}: no replacement fits ({cands})")
        name, c, y = chosen
        moved = math.dist(c, start)
        # task row: swap the name, rebuild the canonical instruction/metadata
        new = dict(row)
        new["objects"] = [name if n == args.old else n for n in row["objects"]]
        for field in ("target", "anchor"):
            if new.get(field) == args.old:
                new[field] = name
        if "referents" in new:
            new["referents"] = [name if n == args.old else n for n in new["referents"]]
        if family in M.SPATIAL:
            # target/referent wording (class vs colour+class) is the generator's own rule
            new["instruction"] = M.instruction_for_row(new, family, set(benchmark.COLORS))
        rows[i] = new
        entry = dict(lay["objects"][slot])
        boxes = L._object_footprint_boxes(name)
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        entry.update(name=name, position=[c[0], c[1], entry["position"][2]], yaw=y, rpy=[0.0, 0.0, y],
                     footprint_boxes=[list(b) for b in boxes], footprint_half_extents=[0.5 * (x1 - x0), 0.5 * (y1 - y0)],
                     footprint_center_offset=[0.5 * (x0 + x1), 0.5 * (y0 + y1)])
        lay["objects"][slot] = entry
        lay["instruction"] = new["instruction"]
        lay.setdefault("edits", []).append({"replaced": args.old, "with": name, "moved_m": round(moved, 4),
                                            "yaw_change_deg": round(math.degrees(y - yaw0), 1),
                                            "at": datetime.now().astimezone().isoformat(timespec="seconds")})
        uses[name] += 1
        report.append((i, family, row["instruction"], "->", new["instruction"], name, f"moved {moved * 100:.1f} cm",
                       f"yaw {math.degrees(y - yaw0):+.0f} deg"))
    for line in report:
        print(*line)
    summary = M.validate(rows, lays, benchmark)
    print(json.dumps(summary["manipulations_per_object"], indent=1))
    if args.dry_run:
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for p in (M.TASKS, M.LAYOUTS):
        shutil.copy(p, p.with_name(p.name + f".before_{args.old.replace(' ', '_')}_{stamp}"))
    M.TASKS.write_text(M.encode(rows))
    M.LAYOUTS.write_text(M.encode(lays))
    print("written", M.TASKS, M.LAYOUTS)


if __name__ == "__main__":
    main()
