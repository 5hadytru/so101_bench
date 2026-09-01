#!/usr/bin/env python3
"""Rename ``Object_<N>`` prims inside object USDs so they cannot shadow scene slots.

IsaacLab resolves each benchmark object slot with the regex
``/World/envs/env_.*/Object_<N>``.  ``.*`` matches across ``/``, so a prim named
``Object_12`` *inside* another asset (Sketchfab GLTF imports name their nodes this
way) matches that pattern too.  Depth-first traversal can therefore hand IsaacLab a
mesh buried inside an earlier slot instead of the real object, which then fails with
"Failed to find a rigid body when resolving '/World/envs/env_.*/Object_12'".

Run with the USD-only environment:

    /home/truman/env_isaaclab/bin/python scripts/rename_object_prims_in_usd_assets.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil

from pxr import Sdf, Usd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJECTS_DIR = REPO_ROOT / "source" / "so101_bench" / "so101_bench" / "assets" / "usd" / "objects"
OBJECT_PRIM_PATTERN = re.compile(r"^Object_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-dir", type=Path, default=DEFAULT_OBJECTS_DIR)
    parser.add_argument("--prefix", default="Node_", help="Replacement prefix for renamed prims.")
    parser.add_argument("--backup-suffix", default=".before_object_prim_rename")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def offending_prim_paths(stage: Usd.Stage) -> list[str]:
    return [
        prim.GetPath().pathString
        for prim in stage.Traverse()
        if OBJECT_PRIM_PATTERN.match(prim.GetName())
    ]


def main() -> None:
    args = parse_args()
    changed = 0
    for usd_path in sorted(args.objects_dir.glob("*.usdc")):
        stage = Usd.Stage.Open(str(usd_path))
        paths = offending_prim_paths(stage)
        if not paths:
            continue

        # Deepest first, so renaming a parent never invalidates a queued child path.
        paths.sort(key=lambda path: path.count("/"), reverse=True)
        renames: list[tuple[str, str]] = []
        edit = Sdf.BatchNamespaceEdit()
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            new_name = f"{args.prefix}{OBJECT_PRIM_PATTERN.match(prim.GetName()).group(1)}"
            parent = prim.GetParent()
            if parent.GetChild(new_name).IsValid():
                raise RuntimeError(f"{usd_path}: cannot rename {path}; sibling {new_name!r} already exists.")
            edit.Add(Sdf.NamespaceEdit.Rename(path, new_name))
            renames.append((path, new_name))

        detail = ", ".join(f"{path} -> {name}" for path, name in renames)
        print(f"{usd_path.name}: {detail}")
        if args.dry_run:
            continue

        backup = usd_path.with_name(usd_path.name + args.backup_suffix)
        if not backup.exists():
            shutil.copy2(usd_path, backup)
        layer = stage.GetRootLayer()
        if not layer.Apply(edit):
            raise RuntimeError(f"{usd_path}: namespace edit failed.")
        layer.Save()
        changed += 1

    if not changed and not args.dry_run:
        print("No object USDs contained Object_<N> prims.")
    elif not args.dry_run:
        print(f"Renamed prims in {changed} asset(s).")


if __name__ == "__main__":
    main()
