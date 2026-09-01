"""Move object material libraries below their USD default prim.

Blender can export ``/_materials`` beside ``/root`` while setting ``/root`` as
the default prim. When that asset is referenced, bindings from meshes below
``/root`` to materials below ``/_materials`` fall outside the reference scope
and are ignored. This utility repairs those files with USD namespace edits,
which also retarget dependent material bindings and shader connections.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Usd, UsdShade


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJECTS_DIR = (
    REPO_ROOT / "source" / "so101_bench" / "so101_bench" / "assets" / "usd" / "objects"
)
USD_SUFFIXES = {".usd", ".usda", ".usdc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-dir", type=Path, default=DEFAULT_OBJECTS_DIR)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Save repairs. Without this flag, only report files needing repair.",
    )
    return parser.parse_args()


def out_of_scope_bindings(stage: Usd.Stage) -> list[tuple[str, list[str]]]:
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        return []

    default_path = default_prim.GetPath()
    problems: list[tuple[str, list[str]]] = []
    for prim in stage.Traverse():
        relationship = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel()
        if not relationship:
            continue
        targets = relationship.GetTargets()
        outside_targets = [str(target) for target in targets if not target.HasPrefix(default_path)]
        if outside_targets:
            problems.append((str(prim.GetPath()), outside_targets))
    return problems


def repair_file(path: Path, apply: bool) -> bool:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {path}")

    problems = out_of_scope_bindings(stage)
    if not problems:
        return False

    default_prim = stage.GetDefaultPrim()
    source = stage.GetPrimAtPath("/_materials")
    destination = default_prim.GetPath().AppendChild("_materials")
    if not source.IsValid():
        raise RuntimeError(f"{path.name}: out-of-scope bindings exist but /_materials does not")
    if stage.GetPrimAtPath(destination).IsValid():
        raise RuntimeError(f"{path.name}: destination already exists: {destination}")

    print(f"{path.name}: {len(problems)} binding(s), /_materials -> {destination}")
    if not apply:
        return True

    editor = Usd.NamespaceEditor(stage)
    if not editor.MovePrimAtPath(source.GetPath(), destination):
        raise RuntimeError(f"{path.name}: could not schedule namespace edit")
    if not editor.CanApplyEdits() or not editor.ApplyEdits():
        raise RuntimeError(f"{path.name}: could not apply namespace edit")
    stage.GetRootLayer().Save()

    reopened = Usd.Stage.Open(str(path))
    remaining = out_of_scope_bindings(reopened)
    if remaining:
        raise RuntimeError(f"{path.name}: repair left {len(remaining)} invalid binding(s)")
    return True


def main() -> int:
    args = parse_args()
    paths = sorted(
        path
        for path in args.objects_dir.resolve().iterdir()
        if path.is_file() and path.suffix.lower() in USD_SUFFIXES
    )
    changed = sum(repair_file(path, args.apply) for path in paths)
    action = "Repaired" if args.apply else "Would repair"
    print(f"{action} {changed} of {len(paths)} object USD file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
