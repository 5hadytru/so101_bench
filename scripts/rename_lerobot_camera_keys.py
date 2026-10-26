#!/usr/bin/env python3
"""Rename LeRobot video feature keys without re-encoding any video data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import pyarrow.parquet as pq


RENAMES = {
    "observation.images.front_init": "observation.images.wrist_init",
    "observation.images.front": "observation.images.wrist",
}


def renamed_key(key: str) -> str:
    # Longest first prevents front from partially rewriting front_init.
    for old, new in RENAMES.items():
        key = key.replace(old, new)
    return key


def rename_mapping_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {renamed_key(str(key)): rename_mapping_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rename_mapping_keys(item) for item in value]
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".camera-rename.tmp")
    temporary.write_text(json.dumps(value, indent=4) + "\n")
    temporary.replace(path)


def rewrite_json(path: Path) -> None:
    write_json_atomic(path, rename_mapping_keys(json.loads(path.read_text())))


def rewrite_jsonl(path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".camera-rename.tmp")
    with path.open() as source, temporary.open("w") as destination:
        for line in source:
            if line.strip():
                destination.write(json.dumps(rename_mapping_keys(json.loads(line)), separators=(",", ":")) + "\n")
    temporary.replace(path)


def rewrite_parquet(path: Path) -> None:
    table = pq.read_table(path)
    names = [renamed_key(name) for name in table.column_names]
    if len(names) != len(set(names)):
        raise ValueError(f"Camera rename would create duplicate columns in {path}")
    temporary = path.with_suffix(path.suffix + ".camera-rename.tmp")
    pq.write_table(table.rename_columns(names), temporary)
    temporary.replace(path)


def video_directories(root: Path, version: str) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    if version == "v3.0":
        parents = [root / "videos"]
    elif version == "v2.1":
        parents = sorted((root / "videos").glob("chunk-*"))
    else:
        raise ValueError(f"Unsupported LeRobot version {version!r} in {root}")
    for parent in parents:
        for old, new in RENAMES.items():
            source, destination = parent / old, parent / new
            if source.exists():
                if destination.exists():
                    raise FileExistsError(destination)
                pairs.append((source, destination))
    return pairs


def migrate(root: Path, backup_dir: Path) -> None:
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    version = str(info["codebase_version"])
    current_keys = set(info["features"])
    if not any(old in current_keys for old in RENAMES):
        raise ValueError(f"No front camera features found in {root}")
    for old, new in RENAMES.items():
        if old in current_keys and new in current_keys:
            raise ValueError(f"Both {old!r} and {new!r} already exist in {root}")

    backup = backup_dir / root.name / "meta"
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copytree(root / "meta", backup)

    directory_pairs = video_directories(root, version)
    expected_old_keys = {old for old in RENAMES if old in current_keys}
    found_old_keys = {source.name for source, _ in directory_pairs}
    if found_old_keys != expected_old_keys:
        raise ValueError(
            f"Video directory mismatch in {root}: features={sorted(expected_old_keys)}, "
            f"directories={sorted(found_old_keys)}"
        )

    for source, destination in directory_pairs:
        source.rename(destination)
    rewrite_json(info_path)
    rewrite_json(root / "meta/stats.json")
    if version == "v3.0":
        for path in sorted((root / "meta/episodes").rglob("*.parquet")):
            rewrite_parquet(path)
    else:
        rewrite_jsonl(root / "meta/episodes_stats.jsonl")
    print(f"Migrated {root} ({version}); metadata backup: {backup.parent}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--backup-dir", type=Path, required=True)
    args = parser.parse_args()
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    for root in args.roots:
        migrate(root, args.backup_dir)


if __name__ == "__main__":
    main()
