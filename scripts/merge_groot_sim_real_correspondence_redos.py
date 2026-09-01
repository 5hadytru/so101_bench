#!/usr/bin/env python3
"""Safely replace original sim-real annotations with scored redo annotations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL = ROOT / "logs/groot_sim_real_correspondence/real_sim_correspondence.jsonl"
DEFAULT_REDOS = ROOT / "logs/groot_sim_real_correspondence/real_sim_correspondence_redos.jsonl"
DEFAULT_MANIFEST = (
    ROOT
    / "outputs/groot_sim_real_correspondence/checkpoint-20000/redos/run_20260811T015034Z/redo_manifest.jsonl"
)
IDENTITY_FIELDS = (
    "trial_id",
    "instruction",
    "objects",
    "ood_key",
    "n_objects",
    "target",
    "referents",
    "direction",
    "clutter",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--redos", type=Path, default=DEFAULT_REDOS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            rows.append(row)
    return rows


def normalized_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in IDENTITY_FIELDS}


def task_success(row: dict[str, Any]) -> bool:
    objects = row.get("objects") or []
    if objects and isinstance(objects[0], dict):
        return all(obj.get("success") is True for obj in objects)
    return row.get("success") is True


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    original_path = args.original.resolve()
    redos_path = args.redos.resolve()
    manifest_path = args.manifest.resolve()
    original = load_jsonl(original_path)
    redos = load_jsonl(redos_path)
    manifest = load_jsonl(manifest_path)
    if len(redos) != len(manifest):
        raise ValueError(f"Redo/manifest count mismatch: {len(redos)} != {len(manifest)}")
    if not original or not redos:
        raise ValueError("Original and redo results must both be non-empty")

    replacements: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    merged = list(original)
    for redo_index, (redo, mapping) in enumerate(zip(redos, manifest, strict=True)):
        if mapping.get("redo_episode_index") != redo_index or mapping.get("redo_episode") != redo_index + 1:
            raise ValueError(f"Non-contiguous redo manifest at redo index {redo_index}")
        original_index = int(mapping.get("original_episode_index", -1))
        if original_index < 0 or original_index >= len(original):
            raise ValueError(f"Invalid original index {original_index} at redo index {redo_index}")
        if original_index in seen_indices:
            raise ValueError(f"Duplicate original index in manifest: {original_index}")
        seen_indices.add(original_index)
        prior = original[original_index]
        if mapping.get("original_episode") != original_index + 1:
            raise ValueError(f"Original episode mismatch at redo index {redo_index}")
        if mapping.get("trial_id") != prior.get("trial_id") or mapping.get("trial_id") != redo.get("trial_id"):
            raise ValueError(f"Trial ID mismatch at redo index {redo_index}")
        if normalized_identity(prior) != normalized_identity(redo):
            raise ValueError(
                f"Full task identity mismatch at redo index {redo_index}:\n"
                f"prior={normalized_identity(prior)!r}\nredo={normalized_identity(redo)!r}"
            )
        replacements.append(
            {
                "redo_episode": redo_index + 1,
                "original_episode": original_index + 1,
                "trial_id": redo.get("trial_id"),
                "prior_success": task_success(prior),
                "redo_success": task_success(redo),
                "changed": prior != redo,
            }
        )
        merged[original_index] = redo

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = (
        args.backup.resolve()
        if args.backup
        else original_path.with_name(f"{original_path.stem}.pre_redo_merge_{stamp}{original_path.suffix}")
    )
    audit_path = (
        args.audit.resolve()
        if args.audit
        else original_path.with_name(f"{original_path.stem}.redo_merge_{stamp}.json")
    )
    audit = {
        "created_at_utc": stamp,
        "original_results": str(original_path),
        "redo_results": str(redos_path),
        "redo_manifest": str(manifest_path),
        "backup": str(backup_path),
        "original_row_count": len(original),
        "replacement_count": len(replacements),
        "changed_row_count": sum(item["changed"] for item in replacements),
        "prior_successes_in_replaced_rows": sum(item["prior_success"] for item in replacements),
        "redo_successes_in_replaced_rows": sum(item["redo_success"] for item in replacements),
        "replacements": replacements,
    }
    print(json.dumps({key: value for key, value in audit.items() if key != "replacements"}, indent=2))
    if args.dry_run:
        return
    if backup_path.exists():
        raise FileExistsError(f"Refusing to overwrite backup: {backup_path}")
    if audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {audit_path}")
    shutil.copy2(original_path, backup_path)
    write_jsonl_atomic(original_path, merged)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Backup: {backup_path}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
