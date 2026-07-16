#!/usr/bin/env python3
"""Update one episode instruction in a task JSONL and its generated layout JSONL."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = REPO_ROOT / "source" / "so101_bench" / "so101_bench" / "benchmark.py"
BENCHMARK_SPEC = importlib.util.spec_from_file_location("so101_bench_benchmark_update_instruction", BENCHMARK_PATH)
if BENCHMARK_SPEC is None or BENCHMARK_SPEC.loader is None:
    raise RuntimeError(f"Could not load benchmark module from {BENCHMARK_PATH}.")
BENCHMARK = importlib.util.module_from_spec(BENCHMARK_SPEC)
sys.modules[BENCHMARK_SPEC.name] = BENCHMARK
BENCHMARK_SPEC.loader.exec_module(BENCHMARK)
row_with_canonical_instruction_metadata = BENCHMARK.row_with_canonical_instruction_metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task_jsonl",
        "--episodes_jsonl",
        type=Path,
        required=True,
        help="Task JSONL file whose matching row's instruction should be updated.",
    )
    parser.add_argument(
        "--layout_jsonl",
        "--episode_layouts_jsonl",
        "--layouts_jsonl",
        type=Path,
        required=True,
        help="Layout JSONL file whose matching row's copied instruction should be updated.",
    )
    parser.add_argument(
        "--trial_id",
        required=True,
        help="trial_id of the episode row to edit. Compared as text so numeric and string JSON ids both work.",
    )
    parser.add_argument(
        "--instruction",
        required=True,
        help="Replacement instruction text.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Report the rows that would change without writing files.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[tuple[int, str, dict]]:
    rows = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line_no, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                rows.append((line_no, line, {}))
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected one JSON object per line.")
            rows.append((line_no, line, row))
    if not rows:
        raise ValueError(f"{path}: file is empty.")
    return rows


def _matching_row_indices(rows: list[tuple[int, str, dict]], trial_id: str) -> list[int]:
    return [
        index
        for index, (_line_no, _line, row) in enumerate(rows)
        if row and "trial_id" in row and str(row["trial_id"]) == trial_id
    ]


def _object_names(row: dict) -> list[str]:
    objects = row.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("'objects' must be a non-empty list.")
    if all(isinstance(name, str) for name in objects):
        return list(objects)
    if all(isinstance(entry, dict) and "name" in entry for entry in objects):
        return [str(entry["name"]) for entry in sorted(objects, key=lambda entry: int(entry.get("slot", 0)))]
    raise ValueError("'objects' must contain either object names or layout object entries.")


def _row_with_updated_instruction_metadata(row: dict, *, instruction: str, source: str) -> dict:
    task_like_row = dict(row)
    task_like_row["objects"] = _object_names(row)
    canonical = row_with_canonical_instruction_metadata(task_like_row, instruction=instruction, source=source)

    updated = dict(row)
    for key in ("target", "referents", "direction"):
        updated.pop(key, None)
    updated["instruction"] = canonical["instruction"]
    if "task_family" in updated and "task_family" in canonical:
        updated["task_family"] = canonical["task_family"]
    for key in ("target", "referents", "direction"):
        if key in canonical:
            updated[key] = canonical[key]
    return updated


def _prepare_instruction_replacement(
    path: Path,
    *,
    trial_id: str,
    instruction: str,
) -> tuple[list[str], int, str, str]:
    rows = _load_jsonl(path)
    matches = _matching_row_indices(rows, trial_id)
    if not matches:
        raise ValueError(f"{path}: no row found for trial_id={trial_id!r}.")
    if len(matches) > 1:
        line_numbers = [rows[index][0] for index in matches]
        raise ValueError(f"{path}: duplicate rows found for trial_id={trial_id!r}: lines {line_numbers}.")

    match_index = matches[0]
    line_no, original_line, row = rows[match_index]
    old_instruction = row.get("instruction")
    if not isinstance(old_instruction, str):
        raise ValueError(f"{path}:{line_no}: matching row has no string 'instruction' field.")

    updated_row = _row_with_updated_instruction_metadata(
        row,
        instruction=instruction,
        source=f"{path}:{line_no}",
    )
    updated_line = json.dumps(updated_row, separators=(",", ":")) + "\n"

    rows[match_index] = (line_no, updated_line, updated_row)
    return [line for _line_no, line, _row in rows], line_no, old_instruction, instruction


def _write_jsonl_lines(path: Path, lines: list[str]) -> None:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.writelines(lines)
    temp_path.replace(path)


def main() -> None:
    args = _parse_args()
    new_instruction = args.instruction.strip()
    if not new_instruction:
        raise ValueError("Replacement instruction must not be empty.")

    replacements = []
    for path in (args.task_jsonl, args.layout_jsonl):
        lines, line_no, old_instruction, updated_instruction = _prepare_instruction_replacement(
            path,
            trial_id=str(args.trial_id),
            instruction=new_instruction,
        )
        replacements.append((path, lines, line_no, old_instruction, updated_instruction))

    if not args.dry_run:
        for path, lines, _line_no, _old_instruction, _updated_instruction in replacements:
            _write_jsonl_lines(path, lines)

    action = "Would update" if args.dry_run else "Updated"
    for path, _lines, line_no, old_instruction, updated_instruction in replacements:
        print(f"{action} {path}:{line_no}")
        print(f"  old: {old_instruction}")
        print(f"  new: {updated_instruction}")

    print(
        "\nNote: the 'instruction' field and semantic task metadata "
        "('target', 'referents', 'direction', and existing 'task_family') are updated together. "
        "Objects and all layout poses are left as-is."
    )


if __name__ == "__main__":
    main()
