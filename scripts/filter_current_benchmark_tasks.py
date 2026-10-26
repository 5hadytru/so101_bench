#!/usr/bin/env python3
"""Write the rows from a task JSONL that the current benchmark catalog accepts."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile


def _load_benchmark_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "so101_bench"
        / "so101_bench"
        / "benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("so101_bench_profile_filter_benchmark", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    args = parser.parse_args()

    benchmark = _load_benchmark_module()
    source_lines = [line for line in args.input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    retained_lines: list[str] = []
    rejected_reasons: dict[str, int] = {}
    retained_objects: set[str] = set()

    for line_number, source_line in enumerate(source_lines, start=1):
        row = json.loads(source_line)
        try:
            benchmark.episode_spec_from_json(row, source=f"{args.input_jsonl}:{line_number}")
        except ValueError as exc:
            reason = str(exc).split(". Expected one of:", maxsplit=1)[0]
            rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
            continue
        retained_lines.append(source_line)
        retained_objects.update(str(name) for name in row.get("objects", ()))

    if not retained_lines:
        raise ValueError(f"No rows in {args.input_jsonl} are valid under the current benchmark catalog.")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl.exists() and args.output_jsonl.stat().st_size > 0:
        raise FileExistsError(f"Refusing to overwrite non-empty filtered task file: {args.output_jsonl}")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output_jsonl.parent, delete=False) as output:
        temporary_path = Path(output.name)
        output.write("".join(f"{line}\n" for line in retained_lines))
    temporary_path.replace(args.output_jsonl)

    print(
        f"[profile] Current-catalog filter retained {len(retained_lines)}/{len(source_lines)} task rows "
        f"and {len(retained_objects)} distinct objects: {args.output_jsonl}"
    )
    if rejected_reasons:
        summary = "; ".join(
            f"{reason} ({count})" for reason, count in sorted(rejected_reasons.items())
        )
        print(f"[profile] Rejected stale rows: {summary}")


if __name__ == "__main__":
    main()
