"""Select completed robot-agent trials without counting infrastructure failures."""

from __future__ import annotations

import json
from pathlib import Path
from primary_eval_exclusions import EXCLUDED_TRIAL_IDS


def valid_result(row: dict) -> bool:
    """Model failures count; provider/runtime failures do not."""
    if row.get("success") is None or row.get("termination_reason") in (None, "error", "runtime_error"):
        return False
    if row.get("infrastructure_error"):
        return False
    agent = row.get("pyrua") or {}
    if agent.get("status") in ("boot_error", "error", "planner_error"):
        return False
    if agent.get("status") == "timeout" and agent.get("http_429", 0):
        return False
    return True


def latest_results(roots: list[Path], excluded=None) -> dict[int, tuple[Path, dict]]:
    """Use the latest valid completed attempt per trial, by file timestamp."""
    if excluded is None:
        excluded = EXCLUDED_TRIAL_IDS
    selected = {}
    for root in roots:
        for path in root.glob("run*/ep*/result.json"):
            row = json.loads(path.read_text())
            trial = int(path.parent.name[2:])
            if trial in excluded or not valid_result(row):
                continue
            previous = selected.get(trial)
            if previous is None or path.stat().st_mtime_ns > previous[0].stat().st_mtime_ns:
                selected[trial] = (path, row)
    return selected
