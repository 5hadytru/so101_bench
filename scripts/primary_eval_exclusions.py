"""primary_eval_v2 rows removed from scoring and from future runs (tasks/primary_eval_v2_excluded.json).

The rows stay in tasks/primary_eval_v2.jsonl (trial_id == row index everywhere); every scorer and
campaign skips these trial ids.
"""

from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
_EXCLUDED_FILE = ROOT / "tasks/primary_eval_v2_excluded.json"  # private evaluation only
EXCLUDED_TRIAL_IDS: frozenset[int] = frozenset(
    json.loads(_EXCLUDED_FILE.read_text())["trial_ids"] if _EXCLUDED_FILE.is_file() else ()
)


def excluded(trial_id: int | str) -> bool:
    return int(str(trial_id).removeprefix("ep")) in EXCLUDED_TRIAL_IDS


@lru_cache(maxsize=1)
def _canonical_excluded_rows():
    source = ROOT/'tasks/primary_eval_v2.jsonl'
    if not EXCLUDED_TRIAL_IDS or not source.is_file():
        return {}
    return {r['trial_id']: r for r in
            (json.loads(s) for s in source.read_text().splitlines() if s.strip())
            if r['trial_id'] in EXCLUDED_TRIAL_IDS}


def primary_episode_excluded(episode) -> bool:
    """Recognize frozen/subset copies without dropping unrelated datasets' IDs."""
    trial = (episode.metadata or {}).get('trial_id')
    row = _canonical_excluded_rows().get(trial)
    return bool(row and tuple(row['objects']) == tuple(episode.objects)
                and row['instruction'] == episode.instruction)
