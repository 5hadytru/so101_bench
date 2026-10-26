from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).parents[3] / "scripts" / "select_groot_profile_tasks.py"
SPEC = importlib.util.spec_from_file_location("select_groot_profile_tasks_test_module", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
SELECTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SELECTOR
SPEC.loader.exec_module(SELECTOR)


def _row(instruction, *, n_objects=4, ood_key="seen"):
    return {
        "instruction": instruction,
        "n_objects": n_objects,
        "objects": [f"object_{index}" for index in range(n_objects)],
        "ood_key": ood_key,
    }


def test_five_row_profile_covers_every_performance_task_shape():
    rows = [
        _row("Place each object in the plastic bin", n_objects=1),
        _row("Place each object in the plastic bin", n_objects=4, ood_key="unseen_seen_class"),
        _row("Move the object forwards", ood_key="unseen_unseen_class"),
        _row("Place the object next to the other object"),
        _row("Place the object between two other objects", ood_key="unseen_seen_class"),
    ]

    selected = SELECTOR.select_profile_indices(rows, 5)
    buckets = {SELECTOR.task_bucket(rows[index]) for index in selected}

    assert selected == [0, 1, 2, 3, 4]
    assert buckets == {"bin_1obj", "bin_4obj", "move", "next_to", "between"}


def test_extra_profile_rows_are_unique():
    rows = [
        _row("Place each object in the plastic bin", n_objects=1),
        _row("Place each object in the plastic bin", n_objects=4, ood_key="unseen_seen_class"),
        _row("Move the object forwards", ood_key="unseen_unseen_class"),
        _row("Place the object next to the other object"),
        _row("Place the object between two other objects", ood_key="unseen_seen_class"),
        _row("Move another object backwards"),
        _row("Place another object next to a third object"),
    ]

    selected = SELECTOR.select_profile_indices(rows, 7)

    assert len(selected) == 7
    assert len(set(selected)) == 7
