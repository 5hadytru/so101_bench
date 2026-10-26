from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

BENCHMARK_PATH = Path(__file__).parents[1] / "so101_bench" / "benchmark.py"
BENCHMARK_SPEC = importlib.util.spec_from_file_location("so101_bench_domain_randomization_test", BENCHMARK_PATH)
assert BENCHMARK_SPEC is not None
assert BENCHMARK_SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(BENCHMARK_SPEC)
sys.modules[BENCHMARK_SPEC.name] = BENCHMARK
BENCHMARK_SPEC.loader.exec_module(BENCHMARK)

OBJECTS = ["white pen", "black pen", "yellow toy car", "cardboard box"]


def _row(domain_randomization: dict | None = None) -> dict:
    row = {
        "objects": OBJECTS,
        "instruction": "Place each object in the plastic bin",
    }
    if domain_randomization is not None:
        row["domain_randomization"] = domain_randomization
    return row


def test_partial_domain_randomization_fills_float_defaults() -> None:
    spec = BENCHMARK.episode_spec_from_json(
        _row(
            {
                "background_mesh_index": 12.0,
                "overhead_camera_position_x_m": 0.021,
                "wrist_camera_horizontal_fov_deg": 55.25,
                "light_intensity": 937.5,
                "robot_color_b": 0.72,
            }
        )
    )

    values = spec.reset_payload()["domain_randomization"]
    assert values["background_mesh_index"] == 12.0
    assert values["overhead_camera_position_x_m"] == 0.021
    assert values["overhead_camera_position_y_m"] == -0.34
    assert values["wrist_camera_horizontal_fov_deg"] == 55.25
    assert values["light_intensity"] == 937.5
    assert values["robot_color_b"] == 0.72
    assert all(isinstance(value, float) for value in values.values())


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"overhead_camera_fvo_deg": 72.0}, "unknown field"),
        ({"background_mesh_index": 2.5}, "integer-valued float from 1 to 22"),
        ({"background_mesh_index": 0}, "integer-valued float from 1 to 22"),
        ({"background_mesh_index": 23}, "integer-valued float from 1 to 22"),
        ({"light_intensity": "800.0"}, "finite number"),
        ({"robot_color_r": 1.2}, "between 0 and 1"),
    ],
)
def test_invalid_domain_randomization_is_rejected(values: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BENCHMARK.episode_spec_from_json(_row(values))


@pytest.mark.parametrize("background", [18, 19, 20, 21, 22])
def test_explicit_robust_backgrounds_are_accepted(background: int) -> None:
    spec = BENCHMARK.episode_spec_from_json(_row({"background_mesh_index": background}))
    assert spec.reset_payload()["domain_randomization"]["background_mesh_index"] == background
