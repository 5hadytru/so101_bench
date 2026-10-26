"""Task, object, and evaluation constants for SO-101 Bench.

The task definitions and failure rules are taken from the attached SO-101 Bench
paper and its appendix. Distances are stored in meters for direct use in Isaac Lab.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

INCH = 0.0254
MULTI_RIGID_BODY_CHILD_NAMES = ("left", "right")

TASK_BIN = "bin"
TASK_NAMED_BIN = "named_bin"
TASK_NEXT_TO = "next_to"
TASK_BETWEEN = "between"
TASK_MOVE = "move"
TASK_MIXED = "mixed"

TASK_FAMILIES = (TASK_BIN, TASK_NAMED_BIN, TASK_NEXT_TO, TASK_BETWEEN, TASK_MOVE)
BIN_TASK_FAMILIES = (TASK_BIN, TASK_NAMED_BIN)

DIRECTIONS = ("left", "right", "forward", "backward")
COLORS = {
    "black",
    "blue",
    "brown",
    "gold",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "tan",
    "white",
    "yellow"
}
# Color words that may appear as the first token of an object label.  ``COLORS``
# is the authoritative registered subset used by task generation.  Keeping the
# broader vocabulary separate lets the parser reject, for example, ``teal mug``
# instead of silently treating "teal" as part of the object class.
KNOWN_LEADING_COLORS = COLORS | {
    "aqua",
    "aquamarine",
    "beige",
    "burgundy",
    "cyan",
    "gold",
    "golden",
    "indigo",
    "ivory",
    "lavender",
    "magenta",
    "maroon",
    "navy",
    "ochre",
    "teal",
    "turquoise",
    "violet",
}


def _leading_color(label: str) -> str | None:
    words = label.split()
    if not words:
        return None
    leading_word = words[0].lower()
    if leading_word in COLORS:
        return leading_word
    if leading_word in KNOWN_LEADING_COLORS:
        raise ValueError(
            f"Unregistered leading color {leading_word!r} in object label {label!r}. "
            "Add it to COLORS before using it in an object label."
        )
    return None


MAX_GRASP_ATTEMPTS = 3
GRASP_ATTEMPT_OBJECT_DISTANCE_M = 4.0 * INCH
BIN_DISPLACEMENT_LIMIT_M = 0.5 * INCH
NON_TARGET_DISPLACEMENT_LIMIT_M = 1.0 * INCH
NON_BIN_TARGET_MIN_DISPLACEMENT_M = 1.0 * INCH
ROBOT_CONTACT_FORCE_THRESHOLD_N = 0.05
# A non-bin object only counts as "lifted off the ground" for postmortem failure
# classification once its root rises this far above its settled resting height.
LIFT_OFF_GROUND_LIMIT_M = 0.5 * INCH
BOUNDARY_DISPLACEMENT_LIMIT_M = 0.5 * INCH
SPATIAL_SUCCESS_DISTANCE_M = 2.0 * INCH
BETWEEN_LINE_TOLERANCE_M = 1.75 * INCH
# Retained as diagnostic bounds for the legacy surface-distance fraction.  Live
# success instead uses the center-distance lens constraint: the target must be
# closer to each referent than the referents are to each other.
BETWEEN_CENTER_FRACTION_MIN = 0.0
BETWEEN_CENTER_FRACTION_MAX = 1.0
MOVE_BOUNDARY_SUCCESS_DISTANCE_M = 2.0 * INCH
MOVE_NO_BOUNDARY_MIN_PROGRESS_M = 2.0 * INCH
MOVE_STRAIGHTNESS_TOLERANCE_M = 2.0 * INCH
# Retained for compatibility with older outcome files and postmortem APIs. The live
# Move task no longer imposes a minimum signed boundary gap or an overshoot failure.
MOVE_PAST_BOUNDARY_TOLERANCE_M = 0.25 * INCH
# A nearest object only counts as the move boundary if it blocks at least this fraction
# of the target's lateral corridor. Below it the object is merely beside the path (a
# glancing clip), so the move is scored on forward progress instead of "reaching" it.
MOVE_BOUNDARY_MIN_LATERAL_OVERLAP_FRACTION = 0.1
DEFAULT_EPISODE_LENGTH_S = 30.0
FOUR_OBJECT_BIN_EPISODE_LENGTH_S = 75.0


def episode_length_s(task_family: str, object_count: int) -> float:
    """Return the timeout for one benchmark episode."""

    if task_family == TASK_BIN and object_count == 4:
        return FOUR_OBJECT_BIN_EPISODE_LENGTH_S
    return DEFAULT_EPISODE_LENGTH_S

# Each object records its supported task categories and whether its USD contains
# multiple rigid bodies. In that case, the object cannot be initialized with
# RigidObjectCfg and must use AssetBaseCfg instead.
OBJECT_SPLITS: dict[str, dict[str, dict[str, bool | list[str]]]] = {
    "seen": {
        # "black glasses": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"], "legacy": True},
        # "silver glasses": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"], "legacy": True},
        "white pen": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "black pen": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "altoids container": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "black remote": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        # "brown stuffed animal": {"multiple_rigid_bodies": False, "deformable": False},
        "blue pliers": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "green clip": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "pink eraser": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "yellow wires": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        # "grey wires": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"], "legacy": True},
        # "black screwdriver": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"], "legacy": True},
        "yellow screwdriver": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "red tape": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "black tape": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "cardboard box": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "flower pot": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "cooking spoon": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["BIN"]},
        "yellow toy car": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "grey toy car": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "green shoes": {"multiple_rigid_bodies": True, "deformable": False, "tasks": ["BIN"]},
        # "black shoes": {"multiple_rigid_bodies": True, "deformable": False, "tasks": ["ALL"], "legacy": True},
        "blue bowl": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "blue scissors": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "silver battery": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "red pepper": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "green sponge": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "white die": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "pink pacifier": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "green spool": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "purple mouthguard": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "red whistle": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "blue kazoo": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "brown bottle opener": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "brown bow tie": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "white air pump": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "pink harmonica": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "purple pill planner": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "orange walkie talkie": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "red candy cane": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "red button": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        # "black bike lock": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "brown ice cream scoop": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "grey toy boat": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "blue paintbrush": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "brown sunglasses case": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "brown trimmer": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "grey flash drive": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "grey camera": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        # "red skateboard tool": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "blue hand brush": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "blue torpedo": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "hose splitter": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "yellow lego": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        # "tan corkscrew": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "yellow juice box": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
    },
    "unseen_seen_class": {
        # "orange glasses": {"multiple_rigid_bodies": False, "deformable": False},
        # "white glasses": {"multiple_rigid_bodies": False, "deformable": False},
        "blue clip": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "blue tape": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "yellow tape": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        # "white stuffed animal": {"multiple_rigid_bodies": False, "deformable": False},
        "blue screwdriver": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        # "pink bowl": {"multiple_rigid_bodies": False, "deformable": False},
        "white bowl": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "black wires": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "brown wires": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "orange toy car": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "blue pen": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "red pen": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "white shoes": {"multiple_rigid_bodies": True, "deformable": False, "tasks": ["BIN"]},
        "red pliers": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "grey scissors": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "green battery": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "grey remote": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "green pepper": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "blue sponge": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "yellow die": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "purple pacifier": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "purple spool": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "orange mouthguard": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "black whistle": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "purple bottle opener": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "pink bow tie": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "green air pump": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "yellow flash drive": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "white pill planner": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "yellow walkie talkie": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "green kazoo": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "blue harmonica": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "gold candy cane": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "blue button": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        # "pink bike lock": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "black ice cream scoop": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "white toy boat": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "brown paintbrush": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "green sunglasses case": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "black trimmer": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "black camera": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        # "black skateboard tool": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "green juice box": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "red lego": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "red torpedo": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "white corkscrew": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "white eraser": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
    },
    "unseen_unseen_class": {
        "blue highlighter": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "purple toothbrush": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "blue controller": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "action figure": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "razor": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "silver tongs": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "playing cards": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "toy fire truck": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "toy monster truck": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
        "toy dinosaur": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
    },
}

# The held-out unseen-object/unseen-class objects are not published: the public split above holds only
# the ones in tasks/real_gr00t_WM_all_filtered.jsonl. A local private_objects.py (never committed)
# supplies the full split, in its original order. It is loaded by path so this works however
# benchmark.py itself is imported.
PUBLIC_UNSEEN_UNSEEN_CLASS: tuple[str, ...] = tuple(OBJECT_SPLITS["unseen_unseen_class"])
_PRIVATE_OBJECTS = Path(__file__).resolve().with_name("private_objects.py")
if _PRIVATE_OBJECTS.is_file():
    import importlib.util as _importlib_util

    _spec = _importlib_util.spec_from_file_location("so101_bench_private_objects", _PRIVATE_OBJECTS)
    _private = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_private)
    OBJECT_SPLITS["unseen_unseen_class"] = dict(_private.UNSEEN_UNSEEN_CLASS)

# Retire from active generation even when the private split overrides the public
# catalog. Legacy scene metadata below permits unchanged distractor/referent rows.
RETIRED_GRASP_OBJECTS = frozenset({"candy bar"})
for _split_objects in OBJECT_SPLITS.values():
    for _retired_name in RETIRED_GRASP_OBJECTS:
        _split_objects.pop(_retired_name, None)

BENCHMARK_OBJECT_NAMES: tuple[str, ...] = tuple(
    object_name for split in OBJECT_SPLITS.values() for object_name in split
)
# Retired objects are excluded from every benchmark split, but metadata for an
# object already present in a partially collected dataset may remain available
# so that the recorded task prefix can still be parsed and replayed.
LEGACY_OBJECT_METADATA: dict[str, dict[str, bool | list[str]]] = {
    "black bike lock": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
    "candy bar": {"multiple_rigid_bodies": False, "deformable": False, "tasks": ["ALL"]},
}
OBJECT_METADATA: dict[str, dict[str, bool | list[str]]] = {
    **{
        object_name: metadata
        for split in OBJECT_SPLITS.values()
        for object_name, metadata in split.items()
    },
    **LEGACY_OBJECT_METADATA,
}
for _object_name in OBJECT_METADATA:
    _leading_color(_object_name)
MOVE_FOOTPRINT_SCHEMA_VERSION = 1
MOVE_FOOTPRINT_GENERATOR_COMMAND = (
    "/home/truman/env_isaaclab_51/bin/python scripts/generate_object_move_footprints.py"
)
OBJECT_MOVE_FOOTPRINT_DIR = Path(__file__).resolve().parent / "assets" / "objects"

FAILURE_TAXONOMY: dict[str, tuple[str, ...]] = {
    "shared_grasp_acquisition": (
        "bad grasp strategy",
        "imprecise grasp",
        "grabbed air",
        "refused to lift",
        "occlusion-induced grasp failure",
        "got stuck on top",
        "dropped/pushed out of range",
    ),
    "bin_placement": (
        "knocked bin",
        "missed bin",
        "not fully in bin",
    ),
    "named_bin": (
        "wrong object targeted",
        "distractor moved",
        "knocked bin",
        "missed bin",
        "not fully in bin",
    ),
    "shared_instruction_following": (
        "repeatedly reached then docked",
        "semantic error",
        "grasped class distractor",
        "grasped color distractor",
        "grasped other object",
        "failure to reset",
        "wrong task",
        "failed to undock",
        "moved an object",
    ),
    "next_to": (
        "placed next to other object",
        "placed next to class distractor",
        "placed next to color distractor",
        "made contact",
        "not close enough",
        "drove/rammed into object",
    ),
    "between": (
        "semantic error",
        "made contact",
        "not centered enough",
        "too close to referent",
        "not close",
    ),
    "move": (
        "not close enough to boundary",
        "trajectory not straight enough",
        "moved boundary",
        "made contact",
    ),
}


def task_instruction(task_family: str, active_labels: list[str], direction: str = "") -> str:
    """Return the natural-language instruction for a benchmark episode."""

    if task_family == TASK_BIN:
        return "Place each object in the plastic bin."
    if task_family == TASK_NAMED_BIN:
        return f"Place the {active_labels[0]} in the plastic bin."
    if task_family == TASK_NEXT_TO:
        return f"Place the {active_labels[0]} next to the {active_labels[1]}."
    if task_family == TASK_BETWEEN:
        return f"Place the {active_labels[0]} between the {active_labels[1]} and the {active_labels[2]}."
    if task_family == TASK_MOVE:
        assert direction
        return f"Move the {active_labels[0]} {direction}."
    raise ValueError(f"Unknown task family: {task_family}")


# Fixed film-gate dimensions used to turn the per-row horizontal FOV into the
# same USD focal-length representation as the existing camera configuration.
OVERHEAD_CAMERA_HORIZONTAL_APERTURE = 34.61329224445431
WRIST_CAMERA_HORIZONTAL_APERTURE = 20.955


@dataclass(frozen=True)
class DomainRandomizationSpec:
    """Resolved per-episode visual settings stored under ``domain_randomization``.

    Every JSON value is a scalar number so task-generation scripts can write a
    completely self-contained row.  Partial objects are also accepted: omitted
    values resolve to the simulator defaults declared here.  The background is
    addressed by its numeric suffix, e.g. ``1.0`` selects
    ``background_1.usdc``.
    """

    background_mesh_index: float = 1.0

    overhead_camera_position_x_m: float = 0.0
    overhead_camera_position_y_m: float = -0.34
    overhead_camera_position_z_m: float = 0.45722
    overhead_camera_roll_deg: float = 51.361
    overhead_camera_pitch_deg: float = -0.169
    overhead_camera_yaw_deg: float = 0.603
    overhead_camera_horizontal_fov_deg: float = 74.0
    overhead_camera_principal_point_x_offset: float = 0.0
    overhead_camera_principal_point_y_offset: float = 0.0
    overhead_camera_exposure_ev: float = 0.0
    overhead_camera_noise_std: float = 0.0
    overhead_camera_blur_sigma_px: float = 0.0

    wrist_camera_position_x_m: float = -0.0025
    wrist_camera_position_y_m: float = 0.0595
    wrist_camera_position_z_m: float = -0.0605
    wrist_camera_roll_deg: float = -45.0
    wrist_camera_pitch_deg: float = 0.0
    wrist_camera_yaw_deg: float = -1.0
    wrist_camera_horizontal_fov_deg: float = 72.5
    wrist_camera_principal_point_x_offset: float = 0.0
    wrist_camera_principal_point_y_offset: float = 0.0
    wrist_camera_exposure_ev: float = 0.0
    wrist_camera_noise_std: float = 0.0
    wrist_camera_blur_sigma_px: float = 0.0

    light_intensity: float = 800.0
    light_color_temperature_k: float = 6500.0
    light_color_r: float = 1.0
    light_color_g: float = 0.96
    light_color_b: float = 0.88

    robot_color_r: float = 0.95
    robot_color_g: float = 0.15
    robot_color_b: float = 0.02

    @classmethod
    def from_json(
        cls,
        value: dict[str, Any] | None,
        *,
        source: str = "JSONL field 'domain_randomization'",
    ) -> DomainRandomizationSpec:
        """Validate a partial JSON object and fill all omitted defaults."""

        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError(f"{source} must be an object, got {value!r}.")

        valid_fields = set(cls.__dataclass_fields__)
        unknown_fields = sorted(set(value) - valid_fields)
        if unknown_fields:
            raise ValueError(
                f"{source} has unknown field(s): {', '.join(unknown_fields)}. "
                f"Expected fields: {', '.join(sorted(valid_fields))}."
            )

        resolved: dict[str, float] = {}
        for name, raw_value in value.items():
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"{source}.{name} must be a finite number, got {raw_value!r}.")
            number = float(raw_value)
            if not math.isfinite(number):
                raise ValueError(f"{source}.{name} must be finite, got {raw_value!r}.")
            resolved[name] = number

        spec = cls(**resolved)
        spec._validate(source=source)
        return spec

    def _validate(self, *, source: str) -> None:
        background_index = self.background_mesh_index
        # Parsing accepts explicit ROBUST holdouts; training samplers stay at 1–18.
        if not background_index.is_integer() or not 1 <= int(background_index) <= 22:
            raise ValueError(f"{source}.background_mesh_index must be an integer-valued float from 1 to 22 (19–22 are ROBUST-only holdouts).")

        for name in ("overhead_camera_horizontal_fov_deg", "wrist_camera_horizontal_fov_deg"):
            value = getattr(self, name)
            if not 0.0 < value < 180.0:
                raise ValueError(f"{source}.{name} must be between 0 and 180 degrees, got {value}.")

        for name in (
            "overhead_camera_principal_point_x_offset",
            "overhead_camera_principal_point_y_offset",
            "wrist_camera_principal_point_x_offset",
            "wrist_camera_principal_point_y_offset",
        ):
            value = getattr(self, name)
            if not -0.5 <= value <= 0.5:
                raise ValueError(f"{source}.{name} must be between -0.5 and 0.5, got {value}.")

        for name in (
            "overhead_camera_noise_std",
            "wrist_camera_noise_std",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{source}.{name} must be between 0 and 1, got {value}.")

        for name in (
            "overhead_camera_blur_sigma_px",
            "wrist_camera_blur_sigma_px",
            "light_intensity",
        ):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{source}.{name} must be non-negative, got {value}.")
        if self.light_color_temperature_k <= 0.0:
            raise ValueError(
                f"{source}.light_color_temperature_k must be positive, got {self.light_color_temperature_k}."
            )

        for name in (
            "light_color_r",
            "light_color_g",
            "light_color_b",
            "robot_color_r",
            "robot_color_g",
            "robot_color_b",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{source}.{name} must be between 0 and 1, got {value}.")

    def to_json(self) -> dict[str, float]:
        """Return the fully resolved scalar-float representation."""

        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class BenchmarkEpisodeSpec:
    """Validated JSONL episode metadata consumed by reset and scene configuration."""

    objects: tuple[str, ...]
    instruction: str
    task_family: str
    target_object_id: int
    referent_object_ids: tuple[int, int]
    direction: str | None = None
    metadata: dict[str, Any] | None = None
    domain_randomization: DomainRandomizationSpec = DomainRandomizationSpec()

    def reset_payload(self) -> dict[str, Any]:
        """Return the JSON-compatible subset needed by the reset event."""

        if not episode_is_executable(self):
            raise ValueError("Episode requires grasping a retired object: candy bar")
        return {
            "objects": list(self.objects),
            "instruction": self.instruction,
            "task_family": self.task_family,
            "active_object_ids": list(range(len(self.objects))),
            "target_object_id": self.target_object_id,
            "referent_object_ids": list(self.referent_object_ids),
            "direction": self.direction,
            "metadata": dict(self.metadata or {}),
            "domain_randomization": self.domain_randomization.to_json(),
        }


def episode_is_executable(episode: BenchmarkEpisodeSpec) -> bool:
    """Old tasks may retain retired props, but may not require grasping them."""
    grasped = episode.objects if episode.task_family == TASK_BIN else (episode.objects[episode.target_object_id],)
    return not RETIRED_GRASP_OBJECTS.intersection(grasped)


def object_metadata(object_name: str) -> dict[str, bool | list[str]]:
    """Return validated metadata for an object name from ``OBJECT_SPLITS``."""

    try:
        return OBJECT_METADATA[object_name]
    except KeyError as exc:
        valid = ", ".join(BENCHMARK_OBJECT_NAMES)
        raise ValueError(f"Unknown benchmark object {object_name!r}. Expected one of: {valid}.") from exc


def object_rigid_body_child_names(object_name: str) -> tuple[str, ...]:
    """Return child prim names that should be treated as independently movable rigid bodies."""

    if object_metadata(object_name)["multiple_rigid_bodies"]:
        return MULTI_RIGID_BODY_CHILD_NAMES
    return ()


def object_usd_stem(object_name: str) -> str:
    """Return the local object USD filename stem used by JSONL labels."""

    object_metadata(object_name)
    return object_name.replace(" ", "_")


def object_move_footprint_path(object_name: str) -> Path:
    """Return the generated move-task footprint metadata path for an object."""

    return OBJECT_MOVE_FOOTPRINT_DIR / f"{object_usd_stem(object_name)}.json"


def load_object_move_footprint_boxes(
    object_name: str,
    *,
    required: bool = True,
) -> tuple[tuple[float, float, float, float], ...]:
    """Load raster-derived local XY rectangles in meters, including the USD scale."""

    path = object_move_footprint_path(object_name)
    if not path.is_file():
        if required:
            raise ValueError(
                f"Missing generated move-task footprint metadata for {object_name!r}: {path}. "
                f"Run `{MOVE_FOOTPRINT_GENERATOR_COMMAND}`."
            )
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read generated move-task footprint metadata for {object_name!r}: {path}. "
            f"Run `{MOVE_FOOTPRINT_GENERATOR_COMMAND}`."
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != MOVE_FOOTPRINT_SCHEMA_VERSION:
        raise ValueError(
            f"Generated move-task footprint metadata for {object_name!r} has an unsupported schema: {path}. "
            f"Run `{MOVE_FOOTPRINT_GENERATOR_COMMAND}`."
        )
    raw_boxes = payload.get("boxes")
    if not isinstance(raw_boxes, list) or not raw_boxes:
        raise ValueError(
            f"Generated move-task footprint metadata for {object_name!r} has no boxes: {path}. "
            f"Run `{MOVE_FOOTPRINT_GENERATOR_COMMAND}`."
        )
    boxes = []
    for raw_box in raw_boxes:
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            raise ValueError(
                f"Invalid move-task footprint box in {path}: {raw_box!r}. "
                f"Run `{MOVE_FOOTPRINT_GENERATOR_COMMAND}`."
            )
        try:
            box = tuple(float(value) for value in raw_box)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid move-task footprint box in {path}: {raw_box!r}. "
                f"Run `{MOVE_FOOTPRINT_GENERATOR_COMMAND}`."
            ) from exc
        if not all(math.isfinite(value) for value in box) or box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError(
                f"Invalid move-task footprint box in {path}: {raw_box!r}. "
                f"Run `{MOVE_FOOTPRINT_GENERATOR_COMMAND}`."
            )
        boxes.append(box)
    return tuple(boxes)


def validate_move_episode_footprints(episodes: list[BenchmarkEpisodeSpec]) -> None:
    """Require generated geometry for every object that can participate in a move task."""

    move_object_names = sorted(
        {
            object_name
            for episode in episodes
            if episode.task_family == TASK_MOVE
            for object_name in episode.objects
        }
    )
    for object_name in move_object_names:
        load_object_move_footprint_boxes(object_name)


def _normalized_instruction(instruction: str) -> str:
    return " ".join(instruction.strip().lower().rstrip(".").split())


def _normalized_object_label(label: str) -> str:
    normalized = _normalized_instruction(label)
    for determiner in ("the ", "a ", "an "):
        if normalized.startswith(determiner):
            return normalized[len(determiner) :]
    return normalized


def _colorless_label(object_name: str) -> str:
    words = object_name.split()
    if _leading_color(object_name) is not None:
        return " ".join(words[1:])
    return object_name


def _canonical_direction(token: str) -> str:
    direction = token.lower()
    aliases = {"forwards": "forward", "backwards": "backward"}
    direction = aliases.get(direction, direction)
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown move direction {token!r}. Expected one of: {', '.join(DIRECTIONS)}.")
    return direction


def infer_task_family(instruction: str) -> str:
    """Infer a supported benchmark task family from an instruction."""

    normalized = _normalized_instruction(instruction)
    if "plastic bin" in normalized and normalized.startswith("place"):
        return TASK_BIN if normalized.startswith("place each object ") else TASK_NAMED_BIN
    if normalized.startswith("place") and " next to " in normalized:
        return TASK_NEXT_TO
    if normalized.startswith("place") and " between " in normalized and " and " in normalized:
        return TASK_BETWEEN
    if normalized.startswith("move "):
        return TASK_MOVE
    raise ValueError(
        f"Instruction {instruction!r} does not match a supported benchmark task. "
        "Expected all-object bin, named-object bin, next-to, between, or directional move phrasing."
    )


def _object_mentions(instruction: str, objects: tuple[str, ...]) -> list[tuple[int, int]]:
    mentions: list[tuple[int, int]] = []
    lowered_instruction = instruction.lower()
    for object_id, object_name in enumerate(objects):
        match = re.search(rf"(?<!\w){re.escape(object_name.lower())}(?!\w)", lowered_instruction)
        if match is not None:
            mentions.append((match.start(), object_id))
    return sorted(mentions)


def _resolve_instruction_object_label(
    label: str,
    objects: tuple[str, ...],
    *,
    source: str,
) -> int:
    normalized_label = _normalized_object_label(label)
    exact_matches = [
        object_id
        for object_id, object_name in enumerate(objects)
        if _normalized_object_label(object_name) == normalized_label
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        names = [objects[object_id] for object_id in exact_matches]
        raise ValueError(f"{source}: object label {label!r} is ambiguous; it matches {names}.")

    colorless_matches = [
        object_id
        for object_id, object_name in enumerate(objects)
        if _normalized_object_label(_colorless_label(object_name)) == normalized_label
    ]
    if len(colorless_matches) == 1:
        return colorless_matches[0]
    if len(colorless_matches) > 1:
        names = [objects[object_id] for object_id in colorless_matches]
        raise ValueError(f"{source}: object label {label!r} is ambiguous; it matches {names}.")

    raise ValueError(f"{source}: object label {label!r} does not match any row object: {list(objects)}.")


def _parse_instruction_object_labels(instruction: str, task_family: str, *, source: str) -> tuple[str, ...]:
    if task_family == TASK_BIN:
        return ()
    if task_family == TASK_NAMED_BIN:
        match = re.fullmatch(
            r"\s*place\s+(?:the\s+)?(.+?)\s+in\s+(?:the\s+)?plastic\s+bin\.?\s*",
            instruction,
            flags=re.IGNORECASE,
        )
        if match is not None:
            return (match.group(1),)
        raise ValueError(
            f"{source}: named-bin instruction must be phrased as 'Place the TARGET in the plastic bin'."
        )
    if task_family == TASK_NEXT_TO:
        match = re.fullmatch(
            r"\s*place\s+(?:the\s+)?(.+?)\s+next\s+to\s+(?:the\s+)?(.+?)\.?\s*",
            instruction,
            flags=re.IGNORECASE,
        )
        if match is not None:
            return (match.group(1), match.group(2))
        raise ValueError(f"{source}: next-to instruction must be phrased as 'Place the TARGET next to the REFERENT'.")
    if task_family == TASK_BETWEEN:
        match = re.fullmatch(
            r"\s*place\s+(?:the\s+)?(.+?)\s+between\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)\.?\s*",
            instruction,
            flags=re.IGNORECASE,
        )
        if match is not None:
            return (match.group(1), match.group(2), match.group(3))
        raise ValueError(
            f"{source}: between instruction must be phrased as "
            "'Place the TARGET between the REFERENT and the REFERENT'."
        )
    match = re.fullmatch(
        r"\s*move\s+(?:the\s+)?(.+?)\s+(left|right|forwards?|backwards?)\.?\s*",
        instruction,
        flags=re.IGNORECASE,
    )
    if match is not None:
        return (match.group(1), match.group(2))
    raise ValueError(f"{source}: move instruction must be phrased as 'Move the TARGET DIRECTION'.")


def row_with_canonical_instruction_metadata(
    row: dict[str, Any],
    *,
    instruction: str | None = None,
    source: str = "JSONL row",
) -> dict[str, Any]:
    """Return ``row`` with task metadata made consistent with its instruction.

    Explicit ``target``, ``referents``, and ``direction`` fields override natural
    language inference during normal loading.  This helper is for edit paths:
    it reparses the new instruction, removes stale overrides that no longer
    apply, and writes only the overrides required for that task family.
    """

    updated = dict(row)
    if instruction is not None:
        updated["instruction"] = instruction

    raw_objects = updated.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects or not all(isinstance(name, str) for name in raw_objects):
        raise ValueError(f"{source}: 'objects' must be a non-empty list of benchmark object names.")
    objects = tuple(raw_objects)
    instruction_value = updated.get("instruction")
    if not isinstance(instruction_value, str) or not instruction_value.strip():
        raise ValueError(f"{source}: 'instruction' must be a non-empty string.")
    instruction_text = instruction_value.strip()
    task_family = infer_task_family(instruction_text)

    for key in ("target", "referents", "direction"):
        updated.pop(key, None)
    updated["instruction"] = instruction_text
    if "task_family" in updated:
        updated["task_family"] = task_family

    labels = _parse_instruction_object_labels(instruction_text, task_family, source=source)
    if task_family == TASK_NAMED_BIN:
        target_id = _resolve_instruction_object_label(labels[0], objects, source=source)
        updated["target"] = objects[target_id]
    elif task_family == TASK_NEXT_TO:
        target_id = _resolve_instruction_object_label(labels[0], objects, source=source)
        referent_id = _resolve_instruction_object_label(labels[1], objects, source=source)
        if referent_id == target_id:
            raise ValueError(f"{source}: next-to episodes need distinct target and referent objects.")
        updated["target"] = objects[target_id]
        updated["referents"] = [objects[referent_id]]
    elif task_family == TASK_BETWEEN:
        target_id = _resolve_instruction_object_label(labels[0], objects, source=source)
        referent_ids = [
            _resolve_instruction_object_label(label, objects, source=source)
            for label in labels[1:]
        ]
        if len(set([target_id, *referent_ids])) != 3:
            raise ValueError(f"{source}: between episodes need a target and two distinct referents.")
        updated["target"] = objects[target_id]
        updated["referents"] = [objects[object_id] for object_id in referent_ids]
    elif task_family == TASK_MOVE:
        target_id = _resolve_instruction_object_label(labels[0], objects, source=source)
        updated["target"] = objects[target_id]
        updated["direction"] = _canonical_direction(labels[1])

    episode_spec_from_json(updated, source=source)
    return updated


def _object_id_from_row_name(objects: tuple[str, ...], row: dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or value not in objects:
        raise ValueError(f"JSONL field {key!r} must name one of the row objects, got {value!r}.")
    return objects.index(value)


def _referent_ids_from_row(objects: tuple[str, ...], row: dict[str, Any]) -> list[int]:
    referents = row.get("referents")
    if referents is None:
        return []
    if not isinstance(referents, list) or not all(isinstance(name, str) for name in referents):
        raise ValueError(f"JSONL field 'referents' must be a list of object names, got {referents!r}.")
    invalid = [name for name in referents if name not in objects]
    if invalid:
        raise ValueError(f"JSONL referents must be present in 'objects', got invalid referents: {invalid}.")
    return [objects.index(name) for name in referents]


def _referent_pair(object_count: int, referent_ids: list[int], fallback: list[int]) -> tuple[int, int]:
    ids = [*referent_ids, *fallback]
    if not ids:
        return (0, 0)
    first = ids[0]
    second = ids[1] if len(ids) > 1 else first
    for object_id in (first, second):
        if object_id < 0 or object_id >= object_count:
            raise ValueError(f"Referent object id {object_id} is out of range for {object_count} objects.")
    return (first, second)


def episode_spec_from_json(row: dict[str, Any], *, source: str = "JSONL row") -> BenchmarkEpisodeSpec:
    """Validate a JSONL row and derive task indices used by the simulator."""

    raw_objects = row.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects or not all(isinstance(name, str) for name in raw_objects):
        raise ValueError(f"{source}: 'objects' must be a non-empty list of benchmark object names.")
    objects = tuple(raw_objects)
    if len(objects) > 4:
        raise ValueError(f"{source}: the benchmark supports at most four tabletop objects, got {len(objects)}.")
    if len(set(objects)) != len(objects):
        raise ValueError(f"{source}: episode object names must be unique, got {list(objects)}.")
    for object_name in objects:
        object_metadata(object_name)

    n_objects = row.get("n_objects")
    if n_objects is not None and n_objects != len(objects):
        raise ValueError(f"{source}: n_objects={n_objects!r} does not match {len(objects)} objects.")

    instruction = row.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{source}: 'instruction' must be a non-empty string.")
    instruction = instruction.strip()
    inferred_task_family = infer_task_family(instruction)
    task_family = row.get("task_family") or inferred_task_family
    if task_family not in TASK_FAMILIES:
        raise ValueError(f"{source}: unsupported task_family {task_family!r}.")
    if task_family != inferred_task_family:
        raise ValueError(
            f"{source}: task_family {task_family!r} does not match the instruction family "
            f"{inferred_task_family!r}."
        )

    mentions = [object_id for _offset, object_id in _object_mentions(instruction, objects)]
    row_target_id = _object_id_from_row_name(objects, row, "target")
    row_referent_ids = _referent_ids_from_row(objects, row)

    if task_family == TASK_BIN:
        if len(objects) not in {1, 4}:
            raise ValueError(f"{source}: bin episodes must contain either one or four objects.")
        target_id = 0
        referents = _referent_pair(len(objects), [], mentions[1:])
        direction = None
    elif task_family == TASK_NAMED_BIN:
        labels = _parse_instruction_object_labels(instruction, task_family, source=source)
        instruction_target_id = _resolve_instruction_object_label(labels[0], objects, source=source)
        if row_target_id is not None and row_target_id != instruction_target_id:
            raise ValueError(
                f"{source}: named-bin target {objects[row_target_id]!r} does not match the "
                f"instruction target {objects[instruction_target_id]!r}."
            )
        target_id = instruction_target_id
        fallback = [object_id for object_id in range(len(objects)) if object_id != target_id]
        referents = _referent_pair(len(objects), row_referent_ids, fallback)
        direction = None
    elif task_family == TASK_NEXT_TO:
        if len(objects) != 4:
            raise ValueError(f"{source}: next-to episodes must contain four objects.")
        if row_target_id is None and len(mentions) < 2:
            raise ValueError(f"{source}: next-to instruction must mention target and referent objects.")
        target_id = row_target_id if row_target_id is not None else mentions[0]
        fallback = [object_id for object_id in mentions if object_id != target_id]
        referents = _referent_pair(len(objects), row_referent_ids, fallback)
        if referents[0] == target_id:
            raise ValueError(f"{source}: next-to episodes need distinct target and referent objects.")
        direction = None
    elif task_family == TASK_BETWEEN:
        if len(objects) != 4:
            raise ValueError(f"{source}: between episodes must contain four objects.")
        if row_target_id is None and len(mentions) < 3:
            raise ValueError(f"{source}: between instruction must mention target and two referent objects.")
        target_id = row_target_id if row_target_id is not None else mentions[0]
        fallback = [object_id for object_id in mentions if object_id != target_id]
        referents = _referent_pair(len(objects), row_referent_ids, fallback)
        if referents[0] == referents[1] or target_id in referents:
            raise ValueError(f"{source}: between episodes need a target and two distinct referents.")
        direction = None
    else:
        if len(objects) != 4:
            raise ValueError(f"{source}: move episodes must contain four objects.")
        if row_target_id is None and not mentions:
            raise ValueError(f"{source}: move instruction must mention the moved object.")
        target_id = row_target_id if row_target_id is not None else mentions[0]
        direction_value = row.get("direction")
        if direction_value is None:
            match = re.search(r"\b(left|right|forwards?|backwards?)\b", instruction, flags=re.IGNORECASE)
            if match is None:
                raise ValueError(f"{source}: move instruction must include a direction.")
            direction_value = match.group(1)
        if not isinstance(direction_value, str):
            raise ValueError(f"{source}: move direction must be a string, got {direction_value!r}.")
        direction = _canonical_direction(direction_value)
        fallback = [object_id for object_id in range(len(objects)) if object_id != target_id]
        referents = _referent_pair(len(objects), row_referent_ids, fallback)

    domain_randomization = DomainRandomizationSpec.from_json(
        row.get("domain_randomization"),
        source=f"{source}: 'domain_randomization'",
    )

    return BenchmarkEpisodeSpec(
        objects=objects,
        instruction=instruction,
        task_family=task_family,
        target_object_id=target_id,
        referent_object_ids=referents,
        direction=direction,
        metadata={
            key: value
            for key, value in row.items()
            if key not in {"objects", "instruction", "domain_randomization"}
        },
        domain_randomization=domain_randomization,
    )


def load_episode_jsonl(path: str | Path) -> list[BenchmarkEpisodeSpec]:
    """Load and validate benchmark episodes from a JSONL file."""

    path = Path(path)
    episodes: list[BenchmarkEpisodeSpec] = []
    with path.open(encoding="utf-8") as jsonl_file:
        for line_no, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object per line.")
            episodes.append(episode_spec_from_json(row, source=f"{path}:{line_no}"))
    if not episodes:
        raise ValueError(f"{path}: JSONL file did not contain any benchmark episodes.")
    validate_move_episode_footprints(episodes)
    return episodes
