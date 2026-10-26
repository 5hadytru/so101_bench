#!/usr/bin/env python3
"""Generate the deterministic visual-domain-randomization UI review matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "tasks/domain_randomization"
DEFAULT_OVERHEAD_CONFIGS_PATH = CONFIGS_DIR / "overhead_camera_configs.json"
DEFAULT_WRIST_CONFIGS_PATH = CONFIGS_DIR / "wrist_camera_configs.json"
DEFAULT_WRIST_CANDIDATES_PATH = CONFIGS_DIR / "wrist_camera_default_candidates.json"
DEFAULT_WRIST_POSE_CANDIDATES_PATH = CONFIGS_DIR / "wrist_camera_pose_candidates.json"

# Both cameras are randomized by picking one of a hand-tuned catalog of whole camera
# configurations rather than by sampling each field independently, so every option is a
# coherent, visually checked camera instead of an arbitrary corner of a hyper-rectangle.
OVERHEAD_CAMERA_FIELDS = (
    "overhead_camera_position_x_m",
    "overhead_camera_position_y_m",
    "overhead_camera_position_z_m",
    "overhead_camera_roll_deg",
    "overhead_camera_pitch_deg",
    "overhead_camera_yaw_deg",
    "overhead_camera_horizontal_fov_deg",
    "overhead_camera_principal_point_x_offset",
    "overhead_camera_principal_point_y_offset",
    "overhead_camera_exposure_ev",
    "overhead_camera_noise_std",
    "overhead_camera_blur_sigma_px",
)

# The wrist camera is bolted to the gripper. Catalog options may include small mount-
# orientation tolerances, but its translation remains fixed in the accepted catalog.
WRIST_MOUNT_FIELDS = (
    "wrist_camera_position_x_m",
    "wrist_camera_position_y_m",
    "wrist_camera_position_z_m",
    "wrist_camera_roll_deg",
    "wrist_camera_pitch_deg",
    "wrist_camera_yaw_deg",
)
WRIST_CAMERA_FIELDS = WRIST_MOUNT_FIELDS + (
    "wrist_camera_horizontal_fov_deg",
    "wrist_camera_principal_point_x_offset",
    "wrist_camera_principal_point_y_offset",
    "wrist_camera_exposure_ev",
    "wrist_camera_noise_std",
    "wrist_camera_blur_sigma_px",
)

# Mount pose plus field of view: used only to pin down the repository default against a
# real camera frame, where lateral offset and field of view trade off against each other.
WRIST_POSE_FIELDS = (
    "wrist_camera_position_x_m",
    "wrist_camera_position_y_m",
    "wrist_camera_position_z_m",
    "wrist_camera_roll_deg",
    "wrist_camera_pitch_deg",
    "wrist_camera_yaw_deg",
    "wrist_camera_horizontal_fov_deg",
)


DEFAULTS = {
    "background_mesh_index": 1.0,
    "overhead_camera_position_x_m": 0.0,
    "overhead_camera_position_y_m": -0.34,
    "overhead_camera_position_z_m": 0.45722,
    "overhead_camera_roll_deg": 51.361,
    "overhead_camera_pitch_deg": -0.169,
    "overhead_camera_yaw_deg": 0.603,
    "overhead_camera_horizontal_fov_deg": 74.0,
    "overhead_camera_principal_point_x_offset": 0.0,
    "overhead_camera_principal_point_y_offset": 0.0,
    "overhead_camera_exposure_ev": 0.0,
    "overhead_camera_noise_std": 0.0,
    "overhead_camera_blur_sigma_px": 0.0,
    "wrist_camera_position_x_m": -0.0025,
    "wrist_camera_position_y_m": 0.0595,
    "wrist_camera_position_z_m": -0.0605,
    "wrist_camera_roll_deg": -45.0,
    "wrist_camera_pitch_deg": 0.0,
    "wrist_camera_yaw_deg": -1.0,
    "wrist_camera_horizontal_fov_deg": 72.5,
    "wrist_camera_principal_point_x_offset": 0.0,
    "wrist_camera_principal_point_y_offset": 0.0,
    "wrist_camera_exposure_ev": 0.0,
    "wrist_camera_noise_std": 0.0,
    "wrist_camera_blur_sigma_px": 0.0,
    "light_intensity": 800.0,
    "light_color_temperature_k": 6500.0,
    "light_color_r": 1.0,
    "light_color_g": 0.96,
    "light_color_b": 0.88,
    "robot_color_r": 0.95,
    "robot_color_g": 0.15,
    "robot_color_b": 0.02,
}

LIGHTING_CONFIGS = (
    ("default", 800.0, 6500.0, (1.0, 0.96, 0.88)),
    ("dim", 600.0, 6500.0, (1.0, 0.96, 0.88)),
    ("bright", 1200.0, 6500.0, (1.0, 0.96, 0.88)),
    ("warm_indoor", 800.0, 3200.0, (1.0, 0.68, 0.45)),
    ("cool_daylight", 800.0, 9000.0, (0.62, 0.80, 1.0)),
    ("dim_warm", 600.0, 3800.0, (1.0, 0.76, 0.55)),
    ("bright_cool", 1100.0, 8500.0, (0.72, 0.86, 1.0)),
)
ROBOT_COLORS = {
    "orange": (0.95, 0.15, 0.02),
    "red": (0.64, 0.012, 0.03),
    "turquoise": (0.15, 0.60, 0.69),
    "yellow": (0.84, 0.69, 0.07),
    "white": (0.95, 0.95, 0.95),
    "black": (0.08, 0.08, 0.08),
}

# Focused material-color sweep for matching the photographed red SO-101.  USD
# diffuse colors are linear RGB, so the candidates center on the linearized
# value of the reference's midtone red and span brightness and undertone
# independently.  This catalog is for visual selection only; choosing a winner
# is a separate, deliberate update to ROBOT_COLORS.
ROBOT_RED_CANDIDATES = {
    "old_bright_baseline": (1.00, 0.020, 0.010),
    "vivid_080": (0.80, 0.015, 0.015),
    "vivid_072": (0.72, 0.015, 0.015),
    "reference_linear": (0.68, 0.010, 0.014),
    "vivid_064": (0.64, 0.015, 0.015),
    "vivid_056": (0.56, 0.015, 0.015),
    "soft_072": (0.72, 0.028, 0.022),
    "soft_064": (0.64, 0.028, 0.022),
    "warm_064": (0.64, 0.035, 0.012),
    "selected_red": (0.64, 0.012, 0.030),
}


def load_camera_configs(
    path: Path,
    fields: tuple[str, ...],
    *,
    optional_fields: tuple[str, ...] = (),
) -> list[tuple[str, dict[str, float]]]:
    """Load and validate a discrete camera-option catalog."""

    raw_configs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_configs, list) or not raw_configs:
        raise ValueError(f"{path} must contain a non-empty JSON array.")

    expected_fields = set(fields)
    configs: list[tuple[str, dict[str, float]]] = []
    names: set[str] = set()
    for index, raw_config in enumerate(raw_configs):
        if not isinstance(raw_config, dict):
            raise ValueError(f"{path}: configuration {index} must be an object.")
        name = raw_config.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}: configuration {index} needs a non-empty string name.")
        if name in names:
            raise ValueError(f"{path}: duplicate overhead-camera configuration name {name!r}.")
        names.add(name)

        provided_fields = set(raw_config) - {"name"}
        missing = sorted(expected_fields - provided_fields - set(optional_fields))
        unknown = sorted(provided_fields - expected_fields)
        if missing or unknown:
            raise ValueError(
                f"{path}: configuration {name!r} has missing fields {missing} "
                f"and unknown fields {unknown}."
            )
        values: dict[str, float] = {}
        for field in provided_fields:
            raw_value = raw_config[field]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"{path}: {name}.{field} must be a finite number.")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"{path}: {name}.{field} must be finite.")
            values[field] = value
        for field in optional_fields:
            values.setdefault(field, DEFAULTS[field])
        configs.append((name, values))

    if configs[0][0] != "default":
        raise ValueError(f"{path}: the first configuration must be named 'default'.")
    expected_default = {field: DEFAULTS[field] for field in expected_fields}
    if configs[0][1] != expected_default:
        raise ValueError(
            f"{path}: the 'default' configuration must exactly match repository defaults."
        )
    return configs


def _row(trial_id: int, test_case: str, overrides: dict[str, float]) -> dict:
    settings = {**DEFAULTS, **overrides}
    assert all(isinstance(value, float) for value in settings.values())
    return {
        "objects": ["black pen", "white pen", "pink eraser", "altoids container"],
        "ood_key": "seen",
        "trial_id": trial_id,
        "domain_randomization_test": test_case,
        "n_objects": 4,
        "instruction": "Place the black pen between the pink eraser and the white pen",
        "target": "black pen",
        "referents": ["pink eraser", "white pen"],
        "domain_randomization": settings,
    }


def build_rows(
    randomization_type: str = "background",
    *,
    overhead_configs_path: Path = DEFAULT_OVERHEAD_CONFIGS_PATH,
    wrist_configs_path: Path = DEFAULT_WRIST_CONFIGS_PATH,
    wrist_candidates_path: Path = DEFAULT_WRIST_CANDIDATES_PATH,
    wrist_pose_candidates_path: Path = DEFAULT_WRIST_POSE_CANDIDATES_PATH,
) -> list[dict]:
    lighting_by_name = {
        name: {
            "light_intensity": intensity,
            "light_color_temperature_k": temperature,
            "light_color_r": color[0],
            "light_color_g": color[1],
            "light_color_b": color[2],
        }
        for name, intensity, temperature, color in LIGHTING_CONFIGS
    }
    cases_by_type: dict[str, list[tuple[str, dict[str, float]]]] = {
        "overhead_camera": [
            (f"overhead_camera_{name}", values)
            for name, values in load_camera_configs(
                overhead_configs_path, OVERHEAD_CAMERA_FIELDS
            )
        ],
        "wrist_camera": [
            (f"wrist_camera_{name}", values)
            for name, values in load_camera_configs(
                wrist_configs_path,
                WRIST_CAMERA_FIELDS,
                optional_fields=WRIST_MOUNT_FIELDS,
            )
        ],
        "default": [("default", {})],
        "wrist_camera_pose": [
            (f"wrist_pose_{name}", values)
            for name, values in load_camera_configs(
                wrist_pose_candidates_path, WRIST_POSE_FIELDS
            )
        ],
        "wrist_camera_default": [
            (f"wrist_candidate_{name}", values)
            for name, values in load_camera_configs(
                wrist_candidates_path,
                WRIST_CAMERA_FIELDS,
                optional_fields=WRIST_MOUNT_FIELDS,
            )
        ],
        "lighting": [
            (
                f"lighting_{name}",
                {
                    "light_intensity": intensity,
                    "light_color_temperature_k": temperature,
                    "light_color_r": color[0],
                    "light_color_g": color[1],
                    "light_color_b": color[2],
                },
            )
            for name, intensity, temperature, color in LIGHTING_CONFIGS
        ],
        "background": [
            (f"background_{index:02d}", {"background_mesh_index": float(index)})
            for index in range(1, 19)
        ],
        "robot_color": [
            (
                f"robot_{name}",
                {"robot_color_r": rgb[0], "robot_color_g": rgb[1], "robot_color_b": rgb[2]},
            )
            for name, rgb in ROBOT_COLORS.items()
        ],
        "robot_red_candidate": [
            (
                f"robot_red_candidate_{name}",
                {"robot_color_r": rgb[0], "robot_color_g": rgb[1], "robot_color_b": rgb[2]},
            )
            for name, rgb in ROBOT_RED_CANDIDATES.items()
        ],
        "robot_color_lighting": [
            (
                f"robot_{color_name}_lighting_{lighting_name}",
                {
                    "robot_color_r": rgb[0],
                    "robot_color_g": rgb[1],
                    "robot_color_b": rgb[2],
                    **lighting_by_name[lighting_name],
                },
            )
            for color_name, rgb in ROBOT_COLORS.items()
            for lighting_name in ("default", "dim")
        ],
    }
    if randomization_type == "all":
        order = (
            "overhead_camera",
            "wrist_camera",
            "lighting",
            "background",
            "robot_color",
        )
        cases = [case for type_name in order for case in cases_by_type[type_name]]
    else:
        cases = cases_by_type[randomization_type]
    return [_row(trial_id, name, overrides) for trial_id, (name, overrides) in enumerate(cases)]


# (label prefix, field prefix, guide title, fields)
CAMERA_CATALOGS = (
    ("overhead_camera", "overhead_camera", "Overhead camera", OVERHEAD_CAMERA_FIELDS),
    ("wrist_camera", "wrist_camera", "Wrist camera", WRIST_CAMERA_FIELDS),
    ("wrist_candidate", "wrist_camera", "Candidate wrist-camera default", WRIST_CAMERA_FIELDS),
    ("wrist_pose", "wrist_camera", "Candidate wrist-camera mount pose", WRIST_POSE_FIELDS),
)


def _camera_summary(settings: dict[str, float], prefix: str, fields: tuple[str, ...]) -> str:
    """Describe how one camera configuration differs from the repository default."""

    differences = [
        f"`{field.removeprefix(prefix + '_')}` {settings[field]:g}"
        for field in fields
        if settings[field] != DEFAULTS[field]
    ]
    if not differences:
        return "repository defaults throughout"
    return ", ".join(differences)


def build_guide(rows: list[dict]) -> str:
    lighting_descriptions = {
        f"lighting_{name}": (
            f"Controlled rig intensity `{intensity:g}`, color temperature `{temperature:g} K`, "
            f"RGB tint `({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})`; cameras, background, and robot default."
        )
        for name, intensity, temperature, color in LIGHTING_CONFIGS
    }
    lines = [
        "# Domain-randomization recording guide",
        "",
        "This guide matches `tasks/test_domain_rand.jsonl`. **LeRobot episode indices are zero-based**; ",
        "the review number is the corresponding one-based episode number shown to a person.",
        "",
        "Every row contains all 33 domain-randomization fields. Unless a variation is named below, that ",
        "field remains at the repository default. All rows use `ood_key: seen`.",
        "",
        "| LeRobot index | Review # | Test label | Applied variation |",
        "|---:|---:|---|---|",
    ]
    for index, row in enumerate(rows):
        name = row["domain_randomization_test"]
        settings = row["domain_randomization"]
        if name == "default":
            description = "Repository defaults for every field; no variation applied."
        elif name in lighting_descriptions:
            description = lighting_descriptions[name]
        elif any(name.startswith(f"{label}_") for label, _, _, _ in CAMERA_CATALOGS):
            label, field_prefix, title, fields = next(
                entry for entry in CAMERA_CATALOGS if name.startswith(f"{entry[0]}_")
            )
            config_name = name.removeprefix(f"{label}_")
            other = "overhead" if field_prefix == "wrist_camera" else "wrist"
            description = (
                f"{title} configuration `{config_name}`: "
                + _camera_summary(settings, field_prefix, fields)
                + f"; {other} camera, light, background, and robot values default."
            )
        elif name.startswith("background_"):
            background_index = int(settings["background_mesh_index"])
            description = f"`background_{background_index}.usdc`; all camera, light, and robot values default."
        elif name.startswith("robot_"):
            color_name = name.removeprefix("robot_")
            lighting_name = None
            if "_lighting_" in color_name:
                color_name, lighting_name = color_name.rsplit("_lighting_", 1)
            rgb = (
                settings["robot_color_r"],
                settings["robot_color_g"],
                settings["robot_color_b"],
            )
            if lighting_name is None:
                description = (
                    f"{color_name} robot, RGB `({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})`; "
                    "otherwise defaults."
                )
            else:
                description = (
                    f"{color_name} robot, RGB `({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})`, "
                    f"under `{lighting_name}` lighting (intensity `{settings['light_intensity']:g}`); "
                    "both cameras and background default."
                )
        else:
            raise ValueError(f"No guide description for {name!r}.")
        lines.append(f"| {index} | {index + 1} | `{name}` | {description} |")

    active_names = {row["domain_randomization_test"] for row in rows}
    for label, field_prefix, title, fields in CAMERA_CATALOGS:
        catalog_rows = [
            row for row in rows
            if row["domain_randomization_test"].startswith(f"{label}_")
        ]
        if not catalog_rows:
            continue
        short_fields = tuple(field.removeprefix(f"{field_prefix}_") for field in fields)
        lines.extend(["", f"## {title} configurations", ""])
        if label == "wrist_pose":
            lines.extend(
                [
                    "These pin down the repository's wrist-camera mount pose against a real camera ",
                    "frame. Lateral offset and field of view are not separable from one image -- a ",
                    "camera further from the grasp axis and a narrower lens both push the gripper ",
                    "sideways -- so the rows cross four `position_x_m` values with three fields of ",
                    "view. Mount y, z, and rotation are held at the repository default throughout.",
                    "",
                    "Compare each against the real wrist frame on two things independently: how far ",
                    "left of centre the fingertips sit, and how much of the frame they span. Pick the ",
                    "row that matches both, and report its LeRobot episode index.",
                    "",
                ]
            )
        elif label == "wrist_candidate":
            lines.extend(
                [
                    "These are candidate values for the repository's wrist-camera default, not ",
                    "randomization options: pick the one whose frames best match the real camera and ",
                    "it becomes the new default. Only the horizontal field of view differs between ",
                    "rows; exposure, noise, blur, and the principal point are zero throughout so the ",
                    "geometry is the only variable.",
                    "",
                    "Each value is derived from a real M12 lens on the 1/2.7-inch OV2710 sensor that ",
                    "the 2MP 1080p modules use (5.856 x 3.276 mm active area). A `_crop` option assumes ",
                    "the driver delivers 640x480 as a 4:3 crop of the sensor; a `_full` option assumes ",
                    "it downscales the full 16:9 width. Stock modules ship 2.8-3.6 mm lenses; the ",
                    "The repository default is 72.5 degrees, selected by matching simulated framing ",
                    "to the real wrist-camera feed; the remaining rows retain lens candidates for ",
                    "comparison.",
                    "",
                    "`lens_2p9_crop` (74 degrees) is the principled pick: a 2.9 mm lens cropped to 4:3 ",
                    "puts the vertical field of view at 58.9 degrees, exactly the sensor's true value, ",
                    "so both image axes match the real optics at once.",
                    "",
                ]
            )
        elif label == "overhead_camera":
            lines.extend(
                [
                    "Each option is one complete camera, not an independently sampled field. Every ",
                    "configuration is aimed at the same workspace point as the default and was checked ",
                    "to keep the table, the robot, and all six bin positions framed at least as well as ",
                    "the default does.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "The wrist camera is bolted to the gripper, so translation remains fixed. Four ",
                    "options apply small one-axis orientation tolerances around the calibrated mount; ",
                    "each option is one complete camera configuration. ",
                    "",
                    "The field-of-view span covers the commodity 32x32 UVC modules the official ",
                    "SO-ARM wrist mount is built around, which are mostly wide-angle: an OV9732/OV2710 ",
                    "module is typically 100-120 degrees diagonal (87-108 horizontal). The repository ",
                    "uses a 72.5-degree horizontal default selected by matching the observed real-camera ",
                    "framing. Other options capture narrower optics and image-quality variation.",
                    "",
                    "Two caveats on what the renderer can represent. Blur is an isotropic Gaussian, so ",
                    "it stands in for defocus and lens softness, not directional motion blur. More ",
                    "importantly, the render is a rectilinear pinhole: real wide-angle modules show heavy ",
                    "barrel distortion that this pipeline cannot produce, so the widest options are ",
                    "geometrically wider without looking fisheye the way the real lens would.",
                    "",
                ]
            )
        lines.extend(
            [
                "| Configuration | " + " | ".join(f"`{field}`" for field in short_fields) + " |",
                "|---" * (len(short_fields) + 1) + "|",
            ]
        )
        for row in catalog_rows:
            settings = row["domain_randomization"]
            config_name = row["domain_randomization_test"].removeprefix(f"{label}_")
            values = " | ".join(f"{settings[field]:g}" for field in fields)
            lines.append(f"| `{config_name}` | {values} |")
        lines.append("")

    lines.extend(["", "## What to report", ""])
    lines.append(
        "For any problematic recording, report the **LeRobot episode index** and what failed. In particular, check:"
    )
    lines.append("")
    if active_names == {"default"}:
        lines.extend(
            [
                "- whether the standardized setup renders as expected end to end;",
                "- whether the wrist and overhead views both frame the workspace correctly.",
            ]
        )
    elif active_names and all(name.startswith("wrist_pose_") for name in active_names):
        lines.extend(
            [
                "Pick the single row to adopt as the repository default and report its LeRobot ",
                "episode index. Judge by:",
                "",
                "- how far left of centre the fingertips sit, against the same measure on the real frame;",
                "- how much of the frame width the fingertips span;",
                "- how much of the finger body is visible above the bottom edge;",
                "- how much table and bin surround the gripper.",
                "",
                "Note the renderer is a rectilinear pinhole, so a real wide lens will show barrel ",
                "distortion that no row here reproduces; match the centre of the frame, not the edges.",
            ]
        )
    elif active_names and all(name.startswith("wrist_candidate_") for name in active_names):
        lines.extend(
            [
                "Pick the single option to adopt as the repository default, and report its LeRobot ",
                "episode index. Judge by:",
                "",
                "- whether both gripper fingers read clearly, with enough jaw visible to see a grasp close;",
                "- how closely the framing matches a real frame from the same camera at the same pose;",
                "- whether the object being approached stays visible as the gripper descends onto it;",
                "- whether the amount of table and arm around the gripper looks like the real feed.",
            ]
        )
    elif active_names and all(name.startswith("wrist_camera_") for name in active_names):
        lines.extend(
            [
                "- whether both gripper fingers and the grasp point stay in frame through the whole reach;",
                "- whether a narrower field of view crops the fingertips or the object being approached;",
                "- whether exposure, blur, or noise make object identity or color hard to read up close;",
                "- whether a principal-point offset produces a visible edge smear;",
                "- whether the view still reads as a plausible wrist camera module.",
            ]
        )
    elif active_names and all(name.startswith("overhead_camera_") for name in active_names):
        lines.extend(
            [
                "- whether the table, the bin, every tabletop object, and the useful part of the robot stay in frame;",
                "- whether the arm occludes the workspace more than the default configuration does;",
                "- whether the viewpoint still reads as a plausible real camera mounting;",
                "- whether exposure, blur, or noise make object identity or color hard to read;",
                "- whether a principal-point offset produces a visible edge smear.",
            ]
        )
    elif active_names and all(name.startswith("background_") for name in active_names):
        lines.extend(
            [
                "- whether the background intersects or occludes the table, robot, bin, or objects;",
                "- whether the complete working area and all tabletop objects remain visible;",
                "- whether the background is disproportionately bright, dark, busy, or reflective;",
                "- whether its scale, alignment, and perspective look plausible from both cameras.",
            ]
        )
    elif active_names and all(name.startswith("robot_red_candidate_") for name in active_names):
        lines.extend(
            [
                "- compare the robot's broad midtone surfaces, not specular highlights or deep shadows;",
                "- report the LeRobot episode index whose red most closely matches the reference photograph;",
                "- note whether the winner is too orange, too magenta, too dark, or still too bright;",
                "- compare both wrist and overhead recordings before choosing, since view-dependent lighting differs.",
            ]
        )
    else:
        lines.extend(
            [
                "- whether all tabletop objects, the working area, and the useful part of the robot remain in frame;",
                "- whether shadows or exposure make object identity/color difficult to read;",
                "- whether principal-point shifts, blur, or noise look unrealistic;",
                "- whether a background intersects, occludes, or visually overwhelms the tabletop scene;",
                "- whether each robot color preserves enough contrast against every important scene element.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tasks/test_domain_rand.jsonl"))
    parser.add_argument("--guide-output", type=Path, default=Path("tasks/test_domain_rand_guide.md"))
    parser.add_argument(
        "--randomization-type",
        choices=(
            "default",
            "background",
            "overhead_camera",
            "wrist_camera",
            "wrist_camera_default",
            "wrist_camera_pose",
            "lighting",
            "robot_color",
            "robot_red_candidate",
            "robot_color_lighting",
            "all",
        ),
        default="background",
        help="Write a focused test for one randomization category (default: background).",
    )
    args = parser.parse_args()
    rows = build_rows(args.randomization_type)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"Wrote {len(rows)} test rows to {args.output}")
    args.guide_output.parent.mkdir(parents=True, exist_ok=True)
    args.guide_output.write_text(build_guide(rows), encoding="utf-8")
    print(f"Wrote episode guide to {args.guide_output}")


if __name__ == "__main__":
    main()
