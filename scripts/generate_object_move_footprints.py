"""Generate top-down move-task footprints from object USD physics meshes.

Run with:

    /home/truman/env_isaaclab_51/bin/python scripts/generate_object_move_footprints.py

The output is a raster-derived union of local XY rectangles. This preserves
concavities and holes without requiring mesh processing dependencies at runtime.
"""

from __future__ import annotations

import argparse
import binascii
import json
import math
from pathlib import Path
import struct
import zlib

import numpy as np

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom, UsdPhysics


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJECTS_DIR = REPO_ROOT / "source" / "so101_bench" / "so101_bench" / "assets" / "usd" / "objects"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "source" / "so101_bench" / "so101_bench" / "assets" / "objects"
USD_SUFFIXES = {".usd", ".usda", ".usdc"}
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-dir", type=Path, default=DEFAULT_OBJECTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--resolution-mm",
        type=float,
        default=1.0,
        help="Raster cell size in millimeters. Default: 1.0",
    )
    parser.add_argument(
        "--object",
        action="append",
        default=[],
        help="Generate only this USD stem, filename, or object label. May be repeated.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Also write a PNG per object showing the rasterized footprint and merged boxes.",
    )
    parser.add_argument(
        "--visualize-dir",
        type=Path,
        default=None,
        help="Directory for visualization PNGs. Defaults to <output-dir>/visualizations.",
    )
    parser.add_argument(
        "--visualize-scale",
        type=int,
        default=4,
        help="Integer upscale factor for visualization pixels. Default: 4",
    )
    return parser.parse_args()


def _physics_meshes(stage: Usd.Stage) -> list[Usd.Prim]:
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    collision_meshes = [prim for prim in meshes if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if collision_meshes:
        return collision_meshes
    named_physics_meshes = [
        prim
        for prim in meshes
        if any("physics" in part.lower() for part in prim.GetPath().pathString.split("/")[:-1])
    ]
    return named_physics_meshes or meshes


def _projected_triangles(stage: Usd.Stage, usd_path: Path) -> tuple[list[np.ndarray], list[str]]:
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        raise ValueError(f"{usd_path}: USD stage has no valid default prim.")

    meshes = _physics_meshes(stage)
    if not meshes:
        raise ValueError(f"{usd_path}: no Mesh prim exists below a prim named 'physics'.")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_to_root = xform_cache.GetLocalToWorldTransform(root).GetInverse()
    triangles: list[np.ndarray] = []
    mesh_paths: list[str] = []
    for prim in meshes:
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not face_counts or not face_indices:
            continue

        mesh_paths.append(prim.GetPath().pathString)
        mesh_to_world = xform_cache.GetLocalToWorldTransform(prim)
        projected_points = np.asarray(
            [
                tuple(world_to_root.Transform(mesh_to_world.Transform(Gf.Vec3d(point))))[:2]
                for point in points
            ],
            dtype=np.float64,
        )
        index_offset = 0
        for face_count in face_counts:
            count = int(face_count)
            face = [int(index) for index in face_indices[index_offset : index_offset + count]]
            index_offset += count
            for triangle_index in range(1, count - 1):
                triangle = projected_points[[face[0], face[triangle_index], face[triangle_index + 1]]]
                first_edge = triangle[1] - triangle[0]
                second_edge = triangle[2] - triangle[0]
                signed_area_twice = float(first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0])
                if abs(signed_area_twice) > 1.0e-12:
                    triangles.append(triangle)

    if not triangles:
        raise ValueError(f"{usd_path}: physics meshes have no non-degenerate top-down triangles.")
    return triangles, mesh_paths


def _rasterize(triangles: list[np.ndarray], resolution_m: float) -> tuple[np.ndarray, float, float]:
    all_points = np.concatenate(triangles, axis=0)
    origin_x = math.floor(float(np.min(all_points[:, 0])) / resolution_m) * resolution_m
    origin_y = math.floor(float(np.min(all_points[:, 1])) / resolution_m) * resolution_m
    max_x = math.ceil(float(np.max(all_points[:, 0])) / resolution_m) * resolution_m
    max_y = math.ceil(float(np.max(all_points[:, 1])) / resolution_m) * resolution_m
    width = max(int(round((max_x - origin_x) / resolution_m)) + 1, 1)
    height = max(int(round((max_y - origin_y) / resolution_m)) + 1, 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    origin = np.asarray([origin_x, origin_y], dtype=np.float64)
    for triangle in triangles:
        pixels = np.rint((triangle - origin) / resolution_m).astype(np.int32)
        _fill_triangle(mask, pixels)
    return mask, origin_x, origin_y


def _fill_triangle(mask: np.ndarray, pixels: np.ndarray) -> None:
    x_min = max(int(np.min(pixels[:, 0])), 0)
    x_max = min(int(np.max(pixels[:, 0])), mask.shape[1] - 1)
    y_min = max(int(np.min(pixels[:, 1])), 0)
    y_max = min(int(np.max(pixels[:, 1])), mask.shape[0] - 1)
    if x_max < x_min or y_max < y_min:
        return

    yy, xx = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
    p0, p1, p2 = pixels.astype(np.float64)
    edge0 = (xx - p1[0]) * (p0[1] - p1[1]) - (yy - p1[1]) * (p0[0] - p1[0])
    edge1 = (xx - p2[0]) * (p1[1] - p2[1]) - (yy - p2[1]) * (p1[0] - p2[0])
    edge2 = (xx - p0[0]) * (p2[1] - p0[1]) - (yy - p0[1]) * (p2[0] - p0[0])
    inside = ((edge0 >= 0.0) & (edge1 >= 0.0) & (edge2 >= 0.0)) | (
        (edge0 <= 0.0) & (edge1 <= 0.0) & (edge2 <= 0.0)
    )
    mask[y_min : y_max + 1, x_min : x_max + 1][inside] = 1


def _row_runs(row: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(row.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _row_run_rectangles(mask: np.ndarray) -> list[list[int]]:
    active: dict[tuple[int, int], list[int]] = {}
    completed: list[list[int]] = []
    for row_index, row in enumerate(mask):
        runs = set(_row_runs(row))
        for run in list(active):
            if run not in runs:
                completed.append(active.pop(run))
        for start, end in runs:
            if (start, end) in active:
                active[(start, end)][3] = row_index + 1
            else:
                active[(start, end)] = [start, row_index, end, row_index + 1]
    completed.extend(active.values())
    return sorted(completed)


def _greedy_rectangles(mask: np.ndarray) -> list[list[int]]:
    remaining = mask.astype(bool).copy()
    height, width = remaining.shape
    completed: list[list[int]] = []
    while True:
        filled = np.argwhere(remaining)
        if filled.size == 0:
            break
        row_start, col_start = (int(value) for value in filled[0])

        width_limit = 0
        while col_start + width_limit < width and remaining[row_start, col_start + width_limit]:
            width_limit += 1

        best_row_end = row_start + 1
        best_col_end = col_start + width_limit
        best_area = width_limit
        row_end = row_start + 1
        while row_end < height and remaining[row_end, col_start]:
            row_width = 0
            while row_width < width_limit and remaining[row_end, col_start + row_width]:
                row_width += 1
            width_limit = min(width_limit, row_width)
            if width_limit <= 0:
                break
            area = width_limit * (row_end - row_start + 1)
            if area > best_area:
                best_area = area
                best_row_end = row_end + 1
                best_col_end = col_start + width_limit
            row_end += 1

        remaining[row_start:best_row_end, col_start:best_col_end] = False
        completed.append([col_start, row_start, best_col_end, best_row_end])

    return sorted(completed)


def _rectangles_to_boxes(
    rectangles: list[list[int]],
    origin_x: float,
    origin_y: float,
    resolution_m: float,
) -> list[list[float]]:
    return [
        [
            round(origin_x + start * resolution_m, 9),
            round(origin_y + row_start * resolution_m, 9),
            round(origin_x + end * resolution_m, 9),
            round(origin_y + row_end * resolution_m, 9),
        ]
        for start, row_start, end, row_end in rectangles
    ]


def _merged_boxes(mask: np.ndarray, origin_x: float, origin_y: float, resolution_m: float) -> list[list[float]]:
    row_rectangles = _row_run_rectangles(mask)
    greedy_rectangles = _greedy_rectangles(mask)
    rectangles = greedy_rectangles if len(greedy_rectangles) < len(row_rectangles) else row_rectangles
    return _rectangles_to_boxes(rectangles, origin_x, origin_y, resolution_m)


def _render_visualization(
    mask: np.ndarray,
    boxes: list[list[float]],
    origin_x: float,
    origin_y: float,
    resolution_m: float,
    scale: int,
) -> np.ndarray:
    height, width = mask.shape
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[mask.astype(bool)] = (110, 110, 110)
    scale = max(int(scale), 1)
    image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    for box_x0, box_y0, box_x1, box_y1 in boxes:
        col0 = int(round((box_x0 - origin_x) / resolution_m)) * scale
        row0 = int(round((box_y0 - origin_y) / resolution_m)) * scale
        col1 = int(round((box_x1 - origin_x) / resolution_m)) * scale
        row1 = int(round((box_y1 - origin_y) / resolution_m)) * scale
        _draw_rectangle(image, col0, row0, col1 - 1, row1 - 1, (60, 200, 60))
    # USD XY is right-handed with +Y up; image rows increase downward, so flip for display.
    return image[::-1]


def _draw_rectangle(
    image: np.ndarray,
    col0: int,
    row0: int,
    col1: int,
    row1: int,
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    col0 = max(0, min(col0, width - 1))
    col1 = max(0, min(col1, width - 1))
    row0 = max(0, min(row0, height - 1))
    row1 = max(0, min(row1, height - 1))
    if col1 < col0 or row1 < row0:
        return
    image[row0, col0 : col1 + 1] = color
    image[row1, col0 : col1 + 1] = color
    image[row0 : row1 + 1, col0] = color
    image[row0 : row1 + 1, col1] = color


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def _write_png(path: Path, image: np.ndarray) -> None:
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {image.shape}.")
    scanlines = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(scanlines))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def _selected(usd_path: Path, requested: set[str]) -> bool:
    if not requested:
        return True
    candidates = {
        usd_path.name.lower(),
        usd_path.stem.lower(),
        usd_path.stem.replace("_", " ").lower(),
    }
    return bool(candidates & requested)


def generate(
    usd_path: Path,
    output_dir: Path,
    resolution_m: float,
    visualize_dir: Path | None = None,
    visualize_scale: int = 4,
) -> Path:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"{usd_path}: could not open USD stage.")
    triangles, mesh_paths = _projected_triangles(stage, usd_path)
    mask, origin_x, origin_y = _rasterize(triangles, resolution_m)
    boxes = _merged_boxes(mask, origin_x, origin_y, resolution_m)
    if not boxes:
        raise ValueError(f"{usd_path}: rasterized footprint is empty.")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_usd": usd_path.name,
        "raster_resolution_m": resolution_m,
        "physics_mesh_paths": mesh_paths,
        "boxes": boxes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{usd_path.stem}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if visualize_dir is not None:
        image = _render_visualization(mask, boxes, origin_x, origin_y, resolution_m, visualize_scale)
        visualize_dir.mkdir(parents=True, exist_ok=True)
        _write_png(visualize_dir / f"{usd_path.stem}.png", image)

    return output_path


def main() -> int:
    args = parse_args()
    resolution_m = float(args.resolution_mm) / 1000.0
    if not math.isfinite(resolution_m) or resolution_m <= 0.0:
        raise ValueError("--resolution-mm must be a positive finite number.")
    requested = {value.lower().removesuffix(".usd").removesuffix(".usda").removesuffix(".usdc") for value in args.object}
    usd_paths = sorted(
        path
        for path in args.objects_dir.iterdir()
        if path.is_file() and path.suffix.lower() in USD_SUFFIXES and _selected(path, requested)
    )
    if not usd_paths:
        raise ValueError(f"No matching USD assets found in {args.objects_dir}.")

    visualize_dir = None
    if args.visualize:
        visualize_dir = args.visualize_dir or (args.output_dir / "visualizations")

    for usd_path in usd_paths:
        output_path = generate(usd_path, args.output_dir, resolution_m, visualize_dir, args.visualize_scale)
        print(f"{usd_path.name}: {output_path}")
        if visualize_dir is not None:
            print(f"{usd_path.name}: {visualize_dir / f'{usd_path.stem}.png'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
