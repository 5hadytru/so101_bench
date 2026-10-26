"""Project saved episode layouts onto the sim overhead camera image (for staging real episodes).

Object and bin footprints from a layout JSONL row are drawn where the sim overhead camera
would see them on the table, so a real overhead camera aligned to the sim view shows exactly
where each physical object and the bin go. Plain numpy/OpenCV; no Isaac imports.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


IMAGE_SIZE = (640, 480)
# Distinct BGR colors per object slot; the bin is drawn in white.
SLOT_COLORS = ((0, 200, 255), (255, 120, 0), (80, 255, 80), (255, 80, 255), (0, 80, 255), (255, 255, 0))


def _rotation_xyz(rpy_deg) -> np.ndarray:
    """Extrinsic x-y-z Euler rotation, as isaacsim's euler_angles_to_quat (and scipy 'xyz')."""

    rx, ry, rz = (math.radians(v) for v in rpy_deg)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


class OverheadProjector:
    """Pinhole model of the sim overhead camera (OpenGL convention: looks along -Z, +Y up)."""

    def __init__(self, position, rpy_deg, horizontal_fov_deg, image_size=IMAGE_SIZE):
        self.position = np.asarray(position, dtype=float)
        self.rotation = _rotation_xyz(rpy_deg)
        self.width, self.height = image_size
        # Square pixels: Isaac derives the vertical aperture from the aspect ratio.
        self.focal_px = 0.5 * self.width / math.tan(math.radians(horizontal_fov_deg) / 2.0)

    @classmethod
    def from_benchmark_defaults(cls, benchmark) -> "OverheadProjector":
        spec = benchmark.DomainRandomizationSpec()
        return cls(
            (spec.overhead_camera_position_x_m, spec.overhead_camera_position_y_m, spec.overhead_camera_position_z_m),
            (spec.overhead_camera_roll_deg, spec.overhead_camera_pitch_deg, spec.overhead_camera_yaw_deg),
            spec.overhead_camera_horizontal_fov_deg,
        )

    def project(self, points_xyz: np.ndarray) -> np.ndarray:
        camera = (np.asarray(points_xyz, dtype=float) - self.position) @ self.rotation  # = R^T (p - t)
        depth = -camera[:, 2]
        u = 0.5 * self.width + self.focal_px * camera[:, 0] / depth
        v = 0.5 * self.height - self.focal_px * camera[:, 1] / depth
        return np.stack([u, v], axis=1)


def _world_corners(local_boxes, center_xy, yaw: float, z: float) -> list[np.ndarray]:
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    polygons = []
    for x0, y0, x1, y1 in local_boxes:
        local = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
        world_x = center_xy[0] + cos_yaw * local[:, 0] - sin_yaw * local[:, 1]
        world_y = center_xy[1] + sin_yaw * local[:, 0] + cos_yaw * local[:, 1]
        polygons.append(np.stack([world_x, world_y, np.full(4, z)], axis=1))
    return polygons


def draw_layout(image_bgr: np.ndarray, layout: dict[str, Any], projector: OverheadProjector, *, table_z: float = 0.0,
                thickness: int = 2) -> np.ndarray:
    """Draw the bin outline and every object's footprint boxes + name onto a copy of the image."""

    out = image_bgr.copy()
    sx, sy = out.shape[1] / projector.width, out.shape[0] / projector.height
    scale = np.array([sx, sy])

    def polyline(world: np.ndarray, color, width: int) -> np.ndarray:
        pixels = projector.project(world) * scale
        cv2.polylines(out, [np.round(pixels).astype(np.int32)], True, color, width, cv2.LINE_AA)
        return pixels

    bin_row = layout.get("bin")
    if bin_row is not None:
        hx, hy = bin_row["footprint_half_extents"]
        polygon = _world_corners([(-hx, -hy, hx, hy)], bin_row["position"][:2], bin_row["rpy"][2], table_z)[0]
        pixels = polyline(polygon, (255, 255, 255), thickness)
        cv2.putText(out, "BIN", tuple(np.round(pixels.mean(0)).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

    for index, obj in enumerate(layout["objects"]):
        color = SLOT_COLORS[index % len(SLOT_COLORS)]
        corners = []
        for polygon in _world_corners(obj["footprint_boxes"], obj["position"][:2], obj["yaw"], table_z):
            corners.append(polyline(polygon, color, thickness))
        center = projector.project(np.array([[*obj["position"][:2], table_z]]))[0] * scale
        cv2.circle(out, tuple(np.round(center).astype(int)), 3, color, -1, cv2.LINE_AA)
        top = np.concatenate(corners).min(axis=0)
        label = f"{index + 1}:{obj['name']}"
        cv2.putText(out, label, (int(top[0]), max(14, int(top[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(out, label, (int(top[0]), max(14, int(top[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    cv2.LINE_AA)
    return out
