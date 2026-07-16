"""Basic tabletop object placement for SO-101 Bench episodes.

Each object root is sampled uniformly from the bin-pose-specific spawn polygon
(``VALID_OBJECT_SPAWN_REGIONS`` in ``so101_bench_env_cfg.py``) and a placement is
accepted only when it satisfies the four shared initial-placement constraints:

1. the object root lies inside the spawn polygon for the chosen bin pose;
2. every object footprint first tries to be at least 2.5 in from every other
   object footprint, then falls back to 2.0 in and finally
   ``MIN_OBJECT_SURFACE_DISTANCE_M`` (1.5 in) if tighter packing is required
   (surface distance -- this also forbids object-object overlap);
3. every object footprint clears the plastic bin by at least
   ``MIN_BIN_SURFACE_DISTANCE_M`` (0.5 in, surface distance); and
4. no object footprint overlaps the SO-101 bounding box.

Object footprints are the exact raster decompositions loaded from
``assets/objects/<name>.json`` (a missing file raises); the plastic bin is the
only rectangular footprint.  ``TASK_BIN`` episodes may use any of the configured
bin poses; every other task family uses bin pose 0.

The sampler is a plain incremental rejection sampler: objects are placed
largest-first, each from a bounded number of random root/yaw draws, with bounded
whole-layout restarts.  Every acceptance test is exact polygon geometry,
accelerated only by axis-aligned bounding-box gates (a whole-footprint gate plus
a vectorised per-box gate) so the fine-grained rasters stay affordable.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import random
from typing import Any

import numpy as np

from so101_bench.benchmark import (
    INCH,
    TASK_BIN,
    BenchmarkEpisodeSpec,
    load_object_move_footprint_boxes,
)

# Surface-distance constraints (see module docstring).
MIN_BIN_SURFACE_DISTANCE_M = 0.5 * INCH
MIN_OBJECT_SURFACE_DISTANCE_M = 1.5 * INCH
OBJECT_SURFACE_DISTANCE_ATTEMPTS_M = (2.5 * INCH, 2.0 * INCH, MIN_OBJECT_SURFACE_DISTANCE_M)

# The plastic bin is the one object modelled by a rectangle rather than a raster.
PLASTIC_BIN_FOOTPRINT_SIZE_IN = (12.675, 7.25)
DEFAULT_BIN_FOOTPRINT_HALF_EXTENTS = (
    0.5 * PLASTIC_BIN_FOOTPRINT_SIZE_IN[0] * INCH,
    0.5 * PLASTIC_BIN_FOOTPRINT_SIZE_IN[1] * INCH,
)
# Fallback half-extents used by teleop tooling when an entry lacks a footprint.
DEFAULT_OBJECT_FOOTPRINT_HALF_EXTENTS = (0.02, 0.02)

DEFAULT_LAYOUT_MAX_ATTEMPTS = 48
DEFAULT_CANDIDATES_PER_OBJECT = 144
# Inter-episode placement diversity (blue-noise / "best-candidate" sampling):
# among the valid candidates found for each object, keep the one whose root is
# farthest from recently placed roots (this episode + a sliding window of prior
# episodes threaded via ``placement_history``).  ``1`` restores first-valid-wins.
DEFAULT_DIVERSITY_CANDIDATES = 8
DIVERSITY_HISTORY_WINDOW = 128
_EPS = 1.0e-9

Point2D = tuple[float, float]
Polygon2D = list[Point2D]
Aabb = tuple[float, float, float, float]


class LayoutGenerationError(RuntimeError):
    """Raised when a collision-free episode layout cannot be sampled."""


# --------------------------------------------------------------------------- #
# Exact polygon geometry (pure Python; general convex or concave polygons)
# --------------------------------------------------------------------------- #
def _xy_polygon(region: list[tuple[float, ...]] | tuple[tuple[float, ...], ...]) -> Polygon2D:
    polygon = [(float(point[0]), float(point[1])) for point in region]
    if len(polygon) < 3:
        raise ValueError(f"Expected a polygon with at least 3 points, got {len(polygon)}.")
    return polygon


def _box_pieces(center: Point2D, yaw: float, boxes: tuple[Aabb, ...]) -> list[Polygon2D]:
    """Rotate/translate each local ``(x0, y0, x1, y1)`` box into a world quad."""

    cx, cy = center
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    pieces: list[Polygon2D] = []
    for x0, y0, x1, y1 in boxes:
        pieces.append(
            [
                (cx + lx * cos_yaw - ly * sin_yaw, cy + lx * sin_yaw + ly * cos_yaw)
                for lx, ly in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            ]
        )
    return pieces


def _point_on_segment(point: Point2D, start: Point2D, end: Point2D) -> bool:
    px, py = point
    ax, ay = start
    bx, by = end
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > _EPS:
        return False
    return min(ax, bx) - _EPS <= px <= max(ax, bx) + _EPS and min(ay, by) - _EPS <= py <= max(ay, by) + _EPS


def _point_in_polygon(point: Point2D, polygon: Polygon2D) -> bool:
    inside = False
    px, py = point
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if _point_on_segment(point, start, end):
            return True
        ax, ay = start
        bx, by = end
        if (ay > py) != (by > py):
            x_at_y = (bx - ax) * (py - ay) / (by - ay) + ax
            if px < x_at_y:
                inside = not inside
    return inside


def _orientation(a: Point2D, b: Point2D, c: Point2D) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a: Point2D, b: Point2D, c: Point2D, d: Point2D) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if abs(o1) <= _EPS and _point_on_segment(c, a, b):
        return True
    if abs(o2) <= _EPS and _point_on_segment(d, a, b):
        return True
    if abs(o3) <= _EPS and _point_on_segment(a, c, d):
        return True
    if abs(o4) <= _EPS and _point_on_segment(b, c, d):
        return True
    return (o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0)


def _point_segment_distance(point: Point2D, start: Point2D, end: Point2D) -> float:
    px, py = point
    ax, ay = start
    dx = end[0] - ax
    dy = end[1] - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= _EPS:
        return math.dist(point, start)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.dist(point, (ax + t * dx, ay + t * dy))


def _polygons_overlap(first: Polygon2D, second: Polygon2D) -> bool:
    for i, first_start in enumerate(first):
        first_end = first[(i + 1) % len(first)]
        for j, second_start in enumerate(second):
            second_end = second[(j + 1) % len(second)]
            if _segments_intersect(first_start, first_end, second_start, second_end):
                return True
    return any(_point_in_polygon(point, second) for point in first) or any(
        _point_in_polygon(point, first) for point in second
    )


def _polygon_surface_distance(first: Polygon2D, second: Polygon2D) -> float:
    if _polygons_overlap(first, second):
        return 0.0
    min_distance = math.inf
    for point in first:
        for index, start in enumerate(second):
            min_distance = min(min_distance, _point_segment_distance(point, start, second[(index + 1) % len(second)]))
    for point in second:
        for index, start in enumerate(first):
            min_distance = min(min_distance, _point_segment_distance(point, start, first[(index + 1) % len(first)]))
    return min_distance


# --------------------------------------------------------------------------- #
# Axis-aligned bounding-box acceleration (numpy)
# --------------------------------------------------------------------------- #
def _piece_aabb(polygon: Polygon2D) -> Aabb:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def _pieces_aabbs(pieces: list[Polygon2D]) -> np.ndarray:
    if not pieces:
        return np.zeros((0, 4))
    return np.array([_piece_aabb(piece) for piece in pieces], dtype=float)


def _aabb_gap(first: Aabb, second: Aabb) -> float:
    """Euclidean gap between two AABBs (0 when they touch or overlap)."""

    dx = max(second[0] - first[2], first[0] - second[2], 0.0)
    dy = max(second[1] - first[3], first[1] - second[3], 0.0)
    return math.hypot(dx, dy)


def _aabb_gap_matrix(a_aabbs: np.ndarray, b_aabbs: np.ndarray) -> np.ndarray:
    """Pairwise AABB gaps between two ``(N, 4)`` / ``(M, 4)`` box sets -> ``(N, M)``."""

    dx = np.maximum(
        np.maximum(b_aabbs[None, :, 0] - a_aabbs[:, None, 2], a_aabbs[:, None, 0] - b_aabbs[None, :, 2]),
        0.0,
    )
    dy = np.maximum(
        np.maximum(b_aabbs[None, :, 1] - a_aabbs[:, None, 3], a_aabbs[:, None, 1] - b_aabbs[None, :, 3]),
        0.0,
    )
    return np.hypot(dx, dy)


def _corner_poly(corners: np.ndarray) -> Polygon2D:
    return [(float(x), float(y)) for x, y in corners]


# --------------------------------------------------------------------------- #
# Piece-set geometry (test-facing + reporting): exact, AABB-gated
# --------------------------------------------------------------------------- #
def _pieces_overlap(first: list[Polygon2D], second: list[Polygon2D]) -> bool:
    if not first or not second:
        return False
    gap = _aabb_gap_matrix(_pieces_aabbs(first), _pieces_aabbs(second))
    for i, j in zip(*np.nonzero(gap <= _EPS)):
        if _polygons_overlap(first[int(i)], second[int(j)]):
            return True
    return False


def _pieces_surface_distance(first: list[Polygon2D], second: list[Polygon2D]) -> float:
    """Exact minimum surface distance between two piece sets (0 if they overlap)."""

    if not first or not second:
        return math.inf
    gap = _aabb_gap_matrix(_pieces_aabbs(first), _pieces_aabbs(second))
    best = math.inf
    row_lower_bound = gap.min(axis=1)
    for i in np.argsort(row_lower_bound):
        if row_lower_bound[i] >= best:
            break
        row = gap[i]
        for j in np.argsort(row):
            if row[j] >= best:
                break
            distance = _polygon_surface_distance(first[int(i)], second[int(j)])
            if distance < best:
                best = distance
            if best <= _EPS:
                return 0.0
    return best


# --------------------------------------------------------------------------- #
# Footprint model (numpy corner arrays for the sampler's hot path)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Footprint:
    corners: np.ndarray  # (N, 4, 2) world-frame box corners
    aabbs: np.ndarray  # (N, 4) per-box [min_x, min_y, max_x, max_y]
    total: Aabb  # union AABB over every box


@lru_cache(maxsize=None)
def _object_footprint_boxes(object_name: str) -> tuple[Aabb, ...]:
    # ``required=True`` raises ValueError when the object has no footprint JSON.
    return load_object_move_footprint_boxes(object_name, required=True)


@lru_cache(maxsize=None)
def _object_local_corners(object_name: str) -> np.ndarray:
    boxes = np.array(_object_footprint_boxes(object_name), dtype=float)  # (N, 4)
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack(
        [
            np.stack([x0, y0], axis=1),
            np.stack([x1, y0], axis=1),
            np.stack([x1, y1], axis=1),
            np.stack([x0, y1], axis=1),
        ],
        axis=1,
    )  # (N, 4, 2)


def _rectangle_local_corners(half_extents: Point2D) -> np.ndarray:
    hx, hy = half_extents
    return np.array([[[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]], dtype=float)  # (1, 4, 2)


def _world_footprint(local_corners: np.ndarray, center: Point2D, yaw: float) -> _Footprint:
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    x = local_corners[..., 0]
    y = local_corners[..., 1]
    corners = np.stack([x * cos_yaw - y * sin_yaw + center[0], x * sin_yaw + y * cos_yaw + center[1]], axis=-1)
    mins = corners.min(axis=1)
    maxs = corners.max(axis=1)
    aabbs = np.concatenate([mins, maxs], axis=1)
    total = (float(mins[:, 0].min()), float(mins[:, 1].min()), float(maxs[:, 0].max()), float(maxs[:, 1].max()))
    return _Footprint(corners=corners, aabbs=aabbs, total=total)


def _footprint_pieces(footprint: _Footprint) -> list[Polygon2D]:
    return [_corner_poly(corner) for corner in footprint.corners]


def _footprints_clear(first: _Footprint, second: _Footprint, threshold: float) -> bool:
    """True when every box of ``first`` is >= ``threshold`` from every box of ``second``."""

    if _aabb_gap(first.total, second.total) >= threshold:
        return True
    gap = _aabb_gap_matrix(first.aabbs, second.aabbs)
    rows, cols = np.nonzero(gap < threshold)
    if rows.size == 0:
        return True
    for k in np.argsort(gap[rows, cols]):  # closest boxes first -> fastest rejection
        i, j = int(rows[k]), int(cols[k])
        if _polygon_surface_distance(_corner_poly(first.corners[i]), _corner_poly(second.corners[j])) < threshold:
            return False
    return True


def _footprint_overlaps_polygon(footprint: _Footprint, polygon: Polygon2D, polygon_aabb: Aabb) -> bool:
    if _aabb_gap(footprint.total, polygon_aabb) > 0.0:
        return False
    box_gaps = _aabb_gap_matrix(footprint.aabbs, np.array([polygon_aabb], dtype=float))[:, 0]
    for i in np.nonzero(box_gaps <= _EPS)[0]:
        if _polygons_overlap(_corner_poly(footprint.corners[int(i)]), polygon):
            return True
    return False


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _sample_point_in_polygon(polygon: Polygon2D, rng: random.Random) -> Point2D:
    x_min, y_min, x_max, y_max = _piece_aabb(polygon)
    for _ in range(1000):
        candidate = (rng.uniform(x_min, x_max), rng.uniform(y_min, y_max))
        if _point_in_polygon(candidate, polygon):
            return candidate
    raise LayoutGenerationError("Could not sample a point inside the valid object spawn polygon.")


def _boxes_extent(boxes: tuple[Aabb, ...]) -> float:
    xs0 = min(box[0] for box in boxes)
    ys0 = min(box[1] for box in boxes)
    xs1 = max(box[2] for box in boxes)
    ys1 = max(box[3] for box in boxes)
    return math.hypot(xs1 - xs0, ys1 - ys0)


@dataclass
class _PlacedObject:
    object_id: int
    name: str
    center: Point2D
    yaw: float
    footprint: _Footprint


def _most_spread(pool: list[_PlacedObject], reference: list[Point2D]) -> _PlacedObject:
    """Pick the pooled candidate whose root is farthest from the reference roots.

    ``reference`` holds recently placed roots (this episode plus a window of
    prior episodes); choosing the max-min-distance candidate yields blue-noise
    coverage across episodes.  With no reference the first candidate wins, which
    keeps single isolated calls deterministic and first-valid.
    """

    if len(pool) == 1 or not reference:
        return pool[0]
    ref = np.asarray(reference, dtype=float)
    best = pool[0]
    best_score = -math.inf
    for candidate in pool:
        score = float(np.hypot(ref[:, 0] - candidate.center[0], ref[:, 1] - candidate.center[1]).min())
        if score > best_score:
            best_score = score
            best = candidate
    return best


def _sample_layout(
    object_names: list[str],
    placement_order: list[int],
    spawn_polygon: Polygon2D,
    bin_footprint: _Footprint,
    robot_polygon: Polygon2D | None,
    robot_aabb: Aabb | None,
    bin_clearance_m: float,
    object_distance_m: float,
    rng: random.Random,
    max_attempts: int,
    candidates_per_object: int,
    history: list[Point2D],
    diversity_candidates: int,
) -> tuple[list[_PlacedObject], int, dict[str, int]]:
    """Seat every object by exact, AABB-gated rejection sampling (largest first).

    For each object we keep sampling until we collect up to ``diversity_candidates``
    valid placements (bounded so hard objects stay cheap), then choose the one
    that spreads best from ``history`` -- see :func:`_most_spread`.
    """

    rejection_counts: dict[str, int] = {}
    local_corners = [_object_local_corners(name) for name in object_names]
    diversity_candidates = max(int(diversity_candidates), 1)
    history_window = history[-DIVERSITY_HISTORY_WINDOW:] if history else []
    # A lone object cannot be helped by whole-layout restarts (nothing else to
    # rearrange), so it gets a single attempt with the full candidate budget.
    attempt_budget = 1 if len(placement_order) == 1 else max_attempts
    # Stop enlarging an object's candidate pool once we are this many draws past
    # the first valid one, so filling the pool never balloons a hard object.
    extra_after_first = 2 * diversity_candidates

    def bump(reason: str) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    def fits(candidate: _Footprint, placed: list[_PlacedObject]) -> bool:
        if robot_polygon is not None and robot_aabb is not None and _footprint_overlaps_polygon(
            candidate, robot_polygon, robot_aabb
        ):
            bump("robot_overlap")
            return False
        if not _footprints_clear(candidate, bin_footprint, bin_clearance_m):
            bump("bin_clearance")
            return False
        for other in placed:
            if not _footprints_clear(candidate, other.footprint, object_distance_m):
                bump("object_distance")
                return False
        return True

    for attempt in range(1, attempt_budget + 1):
        # Diversity is best-effort: only the first whole-layout attempt shops for a
        # well-spread anchor.  If that anchor leaves a tight multi-object layout
        # unplaceable, later attempts fall back to plain first-valid (which packs
        # reliably) rather than restarting into the same crowded-out corner.
        use_diversity = attempt == 1
        placed: list[_PlacedObject] = []
        for order_index, object_id in enumerate(placement_order):
            # Only the anchor (largest, placed first) shops for a well-spread root:
            # that relocates the whole cluster across episodes.  Spreading later
            # objects too would push a tight multi-object layout to the region
            # edges and make it unplaceable, so they take the first valid draw.
            reference = history_window if (use_diversity and order_index == 0) else []
            # Only bother collecting a candidate pool when there is history to
            # spread away from; otherwise the first valid draw wins immediately.
            target_pool = diversity_candidates if reference else 1
            pool: list[_PlacedObject] = []
            first_valid_draw: int | None = None
            for draw in range(max(int(candidates_per_object), 1)):
                center = _sample_point_in_polygon(spawn_polygon, rng)
                yaw = rng.uniform(-math.pi, math.pi)
                footprint = _world_footprint(local_corners[object_id], center, yaw)
                if fits(footprint, placed):
                    pool.append(_PlacedObject(object_id, object_names[object_id], center, yaw, footprint))
                    if first_valid_draw is None:
                        first_valid_draw = draw
                    if len(pool) >= target_pool:
                        break
                if first_valid_draw is not None and draw - first_valid_draw >= extra_after_first:
                    break
            if not pool:
                bump("object_unplaced")
                break
            placed.append(_most_spread(pool, reference))
        if len(placed) == len(placement_order):
            placed.sort(key=lambda obj: obj.object_id)
            return placed, attempt, rejection_counts

    raise LayoutGenerationError(
        f"Could not place {len(placement_order)} object footprint(s) with "
        f">={bin_clearance_m:.5f} m bin clearance and >={object_distance_m:.5f} m "
        f"object spacing after {attempt_budget} attempt(s)."
    )


# --------------------------------------------------------------------------- #
# Layout assembly
# --------------------------------------------------------------------------- #
def _object_layout_entry(placed: _PlacedObject, table_object_z: float) -> dict[str, Any]:
    boxes = _object_footprint_boxes(placed.name)
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    return {
        "slot": placed.object_id,
        "asset_name": f"object_{placed.object_id + 1}",
        "name": placed.name,
        "position": [placed.center[0], placed.center[1], float(table_object_z)],
        "yaw": placed.yaw,
        "rpy": [0.0, 0.0, placed.yaw],
        "footprint_boxes": [list(box) for box in boxes],
        "footprint_half_extents": [0.5 * (x1 - x0), 0.5 * (y1 - y0)],
        "footprint_center_offset": [0.5 * (x0 + x1), 0.5 * (y0 + y1)],
    }


def _min_object_surface_distance(placed: list[_PlacedObject]) -> float | None:
    if len(placed) < 2:
        return None
    pieces = [_footprint_pieces(obj.footprint) for obj in placed]
    best = math.inf
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            best = min(best, _pieces_surface_distance(pieces[i], pieces[j]))
    return best


def _object_surface_distance_attempts(min_object_distance_m: float) -> list[float]:
    """Preferred object spacing attempts, ending at the requested minimum floor."""

    min_object_distance_m = max(float(min_object_distance_m), 0.0)
    distances = [
        distance
        for distance in OBJECT_SURFACE_DISTANCE_ATTEMPTS_M
        if distance + _EPS >= min_object_distance_m
    ]
    if not any(math.isclose(distance, min_object_distance_m, abs_tol=_EPS) for distance in distances):
        distances.append(min_object_distance_m)
    return distances


def generate_episode_layout(
    episode: BenchmarkEpisodeSpec,
    *,
    episode_index: int,
    rng: random.Random,
    bin_random_poses: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...],
    valid_spawn_regions: list[list[tuple[float, float, float]]],
    table_object_z: float,
    seed: int,
    generated_at: str,
    robot_bounding_box: list[tuple[float, ...]] | tuple[tuple[float, ...], ...] | None = None,
    bin_clearance_m: float = MIN_BIN_SURFACE_DISTANCE_M,
    min_object_distance_m: float = MIN_OBJECT_SURFACE_DISTANCE_M,
    max_attempts: int = DEFAULT_LAYOUT_MAX_ATTEMPTS,
    candidates_per_object: int = DEFAULT_CANDIDATES_PER_OBJECT,
    placement_history: list[Point2D] | None = None,
    diversity_candidates: int = DEFAULT_DIVERSITY_CANDIDATES,
) -> dict[str, Any]:
    """Sample a replayable initial scene layout for one episode.

    Pass a shared mutable ``placement_history`` list across a generation run to
    spread object roots away from recent episodes (high inter-episode variance);
    accepted roots are appended to it.  ``diversity_candidates`` controls how
    hard each object shops for a well-spread placement (1 = first-valid-wins).
    """

    if not bin_random_poses:
        raise ValueError("Expected at least one bin pose.")
    if len(bin_random_poses) != len(valid_spawn_regions):
        raise ValueError(
            f"Expected one valid object spawn region per bin pose, got "
            f"{len(valid_spawn_regions)} regions for {len(bin_random_poses)} bin poses."
        )
    if not episode.objects:
        raise ValueError("Cannot generate a layout for an episode with no objects.")

    bin_clearance_m = max(float(bin_clearance_m), 0.0)
    object_distance_attempts_m = _object_surface_distance_attempts(min_object_distance_m)
    history = placement_history if placement_history is not None else []
    object_names = list(episode.objects)
    object_boxes = [_object_footprint_boxes(name) for name in object_names]  # raises on missing JSON
    placement_order = sorted(
        range(len(object_names)),
        key=lambda object_id: (_boxes_extent(object_boxes[object_id]), object_id),
        reverse=True,
    )

    robot_polygon = _xy_polygon(robot_bounding_box) if robot_bounding_box is not None else None
    robot_aabb = _piece_aabb(robot_polygon) if robot_polygon is not None else None
    bin_local_corners = _rectangle_local_corners(DEFAULT_BIN_FOOTPRINT_HALF_EXTENTS)

    # Bin tasks may try any configured bin pose; every other family uses pose 0.
    pose_order = list(range(len(bin_random_poses))) if episode.task_family == TASK_BIN else [0]
    rng.shuffle(pose_order)

    attempted_pose_indices: list[int] = []
    attempted_object_distances_m: list[float] = []
    last_error: LayoutGenerationError | None = None
    for object_distance_m in object_distance_attempts_m:
        attempted_object_distances_m.append(object_distance_m)
        for bin_pose_index in pose_order:
            attempted_pose_indices.append(bin_pose_index)
            bin_translation, bin_rpy = bin_random_poses[bin_pose_index]
            bin_footprint = _world_footprint(
                bin_local_corners,
                (float(bin_translation[0]), float(bin_translation[1])),
                float(bin_rpy[2]),
            )
            spawn_polygon = _xy_polygon(valid_spawn_regions[bin_pose_index])

            try:
                placed, attempts, rejection_counts = _sample_layout(
                    object_names,
                    placement_order,
                    spawn_polygon,
                    bin_footprint,
                    robot_polygon,
                    robot_aabb,
                    bin_clearance_m,
                    object_distance_m,
                    rng,
                    max_attempts,
                    candidates_per_object,
                    history,
                    diversity_candidates,
                )
            except LayoutGenerationError as exc:
                last_error = exc
                continue

            # Record accepted roots so later episodes in the run spread away from them.
            if placement_history is not None:
                placement_history.extend(obj.center for obj in placed)

            bin_pieces = _footprint_pieces(bin_footprint)
            min_bin_distance = min(
                _pieces_surface_distance(_footprint_pieces(obj.footprint), bin_pieces) for obj in placed
            )
            min_object_distance = _min_object_surface_distance(placed)
            metadata = dict(getattr(episode, "metadata", None) or {})
            return {
                "trial_id": metadata.get("trial_id", episode_index),
                "episode_index": episode_index,
                "seed": seed,
                "generated_at": generated_at,
                "task_family": episode.task_family,
                "instruction": episode.instruction,
                "objects": [_object_layout_entry(obj, table_object_z) for obj in placed],
                "bin": {
                    "pose_index": bin_pose_index,
                    "position": [float(bin_translation[0]), float(bin_translation[1]), float(bin_translation[2])],
                    "rpy": [float(bin_rpy[0]), float(bin_rpy[1]), float(bin_rpy[2])],
                    "rpy_deg": [math.degrees(float(angle)) for angle in bin_rpy],
                    "footprint_half_extents": [
                        DEFAULT_BIN_FOOTPRINT_HALF_EXTENTS[0],
                        DEFAULT_BIN_FOOTPRINT_HALF_EXTENTS[1],
                    ],
                    "footprint_center_offset": [0.0, 0.0],
                },
                "placement": {
                    "layout_selection": "basic_constraints",
                    "attempts": attempts,
                    "valid_attempts": 1,
                    "bin_pose_attempts": len(attempted_pose_indices),
                    "object_surface_distance_attempts_m": attempted_object_distances_m.copy(),
                    "valid_spawn_region_index": bin_pose_index,
                    "required_min_object_surface_distance_m": object_distance_m,
                    "min_object_surface_distance_m": min_object_distance,
                    "min_between_object_surface_distance_m": min_object_distance,
                    "min_bin_surface_distance_m": min_bin_distance,
                    "required_min_bin_surface_distance_m": bin_clearance_m,
                    "rejection_counts": rejection_counts,
                    "task_feasibility": None,
                },
            }

    raise LayoutGenerationError(
        f"Could not sample a valid layout for episode {episode_index} across "
        f"{len(attempted_pose_indices)} bin pose attempt(s) for object spacing "
        f"attempts {[round(distance, 5) for distance in attempted_object_distances_m]} m. "
        f"Last error: {last_error}"
    )


def normalize_layout_object_slots(
    layout: dict[str, Any],
    object_names_by_slot: list[str] | tuple[str, ...],
    *,
    episode_index: int | None = None,
) -> dict[str, Any]:
    """Remap named layout entries to the episode's semantic object slots."""

    object_entries = layout.get("objects")
    if not isinstance(object_entries, list):
        return layout

    named_entries = [entry for entry in object_entries if isinstance(entry, dict) and entry.get("name")]
    if not named_entries:
        return layout

    entry_names = [str(entry["name"]) for entry in named_entries]
    prefix = "Layout"
    if episode_index is not None:
        prefix = f"Layout for episode {episode_index}"
    if len(set(entry_names)) != len(entry_names):
        raise ValueError(f"{prefix} has duplicate object names: {entry_names}.")
    if set(entry_names) != set(object_names_by_slot):
        raise ValueError(
            f"{prefix} does not match episode objects. Expected {list(object_names_by_slot)}, got {entry_names}."
        )

    slot_by_name = {object_name: object_id for object_id, object_name in enumerate(object_names_by_slot)}
    normalized_entries = []
    for entry in object_entries:
        if not isinstance(entry, dict):
            normalized_entries.append(entry)
            continue
        normalized_entry = dict(entry)
        object_name = str(normalized_entry.get("name", ""))
        if object_name:
            slot = slot_by_name[object_name]
            normalized_entry["slot"] = slot
            normalized_entry["asset_name"] = f"object_{slot + 1}"
        normalized_entries.append(normalized_entry)

    normalized_layout = dict(layout)
    normalized_layout["objects"] = sorted(
        normalized_entries,
        key=lambda entry: int(entry.get("slot", 0)) if isinstance(entry, dict) else 0,
    )
    return normalized_layout
