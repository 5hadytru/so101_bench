"""Success and failure termination terms for SO-101 Bench."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from so101_bench.benchmark import (
    BETWEEN_LINE_TOLERANCE_M,
    BIN_DISPLACEMENT_LIMIT_M,
    BOUNDARY_DISPLACEMENT_LIMIT_M,
    DIRECTIONS,
    GRASP_ATTEMPT_OBJECT_DISTANCE_M,
    LIFT_OFF_GROUND_LIMIT_M,
    MOVE_BOUNDARY_MIN_LATERAL_OVERLAP_FRACTION,
    MOVE_BOUNDARY_SUCCESS_DISTANCE_M,
    MOVE_NO_BOUNDARY_MIN_PROGRESS_M,
    MOVE_PAST_BOUNDARY_TOLERANCE_M,
    MOVE_STRAIGHTNESS_TOLERANCE_M,
    NON_TARGET_DISPLACEMENT_LIMIT_M,
    SPATIAL_SUCCESS_DISTANCE_M,
    TASK_BETWEEN,
    TASK_BIN,
    TASK_MOVE,
    TASK_NEXT_TO,
    episode_length_s,
)
from .resets import benchmark_object_positions, benchmark_object_yaws, mark_benchmark_robot_start

FAILURE_REASON_NONE = "none"
FAILURE_REASON_MAX_GRASP_ATTEMPTS = "max_grasp_attempts"
FAILURE_REASON_BIN_DISPLACED = "bin_displaced"
FAILURE_REASON_NON_TARGET_MOVED = "non_target_moved"
FAILURE_REASON_MOVE_BOUNDARY_MOVED = "move_boundary_moved"
FAILURE_REASON_MOVE_PAST_BOUNDARY = "move_past_boundary"
FAILURE_REASON_MOVE_TRAJECTORY_NOT_STRAIGHT_ENOUGH = "move_trajectory_not_straight_enough"
FAILURE_REASON_MADE_CONTACT = "made_contact"
FAILURE_REASON_SUCCESS_CONFIRMATION_BREACHED = "success_confirmation_breached"

# Postmortem failure types: a single mutually-exclusive label assigned to every
# non-bin episode at the end of the rollout, based on whether the target (and/or a
# distractor) was ever lifted clear of the table. Unlike the live FAILURE_REASON_*
# rules above, these are computed once at episode end and apply even when the
# episode merely timed out without tripping a live failure term.
POSTMORTEM_NONE = "none"
POSTMORTEM_NOT_APPLICABLE = "not_applicable"
POSTMORTEM_SEMANTIC = "semantic"
POSTMORTEM_FAILED_GRASP = "failed_grasp"
POSTMORTEM_PLACEMENT = "placement"
POSTMORTEM_FAILURE_TYPES = (POSTMORTEM_SEMANTIC, POSTMORTEM_FAILED_GRASP, POSTMORTEM_PLACEMENT)

DEFAULT_SUCCESS_CONFIRM_TIME_S = 3.0
DEFAULT_FAILURE_CONFIRM_TIME_S = 5.0
DEFAULT_CONTACT_GRACE_TIME_S = 5.0
DEFAULT_MOVE_STRAIGHTNESS_FAILURE_CONFIRM_TIME_S = 5.0
DEFAULT_MOVE_PAST_BOUNDARY_FAILURE_CONFIRM_TIME_S = 5.0


@dataclass
class _TerminationStepState:
    positions: torch.Tensor
    yaws: torch.Tensor | None = None
    footprint_vertices: torch.Tensor | None = None
    grasped_object_made_contact: torch.Tensor | None = None


@dataclass(frozen=True)
class TaskConditionDiagnostic:
    """One human-readable benchmark condition status."""

    kind: str
    name: str
    met: bool
    details: str


@dataclass(frozen=True)
class TaskDiagnostics:
    """Condition statuses for one benchmark environment."""

    env_id: int
    task_family: str
    episode_age_s: float
    conditions: tuple[TaskConditionDiagnostic, ...]


def _active_mask(env: ManagerBasedRLEnv, object_asset_names: list[str]) -> torch.Tensor:
    if hasattr(env, "_so101_active_object_mask"):
        return env._so101_active_object_mask
    return torch.ones((env.num_envs, len(object_asset_names)), dtype=torch.bool, device=env.device)


def _object_positions(env: ManagerBasedRLEnv, object_asset_names: list[str]) -> torch.Tensor:
    return benchmark_object_positions(env, object_asset_names)


def _termination_step_state(env: ManagerBasedRLEnv, object_asset_names: list[str]) -> _TerminationStepState:
    """Return lazy object state shared by termination terms in the current step."""

    step_counter = getattr(env, "common_step_counter", None)
    cache_key = None
    if isinstance(step_counter, int):
        cache_key = (step_counter, id(_active_mask(env, object_asset_names)), tuple(object_asset_names))
        cached = getattr(env, "_so101_termination_step_state_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]

    state = _TerminationStepState(positions=_object_positions(env, object_asset_names))
    if cache_key is not None:
        env._so101_termination_step_state_cache = (cache_key, state)
    return state


def _object_half_extents(env: ManagerBasedRLEnv, object_asset_names: list[str]) -> torch.Tensor:
    if hasattr(env, "_so101_object_half_extents"):
        return env._so101_object_half_extents
    fallback = torch.full(
        (env.num_envs, len(object_asset_names), 3),
        0.02,
        dtype=torch.float32,
        device=env.device,
    )
    return fallback


def _bin_half_extents(env: ManagerBasedRLEnv) -> torch.Tensor:
    if hasattr(env, "_so101_bin_half_extents"):
        return env._so101_bin_half_extents
    return torch.tensor((0.125, 0.095, 0.08), dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)


def _object_footprint_half_extents(env: ManagerBasedRLEnv, object_asset_names: list[str]) -> torch.Tensor:
    if hasattr(env, "_so101_object_footprint_half_extents"):
        return env._so101_object_footprint_half_extents
    return _object_half_extents(env, object_asset_names)[..., :2]


def _object_footprint_center_offsets(env: ManagerBasedRLEnv, object_asset_names: list[str]) -> torch.Tensor:
    if hasattr(env, "_so101_object_footprint_center_offsets"):
        return env._so101_object_footprint_center_offsets
    return torch.zeros_like(_object_footprint_half_extents(env, object_asset_names))


def _bin_footprint_half_extents(env: ManagerBasedRLEnv) -> torch.Tensor:
    if hasattr(env, "_so101_bin_footprint_half_extents"):
        return env._so101_bin_footprint_half_extents
    return _bin_half_extents(env)[..., :2]


def _bin_footprint_center_offsets(env: ManagerBasedRLEnv) -> torch.Tensor:
    if hasattr(env, "_so101_bin_footprint_center_offsets"):
        return env._so101_bin_footprint_center_offsets
    return torch.zeros_like(_bin_footprint_half_extents(env))


def _task_is(env: ManagerBasedRLEnv, task_family: str) -> torch.Tensor:
    if not hasattr(env, "_so101_task_family"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return torch.tensor([task == task_family for task in env._so101_task_family], dtype=torch.bool, device=env.device)


def _target_indices(env: ManagerBasedRLEnv) -> torch.Tensor:
    if hasattr(env, "_so101_target_object_ids"):
        return env._so101_target_object_ids
    return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def _referent_indices(env: ManagerBasedRLEnv) -> torch.Tensor:
    if hasattr(env, "_so101_referent_object_ids"):
        return env._so101_referent_object_ids
    return torch.zeros((env.num_envs, 2), dtype=torch.long, device=env.device)


def _direction_indices(env: ManagerBasedRLEnv) -> torch.Tensor:
    if hasattr(env, "_so101_direction_ids"):
        return env._so101_direction_ids
    return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def _gather_by_index(values: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return values[torch.arange(values.shape[0], device=values.device), ids]


def _state_object_yaws(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    step_state: _TerminationStepState,
) -> torch.Tensor:
    if step_state.yaws is None:
        step_state.yaws = benchmark_object_yaws(env, object_asset_names)
    return step_state.yaws


def _state_footprint_vertices(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    step_state: _TerminationStepState,
) -> torch.Tensor:
    if step_state.footprint_vertices is None:
        step_state.footprint_vertices = _footprint_vertices_xy(
            step_state.positions[..., :2],
            _object_footprint_half_extents(env, object_asset_names),
            _object_footprint_center_offsets(env, object_asset_names),
            _state_object_yaws(env, object_asset_names, step_state),
        )
    return step_state.footprint_vertices


def grasped_object_made_contact(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    step_state: _TerminationStepState | None = None,
    force_threshold: float = 0.0,
) -> torch.Tensor:
    """Return whether each episode's currently grasped object contacts another tabletop object."""

    if step_state is None:
        step_state = _termination_step_state(env, object_asset_names)
    if step_state.grasped_object_made_contact is not None:
        return step_state.grasped_object_made_contact

    override = getattr(env, "_so101_grasped_object_made_contact_override", None)
    if override is not None:
        step_state.grasped_object_made_contact = override.to(device=env.device, dtype=torch.bool)
        return step_state.grasped_object_made_contact

    contact_by_object = torch.zeros(
        (env.num_envs, len(object_asset_names)),
        dtype=torch.bool,
        device=env.device,
    )
    sensors = getattr(env.scene, "sensors", {})
    for object_id, asset_name in enumerate(object_asset_names):
        exact_sensor_name = f"{asset_name}_contacts"
        split_sensor_prefix = f"{asset_name}_"
        for sensor_name, sensor in sensors.items():
            if sensor_name != exact_sensor_name and not (
                sensor_name.startswith(split_sensor_prefix) and sensor_name.endswith("_contacts")
            ):
                continue
            force_matrix_w = sensor.data.force_matrix_w
            if force_matrix_w is None:
                continue
            force_magnitudes = torch.linalg.vector_norm(force_matrix_w, dim=-1)
            contact_by_object[:, object_id] |= torch.any(
                force_magnitudes > force_threshold,
                dim=tuple(range(1, force_magnitudes.ndim)),
            )

    grasped_object_ids = getattr(env, "_so101_grasped_object_ids", None)
    if grasped_object_ids is None:
        step_state.grasped_object_made_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return step_state.grasped_object_made_contact

    has_grasped_object = grasped_object_ids >= 0
    safe_object_ids = torch.clamp(grasped_object_ids, min=0)
    step_state.grasped_object_made_contact = (
        _gather_by_index(contact_by_object, safe_object_ids) & has_grasped_object
    )
    return step_state.grasped_object_made_contact


def grasped_object_contact_exceeded_grace_period(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    step_state: _TerminationStepState | None = None,
    grace_time_s: float = DEFAULT_CONTACT_GRACE_TIME_S,
) -> torch.Tensor:
    """Return whether uninterrupted grasped-object contact has exceeded the allowed duration."""

    if step_state is None:
        step_state = _termination_step_state(env, object_asset_names)
    made_contact = grasped_object_made_contact(env, object_asset_names, step_state)
    if not hasattr(env, "_so101_grasped_object_contact_steps"):
        env._so101_grasped_object_contact_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_so101_grasped_object_contact_last_episode_steps"):
        env._so101_grasped_object_contact_last_episode_steps = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )

    episode_steps = env.episode_length_buf.to(device=env.device, dtype=torch.long)
    needs_update = env._so101_grasped_object_contact_last_episode_steps != episode_steps
    updated_steps = torch.where(
        made_contact,
        env._so101_grasped_object_contact_steps + 1,
        torch.zeros_like(env._so101_grasped_object_contact_steps),
    )
    env._so101_grasped_object_contact_steps = torch.where(
        needs_update,
        updated_steps,
        env._so101_grasped_object_contact_steps,
    )
    env._so101_grasped_object_contact_last_episode_steps = torch.where(
        needs_update,
        episode_steps,
        env._so101_grasped_object_contact_last_episode_steps,
    )
    return env._so101_grasped_object_contact_steps.to(dtype=torch.float32) * _env_step_dt(env) > grace_time_s


def _attempt_object_mask(env: ManagerBasedRLEnv, object_asset_names: list[str]) -> torch.Tensor:
    active = _active_mask(env, object_asset_names)
    target_mask = torch.zeros_like(active)
    target_mask[torch.arange(env.num_envs, device=env.device), _target_indices(env)] = True
    return torch.where(_task_is(env, TASK_BIN).unsqueeze(1), active, active & target_mask)


def _env_step_dt(env: ManagerBasedRLEnv) -> float:
    step_dt = getattr(env, "step_dt", None)
    if step_dt is None:
        step_dt = env.cfg.sim.dt * env.cfg.decimation
    return float(step_dt)


def _episode_age_s(env: ManagerBasedRLEnv) -> torch.Tensor:
    step_dt = _env_step_dt(env)
    return env.episode_length_buf.to(dtype=torch.float32) * step_dt


def _episode_age_at_least(env: ManagerBasedRLEnv, seconds: float) -> torch.Tensor:
    return _episode_age_s(env) >= seconds


def _task_success_counters(env: ManagerBasedRLEnv) -> torch.Tensor:
    counters = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    for task_family, counter_name in (
        (TASK_BIN, "_so101_bin_success_counter"),
        (TASK_NEXT_TO, "_so101_next_to_success_counter"),
        (TASK_BETWEEN, "_so101_between_success_counter"),
        (TASK_MOVE, "_so101_move_success_counter"),
    ):
        task_counters = getattr(env, counter_name, None)
        if task_counters is not None:
            counters = torch.where(_task_is(env, task_family), task_counters, counters)
    return counters


def task_time_out(
    env: ManagerBasedRLEnv,
    confirm_time_s: float = DEFAULT_SUCCESS_CONFIRM_TIME_S,
) -> torch.Tensor:
    """Time out episodes unless an in-progress success confirmation window remains intact."""

    active_mask = getattr(env, "_so101_active_object_mask", None)
    if active_mask is None:
        active_counts = [1] * env.num_envs
    else:
        active_counts = active_mask.sum(dim=1).tolist()
    task_families = getattr(env, "_so101_task_family", [TASK_BIN] * env.num_envs)
    timeouts = torch.tensor(
        [
            episode_length_s(task_family, int(active_count))
            for task_family, active_count in zip(task_families, active_counts, strict=True)
        ],
        dtype=torch.float32,
        device=env.device,
    )
    nominal_time_out = _episode_age_s(env) >= timeouts
    success_counters = _task_success_counters(env)
    confirmation_pending = (success_counters > 0) & (
        success_counters < _confirmation_steps(env, confirm_time_s)
    )

    extension_active = getattr(env, "_so101_timeout_success_confirmation_active", None)
    if extension_active is None:
        extension_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    extension_active |= nominal_time_out & confirmation_pending
    env._so101_timeout_success_confirmation_active = extension_active
    env._so101_timeout_success_confirmation_failed = extension_active & (success_counters == 0)

    # A confirmed success is handled by the success term in the same manager pass.
    return nominal_time_out & (~extension_active) & (success_counters == 0)


def _confirmation_steps(
    env: ManagerBasedRLEnv,
    confirm_time_s: float,
    confirm_steps: int | None = None,
) -> int:
    if confirm_steps is not None:
        return confirm_steps
    return max(1, math.ceil(confirm_time_s / _env_step_dt(env)))


def _held_failure(
    env: ManagerBasedRLEnv,
    counter_attr: str,
    instant: torch.Tensor,
    confirm_time_s: float,
) -> torch.Tensor:
    """Gate an instantaneous failure mask behind a continuous-hold confirmation window.

    The per-env counter stored on ``env`` as ``counter_attr`` increments while ``instant``
    is set and resets the moment it clears, so only a deviation that *settles* for the
    confirmation window -- not a transient swing that recovers -- latches as a failure.
    """
    counter = getattr(env, counter_attr, None)
    if counter is None:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    counter = torch.where(instant, counter + 1, torch.zeros_like(counter))
    setattr(env, counter_attr, counter)
    return (counter >= _confirmation_steps(env, confirm_time_s)) & instant


def _grasped_object_contact_exceeded_from_counter(
    env: ManagerBasedRLEnv,
    grace_time_s: float,
) -> torch.Tensor:
    contact_steps = getattr(env, "_so101_grasped_object_contact_steps", None)
    if contact_steps is None:
        contact_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return contact_steps.to(dtype=torch.float32) * _env_step_dt(env) > grace_time_s


def _grasped_object_contact_allows_success(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    step_state: _TerminationStepState,
    grace_time_s: float,
) -> torch.Tensor:
    """Advance the contact timer while requiring separation for success."""

    grasped_object_contact_exceeded_grace_period(env, object_asset_names, step_state, grace_time_s)
    return ~grasped_object_made_contact(env, object_asset_names, step_state)


def bin_success(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    bin_name: str,
    confirm_steps: int | None = None,
    confirm_time_s: float = DEFAULT_SUCCESS_CONFIRM_TIME_S,
    step_state: _TerminationStepState | None = None,
) -> torch.Tensor:
    """Success when every active object's root sits inside the bin's outer XY footprint.

    Containment is a pure XY test: each object's root position is transformed
    into the bin's frame (so the check stays correct even if the bin is yawed)
    and compared against the USD-derived bin footprint. The object's own
    footprint and height are intentionally ignored -- only the root center
    must land in the box.
    """

    if step_state is None:
        step_state = _termination_step_state(env, object_asset_names)
    object_pos_w = step_state.positions
    active = _active_mask(env, object_asset_names)
    bin_asset: RigidObject = env.scene[bin_name]
    bin_pos_w = bin_asset.data.root_pos_w
    bin_quat_inv = math_utils.quat_inv(bin_asset.data.root_quat_w)

    rel = object_pos_w - bin_pos_w.unsqueeze(1)
    rel_local = torch.stack(
        [math_utils.quat_apply(bin_quat_inv, rel[:, object_id, :]) for object_id in range(rel.shape[1])],
        dim=1,
    )

    footprint_half_extents = _bin_footprint_half_extents(env)
    footprint_center_offsets = _bin_footprint_center_offsets(env)
    rel_footprint = rel_local[..., :2] - footprint_center_offsets.unsqueeze(1)
    inside = torch.all(torch.abs(rel_footprint) <= footprint_half_extents.unsqueeze(1), dim=-1)
    all_active_inside = torch.all(torch.where(active, inside, torch.ones_like(inside)), dim=1)
    success_now = all_active_inside & _task_is(env, TASK_BIN)

    if not hasattr(env, "_so101_bin_success_counter"):
        env._so101_bin_success_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._so101_bin_success_counter = torch.where(
        success_now,
        env._so101_bin_success_counter + 1,
        torch.zeros_like(env._so101_bin_success_counter),
    )
    return env._so101_bin_success_counter >= _confirmation_steps(env, confirm_time_s, confirm_steps)


def next_to_success(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    success_distance: float = SPATIAL_SUCCESS_DISTANCE_M,
    contact_grace_time_s: float = DEFAULT_CONTACT_GRACE_TIME_S,
    confirm_steps: int | None = None,
    confirm_time_s: float = DEFAULT_SUCCESS_CONFIRM_TIME_S,
    step_state: _TerminationStepState | None = None,
) -> torch.Tensor:
    """Success for ``Place object 1 next to object 2``."""

    if step_state is None:
        step_state = _termination_step_state(env, object_asset_names)
    positions = step_state.positions
    yaws = _state_object_yaws(env, object_asset_names, step_state)
    is_next_to = _task_is(env, TASK_NEXT_TO)
    surface_distance = _pairwise_object_surface_distance(
        env,
        object_asset_names,
        positions,
        yaws,
        _target_indices(env),
        _referent_indices(env)[:, 0],
        is_next_to,
    )
    success_now = (surface_distance <= success_distance) & _grasped_object_contact_allows_success(
        env, object_asset_names, step_state, contact_grace_time_s
    )
    success_now &= is_next_to

    if not hasattr(env, "_so101_next_to_success_counter"):
        env._so101_next_to_success_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._so101_next_to_success_counter = torch.where(
        success_now,
        env._so101_next_to_success_counter + 1,
        torch.zeros_like(env._so101_next_to_success_counter),
    )
    return env._so101_next_to_success_counter >= _confirmation_steps(env, confirm_time_s, confirm_steps)


def between_success(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    centered_tolerance: float = BETWEEN_LINE_TOLERANCE_M,
    min_segment_fraction: float = 0.1,
    contact_grace_time_s: float = DEFAULT_CONTACT_GRACE_TIME_S,
    confirm_steps: int | None = None,
    confirm_time_s: float = DEFAULT_SUCCESS_CONFIRM_TIME_S,
    step_state: _TerminationStepState | None = None,
) -> torch.Tensor:
    """Success for ``Place object 1 between object 2 and object 3``."""

    if step_state is None:
        step_state = _termination_step_state(env, object_asset_names)
    positions = step_state.positions
    yaws = _state_object_yaws(env, object_asset_names, step_state)
    is_between = _task_is(env, TASK_BETWEEN)
    target_ids = _target_indices(env)
    refs = _referent_indices(env)

    # Centeredness is judged purely from how the target's footprint surface distance to
    # one referent compares with its distance to the other -- d1 / (d1 + d2), where 0.5
    # is equidistant. The [min_segment_fraction, 1 - min_segment_fraction] band rejects a
    # target that hugs one referent, so the referent-to-referent line is not used here.
    distance_to_first = _pairwise_object_surface_distance(
        env, object_asset_names, positions, yaws, target_ids, refs[:, 0], is_between
    )
    distance_to_second = _pairwise_object_surface_distance(
        env, object_asset_names, positions, yaws, target_ids, refs[:, 1], is_between
    )
    total_distance = distance_to_first + distance_to_second
    fraction = torch.where(
        torch.isfinite(total_distance) & (total_distance > 0.0),
        distance_to_first / total_distance,
        torch.full_like(distance_to_first, 0.5),
    )
    centered = (fraction >= min_segment_fraction) & (fraction <= 1.0 - min_segment_fraction)

    # The target must also lie on the line between the two referents, judged from the
    # target's root center (not its footprint surface).
    target = _gather_by_index(positions, target_ids)[:, :2]
    ref_a = _gather_by_index(positions, refs[:, 0])[:, :2]
    ref_b = _gather_by_index(positions, refs[:, 1])[:, :2]
    segment = ref_b - ref_a
    segment_len_sq = torch.clamp(torch.sum(segment * segment, dim=1), min=1.0e-6)
    t = torch.sum((target - ref_a) * segment, dim=1) / segment_len_sq
    projection = ref_a + t.unsqueeze(1) * segment
    perpendicular = torch.linalg.vector_norm(target - projection, dim=1)

    success_now = centered & (perpendicular <= centered_tolerance) & _grasped_object_contact_allows_success(
        env, object_asset_names, step_state, contact_grace_time_s
    )
    success_now &= is_between

    if not hasattr(env, "_so101_between_success_counter"):
        env._so101_between_success_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._so101_between_success_counter = torch.where(
        success_now,
        env._so101_between_success_counter + 1,
        torch.zeros_like(env._so101_between_success_counter),
    )
    return env._so101_between_success_counter >= _confirmation_steps(env, confirm_time_s, confirm_steps)


def _direction_vectors(device: str) -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.0],  # left
            [-1.0, 0.0],  # right
            [0.0, -1.0],  # forward
            [0.0, 1.0],  # backward
        ],
        dtype=torch.float32,
        device=device,
    )


def _direction_axis_and_sign(direction_id: int) -> tuple[int, float]:
    if direction_id == DIRECTIONS.index("left"):
        return (0, 1.0)
    if direction_id == DIRECTIONS.index("right"):
        return (0, -1.0)
    if direction_id == DIRECTIONS.index("forward"):
        return (1, -1.0)
    return (1, 1.0)


def _footprint_centers_xy(root_xy: torch.Tensor, center_offsets: torch.Tensor, yaws: torch.Tensor) -> torch.Tensor:
    cos_yaw = torch.cos(yaws)
    sin_yaw = torch.sin(yaws)
    offset_x = cos_yaw * center_offsets[..., 0] - sin_yaw * center_offsets[..., 1]
    offset_y = sin_yaw * center_offsets[..., 0] + cos_yaw * center_offsets[..., 1]
    return root_xy + torch.stack((offset_x, offset_y), dim=-1)


def _footprint_vertices_xy(
    root_xy: torch.Tensor,
    half_extents: torch.Tensor,
    center_offsets: torch.Tensor,
    yaws: torch.Tensor,
) -> torch.Tensor:
    center = _footprint_centers_xy(root_xy, center_offsets, yaws)
    corner_x = torch.stack(
        (-half_extents[..., 0], half_extents[..., 0], half_extents[..., 0], -half_extents[..., 0]),
        dim=-1,
    )
    corner_y = torch.stack(
        (-half_extents[..., 1], -half_extents[..., 1], half_extents[..., 1], half_extents[..., 1]),
        dim=-1,
    )
    cos_yaw = torch.cos(yaws).unsqueeze(-1)
    sin_yaw = torch.sin(yaws).unsqueeze(-1)
    vertex_x = center[..., 0].unsqueeze(-1) + cos_yaw * corner_x - sin_yaw * corner_y
    vertex_y = center[..., 1].unsqueeze(-1) + sin_yaw * corner_x + cos_yaw * corner_y
    return torch.stack((vertex_x, vertex_y), dim=-1)


def _object_move_footprint_boxes(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    env_id: int,
    object_id: int,
) -> torch.Tensor:
    boxes_by_object = getattr(env, "_so101_object_move_footprint_boxes", None)
    if boxes_by_object is not None and object_id < len(boxes_by_object):
        boxes = boxes_by_object[object_id]
        if boxes.numel() > 0:
            return boxes.reshape(-1, 4)

    half_extents = _object_footprint_half_extents(env, object_asset_names)[env_id, object_id]
    center_offset = _object_footprint_center_offsets(env, object_asset_names)[env_id, object_id]
    return torch.stack(
        (
            center_offset[0] - half_extents[0],
            center_offset[1] - half_extents[1],
            center_offset[0] + half_extents[0],
            center_offset[1] + half_extents[1],
        )
    ).reshape(1, 4)


def _move_footprint_piece_vertices_xy(
    root_xy: torch.Tensor,
    yaw: torch.Tensor,
    boxes: torch.Tensor,
) -> torch.Tensor:
    local_x = torch.stack((boxes[:, 0], boxes[:, 2], boxes[:, 2], boxes[:, 0]), dim=1)
    local_y = torch.stack((boxes[:, 1], boxes[:, 1], boxes[:, 3], boxes[:, 3]), dim=1)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    vertex_x = root_xy[0] + cos_yaw * local_x - sin_yaw * local_y
    vertex_y = root_xy[1] + sin_yaw * local_x + cos_yaw * local_y
    return torch.stack((vertex_x, vertex_y), dim=-1)


def _move_footprint_piece_vertices(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    positions: torch.Tensor,
    yaws: torch.Tensor,
    env_id: int,
    object_id: int,
) -> torch.Tensor:
    return _move_footprint_piece_vertices_xy(
        positions[env_id, object_id, :2],
        yaws[env_id, object_id],
        _object_move_footprint_boxes(env, object_asset_names, env_id, object_id),
    )


def _footprint_edge_endpoints(piece_vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(start, end)`` points for every footprint-piece edge, flattened."""

    starts = piece_vertices.reshape(-1, 2)
    ends = piece_vertices[:, [1, 2, 3, 0], :].reshape(-1, 2)
    return starts, ends


def _min_point_to_segment_distance(
    points: torch.Tensor,
    seg_starts: torch.Tensor,
    seg_ends: torch.Tensor,
) -> torch.Tensor:
    """Distance from each point to each segment, as a ``(num_points, num_segments)`` grid."""

    seg = seg_ends - seg_starts
    seg_len_sq = torch.clamp(torch.sum(seg * seg, dim=-1), min=1.0e-12)
    rel = points.unsqueeze(1) - seg_starts.unsqueeze(0)
    t = torch.clamp(torch.sum(rel * seg.unsqueeze(0), dim=-1) / seg_len_sq.unsqueeze(0), 0.0, 1.0)
    projection = seg_starts.unsqueeze(0) + t.unsqueeze(-1) * seg.unsqueeze(0)
    return torch.linalg.vector_norm(points.unsqueeze(1) - projection, dim=-1)


def _footprint_min_surface_distance_xy(
    first_pieces: torch.Tensor,
    second_pieces: torch.Tensor,
) -> torch.Tensor:
    """Closest XY surface distance between two piecewise rotated-box footprints.

    Each footprint is a union of convex boxes, so the closest approach between two
    such outlines is always realized between a vertex of one and an edge of the
    other; comparing every vertex against every edge (in both directions) therefore
    yields the true minimum. Interior tiling edges only contribute larger candidates,
    so they are harmless to include, and touching/overlapping footprints fall out as
    ~0 -- exactly what the adjacency checks want. Unlike the old whole-object AABB,
    this follows the real outline, so the gap to a screwdriver's thin metal shaft is
    measured against the shaft rather than the bounding box around the whole tool.
    """

    first_starts, first_ends = _footprint_edge_endpoints(first_pieces)
    second_starts, second_ends = _footprint_edge_endpoints(second_pieces)
    first_to_second = _min_point_to_segment_distance(first_pieces.reshape(-1, 2), second_starts, second_ends)
    second_to_first = _min_point_to_segment_distance(second_pieces.reshape(-1, 2), first_starts, first_ends)
    return torch.minimum(first_to_second.min(), second_to_first.min())


def _object_footprint_surface_distance(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    positions: torch.Tensor,
    yaws: torch.Tensor,
    env_id: int,
    first_id: int,
    second_id: int,
) -> torch.Tensor:
    return _footprint_min_surface_distance_xy(
        _move_footprint_piece_vertices(env, object_asset_names, positions, yaws, env_id, first_id),
        _move_footprint_piece_vertices(env, object_asset_names, positions, yaws, env_id, second_id),
    )


def _pairwise_object_surface_distance(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    positions: torch.Tensor,
    yaws: torch.Tensor,
    first_ids: torch.Tensor,
    second_ids: torch.Tensor,
    env_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-env footprint surface distance between two object slots (``inf`` where unmasked)."""

    distances = torch.full((env.num_envs,), float("inf"), dtype=torch.float32, device=env.device)
    for env_id in torch.nonzero(env_mask, as_tuple=False).flatten().tolist():
        distances[env_id] = _object_footprint_surface_distance(
            env,
            object_asset_names,
            positions,
            yaws,
            env_id,
            int(first_ids[env_id].item()),
            int(second_ids[env_id].item()),
        )
    return distances


def _projection_bounds(vertices: torch.Tensor, axis: int) -> tuple[float, float]:
    projection = vertices[..., axis]
    return float(torch.min(projection).item()), float(torch.max(projection).item())


def _projection_intervals_intersect(
    first_vertices: torch.Tensor,
    second_vertices: torch.Tensor,
    axis: int,
) -> bool:
    first_min, first_max = _projection_bounds(first_vertices, axis)
    second_min, second_max = _projection_bounds(second_vertices, axis)
    return first_min <= second_max and second_min <= first_max


def _footprint_front_coord(vertices: torch.Tensor, axis: int, sign: float) -> float:
    min_coord, max_coord = _projection_bounds(vertices, axis)
    return max_coord if sign > 0.0 else min_coord


def _footprint_near_boundary_coord(vertices: torch.Tensor, axis: int, sign: float) -> float:
    min_coord, max_coord = _projection_bounds(vertices, axis)
    return min_coord if sign > 0.0 else max_coord


_DIRECTIONAL_GAP_LATERAL_STEP_M = 0.001


def _cross_section_axis_extents(
    piece_vertices: torch.Tensor,
    axis: int,
    lateral_axis: int,
    lateral_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice each footprint piece by lines ``lateral_axis == value``.

    A convex (rotated) box meets such a line in a segment; the returned ``low`` and
    ``high`` are that segment's extents along ``axis`` per (piece, value), and
    ``covered`` marks where the piece is actually present. Evaluating the slice at
    the target's true lateral position -- instead of reducing each piece to a single
    corner -- is what stops a diagonal boundary's far corner from being mistaken for
    its near surface.
    """

    starts = piece_vertices
    ends = piece_vertices[:, [1, 2, 3, 0], :]
    start_lat = starts[..., lateral_axis].unsqueeze(-1)
    end_lat = ends[..., lateral_axis].unsqueeze(-1)
    start_axis = starts[..., axis].unsqueeze(-1)
    end_axis = ends[..., axis].unsqueeze(-1)
    values = lateral_values.view(1, 1, -1)

    delta = end_lat - start_lat
    parallel = delta.abs() < 1.0e-12
    safe_delta = torch.where(parallel, torch.ones_like(delta), delta)
    fraction = (values - start_lat) / safe_delta
    crosses = (~parallel) & (fraction >= 0.0) & (fraction <= 1.0)
    axis_at = start_axis + fraction * (end_axis - start_axis)

    inf = torch.full_like(axis_at, float("inf"))
    low = torch.where(crosses, axis_at, inf).amin(dim=1)
    high = torch.where(crosses, axis_at, -inf).amax(dim=1)
    covered = crosses.any(dim=1)
    return low, high, covered


def _directional_footprint_gap(
    target_piece_vertices: torch.Tensor,
    boundary_piece_vertices: torch.Tensor,
    axis: int,
    sign: float,
) -> float | None:
    """Signed directional clearance between two footprints along ``axis``.

    Positive means the target's leading edge has not reached the boundary; negative
    means it has crossed past. The clearance is measured per lateral position over
    the region where the two footprints actually overlap laterally, so it respects
    object orientation and footprint concavities. Returns ``None`` when there is no
    lateral overlap (the boundary is not in the target's path).
    """

    lateral_axis = 1 - axis
    target_lateral = target_piece_vertices[..., lateral_axis]
    boundary_lateral = boundary_piece_vertices[..., lateral_axis]
    lateral_lo = float(torch.maximum(target_lateral.min(), boundary_lateral.min()).item())
    lateral_hi = float(torch.minimum(target_lateral.max(), boundary_lateral.max()).item())
    if lateral_hi <= lateral_lo:
        return None

    samples = min(int((lateral_hi - lateral_lo) / _DIRECTIONAL_GAP_LATERAL_STEP_M) + 2, 256)
    lateral_values = torch.linspace(
        lateral_lo,
        lateral_hi,
        samples,
        device=target_piece_vertices.device,
        dtype=target_piece_vertices.dtype,
    )
    target_low, target_high, target_covered = _cross_section_axis_extents(
        target_piece_vertices, axis, lateral_axis, lateral_values
    )
    boundary_low, boundary_high, boundary_covered = _cross_section_axis_extents(
        boundary_piece_vertices, axis, lateral_axis, lateral_values
    )

    inf = float("inf")
    if sign > 0.0:
        target_front = torch.where(target_covered, target_high, torch.full_like(target_high, -inf)).amax(dim=0)
        boundary_surface = torch.where(boundary_covered, boundary_low, torch.full_like(boundary_low, inf)).amin(dim=0)
    else:
        target_front = torch.where(target_covered, target_low, torch.full_like(target_low, inf)).amin(dim=0)
        boundary_surface = torch.where(boundary_covered, boundary_high, torch.full_like(boundary_high, -inf)).amax(dim=0)

    both_present = target_covered.any(dim=0) & boundary_covered.any(dim=0)
    if not bool(both_present.any().item()):
        return None
    gaps = sign * (boundary_surface - target_front)
    gaps = torch.where(both_present, gaps, torch.full_like(gaps, inf))
    return float(gaps.min().item())


def _directional_footprint_ahead_extent(
    target_piece_vertices: torch.Tensor,
    boundary_piece_vertices: torch.Tensor,
    axis: int,
    sign: float,
) -> float | None:
    """How far the boundary's far edge reaches ahead of the target's leading edge.

    Companion to :func:`_directional_footprint_gap`: ``> 0`` means part of the boundary
    lies in the target's forward path, while ``<= 0`` means it sits entirely behind the
    leading edge -- so a negative directional gap there is a trailing object, not a blocker
    in the move's way. Returns ``None`` when the footprints share no lateral overlap.
    """

    lateral_axis = 1 - axis
    target_lateral = target_piece_vertices[..., lateral_axis]
    boundary_lateral = boundary_piece_vertices[..., lateral_axis]
    lateral_lo = float(torch.maximum(target_lateral.min(), boundary_lateral.min()).item())
    lateral_hi = float(torch.minimum(target_lateral.max(), boundary_lateral.max()).item())
    if lateral_hi <= lateral_lo:
        return None

    samples = min(int((lateral_hi - lateral_lo) / _DIRECTIONAL_GAP_LATERAL_STEP_M) + 2, 256)
    lateral_values = torch.linspace(
        lateral_lo,
        lateral_hi,
        samples,
        device=target_piece_vertices.device,
        dtype=target_piece_vertices.dtype,
    )
    target_low, target_high, target_covered = _cross_section_axis_extents(
        target_piece_vertices, axis, lateral_axis, lateral_values
    )
    boundary_low, boundary_high, boundary_covered = _cross_section_axis_extents(
        boundary_piece_vertices, axis, lateral_axis, lateral_values
    )

    inf = float("inf")
    if sign > 0.0:
        target_front = torch.where(target_covered, target_high, torch.full_like(target_high, -inf)).amax(dim=0)
        boundary_far = torch.where(boundary_covered, boundary_high, torch.full_like(boundary_high, -inf)).amax(dim=0)
    else:
        target_front = torch.where(target_covered, target_low, torch.full_like(target_low, inf)).amin(dim=0)
        boundary_far = torch.where(boundary_covered, boundary_low, torch.full_like(boundary_low, inf)).amin(dim=0)

    both_present = target_covered.any(dim=0) & boundary_covered.any(dim=0)
    if not bool(both_present.any().item()):
        return None
    ahead = sign * (boundary_far - target_front)
    ahead = torch.where(both_present, ahead, torch.full_like(ahead, -inf))
    return float(ahead.max().item())


def _lateral_overlap_width(
    target_piece_vertices: torch.Tensor,
    boundary_piece_vertices: torch.Tensor,
    axis: int,
) -> float:
    """Width of the lateral band (perpendicular to ``axis``) where both footprints exist.

    This measures how much of the target's straight-ahead corridor the boundary
    actually blocks, so an object that merely sits beside the path can be told apart
    from one squarely in front of it.
    """

    lateral_axis = 1 - axis
    lateral_lo = float(
        torch.maximum(target_piece_vertices[..., lateral_axis].min(), boundary_piece_vertices[..., lateral_axis].min()).item()
    )
    lateral_hi = float(
        torch.minimum(target_piece_vertices[..., lateral_axis].max(), boundary_piece_vertices[..., lateral_axis].max()).item()
    )
    if lateral_hi <= lateral_lo:
        return 0.0
    samples = min(int((lateral_hi - lateral_lo) / _DIRECTIONAL_GAP_LATERAL_STEP_M) + 2, 256)
    lateral_values = torch.linspace(
        lateral_lo,
        lateral_hi,
        samples,
        device=target_piece_vertices.device,
        dtype=target_piece_vertices.dtype,
    )
    _, _, target_covered = _cross_section_axis_extents(target_piece_vertices, axis, lateral_axis, lateral_values)
    _, _, boundary_covered = _cross_section_axis_extents(boundary_piece_vertices, axis, lateral_axis, lateral_values)
    both_present = target_covered.any(dim=0) & boundary_covered.any(dim=0)
    return float(both_present.float().mean().item()) * (lateral_hi - lateral_lo)


def _footprint_union_near_boundary_coord(piece_vertices: torch.Tensor, axis: int, sign: float) -> float:
    return _footprint_near_boundary_coord(piece_vertices.reshape(-1, 2), axis, sign)


def _ensure_move_boundary_cache(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    table_bounds: dict[str, tuple[float, float]],
    step_state: _TerminationStepState | None = None,
) -> None:
    """Pick the nearest directional object boundary at reset for each move episode."""

    if hasattr(env, "_so101_move_boundary_coords") and hasattr(env, "_so101_move_boundary_ids"):
        return

    positions = getattr(env, "_so101_initial_object_pos_w", None)
    if positions is None:
        positions = step_state.positions if step_state is not None else _object_positions(env, object_asset_names)
    object_yaws = getattr(env, "_so101_initial_object_yaws", None)
    if object_yaws is None:
        object_yaws = (
            _state_object_yaws(env, object_asset_names, step_state)
            if step_state is not None
            else benchmark_object_yaws(env, object_asset_names)
        )
    active = _active_mask(env, object_asset_names)
    target_ids = _target_indices(env)
    direction_ids = _direction_indices(env)

    boundary_coords = torch.full((env.num_envs,), torch.nan, dtype=torch.float32, device=env.device)
    boundary_ids = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    for env_id in range(env.num_envs):
        axis, sign = _direction_axis_and_sign(int(direction_ids[env_id].item()))
        target_id = int(target_ids[env_id].item())
        target_piece_vertices = _move_footprint_piece_vertices(
            env,
            object_asset_names,
            positions,
            object_yaws,
            env_id,
            target_id,
        )
        lateral_axis = 1 - axis
        target_lateral_width = float(
            (target_piece_vertices[..., lateral_axis].max() - target_piece_vertices[..., lateral_axis].min()).item()
        )
        min_lateral_overlap = MOVE_BOUNDARY_MIN_LATERAL_OVERLAP_FRACTION * target_lateral_width
        candidates: list[tuple[float, float, int]] = []

        for object_id in torch.nonzero(active[env_id], as_tuple=False).flatten().tolist():
            if object_id == target_id:
                continue
            boundary_piece_vertices = _move_footprint_piece_vertices(
                env,
                object_asset_names,
                positions,
                object_yaws,
                env_id,
                object_id,
            )
            gap = _directional_footprint_gap(target_piece_vertices, boundary_piece_vertices, axis, sign)
            if gap is None:
                continue
            # A glancing object beside the corridor is not the boundary the move is aimed at.
            if _lateral_overlap_width(target_piece_vertices, boundary_piece_vertices, axis) < min_lateral_overlap:
                continue
            ahead = _directional_footprint_ahead_extent(
                target_piece_vertices, boundary_piece_vertices, axis, sign
            )
            # The boundary must lie in the target's forward path (extend past its leading edge).
            # An object purely behind that edge also has a negative gap but is not in the way.
            # Keep the signed gap so an object already overlapping the lane ahead (gap < 0) is the
            # nearest obstruction and wins selection, rather than being skipped so a clear object
            # further along is treated as the boundary.
            if ahead is None or ahead <= 0.0:
                continue
            candidates.append(
                (
                    gap,
                    _footprint_union_near_boundary_coord(boundary_piece_vertices, axis, sign),
                    object_id,
                )
            )

        if candidates:
            _gap, boundary_coord, boundary_id = min(candidates, key=lambda candidate: candidate[0])
            boundary_coords[env_id] = boundary_coord
            boundary_ids[env_id] = boundary_id

    env._so101_move_boundary_coords = boundary_coords
    env._so101_move_boundary_ids = boundary_ids


def _move_boundary_distance(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    table_bounds: dict[str, tuple[float, float]],
    step_state: _TerminationStepState | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if step_state is None:
        step_state = _termination_step_state(env, object_asset_names)
    _ensure_move_boundary_cache(env, object_asset_names, table_bounds, step_state)
    positions = step_state.positions
    current_yaws = _state_object_yaws(env, object_asset_names, step_state)
    footprint_center_offsets = _object_footprint_center_offsets(env, object_asset_names)
    current_centers = _footprint_centers_xy(positions[..., :2], footprint_center_offsets, current_yaws)
    initial_yaws = getattr(env, "_so101_initial_object_yaws", current_yaws)
    initial_centers = _footprint_centers_xy(
        env._so101_initial_object_pos_w[..., :2],
        footprint_center_offsets,
        initial_yaws,
    )
    target = _gather_by_index(current_centers, _target_indices(env))
    initial = _gather_by_index(initial_centers, _target_indices(env))
    desired = _direction_vectors(env.device)[_direction_indices(env)]
    delta = target - initial
    progress = torch.sum(delta * desired, dim=1)
    lateral = torch.linalg.vector_norm(delta - progress.unsqueeze(1) * desired, dim=1)

    initial_positions = getattr(env, "_so101_initial_object_pos_w", positions)
    boundary_distance = torch.full((env.num_envs,), torch.nan, dtype=torch.float32, device=env.device)
    for env_id in torch.nonzero(env._so101_move_boundary_ids >= 0, as_tuple=False).flatten().tolist():
        target_id = int(_target_indices(env)[env_id].item())
        boundary_id = int(env._so101_move_boundary_ids[env_id].item())
        axis, sign = _direction_axis_and_sign(int(_direction_indices(env)[env_id].item()))
        target_piece_vertices = _move_footprint_piece_vertices(
            env,
            object_asset_names,
            positions,
            current_yaws,
            env_id,
            target_id,
        )
        boundary_piece_vertices = _move_footprint_piece_vertices(
            env,
            object_asset_names,
            initial_positions,
            initial_yaws,
            env_id,
            boundary_id,
        )
        gap = _directional_footprint_gap(target_piece_vertices, boundary_piece_vertices, axis, sign)
        if gap is not None:
            boundary_distance[env_id] = gap
    return boundary_distance, progress, lateral, target


def move_success(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    table_bounds: dict[str, tuple[float, float]] | None = None,
    boundary_distance: float = MOVE_BOUNDARY_SUCCESS_DISTANCE_M,
    no_boundary_min_progress: float = MOVE_NO_BOUNDARY_MIN_PROGRESS_M,
    straightness_tolerance: float = MOVE_STRAIGHTNESS_TOLERANCE_M,
    past_boundary_tolerance: float = MOVE_PAST_BOUNDARY_TOLERANCE_M,
    contact_grace_time_s: float = DEFAULT_CONTACT_GRACE_TIME_S,
    confirm_steps: int | None = None,
    confirm_time_s: float = DEFAULT_SUCCESS_CONFIRM_TIME_S,
    step_state: _TerminationStepState | None = None,
) -> torch.Tensor:
    """Success for ``Move object direction`` against an object boundary or a 2-inch fallback."""

    if table_bounds is None:
        table_bounds = {"x": (0.08, 0.45), "y": (-0.20, 0.20)}

    distance_to_boundary, progress, lateral, _target = _move_boundary_distance(
        env, object_asset_names, table_bounds, step_state
    )
    # The assigned boundary only constrains success while the target still shares a lateral
    # corridor with it; once the target rotates or drifts out of that corridor the directional
    # gap is undefined (NaN). Treat that as "no boundary in the path" and fall back to the
    # plain forward-progress criterion instead of dead-locking the episode on a NaN gap.
    has_boundary = (env._so101_move_boundary_ids >= 0) & torch.isfinite(distance_to_boundary)
    # Touching the boundary (a small negative clearance) still counts as "next to" it,
    # and shares its lower bound with the move_past_boundary failure so the two tile the
    # clearance axis with no dead zone. Straightness is judged on the current (settled)
    # deviation -- the success confirmation window requires it to hold -- so a transient
    # excursion that recovers no longer disqualifies the move.
    close_to_boundary = (distance_to_boundary >= -past_boundary_tolerance) & (
        distance_to_boundary <= boundary_distance
    )
    reached_goal = torch.where(has_boundary, close_to_boundary, progress >= no_boundary_min_progress)

    success_now = (
        (progress > 0.0)
        & reached_goal
        & (lateral <= straightness_tolerance)
        & _grasped_object_contact_allows_success(env, object_asset_names, step_state, contact_grace_time_s)
        & _task_is(env, TASK_MOVE)
    )

    if not hasattr(env, "_so101_move_success_counter"):
        env._so101_move_success_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._so101_move_success_counter = torch.where(
        success_now,
        env._so101_move_success_counter + 1,
        torch.zeros_like(env._so101_move_success_counter),
    )
    return env._so101_move_success_counter >= _confirmation_steps(env, confirm_time_s, confirm_steps)


def task_success(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    bin_name: str,
    table_bounds: dict[str, tuple[float, float]] | None = None,
    min_episode_time_s: float = 5.0,
    confirm_time_s: float = DEFAULT_SUCCESS_CONFIRM_TIME_S,
    move_straightness_tolerance: float = MOVE_STRAIGHTNESS_TOLERANCE_M,
    contact_grace_time_s: float = DEFAULT_CONTACT_GRACE_TIME_S,
) -> torch.Tensor:
    """Dispatch to the success condition for the active benchmark family.

    ``benchmark_failure`` records raw failure-condition violations before its
    five-second confirmation windows are applied.  A violation immediately
    makes success ineligible, even though it only terminates the episode after
    it has remained continuously true for its confirmation interval.
    """

    step_state = _termination_step_state(env, object_asset_names)
    success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    active_families = set(getattr(env, "_so101_task_family", ()))
    if TASK_BIN in active_families:
        success |= bin_success(
            env,
            object_asset_names,
            bin_name,
            confirm_time_s=confirm_time_s,
            step_state=step_state,
        )
    if TASK_NEXT_TO in active_families:
        success |= next_to_success(
            env,
            object_asset_names,
            contact_grace_time_s=contact_grace_time_s,
            confirm_time_s=confirm_time_s,
            step_state=step_state,
        )
    if TASK_BETWEEN in active_families:
        success |= between_success(
            env,
            object_asset_names,
            contact_grace_time_s=contact_grace_time_s,
            confirm_time_s=confirm_time_s,
            step_state=step_state,
        )
    if TASK_MOVE in active_families:
        success |= move_success(
            env,
            object_asset_names,
            table_bounds,
            straightness_tolerance=move_straightness_tolerance,
            contact_grace_time_s=contact_grace_time_s,
            confirm_time_s=confirm_time_s,
            step_state=step_state,
        )

    active_failure_conditions = getattr(env, "_so101_failure_conditions_active", None)
    if active_failure_conditions is not None:
        # Failure is evaluated before success by ``TerminationsCfg``.  Reset a
        # pending success confirmation for affected environments so success
        # also has to be continuously valid without a failure-condition
        # violation.
        for task_family, counter_name in (
            (TASK_BIN, "_so101_bin_success_counter"),
            (TASK_NEXT_TO, "_so101_next_to_success_counter"),
            (TASK_BETWEEN, "_so101_between_success_counter"),
            (TASK_MOVE, "_so101_move_success_counter"),
        ):
            counter = getattr(env, counter_name, None)
            if counter is not None:
                reset_counter = active_failure_conditions & _task_is(env, task_family)
                setattr(env, counter_name, torch.where(reset_counter, torch.zeros_like(counter), counter))
        success &= ~active_failure_conditions

    return success & _episode_age_at_least(env, min_episode_time_s)


def _debug_object_name(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    env_id: int,
    object_id: int,
) -> str:
    asset_name = object_asset_names[object_id]
    for episode in getattr(env, "so101_bench_episodes", ()):
        if int(episode.get("env_id", -1)) != env_id:
            continue
        active_ids = episode.get("active_object_ids", ())
        if object_id not in active_ids:
            break
        label = episode.get("active_labels", ())[active_ids.index(object_id)]
        return f"{asset_name} ({label})"
    return asset_name


def _debug_boundary_name(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    env_id: int,
    boundary_id: int,
) -> str:
    if boundary_id == -1:
        return "none"
    return _debug_object_name(env, object_asset_names, env_id, boundary_id)


def _confirmed_success_diagnostic(
    name: str,
    instant: bool,
    counter: int,
    required_steps: int,
    age_ready: bool,
    details: str,
) -> TaskConditionDiagnostic:
    return TaskConditionDiagnostic(
        kind="success",
        name=name,
        met=instant and counter >= required_steps and age_ready,
        details=f"instant={instant}, held={counter}/{required_steps} steps, age_gate={age_ready}; {details}",
    )


def _bin_task_diagnostic(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    bin_name: str,
    step_state: _TerminationStepState,
    env_id: int,
    min_episode_time_s: float,
    confirm_time_s: float,
) -> TaskConditionDiagnostic:
    active = _active_mask(env, object_asset_names)
    bin_asset: RigidObject = env.scene[bin_name]
    bin_quat_inv = math_utils.quat_inv(bin_asset.data.root_quat_w)
    rel = step_state.positions - bin_asset.data.root_pos_w.unsqueeze(1)
    rel_local = torch.stack(
        [math_utils.quat_apply(bin_quat_inv, rel[:, object_id, :]) for object_id in range(rel.shape[1])],
        dim=1,
    )
    footprint_half_extents = _bin_footprint_half_extents(env)
    footprint_center_offsets = _bin_footprint_center_offsets(env)
    rel_footprint = rel_local[..., :2] - footprint_center_offsets.unsqueeze(1)
    inside = torch.all(torch.abs(rel_footprint) <= footprint_half_extents.unsqueeze(1), dim=-1)
    active_ids = torch.nonzero(active[env_id], as_tuple=False).flatten().tolist()
    active_roots = ", ".join(
        f"{_debug_object_name(env, object_asset_names, env_id, object_id)}: "
        f"inside={bool(inside[env_id, object_id].item())}, "
        f"footprint_xy=({float(rel_footprint[env_id, object_id, 0].item()):.4f}, "
        f"{float(rel_footprint[env_id, object_id, 1].item()):.4f})m"
        for object_id in active_ids
    )
    instant = bool(torch.all(inside[env_id, active_ids]).item())
    return _confirmed_success_diagnostic(
        "all_active_object_roots_in_bin",
        instant,
        int(env._so101_bin_success_counter[env_id].item()),
        _confirmation_steps(env, confirm_time_s),
        bool(_episode_age_at_least(env, min_episode_time_s)[env_id].item()),
        f"required |x|<={float(footprint_half_extents[env_id, 0].item()):.4f}m and "
        f"|y|<={float(footprint_half_extents[env_id, 1].item()):.4f}m; {active_roots}",
    )


def _next_to_task_diagnostic(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    step_state: _TerminationStepState,
    env_id: int,
    min_episode_time_s: float,
    confirm_time_s: float,
    contact_grace_time_s: float = DEFAULT_CONTACT_GRACE_TIME_S,
) -> TaskConditionDiagnostic:
    positions = step_state.positions
    yaws = _state_object_yaws(env, object_asset_names, step_state)
    target_ids = _target_indices(env)
    ref_ids = _referent_indices(env)[:, 0]
    target_id = int(target_ids[env_id].item())
    ref_id = int(ref_ids[env_id].item())
    surface_distance = float(
        _object_footprint_surface_distance(
            env, object_asset_names, positions, yaws, env_id, target_id, ref_id
        ).item()
    )
    made_contact = grasped_object_made_contact(env, object_asset_names, step_state)
    contact_exceeded = _grasped_object_contact_exceeded_from_counter(env, contact_grace_time_s)
    instant = (surface_distance <= SPATIAL_SUCCESS_DISTANCE_M) and not bool(made_contact[env_id].item())
    return _confirmed_success_diagnostic(
        "target_next_to_referent",
        instant,
        int(env._so101_next_to_success_counter[env_id].item()),
        _confirmation_steps(env, confirm_time_s),
        bool(_episode_age_at_least(env, min_episode_time_s)[env_id].item()),
        f"target={_debug_object_name(env, object_asset_names, env_id, target_id)}, "
        f"referent={_debug_object_name(env, object_asset_names, env_id, ref_id)}, "
        f"surface_distance={surface_distance:.4f}m "
        f"(required <={SPATIAL_SUCCESS_DISTANCE_M:.4f}m), "
        f"grasped_object_made_contact={bool(made_contact[env_id].item())}, "
        f"contact_grace_exceeded={bool(contact_exceeded[env_id].item())}",
    )


def _between_task_diagnostic(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    step_state: _TerminationStepState,
    env_id: int,
    min_episode_time_s: float,
    confirm_time_s: float,
    contact_grace_time_s: float = DEFAULT_CONTACT_GRACE_TIME_S,
) -> TaskConditionDiagnostic:
    positions = step_state.positions
    yaws = _state_object_yaws(env, object_asset_names, step_state)
    target_ids = _target_indices(env)
    refs = _referent_indices(env)
    target_id = int(target_ids[env_id].item())
    ref_a_id = int(refs[env_id, 0].item())
    ref_b_id = int(refs[env_id, 1].item())
    distance_to_first = float(
        _object_footprint_surface_distance(
            env, object_asset_names, positions, yaws, env_id, target_id, ref_a_id
        ).item()
    )
    distance_to_second = float(
        _object_footprint_surface_distance(
            env, object_asset_names, positions, yaws, env_id, target_id, ref_b_id
        ).item()
    )
    total_distance = distance_to_first + distance_to_second
    fraction = distance_to_first / total_distance if total_distance > 0.0 else 0.5

    target = _gather_by_index(positions, target_ids)[:, :2]
    ref_a = _gather_by_index(positions, refs[:, 0])[:, :2]
    ref_b = _gather_by_index(positions, refs[:, 1])[:, :2]
    segment = ref_b - ref_a
    segment_len_sq = torch.clamp(torch.sum(segment * segment, dim=1), min=1.0e-6)
    line_fraction = torch.sum((target - ref_a) * segment, dim=1) / segment_len_sq
    projection = ref_a + line_fraction.unsqueeze(1) * segment
    perpendicular = float(torch.linalg.vector_norm(target - projection, dim=1)[env_id].item())
    made_contact = grasped_object_made_contact(env, object_asset_names, step_state)
    contact_exceeded = _grasped_object_contact_exceeded_from_counter(env, contact_grace_time_s)
    centered = 0.2 <= fraction <= 0.8
    instant = centered and (perpendicular <= BETWEEN_LINE_TOLERANCE_M) and not bool(made_contact[env_id].item())
    return _confirmed_success_diagnostic(
        "target_between_referents",
        instant,
        int(env._so101_between_success_counter[env_id].item()),
        _confirmation_steps(env, confirm_time_s),
        bool(_episode_age_at_least(env, min_episode_time_s)[env_id].item()),
        f"target={_debug_object_name(env, object_asset_names, env_id, target_id)}, "
        f"referents=({_debug_object_name(env, object_asset_names, env_id, ref_a_id)}, "
        f"{_debug_object_name(env, object_asset_names, env_id, ref_b_id)}), "
        f"segment_fraction={fraction:.4f} (required 0.20..0.80; "
        f"surface_distances referent1={distance_to_first:.4f}m, referent2={distance_to_second:.4f}m), "
        f"perpendicular_distance={perpendicular:.4f}m "
        f"(required <={BETWEEN_LINE_TOLERANCE_M:.4f}m), "
        f"grasped_object_made_contact={bool(made_contact[env_id].item())}, "
        f"contact_grace_exceeded={bool(contact_exceeded[env_id].item())}",
    )


def _move_task_diagnostic(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    table_bounds: dict[str, tuple[float, float]],
    step_state: _TerminationStepState,
    env_id: int,
    min_episode_time_s: float,
    confirm_time_s: float,
    straightness_tolerance: float,
    contact_grace_time_s: float = DEFAULT_CONTACT_GRACE_TIME_S,
) -> TaskConditionDiagnostic:
    distance_to_boundary, progress, lateral, _target = _move_boundary_distance(
        env, object_asset_names, table_bounds, step_state
    )
    made_contact = grasped_object_made_contact(env, object_asset_names, step_state)
    contact_exceeded = _grasped_object_contact_exceeded_from_counter(env, contact_grace_time_s)
    # Matches move_success: a boundary only applies while its directional gap is defined; an
    # undefined (NaN) gap falls back to the forward-progress criterion.
    boundary_gap_defined = bool(torch.isfinite(distance_to_boundary[env_id]).item())
    has_boundary = (env._so101_move_boundary_ids >= 0) & torch.isfinite(distance_to_boundary)
    close_to_boundary = (
        (distance_to_boundary >= -MOVE_PAST_BOUNDARY_TOLERANCE_M)
        & (distance_to_boundary <= MOVE_BOUNDARY_SUCCESS_DISTANCE_M)
    )
    reached_goal = torch.where(has_boundary, close_to_boundary, progress >= MOVE_NO_BOUNDARY_MIN_PROGRESS_M)
    instant = bool(
        (
            (progress > 0.0)
            & reached_goal
            & (lateral <= straightness_tolerance)
            & (~made_contact)
        )[env_id].item()
    )
    boundary_id = int(env._so101_move_boundary_ids[env_id].item())
    boundary_requirement = (
        f"(object boundary requirement {-MOVE_PAST_BOUNDARY_TOLERANCE_M:.4f}..{MOVE_BOUNDARY_SUCCESS_DISTANCE_M:.4f}m)"
        if boundary_gap_defined
        else "(undefined: target no longer overlaps the boundary laterally; using no-boundary progress criterion)"
    )
    return _confirmed_success_diagnostic(
        "target_moved_to_boundary",
        instant,
        int(env._so101_move_success_counter[env_id].item()),
        _confirmation_steps(env, confirm_time_s),
        bool(_episode_age_at_least(env, min_episode_time_s)[env_id].item()),
        f"boundary={_debug_boundary_name(env, object_asset_names, env_id, boundary_id)}, "
        f"distance_to_boundary={float(distance_to_boundary[env_id].item()):.4f}m "
        f"{boundary_requirement}, "
        f"directional_progress={float(progress[env_id].item()):.4f}m "
        f"(no-boundary requirement >={MOVE_NO_BOUNDARY_MIN_PROGRESS_M:.4f}m), "
        f"current_lateral_error={float(lateral[env_id].item()):.4f}m "
        f"(required <={straightness_tolerance:.4f}m), "
        f"grasped_object_made_contact={bool(made_contact[env_id].item())}, "
        f"contact_grace_exceeded={bool(contact_exceeded[env_id].item())}",
    )


def _task_success_diagnostic(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    bin_name: str,
    table_bounds: dict[str, tuple[float, float]],
    step_state: _TerminationStepState,
    env_id: int,
    min_episode_time_s: float,
    confirm_time_s: float,
    move_straightness_tolerance: float,
    contact_grace_time_s: float,
) -> TaskConditionDiagnostic:
    task_family = env._so101_task_family[env_id]
    if task_family == TASK_BIN:
        return _bin_task_diagnostic(
            env, object_asset_names, bin_name, step_state, env_id, min_episode_time_s, confirm_time_s
        )
    if task_family == TASK_NEXT_TO:
        return _next_to_task_diagnostic(
            env, object_asset_names, step_state, env_id, min_episode_time_s, confirm_time_s, contact_grace_time_s
        )
    if task_family == TASK_BETWEEN:
        return _between_task_diagnostic(
            env, object_asset_names, step_state, env_id, min_episode_time_s, confirm_time_s, contact_grace_time_s
        )
    return _move_task_diagnostic(
        env,
        object_asset_names,
        table_bounds,
        step_state,
        env_id,
        min_episode_time_s,
        confirm_time_s,
        move_straightness_tolerance,
        contact_grace_time_s,
    )


def _update_grasp_attempts(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    jaw_joint_name: str,
    jaw_close_delta: float,
    jaw_open_fraction: float,
    object_distance_threshold: float,
    object_pos_w: torch.Tensor | None = None,
) -> None:
    """Track the nearest grasped object and count one eligible attempt per armed jaw-close cycle."""

    robot = env.scene[robot_cfg.name]
    if not hasattr(env, "_so101_jaw_joint_id"):
        jaw_ids, _ = robot.find_joints(jaw_joint_name)
        env._so101_jaw_joint_id = jaw_ids[0]

    jaw_pos = robot.data.joint_pos[:, env._so101_jaw_joint_id]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    if not hasattr(env, "_so101_grasp_attempt_counts"):
        env._so101_grasp_attempt_counts = torch.zeros(
            (env.num_envs, len(object_asset_names)), dtype=torch.long, device=env.device
        )
    if not hasattr(env, "_so101_grasp_armed"):
        env._so101_grasp_armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_so101_grasped_object_ids"):
        env._so101_grasped_object_ids = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    if getattr(env, "_so101_grasp_arm_jaw_pos", None) is None:
        env._so101_grasp_arm_jaw_pos = jaw_pos.clone()

    jaw_limits = robot.data.joint_pos_limits[:, env._so101_jaw_joint_id]
    jaw_lower = torch.minimum(jaw_limits[:, 0], jaw_limits[:, 1])
    jaw_upper = torch.maximum(jaw_limits[:, 0], jaw_limits[:, 1])
    jaw_open_threshold = jaw_lower + jaw_open_fraction * (jaw_upper - jaw_lower)

    jaw_is_open = jaw_pos >= jaw_open_threshold
    was_armed = env._so101_grasp_armed
    newly_armed = jaw_is_open & (~was_armed)
    arm_jaw_pos = torch.where(newly_armed, jaw_pos, env._so101_grasp_arm_jaw_pos)
    arm_jaw_pos = torch.where(was_armed | newly_armed, torch.maximum(arm_jaw_pos, jaw_pos), arm_jaw_pos)

    close_cycle = (was_armed | newly_armed) & ((arm_jaw_pos - jaw_pos) >= jaw_close_delta)

    if object_pos_w is None:
        object_pos_w = _object_positions(env, object_asset_names)
    object_dist = torch.linalg.vector_norm(object_pos_w - ee_pos_w.unsqueeze(1), dim=2)
    active = _active_mask(env, object_asset_names)
    active_dist = torch.where(active, object_dist, torch.full_like(object_dist, torch.inf))
    nearest_active_dist, nearest_active_object_ids = torch.min(active_dist, dim=1)
    grasp_started = close_cycle & (nearest_active_dist <= object_distance_threshold)

    eligible = _attempt_object_mask(env, object_asset_names)
    masked_dist = torch.where(eligible, object_dist, torch.full_like(object_dist, torch.inf))
    nearest_dist, nearest_object_ids = torch.min(masked_dist, dim=1)
    near_object = nearest_dist <= object_distance_threshold
    counted_attempts = close_cycle & near_object
    counted_env_ids = torch.nonzero(counted_attempts, as_tuple=False).flatten()
    if counted_env_ids.numel() > 0:
        env._so101_grasp_attempt_counts[counted_env_ids, nearest_object_ids[counted_env_ids]] += 1

    grasped_object_ids = torch.where(
        jaw_is_open,
        torch.full_like(env._so101_grasped_object_ids, -1),
        env._so101_grasped_object_ids,
    )
    env._so101_grasped_object_ids = torch.where(
        grasp_started,
        nearest_active_object_ids,
        grasped_object_ids,
    )
    env._so101_grasp_armed = (was_armed | jaw_is_open) & (~close_cycle)
    env._so101_grasp_arm_jaw_pos = arm_jaw_pos


def _ensure_failure_displacement_baseline(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    bin_name: str,
    baseline_time_s: float,
) -> torch.Tensor:
    if not hasattr(env, "_so101_failure_object_pos_w"):
        env._so101_failure_object_pos_w = env._so101_initial_object_pos_w.clone()
    if not hasattr(env, "_so101_failure_bin_pos_w"):
        env._so101_failure_bin_pos_w = env._so101_initial_bin_pos_w.clone()
    if not hasattr(env, "_so101_failure_baseline_recorded"):
        env._so101_failure_baseline_recorded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    ready_to_record = _episode_age_at_least(env, baseline_time_s) & (~env._so101_failure_baseline_recorded)
    if torch.any(ready_to_record):
        mark_benchmark_robot_start(env, object_asset_names, bin_name, env_ids=ready_to_record)

    return env._so101_failure_baseline_recorded


def _update_max_object_lift(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    object_pos_w: torch.Tensor,
    baseline_recorded: torch.Tensor,
) -> torch.Tensor:
    """Accumulate, per object, the highest the root has risen above its settled height.

    The settled resting height is the Z stored in ``_so101_failure_object_pos_w`` once
    the displacement baseline is recorded, so the lift is measured against the object's
    height after it has stopped dropping onto the table -- not its slightly-elevated
    spawn pose. Only active objects in envs whose baseline is in are accumulated; the
    running maximum is what the postmortem classifier later thresholds against.
    """

    if not hasattr(env, "_so101_max_object_lift"):
        env._so101_max_object_lift = torch.zeros(
            (env.num_envs, len(object_asset_names)), dtype=torch.float32, device=env.device
        )
    lift = object_pos_w[..., 2] - env._so101_failure_object_pos_w[..., 2]
    record = baseline_recorded.unsqueeze(1) & _active_mask(env, object_asset_names)
    env._so101_max_object_lift = torch.where(
        record,
        torch.maximum(env._so101_max_object_lift, lift),
        env._so101_max_object_lift,
    )
    return env._so101_max_object_lift


@dataclass(frozen=True)
class PostmortemFailureDiagnostic:
    """End-of-episode failure classification for one environment."""

    env_id: int
    task_family: str
    failure_type: str
    target_object: str
    target_lift_m: float
    lifted_wrong_object: str
    max_non_target_lift_m: float
    lift_threshold_m: float


def benchmark_postmortem_failure_diagnostics(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    lift_threshold: float = LIFT_OFF_GROUND_LIMIT_M,
) -> list[PostmortemFailureDiagnostic]:
    """Classify each non-bin episode into exactly one postmortem failure type.

    The classification is a decision tree over how far each active object was lifted
    clear of the table during the episode (tracked by :func:`_update_max_object_lift`):

    * the target was lifted >= ``lift_threshold`` -> ``placement`` (the robot grasped
      the correct object but settled it in the wrong place);
    * the target was not lifted but some distractor was -> ``semantic`` (the robot
      lifted the wrong object);
    * nothing was lifted high enough -> ``failed_grasp``.

    Bin episodes return ``not_applicable`` (the three types are defined for the
    instruction-following families only). Callers decide whether to keep the label --
    a confirmed success is not a failure, so the label is only meaningful for episodes
    that did not succeed.
    """

    diagnostics: list[PostmortemFailureDiagnostic] = []
    task_families = getattr(env, "_so101_task_family", None)
    max_lift = getattr(env, "_so101_max_object_lift", None)
    active = _active_mask(env, object_asset_names)
    target_ids = _target_indices(env)
    for env_id in range(env.num_envs):
        task_family = task_families[env_id] if task_families is not None else TASK_BIN
        target_id = int(target_ids[env_id].item())
        target_object = _debug_object_name(env, object_asset_names, env_id, target_id)
        if task_family == TASK_BIN or max_lift is None:
            diagnostics.append(
                PostmortemFailureDiagnostic(
                    env_id=env_id,
                    task_family=task_family,
                    failure_type=POSTMORTEM_NOT_APPLICABLE if task_family == TASK_BIN else POSTMORTEM_NONE,
                    target_object=target_object,
                    target_lift_m=0.0,
                    lifted_wrong_object="none",
                    max_non_target_lift_m=0.0,
                    lift_threshold_m=lift_threshold,
                )
            )
            continue

        target_lift = float(max_lift[env_id, target_id].item())
        max_non_target_lift = 0.0
        lifted_wrong_object = "none"
        for object_id in torch.nonzero(active[env_id], as_tuple=False).flatten().tolist():
            if object_id == target_id:
                continue
            object_lift = float(max_lift[env_id, object_id].item())
            if object_lift > max_non_target_lift:
                max_non_target_lift = object_lift
                if object_lift >= lift_threshold:
                    lifted_wrong_object = _debug_object_name(env, object_asset_names, env_id, object_id)

        if target_lift >= lift_threshold:
            failure_type = POSTMORTEM_PLACEMENT
        elif max_non_target_lift >= lift_threshold:
            failure_type = POSTMORTEM_SEMANTIC
        else:
            failure_type = POSTMORTEM_FAILED_GRASP
            lifted_wrong_object = "none"

        diagnostics.append(
            PostmortemFailureDiagnostic(
                env_id=env_id,
                task_family=task_family,
                failure_type=failure_type,
                target_object=target_object,
                target_lift_m=target_lift,
                lifted_wrong_object=lifted_wrong_object,
                max_non_target_lift_m=max_non_target_lift,
                lift_threshold_m=lift_threshold,
            )
        )
    return diagnostics


def benchmark_failure(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    bin_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    jaw_joint_name: str = "Jaw",
    jaw_close_delta: float = 0.06,
    jaw_open_fraction: float = 0.5,
    grasp_attempt_object_distance: float = GRASP_ATTEMPT_OBJECT_DISTANCE_M,
    max_grasp_attempts: int = 3,
    enforce_max_grasp_attempts: bool = True,
    failure_confirm_time_s: float = DEFAULT_FAILURE_CONFIRM_TIME_S,
    bin_displacement_limit: float = BIN_DISPLACEMENT_LIMIT_M,
    non_target_displacement_limit: float = NON_TARGET_DISPLACEMENT_LIMIT_M,
    boundary_displacement_limit: float = BOUNDARY_DISPLACEMENT_LIMIT_M,
    move_straightness_tolerance: float = MOVE_STRAIGHTNESS_TOLERANCE_M,
    move_straightness_failure_confirm_time_s: float = DEFAULT_MOVE_STRAIGHTNESS_FAILURE_CONFIRM_TIME_S,
    move_past_boundary_tolerance: float = MOVE_PAST_BOUNDARY_TOLERANCE_M,
    move_past_boundary_failure_confirm_time_s: float = DEFAULT_MOVE_PAST_BOUNDARY_FAILURE_CONFIRM_TIME_S,
    contact_grace_time_s: float = DEFAULT_CONTACT_GRACE_TIME_S,
    min_episode_time_s: float = 5.0,
    displacement_baseline_time_s: float = 1.0,
    table_bounds: dict[str, tuple[float, float]] | None = None,
) -> torch.Tensor:
    """Cross-task failure conditions from the paper appendix.

    The term covers the measurable simulator-side rules: max grasp attempts,
    bin displacement, non-target object displacement, moved move-boundaries,
    move trajectory straightness, and contact between the currently grasped
    object and another tabletop object.
    Qualitative labels such as semantic error and bad grasp strategy remain
    annotation categories rather than automatic simulator events.
    """

    if not hasattr(env, "_so101_initial_object_pos_w"):
        env._so101_failure_reasons = [FAILURE_REASON_NONE for _ in range(env.num_envs)]
        env._so101_postmortem_failure_diagnostics = []
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if table_bounds is None:
        table_bounds = {"x": (0.08, 0.45), "y": (-0.20, 0.20)}

    step_state = _termination_step_state(env, object_asset_names)
    object_pos_w = step_state.positions
    _update_grasp_attempts(
        env,
        object_asset_names=object_asset_names,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        jaw_joint_name=jaw_joint_name,
        jaw_close_delta=jaw_close_delta,
        jaw_open_fraction=jaw_open_fraction,
        object_distance_threshold=grasp_attempt_object_distance,
        object_pos_w=object_pos_w,
    )
    step_state.grasped_object_made_contact = None
    baseline_recorded = _ensure_failure_displacement_baseline(
        env,
        object_asset_names=object_asset_names,
        bin_name=bin_name,
        baseline_time_s=displacement_baseline_time_s,
    )
    _update_max_object_lift(env, object_asset_names, object_pos_w, baseline_recorded)
    # Recomputed every step from the running max lift so that, on the step an episode
    # ends, the latest classification is already stored before the env auto-resets and
    # zeros the lift buffer -- mirroring how _so101_failure_reasons survives reset.
    env._so101_postmortem_failure_diagnostics = benchmark_postmortem_failure_diagnostics(env, object_asset_names)

    active = _active_mask(env, object_asset_names)
    # The close that raises a target count to three is still a usable attempt.
    exhausted_attempts = env._so101_grasp_attempt_counts > max_grasp_attempts
    instant_attempt_failure = torch.any(exhausted_attempts & _attempt_object_mask(env, object_asset_names), dim=1)
    if not enforce_max_grasp_attempts:
        instant_attempt_failure = torch.zeros_like(instant_attempt_failure)
    attempt_failure = _held_failure(
        env, "_so101_attempt_failure_counter", instant_attempt_failure, failure_confirm_time_s
    )

    bin_asset: RigidObject = env.scene[bin_name]
    # Displacements are judged on the XY (tabletop) plane only: objects spawn slightly
    # above the table and the bin spawns ~0.02m above it, so a Z component would record
    # a phantom displacement at episode start before anything has been touched.
    bin_displacement = torch.linalg.vector_norm(
        bin_asset.data.root_pos_w[..., :2] - env._so101_failure_bin_pos_w[..., :2], dim=1
    )
    instant_bin_failure = bin_displacement > bin_displacement_limit
    bin_failure = _held_failure(
        env,
        "_so101_bin_failure_counter",
        instant_bin_failure,
        failure_confirm_time_s,
    )

    object_displacement = torch.linalg.vector_norm(
        object_pos_w[..., :2] - env._so101_failure_object_pos_w[..., :2], dim=2
    )
    target_ids = _target_indices(env)
    target_mask = torch.zeros_like(active)
    target_mask[torch.arange(env.num_envs, device=env.device), target_ids] = True
    instruction_task = ~_task_is(env, TASK_BIN)
    instant_non_target_moved = torch.any(
        (object_displacement > non_target_displacement_limit) & active & (~target_mask), dim=1
    )
    non_target_moved = _held_failure(
        env,
        "_so101_non_target_failure_counter",
        instant_non_target_moved & instruction_task,
        failure_confirm_time_s,
    )

    boundary_moved = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    move_past_boundary = torch.zeros_like(boundary_moved)
    move_trajectory_not_straight_enough = torch.zeros_like(boundary_moved)
    instant_move_past_boundary = torch.zeros_like(boundary_moved)
    instant_move_trajectory_not_straight_enough = torch.zeros_like(boundary_moved)
    active_families = set(getattr(env, "_so101_task_family", ()))
    if TASK_MOVE in active_families:
        _ensure_move_boundary_cache(env, object_asset_names, table_bounds, step_state)
        boundary_object_ids = env._so101_move_boundary_ids
        object_boundary_env_ids = torch.nonzero(boundary_object_ids >= 0, as_tuple=False).flatten()
        if object_boundary_env_ids.numel() > 0:
            current_yaws = _state_object_yaws(env, object_asset_names, step_state)
            for env_id in object_boundary_env_ids.tolist():
                boundary_object_id = int(boundary_object_ids[env_id].item())
                axis, sign = _direction_axis_and_sign(int(_direction_indices(env)[env_id].item()))
                current_surface_coord = _footprint_union_near_boundary_coord(
                    _move_footprint_piece_vertices(
                        env,
                        object_asset_names,
                        object_pos_w,
                        current_yaws,
                        env_id,
                        boundary_object_id,
                    ),
                    axis,
                    sign,
                )
                boundary_moved[env_id] = (
                    abs(current_surface_coord - float(env._so101_move_boundary_coords[env_id].item()))
                    > boundary_displacement_limit
                )
        distance_to_boundary, _progress, lateral, _target = _move_boundary_distance(
            env, object_asset_names, table_bounds, step_state
        )
        move_task = _task_is(env, TASK_MOVE)
        # A target driven past its boundary must settle there for the confirmation window
        # rather than glancing past in mid-flight: only a held, settled overshoot counts as
        # a failure, mirroring the straightness confirmation below.
        instant_move_past_boundary = (
            (env._so101_move_boundary_ids >= 0)
            & (distance_to_boundary < -move_past_boundary_tolerance)
            & move_task
        )
        move_past_boundary = _held_failure(
            env,
            "_so101_move_past_boundary_failure_counter",
            instant_move_past_boundary,
            move_past_boundary_failure_confirm_time_s,
        )
        # Current (not running-max) deviation: a transient swing that recovers no longer
        # latches a permanent straightness failure. It must be held long enough to
        # count as a settled bad final position rather than an in-flight detour.
        instant_move_trajectory_not_straight_enough = (
            lateral > move_straightness_tolerance
        ) & move_task
        move_trajectory_not_straight_enough = _held_failure(
            env,
            "_so101_move_straightness_failure_counter",
            instant_move_trajectory_not_straight_enough,
            move_straightness_failure_confirm_time_s,
        )
    else:
        if hasattr(env, "_so101_move_straightness_failure_counter"):
            env._so101_move_straightness_failure_counter.zero_()
        if hasattr(env, "_so101_move_past_boundary_failure_counter"):
            env._so101_move_past_boundary_failure_counter.zero_()
    instant_move_boundary_failure = boundary_moved & _task_is(env, TASK_MOVE)
    move_boundary_failure = _held_failure(
        env,
        "_so101_move_boundary_failure_counter",
        instant_move_boundary_failure,
        failure_confirm_time_s,
    )

    made_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if active_families & {TASK_NEXT_TO, TASK_BETWEEN, TASK_MOVE}:
        made_contact = grasped_object_contact_exceeded_grace_period(
            env, object_asset_names, step_state, contact_grace_time_s
        ) & (~_task_is(env, TASK_BIN))

    timeout_confirmation_failure = getattr(env, "_so101_timeout_success_confirmation_failed", None)
    if timeout_confirmation_failure is None:
        timeout_confirmation_failure = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Keep this raw mask separate from the confirmed failure below.  Success
    # must not win during the confirmation interval for any failure condition.
    env._so101_failure_conditions_active = (
        instant_attempt_failure
        | instant_bin_failure
        | (instant_non_target_moved & instruction_task)
        | instant_move_boundary_failure
        | instant_move_past_boundary
        | instant_move_trajectory_not_straight_enough
        | made_contact
        | timeout_confirmation_failure
    )

    failure = (
        attempt_failure
        | bin_failure
        | non_target_moved
        | move_boundary_failure
        | move_past_boundary
        | move_trajectory_not_straight_enough
        | made_contact
    )
    aged_failure = (
        failure & _episode_age_at_least(env, min_episode_time_s) & baseline_recorded
    ) | timeout_confirmation_failure

    env._so101_failure_reasons = [FAILURE_REASON_NONE for _ in range(env.num_envs)]
    for env_id in torch.nonzero(aged_failure, as_tuple=False).flatten().tolist():
        reasons = []
        if bool(attempt_failure[env_id].item()):
            reasons.append(FAILURE_REASON_MAX_GRASP_ATTEMPTS)
        if bool(bin_failure[env_id].item()):
            reasons.append(FAILURE_REASON_BIN_DISPLACED)
        if bool(non_target_moved[env_id].item()):
            reasons.append(FAILURE_REASON_NON_TARGET_MOVED)
        if bool(move_boundary_failure[env_id].item()):
            reasons.append(FAILURE_REASON_MOVE_BOUNDARY_MOVED)
        if bool(move_past_boundary[env_id].item()):
            reasons.append(FAILURE_REASON_MOVE_PAST_BOUNDARY)
        if bool(move_trajectory_not_straight_enough[env_id].item()):
            reasons.append(FAILURE_REASON_MOVE_TRAJECTORY_NOT_STRAIGHT_ENOUGH)
        if bool(made_contact[env_id].item()):
            reasons.append(FAILURE_REASON_MADE_CONTACT)
        if bool(timeout_confirmation_failure[env_id].item()):
            reasons.append(FAILURE_REASON_SUCCESS_CONFIRMATION_BREACHED)
        env._so101_failure_reasons[env_id] = "+".join(reasons)

    return aged_failure


def _gated_failure_diagnostic(
    name: str,
    raw_met: bool,
    age_ready: bool,
    baseline_recorded: bool,
    details: str,
) -> TaskConditionDiagnostic:
    return TaskConditionDiagnostic(
        kind="failure",
        name=name,
        met=raw_met and age_ready and baseline_recorded,
        details=f"raw={raw_met}, age_gate={age_ready}, baseline_recorded={baseline_recorded}; {details}",
    )


def _failure_diagnostics(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    bin_name: str,
    table_bounds: dict[str, tuple[float, float]],
    step_state: _TerminationStepState,
    env_id: int,
    min_episode_time_s: float,
    max_grasp_attempts: int,
    enforce_max_grasp_attempts: bool,
    bin_displacement_limit: float,
    non_target_displacement_limit: float,
    boundary_displacement_limit: float,
    move_straightness_tolerance: float,
    contact_grace_time_s: float,
) -> list[TaskConditionDiagnostic]:
    active = _active_mask(env, object_asset_names)
    age_ready = bool(_episode_age_at_least(env, min_episode_time_s)[env_id].item())
    baseline_recorded = bool(env._so101_failure_baseline_recorded[env_id].item())
    conditions = []

    attempt_mask = _attempt_object_mask(env, object_asset_names)[env_id]
    attempt_ids = torch.nonzero(attempt_mask, as_tuple=False).flatten().tolist()
    attempt_counts = env._so101_grasp_attempt_counts[env_id]
    exhausted_attempts = attempt_counts > max_grasp_attempts
    attempt_failure = bool(enforce_max_grasp_attempts and torch.any(exhausted_attempts & attempt_mask).item())
    attempt_details = ", ".join(
        f"{_debug_object_name(env, object_asset_names, env_id, object_id)}="
        f"{int(attempt_counts[object_id].item())}"
        for object_id in attempt_ids
    )
    attempt_rule = (
        f"failure if any >{max_grasp_attempts}"
        if enforce_max_grasp_attempts
        else f"limit disabled; would fail if any >{max_grasp_attempts}"
    )
    conditions.append(
        _gated_failure_diagnostic(
            "max_grasp_attempts",
            attempt_failure,
            age_ready,
            baseline_recorded,
            f"attempt_counts=[{attempt_details}], allowed_attempts={max_grasp_attempts} ({attempt_rule})",
        )
    )

    bin_asset: RigidObject = env.scene[bin_name]
    bin_displacement = torch.linalg.vector_norm(
        bin_asset.data.root_pos_w[..., :2] - env._so101_failure_bin_pos_w[..., :2],
        dim=1,
    )
    conditions.append(
        _gated_failure_diagnostic(
            "bin_displaced",
            bool((bin_displacement[env_id] > bin_displacement_limit).item()),
            age_ready,
            baseline_recorded,
            f"displacement={float(bin_displacement[env_id].item()):.4f}m "
            f"(failure if >{bin_displacement_limit:.4f}m)",
        )
    )

    object_displacement = torch.linalg.vector_norm(
        step_state.positions[..., :2] - env._so101_failure_object_pos_w[..., :2],
        dim=2,
    )
    target_id = int(_target_indices(env)[env_id].item())
    if env._so101_task_family[env_id] != TASK_BIN:
        non_target_ids = [
            object_id
            for object_id in torch.nonzero(active[env_id], as_tuple=False).flatten().tolist()
            if object_id != target_id
        ]
        non_target_moved = any(
            float(object_displacement[env_id, object_id].item()) > non_target_displacement_limit
            for object_id in non_target_ids
        )
        displacement_details = ", ".join(
            f"{_debug_object_name(env, object_asset_names, env_id, object_id)}="
            f"{float(object_displacement[env_id, object_id].item()):.4f}m"
            for object_id in non_target_ids
        )
        conditions.append(
            _gated_failure_diagnostic(
                "non_target_moved",
                non_target_moved,
                age_ready,
                baseline_recorded,
                f"displacements=[{displacement_details}] "
                f"(failure if any >{non_target_displacement_limit:.4f}m)",
            )
        )

    if env._so101_task_family[env_id] == TASK_MOVE:
        _ensure_move_boundary_cache(env, object_asset_names, table_bounds, step_state)
        boundary_id = int(env._so101_move_boundary_ids[env_id].item())
        boundary_displacement = 0.0
        if boundary_id >= 0:
            axis, sign = _direction_axis_and_sign(int(_direction_indices(env)[env_id].item()))
            current_surface_coord = _footprint_union_near_boundary_coord(
                _move_footprint_piece_vertices(
                    env,
                    object_asset_names,
                    step_state.positions,
                    _state_object_yaws(env, object_asset_names, step_state),
                    env_id,
                    boundary_id,
                ),
                axis,
                sign,
            )
            boundary_displacement = abs(
                current_surface_coord - float(env._so101_move_boundary_coords[env_id].item())
            )
        conditions.append(
            _gated_failure_diagnostic(
                "move_boundary_moved",
                boundary_displacement > boundary_displacement_limit,
                age_ready,
                baseline_recorded,
                f"boundary={_debug_boundary_name(env, object_asset_names, env_id, boundary_id)}, "
                f"displacement={boundary_displacement:.4f}m "
                f"(failure if >{boundary_displacement_limit:.4f}m)",
            )
        )
        distance_to_boundary, _progress, lateral, _target = _move_boundary_distance(
            env, object_asset_names, table_bounds, step_state
        )
        past_boundary_counter = getattr(env, "_so101_move_past_boundary_failure_counter", None)
        if past_boundary_counter is None:
            past_boundary_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        past_boundary_required_steps = _confirmation_steps(
            env,
            DEFAULT_MOVE_PAST_BOUNDARY_FAILURE_CONFIRM_TIME_S,
        )
        past_boundary_instant = boundary_id >= 0 and bool(
            (distance_to_boundary[env_id] < -MOVE_PAST_BOUNDARY_TOLERANCE_M).item()
        )
        conditions.append(
            _gated_failure_diagnostic(
                "move_past_boundary",
                past_boundary_instant
                and int(past_boundary_counter[env_id].item()) >= past_boundary_required_steps,
                age_ready,
                baseline_recorded,
                f"boundary={_debug_boundary_name(env, object_asset_names, env_id, boundary_id)}, "
                f"distance_to_boundary={float(distance_to_boundary[env_id].item()):.4f}m "
                f"(failure if <{-MOVE_PAST_BOUNDARY_TOLERANCE_M:.4f}m), "
                f"held={int(past_boundary_counter[env_id].item())}/{past_boundary_required_steps} steps",
            )
        )
        straightness_counter = getattr(env, "_so101_move_straightness_failure_counter", None)
        if straightness_counter is None:
            straightness_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        straightness_required_steps = _confirmation_steps(
            env,
            DEFAULT_MOVE_STRAIGHTNESS_FAILURE_CONFIRM_TIME_S,
        )
        straightness_instant = bool((lateral[env_id] > move_straightness_tolerance).item())
        conditions.append(
            _gated_failure_diagnostic(
                "move_trajectory_not_straight_enough",
                straightness_instant
                and int(straightness_counter[env_id].item()) >= straightness_required_steps,
                age_ready,
                baseline_recorded,
                f"current_lateral_error={float(lateral[env_id].item()):.4f}m "
                f"(failure if >{move_straightness_tolerance:.4f}m), "
                f"held={int(straightness_counter[env_id].item())}/{straightness_required_steps} steps",
            )
        )

    if env._so101_task_family[env_id] in {TASK_NEXT_TO, TASK_BETWEEN, TASK_MOVE}:
        made_contact = bool(grasped_object_made_contact(env, object_asset_names, step_state)[env_id].item())
        contact_step_counts = getattr(env, "_so101_grasped_object_contact_steps", None)
        if contact_step_counts is None:
            contact_step_counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        contact_steps = int(contact_step_counts[env_id].item())
        contact_exceeded = bool(_grasped_object_contact_exceeded_from_counter(env, contact_grace_time_s)[env_id].item())
        grasped_object_ids = getattr(env, "_so101_grasped_object_ids", None)
        grasped_object_id = -1 if grasped_object_ids is None else int(grasped_object_ids[env_id].item())
        grasped_object_name = (
            _debug_object_name(env, object_asset_names, env_id, grasped_object_id)
            if grasped_object_id >= 0
            else "none"
        )
        conditions.append(
            _gated_failure_diagnostic(
                "made_contact",
                contact_exceeded,
                age_ready,
                baseline_recorded,
                f"grasped_object={grasped_object_name}, current_contact={made_contact}, "
                f"continuous_contact={contact_steps * _env_step_dt(env):.4f}s "
                f"(failure if >{contact_grace_time_s:.4f}s)",
            )
        )

    return conditions


def task_condition_diagnostics(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    bin_name: str,
    table_bounds: dict[str, tuple[float, float]] | None = None,
    success_min_episode_time_s: float = 5.0,
    confirm_time_s: float = DEFAULT_SUCCESS_CONFIRM_TIME_S,
    move_straightness_tolerance: float = MOVE_STRAIGHTNESS_TOLERANCE_M,
    failure_min_episode_time_s: float = 5.0,
    max_grasp_attempts: int = 3,
    enforce_max_grasp_attempts: bool = True,
    bin_displacement_limit: float = BIN_DISPLACEMENT_LIMIT_M,
    non_target_displacement_limit: float = NON_TARGET_DISPLACEMENT_LIMIT_M,
    boundary_displacement_limit: float = BOUNDARY_DISPLACEMENT_LIMIT_M,
    contact_grace_time_s: float = DEFAULT_CONTACT_GRACE_TIME_S,
) -> list[TaskDiagnostics]:
    """Return current success and failure statuses without advancing task state."""

    if not hasattr(env, "_so101_initial_object_pos_w"):
        return []
    if table_bounds is None:
        table_bounds = {"x": (0.08, 0.45), "y": (-0.20, 0.20)}

    step_state = _termination_step_state(env, object_asset_names)
    episode_age = _episode_age_s(env)
    diagnostics = []
    for env_id in range(env.num_envs):
        conditions = [
            _task_success_diagnostic(
                env,
                object_asset_names,
                bin_name,
                table_bounds,
                step_state,
                env_id,
                success_min_episode_time_s,
                confirm_time_s,
                move_straightness_tolerance,
                contact_grace_time_s,
            )
        ]
        conditions.extend(
            _failure_diagnostics(
                env,
                object_asset_names,
                bin_name,
                table_bounds,
                step_state,
                env_id,
                failure_min_episode_time_s,
                max_grasp_attempts,
                enforce_max_grasp_attempts,
                bin_displacement_limit,
                non_target_displacement_limit,
                boundary_displacement_limit,
                move_straightness_tolerance,
                contact_grace_time_s,
            )
        )
        diagnostics.append(
            TaskDiagnostics(
                env_id=env_id,
                task_family=env._so101_task_family[env_id],
                episode_age_s=float(episode_age[env_id].item()),
                conditions=tuple(conditions),
            )
        )
    return diagnostics
