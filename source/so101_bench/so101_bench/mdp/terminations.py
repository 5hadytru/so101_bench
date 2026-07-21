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
    BETWEEN_CENTER_FRACTION_MAX,
    BETWEEN_CENTER_FRACTION_MIN,
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
    TASK_NAMED_BIN,
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

# Legacy postmortem labels.  These are retained because older outcome files and
# downstream analysis use them, but the evidence-based classifier below reports
# them separately as ``legacy_failure_type``.  A maximum root-Z excursion is not
# enough evidence to infer intent: an object can rise because it tipped, bounced,
# or was hit by the object actually being manipulated.
POSTMORTEM_NONE = "none"
POSTMORTEM_NOT_APPLICABLE = "not_applicable"
POSTMORTEM_SEMANTIC = "semantic"
POSTMORTEM_FAILED_GRASP = "failed_grasp"
POSTMORTEM_PLACEMENT = "placement"

# Evidence-based primary failure stages.  These labels describe the last stage
# reached with defensible simulator evidence rather than guessing policy intent
# from lift alone.  ``wrong_object_targeted`` deliberately replaces the broader
# and usually unverifiable claim that an episode was a "semantic" failure.
POSTMORTEM_NO_MANIPULATION_ATTEMPT = "no_manipulation_attempt"
POSTMORTEM_WRONG_OBJECT_TARGETED = "wrong_object_targeted"
POSTMORTEM_TARGET_ACQUISITION_FAILED = "target_acquisition_failed"
POSTMORTEM_OBJECT_ACQUISITION_FAILED = "object_acquisition_failed"
POSTMORTEM_TARGET_DROPPED = "target_dropped"
POSTMORTEM_TARGET_RELEASED_OUTSIDE_GOAL = "target_released_outside_goal"
POSTMORTEM_TRANSPORT_FAILED = "transport_failed"
POSTMORTEM_GOAL_NOT_REACHED = "goal_not_reached"
POSTMORTEM_GOAL_REACHED_BUT_UNCONFIRMED = "goal_reached_but_unconfirmed"
POSTMORTEM_GOAL_REACHED_BUT_UNSTABLE = "goal_reached_but_unstable"
POSTMORTEM_GOAL_REACHED_WITH_RULE_VIOLATION = "goal_reached_with_rule_violation"
POSTMORTEM_GOAL_OVERSHOT = "goal_overshot"
POSTMORTEM_TIMEOUT_DURING_CONFIRMATION = "timeout_during_confirmation"
POSTMORTEM_INCOMPLETE_MULTI_OBJECT_TASK = "incomplete_multi_object_task"
POSTMORTEM_AMBIGUOUS = "ambiguous"
POSTMORTEM_LEGACY_FAILURE_TYPES = (
    POSTMORTEM_SEMANTIC,
    POSTMORTEM_FAILED_GRASP,
    POSTMORTEM_PLACEMENT,
)
POSTMORTEM_FAILURE_TYPES = POSTMORTEM_LEGACY_FAILURE_TYPES + (
    POSTMORTEM_NO_MANIPULATION_ATTEMPT,
    POSTMORTEM_WRONG_OBJECT_TARGETED,
    POSTMORTEM_TARGET_ACQUISITION_FAILED,
    POSTMORTEM_OBJECT_ACQUISITION_FAILED,
    POSTMORTEM_TARGET_DROPPED,
    POSTMORTEM_TARGET_RELEASED_OUTSIDE_GOAL,
    POSTMORTEM_TRANSPORT_FAILED,
    POSTMORTEM_GOAL_NOT_REACHED,
    POSTMORTEM_GOAL_REACHED_BUT_UNCONFIRMED,
    POSTMORTEM_GOAL_REACHED_BUT_UNSTABLE,
    POSTMORTEM_GOAL_REACHED_WITH_RULE_VIOLATION,
    POSTMORTEM_GOAL_OVERSHOT,
    POSTMORTEM_TIMEOUT_DURING_CONFIRMATION,
    POSTMORTEM_INCOMPLETE_MULTI_OBJECT_TASK,
    POSTMORTEM_AMBIGUOUS,
)

POSTMORTEM_CLASSIFICATION_VERSION = 2

# Conservative temporal-evidence thresholds.  A jaw-close merely associates an
# attempt with a nearby object.  Acquisition/manipulation requires motion over
# multiple samples, which avoids calling a one-frame bump a wrong-object grasp.
POSTMORTEM_MIN_STEP_MOTION_M = 7.5e-4
POSTMORTEM_MIN_ASSOCIATED_TRANSPORT_M = 0.01
POSTMORTEM_ACQUISITION_TRANSPORT_M = 0.005
POSTMORTEM_ACQUISITION_HOLD_STEPS = 2
POSTMORTEM_DROP_HOLD_STEPS = 2
POSTMORTEM_MEANINGFUL_GOAL_HOLD_TIME_S = 0.25

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
    contact_by_object: torch.Tensor | None = None
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


def _object_contact_mask(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    step_state: _TerminationStepState,
    force_threshold: float = 0.0,
) -> torch.Tensor:
    """Return an env-by-object mask for contact with another tabletop object."""

    if step_state.contact_by_object is not None:
        return step_state.contact_by_object

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
    step_state.contact_by_object = contact_by_object
    return contact_by_object


def target_object_made_contact(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    step_state: _TerminationStepState | None = None,
    force_threshold: float = 0.0,
) -> torch.Tensor:
    """Return whether the instruction target contacts another tabletop object."""

    if step_state is None:
        step_state = _termination_step_state(env, object_asset_names)
    override = getattr(env, "_so101_target_object_made_contact_override", None)
    if override is not None:
        return override.to(device=env.device, dtype=torch.bool)
    return _gather_by_index(
        _object_contact_mask(env, object_asset_names, step_state, force_threshold),
        _target_indices(env),
    )


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

    grasped_object_ids = getattr(env, "_so101_grasped_object_ids", None)
    if grasped_object_ids is None:
        step_state.grasped_object_made_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return step_state.grasped_object_made_contact

    has_grasped_object = grasped_object_ids >= 0
    safe_object_ids = torch.clamp(grasped_object_ids, min=0)
    step_state.grasped_object_made_contact = (
        _gather_by_index(
            _object_contact_mask(env, object_asset_names, step_state, force_threshold),
            safe_object_ids,
        )
        & has_grasped_object
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
        (TASK_NAMED_BIN, "_so101_bin_success_counter"),
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
    """Require no grasped contact and no target contact earlier in the episode."""

    grasped_object_contact_exceeded_grace_period(env, object_asset_names, step_state, grace_time_s)
    target_contact = target_object_made_contact(env, object_asset_names, step_state)
    target_contact_ever = getattr(env, "_so101_target_object_contact_ever", None)
    if not isinstance(target_contact_ever, torch.Tensor) or tuple(target_contact_ever.shape) != (env.num_envs,):
        target_contact_ever = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    non_bin_task = ~_task_is(env, TASK_BIN)
    env._so101_target_object_contact_ever = target_contact_ever | (target_contact & non_bin_task)
    return (
        ~grasped_object_made_contact(env, object_asset_names, step_state)
        & ~env._so101_target_object_contact_ever
    )


def _target_contact_ever_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    contact_ever = getattr(env, "_so101_target_object_contact_ever", None)
    if isinstance(contact_ever, torch.Tensor) and tuple(contact_ever.shape) == (env.num_envs,):
        return contact_ever.to(device=env.device, dtype=torch.bool)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def bin_success(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    bin_name: str,
    confirm_steps: int | None = None,
    confirm_time_s: float = DEFAULT_SUCCESS_CONFIRM_TIME_S,
    step_state: _TerminationStepState | None = None,
) -> torch.Tensor:
    """Score all-object and named-target bin containment.

    All-object bin episodes require every active root inside. Named-bin episodes
    require only the instructed target root inside. Containment is a pure XY test:
    each relevant root position is transformed
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
    target_inside = inside[
        torch.arange(env.num_envs, device=env.device),
        _target_indices(env),
    ]
    success_now = (all_active_inside & _task_is(env, TASK_BIN)) | (
        target_inside & _task_is(env, TASK_NAMED_BIN)
    )

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
    min_segment_fraction: float = BETWEEN_CENTER_FRACTION_MIN,
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

    # The fraction is retained as a diagnostic. With the configured [0.0, 1.0]
    # band it does not restrict success; line alignment below does.
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
    centered = (fraction >= min_segment_fraction) & (fraction <= BETWEEN_CENTER_FRACTION_MAX)

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
    # The boundary criterion has only an upper bound. A negative signed gap (the target's
    # leading edge has passed the boundary surface) remains below the two-inch maximum.
    # Straightness is judged on the current (settled) deviation; the success confirmation
    # window requires it to hold, so a transient excursion that recovers is permitted.
    close_to_boundary = distance_to_boundary < boundary_distance
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

    # The postmortem tracker is advanced by ``benchmark_failure``, which runs
    # before this term.  Persist the authoritative success-side tolerance so
    # subsequent tracker steps (and offline overrides) use the same geometry.
    env._so101_success_move_straightness_tolerance = float(move_straightness_tolerance)
    step_state = _termination_step_state(env, object_asset_names)
    success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    active_families = set(getattr(env, "_so101_task_family", ()))
    if active_families & {TASK_BIN, TASK_NAMED_BIN}:
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
            (TASK_NAMED_BIN, "_so101_bin_success_counter"),
            (TASK_NEXT_TO, "_so101_next_to_success_counter"),
            (TASK_BETWEEN, "_so101_between_success_counter"),
            (TASK_MOVE, "_so101_move_success_counter"),
        ):
            counter = getattr(env, counter_name, None)
            if counter is not None:
                reset_counter = active_failure_conditions & _task_is(env, task_family)
                setattr(env, counter_name, torch.where(reset_counter, torch.zeros_like(counter), counter))
        success &= ~active_failure_conditions

    policy_control_active = getattr(env, "_so101_policy_control_active", None)
    if isinstance(policy_control_active, torch.Tensor):
        policy_control_active = policy_control_active.to(device=env.device, dtype=torch.bool)
        for counter_name in (
            "_so101_bin_success_counter",
            "_so101_next_to_success_counter",
            "_so101_between_success_counter",
            "_so101_move_success_counter",
        ):
            counter = getattr(env, counter_name, None)
            if isinstance(counter, torch.Tensor):
                setattr(
                    env,
                    counter_name,
                    torch.where(policy_control_active, counter, torch.zeros_like(counter)),
                )
        success &= policy_control_active

    confirmed_success = success & _episode_age_at_least(env, min_episode_time_s)

    # Persist the actual configured confirmation requirement and confirmed
    # state for postmortem attribution.  The benchmark commonly uses a much
    # shorter confirmation window than this function's default, so inferring
    # it later from a hard-coded duration would silently mislabel successes.
    required_steps = _confirmation_steps(env, confirm_time_s)
    episode_steps = env.episode_length_buf.to(device=env.device, dtype=torch.long)
    last_steps = getattr(env, "_so101_success_confirmation_last_episode_step", None)
    if not isinstance(last_steps, torch.Tensor) or tuple(last_steps.shape) != (env.num_envs,):
        last_steps = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    ever_confirmed = getattr(env, "_so101_success_ever_confirmed", None)
    if not isinstance(ever_confirmed, torch.Tensor) or tuple(ever_confirmed.shape) != (env.num_envs,):
        ever_confirmed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    new_episode = (last_steps >= 0) & (episode_steps < last_steps)
    ever_confirmed = torch.where(new_episode, torch.zeros_like(ever_confirmed), ever_confirmed)
    env._so101_success_ever_confirmed = ever_confirmed | confirmed_success
    env._so101_success_confirmed_current = confirmed_success.clone()
    env._so101_success_confirmation_required_steps = torch.full(
        (env.num_envs,), required_steps, dtype=torch.long, device=env.device
    )
    env._so101_success_confirmation_last_episode_step = episode_steps.clone()
    if bool(getattr(env, "_so101_pm_tracker_available", False)):
        # ``benchmark_failure`` runs first in the termination-manager pass.
        # Refresh its serialization hook now that the actual success counter
        # and configured confirmation threshold are known, before auto-reset
        # can clear the episode tensors.
        env._so101_postmortem_failure_diagnostics = benchmark_postmortem_failure_diagnostics(
            env,
            object_asset_names,
            confirm_time_s=confirm_time_s,
        )
    return confirmed_success


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
    named_bin = env._so101_task_family[env_id] == TASK_NAMED_BIN
    scored_ids = (
        [int(_target_indices(env)[env_id].item())]
        if named_bin
        else torch.nonzero(active[env_id], as_tuple=False).flatten().tolist()
    )
    scored_roots = ", ".join(
        f"{_debug_object_name(env, object_asset_names, env_id, object_id)}: "
        f"inside={bool(inside[env_id, object_id].item())}, "
        f"footprint_xy=({float(rel_footprint[env_id, object_id, 0].item()):.4f}, "
        f"{float(rel_footprint[env_id, object_id, 1].item()):.4f})m"
        for object_id in scored_ids
    )
    instant = bool(torch.all(inside[env_id, scored_ids]).item())
    return _confirmed_success_diagnostic(
        "target_root_in_bin" if named_bin else "all_active_object_roots_in_bin",
        instant,
        int(env._so101_bin_success_counter[env_id].item()),
        _confirmation_steps(env, confirm_time_s),
        bool(_episode_age_at_least(env, min_episode_time_s)[env_id].item()),
        f"required |x|<={float(footprint_half_extents[env_id, 0].item()):.4f}m and "
        f"|y|<={float(footprint_half_extents[env_id, 1].item()):.4f}m; {scored_roots}",
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
    target_contact_ever = _target_contact_ever_mask(env)
    contact_exceeded = _grasped_object_contact_exceeded_from_counter(env, contact_grace_time_s)
    contact_rule_breached = bool((made_contact | target_contact_ever)[env_id].item())
    instant = (surface_distance <= SPATIAL_SUCCESS_DISTANCE_M) and not contact_rule_breached
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
        f"target_object_contact_ever={bool(target_contact_ever[env_id].item())}, "
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
    target_contact_ever = _target_contact_ever_mask(env)
    contact_exceeded = _grasped_object_contact_exceeded_from_counter(env, contact_grace_time_s)
    centered = BETWEEN_CENTER_FRACTION_MIN <= fraction <= BETWEEN_CENTER_FRACTION_MAX
    contact_rule_breached = bool((made_contact | target_contact_ever)[env_id].item())
    instant = centered and (perpendicular <= BETWEEN_LINE_TOLERANCE_M) and not contact_rule_breached
    return _confirmed_success_diagnostic(
        "target_between_referents",
        instant,
        int(env._so101_between_success_counter[env_id].item()),
        _confirmation_steps(env, confirm_time_s),
        bool(_episode_age_at_least(env, min_episode_time_s)[env_id].item()),
        f"target={_debug_object_name(env, object_asset_names, env_id, target_id)}, "
        f"referents=({_debug_object_name(env, object_asset_names, env_id, ref_a_id)}, "
        f"{_debug_object_name(env, object_asset_names, env_id, ref_b_id)}), "
        f"segment_fraction={fraction:.4f} (required 0.10..0.90; "
        f"surface_distances referent1={distance_to_first:.4f}m, referent2={distance_to_second:.4f}m), "
        f"perpendicular_distance={perpendicular:.4f}m "
        f"(required <={BETWEEN_LINE_TOLERANCE_M:.4f}m), "
        f"grasped_object_made_contact={bool(made_contact[env_id].item())}, "
        f"target_object_contact_ever={bool(target_contact_ever[env_id].item())}, "
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
    target_contact_ever = _target_contact_ever_mask(env)
    contact_exceeded = _grasped_object_contact_exceeded_from_counter(env, contact_grace_time_s)
    # Matches move_success: a boundary only applies while its directional gap is defined; an
    # undefined (NaN) gap falls back to the forward-progress criterion.
    boundary_gap_defined = bool(torch.isfinite(distance_to_boundary[env_id]).item())
    has_boundary = (env._so101_move_boundary_ids >= 0) & torch.isfinite(distance_to_boundary)
    close_to_boundary = distance_to_boundary < MOVE_BOUNDARY_SUCCESS_DISTANCE_M
    reached_goal = torch.where(has_boundary, close_to_boundary, progress >= MOVE_NO_BOUNDARY_MIN_PROGRESS_M)
    instant = bool(
        (
            (progress > 0.0)
            & reached_goal
            & (lateral <= straightness_tolerance)
            & (~made_contact)
            & (~target_contact_ever)
        )[env_id].item()
    )
    boundary_id = int(env._so101_move_boundary_ids[env_id].item())
    boundary_requirement = (
        f"(object boundary requirement <{MOVE_BOUNDARY_SUCCESS_DISTANCE_M:.4f}m; no minimum)"
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
        f"target_object_contact_ever={bool(target_contact_ever[env_id].item())}, "
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
    if task_family in {TASK_BIN, TASK_NAMED_BIN}:
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


@dataclass(frozen=True)
class _GraspStepEvidence:
    """Raw jaw/EE evidence produced while updating the legacy attempt counter."""

    ee_pos_w: torch.Tensor
    jaw_is_open: torch.Tensor
    close_cycle: torch.Tensor
    grasp_started: torch.Tensor
    nearest_active_object_ids: torch.Tensor
    nearest_active_distance_m: torch.Tensor


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
) -> _GraspStepEvidence:
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
    return _GraspStepEvidence(
        ee_pos_w=ee_pos_w,
        jaw_is_open=jaw_is_open,
        close_cycle=close_cycle,
        grasp_started=grasp_started,
        nearest_active_object_ids=nearest_active_object_ids,
        nearest_active_distance_m=nearest_active_dist,
    )


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
class ManipulationAttemptDiagnostic:
    """Temporal evidence for one jaw-close cycle associated with a nearby object."""

    attempt_id: int
    object_id: int
    object_name: str
    is_target: bool
    close_step: int
    end_step: int | None
    nearest_distance_m: float
    acquired: bool
    acquisition_step: int | None
    released: bool
    dropped: bool
    associated_transport_m: float
    max_comotion_steps: int
    goal_met_during_attempt: bool


@dataclass(frozen=True)
class PostmortemFailureDiagnostic:
    """Evidence-backed end-of-episode attribution for one environment.

    The first eight fields are the version-1 schema and intentionally remain in
    place.  ``failure_type`` is the evidence-based primary stage whenever the
    online tracker is available; ``legacy_failure_type`` contains the old
    lift-only result for consumers that still need to compare against it.
    """

    env_id: int
    task_family: str
    failure_type: str
    target_object: str
    target_lift_m: float
    lifted_wrong_object: str
    max_non_target_lift_m: float
    lift_threshold_m: float
    classification_version: int = POSTMORTEM_CLASSIFICATION_VERSION
    legacy_failure_type: str = POSTMORTEM_NONE
    confidence: float = 0.0
    secondary_failure_types: tuple[str, ...] = ()
    rationale: str = ""
    target_semantics_applicable: bool = True
    target_attempt_count: int = 0
    wrong_object_attempt_count: int = 0
    unassociated_attempt_count: int = 0
    target_acquired: bool = False
    wrong_object_acquired: bool = False
    target_manipulated: bool = False
    target_max_displacement_m: float = 0.0
    target_associated_transport_m: float = 0.0
    target_post_acquisition_transport_m: float = 0.0
    target_drop_count: int = 0
    target_release_count: int = 0
    goal_ever_reached: bool = False
    goal_ever_confirmed: bool = False
    max_goal_hold_steps: int = 0
    final_goal_met: bool = False
    attempts: tuple[ManipulationAttemptDiagnostic, ...] = ()
    evidence: dict[str, object] | None = None


@dataclass(frozen=True)
class _PostmortemGoalStep:
    met: torch.Tensor
    overshot: torch.Tensor
    inside_bin: torch.Tensor
    metrics: list[dict[str, object]]


def _postmortem_goal_step(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    bin_name: str,
    table_bounds: dict[str, tuple[float, float]],
    step_state: _TerminationStepState,
    move_straightness_tolerance: float = MOVE_STRAIGHTNESS_TOLERANCE_M,
    move_past_boundary_tolerance: float = MOVE_PAST_BOUNDARY_TOLERANCE_M,
) -> _PostmortemGoalStep:
    """Compute task geometry without advancing any success/failure counter."""

    active = _active_mask(env, object_asset_names)
    met = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    overshot = torch.zeros_like(met)
    inside_bin = torch.zeros_like(active)
    metrics: list[dict[str, object]] = [{} for _ in range(env.num_envs)]

    # Tests and offline diagnostics can supply exact task geometry when a full
    # Isaac scene (notably quaternion helpers) is intentionally unavailable.
    met_override = getattr(env, "_so101_postmortem_goal_met_override", None)
    if met_override is not None:
        met = met_override.to(device=env.device, dtype=torch.bool).clone()
        overshot_override = getattr(env, "_so101_postmortem_goal_overshot_override", None)
        if overshot_override is not None:
            overshot = overshot_override.to(device=env.device, dtype=torch.bool).clone()
        inside_override = getattr(env, "_so101_postmortem_inside_bin_override", None)
        if inside_override is not None:
            inside_bin = inside_override.to(device=env.device, dtype=torch.bool).clone()
        for env_id in range(env.num_envs):
            metrics[env_id] = {
                "name": "task_goal_override",
                "value": 1.0 if bool(met[env_id].item()) else 0.0,
                "threshold": 1.0,
                "margin": 0.0 if bool(met[env_id].item()) else -1.0,
                "instant": bool(met[env_id].item()),
            }
        return _PostmortemGoalStep(met, overshot, inside_bin, metrics)

    positions = step_state.positions
    families = set(getattr(env, "_so101_task_family", ()))
    yaws: torch.Tensor | None = None
    if families & {TASK_NEXT_TO, TASK_BETWEEN, TASK_MOVE}:
        yaws = _state_object_yaws(env, object_asset_names, step_state)

    if families & {TASK_BIN, TASK_NAMED_BIN}:
        bin_asset: RigidObject = env.scene[bin_name]
        rel = positions - bin_asset.data.root_pos_w.unsqueeze(1)
        bin_quat_inv = math_utils.quat_inv(bin_asset.data.root_quat_w)
        rel_local = torch.stack(
            [math_utils.quat_apply(bin_quat_inv, rel[:, object_id, :]) for object_id in range(rel.shape[1])],
            dim=1,
        )
        rel_footprint = rel_local[..., :2] - _bin_footprint_center_offsets(env).unsqueeze(1)
        inside_bin = torch.all(
            torch.abs(rel_footprint) <= _bin_footprint_half_extents(env).unsqueeze(1), dim=-1
        )
        bin_tasks = _task_is(env, TASK_BIN)
        named_bin_tasks = _task_is(env, TASK_NAMED_BIN)
        met |= torch.all(torch.where(active, inside_bin, torch.ones_like(inside_bin)), dim=1) & bin_tasks
        for env_id in torch.nonzero(bin_tasks, as_tuple=False).flatten().tolist():
            active_count = int(active[env_id].sum().item())
            inside_count = int((inside_bin[env_id] & active[env_id]).sum().item())
            metrics[env_id] = {
                "name": "objects_inside_bin",
                "value": float(inside_count),
                "threshold": float(active_count),
                "margin": float(inside_count - active_count),
                "instant": inside_count == active_count,
            }
        target_ids = _target_indices(env)
        target_inside = inside_bin[
            torch.arange(env.num_envs, device=env.device),
            target_ids,
        ]
        met |= target_inside & named_bin_tasks
        for env_id in torch.nonzero(named_bin_tasks, as_tuple=False).flatten().tolist():
            is_inside = bool(target_inside[env_id].item())
            metrics[env_id] = {
                "name": "target_inside_bin",
                "value": 1.0 if is_inside else 0.0,
                "threshold": 1.0,
                "margin": 0.0 if is_inside else -1.0,
                "instant": is_inside,
                "target_object_id": int(target_ids[env_id].item()),
            }

    if TASK_NEXT_TO in families:
        next_tasks = _task_is(env, TASK_NEXT_TO)
        distances = _pairwise_object_surface_distance(
            env,
            object_asset_names,
            positions,
            yaws,
            _target_indices(env),
            _referent_indices(env)[:, 0],
            next_tasks,
        )
        met |= (distances <= SPATIAL_SUCCESS_DISTANCE_M) & next_tasks
        for env_id in torch.nonzero(next_tasks, as_tuple=False).flatten().tolist():
            value = float(distances[env_id].item())
            metrics[env_id] = {
                "name": "target_referent_surface_distance_m",
                "value": value,
                "threshold": SPATIAL_SUCCESS_DISTANCE_M,
                "margin": SPATIAL_SUCCESS_DISTANCE_M - value,
                "instant": value <= SPATIAL_SUCCESS_DISTANCE_M,
            }

    if TASK_BETWEEN in families:
        between_tasks = _task_is(env, TASK_BETWEEN)
        target_ids = _target_indices(env)
        refs = _referent_indices(env)
        distance_a = _pairwise_object_surface_distance(
            env, object_asset_names, positions, yaws, target_ids, refs[:, 0], between_tasks
        )
        distance_b = _pairwise_object_surface_distance(
            env, object_asset_names, positions, yaws, target_ids, refs[:, 1], between_tasks
        )
        total = distance_a + distance_b
        fraction = torch.where(
            torch.isfinite(total) & (total > 0.0), distance_a / total, torch.full_like(total, 0.5)
        )
        centered = (fraction >= BETWEEN_CENTER_FRACTION_MIN) & (fraction <= BETWEEN_CENTER_FRACTION_MAX)
        target = _gather_by_index(positions, target_ids)[:, :2]
        ref_a = _gather_by_index(positions, refs[:, 0])[:, :2]
        ref_b = _gather_by_index(positions, refs[:, 1])[:, :2]
        segment = ref_b - ref_a
        segment_len_sq = torch.clamp(torch.sum(segment * segment, dim=1), min=1.0e-6)
        projection_fraction = torch.sum((target - ref_a) * segment, dim=1) / segment_len_sq
        projection = ref_a + projection_fraction.unsqueeze(1) * segment
        perpendicular = torch.linalg.vector_norm(target - projection, dim=1)
        between_met = centered & (perpendicular <= BETWEEN_LINE_TOLERANCE_M) & between_tasks
        met |= between_met
        for env_id in torch.nonzero(between_tasks, as_tuple=False).flatten().tolist():
            value = float(perpendicular[env_id].item())
            frac = float(fraction[env_id].item())
            centered_margin = min(
                frac - BETWEEN_CENTER_FRACTION_MIN,
                BETWEEN_CENTER_FRACTION_MAX - frac,
            )
            line_margin = BETWEEN_LINE_TOLERANCE_M - value
            metrics[env_id] = {
                "name": "between_geometry",
                "value": value,
                "threshold": BETWEEN_LINE_TOLERANCE_M,
                "margin": min(line_margin, centered_margin),
                "instant": bool(between_met[env_id].item()),
                "perpendicular_distance_m": value,
                "center_fraction": frac,
                "center_fraction_min": BETWEEN_CENTER_FRACTION_MIN,
                "center_fraction_max": BETWEEN_CENTER_FRACTION_MAX,
            }

    if TASK_MOVE in families:
        move_tasks = _task_is(env, TASK_MOVE)
        gap, progress, lateral, _ = _move_boundary_distance(env, object_asset_names, table_bounds, step_state)
        has_boundary = (env._so101_move_boundary_ids >= 0) & torch.isfinite(gap)
        reached = torch.where(
            has_boundary,
            gap < MOVE_BOUNDARY_SUCCESS_DISTANCE_M,
            progress >= MOVE_NO_BOUNDARY_MIN_PROGRESS_M,
        )
        move_met = (
            (progress > 0.0) & reached & (lateral <= move_straightness_tolerance) & move_tasks
        )
        met |= move_met
        for env_id in torch.nonzero(move_tasks, as_tuple=False).flatten().tolist():
            boundary = bool(has_boundary[env_id].item())
            value = float(gap[env_id].item()) if boundary else float(progress[env_id].item())
            threshold = MOVE_BOUNDARY_SUCCESS_DISTANCE_M if boundary else MOVE_NO_BOUNDARY_MIN_PROGRESS_M
            goal_margin = (
                MOVE_BOUNDARY_SUCCESS_DISTANCE_M - value
                if boundary
                else value - MOVE_NO_BOUNDARY_MIN_PROGRESS_M
            )
            lateral_margin = move_straightness_tolerance - float(lateral[env_id].item())
            metrics[env_id] = {
                "name": "move_boundary_gap_m" if boundary else "move_directional_progress_m",
                "value": value,
                "threshold": threshold,
                "margin": min(goal_margin, lateral_margin),
                "instant": bool(move_met[env_id].item()),
                "directional_progress_m": float(progress[env_id].item()),
                "lateral_error_m": float(lateral[env_id].item()),
                "boundary_gap_m": float(gap[env_id].item()),
                "has_boundary": boundary,
                "overshot": bool(overshot[env_id].item()),
            }

    return _PostmortemGoalStep(met, overshot, inside_bin, metrics)


def task_condition_metrics(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    bin_name: str,
    table_bounds: dict[str, tuple[float, float]] | None = None,
    step_state: _TerminationStepState | None = None,
    confirm_time_s: float = DEFAULT_SUCCESS_CONFIRM_TIME_S,
    move_straightness_tolerance: float = MOVE_STRAIGHTNESS_TOLERANCE_M,
    move_past_boundary_tolerance: float = MOVE_PAST_BOUNDARY_TOLERANCE_M,
    **_ignored: object,
) -> list[dict[str, object]]:
    """Return typed current goal metrics without advancing confirmation counters.

    This is the numeric counterpart to :class:`TaskConditionDiagnostic`; it is
    intended for outcome collection and offline rescoring.  Geometry caches may
    be populated, but success/failure/contact counters are never mutated.
    """

    if table_bounds is None:
        table_bounds = {"x": (0.08, 0.45), "y": (-0.20, 0.20)}
    if step_state is None:
        step_state = _termination_step_state(env, object_asset_names)
    goal = _postmortem_goal_step(
        env,
        object_asset_names,
        bin_name,
        table_bounds,
        step_state,
        move_straightness_tolerance=move_straightness_tolerance,
        move_past_boundary_tolerance=move_past_boundary_tolerance,
    )
    counters = _task_success_counters(env)
    configured_required = getattr(env, "_so101_success_confirmation_required_steps", None)
    confirmed_current = getattr(env, "_so101_success_confirmed_current", None)
    active = _active_mask(env, object_asset_names)
    task_families = getattr(env, "_so101_task_family", [TASK_BIN] * env.num_envs)
    records: list[dict[str, object]] = []
    for env_id in range(env.num_envs):
        if isinstance(configured_required, torch.Tensor):
            required_steps = int(configured_required[env_id].item())
        elif isinstance(configured_required, int):
            required_steps = configured_required
        else:
            required_steps = _confirmation_steps(env, confirm_time_s)
        metric = dict(goal.metrics[env_id])
        for key, value in tuple(metric.items()):
            if isinstance(value, float) and not math.isfinite(value):
                metric[key] = None
        counter = int(counters[env_id].item())
        actual_confirmed = bool(
            isinstance(confirmed_current, torch.Tensor)
            and bool(confirmed_current[env_id].item())
        )
        metric.update(
            {
                "instant": bool(goal.met[env_id].item()),
                "counter": counter,
                "required_steps": required_steps,
                "geometry_confirmed": bool(goal.met[env_id].item()) and counter >= required_steps,
                "success_confirmed_current": actual_confirmed,
            }
        )
        records.append(
            {
                "env_id": env_id,
                "task_family": task_families[env_id],
                "goal": metric,
                "overshot": bool(goal.overshot[env_id].item()),
                "inside_bin_object_ids": tuple(
                    object_id
                    for object_id in torch.nonzero(
                        goal.inside_bin[env_id] & active[env_id], as_tuple=False
                    ).flatten().tolist()
                ),
            }
        )
    return records


def _ensure_postmortem_buffers(env: ManagerBasedRLEnv, num_objects: int) -> None:
    """Create tracker buffers without requiring reset-event changes."""

    specs = {
        "_so101_pm_last_episode_step": ((env.num_envs,), torch.long, -1),
        "_so101_pm_has_prev_sample": ((env.num_envs,), torch.bool, False),
        "_so101_pm_prev_object_pos_w": ((env.num_envs, num_objects, 3), torch.float32, 0.0),
        "_so101_pm_prev_ee_pos_w": ((env.num_envs, 3), torch.float32, 0.0),
        "_so101_pm_attempt_counts_all": ((env.num_envs, num_objects), torch.long, 0),
        "_so101_pm_unassociated_attempt_counts": ((env.num_envs,), torch.long, 0),
        "_so101_pm_acquisition_counts": ((env.num_envs, num_objects), torch.long, 0),
        "_so101_pm_acquired_objects": ((env.num_envs, num_objects), torch.bool, False),
        "_so101_pm_manipulated_objects": ((env.num_envs, num_objects), torch.bool, False),
        "_so101_pm_interaction_steps": ((env.num_envs, num_objects), torch.long, 0),
        "_so101_pm_max_interaction_steps": ((env.num_envs, num_objects), torch.long, 0),
        "_so101_pm_interaction_segment_transport_m": (
            (env.num_envs, num_objects),
            torch.float32,
            0.0,
        ),
        "_so101_pm_max_interaction_segment_transport_m": (
            (env.num_envs, num_objects),
            torch.float32,
            0.0,
        ),
        "_so101_pm_associated_transport_m": ((env.num_envs, num_objects), torch.float32, 0.0),
        "_so101_pm_post_acquisition_transport_m": (
            (env.num_envs, num_objects),
            torch.float32,
            0.0,
        ),
        "_so101_pm_max_object_displacement_m": ((env.num_envs, num_objects), torch.float32, 0.0),
        "_so101_pm_lift_hold_steps": ((env.num_envs, num_objects), torch.long, 0),
        "_so101_pm_max_lift_hold_steps": ((env.num_envs, num_objects), torch.long, 0),
        "_so101_pm_candidate_object_ids": ((env.num_envs,), torch.long, -1),
        "_so101_pm_candidate_transport_m": ((env.num_envs,), torch.float32, 0.0),
        "_so101_pm_candidate_segment_transport_m": ((env.num_envs,), torch.float32, 0.0),
        "_so101_pm_candidate_comotion_steps": ((env.num_envs,), torch.long, 0),
        "_so101_pm_candidate_max_comotion_steps": ((env.num_envs,), torch.long, 0),
        "_so101_pm_candidate_acquired": ((env.num_envs,), torch.bool, False),
        "_so101_pm_candidate_loss_steps": ((env.num_envs,), torch.long, 0),
        "_so101_pm_drop_counts": ((env.num_envs, num_objects), torch.long, 0),
        "_so101_pm_release_counts": ((env.num_envs, num_objects), torch.long, 0),
        "_so101_pm_goal_hold_steps": ((env.num_envs,), torch.long, 0),
        "_so101_pm_max_goal_hold_steps": ((env.num_envs,), torch.long, 0),
        "_so101_pm_goal_ever_reached": ((env.num_envs,), torch.bool, False),
        "_so101_pm_final_goal_met": ((env.num_envs,), torch.bool, False),
        "_so101_pm_max_success_counter": ((env.num_envs,), torch.long, 0),
        "_so101_pm_ever_overshot": ((env.num_envs,), torch.bool, False),
        "_so101_pm_final_overshot": ((env.num_envs,), torch.bool, False),
        "_so101_pm_ever_inside_bin": ((env.num_envs, num_objects), torch.bool, False),
        "_so101_pm_best_goal_margin": ((env.num_envs,), torch.float32, float("-inf")),
        "_so101_pm_best_goal_step": ((env.num_envs,), torch.long, -1),
    }
    for name, (shape, dtype, fill) in specs.items():
        value = getattr(env, name, None)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            setattr(env, name, torch.full(shape, fill, dtype=dtype, device=env.device))
    attempts = getattr(env, "_so101_pm_attempt_history", None)
    if not isinstance(attempts, list) or len(attempts) != env.num_envs:
        env._so101_pm_attempt_history = [[] for _ in range(env.num_envs)]
    metrics = getattr(env, "_so101_postmortem_goal_metrics", None)
    if not isinstance(metrics, list) or len(metrics) != env.num_envs:
        env._so101_postmortem_goal_metrics = [{} for _ in range(env.num_envs)]


def _reset_postmortem_rows(env: ManagerBasedRLEnv, reset: torch.Tensor) -> None:
    """Reset evidence rows when an environment's episode counter rolls back."""

    if not torch.any(reset):
        return
    for name, value in vars(env).items():
        if not name.startswith("_so101_pm_") or not isinstance(value, torch.Tensor):
            continue
        if value.shape[0] != env.num_envs or name == "_so101_pm_last_episode_step":
            continue
        if value.dtype == torch.bool:
            value[reset] = False
        elif value.dtype.is_floating_point:
            value[reset] = float("-inf") if name == "_so101_pm_best_goal_margin" else 0.0
        else:
            value[reset] = -1 if name in {"_so101_pm_candidate_object_ids", "_so101_pm_best_goal_step"} else 0
    for env_id in torch.nonzero(reset, as_tuple=False).flatten().tolist():
        env._so101_pm_attempt_history[env_id] = []
        env._so101_postmortem_goal_metrics[env_id] = {}
    for name in ("_so101_success_ever_confirmed", "_so101_success_confirmed_current"):
        value = getattr(env, name, None)
        if isinstance(value, torch.Tensor) and tuple(value.shape) == (env.num_envs,):
            value[reset] = False


def _finish_postmortem_attempt(
    env: ManagerBasedRLEnv,
    env_id: int,
    episode_step: int,
    *,
    released: bool = False,
    dropped: bool = False,
) -> None:
    candidate_id = int(env._so101_pm_candidate_object_ids[env_id].item())
    if candidate_id < 0:
        return
    history = env._so101_pm_attempt_history[env_id]
    if history:
        attempt = history[-1]
        attempt["end_step"] = episode_step
        attempt["released"] = released
        attempt["dropped"] = dropped
        attempt["associated_transport_m"] = float(env._so101_pm_candidate_transport_m[env_id].item())
        attempt["max_comotion_steps"] = int(env._so101_pm_candidate_max_comotion_steps[env_id].item())
    if released and bool(env._so101_pm_candidate_acquired[env_id].item()):
        env._so101_pm_release_counts[env_id, candidate_id] += 1
    env._so101_pm_candidate_object_ids[env_id] = -1
    env._so101_pm_candidate_transport_m[env_id] = 0.0
    env._so101_pm_candidate_segment_transport_m[env_id] = 0.0
    env._so101_pm_candidate_comotion_steps[env_id] = 0
    env._so101_pm_candidate_max_comotion_steps[env_id] = 0
    env._so101_pm_candidate_acquired[env_id] = False
    env._so101_pm_candidate_loss_steps[env_id] = 0


def _update_postmortem_evidence(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    bin_name: str,
    table_bounds: dict[str, tuple[float, float]],
    step_state: _TerminationStepState,
    grasp_step: _GraspStepEvidence,
    baseline_recorded: torch.Tensor,
    object_distance_threshold: float,
    move_straightness_tolerance: float = MOVE_STRAIGHTNESS_TOLERANCE_M,
    move_past_boundary_tolerance: float = MOVE_PAST_BOUNDARY_TOLERANCE_M,
) -> None:
    """Accumulate idempotent temporal evidence for postmortem attribution.

    Object acquisition requires sustained near-EE co-motion (or sustained lift
    while associated with a jaw-close).  Open-jaw pushing is still recognized
    through the same co-motion test, so valid move/next-to/between strategies do
    not become ``failed_grasp`` merely because the object was never lifted.
    """

    num_objects = len(object_asset_names)
    _ensure_postmortem_buffers(env, num_objects)
    previous_steps = env._so101_pm_last_episode_step
    episode_buf = getattr(env, "episode_length_buf", None)
    if isinstance(episode_buf, torch.Tensor):
        episode_steps = episode_buf.to(device=env.device, dtype=torch.long)
        reset = (previous_steps >= 0) & (episode_steps < previous_steps)
        needs_update = episode_steps != previous_steps
    else:
        episode_steps = previous_steps + 1
        reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        needs_update = torch.ones_like(reset)
    _reset_postmortem_rows(env, reset)
    if not torch.any(needs_update):
        return

    configured_success_straightness = float(
        getattr(
            env,
            "_so101_success_move_straightness_tolerance",
            move_straightness_tolerance,
        )
    )
    goal = _postmortem_goal_step(
        env,
        object_asset_names,
        bin_name,
        table_bounds,
        step_state,
        move_straightness_tolerance=configured_success_straightness,
        move_past_boundary_tolerance=move_past_boundary_tolerance,
    )
    active = _active_mask(env, object_asset_names)
    policy_control_active = getattr(env, "_so101_policy_control_active", None)
    robot_started = getattr(env, "_so101_robot_started_moving", None)
    attribution_active = (
        policy_control_active.to(device=env.device, dtype=torch.bool)
        if isinstance(policy_control_active, torch.Tensor)
        else robot_started.to(device=env.device, dtype=torch.bool)
        if isinstance(robot_started, torch.Tensor)
        else baseline_recorded
    )
    attribution_update = needs_update & attribution_active
    object_pos_w = step_state.positions
    ee_pos_w = grasp_step.ee_pos_w
    has_prev = env._so101_pm_has_prev_sample & needs_update
    object_delta = object_pos_w - env._so101_pm_prev_object_pos_w
    ee_delta = ee_pos_w - env._so101_pm_prev_ee_pos_w
    object_step_m = torch.linalg.vector_norm(object_delta, dim=2)
    ee_step_m = torch.linalg.vector_norm(ee_delta, dim=1)
    object_ee_distance = torch.linalg.vector_norm(object_pos_w - ee_pos_w.unsqueeze(1), dim=2)
    dot = torch.sum(object_delta * ee_delta.unsqueeze(1), dim=2)
    denom = torch.clamp(object_step_m * ee_step_m.unsqueeze(1), min=1.0e-9)
    cosine = dot / denom
    motion_ratio = object_step_m / torch.clamp(ee_step_m.unsqueeze(1), min=1.0e-9)
    co_motion = (
        has_prev.unsqueeze(1)
        & baseline_recorded.unsqueeze(1)
        & attribution_active.unsqueeze(1)
        & active
        & (object_ee_distance <= object_distance_threshold * 1.5)
        & (object_step_m >= POSTMORTEM_MIN_STEP_MOTION_M)
        & (ee_step_m.unsqueeze(1) >= POSTMORTEM_MIN_STEP_MOTION_M)
        & (cosine >= 0.5)
        & (motion_ratio >= 0.1)
        & (motion_ratio <= 3.0)
    )

    env._so101_pm_interaction_steps = torch.where(
        needs_update.unsqueeze(1) & co_motion,
        env._so101_pm_interaction_steps + 1,
        torch.where(
            needs_update.unsqueeze(1),
            torch.zeros_like(env._so101_pm_interaction_steps),
            env._so101_pm_interaction_steps,
        ),
    )
    env._so101_pm_max_interaction_steps = torch.maximum(
        env._so101_pm_max_interaction_steps, env._so101_pm_interaction_steps
    )
    env._so101_pm_interaction_segment_transport_m = torch.where(
        needs_update.unsqueeze(1) & co_motion,
        env._so101_pm_interaction_segment_transport_m + object_step_m,
        torch.where(
            needs_update.unsqueeze(1),
            torch.zeros_like(env._so101_pm_interaction_segment_transport_m),
            env._so101_pm_interaction_segment_transport_m,
        ),
    )
    env._so101_pm_max_interaction_segment_transport_m = torch.maximum(
        env._so101_pm_max_interaction_segment_transport_m,
        env._so101_pm_interaction_segment_transport_m,
    )
    env._so101_pm_associated_transport_m += torch.where(
        needs_update.unsqueeze(1) & co_motion, object_step_m, torch.zeros_like(object_step_m)
    )
    meaningfully_manipulated = (
        (env._so101_pm_max_interaction_steps >= POSTMORTEM_ACQUISITION_HOLD_STEPS)
        & (
            env._so101_pm_max_interaction_segment_transport_m
            >= POSTMORTEM_MIN_ASSOCIATED_TRANSPORT_M
        )
    )
    env._so101_pm_manipulated_objects |= meaningfully_manipulated

    if hasattr(env, "_so101_failure_object_pos_w"):
        displacement = torch.linalg.vector_norm(
            object_pos_w[..., :2] - env._so101_failure_object_pos_w[..., :2], dim=2
        )
        env._so101_pm_max_object_displacement_m = torch.where(
            baseline_recorded.unsqueeze(1) & attribution_active.unsqueeze(1) & active,
            torch.maximum(env._so101_pm_max_object_displacement_m, displacement),
            env._so101_pm_max_object_displacement_m,
        )
        lift_now = object_pos_w[..., 2] - env._so101_failure_object_pos_w[..., 2]
        lifted = (
            baseline_recorded.unsqueeze(1)
            & attribution_active.unsqueeze(1)
            & active
            & (lift_now >= LIFT_OFF_GROUND_LIMIT_M)
        )
        env._so101_pm_lift_hold_steps = torch.where(
            needs_update.unsqueeze(1) & lifted,
            env._so101_pm_lift_hold_steps + 1,
            torch.where(
                needs_update.unsqueeze(1),
                torch.zeros_like(env._so101_pm_lift_hold_steps),
                env._so101_pm_lift_hold_steps,
            ),
        )
        env._so101_pm_max_lift_hold_steps = torch.maximum(
            env._so101_pm_max_lift_hold_steps, env._so101_pm_lift_hold_steps
        )

    current_success_counter = _task_success_counters(env)
    env._so101_pm_max_success_counter = torch.maximum(
        env._so101_pm_max_success_counter, current_success_counter
    )
    env._so101_pm_goal_hold_steps = torch.where(
        attribution_update & goal.met,
        env._so101_pm_goal_hold_steps + 1,
        torch.where(
            attribution_update,
            torch.zeros_like(env._so101_pm_goal_hold_steps),
            env._so101_pm_goal_hold_steps,
        ),
    )
    env._so101_pm_max_goal_hold_steps = torch.maximum(
        env._so101_pm_max_goal_hold_steps, env._so101_pm_goal_hold_steps
    )
    env._so101_pm_goal_ever_reached |= attribution_update & goal.met
    env._so101_pm_final_goal_met = torch.where(
        attribution_update, goal.met, env._so101_pm_final_goal_met
    )
    env._so101_pm_ever_overshot |= attribution_update & goal.overshot
    env._so101_pm_final_overshot = torch.where(
        attribution_update, goal.overshot, env._so101_pm_final_overshot
    )
    env._so101_pm_ever_inside_bin |= attribution_update.unsqueeze(1) & goal.inside_bin & active
    for env_id in torch.nonzero(attribution_update, as_tuple=False).flatten().tolist():
        metric = dict(goal.metrics[env_id])
        for key, value in tuple(metric.items()):
            if isinstance(value, float) and not math.isfinite(value):
                metric[key] = None
        metric["counter"] = int(current_success_counter[env_id].item())
        env._so101_postmortem_goal_metrics[env_id] = metric
        margin = metric.get("margin")
        if isinstance(margin, (int, float)) and math.isfinite(float(margin)):
            if float(margin) > float(env._so101_pm_best_goal_margin[env_id].item()):
                env._so101_pm_best_goal_margin[env_id] = float(margin)
                env._so101_pm_best_goal_step[env_id] = int(episode_steps[env_id].item())

    # Start attempts on any active object, not only the instructed target.  The
    # legacy max-attempt rule deliberately continues to count only eligible
    # task objects; this all-object history is diagnostic evidence only.
    for env_id in torch.nonzero(
        attribution_update & grasp_step.close_cycle, as_tuple=False
    ).flatten().tolist():
        step = int(episode_steps[env_id].item())
        _finish_postmortem_attempt(env, env_id, step)
        if not bool(grasp_step.grasp_started[env_id].item()):
            env._so101_pm_unassociated_attempt_counts[env_id] += 1
            continue
        object_id = int(grasp_step.nearest_active_object_ids[env_id].item())
        env._so101_pm_attempt_counts_all[env_id, object_id] += 1
        env._so101_pm_candidate_object_ids[env_id] = object_id
        env._so101_pm_attempt_history[env_id].append(
            {
                "attempt_id": len(env._so101_pm_attempt_history[env_id]),
                "object_id": object_id,
                "close_step": step,
                "end_step": None,
                "nearest_distance_m": float(grasp_step.nearest_active_distance_m[env_id].item()),
                "acquired": False,
                "acquisition_step": None,
                "released": False,
                "dropped": False,
                "associated_transport_m": 0.0,
                "max_comotion_steps": 0,
                "goal_met_during_attempt": bool(goal.met[env_id].item()),
            }
        )

    for env_id in torch.nonzero(attribution_update, as_tuple=False).flatten().tolist():
        step = int(episode_steps[env_id].item())
        candidate_id = int(env._so101_pm_candidate_object_ids[env_id].item())
        if candidate_id < 0:
            continue
        if bool(grasp_step.jaw_is_open[env_id].item()):
            _finish_postmortem_attempt(env, env_id, step, released=True)
            continue

        candidate_comoving = bool(co_motion[env_id, candidate_id].item()) and not bool(
            grasp_step.close_cycle[env_id].item()
        )
        if candidate_comoving:
            env._so101_pm_candidate_comotion_steps[env_id] += 1
            env._so101_pm_candidate_transport_m[env_id] += object_step_m[env_id, candidate_id]
            env._so101_pm_candidate_segment_transport_m[env_id] += object_step_m[
                env_id, candidate_id
            ]
        else:
            env._so101_pm_candidate_comotion_steps[env_id] = 0
            env._so101_pm_candidate_segment_transport_m[env_id] = 0.0
        env._so101_pm_candidate_max_comotion_steps[env_id] = torch.maximum(
            env._so101_pm_candidate_max_comotion_steps[env_id],
            env._so101_pm_candidate_comotion_steps[env_id],
        )
        sustained_comotion = (
            int(env._so101_pm_candidate_comotion_steps[env_id].item())
            >= POSTMORTEM_ACQUISITION_HOLD_STEPS
            and float(env._so101_pm_candidate_segment_transport_m[env_id].item())
            >= POSTMORTEM_ACQUISITION_TRANSPORT_M
        )
        sustained_associated_lift = (
            int(env._so101_pm_lift_hold_steps[env_id, candidate_id].item())
            >= POSTMORTEM_ACQUISITION_HOLD_STEPS
            and float(object_ee_distance[env_id, candidate_id].item()) <= object_distance_threshold * 1.5
        )
        newly_acquired = (sustained_comotion or sustained_associated_lift) and not bool(
            env._so101_pm_candidate_acquired[env_id].item()
        )
        if newly_acquired:
            env._so101_pm_candidate_acquired[env_id] = True
            env._so101_pm_acquired_objects[env_id, candidate_id] = True
            env._so101_pm_manipulated_objects[env_id, candidate_id] = True
            env._so101_pm_acquisition_counts[env_id, candidate_id] += 1
            attempt = env._so101_pm_attempt_history[env_id][-1]
            attempt["acquired"] = True
            attempt["acquisition_step"] = step
        if bool(env._so101_pm_candidate_acquired[env_id].item()) and candidate_comoving:
            env._so101_pm_post_acquisition_transport_m[env_id, candidate_id] += object_step_m[
                env_id, candidate_id
            ]

        attempt = env._so101_pm_attempt_history[env_id][-1]
        attempt["goal_met_during_attempt"] |= bool(goal.met[env_id].item())
        attempt["associated_transport_m"] = float(env._so101_pm_candidate_transport_m[env_id].item())
        attempt["max_comotion_steps"] = int(env._so101_pm_candidate_max_comotion_steps[env_id].item())

        # World-frame relative-vector drift is not a reliable grasp-loss test:
        # coordinated wrist/object rotation changes that vector even for a
        # rigid grasp.  Require sustained spatial separation corroborated by
        # loss of co-motion.
        association_lost = (
            float(object_ee_distance[env_id, candidate_id].item())
            > object_distance_threshold * 2.0
            and not candidate_comoving
        )
        if bool(env._so101_pm_candidate_acquired[env_id].item()) and association_lost:
            env._so101_pm_candidate_loss_steps[env_id] += 1
        else:
            env._so101_pm_candidate_loss_steps[env_id] = 0
        if int(env._so101_pm_candidate_loss_steps[env_id].item()) >= POSTMORTEM_DROP_HOLD_STEPS:
            env._so101_pm_drop_counts[env_id, candidate_id] += 1
            _finish_postmortem_attempt(env, env_id, step, dropped=True)

    env._so101_pm_prev_object_pos_w = torch.where(
        needs_update.view(-1, 1, 1), object_pos_w, env._so101_pm_prev_object_pos_w
    )
    env._so101_pm_prev_ee_pos_w = torch.where(
        needs_update.unsqueeze(1), ee_pos_w, env._so101_pm_prev_ee_pos_w
    )
    env._so101_pm_has_prev_sample |= needs_update
    env._so101_pm_last_episode_step = torch.where(needs_update, episode_steps, previous_steps)
    env._so101_pm_tracker_available = True


def _legacy_postmortem_type(
    task_family: str,
    target_lift: float,
    max_non_target_lift: float,
    lift_threshold: float,
) -> str:
    if task_family == TASK_BIN:
        return POSTMORTEM_NOT_APPLICABLE
    if target_lift >= lift_threshold:
        return POSTMORTEM_PLACEMENT
    if max_non_target_lift >= lift_threshold:
        return POSTMORTEM_SEMANTIC
    return POSTMORTEM_FAILED_GRASP


def _attempt_diagnostics(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    env_id: int,
    target_id: int,
) -> tuple[ManipulationAttemptDiagnostic, ...]:
    result = []
    for attempt in getattr(env, "_so101_pm_attempt_history", [[] for _ in range(env.num_envs)])[env_id]:
        object_id = int(attempt["object_id"])
        result.append(
            ManipulationAttemptDiagnostic(
                attempt_id=int(attempt["attempt_id"]),
                object_id=object_id,
                object_name=_debug_object_name(env, object_asset_names, env_id, object_id),
                is_target=object_id == target_id,
                close_step=int(attempt["close_step"]),
                end_step=None if attempt["end_step"] is None else int(attempt["end_step"]),
                nearest_distance_m=float(attempt["nearest_distance_m"]),
                acquired=bool(attempt["acquired"]),
                acquisition_step=(
                    None if attempt["acquisition_step"] is None else int(attempt["acquisition_step"])
                ),
                released=bool(attempt["released"]),
                dropped=bool(attempt["dropped"]),
                associated_transport_m=float(attempt["associated_transport_m"]),
                max_comotion_steps=int(attempt["max_comotion_steps"]),
                goal_met_during_attempt=bool(attempt["goal_met_during_attempt"]),
            )
        )
    return tuple(result)


def benchmark_postmortem_failure_diagnostics(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    lift_threshold: float = LIFT_OFF_GROUND_LIMIT_M,
    confirm_time_s: float = DEFAULT_SUCCESS_CONFIRM_TIME_S,
) -> list[PostmortemFailureDiagnostic]:
    """Attribute outcomes from attempts, acquisition, co-motion, progress, and goal history.

    The classifier uses lift only as supporting evidence.  If called on an old
    environment/log that has no version-2 tracker, it returns the historical
    lift-only ``failure_type`` with low confidence so existing consumers remain
    functional and can identify the fallback from ``classification_version``.
    """

    diagnostics: list[PostmortemFailureDiagnostic] = []
    task_families = getattr(env, "_so101_task_family", None)
    max_lift = getattr(env, "_so101_max_object_lift", None)
    active = _active_mask(env, object_asset_names)
    target_ids = _target_indices(env)
    tracker_available = bool(getattr(env, "_so101_pm_tracker_available", False))
    success_counters = _task_success_counters(env)
    default_required_success_steps = _confirmation_steps(env, confirm_time_s) if tracker_available else 0
    configured_required_steps = getattr(env, "_so101_success_confirmation_required_steps", None)
    confirmed_current_state = getattr(env, "_so101_success_confirmed_current", None)
    ever_confirmed_state = getattr(env, "_so101_success_ever_confirmed", None)

    for env_id in range(env.num_envs):
        task_family = task_families[env_id] if task_families is not None else TASK_BIN
        target_id = int(target_ids[env_id].item())
        target_object = _debug_object_name(env, object_asset_names, env_id, target_id)
        target_lift = 0.0 if max_lift is None else float(max_lift[env_id, target_id].item())
        max_non_target_lift = 0.0
        lifted_wrong_object = "none"
        for object_id in torch.nonzero(active[env_id], as_tuple=False).flatten().tolist():
            if object_id == target_id:
                continue
            object_lift = 0.0 if max_lift is None else float(max_lift[env_id, object_id].item())
            if object_lift > max_non_target_lift:
                max_non_target_lift = object_lift
                lifted_wrong_object = (
                    _debug_object_name(env, object_asset_names, env_id, object_id)
                    if object_lift >= lift_threshold
                    else "none"
                )
        legacy_type = _legacy_postmortem_type(
            task_family, target_lift, max_non_target_lift, lift_threshold
        )

        if not tracker_available:
            diagnostics.append(
                PostmortemFailureDiagnostic(
                    env_id=env_id,
                    task_family=task_family,
                    failure_type=legacy_type,
                    target_object=target_object,
                    target_lift_m=target_lift,
                    lifted_wrong_object=lifted_wrong_object,
                    max_non_target_lift_m=max_non_target_lift,
                    lift_threshold_m=lift_threshold,
                    classification_version=1,
                    legacy_failure_type=legacy_type,
                    confidence=0.25,
                    rationale="Legacy fallback: temporal manipulation evidence was unavailable.",
                    target_semantics_applicable=task_family != TASK_BIN,
                    evidence={"temporal_evidence_available": False},
                )
            )
            continue

        attempts_all = env._so101_pm_attempt_counts_all[env_id]
        acquired_all = env._so101_pm_acquired_objects[env_id]
        manipulated_all = env._so101_pm_manipulated_objects[env_id]
        target_attempts = int(attempts_all[target_id].item())
        wrong_mask = active[env_id].clone()
        if task_family == TASK_BIN:
            # Every active bin object is an intended target.  Treating the
            # arbitrary metadata target slot as the only correct one produced
            # spurious ``wrong_object_targeted`` evidence for valid bin work.
            wrong_mask.zero_()
            associated_attempts = int(attempts_all[active[env_id]].sum().item())
        else:
            wrong_mask[target_id] = False
            associated_attempts = target_attempts + int(attempts_all[wrong_mask].sum().item())
        wrong_attempts = int(attempts_all[wrong_mask].sum().item())
        unassociated_attempts = int(env._so101_pm_unassociated_attempt_counts[env_id].item())
        target_acquired = bool(acquired_all[target_id].item())
        wrong_acquired = bool(torch.any(acquired_all & wrong_mask).item())
        target_manipulated = bool(manipulated_all[target_id].item())
        wrong_manipulated = bool(torch.any(manipulated_all & wrong_mask).item())
        target_displacement = float(env._so101_pm_max_object_displacement_m[env_id, target_id].item())
        target_transport = float(env._so101_pm_associated_transport_m[env_id, target_id].item())
        target_post_acquisition_transport = float(
            env._so101_pm_post_acquisition_transport_m[env_id, target_id].item()
        )
        target_drop_count = int(env._so101_pm_drop_counts[env_id, target_id].item())
        target_release_count = int(env._so101_pm_release_counts[env_id, target_id].item())
        goal_ever_reached = bool(env._so101_pm_goal_ever_reached[env_id].item())
        max_goal_hold = int(env._so101_pm_max_goal_hold_steps[env_id].item())
        final_goal_met = bool(env._so101_pm_final_goal_met[env_id].item())
        if isinstance(configured_required_steps, torch.Tensor):
            required_success_steps = int(configured_required_steps[env_id].item())
        elif isinstance(configured_required_steps, int):
            required_success_steps = configured_required_steps
        else:
            required_success_steps = default_required_success_steps
        max_success_counter = max(
            int(env._so101_pm_max_success_counter[env_id].item()),
            int(success_counters[env_id].item()),
        )
        # Once task_success has persisted its age/failure-gated result, that is
        # authoritative.  Raw counters alone can cross the threshold while an
        # age gate or live failure rule still makes success false.
        if isinstance(ever_confirmed_state, torch.Tensor):
            goal_ever_confirmed = bool(ever_confirmed_state[env_id].item())
        else:
            goal_ever_confirmed = max_success_counter >= required_success_steps
        if isinstance(confirmed_current_state, torch.Tensor):
            goal_confirmed_current = bool(confirmed_current_state[env_id].item())
        else:
            goal_confirmed_current = int(success_counters[env_id].item()) >= required_success_steps
        meaningful_goal_steps = max(
            2, math.ceil(POSTMORTEM_MEANINGFUL_GOAL_HOLD_TIME_S / _env_step_dt(env))
        )
        meaningful_goal_reached = max_goal_hold >= meaningful_goal_steps
        ever_overshot = bool(env._so101_pm_ever_overshot[env_id].item())
        final_overshot = bool(env._so101_pm_final_overshot[env_id].item())
        timeout_confirmation = bool(
            getattr(
                env,
                "_so101_timeout_success_confirmation_failed",
                torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            )[env_id].item()
        )
        failure_reason = (
            getattr(env, "_so101_failure_reasons", [FAILURE_REASON_NONE] * env.num_envs)[env_id]
        )
        raw_failure_state = getattr(env, "_so101_failure_conditions_active", None)
        raw_failure_active = bool(
            raw_failure_state[env_id].item()
            if isinstance(raw_failure_state, torch.Tensor)
            else False
        )
        raw_failure_reasons_state = getattr(
            env,
            "_so101_failure_conditions_active_reasons",
            [tuple() for _ in range(env.num_envs)],
        )
        raw_failure_reasons = tuple(raw_failure_reasons_state[env_id])
        live_overshot = FAILURE_REASON_MOVE_PAST_BOUNDARY in str(failure_reason).split("+")

        secondary: list[str] = []
        if wrong_acquired:
            secondary.append("wrong_object_acquired")
        elif wrong_manipulated:
            secondary.append("distractor_manipulated")
        if task_family != TASK_BIN and max_non_target_lift >= lift_threshold:
            secondary.append("wrong_object_lifted_or_disturbed")
        if task_family != TASK_BIN and target_drop_count:
            secondary.append("target_dropped")
        if task_family != TASK_BIN and target_release_count:
            secondary.append("target_released")
        if meaningful_goal_reached and not final_goal_met:
            secondary.append("goal_was_transient")
        if target_lift >= lift_threshold:
            secondary.append("target_lifted")
        if ever_overshot and not final_overshot:
            secondary.append("transient_overshoot_recovered")
        for reason in str(failure_reason).split("+"):
            if reason and reason != FAILURE_REASON_NONE:
                secondary.append(f"live_rule:{reason}")
        confirmed_failure_reasons = set(str(failure_reason).split("+"))
        for reason in raw_failure_reasons:
            if reason and reason not in confirmed_failure_reasons:
                secondary.append(f"raw_live_rule:{reason}")

        any_attempts = associated_attempts + unassociated_attempts
        live_rule_violation = failure_reason != FAILURE_REASON_NONE or raw_failure_active
        if goal_confirmed_current and final_goal_met and not live_rule_violation:
            failure_type = POSTMORTEM_NONE
            confidence = 1.0
            rationale = "The task goal reached its confirmation threshold and remained met."
        elif task_family == TASK_MOVE and (final_overshot or live_overshot):
            failure_type = POSTMORTEM_GOAL_OVERSHOT
            confidence = 0.99
            rationale = "The target crossed beyond the assigned move boundary."
        elif timeout_confirmation:
            failure_type = POSTMORTEM_TIMEOUT_DURING_CONFIRMATION
            confidence = 0.98
            rationale = "A goal confirmation was in progress at timeout and then breached."
        elif final_goal_met and live_rule_violation:
            failure_type = POSTMORTEM_GOAL_REACHED_WITH_RULE_VIOLATION
            confidence = 0.98
            active_rule_names = raw_failure_reasons or tuple(
                reason
                for reason in str(failure_reason).split("+")
                if reason and reason != FAILURE_REASON_NONE
            )
            active_rule_text = ", ".join(active_rule_names) if active_rule_names else "unknown rule"
            rationale = (
                "Final goal geometry was met, but an active benchmark rule made the state "
                f"ineligible for success: {active_rule_text}."
            )
        elif final_goal_met and not goal_confirmed_current:
            failure_type = POSTMORTEM_GOAL_REACHED_BUT_UNCONFIRMED
            confidence = 0.98
            rationale = (
                f"Final goal geometry was met, but confirmation held for only "
                f"{max_success_counter}/{required_success_steps} required steps."
            )
        elif task_family != TASK_BIN and target_drop_count > 0 and not final_goal_met:
            failure_type = POSTMORTEM_TARGET_DROPPED
            confidence = 0.88
            rationale = (
                "The acquired target became spatially separated from the end effector and lost sustained "
                "co-motion before reaching a stable goal."
            )
        elif task_family != TASK_BIN and target_release_count > 0 and not final_goal_met:
            failure_type = POSTMORTEM_TARGET_RELEASED_OUTSIDE_GOAL
            confidence = 0.9
            rationale = (
                "The acquired target was released, but the task goal was not met at the final state."
            )
        elif (
            goal_ever_confirmed
            and not goal_confirmed_current
        ) or (meaningful_goal_reached and not final_goal_met):
            failure_type = POSTMORTEM_GOAL_REACHED_BUT_UNSTABLE
            confidence = 0.95 if goal_ever_confirmed else 0.85
            rationale = "Goal geometry was sustained during the episode but was not preserved."
        elif task_family == TASK_BIN:
            active_count = int(active[env_id].sum().item())
            manipulated_count = int((manipulated_all & active[env_id]).sum().item())
            acquired_count = int((acquired_all & active[env_id]).sum().item())
            ever_inside_count = int(env._so101_pm_ever_inside_bin[env_id, active[env_id]].sum().item())
            if any_attempts == 0 and manipulated_count == 0:
                failure_type = POSTMORTEM_NO_MANIPULATION_ATTEMPT
                confidence = 0.92
                rationale = "No active object had a jaw-close attempt or sustained near-EE co-motion."
            elif manipulated_count == 0 and acquired_count == 0:
                failure_type = POSTMORTEM_OBJECT_ACQUISITION_FAILED
                confidence = min(0.9, 0.66 + 0.06 * any_attempts)
                rationale = (
                    f"{any_attempts} object-associated attempt(s) occurred, but no active bin object had "
                    "sustained acquisition or manipulation evidence."
                )
            elif active_count == 1:
                failure_type = POSTMORTEM_GOAL_NOT_REACHED
                confidence = 0.9
                rationale = "The required bin object was manipulated but never reached a confirmed bin goal."
            else:
                failure_type = POSTMORTEM_INCOMPLETE_MULTI_OBJECT_TASK
                confidence = 0.92
                rationale = (
                    f"Multi-object bin progress was incomplete: manipulated {manipulated_count}/{active_count}, "
                    f"acquired {acquired_count}/{active_count}, and ever inside {ever_inside_count}/{active_count}."
                )
        elif (wrong_acquired or wrong_manipulated) and not (target_acquired or target_manipulated):
            failure_type = POSTMORTEM_WRONG_OBJECT_TARGETED
            confidence = 0.86 if wrong_acquired else 0.76
            rationale = (
                "A non-target object had sustained gripper-associated motion while the target was never acquired "
                "or meaningfully manipulated."
            )
        elif target_attempts > 0 and not (target_acquired or target_manipulated):
            failure_type = POSTMORTEM_TARGET_ACQUISITION_FAILED
            confidence = min(0.92, 0.68 + 0.08 * target_attempts)
            rationale = "The gripper closed near the target, but no sustained target/EE co-motion followed."
        elif (
            target_acquired
            and target_post_acquisition_transport < POSTMORTEM_MIN_ASSOCIATED_TRANSPORT_M
        ):
            failure_type = POSTMORTEM_TRANSPORT_FAILED
            confidence = 0.86
            rationale = "The target was acquired but had less than 1 cm of post-acquisition co-motion."
        elif target_acquired or target_manipulated:
            failure_type = POSTMORTEM_GOAL_NOT_REACHED
            confidence = 0.88
            rationale = "The target was meaningfully manipulated, but task geometry never held at the goal."
        elif any_attempts == 0:
            failure_type = POSTMORTEM_NO_MANIPULATION_ATTEMPT
            confidence = 0.92
            rationale = "No jaw-close attempt or sustained near-EE object motion was observed."
        else:
            failure_type = POSTMORTEM_AMBIGUOUS
            confidence = 0.35
            rationale = "The available attempt and motion evidence conflicts or is too weak for a precise stage label."

        wrong_object_ids = torch.nonzero(wrong_mask, as_tuple=False).flatten().tolist()
        strongest_wrong_id = None
        if wrong_object_ids:
            strongest_wrong_id = max(
                wrong_object_ids,
                key=lambda object_id: (
                    int(acquired_all[object_id].item()),
                    float(env._so101_pm_associated_transport_m[env_id, object_id].item()),
                    int(attempts_all[object_id].item()),
                ),
            )
        active_ids = torch.nonzero(active[env_id], as_tuple=False).flatten().tolist()
        ever_inside_ids = [
            object_id for object_id in active_ids if bool(env._so101_pm_ever_inside_bin[env_id, object_id].item())
        ]
        never_manipulated_ids = [
            object_id for object_id in active_ids if not bool(manipulated_all[object_id].item())
        ]
        best_goal_margin = float(env._so101_pm_best_goal_margin[env_id].item())
        policy_control_state = getattr(env, "_so101_policy_control_active", None)
        if not isinstance(policy_control_state, torch.Tensor):
            policy_control_state = getattr(env, "_so101_robot_started_moving", None)
        evidence = {
            "temporal_evidence_available": True,
            "policy_control_active_at_classification": (
                bool(policy_control_state[env_id].item())
                if isinstance(policy_control_state, torch.Tensor)
                else None
            ),
            "policy_control_start_step": int(
                getattr(
                    env,
                    "_so101_robot_start_step",
                    torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
                )[env_id].item()
            ),
            "attempt_count_by_object": tuple(int(value) for value in attempts_all.tolist()),
            "active_object_attempt_count": associated_attempts,
            "unassociated_attempt_count": unassociated_attempts,
            "acquisition_count_by_object": tuple(
                int(value) for value in env._so101_pm_acquisition_counts[env_id].tolist()
            ),
            "manipulated_object_ids": tuple(
                object_id for object_id in active_ids if bool(manipulated_all[object_id].item())
            ),
            "max_displacement_m_by_object": tuple(
                float(value) for value in env._so101_pm_max_object_displacement_m[env_id].tolist()
            ),
            "associated_transport_m_by_object": tuple(
                float(value) for value in env._so101_pm_associated_transport_m[env_id].tolist()
            ),
            "maximum_contiguous_interaction_transport_m_by_object": tuple(
                float(value)
                for value in env._so101_pm_max_interaction_segment_transport_m[env_id].tolist()
            ),
            "post_acquisition_transport_m_by_object": tuple(
                float(value)
                for value in env._so101_pm_post_acquisition_transport_m[env_id].tolist()
            ),
            "max_lift_hold_steps_by_object": tuple(
                int(value) for value in env._so101_pm_max_lift_hold_steps[env_id].tolist()
            ),
            "drop_count_by_object": tuple(int(value) for value in env._so101_pm_drop_counts[env_id].tolist()),
            "release_count_by_object": tuple(
                int(value) for value in env._so101_pm_release_counts[env_id].tolist()
            ),
            "strongest_wrong_object_id": strongest_wrong_id,
            "strongest_wrong_object": (
                "none"
                if strongest_wrong_id is None
                else _debug_object_name(env, object_asset_names, env_id, strongest_wrong_id)
            ),
            "goal_required_confirmation_steps": required_success_steps,
            "goal_meaningful_hold_steps": meaningful_goal_steps,
            "max_success_counter": max_success_counter,
            "best_goal_margin": best_goal_margin if math.isfinite(best_goal_margin) else None,
            "best_goal_step": int(env._so101_pm_best_goal_step[env_id].item()),
            "current_goal_metric": dict(env._so101_postmortem_goal_metrics[env_id]),
            "ever_inside_bin_object_ids": tuple(ever_inside_ids),
            "never_manipulated_object_ids": tuple(never_manipulated_ids),
            "target_vs_wrong_semantics_applicable": task_family != TASK_BIN,
            "live_failure_reason": str(failure_reason),
            "raw_live_failure_active": raw_failure_active,
            "raw_live_failure_reasons": raw_failure_reasons,
        }
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
                legacy_failure_type=legacy_type,
                confidence=confidence,
                secondary_failure_types=tuple(dict.fromkeys(secondary)),
                rationale=rationale,
                target_semantics_applicable=task_family != TASK_BIN,
                target_attempt_count=target_attempts,
                wrong_object_attempt_count=wrong_attempts,
                unassociated_attempt_count=unassociated_attempts,
                target_acquired=target_acquired,
                wrong_object_acquired=wrong_acquired,
                target_manipulated=target_manipulated,
                target_max_displacement_m=target_displacement,
                target_associated_transport_m=target_transport,
                target_post_acquisition_transport_m=target_post_acquisition_transport,
                target_drop_count=target_drop_count,
                target_release_count=target_release_count,
                goal_ever_reached=goal_ever_reached,
                goal_ever_confirmed=goal_ever_confirmed,
                max_goal_hold_steps=max_goal_hold,
                final_goal_met=final_goal_met,
                attempts=_attempt_diagnostics(env, object_asset_names, env_id, target_id),
                evidence=evidence,
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
    object and another tabletop object. Passing a Move boundary is permitted;
    the boundary goal now has only a strict two-inch maximum signed gap.
    These live rules remain separate from the evidence-based postmortem stage
    attribution, which is diagnostic only and never terminates an episode.
    """

    if not hasattr(env, "_so101_initial_object_pos_w"):
        env._so101_failure_reasons = [FAILURE_REASON_NONE for _ in range(env.num_envs)]
        env._so101_postmortem_failure_diagnostics = []
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if table_bounds is None:
        table_bounds = {"x": (0.08, 0.45), "y": (-0.20, 0.20)}

    step_state = _termination_step_state(env, object_asset_names)
    object_pos_w = step_state.positions
    grasp_step = _update_grasp_attempts(
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
    if grasp_step is not None:
        _update_postmortem_evidence(
            env,
            object_asset_names=object_asset_names,
            bin_name=bin_name,
            table_bounds=table_bounds,
            step_state=step_state,
            grasp_step=grasp_step,
            baseline_recorded=baseline_recorded,
            object_distance_threshold=grasp_attempt_object_distance,
            move_straightness_tolerance=move_straightness_tolerance,
            move_past_boundary_tolerance=move_past_boundary_tolerance,
        )

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
        _distance_to_boundary, _progress, lateral, _target = _move_boundary_distance(
            env, object_asset_names, table_bounds, step_state
        )
        move_task = _task_is(env, TASK_MOVE)
        # Passing the boundary is no longer a failure; the Move goal has only a maximum
        # signed gap. Keep the legacy counter cleared for outcome-schema compatibility.
        if hasattr(env, "_so101_move_past_boundary_failure_counter"):
            env._so101_move_past_boundary_failure_counter.zero_()
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
    if active_families & {TASK_NAMED_BIN, TASK_NEXT_TO, TASK_BETWEEN, TASK_MOVE}:
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
    raw_reason_masks = (
        (FAILURE_REASON_MAX_GRASP_ATTEMPTS, instant_attempt_failure),
        (FAILURE_REASON_BIN_DISPLACED, instant_bin_failure),
        (FAILURE_REASON_NON_TARGET_MOVED, instant_non_target_moved & instruction_task),
        (FAILURE_REASON_MOVE_BOUNDARY_MOVED, instant_move_boundary_failure),
        (FAILURE_REASON_MOVE_PAST_BOUNDARY, instant_move_past_boundary),
        (
            FAILURE_REASON_MOVE_TRAJECTORY_NOT_STRAIGHT_ENOUGH,
            instant_move_trajectory_not_straight_enough,
        ),
        (FAILURE_REASON_MADE_CONTACT, made_contact),
        (FAILURE_REASON_SUCCESS_CONFIRMATION_BREACHED, timeout_confirmation_failure),
    )
    env._so101_failure_conditions_active_reasons = [
        tuple(
            reason
            for reason, mask in raw_reason_masks
            if bool(mask[env_id].item())
        )
        for env_id in range(env.num_envs)
    ]

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

    # Recompute after live reasons are known.  The temporal updater is guarded by
    # ``episode_length_buf``, so collector/rescorer calls on the same final step
    # may safely refresh this diagnostic without duplicating any evidence.
    env._so101_postmortem_failure_diagnostics = benchmark_postmortem_failure_diagnostics(
        env, object_asset_names
    )

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
        _distance_to_boundary, _progress, lateral, _target = _move_boundary_distance(
            env, object_asset_names, table_bounds, step_state
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

    if env._so101_task_family[env_id] in {TASK_NAMED_BIN, TASK_NEXT_TO, TASK_BETWEEN, TASK_MOVE}:
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
    move_past_boundary_tolerance: float = MOVE_PAST_BOUNDARY_TOLERANCE_M,
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
