"""Closed-loop phase-governed trajectory retiming.

This module deliberately has no Isaac Lab dependencies.  A caller supplies
pre-stretched reference and command paths in simulator joint radians, requests
one command before each environment step, and reports the resulting joint state
with :meth:`PhaseGovernedTrajectoryTracker.observe_post_step`.

Both input paths use node-aligned semantics:

* node 0 is the pre-command initial state/hold command;
* node ``i + 1`` is the expected post-step state and command for output frame
  ``i``.

Consequently, zero-error scale-one playback emits command node 1 first.  The
feedback error is measured at the current phase while the nominal command is
sampled at the next governed phase.  This avoids adding a second temporal lead
to the demonstrated action/state tracking offset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator


DEFAULT_SOFT_ERROR_RAD = (0.020, 0.020, 0.020, 0.026, 0.052)
DEFAULT_HARD_ERROR_RAD = (0.052, 0.052, 0.052, 0.070, 0.105)
DEFAULT_MAX_CORRECTION_RAD = (0.105, 0.105, 0.105, 0.140, 0.210)
DEFAULT_MAX_JOINT_SPEED_RAD_S = (1.5, 1.7, 1.7, 1.5, 2.2, 1.5)


def _finite_tuple(
    name: str,
    values: Sequence[float],
    *,
    positive: bool,
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    invalid = (
        any(not math.isfinite(value) for value in result)
        or (positive and any(value <= 0.0 for value in result))
        or (not positive and any(value < 0.0 for value in result))
    )
    if invalid:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} values must be finite and {qualifier}, got {result!r}.")
    return result


@dataclass(frozen=True)
class PhaseGovernorConfig:
    """Deterministic bounds and thresholds for the phase governor.

    Error and correction tuples cover the governed arm joints.  Speed and
    optional joint-limit tuples cover every trajectory joint, including the
    ungoverned gripper.  Gripper tracking error never affects phase progress.
    """

    control_dt: float
    feedback_gain: float = 0.70
    recovery_integral_gain_s: float = 1.50
    soft_error_rad: tuple[float, ...] = DEFAULT_SOFT_ERROR_RAD
    hard_error_rad: tuple[float, ...] = DEFAULT_HARD_ERROR_RAD
    max_correction_rad: tuple[float, ...] = DEFAULT_MAX_CORRECTION_RAD
    max_joint_speed_rad_s: tuple[float, ...] = DEFAULT_MAX_JOINT_SPEED_RAD_S
    hard_pause_entry_ratio: float = 0.95
    minimum_advancing_phase_rate: float = 0.10
    joint_lower_limits_rad: tuple[float, ...] | None = None
    joint_upper_limits_rad: tuple[float, ...] | None = None
    pause_release_steps: int = 3
    terminal_settle_steps: int = 3
    max_final_settle_steps: int = 15
    min_recovery_steps: int = 60
    recovery_fraction: float = 0.25
    arm_joint_count: int = 5

    def __post_init__(self) -> None:
        if not math.isfinite(self.control_dt) or self.control_dt <= 0.0:
            raise ValueError(f"control_dt must be finite and positive, got {self.control_dt!r}.")
        if not math.isfinite(self.feedback_gain) or self.feedback_gain < 0.0:
            raise ValueError(
                f"feedback_gain must be finite and non-negative, got {self.feedback_gain!r}."
            )
        if (
            not math.isfinite(self.recovery_integral_gain_s)
            or self.recovery_integral_gain_s < 0.0
        ):
            raise ValueError(
                "recovery_integral_gain_s must be finite and non-negative, "
                f"got {self.recovery_integral_gain_s!r}."
            )
        if (
            not math.isfinite(self.hard_pause_entry_ratio)
            or self.hard_pause_entry_ratio <= 0.0
            or self.hard_pause_entry_ratio >= 1.0
        ):
            raise ValueError(
                "hard_pause_entry_ratio must be finite and strictly between 0 and 1, "
                f"got {self.hard_pause_entry_ratio!r}."
            )
        if (
            not math.isfinite(self.minimum_advancing_phase_rate)
            or self.minimum_advancing_phase_rate <= 0.0
            or self.minimum_advancing_phase_rate > 1.0
        ):
            raise ValueError(
                "minimum_advancing_phase_rate must be finite and in (0, 1], "
                f"got {self.minimum_advancing_phase_rate!r}."
            )
        if self.arm_joint_count < 1:
            raise ValueError(f"arm_joint_count must be positive, got {self.arm_joint_count}.")

        soft = _finite_tuple("soft_error_rad", self.soft_error_rad, positive=True)
        hard = _finite_tuple("hard_error_rad", self.hard_error_rad, positive=True)
        correction = _finite_tuple(
            "max_correction_rad", self.max_correction_rad, positive=False
        )
        speeds = _finite_tuple(
            "max_joint_speed_rad_s", self.max_joint_speed_rad_s, positive=True
        )
        object.__setattr__(self, "soft_error_rad", soft)
        object.__setattr__(self, "hard_error_rad", hard)
        object.__setattr__(self, "max_correction_rad", correction)
        object.__setattr__(self, "max_joint_speed_rad_s", speeds)

        for name, values in (
            ("soft_error_rad", soft),
            ("hard_error_rad", hard),
            ("max_correction_rad", correction),
        ):
            if len(values) != self.arm_joint_count:
                raise ValueError(
                    f"{name} must have one value per governed arm joint; expected "
                    f"{self.arm_joint_count}, got {len(values)}."
                )
        if any(hard_value <= soft_value for soft_value, hard_value in zip(soft, hard)):
            raise ValueError(
                "Each hard_error_rad value must exceed the corresponding "
                f"soft_error_rad value, got soft={soft!r}, hard={hard!r}."
            )
        if len(speeds) <= self.arm_joint_count:
            raise ValueError(
                "max_joint_speed_rad_s must cover the arm and at least one ungoverned "
                f"gripper joint, got {len(speeds)} values."
            )

        lower = self.joint_lower_limits_rad
        upper = self.joint_upper_limits_rad
        if (lower is None) != (upper is None):
            raise ValueError(
                "joint_lower_limits_rad and joint_upper_limits_rad must be provided together."
            )
        if lower is not None and upper is not None:
            lower_tuple = tuple(float(value) for value in lower)
            upper_tuple = tuple(float(value) for value in upper)
            if any(not math.isfinite(value) for value in lower_tuple + upper_tuple):
                raise ValueError("Joint limits must contain only finite values.")
            if len(lower_tuple) != len(speeds) or len(upper_tuple) != len(speeds):
                raise ValueError(
                    "Joint-limit tuples must match max_joint_speed_rad_s length; got "
                    f"{len(lower_tuple)}/{len(upper_tuple)} versus {len(speeds)}."
                )
            if any(lo >= hi for lo, hi in zip(lower_tuple, upper_tuple)):
                raise ValueError(
                    "Every lower joint limit must be less than its upper limit, got "
                    f"lower={lower_tuple!r}, upper={upper_tuple!r}."
                )
            object.__setattr__(self, "joint_lower_limits_rad", lower_tuple)
            object.__setattr__(self, "joint_upper_limits_rad", upper_tuple)

        for name, value, minimum in (
            ("pause_release_steps", self.pause_release_steps, 1),
            ("terminal_settle_steps", self.terminal_settle_steps, 1),
            ("max_final_settle_steps", self.max_final_settle_steps, 1),
            ("min_recovery_steps", self.min_recovery_steps, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
        if self.max_final_settle_steps < self.terminal_settle_steps:
            raise ValueError(
                "max_final_settle_steps must be at least terminal_settle_steps, got "
                f"{self.max_final_settle_steps} < {self.terminal_settle_steps}."
            )
        if not math.isfinite(self.recovery_fraction) or self.recovery_fraction < 0.0:
            raise ValueError(
                "recovery_fraction must be finite and non-negative, got "
                f"{self.recovery_fraction!r}."
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseGovernorStep:
    """One pre-step controller decision and its diagnostic evidence."""

    command: np.ndarray
    raw_command: np.ndarray
    nominal_command: np.ndarray
    reference: np.ndarray
    target_reference: np.ndarray
    tracking_error: np.ndarray
    correction: np.ndarray
    phase: float
    proposed_phase: float
    next_phase: float
    governor_phase_rate: float
    phase_rate: float
    feedforward_scale: float
    normalized_arm_error: float
    max_arm_error_rad: float
    phase_rate_floored: bool
    correction_clipped: bool
    velocity_clipped: bool
    joint_limit_clipped: bool
    correction_clipped_mask: np.ndarray
    velocity_clipped_mask: np.ndarray
    joint_limit_clipped_mask: np.ndarray
    ungoverned_slew_limited_mask: np.ndarray
    paused: bool
    hard_paused: bool
    ungoverned_slew_limited: bool
    pause_release_streak: int
    at_end: bool

    @property
    def lookahead_reference(self) -> np.ndarray:
        """Compatibility alias for the phase-aligned target reference."""

        return self.target_reference

    @property
    def jaw_slew_limited(self) -> bool:
        """Whether commanded gripper lag constrained this phase increment."""

        return self.ungoverned_slew_limited


@dataclass(frozen=True)
class PhaseGovernorObservation:
    """Result of consuming the post-step state produced by a command."""

    tracking_error: np.ndarray
    max_arm_error_rad: float
    terminal_error_rad: np.ndarray
    terminal_max_arm_error_rad: float
    terminal_settle_streak: int
    final_settle_frames: int
    complete: bool
    limit_reached: bool
    completion_reason: str | None


def smoothly_retime_waypoint_path(
    source_waypoints: np.ndarray,
    *,
    initial_waypoint: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Time-warp a joint path with shape-preserving C1 interpolation.

    The returned path stays at the caller's control frequency and therefore has
    approximately ``scale`` times as many frames.  PCHIP avoids coordinate
    overshoot while eliminating repeated-frame velocity discontinuities.  The
    initial node is not returned; callers that need node-aligned paths prepend
    ``initial_waypoint`` themselves.
    """

    waypoints = np.asarray(source_waypoints, dtype=np.float64)
    initial = np.asarray(initial_waypoint, dtype=np.float64)
    if waypoints.ndim != 2 or waypoints.shape[0] < 1:
        raise ValueError(f"Expected non-empty [frames, joints] waypoints, got {waypoints.shape}.")
    if initial.shape != waypoints.shape[1:]:
        raise ValueError(
            f"Initial waypoint shape {initial.shape} does not match joint shape {waypoints.shape[1:]}."
        )
    if not np.all(np.isfinite(waypoints)) or not np.all(np.isfinite(initial)):
        raise ValueError("Trajectory waypoints must contain only finite values.")
    if not math.isfinite(scale) or scale < 1.0:
        raise ValueError(f"Expected finite retime scale >= 1.0, got {scale!r}.")

    source_frames = int(waypoints.shape[0])
    target_frames = max(source_frames, int(round(source_frames * scale)))
    if target_frames == source_frames:
        return waypoints.astype(np.float32, copy=True)

    path = np.concatenate((initial[None, :], waypoints), axis=0)
    source_times = np.arange(source_frames + 1, dtype=np.float64)
    target_times = np.arange(1, target_frames + 1, dtype=np.float64) * (
        source_frames / target_frames
    )
    result = PchipInterpolator(source_times, path, axis=0)(target_times)
    result[-1] = waypoints[-1]
    return np.asarray(result, dtype=np.float32)


def build_phase_governed_paths(
    source_actions: np.ndarray,
    source_observed_states: np.ndarray,
    initial_command: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build node-aligned reference and nominal paths for closed-loop replay.

    LeRobot rows pair ``action[i]`` with the state observed after applying that
    action.  Node zero is therefore the pre-command initial pose, while source
    row ``i`` maps to path node ``i + 1``.

    For arm joints, the slowed nominal target follows the demonstrated physical
    state path while retaining ``1 / scale`` of the demonstrated action/state
    tracking offset.  The final joint is the gripper: its original command path
    is retimed directly so actuator/contact lag in its observed state cannot
    weaken grasp and release intent.
    """

    actions = np.asarray(source_actions, dtype=np.float64)
    states = np.asarray(source_observed_states, dtype=np.float64)
    initial = np.asarray(initial_command, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] < 1:
        raise ValueError(
            f"source_actions must be non-empty [frames, joints], got {actions.shape}."
        )
    if states.ndim != 2 or states.shape != actions.shape:
        raise ValueError(
            "source_observed_states must have the same [frames, joints] shape as "
            f"source_actions; got {states.shape} and {actions.shape}."
        )
    if actions.shape[1] < 2:
        raise ValueError(
            "Trajectories must contain at least one arm joint and one gripper "
            f"joint, got {actions.shape[1]} joint(s)."
        )
    if initial.shape != actions.shape[1:]:
        raise ValueError(
            f"initial_command must have shape {actions.shape[1:]}, got {initial.shape}."
        )
    if (
        not np.all(np.isfinite(actions))
        or not np.all(np.isfinite(states))
        or not np.all(np.isfinite(initial))
    ):
        raise ValueError("Actions, observed states, and initial command must be finite.")
    if not math.isfinite(scale) or scale < 1.0:
        raise ValueError(f"Expected finite retime scale >= 1.0, got {scale!r}.")

    initial_node = np.asarray(initial, dtype=np.float32)[None, :]
    if scale == 1.0:
        # Avoid even harmless floating-point cancellation in
        # ``state + (action - state)``: scale-one is an exact identity contract.
        return (
            np.concatenate(
                (initial_node, np.asarray(states, dtype=np.float32)),
                axis=0,
            ),
            np.concatenate(
                (initial_node.copy(), np.asarray(actions, dtype=np.float32)),
                axis=0,
            ),
        )

    nominal_waypoints = states + (actions - states) / scale
    nominal_waypoints[:, -1] = actions[:, -1]

    retimed_reference = smoothly_retime_waypoint_path(
        states,
        initial_waypoint=initial,
        scale=scale,
    )
    retimed_nominal = smoothly_retime_waypoint_path(
        nominal_waypoints,
        initial_waypoint=initial,
        scale=scale,
    )
    reference_nodes = np.concatenate((initial_node, retimed_reference), axis=0)
    nominal_nodes = np.concatenate((initial_node.copy(), retimed_nominal), axis=0)
    # Make the node-zero contract explicit even if a future interpolation helper
    # changes its output dtype or endpoint handling.
    reference_nodes[0] = initial_node[0]
    nominal_nodes[0] = initial_node[0]
    return reference_nodes, nominal_nodes


class PhaseGovernedTrajectoryTracker:
    """Track a pre-stretched geometric path with bounded closed-loop feedback."""

    _PHASE_EPSILON = 1.0e-9

    def __init__(
        self,
        reference_positions: np.ndarray,
        nominal_commands: np.ndarray,
        config: PhaseGovernorConfig,
    ) -> None:
        reference = np.asarray(reference_positions, dtype=np.float64)
        nominal = np.asarray(nominal_commands, dtype=np.float64)
        if reference.ndim != 2 or reference.shape[0] < 2:
            raise ValueError(
                "reference_positions must be [M + 1, joints] with at least an "
                f"initial and final node, got {reference.shape}."
            )
        if nominal.ndim != 2 or nominal.shape[0] < 2:
            raise ValueError(
                "nominal_commands must be [M + 1, joints] with at least an "
                f"initial and first command node, got {nominal.shape}."
            )
        if reference.shape != nominal.shape:
            raise ValueError(
                "reference_positions and nominal_commands must have identical "
                f"[M + 1, joints] shapes, got {reference.shape} and {nominal.shape}."
            )
        if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(nominal)):
            raise ValueError("Reference and nominal trajectories must contain only finite values.")

        joint_count = reference.shape[1]
        if config.arm_joint_count >= joint_count:
            raise ValueError(
                "arm_joint_count must leave at least one ungoverned gripper joint, got "
                f"{config.arm_joint_count} for {joint_count} total joints."
            )
        if len(config.max_joint_speed_rad_s) != joint_count:
            raise ValueError(
                "max_joint_speed_rad_s must have exactly one value per trajectory joint, got "
                f"{len(config.max_joint_speed_rad_s)} for {joint_count} joints."
            )
        if config.joint_lower_limits_rad is not None and (
            len(config.joint_lower_limits_rad) != joint_count
            or len(config.joint_upper_limits_rad or ()) != joint_count
        ):
            raise ValueError("Joint-limit tuples must have exactly one value per trajectory joint.")

        self.reference_positions = reference
        self.nominal_commands = nominal
        self.config = config
        self.phase = 0.0
        self.steps = 0
        self.complete = False
        self.limit_reached = False
        self.completion_reason: str | None = None

        self.allowed_recovery_frames = max(
            config.min_recovery_steps,
            int(math.ceil(self.nominal_frames * config.recovery_fraction)),
        )
        self.max_steps = self.nominal_frames + self.allowed_recovery_frames

        self._hard_paused = False
        self._pause_release_streak = 0
        self._recovery_integral_correction = np.zeros(
            config.arm_joint_count,
            dtype=np.float64,
        )
        self._peak_recovery_integral_correction = np.zeros(
            config.arm_joint_count,
            dtype=np.float64,
        )
        self._current_pause_streak = 0
        self._longest_pause_streak = 0
        self._terminal_settle_streak = 0
        self._final_settle_frames = 0
        self._awaiting_post_step = False

        self._paused_frames = 0
        self._partial_rate_frames = 0
        self._phase_rate_floored_frames = 0
        self._phase_rate_sum = 0.0
        self._minimum_phase_rate = 1.0
        self._correction_clipped_frames = 0
        self._velocity_clipped_frames = 0
        self._joint_limit_clipped_frames = 0
        self._ungoverned_slew_limited_frames = 0
        self._correction_clipped_counts = np.zeros(joint_count, dtype=np.int64)
        self._velocity_clipped_counts = np.zeros(joint_count, dtype=np.int64)
        self._joint_limit_clipped_counts = np.zeros(joint_count, dtype=np.int64)
        self._ungoverned_slew_limited_counts = np.zeros(joint_count, dtype=np.int64)
        self._post_step_arm_errors: list[np.ndarray] = []

    @property
    def nominal_frames(self) -> int:
        return int(self.nominal_commands.shape[0] - 1)

    @property
    def emitted_frames(self) -> int:
        return self.steps

    @property
    def end_phase(self) -> float:
        return float(self.nominal_frames)

    @property
    def awaiting_post_step(self) -> bool:
        return self._awaiting_post_step

    @staticmethod
    def _interpolate(values: np.ndarray, phase: float) -> np.ndarray:
        bounded = float(np.clip(phase, 0.0, len(values) - 1))
        lower = int(math.floor(bounded))
        upper = min(lower + 1, len(values) - 1)
        weight = bounded - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    def _validate_joint_vector(
        self,
        name: str,
        values: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        expected_shape = self.reference_positions.shape[1:]
        if array.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        return array

    def _normalized_error(self, arm_error_abs: np.ndarray) -> float:
        soft = np.asarray(self.config.soft_error_rad, dtype=np.float64)
        hard = np.asarray(self.config.hard_error_rad, dtype=np.float64)
        normalized = np.clip((arm_error_abs - soft) / (hard - soft), 0.0, 1.0)
        return float(np.max(normalized))

    def _phase_rate(
        self,
        arm_error_abs: np.ndarray,
    ) -> tuple[float, float, bool]:
        soft = np.asarray(self.config.soft_error_rad, dtype=np.float64)

        if self._hard_paused:
            if np.all(arm_error_abs <= soft):
                self._pause_release_streak += 1
            else:
                self._pause_release_streak = 0
            if self._pause_release_streak >= self.config.pause_release_steps:
                self._hard_paused = False
                self._pause_release_streak = 0
            else:
                return 0.0, self._normalized_error(arm_error_abs), False

        normalized = self._normalized_error(arm_error_abs)
        # Smoothstep has zero slope and approaches zero phase speed at the hard
        # boundary.  A persistent actuator error just below that boundary can
        # therefore make the old controller crawl forever without ever entering
        # its recovery mode.  Enter recovery slightly early, with hysteresis
        # supplied by the existing all-soft release condition.
        if normalized >= self.config.hard_pause_entry_ratio:
            self._hard_paused = True
            self._pause_release_streak = 0
            return 0.0, normalized, False

        smoothstep = normalized * normalized * (3.0 - 2.0 * normalized)
        unconstrained_rate = 1.0 - smoothstep
        rate = max(
            self.config.minimum_advancing_phase_rate,
            unconstrained_rate,
        )
        return rate, normalized, rate > unconstrained_rate + self._PHASE_EPSILON

    def _joint_bounds(self, joint_count: int) -> tuple[np.ndarray, np.ndarray]:
        if self.config.joint_lower_limits_rad is None:
            return (
                np.full(joint_count, -np.inf, dtype=np.float64),
                np.full(joint_count, np.inf, dtype=np.float64),
            )
        assert self.config.joint_upper_limits_rad is not None
        return (
            np.asarray(self.config.joint_lower_limits_rad, dtype=np.float64),
            np.asarray(self.config.joint_upper_limits_rad, dtype=np.float64),
        )

    def _constrain_phase_for_ungoverned_slew(
        self,
        phase: float,
        proposed_phase: float,
        previous_command: np.ndarray,
    ) -> tuple[float, bool, np.ndarray]:
        """Keep ungoverned command nodes within one command-slew budget.

        The check intentionally uses the previously *commanded* position rather
        than the measured gripper state.  Contact may prevent the physical jaw
        from attaining its command and must not freeze arm-path progress.

        A phase increment spans at most one output frame, but it can cross one
        interpolation knot when the current phase is fractional.  Checking each
        linear segment endpoint makes the feasible prefix monotone; bisection
        then finds the first slew boundary without skipping a non-monotonic
        command excursion at a knot.
        """

        joint_count = self.reference_positions.shape[1]
        limited_mask = np.zeros(joint_count, dtype=np.bool_)
        if proposed_phase <= phase + self._PHASE_EPSILON:
            return proposed_phase, False, limited_mask

        arm_count = self.config.arm_joint_count
        max_delta = (
            np.asarray(self.config.max_joint_speed_rad_s, dtype=np.float64)
            * self.config.control_dt
        )

        def violations(candidate_phase: float) -> np.ndarray:
            nominal = self._interpolate(self.nominal_commands, candidate_phase)
            mask = np.zeros(joint_count, dtype=np.bool_)
            mask[arm_count:] = (
                np.abs(nominal[arm_count:] - previous_command[arm_count:])
                > max_delta[arm_count:] + 1.0e-12
            )
            return mask

        start_violations = violations(phase)
        if np.any(start_violations):
            return phase, True, start_violations

        segment_ends: list[float] = []
        knot = math.floor(phase) + 1
        while knot < proposed_phase - self._PHASE_EPSILON:
            segment_ends.append(float(knot))
            knot += 1
        segment_ends.append(proposed_phase)

        last_feasible = phase
        for segment_end in segment_ends:
            end_violations = violations(segment_end)
            if not np.any(end_violations):
                last_feasible = segment_end
                continue

            infeasible = segment_end
            limiting_mask = end_violations
            for _ in range(60):
                midpoint = 0.5 * (last_feasible + infeasible)
                midpoint_violations = violations(midpoint)
                if np.any(midpoint_violations):
                    infeasible = midpoint
                    limiting_mask = midpoint_violations
                else:
                    last_feasible = midpoint
            if last_feasible <= phase + self._PHASE_EPSILON:
                last_feasible = phase
            return last_feasible, True, limiting_mask

        return proposed_phase, False, limited_mask

    def step(
        self,
        actual_position: Sequence[float] | np.ndarray,
        previous_command: Sequence[float] | np.ndarray,
    ) -> PhaseGovernorStep:
        """Construct one command; the caller must report its post-step state."""

        if self.complete:
            raise RuntimeError("Cannot step a completed phase-governed trajectory.")
        if self._awaiting_post_step:
            raise RuntimeError(
                "observe_post_step() must consume the previous command's resulting "
                "state before step() is called again."
            )

        actual = self._validate_joint_vector("actual_position", actual_position)
        previous = self._validate_joint_vector("previous_command", previous_command)
        arm_count = self.config.arm_joint_count

        phase = self.phase
        reference = self._interpolate(self.reference_positions, phase)
        tracking_error = reference - actual
        arm_error_abs = np.abs(tracking_error[:arm_count])
        max_arm_error = float(np.max(arm_error_abs))

        at_end = math.isclose(
            phase, self.end_phase, rel_tol=0.0, abs_tol=self._PHASE_EPSILON
        )
        if at_end:
            governor_phase_rate = 0.0
            normalized_error = self._normalized_error(arm_error_abs)
            phase_rate_floored = False
        else:
            (
                governor_phase_rate,
                normalized_error,
                phase_rate_floored,
            ) = self._phase_rate(arm_error_abs)
        soft_error = np.asarray(self.config.soft_error_rad, dtype=np.float64)
        if not self._hard_paused:
            # Do not carry a contact/recovery force into normal path motion or
            # through the hysteretic release samples.
            self._recovery_integral_correction.fill(0.0)
        else:
            # Anti-windup is joint-local: one blocked joint must not cause small
            # in-tolerance errors on other joints to accumulate into saturated
            # and potentially limit-clamped commands.
            self._recovery_integral_correction[arm_error_abs <= soft_error] = 0.0
        proposed_phase = min(self.end_phase, phase + governor_phase_rate)
        (
            next_phase,
            ungoverned_slew_limited,
            ungoverned_slew_limited_mask,
        ) = self._constrain_phase_for_ungoverned_slew(
            phase,
            proposed_phase,
            previous,
        )
        phase_rate = next_phase - phase
        target_reference = self._interpolate(self.reference_positions, next_phase)
        nominal_command = self._interpolate(self.nominal_commands, next_phase)

        correction_unclipped = (
            self.config.feedback_gain * tracking_error[:arm_count]
            + self._recovery_integral_correction
        )
        correction_limits = np.asarray(self.config.max_correction_rad, dtype=np.float64)
        correction_arm = np.clip(
            correction_unclipped, -correction_limits, correction_limits
        )
        correction = np.zeros_like(actual)
        correction[:arm_count] = correction_arm
        correction_clipped_mask = np.zeros_like(actual, dtype=np.bool_)
        correction_clipped_mask[:arm_count] = (
            np.abs(correction_unclipped - correction_arm) > 1.0e-12
        )

        # A hard pause can otherwise deadlock against a stable contact load:
        # holding the current geometric reference may not generate enough
        # command offset to attain that reference.  Accumulate a bounded
        # integral recovery correction for the *next* command.  This preserves
        # a bumpless first pause frame, supplies the missing static/contact
        # effort over subsequent frames, and is cleared as soon as tracking is
        # soft enough to begin the release hysteresis.
        if self._hard_paused:
            active_recovery_mask = arm_error_abs > soft_error
            self._recovery_integral_correction[active_recovery_mask] += (
                self.config.recovery_integral_gain_s
                * self.config.control_dt
                * tracking_error[:arm_count][active_recovery_mask]
            )
            self._recovery_integral_correction = np.clip(
                self._recovery_integral_correction,
                -correction_limits,
                correction_limits,
            )
            self._peak_recovery_integral_correction = np.maximum(
                self._peak_recovery_integral_correction,
                np.abs(self._recovery_integral_correction),
            )

        # The demonstrated arm command/state offset is primarily dynamic lead.
        # Scale it by the phase speed actually permitted this frame.  At a hard
        # pause (or while settling at the endpoint) the command therefore
        # converges on the geometric reference instead of sustaining a bias
        # that can prevent the tracking error from ever crossing the resume
        # threshold.  The jaw deliberately remains on its nominal command path.
        feedforward_scale = phase_rate
        raw_command = nominal_command.copy()
        raw_command[:arm_count] = (
            target_reference[:arm_count]
            + feedforward_scale
            * (
                nominal_command[:arm_count]
                - target_reference[:arm_count]
            )
            + correction_arm
        )

        max_delta = (
            np.asarray(self.config.max_joint_speed_rad_s, dtype=np.float64)
            * self.config.control_dt
        )
        requested_delta = raw_command - previous
        bounded_delta = np.clip(requested_delta, -max_delta, max_delta)
        velocity_bounded_command = previous + bounded_delta
        velocity_clipped_mask = np.abs(requested_delta - bounded_delta) > 1.0e-12

        lower, upper = self._joint_bounds(len(actual))
        command = np.clip(velocity_bounded_command, lower, upper)
        joint_limit_clipped_mask = (
            np.abs(velocity_bounded_command - command) > 1.0e-12
        )

        self.phase = next_phase
        self.steps += 1
        self._awaiting_post_step = True

        paused = phase_rate <= self._PHASE_EPSILON and not at_end
        if paused:
            self._paused_frames += 1
            self._current_pause_streak += 1
            self._longest_pause_streak = max(
                self._longest_pause_streak, self._current_pause_streak
            )
        else:
            self._current_pause_streak = 0
            if phase_rate < 1.0 - self._PHASE_EPSILON and not at_end:
                self._partial_rate_frames += 1
        if phase_rate_floored:
            self._phase_rate_floored_frames += 1
        self._phase_rate_sum += phase_rate
        self._minimum_phase_rate = min(self._minimum_phase_rate, phase_rate)

        correction_clipped = bool(np.any(correction_clipped_mask))
        velocity_clipped = bool(np.any(velocity_clipped_mask))
        joint_limit_clipped = bool(np.any(joint_limit_clipped_mask))
        if correction_clipped:
            self._correction_clipped_frames += 1
        if velocity_clipped:
            self._velocity_clipped_frames += 1
        if joint_limit_clipped:
            self._joint_limit_clipped_frames += 1
        if ungoverned_slew_limited:
            self._ungoverned_slew_limited_frames += 1
        self._correction_clipped_counts += correction_clipped_mask
        self._velocity_clipped_counts += velocity_clipped_mask
        self._joint_limit_clipped_counts += joint_limit_clipped_mask
        self._ungoverned_slew_limited_counts += ungoverned_slew_limited_mask

        return PhaseGovernorStep(
            command=np.asarray(command, dtype=np.float32),
            raw_command=np.asarray(raw_command, dtype=np.float32),
            nominal_command=np.asarray(nominal_command, dtype=np.float32),
            reference=np.asarray(reference, dtype=np.float32),
            target_reference=np.asarray(target_reference, dtype=np.float32),
            tracking_error=np.asarray(tracking_error, dtype=np.float32),
            correction=np.asarray(correction, dtype=np.float32),
            phase=phase,
            proposed_phase=proposed_phase,
            next_phase=next_phase,
            governor_phase_rate=governor_phase_rate,
            phase_rate=phase_rate,
            feedforward_scale=feedforward_scale,
            normalized_arm_error=normalized_error,
            max_arm_error_rad=max_arm_error,
            phase_rate_floored=phase_rate_floored,
            correction_clipped=correction_clipped,
            velocity_clipped=velocity_clipped,
            joint_limit_clipped=joint_limit_clipped,
            correction_clipped_mask=correction_clipped_mask.copy(),
            velocity_clipped_mask=velocity_clipped_mask.copy(),
            joint_limit_clipped_mask=joint_limit_clipped_mask.copy(),
            ungoverned_slew_limited_mask=ungoverned_slew_limited_mask.copy(),
            paused=paused,
            hard_paused=self._hard_paused,
            ungoverned_slew_limited=ungoverned_slew_limited,
            pause_release_streak=self._pause_release_streak,
            at_end=at_end,
        )

    def observe_post_step(
        self,
        actual_position: Sequence[float] | np.ndarray,
    ) -> PhaseGovernorObservation:
        """Consume the state resulting from the most recently emitted command."""

        if self.complete:
            raise RuntimeError("Cannot observe a completed phase-governed trajectory.")
        if not self._awaiting_post_step:
            raise RuntimeError("step() must emit a command before observe_post_step() is called.")

        actual = self._validate_joint_vector("actual_position", actual_position)
        self._awaiting_post_step = False
        arm_count = self.config.arm_joint_count
        reference = self._interpolate(self.reference_positions, self.phase)
        tracking_error = reference - actual
        arm_error = tracking_error[:arm_count]
        arm_error_abs = np.abs(arm_error)
        max_arm_error = float(np.max(arm_error_abs))
        self._post_step_arm_errors.append(arm_error.copy())

        terminal_error = self.reference_positions[-1] - actual
        terminal_arm_error_abs = np.abs(terminal_error[:arm_count])
        terminal_max_error = float(np.max(terminal_arm_error_abs))
        at_end = math.isclose(
            self.phase, self.end_phase, rel_tol=0.0, abs_tol=self._PHASE_EPSILON
        )
        if at_end:
            self._final_settle_frames += 1
            soft = np.asarray(self.config.soft_error_rad, dtype=np.float64)
            if np.all(terminal_arm_error_abs <= soft):
                self._terminal_settle_streak += 1
            else:
                self._terminal_settle_streak = 0
            if self._terminal_settle_streak >= self.config.terminal_settle_steps:
                self.complete = True
                self.completion_reason = "settled"
            elif self._final_settle_frames >= self.config.max_final_settle_steps:
                self.complete = True
                self.limit_reached = True
                self.completion_reason = "final_settle_limit"
        else:
            self._terminal_settle_streak = 0
            self._final_settle_frames = 0

        if not self.complete and self.steps >= self.max_steps:
            self.complete = True
            self.limit_reached = True
            self.completion_reason = "recovery_limit"

        return PhaseGovernorObservation(
            tracking_error=np.asarray(tracking_error, dtype=np.float32),
            max_arm_error_rad=max_arm_error,
            terminal_error_rad=np.asarray(terminal_error, dtype=np.float32),
            terminal_max_arm_error_rad=terminal_max_error,
            terminal_settle_streak=self._terminal_settle_streak,
            final_settle_frames=self._final_settle_frames,
            complete=self.complete,
            limit_reached=self.limit_reached,
            completion_reason=self.completion_reason,
        )

    def diagnostics(self) -> dict[str, object]:
        """Return deterministic aggregate diagnostics for the current episode."""

        arm_count = self.config.arm_joint_count
        if self._post_step_arm_errors:
            errors = np.stack(self._post_step_arm_errors, axis=0)
            absolute = np.abs(errors)
            rms = np.sqrt(np.mean(errors * errors, axis=0))
            p95 = np.quantile(absolute, 0.95, axis=0)
            maximum = np.max(absolute, axis=0)
        else:
            rms = np.zeros(arm_count, dtype=np.float64)
            p95 = np.zeros(arm_count, dtype=np.float64)
            maximum = np.zeros(arm_count, dtype=np.float64)

        return {
            "nominal_frames": self.nominal_frames,
            "emitted_frames": self.emitted_frames,
            "allowed_recovery_frames": self.allowed_recovery_frames,
            "max_frames": self.max_steps,
            "recovery_frames": max(0, self.emitted_frames - self.nominal_frames),
            "phase": self.phase,
            "end_phase": self.end_phase,
            "phase_completion_fraction": self.phase / self.end_phase,
            "realized_duration_factor": self.emitted_frames / self.nominal_frames,
            "mean_phase_rate": (
                self._phase_rate_sum / self.emitted_frames
                if self.emitted_frames
                else 0.0
            ),
            "minimum_phase_rate": (
                self._minimum_phase_rate if self.emitted_frames else 0.0
            ),
            "paused_frames": self._paused_frames,
            "partial_rate_frames": self._partial_rate_frames,
            "phase_rate_floored_frames": self._phase_rate_floored_frames,
            "recovery_integral_correction_rad": (
                self._recovery_integral_correction.tolist()
            ),
            "peak_recovery_integral_correction_rad": (
                self._peak_recovery_integral_correction.tolist()
            ),
            "longest_pause_frames": self._longest_pause_streak,
            "hard_paused": self._hard_paused,
            "pause_release_streak": self._pause_release_streak,
            "final_settle_frames": self._final_settle_frames,
            "terminal_settle_streak": self._terminal_settle_streak,
            "correction_clipped_frames": self._correction_clipped_frames,
            "velocity_clipped_frames": self._velocity_clipped_frames,
            "joint_limit_clipped_frames": self._joint_limit_clipped_frames,
            "ungoverned_slew_limited_frames": self._ungoverned_slew_limited_frames,
            "jaw_slew_limited_frames": self._ungoverned_slew_limited_frames,
            "correction_clipped_counts": self._correction_clipped_counts.tolist(),
            "velocity_clipped_counts": self._velocity_clipped_counts.tolist(),
            "joint_limit_clipped_counts": self._joint_limit_clipped_counts.tolist(),
            "ungoverned_slew_limited_counts": (
                self._ungoverned_slew_limited_counts.tolist()
            ),
            "arm_tracking_error_rms_rad": rms.tolist(),
            "arm_tracking_error_p95_rad": p95.tolist(),
            "arm_tracking_error_max_rad": maximum.tolist(),
            "max_arm_tracking_error_rad": float(np.max(maximum)),
            "complete": self.complete,
            "limit_reached": self.limit_reached,
            "completion_reason": self.completion_reason,
            "awaiting_post_step": self._awaiting_post_step,
        }
