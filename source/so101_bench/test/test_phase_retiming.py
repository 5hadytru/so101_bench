from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "so101_bench"
    / "utils"
    / "phase_retiming.py"
)
SPEC = importlib.util.spec_from_file_location("so101_phase_retiming_test_module", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
RETIMING = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RETIMING
SPEC.loader.exec_module(RETIMING)


def arm_tuple(value: float) -> tuple[float, ...]:
    return (value,) * 5


def config(**overrides):
    values = {
        "control_dt": 0.1,
        "feedback_gain": 0.5,
        "soft_error_rad": arm_tuple(0.05),
        "hard_error_rad": arm_tuple(0.15),
        "max_correction_rad": arm_tuple(0.20),
        "max_joint_speed_rad_s": (100.0,) * 6,
        "pause_release_steps": 3,
        "terminal_settle_steps": 3,
        "max_final_settle_steps": 15,
        "min_recovery_steps": 60,
        "recovery_fraction": 0.25,
    }
    values.update(overrides)
    return RETIMING.PhaseGovernorConfig(**values)


def paths(frames: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return M+1 node-aligned reference and command paths."""

    reference = np.zeros((frames + 1, 6), dtype=np.float32)
    reference[:, 0] = np.arange(frames + 1, dtype=np.float32) * 0.1
    reference[:, 1] = np.arange(frames + 1, dtype=np.float32) * -0.02
    nominal = reference.copy()
    # A demonstrated arm command/state tracking offset.
    nominal[1:, 0] += 0.03
    # Node zero is the initial hold; command node one drives output frame zero.
    nominal[:, 5] = np.linspace(0.4, 0.1, frames + 1)
    return reference, nominal


class SmoothRetimingTests(unittest.TestCase):
    def test_retiming_preserves_endpoints_and_target_length(self) -> None:
        source = np.asarray([[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        result = RETIMING.smoothly_retime_waypoint_path(
            source,
            initial_waypoint=np.asarray([0.0, 1.0], dtype=np.float32),
            scale=2.0,
        )

        self.assertEqual(result.shape, (6, 2))
        np.testing.assert_allclose(result[-1], source[-1])
        self.assertTrue(np.all(np.diff(result[:, 0]) >= 0.0))

    def test_retiming_rejects_nonfinite_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            RETIMING.smoothly_retime_waypoint_path(
                np.asarray([[np.nan]], dtype=np.float32),
                initial_waypoint=np.asarray([0.0], dtype=np.float32),
                scale=2.0,
            )


class PhaseGovernedPathBuilderTests(unittest.TestCase):
    def test_scale_one_returns_exact_node_aligned_states_and_actions(self) -> None:
        initial = np.asarray([0.0, -0.1, 0.4], dtype=np.float32)
        states = np.asarray(
            [
                [0.1, -0.2, 0.35],
                [0.2, -0.3, 0.25],
                [0.4, -0.4, 0.15],
            ],
            dtype=np.float32,
        )
        actions = np.asarray(
            [
                [0.12, -0.18, 0.30],
                [0.24, -0.27, 0.20],
                [0.45, -0.36, 0.10],
            ],
            dtype=np.float32,
        )

        reference, nominal = RETIMING.build_phase_governed_paths(
            actions,
            states,
            initial,
            1.0,
        )

        np.testing.assert_array_equal(
            reference,
            np.concatenate((initial[None, :], states), axis=0),
        )
        np.testing.assert_array_equal(
            nominal,
            np.concatenate((initial[None, :], actions), axis=0),
        )

    def test_scaled_path_uses_tracking_compensation_but_retains_jaw_commands(self) -> None:
        initial = np.asarray([0.0, 0.0, 0.8], dtype=np.float32)
        states = np.asarray(
            [
                [0.10, -0.10, 0.75],
                [0.20, -0.20, 0.70],
                [0.30, -0.30, 0.65],
            ],
            dtype=np.float32,
        )
        actions = np.asarray(
            [
                [0.14, -0.04, 0.60],
                [0.28, -0.10, 0.30],
                [0.42, -0.16, 0.10],
            ],
            dtype=np.float32,
        )
        scale = 2.0

        reference, nominal = RETIMING.build_phase_governed_paths(
            actions,
            states,
            initial,
            scale,
        )

        compensated = states + (actions - states) / scale
        compensated[:, -1] = actions[:, -1]
        expected_reference = RETIMING.smoothly_retime_waypoint_path(
            states,
            initial_waypoint=initial,
            scale=scale,
        )
        expected_nominal = RETIMING.smoothly_retime_waypoint_path(
            compensated,
            initial_waypoint=initial,
            scale=scale,
        )
        expected_jaw_commands = RETIMING.smoothly_retime_waypoint_path(
            actions[:, -1, None],
            initial_waypoint=initial[-1, None],
            scale=scale,
        )[:, 0]

        self.assertEqual(reference.shape, (7, 3))
        self.assertEqual(nominal.shape, reference.shape)
        np.testing.assert_array_equal(reference[0], initial)
        np.testing.assert_array_equal(nominal[0], initial)
        np.testing.assert_allclose(reference[1:], expected_reference)
        np.testing.assert_allclose(nominal[1:], expected_nominal)
        np.testing.assert_allclose(nominal[1:, -1], expected_jaw_commands)
        self.assertFalse(np.allclose(nominal[1:, -1], reference[1:, -1]))

    def test_builder_rejects_mismatched_nonfinite_and_invalid_scale_inputs(self) -> None:
        actions = np.zeros((2, 3), dtype=np.float32)
        states = np.zeros_like(actions)
        initial = np.zeros(3, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "same"):
            RETIMING.build_phase_governed_paths(
                actions,
                states[:-1],
                initial,
                2.0,
            )
        invalid = actions.copy()
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            RETIMING.build_phase_governed_paths(
                invalid,
                states,
                initial,
                2.0,
            )
        with self.assertRaisesRegex(ValueError, "scale"):
            RETIMING.build_phase_governed_paths(
                actions,
                states,
                initial,
                0.5,
            )


class PhaseGovernedTrajectoryTrackerTests(unittest.TestCase):
    def test_zero_error_emits_command_node_one_first_without_extra_lead(self) -> None:
        reference, nominal = paths()
        tracker = RETIMING.PhaseGovernedTrajectoryTracker(reference, nominal, config())

        first = tracker.step(
            actual_position=reference[0],
            previous_command=nominal[0],
        )

        self.assertEqual(first.phase, 0.0)
        self.assertEqual(first.next_phase, 1.0)
        self.assertEqual(first.phase_rate, 1.0)
        self.assertEqual(first.feedforward_scale, 1.0)
        np.testing.assert_allclose(first.command, nominal[1], atol=1.0e-7)
        np.testing.assert_allclose(first.target_reference, reference[1], atol=1.0e-7)
        self.assertEqual(first.command[5], nominal[1, 5])

        observed = tracker.observe_post_step(reference[1])
        self.assertFalse(observed.complete)
        second = tracker.step(
            actual_position=reference[1],
            previous_command=first.command,
        )
        np.testing.assert_allclose(second.command, nominal[2], atol=1.0e-7)

    def test_hard_pause_fades_arm_residual_and_allows_hysteretic_recovery(self) -> None:
        reference, nominal = paths()
        tracker = RETIMING.PhaseGovernedTrajectoryTracker(
            reference,
            nominal,
            config(feedback_gain=0.0),
        )

        first = tracker.step(reference[0], nominal[0])
        tracker.observe_post_step(reference[1])
        self.assertGreater(
            abs(nominal[1, 0] - reference[1, 0]),
            0.0,
        )

        bad = reference[1].copy()
        bad[0] -= 0.20
        paused = tracker.step(bad, first.command)
        self.assertTrue(paused.hard_paused)
        self.assertEqual(paused.phase, 1.0)
        self.assertEqual(paused.next_phase, 1.0)
        self.assertEqual(paused.feedforward_scale, 0.0)
        np.testing.assert_allclose(
            paused.raw_command[:5],
            reference[1, :5],
            atol=1.0e-7,
        )
        self.assertNotAlmostEqual(
            paused.raw_command[0],
            nominal[1, 0],
        )
        tracker.observe_post_step(reference[1])

        previous = paused.command
        for expected_streak in (1, 2):
            held = tracker.step(reference[1], previous)
            self.assertEqual(held.feedforward_scale, 0.0)
            self.assertEqual(held.next_phase, 1.0)
            np.testing.assert_allclose(held.raw_command[:5], reference[1, :5])
            tracker.observe_post_step(reference[1])
            previous = held.command
            self.assertEqual(held.pause_release_streak, expected_streak)

        resumed = tracker.step(reference[1], previous)
        self.assertEqual(resumed.feedforward_scale, 1.0)
        self.assertEqual(resumed.next_phase, 2.0)
        np.testing.assert_allclose(resumed.raw_command, nominal[2], atol=1.0e-7)

    def test_smooth_governor_uses_worst_normalized_arm_error(self) -> None:
        reference, nominal = paths()
        tracker = RETIMING.PhaseGovernedTrajectoryTracker(reference, nominal, config())
        actual = reference[0].copy()
        actual[0] -= 0.10

        result = tracker.step(actual_position=actual, previous_command=nominal[0])

        # Error is halfway between the soft and hard bounds; smoothstep(0.5)
        # is also 0.5.
        self.assertAlmostEqual(result.normalized_arm_error, 0.5)
        self.assertAlmostEqual(result.phase_rate, 0.5)
        self.assertAlmostEqual(result.next_phase, 0.5)
        self.assertFalse(result.paused)

    def test_near_hard_error_enters_recovery_before_the_smoothstep_asymptote(self) -> None:
        reference, nominal = paths()
        tracker = RETIMING.PhaseGovernedTrajectoryTracker(reference, nominal, config())
        actual = reference[0].copy()
        # soft=0.05, hard=0.15: 0.149 rad is normalized error 0.99, but
        # remains strictly below the old exact hard-pause boundary.
        actual[2] -= 0.149

        result = tracker.step(actual_position=actual, previous_command=nominal[0])

        self.assertAlmostEqual(result.normalized_arm_error, 0.99)
        self.assertEqual(result.phase_rate, 0.0)
        self.assertEqual(result.next_phase, 0.0)
        self.assertTrue(result.paused)
        self.assertTrue(result.hard_paused)
        self.assertFalse(result.phase_rate_floored)

    def test_advancing_phase_rate_has_a_positive_floor_below_pause_entry(self) -> None:
        reference, nominal = paths()
        tracker = RETIMING.PhaseGovernedTrajectoryTracker(reference, nominal, config())
        actual = reference[0].copy()
        # Normalized error 0.94 is below the 0.95 pause-entry ratio, while its
        # unconstrained smoothstep rate is only about 0.0104.
        actual[2] -= 0.144

        result = tracker.step(actual_position=actual, previous_command=nominal[0])

        self.assertAlmostEqual(result.normalized_arm_error, 0.94, places=6)
        self.assertAlmostEqual(result.phase_rate, 0.10)
        self.assertAlmostEqual(result.next_phase, 0.10)
        self.assertFalse(result.paused)
        self.assertFalse(result.hard_paused)
        self.assertTrue(result.phase_rate_floored)
        self.assertEqual(tracker.diagnostics()["phase_rate_floored_frames"], 1)

    def test_hybrid_governor_parameters_reject_asymptotic_or_invalid_values(self) -> None:
        for invalid_ratio in (0.0, 1.0, math.inf):
            with self.subTest(hard_pause_entry_ratio=invalid_ratio):
                with self.assertRaisesRegex(ValueError, "hard_pause_entry_ratio"):
                    config(hard_pause_entry_ratio=invalid_ratio)
        for invalid_rate in (0.0, 1.01, math.nan):
            with self.subTest(minimum_advancing_phase_rate=invalid_rate):
                with self.assertRaisesRegex(
                    ValueError, "minimum_advancing_phase_rate"
                ):
                    config(minimum_advancing_phase_rate=invalid_rate)
        for invalid_gain in (-0.01, math.inf):
            with self.subTest(recovery_integral_gain_s=invalid_gain):
                with self.assertRaisesRegex(
                    ValueError, "recovery_integral_gain_s"
                ):
                    config(recovery_integral_gain_s=invalid_gain)

    def test_hard_pause_requires_three_good_pre_step_observations_to_resume(self) -> None:
        reference, nominal = paths()
        tracker = RETIMING.PhaseGovernedTrajectoryTracker(reference, nominal, config())
        bad = reference[0].copy()
        bad[0] -= 0.20

        paused = tracker.step(actual_position=bad, previous_command=nominal[0])
        self.assertTrue(paused.paused)
        self.assertTrue(paused.hard_paused)
        self.assertEqual(paused.next_phase, 0.0)
        tracker.observe_post_step(reference[0])

        previous = paused.command
        for expected_streak in (1, 2):
            held = tracker.step(actual_position=reference[0], previous_command=previous)
            self.assertEqual(held.phase_rate, 0.0)
            self.assertTrue(held.hard_paused)
            self.assertEqual(held.pause_release_streak, expected_streak)
            tracker.observe_post_step(reference[0])
            previous = held.command

        resumed = tracker.step(actual_position=reference[0], previous_command=previous)
        self.assertEqual(resumed.phase_rate, 1.0)
        self.assertFalse(resumed.hard_paused)
        self.assertEqual(resumed.next_phase, 1.0)

    def test_hard_pause_integral_builds_bounded_contact_recovery_effort(self) -> None:
        reference, nominal = paths()
        tracker = RETIMING.PhaseGovernedTrajectoryTracker(
            reference,
            nominal,
            config(
                feedback_gain=0.0,
                recovery_integral_gain_s=1.0,
                max_correction_rad=arm_tuple(0.03),
            ),
        )
        blocked = reference[0].copy()
        blocked[2] -= 0.149
        blocked[3] -= 0.01  # In tolerance; must not wind up with the elbow.

        first = tracker.step(blocked, nominal[0])
        self.assertEqual(first.correction[2], 0.0)
        tracker.observe_post_step(blocked)

        second = tracker.step(blocked, first.command)
        self.assertAlmostEqual(second.correction[2], 0.0149, places=6)
        tracker.observe_post_step(blocked)

        third = tracker.step(blocked, second.command)
        self.assertAlmostEqual(third.correction[2], 0.0298, places=6)
        tracker.observe_post_step(blocked)

        fourth = tracker.step(blocked, third.command)
        self.assertAlmostEqual(fourth.correction[2], 0.03, places=6)
        diagnostics = tracker.diagnostics()
        self.assertAlmostEqual(
            diagnostics["peak_recovery_integral_correction_rad"][2],
            0.03,
            places=6,
        )
        self.assertEqual(
            diagnostics["peak_recovery_integral_correction_rad"][3],
            0.0,
        )

    def test_jaw_error_never_gates_phase_or_receives_feedback(self) -> None:
        reference, nominal = paths()
        tracker = RETIMING.PhaseGovernedTrajectoryTracker(reference, nominal, config())
        actual = reference[0].copy()
        actual[5] = -10.0

        result = tracker.step(actual_position=actual, previous_command=nominal[0])

        self.assertEqual(result.phase_rate, 1.0)
        self.assertFalse(result.paused)
        self.assertFalse(result.ungoverned_slew_limited)
        self.assertEqual(result.correction[5], 0.0)
        self.assertAlmostEqual(result.raw_command[5], nominal[1, 5])

    def test_commanded_jaw_slew_budget_constrains_phase_without_clipping(self) -> None:
        reference, nominal = paths()
        # The first jaw segment requests 0.075 rad.  A 0.05-rad command budget
        # permits exactly two thirds of the phase increment.
        tracker = RETIMING.PhaseGovernedTrajectoryTracker(
            reference,
            nominal,
            config(max_joint_speed_rad_s=(100.0,) * 5 + (0.5,)),
        )

        first = tracker.step(reference[0], nominal[0])

        self.assertEqual(first.proposed_phase, 1.0)
        self.assertAlmostEqual(first.next_phase, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(first.phase_rate, 2.0 / 3.0, places=6)
        self.assertTrue(first.ungoverned_slew_limited)
        self.assertTrue(first.jaw_slew_limited)
        self.assertTrue(first.ungoverned_slew_limited_mask[5])
        self.assertFalse(np.any(first.ungoverned_slew_limited_mask[:5]))
        self.assertFalse(first.velocity_clipped_mask[5])
        self.assertAlmostEqual(first.command[5], nominal[0, 5] - 0.05, places=7)

        post_reference = tracker._interpolate(reference, first.next_phase)
        tracker.observe_post_step(post_reference)
        second = tracker.step(post_reference, first.command)
        self.assertGreater(second.next_phase, first.next_phase)
        self.assertLessEqual(second.next_phase - first.next_phase, 1.0)

    def test_feedback_correction_is_per_joint_bounded(self) -> None:
        reference, nominal = paths()
        tracker = RETIMING.PhaseGovernedTrajectoryTracker(
            reference,
            nominal,
            config(
                feedback_gain=10.0,
                hard_error_rad=arm_tuple(1.0),
                max_correction_rad=(0.02, 0.03, 0.04, 0.05, 0.06),
            ),
        )
        actual = reference[0].copy()
        actual[:5] -= 0.10

        result = tracker.step(actual_position=actual, previous_command=nominal[0])

        self.assertTrue(result.correction_clipped)
        np.testing.assert_allclose(
            result.correction[:5],
            np.asarray([0.02, 0.03, 0.04, 0.05, 0.06]),
        )
        self.assertFalse(result.correction_clipped_mask[5])

    def test_velocity_and_joint_limits_are_applied_and_diagnosed(self) -> None:
        reference, nominal = paths()
        lower = (-1.0, -1.0, -1.0, -1.0, -1.0, 0.0)
        upper = (0.05, 1.0, 1.0, 1.0, 1.0, 0.35)
        tracker = RETIMING.PhaseGovernedTrajectoryTracker(
            reference,
            nominal,
            config(
                max_joint_speed_rad_s=(0.2,) * 6,
                joint_lower_limits_rad=lower,
                joint_upper_limits_rad=upper,
            ),
        )

        result = tracker.step(
            actual_position=reference[0],
            previous_command=np.asarray([0.049, 0.0, 0.0, 0.0, 0.0, 0.36]),
        )

        self.assertTrue(result.velocity_clipped)
        self.assertTrue(result.joint_limit_clipped)
        self.assertLessEqual(result.command[0], upper[0])
        self.assertLessEqual(result.command[5], upper[5])
        self.assertTrue(result.velocity_clipped_mask[0])
        self.assertTrue(result.joint_limit_clipped_mask[5])

    def test_post_step_observation_order_is_enforced(self) -> None:
        reference, nominal = paths()
        tracker = RETIMING.PhaseGovernedTrajectoryTracker(reference, nominal, config())

        with self.assertRaisesRegex(RuntimeError, r"step\(\) must"):
            tracker.observe_post_step(reference[0])

        tracker.step(reference[0], nominal[0])
        with self.assertRaisesRegex(RuntimeError, "observe_post_step"):
            tracker.step(reference[1], nominal[1])

    def test_terminal_completion_requires_three_post_step_settle_samples(self) -> None:
        reference, nominal = paths(frames=1)
        tracker = RETIMING.PhaseGovernedTrajectoryTracker(reference, nominal, config())

        first = tracker.step(reference[0], nominal[0])
        self.assertEqual(first.next_phase, tracker.end_phase)
        observed = tracker.observe_post_step(reference[-1])
        self.assertEqual(observed.terminal_settle_streak, 1)
        self.assertFalse(observed.complete)

        previous = first.command
        for expected_streak, expected_complete in ((2, False), (3, True)):
            held = tracker.step(reference[-1], previous)
            self.assertTrue(held.at_end)
            observed = tracker.observe_post_step(reference[-1])
            self.assertEqual(observed.terminal_settle_streak, expected_streak)
            self.assertEqual(observed.complete, expected_complete)
            previous = held.command

        self.assertEqual(observed.completion_reason, "settled")
        self.assertFalse(observed.limit_reached)

    def test_final_settle_limit_is_a_controller_failure(self) -> None:
        reference, nominal = paths(frames=1)
        tracker = RETIMING.PhaseGovernedTrajectoryTracker(
            reference,
            nominal,
            config(terminal_settle_steps=2, max_final_settle_steps=3),
        )
        previous = nominal[0]
        actual = reference[-1].copy()
        actual[0] += 0.20

        # The first command advances from the valid initial pose to the end.
        first = tracker.step(reference[0], previous)
        result = tracker.observe_post_step(actual)
        previous = first.command
        for _ in range(2):
            held = tracker.step(actual, previous)
            result = tracker.observe_post_step(actual)
            previous = held.command

        self.assertTrue(result.complete)
        self.assertTrue(result.limit_reached)
        self.assertEqual(result.completion_reason, "final_settle_limit")

    def test_total_recovery_limit_stops_an_untrackable_path(self) -> None:
        reference, nominal = paths(frames=2)
        tracker = RETIMING.PhaseGovernedTrajectoryTracker(
            reference,
            nominal,
            config(min_recovery_steps=0, recovery_fraction=0.0),
        )
        bad = reference[0].copy()
        bad[0] -= 1.0
        previous = nominal[0]

        for expected_complete in (False, True):
            step = tracker.step(bad, previous)
            self.assertEqual(step.next_phase, 0.0)
            observed = tracker.observe_post_step(bad)
            self.assertEqual(observed.complete, expected_complete)
            previous = step.command

        self.assertTrue(observed.limit_reached)
        self.assertEqual(observed.completion_reason, "recovery_limit")

    def test_diagnostics_aggregate_errors_pauses_and_clamps(self) -> None:
        reference, nominal = paths(frames=2)
        tracker = RETIMING.PhaseGovernedTrajectoryTracker(
            reference,
            nominal,
            config(
                max_joint_speed_rad_s=(0.01,) * 6,
                min_recovery_steps=5,
            ),
        )
        bad = reference[0].copy()
        bad[0] -= 0.20

        step = tracker.step(bad, nominal[0])
        tracker.observe_post_step(bad)
        diagnostics = tracker.diagnostics()

        self.assertEqual(diagnostics["paused_frames"], 1)
        self.assertEqual(diagnostics["longest_pause_frames"], 1)
        self.assertEqual(diagnostics["velocity_clipped_frames"], 1)
        self.assertGreater(diagnostics["max_arm_tracking_error_rad"], 0.0)
        self.assertEqual(diagnostics["allowed_recovery_frames"], 5)
        self.assertEqual(len(diagnostics["arm_tracking_error_rms_rad"]), 5)

    def test_diagnostics_count_jaw_limited_phase_frames(self) -> None:
        reference, nominal = paths()
        tracker = RETIMING.PhaseGovernedTrajectoryTracker(
            reference,
            nominal,
            config(max_joint_speed_rad_s=(100.0,) * 5 + (0.5,)),
        )

        result = tracker.step(reference[0], nominal[0])
        tracker.observe_post_step(
            tracker._interpolate(reference, result.next_phase)
        )
        diagnostics = tracker.diagnostics()

        self.assertEqual(diagnostics["ungoverned_slew_limited_frames"], 1)
        self.assertEqual(diagnostics["jaw_slew_limited_frames"], 1)
        self.assertEqual(diagnostics["ungoverned_slew_limited_counts"][:5], [0] * 5)
        self.assertEqual(diagnostics["ungoverned_slew_limited_counts"][5], 1)

    def test_trackers_keep_independent_phase_and_pause_state(self) -> None:
        reference, nominal = paths()
        first = RETIMING.PhaseGovernedTrajectoryTracker(reference, nominal, config())
        second = RETIMING.PhaseGovernedTrajectoryTracker(reference, nominal, config())
        bad = reference[0].copy()
        bad[0] -= 1.0

        first_result = first.step(reference[0], nominal[0])
        second_result = second.step(bad, nominal[0])

        self.assertEqual(first_result.next_phase, 1.0)
        self.assertEqual(second_result.next_phase, 0.0)
        self.assertFalse(first_result.hard_paused)
        self.assertTrue(second_result.hard_paused)

    def test_configuration_and_inputs_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "hard_error_rad"):
            config(
                soft_error_rad=arm_tuple(0.2),
                hard_error_rad=arm_tuple(0.1),
            )
        with self.assertRaisesRegex(ValueError, "provided together"):
            config(joint_lower_limits_rad=(-1.0,) * 6)

        reference, nominal = paths()
        with self.assertRaisesRegex(ValueError, "identical"):
            RETIMING.PhaseGovernedTrajectoryTracker(
                reference,
                nominal[:-1],
                config(),
            )
        tracker = RETIMING.PhaseGovernedTrajectoryTracker(reference, nominal, config())
        invalid = reference[0].copy()
        invalid[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            tracker.step(invalid, nominal[0])


if __name__ == "__main__":
    unittest.main()
