from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch


PACKAGE_ROOT = Path(__file__).parents[1] / "so101_bench"

PACKAGE = sys.modules.setdefault("so101_bench", types.ModuleType("so101_bench"))
PACKAGE.__path__ = [str(PACKAGE_ROOT)]

BENCHMARK_SPEC = importlib.util.spec_from_file_location("so101_bench.benchmark", PACKAGE_ROOT / "benchmark.py")
assert BENCHMARK_SPEC is not None
assert BENCHMARK_SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(BENCHMARK_SPEC)
sys.modules[BENCHMARK_SPEC.name] = BENCHMARK
BENCHMARK_SPEC.loader.exec_module(BENCHMARK)

ISAACLAB = sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
ISAACLAB.__path__ = []
ISAACLAB_UTILS = sys.modules.setdefault("isaaclab.utils", types.ModuleType("isaaclab.utils"))
ISAACLAB_UTILS.__path__ = []
sys.modules.setdefault("isaaclab.utils.math", types.ModuleType("isaaclab.utils.math"))

ISAACLAB_ASSETS = sys.modules.setdefault("isaaclab.assets", types.ModuleType("isaaclab.assets"))
ISAACLAB_ASSETS.RigidObject = object
ISAACLAB_ENVS = sys.modules.setdefault("isaaclab.envs", types.ModuleType("isaaclab.envs"))
ISAACLAB_ENVS.ManagerBasedRLEnv = object
ISAACLAB_MANAGERS = sys.modules.setdefault("isaaclab.managers", types.ModuleType("isaaclab.managers"))


class SceneEntityCfg:
    def __init__(self, name: str) -> None:
        self.name = name


ISAACLAB_MANAGERS.SceneEntityCfg = SceneEntityCfg

MDP_PACKAGE = sys.modules.setdefault("so101_bench.mdp", types.ModuleType("so101_bench.mdp"))
MDP_PACKAGE.__path__ = [str(PACKAGE_ROOT / "mdp")]
RESETS = types.ModuleType("so101_bench.mdp.resets")
RESETS.benchmark_object_positions = lambda *_args, **_kwargs: None
RESETS.benchmark_object_yaws = lambda *_args, **_kwargs: None
RESETS.mark_benchmark_robot_start = lambda *_args, **_kwargs: None
# Returning None means "no slot is known to be parked", so contact masks scan every
# slot unless a test overrides this to exercise the pre-spawned-pool skip.
RESETS.slots_live_in_any_env = lambda *_args, **_kwargs: None
sys.modules[RESETS.__name__] = RESETS

TERMINATIONS_SPEC = importlib.util.spec_from_file_location(
    "so101_bench.mdp.terminations",
    PACKAGE_ROOT / "mdp" / "terminations.py",
)
assert TERMINATIONS_SPEC is not None
assert TERMINATIONS_SPEC.loader is not None
TERMINATIONS = importlib.util.module_from_spec(TERMINATIONS_SPEC)
sys.modules[TERMINATIONS_SPEC.name] = TERMINATIONS
TERMINATIONS_SPEC.loader.exec_module(TERMINATIONS)


class TerminationOptimizationTests(unittest.TestCase):
    def test_named_bin_scores_only_the_instructed_target(self) -> None:
        original_math_utils = TERMINATIONS.math_utils

        class FakeEnv:
            num_envs = 2
            device = "cpu"
            _so101_task_family = [BENCHMARK.TASK_BIN, BENCHMARK.TASK_NAMED_BIN]
            _so101_active_object_mask = torch.tensor([[True, True], [True, True]])
            _so101_target_object_ids = torch.tensor([0, 1])
            _so101_bin_success_counter = torch.zeros(2, dtype=torch.long)
            _so101_bin_footprint_half_extents = torch.tensor([[0.1, 0.1], [0.1, 0.1]])
            _so101_bin_footprint_center_offsets = torch.zeros(2, 2)
            scene = {
                "bin": types.SimpleNamespace(
                    data=types.SimpleNamespace(
                        root_pos_w=torch.zeros(2, 3),
                        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
                    )
                )
            }

        positions = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
                [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        )
        try:
            TERMINATIONS.math_utils = types.SimpleNamespace(
                quat_inv=lambda quat: quat,
                quat_apply=lambda _quat, vector: vector,
            )
            result = TERMINATIONS.bin_success(
                FakeEnv(),
                ["object_1", "object_2"],
                "bin",
                confirm_steps=1,
                step_state=TERMINATIONS._TerminationStepState(positions=positions),
            )
        finally:
            TERMINATIONS.math_utils = original_math_utils

        self.assertEqual(result.tolist(), [False, True])

    def test_named_bin_grasp_attempt_limit_applies_only_to_target(self) -> None:
        class FakeEnv:
            num_envs = 1
            device = "cpu"
            _so101_task_family = [BENCHMARK.TASK_NAMED_BIN]
            _so101_active_object_mask = torch.tensor([[True, True, True, True]])
            _so101_target_object_ids = torch.tensor([2])

        mask = TERMINATIONS._attempt_object_mask(
            FakeEnv(),
            ["object_1", "object_2", "object_3", "object_4"],
        )

        self.assertEqual(mask.tolist(), [[False, False, True, False]])

    def test_named_bin_postmortem_goal_uses_target_only(self) -> None:
        original_math_utils = TERMINATIONS.math_utils

        class FakeEnv:
            num_envs = 1
            device = "cpu"
            _so101_task_family = [BENCHMARK.TASK_NAMED_BIN]
            _so101_active_object_mask = torch.tensor([[True, True]])
            _so101_target_object_ids = torch.tensor([1])
            _so101_bin_footprint_half_extents = torch.tensor([[0.1, 0.1]])
            _so101_bin_footprint_center_offsets = torch.zeros(1, 2)
            scene = {
                "bin": types.SimpleNamespace(
                    data=types.SimpleNamespace(
                        root_pos_w=torch.zeros(1, 3),
                        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                    )
                )
            }

        try:
            TERMINATIONS.math_utils = types.SimpleNamespace(
                quat_inv=lambda quat: quat,
                quat_apply=lambda _quat, vector: vector,
            )
            goal = TERMINATIONS._postmortem_goal_step(
                FakeEnv(),
                ["object_1", "object_2"],
                "bin",
                {"x": (0.0, 1.0), "y": (0.0, 1.0)},
                TERMINATIONS._TerminationStepState(
                    positions=torch.tensor([[[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]])
                ),
                move_straightness_tolerance=0.05,
                move_past_boundary_tolerance=0.01,
            )
        finally:
            TERMINATIONS.math_utils = original_math_utils

        self.assertEqual(goal.met.tolist(), [True])
        self.assertEqual(goal.metrics[0]["name"], "target_inside_bin")
        self.assertEqual(goal.metrics[0]["target_object_id"], 1)

    def test_task_timeout_uses_long_limit_only_for_four_object_bin_tasks(self) -> None:
        class FakeEnv:
            num_envs = 3
            device = "cpu"
            step_dt = 1.0
            episode_length_buf = torch.tensor(
                [
                    int(BENCHMARK.DEFAULT_EPISODE_LENGTH_S),
                    int(BENCHMARK.FOUR_OBJECT_BIN_EPISODE_LENGTH_S) - 1,
                    int(BENCHMARK.DEFAULT_EPISODE_LENGTH_S),
                ]
            )
            _so101_task_family = [BENCHMARK.TASK_BIN, BENCHMARK.TASK_BIN, BENCHMARK.TASK_MOVE]
            _so101_active_object_mask = torch.tensor(
                [
                    [True, False, False, False],
                    [True, True, True, True],
                    [True, True, True, True],
                ]
            )

        env = FakeEnv()
        self.assertEqual(TERMINATIONS.task_time_out(env).tolist(), [True, False, True])
        env.episode_length_buf[1] = int(BENCHMARK.FOUR_OBJECT_BIN_EPISODE_LENGTH_S)
        self.assertEqual(TERMINATIONS.task_time_out(env).tolist(), [True, True, True])

    def test_task_timeout_scales_budget_for_retimed_replay(self) -> None:
        original_episode_length_s = TERMINATIONS.episode_length_s

        class FakeEnv:
            num_envs = 1
            device = "cpu"
            step_dt = 1.0
            episode_length_buf = torch.tensor([27])
            _so101_task_family = [BENCHMARK.TASK_NEXT_TO]
            _so101_active_object_mask = torch.tensor([[True]])

        env = FakeEnv()
        try:
            TERMINATIONS.episode_length_s = lambda *_args: 20.0
            self.assertEqual(TERMINATIONS.task_time_out(env).tolist(), [True])
            self.assertEqual(
                TERMINATIONS.task_time_out(env, duration_scale=1.4).tolist(),
                [False],
            )
            env.episode_length_buf[0] = 28
            self.assertEqual(
                TERMINATIONS.task_time_out(env, duration_scale=1.4).tolist(),
                [True],
            )
        finally:
            TERMINATIONS.episode_length_s = original_episode_length_s

    def test_task_timeout_extends_pending_success_confirmation(self) -> None:
        original_episode_length_s = TERMINATIONS.episode_length_s

        class FakeEnv:
            num_envs = 1
            device = "cpu"
            step_dt = 1.0
            episode_length_buf = torch.tensor([20])
            _so101_task_family = [BENCHMARK.TASK_BIN]
            _so101_active_object_mask = torch.tensor([[True]])
            _so101_bin_success_counter = torch.tensor([1])

        env = FakeEnv()
        try:
            TERMINATIONS.episode_length_s = lambda *_args: 20.0
            self.assertEqual(TERMINATIONS.task_time_out(env, confirm_time_s=3.0).tolist(), [False])
            env._so101_bin_success_counter[0] = 2
            self.assertEqual(TERMINATIONS.task_time_out(env, confirm_time_s=3.0).tolist(), [False])
            env._so101_bin_success_counter[0] = 3
            self.assertEqual(TERMINATIONS.task_time_out(env, confirm_time_s=3.0).tolist(), [False])
        finally:
            TERMINATIONS.episode_length_s = original_episode_length_s

        self.assertEqual(env._so101_timeout_success_confirmation_active.tolist(), [True])
        self.assertEqual(env._so101_timeout_success_confirmation_failed.tolist(), [False])

    def test_task_timeout_marks_breached_confirmation_extension_as_failure(self) -> None:
        original_episode_length_s = TERMINATIONS.episode_length_s

        class FakeEnv:
            num_envs = 1
            device = "cpu"
            step_dt = 1.0
            episode_length_buf = torch.tensor([20])
            _so101_task_family = [BENCHMARK.TASK_BIN]
            _so101_active_object_mask = torch.tensor([[True]])
            _so101_bin_success_counter = torch.tensor([1])

        env = FakeEnv()
        try:
            TERMINATIONS.episode_length_s = lambda *_args: 20.0
            self.assertEqual(TERMINATIONS.task_time_out(env, confirm_time_s=3.0).tolist(), [False])
            env._so101_bin_success_counter[0] = 0
            self.assertEqual(TERMINATIONS.task_time_out(env, confirm_time_s=3.0).tolist(), [False])
        finally:
            TERMINATIONS.episode_length_s = original_episode_length_s

        self.assertEqual(env._so101_timeout_success_confirmation_failed.tolist(), [True])

    def test_step_state_reuses_positions_and_lazily_computes_yaws(self) -> None:
        calls = {"positions": 0, "yaws": 0}
        original_positions = TERMINATIONS._object_positions
        original_yaws = TERMINATIONS.benchmark_object_yaws

        class FakeEnv:
            common_step_counter = 7
            num_envs = 1
            device = "cpu"
            _so101_active_object_mask = torch.tensor([[True]])

        env = FakeEnv()

        def positions(*_args):
            calls["positions"] += 1
            return torch.zeros(1, 1, 3)

        def yaws(*_args):
            calls["yaws"] += 1
            return torch.zeros(1, 1)

        try:
            TERMINATIONS._object_positions = positions
            TERMINATIONS.benchmark_object_yaws = yaws
            first = TERMINATIONS._termination_step_state(env, ["object_1"])
            second = TERMINATIONS._termination_step_state(env, ["object_1"])
            TERMINATIONS._state_object_yaws(env, ["object_1"], first)
            TERMINATIONS._state_object_yaws(env, ["object_1"], second)
            env.common_step_counter += 1
            TERMINATIONS._termination_step_state(env, ["object_1"])
        finally:
            TERMINATIONS._object_positions = original_positions
            TERMINATIONS.benchmark_object_yaws = original_yaws

        self.assertIs(first, second)
        self.assertEqual(calls, {"positions": 2, "yaws": 1})

    def test_task_success_dispatches_only_present_families(self) -> None:
        calls: list[str] = []
        originals = {
            "bin_success": TERMINATIONS.bin_success,
            "next_to_success": TERMINATIONS.next_to_success,
            "between_success": TERMINATIONS.between_success,
            "move_success": TERMINATIONS.move_success,
            "_termination_step_state": TERMINATIONS._termination_step_state,
        }

        class FakeEnv:
            num_envs = 2
            device = "cpu"
            step_dt = 1.0
            episode_length_buf = torch.tensor([10, 10])
            _so101_task_family = [BENCHMARK.TASK_BIN, BENCHMARK.TASK_MOVE]

        def success(name: str):
            def record(*_args, **_kwargs):
                calls.append(name)
                return torch.tensor([True, True])

            return record

        try:
            TERMINATIONS.bin_success = success("bin")
            TERMINATIONS.next_to_success = success("next_to")
            TERMINATIONS.between_success = success("between")
            TERMINATIONS.move_success = success("move")
            TERMINATIONS._termination_step_state = lambda *_args: object()
            result = TERMINATIONS.task_success(FakeEnv(), ["object_1"], "bin")
        finally:
            for name, value in originals.items():
                setattr(TERMINATIONS, name, value)

        self.assertEqual(calls, ["bin", "move"])
        self.assertEqual(result.tolist(), [True, True])

    def test_task_success_is_blocked_by_an_unconfirmed_failure_condition(self) -> None:
        originals = {
            "bin_success": TERMINATIONS.bin_success,
            "move_success": TERMINATIONS.move_success,
            "_termination_step_state": TERMINATIONS._termination_step_state,
        }

        class FakeEnv:
            num_envs = 2
            device = "cpu"
            step_dt = 1.0
            episode_length_buf = torch.tensor([10, 10])
            _so101_task_family = [BENCHMARK.TASK_BIN, BENCHMARK.TASK_MOVE]
            _so101_bin_success_counter = torch.tensor([3, 0])
            _so101_move_success_counter = torch.tensor([0, 3])
            # This mask is written before benchmark_failure applies its
            # five-second continuous-hold confirmation gate.
            _so101_failure_conditions_active = torch.tensor([True, False])

        try:
            TERMINATIONS.bin_success = lambda *_args, **_kwargs: torch.tensor([True, True])
            TERMINATIONS.move_success = lambda *_args, **_kwargs: torch.tensor([True, True])
            TERMINATIONS._termination_step_state = lambda *_args: object()
            env = FakeEnv()
            result = TERMINATIONS.task_success(env, ["object_1"], "bin")
        finally:
            for name, value in originals.items():
                setattr(TERMINATIONS, name, value)

        self.assertEqual(result.tolist(), [False, True])
        self.assertEqual(env._so101_bin_success_counter.tolist(), [0, 0])
        self.assertEqual(env._so101_move_success_counter.tolist(), [0, 3])

    def test_task_success_persists_actual_confirmation_requirement(self) -> None:
        originals = {
            "next_to_success": TERMINATIONS.next_to_success,
            "_termination_step_state": TERMINATIONS._termination_step_state,
            "benchmark_postmortem_failure_diagnostics": TERMINATIONS.benchmark_postmortem_failure_diagnostics,
        }
        postmortem_calls = []

        class FakeEnv:
            num_envs = 1
            device = "cpu"
            step_dt = 0.1
            episode_length_buf = torch.tensor([50])
            _so101_task_family = [BENCHMARK.TASK_NEXT_TO]
            _so101_pm_tracker_available = True

        try:
            TERMINATIONS.next_to_success = lambda *_args, **_kwargs: torch.tensor([True])
            TERMINATIONS._termination_step_state = lambda *_args: object()
            TERMINATIONS.benchmark_postmortem_failure_diagnostics = (
                lambda *_args, **kwargs: postmortem_calls.append(kwargs) or ["refreshed"]
            )
            env = FakeEnv()
            result = TERMINATIONS.task_success(
                env,
                ["object_1"],
                "bin",
                min_episode_time_s=0.0,
                confirm_time_s=0.25,
            )
        finally:
            for name, value in originals.items():
                setattr(TERMINATIONS, name, value)

        self.assertEqual(result.tolist(), [True])
        self.assertEqual(env._so101_success_confirmation_required_steps.tolist(), [3])
        self.assertEqual(env._so101_success_confirmed_current.tolist(), [True])
        self.assertEqual(env._so101_success_ever_confirmed.tolist(), [True])
        self.assertEqual(env._so101_postmortem_failure_diagnostics, ["refreshed"])
        self.assertEqual(postmortem_calls, [{"confirm_time_s": 0.25}])

    def test_move_task_diagnostic_reports_boundary_distance_and_source(self) -> None:
        originals = {
            "_move_boundary_distance": TERMINATIONS._move_boundary_distance,
            "grasped_object_made_contact": TERMINATIONS.grasped_object_made_contact,
            "target_object_made_contact": TERMINATIONS.target_object_made_contact,
        }

        class FakeEnv:
            num_envs = 1
            device = "cpu"
            step_dt = 0.1
            episode_length_buf = torch.tensor([50])
            _so101_move_boundary_ids = torch.tensor([-1])
            _so101_move_success_counter = torch.tensor([2])

        try:
            TERMINATIONS._move_boundary_distance = lambda *_args: (
                torch.tensor([0.031]),
                torch.tensor([0.082]),
                torch.tensor([0.004]),
                torch.tensor([[0.0, 0.0]]),
            )
            TERMINATIONS.grasped_object_made_contact = lambda *_args: torch.tensor([False])
            TERMINATIONS.target_object_made_contact = lambda *_args: torch.tensor([False])
            diagnostic = TERMINATIONS._move_task_diagnostic(
                FakeEnv(),
                ["object_1"],
                {"x": (-0.14, 0.25), "y": (-0.1, 0.155)},
                TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
                env_id=0,
                min_episode_time_s=0.5,
                confirm_time_s=0.2,
                straightness_tolerance=0.0508,
            )
        finally:
            for name, value in originals.items():
                setattr(TERMINATIONS, name, value)

        self.assertTrue(diagnostic.met)
        self.assertEqual(diagnostic.name, "target_moved_to_boundary")
        self.assertIn("boundary=none", diagnostic.details)
        self.assertIn("distance_to_boundary=0.0310m", diagnostic.details)
        self.assertIn("lateral_error=0.0040m", diagnostic.details)

    def test_move_boundary_cache_allows_no_object_boundary(self) -> None:
        class FakeEnv:
            num_envs = 1
            device = "cpu"
            scene = types.SimpleNamespace()
            _so101_initial_object_pos_w = torch.tensor([[[0.0, 0.0, 0.001]]])
            _so101_initial_object_yaws = torch.zeros(1, 1)
            _so101_object_footprint_half_extents = torch.full((1, 1, 2), 0.01)
            _so101_object_footprint_center_offsets = torch.zeros(1, 1, 2)
            _so101_active_object_mask = torch.tensor([[True]])
            _so101_target_object_ids = torch.tensor([0])
            _so101_direction_ids = torch.tensor([0])

        env = FakeEnv()
        TERMINATIONS._ensure_move_boundary_cache(
            env,
            ["object_1"],
            {"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
        )

        self.assertEqual(env._so101_move_boundary_ids.tolist(), [-1])
        self.assertTrue(torch.isnan(env._so101_move_boundary_coords).all().item())

    def test_move_boundary_cache_ignores_laterally_non_overlapping_object(self) -> None:
        class FakeEnv:
            num_envs = 1
            device = "cpu"
            scene = types.SimpleNamespace()
            _so101_initial_object_pos_w = torch.tensor([[[0.0, 0.0, 0.001], [0.08, 0.04, 0.001]]])
            _so101_initial_object_yaws = torch.zeros(1, 2)
            _so101_object_footprint_half_extents = torch.full((1, 2, 2), 0.01)
            _so101_object_footprint_center_offsets = torch.zeros(1, 2, 2)
            _so101_active_object_mask = torch.tensor([[True, True]])
            _so101_target_object_ids = torch.tensor([0])
            _so101_direction_ids = torch.tensor([0])

        env = FakeEnv()
        TERMINATIONS._ensure_move_boundary_cache(
            env,
            ["object_1", "object_2"],
            {"x": (-0.02, 0.10), "y": (-0.02, 0.06)},
        )

        self.assertEqual(env._so101_move_boundary_ids.tolist(), [-1])
        self.assertTrue(torch.isnan(env._so101_move_boundary_coords).all().item())

    def test_move_boundary_cache_ignores_glancing_lateral_overlap(self) -> None:
        # object_2 sits in front (along +x) but is offset in y so its footprint clips
        # only ~1mm of the target's 20mm-wide corridor -- below the 10% threshold, so the
        # move falls back to forward-progress scoring instead of "reaching" this object.
        class FakeEnv:
            num_envs = 1
            device = "cpu"
            scene = types.SimpleNamespace()
            _so101_initial_object_pos_w = torch.tensor([[[0.0, 0.0, 0.001], [0.08, 0.019, 0.001]]])
            _so101_initial_object_yaws = torch.zeros(1, 2)
            _so101_object_footprint_half_extents = torch.full((1, 2, 2), 0.01)
            _so101_object_footprint_center_offsets = torch.zeros(1, 2, 2)
            _so101_active_object_mask = torch.tensor([[True, True]])
            _so101_target_object_ids = torch.tensor([0])
            _so101_direction_ids = torch.tensor([0])

        env = FakeEnv()
        TERMINATIONS._ensure_move_boundary_cache(
            env,
            ["object_1", "object_2"],
            {"x": (-0.02, 0.10), "y": (-0.02, 0.06)},
        )

        self.assertEqual(env._so101_move_boundary_ids.tolist(), [-1])

    def test_move_boundary_cache_keeps_sufficiently_aligned_object(self) -> None:
        # Same geometry but object_2 now overlaps ~15mm of the corridor: a real boundary.
        class FakeEnv:
            num_envs = 1
            device = "cpu"
            scene = types.SimpleNamespace()
            _so101_initial_object_pos_w = torch.tensor([[[0.0, 0.0, 0.001], [0.08, 0.005, 0.001]]])
            _so101_initial_object_yaws = torch.zeros(1, 2)
            _so101_object_footprint_half_extents = torch.full((1, 2, 2), 0.01)
            _so101_object_footprint_center_offsets = torch.zeros(1, 2, 2)
            _so101_active_object_mask = torch.tensor([[True, True]])
            _so101_target_object_ids = torch.tensor([0])
            _so101_direction_ids = torch.tensor([0])

        env = FakeEnv()
        TERMINATIONS._ensure_move_boundary_cache(
            env,
            ["object_1", "object_2"],
            {"x": (-0.02, 0.10), "y": (-0.02, 0.06)},
        )

        self.assertEqual(env._so101_move_boundary_ids.tolist(), [1])

    def test_move_boundary_cache_selects_forward_blocker_over_distant_object(self) -> None:
        # object_2 overlaps the target's forward lane (negative gap) while object_3 sits clear
        # further along. The nearest obstruction -- the blocker -- must be the boundary, not the
        # distant object reached by stepping past the block.
        class FakeEnv:
            num_envs = 1
            device = "cpu"
            scene = types.SimpleNamespace()
            _so101_initial_object_pos_w = torch.tensor(
                [[[0.0, 0.0, 0.001], [0.03, 0.0, 0.001], [0.16, 0.0, 0.001]]]
            )
            _so101_initial_object_yaws = torch.zeros(1, 3)
            _so101_object_footprint_half_extents = torch.tensor(
                [[[0.01, 0.01], [0.03, 0.01], [0.01, 0.01]]]
            )
            _so101_object_footprint_center_offsets = torch.zeros(1, 3, 2)
            _so101_active_object_mask = torch.tensor([[True, True, True]])
            _so101_target_object_ids = torch.tensor([0])
            _so101_direction_ids = torch.tensor([0])

        env = FakeEnv()
        TERMINATIONS._ensure_move_boundary_cache(
            env,
            ["object_1", "object_2", "object_3"],
            {"x": (-0.2, 0.3), "y": (-0.2, 0.2)},
        )

        self.assertEqual(env._so101_move_boundary_ids.tolist(), [1])

    def test_move_boundary_cache_ignores_object_behind_leading_edge(self) -> None:
        # object_2 sits entirely behind the target's leading edge (negative gap but not ahead);
        # it is not in the move's path, so the clear object_3 ahead is the boundary.
        class FakeEnv:
            num_envs = 1
            device = "cpu"
            scene = types.SimpleNamespace()
            _so101_initial_object_pos_w = torch.tensor(
                [[[0.0, 0.0, 0.001], [-0.05, 0.0, 0.001], [0.16, 0.0, 0.001]]]
            )
            _so101_initial_object_yaws = torch.zeros(1, 3)
            _so101_object_footprint_half_extents = torch.tensor(
                [[[0.01, 0.01], [0.03, 0.01], [0.01, 0.01]]]
            )
            _so101_object_footprint_center_offsets = torch.zeros(1, 3, 2)
            _so101_active_object_mask = torch.tensor([[True, True, True]])
            _so101_target_object_ids = torch.tensor([0])
            _so101_direction_ids = torch.tensor([0])

        env = FakeEnv()
        TERMINATIONS._ensure_move_boundary_cache(
            env,
            ["object_1", "object_2", "object_3"],
            {"x": (-0.2, 0.3), "y": (-0.2, 0.2)},
        )

        self.assertEqual(env._so101_move_boundary_ids.tolist(), [2])

    def test_move_boundary_cache_ignores_empty_space_inside_concave_footprint(self) -> None:
        class FakeEnv:
            num_envs = 1
            device = "cpu"
            scene = types.SimpleNamespace()
            _so101_initial_object_pos_w = torch.tensor([[[0.0, 0.0, 0.001], [0.08, 0.0, 0.001]]])
            _so101_initial_object_yaws = torch.zeros(1, 2)
            _so101_object_footprint_half_extents = torch.tensor([[[0.01, 0.01], [0.01, 0.03]]])
            _so101_object_footprint_center_offsets = torch.zeros(1, 2, 2)
            _so101_object_move_footprint_boxes = [
                torch.tensor([[-0.01, -0.01, 0.01, 0.01]]),
                torch.tensor([[-0.01, -0.03, 0.01, -0.02], [-0.01, 0.02, 0.01, 0.03]]),
            ]
            _so101_active_object_mask = torch.tensor([[True, True]])
            _so101_target_object_ids = torch.tensor([0])
            _so101_direction_ids = torch.tensor([0])

        env = FakeEnv()
        TERMINATIONS._ensure_move_boundary_cache(
            env,
            ["object_1", "object_2"],
            {"x": (-0.02, 0.10), "y": (-0.04, 0.04)},
        )

        self.assertEqual(env._so101_move_boundary_ids.tolist(), [-1])
        self.assertTrue(torch.isnan(env._so101_move_boundary_coords).all().item())

    def test_move_boundary_distance_detects_crossing_a_concave_footprint_piece(self) -> None:
        class FakeEnv:
            num_envs = 1
            device = "cpu"
            scene = types.SimpleNamespace()
            _so101_initial_object_pos_w = torch.tensor([[[0.0, 0.025, 0.001], [0.08, 0.0, 0.001]]])
            _so101_initial_object_yaws = torch.zeros(1, 2)
            _so101_object_footprint_half_extents = torch.tensor([[[0.01, 0.01], [0.01, 0.03]]])
            _so101_object_footprint_center_offsets = torch.zeros(1, 2, 2)
            _so101_object_move_footprint_boxes = [
                torch.tensor([[-0.01, -0.01, 0.01, 0.01]]),
                torch.tensor([[-0.01, -0.03, 0.01, -0.02], [-0.01, 0.02, 0.01, 0.03]]),
            ]
            _so101_active_object_mask = torch.tensor([[True, True]])
            _so101_target_object_ids = torch.tensor([0])
            _so101_direction_ids = torch.tensor([0])

        env = FakeEnv()
        current_positions = torch.tensor([[[0.075, 0.025, 0.001], [0.08, 0.0, 0.001]]])
        distance, _progress, _lateral, _target = TERMINATIONS._move_boundary_distance(
            env,
            ["object_1", "object_2"],
            {"x": (-0.02, 0.10), "y": (-0.04, 0.04)},
            TERMINATIONS._TerminationStepState(positions=current_positions, yaws=torch.zeros(1, 2)),
        )

        self.assertEqual(env._so101_move_boundary_ids.tolist(), [1])
        self.assertLess(float(distance[0].item()), -0.002)

    def test_directional_footprint_gap_measures_clearance_at_target_lateral_band(self) -> None:
        # A small target whose leading edge (max x) is at 0.02, moving +x ("left").
        target = torch.tensor([[[0.0, 0.0], [0.02, 0.0], [0.02, 0.02], [0.0, 0.02]]])
        # A diagonal boundary strip: its nearest *corner* (min x) is 0.01 -- behind the
        # target front -- but that corner sits far away at y=0.1. Across the target's
        # own lateral band (y in [0, 0.02]) the boundary surface is ~0.046, well ahead.
        boundary = torch.tensor([[[0.05, 0.0], [0.06, 0.0], [0.02, 0.1], [0.01, 0.1]]])

        gap = TERMINATIONS._directional_footprint_gap(target, boundary, axis=0, sign=1.0)

        # Reducing the boundary to its nearest corner would report a spurious crossing
        # (0.01 - 0.02 = -0.01); the per-lateral clearance is comfortably positive.
        self.assertIsNotNone(gap)
        self.assertGreater(gap, 0.0)

    def test_directional_footprint_gap_none_without_lateral_overlap(self) -> None:
        target = torch.tensor([[[0.0, 0.0], [0.02, 0.0], [0.02, 0.02], [0.0, 0.02]]])
        boundary = torch.tensor([[[0.05, 0.1], [0.07, 0.1], [0.07, 0.12], [0.05, 0.12]]])
        self.assertIsNone(TERMINATIONS._directional_footprint_gap(target, boundary, axis=0, sign=1.0))

    def test_move_success_uses_two_inch_fallback_without_boundary(self) -> None:
        originals = {
            "_move_boundary_distance": TERMINATIONS._move_boundary_distance,
            "_grasped_object_contact_allows_success": TERMINATIONS._grasped_object_contact_allows_success,
        }

        class FakeEnv:
            num_envs = 1
            device = "cpu"
            _so101_task_family = [BENCHMARK.TASK_MOVE]
            _so101_move_boundary_ids = torch.tensor([-1])

        env = FakeEnv()
        progress = torch.tensor([BENCHMARK.SPATIAL_SUCCESS_DISTANCE_M])
        try:
            TERMINATIONS._move_boundary_distance = lambda *_args: (
                torch.tensor([float("nan")]),
                progress,
                torch.tensor([0.001]),
                torch.tensor([[0.0, 0.0]]),
            )
            TERMINATIONS._grasped_object_contact_allows_success = lambda *_args: torch.tensor([True])
            success = TERMINATIONS.move_success(
                env,
                ["object_1"],
                confirm_steps=1,
                step_state=TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
            )
        finally:
            for name, value in originals.items():
                setattr(TERMINATIONS, name, value)

        self.assertEqual(success.tolist(), [True])

    def test_next_to_success_requires_one_inch_target_displacement(self) -> None:
        originals = {
            "_state_object_yaws": TERMINATIONS._state_object_yaws,
            "_pairwise_object_surface_distance": TERMINATIONS._pairwise_object_surface_distance,
            "_grasped_object_contact_allows_success": TERMINATIONS._grasped_object_contact_allows_success,
        }

        class FakeEnv:
            num_envs = 1
            device = "cpu"
            _so101_task_family = [BENCHMARK.TASK_NEXT_TO]
            _so101_target_object_ids = torch.tensor([0])
            _so101_referent_object_ids = torch.tensor([[1, 1]])
            _so101_initial_object_pos_w = torch.zeros(1, 2, 3)

        env = FakeEnv()
        positions = torch.zeros(1, 2, 3)
        try:
            TERMINATIONS._state_object_yaws = lambda *_args: torch.zeros(1, 2)
            TERMINATIONS._pairwise_object_surface_distance = lambda *_args: torch.tensor([0.0])
            TERMINATIONS._grasped_object_contact_allows_success = lambda *_args: torch.tensor([True])
            positions[0, 0, 0] = BENCHMARK.NON_BIN_TARGET_MIN_DISPLACEMENT_M - 1.0e-5
            below_minimum = TERMINATIONS.next_to_success(
                env,
                ["object_1", "object_2"],
                confirm_steps=1,
                step_state=TERMINATIONS._TerminationStepState(positions=positions),
            )
            positions[0, 0, 0] = BENCHMARK.NON_BIN_TARGET_MIN_DISPLACEMENT_M
            at_minimum = TERMINATIONS.next_to_success(
                env,
                ["object_1", "object_2"],
                confirm_steps=1,
                step_state=TERMINATIONS._TerminationStepState(positions=positions),
            )
        finally:
            for name, value in originals.items():
                setattr(TERMINATIONS, name, value)

        self.assertEqual(below_minimum.tolist(), [False])
        self.assertEqual(at_minimum.tolist(), [True])

    def test_between_success_uses_one_point_seven_five_inch_line_tolerance(self) -> None:
        originals = {
            "_state_object_yaws": TERMINATIONS._state_object_yaws,
            "_pairwise_object_surface_distance": TERMINATIONS._pairwise_object_surface_distance,
            "_grasped_object_contact_allows_success": TERMINATIONS._grasped_object_contact_allows_success,
        }

        class FakeEnv:
            num_envs = 1
            device = "cpu"
            _so101_task_family = [BENCHMARK.TASK_BETWEEN]
            _so101_target_object_ids = torch.tensor([0])
            _so101_referent_object_ids = torch.tensor([[1, 2]])
            _so101_initial_object_pos_w = torch.zeros(1, 3, 3)

        env = FakeEnv()
        tolerance = 1.75 * BENCHMARK.INCH
        self.assertAlmostEqual(BENCHMARK.BETWEEN_LINE_TOLERANCE_M, tolerance)
        positions = torch.tensor(
            [[[BENCHMARK.NON_BIN_TARGET_MIN_DISPLACEMENT_M, tolerance, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]]
        )
        try:
            TERMINATIONS._state_object_yaws = lambda *_args: torch.zeros(1, 3)
            TERMINATIONS._pairwise_object_surface_distance = lambda *_args: torch.tensor([0.1])
            TERMINATIONS._grasped_object_contact_allows_success = lambda *_args: torch.tensor([True])
            at_tolerance = TERMINATIONS.between_success(
                env,
                ["object_1", "object_2", "object_3"],
                confirm_steps=1,
                step_state=TERMINATIONS._TerminationStepState(positions=positions),
            )
            positions[0, 0, 1] = tolerance + 1.0e-5
            above_tolerance = TERMINATIONS.between_success(
                env,
                ["object_1", "object_2", "object_3"],
                confirm_steps=1,
                step_state=TERMINATIONS._TerminationStepState(positions=positions),
            )
            # This target is exactly on the infinite referent line, but lies beyond
            # the first referent.  Perpendicular distance alone must not pass it.
            positions[0, 0, 0] = -0.03
            positions[0, 0, 1] = 0.0
            outside_referent_span = TERMINATIONS.between_success(
                env,
                ["object_1", "object_2", "object_3"],
                confirm_steps=1,
                step_state=TERMINATIONS._TerminationStepState(positions=positions),
            )
        finally:
            for name, value in originals.items():
                setattr(TERMINATIONS, name, value)

        self.assertEqual(at_tolerance.tolist(), [True])
        self.assertEqual(above_tolerance.tolist(), [False])
        self.assertEqual(outside_referent_span.tolist(), [False])

    def test_move_success_requires_one_inch_total_target_displacement(self) -> None:
        originals = {
            "_move_boundary_distance": TERMINATIONS._move_boundary_distance,
            "_grasped_object_contact_allows_success": TERMINATIONS._grasped_object_contact_allows_success,
        }

        class FakeEnv:
            num_envs = 1
            device = "cpu"
            _so101_task_family = [BENCHMARK.TASK_MOVE]
            _so101_move_boundary_ids = torch.tensor([0])

        env = FakeEnv()
        progress = torch.tensor([BENCHMARK.NON_BIN_TARGET_MIN_DISPLACEMENT_M - 1.0e-5])
        try:
            TERMINATIONS._move_boundary_distance = lambda *_args: (
                torch.tensor([0.0]),
                progress,
                torch.tensor([0.0]),
                torch.tensor([[0.0, 0.0]]),
            )
            TERMINATIONS._grasped_object_contact_allows_success = lambda *_args: torch.tensor([True])
            below_minimum = TERMINATIONS.move_success(
                env,
                ["object_1"],
                confirm_steps=1,
                step_state=TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
            )
            progress[:] = BENCHMARK.NON_BIN_TARGET_MIN_DISPLACEMENT_M
            at_minimum = TERMINATIONS.move_success(
                env,
                ["object_1"],
                confirm_steps=1,
                step_state=TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
            )
        finally:
            for name, value in originals.items():
                setattr(TERMINATIONS, name, value)

        self.assertEqual(below_minimum.tolist(), [False])
        self.assertEqual(at_minimum.tolist(), [True])

    def test_move_success_uses_current_not_latched_trajectory_deviation(self) -> None:
        originals = {
            "_move_boundary_distance": TERMINATIONS._move_boundary_distance,
            "_grasped_object_contact_allows_success": TERMINATIONS._grasped_object_contact_allows_success,
        }

        class FakeEnv:
            num_envs = 1
            device = "cpu"
            _so101_task_family = [BENCHMARK.TASK_MOVE]
            _so101_move_boundary_ids = torch.tensor([-1])

        env = FakeEnv()
        lateral = torch.tensor([0.051])
        try:
            TERMINATIONS._move_boundary_distance = lambda *_args: (
                torch.tensor([float("nan")]),
                torch.tensor([BENCHMARK.MOVE_NO_BOUNDARY_MIN_PROGRESS_M]),
                lateral,
                torch.tensor([[0.0, 0.0]]),
            )
            TERMINATIONS._grasped_object_contact_allows_success = lambda *_args: torch.tensor([True])
            # A current deviation beyond tolerance blocks success.
            deviated = TERMINATIONS.move_success(
                env,
                ["object_1"],
                confirm_steps=1,
                step_state=TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
            )
            # Once the object settles back on-axis, the earlier excursion no longer latches.
            lateral[:] = 0.0
            recovered = TERMINATIONS.move_success(
                env,
                ["object_1"],
                confirm_steps=1,
                step_state=TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
            )
        finally:
            for name, value in originals.items():
                setattr(TERMINATIONS, name, value)

        self.assertEqual(deviated.tolist(), [False])
        self.assertEqual(recovered.tolist(), [True])

    def test_move_success_has_only_a_strict_two_inch_boundary_maximum(self) -> None:
        originals = {
            "_move_boundary_distance": TERMINATIONS._move_boundary_distance,
            "_grasped_object_contact_allows_success": TERMINATIONS._grasped_object_contact_allows_success,
        }

        class FakeEnv:
            num_envs = 1
            device = "cpu"
            _so101_task_family = [BENCHMARK.TASK_MOVE]
            _so101_move_boundary_ids = torch.tensor([0])

        env = FakeEnv()
        distance = torch.tensor([BENCHMARK.MOVE_BOUNDARY_SUCCESS_DISTANCE_M - 0.001])
        try:
            TERMINATIONS._move_boundary_distance = lambda *_args: (
                distance,
                torch.tensor([0.05]),
                torch.tensor([0.0]),
                torch.tensor([[0.0, 0.0]]),
            )
            TERMINATIONS._grasped_object_contact_allows_success = lambda *_args: torch.tensor([True])
            inside_maximum = TERMINATIONS.move_success(
                env,
                ["object_1"],
                confirm_steps=1,
                step_state=TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
            )
            distance[:] = -0.10  # Far past the boundary still has no lower-bound violation.
            past = TERMINATIONS.move_success(
                env,
                ["object_1"],
                confirm_steps=1,
                step_state=TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
            )
            distance[:] = BENCHMARK.MOVE_BOUNDARY_SUCCESS_DISTANCE_M
            at_maximum = TERMINATIONS.move_success(
                env,
                ["object_1"],
                confirm_steps=1,
                step_state=TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
            )
        finally:
            for name, value in originals.items():
                setattr(TERMINATIONS, name, value)

        self.assertEqual(inside_maximum.tolist(), [True])
        self.assertEqual(past.tolist(), [True])
        self.assertEqual(at_maximum.tolist(), [False])

    def test_benchmark_failure_reports_move_trajectory_not_straight_enough_after_hold(self) -> None:
        originals = {
            "_termination_step_state": TERMINATIONS._termination_step_state,
            "_update_grasp_attempts": TERMINATIONS._update_grasp_attempts,
            "_ensure_failure_displacement_baseline": TERMINATIONS._ensure_failure_displacement_baseline,
            "_ensure_move_boundary_cache": TERMINATIONS._ensure_move_boundary_cache,
            "_move_boundary_distance": TERMINATIONS._move_boundary_distance,
            "grasped_object_contact_exceeded_grace_period": TERMINATIONS.grasped_object_contact_exceeded_grace_period,
        }

        class FakeEnv:
            num_envs = 1
            device = "cpu"
            step_dt = 1.0
            episode_length_buf = torch.tensor([10])
            _so101_task_family = [BENCHMARK.TASK_MOVE]
            _so101_initial_object_pos_w = torch.zeros(1, 1, 3)
            _so101_failure_object_pos_w = torch.zeros(1, 1, 3)
            _so101_failure_bin_pos_w = torch.zeros(1, 3)
            _so101_active_object_mask = torch.tensor([[True]])
            _so101_target_object_ids = torch.tensor([0])
            _so101_grasp_attempt_counts = torch.zeros(1, 1, dtype=torch.long)
            scene = {
                "bin": types.SimpleNamespace(
                    data=types.SimpleNamespace(root_pos_w=torch.zeros(1, 3))
                )
            }

        env = FakeEnv()
        step_state = TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3))
        try:
            TERMINATIONS._termination_step_state = lambda *_args: step_state
            TERMINATIONS._update_grasp_attempts = lambda *_args, **_kwargs: None
            TERMINATIONS._ensure_failure_displacement_baseline = lambda *_args, **_kwargs: torch.tensor([True])

            def ensure_boundary_cache(fake_env, *_args):
                fake_env._so101_move_boundary_ids = torch.tensor([-1])
                fake_env._so101_move_boundary_coords = torch.tensor([float("nan")])

            TERMINATIONS._ensure_move_boundary_cache = ensure_boundary_cache
            lateral = torch.tensor([0.051])
            TERMINATIONS._move_boundary_distance = lambda *_args: (
                torch.tensor([float("nan")]),
                torch.tensor([0.01]),
                lateral,
                torch.tensor([[0.01, 0.051]]),
            )
            TERMINATIONS.grasped_object_contact_exceeded_grace_period = lambda *_args: torch.tensor([False])
            first_failure = TERMINATIONS.benchmark_failure(
                env,
                ["object_1"],
                "bin",
                min_episode_time_s=0.0,
                move_straightness_tolerance=0.05,
            )
            first_active_failure_condition = env._so101_failure_conditions_active.clone()
            second_failure = TERMINATIONS.benchmark_failure(
                env,
                ["object_1"],
                "bin",
                min_episode_time_s=0.0,
                move_straightness_tolerance=0.05,
            )
            third_failure = TERMINATIONS.benchmark_failure(
                env,
                ["object_1"],
                "bin",
                min_episode_time_s=0.0,
                move_straightness_tolerance=0.05,
            )
            fourth_failure = TERMINATIONS.benchmark_failure(
                env,
                ["object_1"],
                "bin",
                min_episode_time_s=0.0,
                move_straightness_tolerance=0.05,
            )
            failure = TERMINATIONS.benchmark_failure(
                env,
                ["object_1"],
                "bin",
                min_episode_time_s=0.0,
                move_straightness_tolerance=0.05,
            )
            deviated_reason = list(env._so101_failure_reasons)
            lateral[:] = 0.0
            recovered_failure = TERMINATIONS.benchmark_failure(
                env,
                ["object_1"],
                "bin",
                min_episode_time_s=0.0,
                move_straightness_tolerance=0.05,
            )
        finally:
            for name, value in originals.items():
                setattr(TERMINATIONS, name, value)

        self.assertEqual(first_failure.tolist(), [False])
        self.assertEqual(first_active_failure_condition.tolist(), [True])
        self.assertEqual(second_failure.tolist(), [False])
        self.assertEqual(third_failure.tolist(), [False])
        self.assertEqual(fourth_failure.tolist(), [False])
        self.assertEqual(failure.tolist(), [True])
        self.assertEqual(deviated_reason, ["move_trajectory_not_straight_enough"])
        # The deviation is no longer latched: once the object is back on-axis the
        # straightness failure clears instead of persisting for the rest of the episode.
        self.assertEqual(recovered_failure.tolist(), [False])
        self.assertEqual(env._so101_failure_reasons, ["none"])

    def test_benchmark_failure_can_disable_max_grasp_attempts(self) -> None:
        originals = {
            "_termination_step_state": TERMINATIONS._termination_step_state,
            "_update_grasp_attempts": TERMINATIONS._update_grasp_attempts,
            "_ensure_failure_displacement_baseline": TERMINATIONS._ensure_failure_displacement_baseline,
        }

        class FakeEnv:
            num_envs = 1
            device = "cpu"
            step_dt = 1.0
            episode_length_buf = torch.tensor([10])
            _so101_task_family = [BENCHMARK.TASK_BIN]
            _so101_initial_object_pos_w = torch.zeros(1, 1, 3)
            _so101_failure_object_pos_w = torch.zeros(1, 1, 3)
            _so101_failure_bin_pos_w = torch.zeros(1, 3)
            _so101_active_object_mask = torch.tensor([[True]])
            _so101_target_object_ids = torch.tensor([0])
            _so101_grasp_attempt_counts = torch.tensor([[4]], dtype=torch.long)
            scene = {
                "bin": types.SimpleNamespace(
                    data=types.SimpleNamespace(root_pos_w=torch.zeros(1, 3))
                )
            }

        env = FakeEnv()
        step_state = TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3))
        try:
            TERMINATIONS._termination_step_state = lambda *_args: step_state
            TERMINATIONS._update_grasp_attempts = lambda *_args, **_kwargs: None
            TERMINATIONS._ensure_failure_displacement_baseline = lambda *_args, **_kwargs: torch.tensor([True])

            capped_failure = TERMINATIONS.benchmark_failure(
                env,
                ["object_1"],
                "bin",
                min_episode_time_s=0.0,
                max_grasp_attempts=3,
                enforce_max_grasp_attempts=True,
                failure_confirm_time_s=0.0,
            )
            disabled_failure = TERMINATIONS.benchmark_failure(
                env,
                ["object_1"],
                "bin",
                min_episode_time_s=0.0,
                max_grasp_attempts=3,
                enforce_max_grasp_attempts=False,
                failure_confirm_time_s=0.0,
            )
        finally:
            for name, value in originals.items():
                setattr(TERMINATIONS, name, value)

        self.assertEqual(capped_failure.tolist(), [True])
        self.assertEqual(disabled_failure.tolist(), [False])
        self.assertEqual(env._so101_grasp_attempt_counts.tolist(), [[4]])

    def test_grasped_object_made_contact_reads_filtered_object_contact_sensors(self) -> None:
        class Sensor:
            def __init__(self, force_matrix_w: torch.Tensor) -> None:
                self.data = types.SimpleNamespace(force_matrix_w=force_matrix_w)

        class FakeEnv:
            num_envs = 2
            device = "cpu"
            _so101_grasped_object_ids = torch.tensor([0, 1])
            scene = types.SimpleNamespace(
                sensors={
                    "object_1_contacts": Sensor(
                        torch.tensor([[[[1.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0]]]])
                    ),
                    "object_2_contacts": Sensor(
                        torch.tensor([[[[0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0]]]])
                    ),
                }
            )

        contact = TERMINATIONS.grasped_object_made_contact(
            FakeEnv(),
            ["object_1", "object_2"],
            TERMINATIONS._TerminationStepState(positions=torch.zeros(2, 2, 3)),
        )

        self.assertEqual(contact.tolist(), [True, False])

    def test_robot_contact_mask_uses_dedicated_sensors_and_force_threshold(self) -> None:
        class Sensor:
            def __init__(self, force_matrix_w: torch.Tensor) -> None:
                self.data = types.SimpleNamespace(force_matrix_w=force_matrix_w)

        class FakeEnv:
            num_envs = 2
            device = "cpu"
            scene = types.SimpleNamespace(
                sensors={
                    "object_1_robot_contact": Sensor(
                        torch.tensor([[[[0.049, 0.0, 0.0]]], [[[0.051, 0.0, 0.0]]]])
                    ),
                    # A large object-object force must not leak into the robot channel.
                    "object_1_contacts": Sensor(
                        torch.tensor([[[[5.0, 0.0, 0.0]]], [[[5.0, 0.0, 0.0]]]])
                    ),
                }
            )

        contact = TERMINATIONS._object_robot_contact_mask(
            FakeEnv(),
            ["object_1"],
            TERMINATIONS._TerminationStepState(positions=torch.zeros(2, 1, 3)),
        )

        self.assertEqual(contact.tolist(), [[False], [True]])

    def test_contact_masks_skip_pool_slots_no_env_is_using(self) -> None:
        """A pre-spawned pool must not pay a sensor read for slots that are parked.

        Reading a parked slot's filtered contact buffer costs a PhysX round trip per
        slot per step, which is what makes a 150-object task JSONL slow. The parked
        slot must also report no contact rather than whatever its neighbours in the
        parking area happen to be overlapping.
        """

        reads: list[str] = []

        class Sensor:
            def __init__(self, name: str) -> None:
                self._name = name

            @property
            def data(self):
                reads.append(self._name)
                return types.SimpleNamespace(
                    force_matrix_w=torch.tensor([[[[5.0, 0.0, 0.0]]]])
                )

        class FakeEnv:
            num_envs = 1
            device = "cpu"
            scene = types.SimpleNamespace(
                sensors={
                    "object_1_robot_contact": Sensor("object_1_robot_contact"),
                    "object_2_robot_contact": Sensor("object_2_robot_contact"),
                }
            )

        original = TERMINATIONS.slots_live_in_any_env
        TERMINATIONS.slots_live_in_any_env = lambda _env, _count: [True, False]
        try:
            contact = TERMINATIONS._object_robot_contact_mask(
                FakeEnv(),
                ["object_1", "object_2"],
                TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 2, 3)),
            )
        finally:
            TERMINATIONS.slots_live_in_any_env = original

        self.assertEqual(contact.tolist(), [[True, False]])
        self.assertEqual(reads, ["object_1_robot_contact"])

    def test_robot_clearance_scores_all_generic_bin_objects_but_only_named_target(self) -> None:
        class FakeEnv:
            num_envs = 3
            device = "cpu"
            _so101_task_family = [
                BENCHMARK.TASK_NEXT_TO,
                BENCHMARK.TASK_BIN,
                BENCHMARK.TASK_NAMED_BIN,
            ]
            _so101_target_object_ids = torch.tensor([0, 0, 0])
            _so101_active_object_mask = torch.tensor([[True, True], [True, True], [True, True]])

        state = TERMINATIONS._TerminationStepState(
            positions=torch.zeros(3, 2, 3),
            robot_contact_by_object=torch.tensor(
                [
                    [True, False],
                    [False, True],
                    [False, True],
                ]
            ),
        )
        clear = TERMINATIONS._scored_objects_clear_of_robot(
            FakeEnv(),
            ["object_1", "object_2"],
            state,
        )

        self.assertEqual(clear.tolist(), [False, False, True])

    def test_bin_success_counter_starts_only_after_robot_separation(self) -> None:
        original_math_utils = TERMINATIONS.math_utils

        class FakeEnv:
            num_envs = 1
            device = "cpu"
            _so101_task_family = [BENCHMARK.TASK_NAMED_BIN]
            _so101_active_object_mask = torch.tensor([[True]])
            _so101_target_object_ids = torch.tensor([0])
            _so101_bin_footprint_half_extents = torch.tensor([[0.1, 0.1]])
            _so101_bin_footprint_center_offsets = torch.zeros(1, 2)
            scene = {
                "bin": types.SimpleNamespace(
                    data=types.SimpleNamespace(
                        root_pos_w=torch.zeros(1, 3),
                        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                    )
                )
            }

        env = FakeEnv()
        try:
            TERMINATIONS.math_utils = types.SimpleNamespace(
                quat_inv=lambda quat: quat,
                quat_apply=lambda _quat, vector: vector,
            )
            touching = TERMINATIONS.bin_success(
                env,
                ["object_1"],
                "bin",
                confirm_steps=1,
                step_state=TERMINATIONS._TerminationStepState(
                    positions=torch.zeros(1, 1, 3),
                    robot_contact_by_object=torch.tensor([[True]]),
                ),
            )
            separated = TERMINATIONS.bin_success(
                env,
                ["object_1"],
                "bin",
                confirm_steps=1,
                step_state=TERMINATIONS._TerminationStepState(
                    positions=torch.zeros(1, 1, 3),
                    robot_contact_by_object=torch.tensor([[False]]),
                ),
            )
        finally:
            TERMINATIONS.math_utils = original_math_utils

        self.assertEqual(touching.tolist(), [False])
        self.assertEqual(separated.tolist(), [True])

    def test_grasped_object_contact_grace_period_allows_half_second_and_resets_after_separation(self) -> None:
        original_grasped_contact = TERMINATIONS.grasped_object_made_contact
        original_target_contact = TERMINATIONS.target_object_made_contact

        class FakeEnv:
            num_envs = 1
            device = "cpu"
            step_dt = 0.1
            episode_length_buf = torch.tensor([0])

        env = FakeEnv()
        contact = torch.tensor([True])
        try:
            TERMINATIONS.grasped_object_made_contact = lambda *_args: contact
            TERMINATIONS.target_object_made_contact = lambda *_args: torch.tensor([False])
            for step in range(1, 6):
                env.episode_length_buf[:] = step
                exceeded = TERMINATIONS.grasped_object_contact_exceeded_grace_period(
                    env,
                    ["object_1"],
                    TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
                    grace_time_s=0.5,
                )
                self.assertEqual(exceeded.tolist(), [False])

            # Re-reading the same control step must not consume another grace-window tick.
            exceeded = TERMINATIONS.grasped_object_contact_exceeded_grace_period(
                env,
                ["object_1"],
                TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
                grace_time_s=0.5,
            )
            self.assertEqual(exceeded.tolist(), [False])

            env.episode_length_buf[:] = 6
            exceeded = TERMINATIONS.grasped_object_contact_exceeded_grace_period(
                env,
                ["object_1"],
                TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
                grace_time_s=0.5,
            )
            self.assertEqual(exceeded.tolist(), [True])

            contact[:] = False
            env.episode_length_buf[:] = 7
            exceeded = TERMINATIONS.grasped_object_contact_exceeded_grace_period(
                env,
                ["object_1"],
                TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
                grace_time_s=0.5,
            )
            self.assertEqual(exceeded.tolist(), [False])
            self.assertEqual(env._so101_grasped_object_contact_steps.tolist(), [0])
        finally:
            TERMINATIONS.grasped_object_made_contact = original_grasped_contact
            TERMINATIONS.target_object_made_contact = original_target_contact

    def test_current_grasped_object_contact_blocks_success_during_grace_period(self) -> None:
        originals = {
            "grasped_object_contact_exceeded_grace_period": TERMINATIONS.grasped_object_contact_exceeded_grace_period,
            "grasped_object_made_contact": TERMINATIONS.grasped_object_made_contact,
            "target_object_made_contact": TERMINATIONS.target_object_made_contact,
        }
        calls = []

        def contact_exceeded(*_args):
            calls.append("timer_advanced")
            return torch.tensor([False])

        try:
            TERMINATIONS.grasped_object_contact_exceeded_grace_period = contact_exceeded
            TERMINATIONS.grasped_object_made_contact = lambda *_args: torch.tensor([True])
            TERMINATIONS.target_object_made_contact = lambda *_args: torch.tensor([False])
            allowed = TERMINATIONS._grasped_object_contact_allows_success(
                types.SimpleNamespace(num_envs=1, device="cpu"),
                ["object_1"],
                TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
                grace_time_s=0.5,
            )
        finally:
            for name, value in originals.items():
                setattr(TERMINATIONS, name, value)

        self.assertEqual(calls, ["timer_advanced"])
        self.assertEqual(allowed.tolist(), [False])

    def test_target_object_contact_blocks_success_only_until_separation(self) -> None:
        originals = {
            "grasped_object_contact_exceeded_grace_period": TERMINATIONS.grasped_object_contact_exceeded_grace_period,
            "grasped_object_made_contact": TERMINATIONS.grasped_object_made_contact,
            "target_object_made_contact": TERMINATIONS.target_object_made_contact,
        }
        env = types.SimpleNamespace(
            num_envs=1,
            device="cpu",
            _so101_task_family=[BENCHMARK.TASK_NEXT_TO],
        )
        target_contact = torch.tensor([True])
        try:
            TERMINATIONS.grasped_object_contact_exceeded_grace_period = lambda *_args: torch.tensor([False])
            TERMINATIONS.grasped_object_made_contact = lambda *_args: torch.tensor([False])
            TERMINATIONS.target_object_made_contact = lambda *_args: target_contact
            allowed_during_contact = TERMINATIONS._grasped_object_contact_allows_success(
                env,
                ["object_1"],
                TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
                grace_time_s=0.5,
            )
            target_contact[:] = False
            allowed_after_separation = TERMINATIONS._grasped_object_contact_allows_success(
                env,
                ["object_1"],
                TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
                grace_time_s=0.5,
            )
        finally:
            for name, value in originals.items():
                setattr(TERMINATIONS, name, value)

        self.assertEqual(allowed_during_contact.tolist(), [False])
        self.assertEqual(allowed_after_separation.tolist(), [True])
        self.assertEqual(env._so101_target_object_contact_ever.tolist(), [True])

    def test_sustained_target_contact_exceeds_grace_without_grasp_tracker_acquisition(self) -> None:
        originals = {
            "grasped_object_made_contact": TERMINATIONS.grasped_object_made_contact,
            "target_object_made_contact": TERMINATIONS.target_object_made_contact,
        }

        class FakeEnv:
            num_envs = 1
            device = "cpu"
            step_dt = 0.1
            episode_length_buf = torch.tensor([0])
            _so101_task_family = [BENCHMARK.TASK_BETWEEN]

        env = FakeEnv()
        target_contact = torch.tensor([True])
        try:
            TERMINATIONS.grasped_object_made_contact = lambda *_args: torch.tensor([False])
            TERMINATIONS.target_object_made_contact = lambda *_args: target_contact
            for step in range(1, 6):
                env.episode_length_buf[:] = step
                exceeded = TERMINATIONS.grasped_object_contact_exceeded_grace_period(
                    env,
                    ["object_1"],
                    TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
                    grace_time_s=0.5,
                )
                self.assertEqual(exceeded.tolist(), [False])

            env.episode_length_buf[:] = 6
            exceeded = TERMINATIONS.grasped_object_contact_exceeded_grace_period(
                env,
                ["object_1"],
                TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
                grace_time_s=0.5,
            )
            self.assertEqual(exceeded.tolist(), [True])

            target_contact[:] = False
            env.episode_length_buf[:] = 7
            exceeded = TERMINATIONS.grasped_object_contact_exceeded_grace_period(
                env,
                ["object_1"],
                TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
                grace_time_s=0.5,
            )
            self.assertEqual(exceeded.tolist(), [False])
            self.assertEqual(env._so101_grasped_object_contact_steps.tolist(), [0])
        finally:
            for name, value in originals.items():
                setattr(TERMINATIONS, name, value)

    def test_named_bin_contact_is_excluded_from_non_bin_contact_timer(self) -> None:
        originals = {
            "grasped_object_made_contact": TERMINATIONS.grasped_object_made_contact,
            "target_object_made_contact": TERMINATIONS.target_object_made_contact,
        }

        class FakeEnv:
            num_envs = 1
            device = "cpu"
            step_dt = 0.1
            episode_length_buf = torch.tensor([1])
            _so101_task_family = [BENCHMARK.TASK_NAMED_BIN]

        env = FakeEnv()
        try:
            TERMINATIONS.grasped_object_made_contact = lambda *_args: torch.tensor([True])
            TERMINATIONS.target_object_made_contact = lambda *_args: torch.tensor([True])
            exceeded = TERMINATIONS.grasped_object_contact_exceeded_grace_period(
                env,
                ["object_1"],
                TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 1, 3)),
                grace_time_s=0.0,
            )
        finally:
            for name, value in originals.items():
                setattr(TERMINATIONS, name, value)

        self.assertEqual(exceeded.tolist(), [False])
        self.assertEqual(env._so101_grasped_object_contact_steps.tolist(), [0])

    def test_grasp_tracker_records_nearest_active_object_and_clears_on_open(self) -> None:
        class Robot:
            def __init__(self) -> None:
                self.data = types.SimpleNamespace(
                    joint_pos=torch.tensor([[1.0]]),
                    joint_pos_limits=torch.tensor([[[0.0, 1.0]]]),
                )

            def find_joints(self, _name: str):
                return [0], ["Jaw"]

        robot = Robot()
        env = types.SimpleNamespace(
            num_envs=1,
            device="cpu",
            scene={
                "robot": robot,
                "ee_frame": types.SimpleNamespace(
                    data=types.SimpleNamespace(target_pos_w=torch.tensor([[[0.0, 0.0, 0.0]]]))
                ),
            },
            _so101_task_family=[BENCHMARK.TASK_NEXT_TO],
            _so101_target_object_ids=torch.tensor([0]),
            _so101_active_object_mask=torch.tensor([[True, True]]),
        )
        object_pos_w = torch.tensor([[[0.5, 0.0, 0.0], [0.01, 0.0, 0.0]]])
        kwargs = {
            "object_asset_names": ["object_1", "object_2"],
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "jaw_joint_name": "Jaw",
            "jaw_close_delta": 0.5,
            "jaw_open_fraction": 0.5,
            "object_distance_threshold": 0.1,
            "object_pos_w": object_pos_w,
        }

        TERMINATIONS._update_grasp_attempts(env, **kwargs)
        robot.data.joint_pos[:] = 0.0
        TERMINATIONS._update_grasp_attempts(env, **kwargs)
        self.assertEqual(env._so101_grasped_object_ids.tolist(), [1])

        robot.data.joint_pos[:] = 1.0
        TERMINATIONS._update_grasp_attempts(env, **kwargs)
        self.assertEqual(env._so101_grasped_object_ids.tolist(), [-1])


class PostmortemFailureTypeTests(unittest.TestCase):
    @staticmethod
    def _env(task_family: str, lift: list[float]):
        num_objects = len(lift)
        return types.SimpleNamespace(
            num_envs=1,
            device="cpu",
            _so101_task_family=[task_family],
            _so101_max_object_lift=torch.tensor([lift], dtype=torch.float32),
            _so101_active_object_mask=torch.ones((1, num_objects), dtype=torch.bool),
            _so101_target_object_ids=torch.tensor([0], dtype=torch.long),
            so101_bench_episodes=[
                {
                    "env_id": 0,
                    "active_object_ids": list(range(num_objects)),
                    "active_labels": [f"obj{i}" for i in range(num_objects)],
                }
            ],
        )

    def _classify(self, task_family: str, lift: list[float]):
        names = [f"object_{i + 1}" for i in range(len(lift))]
        return TERMINATIONS.benchmark_postmortem_failure_diagnostics(self._env(task_family, lift), names)[0]

    def _tracked_env(self, task_family: str, lift: list[float]):
        env = self._env(task_family, lift)
        env.step_dt = 0.1
        env._so101_failure_reasons = [TERMINATIONS.FAILURE_REASON_NONE]
        TERMINATIONS._ensure_postmortem_buffers(env, len(lift))
        env._so101_pm_tracker_available = True
        return env

    def test_target_lifted_is_placement(self) -> None:
        threshold = TERMINATIONS.LIFT_OFF_GROUND_LIMIT_M
        diagnostic = self._classify(BENCHMARK.TASK_NEXT_TO, [threshold + 0.01, 0.0, 0.0, 0.0])
        self.assertEqual(diagnostic.failure_type, TERMINATIONS.POSTMORTEM_PLACEMENT)

    def test_only_distractor_lifted_is_semantic(self) -> None:
        threshold = TERMINATIONS.LIFT_OFF_GROUND_LIMIT_M
        diagnostic = self._classify(BENCHMARK.TASK_BETWEEN, [0.0, threshold + 0.02, threshold + 0.05, 0.0])
        self.assertEqual(diagnostic.failure_type, TERMINATIONS.POSTMORTEM_SEMANTIC)
        # The reported wrong object is the highest-lifted distractor.
        self.assertIn("object_3", diagnostic.lifted_wrong_object)

    def test_nothing_lifted_is_failed_grasp(self) -> None:
        threshold = TERMINATIONS.LIFT_OFF_GROUND_LIMIT_M
        diagnostic = self._classify(BENCHMARK.TASK_MOVE, [threshold - 0.001, threshold - 0.001, 0.0, 0.0])
        self.assertEqual(diagnostic.failure_type, TERMINATIONS.POSTMORTEM_FAILED_GRASP)
        self.assertEqual(diagnostic.lifted_wrong_object, "none")

    def test_target_lifted_takes_priority_over_distractor(self) -> None:
        threshold = TERMINATIONS.LIFT_OFF_GROUND_LIMIT_M
        diagnostic = self._classify(BENCHMARK.TASK_NEXT_TO, [threshold + 0.01, threshold + 0.05, 0.0, 0.0])
        self.assertEqual(diagnostic.failure_type, TERMINATIONS.POSTMORTEM_PLACEMENT)

    def test_bin_task_is_not_applicable(self) -> None:
        threshold = TERMINATIONS.LIFT_OFF_GROUND_LIMIT_M
        diagnostic = self._classify(BENCHMARK.TASK_BIN, [threshold + 0.5, 0.0, 0.0, 0.0])
        self.assertEqual(diagnostic.failure_type, TERMINATIONS.POSTMORTEM_NOT_APPLICABLE)

    def test_update_max_object_lift_tracks_running_max_after_baseline(self) -> None:
        env = types.SimpleNamespace(
            num_envs=1,
            device="cpu",
            _so101_active_object_mask=torch.tensor([[True, True]]),
            _so101_failure_object_pos_w=torch.tensor([[[0.0, 0.0, 0.10], [0.0, 0.0, 0.10]]]),
        )
        names = ["object_1", "object_2"]
        baseline = torch.tensor([True])
        # Object 0 rises to 0.15 (lift 0.05), object 1 stays put.
        TERMINATIONS._update_max_object_lift(env, names, torch.tensor([[[0.0, 0.0, 0.15], [0.0, 0.0, 0.10]]]), baseline)
        # Object 0 drops back; running max must be retained.
        TERMINATIONS._update_max_object_lift(env, names, torch.tensor([[[0.0, 0.0, 0.10], [0.0, 0.0, 0.10]]]), baseline)
        self.assertAlmostEqual(float(env._so101_max_object_lift[0, 0].item()), 0.05, places=5)
        # Before the baseline is recorded, lift is not accumulated.
        TERMINATIONS._update_max_object_lift(
            env, names, torch.tensor([[[0.0, 0.0, 0.30], [0.0, 0.0, 0.30]]]), torch.tensor([False])
        )
        self.assertAlmostEqual(float(env._so101_max_object_lift[0, 1].item()), 0.0, places=5)

    def test_accidental_distractor_lift_is_not_wrong_object_targeted(self) -> None:
        threshold = TERMINATIONS.LIFT_OFF_GROUND_LIMIT_M
        env = self._tracked_env(BENCHMARK.TASK_NEXT_TO, [0.0, threshold + 0.05])

        diagnostic = TERMINATIONS.benchmark_postmortem_failure_diagnostics(
            env, ["object_1", "object_2"]
        )[0]

        self.assertEqual(diagnostic.failure_type, TERMINATIONS.POSTMORTEM_NO_MANIPULATION_ATTEMPT)
        self.assertEqual(diagnostic.legacy_failure_type, TERMINATIONS.POSTMORTEM_SEMANTIC)
        self.assertIn("wrong_object_lifted_or_disturbed", diagnostic.secondary_failure_types)

    def test_wrong_object_requires_temporally_associated_manipulation(self) -> None:
        env = self._tracked_env(BENCHMARK.TASK_BETWEEN, [0.0, 0.0, 0.0])
        env._so101_pm_attempt_counts_all[0, 2] = 1
        env._so101_pm_acquired_objects[0, 2] = True
        env._so101_pm_manipulated_objects[0, 2] = True
        env._so101_pm_associated_transport_m[0, 2] = 0.08

        diagnostic = TERMINATIONS.benchmark_postmortem_failure_diagnostics(
            env, ["object_1", "object_2", "object_3"]
        )[0]

        self.assertEqual(diagnostic.failure_type, TERMINATIONS.POSTMORTEM_WRONG_OBJECT_TARGETED)
        self.assertTrue(diagnostic.wrong_object_acquired)
        self.assertGreaterEqual(diagnostic.confidence, 0.8)

    def test_push_only_target_manipulation_is_not_failed_grasp(self) -> None:
        env = self._tracked_env(BENCHMARK.TASK_MOVE, [0.0, 0.0])
        env._so101_pm_manipulated_objects[0, 0] = True
        env._so101_pm_associated_transport_m[0, 0] = 0.06
        env._so101_pm_max_object_displacement_m[0, 0] = 0.06

        diagnostic = TERMINATIONS.benchmark_postmortem_failure_diagnostics(
            env, ["object_1", "object_2"]
        )[0]

        self.assertEqual(diagnostic.failure_type, TERMINATIONS.POSTMORTEM_GOAL_NOT_REACHED)
        self.assertTrue(diagnostic.target_manipulated)
        self.assertFalse(diagnostic.target_acquired)

    def test_target_close_without_comotion_is_acquisition_failure(self) -> None:
        env = self._tracked_env(BENCHMARK.TASK_NEXT_TO, [0.0, 0.0])
        env._so101_pm_attempt_counts_all[0, 0] = 2

        diagnostic = TERMINATIONS.benchmark_postmortem_failure_diagnostics(
            env, ["object_1", "object_2"]
        )[0]

        self.assertEqual(diagnostic.failure_type, TERMINATIONS.POSTMORTEM_TARGET_ACQUISITION_FAILED)
        self.assertEqual(diagnostic.target_attempt_count, 2)

    def test_acquired_target_released_outside_goal_is_explicit(self) -> None:
        env = self._tracked_env(BENCHMARK.TASK_NEXT_TO, [0.0, 0.0])
        env._so101_pm_attempt_counts_all[0, 0] = 1
        env._so101_pm_acquired_objects[0, 0] = True
        env._so101_pm_manipulated_objects[0, 0] = True
        env._so101_pm_release_counts[0, 0] = 1

        diagnostic = TERMINATIONS.benchmark_postmortem_failure_diagnostics(
            env, ["object_1", "object_2"]
        )[0]

        self.assertEqual(
            diagnostic.failure_type,
            TERMINATIONS.POSTMORTEM_TARGET_RELEASED_OUTSIDE_GOAL,
        )
        self.assertEqual(diagnostic.target_release_count, 1)

    def test_transient_sustained_goal_is_unstable(self) -> None:
        env = self._tracked_env(BENCHMARK.TASK_NEXT_TO, [0.0, 0.0])
        env._so101_pm_goal_ever_reached[0] = True
        env._so101_pm_max_goal_hold_steps[0] = 4
        env._so101_pm_final_goal_met[0] = False

        diagnostic = TERMINATIONS.benchmark_postmortem_failure_diagnostics(
            env, ["object_1", "object_2"]
        )[0]

        self.assertEqual(diagnostic.failure_type, TERMINATIONS.POSTMORTEM_GOAL_REACHED_BUT_UNSTABLE)
        self.assertTrue(diagnostic.goal_ever_reached)

    def test_confirmed_goal_that_remains_met_has_no_failure(self) -> None:
        env = self._tracked_env(BENCHMARK.TASK_NEXT_TO, [0.0, 0.0])
        env._so101_pm_goal_ever_reached[0] = True
        env._so101_pm_final_goal_met[0] = True
        env._so101_pm_max_success_counter[0] = 3
        env._so101_success_confirmation_required_steps = torch.tensor([3])
        env._so101_success_confirmed_current = torch.tensor([True])
        env._so101_success_ever_confirmed = torch.tensor([True])

        diagnostic = TERMINATIONS.benchmark_postmortem_failure_diagnostics(
            env, ["object_1", "object_2"]
        )[0]

        self.assertEqual(diagnostic.failure_type, TERMINATIONS.POSTMORTEM_NONE)
        self.assertTrue(diagnostic.goal_ever_confirmed)

    def test_raw_counter_does_not_override_persisted_unconfirmed_state(self) -> None:
        env = self._tracked_env(BENCHMARK.TASK_NEXT_TO, [0.0, 0.0])
        env._so101_pm_goal_ever_reached[0] = True
        env._so101_pm_final_goal_met[0] = True
        env._so101_pm_max_success_counter[0] = 3
        env._so101_next_to_success_counter = torch.tensor([3])
        env._so101_success_confirmation_required_steps = torch.tensor([3])
        env._so101_success_confirmed_current = torch.tensor([False])
        env._so101_success_ever_confirmed = torch.tensor([False])

        diagnostic = TERMINATIONS.benchmark_postmortem_failure_diagnostics(
            env, ["object_1", "object_2"]
        )[0]

        self.assertEqual(
            diagnostic.failure_type,
            TERMINATIONS.POSTMORTEM_GOAL_REACHED_BUT_UNCONFIRMED,
        )
        self.assertFalse(diagnostic.goal_ever_confirmed)

    def test_final_goal_with_live_rule_violation_is_not_called_unstable(self) -> None:
        env = self._tracked_env(BENCHMARK.TASK_NEXT_TO, [0.0, 0.0])
        env._so101_pm_goal_ever_reached[0] = True
        env._so101_pm_final_goal_met[0] = True
        env._so101_success_confirmation_required_steps = torch.tensor([3])
        env._so101_success_confirmed_current = torch.tensor([False])
        env._so101_success_ever_confirmed = torch.tensor([False])
        env._so101_failure_reasons = [TERMINATIONS.FAILURE_REASON_NON_TARGET_MOVED]

        diagnostic = TERMINATIONS.benchmark_postmortem_failure_diagnostics(
            env, ["object_1", "object_2"]
        )[0]

        self.assertEqual(
            diagnostic.failure_type,
            TERMINATIONS.POSTMORTEM_GOAL_REACHED_WITH_RULE_VIOLATION,
        )

    def test_final_goal_with_raw_pending_rule_violation_is_rule_violation(self) -> None:
        env = self._tracked_env(BENCHMARK.TASK_NEXT_TO, [0.0, 0.0])
        env._so101_pm_goal_ever_reached[0] = True
        env._so101_pm_final_goal_met[0] = True
        env._so101_success_confirmation_required_steps = torch.tensor([3])
        env._so101_success_confirmed_current = torch.tensor([False])
        env._so101_success_ever_confirmed = torch.tensor([False])
        env._so101_failure_conditions_active = torch.tensor([True])
        env._so101_failure_conditions_active_reasons = [
            (TERMINATIONS.FAILURE_REASON_NON_TARGET_MOVED,)
        ]

        diagnostic = TERMINATIONS.benchmark_postmortem_failure_diagnostics(
            env, ["object_1", "object_2"]
        )[0]

        self.assertEqual(
            diagnostic.failure_type,
            TERMINATIONS.POSTMORTEM_GOAL_REACHED_WITH_RULE_VIOLATION,
        )
        self.assertTrue(diagnostic.evidence["raw_live_failure_active"])
        self.assertIn(
            "raw_live_rule:non_target_moved",
            diagnostic.secondary_failure_types,
        )

    def test_bin_objects_are_all_intended_targets(self) -> None:
        env = self._tracked_env(BENCHMARK.TASK_BIN, [0.0, 0.0])
        env._so101_pm_attempt_counts_all[0, 1] = 1
        env._so101_pm_acquired_objects[0, 1] = True
        env._so101_pm_manipulated_objects[0, 1] = True
        env._so101_pm_ever_inside_bin[0, 1] = True

        diagnostic = TERMINATIONS.benchmark_postmortem_failure_diagnostics(
            env, ["object_1", "object_2"]
        )[0]

        self.assertEqual(
            diagnostic.failure_type,
            TERMINATIONS.POSTMORTEM_INCOMPLETE_MULTI_OBJECT_TASK,
        )
        self.assertFalse(diagnostic.wrong_object_acquired)
        self.assertFalse(diagnostic.evidence["target_vs_wrong_semantics_applicable"])

    def test_bin_attempt_on_non_metadata_target_counts_as_acquisition_attempt(self) -> None:
        env = self._tracked_env(BENCHMARK.TASK_BIN, [0.0, 0.0])
        env._so101_pm_attempt_counts_all[0, 1] = 1

        diagnostic = TERMINATIONS.benchmark_postmortem_failure_diagnostics(
            env, ["object_1", "object_2"]
        )[0]

        self.assertEqual(
            diagnostic.failure_type,
            TERMINATIONS.POSTMORTEM_OBJECT_ACQUISITION_FAILED,
        )
        self.assertEqual(diagnostic.evidence["active_object_attempt_count"], 1)

    def test_recovered_transient_overshoot_is_secondary_not_primary(self) -> None:
        env = self._tracked_env(BENCHMARK.TASK_MOVE, [0.0, 0.0])
        env._so101_pm_ever_overshot[0] = True
        env._so101_pm_final_overshot[0] = False
        env._so101_pm_manipulated_objects[0, 0] = True
        env._so101_pm_associated_transport_m[0, 0] = 0.05

        diagnostic = TERMINATIONS.benchmark_postmortem_failure_diagnostics(
            env, ["object_1", "object_2"]
        )[0]

        self.assertEqual(diagnostic.failure_type, TERMINATIONS.POSTMORTEM_GOAL_NOT_REACHED)
        self.assertIn("transient_overshoot_recovered", diagnostic.secondary_failure_types)

    def test_task_condition_metrics_returns_typed_counter_and_goal_fields(self) -> None:
        env = self._tracked_env(BENCHMARK.TASK_NEXT_TO, [0.0, 0.0])
        env._so101_postmortem_goal_met_override = torch.tensor([True])
        env._so101_next_to_success_counter = torch.tensor([2])
        env._so101_success_confirmation_required_steps = torch.tensor([3])
        state = TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 2, 3))

        record = TERMINATIONS.task_condition_metrics(
            env,
            ["object_1", "object_2"],
            "bin",
            step_state=state,
        )[0]

        self.assertEqual(record["task_family"], BENCHMARK.TASK_NEXT_TO)
        self.assertEqual(record["goal"]["counter"], 2)
        self.assertEqual(record["goal"]["required_steps"], 3)
        self.assertIs(record["goal"]["instant"], True)
        self.assertIs(record["goal"]["geometry_confirmed"], False)

    def test_postmortem_between_geometry_uses_disabled_center_band(self) -> None:
        env = self._tracked_env(BENCHMARK.TASK_BETWEEN, [0.0, 0.0, 0.0])
        env._so101_referent_object_ids = torch.tensor([[1, 2]], dtype=torch.long)
        state = TERMINATIONS._TerminationStepState(
            positions=torch.tensor([[[0.15, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        )
        original_yaws = TERMINATIONS._state_object_yaws
        original_distances = TERMINATIONS._pairwise_object_surface_distance
        # Fraction 0.99 would have failed the former [0.1, 0.9] gate.
        distances = iter((torch.tensor([0.99]), torch.tensor([0.01])))
        try:
            TERMINATIONS._state_object_yaws = lambda *_args, **_kwargs: torch.zeros(1, 3)
            TERMINATIONS._pairwise_object_surface_distance = (
                lambda *_args, **_kwargs: next(distances)
            )
            goal = TERMINATIONS._postmortem_goal_step(
                env,
                ["object_1", "object_2", "object_3"],
                "bin",
                {"x": (0.0, 1.0), "y": (-1.0, 1.0)},
                state,
            )
        finally:
            TERMINATIONS._state_object_yaws = original_yaws
            TERMINATIONS._pairwise_object_surface_distance = original_distances

        self.assertTrue(bool(goal.met[0].item()))
        self.assertEqual(goal.metrics[0]["center_fraction_min"], 0.0)
        self.assertEqual(goal.metrics[0]["center_fraction_max"], 1.0)

    def test_postmortem_move_geometry_has_no_boundary_gap_minimum(self) -> None:
        env = self._tracked_env(BENCHMARK.TASK_MOVE, [0.0, 0.0])
        env._so101_move_boundary_ids = torch.tensor([1], dtype=torch.long)
        state = TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 2, 3))
        original_yaws = TERMINATIONS._state_object_yaws
        original_distance = TERMINATIONS._move_boundary_distance
        try:
            TERMINATIONS._state_object_yaws = lambda *_args, **_kwargs: torch.zeros(1, 2)
            TERMINATIONS._move_boundary_distance = lambda *_args, **_kwargs: (
                torch.tensor([-0.023]),
                torch.tensor([0.10]),
                torch.tensor([0.03]),
                torch.zeros(1, 2),
            )
            goal = TERMINATIONS._postmortem_goal_step(
                env,
                ["object_1", "object_2"],
                "bin",
                {"x": (0.0, 1.0), "y": (-1.0, 1.0)},
                state,
                move_straightness_tolerance=0.02,
                move_past_boundary_tolerance=0.02,
            )
        finally:
            TERMINATIONS._state_object_yaws = original_yaws
            TERMINATIONS._move_boundary_distance = original_distance

        # The gap is far past the former lower bound, but still below the maximum.
        # Straightness remains independently enforced here and keeps the goal unmet.
        self.assertFalse(bool(goal.met[0].item()))
        self.assertFalse(bool(goal.overshot[0].item()))

    def test_temporal_tracker_is_idempotent_and_infers_acquisition_from_comotion(self) -> None:
        env = self._tracked_env(BENCHMARK.TASK_MOVE, [0.0, 0.0])
        env.episode_length_buf = torch.tensor([1], dtype=torch.long)
        env._so101_failure_object_pos_w = torch.zeros(1, 2, 3)
        env._so101_move_success_counter = torch.zeros(1, dtype=torch.long)
        env._so101_postmortem_goal_met_override = torch.tensor([False])
        names = ["object_1", "object_2"]
        baseline = torch.tensor([True])

        def grasp_step(x: float, *, close: bool) -> object:
            return TERMINATIONS._GraspStepEvidence(
                ee_pos_w=torch.tensor([[x, 0.0, 0.0]]),
                jaw_is_open=torch.tensor([False]),
                close_cycle=torch.tensor([close]),
                grasp_started=torch.tensor([close]),
                nearest_active_object_ids=torch.tensor([0]),
                nearest_active_distance_m=torch.tensor([0.0]),
            )

        def update(x: float, *, close: bool = False) -> None:
            state = TERMINATIONS._TerminationStepState(
                positions=torch.tensor([[[x, 0.0, 0.0], [0.2, 0.0, 0.0]]])
            )
            TERMINATIONS._update_postmortem_evidence(
                env,
                names,
                "bin",
                {"x": (0.0, 1.0), "y": (-1.0, 1.0)},
                state,
                grasp_step(x, close=close),
                baseline,
                object_distance_threshold=0.1,
            )

        update(0.0, close=True)
        env.episode_length_buf[0] = 2
        update(0.003)
        env.episode_length_buf[0] = 3
        update(0.006)
        # A collector/rescorer refresh at the same episode step must be a no-op.
        update(0.009)

        self.assertEqual(env._so101_pm_attempt_counts_all.tolist(), [[1, 0]])
        self.assertEqual(env._so101_pm_acquisition_counts.tolist(), [[1, 0]])
        self.assertTrue(bool(env._so101_pm_acquired_objects[0, 0].item()))
        self.assertAlmostEqual(float(env._so101_pm_candidate_transport_m[0].item()), 0.006, places=5)
        self.assertEqual(len(env._so101_pm_attempt_history[0]), 1)

        # The tracker owns its reset lifecycle because resets.py intentionally
        # knows nothing about these optional diagnostic buffers.
        env.episode_length_buf[0] = 1
        update(0.0)
        self.assertEqual(env._so101_pm_attempt_counts_all.tolist(), [[0, 0]])
        self.assertEqual(env._so101_pm_acquisition_counts.tolist(), [[0, 0]])
        self.assertEqual(env._so101_pm_attempt_history[0], [])

    def test_acquisition_requires_transport_in_one_contiguous_comotion_segment(self) -> None:
        env = self._tracked_env(BENCHMARK.TASK_MOVE, [0.0, 0.0])
        env._so101_failure_object_pos_w = torch.zeros(1, 2, 3)
        env._so101_move_success_counter = torch.zeros(1, dtype=torch.long)
        env._so101_postmortem_goal_met_override = torch.tensor([False])
        names = ["object_1", "object_2"]
        baseline = torch.tensor([True])

        def update(step: int, x: float, *, close: bool = False) -> None:
            env.episode_length_buf = torch.tensor([step], dtype=torch.long)
            state = TERMINATIONS._TerminationStepState(
                positions=torch.tensor([[[x, 0.0, 0.0], [0.2, 0.0, 0.0]]])
            )
            grasp = TERMINATIONS._GraspStepEvidence(
                ee_pos_w=torch.tensor([[x, 0.0, 0.0]]),
                jaw_is_open=torch.tensor([False]),
                close_cycle=torch.tensor([close]),
                grasp_started=torch.tensor([close]),
                nearest_active_object_ids=torch.tensor([0]),
                nearest_active_distance_m=torch.tensor([0.0]),
            )
            TERMINATIONS._update_postmortem_evidence(
                env,
                names,
                "bin",
                {"x": (0.0, 1.0), "y": (-1.0, 1.0)},
                state,
                grasp,
                baseline,
                object_distance_threshold=0.1,
            )

        update(1, 0.0, close=True)
        update(2, 0.001)
        update(3, 0.002)  # Two-frame segment, but only 2 mm.
        update(4, 0.002)  # Break the segment.
        update(5, 0.005)  # Cumulative attempt transport reaches 5 mm.

        self.assertFalse(bool(env._so101_pm_acquired_objects[0, 0].item()))
        self.assertAlmostEqual(float(env._so101_pm_candidate_transport_m[0].item()), 0.005, places=5)
        self.assertAlmostEqual(
            float(env._so101_pm_candidate_segment_transport_m[0].item()), 0.003, places=5
        )

    def test_pre_control_goal_geometry_is_not_attributed_to_the_policy(self) -> None:
        env = self._tracked_env(BENCHMARK.TASK_MOVE, [0.0, 0.0])
        env._so101_failure_object_pos_w = torch.zeros(1, 2, 3)
        env._so101_move_success_counter = torch.zeros(1, dtype=torch.long)
        env._so101_postmortem_goal_met_override = torch.tensor([True])
        env._so101_robot_started_moving = torch.tensor([False])
        names = ["object_1", "object_2"]
        state = TERMINATIONS._TerminationStepState(positions=torch.zeros(1, 2, 3))
        grasp = TERMINATIONS._GraspStepEvidence(
            ee_pos_w=torch.zeros(1, 3),
            jaw_is_open=torch.tensor([True]),
            close_cycle=torch.tensor([False]),
            grasp_started=torch.tensor([False]),
            nearest_active_object_ids=torch.tensor([0]),
            nearest_active_distance_m=torch.tensor([0.0]),
        )

        env.episode_length_buf = torch.tensor([1], dtype=torch.long)
        TERMINATIONS._update_postmortem_evidence(
            env,
            names,
            "bin",
            {"x": (0.0, 1.0), "y": (-1.0, 1.0)},
            state,
            grasp,
            torch.tensor([True]),
            object_distance_threshold=0.1,
        )
        self.assertFalse(bool(env._so101_pm_goal_ever_reached[0].item()))

        env._so101_robot_started_moving[0] = True
        env.episode_length_buf[0] = 2
        TERMINATIONS._update_postmortem_evidence(
            env,
            names,
            "bin",
            {"x": (0.0, 1.0), "y": (-1.0, 1.0)},
            state,
            grasp,
            torch.tensor([True]),
            object_distance_threshold=0.1,
        )
        self.assertTrue(bool(env._so101_pm_goal_ever_reached[0].item()))


if __name__ == "__main__":
    unittest.main()
