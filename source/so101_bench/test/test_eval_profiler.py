from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


MODULE_PATH = Path(__file__).parents[1] / "so101_bench" / "utils" / "eval_profiler.py"
SPEC = importlib.util.spec_from_file_location("so101_eval_profiler_test_module", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
PROFILER_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROFILER_MODULE
SPEC.loader.exec_module(PROFILER_MODULE)
EvalProfiler = PROFILER_MODULE.EvalProfiler


class _CallableOwner:
    def work(self, value=1):
        return value + 1


class _ObservationManager:
    def compute_group(self, group_name, update_history=False):
        return {group_name: update_history}

    def compute(self, update_history=False):
        return self.compute_group("visual", update_history=update_history)


class _TerminationManager(_CallableOwner):
    def __init__(self):
        self._term_names = ["failure"]
        self._term_cfgs = [SimpleNamespace(func=lambda _env: False)]

    def compute(self):
        return self._term_cfgs[0].func(None)


class _Scene(SimpleNamespace):
    def __getitem__(self, name):
        return getattr(self, name)


class _DummyEnv:
    def __init__(self):
        self.unwrapped = self
        self.action_manager = SimpleNamespace(process_action=lambda _action: None, apply_action=lambda: None)
        self.sim = SimpleNamespace(step=lambda **_kwargs: None, render=lambda: None, forward=lambda: None)
        robot = _CallableOwner()
        robot.update = robot.work
        self.scene = _Scene(
            write_data_to_sim=lambda: None,
            update=lambda _dt: None,
            _sensors={"camera_wrist": _CallableOwner()},
            robot=robot,
        )
        self.scene._sensors["camera_wrist"]._update_buffers_impl = self.scene._sensors["camera_wrist"].work
        self.termination_manager = _TerminationManager()
        self.reward_manager = SimpleNamespace(compute=lambda **_kwargs: None)
        self.observation_manager = _ObservationManager()
        self.command_manager = SimpleNamespace(compute=lambda **_kwargs: None)
        self.event_manager = SimpleNamespace(apply=lambda **_kwargs: None)

    def _reset_idx(self, _env_ids):
        return None


def test_profile_snapshot_episode_and_instrumentation(tmp_path):
    summary_path = tmp_path / "profile.json"
    episodes_path = tmp_path / "episodes.jsonl"
    profiler = EvalProfiler(summary_path, episodes_jsonl=episodes_path, metadata={"test": True})
    env = _DummyEnv()
    profiler.install_simulation_instrumentation(env)

    profiler.start_episode(1)
    with profiler.measure("simulation.env_step.total"):
        env.action_manager.process_action(None)
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.sim.render()
        env.scene.update(1.0 / 240.0)
        env.scene["robot"].update(1.0 / 240.0)
        env.termination_manager.compute()
        env.observation_manager.compute(update_history=True)
        env.event_manager.apply(mode="interval")
        env.scene._sensors["camera_wrist"]._update_buffers_impl()
        env._reset_idx([0])
        env.sim.render()
        env.event_manager.apply(mode="interval")
        env.observation_manager.compute(update_history=True)
    with profiler.measure("episode.explicit_reset"):
        env._reset_idx([0])
    profiler.increment("simulation.control_steps")
    profiler.observe("recording.encoder.queue_depth.front", 3)
    profiler.finish_episode(
        episode=1,
        success=False,
        termination_reason="time_out",
        control_steps=1,
        simulated_seconds=1.0 / 30.0,
    )
    profiler.finish()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["completed_episodes"] == 1
    assert summary["counters"]["simulation.control_steps"] == 1
    assert summary["gauges"]["recording.encoder.queue_depth.front"]["max"] == 3
    assert summary["timings"]["simulation.physics.step"]["count"] == 1
    assert summary["timings"]["simulation.render.control_step"]["count"] == 1
    assert summary["timings"]["simulation.render.automatic_reset"]["count"] == 1
    assert summary["timings"]["simulation.scene.write.control_step"]["count"] == 1
    assert summary["timings"]["simulation.events.interval.control_step"]["count"] == 1
    assert summary["timings"]["simulation.events.interval.automatic_reset"]["count"] == 1
    assert summary["timings"]["simulation.observation.group.visual.control_step"]["count"] == 1
    assert summary["timings"]["simulation.observation.group.visual.automatic_reset"]["count"] == 1
    assert summary["timings"]["simulation.termination.term.failure"]["count"] == 1
    assert summary["timings"]["simulation.sensor.camera_wrist.update_buffers.control_step"]["count"] == 1
    assert summary["timings"]["simulation.scene.robot.update"]["count"] == 1
    assert summary["timings"]["simulation.reset_idx.automatic_terminal"]["count"] == 1
    assert summary["timings"]["simulation.reset_idx.explicit"]["count"] == 1

    episode = json.loads(episodes_path.read_text(encoding="utf-8"))
    assert episode["episode"] == 1
    assert episode["control_steps"] == 1
    assert episode["counters"]["simulation.control_steps"] == 1
    assert episode["timings"]["simulation.env_step.total"]["count"] == 1


def test_dynamic_method_preserves_return_value(tmp_path):
    profiler = EvalProfiler(tmp_path / "profile.json")
    owner = _CallableOwner()
    assert profiler.wrap_dynamic_method(owner, "work", lambda args, _kwargs: f"work.{args[0]}")
    assert owner.work(4) == 5
    profiler.finish()

    summary = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    assert summary["timings"]["work.4"]["count"] == 1
