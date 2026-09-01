"""Low-overhead wall-clock profiling for SO-101 evaluation.

The profiler instruments already-existing call boundaries. It does not invoke
layout generation, task diagnostics, rendering, simulation, or inference on
its own. Timings are inclusive and may be nested; callers must not sum them as
if they were mutually exclusive.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Iterator


@dataclass
class _TimingAggregate:
    count: int = 0
    total_s: float = 0.0
    min_s: float = math.inf
    max_s: float = 0.0
    sum_squares: float = 0.0
    samples_s: list[float] = field(default_factory=list)
    _sample_rng: random.Random = field(default_factory=lambda: random.Random(0), repr=False)

    _MAX_SAMPLES = 4096

    def add(self, elapsed_s: float) -> None:
        self.count += 1
        self.total_s += elapsed_s
        self.min_s = min(self.min_s, elapsed_s)
        self.max_s = max(self.max_s, elapsed_s)
        self.sum_squares += elapsed_s * elapsed_s
        if len(self.samples_s) < self._MAX_SAMPLES:
            self.samples_s.append(elapsed_s)
        else:
            # Deterministic reservoir sampling keeps tail-latency estimates bounded
            # without consuming the evaluator's global random state.
            sample_index = self._sample_rng.randrange(self.count)
            if sample_index < self._MAX_SAMPLES:
                self.samples_s[sample_index] = elapsed_s

    def values(self) -> tuple[int, float, float, float, float]:
        return self.count, self.total_s, self.min_s, self.max_s, self.sum_squares

    def as_dict(self) -> dict[str, float | int]:
        mean = self.total_s / self.count if self.count else 0.0
        variance = max(0.0, self.sum_squares / self.count - mean * mean) if self.count else 0.0
        ordered_samples = sorted(self.samples_s)

        def percentile(fraction: float) -> float:
            if not ordered_samples:
                return 0.0
            index = round(fraction * (len(ordered_samples) - 1))
            return ordered_samples[index]

        return {
            "count": self.count,
            "total_s": self.total_s,
            "mean_s": mean,
            "std_s": math.sqrt(variance),
            "min_s": self.min_s if self.count else 0.0,
            "max_s": self.max_s if self.count else 0.0,
            "p50_s": percentile(0.50),
            "p95_s": percentile(0.95),
            "p99_s": percentile(0.99),
            "percentile_sample_count": len(ordered_samples),
            "rate_hz": self.count / self.total_s if self.total_s > 0.0 else 0.0,
        }


@dataclass
class _NumericAggregate:
    count: int = 0
    total: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": self.total / self.count if self.count else 0.0,
            "min": self.min_value if self.count else 0.0,
            "max": self.max_value if self.count else 0.0,
            "sum": self.total,
        }


class EvalProfiler:
    """Collect inclusive timings and persist crash-tolerant profile snapshots."""

    def __init__(
        self,
        output_json: Path,
        *,
        episodes_jsonl: Path | None = None,
        cuda_synchronize: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.output_json = output_json.expanduser().resolve()
        self.episodes_jsonl = (
            episodes_jsonl.expanduser().resolve()
            if episodes_jsonl is not None
            else self.output_json.with_name(f"{self.output_json.stem}_episodes.jsonl")
        )
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        self.episodes_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if self.output_json == self.episodes_jsonl:
            raise ValueError("Profiler summary JSON and per-episode JSONL must use different paths.")
        if self.episodes_jsonl.exists() and self.episodes_jsonl.stat().st_size > 0:
            raise FileExistsError(f"Refusing to append to non-empty profiler log: {self.episodes_jsonl}")
        self.episodes_jsonl.touch()
        self.cuda_synchronize = cuda_synchronize
        self.metadata = dict(metadata or {})
        self.started_at_utc = datetime.now(timezone.utc).isoformat()
        self.started_perf_counter = time.perf_counter()
        self.timings: dict[str, _TimingAggregate] = {}
        self.counters: dict[str, float] = {}
        self.gauges: dict[str, _NumericAggregate] = {}
        self._episode_number: int | None = None
        self._episode_started_perf_counter: float | None = None
        self._episode_baseline: dict[str, tuple[int, float, float, float, float]] = {}
        self._episode_counter_baseline: dict[str, float] = {}
        self._wrapped_methods: list[str] = []
        self._unavailable_methods: list[str] = []
        self._active_measurements: list[str] = []
        self._automatic_reset_seen_in_env_step = False
        self.completed_episodes = 0
        self.write_snapshot(status="running")

    def _sync_cuda(self) -> None:
        if not self.cuda_synchronize:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            # Profiling must never make an otherwise valid CPU evaluation fail.
            return

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        self._sync_cuda()
        started = time.perf_counter()
        if name == "simulation.env_step.total":
            self._automatic_reset_seen_in_env_step = False
        self._active_measurements.append(name)
        try:
            yield
        finally:
            self._sync_cuda()
            self._active_measurements.pop()
            self.record(name, time.perf_counter() - started)

    def record(self, name: str, elapsed_s: float) -> None:
        aggregate = self.timings.get(name)
        if aggregate is None:
            aggregate = _TimingAggregate()
            self.timings[name] = aggregate
        aggregate.add(float(elapsed_s))

    def increment(self, name: str, value: float = 1.0) -> None:
        self.counters[name] = self.counters.get(name, 0.0) + float(value)

    def observe(self, name: str, value: float) -> None:
        aggregate = self.gauges.get(name)
        if aggregate is None:
            aggregate = _NumericAggregate()
            self.gauges[name] = aggregate
        aggregate.add(float(value))

    def timed_call(self, name: str, function: Callable[..., Any], *args, **kwargs):
        with self.measure(name):
            return function(*args, **kwargs)

    def wrap_method(self, owner: Any, method_name: str, metric_name: str) -> bool:
        """Time a callable instance attribute without changing its arguments or result."""

        label = f"{type(owner).__name__}.{method_name} -> {metric_name}"
        try:
            original = getattr(owner, method_name)
            if not callable(original):
                raise TypeError(f"attribute is not callable: {method_name}")

            def timed(*args, **kwargs):
                return self.timed_call(metric_name, original, *args, **kwargs)

            setattr(owner, method_name, timed)
        except Exception as exc:
            self._unavailable_methods.append(f"{label}: {exc}")
            return False
        self._wrapped_methods.append(label)
        return True

    def wrap_dynamic_method(
        self,
        owner: Any,
        method_name: str,
        metric_name: Callable[[tuple[Any, ...], dict[str, Any]], str],
    ) -> bool:
        """Time a method under a metric selected from that call's arguments."""

        label = f"{type(owner).__name__}.{method_name} -> dynamic metric"
        try:
            original = getattr(owner, method_name)
            if not callable(original):
                raise TypeError(f"attribute is not callable: {method_name}")

            def timed(*args, **kwargs):
                return self.timed_call(metric_name(args, kwargs), original, *args, **kwargs)

            setattr(owner, method_name, timed)
        except Exception as exc:
            self._unavailable_methods.append(f"{label}: {exc}")
            return False
        self._wrapped_methods.append(label)
        return True

    def wrap_configured_callable(self, cfg: Any, metric_name: str) -> bool:
        """Time a manager term's configured callable while preserving reset helpers."""

        label = f"{type(cfg).__name__}.func -> {metric_name}"
        try:
            original = cfg.func

            def timed(*args, **kwargs):
                return self.timed_call(metric_name, original, *args, **kwargs)

            # Class-based manager terms expose reset/serialize on the callable.
            # Forward those attributes if present so instrumentation is transparent.
            for attribute in ("reset", "serialize"):
                if hasattr(original, attribute):
                    setattr(timed, attribute, getattr(original, attribute))
            cfg.func = timed
        except Exception as exc:
            self._unavailable_methods.append(f"{label}: {exc}")
            return False
        self._wrapped_methods.append(label)
        return True

    @staticmethod
    def _metric_component(value: Any) -> str:
        return "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in str(value)
        )

    def _scope_suffix(self) -> str:
        if "simulation.reset_idx.automatic_terminal" in self._active_measurements or (
            "simulation.env_step.total" in self._active_measurements
            and self._automatic_reset_seen_in_env_step
        ):
            return "automatic_reset"
        if "episode.explicit_reset" in self._active_measurements:
            return "explicit_reset"
        if "simulation.env_step.total" in self._active_measurements:
            return "control_step"
        if "runtime.paused_iteration" in self._active_measurements:
            return "paused"
        return "outside_step"

    def _reset_metric_name(self) -> str:
        if "simulation.env_step.total" in self._active_measurements:
            self._automatic_reset_seen_in_env_step = True
            return "simulation.reset_idx.automatic_terminal"
        return "simulation.reset_idx.explicit"

    def install_simulation_instrumentation(self, env: Any) -> None:
        """Instrument Isaac Lab call boundaries used inside ``env.step``."""

        unwrapped = env.unwrapped
        for owner, method_name, metric_name in (
            (unwrapped.action_manager, "process_action", "simulation.action.process"),
            (unwrapped.action_manager, "apply_action", "simulation.action.apply"),
            (unwrapped.sim, "step", "simulation.physics.step"),
            (unwrapped.scene, "update", "simulation.scene.update"),
            (unwrapped.termination_manager, "compute", "simulation.termination"),
            (unwrapped.reward_manager, "compute", "simulation.reward"),
            (unwrapped.command_manager, "compute", "simulation.command"),
        ):
            self.wrap_method(owner, method_name, metric_name)

        for owner, method_name, metric_base in (
            (unwrapped.scene, "write_data_to_sim", "simulation.scene.write"),
            (unwrapped.sim, "render", "simulation.render"),
            (unwrapped.sim, "forward", "simulation.forward"),
            (unwrapped.observation_manager, "compute", "simulation.observation"),
        ):
            self.wrap_dynamic_method(
                owner,
                method_name,
                lambda _args, _kwargs, base=metric_base: f"{base}.{self._scope_suffix()}",
            )

        self.wrap_dynamic_method(
            unwrapped.event_manager,
            "apply",
            lambda args, kwargs: (
                "simulation.events."
                + self._metric_component(kwargs.get("mode", args[0] if args else "unknown"))
                + f".{self._scope_suffix()}"
            ),
        )

        self.wrap_dynamic_method(
            unwrapped,
            "_reset_idx",
            lambda _args, _kwargs: self._reset_metric_name(),
        )

        self.wrap_dynamic_method(
            unwrapped.observation_manager,
            "compute_group",
            lambda args, kwargs: (
                "simulation.observation.group."
                + self._metric_component(kwargs.get("group_name", args[0] if args else "unknown"))
                + f".{self._scope_suffix()}"
            ),
        )

        termination_manager = unwrapped.termination_manager
        for term_name, term_cfg in zip(
            getattr(termination_manager, "_term_names", ()),
            getattr(termination_manager, "_term_cfgs", ()),
        ):
            self.wrap_configured_callable(
                term_cfg,
                f"simulation.termination.term.{self._metric_component(term_name)}",
            )

        # Sensors are lazy in this environment. Their buffer refresh is therefore
        # usually nested under a termination or observation call, not scene.update.
        for sensor_name, sensor in getattr(unwrapped.scene, "_sensors", {}).items():
            self.wrap_dynamic_method(
                sensor,
                "_update_buffers_impl",
                lambda _args, _kwargs, name=self._metric_component(sensor_name): (
                    f"simulation.sensor.{name}.update_buffers.{self._scope_suffix()}"
                ),
            )

        try:
            robot = unwrapped.scene["robot"]
        except Exception as exc:
            self._unavailable_methods.append(f"scene['robot'].update -> simulation.scene.robot.update: {exc}")
        else:
            self.wrap_method(robot, "update", "simulation.scene.robot.update")

    def start_episode(self, episode: int) -> None:
        if self._episode_number is not None:
            raise RuntimeError(f"Profiler episode {self._episode_number} is already active.")
        self._episode_number = episode
        self._episode_started_perf_counter = time.perf_counter()
        self._episode_baseline = {name: aggregate.values() for name, aggregate in self.timings.items()}
        self._episode_counter_baseline = dict(self.counters)

    def finish_episode(
        self,
        *,
        episode: int,
        success: bool,
        termination_reason: str,
        control_steps: int,
        simulated_seconds: float,
    ) -> None:
        if self._episode_number != episode or self._episode_started_perf_counter is None:
            raise RuntimeError(
                f"Profiler episode mismatch: active={self._episode_number}, completed={episode}."
            )
        wall_s = time.perf_counter() - self._episode_started_perf_counter
        timing_delta: dict[str, dict[str, float | int]] = {}
        all_names = set(self._episode_baseline) | set(self.timings)
        for name in sorted(all_names):
            before = self._episode_baseline.get(name, (0, 0.0, math.inf, 0.0, 0.0))
            after = self.timings.get(name, _TimingAggregate()).values()
            count = after[0] - before[0]
            total_s = after[1] - before[1]
            if count <= 0:
                continue
            timing_delta[name] = {
                "count": count,
                "total_s": total_s,
                "mean_s": total_s / count,
            }
        payload = {
            "episode": episode,
            "success": success,
            "termination_reason": termination_reason,
            "control_steps": control_steps,
            "simulated_seconds": simulated_seconds,
            "wall_seconds": wall_s,
            "real_time_factor": simulated_seconds / wall_s if wall_s > 0.0 else 0.0,
            "wall_seconds_per_simulated_second": wall_s / simulated_seconds if simulated_seconds > 0.0 else 0.0,
            "timings": timing_delta,
            "counters": {
                name: value - self._episode_counter_baseline.get(name, 0.0)
                for name, value in sorted(self.counters.items())
                if value - self._episode_counter_baseline.get(name, 0.0) != 0.0
            },
        }
        with self.episodes_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
        self.completed_episodes += 1
        self._episode_number = None
        self._episode_started_perf_counter = None
        self._episode_baseline = {}
        self._episode_counter_baseline = {}
        self.write_snapshot(status="running")

    def _payload(self, status: str) -> dict[str, Any]:
        elapsed_s = time.perf_counter() - self.started_perf_counter
        active_episode_wall_s = (
            time.perf_counter() - self._episode_started_perf_counter
            if self._episode_started_perf_counter is not None
            else None
        )
        timings = {name: aggregate.as_dict() for name, aggregate in sorted(self.timings.items())}
        for values in timings.values():
            values["fraction_of_run_wall"] = values["total_s"] / elapsed_s if elapsed_s > 0.0 else 0.0
        return {
            "status": status,
            "started_at_utc": self.started_at_utc,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_wall_seconds": elapsed_s,
            "completed_episodes": self.completed_episodes,
            "active_episode": self._episode_number,
            "active_episode_wall_seconds": active_episode_wall_s,
            "cuda_synchronize": self.cuda_synchronize,
            "timings_are_inclusive_and_may_overlap": True,
            "metadata": self.metadata,
            "instrumented_methods": self._wrapped_methods,
            "unavailable_methods": self._unavailable_methods,
            "episode_profile_jsonl": str(self.episodes_jsonl),
            "timings": timings,
            "counters": dict(sorted(self.counters.items())),
            "gauges": {name: aggregate.as_dict() for name, aggregate in sorted(self.gauges.items())},
        }

    def write_snapshot(self, status: str = "running") -> None:
        payload = self._payload(status)
        temporary = self.output_json.with_name(f".{self.output_json.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.output_json)

    def finish(self, status: str = "completed") -> None:
        self.write_snapshot(status=status)
