#!/usr/bin/env python3
"""Render a concise, human-readable report from a GR00T eval profile JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as profile_file:
        return json.load(profile_file)


def _timing_rows(
    timings: dict[str, dict[str, float | int]],
    names: list[str],
) -> list[tuple[str, dict[str, float | int]]]:
    return [(name, timings[name]) for name in names if name in timings]


def _prefixed_rows(
    timings: dict[str, dict[str, float | int]],
    prefixes: tuple[str, ...],
    *,
    excluded: set[str] | None = None,
) -> list[tuple[str, dict[str, float | int]]]:
    excluded = excluded or set()
    return [
        (name, values)
        for name, values in sorted(timings.items())
        if name not in excluded and name.startswith(prefixes)
    ]


def _print_timing_table(
    title: str,
    rows: list[tuple[str, dict[str, float | int]]],
    run_wall_s: float,
    *,
    reference_s: float | None = None,
) -> None:
    if not rows:
        return
    print(f"\n{title}")
    reference_heading = "scope %" if reference_s is not None else ""
    print(
        f"{'metric':55} {'calls':>9} {'total s':>10} {'mean ms':>10} "
        f"{'p95 ms':>10} {'run %':>7} {reference_heading:>8}"
    )
    print("-" * 123)
    for name, values in rows:
        total_s = float(values.get("total_s", 0.0))
        mean_ms = 1000.0 * float(values.get("mean_s", 0.0))
        p95_ms = 1000.0 * float(values.get("p95_s", values.get("mean_s", 0.0)))
        wall_fraction = 100.0 * total_s / run_wall_s if run_wall_s > 0.0 else 0.0
        reference_fraction = 100.0 * total_s / reference_s if reference_s and reference_s > 0.0 else None
        reference_cell = f"{reference_fraction:8.2f}" if reference_fraction is not None else " " * 8
        print(
            f"{name:55} {int(values.get('count', 0)):9d} {total_s:10.3f} "
            f"{mean_ms:10.3f} {p95_ms:10.3f} {wall_fraction:7.2f} {reference_cell}"
        )


def _episode_summary(profile: dict[str, Any]) -> None:
    episode_path_text = profile.get("episode_profile_jsonl")
    if not episode_path_text:
        return
    episode_path = Path(episode_path_text)
    if not episode_path.is_file():
        return
    rows = []
    with episode_path.open(encoding="utf-8") as episode_file:
        for line in episode_file:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return
    simulated_s = sum(float(row.get("simulated_seconds", 0.0)) for row in rows)
    wall_s = sum(float(row.get("wall_seconds", 0.0)) for row in rows)
    wall_per_sim = [
        float(row["wall_seconds_per_simulated_second"])
        for row in rows
        if float(row.get("simulated_seconds", 0.0)) > 0.0
    ]
    aggregate_rtf = simulated_s / wall_s if wall_s > 0.0 else 0.0
    print("\nEnd-to-end episode throughput (includes reset, policy, recording, and finalization)")
    print(f"  completed episode profiles: {len(rows)}")
    print(f"  simulated / episode wall:   {simulated_s:.2f}s / {wall_s:.2f}s")
    print(f"  aggregate real-time factor: {aggregate_rtf:.4f}x")
    if wall_per_sim:
        print(f"  wall seconds / sim second:  {wall_s / simulated_s:.3f} aggregate")
        print(f"  median by episode:          {statistics.median(wall_per_sim):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_json", type=Path, help="eval_profile.json written by scripts/groot_eval.py")
    args = parser.parse_args()

    profile = _load_json(args.profile_json.expanduser().resolve())
    timings = profile.get("timings", {})
    run_wall_s = float(profile.get("run_wall_seconds", 0.0))
    metadata = profile.get("metadata", {})

    print("GR00T evaluation performance profile")
    print(f"  status:              {profile.get('status', 'unknown')}")
    print(f"  run wall time:       {run_wall_s:.2f}s")
    print(f"  completed episodes:  {profile.get('completed_episodes', 0)}")
    print(f"  physics/control Hz:  {metadata.get('physics_hz', '?')} / {metadata.get('control_hz', '?')}")
    print(f"  action horizon:      {metadata.get('action_horizon', '?')}")
    print(f"  pre-spawned objects: {metadata.get('object_asset_count', '?')}")
    print(f"  recording enabled:   {metadata.get('record_dataset', '?')}")
    print(f"  Isaac-boundary sync: {profile.get('cuda_synchronize', False)}")
    print("  layout work:         excluded")
    unavailable_methods = profile.get("unavailable_methods", [])
    if unavailable_methods:
        print(f"  unavailable hooks:   {len(unavailable_methods)} (details below)")

    headline_names = [
        "simulation.env_step.total",
        "policy.control_step.total",
        "recording.frame.total",
        "recording.episode_finalize",
        "diagnostics.episode_jsonl_write",
    ]
    headline_rows = _timing_rows(timings, headline_names)
    _print_timing_table("End-to-end call boundaries", headline_rows, run_wall_s)

    lifecycle_names = [
        "setup.environment_create",
        "setup.camera_discovery",
        "setup.recording_construct",
        "setup.policy_connect",
        "setup.recording_initialize",
        "episode.apply_precomputed_configuration",
        "episode.explicit_reset",
        "episode.policy_reset",
        "recording.episode_start",
        "cleanup.recording_finalize",
        "cleanup.environment_close",
        "profiler.encoder_telemetry",
    ]
    lifecycle_rows = _timing_rows(timings, lifecycle_names)
    _print_timing_table("Setup, reset, and cleanup", lifecycle_rows, run_wall_s)

    simulation_names = [
        "simulation.action.process",
        "simulation.action.apply",
        "simulation.scene.write.control_step",
        "simulation.physics.step",
        "simulation.render.control_step",
        "simulation.scene.update",
        "simulation.termination",
        "simulation.reward",
        "simulation.command",
        "simulation.events.interval.control_step",
        "simulation.observation.control_step",
        "simulation.reset_idx.automatic_terminal",
    ]
    simulation_rows = _timing_rows(timings, simulation_names)
    env_step_s = float(timings.get("simulation.env_step.total", {}).get("total_s", 0.0))
    _print_timing_table(
        "Simulation components (scope % = share of env.step wall)",
        simulation_rows,
        run_wall_s,
        reference_s=env_step_s,
    )

    nested_simulation_rows = _prefixed_rows(
        timings,
        (
            "simulation.observation.group.",
            "simulation.termination.term.",
            "simulation.sensor.",
            "simulation.scene.robot.update",
        ),
    )
    _print_timing_table(
        "Nested sim detail (do not add to parent rows)",
        nested_simulation_rows,
        run_wall_s,
        reference_s=env_step_s,
    )

    reset_names = [
        "simulation.reset_idx.explicit",
        "simulation.events.reset.explicit_reset",
        "simulation.events.reset.automatic_reset",
        "simulation.events.interval.automatic_reset",
        "simulation.scene.write.explicit_reset",
        "simulation.forward.explicit_reset",
        "simulation.render.explicit_reset",
        "simulation.observation.explicit_reset",
        "simulation.observation.automatic_reset",
        "simulation.render.automatic_reset",
        "simulation.render.paused",
        "simulation.render.outside_step",
    ]
    reset_rows = _timing_rows(timings, reset_names)
    _print_timing_table("Reset and out-of-step sim detail", reset_rows, run_wall_s)

    physics_pipeline_s = sum(
        float(timings.get(name, {}).get("total_s", 0.0))
        for name in ("simulation.physics.step", "simulation.scene.update")
    )
    camera_buffer_s = sum(
        float(values.get("total_s", 0.0))
        for name, values in timings.items()
        if name.startswith("simulation.sensor.camera_") and name.endswith(".control_step")
    )
    render_pipeline_s = float(timings.get("simulation.render.control_step", {}).get("total_s", 0.0)) + camera_buffer_s
    if env_step_s > 0.0:
        print("\nApproximate host-attributed pipelines (not pure GPU kernel time)")
        print(
            f"  physics step + scene update:       {physics_pipeline_s:.3f}s "
            f"({100.0 * physics_pipeline_s / env_step_s:.2f}% of env.step)"
        )
        print(
            f"  render + camera buffer retrieval:  {render_pipeline_s:.3f}s "
            f"({100.0 * render_pipeline_s / env_step_s:.2f}% of env.step)"
        )

    policy_rows = _prefixed_rows(
        timings,
        ("policy.",),
        excluded={"policy.control_step.total"},
    )
    policy_control_s = float(timings.get("policy.control_step.total", {}).get("total_s", 0.0))
    _print_timing_table(
        "Policy/inference detail (scope % = share of policy control wall)",
        policy_rows,
        run_wall_s,
        reference_s=policy_control_s,
    )

    recording_rows = _prefixed_rows(
        timings,
        ("recording.",),
        excluded={"recording.frame.total", "recording.episode_finalize"},
    )
    recording_frame_s = float(timings.get("recording.frame.total", {}).get("total_s", 0.0))
    _print_timing_table(
        "Recording detail (scope % = share of recording frame wall)",
        recording_rows,
        run_wall_s,
        reference_s=recording_frame_s,
    )

    gauges = profile.get("gauges", {})
    if gauges:
        print("\nRecorder encoder queue telemetry")
        print(f"{'metric':74} {'samples':>9} {'mean':>10} {'max':>10}")
        print("-" * 108)
        for name, values in sorted(gauges.items()):
            print(
                f"{name:74} {int(values.get('count', 0)):9d} "
                f"{float(values.get('mean', 0.0)):10.2f} {float(values.get('max', 0.0)):10.2f}"
            )

    counters = profile.get("counters", {})
    counter_names = [
        "simulation.control_steps",
        "simulation.terminal_control_steps",
        "simulation.simulated_seconds",
        "policy.action_chunks",
        "policy.actions_received",
        "policy.action_cache.hits",
        "policy.action_cache.misses",
        "policy.rpc.get_action.request_bytes",
        "policy.rpc.get_action.response_bytes",
    ]
    present_counters = [(name, counters[name]) for name in counter_names if name in counters]
    present_counters.extend(
        (name, value)
        for name, value in sorted(counters.items())
        if name.startswith("recording.encoder.dropped_frames_total.")
    )
    if present_counters:
        print("\nCounts and volumes")
        for name, value in present_counters:
            print(f"  {name:48} {value:g}")

    _episode_summary(profile)
    if unavailable_methods:
        print("\nUnavailable instrumentation hooks")
        for unavailable in unavailable_methods:
            print(f"  - {unavailable}")
    print("\nInterpretation notes")
    print("  - Timings are inclusive and nested; wall percentages are individually meaningful but are not additive.")
    print("  - physics.step can defer GPU work to scene.update; use their combined pipeline before blaming the solver.")
    print("  - sim.render includes Fabric/Kit/RTX work; deferred work can land in camera-buffer retrieval.")
    print(
        "  - Lazy sensor refreshes are charged where data is read, usually visual observations or termination checks."
    )
    print("  - policy.server.handler_reported includes preprocessing, model execution, and action postprocessing.")
    print("  - RPC receive_wait also contains server queueing/serialization and local TCP transport.")
    print("  - Default timings preserve throughput but asynchronous GPU work can land at a later blocking boundary.")
    print(
        "  - CUDA sync mode covers evaluator/Isaac boundaries; "
        "internal policy/recorder submetrics stay host-observed."
    )
    print("  - Encoder thread CPU work overlaps the main loop; queue depth/drop counts expose harmful backpressure.")


if __name__ == "__main__":
    main()
