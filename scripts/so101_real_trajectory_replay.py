#!/usr/bin/env python3
"""Replay one LeRobot trajectory on a real SO-101 follower arm.

This is a hardware characterization tool, not an unattended evaluation runner.
It linearly samples the recorded trajectory at a fixed wall-clock control rate,
logs requested/sent/measured positions, and aborts on sustained tracking error.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import inspect
import math
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any

import numpy as np


JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
FEATURE_NAMES = tuple(f"{joint}.pos" for joint in JOINTS)
POSITION_MINS = np.asarray((-100, -100, -100, -100, -100, 0), dtype=np.float64)
POSITION_MAXS = np.asarray((100, 100, 100, 100, 100, 100), dtype=np.float64)

# The same LeRobot-position limits used by so101_follower_teleop.py. These
# suppress isolated corrupt/discontinuous frames without changing ordinary
# recorded motion at --speed 1.
DEFAULT_REPLAY_VELOCITY_LIMITS = np.asarray((110, 140, 150, 125, 110, 120), dtype=np.float64)
DEFAULT_APPROACH_VELOCITY_LIMITS = np.asarray((20, 20, 20, 20, 30, 30), dtype=np.float64)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a finite positive number, got {value!r}.")
    return parsed


def _six_positive_floats(value: str) -> np.ndarray:
    parts = [part.strip() for part in value.replace(";", ",").split(",")]
    if len(parts) != len(JOINTS):
        raise argparse.ArgumentTypeError(f"Expected six comma-separated values, got {value!r}.")
    parsed = np.asarray([_positive_float(part) for part in parts], dtype=np.float64)
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", default="5hadytru/so101_bench_sim_5")
    parser.add_argument("--repo_root", type=Path, default=Path("data/lerobot/so101_bench_sim_5"))
    parser.add_argument("--episode", "--dataset_episode_index", type=int, default=0)
    parser.add_argument("--follower_port", default="/dev/ttyACM0")
    parser.add_argument("--follower_id", default="follower_arm_1")
    parser.add_argument(
        "--speed",
        type=_positive_float,
        default=0.25,
        help="Trajectory time multiplier. 0.25 replays at one-quarter recorded speed.",
    )
    parser.add_argument("--control_hz", type=_positive_float, default=30.0)
    parser.add_argument(
        "--replay_velocity_limits",
        type=_six_positive_floats,
        default=DEFAULT_REPLAY_VELOCITY_LIMITS.copy(),
        metavar="PAN,LIFT,ELBOW,WRIST,ROLL,GRIPPER",
        help="Per-joint wall-clock command limits in LeRobot units/s.",
    )
    parser.add_argument(
        "--approach_velocity_limits",
        type=_six_positive_floats,
        default=DEFAULT_APPROACH_VELOCITY_LIMITS.copy(),
        metavar="PAN,LIFT,ELBOW,WRIST,ROLL,GRIPPER",
    )
    parser.add_argument(
        "--tracking_error_limit",
        type=_positive_float,
        default=25.0,
        help="Abort if any joint exceeds this sent-vs-measured error for consecutive checks.",
    )
    parser.add_argument("--tracking_error_consecutive", type=int, default=3)
    parser.add_argument(
        "--tracking_error_grace_s",
        type=float,
        default=0.5,
        help="Do not enforce tracking error during this initial portion of replay.",
    )
    parser.add_argument("--initial_hold_s", type=float, default=1.0)
    parser.add_argument("--final_hold_s", type=float, default=1.0)
    parser.add_argument(
        "--log_csv",
        type=Path,
        default=None,
        help="Output log path; defaults to outputs/real_trajectory_replay/episode_N_TIMESTAMP.csv.",
    )
    parser.add_argument(
        "--calibrate_on_connect",
        action="store_true",
        help="Allow interactive LeRobot calibration. Off by default to use follower_arm_1's existing calibration.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the final typed REPLAY confirmation (guided hand-positioning is still required).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load and validate the episode, print its motion statistics, and never connect to hardware.",
    )
    return parser


def _canonical_name(name: Any) -> str:
    value = str(name)
    if value in JOINTS:
        return f"{value}.pos"
    return value


def _load_episode(repo_id: str, root: Path, episode_index: int) -> tuple[np.ndarray, float]:
    if episode_index < 0:
        raise ValueError(f"--episode must be nonnegative, got {episode_index}.")
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    kwargs: dict[str, Any] = {}
    signature = inspect.signature(LeRobotDataset)
    if "root" in signature.parameters:
        kwargs["root"] = root
    if "episodes" in signature.parameters:
        kwargs["episodes"] = [episode_index]
    if "download_videos" in signature.parameters:
        kwargs["download_videos"] = False
    dataset = LeRobotDataset(repo_id, **kwargs)

    fps = getattr(dataset, "fps", None) or getattr(getattr(dataset, "meta", None), "fps", None) or 30
    features = getattr(dataset, "features", {})
    action_feature = features.get("action", {})
    names = action_feature.get("names") or FEATURE_NAMES
    names = [_canonical_name(name) for name in names]
    index_by_name = {name: index for index, name in enumerate(names)}
    missing = [name for name in FEATURE_NAMES if name not in index_by_name]
    if missing:
        raise ValueError(f"Dataset action names are missing {missing}; found {names}.")

    rows = dataset.select_columns("action") if hasattr(dataset, "select_columns") else dataset.hf_dataset.select_columns("action")
    num_frames = int(getattr(dataset, "num_frames", len(rows)))
    actions = np.empty((num_frames, len(JOINTS)), dtype=np.float64)
    for frame_index in range(num_frames):
        raw = rows[frame_index]["action"]
        if isinstance(raw, dict):
            raw_names = [_canonical_name(name) for name in raw]
            raw_values = np.asarray(list(raw.values()), dtype=np.float64).reshape(-1)
            raw_index = {name: index for index, name in enumerate(raw_names)}
            actions[frame_index] = [raw_values[raw_index[name]] for name in FEATURE_NAMES]
        else:
            raw_values = np.asarray(raw, dtype=np.float64).reshape(-1)
            actions[frame_index] = [raw_values[index_by_name[name]] for name in FEATURE_NAMES]

    if num_frames < 2:
        raise ValueError(f"Episode {episode_index} contains only {num_frames} frame(s).")
    if not np.isfinite(actions).all():
        frame, joint = np.argwhere(~np.isfinite(actions))[0]
        raise ValueError(f"Non-finite action at frame {frame}, joint {JOINTS[joint]}.")
    out_of_range = (actions < POSITION_MINS - 1e-3) | (actions > POSITION_MAXS + 1e-3)
    if out_of_range.any():
        frame, joint = np.argwhere(out_of_range)[0]
        raise ValueError(
            f"Out-of-range action at frame {frame}, {JOINTS[joint]}={actions[frame, joint]:.3f}."
        )
    return actions, float(fps)


def _print_summary(actions: np.ndarray, fps: float, speed: float) -> None:
    rates = np.abs(np.diff(actions, axis=0)) * fps
    duration = (len(actions) - 1) / fps
    print(f"[INFO] Frames: {len(actions)}, dataset FPS: {fps:g}")
    print(f"[INFO] Recorded duration: {duration:.2f}s; replay duration at {speed:g}x: {duration / speed:.2f}s")
    print("[INFO] Recorded per-joint action-rate p95 / p99 / max (LeRobot units/s):")
    for joint_index, joint in enumerate(JOINTS):
        p95, p99 = np.quantile(rates[:, joint_index], (0.95, 0.99))
        print(f"  {joint:14s} {p95:7.1f} / {p99:7.1f} / {rates[:, joint_index].max():7.1f}")
    print("[INFO] First action: " + ", ".join(f"{name}={value:.1f}" for name, value in zip(JOINTS, actions[0])))


def _import_robot_types():
    try:
        from lerobot.robots.so101_follower import SO101FollowerConfig
    except ImportError:
        try:
            from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
        except ImportError:
            from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.robots import make_robot_from_config

    return SO101FollowerConfig, make_robot_from_config


def _make_robot(args: argparse.Namespace):
    config_type, make_robot_from_config = _import_robot_types()
    signature = inspect.signature(config_type)
    kwargs: dict[str, Any] = {"port": args.follower_port}
    if "id" in signature.parameters:
        kwargs["id"] = args.follower_id
    if "cameras" in signature.parameters:
        kwargs["cameras"] = {}
    if "disable_torque_on_disconnect" in signature.parameters:
        kwargs["disable_torque_on_disconnect"] = True
    if "use_degrees" in signature.parameters:
        kwargs["use_degrees"] = False
    if "max_relative_target" in signature.parameters:
        kwargs["max_relative_target"] = None
    return make_robot_from_config(config_type(**kwargs))


def _observation_vector(robot) -> np.ndarray:
    observation = robot.get_observation()
    values = []
    for name in FEATURE_NAMES:
        if name in observation:
            values.append(float(observation[name]))
        elif name.removesuffix(".pos") in observation:
            values.append(float(observation[name.removesuffix(".pos")]))
        else:
            raise KeyError(f"Follower observation lacks {name!r}; got {list(observation)}")
    return np.asarray(values, dtype=np.float64)


def _action_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(FEATURE_NAMES, values)}


def _send(robot, values: np.ndarray) -> np.ndarray:
    sent = robot.send_action(_action_dict(values))
    return np.asarray([float(sent.get(name, values[index])) for index, name in enumerate(FEATURE_NAMES)])


def _disable_torque(robot) -> None:
    bus = getattr(robot, "bus", None)
    if bus is not None and hasattr(bus, "disable_torque"):
        bus.disable_torque()
    elif hasattr(robot, "disable_torque"):
        robot.disable_torque()
    else:
        raise RuntimeError("Could not find a follower torque-disable method.")


def _enable_torque(robot) -> None:
    bus = getattr(robot, "bus", None)
    if bus is not None and hasattr(bus, "enable_torque"):
        bus.enable_torque()
    elif hasattr(robot, "enable_torque"):
        robot.enable_torque()
    else:
        raise RuntimeError("Could not find a follower torque-enable method.")


def _seed_goal_at_current_position(robot, current: np.ndarray) -> None:
    bus = getattr(robot, "bus", None)
    if bus is None or not hasattr(bus, "sync_write"):
        raise RuntimeError("Follower motor bus does not expose sync_write; refusing unsafe torque enable.")
    bus.sync_write("Goal_Position", {joint: float(value) for joint, value in zip(JOINTS, current)})


def _sleep_to_deadline(deadline: float) -> None:
    remaining = deadline - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def _rate_limit(target: np.ndarray, previous: np.ndarray, limits: np.ndarray, dt: float) -> np.ndarray:
    max_delta = limits * dt
    return previous + np.clip(target - previous, -max_delta, max_delta)


def _approach(robot, start: np.ndarray, target: np.ndarray, limits: np.ndarray, hz: float, stop: threading.Event) -> np.ndarray:
    delta = np.abs(target - start)
    duration = max(1.0, float(np.max(delta / limits)))
    steps = max(1, int(math.ceil(duration * hz)))
    print(f"[INFO] Approaching first action over {duration:.2f}s ({steps} control steps). Press Ctrl-C or ENTER to abort.")
    sent = start.copy()
    begin = time.perf_counter()
    for step in range(1, steps + 1):
        if stop.is_set():
            raise KeyboardInterrupt
        fraction = step / steps
        sent = _send(robot, start + fraction * (target - start))
        _sleep_to_deadline(begin + step / hz)
    return sent


def _start_enter_abort_listener(stop: threading.Event) -> None:
    def listen() -> None:
        try:
            input()
        except EOFError:
            return
        print("\n[ABORT] ENTER pressed; stopping replay.")
        stop.set()

    threading.Thread(target=listen, name="enter-abort", daemon=True).start()


def _default_log_path(episode: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/real_trajectory_replay") / f"episode_{episode:06d}_{stamp}.csv"


def _replay(
    robot,
    actions: np.ndarray,
    dataset_fps: float,
    args: argparse.Namespace,
    stop: threading.Event,
    writer: csv.writer,
) -> None:
    control_dt = 1.0 / args.control_hz
    source_duration = (len(actions) - 1) / dataset_fps
    wall_duration = source_duration / args.speed
    previous_sent = actions[0].copy()
    consecutive_errors = 0
    error_samples: list[np.ndarray] = []
    limited_steps = 0
    late_steps = 0
    step = 0
    start = time.perf_counter()

    print(f"[INFO] Replaying episode {args.episode} at {args.speed:g}x for approximately {wall_duration:.2f}s.")
    while not stop.is_set():
        loop_start = time.perf_counter()
        wall_elapsed = loop_start - start
        source_time = min(wall_elapsed * args.speed, source_duration)
        source_position = source_time * dataset_fps
        lower = min(int(math.floor(source_position)), len(actions) - 1)
        upper = min(lower + 1, len(actions) - 1)
        alpha = source_position - lower
        requested = (1.0 - alpha) * actions[lower] + alpha * actions[upper]
        target = _rate_limit(requested, previous_sent, args.replay_velocity_limits, control_dt)
        if not np.allclose(target, requested, atol=1e-6, rtol=0):
            limited_steps += 1

        measured_before = _observation_vector(robot)
        prior_error = np.abs(previous_sent - measured_before)
        error_samples.append(prior_error)
        sent = _send(robot, target)
        max_error = float(prior_error.max())
        max_error_joint = JOINTS[int(prior_error.argmax())]
        writer.writerow(
            [
                step,
                f"{wall_elapsed:.6f}",
                f"{source_time:.6f}",
                lower,
                upper,
                f"{alpha:.6f}",
                *requested.tolist(),
                *sent.tolist(),
                *measured_before.tolist(),
                *prior_error.tolist(),
            ]
        )

        if wall_elapsed >= args.tracking_error_grace_s and max_error > args.tracking_error_limit:
            consecutive_errors += 1
            print(
                f"[WARN] Tracking error {max_error:.1f} on {max_error_joint} "
                f"({consecutive_errors}/{args.tracking_error_consecutive})."
            )
            if consecutive_errors >= args.tracking_error_consecutive:
                raise RuntimeError(
                    f"Sustained tracking error exceeded {args.tracking_error_limit:g} on {max_error_joint}."
                )
        else:
            consecutive_errors = 0

        previous_sent = sent
        step += 1
        if source_time >= source_duration:
            break
        deadline = start + step * control_dt
        if time.perf_counter() > deadline:
            late_steps += 1
        _sleep_to_deadline(deadline)

    if stop.is_set():
        raise KeyboardInterrupt
    errors = np.stack(error_samples)
    elapsed = max(time.perf_counter() - start, 1e-9)
    print(f"[RESULT] Effective control rate: {step / elapsed:.1f} Hz; late steps: {late_steps}/{step}.")
    print(f"[RESULT] Safety velocity limiter changed {limited_steps}/{step} requested control steps.")
    print("[RESULT] Sent-vs-measured tracking error p95 / max (LeRobot units):")
    for joint_index, joint in enumerate(JOINTS):
        print(
            f"  {joint:14s} {np.quantile(errors[:, joint_index], 0.95):7.2f} / "
            f"{errors[:, joint_index].max():7.2f}"
        )
    if args.final_hold_s > 0:
        print(f"[INFO] Holding final action for {args.final_hold_s:.2f}s.")
        hold_end = time.perf_counter() + args.final_hold_s
        while time.perf_counter() < hold_end and not stop.is_set():
            _send(robot, previous_sent)
            time.sleep(min(control_dt, max(0.0, hold_end - time.perf_counter())))


def main() -> int:
    args = _parser().parse_args()
    if args.tracking_error_consecutive < 1:
        raise ValueError("--tracking_error_consecutive must be at least 1.")
    if args.tracking_error_grace_s < 0 or args.initial_hold_s < 0 or args.final_hold_s < 0:
        raise ValueError("Hold and grace durations must be nonnegative.")

    actions, dataset_fps = _load_episode(args.repo_id, args.repo_root, args.episode)
    _print_summary(actions, dataset_fps, args.speed)
    if args.dry_run:
        print("[INFO] Dry run complete; no hardware connection was attempted.")
        return 0

    print("\n[SAFETY] Clear the workspace and keep a hand on the power cutoff.")
    print("[SAFETY] On exit, LeRobot disables torque; support the arm so it cannot fall.")
    if not args.yes:
        confirmation = input("Type REPLAY to connect to the real arm: ").strip()
        if confirmation != "REPLAY":
            print("[INFO] Confirmation not received; exiting without connecting.")
            return 2

    log_path = args.log_csv or _default_log_path(args.episode)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    robot = _make_robot(args)
    connected = False

    try:
        connect_signature = inspect.signature(robot.connect)
        if "calibrate" in connect_signature.parameters:
            robot.connect(calibrate=args.calibrate_on_connect)
        else:
            robot.connect()
        connected = True
        print(f"[INFO] Connected follower '{args.follower_id}' on {args.follower_port}.")

        _disable_torque(robot)
        print("\n[POSITION] Torque is OFF. Support the arm and place it near the printed first action.")
        input("Press ENTER when the arm is supported and positioned; torque will then turn on: ")
        current = _observation_vector(robot)
        print("[INFO] Current pose: " + ", ".join(f"{name}={value:.1f}" for name, value in zip(JOINTS, current)))
        _seed_goal_at_current_position(robot, current)
        _enable_torque(robot)
        print("[INFO] Torque enabled at the measured current pose.")

        _start_enter_abort_listener(stop)
        first_sent = _approach(
            robot,
            current,
            actions[0],
            args.approach_velocity_limits,
            args.control_hz,
            stop,
        )
        if args.initial_hold_s > 0:
            print(f"[INFO] Holding first action for {args.initial_hold_s:.2f}s.")
            hold_end = time.perf_counter() + args.initial_hold_s
            while time.perf_counter() < hold_end and not stop.is_set():
                first_sent = _send(robot, first_sent)
                time.sleep(min(1.0 / args.control_hz, max(0.0, hold_end - time.perf_counter())))

        with log_path.open("w", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(
                [
                    "control_step",
                    "wall_time_s",
                    "source_time_s",
                    "source_frame_lower",
                    "source_frame_upper",
                    "source_alpha",
                    *(f"requested.{joint}" for joint in JOINTS),
                    *(f"sent.{joint}" for joint in JOINTS),
                    *(f"measured_before_send.{joint}" for joint in JOINTS),
                    *(f"tracking_error.{joint}" for joint in JOINTS),
                ]
            )
            _replay(robot, actions, dataset_fps, args, stop, writer)
        print(f"[DONE] Replay completed. Tracking log: {log_path}")
        return 0
    except KeyboardInterrupt:
        print("\n[ABORT] Replay interrupted.")
        return 130
    except Exception as exc:
        print(f"\n[ABORT] {exc}", file=sys.stderr)
        return 1
    finally:
        if connected:
            try:
                robot.disconnect()
                print("[INFO] Follower disconnected; torque should now be off. Support the arm.")
            except Exception as exc:
                print(f"[WARN] Failed to disconnect cleanly: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
