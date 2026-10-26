#!/usr/bin/env python3
"""Measure how the real SO-101 follower's motor ticks line up with the sim USD joints.

`measure` and `live` are read-only on the motor bus; only `verify` enables torque. Hand-place
the limp arm into reference poses; at the USD zero pose every motor case is square to
the base, so a phone level (zeroed on the table) on a motor case gives each pitch/roll
joint's zero exactly. Pan uses a straightedge, the gripper uses jaw-tip contact.

    python scripts/so101_sim_alignment.py measure            # guided capture, writes the JSON
    python scripts/so101_sim_alignment.py live               # stream sim angles under the saved alignment
    python scripts/so101_sim_alignment.py verify             # TORQUE ON: slowly visit sim poses, log tracking error

Run with /home/truman/env_isaaclab_51/bin/python (it has LeRobot). The JSON is consumed by
`so101_bench.utils.real_sim_alignment.RealSimAlignment`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import queue
import statistics
import sys
import threading
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = REPO_ROOT / "source" / "so101_bench" / "so101_bench" / "utils"


def _load_module(name: str, path: Path):
    # The so101_bench package imports Isaac Lab; these two modules are plain Python.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


alignment_lib = _load_module("_so101_real_sim_alignment", UTILS_DIR / "real_sim_alignment.py")
calibration_lib = _load_module("_so101_lerobot_calibration", UTILS_DIR / "lerobot_calibration.py")

JOINTS = alignment_lib.JOINT_ORDER
TICKS_PER_DEGREE = alignment_lib.TICKS_PER_DEGREE

# Jaw angle at which the USD fingertips meet (vertex gap 1.2 mm at -11 deg, 0.3 mm at -12 deg).
SIM_JAW_TOUCH_DEG = -12.0
SAMPLES_PER_CAPTURE = 15
SAMPLE_PERIOD_S = 0.05
MAX_CAPTURE_STD_TICKS = 4.0
REPEAT_DISAGREEMENT_DEG = 1.5
CHECK_TOLERANCE_DEG = 2.0

LEVEL_SETUP = (
    "Set your phone level's reference on the table right next to the base (iPhone Measure > Level: lay it "
    "flat and tap the screen). A face is 'square' when it reads 0 deg flat or 0 deg on its edge."
)


@dataclass(frozen=True)
class Step:
    key: str
    chain: tuple[str, ...]
    target_deg: float
    solves: str | None
    instructions: str


ZERO_STEPS = (
    Step(
        "lift_zero",
        ("shoulder_lift",),
        0.0,
        "shoulder_lift",
        "UPPER ARM VERTICAL. Put the level on a flat face of the ELBOW motor case (the motor at the top of "
        "the upper arm), on a face parallel to that motor's shaft, not the horn face. Hold the upper arm so "
        "that face is square. The forearm can be anywhere.",
    ),
    Step(
        "elbow_zero",
        ("shoulder_lift", "elbow_flex"),
        0.0,
        "elbow_flex",
        "FOREARM HORIZONTAL, POINTING FORWARD. Level on the WRIST-FLEX motor case at the end of the forearm. "
        "The upper arm can sit anywhere; its angle is subtracted.",
    ),
    Step(
        "wrist_flex_zero",
        ("shoulder_lift", "elbow_flex", "wrist_flex"),
        0.0,
        "wrist_flex",
        "WRIST STRAIGHT AND HORIZONTAL. Level on the WRIST-ROLL motor case (in the wrist; its shaft runs along "
        "the arm). Upper arm and forearm can be anywhere.",
    ),
    Step(
        "roll_zero",
        ("wrist_roll",),
        0.0,
        "wrist_roll",
        "GRIPPER UPRIGHT: moving jaw on TOP, wrist camera on the robot's LEFT. Level on the top face of the "
        "GRIPPER motor case. Keep the wrist within ~10 deg of horizontal while you do this.",
    ),
    Step(
        "pan_zero",
        ("shoulder_pan",),
        0.0,
        "shoulder_pan",
        "ARM POINTING STRAIGHT OUT THE FRONT of the base. Lay a ruler against the flat side of the base "
        "so it extends forward; the shoulder bracket / forearm sides should run parallel to it. Sight from "
        "directly above.",
    ),
    Step(
        "gripper_closed",
        ("gripper",),
        SIM_JAW_TOUCH_DEG,
        "gripper",
        "JAW TIPS JUST TOUCHING. Close the gripper by hand until the tips meet. Don't squeeze.",
    ),
)

CHECK_STEPS = (
    Step(
        "lift_check",
        ("shoulder_lift",),
        90.0,
        None,
        "CHECK: UPPER ARM HORIZONTAL POINTING FORWARD (elbow motor case square again).",
    ),
    Step(
        "elbow_check",
        ("shoulder_lift", "elbow_flex"),
        -90.0,
        None,
        "CHECK: FOREARM POINTING STRAIGHT UP (wrist-flex motor case square).",
    ),
    Step(
        "wrist_flex_check",
        ("shoulder_lift", "elbow_flex", "wrist_flex"),
        90.0,
        None,
        "CHECK: GRIPPER POINTING STRAIGHT DOWN (wrist-roll motor case square).",
    ),
    Step(
        "roll_check",
        ("wrist_roll",),
        90.0,
        None,
        "CHECK: ROLLED A QUARTER TURN so the moving jaw is on the robot's LEFT and the camera underneath "
        "(gripper motor case square, wrist roughly horizontal).",
    ),
    Step(
        "pan_check",
        ("shoulder_pan",),
        90.0,
        None,
        "CHECK: ARM POINTING STRAIGHT TO THE ROBOT'S RIGHT, parallel to the base's front face.",
    ),
    Step(
        "gripper_open",
        ("gripper",),
        calibration_lib.USD_SIM_JOINT_LIMITS_DEG["gripper"][1],
        None,
        "INFO: GRIPPER FULLY OPEN against its stop (compared to the USD's upper limit; not pass/fail).",
    ),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("measure", "live", "verify"))
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot_id", default="follower_arm_1")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to assets/real_robot/<id>_sim_alignment.json.")
    parser.add_argument("--repeats", type=int, default=2, help="Captures per zero step; re-place the arm between them.")
    parser.add_argument("--skip_checks", action="store_true")
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated joints to re-measure (e.g. 'gripper' after adding finger pads); "
        "the rest are kept from the saved alignment.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Wrist camera index for verify snapshots.")
    return parser


class ReadOnlyArm:
    def __init__(self, port: str):
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        self.bus = FeetechMotorsBus(
            port=port,
            motors={name: Motor(index + 1, "sts3215", MotorNormMode.RANGE_M100_100) for index, name in enumerate(JOINTS)},
        )
        self.bus.connect()
        torque = self.bus.sync_read("Torque_Enable", normalize=False)
        if any(torque.values()):
            self.close()
            raise SystemExit(
                f"Torque is enabled ({torque}). Release the arm safely first: run "
                "scripts/so101_real_policy_runner.py --no_policy and press q (docks, then torque off). "
                "Power-cycling drops the arm wherever it is."
            )
        self.calibration = self.bus.read_calibration()

    def ticks(self) -> dict[str, int]:
        return self.bus.sync_read("Present_Position", normalize=False)

    def close(self) -> None:
        self.bus.port_handler.closePort()


class ProvisionalMapping:
    """The mapping teleop used: motor-file ranges -> `.pos` -> hardcoded sim mapping."""

    def __init__(self, motor_calibration):
        self.motor_calibration = motor_calibration

    def sim_degrees(self, joint: str, ticks: float) -> float:
        calibration = self.motor_calibration[joint]
        normalized = (ticks - calibration.range_min) / (calibration.range_max - calibration.range_min)
        position = normalized * 100.0 if joint == "gripper" else normalized * 200.0 - 100.0
        return calibration_lib.lerobot_position_to_sim_degrees(joint, position)


def _chain_degrees(chain, ticks, zeros, provisional) -> tuple[float, bool]:
    total = 0.0
    exact = True
    for joint in chain:
        if joint in zeros:
            total += (ticks[joint] - zeros[joint]) / TICKS_PER_DEGREE
        else:
            total += provisional.sim_degrees(joint, ticks[joint])
            exact = False
    return total, exact


def _stdin_reader(lines: queue.Queue[str]) -> None:
    for line in sys.stdin:
        lines.put(line.strip().lower())
    lines.put("q")


def _capture(arm: ReadOnlyArm) -> tuple[dict[str, float], dict[str, float]]:
    samples = []
    for _ in range(SAMPLES_PER_CAPTURE):
        samples.append(arm.ticks())
        time.sleep(SAMPLE_PERIOD_S)
    mean = {joint: statistics.fmean(sample[joint] for sample in samples) for joint in JOINTS}
    std = {joint: statistics.pstdev(sample[joint] for sample in samples) for joint in JOINTS}
    return mean, std


def _run_step(arm, step, lines, zeros, provisional, repeats) -> list[dict[str, float]] | None:
    """Live-display the chain angle; Enter captures, 's' skips, 'q' quits."""

    print(f"\n=== {step.key}: target {step.target_deg:+.1f} deg on {' + '.join(step.chain)}")
    print(step.instructions)
    print("Enter = capture   s = skip   q = quit")
    captures: list[dict[str, float]] = []
    while len(captures) < repeats:
        while True:
            try:
                command = lines.get_nowait()
                break
            except queue.Empty:
                ticks = arm.ticks()
                angle, exact = _chain_degrees(step.chain, ticks, zeros, provisional)
                label = "" if exact else " (provisional)"
                raw = " ".join(f"{joint}={ticks[joint]}" for joint in step.chain)
                print(f"\r  now {angle:+7.2f} deg{label}   raw {raw}      ", end="", flush=True)
                time.sleep(0.1)
        print()
        if command == "q":
            raise KeyboardInterrupt
        if command == "s":
            return None
        mean, std = _capture(arm)
        worst = max(std[joint] for joint in step.chain)
        if worst > MAX_CAPTURE_STD_TICKS:
            print(f"  arm moved during capture (std {worst:.1f} ticks); hold still and press Enter again.")
            continue
        captures.append(mean)
        angle, _ = _chain_degrees(step.chain, mean, zeros, provisional)
        print(f"  captured {len(captures)}/{repeats}: {angle:+.2f} deg")
        if len(captures) < repeats:
            print("  Move the arm away, re-place it, and press Enter again.")
    return captures


def _solve_zero(step: Step, captures, zeros) -> float:
    estimates = []
    for ticks in captures:
        others = sum((ticks[joint] - zeros[joint]) / TICKS_PER_DEGREE for joint in step.chain if joint != step.solves)
        estimates.append(ticks[step.solves] - (step.target_deg - others) * TICKS_PER_DEGREE)
    spread = (max(estimates) - min(estimates)) / TICKS_PER_DEGREE
    if spread > REPEAT_DISAGREEMENT_DEG:
        print(f"  WARNING: repeats disagree by {spread:.2f} deg; consider redoing this step.")
    else:
        print(f"  repeats agree within {spread:.2f} deg")
    return statistics.fmean(estimates)


def measure(args) -> None:
    arm = ReadOnlyArm(args.port)
    provisional = ProvisionalMapping(arm.calibration)
    output = args.output or alignment_lib.default_alignment_path(args.robot_id)
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(lines,), daemon=True).start()

    print("Motor calibration currently in EEPROM:")
    for joint in JOINTS:
        calibration = arm.calibration[joint]
        print(f"  {joint:14s} homing={calibration.homing_offset:6d} range={calibration.range_min}..{calibration.range_max}")
    print("\nTorque is off: support the arm by hand at every step.")
    print(LEVEL_SETUP)

    zeros: dict[str, float] = {}
    captures_log: dict[str, list[dict[str, float]]] = {}
    checks: dict[str, float] = {}
    steps = ZERO_STEPS
    if args.only:
        only = {joint.strip() for joint in args.only.split(",")}
        if unknown := only - set(JOINTS):
            raise SystemExit(f"Unknown joints in --only: {sorted(unknown)}")
        previous = alignment_lib.RealSimAlignment.load(output)
        if mismatches := previous.calibration_mismatches(arm.calibration):
            raise SystemExit("Motor calibration changed; re-measure everything:\n  " + "\n  ".join(mismatches))
        zeros = {joint: value.zero_tick for joint, value in previous.joints.items() if joint not in only}
        captures_log = {key: value for key, value in previous.notes.get("captures_ticks", {}).items()}
        steps = tuple(step for step in ZERO_STEPS if step.solves in only)
        args.skip_checks = True
    try:
        for step in steps:
            while True:
                captures = _run_step(arm, step, lines, zeros, provisional, args.repeats)
                if captures is not None:
                    break
                print(f"  {step.key} is required to solve {step.solves}; it cannot be skipped.")
            zeros[step.solves] = _solve_zero(step, captures, zeros)
            captures_log[step.key] = captures
        if not args.skip_checks:
            for step in CHECK_STEPS:
                captures = _run_step(arm, step, lines, zeros, provisional, 1)
                if captures is None:
                    continue
                captures_log[step.key] = captures
                checks[step.key], _ = _chain_degrees(step.chain, captures[0], zeros, provisional)
    except KeyboardInterrupt:
        print("\nStopped.")
        if len(zeros) < len(JOINTS):
            raise SystemExit("Not all joints were measured; nothing saved.")
    finally:
        arm.close()

    print("\n=== Result (zero tick = Present_Position at the USD zero pose)")
    for joint in JOINTS:
        shift = provisional.sim_degrees(joint, zeros[joint])
        print(f"  {joint:14s} zero_tick={zeros[joint]:8.1f}   teleop mapping put this pose at {shift:+7.2f} deg")
    for key, measured in checks.items():
        step = next(step for step in CHECK_STEPS if step.key == key)
        error = measured - step.target_deg
        verdict = "info" if step.key == "gripper_open" else ("ok" if abs(error) <= CHECK_TOLERANCE_DEG else "CHECK FAILED")
        print(f"  {key:16s} measured {measured:+8.2f} deg vs {step.target_deg:+7.2f}  ({error:+.2f})  {verdict}")

    alignment = alignment_lib.RealSimAlignment(
        robot_id=args.robot_id,
        joints={joint: alignment_lib.JointAlignment(zero_tick=round(zeros[joint], 2)) for joint in JOINTS},
        motor_calibration={
            joint: alignment_lib.MotorCalibrationSnapshot(
                homing_offset=arm.calibration[joint].homing_offset,
                range_min=arm.calibration[joint].range_min,
                range_max=arm.calibration[joint].range_max,
            )
            for joint in JOINTS
        },
        notes={"sim_jaw_touch_deg": SIM_JAW_TOUCH_DEG, "checks_deg": checks, "captures_ticks": captures_log},
    )
    if output.exists():
        backup = output.with_suffix(f".{time.strftime('%Y%m%d_%H%M%S')}.bak.json")
        output.rename(backup)
        print(f"Previous alignment moved to {backup}")
    alignment.save(output)
    print(f"Saved {output}")


def live(args) -> None:
    path = args.output or alignment_lib.default_alignment_path(args.robot_id)
    alignment = alignment_lib.RealSimAlignment.load(path)
    arm = ReadOnlyArm(args.port)
    try:
        if mismatches := alignment.calibration_mismatches(arm.calibration):
            raise SystemExit("Motor calibration changed since the alignment was measured:\n  " + "\n  ".join(mismatches))
        print(f"Sim joint angles (deg) under {path}; Ctrl-C to stop.")
        print("  ".join(f"{joint[:11]:>11s}" for joint in JOINTS))
        while True:
            ticks = arm.ticks()
            print(
                "\r" + "  ".join(f"{alignment.ticks_to_sim_degrees(joint, ticks[joint]):+11.2f}" for joint in JOINTS),
                end="",
                flush=True,
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        arm.close()


# Sim degrees (pan, lift, elbow, wrist_flex, roll, gripper). Lowest arm point stays >= 7.5 cm above the
# base bottom in the USD; every move goes through the high L pose.
VERIFY_POSES = {
    "L": (0, 0, 0, 0, 0, 20),
    "L_pan45_open": (45, 0, 0, 0, 0, 60),
    "reach": (0, 30, -30, 60, 0, 30),
    "left_roll90": (-40, 10, 0, 40, 90, 10),
    "right_roll-60": (60, 20, -20, 60, -60, 40),
}
VERIFY_MAX_SPEED_DEG_S = 20.0
VERIFY_CONTROL_HZ = 50.0
VERIFY_HOLD_S = 6.0
VERIFY_ABORT_ERROR_DEG = 12.0
VERIFY_ABORT_AFTER_S = 0.75


class _Abort(Exception):
    pass


def verify(args) -> None:
    """Torque ON: step the real arm slowly through sim poses, log tracking error and wrist frames."""

    import json
    import signal

    import cv2

    path = args.output or alignment_lib.default_alignment_path(args.robot_id)
    alignment = alignment_lib.RealSimAlignment.load(path)
    out_dir = REPO_ROOT / "outputs" / "real_sim_alignment" / f"verify_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    arm = ReadOnlyArm(args.port)
    bus = arm.bus
    if mismatches := alignment.calibration_mismatches(arm.calibration):
        arm.close()
        raise SystemExit("Motor calibration changed since the alignment was measured:\n  " + "\n  ".join(mismatches))
    camera = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def sim_now() -> dict[str, float]:
        ticks = arm.ticks()
        return {joint: alignment.ticks_to_sim_degrees(joint, ticks[joint]) for joint in JOINTS}

    def command(sim_deg: dict[str, float]) -> None:
        bus.sync_write(
            "Goal_Position",
            {joint: int(round(alignment.sim_degrees_to_ticks(joint, sim_deg[joint]))) for joint in JOINTS},
            normalize=False,
        )

    def move_to(target: dict[str, float]) -> None:
        start = sim_now()
        duration = max(abs(target[joint] - start[joint]) for joint in JOINTS) / VERIFY_MAX_SPEED_DEG_S
        steps = max(1, int(duration * VERIFY_CONTROL_HZ))
        over_since = None
        for step in range(1, steps + 1):
            s = step / steps
            s = s * s * (3.0 - 2.0 * s)
            goal = {joint: start[joint] + s * (target[joint] - start[joint]) for joint in JOINTS}
            command(goal)
            time.sleep(1.0 / VERIFY_CONTROL_HZ)
            present = sim_now()
            worst = max(abs(goal[joint] - present[joint]) for joint in JOINTS if joint != "gripper")
            if worst > VERIFY_ABORT_ERROR_DEG:
                over_since = over_since or time.monotonic()
                if time.monotonic() - over_since > VERIFY_ABORT_AFTER_S:
                    raise _Abort(f"tracking error {worst:.1f} deg (blocked?)")
            else:
                over_since = None

    def hold_and_log(name: str, target: dict[str, float]) -> dict[str, float]:
        time.sleep(VERIFY_HOLD_S - 1.0)
        samples = []
        for _ in range(10):
            samples.append(sim_now())
            time.sleep(0.1)
        measured = {joint: statistics.fmean(sample[joint] for sample in samples) for joint in JOINTS}
        error = {joint: measured[joint] - target[joint] for joint in JOINTS}
        for _ in range(5):
            camera.grab()
        ok, frame = camera.read()
        if ok:
            cv2.imwrite(str(out_dir / f"{name}_wrist.jpg"), frame)
        print(f"  {name:14s} " + "  ".join(f"{joint}={error[joint]:+5.1f}" for joint in JOINTS), flush=True)
        return error

    rest = {joint: value for joint, value in sim_now().items()}
    log: dict[str, dict] = {"alignment": str(path), "rest_sim_deg": rest, "poses": {}}
    try:
        print("Enabling torque at the current pose.", flush=True)
        command(rest)
        bus.enable_torque()
        print("Tracking error at each hold (measured - commanded, sim deg):", flush=True)
        for name, values in VERIFY_POSES.items():
            target = dict(zip(JOINTS, map(float, values)))
            if name != "L":
                move_to(dict(zip(JOINTS, map(float, VERIFY_POSES["L"]))))
            move_to(target)
            log["poses"][name] = {"target": target, "error": hold_and_log(name, target)}
        move_to(dict(zip(JOINTS, map(float, VERIFY_POSES["L"]))))
        move_to(rest)
        time.sleep(1.0)
        print("Back at rest.", flush=True)
    except (_Abort, KeyboardInterrupt) as exc:
        print(f"\nABORT: {exc or 'interrupted'}; holding where it is for 3 s, then torque off.", flush=True)
        command(sim_now())
        time.sleep(3.0)
        log["aborted"] = str(exc) or "interrupted"
    finally:
        bus.disable_torque()
        camera.release()
        arm.close()
        (out_dir / "verify_log.json").write_text(json.dumps(log, indent=2) + "\n")
        print(f"Torque off. Log and wrist frames in {out_dir}", flush=True)


def main() -> None:
    args = _parser().parse_args()
    {"measure": measure, "live": live, "verify": verify}[args.mode](args)


if __name__ == "__main__":
    main()
