"""Map SO-101 sim joint angles to the real follower's raw motor ticks, and back.

The dataset/policy `.pos` space is defined by `lerobot_calibration`: it is sim
joint space under a fixed bijection, independent of how the real arm happens to
be calibrated. This module is only for driving real hardware. Each joint is

    ticks = zero_tick + sign * sim_degrees * 4096 / 360

where `zero_tick` is the Present_Position (in the motor's current homing-offset
frame) at which the real joint matches the USD zero pose. The STS3215 encoder is
4096 ticks/turn on the output shaft, so only the zero is measured.

`scripts/so101_sim_alignment.py measure` writes the JSON this module loads.

Kept free of torch/Isaac imports so hardware scripts can load it by path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any


STS3215_TICKS_PER_TURN = 4096
TICKS_PER_DEGREE = STS3215_TICKS_PER_TURN / 360.0

JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

DEFAULT_ALIGNMENT_DIR = Path(__file__).resolve().parents[1] / "assets" / "real_robot"


def default_alignment_path(robot_id: str) -> Path:
    return DEFAULT_ALIGNMENT_DIR / f"{robot_id}_sim_alignment.json"


@dataclass(frozen=True)
class JointAlignment:
    zero_tick: float
    sign: int = 1


@dataclass(frozen=True)
class MotorCalibrationSnapshot:
    """The motor EEPROM calibration the zero ticks were measured against."""

    homing_offset: int
    range_min: int
    range_max: int


@dataclass
class RealSimAlignment:
    robot_id: str
    joints: dict[str, JointAlignment]
    motor_calibration: dict[str, MotorCalibrationSnapshot]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    notes: dict[str, Any] = field(default_factory=dict)

    def sim_degrees_to_ticks(self, joint: str, sim_degrees: float, *, clamp: bool = True) -> float:
        alignment = self.joints[joint]
        ticks = alignment.zero_tick + alignment.sign * sim_degrees * TICKS_PER_DEGREE
        if clamp:
            calibration = self.motor_calibration[joint]
            ticks = min(calibration.range_max, max(calibration.range_min, ticks))
        return ticks

    def ticks_to_sim_degrees(self, joint: str, ticks: float) -> float:
        alignment = self.joints[joint]
        return alignment.sign * (ticks - alignment.zero_tick) / TICKS_PER_DEGREE

    def sim_radians_to_ticks(self, sim_radians: dict[str, float], *, clamp: bool = True) -> dict[str, int]:
        return {
            joint: int(round(self.sim_degrees_to_ticks(joint, math.degrees(value), clamp=clamp)))
            for joint, value in sim_radians.items()
        }

    def ticks_to_sim_radians(self, ticks: dict[str, float]) -> dict[str, float]:
        return {joint: math.radians(self.ticks_to_sim_degrees(joint, value)) for joint, value in ticks.items()}

    def calibration_mismatches(self, motor_calibration: dict[str, Any]) -> list[str]:
        """Compare against `FeetechMotorsBus.read_calibration()`; non-empty means re-measure."""

        mismatches = []
        for joint, expected in self.motor_calibration.items():
            actual = motor_calibration[joint]
            if actual.homing_offset != expected.homing_offset:
                mismatches.append(
                    f"{joint}: motor homing_offset={actual.homing_offset}, alignment measured with "
                    f"{expected.homing_offset}"
                )
        return mismatches

    def to_json(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "created_at": self.created_at,
            "ticks_per_degree": TICKS_PER_DEGREE,
            "joints": {joint: asdict(self.joints[joint]) for joint in JOINT_ORDER if joint in self.joints},
            "motor_calibration": {
                joint: asdict(self.motor_calibration[joint])
                for joint in JOINT_ORDER
                if joint in self.motor_calibration
            },
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RealSimAlignment:
        return cls(
            robot_id=data["robot_id"],
            created_at=data["created_at"],
            joints={joint: JointAlignment(**values) for joint, values in data["joints"].items()},
            motor_calibration={
                joint: MotorCalibrationSnapshot(**values) for joint, values in data["motor_calibration"].items()
            },
            notes=data.get("notes", {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> RealSimAlignment:
        return cls.from_json(json.loads(Path(path).read_text()))
