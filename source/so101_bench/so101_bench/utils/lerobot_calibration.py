"""LeRobot calibration helpers for the real SO-101 used to collect the dataset."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MotorCalibration:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


STS3215_MAX_POSITION = 4095.0
STS3215_CENTER_POSITION = STS3215_MAX_POSITION / 2.0
STS3215_DEGREES_PER_TICK = 360.0 / STS3215_MAX_POSITION
SIM_LIMIT_MARGIN_DEG = 1.0e-3

LEROBOT_JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

LEROBOT_JOINT_FEATURE_ORDER = [f"{joint_name}.pos" for joint_name in LEROBOT_JOINT_ORDER]

LEROBOT_TO_USD_JOINT_NAMES = {
    "shoulder_pan": "Rotation",
    "shoulder_lift": "Pitch",
    "elbow_flex": "Elbow",
    "wrist_flex": "Wrist_Pitch",
    "wrist_roll": "Wrist_Roll",
    "gripper": "Jaw",
}

# These values were tuned according to the real robot (which had actual joint mins and maxes (i.e based on self-collision))
# These values are reflected in the updated SO-101 USD
# OLD values correspond to the original SO-101 USD
USD_SIM_JOINT_LIMITS_DEG = {
    "shoulder_pan": (-115.0, 122.0), # OLD: -110, 110
    "shoulder_lift": (-105.5, 101.9), # OLD: -100, 100
    "elbow_flex": (-102.0, 90.25), # OLD: -100.0, 90.0
    "wrist_flex": (-103.15, 101.25), # OLD: -95, 95
    "wrist_roll": (-168.0, 168.0), # OLD: -160, 160
    "gripper": (-12.75, 115.5), # OLD: -10.0, 100.0
}

LEROBOT_INITIAL_JOINT_POS = {
    "shoulder_pan": -6.5,
    "shoulder_lift": -98.8,
    "elbow_flex": 81.2,
    "wrist_flex": 71.2,
    "wrist_roll": -0.8,
    "gripper": 0.9,
}

REAL_SO101_CALIBRATION = {
    "shoulder_pan": MotorCalibration(id=1, drive_mode=0, homing_offset=-353, range_min=735, range_max=3443),
    "shoulder_lift": MotorCalibration(id=2, drive_mode=0, homing_offset=-1765, range_min=874, range_max=3230),
    "elbow_flex": MotorCalibration(id=3, drive_mode=0, homing_offset=-713, range_min=862, range_max=3077),
    "wrist_flex": MotorCalibration(id=4, drive_mode=0, homing_offset=-325, range_min=887, range_max=3178),
    "wrist_roll": MotorCalibration(id=5, drive_mode=0, homing_offset=481, range_min=187, range_max=4022),
    "gripper": MotorCalibration(id=6, drive_mode=0, homing_offset=339, range_min=1495, range_max=2981),
}


def lerobot_position_bounds(joint_name: str) -> tuple[float, float]:
    if joint_name == "gripper":
        return 0.0, 100.0
    return -100.0, 100.0


def lerobot_position_to_motor_position(joint_name: str, lerobot_position: float) -> float:
    """Convert a LeRobot `.pos` value into calibrated Feetech present-position ticks."""

    calibration = REAL_SO101_CALIBRATION[joint_name]
    lower, upper = lerobot_position_bounds(joint_name)

    if joint_name == "gripper":
        value = 100.0 - lerobot_position if calibration.drive_mode else lerobot_position
        bounded_value = min(upper, max(lower, value))
        normalized = bounded_value / 100.0
    else:
        value = -lerobot_position if calibration.drive_mode else lerobot_position
        bounded_value = min(upper, max(lower, value))
        normalized = (bounded_value + 100.0) / 200.0

    return normalized * (calibration.range_max - calibration.range_min) + calibration.range_min


def motor_position_to_lerobot_position(joint_name: str, motor_position: float) -> float:
    """Convert calibrated Feetech present-position ticks into a LeRobot `.pos` value."""

    calibration = REAL_SO101_CALIBRATION[joint_name]
    bounded_position = min(calibration.range_max, max(calibration.range_min, motor_position))
    normalized = (bounded_position - calibration.range_min) / (calibration.range_max - calibration.range_min)

    if joint_name == "gripper":
        lerobot_position = normalized * 100.0
        return 100.0 - lerobot_position if calibration.drive_mode else lerobot_position

    lerobot_position = normalized * 200.0 - 100.0
    return -lerobot_position if calibration.drive_mode else lerobot_position


def motor_position_to_sim_degrees(motor_position: float) -> float:
    return (motor_position - STS3215_CENTER_POSITION) * STS3215_DEGREES_PER_TICK


def sim_degrees_to_motor_position(sim_degrees: float) -> float:
    return sim_degrees / STS3215_DEGREES_PER_TICK + STS3215_CENTER_POSITION


def clamp_to_usd_joint_limits_degrees(joint_name: str, sim_degrees: float) -> float:
    lower, upper = USD_SIM_JOINT_LIMITS_DEG[joint_name]
    return min(upper - SIM_LIMIT_MARGIN_DEG, max(lower + SIM_LIMIT_MARGIN_DEG, sim_degrees))


def lerobot_position_to_sim_degrees(joint_name: str, lerobot_position: float) -> float:
    if joint_name == "gripper":
        lower, upper = lerobot_position_bounds(joint_name)
        bounded_position = min(upper, max(lower, lerobot_position))
        jaw_lower, jaw_upper = USD_SIM_JOINT_LIMITS_DEG[joint_name]
        jaw_degrees = jaw_lower + (bounded_position / 100.0) * (jaw_upper - jaw_lower)
        return clamp_to_usd_joint_limits_degrees(joint_name, jaw_degrees)

    motor_position = lerobot_position_to_motor_position(joint_name, lerobot_position)
    return clamp_to_usd_joint_limits_degrees(joint_name, motor_position_to_sim_degrees(motor_position))


def sim_degrees_to_lerobot_position(joint_name: str, sim_degrees: float) -> float:
    bounded_degrees = clamp_to_usd_joint_limits_degrees(joint_name, sim_degrees)
    if joint_name == "gripper":
        jaw_lower, jaw_upper = USD_SIM_JOINT_LIMITS_DEG[joint_name]
        normalized = (bounded_degrees - jaw_lower) / (jaw_upper - jaw_lower)
        return min(100.0, max(0.0, normalized * 100.0))

    motor_position = sim_degrees_to_motor_position(bounded_degrees)
    return motor_position_to_lerobot_position(joint_name, motor_position)


def lerobot_position_to_sim_radians(joint_name: str, lerobot_position: float) -> float:
    return math.radians(lerobot_position_to_sim_degrees(joint_name, lerobot_position))


def lerobot_pose_to_sim_joint_pos(lerobot_joint_pos: dict[str, float]) -> dict[str, float]:
    return {
        LEROBOT_TO_USD_JOINT_NAMES[joint_name]: lerobot_position_to_sim_radians(joint_name, lerobot_position)
        for joint_name, lerobot_position in lerobot_joint_pos.items()
    }


def lerobot_joint_sim_limits_degrees(joint_name: str) -> tuple[float, float]:
    lower, upper = lerobot_position_bounds(joint_name)
    return lerobot_position_to_sim_degrees(joint_name, lower), lerobot_position_to_sim_degrees(joint_name, upper)
