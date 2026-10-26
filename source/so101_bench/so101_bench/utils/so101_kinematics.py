"""Numpy forward/inverse kinematics for the SO-101 arm in ``SO-ARM101-USD.usd``.

The joint frames are copied from the USD revolute joints (``physics:localPos0/1`` and
``physics:localRot0/1``, all axes +Z), so FK is expressed in the ``base`` rigid-body
frame. Callers supply the world pose of that body from the simulator.

Plain Python on purpose: importing ``so101_bench`` pulls in Isaac Lab, so tools that
only need geometry load this file by path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares

ARM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")

# (parent body, child body, localPos0, localRot0 wxyz, localPos1, localRot1 wxyz, limits deg)
_USD_JOINTS = {
    "Rotation": (
        (0.0207909, -0.0230745, 0.0948817),
        (-8.963205e-7, 0.7071055, 0.7071081, -8.963237e-7),
        (3.469447e-18, 3.469447e-18, 0.0),
        (1.0, -3.677072e-14, 3.8411286e-17, 4.588883e-14),
        (-115.0, 122.0),
    ),
    "Pitch": (
        (-0.0303992, -0.018277798, -0.05420001),
        (-0.49999818, 0.49999997, 0.5, 0.5000018),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (-105.5, 101.9000015258789),
    ),
    "Elbow": (
        (-0.11257002, -0.028000006, -3.6395834e-11),
        (0.7071056, 3.8406025e-8, -3.000745e-9, 0.707108),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (-102.0, 90.25),
    ),
    "Wrist_Pitch": (
        (-0.13490003, 0.00519998, -2.2649212e-8),
        (-0.7071055, 1.0536122e-7, 4.9158423e-8, 0.707108),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (-103.1500015258789, 101.25),
    ),
    "Wrist_Roll": (
        (4.4612474e-9, -0.061100032, 0.018100007),
        (6.152631e-7, 0.0000010294956, 0.7071081, 0.70710546),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (-168.0, 168.0),
    ),
}
JAW_LIMITS_DEG = (-12.75, 115.5)

# Gripper-body frame: the fingers point along -Z and the moving jaw opens along +X.
# Measured on the collision meshes: the fixed finger's inner face is at x = -0.0079 and
# its tip at z = -0.1044. The TCP sits 8 mm off that face, 12.4 mm above the tip.
FIXED_FINGER_FACE_X = -0.0079
TCP_IN_GRIPPER = np.array([0.0001, 0.0, -0.0920])
FINGERTIP_IN_GRIPPER = np.array([0.0001, 0.0, -0.1060])

# Jaw angle (deg) -> gap between the finger pads (m), from the same meshes.
_JAW_GAP_TABLE = (
    (-12.0, -0.0008), (-8.0, 0.0046), (-4.0, 0.0102), (0.0, 0.0158), (4.0, 0.0209),
    (8.0, 0.0259), (12.0, 0.0309), (16.0, 0.0353), (20.0, 0.0396), (24.0, 0.0439),
    (28.0, 0.0481), (32.0, 0.0522), (36.0, 0.0577), (40.0, 0.0628), (44.0, 0.0688),
    (48.0, 0.0734), (52.0, 0.0809),
)
MAX_GRIPPER_OPENING_M = _JAW_GAP_TABLE[-1][1]
# A zero opening command squeezes: the target sits below the angle where the pads meet.
JAW_SQUEEZE_DEG = -12.5


def _quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _transform(position, rotation: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


_JOINT_FRAMES = {
    name: (
        _transform(p0, _quat_wxyz_to_matrix(r0)),
        np.linalg.inv(_transform(p1, _quat_wxyz_to_matrix(r1))),
    )
    for name, (p0, r0, p1, r1, _limits) in _USD_JOINTS.items()
}
# The moving jaw's joint frame in the gripper body (USD ``joints/Jaw``; its localPos1/Rot1 are identity).
_JAW_FRAME = _transform((0.020200018, 0.018800007, -0.02340001), _quat_wxyz_to_matrix((0.7071055, 0.707108, 6.31823e-8, 1.6858581e-7)))
ARM_LIMITS_RAD = np.radians(np.array([_USD_JOINTS[name][4] for name in ARM_JOINTS]))


def link_poses_in_base(arm_q) -> list[np.ndarray]:
    """4x4 poses of ``shoulder``, ``upper_arm``, ``lower_arm``, ``wrist`` and ``gripper`` in the
    ``base`` body frame."""

    result = np.eye(4)
    poses = []
    for name, angle in zip(ARM_JOINTS, arm_q, strict=True):
        parent_frame, child_frame_inv = _JOINT_FRAMES[name]
        result = result @ parent_frame @ _transform((0.0, 0.0, 0.0), _rot_z(float(angle))) @ child_frame_inv
        poses.append(result)
    return poses


def gripper_pose_in_base(arm_q) -> np.ndarray:
    """4x4 pose of the ``gripper`` body in the ``base`` body frame."""

    return link_poses_in_base(arm_q)[-1]


def jaw_pose_in_gripper(jaw_rad: float) -> np.ndarray:
    """4x4 pose of the ``jaw`` body (the moving finger) in the ``gripper`` body frame."""

    return _JAW_FRAME @ _transform((0.0, 0.0, 0.0), _rot_z(float(jaw_rad)))


@dataclass(frozen=True)
class ToolPose:
    tcp: np.ndarray
    fingertip: np.ndarray
    tool_axis: np.ndarray
    jaw_axis: np.ndarray
    pitch_deg: float
    jaw_yaw_deg: float | None


def tool_pose(world_from_base: np.ndarray, arm_q) -> ToolPose:
    world_from_gripper = world_from_base @ gripper_pose_in_base(arm_q)
    rotation = world_from_gripper[:3, :3]
    tool_axis = -rotation[:, 2]
    jaw_axis = rotation[:, 0]
    pitch = math.degrees(math.acos(float(np.clip(-tool_axis[2], -1.0, 1.0))))
    horizontal = math.hypot(jaw_axis[0], jaw_axis[1])
    yaw = math.degrees(math.atan2(jaw_axis[1], jaw_axis[0])) if horizontal > 0.3 else None
    return ToolPose(
        tcp=world_from_gripper[:3, :3] @ TCP_IN_GRIPPER + world_from_gripper[:3, 3],
        fingertip=world_from_gripper[:3, :3] @ FINGERTIP_IN_GRIPPER + world_from_gripper[:3, 3],
        tool_axis=tool_axis,
        jaw_axis=jaw_axis,
        pitch_deg=pitch,
        jaw_yaw_deg=yaw,
    )


def _wrap_half_turn(angle_deg: float) -> float:
    """Wrap to [-90, 90): a jaw heading and its reverse close along the same line."""

    return (angle_deg + 90.0) % 180.0 - 90.0


@dataclass(frozen=True)
class IKResult:
    arm_q: np.ndarray
    position_error_m: float
    pitch_error_deg: float | None
    jaw_yaw_error_deg: float | None
    achieved: ToolPose


def solve_ik(
    world_from_base: np.ndarray,
    tcp_target,
    seed_q,
    *,
    pitch_deg: float | None = None,
    jaw_yaw_deg: float | None = None,
    limit_margin_deg: float = 2.0,
) -> IKResult:
    """Damped least squares over the five arm joints, seeded from the current target.

    Position is weighted per millimetre and angles per two degrees. ``None`` leaves that
    angle free. The jaw heading is compared modulo 180 degrees.
    """

    target = np.asarray(tcp_target, dtype=float)
    margin = math.radians(limit_margin_deg)
    lower, upper = ARM_LIMITS_RAD[:, 0] + margin, ARM_LIMITS_RAD[:, 1] - margin
    seed = np.clip(np.asarray(seed_q, dtype=float), lower, upper)

    def residuals(q):
        pose = tool_pose(world_from_base, q)
        values = list((pose.tcp - target) * 1000.0)
        if pitch_deg is not None:
            values.append((pose.pitch_deg - pitch_deg) / 2.0)
        if jaw_yaw_deg is not None:
            values.append(0.0 if pose.jaw_yaw_deg is None else _wrap_half_turn(pose.jaw_yaw_deg - jaw_yaw_deg) / 2.0)
        values.extend(np.degrees(q - seed) * 0.01)
        return np.asarray(values)

    seeds = [seed]
    for roll_offset in (math.pi, -math.pi):
        flipped = seed.copy()
        flipped[4] += roll_offset
        if lower[4] <= flipped[4] <= upper[4]:
            seeds.append(flipped)
    best = None
    for start in seeds:
        solution = least_squares(residuals, start, bounds=(lower, upper), method="trf", max_nfev=200)
        if best is None or solution.cost < best.cost:
            best = solution
    pose = tool_pose(world_from_base, best.x)
    return IKResult(
        arm_q=best.x,
        position_error_m=float(np.linalg.norm(pose.tcp - target)),
        pitch_error_deg=None if pitch_deg is None else abs(pose.pitch_deg - pitch_deg),
        jaw_yaw_error_deg=(
            None
            if jaw_yaw_deg is None
            else (90.0 if pose.jaw_yaw_deg is None else abs(_wrap_half_turn(pose.jaw_yaw_deg - jaw_yaw_deg)))
        ),
        achieved=pose,
    )


def jaw_angle_for_opening(opening_m: float) -> float:
    """Jaw joint target (rad) for a finger-pad gap; zero or less squeezes shut."""

    if opening_m <= 0.0:
        return math.radians(JAW_SQUEEZE_DEG)
    angles, gaps = zip(*_JAW_GAP_TABLE)
    return math.radians(float(np.interp(min(opening_m, MAX_GRIPPER_OPENING_M), gaps, angles)))


def opening_for_jaw_angle(jaw_rad: float) -> float:
    angles, gaps = zip(*_JAW_GAP_TABLE)
    return max(0.0, float(np.interp(math.degrees(jaw_rad), angles, gaps)))
