# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This file is adapted from the public NVIDIA Isaac Sim SO-101 workshop:
# https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop

from __future__ import annotations

import os

import numpy as np
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaaclab.assets.articulation import ArticulationCfg

from so101_bench.utils.lerobot_calibration import (
    LEROBOT_INITIAL_JOINT_POS,
    lerobot_pose_to_sim_joint_pos,
)

ASSET_DIR = os.path.dirname(os.path.abspath(__file__))

# Light SO-101 joint speed caps in rad/s. These leave normal teleop responsive
# while preventing extreme target jumps from producing unrealistic tracking.
SO101_VELOCITY_LIMITS_SIM = {
    "Rotation": 3.0,
    "Pitch": 3.3,
    "Elbow": 3.1,
    "Wrist_Pitch": 2.9,
    "Wrist_Roll": 4.2,
    "Jaw": 3.2,
}


def euler_angles_to_quat(euler_angles: np.ndarray, degrees: bool = False) -> np.ndarray:
    quat_xyzw = Rotation.from_euler("xyz", euler_angles, degrees=degrees).as_quat()
    return quat_xyzw[[3, 0, 1, 2]]

SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/usd/SO-ARM101-USD.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=lerobot_pose_to_sim_joint_pos(LEROBOT_INITIAL_JOINT_POS),
        pos=(0.05209, 0.18061, -0.03102),
        rot=euler_angles_to_quat(np.array([0.0, 0.0, 6.195]), degrees=True),
    ),
    # One actuator group with per-joint gains. PhysX receives exactly the same drive
    # parameters as six single-joint groups would, but Isaac Lab's per-substep
    # ``_apply_actuator_model`` loop runs once instead of six times.
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
            effort_limit_sim={
                "Rotation": 30.0,
                "Pitch": 30.0,
                "Elbow": 30.0,
                "Wrist_Pitch": 30.0,
                "Wrist_Roll": 30.0,
                "Jaw": 15.0,
            },
            velocity_limit_sim=dict(SO101_VELOCITY_LIMITS_SIM),
            stiffness={
                "Rotation": 55.0,
                "Pitch": 30.0,
                "Elbow": 25.0,
                "Wrist_Pitch": 12.0,
                "Wrist_Roll": 7.0,
                "Jaw": 2.0,
            },
            damping={
                "Rotation": 0.7,
                "Pitch": 0.8,
                "Elbow": 0.7,
                "Wrist_Pitch": 0.5,
                "Wrist_Roll": 0.5,
                "Jaw": 0.3,
            },
        ),
    },
)
