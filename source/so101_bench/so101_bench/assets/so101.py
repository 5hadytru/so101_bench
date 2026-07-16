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
    actuators={
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"],
            effort_limit_sim=30.0,
            velocity_limit_sim=SO101_VELOCITY_LIMITS_SIM["Rotation"],
            stiffness=55.0,
            damping=0.7,
        ),
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"],
            effort_limit_sim=30.0,
            velocity_limit_sim=SO101_VELOCITY_LIMITS_SIM["Pitch"],
            stiffness=30.0,
            damping=0.8,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"],
            effort_limit_sim=30.0,
            velocity_limit_sim=SO101_VELOCITY_LIMITS_SIM["Elbow"],
            stiffness=25.0,
            damping=0.7,
        ),
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"],
            effort_limit_sim=30.0,
            velocity_limit_sim=SO101_VELOCITY_LIMITS_SIM["Wrist_Pitch"],
            stiffness=12.0,
            damping=0.5,
        ),
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"],
            effort_limit_sim=30.0,
            velocity_limit_sim=SO101_VELOCITY_LIMITS_SIM["Wrist_Roll"],
            stiffness=7.0,
            damping=0.5,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["Jaw"],
            effort_limit_sim=15.0,
            velocity_limit_sim=SO101_VELOCITY_LIMITS_SIM["Jaw"],
            stiffness=2.0,
            damping=0.3,
        ),
    },
)
