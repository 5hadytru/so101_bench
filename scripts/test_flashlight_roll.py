"""Headless physics-only check: does the yellow flashlight roll after settling?

Drops the object onto a ground plane at the benchmark spawn clearance and reports
how far its root travels and how much it spins, for a few angular-damping values.
No cameras -> avoids the headless rendering-kit crash.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ang_damp", type=float, default=0.0)
parser.add_argument("--lin_damp", type=float, default=0.0)
parser.add_argument("--seconds", type=float, default=4.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext

from so101_bench.tasks.direct.so101_bench.so101_bench_env_cfg import (
    ASSETS_PATH,
    CONTACT_OFFSET,
    REST_OFFSET,
    CONTACT_SOLVER_POSITION_ITERATIONS,
    CONTACT_SOLVER_VELOCITY_ITERATIONS,
    MAX_DEPENETRATION_VELOCITY,
    MAX_OBJECT_ANGULAR_VELOCITY,
    MAX_OBJECT_LINEAR_VELOCITY,
    TABLE_OBJECT_Z,
)

USD = f"{ASSETS_PATH}/usd/objects/yellow_flashlight.usdc"


def run_one(angular_damping: float, linear_damping: float, seconds: float = 4.0) -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 240.0, device=args_cli.device))
    # ground
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.spawn_light("/World/light", sim_utils.DomeLightCfg(intensity=1000.0))

    cfg = RigidObjectCfg(
        prim_path="/World/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=CONTACT_SOLVER_POSITION_ITERATIONS,
                solver_velocity_iteration_count=CONTACT_SOLVER_VELOCITY_ITERATIONS,
                max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
                max_linear_velocity=MAX_OBJECT_LINEAR_VELOCITY,
                max_angular_velocity=MAX_OBJECT_ANGULAR_VELOCITY,
                angular_damping=angular_damping,
                linear_damping=linear_damping,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=CONTACT_OFFSET, rest_offset=REST_OFFSET
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_OBJECT_Z)),
    )
    obj = RigidObject(cfg)
    sim.reset()

    start = obj.data.root_pos_w.clone()
    steps = int(seconds * 240)
    max_xy = 0.0
    for _ in range(steps):
        sim.step()
        obj.update(1.0 / 240.0)
        d = torch.linalg.norm(obj.data.root_pos_w[0, :2] - start[0, :2]).item()
        max_xy = max(max_xy, d)
    end = obj.data.root_pos_w.clone()
    travel = torch.linalg.norm(end[0, :2] - start[0, :2]).item()
    ang = torch.linalg.norm(obj.data.root_ang_vel_w[0]).item()
    print(
        f"ang_damp={angular_damping:5.1f} lin_damp={linear_damping:4.1f} | "
        f"xy_travel={travel*100:6.2f} cm  max_xy={max_xy*100:6.2f} cm  "
        f"final|w|={ang:5.2f} rad/s"
    )

run_one(args_cli.ang_damp, args_cli.lin_damp, args_cli.seconds)
simulation_app.close()
