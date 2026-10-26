"""Headless physics-only check: do objects pop out of the bin when one is dropped on a pile?

Settles three objects inside the plastic bin, then drops a fourth onto the pile at a
realistic release speed and reports each object's max height and max speed afterwards.
Run with --ccd 1 to author physxRigidBody:enableSpeculativeCCD on all bodies before
simulating, to compare against the --ccd 0 baseline.
No cameras -> avoids the headless rendering-kit crash.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ccd", type=int, default=0, help="1 = enable speculative CCD on all bodies")
parser.add_argument("--drop_height", type=float, default=0.25, help="release height above ground (m)")
parser.add_argument("--drop_speed", type=float, default=1.0, help="initial downward speed (m/s)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from pxr import Sdf, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg

from so101_bench.tasks.direct.so101_bench.so101_bench_env_cfg import (
    ASSETS_PATH,
    CONTACT_OFFSET,
    CONTACT_SOLVER_POSITION_ITERATIONS,
    CONTACT_SOLVER_VELOCITY_ITERATIONS,
    MAX_BIN_ANGULAR_VELOCITY,
    MAX_BIN_LINEAR_VELOCITY,
    MAX_DEPENETRATION_VELOCITY,
    MAX_OBJECT_ANGULAR_VELOCITY,
    MAX_OBJECT_LINEAR_VELOCITY,
    PHYSICS_DT,
    REST_OFFSET,
)

BIN_TOP_Z = 0.098
SETTLE_SECONDS = 2.0
DROP_SECONDS = 3.0
PILE = [
    ("black_tape", (0.05, 0.0, 0.04)),
    ("altoids_container", (-0.05, 0.03, 0.04)),
    ("grey_toy_car", (-0.04, -0.04, 0.04)),
]
DROPPER = ("red_pen", (0.0, 0.0))


def _rigid_props(max_lin: float, max_ang: float) -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=CONTACT_SOLVER_POSITION_ITERATIONS,
        solver_velocity_iteration_count=CONTACT_SOLVER_VELOCITY_ITERATIONS,
        max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
        max_linear_velocity=max_lin,
        max_angular_velocity=max_ang,
    )


def _collision_props() -> sim_utils.CollisionPropertiesCfg:
    return sim_utils.CollisionPropertiesCfg(
        collision_enabled=True, contact_offset=CONTACT_OFFSET, rest_offset=REST_OFFSET
    )


def _object_cfg(prim_path: str, usd_name: str, pos: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSETS_PATH}/usd/objects/{usd_name}.usdc",
            rigid_props=_rigid_props(MAX_OBJECT_LINEAR_VELOCITY, MAX_OBJECT_ANGULAR_VELOCITY),
            collision_props=_collision_props(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def _enable_speculative_ccd() -> None:
    stage = sim_utils.get_current_stage()
    count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.CreateAttribute("physxRigidBody:enableSpeculativeCCD", Sdf.ValueTypeNames.Bool).Set(True)
            count += 1
    print(f"[test] speculative CCD enabled on {count} rigid bodies")


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.spawn_light("/World/light", sim_utils.DomeLightCfg(intensity=1000.0))

    bin_obj = RigidObject(
        RigidObjectCfg(
            prim_path="/World/PlasticBin",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ASSETS_PATH}/usd/plastic_bin.usdc",
                rigid_props=_rigid_props(MAX_BIN_LINEAR_VELOCITY, MAX_BIN_ANGULAR_VELOCITY),
                collision_props=_collision_props(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.005)),
        )
    )
    objects = {
        name: RigidObject(_object_cfg(f"/World/Pile_{name}", name, pos)) for name, pos in PILE
    }
    drop_name, (drop_x, drop_y) = DROPPER
    dropper = RigidObject(
        _object_cfg(f"/World/Drop_{drop_name}", drop_name, (drop_x, drop_y, args_cli.drop_height))
    )

    if args_cli.ccd:
        _enable_speculative_ccd()

    sim.reset()
    all_objects = {**objects, drop_name: dropper, "bin": bin_obj}

    # hold the dropper in place while the pile settles
    hold_pose = dropper.data.root_state_w.clone()
    hold_pose[:, 7:] = 0.0
    for _ in range(int(SETTLE_SECONDS / PHYSICS_DT)):
        dropper.write_root_state_to_sim(hold_pose)
        sim.step()
        for obj in all_objects.values():
            obj.update(PHYSICS_DT)

    # release with a downward push, as if let go mid-move
    release = hold_pose.clone()
    release[:, 9] = -args_cli.drop_speed
    dropper.write_root_state_to_sim(release)

    max_z = {name: 0.0 for name in all_objects}
    max_speed = {name: 0.0 for name in all_objects}
    for _ in range(int(DROP_SECONDS / PHYSICS_DT)):
        sim.step()
        for name, obj in all_objects.items():
            obj.update(PHYSICS_DT)
            max_z[name] = max(max_z[name], obj.data.root_pos_w[0, 2].item())
            max_speed[name] = max(
                max_speed[name], torch.linalg.norm(obj.data.root_lin_vel_w[0]).item()
            )

    print(f"\n[test] ccd={bool(args_cli.ccd)} drop={drop_name} from {args_cli.drop_height:.2f}m "
          f"at {args_cli.drop_speed:.1f}m/s | bin top ~{BIN_TOP_Z:.3f}m")
    popped = False
    for name in all_objects:
        flag = ""
        if name not in ("bin", drop_name) and max_z[name] > BIN_TOP_Z + 0.03:
            flag = "  <-- POPPED OUT"
            popped = True
        print(f"  {name:18s} max_z={max_z[name]*100:6.2f} cm   max|v|={max_speed[name]:5.2f} m/s{flag}")
    final_z = {n: o.data.root_pos_w[0, 2].item() for n, o in all_objects.items()}
    inside = all(final_z[n] < BIN_TOP_Z for n in objects) and final_z[drop_name] < BIN_TOP_Z
    print(f"[test] pile popped: {popped} | everything final-resting inside bin: {inside}")


main()
simulation_app.close()
