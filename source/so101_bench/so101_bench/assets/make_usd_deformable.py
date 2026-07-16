"""Apply deformable-body physics presets to SO-101 Bench object USDs.

Run with:

    TERM=xterm env -u CONDA_PREFIX VIRTUAL_ENV=/home/truman/env_isaaclab_51 \
        PATH=/home/truman/env_isaaclab_51/bin:$PATH \
        /home/truman/IsaacLab/isaaclab.sh -p \
        /home/truman/so101_bench/source/so101_bench/so101_bench/assets/make_usd_deformable.py \
        --preset stuffed_animal

The script expects a single mesh in the USD, which is the shape Isaac/PhysX uses
to cook the deformable simulation and collision meshes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


ASSETS_DIR = Path(__file__).resolve().parent
OBJECT_USD_DIR = ASSETS_DIR / "usd" / "objects"
UNSUPPORTED_DEFORMABLE_ATTRIBUTES = ("physxDeformable:maxDepenetrationVelocity",)


@dataclass(frozen=True)
class DeformablePreset:
    default_usd_name: str
    material_name: str
    default_deformable_root_path: str | None
    body: dict[str, object]
    material: dict[str, object]


PRESETS: dict[str, DeformablePreset] = {
    "stuffed_animal": DeformablePreset(
        default_usd_name="brown_stuffed_animal.usdc",
        material_name="stuffed_animal_deformable_physics",
        default_deformable_root_path=None,
        body={
            "deformable_enabled": True,
            "rest_offset": 0.002,
            "contact_offset": 0.01,
            "solver_position_iteration_count": 112,
            "vertex_velocity_damping": 0.28,
            "simulation_hexahedral_resolution": 10,
            "collision_simplification": True,
            "collision_simplification_remeshing": True,
            "collision_simplification_remeshing_resolution": 24,
            "collision_simplification_target_triangle_count": 700,
            "collision_simplification_force_conforming": True,
            "self_collision": False,
        },
        material={
            "density": 35.0,
            "dynamic_friction": 70.0,
            "youngs_modulus": 5.0e9,
            "poissons_ratio": 0.45,
            "elasticity_damping": 1.0,
            "damping_scale": 1.0,
        },
    ),
    "sponge": DeformablePreset(
        default_usd_name="sponge.usdc",
        material_name="sponge_deformable_physics",
        default_deformable_root_path=None,
        body={
            "deformable_enabled": True,
            "rest_offset": 0.0,
            "contact_offset": 0.003,
            "solver_position_iteration_count": 112,
            "vertex_velocity_damping": 0.14,
            "simulation_hexahedral_resolution": 18,
            "collision_simplification": True,
            "collision_simplification_remeshing": True,
            "collision_simplification_remeshing_resolution": 36,
            "collision_simplification_target_triangle_count": 1400,
            "collision_simplification_force_conforming": True,
            "self_collision": False,
        },
        material={
            "density": 28.0,
            "dynamic_friction": 10.0,
            "youngs_modulus": 8.0e6,
            "poissons_ratio": 0.22,
            "elasticity_damping": 0.18,
            "damping_scale": 1.0,
        },
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        required=True,
        choices=sorted(PRESETS),
        help="Deformable physics preset to apply.",
    )
    parser.add_argument(
        "--usd-path",
        type=Path,
        default=None,
        help="USD file to modify. Defaults to the preset's canonical object USD.",
    )
    parser.add_argument(
        "--deformable-root-path",
        default=None,
        help="Prim path whose single descendant Mesh should receive deformable-body schemas.",
    )
    return parser.parse_args()


args_cli = parse_args()

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas import define_deformable_body_properties
from isaaclab.sim.utils.stage import get_current_stage, open_stage, update_stage
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade


def log(message: str) -> None:
    print(message, flush=True)


def usd_path_for_args() -> Path:
    preset = PRESETS[args_cli.preset]
    usd_path = args_cli.usd_path or OBJECT_USD_DIR / preset.default_usd_name
    return usd_path.expanduser().resolve()


def single_mesh_prim(stage: Usd.Stage, root_path: str | None = None) -> Usd.Prim:
    root_prim = stage.GetPrimAtPath(root_path) if root_path else stage.GetPseudoRoot()
    if not root_prim.IsValid():
        raise RuntimeError(f"Deformable root path is invalid: {root_path}")

    mesh_prims = [prim for prim in Usd.PrimRange(root_prim) if prim.IsA(UsdGeom.Mesh)]
    log("Mesh prims:")
    for prim in mesh_prims:
        log(f"  {prim.GetPath()}")

    if len(mesh_prims) != 1:
        raise RuntimeError(
            f"Expected exactly one Mesh prim, found {len(mesh_prims)}. "
            "Join the object into one mesh in Blender, triangulate it, apply transforms, and export again."
        )
    return mesh_prims[0]


def infer_deformable_root_path(mesh_prim: Usd.Prim) -> str:
    parent_path = mesh_prim.GetPath().GetParentPath()
    if parent_path.isEmpty or parent_path.pathString == "/":
        return mesh_prim.GetPath().pathString
    return parent_path.pathString


def material_path(stage: Usd.Stage, preset: DeformablePreset) -> str:
    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        root_path = default_prim.GetPath().pathString
    else:
        root_prims = [prim for prim in stage.GetPseudoRoot().GetChildren() if prim.IsValid()]
        if not root_prims:
            raise RuntimeError("USD has no root prims; cannot place a physics material.")
        root_path = root_prims[0].GetPath().pathString
    return f"{root_path}/_materials/{preset.material_name}"


def remove_rigid_body_schemas(stage: Usd.Stage) -> None:
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
        if prim.HasAPI(UsdPhysics.MassAPI):
            prim.RemoveAPI(UsdPhysics.MassAPI)


def clear_unsupported_deformable_attributes(mesh_prim: Usd.Prim) -> None:
    for attribute_name in UNSUPPORTED_DEFORMABLE_ATTRIBUTES:
        attribute = mesh_prim.GetAttribute(attribute_name)
        if attribute.IsValid() and attribute.HasAuthoredValueOpinion():
            attribute.Clear()
        if mesh_prim.HasProperty(attribute_name):
            mesh_prim.RemoveProperty(attribute_name)


def bind_physics_material(mesh_prim: Usd.Prim, material_prim: Usd.Prim) -> None:
    physics_material = UsdShade.Material(material_prim)
    if not physics_material:
        raise RuntimeError(f"Prim exists but is not a UsdShade.Material: {material_prim.GetPath()}")

    binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    binding_api.Bind(
        physics_material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )


def verify_deformable_setup(stage: Usd.Stage, mesh_path: str, physics_material_path: str) -> None:
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    if not mesh_prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI):
        raise RuntimeError(f"Failed to apply PhysxDeformableBodyAPI to {mesh_path}")

    if mesh_prim.HasAPI(UsdPhysics.RigidBodyAPI) or mesh_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        raise RuntimeError(f"Mesh still has rigid-body schemas after conversion: {mesh_path}")

    material_prim = stage.GetPrimAtPath(physics_material_path)
    if not material_prim.HasAPI(PhysxSchema.PhysxDeformableBodyMaterialAPI):
        raise RuntimeError(f"Failed to apply PhysxDeformableBodyMaterialAPI to {physics_material_path}")

    bound_material, _relationship = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial("physics")
    if not bound_material or bound_material.GetPath().pathString != physics_material_path:
        raise RuntimeError(f"Mesh {mesh_path} is not bound to physics material {physics_material_path}")


def main() -> None:
    preset = PRESETS[args_cli.preset]
    usd_path = usd_path_for_args()
    if not usd_path.is_file():
        raise FileNotFoundError(f"USD does not exist: {usd_path}")

    if not open_stage(str(usd_path)):
        raise RuntimeError(f"Could not open USD as current stage: {usd_path}")

    for _ in range(5):
        update_stage()

    stage = get_current_stage()
    requested_root_path = args_cli.deformable_root_path or preset.default_deformable_root_path
    mesh_prim = single_mesh_prim(stage, requested_root_path)
    mesh_path = mesh_prim.GetPath().pathString
    deformable_root_path = requested_root_path or infer_deformable_root_path(mesh_prim)
    root_prim = stage.GetPrimAtPath(deformable_root_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Deformable root path is invalid: {deformable_root_path}")

    physics_material_path = material_path(stage, preset)

    log(f"Current stage: {stage.GetRootLayer().identifier}")
    log(f"Preset: {args_cli.preset}")
    log(f"Default prim: {stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None}")
    log(f"Deformable root path: {deformable_root_path}")
    log(f"Deformable mesh path: {mesh_path}")

    remove_rigid_body_schemas(stage)
    define_deformable_body_properties(
        deformable_root_path,
        sim_utils.DeformableBodyPropertiesCfg(**preset.body),
    )
    clear_unsupported_deformable_attributes(mesh_prim)
    update_stage()

    material_cfg = sim_utils.DeformableBodyMaterialCfg(**preset.material)
    material_prim = material_cfg.func(physics_material_path, material_cfg)
    bind_physics_material(mesh_prim, material_prim)
    update_stage()

    verify_deformable_setup(stage, mesh_path, physics_material_path)

    root_layer = stage.GetRootLayer()
    if not root_layer.Save():
        raise RuntimeError(f"Failed to save root layer: {root_layer.identifier}")

    log(f"Saved deformable USD: {root_layer.identifier}")
    log(f"Physics material path: {physics_material_path}")


try:
    main()
finally:
    simulation_app.close()
