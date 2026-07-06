# make_usd_deformable.py
# Usage: ~/IsaacLab/isaaclab.sh -p source/so101_bench/so101_bench/assets/make_usd_deformable.py [usd_path]
# Defaults to the brown_stuffed_animal.usdc under this file's assets/usd/objects/ directory.

# object_4 = DeformableObjectCfg(
#     prim_path="{ENV_REGEX_NS}/Object_4",
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ASSETS_PATH}/usd/objects/brown_stuffed_animal.usdc",
#     ),
#     init_state=DeformableObjectCfg.InitialStateCfg(
#         pos=(OBJECT_FIXED_POSES[3][0], OBJECT_FIXED_POSES[3][1], TABLE_OBJECT_Z),
#         rot=(1.0, 0.0, 0.0, 0.0),
#     ),
#     debug_vis=False,
# )

import sys
from pathlib import Path

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas import define_deformable_body_properties
from isaaclab.sim.utils.stage import (
    open_stage,
    get_current_stage,
    update_stage,
)
from pxr import UsdGeom, PhysxSchema, UsdShade

# Default to the mesh shipped alongside this script; allow an optional CLI override.
DEFAULT_USD_PATH = Path(__file__).resolve().parent / "usd" / "objects" / "brown_stuffed_animal.usdc"
USD_PATH = str(Path(sys.argv[1]).resolve()) if len(sys.argv) > 1 else str(DEFAULT_USD_PATH)

# From your Isaac Sim tree:
# root/defaultPrim -> node_0_001 Xform -> node_0_001 Mesh
DEFORMABLE_ROOT_PATH = "/root/node_0_001"

# Put the physics material next to the existing visual material scope.
STUFFED_ANIMAL_PHYSICS_MATERIAL_PATH = (
    "/root/_materials/stuffed_animal_deformable_physics"
)


# ---------------------------------------------------------------------
# Open USD as current Kit/Isaac stage.
# Do not use Usd.Stage.Open(...) for the Isaac Lab schema helper.
# ---------------------------------------------------------------------
if not open_stage(USD_PATH):
    raise RuntimeError(f"Could not open USD as current stage: {USD_PATH}")

for _ in range(5):
    update_stage()

stage = get_current_stage()

print("Current stage:", stage.GetRootLayer().identifier)
print(
    "Default prim:",
    stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None,
)

root_prim = stage.GetPrimAtPath(DEFORMABLE_ROOT_PATH)
print("Root prim valid:", root_prim.IsValid(), DEFORMABLE_ROOT_PATH)

if not root_prim.IsValid():
    raise RuntimeError(f"Deformable root path is invalid: {DEFORMABLE_ROOT_PATH}")

mesh_prims = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]

print("Mesh prims:")
for prim in mesh_prims:
    print(" ", prim.GetPath())

if len(mesh_prims) != 1:
    raise RuntimeError(
        f"Expected exactly one Mesh prim, found {len(mesh_prims)}. "
        "Join the object into one mesh in Blender, triangulate it, apply transforms, and export again."
    )

mesh_path = str(mesh_prims[0].GetPath())


# ---------------------------------------------------------------------
# Deformable-body geometry / solver / collision-cooking params.
#
# These are stable first-pass values for a small plush-ish object:
# - low-ish hex resolution so the object is not insanely expensive
# - simplified collision so grasp/contact stays stable
# - no self collision for now
# ---------------------------------------------------------------------
deformable_body_cfg = sim_utils.DeformableBodyPropertiesCfg(
    deformable_enabled=True,
    rest_offset=0.0,
    # Slightly larger contact shell to help the gripper maintain contact.
    contact_offset=0.0035,
    # More stable contact under lift.
    solver_position_iteration_count=48,
    vertex_velocity_damping=0.07,
    max_depenetration_velocity=0.25,
    simulation_hexahedral_resolution=12,
    collision_simplification=True,
    collision_simplification_remeshing=True,
    collision_simplification_remeshing_resolution=28,
    collision_simplification_target_triangle_count=900,
    self_collision=False,
)

define_deformable_body_properties(
    DEFORMABLE_ROOT_PATH,
    deformable_body_cfg,
)

update_stage()

# ---------------------------------------------------------------------
# Stuffed-animal physics material.
#
# Defaults in Isaac Lab are much stiffer than plush.
# This is intentionally "stable plush", not physically perfect cloth/fur:
# - lower Young's modulus => softer/squishier
# - high-ish dynamic friction => gripper can hold it
# - damping prevents floppy/jelly oscillations
# ---------------------------------------------------------------------
stuffed_animal_material_cfg = sim_utils.DeformableBodyMaterialCfg(
    density=45.0,
    # Main change: more grip.
    dynamic_friction=4.0,
    # Slightly softer than 2.5e5 so it forms a better contact patch.
    youngs_modulus=1.8e5,
    # Still fairly volume-preserving, but not hard-rubber stiff.
    poissons_ratio=0.36,
    elasticity_damping=0.09,
    damping_scale=1.0,
)

# This creates or updates the material prim. It is safe to rerun if the prim
# already exists, as long as it is a Material prim.
material_prim = stuffed_animal_material_cfg.func(
    STUFFED_ANIMAL_PHYSICS_MATERIAL_PATH,
    stuffed_animal_material_cfg,
)

print("Created/updated deformable physics material:", material_prim.GetPath())

# Bind the physics material directly using USD material binding.
# This authors the relationship:
# material:binding:physics -> /root/_materials/stuffed_animal_deformable_physics
mesh_prim = stage.GetPrimAtPath(mesh_path)
material_prim = stage.GetPrimAtPath(STUFFED_ANIMAL_PHYSICS_MATERIAL_PATH)

if not mesh_prim.IsValid():
    raise RuntimeError(f"Invalid mesh prim: {mesh_path}")

if not material_prim.IsValid():
    raise RuntimeError(f"Invalid material prim: {STUFFED_ANIMAL_PHYSICS_MATERIAL_PATH}")

physics_material = UsdShade.Material(material_prim)
if not physics_material:
    raise RuntimeError(
        f"Prim exists but is not a UsdShade.Material: {STUFFED_ANIMAL_PHYSICS_MATERIAL_PATH}"
    )

binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
binding_api.Bind(
    physics_material,
    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    materialPurpose="physics",
)

print(
    "Bound deformable physics material:",
    STUFFED_ANIMAL_PHYSICS_MATERIAL_PATH,
    "->",
    mesh_path,
)


# ---------------------------------------------------------------------
# Verification.
# ---------------------------------------------------------------------
mesh_prim = stage.GetPrimAtPath(mesh_path)

print("Applied schemas on mesh:")
for schema in mesh_prim.GetAppliedSchemas():
    print(" ", schema)

if not mesh_prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI):
    raise RuntimeError(f"Failed to apply PhysxDeformableBodyAPI to {mesh_path}")

material_prim = stage.GetPrimAtPath(STUFFED_ANIMAL_PHYSICS_MATERIAL_PATH)
print("Applied schemas on stuffed-animal physics material:")
for schema in material_prim.GetAppliedSchemas():
    print(" ", schema)

if not material_prim.HasAPI(PhysxSchema.PhysxDeformableBodyMaterialAPI):
    raise RuntimeError(
        f"Failed to apply PhysxDeformableBodyMaterialAPI to "
        f"{STUFFED_ANIMAL_PHYSICS_MATERIAL_PATH}"
    )


# ---------------------------------------------------------------------
# Save existing opened root layer.
# Do NOT use save_stage(USD_PATH), because that tries to create a new layer.
# ---------------------------------------------------------------------
root_layer = stage.GetRootLayer()
if not root_layer.Save():
    raise RuntimeError(f"Failed to save root layer: {root_layer.identifier}")

print(f"Saved deformable USD: {root_layer.identifier}")
print(f"Deformable mesh path: {mesh_path}")
print(f"Physics material path: {STUFFED_ANIMAL_PHYSICS_MATERIAL_PATH}")

simulation_app.close()
