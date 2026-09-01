"""Check object USD mesh material bindings and visual color textures.

Run with:

    /home/truman/IsaacLab/isaaclab.sh -p scripts/check_object_usd_assets.py
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJECTS_DIR = (
    REPO_ROOT / "source" / "so101_bench" / "so101_bench" / "assets" / "usd" / "objects"
)
USD_SUFFIXES = {".usd", ".usda", ".usdc"}
COLOR_INPUT_TERMS = ("albedo", "basecolor", "base_color", "color", "diffuse")
NON_COLOR_INPUT_TERMS = ("clearcoat", "metal", "normal", "occlusion", "rough")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objects-dir",
        type=Path,
        default=DEFAULT_OBJECTS_DIR,
        help=f"Directory containing object USD files. Default: {DEFAULT_OBJECTS_DIR}",
    )
    return parser.parse_args()


args_cli = parse_args()


def mesh_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    """Return all Mesh prims in the stage."""
    return [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]


def bound_material(prim: Usd.Prim, purposes: Iterable[str]) -> UsdShade.Material | None:
    """Return the first material bound for one of the requested USD purposes."""
    binding_api = UsdShade.MaterialBindingAPI(prim)
    for purpose in purposes:
        material, _relationship = binding_api.ComputeBoundMaterial(purpose)
        if material and material.GetPrim().IsValid():
            return material
    return None


def input_name_is_color_input(shader_input: UsdShade.Input) -> bool:
    """Heuristically identify shader inputs that carry visible surface color."""
    name = shader_input.GetBaseName().lower()
    return any(term in name for term in COLOR_INPUT_TERMS) and not any(
        term in name for term in NON_COLOR_INPUT_TERMS
    )


def asset_paths_from_value(value: object) -> list[Sdf.AssetPath]:
    """Extract asset paths from scalar and array-valued USD input values."""
    if isinstance(value, Sdf.AssetPath):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, Sdf.AssetPath)]
    return []


def input_asset_paths(
    shader_input: UsdShade.Input,
    visited_inputs: set[str] | None = None,
) -> set[Sdf.AssetPath]:
    """Follow a shader input upstream and collect referenced asset paths."""
    if visited_inputs is None:
        visited_inputs = set()

    input_path = shader_input.GetAttr().GetPath().pathString
    if input_path in visited_inputs:
        return set()
    visited_inputs.add(input_path)

    asset_paths = set(asset_paths_from_value(shader_input.Get()))
    connected_sources, _invalid_source_paths = shader_input.GetConnectedSources()
    for connected_source in connected_sources:
        source_prim = connected_source.source.GetPrim()
        source_shader = UsdShade.Shader(source_prim)
        if not source_shader:
            continue
        for source_input in source_shader.GetInputs():
            asset_paths.update(input_asset_paths(source_input, visited_inputs))
    return asset_paths


def all_material_asset_paths(material: UsdShade.Material) -> set[Sdf.AssetPath]:
    """Collect every shader asset reference authored under a material."""
    asset_paths: set[Sdf.AssetPath] = set()
    for prim in Usd.PrimRange(material.GetPrim()):
        shader = UsdShade.Shader(prim)
        if not shader:
            continue
        for shader_input in shader.GetInputs():
            asset_paths.update(asset_paths_from_value(shader_input.Get()))
    return asset_paths


def visual_color_asset_paths(material: UsdShade.Material) -> set[Sdf.AssetPath]:
    """Collect asset paths connected to visual color-related shader inputs."""
    asset_paths: set[Sdf.AssetPath] = set()
    for prim in Usd.PrimRange(material.GetPrim()):
        shader = UsdShade.Shader(prim)
        if not shader:
            continue
        for shader_input in shader.GetInputs():
            if input_name_is_color_input(shader_input):
                asset_paths.update(input_asset_paths(shader_input))

    if asset_paths:
        return asset_paths

    # Some exporters name texture file inputs without semantic color input names.
    # In that case, the color texture naming convention is the next strongest signal.
    return {
        asset_path
        for asset_path in all_material_asset_paths(material)
        if "color" in Path(asset_path.path).name.lower()
    }


def resolve_asset_path(stage: Usd.Stage, asset_path: Sdf.AssetPath) -> Path | None:
    """Resolve an authored USD asset path against the layer that owns the object."""
    if asset_path.resolvedPath:
        return Path(asset_path.resolvedPath).resolve()

    raw_path = asset_path.path
    if not raw_path:
        return None

    raw_path_obj = Path(raw_path)
    if raw_path_obj.is_absolute():
        return raw_path_obj.resolve()

    root_layer_path = Path(stage.GetRootLayer().realPath)
    return (root_layer_path.parent / raw_path_obj).resolve()


def texture_paths_in_dir(
    stage: Usd.Stage,
    asset_paths: Iterable[Sdf.AssetPath],
    textures_dir: Path,
) -> list[Path]:
    """Return existing asset paths that resolve inside the expected textures dir."""
    textures_dir = textures_dir.resolve()
    texture_paths: set[Path] = set()
    for asset_path in asset_paths:
        resolved_path = resolve_asset_path(stage, asset_path)
        if resolved_path is None or not resolved_path.is_file():
            continue
        if resolved_path.is_relative_to(textures_dir):
            texture_paths.add(resolved_path)
    return sorted(texture_paths)


def print_paths(label: str, paths: Iterable[Path]) -> None:
    paths = list(paths)
    print(f"    {label}:")
    for path in paths:
        print(f"      {path.name}")


def check_object_usd(usd_path: Path, textures_dir: Path) -> bool:
    """Print checks for one object USD and return True when it passes."""
    print(f"\n{usd_path.name}")
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        print("  FAIL could not open USD stage")
        return False

    passes = True
    all_meshes = mesh_prims(stage)
    if not all_meshes:
        print("  FAIL no Mesh prim found")
        passes = False

    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        print("  FAIL no default prim")
        passes = False
    else:
        default_path = default_prim.GetPath()
        for prim in stage.Traverse():
            relationship = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel()
            if not relationship:
                continue
            outside_targets = [
                target for target in relationship.GetTargets() if not target.HasPrefix(default_path)
            ]
            if outside_targets:
                print(
                    f"  FAIL {prim.GetPath()} has material target(s) outside "
                    f"default prim {default_path}: {outside_targets}"
                )
                passes = False

    physics_bound_meshes: list[tuple[Usd.Prim, UsdShade.Material]] = []
    for mesh_prim in all_meshes:
        material = bound_material(mesh_prim, ("physics",))
        if material is not None and material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
            physics_bound_meshes.append((mesh_prim, material))

    if not physics_bound_meshes:
        print("  FAIL no Mesh prim has a physics-purpose material binding")
        passes = False
    else:
        print("  physics material bindings:")
        for mesh_prim, material in physics_bound_meshes:
            print(f"    PASS {mesh_prim.GetPath()}: {material.GetPrim().GetName()} ({material.GetPath()})")

    visual_texture_paths: set[Path] = set()
    textured_meshes: list[tuple[Usd.Prim, UsdShade.Material, list[Path]]] = []
    for mesh_prim in all_meshes:
        material = bound_material(mesh_prim, (UsdShade.Tokens.allPurpose,))
        if material is None:
            continue

        mesh_texture_paths = texture_paths_in_dir(
            stage,
            visual_color_asset_paths(material),
            textures_dir,
        )
        if mesh_texture_paths:
            textured_meshes.append((mesh_prim, material, mesh_texture_paths))
            visual_texture_paths.update(mesh_texture_paths)

    if not textured_meshes:
        print(f"  FAIL no Mesh prim is connected to a color texture in {textures_dir}")
        passes = False
    else:
        print("  visual color textures:")
        for mesh_prim, material, mesh_texture_paths in textured_meshes:
            print(f"    PASS {mesh_prim.GetPath()}: {material.GetPrim().GetName()}")
            print_paths("textures", mesh_texture_paths)

    if visual_texture_paths:
        print_paths("object color texture files", visual_texture_paths)
    return passes


def main() -> int:
    objects_dir = args_cli.objects_dir.resolve()
    textures_dir = objects_dir / "textures"
    usd_paths = sorted(
        path
        for path in objects_dir.iterdir()
        if path.is_file() and path.suffix.lower() in USD_SUFFIXES
    )

    print(f"Objects directory: {objects_dir}")
    print(f"Textures directory: {textures_dir.resolve()}")
    print(f"Object USD files: {len(usd_paths)}")

    if not textures_dir.is_dir():
        print(f"FAIL textures directory does not exist: {textures_dir}")
        return 1
    if not usd_paths:
        print(f"FAIL no USD files found in: {objects_dir}")
        return 1

    failed_paths = [path for path in usd_paths if not check_object_usd(path, textures_dir)]
    print("\nSummary")
    if failed_paths:
        print(f"FAIL {len(failed_paths)} of {len(usd_paths)} object USD files failed")
        for path in failed_paths:
            print(f"  {path.name}")
        return 1

    print(f"PASS checked {len(usd_paths)} object USD files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
