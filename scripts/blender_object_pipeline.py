#!/usr/bin/env python3
"""Build SO-101 Bench visual and physics meshes from a prepared Blender file.

Run this script with Blender, not regular Python::

    blender --background --factory-startup \
        --python scripts/blender_object_pipeline.py -- \
        path/to/object.blend --pipeline 2 --asset-name new_object

Pipeline 1 decimates the textured source directly. Pipeline 2 voxel-remeshes and
decimates a visual copy, creates new UVs and textures, then bakes base color and
roughness from the untouched source. Both pipelines create a voxel-remeshed,
decimated physics mesh, save a processed blend file, render visual/physics
previews, and export an Isaac-compatible USD hierarchy:

    /root/visual/<asset>_visual
    /root/physics/<asset>_physics

The physics prim is invisible, has collision schemas, and is bound to a
physics-purpose material. Pipeline 2's processed blend also retains the source
under /root/original; the source is removed only from the exported USD.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

try:
    import bpy
    from mathutils import Vector
except ImportError as exc:  # pragma: no cover - only useful for a friendly shell error
    raise SystemExit(
        "This script must run inside Blender. See the invocation in --help or the module docstring."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJECTS_DIR = (
    REPO_ROOT / "source" / "so101_bench" / "so101_bench" / "assets" / "usd" / "objects"
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("expected a positive number")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("expected a non-negative number")
    return parsed


def unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("expected a number in [0, 1]")
    return parsed


def angle_degrees(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 180.0:
        raise argparse.ArgumentTypeError("expected an angle in [0, 180] degrees")
    return parsed


def blender_script_args() -> list[str]:
    """Return arguments after Blender's conventional ``--`` separator."""
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("blend_file", type=Path, help="Prepared .blend containing the source mesh.")
    parser.add_argument("--pipeline", type=int, choices=(1, 2), required=True)
    parser.add_argument(
        "--asset-name",
        help="Lowercase USD/file stem. Defaults to the input stem (with _processed removed).",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source-object",
        action="append",
        dest="source_objects",
        help="Mesh object to use; repeat to join several source parts.",
    )
    source_group.add_argument(
        "--join-all-meshes",
        action="store_true",
        help="Join every mesh object in the blend as the source.",
    )

    output = parser.add_argument_group("outputs")
    output.add_argument("--output-blend", type=Path, help="Processed .blend output path.")
    output.add_argument("--output-usd", type=Path, help="USD output path (.usd/.usda/.usdc).")
    output.add_argument(
        "--objects-dir",
        type=Path,
        default=DEFAULT_OBJECTS_DIR,
        help="Default USD directory when --output-usd is omitted.",
    )
    output.add_argument(
        "--texture-dir",
        type=Path,
        help="Baked texture directory. Defaults to <USD directory>/textures.",
    )
    output.add_argument(
        "--preview-dir",
        type=Path,
        help="Preview directory. Defaults to the processed blend's directory.",
    )
    output.add_argument(
        "--export-textures-mode",
        choices=("KEEP", "PRESERVE", "NEW"),
        default="KEEP",
        help=(
            "Blender USD texture export policy. Pipeline 1 first stages uniquely named "
            "texture files, so KEEP is the collision-safe default."
        ),
    )
    output.add_argument("--overwrite", action="store_true", help="Allow replacing output files.")

    visual = parser.add_argument_group("visual processing")
    visual.add_argument(
        "--visual-remesh-faces",
        type=positive_int,
        default=50_000,
        help="Pipeline 2 approximate triangle target for voxel remeshing.",
    )
    visual.add_argument(
        "--visual-remesh-voxel-size",
        type=positive_float,
        help="Use this exact voxel size instead of searching for --visual-remesh-faces.",
    )
    visual.add_argument("--visual-remesh-adaptivity", type=unit_float, default=0.0)
    visual.add_argument(
        "--visual-remesh-remove-disconnected",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    visual.add_argument(
        "--visual-decimate-faces",
        type=positive_int,
        default=10_000,
        help="Triangle target when --visual-decimate-mode=collapse.",
    )
    visual.add_argument(
        "--visual-decimate-mode",
        choices=("collapse", "planar"),
        default="collapse",
        help="Collapse targets a face count; planar dissolves faces below an angle.",
    )
    visual.add_argument(
        "--visual-decimate-angle-deg",
        type=angle_degrees,
        default=5.0,
        help="Maximum planar dissolve angle; used only in planar mode.",
    )
    visual.add_argument(
        "--delete-loose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Edit Mode > Clean Up > Delete Loose after pipeline 1 decimation.",
    )

    uv = parser.add_argument_group("pipeline 2 UV and material")
    uv.add_argument("--uv-angle-limit-deg", type=positive_float, default=66.0)
    uv.add_argument("--uv-island-margin", type=nonnegative_float, default=0.02)
    uv.add_argument("--uv-area-weight", type=unit_float, default=0.0)
    uv.add_argument(
        "--uv-margin-method", choices=("SCALED", "ADD", "FRACTION"), default="FRACTION"
    )
    uv.add_argument("--material-roughness", type=unit_float, default=0.5)

    bake = parser.add_argument_group("pipeline 2 baking")
    bake.add_argument("--color-resolution", type=positive_int, default=2048)
    bake.add_argument("--roughness-resolution", type=positive_int, default=2048)
    bake.add_argument("--bake-margin", type=nonnegative_float, default=16.0, help="Pixel margin.")
    bake.add_argument("--bake-cage-extrusion", type=nonnegative_float, default=0.01)
    bake.add_argument("--bake-max-ray-distance", type=nonnegative_float, default=0.0)
    bake.add_argument("--bake-samples", type=positive_int, default=32)
    bake.add_argument("--bake-device", choices=("CPU", "GPU"), default="CPU")

    physics = parser.add_argument_group("physics processing and USD material")
    physics.add_argument(
        "--physics-remesh-faces",
        type=positive_int,
        default=10_000,
        help="Approximate triangle target for voxel remeshing.",
    )
    physics.add_argument(
        "--physics-remesh-voxel-size",
        type=positive_float,
        help="Use this exact voxel size instead of searching for --physics-remesh-faces.",
    )
    physics.add_argument("--physics-remesh-adaptivity", type=unit_float, default=0.0)
    physics.add_argument(
        "--physics-remesh-remove-disconnected",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    physics.add_argument("--physics-decimate-faces", type=positive_int, default=1_000)
    physics.add_argument(
        "--physics-approximation",
        choices=(
            "none",
            "convexHull",
            "convexDecomposition",
            "meshSimplification",
            "boundingCube",
            "boundingSphere",
            "sdf",
        ),
        default="sdf",
    )
    physics.add_argument("--static-friction", type=nonnegative_float, default=0.5)
    physics.add_argument("--dynamic-friction", type=nonnegative_float, default=0.32)
    physics.add_argument("--restitution", type=unit_float, default=0.04)
    physics.add_argument("--density", type=nonnegative_float, default=0.0)
    physics.add_argument("--mass", type=positive_float, default=0.05, help="Rigid-body mass in kg.")

    quality = parser.add_argument_group("search and preview quality")
    quality.add_argument("--remesh-search-steps", type=positive_int, default=8)
    quality.add_argument("--remesh-target-tolerance", type=unit_float, default=0.10)
    quality.add_argument("--preview-resolution", type=positive_int, default=512)
    quality.add_argument("--preview-samples", type=positive_int, default=32)
    return parser


def normalized_asset_name(raw_name: str) -> str:
    name = re.sub(r"_processed$", "", raw_name.strip(), flags=re.IGNORECASE)
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower()
    if not name:
        raise ValueError("Asset name is empty after normalization.")
    if name[0].isdigit():
        name = f"object_{name}"
    return name


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def resolve_outputs(args: argparse.Namespace) -> None:
    args.blend_file = resolve_path(args.blend_file)
    if not args.blend_file.is_file():
        raise FileNotFoundError(f"Input blend does not exist: {args.blend_file}")
    if args.blend_file.suffix.lower() != ".blend":
        raise ValueError(f"Expected a .blend input, got: {args.blend_file}")

    args.asset_name = normalized_asset_name(args.asset_name or args.blend_file.stem)
    args.objects_dir = resolve_path(args.objects_dir)
    args.output_blend = resolve_path(
        args.output_blend or args.blend_file.with_name(f"{args.asset_name}_processed.blend")
    )
    args.output_usd = resolve_path(
        args.output_usd or args.objects_dir / f"{args.asset_name}.usdc"
    )
    if args.output_usd.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError(f"USD output needs a .usd, .usda, or .usdc suffix: {args.output_usd}")
    args.texture_dir = resolve_path(args.texture_dir or args.output_usd.parent / "textures")
    args.preview_dir = resolve_path(args.preview_dir or args.output_blend.parent)
    args.color_path = args.texture_dir / f"{args.asset_name}_colors_1.png"
    args.roughness_path = args.texture_dir / f"{args.asset_name}_rough_1.png"
    args.visual_preview_path = args.preview_dir / f"{args.asset_name}_visual.png"
    args.physics_preview_path = args.preview_dir / f"{args.asset_name}_physics.png"

    outputs = [
        args.output_blend,
        args.output_usd,
        args.visual_preview_path,
        args.physics_preview_path,
    ]
    if args.pipeline == 2:
        outputs.extend((args.color_path, args.roughness_path))
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        formatted = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist (pass --overwrite to replace):\n  {formatted}")
    if args.output_blend == args.blend_file and not args.overwrite:
        raise ValueError("Refusing to replace the input blend without --overwrite.")

    for directory in {
        args.output_blend.parent,
        args.output_usd.parent,
        args.texture_dir,
        args.preview_dir,
    }:
        directory.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[object-pipeline] {message}", flush=True)


def eevee_engine_name() -> str:
    """Return the Eevee engine identifier used by this Blender version."""
    engine_property = bpy.context.scene.render.bl_rna.properties["engine"]
    identifiers = {item.identifier for item in engine_property.enum_items}
    if "BLENDER_EEVEE_NEXT" in identifiers:
        return "BLENDER_EEVEE_NEXT"
    return "BLENDER_EEVEE"


def ensure_object_mode() -> None:
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def activate_only(active: bpy.types.Object, selected: Iterable[bpy.types.Object] = ()) -> None:
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    for obj in selected:
        obj.select_set(True)
    active.select_set(True)
    bpy.context.view_layer.objects.active = active


def source_meshes(args: argparse.Namespace) -> list[bpy.types.Object]:
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if args.source_objects:
        missing = [name for name in args.source_objects if bpy.data.objects.get(name) is None]
        if missing:
            raise ValueError(f"Source object(s) not found: {', '.join(missing)}")
        sources = [bpy.data.objects[name] for name in args.source_objects]
        not_mesh = [obj.name for obj in sources if obj.type != "MESH"]
        if not_mesh:
            raise ValueError(f"Source object(s) are not meshes: {', '.join(not_mesh)}")
        return sources
    if args.join_all_meshes:
        if not meshes:
            raise ValueError("The blend has no mesh objects.")
        return meshes

    selected = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if selected:
        return selected
    if len(meshes) == 1:
        return meshes
    choices = ", ".join(obj.name for obj in meshes) or "<none>"
    raise ValueError(
        "Could not choose the source automatically. Select the desired mesh(es) in the saved "
        f"blend, pass --source-object, or pass --join-all-meshes. Meshes: {choices}"
    )


def move_to_scene_collection(obj: bpy.types.Object) -> None:
    master = bpy.context.scene.collection
    if master not in obj.users_collection:
        master.objects.link(obj)
    for collection in tuple(obj.users_collection):
        if collection != master:
            collection.objects.unlink(obj)


def consolidate_source(sources: list[bpy.types.Object], asset_name: str) -> bpy.types.Object:
    """Apply existing modifiers, join source parts, detach parents, and apply scale."""
    activate_only(sources[0], sources)
    bpy.ops.object.convert(target="MESH")
    # convert() can replace selected object references; get the current selection again.
    converted = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    active = bpy.context.view_layer.objects.active
    if active is None or active.type != "MESH":
        active = converted[0]
        bpy.context.view_layer.objects.active = active
    if len(converted) > 1:
        bpy.ops.object.join()
    source = bpy.context.view_layer.objects.active
    world = source.matrix_world.copy()
    source.parent = None
    source.matrix_world = world
    move_to_scene_collection(source)
    activate_only(source)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    source.name = f"{asset_name}_original"
    source.data.name = f"{asset_name}_original_mesh"
    if not source.data.polygons:
        raise ValueError("The consolidated source mesh has no faces.")
    return source


def new_empty(name: str, parent: bpy.types.Object | None = None) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    obj.empty_display_type = "PLAIN_AXES"
    obj.parent = parent
    return obj


def reparent_keep_world(obj: bpy.types.Object, parent: bpy.types.Object) -> None:
    world = obj.matrix_world.copy()
    obj.parent = parent
    obj.matrix_world = world


def duplicate_mesh(
    source: bpy.types.Object, name: str, data_name: str, parent: bpy.types.Object
) -> bpy.types.Object:
    duplicate = source.copy()
    duplicate.data = source.data.copy()
    bpy.context.scene.collection.objects.link(duplicate)
    duplicate.name = name
    duplicate.data.name = data_name
    reparent_keep_world(duplicate, parent)
    return duplicate


def delete_object(obj: bpy.types.Object | None) -> None:
    if obj is not None and obj.name in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)


def delete_mesh_object(obj: bpy.types.Object | None) -> None:
    mesh = obj.data if obj is not None and obj.type == "MESH" else None
    delete_object(obj)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def triangle_count_for_mesh(mesh: bpy.types.Mesh) -> int:
    mesh.calc_loop_triangles()
    return len(mesh.loop_triangles)


def triangle_count(obj: bpy.types.Object) -> int:
    return triangle_count_for_mesh(obj.data)


def evaluated_triangle_count(obj: bpy.types.Object) -> int:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return triangle_count_for_mesh(mesh)
    finally:
        evaluated.to_mesh_clear()


def apply_modifier(obj: bpy.types.Object, modifier: bpy.types.Modifier) -> None:
    activate_only(obj)
    result = bpy.ops.object.modifier_apply(modifier=modifier.name)
    if "FINISHED" not in result:
        raise RuntimeError(f"Failed to apply modifier {modifier.name!r} to {obj.name!r}.")


def decimate_to_faces(obj: bpy.types.Object, target_faces: int, label: str) -> None:
    before = triangle_count(obj)
    if before <= target_faces:
        log(f"{label} decimate skipped: {before:,} triangles <= {target_faces:,} target")
        return
    modifier = obj.modifiers.new(name=f"{label} Decimate", type="DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = max(0.0, min(1.0, target_faces / before))
    modifier.use_collapse_triangulate = True
    apply_modifier(obj, modifier)
    log(f"{label} decimate: {before:,} -> {triangle_count(obj):,} triangles")


def decimate_visual(obj: bpy.types.Object, args: argparse.Namespace) -> None:
    if args.visual_decimate_mode == "collapse":
        decimate_to_faces(obj, args.visual_decimate_faces, "visual")
        return

    before = triangle_count(obj)
    modifier = obj.modifiers.new(name="visual Planar Decimate", type="DECIMATE")
    modifier.decimate_type = "DISSOLVE"
    modifier.angle_limit = math.radians(args.visual_decimate_angle_deg)
    apply_modifier(obj, modifier)
    log(
        f"visual planar decimate at {args.visual_decimate_angle_deg:g} degrees: "
        f"{before:,} -> {triangle_count(obj):,} triangles"
    )


def delete_loose_geometry(obj: bpy.types.Object) -> None:
    activate_only(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


def remesh_to_faces(
    obj: bpy.types.Object,
    target_faces: int,
    voxel_size: float | None,
    adaptivity: float,
    remove_disconnected: bool,
    search_steps: int,
    tolerance: float,
    label: str,
) -> None:
    modifier = obj.modifiers.new(name=f"{label} Voxel Remesh", type="REMESH")
    modifier.mode = "VOXEL"
    modifier.adaptivity = adaptivity
    modifier.use_remove_disconnected = remove_disconnected
    modifier.use_smooth_shade = label == "visual"

    max_dimension = max(obj.dimensions)
    if max_dimension <= 0.0:
        raise ValueError(f"{label.capitalize()} mesh has zero-size bounds.")

    if voxel_size is not None:
        modifier.voxel_size = voxel_size
        count = evaluated_triangle_count(obj)
        log(f"{label} remesh using voxel size {voxel_size:.8g}: ~{count:,} triangles")
    else:
        # A cube has about 6 * (dimension / voxel_size)^2 surface triangles.
        # This is a useful scale-independent starting point; evaluated feedback
        # then converges toward the requested count for non-cube shapes.
        current_size = max_dimension * math.sqrt(6.0 / target_faces)
        minimum_size = max_dimension / 2048.0
        maximum_size = max_dimension / 2.0
        current_size = min(max(current_size, minimum_size), maximum_size)
        best: tuple[float, int, float] | None = None

        for _ in range(search_steps):
            modifier.voxel_size = current_size
            count = evaluated_triangle_count(obj)
            relative_error = abs(count - target_faces) / target_faces
            candidate = (relative_error, count, current_size)
            if best is None or candidate[0] < best[0]:
                best = candidate
            if relative_error <= tolerance:
                break
            ratio = max(count, 1) / target_faces
            adjustment = min(2.0, max(0.5, math.sqrt(ratio)))
            next_size = min(max(current_size * adjustment, minimum_size), maximum_size)
            if math.isclose(next_size, current_size, rel_tol=1.0e-6):
                break
            current_size = next_size

        assert best is not None
        _, estimated_count, chosen_size = best
        modifier.voxel_size = chosen_size
        log(
            f"{label} remesh chose voxel size {chosen_size:.8g}: "
            f"~{estimated_count:,} triangles for {target_faces:,} target"
        )

    apply_modifier(obj, modifier)
    if not obj.data.polygons:
        raise RuntimeError(
            f"{label.capitalize()} voxel remesh produced an empty mesh. Use a smaller voxel size."
        )
    log(f"{label} remesh result: {triangle_count(obj):,} triangles")


def smart_uv_project(obj: bpy.types.Object, args: argparse.Namespace) -> None:
    while obj.data.uv_layers:
        obj.data.uv_layers.remove(obj.data.uv_layers[0])
    obj.data.uv_layers.new(name="UVMap")
    activate_only(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        bpy.ops.mesh.select_all(action="SELECT")
        result = bpy.ops.uv.smart_project(
            angle_limit=math.radians(args.uv_angle_limit_deg),
            margin_method=args.uv_margin_method,
            island_margin=args.uv_island_margin,
            area_weight=args.uv_area_weight,
            correct_aspect=True,
            scale_to_bounds=False,
        )
        if "FINISHED" not in result:
            raise RuntimeError("Smart UV Project did not finish.")
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


def set_image_colorspace(image: bpy.types.Image, colorspace: str) -> None:
    try:
        image.colorspace_settings.name = colorspace
    except TypeError:
        log(f"Warning: Blender does not provide the {colorspace!r} image color space.")


def create_bake_material(
    visual: bpy.types.Object, args: argparse.Namespace
) -> tuple[bpy.types.Material, bpy.types.Image, bpy.types.Image, bpy.types.Node, bpy.types.Node]:
    visual.data.materials.clear()
    material = bpy.data.materials.new(name=f"{args.asset_name}_material")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (620, 80)
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (300, 80)
    shader.inputs["Roughness"].default_value = args.material_roughness
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    color_image = bpy.data.images.new(
        name=f"{args.asset_name}_colors_1",
        width=args.color_resolution,
        height=args.color_resolution,
        alpha=False,
        float_buffer=False,
    )
    color_image.generated_color = (0.18, 0.18, 0.18, 1.0)
    color_image.file_format = "PNG"
    color_image.filepath_raw = str(args.color_path)
    set_image_colorspace(color_image, "sRGB")
    color_node = nodes.new("ShaderNodeTexImage")
    color_node.name = "Baked Base Color"
    color_node.label = "Baked Base Color"
    color_node.location = (-100, 160)
    color_node.image = color_image

    roughness_image = bpy.data.images.new(
        name=f"{args.asset_name}_rough_1",
        width=args.roughness_resolution,
        height=args.roughness_resolution,
        alpha=False,
        float_buffer=False,
    )
    roughness_image.generated_color = (
        args.material_roughness,
        args.material_roughness,
        args.material_roughness,
        1.0,
    )
    roughness_image.file_format = "PNG"
    roughness_image.filepath_raw = str(args.roughness_path)
    set_image_colorspace(roughness_image, "Non-Color")
    roughness_node = nodes.new("ShaderNodeTexImage")
    roughness_node.name = "Baked Roughness"
    roughness_node.label = "Baked Roughness"
    roughness_node.location = (-100, -100)
    roughness_node.image = roughness_image

    visual.data.materials.append(material)
    return material, color_image, roughness_image, color_node, roughness_node


def bake_maps(
    original: bpy.types.Object,
    visual: bpy.types.Object,
    material: bpy.types.Material,
    color_image: bpy.types.Image,
    roughness_image: bpy.types.Image,
    color_node: bpy.types.Node,
    roughness_node: bpy.types.Node,
    args: argparse.Namespace,
) -> None:
    scene = bpy.context.scene
    # Texture baking is implemented by Cycles.
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.bake_samples
    scene.cycles.device = args.bake_device
    activate_only(visual, (original, visual))

    common = {
        "use_selected_to_active": True,
        "cage_extrusion": args.bake_cage_extrusion,
        "max_ray_distance": args.bake_max_ray_distance,
        "margin": round(args.bake_margin),
        "margin_type": "EXTEND",
        "use_clear": True,
        "target": "IMAGE_TEXTURES",
    }

    material.node_tree.nodes.active = color_node
    color_node.select = True
    roughness_node.select = False
    log(f"Baking base color at {args.color_resolution}x{args.color_resolution}")
    result = bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, **common)
    if "FINISHED" not in result:
        raise RuntimeError("Base-color bake failed.")
    color_image.save()

    material.node_tree.nodes.active = roughness_node
    color_node.select = False
    roughness_node.select = True
    log(f"Baking roughness at {args.roughness_resolution}x{args.roughness_resolution}")
    result = bpy.ops.object.bake(type="ROUGHNESS", **common)
    if "FINISHED" not in result:
        raise RuntimeError("Roughness bake failed.")
    roughness_image.save()

    # Link only after baking so the target images are not also dependencies of
    # the material graph while Blender writes into them.
    shader = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    material.node_tree.links.new(color_node.outputs["Color"], shader.inputs["Base Color"])
    material.node_tree.links.new(roughness_node.outputs["Color"], shader.inputs["Roughness"])

    # Make the saved blend and USD portable relative to the processed blend.
    color_image.filepath = "//" + os.path.relpath(args.color_path, args.output_blend.parent)
    roughness_image.filepath = "//" + os.path.relpath(args.roughness_path, args.output_blend.parent)


def world_bounds(obj: bpy.types.Object) -> tuple[Vector, float]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    low = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    high = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    center = (low + high) / 2.0
    maximum_dimension = max(high - low)
    if maximum_dimension <= 0.0:
        raise ValueError(f"Cannot frame zero-size object {obj.name!r}.")
    return center, maximum_dimension


def point_camera(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def make_area_light(name: str, location: Vector, target: Vector, energy: float, size: float):
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    point_camera(obj, target)
    return obj


def render_previews(
    visual: bpy.types.Object,
    physics: bpy.types.Object,
    args: argparse.Namespace,
) -> None:
    scene = bpy.context.scene
    hidden_state = {obj: obj.hide_render for obj in scene.objects}
    old_camera = scene.camera
    old_world = scene.world
    old_engine = scene.render.engine
    old_filepath = scene.render.filepath
    old_override = bpy.context.view_layer.material_override
    temporary_objects: list[bpy.types.Object] = []
    temporary_data: list[object] = []

    try:
        for obj in hidden_state:
            obj.hide_render = True

        camera_data = bpy.data.cameras.new("Pipeline Preview Camera")
        temporary_data.append(camera_data)
        camera_data.type = "ORTHO"
        camera = bpy.data.objects.new("Pipeline Preview Camera", camera_data)
        bpy.context.scene.collection.objects.link(camera)
        temporary_objects.append(camera)
        scene.camera = camera

        preview_world = bpy.data.worlds.new("Pipeline Preview World")
        temporary_data.append(preview_world)
        preview_world.use_nodes = True
        preview_world.node_tree.nodes["Background"].inputs["Color"].default_value = (
            0.035,
            0.035,
            0.035,
            1.0,
        )
        preview_world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
        scene.world = preview_world
        scene.render.engine = eevee_engine_name()
        scene.eevee.taa_render_samples = args.preview_samples
        scene.render.resolution_x = args.preview_resolution
        scene.render.resolution_y = args.preview_resolution
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.film_transparent = False
        scene.render.filepath = str(args.visual_preview_path)
        scene.render.engine = eevee_engine_name()
        scene.render.image_settings.color_depth = "8"

        physics_material = bpy.data.materials.new("Pipeline Physics Preview Material")
        temporary_data.append(physics_material)
        physics_material.diffuse_color = (0.08, 0.35, 0.8, 1.0)
        physics_material.use_nodes = True
        physics_shader = physics_material.node_tree.nodes.get("Principled BSDF")
        physics_shader.inputs["Base Color"].default_value = (0.04, 0.25, 0.8, 1.0)
        physics_shader.inputs["Metallic"].default_value = 0.05
        physics_shader.inputs["Roughness"].default_value = 0.32

        for target, path, override in (
            (visual, args.visual_preview_path, None),
            (physics, args.physics_preview_path, physics_material),
        ):
            target.hide_render = False
            center, size = world_bounds(target)
            direction = Vector((1.4, -1.7, 1.15)).normalized()
            camera.location = center + direction * size * 3.0
            camera_data.ortho_scale = size * 1.45
            camera_data.lens = 52.0
            camera_data.clip_start = max(size / 1000.0, 1.0e-5)
            camera_data.clip_end = max(size * 20.0, 10.0)
            point_camera(camera, center)

            key = make_area_light(
                "Pipeline Preview Key",
                center + Vector((1.5, -2.0, 2.2)).normalized() * size * 3.0,
                center,
                900.0,
                size * 2.0,
            )
            fill = make_area_light(
                "Pipeline Preview Fill",
                center + Vector((-2.0, -0.5, 1.0)).normalized() * size * 2.5,
                center,
                500.0,
                size * 2.5,
            )
            temporary_objects.extend((key, fill))
            temporary_data.extend((key.data, fill.data))
            key.hide_render = False
            fill.hide_render = False
            camera.hide_render = False
            bpy.context.view_layer.material_override = override
            scene.render.filepath = str(path)
            log(f"Rendering {target.name} preview: {path}")
            bpy.ops.render.render(write_still=True)
            target.hide_render = True
            delete_object(key)
            delete_object(fill)
            temporary_objects.remove(key)
            temporary_objects.remove(fill)
    finally:
        bpy.context.view_layer.material_override = old_override
        scene.camera = old_camera
        scene.world = old_world
        scene.render.engine = old_engine
        scene.render.filepath = old_filepath
        for obj, hidden in hidden_state.items():
            if obj.name in bpy.data.objects:
                obj.hide_render = hidden
        for obj in temporary_objects:
            delete_object(obj)
        for datablock in temporary_data:
            if isinstance(datablock, bpy.types.Camera) and datablock.name in bpy.data.cameras:
                bpy.data.cameras.remove(datablock)
            elif isinstance(datablock, bpy.types.World) and datablock.name in bpy.data.worlds:
                bpy.data.worlds.remove(datablock)
            elif isinstance(datablock, bpy.types.Material) and datablock.name in bpy.data.materials:
                bpy.data.materials.remove(datablock)
            elif isinstance(datablock, bpy.types.Light) and datablock.name in bpy.data.lights:
                bpy.data.lights.remove(datablock)


def save_processed_blend(path: Path) -> None:
    ensure_object_mode()
    log(f"Saving processed blend: {path}")
    result = bpy.ops.wm.save_as_mainfile(filepath=str(path), check_existing=False)
    if "FINISHED" not in result:
        raise RuntimeError(f"Could not save processed blend: {path}")


def image_format_extension(image: bpy.types.Image) -> str:
    extensions = {
        "BMP": ".bmp",
        "JPEG": ".jpg",
        "JPEG2000": ".jp2",
        "OPEN_EXR": ".exr",
        "PNG": ".png",
        "TARGA": ".tga",
        "TIFF": ".tif",
        "WEBP": ".webp",
    }
    extension = extensions.get(image.file_format)
    if extension is not None:
        return extension
    return Path(bpy.path.abspath(image.filepath)).suffix.lower() or ".png"


def image_texture_role(node: bpy.types.Node) -> str:
    """Infer a stable filename role from an image node's downstream sockets."""
    roles = set()
    pending = list(node.outputs)
    visited_nodes = {node}
    while pending:
        output = pending.pop()
        for link in output.links:
            destination = link.to_node
            if destination.type == "BSDF_PRINCIPLED":
                input_name = link.to_socket.name.lower()
                if any(term in input_name for term in ("base color", "diffuse", "color")):
                    roles.add("color")
                elif "normal" in input_name:
                    roles.add("normal")
                elif "rough" in input_name:
                    roles.add("roughness")
                elif "metal" in input_name:
                    roles.add("metallic")
                elif "occlusion" in input_name:
                    roles.add("occlusion")
                else:
                    roles.add(re.sub(r"[^a-z0-9]+", "_", input_name).strip("_"))
            elif destination not in visited_nodes:
                visited_nodes.add(destination)
                pending.extend(destination.outputs)

    if {"occlusion", "roughness", "metallic"}.issubset(roles):
        return "orm"
    if {"roughness", "metallic"}.issubset(roles):
        return "metallic_roughness"
    return "_".join(sorted(roles)) if roles else "texture"


def write_image_source(image: bpy.types.Image, destination: Path) -> None:
    """Write the exact packed/external pixels used by a Blender image."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if image.packed_file is not None:
        destination.write_bytes(image.packed_file.data)
        return

    source = Path(bpy.path.abspath(image.filepath)).resolve()
    if source.is_file():
        if source != destination.resolve():
            shutil.copy2(source, destination)
        return

    if not image.has_data:
        raise RuntimeError(f"Image {image.name!r} has no packed data or readable source file.")
    previous_path = image.filepath_raw
    try:
        image.filepath_raw = str(destination)
        image.save()
    finally:
        image.filepath_raw = previous_path


def stage_unique_visual_textures(visual: bpy.types.Object, args: argparse.Namespace) -> None:
    """Give every Pipeline 1 texture an object-scoped filename before USD export."""
    entries: list[tuple[bpy.types.Image, str]] = []
    seen_images = set()
    for material_slot in visual.material_slots:
        material = material_slot.material
        if material is None or not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE" or node.image is None or node.image in seen_images:
                continue
            seen_images.add(node.image)
            entries.append((node.image, image_texture_role(node)))

    if not entries:
        raise RuntimeError(f"Pipeline 1 visual object {visual.name!r} has no image textures.")

    role_counts: dict[str, int] = {}
    for image, role in entries:
        role_counts[role] = role_counts.get(role, 0) + 1
        filename = (
            f"{args.asset_name}_{role}_{role_counts[role]}{image_format_extension(image)}"
        )
        destination = args.texture_dir / filename
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Texture output already exists (pass --overwrite to replace): {destination}"
            )
        write_image_source(image, destination)
        image.name = destination.stem
        image.filepath = "//" + os.path.relpath(destination, args.output_blend.parent)
        log(f"Staged visual texture: {destination}")


def select_export_hierarchy(root: bpy.types.Object) -> None:
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")

    def visit(obj: bpy.types.Object) -> None:
        obj.select_set(True)
        for child in obj.children:
            visit(child)

    visit(root)
    bpy.context.view_layer.objects.active = root


def export_usd(root: bpy.types.Object, args: argparse.Namespace) -> None:
    select_export_hierarchy(root)
    log(f"Exporting USD: {args.output_usd}")
    result = bpy.ops.wm.usd_export(
        filepath=str(args.output_usd),
        selected_objects_only=True,
        export_animation=False,
        export_uvmaps=True,
        export_normals=True,
        export_materials=True,
        generate_preview_surface=True,
        export_textures_mode=args.export_textures_mode,
        overwrite_textures=args.overwrite,
        relative_paths=True,
        root_prim_path="/",
        export_meshes=True,
        export_lights=False,
        export_cameras=False,
        export_curves=False,
        export_points=False,
        export_volumes=False,
        triangulate_meshes=False,
        convert_scene_units="METERS",
        meters_per_unit=1.0,
        author_blender_name=False,
        merge_parent_xform=True,
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"USD export failed: {args.output_usd}")


def configure_usd(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """Add collision/material metadata and return visual/physics mesh paths."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(str(args.output_usd))
    if stage is None:
        raise RuntimeError(f"Could not reopen exported USD: {args.output_usd}")
    root_prim = stage.GetPrimAtPath("/root")
    if not root_prim.IsValid():
        roots = stage.GetPseudoRoot().GetChildren()
        root_prim = next((prim for prim in roots if prim.GetName().lower() == "root"), None)
    if root_prim is None or not root_prim.IsValid():
        raise RuntimeError("Exported USD does not contain the expected /root prim.")
    stage.SetDefaultPrim(root_prim)

    # Blender exports its shared material library as a sibling of /root. Because
    # Isaac Lab references only the default prim, bindings from /root meshes to
    # /_materials are outside the reference's composition scope and get ignored.
    # Moving the library below the default prim keeps the materials and lets USD
    # retarget all dependent relationships/connections atomically.
    exported_materials = stage.GetPrimAtPath("/_materials")
    if exported_materials.IsValid():
        destination = root_prim.GetPath().AppendChild("_materials")
        if stage.GetPrimAtPath(destination).IsValid():
            raise RuntimeError(
                f"Cannot move /_materials to {destination}: destination already exists."
            )
        namespace_editor = Usd.NamespaceEditor(stage)
        if not namespace_editor.MovePrimAtPath(exported_materials.GetPath(), destination):
            raise RuntimeError(f"Could not schedule material move to {destination}.")
        if not namespace_editor.CanApplyEdits():
            raise RuntimeError(f"Cannot move exported materials to {destination}.")
        if not namespace_editor.ApplyEdits():
            raise RuntimeError(f"Could not move exported materials to {destination}.")

    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rigid_body_api.CreateRigidBodyEnabledAttr(True)
    rigid_body_api.CreateKinematicEnabledAttr(False)
    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.CreateMassAttr(args.mass)

    visual_paths: list[str] = []
    physics_prims = []
    root_path = root_prim.GetPath().pathString
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        relative_parts = prim.GetPath().pathString[len(root_path) :].lower().split("/")
        if "physics" in relative_parts:
            physics_prims.append(prim)
        elif "visual" in relative_parts:
            visual_paths.append(prim.GetPath().pathString)

    if not visual_paths or not physics_prims:
        raise RuntimeError(
            "USD hierarchy check failed: expected at least one mesh below both /root/visual "
            "and /root/physics."
        )

    physics_branch = stage.GetPrimAtPath(f"{root_path}/physics")
    if physics_branch.IsValid():
        UsdGeom.Imageable(physics_branch).MakeInvisible()

    material_path = root_prim.GetPath().AppendChild("PhysicsMaterial")
    physics_material = UsdShade.Material.Define(stage, material_path)
    material_api = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    material_api.CreateStaticFrictionAttr(args.static_friction)
    material_api.CreateDynamicFrictionAttr(args.dynamic_friction)
    material_api.CreateRestitutionAttr(args.restitution)
    material_api.CreateDensityAttr(args.density)

    physics_paths: list[str] = []
    for prim in physics_prims:
        physics_paths.append(prim.GetPath().pathString)
        UsdGeom.Imageable(prim).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
            args.physics_approximation
        )
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(
            physics_material,
            bindingStrength=UsdShade.Tokens.weakerThanDescendants,
            materialPurpose="physics",
        )

    stage.GetRootLayer().Save()
    return sorted(visual_paths), sorted(physics_paths)


def main() -> None:
    args = build_parser().parse_args(blender_script_args())
    resolve_outputs(args)
    log(f"Opening source blend: {args.blend_file}")
    bpy.ops.wm.open_mainfile(filepath=str(args.blend_file), load_ui=False, use_scripts=False)

    source = consolidate_source(source_meshes(args), args.asset_name)
    root = new_empty("root")
    original_branch = new_empty("original", root)
    visual_branch = new_empty("visual", root)
    physics_branch = new_empty("physics", root)
    reparent_keep_world(source, original_branch)
    visual = duplicate_mesh(
        source,
        f"{args.asset_name}_visual",
        f"{args.asset_name}_visual_mesh",
        visual_branch,
    )
    physics = duplicate_mesh(
        source,
        f"{args.asset_name}_physics",
        f"{args.asset_name}_physics_mesh",
        physics_branch,
    )

    if args.pipeline == 1:
        log("Running pipeline 1: direct textured decimation")
        decimate_visual(source, args)
        if args.delete_loose:
            delete_loose_geometry(source)
        delete_mesh_object(visual)
        reparent_keep_world(source, visual_branch)
        source.name = f"{args.asset_name}_visual"
        source.data.name = f"{args.asset_name}_visual_mesh"
        delete_object(original_branch)
        original_branch = None
        visual = source
        original = None
    else:
        log("Running pipeline 2: visual remesh, decimation, UVs, and rebake")
        original = source
        visual.data.materials.clear()
        remesh_to_faces(
            visual,
            args.visual_remesh_faces,
            args.visual_remesh_voxel_size,
            args.visual_remesh_adaptivity,
            args.visual_remesh_remove_disconnected,
            args.remesh_search_steps,
            args.remesh_target_tolerance,
            "visual",
        )
        decimate_visual(visual, args)
        smart_uv_project(visual, args)
        material_data = create_bake_material(visual, args)
        bake_maps(original, visual, *material_data, args)

    log("Building material-free voxel physics mesh")
    physics.data.materials.clear()
    remesh_to_faces(
        physics,
        args.physics_remesh_faces,
        args.physics_remesh_voxel_size,
        args.physics_remesh_adaptivity,
        args.physics_remesh_remove_disconnected,
        args.remesh_search_steps,
        args.remesh_target_tolerance,
        "physics",
    )
    decimate_to_faces(physics, args.physics_decimate_faces, "physics")

    if args.pipeline == 1:
        stage_unique_visual_textures(visual, args)

    render_previews(visual, physics, args)
    save_processed_blend(args.output_blend)

    if args.pipeline == 2:
        # The saved blend intentionally keeps the high-poly original for future rebakes.
        delete_mesh_object(original)
        delete_object(original_branch)

    export_usd(root, args)
    visual_paths, physics_paths = configure_usd(args)
    log("Done")
    log(f"Blend: {args.output_blend}")
    log(f"USD: {args.output_usd}")
    log(f"Visual preview: {args.visual_preview_path}")
    log(f"Physics preview: {args.physics_preview_path}")
    if args.pipeline == 2:
        log(f"Color texture: {args.color_path}")
        log(f"Roughness texture: {args.roughness_path}")
    log(f"USD visual meshes: {', '.join(visual_paths)}")
    log(f"USD physics meshes: {', '.join(physics_paths)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[object-pipeline] ERROR: {exc}", file=sys.stderr, flush=True)
        raise
