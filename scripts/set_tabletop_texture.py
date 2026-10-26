#!/usr/bin/env python3
"""Apply/restore the reference charcoal tabletop texture with byte-exact backups.

Run using ~/env_isaaclab_51/bin/python scripts/set_tabletop_texture.py apply|restore|status.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "source/so101_bench/so101_bench/assets/usd/backgrounds"
ARCHIVE = REPO / "outputs/tabletop_texture_20260915"
MANIFEST = ARCHIVE / "manifest.json"
IDS = (1, 2, 3, 4, 9, 10, 11, 16)
TEXTURE = "textures/tabletop_charcoal_plastic_v1.png"
MATERIAL = "/root/_materials/Material_004"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def usd_modules():
    # Load the USD libraries bundled with the existing Isaac Lab environment.
    if not os.environ.get("SO101_TABLETOP_USD_READY"):
        isaac = Path(sys.prefix) / "lib/python3.11/site-packages/isaacsim"
        libs = next((isaac / "extscache").glob("omni.usd.libs-*"))
        env = dict(os.environ, SO101_TABLETOP_USD_READY="1")
        env["PYTHONPATH"] = str(libs) + os.pathsep + env.get("PYTHONPATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            (str(libs / "bin"), str(isaac / "kit"), env.get("LD_LIBRARY_PATH", ""))
        )
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    from pxr import Sdf, Usd, UsdShade
    return Sdf, Usd, UsdShade


def replace_copy(source, destination):
    temporary = destination.with_suffix(destination.suffix + ".tabletop-tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "restore", "status", "revise"))
    parser.add_argument("--texture", type=Path, help="New PNG inside the background asset directory, for revise.")
    args = parser.parse_args()
    if args.action == "revise":
        if args.texture is None:
            parser.error("revise requires --texture")
        Sdf, Usd, UsdShade = usd_modules()
    if not MANIFEST.exists():
        if args.action != "apply":
            raise SystemExit("No tabletop backup manifest exists yet.")
        Sdf, Usd, UsdShade = usd_modules()
        assert (ASSETS / TEXTURE).is_file(), "Missing generated texture"
        originals, updated = ARCHIVE / "originals", ARCHIVE / "updated"
        originals.mkdir(parents=True, exist_ok=True)
        updated.mkdir(parents=True, exist_ok=True)
        records = []
        # Prepare and verify every asset before replacing any live asset.
        for index in IDS:
            name = f"background_{index}.usdc"
            original = originals / name
            if original.exists():
                if digest(original) != digest(ASSETS / name):
                    raise RuntimeError(f"Unmanifested backup differs; refusing to overwrite {original}")
            else:
                shutil.copy2(ASSETS / name, original)
            stage = Usd.Stage.Open(str(ASSETS / name))
            material = UsdShade.Material(stage.GetPrimAtPath(MATERIAL))
            mesh = stage.GetPrimAtPath("/root/root/table_visual/node_0_003")
            assert material and mesh
            assert UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0].GetPath() == material.GetPath()
            assert not mesh.GetAttribute("primvars:Col").HasAuthoredValueOpinion()
            shader = UsdShade.Shader(stage.GetPrimAtPath(MATERIAL + "/Principled_BSDF"))
            original_layer = stage.GetRootLayer().ExportToString()
            uv = UsdShade.Shader.Define(stage, MATERIAL + "/TabletopUV")
            uv.CreateIdAttr("UsdPrimvarReader_float2")
            uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            mapping = UsdShade.Shader.Define(stage, MATERIAL + "/TabletopTextureScale")
            mapping.CreateIdAttr("UsdTransform2d")
            mapping.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
            mapping.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set((16.0, 16.0))
            mapping.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            texture = UsdShade.Shader.Define(stage, MATERIAL + "/TabletopColor")
            texture.CreateIdAttr("UsdUVTexture")
            texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./" + TEXTURE))
            # Auto treats this 8-bit RGB color PNG as sRGB. It also keeps the
            # room-specific sRGB brightness boosts (3x/10x) off this new table map.
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("auto")
            texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(mapping.ConnectableAPI(), "result")
            texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            diffuse = shader.GetInput("diffuseColor")
            old_connections = diffuse.GetAttr().GetConnections()
            diffuse.ConnectToSource(texture.ConnectableAPI(), "rgb")
            stage.GetRootLayer().Export(str(updated / name))
            # Removing just our connection and three nodes must reproduce the original layer exactly.
            diffuse.GetAttr().SetConnections(old_connections)
            stage.RemovePrim(texture.GetPath())
            stage.RemovePrim(mapping.GetPath())
            stage.RemovePrim(uv.GetPath())
            assert stage.GetRootLayer().ExportToString() == original_layer, name
            records.append({"name": name, "original": digest(original), "updated": digest(updated / name)})
        MANIFEST.write_text(json.dumps({"texture": TEXTURE, "texture_sha256": digest(ASSETS / TEXTURE), "files": records}, indent=2) + "\n")

    manifest = json.loads(MANIFEST.read_text())
    records = manifest["files"]
    # Validate the whole transaction before changing anything; protect subsequent user edits.
    for record in records:
        current = digest(ASSETS / record["name"])
        if current not in (record["original"], record["updated"]):
            raise SystemExit(f"{record['name']} has newer edits; refusing to overwrite it.")
        for version, folder in (("original", "originals"), ("updated", "updated")):
            assert digest(ARCHIVE / folder / record["name"]) == record[version], "Backup checksum mismatch"
    if args.action in ("apply", "revise"):
        assert digest(ASSETS / manifest["texture"]) == manifest["texture_sha256"], "Texture checksum mismatch"
    if args.action == "revise":
        texture_path = args.texture.resolve()
        relative_texture = texture_path.relative_to(ASSETS.resolve()).as_posix()
        assert texture_path.is_file() and texture_path.suffix.lower() == ".png"
        revision = ARCHIVE / "revisions" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        revision.mkdir(parents=True)
        shutil.copy2(MANIFEST, revision / "manifest.json")
        shutil.copytree(ARCHIVE / "updated", revision / "updated")
        pending = revision / "replacement"
        pending.mkdir()
        for record in records:
            layer = Sdf.Layer.FindOrOpen(str(ARCHIVE / "updated" / record["name"]))
            before = layer.ExportToString()
            attribute = layer.GetAttributeAtPath(MATERIAL + "/TabletopColor.inputs:file")
            old_texture = attribute.default
            attribute.default = Sdf.AssetPath("./" + relative_texture)
            layer.Export(str(pending / record["name"]))
            attribute.default = old_texture
            assert layer.ExportToString() == before
            record["updated"] = digest(pending / record["name"])
        manifest["texture"] = relative_texture
        manifest["texture_sha256"] = digest(texture_path)
        # Retain the preceding version before publishing the new manifest/assets.
        for record in records:
            replace_copy(pending / record["name"], ARCHIVE / "updated" / record["name"])
        manifest_tmp = MANIFEST.with_suffix(".tmp")
        manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest_tmp.replace(MANIFEST)
        args.action = "apply"
    for record in records:
        destination = ASSETS / record["name"]
        if args.action != "status":
            folder = "updated" if args.action == "apply" else "originals"
            replace_copy(ARCHIVE / folder / record["name"], destination)
        state = "updated" if digest(destination) == record["updated"] else "original"
        print(f"{record['name']}: {state}")


if __name__ == "__main__":
    main()
