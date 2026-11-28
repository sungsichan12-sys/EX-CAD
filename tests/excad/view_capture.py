# SPDX-License-Identifier: GPL-2.0-or-later
"""Round-trip Blender's actual camera RGB and Z passes without changing the user's scene."""

import json
import os
from pathlib import Path
import tempfile

import bpy
import numpy as np

from excad_addon.ai_bridge import AiBridgeServer
from excad_addon.feature_tree import _solid
from excad_addon.polyhedron_engine import _load_rgba_and_mask
from excad_addon.uv_bake import read_depth
from excad_addon.view_capture import _publish


bridge = AiBridgeServer.instance()


def run(cmd, **fields):
    reply = bridge._dispatch_dict(dict(cmd=cmd, **fields))
    assert reply["ok"], (cmd, reply)
    return reply


def database():
    return {key: set(getattr(bpy.data, key).keys()) for key in ("scenes", "node_groups", "objects", "cameras", "meshes", "collections")}


scene = bpy.context.scene
scene.render.engine = os.getenv("EXCAD_CAPTURE_ENGINE", "CYCLES")
scene.cycles.samples = 1
scene.cycles.use_denoising = False
scene.display.shading.light = "FLAT"
scene.display.shading.color_type = "MATERIAL"
scene.display.shading.show_specular_highlight = False
scene.render.threads_mode, scene.render.threads = "FIXED", 2
scene.view_settings.view_transform = "Standard"
cube = scene.objects["Cube"]
cube.scale = (0.4, 0.3, 0.2)
center = (0.15, -0.2, 0.4)
cube.location = center
material = bpy.data.materials.new("Emission registration")
material.use_nodes = True
material.diffuse_color = (0.16, 0.05, 0.65, 1)
nodes = material.node_tree.nodes
emission = nodes.new("ShaderNodeEmission")
emission.inputs["Color"].default_value = (0.16, 0.05, 0.65, 1)
material.node_tree.links.new(emission.outputs[0], nodes.get("Material Output").inputs["Surface"])
cube.data.materials.clear()
cube.data.materials.append(material)
args = dict(poly_type="CUBE_6", size=2, center=center)

run("poly_create_rig", **args)
rig = bpy.data.collections["EXCAD_Polyhedron_Rig"]
marker = bpy.data.objects.new("User alignment note", None)
rig.objects.link(marker)
camera = scene.camera = rig.objects["EXCAD_Cam_02"]
cage = rig.objects["EXCAD_Poly_Cage"]
assert cage.hide_render
run("poly_create_rig", poly_type="DODECAHEDRON_12", size=2, center=center)
assert len(cage.data.polygons) == 12
run("poly_create_rig", **args)
assert rig.objects[marker.name] == marker and scene.camera == camera and len(cage.data.polygons) == 6
assert np.allclose(run("render_inspect", size=64)["bbox"], [np.array(center) - cube.scale, np.array(center) + cube.scale])
# Verify settings and output nodes belonging to the active scene survive captures.
scene.render.pixel_aspect_x = 2
scene.render.resolution_x = 387
scene.render.resolution_y = 219
scene.render.image_settings.file_format = "JPEG"
scene.render.image_settings.quality = 72
scene.render.use_border = True
group = bpy.data.node_groups.new("User composite", "CompositorNodeTree")
scene.compositing_node_group = group
scene.timeline_markers.new("Camera cut", frame=1).camera = camera
state = database()
digest = run("scene_digest", include_all=True)["h"]

with tempfile.TemporaryDirectory() as tmp:
    directory = Path(tmp)
    opts = dict(args, output_dir=tmp, img_res=80)
    report = run("poly_render", write_depth=True, **opts)
    manifest = json.loads(Path(report["manifest"]).read_text(encoding="utf8"))
    assert report["views"] == 6 and manifest["write_depth"]
    assert database() == state and scene.camera == camera and scene.compositing_node_group == group
    assert run("scene_digest", since_hash=digest, include_all=True).get("unchanged")
    assert scene.timeline_markers[0].camera == camera and bpy.context.scene == scene
    assert (scene.render.resolution_x, scene.render.resolution_y, scene.render.pixel_aspect_x) == (387, 219, 2)
    assert scene.render.use_border and scene.render.image_settings.file_format == "JPEG" and scene.render.image_settings.quality == 72
    for face in manifest["views"]:
        image = directory / face["filename"]
        rgb, mask = _load_rgba_and_mask(str(image))
        depth = read_depth(image.with_name(image.stem + "_depth.png"))
        expected = 0.5 - np.dot(np.abs(face["normal"]), cube.scale) / 2
        assert mask[40, 40] and not mask[0, 0] and np.allclose(rgb[40, 40], (0.16, 0.05, 0.65), atol=0.012)
        assert abs(depth[40, 40] - expected) < 0.0001 and depth[0, 0] == 1.0, (face, depth[40, 40], expected)
    model = run("poly_reconstruct", image_dir=tmp, name="Round trip", resolution=56, texture_size=256, **args)
    obj = bpy.data.objects[model["name"]]
    _solid(obj.data)
    assert np.allclose(obj.dimensions, 2 * np.array(cube.scale), atol=0.04), obj.dimensions[:]

    # Recapturing without depth removes only the sidecars previously produced by EX-CAD.
    note = directory / "calibration.txt"
    note.write_text("reference", encoding="utf8")
    run("poly_create_rig", output_dir=tmp, **args)
    run("poly_render", write_depth=False, **opts)
    assert not list(directory.glob("face_*_depth.png")) and note.read_text(encoding="utf8") == "reference"
    previous = (directory / "face_00.png").read_bytes()
    # A late filesystem error restores previously replaced images and leaves the directory intact.
    with tempfile.TemporaryDirectory(dir=tmp) as other:
        staging = Path(other)
        (staging / "face_00.png").write_bytes(b"new capture")
        (staging / "zz_locked.png").write_bytes(b"blocked")
        manifest_file = directory / "views_manifest.json"
        (staging / "views_manifest.json").write_bytes(manifest_file.read_bytes())
        (directory / "zz_locked.png").mkdir()
        try:
            _publish(staging, directory)
        except IsADirectoryError:
            pass
        else:
            raise AssertionError("Expected the occupied destination to reject the capture")
        assert (directory / "face_00.png").read_bytes() == previous and (directory / "face_05.png").is_file()
other = bpy.data.scenes.new("Second shoot")
other.collection.children.link(rig)
other.camera = camera
run("poly_create_rig", poly_type="OCTAHEDRON_8", size=3, center=center)
assert other.camera == camera and other.camera.data.ortho_scale == 2 and len(cage.data.polygons) == 6
assert scene.camera != camera and scene.camera.data.ortho_scale == 3 and scene.timeline_markers[0].camera == scene.camera
assert other.objects[marker.name] == scene.objects[marker.name]
assert len(bridge._dispatch('{"cmd":"schema"}').encode()) <= 1200
print("CAMERA COLOR AND METRIC DEPTH PASS", bpy.app.version_string, scene.render.engine)
