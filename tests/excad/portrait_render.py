# SPDX-License-Identifier: GPL-2.0-or-later
"""Open the character.py dressed.blend fixture and verify its headshot in native Cycles."""

import os
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix

from excad_addon.ai_bridge import AiBridgeServer


def run(cmd, **args):
    reply = AiBridgeServer.instance()._dispatch_dict(dict(cmd=cmd, **args))
    assert reply["ok"], (cmd, reply)
    return reply


def eye_frame():
    graph = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene
    coordinates = []
    for obj in scene.objects:
        if obj.get("excad_human") != scene.objects["Portrait"] or "excad_eye_side" not in obj:
            continue
        ev = obj.evaluated_get(graph)
        first, last = ev.data.vertices[0].co, ev.data.vertices[-1].co
        center = ev.matrix_world @ ((first + last) / 2)
        coordinates.append(world_to_camera_view(scene, scene.camera, center)[:2])
    assert len(coordinates) == 2
    midpoint = np.mean(coordinates, axis=0)
    assert 0.39 < midpoint[0] < 0.61 and 0.59 < midpoint[1] < 0.79, midpoint
    return np.sort(coordinates, axis=0)


scene = bpy.context.scene
subject = scene.objects["Portrait"]
run("character_stylize", object=subject.name, expression="SERIOUS")
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 600, 704, 100
start = run("scene_digest")["h"]
lights = run("light_studio", target=subject.name, style="HEADSHOT")
assert scene.camera.name == lights["camera"] and len(lights["lights"]) == 3
assert {row[0] for row in run("scene_digest", since_hash=start)["changed"]} >= {lights["camera"], *lights["lights"]}
bpy.context.view_layer.update()
original = eye_frame()
run("checkpoint", name="Studio composition")
rig = subject.find_armature()
rig.matrix_world = Matrix.Translation((2, -1.6, 0.4)) @ Matrix.Rotation(0.59, 4, "Z") @ rig.matrix_world
run("oneshot_pose", object=subject.name, pose="HEROIC")
again = run("light_studio", target=subject.name, style="HEADSHOT")
assert again == lights
bpy.context.view_layer.update()
assert np.max(np.abs(original - eye_frame())) < 0.025
run("rollback", name="Studio composition")
scene = bpy.context.scene
assert scene.camera.name == lights["camera"]
assert np.allclose(eye_frame(), original, atol=1e-5)

world = scene.world or bpy.data.worlds.new("Portrait background")
scene.world = world
world.use_nodes = True
background = world.node_tree.nodes.get("Background")
background.inputs["Color"].default_value = (0.085, 0.12, 0.155, 1)
background.inputs["Strength"].default_value = 0.32
subject = scene.objects["Portrait"]
surface = next((m for m in subject.modifiers if m.type == "SUBSURF"), None)
if surface is None:
    surface = subject.modifiers.new("Portrait surface detail", "SUBSURF")
surface.levels, surface.render_levels = 1, 2
scene.render.engine = "CYCLES"
scene.cycles.device, scene.cycles.samples, scene.cycles.use_denoising = "CPU", 64, True
scene.render.film_transparent = False
scene.render.image_settings.file_format, scene.render.image_settings.color_mode = "PNG", "RGB"
out = Path(os.getenv("EXCAD_TEST_OUTPUT", bpy.app.tempdir)) / "phase2-portrait.png"
out.parent.mkdir(exist_ok=True, parents=True)
scene.render.filepath = str(out)
assert "FINISHED" in bpy.ops.render.render(write_still=True)
assert out.is_file() and out.stat().st_size > 20_000
print("PORTRAIT RENDER PASS", scene.camera.name, original.tolist(), out, flush=True)
