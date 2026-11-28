# SPDX-License-Identifier: GPL-2.0-or-later
"""Generate real multi-part GLBs, normalize and skin them, then pose and reimport an authored rig."""

import tempfile
from pathlib import Path

import bmesh
import bpy
import numpy as np

from excad_addon import ai_3d_provider as providers


def ellipsoid(name, center, scale, parent=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8)
    obj = bpy.context.object
    obj.name = name
    obj.parent = parent
    obj.location, obj.scale = center, scale
    return obj


def export(path, objects, animated=False):
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(path), use_selection=True, export_format="GLB", export_animations=animated)


def coordinates(obj):
    graph = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(graph)
    return np.array([tuple(ev.matrix_world @ vertex.co) for vertex in ev.data.vertices])


def weights(obj, rig):
    groups = {g.index for g in obj.vertex_groups if g.name in rig.data.bones and rig.data.bones[g.name].use_deform}
    total = [sum(g.weight for g in v.groups if g.group in groups) for v in obj.data.vertices]
    assert np.allclose(total, 1, atol=0.002), (obj.name, min(total), max(total))


for obj in list(bpy.context.scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
root = bpy.data.objects.new("Imported pivot", None)
bpy.context.collection.objects.link(root)
root.location = (3.2, -1.1, 0.4)
objects = [root]
objects.append(ellipsoid("Torso", (0, 0, 1.06), (0.26, 0.15, 0.41), root))
objects.append(ellipsoid("Skull", (0, -0.015, 1.65), (0.16, 0.14, 0.20), root))
for side in (-1, 1):
    objects.append(ellipsoid(f"Leg {side}", (side * 0.135, 0, 0.46), (0.115, 0.12, 0.46), root))
    objects.append(ellipsoid(f"Arm {side}", (side * 0.34, 0, 1.10), (0.085, 0.095, 0.31), root))
with tempfile.TemporaryDirectory() as tmp:
    unskinned = Path(tmp, "mannequin.glb")
    export(unskinned, objects)
    for obj in reversed(objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    model = providers.import_model(unskinned, name="Shared rig", height=0.12, rig="HUMANOID")
    parts = [bpy.data.objects[n] for n in model["meshes"]]
    rigs = {p.find_armature() for p in parts}
    assert len(parts) == 6 and len(rigs) == 1 and None not in rigs, (len(parts), rigs)
    rig = rigs.pop()
    assert bpy.data.objects[model["name"]] == rig
    low, high = providers._dimensions(parts)
    assert abs(low.z) < 1e-5 and abs(high.z - 0.12) < 1e-5
    assert abs(rig.data.bones["head"].tail_local.z - 0.12 * 0.99) < 1e-4
    for part in parts:
        weights(part, rig)
    head = next(obj for obj in parts if obj.name.startswith("Skull"))
    leg = next(obj for obj in parts if obj.name.startswith("Leg"))
    rest_head, rest_leg = coordinates(head), coordinates(leg)
    bone = rig.pose.bones["head"]
    bone.rotation_mode = "XYZ"
    bone.rotation_euler.y = 0.5
    bpy.context.view_layer.update()
    assert np.max(np.linalg.norm(coordinates(head) - rest_head, axis=1)) > 0.001
    assert np.max(np.abs(coordinates(leg) - rest_leg)) < 1e-5
    bone.rotation_euler = (0, 0, 0)
    bpy.context.view_layer.update()

    # An unbound accessory beside an authored skin should follow the authored armature on import.
    cuff = ellipsoid("Headband", (0, 0, 0.109), (0.011, 0.009, 0.002))
    skinned = Path(tmp, "authored.glb")
    export(skinned, [*rig.children_recursive, rig, cuff])
    imported = providers.import_model(skinned, name="Authored character", height=0.38, rig="HUMANOID")
    meshes = [bpy.data.objects[n] for n in imported["meshes"]]
    bones = {m.find_armature() for m in meshes}
    assert len(meshes) == 7 and len(bones) == 1 and None not in bones, [
        (m.name, m.find_armature().name if m.find_armature() else None) for m in meshes
    ]
    original = bones.pop()
    assert original != rig and original.pose.bones.get("head")
    for part in meshes:
        weights(part, original)
    accessory = next(m for m in meshes if m.name.startswith("Headband"))
    rest = coordinates(accessory)
    original.pose.bones["head"].rotation_mode = "XYZ"
    original.pose.bones["head"].rotation_euler.x = 0.4
    bpy.context.view_layer.update()
    assert np.max(np.abs(coordinates(accessory) - rest)) > 0.001
    prop = ellipsoid("Animated source", (0, 0, 2), (0.4, 0.5, 1))
    prop.keyframe_insert(data_path="location", frame=1)
    prop.location.x = 2
    prop.keyframe_insert(data_path="location", frame=5)
    bpy.context.scene.frame_set(1)
    animated = Path(tmp, "animated.glb")
    export(animated, [prop], animated=True)
    model = providers.import_model(animated, name="Moving object", height=0.3)
    piece = bpy.data.objects[model["meshes"][0]]
    measured = []
    for frame in (1, 5, 1):
        bpy.context.scene.frame_set(frame)
        low, high = providers._dimensions([piece])
        assert abs(low.z) < 1e-5 and abs(high.z - 0.3) < 1e-5
        measured.append((low.x + high.x) / 2)
    assert np.allclose(measured, (0, 0.3, 0), atol=1e-4), measured
    # A texture discontinuity shares one geometric edge. Removing a face
    # creates a real opening which the GLB round-trip must keep.
    from excad_addon.scene_mesh import read_surface

    ball = ellipsoid("UV chart sphere", (2, 0, 1), (0.3, 0.3, 0.3))
    for poly in ball.data.polygons:
        poly.use_smooth = True
    for opened in (False, True):
        if opened:
            bm = bmesh.new()
            bm.from_mesh(ball.data)
            face = next(f for f in bm.faces if len(f.verts) == 4)
            bmesh.ops.delete(bm, geom=[face], context="FACES")
            bm.to_mesh(ball.data)
            bm.free()
        path = Path(tmp, "uv-charts.glb")
        export(path, [ball])
        result = providers.import_model(path, height=0.6, name="Open chart" if opened else "Closed chart")
        obj = bpy.data.objects[result["meshes"][0]]
        surface = read_surface(obj, bpy.context.evaluated_depsgraph_get())
        assert surface.boundary_edges == (4 if opened else 0) and surface.nonmanifold_edges == 0
        layer = obj.data.uv_layers.active
        by_vertex = {}
        for corner in obj.data.loops:
            by_vertex.setdefault(corner.vertex_index, []).append(layer.uv[corner.index].vector.x)
        assert any(max(us) - min(us) > 0.95 for us in by_vertex.values())
print("HUMANOID GLB PASS", bpy.app.version_string)
