# SPDX-License-Identifier: GPL-2.0-or-later
"""Rebuild painted SDFs and dependent CAD features through the same bridge used by MCP."""

import json
import tempfile
from pathlib import Path

import bpy
import numpy as np

from excad_addon.ai_bridge import AiBridgeServer
from excad_addon.feature_tree import _solid


def run(cmd, **args):
    r = AiBridgeServer.instance()._dispatch_dict(dict(cmd=cmd, **args))
    assert r["ok"], (cmd, r)
    return r


def coords(mesh):
    xyz = np.empty((len(mesh.vertices), 3), np.float32)
    mesh.vertices.foreach_get("co", xyz.ravel())
    return xyz


def unit(mesh):
    xyz = coords(mesh)
    return (xyz - xyz.min(0)) / np.ptp(xyz, axis=0)


def empty(name):
    obj = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(obj)
    return obj


for obj in list(bpy.context.scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
tree = {"op": "mirror_x", "child": {"type": "ellipsoid", "radii": [0.38, 0.26, 0.40]}}
name = run("sdf_generate", name="Shade", tree=tree, resolution=36)["name"]
obj = bpy.data.objects[name]
obj.parent = empty("Fixture")
empty("Tag").parent = obj
obj.location, obj.rotation_euler, obj.scale = (1, -0.3, 2), (0.26, 0.34, -0.19), (1.2, 0.9, 1.1)
obj["serial"] = "surface test"
bpy.context.view_layer.update()
matrix = np.array(obj.matrix_world)
pos = unit(obj.data)
uv = obj.data.uv_layers.new(name="Transfer chart")
paint = obj.data.color_attributes.new("Lacquer", "FLOAT_COLOR", "POINT")
edge_color = obj.data.color_attributes.new("Stitch", "BYTE_COLOR", "CORNER")
# Attribute wrappers may be invalidated when Blender adds a layer.
paint = obj.data.color_attributes["Lacquer"]
for i, pt in enumerate(pos):
    paint.data[i].color = (pt[0], pt[2], 0.3, 1)
for loop in obj.data.loops:
    pt = pos[loop.vertex_index]
    uv.uv[loop.index].vector = (pt[0] * 0.46 + 0.22, pt[2] * 0.53 + 0.16)
    edge_color.data[loop.index].color = (pt[1], 0.32, 0.1, 1)
obj.data.color_attributes.active_color = paint
group = obj.vertex_groups.new(name="Hinge")
for v, p in zip(obj.data.vertices, pos):
    group.add([v.index], 0.2 + 0.65 * p[1], "REPLACE")
group.lock_weight = True
for mat in ("Enamel", "Rim"):
    obj.data.materials.append(bpy.data.materials.new(mat))
for f in obj.data.polygons:
    f.material_index = int(f.center.z > 0)
obj.material_slots[1].link = "OBJECT"
obj.material_slots[1].material = bpy.data.materials.new("Gold rim")
obj.active_material_index = 1
linked = obj.copy()
bpy.context.collection.objects.link(linked)
linked.name = "Original shade"
old_count = len(obj.data.vertices)
with tempfile.TemporaryDirectory() as tmp:
    path = str(Path(tmp) / "painted.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    obj = bpy.data.objects[name]
    before = run("scene_digest")["h"]
    run("checkpoint", name="Before radius")
    run(
        "batch",
        steps=[
            {
                "cmd": "sdf_generate",
                "object": name,
                "parameters": {"child.radii.1": 0.45, "child.radii.2": 0.31},
                "resolution": 54,
            },
            {"cmd": "sdf_generate", "object": name, "parameters": {"child.radii": [0.42, 0.45, 0.31]}},
        ],
    )
    _solid(obj.data)
    assert run("scene_digest", since_hash=before)["changed"]
    assert np.allclose(np.ptp(coords(obj.data), axis=0), [0.84, 0.9, 0.62], atol=0.006)
    assert (
        np.allclose(obj.matrix_world, matrix) and obj.parent.name == "Fixture" and bpy.data.objects["Tag"].parent == obj
    )
    assert obj["serial"] == "surface test" and bpy.data.objects["Original shade"].data != obj.data
    assert len(bpy.data.objects["Original shade"].data.vertices) == old_count != len(obj.data.vertices)
    assert obj.active_material_index == 1 and obj.material_slots[1].link == "OBJECT"
    assert obj.material_slots[1].material.name == "Gold rim" and obj.data.materials[1].name == "Rim"
    for f in obj.data.polygons:
        if abs(f.center.z) > 0.05:
            assert f.material_index == int(f.center.z > 0)
    pos = unit(obj.data)
    values = np.array([v.color[:2] for v in obj.data.color_attributes["Lacquer"].data])
    assert np.max(np.abs(values - pos[:, [0, 2]])) < 0.025
    assert obj.vertex_groups["Hinge"].lock_weight
    weights = [obj.vertex_groups["Hinge"].weight(i) for i in range(len(pos))]
    assert np.max(np.abs(weights - (0.2 + 0.65 * pos[:, 1]))) < 0.02
    loops = np.array([l.vertex_index for l in obj.data.loops])
    values = np.array([v.vector[:] for v in obj.data.uv_layers["Transfer chart"].uv])
    assert np.max(np.abs(values - (pos[loops][:, [0, 2]] * [0.46, 0.53] + [0.22, 0.16]))) < 0.015
    assert obj.data.color_attributes.active_color.name == "Lacquer" and "Stitch" in obj.data.color_attributes
    edited = run("scene_digest")["h"]
    run("sdf_generate", object=name)
    assert run("scene_digest", since_hash=edited).get("unchanged")
    objects, meshes = set(bpy.data.objects), set(bpy.data.meshes)
    for key, value in (("child.radii.1", -1), ("child.radii.15", 4), ("child.radii.0", "red"), ("child.type", 2)):
        assert not AiBridgeServer.instance()._dispatch_dict(
            dict(cmd="sdf_generate", object=name, parameters={key: value})
        )["ok"]
        assert run("scene_digest", since_hash=edited).get("unchanged")
        assert set(bpy.data.objects) == objects and set(bpy.data.meshes) == meshes
    recipe = json.loads(obj["excad_sdf_recipe"])
    assert recipe["resolution"] == 54
    run("sdf_generate", object=name, tree=dict(recipe["tree"], label="painted shade"))
    assert run("scene_digest", since_hash=edited)["changed"]
    run("undo")
    assert run("scene_digest", since_hash=edited).get("unchanged")
    run("rollback", name="Before radius")
    assert run("scene_digest", since_hash=before).get("unchanged")
    assert bpy.data.objects[name].data == bpy.data.objects["Original shade"].data

    # A face-attached CAD cut must survive a change in the underlying SDF dimensions.
    stock = run(
        "sdf_generate",
        name="Cast block",
        tree={"type": "round_box", "size": [1.4, 0.9, 0.54], "radius": 0.09},
        resolution=36,
    )["name"]
    run("sketch_on_face", object=stock, face="TOP", sketch_id="Mounting hole")
    run("sketch_add_circle", u=0, v=0, radius=0.14)
    run("extrude", object=stock, id="bore", op="CUT", depth=-0.9)
    block = bpy.data.objects[stock]
    assert not block.ray_cast((0, 0, 1), (0, 0, -1))[0]
    run("sdf_generate", object=stock, parameters={"size.0": 2.2})
    assert not block.ray_cast((0, 0, 1), (0, 0, -1))[0] and block.ray_cast((0.9, 0, 1), (0, 0, -1))[0]
    assert abs(np.ptp(coords(block.data), axis=0)[0] - 2.2) < 0.008
    block.location, block.rotation_euler, block.scale = (2.5, -1, 0.4), (0.3, 0.5, -0.2), (-1.2, 1.1, 0.8)
    bpy.context.view_layer.update()
    placement = np.array(block.matrix_world)
    run("sdf_generate", object=stock, parameters={"size.1": 1.05})
    assert np.allclose(block.matrix_world, placement) and not block.ray_cast((0, 0, 1), (0, 0, -1))[0]
    recipe_path = run("save_feature_tree", object=stock, path=str(Path(tmp) / "mixed.json"))["path"]
    raw = json.loads(Path(recipe_path).read_text(encoding="utf8"))
    assert raw["base"]["sdf"]["tree"]["size"][0] == 2.2 and raw["features"][0]["id"] == "bore"
    duplicate = bpy.data.objects[run("feature_tree", name="Replay block", path=recipe_path)["name"]]
    _solid(duplicate.data)
    assert not duplicate.ray_cast((0, 0, 1), (0, 0, -1))[0]
    run("rebuild_feature_tree", object=duplicate.name, parameters={"bore.depth": -0.12})
    assert duplicate.ray_cast((0, 0, 1), (0, 0, -1))[0]
# An older .blend has no SDF recipe. An explicit tree initializes it in place.
bpy.ops.mesh.primitive_cube_add()
legacy = bpy.context.object
legacy.name = "Legacy mesh"
assert not AiBridgeServer.instance()._dispatch_dict(
    dict(cmd="sdf_generate", object=legacy.name, parameters={"radius": 0.3})
)["ok"]
run("sdf_generate", object=legacy.name, tree={"type": "sphere", "radius": 0.32}, resolution=32)
assert len(legacy.data.polygons) > 100 and "excad_sdf_recipe" in legacy
legacy.shape_key_add(name="Basis")
key_mesh = legacy.data
assert not AiBridgeServer.instance()._dispatch_dict(
    dict(cmd="sdf_generate", object=legacy.name, parameters={"radius": 0.45})
)["ok"]
assert legacy.data == key_mesh and "Basis" in legacy.data.shape_keys.key_blocks
print("SDF REBUILD PASS", bpy.app.version_string, flush=True)
