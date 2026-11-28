# SPDX-License-Identifier: GPL-2.0-or-later
"""Verify surface-aligned scatter on uneven, reflected terrain and stacked shelves."""

import math

import bpy
import numpy as np
from mathutils import Vector

from excad_addon.ai_bridge import AiBridgeServer
from excad_addon import scene_mesh


def run(command, **fields):
    result = AiBridgeServer.instance()._dispatch_dict(dict(cmd=command, **fields))
    assert result["ok"], (command, result)
    return result


def make_mesh(name, vertices, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


for obj in list(bpy.context.scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

side = 17
grid = np.linspace(-1.8, 1.8, side)
terrain = [
    (x, y, 0.56 + 0.11 * math.sin(2.7 * x) * math.sin(2.3 * y) + 0.09 * math.cos(2.8 * y)) for y in grid for x in grid
]
top = [(i, i + 1, i + side + 1, i + side) for i in range(side * (side - 1)) if i % side < side - 1]
rim = [
    *range(side),
    *range(2 * side - 1, side * side, side),
    *range(side * side - 2, side * (side - 1) - 1, -1),
    *range(side * (side - 2), 0, -side),
]
base = len(terrain)
terrain.extend((terrain[i][0], terrain[i][1], 0) for i in rim)
for j, i in enumerate(rim):
    k = (j + 1) % len(rim)
    top.append((i, base + j, base + k, rim[k]))
top.append(tuple(base + j for j in reversed(range(len(rim)))))
land = make_mesh("Hillside", terrain, top)
land.scale.x = -1

bpy.ops.mesh.primitive_cube_add(size=1)
tile = bpy.context.object
tile.name = "Survey marker"
tile.location = (5, 0, 0.08)
tile.scale = (-0.40, 0.26, 0.16)
for aligned in (True, False):
    names = run(
        "scatter_on_surface",
        asset=tile.name,
        surface_obj=land.name,
        count=14,
        min_dist=0.58,
        seed=91,
        align_normal=aligned,
        max_slope=40,
    )["objects"]
    parts = {s.obj.name: s for s in scene_mesh.surfaces()}
    ground = parts[land.name]
    assert ground.closed
    for name in names:
        part = parts[name]
        up = (part.obj.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
        if aligned:
            assert up.z >= math.cos(math.radians(40)) - 1e-5
        else:
            assert (up - Vector((0, 0, 1))).length < 1e-6
        # The fitter moves along the sampled normal. Independently measure
        # the vertical gap to the complete terrain, including its edges.
        lift = float(ground.bounds[1, 2] - part.bounds[0, 2] + 0.2)
        clearance = scene_mesh.vertical_clearance(part, ground, (0, 0, lift))
        assert abs(clearance - lift) < 2e-5, (name, clearance - lift)
    result = run("render_inspect")
    assert not result["warnings"], result["warnings"]
    sheet = make_mesh("Hillside surface", terrain[:base], top[: (side - 1) ** 2])
    sheet.matrix_world = land.matrix_world
    assert not run("check_intersections", objects=[sheet.name, *names])["intersections"]
    bpy.data.objects[names[0]].location.z -= 0.03
    assert run("check_intersections", objects=[sheet.name, names[0]])["intersections"]
    bpy.data.objects.remove(sheet, do_unlink=True)
    for name in names:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

# A support within the search radius but disconnected from the sampled sheet
# must leave a marker on its original shelf.
plates = []
for height in (0.2, 0.40):
    plates.extend((x, y, height) for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1)))
shelves = make_mesh("Separated shelves", plates, [(0, 1, 2, 3), (4, 5, 6, 7)])
shelves.location.x = 10
tile.scale = (0.9, 0.5, 0.06)
names = run(
    "scatter_on_surface",
    asset=tile.name,
    surface_obj=shelves.name,
    count=12,
    min_dist=0.1,
    max_slope=1,
    seed=37,
)["objects"]
bottoms = {round(float(s.bounds[0, 2]), 4) for s in scene_mesh.surfaces() if s.obj.name in names}
assert bottoms == {0.2, 0.4}, bottoms

# Both sides of a vertical sheet can support a bracket. Cutting through
# the sheet should still register on either side.
wall = make_mesh("Vertical sheet", [(0, -1, -1), (0, 1, -1), (0, 1, 1), (0, -1, 1)], [(0, 1, 2, 3)])
tile.scale = (0.2, 0.3, 0.3)
for side_sign in (-1, 1):
    for distance, crossing in ((0.1, False), (0.099, True)):
        tile.location = (side_sign * distance, 0, 0)
        assert bool(run("check_intersections", objects=[wall.name, tile.name])["intersections"]) == crossing
before = set(bpy.data.objects)
bad = AiBridgeServer.instance()._dispatch_dict(
    dict(cmd="scatter_on_surface", asset=tile.name, surface_obj=land.name, count=1, max_slope=-5)
)
assert not bad["ok"] and set(bpy.data.objects) == before
print("SCATTER TERRAIN PASS", bpy.app.version_string, len(names), "markers on two disconnected shelves")
