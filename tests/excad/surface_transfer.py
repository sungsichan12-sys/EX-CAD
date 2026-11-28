# SPDX-License-Identifier: GPL-2.0-or-later
"""Check UV chart registration and seams after an edit or a real GLB voxel import."""

import tempfile
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

from excad_addon.ai_3d_provider import import_model
from excad_addon.feature_tree import _solid
from excad_addon.uv_transfer import transfer_uv


def chart(point):
    axis = np.argmax(np.abs(point))
    return axis * 2 + int(point[axis] < 0)


def chart_uv(point, sector):
    axis = sector // 2
    return (
        (sector % 3 + 0.5 + 0.42 * point[(axis + 1) % 3]) / 3,
        (sector // 3 + 0.5 + 0.42 * point[(axis + 2) % 3]) / 2,
    )


def check_chart(mesh, registration=False):
    layer = mesh.uv_layers["Six charts"]
    uv = np.empty((len(mesh.loops), 2), np.float32)
    layer.uv.foreach_get("vector", uv.ravel())
    sectors = np.floor(uv[:, 0] * 3).astype(int) + 3 * np.floor(uv[:, 1] * 2).astype(int)
    assert set(sectors) == set(range(6))
    starts = np.empty(len(mesh.polygons), np.int32)
    mesh.polygons.foreach_get("loop_start", starts)
    assert np.array_equal(np.minimum.reduceat(sectors, starts), np.maximum.reduceat(sectors, starts))
    assert sum(edge.use_seam for edge in mesh.edges) >= 20
    if registration:
        errors = []
        for p in mesh.polygons:
            center = np.asarray(p.center) / (1, 2, 0.6)
            lengths = sorted(abs(center))
            if lengths[-1] - lengths[-2] < 0.1:
                continue
            tag = chart(center)
            for index, loop in zip(p.vertices, p.loop_indices):
                point = np.asarray(mesh.vertices[index].co) / (1, 2, 0.6)
                errors.append(np.linalg.norm(uv[loop] - chart_uv(point, tag)))
        assert len(errors) > 1000 and max(errors) < 0.008, max(errors)


bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24)
source = bpy.context.object
source.name = "Painted sphere"
source.data.uv_layers.active.name = "Six charts"
for p in source.data.polygons:
    tag = chart(p.center)
    for loop, index in zip(p.loop_indices, p.vertices):
        source.data.uv_layers.active.uv[loop].vector = chart_uv(source.data.vertices[index].co, tag)
other = source.data.uv_layers.new(name="Second UV")
source.data.uv_layers.active = other
source.data.uv_layers["Six charts"].active_render = True
bpy.ops.mesh.primitive_uv_sphere_add(segments=66, ring_count=33)
edited = bpy.context.object
edited.data.uv_layers.remove(edited.data.uv_layers.active)
edited.data.transform(Matrix.Diagonal(Vector((1.0, 2.0, 0.6, 1.0))))
transfer_uv(source.data, edited.data, normalize=True)
check_chart(edited.data, registration=True)
assert edited.data.uv_layers.active.name == "Second UV" and edited.data.uv_layers["Six charts"].active_render

with tempfile.TemporaryDirectory() as directory:
    glb = Path(directory) / "painted.glb"
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = source
    source.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(glb), use_selection=True, export_texcoords=True)
    model = import_model(glb, height=0.8, remesh=True, name="Voxel sphere")
    mesh = bpy.data.objects[model["meshes"][0]].data
    _solid(mesh)
    # glTF indexes texture coordinates and the Blender importer supplies their names.
    mesh.uv_layers[0].name = "Six charts"
    check_chart(mesh)
    print("SURFACE TRANSFER PASS", len(mesh.vertices), "vertices", len(mesh.polygons), "faces", flush=True)
