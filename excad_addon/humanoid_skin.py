# SPDX-License-Identifier: GPL-2.0-or-later
"""Skin multi-part generated characters to one armature in their original rest coordinates."""

import bpy
import numpy as np
from mathutils import Vector

from . import scene_mesh


def _modifier(obj, rig):
    mod = next((m for m in obj.modifiers if m.type == "ARMATURE" and m.object == rig), None)
    if mod is None:
        mod = obj.modifiers.new("Character pose", "ARMATURE")
        mod.object = rig
    mod.use_deform_preserve_volume = True


def _complete_weights(obj, rig):
    bones = [b for b in rig.data.bones if b.use_deform and (b.tail_local - b.head_local).length > 1e-8]
    if not bones:
        raise ValueError("The imported character armature has no deform bones")
    names = {b.name for b in bones}
    groups = {g.index: g for g in obj.vertex_groups if g.name in names}
    missing = []
    for vertex in obj.data.vertices:
        influences = [(groups[g.group], g.weight) for g in vertex.groups if g.group in groups and g.weight > 0]
        total = sum(w for _, w in influences)
        if total < 1e-7:
            missing.append(vertex.index)
        elif abs(total - 1) > 1e-4:
            for group, weight in influences:
                group.add([vertex.index], weight / total, "REPLACE")
    if not missing:
        return
    # Bone heat can leave disconnected islands unweighted. Blend their four
    # nearest finite bone segments in armature rest coordinates.
    starts = np.array([b.head_local[:] for b in bones])
    vectors = np.array([b.tail_local[:] for b in bones]) - starts
    length_squared = np.einsum("ij,ij->i", vectors, vectors)
    radius_squared = (0.022 * np.ptp(np.vstack((starts, starts + vectors)), axis=0).max()) ** 2
    transform = np.asarray(rig.matrix_world.inverted_safe() @ obj.matrix_world)
    coords = np.empty((len(obj.data.vertices), 3), np.float32)
    obj.data.vertices.foreach_get("co", coords.ravel())
    groups_by_bone = [obj.vertex_groups.get(b.name) or obj.vertex_groups.new(name=b.name) for b in bones]
    for offset in range(0, len(missing), 2048):
        ids = missing[offset : offset + 2048]
        points = coords[ids] @ transform[:3, :3].T + transform[:3, 3]
        delta = points[:, None, :] - starts
        along = np.clip(np.einsum("nbi,bi->nb", delta, vectors) / length_squared, 0, 1)
        distance = delta - along[:, :, None] * vectors
        squared = np.einsum("nbi,nbi->nb", distance, distance)
        nearest = np.argsort(squared, axis=1)[:, :4]
        weights = 1 / (np.take_along_axis(squared, nearest, axis=1) + radius_squared) ** 2
        weights /= weights.sum(axis=1, keepdims=True)
        weights[weights < 0.002] = 0
        weights /= weights.sum(axis=1, keepdims=True)
        for vertex, indices, values in zip(ids, nearest, weights):
            for index, weight in zip(indices, values):
                if weight:
                    groups_by_bone[index].add([vertex], float(weight), "REPLACE")


def bind_heat(obj, rig, keep_parent):
    if obj.data.users > 1:
        obj.data = obj.data.copy()
    parent = obj.parent
    kind, bone = obj.parent_type, obj.parent_bone
    basis, inverse = obj.matrix_basis.copy(), obj.matrix_parent_inverse.copy()
    world = obj.matrix_world.copy()
    for other in list(bpy.context.selected_objects):
        other.select_set(False)
    obj.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    result = bpy.ops.object.parent_set(type="ARMATURE_AUTO", keep_transform=True)
    if "FINISHED" not in result:
        raise ValueError(f"Could not bind {obj.name} to the character rig")
    if keep_parent:
        obj.parent, obj.parent_type, obj.parent_bone = parent, kind, bone
        obj.matrix_parent_inverse, obj.matrix_basis = inverse, basis
    else:
        obj.matrix_world = world
    bpy.context.view_layer.update()
    _modifier(obj, rig)
    _complete_weights(obj, rig)


def fit_imported(meshes, loose, name):
    from .human_engine import create_humanoid_armature
    from .wardrobe import transfer_weights

    loose = set(loose)
    graph = bpy.context.evaluated_depsgraph_get()
    items = [s for obj in meshes if (s := scene_mesh.read_surface(obj, graph))]
    bound = [s for s in items if s.obj not in loose and s.obj.find_armature()]
    bounds = np.array([s.bounds for s in items])
    low, high = bounds[:, 0].min(0), bounds[:, 1].max(0)
    created = None
    if not bound:
        created = create_humanoid_armature((Vector(low), Vector(high)), name + " rig")
        main = max(items, key=lambda s: (s.bounds[1, 2] - s.bounds[0, 2], len(s.points)))
        # A full-body surface is a weight source for separate clothing and accessories.
        # A set of disconnected body parts is weighted against the shared skeleton.
        if main.bounds[1, 2] - main.bounds[0, 2] >= 0.82 * (high[2] - low[2]):
            bind_heat(main.obj, created, keep_parent=True)
            bound.append(main)
    sources = {s.obj for s in bound}
    for part in items:
        obj = part.obj
        if obj not in loose or obj in sources:
            continue
        rig = created
        if obj.data.users > 1:
            obj.data = obj.data.copy()
        if bound:
            center = Vector(part.bounds.mean(0))
            source = min(bound, key=lambda s: s.bvh.find_nearest(center)[3])
            rig = source.obj.find_armature()
            transfer_weights(source.obj, obj)
        _modifier(obj, rig)
        _complete_weights(obj, rig)
    return created
