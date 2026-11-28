# SPDX-License-Identifier: GPL-2.0-or-later
"""Evaluated, rig-aware hair, clothing and portrait details for MPFB basemeshes."""

from contextlib import contextmanager
import gzip
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion, Vector

from .character_stylizer import _pbr_mat, _vgroup_center
from .face_fit import set_targets
from .human_assets import equip_asset
from .human_engine import _import_mpfb

SKINS = {
    "FAIR_IVORY": (0.80, 0.54, 0.42),
    "WARM_BEIGE": (0.60, 0.34, 0.22),
    "TAN": (0.36, 0.18, 0.095),
    "DEEP": (0.15, 0.067, 0.038),
}
HAIR = {
    "SHORT": "short02",
    "BOB": "bob01",
    "PONYTAIL": "ponytail01",
    "LONG_STRAIGHT": "long01",
    "LONG_WAVY": "long01",
    "CURTAIN_BANGS": "long01",
}
CLOTHES = {
    "TSHIRT": ("shirts", "toigo_basic_tucked_t-shirt"),
    "PANTS": ("pants", "toigo_wool_pants"),
    "SKIRT": ("skirts", "frankyaye_mini_skirt_01"),
    "SHOES": ("system", "shoes01"),
}
EXPRESSIONS = {
    "SMILE": {"mouth-corner-puller": 0.62, "mouth-upward-retraction": 0.24},
    "SERIOUS": {"mouth-compression": 0.38, "eyebrows-left-down": 0.25, "eyebrows-right-down": 0.25},
    "SURPRISED": {
        "mouth-open": 0.70,
        "eyebrows-left-up": 0.58,
        "eyebrows-right-up": 0.58,
        "eye-left-opened-up": 0.30,
        "eye-right-opened-up": 0.30,
    },
    "WINK": {"eye-left-closure": 0.92, "eyebrows-right-up": 0.16},
}


@contextmanager
def rest_mesh(human):
    rig = human.find_armature()
    pose = rig.data.pose_position if rig else None
    disabled = [m for m in human.modifiers if m.show_viewport and m.type in {"SUBSURF", "MASK"}]
    try:
        if rig:
            rig.data.pose_position = "REST"
        for mod in disabled:
            mod.show_viewport = False
        bpy.context.view_layer.update()
        evaluated = human.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        try:
            if len(mesh.vertices) != len(human.data.vertices):
                raise ValueError("Character styling requires indexed MPFB topology")
            yield mesh
        finally:
            evaluated.to_mesh_clear()
    finally:
        for mod in disabled:
            mod.show_viewport = True
        if rig:
            rig.data.pose_position = pose
        bpy.context.view_layer.update()


def _attach(obj, human):
    obj.parent = human
    obj["excad_wearer"] = human.name
    obj["excad_human"] = human
    obj.matrix_parent_inverse.identity()
    rig = human.find_armature()
    if rig and not any(m.type == "ARMATURE" for m in obj.modifiers):
        mod = obj.modifiers.new("Character pose", "ARMATURE")
        mod.object = rig
        mod.use_deform_preserve_volume = True
    return obj


def head_weight(obj, human):
    rig = human.find_armature()
    head = (
        next((b.name for b in rig.data.bones if b.name.lower().split(":")[-1] in {"head", "mixamorighead"}), None)
        if rig
        else "head"
    )
    if not head:
        raise ValueError("Selected armature has no head bone")
    if rig:
        for group in list(obj.vertex_groups):
            if group.name in rig.data.bones and group.name != head:
                obj.vertex_groups.remove(group)
    group = obj.vertex_groups.get(head) or obj.vertex_groups.new(name=head)
    group.add(list(range(len(obj.data.vertices))), 1, "REPLACE")


def _object(human, name, points, faces, mat):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(points, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(mat)
    for p in mesh.polygons:
        p.use_smooth = True
    return _attach(obj, human)


def _eye_centers(human, mesh):
    left = _vgroup_center(human, mesh, ("helper-l-eye", "l-eye"))
    right = _vgroup_center(human, mesh, ("helper-r-eye", "r-eye"))
    if left is None or right is None:
        raise ValueError("Character has no MPFB eye landmarks")
    return left, right


def _curtains(human, mesh, material):
    left, right = _eye_centers(human, mesh)
    center = (left + right) / 2
    span = (left - right).length
    verts, faces, coords = [], [], []
    surface = _body_surface(human, mesh)
    for side in (-1, 1):
        for lock in range(8):
            start = len(verts)
            layer = lock / 7
            for j in range(17):
                t = j / 16
                x = side * span * (0.02 + 0.16 * layer + (1.13 + 0.10 * layer) * math.sin(t * math.pi / 2))
                z = span * (1.22 + 0.16 * layer - (1.69 + 0.35 * layer) * t * t)
                p = center + Vector((x, -2 * span, z))
                forehead = surface.ray_cast(p, (0, 1, 0), 4 * span)[0]
                p.y = (forehead.y if forehead else center.y + span * 0.17) - span * (0.065 + 0.04 * layer)
                w = span * 0.075 * (0.03 + math.sin(math.pi * t) ** 0.6)
                verts.extend((p + Vector((-w, 0, 0)), p + Vector((0, -0.22 * w, 0)), p + Vector((w, 0, 0))))
                # Sample an authored lock from the same CC0 atlas as the rest of the hair.
                coords.extend((0.14 + layer * 0.07 + d, 0.94 - 0.87 * t) for d in (-0.035, 0, 0.035))
                if j:
                    k = start + (j - 1) * 3
                    faces.extend(((k, k + 1, k + 4, k + 3), (k + 1, k + 2, k + 5, k + 4)))
    obj = _object(human, "Curtain fringe", verts, faces, material)
    uv = obj.data.uv_layers.new(name="Hair locks")
    for loop in obj.data.loops:
        uv.uv[loop.index].vector = coords[loop.vertex_index]
    head_weight(obj, human)
    return obj


def _body_surface(human, mesh):
    from mathutils.bvhtree import BVHTree

    group = human.vertex_groups["body"].index
    vertices = {v.index for v in human.data.vertices if any(g.group == group for g in v.groups)}
    faces = [tuple(f.vertices) for f in mesh.polygons if all(i in vertices for i in f.vertices)]
    return BVHTree.FromPolygons([v.co for v in mesh.vertices], faces)


def _drape_hair(human, obj, mesh, style):
    surface = _body_surface(human, mesh)
    eye_l, eye_r = _eye_centers(human, mesh)
    span = (eye_l - eye_r).length
    neck = _vgroup_center(human, mesh, ("joint-neck",))
    if neck is None:
        return
    transform = human.matrix_world.inverted() @ obj.matrix_world
    inverse = transform.to_3x3().inverted()
    if not obj.data.shape_keys:
        obj.shape_key_add(name="Basis")
    mixed = obj.shape_key_add(name="EXCAD hair measurement", from_mix=True)
    points = [transform @ v.co for v in mixed.data]
    obj.shape_key_remove(mixed)
    key = obj.shape_key_add(name="EXCAD hair drape", from_mix=False)
    for i, original in enumerate(points):
        p = original.copy()
        length = max(0, neck.z - p.z)
        blend = min(1.0, max(0, (neck.z + span * 0.85 - p.z) / (span * 0.80)))
        if blend:
            # Retain the head weighting. Bring long tips to the upper back and
            # let both front and back locks clear the moving shoulder line.
            p.z += length * (0.25 if style == "PONYTAIL" else 0.39)
            start = Vector((p.x, neck.y + span * 8, p.z))
            hit = surface.ray_cast(start, (0, -1, 0), span * 16)[0]
            if hit:
                gap = (0.46 if style == "PONYTAIL" else 0.39) * span
                p.y += max(0, hit.y + gap - p.y) * blend
        if style == "LONG_WAVY":
            t = min(1, max(0, (neck.z + span - p.z) / (3.8 * span)))
            side = -1 if p.x < 0 else 1
            p.x += side * span * 0.37 * t * math.sin(13 * t + 0.25 * side)
            p.y += span * 0.10 * t * math.sin(16 * t)
        key.data[i].co += inverse @ (p - original)
    key.value = 1


def refit_fringe(human):
    fringes = [o for o in bpy.context.scene.objects if o.get("excad_human") == human and o.get("excad_fringe")]
    if not fringes:
        return
    with rest_mesh(human) as mesh:
        surface = _body_surface(human, mesh)
        left, right = _eye_centers(human, mesh)
        eye, span = (left + right) / 2, (left - right).length
        axis = eye + Vector((0, 0.7 * span, 0))
        for obj in fringes:
            to_head = human.matrix_world.inverted() @ obj.matrix_world
            to_hair = to_head.to_3x3().inverted()
            if not obj.data.shape_keys:
                obj.shape_key_add(name="Basis")
            key = obj.data.shape_keys.key_blocks.get("EXCAD fringe clearance")
            if key:
                key.value = 0
            mixed = obj.shape_key_add(name="EXCAD fringe measurement", from_mix=True)
            points = [to_head @ v.co for v in mixed.data]
            obj.shape_key_remove(mixed)
            key = key or obj.shape_key_add(name="EXCAD fringe clearance", from_mix=False)
            basis = obj.data.shape_keys.reference_key
            for i, point in enumerate(points):
                key.data[i].co = basis.data[i].co
                height = (point.z - eye.z) / span
                outward = Vector((point.x - axis.x, point.y - axis.y, 0)).normalized()
                if height > 0.85 or height < -2.0 or outward.y > 0:
                    continue
                # A lock can span an open mouth. Use the adjacent cheek/lip
                # envelope so a strand cannot pass through the mouth to the teeth.
                offset = 0.0
                # Keep the curved fringe over the brow with room for the lashes.
                gap = span * (0.04 + 0.18 * math.exp(-(((height - 0.02) / 0.46) ** 2)))
                for dz in (-0.09, 0, 0.09) if height < 0.35 else (0,):
                    start = point + outward * (4 * span) + Vector((0, 0, dz * span))
                    hit = surface.ray_cast(start, -outward, 8 * span)[0]
                    if hit is not None:
                        offset = max(offset, (hit - point).dot(outward) + gap)
                fade = min(1.0, (0.85 - height) / 0.4, (height + 2.0) / 0.35, -outward.y / 0.2)
                key.data[i].co += to_hair @ (outward * (offset * fade))
            key.value = 1


def hair_style(human, style):
    from .portrait_materials import hair_fibers

    obj = equip_asset(human, HAIR[style])
    hair_fibers(obj)
    head_weight(obj, human)
    result = [obj]
    if style == "BOB":
        obj["excad_fringe"] = True
        refit_fringe(human)
    if style in {"LONG_WAVY", "LONG_STRAIGHT", "PONYTAIL", "CURTAIN_BANGS"}:
        with rest_mesh(human) as mesh:
            _drape_hair(human, obj, mesh, style)
            if style == "CURTAIN_BANGS":
                result.append(_curtains(human, mesh, obj.data.materials[0]))
        obj["excad_draped_hair"] = True
    return result


def transfer_weights(human, obj):
    rig = human.find_armature()
    if rig is None:
        return
    # Evaluated helper geometry can be closer than the skin. Transfer from the body's
    # visible skin with clothes masks temporarily disabled, in the same rest pose.
    masked = [m for m in human.modifiers if m.type == "MASK" and "helper" not in m.name.lower() and m.show_viewport]
    pose = rig.data.pose_position
    existing_groups = {group.name for group in obj.vertex_groups}
    try:
        rig.data.pose_position = "REST"
        for m in masked:
            m.show_viewport = False
        bpy.context.view_layer.update()
        with bpy.context.temp_override(
            object=human, active_object=human, selected_objects=[human, obj], selected_editable_objects=[human, obj]
        ):
            result = bpy.ops.object.data_transfer(
                data_type="VGROUP_WEIGHTS",
                use_create=True,
                vert_mapping="POLYINTERP_NEAREST",
                layers_select_src="ALL",
                layers_select_dst="NAME",
                mix_mode="REPLACE",
            )
        if "FINISHED" not in result:
            raise ValueError("Could not transfer garment weights")
        for group in list(obj.vertex_groups):
            if group.name not in existing_groups and group.name not in rig.data.bones:
                obj.vertex_groups.remove(group)
    finally:
        for m in masked:
            m.show_viewport = True
        rig.data.pose_position = pose
        bpy.context.view_layer.update()


def _clip_panel(points, distance):
    """Clip one garment panel to a seam shared by its neighbouring panels."""
    clipped = []
    for i, point in enumerate(points):
        previous = points[i - 1]
        a, b = distance(previous), distance(point)
        if (a >= 0) != (b >= 0):
            clipped.append(previous.lerp(point, a / (a - b)))
        if b >= 0:
            clipped.append(point)
    return clipped


def _tailored(human, style):
    with rest_mesh(human) as mesh:

        def group_vertices(name):
            group = human.vertex_groups[name].index
            return {v.index for v in human.data.vertices if any(g.group == group for g in v.groups)}

        body = group_vertices("body")
        cloth = group_vertices("helper-tights")
        height = max(mesh.vertices[i].co.z for i in body) - min(mesh.vertices[i].co.z for i in body)

        def joint(name):
            point = _vgroup_center(human, mesh, ("joint-" + name,))
            if point is None:
                raise ValueError("Missing MPFB garment landmark " + name)
            return point

        neck, hip = joint("neck"), joint("pelvis")
        wrists = [(joint(side + "-hand"), joint(side + "-elbow")) for side in ("l", "r")]
        wrists = [(p, (p - elbow).normalized()) for p, elbow in wrists]
        allowance = {"JACKET": 0.012, "HOODIE": 0.010, "VNECK_SWEATER": 0.006}[style] * height
        hem = hip.z + {"JACKET": -0.018, "HOODIE": -0.008, "VNECK_SWEATER": 0.004}[style] * height

        def neckline(p):
            width = abs(p.x - neck.x) / height
            front = min(1, max(0, (neck.y - p.y) / (0.04 * height)))
            drop = {"JACKET": 0.078, "HOODIE": 0.026, "VNECK_SWEATER": 0.075}[style]
            return neck.z + 0.003 * height - height * front * max(0, drop - width * 1.25) - p.z

        def placket(p):
            # Two open fronts with a tapered lapel. The back panel stays continuous.
            return max(abs(p.x - neck.x) - height * 0.007, p.y - neck.y)

        offsets = {i: mesh.vertices[i].co + mesh.vertices[i].normal * allowance for i in cloth}
        cuts = [lambda p: p.z - hem, neckline]
        for wrist, along in wrists:
            cuts.append(lambda p, w=wrist, d=along: -(p - w).dot(d) - 0.009 * height)
        if style == "JACKET":
            cuts.append(placket)
        verts, faces, index = [], [], {}

        def vertex(p):
            key = tuple(round(c, 6) for c in p)
            if key not in index:
                index[key] = len(verts)
                verts.append(p)
            return index[key]

        for poly in mesh.polygons:
            if not all(i in cloth for i in poly.vertices):
                continue
            panel = [offsets[i] for i in poly.vertices]
            for cut in cuts:
                panel = _clip_panel(panel, cut)
                if len(panel) < 3:
                    break
            if len(panel) >= 3:
                face = tuple(dict.fromkeys(vertex(p) for p in panel))
                if len(face) >= 3:
                    faces.append(face)
        if style == "HOODIE":
            start = len(verts)
            for j in range(8):
                t = j / 7
                for i in range(41):
                    a = math.pi * i / 40
                    verts.append(
                        neck
                        + height
                        * Vector(
                            (
                                (0.053 + 0.045 * t) * math.cos(a),
                                0.022 + (0.035 + 0.069 * t) * math.sin(a),
                                0.003 - 0.024 * t - (0.018 + 0.077 * t) * math.sin(a),
                            )
                        )
                    )
            for j in range(7):
                for i in range(40):
                    a = start + j * 41 + i
                    faces.append((a, a + 1, a + 42, a + 41))
        colors = {"VNECK_SWEATER": (0.51, 0.36, 0.22), "JACKET": (0.060, 0.092, 0.14), "HOODIE": (0.105, 0.245, 0.22)}
        color = colors[style]
        obj = _object(
            human, style.replace("_", " ").title(), verts, faces, _pbr_mat("EXCAD " + style, color, 0.78, sheen=0.3)
        )
    counts = {}
    for p in obj.data.polygons:
        for edge in p.edge_keys:
            counts[edge] = counts.get(edge, 0) + 1
    border_edges = [edge for edge, n in counts.items() if n == 1]
    borders = {v for edge in border_edges for v in edge}
    seam = obj.vertex_groups.new(name="EXCAD seams")
    seam.add(list(borders), 1, "REPLACE")
    trim = _pbr_mat("EXCAD " + style + " trim", tuple(c * 0.62 for c in color), 0.84, sheen=0.2)
    obj.data.materials.append(trim)
    from .portrait_materials import finish_fabric

    finish_fabric(obj, style, color, height, border_edges)
    transfer_weights(human, obj)
    sub = obj.modifiers.new("Finished neckline and cuffs", "SUBSURF")
    sub.levels = sub.render_levels = 1
    solid = obj.modifiers.new("Fabric thickness", "SOLIDIFY")
    solid.thickness, solid.offset = height * 0.0018, 1
    solid.material_offset_rim = 1
    obj["excad_outer_fabric"] = True
    return obj


def _fill_coverage_gaps(covered, ambiguous, neighbors):
    """Close skin islands enclosed by fabric. Leave every region reaching exposed skin open."""
    while ambiguous:
        start = ambiguous.pop()
        patch, pending = {start}, [start]
        enclosed, touches_cover = True, False
        while pending:
            for other in neighbors.get(pending.pop(), ()):
                if other in ambiguous:
                    ambiguous.remove(other)
                    patch.add(other)
                    pending.append(other)
                elif other not in patch:
                    if other in covered:
                        touches_cover = True
                    else:
                        enclosed = False
        if enclosed and touches_cover:
            covered.update(patch)


def garment_clearance(human, garment):
    """Move inward cloth vertices to the fitted skin and mask skin covered by fabric.

    Test coverage away from open hems. A nearest point on a sleeve edge by itself
    is insufficient evidence that the hand underneath should be hidden.
    """
    from mathutils.bvhtree import BVHTree
    from mathutils.kdtree import KDTree

    with rest_mesh(human) as skin:
        body_group = human.vertex_groups["body"].index
        body_ids = {v.index for v in human.data.vertices if any(g.group == body_group for g in v.groups)}
        points = [v.co.copy() for v in skin.vertices]
        faces = [tuple(f.vertices) for f in skin.polygons if all(i in body_ids for i in f.vertices)]
        surface = BVHTree.FromPolygons(points, faces)
        height = max(points[i].z for i in body_ids) - min(points[i].z for i in body_ids)
        to_skin = human.matrix_world.inverted() @ garment.matrix_world
        to_garment = to_skin.to_3x3().inverted()
        if not garment.data.shape_keys:
            garment.shape_key_add(name="Basis")
        # Mixed (fitted) coordinates for measurement, a basis-relative displacement
        # for the correction. Keeping these apart prevents applying a fit twice.
        mix = garment.shape_key_add(name="EXCAD fit measurement", from_mix=True)
        cloth = [to_skin @ v.co for v in mix.data]
        garment.shape_key_remove(mix)
        key = garment.shape_key_add(name="EXCAD fabric clearance", from_mix=False)
        margin = max(0.001, height * 0.0035)
        for i, p in enumerate(cloth):
            hit, n, _, distance = surface.find_nearest(p)
            if hit is None or distance > height * 0.026:
                continue
            signed = (p - hit).dot(n)
            if signed < margin:
                shift = n * (margin - signed)
                key.data[i].co += to_garment @ shift
                cloth[i] = p + shift
        key.value = 1
        polygons = [tuple(p.vertices) for p in garment.data.polygons]
        cover = BVHTree.FromPolygons(cloth, polygons)
        edge_count = {}
        for face in polygons:
            for a, b in zip(face, face[1:] + face[:1]):
                pair = tuple(sorted((a, b)))
                edge_count[pair] = edge_count.get(pair, 0) + 1
        borders = [(cloth[a], cloth[b]) for (a, b), count in edge_count.items() if count == 1]
        kd = None
        if borders:
            samples = []
            for a, b in borders:
                steps = max(1, math.ceil((a - b).length / (height * 0.004)))
                samples.extend(a.lerp(b, i / steps) for i in range(steps + 1))
            kd = KDTree(len(samples))
            for i, p in enumerate(samples):
                kd.insert(p, i)
            kd.balance()
        incident, neighbors = {}, {}
        for face in faces:
            for a, b in zip(face, face[1:] + face[:1]):
                length = (points[a] - points[b]).length
                incident[a] = max(incident.get(a, 0), length)
                incident[b] = max(incident.get(b, 0), length)
                neighbors.setdefault(a, set()).add(b)
                neighbors.setdefault(b, set()).add(a)
        covered, ambiguous = set(), set()
        for i in body_ids:
            hit, n, _, distance = cover.find_nearest(points[i])
            if hit is None or distance > height * 0.05:
                continue
            normal = skin.vertices[i].normal
            hem_margin = max(height * 0.003, incident.get(i, 0) * 0.68)
            if normal.dot(n) < 0.25:
                # At an armpit the opposite sleeve wall can be the nearest
                # face. An outward ray finds the panel covering this skin.
                nearest = hit
                hit, n, _, _ = cover.ray_cast(points[i], normal, height * 0.05)
                if hit is None or normal.dot(n) < 0.25:
                    if kd is None or kd.find(nearest)[2] >= hem_margin:
                        ambiguous.add(i)
                    continue
            if kd and kd.find(hit)[2] < hem_margin:
                continue
            covered.add(i)
        _fill_coverage_gaps(covered, ambiguous, neighbors)
    if covered:
        name = "EXCAD.Cover." + garment.name
        group = human.vertex_groups.new(name=name)
        group.add(sorted(covered), 1, "REPLACE")
        mod = human.modifiers.new(group.name, "MASK")
        mod.vertex_group = group.name
        mod.invert_vertex_group = True
        # Subdivision interpolates the binary coverage across a hem. Removing
        # every nonzero weight cuts a row of visible skin outside a V-neck.
        mod.threshold = 0.5
        garment["excad_coverage_group"] = group.name


def _clear_layer_faces(human, outer, skin, key, mixed, surfaces, neck, span):
    """Clear inner fabric along the skin normal, including crossings between vertices."""
    from mathutils.bvhtree import BVHTree

    body = _body_surface(human, skin)
    to_skin = human.matrix_world.inverted()
    normal_to_world = to_skin.to_3x3().transposed()
    to_outer = outer.matrix_world.to_3x3().inverted()
    basis = outer.data.shape_keys.reference_key
    points = [outer.matrix_world @ (p + v.co - b.co) for p, v, b in zip(mixed, key.data, basis.data)]
    outer.data.calc_loop_triangles()
    triangles = [tuple(t.vertices) for t in outer.data.loop_triangles]
    scale = (human.matrix_world.to_3x3() @ Vector((0, 0, span))).length
    for _ in range(8):
        tree = BVHTree.FromPolygons(points, triangles, all_triangles=True)
        corrections = {}
        for surface in surfaces:
            if surface is None:
                continue
            for hi, fi in tree.overlap(surface.bvh):
                a, b, c = (Vector(p) for p in surface.points[surface.triangles[fi]])
                normal = (b - a).cross(c - a).normalized()
                near = body.find_nearest(to_skin @ ((a + b + c) / 3))
                if near[0] is None:
                    continue
                if normal.dot(normal_to_world @ near[1]) < 0:
                    normal = -normal
                for index in triangles[hi]:
                    distance = scale * 0.008 - (points[index] - a).dot(normal)
                    if distance > 0 and (index not in corrections or distance > corrections[index][0]):
                        corrections[index] = (distance, normal * min(scale * 0.060, distance))
        if not corrections:
            break
        for index, (_, shift) in corrections.items():
            points[index] += shift
            key.data[index].co += to_outer @ shift
    if outer.get("excad_draped_hair"):
        # Leave articulation space over the moving shoulder. Head-weighted locks
        # rotate at the neck while the collar follows the chest and upper arms.
        ease = to_outer @ human.matrix_world.to_3x3() @ Vector((0, 0.010 * span, 0.040 * span))
        for i, point in enumerate(points):
            local = to_skin @ point
            z = (local.z - neck.z) / span
            lateral = abs(local.x - neck.x) / span
            if abs(z) >= 0.45 or not 0.03 < lateral < 0.39:
                continue
            distances = (s.bvh.find_nearest(point, scale * 0.24)[3] for s in surfaces if s is not None)
            nearest = min((d / scale for d in distances if d is not None), default=0.24)
            fade = min(1.0, (lateral - 0.03) / 0.08, (0.39 - lateral) / 0.10, (0.24 - nearest) / 0.07)
            key.data[i].co += ease * (math.exp(-((z / 0.14) ** 2)) * max(0, fade))


def _remove_layer_mask(obj):
    name = obj.get("excad_underlay_group")
    if not name:
        return
    for mod in list(obj.modifiers):
        if mod.type == "MASK" and mod.vertex_group == name:
            obj.modifiers.remove(mod)
    if group := obj.vertex_groups.get(name):
        obj.vertex_groups.remove(group)
    del obj["excad_underlay_group"]


def _mask_under_jacket(jacket, shirt, span):
    """Hide covered shirt panels, retaining the shirt at the open fronts, collar and hems."""
    from mathutils.bvhtree import BVHTree
    from mathutils.kdtree import KDTree

    graph = bpy.context.evaluated_depsgraph_get()
    # Calculate coverage on the fitted control cage. Subdivision and shell
    # thickness add vertices which cannot be addressed by the source vertex group.
    topology = [m for m in shirt.modifiers if m.type in {"SUBSURF", "SOLIDIFY", "MASK"} and m.show_viewport]
    for modifier in topology:
        modifier.show_viewport = False
    bpy.context.view_layer.update()
    outer, inner = jacket.evaluated_get(graph), shirt.evaluated_get(graph)
    jacket_mesh = shirt_mesh = None
    try:
        jacket_mesh, shirt_mesh = outer.to_mesh(), inner.to_mesh()
        if len(shirt_mesh.vertices) != len(shirt.data.vertices):
            return
        points = [outer.matrix_world @ v.co for v in jacket_mesh.vertices]
        faces = [tuple(f.vertices) for f in jacket_mesh.polygons]
        surface = BVHTree.FromPolygons(points, faces)
        counts = {}
        for poly in jacket_mesh.polygons:
            for edge in poly.edge_keys:
                counts[edge] = counts.get(edge, 0) + 1
        seam = jacket.vertex_groups.get("EXCAD seams")
        hem = {
            v.index
            for v in jacket_mesh.vertices
            if seam and any(g.group == seam.index and g.weight > 0.3 for g in v.groups)
        }
        boundary = []
        for (a, b), count in counts.items():
            if count == 1 or (a in hem and b in hem):
                steps = max(1, math.ceil((points[a] - points[b]).length / (span * 0.012)))
                boundary.extend(points[a].lerp(points[b], i / steps) for i in range(steps + 1))
        if not boundary:
            return
        edges = KDTree(len(boundary))
        for i, p in enumerate(boundary):
            edges.insert(p, i)
        edges.balance()
        normals = inner.matrix_world.to_3x3().inverted().transposed()
        # Masking a control vertex can remove the whole adjoining shirt panel.
        # Protect the full panel width around the open lapels and collar.
        shirt_points = [inner.matrix_world @ v.co for v in shirt_mesh.vertices]
        reach = [0.0] * len(shirt_points)
        for face in shirt_mesh.polygons:
            ids = face.vertices[:]
            for i in ids:
                reach[i] = max(reach[i], max((shirt_points[j] - shirt_points[i]).length for j in ids))
        hide = []
        for vertex in shirt_mesh.vertices:
            point = shirt_points[vertex.index]
            normal = (normals @ vertex.normal).normalized()
            hit, direction, _, _ = surface.ray_cast(point - normal * (0.01 * span), normal, 0.24 * span)
            margin = max(0.045 * span, reach[vertex.index])
            if hit is not None and abs(normal.dot(direction)) > 0.25 and edges.find(hit)[2] > margin:
                hide.append(vertex.index)
            elif shirt.get("excad_outer_fabric"):
                # Both tailored pieces use the same body panels. At a concave
                # armhole their normals can miss the enclosing jacket entirely.
                near, _, _, _ = surface.find_nearest(point, 0.12 * span)
                if near is not None and edges.find(near)[2] > max(margin, 0.09 * span):
                    hide.append(vertex.index)
        hidden = set(hide)
        frame = inner.matrix_world.to_3x3()
        front_views = [(frame @ Vector((x, -1, 0))).normalized() for x in (-0.75, -0.35, 0, 0.35, 0.75)]
        for face in shirt_mesh.polygons:
            ids = face.vertices[:]
            if hidden.isdisjoint(ids):
                continue
            center = inner.matrix_world @ face.center
            if edges.find(center)[2] > 0.18 * span + max(reach[i] for i in ids):
                continue
            normal = (normals @ face.normal).normalized()
            samples = [center, *(shirt_points[i] for i in ids)]
            # A shirt triangle may extend past an open lapel even if a ray
            # along its own tilted normal reaches the jacket. Keep that face
            # whenever the actual opening exposes it to a frontal view.
            if any(
                surface.ray_cast(p + view * (0.001 * span), view, 0.35 * span)[0] is None
                for view in front_views
                if normal.dot(view) > 0.2
                for p in samples
            ):
                hidden.difference_update(ids)
        hide = list(hidden)
    finally:
        if jacket_mesh is not None:
            outer.to_mesh_clear()
        if shirt_mesh is not None:
            inner.to_mesh_clear()
        for modifier in topology:
            modifier.show_viewport = True
        bpy.context.view_layer.update()
    if hide:
        group = shirt.vertex_groups.new(name="EXCAD under jacket")
        group.add(hide, 1, "REPLACE")
        mod = shirt.modifiers.new(group.name, "MASK")
        mod.vertex_group = group.name
        mod.invert_vertex_group = True
        # Subdivide the complete shirt first to preserve its shape at trimmed
        # panels. Fabric thickness then closes the newly exposed hems.
        shell = next((m for m in topology if m.type == "SOLIDIFY"), None)
        if shell:
            shirt.modifiers.move(shirt.modifiers.find(mod.name), shirt.modifiers.find(shell.name))
        shirt["excad_underlay_group"] = group.name


def refit_layers(human, omit=()):
    """Fit outside garments and hanging hair to the final outfit in rest space."""
    from . import hair_contact
    from .scene_mesh import read_surface

    pieces = [o for o in bpy.context.scene.objects if o.get("excad_human") == human and o not in omit]
    for piece in pieces:
        _remove_layer_mask(piece)
        if piece.get("excad_draped_hair"):
            hair_contact.clear(piece)
    if not any(o.get("excad_outer_fabric") or o.get("excad_draped_hair") for o in pieces):
        return
    with rest_mesh(human) as skin_mesh:
        pelvis = _vgroup_center(human, skin_mesh, ("joint-pelvis",))
        neck = _vgroup_center(human, skin_mesh, ("joint-neck",))
        span = (neck - pelvis).length
        if span < 1e-6:
            return
        for outer in sorted(
            pieces, key=lambda o: (bool(o.get("excad_draped_hair")), o.get("excad_slot") == "outerwear")
        ):
            hair = outer.get("excad_draped_hair")
            if not hair and not outer.get("excad_outer_fabric"):
                continue
            slots = (
                {"top", "outerwear"}
                if hair
                else {"bottom", "top"}
                if outer.get("excad_slot") == "outerwear"
                else {"bottom"}
            )
            obstacles = [o for o in pieces if o != outer and o.get("excad_slot") in slots]
            key = outer.data.shape_keys.key_blocks.get("EXCAD garment layers") if outer.data.shape_keys else None
            if key:
                key.value = 0
            if not obstacles:
                continue
            bpy.context.view_layer.update()
            graph = bpy.context.evaluated_depsgraph_get()
            surfaces = [read_surface(o, graph) for o in obstacles]
            if not outer.data.shape_keys:
                outer.shape_key_add(name="Basis")
            mix = outer.shape_key_add(name="EXCAD drape measurement", from_mix=True)
            points = [v.co.copy() for v in mix.data]
            outer.shape_key_remove(mix)
            key = key or outer.shape_key_add(name="EXCAD garment layers", from_mix=False)
            basis = outer.data.shape_keys.reference_key
            to_human = human.matrix_world.inverted() @ outer.matrix_world
            to_world = human.matrix_world.to_3x3()
            from_world = outer.matrix_world.to_3x3().inverted()
            for i, local in enumerate(points):
                key.data[i].co = basis.data[i].co
                p = to_human @ local
                if hair and p.z > neck.z + 0.10 * span:
                    continue
                direction = Vector((0, 1, 0)) if hair else Vector((p.x - pelvis.x, p.y - pelvis.y, 0)).normalized()
                if direction.length_squared < 0.5:
                    continue
                world = outer.matrix_world @ local
                out = (to_world @ direction).normalized()
                reach = (to_world @ (span * direction)).length
                clearance = 0.026 * reach
                push = 0
                for obstacle in surfaces:
                    if obstacle is None:
                        continue
                    start, end = world + out * reach, world - out * reach * 0.3
                    if any(
                        max(start[k], end[k]) + clearance < obstacle.bounds[0, k]
                        or min(start[k], end[k]) - clearance > obstacle.bounds[1, k]
                        for k in range(3)
                    ):
                        continue
                    hit = obstacle.bvh.ray_cast(start, -out, reach * 1.3)[0]
                    if hit:
                        amount = (hit - world).dot(out) + clearance
                        if 0 < amount < reach * 0.24:
                            push = max(push, amount)
                key.data[i].co += from_world @ (out * push)
            if hair or outer.get("excad_slot") == "outerwear":
                _clear_layer_faces(human, outer, skin_mesh, key, points, surfaces, neck, span)
            key.value = 1
            if hair:
                fitted = [p + v.co - b.co for p, v, b in zip(points, key.data, basis.data)]
                hair_contact.fit(outer, obstacles, fitted, to_human, neck, span)
            if outer.get("excad_slot") == "outerwear":
                bpy.context.view_layer.update()
                world_span = (to_world @ Vector((0, 0, span))).length
                for shirt in obstacles:
                    if shirt.get("excad_slot") == "top":
                        _mask_under_jacket(outer, shirt, world_span)


def garment(human, style):
    if style in CLOTHES:
        pack, name = CLOTHES[style]
        obj = equip_asset(human, name, pack)
        transfer_weights(human, obj)
        garment_clearance(human, obj)
        return obj
    if style not in {"VNECK_SWEATER", "HOODIE", "JACKET"}:
        raise ValueError(f"Unknown garment {style}")
    obj = _tailored(human, style)
    garment_clearance(human, obj)
    return obj


def skin(human, tone, lip_tint):
    from .portrait_materials import skin_detail

    color = SKINS[tone]
    mat = _pbr_mat(f"{human.name} Portrait Skin", color, roughness=0.44)
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Subsurface Weight"].default_value = 0.16
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.43, 0.23)
    bsdf.inputs["Subsurface Scale"].default_value = 0.0032
    skin_detail(human, mat)
    attr = human.data.color_attributes.get("EXCAD Lips")
    if attr is None or human.data.get("excad_lip_mask_version") != 1:
        if attr is None:
            attr = human.data.color_attributes.new("EXCAD Lips", "FLOAT_COLOR", "POINT")
        colors = np.zeros((len(attr.data), 4), dtype=np.float32)
        colors[:, 3] = 1.0
        directory = Path(_import_mpfb()[3].get_mpfb_data("targets")) / "mouth"
        for lip in ("upperlip", "lowerlip"):
            with gzip.open(directory / f"mouth-{lip}-volume-incr.target.gz", "rt") as inp:
                for line in inp:
                    cols = line.split()
                    if len(cols) == 4 and cols[0].isdigit():
                        weight = min(1, math.sqrt(sum(float(v) ** 2 for v in cols[1:])) * 30)
                        index = int(cols[0])
                        colors[index, :3] = max(colors[index, 0], weight)
        attr.data.foreach_set("color", colors.ravel())
        human.data["excad_lip_mask_version"] = 1
    nodes = mat.node_tree.nodes
    mask = nodes.get("Lip mask") or nodes.new("ShaderNodeVertexColor")
    mask.name, mask.layer_name = "Lip mask", attr.name
    mix = nodes.get("Lip tint") or nodes.new("ShaderNodeMixRGB")
    mix.name = "Lip tint"
    mix.inputs[1].default_value = (*color, 1)
    mix.inputs[2].default_value = (*lip_tint, 1)
    from .human_assets import install_asset_pack

    pack = install_asset_pack("system")
    entry = pack / "skins" / "young_caucasian_female" / "young_caucasian_female.mhmat"
    texture = next(
        line.split(maxsplit=1)[1]
        for line in entry.read_text(encoding="utf8").splitlines()
        if line.startswith("diffuseTexture ")
    )
    albedo = nodes.get("Skin albedo") or nodes.new("ShaderNodeTexImage")
    albedo.name = "Skin albedo"
    albedo.image = bpy.data.images.load(str(entry.parent / texture), check_existing=True)
    tint = nodes.get("Skin tone") or nodes.new("ShaderNodeVectorMath")
    tint.name, tint.operation = "Skin tone", "MULTIPLY"
    factor = [v / r for v, r in zip(color, (0.73, 0.48, 0.37))]
    tint.inputs[1].default_value = factor
    mat["excad_albedo_multiplier"] = factor
    mat.node_tree.links.new(albedo.outputs["Color"], tint.inputs[0])
    mat.node_tree.links.new(tint.outputs[0], mix.inputs[1])
    mat.node_tree.links.new(mask.outputs["Color"], mix.inputs[0])
    mat.node_tree.links.new(mix.outputs[0], bsdf.inputs["Base Color"])
    if human.data.materials:
        human.data.materials[0] = mat
    else:
        human.data.materials.append(mat)
    from .inspect_shading import warm_materials

    warm_materials(human)


def _eye_materials(human, iris_color):
    from .human_assets import install_asset_pack

    atlas = install_asset_pack("system") / "eyes" / "materials" / "grey_eye.png"
    image = bpy.data.images.load(str(atlas), check_existing=True)
    white = _pbr_mat("EXCAD Sclera", (0.82, 0.84, 0.80), 0.31)
    iris = _pbr_mat(human.name + " Iris", iris_color, 0.30)
    pupil = _pbr_mat("EXCAD Pupil", (0.003, 0.004, 0.005), 0.28)
    # Mean linear RGB sampled across the grey iris before applying its requested tint.
    iris_factor = [v / average for v, average in zip(iris_color, (0.2475, 0.3086, 0.2677))]
    for mat, factor in ((white, (1.40, 1.44, 1.42)), (iris, iris_factor), (pupil, None)):
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        bsdf.inputs["Specular IOR Level"].default_value = 0.22
        bsdf.inputs["Coat Weight"].default_value = 1.0
        bsdf.inputs["Coat Roughness"].default_value = 0.045
        bsdf.inputs["Coat IOR"].default_value = 1.336
        if factor is not None:
            tex = nodes.get("Eye atlas") or nodes.new("ShaderNodeTexImage")
            tex.name, tex.image = "Eye atlas", image
            tint = nodes.get("Eye pigmentation") or nodes.new("ShaderNodeVectorMath")
            tint.name, tint.operation = "Eye pigmentation", "MULTIPLY"
            tint.inputs[1].default_value = factor
            mat["excad_albedo_multiplier"] = factor
            mat.node_tree.links.new(tex.outputs["Color"], tint.inputs[0])
            mat.node_tree.links.new(tint.outputs[0], bsdf.inputs["Base Color"])
    return white, iris, pupil


def eyes(human, iris_color, iris_size):
    if not 0.2 <= iris_size <= 0.75:
        raise ValueError("iris_size is the fraction of eyeball radius, from 0.2 to 0.75")
    white, iris, pupil = _eye_materials(human, iris_color)
    with rest_mesh(human) as mesh:
        centers = _eye_centers(human, mesh)
    radius = (centers[0] - centers[1]).length * 0.194
    edge = math.asin(iris_size)
    # Each ring carries its polar angle, its source radius in atlas pixels,
    # and the material on the band connecting it to the previous ring.
    rings = [
        (math.asin(iris_size * f), px, slot)
        for f, px, slot in ((0.32, 32, 2), (0.5, 54, 1), (0.75, 83, 1), (1.0, 119, 1))
    ]
    for i in range(1, 25):
        theta = edge + (math.pi - edge) * i / 25
        progress = min(1.0, (theta - edge) / (math.pi / 2 - edge))
        rings.append((theta, 119 + 153 * progress, 0))

    result = []
    sectors = 40
    for side, center in enumerate(centers):
        cx, cy = ((300.5, 1024 - 718.5), (723.5, 1024 - 303.5))[side]
        coords = [center + Vector((0, -radius * 1.04, 0))]
        texcoords = [(cx / 1024, cy / 1024)]
        faces, slots = [], []
        for j, (theta, atlas_radius, slot) in enumerate(rings):
            ring = math.sin(theta)
            bulge = 0.04 * (1 - (ring / iris_size) ** 2) ** 2 if theta < edge else 0
            for i in range(sectors):
                phi = math.tau * i / sectors
                x, z = math.cos(phi), math.sin(phi)
                coords.append(center + radius * Vector((ring * x, -math.cos(theta) - bulge, ring * z)))
                texcoords.append(((cx + atlas_radius * x) / 1024, (cy + atlas_radius * z) / 1024))
                a, b = 1 + j * sectors + i, 1 + j * sectors + (i + 1) % sectors
                faces.append((0, a, b) if j == 0 else (a - sectors, a, b, b - sectors))
                slots.append(slot)
        end = len(coords)
        coords.append(center + Vector((0, radius, 0)))
        for i in range(sectors):
            faces.append((end - sectors + i, end, end - sectors + (i + 1) % sectors))
            slots.append(0)
        obj = _object(human, "Portrait eye", coords, faces, white)
        obj.data.materials.append(iris)
        obj.data.materials.append(pupil)
        obj["excad_eye_side"], obj["excad_eye_center"], obj["excad_eye_radius"] = side, list(center), radius
        uv = obj.data.uv_layers.new(name="Eye atlas")
        for face, slot in zip(obj.data.polygons, slots):
            face.material_index = slot
            for loop, idx in zip(face.loop_indices, face.vertices):
                # The rear pole stays on the sclera in each sector of the atlas.
                uv.uv[loop].vector = texcoords[idx if idx != end else face.vertices[0]]
        head_weight(obj, human)
        result.append(obj)
    from .inspect_shading import warm_materials

    warm_materials(result[0])
    return result


def refit_eyes(human):
    eyes = [o for o in bpy.context.scene.objects if o.get("excad_human") == human and "excad_eye_side" in o]
    if not eyes:
        return
    with rest_mesh(human) as mesh:
        centers = _eye_centers(human, mesh)
    radius = (centers[0] - centers[1]).length * 0.194
    for eye in eyes:
        center = centers[eye["excad_eye_side"]]
        old = Vector(eye["excad_eye_center"])
        if (center - old).length < 1e-7 and abs(radius / eye["excad_eye_radius"] - 1) < 1e-7:
            continue
        if eye.data.users > 1:
            eye.data = eye.data.copy()
        if not eye.data.shape_keys:
            eye.shape_key_add(name="Basis")
        keys = eye.data.shape_keys
        fitted = keys.key_blocks.get("EXCAD eye fit")
        if fitted is None:
            # Calibrate once from the saved eye, including meshes made by older versions.
            fitted = eye.shape_key_add(name="EXCAD eye fit", from_mix=False)
            eye["excad_eye_rest_center"] = list(old)
            eye["excad_eye_rest_radius"] = eye["excad_eye_radius"]
        local = eye.matrix_world.inverted() @ human.matrix_world
        scale = radius / eye["excad_eye_rest_radius"]
        offset = (local @ center) - (local @ Vector(eye["excad_eye_rest_center"])) * scale
        for dest, source in zip(fitted.data, keys.reference_key.data):
            dest.co = source.co * scale + offset
        fitted.value = 1
        eye["excad_eye_center"], eye["excad_eye_radius"] = list(center), radius


def expression(human, preset):
    # The bundled facial units use a shared topology; this is an expression library,
    # independent of portrait appearance and of the subject in a reference image.
    targets = {"expression/units/caucasian/" + name: value for name, value in EXPRESSIONS[preset].items()}
    set_targets(human, targets, "excad_expression")


def hand_pose(human, preset):
    rig = human.find_armature()
    if not rig:
        raise ValueError("Finger poses require a character armature")
    amount = {"OPEN": 0.0, "RELAXED": 0.22, "FIST": 1.15, "POINT": 1.02}[preset]
    changes = 0
    for side in ("l", "r"):
        for finger_id, finger in enumerate(("thumb", "index", "middle", "ring", "pinky"), 1):
            for joint in range(1, 4):
                candidates = (
                    f"{finger}_{joint:02}_{side}",
                    f"finger{finger_id}-{joint}.{side.upper()}",
                    f"mixamorig:{'Left' if side == 'l' else 'Right'}Hand{finger.title()}{joint}",
                )
                bone = next((rig.pose.bones[n] for n in candidates if n in rig.pose.bones), None)
                if not bone:
                    continue
                angle = 0 if preset == "POINT" and finger == "index" else amount * (0.52 if finger == "thumb" else 1)
                bone.rotation_mode = "QUATERNION"
                bone.rotation_quaternion = Quaternion((1, 0, 0), angle)
                changes += 1
    if not changes:
        raise ValueError("Armature has no recognized finger bones")
    bpy.context.view_layer.update()
    return changes


def remove_wearables(human, objects):
    keys = ("excad_mask_group", "excad_coverage_group")
    removed_masks = {o.get(k) for o in objects for k in keys} - {None}
    for obj in objects:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if isinstance(mesh, bpy.types.Mesh) and not mesh.users:
            bpy.data.meshes.remove(mesh)
    active_masks = {o.get(k) for o in bpy.context.scene.objects if o.get("excad_human") == human for k in keys}
    for name in removed_masks - active_masks:
        for mod in list(human.modifiers):
            if mod.type == "MASK" and mod.vertex_group == name:
                human.modifiers.remove(mod)
        group = human.vertex_groups.get(name)
        if group is not None:
            human.vertex_groups.remove(group)
