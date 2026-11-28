# SPDX-License-Identifier: GPL-2.0-or-later
"""Closed solids in construction coordinates. Evaluated results are committed by FeatureTree."""

import math
from contextlib import contextmanager

import bmesh
import bpy
from mathutils import Matrix, Quaternion, Vector, geometry

from .loft_sections import ordered_sections
from .sketch_profiles import contains, offset_loop, signed_area


DIRECTIONS = {
    "TOP": (0, 0, 1),
    "BOTTOM": (0, 0, -1),
    "FRONT": (0, -1, 0),
    "BACK": (0, 1, 0),
    "LEFT": (-1, 0, 0),
    "RIGHT": (1, 0, 0),
}
AXES = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}


def axis_vector(axis):
    value = Vector(AXES[axis.upper()] if isinstance(axis, str) else axis)
    if len(value) != 3 or value.length < 1e-8:
        raise ValueError("Expected a nonzero 3D axis")
    return value.normalized()


def selected_face(obj, side, placement=None):
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.data
    world = evaluated.matrix_world if placement is None else placement @ evaluated.matrix_world
    normal_matrix = world.to_3x3().inverted().transposed()
    # Combine only adjacent coplanar polygons. Patterned or disconnected pads on
    # the same plane must each retain their own face center.
    parent = list(range(len(mesh.polygons)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners, normals = {}, []
    for polygon in mesh.polygons:
        normal = (normal_matrix @ polygon.normal).normalized()
        normals.append(normal)
        center = world @ polygon.center
        plane = tuple(round(v, 5) for v in normal) + (round(normal.dot(center), 5),)
        for edge in polygon.edge_keys:
            other = owners.setdefault((plane, edge), polygon.index)
            if other != polygon.index:
                parent[root(polygon.index)] = root(other)
    groups = {}
    mesh.calc_loop_triangles()
    points = [world @ vertex.co for vertex in mesh.vertices]
    for triangle in mesh.loop_triangles:
        a, b, c = (points[i] for i in triangle.vertices)
        area = (b - a).cross(c - a).length / 2
        if area <= 1e-12:
            continue
        index = root(triangle.polygon_index)
        region = groups.setdefault(index, [0, Vector((0, 0, 0)), normals[index]])
        region[0] += area
        region[1] += ((a + b + c) / 3) * area
    if not groups:
        raise ValueError("No planar support on the selected object")
    faces = [(area, center / area, n) for area, center, n in groups.values()]
    if side == "LARGEST_PLANAR":
        _, origin, normal = max(faces, key=lambda f: f[0])
    else:
        direction = Vector(DIRECTIONS[side])
        alignment = max(f[2].dot(direction) for f in faces)
        if alignment <= 1e-5:
            raise ValueError(f"No face points {side}")
        facing = [f for f in faces if f[2].dot(direction) > alignment - 1e-5]
        _, origin, normal = max(facing, key=lambda f: (round(f[1].dot(direction), 5), f[0]))
    reference = Vector((1, 0, 0)) if abs(normal.x) < 0.9 else Vector((0, 1, 0))
    u = (reference - normal * reference.dot(normal)).normalized()
    v = normal.cross(u)
    frame = Matrix((u, v, normal)).transposed().to_4x4()
    frame.translation = origin
    return [list(row) for row in frame]


def caps(loops):
    if len(loops) == 1:
        return [tuple(range(len(loops[0])))]
    points = [v for loop in loops for v in loop]
    edges, start = [], 0
    for loop in loops:
        n = len(loop)
        edges.extend((start + i, start + (i + 1) % n) for i in range(n))
        start += n
    vertices, _, faces, original, _, _ = geometry.delaunay_2d_cdt(points, edges, [], 0, 1e-7)
    triangles = []
    for face in faces:
        center = sum((vertices[i] for i in face), Vector((0, 0))) / len(face)
        if sum(contains(center, loop) for loop in loops) % 2:
            if any(len(original[i]) != 1 for i in face):
                raise ValueError("Sketch boundaries touch or intersect")
            triangles.append(tuple(original[i][0] for i in face))
    if not triangles:
        raise ValueError("Sketch has no filled region")
    return triangles


@contextmanager
def construction():
    objects = []
    try:
        yield objects
    finally:
        for obj in reversed(objects):
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)


def new_mesh(objects, verts, faces, orient=True):
    mesh = bpy.data.meshes.new("EXCAD_Construction")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("EXCAD_Construction", mesh)
    bpy.context.collection.objects.link(obj)
    objects.append(obj)
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
        bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1e-8)
        if orient:
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(mesh)
    finally:
        bm.free()
    return obj


def skin_sections(objects, profiles, rings, closed=False, end_profiles=None):
    n = sum(map(len, profiles))
    faces = []
    for j in range(len(rings) if closed else len(rings) - 1):
        next_j = (j + 1) % len(rings)
        offset = 0
        for loop in profiles:
            for i in range(len(loop)):
                a, b = offset + i, offset + (i + 1) % len(loop)
                faces.append((j * n + a, j * n + b, next_j * n + b, next_j * n + a))
            offset += len(loop)
    if not closed:
        start_cap = caps(profiles)
        faces.extend(tuple(reversed(face)) for face in start_cap)
        end_cap = caps(end_profiles) if end_profiles is not None else start_cap
        faces.extend(tuple((len(rings) - 1) * n + i for i in face) for face in end_cap)
    return new_mesh(objects, [v for ring in rings for v in ring], faces)


def profiles(doc, segments=64):
    if not doc.solve()[0]:
        raise ValueError("Sketch constraints are inconsistent")
    loops = doc.build_2d_loops(segments)
    if not loops or any(len(loop) < 3 or abs(signed_area(loop)) < 1e-10 for loop in loops):
        raise ValueError("Expected a non-degenerate closed sketch")
    return loops


def extrude(objects, doc, depth=1, symmetric=False, draft_angle=0, start_overlap=0):
    if abs(depth) < 1e-8 or abs(draft_angle) >= 89:
        raise ValueError("Extrusion needs a nonzero depth and a draft below 89 degrees")
    loops = profiles(doc)
    rings = []
    sections = []
    start = -math.copysign(start_overlap, depth)
    for z in (-depth / 2, depth / 2) if symmetric else (start, depth):
        section = []
        for i, loop in enumerate(loops):
            is_hole = sum(contains(loop[0], other) for j, other in enumerate(loops) if j != i) % 2
            inset = z * math.tan(math.radians(draft_angle)) * (1 if is_hole else -1)
            section.append(offset_loop(loop, inset) if inset else loop)
        sections.append(section)
        rings.append([doc.plane_to_world(u, v, z) for loop in section for u, v in loop])
    return skin_sections(objects, sections[0], rings, end_profiles=sections[-1] if draft_angle else None)


def revolve(objects, doc, axis=None, angle_deg=360, segments=64, origin=None):
    if not 0 < abs(angle_deg) <= 360 or not 3 <= segments <= 2048:
        raise ValueError("Invalid revolve span or segment count")
    loops = profiles(doc)
    points = [Vector(doc.plane_to_world(*p, 0)) for loop in loops for p in loop]
    pivot = Vector(origin if origin is not None else doc.plane_to_world(0, 0, 0))
    direction = (
        axis_vector(axis)
        if axis is not None
        else (Vector(doc.plane_to_world(0, 1, 0)) - Vector(doc.plane_to_world(0, 0, 0))).normalized()
    )
    closed = abs(angle_deg) == 360
    rings = []
    for i in range(segments + int(not closed)):
        rotation = Quaternion(direction, math.radians(angle_deg) * i / segments)
        rings.append([pivot + rotation @ (p - pivot) for p in points])
    return skin_sections(objects, loops, rings, closed)


def resample(loop, n):
    lengths = [(loop[(i + 1) % len(loop)] - p).length for i, p in enumerate(loop)]
    if min(lengths) < 1e-8 or n < len(loop):
        raise ValueError("Loft section contains a collapsed edge")
    # Keep every input corner, even when a rectangular section meets a circle.
    counts = [1] * len(loop)
    for _ in range(n - len(loop)):
        edge = max(range(len(loop)), key=lambda i: lengths[i] / counts[i])
        counts[edge] += 1
    return [p.lerp(loop[(i + 1) % len(loop)], j / counts[i]) for i, p in enumerate(loop) for j in range(counts[i])]


def loft(objects, docs, segments=64, interpolation="SMOOTH", steps=4):
    if interpolation not in {"SMOOTH", "LINEAR"} or not 1 <= steps <= 32:
        raise ValueError("Use SMOOTH or LINEAR loft interpolation and 1..32 steps")
    sections = [profiles(doc, segments) for doc in docs]
    if len(sections) < 2 or any(len(loops) != len(sections[0]) for loops in sections):
        raise ValueError("Loft needs two or more sections with matching boundary counts")
    sections = ordered_sections(docs, sections)
    normals = []
    for doc in docs:
        origin = Vector(doc.plane_to_world(0, 0, 0))
        u, v = (Vector(doc.plane_to_world(*p, 0)) - origin for p in ((1, 0), (0, 1)))
        normals.append(u.cross(v).normalized())
    flipped = [False]
    for a, b in zip(normals, normals[1:]):
        flipped.append(flipped[-1] ^ (a.dot(b) < 0))
    rings = [[] for doc in docs]
    base, end = [], []
    for boundary in zip(*sections):
        n = max(len(loop) for loop in boundary)
        previous = None
        for i, (doc, loop) in enumerate(zip(docs, boundary)):
            if (signed_area(loop) < 0) != flipped[i]:
                loop = list(reversed(loop))
            pts = resample(loop, n) if len(loop) != n else loop
            world = [Vector(doc.plane_to_world(*p, 0)) for p in pts]
            if previous:
                shift = min(
                    range(n), key=lambda j: sum((a - world[(k + j) % n]).length_squared for k, a in enumerate(previous))
                )
                world = world[shift:] + world[:shift]
                pts = pts[shift:] + pts[:shift]
            else:
                base.append(pts)
            if i == len(docs) - 1:
                end.append(pts)
            rings[i].extend(world)
            previous = world
    if interpolation == "SMOOTH" and len(rings) > 2 and steps > 1:
        smooth = []
        padded = [rings[0], *rings, rings[-1]]
        for i in range(len(rings) - 1):
            a, b, c, d = padded[i : i + 4]
            for j in range(steps):
                t = j / steps
                smooth.append(
                    [
                        0.5
                        * (2 * q + (-p + r) * t + (2 * p - 5 * q + 4 * r - s) * t * t + (-p + 3 * q - 3 * r + s) * t**3)
                        for p, q, r, s in zip(a, b, c, d)
                    ]
                )
        rings = smooth + [rings[-1]]
    return skin_sections(objects, base, rings, end_profiles=end)


def _wire_path(obj):
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        neighbors = [[] for _ in mesh.vertices]
        for edge in mesh.edges:
            a, b = edge.vertices
            neighbors[a].append(b)
            neighbors[b].append(a)
        ends = [i for i, adjacent in enumerate(neighbors) if len(adjacent) == 1]
        if (
            not neighbors or mesh.polygons or len(ends) not in (0, 2)
            or any(len(adjacent) not in (1, 2) for adjacent in neighbors)
        ):
            raise ValueError("Evaluated sweep path must be a single unbranched curve")
        index, previous = ends[0] if ends else 0, None
        visited, ordered = set(), []
        while index is not None and index not in visited:
            ordered.append(evaluated.matrix_world @ mesh.vertices[index].co)
            visited.add(index)
            onward = [other for other in neighbors[index] if other != previous]
            previous, index = index, min(onward, default=None)
        if len(visited) != len(neighbors):
            raise ValueError("Evaluated sweep path has disconnected components")
        if not ends:
            ordered.append(ordered[0].copy())
        return ordered
    finally:
        evaluated.to_mesh_clear()


def path_points(path):
    if not isinstance(path, str):
        return [Vector(p) for p in path]
    obj = bpy.data.objects[path]
    if obj.type != "CURVE" or len(obj.data.splines) != 1:
        raise ValueError("Sweep requires a single curve spline")
    curve = obj.data
    if curve.dimensions == "3D" and not (curve.bevel_depth or curve.extrude or curve.bevel_object):
        return _wire_path(obj)
    # Evaluate the guide without its decorative thickness. Copying the object
    # retains its deformers and copying its data preserves the current shape keys.
    temp = obj.copy()
    data = curve.copy()
    try:
        temp.data = data
        data.bevel_mode, data.bevel_depth, data.extrude, data.dimensions = "ROUND", 0, 0, "3D"
        bpy.context.collection.objects.link(temp)
        return _wire_path(temp)
    finally:
        bpy.data.objects.remove(temp, do_unlink=True)
        bpy.data.curves.remove(data)


def sweep(objects, doc, path_curve):
    loops = profiles(doc)
    points = path_points(path_curve)
    if len(points) < 2 or any((a - b).length < 1e-7 for a, b in zip(points, points[1:])):
        raise ValueError("Sweep needs a path with distinct consecutive points")
    closed = (points[0] - points[-1]).length < 1e-7
    if closed:
        points.pop()
    previous = None
    frames = []
    for i, p in enumerate(points):
        before = points[(i - 1) % len(points)] if i or closed else p
        after = points[(i + 1) % len(points)] if i + 1 < len(points) or closed else p
        tangent = ((p - before).normalized() + (after - p).normalized()).normalized()
        if tangent.length < 0.5:
            raise ValueError("Sweep path reverses direction")
        if previous is None:
            guide = Vector((1, 0, 0)) if abs(tangent.x) < 0.9 else Vector((0, 1, 0))
            u = (guide - tangent * guide.dot(tangent)).normalized()
        else:
            u = previous.rotation_difference(tangent) @ u
        frames.append((p, tangent, u))
        previous = tangent
    twist = 0.0
    if closed:
        first_tangent, first_u = frames[0][1:]
        closing_u = previous.rotation_difference(first_tangent) @ u
        twist = math.atan2(first_tangent.dot(closing_u.cross(first_u)), closing_u.dot(first_u))
    lengths = [0.0]
    for a, b in zip(points, points[1:]):
        lengths.append(lengths[-1] + (b - a).length)
    perimeter = lengths[-1] + ((points[0] - points[-1]).length if closed else 0)
    rings = []
    for distance, (p, tangent, u) in zip(lengths, frames):
        if closed:
            u = Quaternion(tangent, twist * distance / perimeter) @ u
        v = tangent.cross(u)
        rings.append([p + u * x + v * y for loop in loops for x, y in loop])
    return skin_sections(objects, loops, rings, closed)


def bake_modifier(obj):
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = bpy.data.meshes.new_from_object(evaluated)
    old = obj.data
    obj.modifiers.clear()
    obj.data = mesh
    if old.users == 0:
        bpy.data.meshes.remove(old)


def boolean(body, tool, op):
    mod = body.modifiers.new("Boolean", "BOOLEAN")
    mod.operation = {"ADD": "UNION", "CUT": "DIFFERENCE", "INTERSECT": "INTERSECT"}[op]
    mod.solver = "EXACT"
    mod.object = tool
    bake_modifier(body)


def finish_edges(body, edges, width, segments):
    if width <= 0 or edges not in {"ALL", "SHARP", "TOP", "BOTTOM"}:
        raise ValueError("Expected positive edge width and ALL, SHARP, TOP or BOTTOM")
    bm = bmesh.new()
    try:
        bm.from_mesh(body.data)
        bm.normal_update()
        chosen = [e for e in bm.edges if e.is_manifold and e.calc_face_angle() > (0.52 if edges == "SHARP" else 1e-5)]
        if edges in {"TOP", "BOTTOM"}:
            z = (max if edges == "TOP" else min)(v.co.z for v in bm.verts)
            chosen = [e for e in chosen if all(abs(v.co.z - z) < 1e-5 for v in e.verts)]
        if not chosen:
            raise ValueError("No matching edges")
        bmesh.ops.bevel(bm, geom=chosen, offset=width, segments=segments, profile=0.5, affect="EDGES", clamp_overlap=True)
        bm.to_mesh(body.data)
    finally:
        bm.free()


def shell(body, thickness, open_faces):
    if thickness <= 0:
        raise ValueError("Wall thickness must be positive")
    frames = [Matrix(selected_face(body, s)) for s in open_faces]
    bm = bmesh.new()
    try:
        bm.from_mesh(body.data)
        bm.normal_update()
        remove = []
        for frame in frames:
            n = frame.col[2].xyz
            remove.extend(
                f
                for f in bm.faces
                if f.normal.dot(n) > 0.99999 and abs(n.dot(f.calc_center_median() - frame.translation)) < 1e-5
            )
        bmesh.ops.delete(bm, geom=list(set(remove)), context="FACES")
        bm.to_mesh(body.data)
    finally:
        bm.free()
    mod = body.modifiers.new("Shell wall", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = -1
    mod.use_even_offset = True
    bake_modifier(body)
