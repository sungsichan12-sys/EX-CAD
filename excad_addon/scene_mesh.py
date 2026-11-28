# SPDX-License-Identifier: GPL-2.0-or-later
"""Evaluated world-space surfaces used by inspection and scene layout."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import combinations

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


@dataclass
class Surface:
    obj: object
    points: np.ndarray
    triangles: np.ndarray
    colors: np.ndarray
    bounds: np.ndarray
    boundary_edges: int
    nonmanifold_edges: int
    shading: object = None
    boundary_vertices: object = None
    local_points: object = None
    matrix: object = None
    _bvh: object = None
    _samples: object = None
    _hem: object = None

    @property
    def bvh(self):
        if self._bvh is None:
            self._bvh = BVHTree.FromPolygons(self.points.tolist(), self.triangles.tolist(), all_triangles=True)
        return self._bvh

    @property
    def closed(self):
        return self.boundary_edges == 0 and self.nonmanifold_edges == 0

    def component_samples(self):
        """Keep one surface point per connected component, including isolated islands."""
        if self._samples is None:
            parent = np.arange(len(self.points), dtype=np.int32)
            while True:
                roots = parent[self.triangles]
                lowest = roots.min(axis=1)
                if np.all(roots == lowest[:, None]):
                    break
                for column in roots.T:
                    np.minimum.at(parent, column, lowest)
                while not np.array_equal(parent, parent[parent]):
                    parent = parent[parent]
            indices = np.unique(parent[self.triangles[:, 0]], return_index=True)[1]
            self._samples = self.points[self.triangles[indices]].mean(axis=1)
        return self._samples

    def hem_faces(self):
        if self._hem is None:
            border = np.zeros(len(self.points), dtype=bool)
            if self.boundary_vertices is not None:
                border[self.boundary_vertices] = True
            for _ in range(2):
                near = border[self.triangles].any(axis=1)
                border[self.triangles[near]] = True
            self._hem = border[self.triangles].any(axis=1)
        return self._hem


def resolved_materials(mesh, evaluated):
    slots = evaluated.material_slots
    return [
        slots[i].material if i < len(slots) and slots[i].link == "OBJECT" else mat
        for i, mat in enumerate(mesh.materials)
    ]


@contextmanager
def evaluated_mesh(obj):
    """Borrow the evaluated mesh; Blender needs a temporary copy for converted and edit-mode objects."""
    if obj.type == "MESH" and obj.original.mode != "EDIT":
        yield obj.data
        return
    mesh = obj.to_mesh()
    try:
        yield mesh
    finally:
        obj.to_mesh_clear()


def nearby_pairs(items, tolerance=1e-5):
    """Yield original item indices whose world bounds overlap, including face contact."""
    if len(items) < 8:
        yield from combinations(range(len(items)), 2)
        return
    boxes = np.asarray([item.bounds for item in items])
    low, high = boxes[:, 0], boxes[:, 1]
    tolerance = max(tolerance, float(np.abs(boxes).max()) * 2 * np.finfo(np.float32).eps)
    span = high.max(0) - low.min(0)
    axis = int(np.argmax(span / np.maximum((high - low).mean(0), tolerance)))
    order = np.argsort(low[:, axis], kind="stable")
    ends = np.searchsorted(low[order, axis], high[order, axis] + tolerance, side="right")
    for rank, (index, end) in enumerate(zip(order, ends)):
        candidates = order[rank + 1 : end]
        if len(candidates):
            contact = np.all(
                (low[candidates] <= high[index] + tolerance) & (high[candidates] >= low[index] - tolerance), axis=1
            )
            for other in candidates[contact]:
                yield (index, other) if index < other else (other, index)


class MeshGeometry:
    """Local attributes of an evaluated mesh shared by its linked copies during one query."""

    def __init__(self, mesh, shaded):
        mesh.calc_loop_triangles()
        self.coords = np.empty((len(mesh.vertices), 3), np.float32)
        mesh.vertices.foreach_get("co", self.coords.ravel())
        self.corners = np.empty((len(mesh.loop_triangles), 3), np.int32)
        mesh.loop_triangles.foreach_get("loops", self.corners.ravel())
        vertex_ids = np.empty(len(mesh.loops), np.int32)
        mesh.attributes[".corner_vert"].data.foreach_get("value", vertex_ids)
        self.triangles = vertex_ids[self.corners]
        self.slots = np.zeros(len(self.triangles), np.int32)
        assignment = mesh.attributes.get("material_index")
        if assignment and len(mesh.materials) > 1:
            material_ids = np.empty(len(mesh.polygons), np.int32)
            assignment.data.foreach_get("value", material_ids)
            mesh.loop_triangles.foreach_get("polygon_index", self.slots)
            self.slots = material_ids[self.slots]
        edge_ids = np.empty(len(mesh.loops), np.int32)
        mesh.attributes[".corner_edge"].data.foreach_get("value", edge_ids)
        self.edge_uses = np.bincount(edge_ids, minlength=len(mesh.edges))
        self.boundary_edges = int(np.count_nonzero(self.edge_uses == 1))
        self.nonmanifold_edges = int(np.count_nonzero((self.edge_uses == 0) | (self.edge_uses > 2)))
        self.normals = None
        if shaded:
            self.normals = np.empty((len(mesh.corner_normals), 3), np.float32)
            mesh.corner_normals.foreach_get("vector", self.normals.ravel())
        self.styles = {}

    def appearance(self, mesh, materials):
        from .inspect_shading import material_color, material_shading

        key = tuple(materials)
        if key not in self.styles:
            base = (0.42, 0.57, 0.70)
            palette = np.asarray([material_color(m, base) for m in materials] or [base], dtype=np.float32)
            colors = palette[np.clip(self.slots, 0, len(palette) - 1)]
            detail = None
            if self.normals is not None:
                detail = material_shading(mesh, self.slots, materials, self.corners, self.triangles)
            self.styles[key] = colors, detail
        return self.styles[key]


def _garment_boundary(obj, mesh, edge_uses):
    edges = np.empty((len(mesh.edges), 2), np.int32)
    mesh.edges.foreach_get("vertices", edges.ravel())
    boundary = np.unique(edges[edge_uses == 1])
    seam = obj.vertex_groups.get("EXCAD seams")
    if seam:
        marked = [v.index for v in mesh.vertices if any(g.group == seam.index and g.weight >= 0.15 for g in v.groups)]
        boundary = np.union1d(boundary, np.asarray(marked, dtype=np.int32))
    return boundary


def read_surface(obj, graph, shading=False, cache=None):
    evaluated = obj.evaluated_get(graph)
    with evaluated_mesh(evaluated) as mesh:
        if not mesh or not mesh.polygons:
            return None
        # to_mesh() copies are temporary. Their addresses can be reused after to_mesh_clear().
        shared = cache is not None and evaluated.type == "MESH" and obj.mode != "EDIT"
        key = mesh.as_pointer()
        geometry = cache.get(key) if shared else None
        if geometry is None:
            geometry = MeshGeometry(mesh, shading)
            if shared:
                cache[key] = geometry
        matrix = np.asarray(evaluated.matrix_world, dtype=np.float32)
        points = geometry.coords @ matrix[:3, :3].T + matrix[:3, 3]
        colors, style = geometry.appearance(mesh, resolved_materials(mesh, evaluated))
        detail = boundary = None
        if shading:
            normals = geometry.normals @ np.linalg.pinv(matrix[:3, :3])
            normals /= np.maximum(1e-9, np.linalg.norm(normals, axis=1, keepdims=True))
            detail = (normals[geometry.corners], *style)
            if obj.get("excad_slot") in {"top", "outerwear", "bottom", "footwear"}:
                boundary = _garment_boundary(obj, mesh, geometry.edge_uses)
        return Surface(
            obj,
            points,
            geometry.triangles,
            colors,
            np.stack((points.min(0), points.max(0))),
            geometry.boundary_edges,
            geometry.nonmanifold_edges,
            detail,
            boundary,
            geometry.coords if shading else None,
            evaluated.matrix_world.copy() if shading else None,
        )


def visible_geometry(objects):
    custom_shapes = {
        bone.custom_shape
        for rig in bpy.context.scene.objects
        if rig.type == "ARMATURE"
        for bone in rig.pose.bones
        if bone.custom_shape
    }
    for obj in objects:
        if (
            obj.type in {"MESH", "CURVE", "SURFACE", "FONT", "META"}
            and obj.visible_get()
            and obj not in custom_shapes
            and not obj.get("excad_view_cage")
        ):
            yield obj


def surfaces(target="ALL", shading=False):
    bpy.context.view_layer.update()
    scene = bpy.context.scene
    if target == "ALL":
        objects = scene.objects
    else:
        obj = scene.objects[target]
        objects = [obj, *obj.children_recursive]
        objects.extend(
            o
            for o in scene.objects
            if (o.get("excad_human") == obj or o.get("excad_wearer") == obj.name) and o not in objects
        )
    graph = bpy.context.evaluated_depsgraph_get()
    cache = {}
    return [item for obj in visible_geometry(objects) if (item := read_surface(obj, graph, shading, cache))]


_INSIDE_RAYS = tuple(
    Vector(d).normalized()
    for d in ((0.62371, 0.72193, 0.29977), (-0.31173, 0.80013, -0.20517), (0.92159, -0.37009, 0.12371))
)


def _ray_hits(tree, origin, direction, precision):
    point = origin
    hits, previous, travel, step = 0, None, 0.0, precision
    while True:
        location, _, face, _ = tree.ray_cast(point, direction)
        if face is None:
            return hits
        if face == previous:
            step *= 2
        else:
            hits += 1
            step = precision
        previous = face
        travel = max(travel, (location - origin).dot(direction)) + step
        point = origin + direction * travel


def _inside(surface, point, eps):
    if np.any(point <= surface.bounds[0] + eps) or np.any(point >= surface.bounds[1] - eps):
        return False
    tree = surface.bvh
    p = Vector(point)
    _, _, index, distance = tree.find_nearest(p)
    if index is None or distance <= eps:
        return False
    # The clearance tolerance applies at the query point. Farther along the
    # ray, even two skins closer than this tolerance must count as two hits.
    precision = max(1e-5, float(np.abs(surface.bounds).max())) * float(2 * np.finfo(np.float32).eps)
    first = _ray_hits(tree, p, _INSIDE_RAYS[0], precision)
    if first == 0:
        return False
    second = _ray_hits(tree, p, _INSIDE_RAYS[1], precision)
    if first % 2 == second % 2:
        return bool(first % 2)
    # A seam or a grazing triangle can make one ray ambiguous.
    return bool(_ray_hits(tree, p, _INSIDE_RAYS[2], precision) % 2)


def _crosses(a, b, pairs, eps):
    """Distinguish interpenetration from two closed solids sharing a sloped face."""
    if not (a.closed and b.closed):
        return True
    ia, ib = np.asarray(pairs, dtype=np.int32).T
    ta, tb = a.points[a.triangles[ia]].astype(np.float64), b.points[b.triangles[ib]].astype(np.float64)

    def normals(triangles):
        v = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-20)

    na, nb = normals(ta), normals(tb)
    da = np.einsum("ijk,ik->ij", ta - tb[:, :1], nb)
    db = np.einsum("ijk,ik->ij", tb - ta[:, :1], na)
    straddle_a = (da.min(axis=1) < -eps) & (da.max(axis=1) > eps)
    straddle_b = (db.min(axis=1) < -eps) & (db.max(axis=1) > eps)
    for i in np.flatnonzero(straddle_a & straddle_b):
        axis = np.cross(na[i], nb[i])
        sine = np.linalg.norm(axis)
        axis /= max(sine, 1e-12)

        def interval(triangle, distances):
            points = []
            for j in range(3):
                d, e = distances[j - 1], distances[j]
                if (d > 0) != (e > 0):
                    point = triangle[j - 1] + (triangle[j] - triangle[j - 1]) * (d / (d - e))
                    points.append(point)
            values = np.asarray(points) @ axis
            return values.min(), values.max(), points[0]

        low_a, high_a, start = interval(ta[i], da[i])
        low_b, high_b, _ = interval(tb[i], db[i])
        low, high = max(low_a, low_b), min(high_a, high_b)
        if high - low > eps:
            center = start + axis * ((low + high) * 0.5 - np.dot(start, axis))
            # Almost tangent triangles can share a long interval at their edge
            # with negligible penetration. Measure their available depth normal
            # to the other face before classifying the interval as a crossing.
            triangles = np.stack((ta[i], tb[i]))
            edges = np.roll(triangles, -1, axis=1) - triangles
            margins = np.linalg.norm(np.cross(edges, center - triangles), axis=2)
            margins /= np.maximum(np.linalg.norm(edges, axis=2), 1e-20)
            if margins.min() * sine > eps:
                return True
    # Aligned or coincident boundaries may overlap without a transverse triangle
    # crossing. Sample both sides of the overlapping triangle patch inside both solids.
    coplanar = (np.abs(da).max(axis=1) <= eps) & (np.abs(db).max(axis=1) <= eps)
    for first, second, normal in zip(ta[coplanar], tb[coplanar], na[coplanar]):
        point = _patch_center(first, second, normal, eps)
        if point is not None:
            for side in (-1, 1):
                sample = point + normal * (side * 5 * eps)
                if _inside(a, sample, eps) and _inside(b, sample, eps):
                    return True
    # A triangle edge may track a mesh seam all the way through the other solid.
    for subject, container, ids in ((a, b, ia), (b, a, ib)):
        triangles = subject.triangles[np.unique(ids)]
        vertices = subject.points[np.unique(triangles)]
        corners = subject.points[triangles]
        midpoints = corners.mean(axis=1)
        if any(_inside(container, p, eps) for p in np.concatenate((vertices, midpoints))):
            return True
        # BVH overlap can return only transverse edge neighbours for identical
        # rotated shells, omitting the coplanar face pairs themselves.
        for point, normal in zip(midpoints, normals(corners)):
            _, other_normal, face, _ = container.bvh.find_nearest(Vector(point), eps)
            if face is not None and abs(np.dot(normal, other_normal)) > 0.999:
                for side in (-1, 1):
                    sample = point + normal * (side * 5 * eps)
                    if _inside(subject, sample, eps) and _inside(container, sample, eps):
                        return True
    return False


def _patch_center(first, second, normal, eps):
    # Clip coplanar triangles in their plane. Their individual centers can both
    # be outside the overlap when only a narrow strip is shared.
    polygon = list(first)
    for i in range(3):
        start, end = second[i - 1], second[i]
        inward = np.cross(end - start, normal)
        if np.dot(second[(i + 1) % 3] - start, inward) < 0:
            inward = -inward
        polygon_out = []
        for j, current in enumerate(polygon):
            previous = polygon[j - 1]
            a, b = np.dot(previous - start, inward), np.dot(current - start, inward)
            if (a >= 0) != (b >= 0):
                polygon_out.append(previous + (current - previous) * (a / (a - b)))
            if b >= 0:
                polygon_out.append(current)
        polygon = polygon_out
        if len(polygon) < 3:
            return None
    center = np.mean(polygon, axis=0)
    area = sum(abs(np.dot(normal, np.cross(p - center, polygon[i - 1] - center))) for i, p in enumerate(polygon))
    return center if area > eps * eps else None


def _footprint_faces(points, triangles, low, high):
    for axis in range(2):
        values = points[triangles, axis]
        triangles = triangles[(values.max(axis=1) >= low[axis]) & (values.min(axis=1) <= high[axis])]
    return triangles


def _surface_over(surface, bounds, eps):
    low, high = bounds[0, :2] - eps, bounds[1, :2] + eps
    if np.all(surface.bounds[0, :2] >= low) and np.all(surface.bounds[1, :2] <= high):
        return surface
    faces = _footprint_faces(surface.points, surface.triangles, low, high)
    if not len(faces):
        return None
    if len(faces) == len(surface.triangles):
        return surface
    used, indices = np.unique(faces, return_inverse=True)
    points = surface.points[used]
    return replace(
        surface, points=points, triangles=indices.reshape(-1, 3), bounds=np.stack((points.min(0), points.max(0))),
        _bvh=None,
    )


def _separated_along_z(upper, lower, eps):
    # Only the overlapping footprint can support the other mesh. Peaks on a
    # distant hillside would otherwise raise every ray and edge search.
    upper, lower = _surface_over(upper, lower.bounds, eps), _surface_over(lower, upper.bounds, eps)
    if upper is None or lower is None:
        return True
    lift = float(lower.bounds[1, 2] - upper.bounds[0, 2])
    if lift <= eps:
        return True
    clearance = vertical_clearance(upper, lower, (0, 0, lift), best=lift + 2 * eps, stop_at=lift - 2 * eps)
    return clearance >= lift - eps


def _touches_sheet(sheet, solid, face_ids, eps):
    triangles = sheet.points[sheet.triangles[np.unique(face_ids)]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    axis = int(np.argmax(np.abs(normals).sum(axis=0)))
    contact = triangles[:, :, axis].mean()
    if axis != 2:
        axes = ((1, 2, 0), (2, 0, 1))[axis]
        sheet = replace(sheet, points=sheet.points[:, axes], bounds=sheet.bounds[:, axes], _bvh=None)
        solid = replace(solid, points=solid.points[:, axes], bounds=solid.bounds[:, axes], _bvh=None)
    upper, lower = (solid, sheet) if solid.bounds[:, 2].mean() >= contact else (sheet, solid)
    return _separated_along_z(upper, lower, eps)


def overlap_kind(a, b, epsilon=1e-5):
    scale = max(float(np.abs(a.bounds).max()), float(np.abs(b.bounds).max()))
    epsilon = max(epsilon, scale * np.finfo(np.float32).eps * 2)
    depth = np.minimum(a.bounds[1], b.bounds[1]) - np.maximum(a.bounds[0], b.bounds[0])
    if np.min(depth) < -epsilon or (a.closed and b.closed and np.min(depth) <= epsilon):
        return None
    pairs = a.bvh.overlap(b.bvh)
    if pairs and a.closed != b.closed:
        # A mesh with an intentional boundary, such as a terrain surface, can
        # touch a closed sole or bracket without cutting into it.
        sheet, solid, column = (b, a, 1) if a.closed else (a, b, 0)
        if _touches_sheet(sheet, solid, np.asarray(pairs)[:, column], epsilon):
            return None
    if pairs and a.closed and b.closed and depth[2] < 0.1 * min(np.ptp(a.bounds[:, 2]), np.ptp(b.bounds[:, 2])):
        # At grazing support, BVH can report a short line crossing at the rim
        # although the solids only meet. Check the actual vertical separation.
        upper, lower = (a, b) if a.bounds[0, 2] > b.bounds[0, 2] else (b, a)
        if _separated_along_z(upper, lower, epsilon):
            return None
    if pairs and _crosses(a, b, pairs, epsilon):
        return "crossing"
    for subject, container in ((a, b), (b, a)):
        if container.closed and any(_inside(container, p, epsilon) for p in subject.component_samples()):
            return "contained"
    return None


def assembly_root(obj):
    while obj.parent:
        obj = obj.parent
    return obj


def _project_vertices(subject, other, shift, sign, best, stop_at):
    """Measure vertical separation at projected vertices, in the other mesh's space."""
    points = subject.points + shift
    inside = np.all(
        (points[:, :2] >= other.bounds[0, :2] - 1e-6) & (points[:, :2] <= other.bounds[1, :2] + 1e-6), axis=1
    )
    points = points[inside]
    cap = other.bounds[1 if sign < 0 else 0, 2]
    origin_z = float(cap - sign * max(1.0, np.ptp(other.bounds[:, 2])))
    ray = Vector((0, 0, sign))
    for i in np.argsort(-sign * points[:, 2]):
        point = points[i]
        if -sign * (point[2] - cap) >= best:
            break
        hit = other.bvh.ray_cast(Vector((point[0], point[1], origin_z)), ray)[0]
        if hit is not None:
            best = min(best, -sign * float(point[2] - hit.z))
            if best <= stop_at:
                break
    return best


def _project_edges(surface, shift, limit, bottom, low_xy, high_xy):
    tris = surface.triangles
    heights = surface.points[tris, 2] + shift[2]
    tri = tris[heights.min(axis=1) <= limit] if bottom else tris[heights.max(axis=1) >= limit]
    # A small object on a large terrain can share a height with thousands of
    # remote edges. Only triangles inside the shared footprint can touch.
    low, high = low_xy - shift[:2], high_xy - shift[:2]
    tri = _footprint_faces(surface.points, tri, low, high)
    edges = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    start, end = edges.min(axis=1).astype(np.int64), edges.max(axis=1)
    count = len(surface.points)
    keys = np.unique(start * count + end)
    pairs = np.column_stack((keys // count, keys % count))
    return surface.points[pairs] + shift


def vertical_clearance(upper, lower, shift=(0, 0, 0), best=float("inf"), *, stop_at=1e-6):
    """First contact distance when moving a translated upper mesh straight down.

    Over each pair of projected triangles the difference in height is linear.
    Its minimum is at a projected vertex or at the intersection of two edges.
    A caller checking separation can stop once any clearance proves penetration.
    """
    shift = np.asarray(shift, dtype=np.float64)
    lo, hi = upper.bounds + shift
    if (
        lo[2] - lower.bounds[1, 2] >= best
        or np.any(hi[:2] < lower.bounds[0, :2])
        or np.any(lo[:2] > lower.bounds[1, :2])
    ):
        return best
    best = _project_vertices(upper, lower, shift, -1, best, stop_at)
    if best > stop_at:
        best = _project_vertices(lower, upper, -shift, 1, best, stop_at)
    if best <= stop_at:
        return best
    low_xy = np.maximum(lo[:2], lower.bounds[0, :2]) - 1e-6
    high_xy = np.minimum(hi[:2], lower.bounds[1, :2]) + 1e-6
    above = _project_edges(upper, shift, lower.bounds[1, 2] + best, True, low_xy, high_xy)
    below = _project_edges(lower, np.zeros(3), lo[2] - best, False, low_xy, high_xy)
    if not len(above) or not len(below):
        return best
    # Fewer scalar iterations for a dense asset sitting on a simple support.
    sign = 1
    if len(above) > len(below):
        above, below, sign = below, above, -1
    minima, maxima = below[:, :, :2].min(axis=1), below[:, :, :2].max(axis=1)

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    for a, b in above:
        mask = np.all(maxima >= np.minimum(a, b)[:2], axis=1) & np.all(minima <= np.maximum(a, b)[:2], axis=1)
        edges = below[mask]
        if not len(edges):
            continue
        start, end = edges[:, 0], edges[:, 1]
        r, s, q = b - a, end - start, start - a
        denom = cross(r, s)
        valid = np.abs(denom) > 1e-14
        t = np.divide(cross(q, s), denom, out=np.zeros_like(denom), where=valid)
        u = np.divide(cross(q, r), denom, out=np.zeros_like(denom), where=valid)
        valid &= (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        if np.any(valid):
            distance = sign * (a[2] + t * r[2] - start[:, 2] - u * s[:, 2])
            best = min(best, float(distance[valid].min()))
            if best <= stop_at:
                break
    return best


def intersections(items, epsilon=1e-5):
    result = []
    for a, b in combinations(items, 2):
        kind = overlap_kind(a, b, epsilon)
        if kind:
            result.append({"objects": [a.obj.name, b.obj.name], "kind": kind})
    return result
