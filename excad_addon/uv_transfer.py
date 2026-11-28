# SPDX-License-Identifier: GPL-2.0-or-later
"""Transfer painted UV charts onto a rebuilt surface without crossing atlas seams."""

import numpy as np
from mathutils.bvhtree import BVHTree


def _array(items, prop, width, dtype):
    values = np.empty((len(items), width), dtype=dtype)
    items.foreach_get(prop, values.ravel())
    return values.reshape(-1) if width == 1 else values


def _islands(mesh, uv, polygon_ids, loop_vertices):
    """Join adjacent faces only when both UV endpoints agree along their shared edge."""
    starts = _array(mesh.polygons, "loop_start", 1, np.int32)
    sizes = _array(mesh.polygons, "loop_total", 1, np.int32)
    following = np.arange(len(mesh.loops)) + 1
    following[starts + sizes - 1] = starts
    forward = loop_vertices < loop_vertices[following]
    quantized = np.rint(uv.astype(np.float64) * 1e6).astype(np.int64)
    ends = quantized[following]
    keys = np.column_stack(
        (
            _array(mesh.loops, "edge_index", 1, np.int32),
            np.where(forward[:, None], quantized, ends),
            np.where(forward[:, None], ends, quantized),
        )
    )
    order = np.lexsort(keys.T[::-1])
    shared = (keys[order[1:]] == keys[order[:-1]]).all(axis=1)
    a, b = polygon_ids[order[:-1][shared]], polygon_ids[order[1:][shared]]
    parent = np.arange(len(mesh.polygons))
    while True:
        lo, hi = np.minimum(parent[a], parent[b]), np.maximum(parent[a], parent[b])
        if np.array_equal(lo, hi):
            return parent
        np.minimum.at(parent, hi, lo)
        while not np.array_equal(parent, parent[parent]):
            parent = parent[parent]


def _nearest(tree, points):
    faces = np.empty(len(points), np.int32)
    projected = np.empty((len(points), 3), np.float32)
    for i, p in enumerate(points):
        pos, _, faces[i], _ = tree.find_nearest(p)
        projected[i] = pos
    return projected, faces


def _interpolate(triangles, points, uv):
    a, b, c = triangles.transpose(1, 0, 2)
    u, v, p = b - a, c - a, points - a

    def dot(x, y):
        return np.einsum("ij,ij->i", x, y)

    uu, uv_dot, vv, pu, pv = dot(u, u), dot(u, v), dot(v, v), dot(p, u), dot(p, v)
    denom = uu * vv - uv_dot * uv_dot
    reciprocal = np.divide(1.0, denom, out=np.zeros_like(denom), where=denom > 0)
    w1 = ((pu * vv - pv * uv_dot) * reciprocal)[:, None]
    w2 = ((pv * uu - pu * uv_dot) * reciprocal)[:, None]
    return uv[:, 0] + w1 * (uv[:, 1] - uv[:, 0]) + w2 * (uv[:, 2] - uv[:, 0])


def transfer_uv(source, result, normalize=False):
    """Choose a source chart per destination polygon, then project its individual corners."""
    if not source.uv_layers or not source.polygons or not result.polygons:
        return
    source.calc_loop_triangles()
    src = _array(source.vertices, "co", 3, np.float32)
    dst = _array(result.vertices, "co", 3, np.float32)
    centers = _array(result.polygons, "center", 3, np.float32)
    if normalize:
        # A dimensional edit can grow each model axis independently.
        src = (src - src.min(0)) / np.maximum(np.ptp(src, axis=0), 1e-8)
        low, span = dst.min(0), np.maximum(np.ptp(dst, axis=0), 1e-8)
        dst, centers = (dst - low) / span, (centers - low) / span

    triangles = _array(source.loop_triangles, "vertices", 3, np.int32)
    triangle_loops = _array(source.loop_triangles, "loops", 3, np.int32)
    triangle_faces = _array(source.loop_triangles, "polygon_index", 1, np.int32)
    tree = BVHTree.FromPolygons(src.tolist(), triangles.tolist(), all_triangles=True)
    nearest, vertex_tri = _nearest(tree, dst)
    _, face_tri = _nearest(tree, centers)
    src_vertices = _array(source.loops, "vertex_index", 1, np.int32)
    dst_vertices = _array(result.loops, "vertex_index", 1, np.int32)
    src_faces = np.repeat(np.arange(len(source.polygons)), _array(source.polygons, "loop_total", 1, np.int32))
    dst_faces = np.repeat(np.arange(len(result.polygons)), _array(result.polygons, "loop_total", 1, np.int32))
    for layer in source.uv_layers:
        uv = _array(layer.uv, "vector", 2, np.float32)
        chart = _islands(source, uv, src_faces, src_vertices)[triangle_faces]
        target_chart = chart[face_tri[dst_faces]]
        tri = vertex_tri[dst_vertices].copy()
        points = nearest[dst_vertices].copy()
        boundary = np.flatnonzero(chart[tri] != target_chart)
        if len(boundary):
            # A vertex on a new seam needs an independent projection on each side.
            order = np.argsort(chart)
            sorted_charts = chart[order]
            boundary = boundary[np.argsort(target_chart[boundary])]
            splits = np.flatnonzero(np.diff(target_chart[boundary])) + 1
            for corners in np.split(boundary, splits):
                island = target_chart[corners[0]]
                start, end = np.searchsorted(sorted_charts, (island, island + 1))
                ids = order[start:end]
                used, indices = np.unique(triangles[ids], return_inverse=True)
                local = BVHTree.FromPolygons(src[used].tolist(), indices.reshape(-1, 3).tolist(), all_triangles=True)
                vertices, inverse = np.unique(dst_vertices[corners], return_inverse=True)
                local_points, local_triangles = _nearest(local, dst[vertices])
                tri[corners] = ids[local_triangles[inverse]]
                points[corners] = local_points[inverse]
        result_uv = _interpolate(src[triangles[tri]].astype(np.float64), points, uv[triangle_loops[tri]])
        dest = result.uv_layers.get(layer.name) or result.uv_layers.new(name=layer.name, do_init=False)
        dest.uv.foreach_set("vector", np.ascontiguousarray(result_uv, dtype=np.float32).ravel())
        if layer.active_render:
            dest.active_render = True
            edges = _array(result.loops, "edge_index", 1, np.int32)
            by_edge = np.argsort(edges)
            a, b = by_edge[:-1], by_edge[1:]
            split = (edges[a] == edges[b]) & (target_chart[a] != target_chart[b])
            seam = np.zeros(len(result.edges), np.bool_)
            seam[edges[a[split]]] = True
            result.edges.foreach_set("use_seam", seam)
    result.uv_layers.active = result.uv_layers[source.uv_layers.active.name]
    result.update()
