# SPDX-License-Identifier: GPL-2.0-or-later
"""Fit an asset's actual footprint to the connected surface around a sampled triangle."""

import math

import numpy as np
from mathutils import Matrix, Quaternion, Vector

from . import scene_mesh


def _contact_mesh(points, triangles):
    return scene_mesh.Surface(
        obj=None,
        points=points,
        triangles=triangles,
        colors=None,
        bounds=np.stack((points.min(0), points.max(0))),
        boundary_edges=None,
        nonmanifold_edges=None,
    )


def _connected(triangles, seed):
    """Indices of the triangles connected to the seed by a shared edge."""
    following = np.roll(triangles, -1, axis=1)
    stride = int(triangles.max()) + 1
    edges = (np.minimum(triangles, following).astype(np.int64) * stride + np.maximum(triangles, following)).ravel()
    order = np.argsort(edges)
    shared = edges[order[1:]] == edges[order[:-1]]
    a, b = order[1:][shared] // 3, order[:-1][shared] // 3
    parent = np.arange(len(triangles))
    while True:
        first, second = parent[a], parent[b]
        if np.array_equal(first, second):
            return parent == parent[seed]
        np.minimum.at(parent, np.maximum(first, second), np.minimum(first, second))
        while not np.array_equal(parent, parent[parent]):
            parent = parent[parent]


def _contact_patch(surface, seed, point, basis, asset):
    radius = float(np.linalg.norm(asset.points, axis=1).max())
    hits = surface.bvh.find_nearest_range(Vector(point), radius + max(radius * 1e-5, 1e-6))
    ids = np.unique([seed, *(face for _, _, face, _ in hits)])
    vertices, indices = np.unique(surface.triangles[ids], return_inverse=True)
    triangles = indices.reshape(-1, 3)
    coords = (surface.points[vertices] - point) @ basis.T
    corners = coords[triangles]
    faces_up = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])[:, 2]
    selected = int(np.searchsorted(ids, seed))
    # Follow only the sampled sheet. A nearby disconnected shelf, the back of
    # a thin slab, or the far wall of a cavity must not pull copies across it.
    reachable = (faces_up * np.sign(faces_up[selected]) > 0) & np.all(
        (corners[:, :, :2].min(1) <= asset.bounds[1, :2]) & (corners[:, :, :2].max(1) >= asset.bounds[0, :2]),
        axis=1,
    )
    reachable[selected] = True
    triangles = triangles[reachable]
    seed_index = int(np.count_nonzero(reachable[:selected]))
    triangles = triangles[_connected(triangles, seed_index)]
    used, indices = np.unique(triangles, return_inverse=True)
    return _contact_mesh(coords[used], indices.reshape(-1, 3))


class ScatterAsset:
    def __init__(self, source, parts):
        origin, self.rotation, self.scale = source.matrix_world.decompose()
        up = Vector((0, 0, 1))
        axis = source.matrix_world.to_3x3() @ up
        self.up = axis.normalized() if axis.length_squared else self.rotation @ up
        self.points = np.concatenate([p.points for p in parts]) - origin
        offsets = np.cumsum([0, *(len(p.points) for p in parts[:-1])], dtype=np.int32)
        self.triangles = np.concatenate([p.triangles + offset for p, offset in zip(parts, offsets)])

    def placement(self, surface, seed, point, normal, align):
        n = Vector(normal)
        u = n.orthogonal().normalized()
        basis = np.asarray((u, n.cross(u), n))
        rotate = self.up.rotation_difference(n) if align else Quaternion()
        projected = self.points @ np.asarray(rotate.to_matrix()).T @ basis.T
        low, high = projected.min(0), projected.max(0)
        anchor = (low + high) / 2
        anchor[2] = low[2]
        asset = _contact_mesh(projected - anchor, self.triangles)
        patch = _contact_patch(surface, seed, point, basis, asset)
        # Start clear of the patch. The first contact includes sole vertices,
        # surface peaks and intersections of edges projected along the normal.
        lift = max(0.0, float(patch.bounds[1, 2] - asset.bounds[0, 2]))
        clearance = scene_mesh.vertical_clearance(asset, patch, (0, 0, lift))
        if not math.isfinite(clearance):
            return None
        contact = point + normal * (lift - clearance)
        return Matrix.LocRotScale(Vector(contact - anchor @ basis), rotate @ self.rotation, self.scale)
