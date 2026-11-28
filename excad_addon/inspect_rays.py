# SPDX-License-Identifier: GPL-2.0-or-later
"""Ray queries for a four-view preview of evaluated geometry and dense linked copies."""

import numpy as np
from mathutils.bvhtree import BVHTree


def _world_tree(shading):
    return BVHTree.FromPolygons(shading.points.tolist(), shading.triangles.tolist(), all_triangles=True)


class PreviewRays:
    def __init__(self, items, shading):
        unique = {id(s.triangles): len(s.triangles) for s in items}
        duplicated = sum(len(s.triangles) for s in items) - sum(unique.values())
        self.tree = None
        if duplicated < 60_000:
            self.tree = items[0].bvh if len(items) == 1 else _world_tree(shading)
            return

        # Build each linked mesh's tree in local coordinates once. Diagnostic
        # intersections still use exact, evaluated world-space triangles.
        self.shading = shading
        self.instances = []
        trees = {}
        offset = 0
        for s in items:
            try:
                inverse = s.matrix.inverted()
            except ValueError:
                inverse = None
            key = id(s.triangles)
            if inverse is None:
                tree = s.bvh
            else:
                if key not in trees:
                    trees[key] = BVHTree.FromPolygons(
                        s.local_points.tolist(),
                        s.triangles.tolist(),
                        all_triangles=True,
                    )
                tree = trees[key]
            self.instances.append((tree, inverse, offset))
            offset += len(s.triangles)
        boxes = np.asarray([s.bounds for s in items])
        self.centers = boxes.mean(axis=1)
        self.extents = (boxes[:, 1] - boxes[:, 0]) * 0.5

    def _project_bounds(self, start, down, across, size):
        """Conservative pixel rectangles for the orthographic view."""
        basis = np.stack((across, down), axis=1)
        basis /= np.sum(basis * basis, axis=0)
        centers = (self.centers - start) @ basis
        radii = self.extents @ np.abs(basis)
        self.low = np.maximum(np.floor(centers - radii).astype(int) - 1, 0)
        self.high = np.minimum(np.ceil(centers + radii).astype(int) + 2, size)
        # For heavily superimposed meshes a whole-scene tree skips occluded
        # objects more cheaply than casting a ray into every copy's rectangle.
        if np.prod(np.maximum(self.high - self.low, 0), axis=1).sum() > 8 * size * size:
            self.tree = _world_tree(self.shading)

    def grid(self, rows, ray, size):
        start, down, across = rows
        ids = np.full((size, size), -1, np.int32)
        positions = np.zeros((size, size, 3), np.float32)
        if self.tree is None:
            self._project_bounds(start, down, across, size)
        if self.tree is not None:
            for y in range(size):
                point = start + down * y
                for x in range(size):
                    hit, _, face, _ = self.tree.ray_cast(point, ray)
                    if face is not None:
                        ids[y, x], positions[y, x] = face, hit
                    point += across
            return ids, positions

        self.view_rays = []
        depths = np.full((size, size), np.inf)
        for (tree, inverse, base), (x0, y0), (x1, y1) in zip(
            self.instances, self.low, self.high
        ):
            origin, dy, dx, direction = start, down, across, ray
            if inverse is not None:
                linear = inverse.to_3x3()
                origin, dy, dx, direction = (
                    inverse @ start,
                    linear @ down,
                    linear @ across,
                    linear @ ray,
                )
            # BVHTree reports a distance along a normalized local direction.
            to_world = 1.0 / direction.length
            direction = direction.normalized()
            self.view_rays.append((direction, to_world))
            origin = origin + dx * int(x0)
            for y in range(y0, y1):
                point = origin + dy * y
                for x in range(x0, x1):
                    _, _, face, distance = tree.ray_cast(point, direction)
                    if face is not None and distance * to_world < depths[y, x]:
                        depths[y, x], ids[y, x] = distance * to_world, base + face
                    point += dx
        y, x = np.nonzero(ids >= 0)
        positions[y, x] = (
            np.asarray(start)
            + x[:, None] * across
            + y[:, None] * down
            + depths[y, x, None] * ray
        )
        return ids, positions

    def behind(self, origin, ray, x, y):
        """Continue a cutout pixel through overlapping meshes along its original ray."""
        if self.tree is not None:
            hit, _, face, _ = self.tree.ray_cast(origin, ray)
            return hit, face
        pixel = (x, y)
        candidates = np.flatnonzero(
            np.all((self.low <= pixel) & (pixel < self.high), axis=1)
        )
        nearest, best = float("inf"), None
        for i in candidates:
            tree, inverse, base = self.instances[i]
            direction, to_world = self.view_rays[i]
            local = inverse @ origin if inverse is not None else origin
            _, _, face, distance = tree.ray_cast(local, direction)
            if face is not None and distance * to_world < nearest:
                nearest, best = distance * to_world, base + face
        return (origin + ray * nearest, best) if best is not None else (None, None)
