# SPDX-License-Identifier: GPL-2.0-or-later
"""Fuse registered orthographic silhouettes and near-surface depths into a continuous field."""

import math

import numpy as np


def image_coordinates(points, face, size, width, height):
    # The orthographic frame ends at pixel edges; integer array indices address pixel centers.
    px = (points @ np.asarray(face["right"], np.float32) / size + 0.5) * width - 0.5
    py = (points @ np.asarray(face["up"], np.float32) / size + 0.5) * height - 0.5
    return px, py


def _silhouette(mask, resolution):
    """A signed narrow band in camera space; negative samples lie in the foreground."""
    height, width = mask.shape
    pixel = 1 / max(width, height)
    field = np.where(mask, -0.5, 0.5).astype(np.float32)
    inside, outside = mask, ~mask

    def shrink(region):
        padded = np.pad(region, 1, mode="edge")
        return region & padded[1:-1, :-2] & padded[1:-1, 2:] & padded[:-2, 1:-1] & padded[2:, 1:-1]

    for _ in range(min(16, math.ceil(3 / (resolution * pixel)))):
        inside, outside = shrink(inside), shrink(outside)
        field += outside.astype(np.float32) - inside
    return field * pixel


def _observation(points, face, size, mask, silhouette, depth):
    height, width = mask.shape
    px, py = image_coordinates(points, face, size, width, height)
    outside = (px < -0.5) | (px > width - 0.5) | (py < -0.5) | (py > height - 0.5)
    px, py = np.clip(px, 0, width - 1), np.clip(py, 0, height - 1)
    x0, y0 = px.astype(np.int32), py.astype(np.int32)
    fx, fy = px - x0, py - y0
    field = np.zeros(len(points), np.float32)
    if depth is not None:
        total, weights = np.zeros_like(field), np.zeros_like(field)
        near, far = np.ones_like(field), np.zeros_like(field)
    for x, wx in ((x0, 1 - fx), (np.minimum(x0 + 1, width - 1), fx)):
        for y, wy in ((y0, 1 - fy), (np.minimum(y0 + 1, height - 1), fy)):
            weight = wx * wy
            field += weight * silhouette[y, x]
            if depth is not None:
                valid = mask[y, x] & (weight > 1e-6)
                sample = depth[y, x]
                total += np.where(valid, weight * sample, 0)
                weights += weight * valid
                near = np.minimum(near, np.where(valid, sample, 1))
                far = np.maximum(far, np.where(valid, sample, 0))
    if depth is not None:
        observed = total / np.maximum(weights, 1e-8)
        # A pixel footprint at an occluding rim can contain both the foreground
        # and the cavity floor. Interpolating the jump would carve away the rim.
        observed = np.where(far - near > 3 / min(width, height), near, observed)
        distance = 0.5 - points @ np.asarray(face["normal"], np.float32) / size
        field = np.maximum(field, np.where(weights > 0, observed - distance, -1))
    field[outside] = 1
    return field


def fuse(views, size, center, resolution):
    from .sdf_engine import surface_nets

    step = size / resolution
    coordinates = step * (np.arange(resolution, dtype=np.float32) + 0.5) - size / 2
    local = np.stack(np.meshgrid(coordinates, coordinates, coordinates, indexing="ij"), axis=-1).reshape(-1, 3)
    field = np.full(len(local), -1, dtype=np.float32)
    for face, mask, depth in views:
        band = _silhouette(mask, resolution)
        for start in range(0, len(local), 65536):
            batch = slice(start, start + 65536)
            field[batch] = np.maximum(field[batch], _observation(local[batch], face, size, mask, band, depth))
    if not (field < 0).any():
        raise ValueError("Visual hull intersection is empty; check the alignment, masks and depth range")
    volume = field.reshape((resolution,) * 3)
    # A silhouette may touch the reconstruction box. An exterior ring closes its boundary.
    padded = np.pad(volume, 1, constant_values=1 / resolution)
    return surface_nets(padded, np.asarray(center) - size / 2 - step / 2, step)
