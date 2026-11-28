# SPDX-License-Identifier: GPL-2.0-or-later
"""Render the sketch draw callbacks to a real GPU framebuffer without opening a Blender window."""

import os
from pathlib import Path

import bpy
import gpu
import numpy as np
from mathutils import Matrix, Vector

from excad_addon.ai_bridge import AiBridgeServer
from excad_addon.inspection import _png
from excad_addon import sketch_overlay

bridge = AiBridgeServer.instance()
steps = [
    {"cmd": "sketch_new", "sketch_id": "Overlay", "plane": "XY"},
    {"cmd": "sketch_add_circle", "cu": -0.95, "cv": -0.1, "radius": 0.38},
    {"cmd": "sketch_constraint", "ctype": "FIXED", "a": 0},
    {"cmd": "sketch_slot", "center": [0.76, -0.1], "length": 1.25, "width": 0.52},
    {"cmd": "sketch_offset_loop", "source": 0, "distance": 0.12},
]
response = bridge._dispatch_dict({"cmd": "batch", "steps": steps})
assert response["ok"], response
lines, dimensions = sketch_overlay.drawing()
assert len(lines[True]) == 2 * 64 * 2 and len(lines[False]) > 0
assert {label for _, label in dimensions} == {"R 0.38", "1.25 × 0.52", "Offset 0.12"}

width, height = 512, 224
scale = 117


def to_pixels(point):
    return Vector((width / 2 + scale * point.x, height / 2 + scale * point.y))


gpu.init()
target = gpu.types.GPUOffScreen(width, height)
try:
    with target.bind(), gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(0.30, 0.34, 0.40, 1))
        gpu.matrix.load_identity()
        gpu.matrix.load_projection_matrix(Matrix.Diagonal((2 * scale / width, 2 * scale / height, 1, 1)))
        sketch_overlay._draw_lines()
        gpu.matrix.load_projection_matrix(
            Matrix(((2 / width, 0, 0, -1), (0, 2 / height, 0, -1), (0, 0, 1, 0), (0, 0, 0, 1)))
        )
        sketch_overlay._draw_labels(to_pixels)
        raw = fb.read_color(0, 0, width, height, 4, 0, "UBYTE")
        rgb = np.array(raw, dtype=np.uint8).reshape(height, width, 4)[::-1, :, :3].copy()
finally:
    target.free()
out = Path(os.getenv("EXCAD_TEST_OUTPUT", bpy.app.tempdir))
out.mkdir(parents=True, exist_ok=True)
_png(out / "phase1-sketch-overlay.png", rgb)
r, g, b = rgb.astype(np.int16).transpose(2, 0, 1)
assert np.count_nonzero((b - r > 90) & (b - g > 35)) > 100, "Missing free-geometry pixels"
assert np.count_nonzero((r < 25) & (g < 25) & (b < 25)) > 100, "Missing fully constrained geometry"
assert np.count_nonzero((r > 160) & (g > 140) & (b < 155)) > 40, "Missing dimension text"
print("GPU OVERLAY PASS", gpu.platform.backend_type_get(), gpu.platform.renderer_get())
