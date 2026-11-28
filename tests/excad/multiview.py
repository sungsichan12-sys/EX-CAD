# SPDX-License-Identifier: GPL-2.0-or-later
"""Reconstruct and texture a known concave model from 12 and 20 orthographic observations."""

import math
import os
from pathlib import Path
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Vector

from excad_addon.ai_bridge import AiBridgeServer
from excad_addon.feature_tree import _solid
from excad_addon.image_colors import linear_to_srgb
from excad_addon.inspect_shading import Shading
from excad_addon.inspection import _png
from excad_addon.polyhedron_engine import get_polyhedron_faces
from excad_addon.scene_mesh import surfaces


bridge = AiBridgeServer.instance()
for obj in list(bpy.context.scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)


def run(cmd, **fields):
    reply = bridge._dispatch_dict(dict(cmd=cmd, **fields))
    assert reply["ok"], (cmd, reply)
    return reply


def glaze(p, n):
    """Asymmetric slip decoration in world space; the interior has a different glaze."""
    if n.z > 0.5 or n.x * p.x + n.y * p.y < -0.05:
        return (219, 192, 145)
    band = p.z - 0.065 * math.sin(3 * math.atan2(p.y, p.x) + 0.65)
    if band > 0.23 or -0.12 < band < -0.095:
        return (204, 124, 58)
    if -0.095 <= band < 0.02:
        return (220, 205, 155)
    return (32, 87, 112)


def photograph(surface, directory, poly_type, width, size):
    for face in get_polyhedron_faces(poly_type):
        eye, right, up = (Vector(face[k]) for k in ("normal", "right", "up"))
        rgb = np.full((width, width, 3), 255, np.uint8)
        depth = np.full_like(rgb, 255)
        for y in range(width):
            for x in range(width):
                ray_origin = size * (0.5 * eye + ((x + 0.5) / width - 0.5) * right + (0.5 - (y + 0.5) / width) * up)
                p, n, face_id, distance = surface.bvh.ray_cast(ray_origin, -eye)
                if face_id is not None:
                    rgb[y, x] = glaze(p, n)
                    depth[y, x] = round(255 * distance / size)
        image = directory / face["filename"]
        _png(image, rgb)
        _png(image.with_name(image.stem + "_depth.png"), depth)


def check_pattern(reference, model):
    shading = Shading([model])
    errors = []
    for theta in (0.1, 1.15, 2.8, 4.1, 5.2):
        radial = Vector((math.cos(theta), math.sin(theta), 0))
        for z in (-0.22, -0.05, 0.13, 0.28):
            origin = radial + Vector((0, 0, z))
            p, n, original_id, _ = reference.bvh.ray_cast(origin, -radial)
            q, _, triangle, _ = model.bvh.ray_cast(origin, -radial)
            assert original_id is not None and triangle is not None
            # Isolate regions with a single expected color a voxel either side.
            if glaze(p + Vector((0, 0, 0.03)), n) != glaze(p - Vector((0, 0, 0.03)), n):
                continue
            _, color, _ = shading.sample(np.array([triangle]), np.array([q]))
            errors.append(np.max(np.abs(255 * linear_to_srgb(color[0]) - glaze(p, n))))
    assert len(errors) >= 12 and np.quantile(errors, 0.80) < 32, errors
    return round(float(np.median(errors)), 2)


outer = {"type": "cone", "height": 0.72, "radius": 0.36, "top_radius": 0.58}
inner = {"type": "cone", "height": 0.70, "radius": 0.30, "top_radius": 0.52, "center": [0, 0, 0.13]}
resolution, width = 68, 112
if "--" in sys.argv:
    rng = np.random.default_rng(int(sys.argv[sys.argv.index("--") + 1]))
    outer.update(
        height=rng.uniform(0.70, 0.80),
        radius=rng.uniform(0.34, 0.40),
        top_radius=rng.uniform(0.56, 0.62),
    )
    wall = rng.uniform(0.055, 0.085)
    inner.update(
        height=outer["height"],
        radius=outer["radius"] - wall,
        top_radius=outer["top_radius"] - wall,
        center=[0, 0, rng.uniform(0.12, 0.17)],
    )
    resolution, width = int(rng.integers(62, 79)), int(rng.integers(100, 133))
tree = {"op": "smooth_subtract", "k": 0.018, "children": [outer, inner]}
source = run("sdf_generate", name="Source vessel", tree=tree, resolution=112)["name"]
ref = surfaces(source)[0]
down = Vector((0, 0, -1))
top = Vector((0, 0, 1))
base = ref.bvh.ray_cast(top, down)[0].z
output = Path(os.getenv("EXCAD_TEST_OUTPUT", bpy.app.tempdir))
output.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    for poly_type, count in (("DODECAHEDRON_12", 12), ("ICOSAHEDRON_20", 20)):
        directory = Path(tmp, poly_type)
        directory.mkdir()
        size = 1.65
        photograph(ref, directory, poly_type, width=width, size=size)
        options = dict(image_dir=str(directory), poly_type=poly_type, size=size, center=[0, 0, 0], resolution=resolution)
        hull = run("poly_reconstruct", name="Silhouette hull", use_depth=False, project_colors=False, **options)
        occluded = surfaces(hull["name"])[0].bvh.ray_cast(top, down)[0].z
        result = run("poly_reconstruct", name="Decorated vessel", **options)
        obj = bpy.data.objects[result["name"]]
        _solid(obj.data)
        assert result["views_used"] == count
        item = surfaces(obj.name, shading=True)[0]
        center_hit = item.bvh.ray_cast(top, down)[0].z
        assert abs(center_hit - base) < 0.073 and occluded - center_hit > 0.34, (poly_type, base, center_hit, occluded)
        radii_error = []
        for angle in np.linspace(0, 2 * math.pi, 128, endpoint=False):
            ray = Vector((math.cos(angle), math.sin(angle), 0))
            origin = ray + Vector((0, 0, 0.13))
            radius = ref.bvh.ray_cast(origin, -ray)[3]
            radii_error.append(item.bvh.ray_cast(origin, -ray)[3] - radius)
        rms = float(np.sqrt(np.mean(np.square(radii_error))))
        assert rms < size / options["resolution"] * 0.45, (poly_type, rms)
        error = check_pattern(ref, item)
        image = obj.active_material.node_tree.nodes.get("Image Texture").image
        assert list(image.size) == [2048, 2048] and obj.data.uv_layers["EXCAD Atlas"]
        run("place_relative", obj=obj.name, relation="ALIGN_FLOOR", gap=0)
        path = output / "phase3-uv.png" if count == 20 else directory / "preview.png"
        report = run("render_inspect", target=obj.name, size=224, path=str(path))
        assert not report["warnings"], report
        print("MULTIVIEW", count, "floor error", round(abs(center_hit - base), 4), "side RMS", round(rms, 4),
              "median sRGB error", error, flush=True)
        for name in (obj.name, hull["name"]):
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
print("MULTIVIEW PASS", bpy.app.version_string)
