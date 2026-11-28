# SPDX-License-Identifier: GPL-2.0-or-later
"""Seeded dimensional edits and scene rollback checks on real Blender data."""

import json
import math
import os
import random
import sys
import time

import bmesh
import bpy

from excad_addon.ai_bridge import AiBridgeServer
from excad_addon.feature_tree import _solid

seed = int(sys.argv[sys.argv.index("--") + 1]) if "--" in sys.argv else 1
iterations = int(os.getenv("EXCAD_CASES", "8"))
rng = random.Random(seed)
bridge = AiBridgeServer.instance()
for obj in list(bpy.context.scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)


def run(cmd, **kw):
    result = bridge._dispatch_dict(dict(cmd=cmd, **kw))
    assert result.get("ok"), (seed, case, cmd, result)
    return result


def volume(obj):
    _solid(obj.data)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    result = abs(bm.calc_volume(signed=True))
    bm.free()
    return result


def section(width, length, height):
    radius = width / 2
    return (length * width + math.pi * radius * radius) * height


started = time.perf_counter()
max_preview = 0
for case in range(iterations):
    w, d, h = rng.uniform(0.6, 4), rng.uniform(0.7, 3), rng.uniform(0.3, 1.4)
    slot_width = min(w, d) * rng.uniform(0.10, 0.19)
    length = w * rng.uniform(0.25, 0.45)
    spec = {
        "version": 1,
        "sketches": {
            "plate": {"shapes": [{"type": "RECT", "width": w, "height": d}]},
            "bore": {
                "support": {"object": "SELF", "face": "TOP"},
                "shapes": [{"type": "SLOT", "start": [-length / 2, 0], "end": [length / 2, 0], "width": slot_width}],
            },
        },
        "features": [
            {"id": "pad", "type": "EXTRUDE", "sketch": "plate", "depth": h},
            {"id": "cut", "type": "EXTRUDE", "sketch": "bore", "depth": -2 * h, "op": "CUT"},
        ],
    }
    run("feature_tree", name="Plate", recipe=spec)
    obj = bpy.data.objects["Plate"]
    expected = w * d * h - section(slot_width, length, h)
    assert math.isclose(volume(obj), expected, rel_tol=0.001), (seed, case, expected, volume(obj))
    assert not obj.ray_cast((0, 0, h + 1), (0, 0, -1))[0], (seed, case, "obstructed slot")
    run("checkpoint", name="dimension")
    digest = run("scene_digest")["h"]
    next_h = h * rng.uniform(0.5, 1.7)
    run("rebuild_feature_tree", object="Plate", parameters={"pad.depth": next_h})
    assert math.isclose(volume(obj), expected * next_h / h, rel_tol=0.001)
    run("rollback", name="dimension")
    assert run("scene_digest", since_hash=digest).get("unchanged"), (seed, case, "rollback digest")
    # Parameter ranges cross surface-net cell corners and exercise narrow blends.
    major = rng.uniform(0.25, 0.75)
    node = {
        "op": "smooth_union",
        "k": rng.uniform(0.01, 0.16),
        "children": [
            {"type": "torus", "major_radius": major, "minor_radius": major * rng.uniform(0.11, 0.33)},
            {
                "type": "radial_array",
                "count": rng.randrange(3, 8),
                "child": {
                    "type": "capsule",
                    "a": [0, 0, 0],
                    "b": [major, 0, 0],
                    "radius": major * rng.uniform(0.07, 0.18),
                },
            },
        ],
    }
    part = run("sdf_generate", tree=node, resolution=rng.randint(34, 76), name="Wheel")
    wheel = bpy.data.objects[part["name"]]
    assert volume(wheel) > 0
    run("place_relative", obj=wheel.name, target="Plate", gap=0)
    assert not run("check_intersections", objects=["Plate", wheel.name])["intersections"]
    preview = run("render_inspect", target="ALL", size=144, path=os.path.join(bpy.app.tempdir, "excad-endurance.png"))
    max_preview = max(max_preview, preview["elapsed_ms"])
    assert not preview["warnings"], (seed, case, preview["warnings"])
    bpy.data.objects.remove(wheel, do_unlink=True)
print(
    "ENDURANCE_RESULT",
    json.dumps(
        {
            "seed": seed,
            "cases": iterations,
            "max_preview_ms": max_preview,
            "elapsed_s": round(time.perf_counter() - started, 2),
            "version": bpy.app.version_string,
        }
    ),
    flush=True,
)
