# SPDX-License-Identifier: GPL-2.0-or-later
"""An assembly of 625 separate studs must report a single overlap and a single unsupported part."""

from pathlib import Path
import tempfile

import bpy
import numpy as np

from excad_addon.ai_bridge import AiBridgeServer


bridge = AiBridgeServer.instance()


def run(cmd, **fields):
    result = bridge._dispatch_dict(dict(cmd=cmd, **fields))
    assert result["ok"], (cmd, result)
    return result


for obj in list(bpy.context.scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
side = 25
bpy.ops.mesh.primitive_cube_add(size=1)
stand = bpy.context.object
stand.name = "Fixture"
stand.location.z = 0.05
stand.scale = (0.14 * side + 0.12, 0.14 * side + 0.12, 0.1)
bpy.ops.mesh.primitive_cube_add(size=0.08)
first = bpy.context.object
studs = []
for row in range(side):
    for column in range(side):
        obj = first if not studs else first.copy()
        if studs:
            bpy.context.collection.objects.link(obj)
        obj.name = f"Stud {len(studs):03d}"
        obj.location = ((column - 12) * 0.14, (row - 12) * 0.14, 0.14)
        studs.append(obj)

studs[-1].location.x -= 0.08
floating = first.copy()
bpy.context.collection.objects.link(floating)
floating.name, floating.location = "Floating", (3.5, 0, 0.53)
bpy.context.view_layer.update()
clashing = [o.name for o in studs[-2:]]
assert run("check_intersections")["intersections"] == [{"objects": clashing, "kind": "crossing"}]
with tempfile.TemporaryDirectory() as tmp:
    path = str(Path(tmp) / "fixture.png")
    info = run("render_inspect", path=path)
    assert [(r["kind"], r.get("objects", r.get("object"))) for r in info["warnings"]] == [
        ("overlap", clashing),
        ("above_floor", floating.name),
    ], info["warnings"]
    initial = info["elapsed_ms"]
    closeup = run("render_inspect", target=first.name, path=path)
    assert not closeup["warnings"] and np.isclose(closeup["bbox"][0][2], 0.10)
    assert run("render_inspect", target=floating.name, path=path)["warnings"][0]["kind"] == "above_floor"

    # A selected fitting can be above the floor while another part of its assembly is grounded.
    joint = bpy.data.objects.new("Fixture assembly", None)
    bpy.context.collection.objects.link(joint)
    stand.parent = joint
    first.parent = joint
    assert not run("render_inspect", target=first.name, path=path)["warnings"]
    run("place_relative", obj=floating.name, relation="ALIGN_FLOOR")
    fixed = studs[-2].matrix_world.copy()
    result = run("resolve_overlaps", fixed=[stand.name, studs[-2].name])
    assert result["moved"] == [studs[-1].name] and studs[-2].matrix_world == fixed
    assert not run("check_intersections")["intersections"]
    assert not run("render_inspect", path=path)["warnings"]
print("DENSE LAYOUT PASS", bpy.app.version_string, "objects", len(studs) + 2, "initial preview ms", initial)
