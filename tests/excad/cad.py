# SPDX-License-Identifier: GPL-2.0-or-later
import json
import math
import os
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

from excad_addon.ai_bridge import AiBridgeServer
from excad_addon.sketch_engine import SketchDocument
from excad_addon.sketch_overlay import constrained_variables
from excad_addon import feature_geometry

bridge = AiBridgeServer.instance()
for obj in list(bpy.context.scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)


def run(cmd, **params):
    result = bridge._dispatch_dict(dict(cmd=cmd, **params))
    assert result["ok"], (cmd, result)
    return result


def volume(name):
    obj = bpy.data.objects[name]
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    assert bm.faces and bm.verts and all(e.is_manifold for e in bm.edges) and all(v.is_manifold for v in bm.verts), name
    result = abs(bm.calc_volume(signed=True))
    bm.free()
    return result


def sketch(name="profile", **args):
    run("sketch_new", sketch_id=name, **args)


def tree(name, sketches, features):
    return run("feature_tree", name=name, recipe={"version": 1, "sketches": sketches, "features": features})


sketch()
run("sketch_arc", points=[[1, 0], [0, 1], [-1, 0]])
run("sketch_add_line", u0=-1, v0=0, u1=1, v1=0)
run("extrude", name="Half", depth=1)
assert abs(volume("Half") - 1.5708) < 0.005
sketch()
run("sketch_arc", center=[0, 0], radius=1, start_deg=0, angle_deg=180)
run("sketch_add_line", u0=-1, v0=0, u1=1, v1=0)
run("extrude", name="OtherHalf", depth=1, symmetric=True)
assert abs(volume("Half") - volume("OtherHalf")) < 1e-5
sketch()
run("sketch_arc", center=[0, 0], radius=1, start_deg=0, angle_deg=180)
run("sketch_mirror2d", axis="X", source_entity=0)
run("extrude", name="Two arcs", depth=1)
assert abs(volume("Two arcs") - 3.14159) < 0.009
sketch()
run("sketch_slot", center=[0, 0], angle_deg=45, length=2, width=0.4)
run("extrude", name="Centered slot", depth=1)
assert abs(volume("Centered slot") - (1.6 * 0.4 + 3.14159 * 0.2**2)) < 0.001
sketch()
run("sketch_regular_polygon", sides=6, radius=0.5, center=[1, 0])
run("sketch_mirror2d", axis="Y")
run("extrude", name="Mirrored", depth=0.3)
assert volume("Mirrored") > 0.35
sketch()
run("sketch_add_rect", u0=-1, v0=-1, width=2, height=2)
run("sketch_offset_loop", distance=-0.2)
run("extrude", name="Ring", depth=1)
assert abs(volume("Ring") - (4 - 1.6**2)) < 1e-5
sketch()
run("sketch_spline", points=[[0, 0], [0.5, -0.1], [0.8, 0.3], [0.1, 0.6]], closed=True)
run("extrude", name="Spline", depth=0.2, draft_angle=2)
assert volume("Spline") > 0.01

# Independent residual checks start from an incorrect solution.
for kind in ("MIDPOINT", "SYMMETRIC", "ANGLE", "TANGENT", "CONCENTRIC"):
    d = SketchDocument()
    if kind in {"TANGENT", "CONCENTRIC"}:
        if kind == "TANGENT":
            d.add_line(-2, 0, 2, 0)
            d.points[0].fixed = d.points[1].fixed = True
            b = d.add_circle(0, 0.7, 1)
        else:
            d.add_circle(0, 0, 1)
            d.points[0].fixed = True
            b = d.add_circle(0.3, 0.1, 0.5)
        d.add_constraint(kind, 0, b)
    else:
        for i, p in enumerate(((0.2, 0.3), (-1, 0), (1, 0), (1, 1))):
            d.add_point(*p, fixed=i > 0)
        d.add_constraint(kind, 0, 1, 2, 3, value=45)
    solved, _, residual = d.solve()
    assert solved and residual < 1e-6, (kind, residual)

# A translated three-point arc must retain enough precision for the numerical
# Jacobian. Check control-point radii and the tangent line in sketch units.
cx, cy = 130.31857, -187.12389
d = SketchDocument()
anchor = d.add_circle(cx, cy, 0.6)
d.points[0].fixed = True
arc = d.add_shape("ARC", points=[(cx + 1.13, cy), (cx + 0.1, cy + 0.96), (cx - 0.89, cy)])
d.add_constraint("CONCENTRIC", anchor, arc)
assert d.solve()[0]
radii = [math.hypot(p.u - cx, p.v - cy) for p in d.points[1:]]
assert max(radii) - min(radii) < 2e-7, radii

d = SketchDocument()
radius = 0.77
arc = d.add_shape("ARC", points=[(cx + radius, cy), (cx, cy + radius), (cx - radius, cy)])
for p in d.points:
    p.fixed = True
line = d.add_line(cx - 1, cy + 0.98, cx + 1, cy + 0.98)
d.add_constraint("HORIZONTAL", d.entities[line].p0, d.entities[line].p1)
d.add_constraint("TANGENT", line, arc)
assert d.solve()[0]
assert all(abs(p.v - cy - radius) < 2e-7 for p in d.points[-2:])

sketch()
run("sketch_add_circle", radius=0.5)
run("sketch_constraint", ctype="FIXED", a=0)
assert constrained_variables(SketchDocument.active()) == {0, 1, 2}

rect = {"shapes": [{"type": "RECT", "width": 3, "height": 1.8}]}
tree("Housing", {"base": rect}, [{"id": "base", "type": "EXTRUDE", "sketch": "base", "depth": 0.65}])
for side in ("TOP", "BOTTOM", "FRONT", "BACK", "LEFT", "RIGHT", "LARGEST_PLANAR"):
    assert run("sketch_on_face", object="Housing", face=side, sketch_id="opening")["plane"] == "FACE"
run("sketch_on_face", object="Housing", face="TOP", sketch_id="opening")
run("sketch_slot", start=[-0.65, 0], end=[0.65, 0], width=0.48)
run("extrude", object="Housing", op="CUT", depth=-0.8)
v = volume("Housing")
assert abs(v - (3 * 1.8 - 1.3 * 0.48 - 3.14159 * 0.24**2) * 0.65) < 0.001
run("fillet", object="Housing", edges="TOP", radius=0.065)
run("chamfer", object="Housing", edges="BOTTOM", width=0.035)
assert volume("Housing") < v
bpy.context.view_layer.update()
assert not bpy.data.objects["Housing"].ray_cast((0, 0, 2), (0, 0, -1))[0], "Slot roof obstructs through-cut"
out = Path(os.getenv("EXCAD_TEST_OUTPUT", bpy.app.tempdir))
out.mkdir(exist_ok=True, parents=True)
finish = run("set_material", object="Housing", preset="BRUSHED_METAL", color=[0.35, 0.48, 0.58])["material"]
run("light_studio", target="Housing", style="PRODUCT", power=1.15)
probe = run("render_inspect", target="Housing", size=256, path=str(out / "phase1-cad.png"))
assert not probe["warnings"], probe
print("CAD INSPECTION", probe)
recipe_path = Path(bpy.app.tempdir) / "feature_tree.json"
run("save_feature_tree", object="Housing", path=str(recipe_path))
assert any(
    s.get("support", {}).get("object") == "SELF"
    for s in json.loads(recipe_path.read_text())["sketches"].values()
    if s.get("support")
)
before = volume("Housing")
run("rebuild_feature_tree", object="Independent housing", path=str(recipe_path))
assert abs(volume("Independent housing") - before) < 1e-5
run("rebuild_feature_tree", object="Housing", path=str(recipe_path), parameters={"base.depth": 0.72})
assert volume("Housing") > before
rebuilt = bpy.data.objects["Housing"]
assert rebuilt.active_material == bpy.data.materials[finish], (list(rebuilt.data.materials), rebuilt.active_material_index)
fresh = run(
    "feature_tree",
    name="Dimensioned",
    recipe={
        "sketches": {"plate": rect},
        "features": [{"id": "height", "type": "EXTRUDE", "sketch": "plate", "depth": 1}],
    },
)
dimensioned = bpy.data.objects[fresh["name"]]
side_paint = bpy.data.materials.new("Body paint")
cap_paint = bpy.data.materials.new("Cap paint")
override = bpy.data.materials.new("Instance cap paint")
dimensioned.data.materials.append(side_paint)
dimensioned.data.materials.append(cap_paint)
for polygon in dimensioned.data.polygons:
    polygon.material_index = int(polygon.normal.z > 0.99)
dimensioned.material_slots[1].link = "OBJECT"
dimensioned.material_slots[1].material = override
dimensioned.active_material_index = 1
linked = dimensioned.copy()
bpy.context.collection.objects.link(linked)
original_data = linked.data
run(
    "rebuild_feature_tree", object="Dimensioned", parameters={"sketches.plate.shapes.0.width": 2.0, "height.depth": 0.5}
)
assert abs(volume("Dimensioned") - 1.8) < 1e-5
run("fillet", object=dimensioned.name, edges="TOP", radius=0.04)
assert linked.data == original_data and linked.data != dimensioned.data
assert list(dimensioned.data.materials) == [side_paint, cap_paint] and dimensioned.active_material == override
assert all(p.material_index == 1 for p in dimensioned.data.polygons if p.normal.z > 0.999)
assert all(p.material_index == 0 for p in dimensioned.data.polygons if p.normal.z < 0.01)
# On a 90-degree outside edge the five-segment fillet lies on a quarter circle.
for degrees in (18, 36, 54, 72):
    theta = math.radians(degrees)
    x = 1 - 0.04 + 0.04 * math.cos(theta)
    hit, position, _, _ = dimensioned.ray_cast((x, 0, 1), (0, 0, -1))
    assert hit and abs(position.z - (0.46 + 0.04 * math.sin(theta))) < 0.0005
bad = bridge._dispatch_dict({"cmd": "rebuild_feature_tree", "object": "Housing", "parameters": {"base.depth": 0}})
assert not bad["ok"] and volume("Housing") > before
sketch()
run("sketch_add_rect", u0=0, v0=-1, width=2, height=2)
run("extrude", object="Housing", depth=2, op="INTERSECT")
assert 0 < volume("Housing") < before

tree(
    "Case",
    {"base": rect},
    [{"type": "EXTRUDE", "sketch": "base", "depth": 1}, {"type": "SHELL", "thickness": 0.1, "open_faces": ["TOP"]}],
)
assert 0 < volume("Case") < 2
sketch("bottom")
run("sketch_add_circle", radius=0.4)
sketch("top", offset=1)
run("sketch_regular_polygon", sides=6, radius=0.65)
run("loft", name="Vase", sketch_ids=["bottom", "top"])
assert volume("Vase") > 0.25
sections = {
    f"s{i}": {"offset": i * 0.4, "shapes": [{"type": "RECT", "width": w, "height": w * 0.7}]}
    for i, w in enumerate((0.4, 1.0, 0.55))
}
tree("Smooth transitions", sections, [{"type": "LOFT", "sketch_ids": list(sections), "steps": 5}])
assert len({round(v.co.z, 4) for v in bpy.data.objects["Smooth transitions"].data.vertices}) == 11
assert volume("Smooth transitions") > 0.1
# The two bores trade diameters. The upper section is drawn in reverse order
# on a downward-facing plane, as when selecting the opposite face of a fixture.
def drilled(radii):
    return [
        {"type": "RECT", "width": 3, "height": 1.6},
        *[{"type": "CIRCLE", "center": [x, 0], "radius": r} for x, r in zip((-0.78, 0.78), radii)],
    ]


lower_radii, upper_radii = (0.20, 0.36), (0.35, 0.17)
downward = Matrix.Translation((0, 0, 1)) @ Matrix.Rotation(math.pi, 4, "X")
drilled_sections = {
    "base": {"shapes": drilled(lower_radii)},
    "cap": {"frame": [row[:] for row in downward], "shapes": list(reversed(drilled(upper_radii)))},
}
tree("Tapered bores", drilled_sections, [{"type": "LOFT", "sketch_ids": ["base", "cap"]}])
part = bpy.data.objects["Tapered bores"]
assert [part.ray_cast((x, 0, -0.2), (0, 0, 1))[0] for x in (-0.78, 0, 0.78)] == [False, True, False]
expected = 4.8 - sum(math.pi * (a * a + a * b + b * b) / 3 for a, b in zip(lower_radii, upper_radii))
assert math.isclose(volume(part.name), expected, rel_tol=0.001)
for x, a, b in zip((-0.78, 0.78), lower_radii, upper_radii):
    hit, point, _, _ = part.ray_cast((x, 0, 0.5), (0, 1, 0))
    assert hit and abs(point.y - (a + b) / 2) < 0.001, point
disconnected = drilled(upper_radii)
disconnected[-1]["center"] = [2.5, 0]
rejected = bridge._dispatch_dict({
    "cmd": "feature_tree", "name": part.name,
    "recipe": {"sketches": {"a": drilled_sections["base"], "b": {"offset": 1, "shapes": disconnected}},
               "features": [{"type": "LOFT", "sketch_ids": ["a", "b"]}]},
})
assert not rejected["ok"] and "Loft sections" in rejected["error"] and math.isclose(volume(part.name), expected, rel_tol=0.001)
sketch("profile", plane="XZ")
run("sketch_add_rect", u0=0, v0=0, width=0.7, height=0.35)
run("revolve", name="Dial", axis="Z", angle_deg=270, segments=40)
assert volume("Dial") > 0.35
sketch()
run("sketch_add_circle", radius=0.1)
curve = bpy.data.curves.new("HandlePath", "CURVE")
curve.dimensions = "3D"
sp = curve.splines.new("POLY")
sp.points.add(3)
for p, co in zip(sp.points, [(0, 0, 0), (0, 0, 0.5), (1, 0, 0.5), (1, 0, 0)]):
    p.co = (*co, 1)
path_obj = bpy.data.objects.new("HandlePath", curve)
bpy.context.collection.objects.link(path_obj)
run("sweep", name="Handle", profile_sketch="profile", path_curve=path_obj.name)
assert volume("Handle") > 0.04
path_obj.shape_key_add(name="Basis")
taller = path_obj.shape_key_add(name="Taller path")
for index in (1, 2):
    taller.data[index].co.z += 0.24
taller.value = 1
path_obj.location.x = 0.3
for bevel in (0, 0.035):
    curve.bevel_depth = bevel
    counts = len(bpy.data.curves), len(bpy.data.shape_keys), len(bpy.data.objects)
    evaluated = feature_geometry.path_points(path_obj.name)
    assert len(evaluated) == 4 and abs(max(p.z for p in evaluated) - 0.74) < 1e-6
    assert abs(evaluated[0].x - 0.3) < 1e-6 and abs(evaluated[-1].x - 1.3) < 1e-6
    assert counts == (len(bpy.data.curves), len(bpy.data.shape_keys), len(bpy.data.objects))
run("sweep", name="Taller handle", profile_sketch="profile", path_curve=path_obj.name)
assert volume("Taller handle") > 1.15 * volume("Handle")
tree(
    "Guide rail", {"section": {"shapes": [{"type": "CIRCLE", "radius": 0.1}]}},
    [{"id": "rail", "type": "SWEEP", "sketch": "section", "path_curve": path_obj.name}],
)
rail_volume = volume("Guide rail")
assert math.isclose(rail_volume, volume("Taller handle"), rel_tol=1e-5)
taller.value = 0
portable = Path(bpy.app.tempdir) / "evaluated-rail.json"
run("save_feature_tree", object="Guide rail", path=str(portable))
exported = json.loads(portable.read_text())["features"][0]
assert isinstance(exported["path_curve"], list) and "path_reference" not in exported
assert abs(max(p[2] for p in exported["path_curve"]) - 0.74) < 1e-6
run("rebuild_feature_tree", object="Guide rail")
short_volume = volume("Guide rail")
assert rail_volume > short_volume * 1.15
bpy.data.objects.remove(path_obj, do_unlink=True)
run("rebuild_feature_tree", name="Portable rail", path=str(portable))
assert math.isclose(volume("Portable rail"), rail_volume, rel_tol=1e-5)
run("rebuild_feature_tree", object="Guide rail", parameters={"sketches.section.shapes.0.radius": 0.12})
assert math.isclose(volume("Guide rail"), short_volume * 1.2**2, rel_tol=1e-4)
rejected = bridge._dispatch_dict({"cmd": "rebuild_feature_tree", "object": "Guide rail",
                                  "parameters": {"rail.path_curve": "Different missing curve"}})
assert not rejected["ok"] and math.isclose(volume("Guide rail"), short_volume * 1.2**2, rel_tol=1e-4)
path = [[1, 0, 0], [0, 1, 0.45], [-1, 0, 0], [0, -1, -0.15], [1, 0, 0]]
tree(
    "Closed tube",
    {"section": {"shapes": [{"type": "CIRCLE", "radius": 0.07}]}},
    [{"type": "SWEEP", "sketch": "section", "path_curve": path}],
)
assert volume("Closed tube") > 0.04
# Capture which face is currently facing up on a rotated body. The cut keeps
# that material face after a subsequent motion and regenerates at its new center.
tree(
    "Mounted",
    {"base": {"shapes": [{"type": "RECT", "width": 2, "height": 1}]}},
    [{"id": "pad", "type": "EXTRUDE", "sketch": "base", "depth": 0.6}],
)
mounted = bpy.data.objects["Mounted"]
mounted.matrix_world = Matrix.Translation((3, 1, 2)) @ Matrix.Rotation(math.pi / 2, 4, "Y")
run("sketch_on_face", object="Mounted", face="TOP", sketch_id="fixture hole")
run("sketch_add_circle", radius=0.12)
run("extrude", object="Mounted", op="CUT", depth=-2.2)
assert not mounted.ray_cast((-2, 0, 0.3), (1, 0, 0))[0]
mounted.matrix_world = Matrix.Rotation(0.4, 4, "Z")
run("rebuild_feature_tree", object="Mounted", parameters={"pad.depth": 0.8})
assert not mounted.ray_cast((-2, 0, 0.4), (1, 0, 0))[0]
assert mounted.ray_cast((-2, 0, 0.65), (1, 0, 0))[0]
# A guide drawn on a placed body lives in the current world frame. Later
# placement of the finished part must carry the cut with it.
drill = bpy.data.curves.new("Drill axis", "CURVE")
drill.dimensions = "3D"
axis = drill.splines.new("POLY")
axis.points.add(1)
for p, z in zip(axis.points, (-0.1, 1.1)):
    p.co = (0.62, 0.28, z, 1)
guide = bpy.data.objects.new(drill.name, drill)
bpy.context.collection.objects.link(guide)
guide.matrix_world = mounted.matrix_world
sketch("drill bit")
run("sketch_add_circle", radius=0.05)
run("sweep", object="Mounted", profile_sketch="drill bit", path_curve=guide.name, op="CUT")
assert not mounted.ray_cast((0.62, 0.28, 1), (0, 0, -1))[0]
mounted.matrix_world = Matrix.Translation((3, 2, -1)) @ Matrix.Rotation(-0.8, 4, "Y")
run("rebuild_feature_tree", object="Mounted", parameters={"pad.depth": 0.75})
assert not mounted.ray_cast((0.62, 0.28, 1), (0, 0, -1))[0]
sketch()
run("sketch_add_circle", cu=1, radius=0.10)
run("extrude", name="Bolts", depth=0.2)
unit = volume("Bolts")
run("pattern", object="Bolts", type="CIRCULAR", axis="Z", count=4)
run("pattern", object="Bolts", type="LINEAR", axis="X", spacing=3, count=2)
assert abs(volume("Bolts") / unit - 8) < 0.01
origin = Matrix(feature_geometry.selected_face(bpy.data.objects["Bolts"], "TOP")).translation
assert any(
    (origin - Vector((x + dx, y, 0.2))).length < 1e-4 for dx in (0, 3) for x, y in ((1, 0), (0, 1), (-1, 0), (0, -1))
), origin
run("mirror3d", object="Bolts", axis="Y", offset=3)
assert abs(volume("Bolts") / unit - 16) < 0.02
assert not any(o.name.startswith("EXCAD_Construction") for o in bpy.data.objects)
assert len(bridge._dispatch('{"cmd":"schema"}').encode()) <= 1200
print("CAD PASS", bpy.app.version_string)
