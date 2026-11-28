# SPDX-License-Identifier: GPL-2.0-or-later
"""Set EXCAD_TEST_PHOTO to a local front-facing portrait and enable the MPFB extension."""

import json
import os
from pathlib import Path

import addon_utils
import bpy
import numpy as np
from mathutils import Vector

addon_utils.enable("bl_ext.blender_org.mpfb", default_set=True)
from excad_addon.ai_bridge import AiBridgeServer

run_bridge = AiBridgeServer.instance()._dispatch_dict
for obj in list(bpy.context.scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)


def run(cmd, **args):
    res = run_bridge(dict(cmd=cmd, **args))
    assert res["ok"], (cmd, res)
    return res


def wear(**params):
    return run("character_stylize", object="Portrait", **params)


def variant(style, garments, hair=None):
    folder = os.getenv("EXCAD_TEST_VARIANTS")
    if not folder:
        return
    extra = {"clothes": garments, "iris_color": [0.15, 0.075, 0.025], "lashes": True}
    if hair:
        extra["hair"] = hair
    wear(**extra)
    run("place_relative", obj="Portrait", relation="ALIGN_FLOOR")
    preview = run("render_inspect", target="Portrait", size=196, path=str(Path(folder) / (style + ".png")))
    print("VARIANT", style, preview["warnings"], flush=True)
    assert not preview["warnings"], (style, preview["warnings"])


result = run("human_generate", name="Portrait", rig="game_engine", subdiv_level=0, apply_skin=False)
assert result["engine"] == "MPFB2" and result["rig"], result
human = bpy.data.objects["Portrait"]
rig = human.find_armature()
photo = os.environ["EXCAD_TEST_PHOTO"]
fit = run("fit_face_from_image", object=human.name, image_path=photo)
assert len(fit["measurements"]) == 8 and all(0.15 <= v <= 0.5 for v in fit["weights"].values())
for name, value in fit["weights"].items():
    assert abs(human.data.shape_keys.key_blocks[name].value - value) < 1e-5
    if "/l-" in name:
        assert fit["weights"][name.replace("/l-", "/r-")] == value
print("PHOTO FIT", fit, flush=True)
for tone in ("FAIR_IVORY", "TAN", "DEEP", "WARM_BEIGE"):
    wear(skin=tone)
shader = human.data.materials[0].node_tree.nodes["Principled BSDF"]
assert shader.inputs["Subsurface Weight"].default_value > 0 and shader.inputs["Normal"].is_linked
assert shader.inputs["Roughness"].is_linked
lip_mask = [point.color[0] for point in human.data.color_attributes["EXCAD Lips"].data]
assert max(lip_mask) > 0.25
assert 0 < sum(weight > 0 for weight in lip_mask) < len(lip_mask) * 0.04
for name in ("SMILE", "SURPRISED", "WINK", "SERIOUS", "SMILE"):
    wear(expression=name)
    assert json.loads(human["excad_expression"])
for name in ("FIST", "POINT", "OPEN", "RELAXED"):
    assert wear(hands=name)["posed_fingers"] == 30
for name in ("system", "shirts", "pants", "skirts"):
    dest = Path(run("install_asset_pack", pack=name)["path"])
    assert dest.is_dir() and any(dest.rglob("*.mhclo"))
assert run("equip_asset", human="Portrait", name="eyebrow001")["name"]
assert Path(run("install_asset_pack", pack="all")["path"]).is_dir()

# Isolate outfits between trials, including MPFB's body occlusion masks.
# Reload the same fitted character so repeated poses do not mask other tests.
fixture = str(Path(bpy.app.tempdir) / "excad-wearable-test.blend")
bpy.ops.wm.save_as_mainfile(filepath=fixture, copy=True)
for style in ("SHORT", "PONYTAIL", "LONG_STRAIGHT", "LONG_WAVY", "CURTAIN_BANGS", "BOB"):
    bpy.ops.wm.open_mainfile(filepath=fixture)
    added = wear(hair=style)["added"]
    for name in added:
        obj = bpy.data.objects[name]
        assert obj.data.polygons and obj.vertex_groups.get("head") and obj.find_armature(), name
        shader = obj.data.materials[0].node_tree.nodes["Principled BSDF"]
        assert shader.inputs["Tangent"].is_linked and shader.inputs["Anisotropic"].default_value > 0
    print("HAIR", style, flush=True)
    variant(style, ["TSHIRT", "PANTS", "SHOES"])
for style in ("TSHIRT", "VNECK_SWEATER", "HOODIE", "JACKET", "PANTS", "SKIRT", "SHOES"):
    bpy.ops.wm.open_mainfile(filepath=fixture)
    obj = bpy.data.objects[wear(clothes=style)["added"][0]]
    assert obj.data.polygons and obj.find_armature() and any(v.groups for v in obj.data.vertices), style
    if style in {"VNECK_SWEATER", "JACKET", "HOODIE"}:
        distance = [v.color[0] for v in obj.data.color_attributes["EXCAD Hem distance"].data]
        assert min(distance) == 0 and max(distance) > 0.04
        assert obj.data.materials[0].node_tree.nodes["Fabric weave"].outputs["Normal"].is_linked
        assert all(p.material_index == 0 for p in obj.data.polygons)
    print("GARMENT", style, flush=True)
    complement = (
        ["TSHIRT", "SHOES"]
        if style in {"PANTS", "SKIRT"}
        else ["TSHIRT", "PANTS"]
        if style == "SHOES"
        else ["PANTS", "SHOES"]
    )
    variant(style, complement, "BOB")
bpy.ops.wm.open_mainfile(filepath=fixture)
wear(hair="PONYTAIL", clothes=["TSHIRT", "PANTS", "SHOES"], iris_color=[0.15, 0.075, 0.025], lashes=True, teeth=True)
human = bpy.data.objects["Portrait"]
run("place_relative", obj=human.name, relation="ALIGN_FLOOR")
# Restyling replaces hair, preserves wardrobe and keeps its deformation rig.
wear(hair="SHORT")
assert len([o for o in bpy.context.scene.objects if o.get("excad_slot") == "hair"]) == 1
wear(hair="PONYTAIL")
assert bpy.data.objects.get("Portrait.toigo_wool_pants")
out = Path(os.getenv("EXCAD_TEST_OUTPUT", bpy.app.tempdir))
out.mkdir(parents=True, exist_ok=True)
inspect = run("render_inspect", target="Portrait", size=320, path=str(out / "phase2-character.png"))
print("CHARACTER INSPECTION", inspect, flush=True)
assert Path(inspect["path"]).stat().st_size > 1000
assert not inspect["warnings"], inspect["warnings"]
assert {c["kind"] for c in inspect["contacts"]} == {"attachment", "garment_seam"}
assert not run("check_intersections")["intersections"]
assert not run("resolve_overlaps")["moved"]
if os.getenv("EXCAD_TEST_VARIANTS"):
    scene_file = str(Path(os.environ["EXCAD_TEST_VARIANTS"]) / "dressed.blend")
    bpy.ops.wm.save_as_mainfile(filepath=scene_file, copy=True)
before = run("scene_digest")["h"]
skin_file = human.data.materials[0].node_tree.nodes["Skin albedo"].image.filepath
run("checkpoint", name="Dressed portrait")
wear(iris_size=0.58, expression="SURPRISED", hands="FIST")
assert run("scene_digest", since_hash=before).get("changed")
run("rollback", name="Dressed portrait")
restored = run("scene_digest", since_hash=before)
assert restored.get("unchanged"), restored
human = bpy.context.scene.objects["Portrait"]
assert human.find_armature() and human.data.shape_keys
assert human.data.materials[0].node_tree.nodes["Skin albedo"].image.filepath == skin_file
from excad_addon import scene_mesh

hair = next(o.name for o in bpy.context.scene.objects if o.get("excad_slot") == "hair")
rest = scene_mesh.surfaces(hair)[0].points.copy()
assert run("oneshot_pose", object="Portrait", pose="HEROIC")["posed_bones"] >= 5
posed = scene_mesh.surfaces(hair)[0].points
assert np.max(np.linalg.norm(posed - rest, axis=1)) > 0.005
assert wear(hands="POINT")["posed_fingers"] == 30
current = run("scene_digest")["h"]
assert not run_bridge({"cmd": "oneshot_pose", "object": "Portrait", "pose": "UNKNOWN"})["ok"]
assert run("scene_digest", since_hash=current).get("unchanged")
if os.getenv("EXCAD_TEST_VARIANTS"):
    rendered = run(
        "render_inspect", target="Portrait", size=260, path=str(Path(os.environ["EXCAD_TEST_VARIANTS"]) / "posed.png")
    )
    print("POSED", rendered["warnings"], flush=True)
    assert not rendered["warnings"], rendered["warnings"]
assert run("oneshot_pose", object="Portrait", pose="RESET")["posed_bones"] == len(human.find_armature().pose.bones)
assert np.allclose(scene_mesh.surfaces(hair)[0].points, rest, atol=1e-5)
wear(iris_size=0.62)
iris = bpy.data.materials[human.name + " Iris"]
assert np.allclose(iris.diffuse_color[:3], [0.15, 0.075, 0.025])
from excad_addon.face_fit import apply_measurements
from excad_addon.wardrobe import _eye_centers, rest_mesh

eyes = [o for o in bpy.context.scene.objects if "excad_eye_side" in o]
assert len(eyes) == 2
for eye in eyes:
    uv = eye.data.uv_layers.active
    assert uv and uv.name == "Eye atlas"
    assert {p.material_index for p in eye.data.polygons} == {0, 1, 2}
    center = np.asarray(eye["excad_eye_center"])
    front = min(v.co.y for v in eye.data.vertices)
    assert np.isclose(center[1] - front, 1.04 * eye["excad_eye_radius"], rtol=1e-4)
    for mat in list(eye.data.materials)[:2]:
        assert mat.node_tree.nodes["Eye atlas"].image.size[:] == (1024, 1024)
        assert mat.node_tree.nodes["Principled BSDF"].inputs["Coat Weight"].default_value > 0.9
    first_uv = np.asarray(uv.uv[0].vector)
    rim_uv = np.asarray(uv.uv[1].vector)
    assert np.isclose(np.linalg.norm(first_uv - rim_uv) * 1024, 32, atol=0.01)
old_centers = np.array([o["excad_eye_center"][:] for o in eyes])
apply_measurements(human, dict(fit["measurements"], eye_spacing=0.30))
with rest_mesh(human) as mesh:
    centers = _eye_centers(human, mesh)
for eye in eyes:
    assert np.allclose(eye["excad_eye_center"][:], centers[eye["excad_eye_side"]], atol=1e-6)
assert np.max(np.abs(old_centers - np.array([o["excad_eye_center"][:] for o in eyes]))) > 1e-4

# Alternating two fits must restore the evaluated eyes as precisely as the face.
fitted_hash = run("scene_digest")["h"]
apply_measurements(human, dict(fit["measurements"], eye_spacing=0.54))
apply_measurements(human, dict(fit["measurements"], eye_spacing=0.30))
assert run("scene_digest", since_hash=fitted_hash).get("unchanged")

# The body, CC0 assets and tailored clothes do not all have the same direct
# parent. A turn must preserve the posed geometry of the complete person.
rig = human.find_armature()
rig_before = rig.matrix_world.copy()
body_local = rig_before.inverted() @ human.matrix_world
aim_target = bpy.data.objects.new("Look at", None)
bpy.context.collection.objects.link(aim_target)
aim_target.location = human.matrix_world.translation + Vector((-2.0, -1.1, 0.16))
original = {s.obj.name: s.points[::37].copy() for s in scene_mesh.surfaces(human.name)}
run("look_at", obj=human.name, target=aim_target.name)
assert np.allclose(rig.matrix_world.inverted() @ human.matrix_world, body_local, atol=1e-6)
aim = (aim_target.matrix_world.translation - human.matrix_world.translation).normalized()
assert (human.matrix_world.to_quaternion() @ Vector((0, -1, 0))).dot(aim) > 0.9999
turn = np.asarray(rig.matrix_world @ rig_before.inverted())
for surface in scene_mesh.surfaces(human.name):
    expected = original[surface.obj.name] @ turn[:3, :3].T + turn[:3, 3]
    assert np.allclose(surface.points[::37], expected, atol=1e-5), surface.obj.name
rig.matrix_world = rig_before
bpy.data.objects.remove(aim_target, do_unlink=True)
bpy.context.view_layer.update()

wear(clothes=[], lashes=False)
assert all(o.get("excad_slot") not in {"top", "bottom", "footwear", "lashes"} for o in bpy.context.scene.objects)
assert any(o.get("excad_slot") == "hair" for o in bpy.context.scene.objects)
assert len(AiBridgeServer.instance()._dispatch('{"cmd":"schema"}').encode()) <= 1200
print("CHARACTER PASS", bpy.app.version_string)
