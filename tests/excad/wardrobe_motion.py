# SPDX-License-Identifier: GPL-2.0-or-later
"""Check assembled clothes and head-weighted hair across MPFB rigs and body proportions."""

import os
from pathlib import Path
import random
import sys
import tempfile

import addon_utils
import bpy
import numpy as np
from mathutils import Vector

from excad_addon.ai_bridge import AiBridgeServer
from excad_addon.character_stylizer import _vgroup_center
from excad_addon.oneshot_generators import _BONE_ALIASES
from excad_addon.scene_mesh import read_surface
from excad_addon.wardrobe import rest_mesh

addon_utils.enable("bl_ext.blender_org.mpfb", default_set=True)
bridge = AiBridgeServer.instance()
cases = [
    ("REALISTIC_MALE", "default", "SHORT", ["TSHIRT", "JACKET", "PANTS", "SHOES"], "TAN", "SMILE", {}),
    ("HEROIC_SCULPT", "game_engine", "PONYTAIL", ["HOODIE", "PANTS", "SHOES"], "DEEP", "SERIOUS", {}),
    ("REALISTIC_FEMALE", "default", "LONG_WAVY", ["VNECK_SWEATER", "SKIRT", "SHOES"], "WARM_BEIGE", "WINK", {}),
    (
        "CUSTOM",
        "game_engine",
        "CURTAIN_BANGS",
        ["TSHIRT", "JACKET", "PANTS", "SHOES"],
        "FAIR_IVORY",
        "SURPRISED",
        {"gender": 0.6, "weight": 0.78, "muscle": 0.3, "height": 0.7, "age": 0.62},
    ),
    ("ATHLETIC_MALE", "default", "BOB", ["VNECK_SWEATER", "PANTS", "SHOES"], "DEEP", "SERIOUS", {}),
    ("ATHLETIC_FEMALE", "game_engine", "LONG_STRAIGHT", ["HOODIE", "SKIRT", "SHOES"], "FAIR_IVORY", "SMILE", {}),
]


def seeded_case(seed):
    rng = random.Random(seed)
    top = rng.choice(
        [
            ["TSHIRT"],
            ["VNECK_SWEATER"],
            ["HOODIE"],
            ["TSHIRT", "JACKET"],
            ["VNECK_SWEATER", "JACKET"],
        ]
    )
    clothes = [*top, rng.choice(["SKIRT", "PANTS"]), "SHOES"]
    hair = rng.choice(["SHORT", "BOB", "PONYTAIL", "LONG_STRAIGHT", "LONG_WAVY", "CURTAIN_BANGS"])
    rig = rng.choice(["default", "game_engine"])
    body = {
        key: rng.uniform(low, high)
        for key, low, high in (
            ("gender", 0.05, 0.95),
            ("age", 0.30, 0.72),
            ("height", 0.22, 0.85),
            ("muscle", 0.16, 0.90),
            ("weight", 0.18, 0.88),
            ("proportions", 0.30, 0.78),
        )
    }
    skin = rng.choice(["FAIR_IVORY", "TAN", "DEEP", "WARM_BEIGE"])
    return (
        "CUSTOM",
        rig,
        hair,
        clothes,
        skin,
        rng.choice(["SMILE", "SERIOUS", "WINK", "SURPRISED"]),
        body,
    )


if "--" in sys.argv:
    cases = [seeded_case(int(seed)) for seed in sys.argv[sys.argv.index("--") + 1 :]]


def run(cmd, **args):
    response = bridge._dispatch_dict(dict(cmd=cmd, **args))
    assert response["ok"], (cmd, response)
    return response


def check_open_lapels(shirt, jacket):
    """An opening exposing the whole shirt must still expose it after the underlay mask."""
    graph = bpy.context.evaluated_depsgraph_get()
    mask = next(m for m in shirt.modifiers if m.type == "MASK" and m.vertex_group == shirt["excad_underlay_group"])
    visible, coat = read_surface(shirt, graph), read_surface(jacket, graph)
    mask.show_viewport = False
    try:
        bpy.context.view_layer.update()
        whole = read_surface(shirt, graph)
    finally:
        mask.show_viewport = True
        bpy.context.view_layer.update()
    low, high = whole.bounds
    center, span = Vector((low + high) / 2), high[2] - low[2]
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    for slope in (-0.4, 0, 0.4):
        eye = Vector((slope, -1, 0)).normalized()
        right = Vector((-eye.y, eye.x, 0))
        along = (corners - np.asarray(center)) @ right
        missing, exposed = 0, 0
        for x in np.linspace(along.min(), along.max(), 58):
            for z in np.linspace(low[2], high[2], 64):
                start = center + right * x + eye * (2 * span)
                start.z = z
                distance = whole.bvh.ray_cast(start, -eye)[3]
                if distance is None:
                    continue
                outside = coat.bvh.ray_cast(start, -eye)[3]
                if outside is not None and outside < distance + 0.12 * span:
                    continue
                exposed += 1
                actual = visible.bvh.ray_cast(start, -eye)[3]
                missing += actual is None or actual > distance + 0.01 * span
        assert exposed > 10 and missing <= max(1, exposed * 0.005), (slope, missing, exposed)


def check_exposed_skin(person, shirt):
    body = person.vertex_groups["body"].index
    coverage = person.vertex_groups[shirt["excad_coverage_group"]].index
    ids = np.array([v.index for v in person.data.vertices if any(g.group == body for g in v.groups)])
    with rest_mesh(person) as mesh:
        points = np.asarray([mesh.vertices[int(i)].co for i in ids])
        span = np.ptp(points[:, 2])
        neck = _vgroup_center(person, mesh, ("joint-neck",))
        regions = [ids[points[:, 2] > neck.z + 0.075 * span]]
        for side in ("l", "r"):
            wrist = _vgroup_center(person, mesh, (f"joint-{side}-hand",))
            elbow = _vgroup_center(person, mesh, (f"joint-{side}-elbow",))
            offset = points - wrist
            along = offset @ (wrist - elbow).normalized()
            regions.append(ids[(along > 0.045 * span) & (np.linalg.norm(offset, axis=1) < 0.14 * span)])
    for exposed in regions:
        assert len(exposed) > 10
        assert all(
            not any(g.group == coverage for g in person.data.vertices[int(i)].groups) for i in exposed
        ), "Clothing covered the face or fingers"


def check_neckline(person, shirt):
    """Compare the visible collar opening with the same skin before the shirt mask."""
    graph = bpy.context.evaluated_depsgraph_get()
    skin, fabric = read_surface(person, graph), read_surface(shirt, graph)
    mask = next(m for m in person.modifiers if m.type == "MASK" and m.vertex_group == shirt["excad_coverage_group"])
    mask.show_viewport = False
    try:
        bpy.context.view_layer.update()
        full = read_surface(person, graph)
    finally:
        mask.show_viewport = True
        bpy.context.view_layer.update()
    with rest_mesh(person) as mesh:
        neck = person.matrix_world @ _vgroup_center(person, mesh, ("joint-neck",))
    height = float(full.bounds[1, 2] - full.bounds[0, 2])
    for angle in (-0.32, 0.32):
        eye = Vector((angle, -1, 0.23)).normalized()
        right = eye.cross((0, 0, 1)).normalized()
        up = right.cross(eye).normalized()
        exposed = missing = 0
        for x in np.linspace(-0.074, 0.074, 86):
            for z in np.linspace(-0.094, 0.025, 72):
                start = neck + height * (eye + right * x + up * z)
                depth = full.bvh.ray_cast(start, -eye)[3]
                if depth is None:
                    continue
                cloth = fabric.bvh.ray_cast(start, -eye)[3]
                if cloth is not None and cloth <= depth + 1e-4:
                    continue
                exposed += 1
                visible = skin.bvh.ray_cast(start, -eye)[3]
                closest = min(v for v in (visible, cloth, height * 3) if v is not None)
                missing += closest > depth + 0.001
        assert exposed > 200 and missing <= 1, (angle, exposed, missing)


with tempfile.TemporaryDirectory() as tmp:
    image_dir = Path(os.getenv("EXCAD_TEST_VARIANTS", tmp))
    image_dir.mkdir(parents=True, exist_ok=True)
    for obj in list(bpy.context.scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    fixture = str(Path(tmp) / "empty.blend")
    bpy.ops.wm.save_as_mainfile(filepath=fixture, copy=True)
    measured = []
    for preset, rig, hair, clothes, skin, expression, options in cases:
        bpy.ops.wm.open_mainfile(filepath=fixture)
        run(
            "batch",
            steps=[
                dict(
                    cmd="human_generate",
                    name="Person",
                    preset=preset,
                    rig=rig,
                    apply_skin=False,
                    subdiv_level=1,
                    **options,
                ),
                dict(cmd="fit_face_from_image", object="Person", image_path=os.environ["EXCAD_TEST_PHOTO"]),
                dict(
                    cmd="character_stylize",
                    object="Person",
                    hair=hair,
                    clothes=clothes,
                    skin=skin,
                    expression=expression,
                    iris_color=[0.18, 0.08, 0.035],
                    teeth=True,
                    lashes=True,
                    hands="RELAXED",
                ),
            ],
        )
        obj = bpy.context.scene.objects["Person"]
        bones = obj.find_armature().pose.bones
        parts = {o.get("excad_slot"): o for o in bpy.context.scene.objects if o.get("excad_human") == obj}
        check_exposed_skin(obj, parts["top"])
        if "VNECK_SWEATER" in clothes and "JACKET" not in clothes:
            check_neckline(obj, parts["top"])
        if "JACKET" in clothes:
            check_open_lapels(parts["top"], parts["outerwear"])

        def bone(role):
            return next(bones[name] for name in _BONE_ALIASES[role] if name in bones)

        for pose in ("RESET", "RELAXED", "HEROIC", "WALK", "WAVE"):
            assert run("oneshot_pose", object=obj.name, pose=pose)["posed_bones"] > 0
            run("place_relative", obj=obj.name, relation="ALIGN_FLOOR")
            preview = run("render_inspect", target=obj.name, path=str(image_dir / f"{preset}-{pose}.png"))
            assert not preview["warnings"], (preset, pose, preview["warnings"])
            measured.append(preview["elapsed_ms"])
        # The waving wrist belongs above and outside the right shoulder in both rigs.
        shoulder = bone("upperarm_r").head
        outward = (shoulder - bone("upperarm_l").head).normalized()
        up = bone("head").head - (shoulder + bone("upperarm_l").head) / 2
        wrist = bone("forearm_r").tail
        assert (wrist - shoulder).dot(up) > 0 and (wrist - shoulder).dot(outward) > 0
        if "JACKET" in clothes:
            shirt = next(o for o in bpy.context.scene.objects if o.get("excad_slot") == "top")
            mask = next(
                m for m in shirt.modifiers if m.type == "MASK" and m.vertex_group == shirt["excad_underlay_group"]
            )
            covered = len(shirt.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices)
            mask.show_viewport = False
            bpy.context.view_layer.update()
            full = len(shirt.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices)
            assert 0 < covered < full
            mask.show_viewport = True
            bpy.context.view_layer.update()
            digest = run("scene_digest")["h"]
            run("checkpoint", name="Posed layers")
            run("character_stylize", object=obj.name, remove=["outerwear"])
            assert not shirt.get("excad_underlay_group")
            assert len(shirt.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices) == full
            assert not any(o.get("excad_slot") == "outerwear" for o in bpy.context.scene.objects)
            run("rollback", name="Posed layers")
            assert run("scene_digest", since_hash=digest).get("unchanged")
            assert not run("render_inspect", target="Person", path=str(image_dir / "restored.png"))["warnings"]
        print("WARDROBE", preset, rig, flush=True)
    print("WARDROBE MOTION PASS", len(measured), "previews", max(measured), "max ms", flush=True)
