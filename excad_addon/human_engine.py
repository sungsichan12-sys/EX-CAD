# SPDX-FileCopyrightText: 2026 EX-CAD Authors
# SPDX-License-Identifier: GPL-2.0-or-later

import math
import os
import bpy
import bmesh
from mathutils import Matrix, Vector


HUMAN_PRESETS = {
    "ATHLETIC_MALE": {
        "gender": 1.0,
        "age": 0.42,
        "muscle": 0.82,
        "weight": 0.42,
        "proportions": 0.65,
        "height": 0.62,
        "cupsize": 0.0,
        "firmness": 0.8,
    },
    "ATHLETIC_FEMALE": {
        "gender": 0.0,
        "age": 0.40,
        "muscle": 0.68,
        "weight": 0.40,
        "proportions": 0.62,
        "height": 0.50,
        "cupsize": 0.48,
        "firmness": 0.78,
    },
    "REALISTIC_MALE": {
        "gender": 1.0,
        "age": 0.50,
        "muscle": 0.50,
        "weight": 0.52,
        "proportions": 0.50,
        "height": 0.52,
        "cupsize": 0.0,
        "firmness": 0.50,
    },
    "REALISTIC_FEMALE": {
        "gender": 0.0,
        "age": 0.48,
        "muscle": 0.42,
        "weight": 0.48,
        "proportions": 0.52,
        "height": 0.46,
        "cupsize": 0.50,
        "firmness": 0.55,
    },
    "HEROIC_SCULPT": {
        "gender": 0.95,
        "age": 0.45,
        "muscle": 0.95,
        "weight": 0.45,
        "proportions": 0.78,
        "height": 0.75,
        "cupsize": 0.0,
        "firmness": 0.90,
    },
    "TEEN": {
        "gender": 0.35,
        "age": 0.22,
        "muscle": 0.40,
        "weight": 0.40,
        "proportions": 0.50,
        "height": 0.42,
        "cupsize": 0.30,
        "firmness": 0.65,
    },
}

ETHNICITY_RACE_MAP = {
    "BALANCED": {"asian": 0.33, "caucasian": 0.34, "african": 0.33},
    "ASIAN": {"asian": 0.90, "caucasian": 0.05, "african": 0.05},
    "CAUCASIAN": {"asian": 0.05, "caucasian": 0.90, "african": 0.05},
    "AFRICAN": {"asian": 0.05, "caucasian": 0.05, "african": 0.90},
}


def _import_mpfb():
    try:
        from bl_ext.blender_org.mpfb.services.humanservice import HumanService
        from bl_ext.blender_org.mpfb.services.targetservice import TargetService
        from bl_ext.blender_org.mpfb.services.materialservice import MaterialService
        from bl_ext.blender_org.mpfb.services.locationservice import LocationService
        return HumanService, TargetService, MaterialService, LocationService
    except Exception:
        return None


def is_mpfb_available() -> bool:
    return _import_mpfb() is not None


def _apply_fallback_skin_material(obj, name: str):
    mat = bpy.data.materials.new(name=f"{name}_Skin")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.76, 0.58, 0.49, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.48
        if "Subsurface Weight" in bsdf.inputs:
            bsdf.inputs["Subsurface Weight"].default_value = 0.18
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def _fallback_sdf_human(height=1.75, muscularity=0.5, name="EXCAD_Human"):
    scale = max(0.5, min(3.0, height)) / 1.75
    musc = max(0.0, min(1.5, muscularity))
    bm = bmesh.new()

    def ell(center, radii):
        ret = bmesh.ops.create_uvsphere(bm, u_segments=20, v_segments=12, radius=1.0)
        mat = Matrix.Translation(Vector(center)) @ Matrix.Diagonal((radii[0], radii[1], radii[2], 1.0))
        bmesh.ops.transform(bm, matrix=mat, verts=ret["verts"])

    def cap(a, b, ra, rb):
        va, vb = Vector(a), Vector(b)
        axis = vb - va
        length = axis.length
        if length < 1e-5:
            ell(a, (ra, ra, ra))
            return
        rot = axis.normalized().to_track_quat('Z', 'Y').to_matrix().to_4x4()
        rings = []
        for r in range(9):
            t = r / 8.0
            rad = ra + (rb - ra) * t
            z = t * length
            rings.append([
                bm.verts.new(va + (rot @ Vector((rad * math.cos(2 * math.pi * s / 16), rad * math.sin(2 * math.pi * s / 16), z))))
                for s in range(16)
            ])
        for r in range(8):
            for s in range(16):
                sn = (s + 1) % 16
                bm.faces.new((rings[r][s], rings[r][sn], rings[r + 1][sn], rings[r + 1][s]))
        ell(a, (ra, ra, ra))
        ell(b, (rb, rb, rb))

    ell((0.0, 0.01, 1.63), (0.082, 0.098, 0.112))
    ell((0.0, -0.028, 1.565), (0.058, 0.068, 0.072))
    cap((0.0, 0.0, 1.44), (0.0, 0.008, 1.54), 0.056, 0.048)
    ell((0.0, 0.005, 1.29), (0.145, 0.105 + 0.015 * musc, 0.165))
    ell((0.0, -0.005, 1.09), (0.118 + 0.01 * musc, 0.092, 0.125))
    ell((0.0, 0.0, 0.93), (0.142, 0.102, 0.095))
    for sx in (-1.0, 1.0):
        cap((sx * 0.035, 0.01, 1.47), (sx * 0.175, 0.01, 1.41), 0.048, 0.038)
        ell((sx * 0.195, 0.01, 1.40), (0.056 + 0.014 * musc,) * 3)
        cap((sx * 0.195, 0.01, 1.40), (sx * 0.275, 0.025, 1.13), 0.048, 0.040)
        cap((sx * 0.275, 0.025, 1.13), (sx * 0.335, -0.015, 0.87), 0.040, 0.029)
        cap((sx * 0.092, 0.0, 0.89), (sx * 0.095, -0.008, 0.50), 0.078, 0.054)
        cap((sx * 0.095, -0.008, 0.50), (sx * 0.092, 0.015, 0.09), 0.054, 0.036)

    bmesh.ops.transform(bm, matrix=Matrix.Diagonal((scale, scale, scale, 1.0)), verts=bm.verts)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    obj.data.remesh_voxel_size = 0.025
    bpy.ops.object.voxel_remesh()
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return obj


def create_studio_human(
    preset: str = "ATHLETIC_FEMALE",
    gender: float = 0.5,
    age: float = 0.45,
    muscle: float = 0.65,
    weight: float = 0.45,
    height: float = 0.52,
    proportions: float = 0.6,
    ethnicity: str = "BALANCED",
    rig_name: str = "game_engine",
    apply_skin: bool = True,
    subdiv_level: int = 1,
    name: str = "EXCAD_Human",
):
    mpfb = _import_mpfb()
    if mpfb is None:
        obj = _fallback_sdf_human(height=1.5 + 0.5 * height, muscularity=muscle, name=name)
        if apply_skin:
            _apply_fallback_skin_material(obj, name)
        rig_obj = auto_rig_any_human_mesh(obj, name=f"{name}.rig") if (rig_name and rig_name != "NONE") else None
        obj["excad_character"] = True
        return obj, rig_obj, "FALLBACK_SDF"

    HumanService, TargetService, MaterialService, _ = mpfb
    macro = TargetService.get_default_macro_info_dict()
    if preset in HUMAN_PRESETS:
        macro.update(HUMAN_PRESETS[preset])
    else:
        macro["gender"] = float(gender)
        macro["age"] = float(age)
        macro["muscle"] = float(muscle)
        macro["weight"] = float(weight)
        macro["height"] = float(height)
        macro["proportions"] = float(proportions)
    macro["race"] = dict(ETHNICITY_RACE_MAP.get(ethnicity, ETHNICITY_RACE_MAP["BALANCED"]))

    basemesh = HumanService.create_human(
        mask_helpers=True,
        detailed_helpers=True,
        extra_vertex_groups=True,
        feet_on_ground=True,
        scale=0.1,
        macro_detail_dict=macro,
    )
    basemesh.name = name
    for poly in basemesh.data.polygons:
        poly.use_smooth = True

    if apply_skin:
        try:
            MaterialService.create_v2_skin_material(f"{name}_Skin", basemesh)
        except Exception:
            _apply_fallback_skin_material(basemesh, name)

    rig_obj = None
    if rig_name and rig_name != "NONE":
        if rig_name.startswith("rigify."):
            try:
                bpy.ops.preferences.addon_enable(module="rigify")
            except Exception as exc:
                raise RuntimeError(f"Failed to enable Rigify addon: {exc}") from exc
        rig_obj = HumanService.add_builtin_rig(basemesh, rig_name, import_weights=True)
        if rig_obj is not None:
            rig_obj.data.display_type = 'STICK'
            rig_obj.show_in_front = True
            for m in basemesh.modifiers:
                if m.type == 'ARMATURE':
                    m.use_deform_preserve_volume = True

    if subdiv_level > 0:
        sub = basemesh.modifiers.new("EXCAD_Subsurf", 'SUBSURF')
        sub.levels = int(subdiv_level)
        sub.render_levels = max(2, int(subdiv_level))

    for o in bpy.context.selected_objects:
        o.select_set(False)
    basemesh.select_set(True)
    bpy.context.view_layer.objects.active = basemesh

    # Ensure ARMATURE modifier runs before SUBSURF
    for m in list(basemesh.modifiers):
        if m.type == 'ARMATURE':
            bpy.ops.object.modifier_move_to_index(modifier=m.name, index=0)
            break

    basemesh["excad_character"] = True
    return basemesh, rig_obj, "MPFB2"


def list_target_categories() -> list[str]:
    mpfb = _import_mpfb()
    if mpfb is None:
        return []
    _, _, _, LocationService = mpfb
    targets_dir = LocationService.get_mpfb_data("targets")
    if not os.path.isdir(targets_dir):
        return []
    return sorted(
        d for d in os.listdir(targets_dir)
        if not d.startswith("_") and os.path.isdir(os.path.join(targets_dir, d))
    )


def list_targets_in_category(category: str) -> list[str]:
    mpfb = _import_mpfb()
    if mpfb is None:
        return []
    _, _, _, LocationService = mpfb
    cat_dir = os.path.join(LocationService.get_mpfb_data("targets"), category)
    if not os.path.isdir(cat_dir):
        return []
    out = []
    for root, _, files in os.walk(cat_dir):
        for fn in files:
            if fn.endswith(".target") or fn.endswith(".target.gz"):
                rel = os.path.relpath(os.path.join(root, fn), cat_dir)
                out.append(rel)
    return sorted(out)


def apply_morph_target(obj, category: str, target_relpath: str, weight: float = 0.5) -> str:
    mpfb = _import_mpfb()
    if mpfb is None:
        raise RuntimeError("MPFB2 extension is not available")
    HumanService, TargetService, _, LocationService = mpfb
    full_path = os.path.join(LocationService.get_mpfb_data("targets"), category, target_relpath)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Target not found: {full_path}")
    target_name = os.path.splitext(os.path.basename(full_path))[0].replace(".target", "")
    TargetService.load_target(obj, full_path, weight=float(weight), name=target_name)
    if obj.find_armature() is not None:
        HumanService.refit(obj)
    from .wardrobe import refit_eyes
    refit_eyes(obj)
    return target_name


def refit_human_rig(obj) -> bool:
    mpfb = _import_mpfb()
    if mpfb is None:
        return False
    HumanService, _, _, _ = mpfb
    if obj is None or obj.find_armature() is None:
        return False
    HumanService.refit(obj)
    from .wardrobe import refit_eyes
    refit_eyes(obj)
    return True


def create_humanoid_armature(corners, name):
    """Place the shared upright skeleton in world space at the scale of the model."""
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    z0 = min(zs)
    h = max(1e-6, max(zs) - z0)
    w = max(1e-6, max(xs) - min(xs))

    arm_data = bpy.data.armatures.new(name)
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.collection.objects.link(arm_obj)
    arm_data.display_type = 'STICK'
    arm_obj.show_in_front = True

    for o in bpy.context.selected_objects:
        o.select_set(False)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')

    eb = arm_data.edit_bones

    def add_bone(bname, head, tail, parent=None, connect=False):
        b = eb.new(bname)
        b.head = Vector(head)
        b.tail = Vector(tail)
        if parent is not None:
            b.parent = parent
            b.use_connect = connect
        return b

    hips = add_bone("hips", (cx, cy, z0 + 0.50 * h), (cx, cy, z0 + 0.58 * h))
    spine = add_bone("spine", (cx, cy, z0 + 0.58 * h), (cx, cy, z0 + 0.70 * h), hips, True)
    chest = add_bone("chest", (cx, cy, z0 + 0.70 * h), (cx, cy, z0 + 0.82 * h), spine, True)
    neck = add_bone("neck", (cx, cy, z0 + 0.82 * h), (cx, cy, z0 + 0.88 * h), chest, True)
    add_bone("head", (cx, cy, z0 + 0.88 * h), (cx, cy, z0 + 0.99 * h), neck, True)

    sh_x = min(0.16 * h, 0.22 * w)
    el_x = min(0.28 * h, 0.36 * w)
    wr_x = min(0.38 * h, 0.48 * w)
    hip_x = min(0.08 * h, 0.14 * w)

    for side, sx in (("L", 1.0), ("R", -1.0)):
        sh = add_bone(f"shoulder.{side}", (cx + sx * 0.03 * h, cy, z0 + 0.80 * h), (cx + sx * sh_x, cy, z0 + 0.80 * h), chest, False)
        ua = add_bone(f"upper_arm.{side}", (cx + sx * sh_x, cy, z0 + 0.80 * h), (cx + sx * el_x, cy + 0.01 * h, z0 + 0.64 * h), sh, True)
        add_bone(f"forearm.{side}", (cx + sx * el_x, cy + 0.01 * h, z0 + 0.64 * h), (cx + sx * wr_x, cy - 0.01 * h, z0 + 0.49 * h), ua, True)

        th = add_bone(f"thigh.{side}", (cx + sx * hip_x, cy, z0 + 0.51 * h), (cx + sx * hip_x, cy - 0.01 * h, z0 + 0.28 * h), hips, False)
        kn = add_bone(f"shin.{side}", (cx + sx * hip_x, cy - 0.01 * h, z0 + 0.28 * h), (cx + sx * hip_x, cy + 0.01 * h, z0 + 0.05 * h), th, True)
        add_bone(f"foot.{side}", (cx + sx * hip_x, cy + 0.01 * h, z0 + 0.05 * h), (cx + sx * hip_x, cy - 0.09 * h, z0 + 0.02 * h), kn, True)

    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj


def auto_rig_any_human_mesh(obj, name: str = "EXCAD_AutoRig"):
    if obj is None or obj.type != 'MESH':
        raise ValueError("Target object must be a mesh")
    if bpy.context.mode != "OBJECT":
        raise ValueError("Switch to Object Mode before fitting the armature")
    from .humanoid_skin import bind_heat

    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    arm_obj = create_humanoid_armature(corners, name)
    bind_heat(obj, arm_obj, keep_parent=False)
    return arm_obj
