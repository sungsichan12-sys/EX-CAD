# SPDX-FileCopyrightText: 2026 EX-CAD Authors
# SPDX-License-Identifier: GPL-2.0-or-later

import math
import bpy
import bmesh
from mathutils import Euler, Matrix, Vector

from . import human_engine


# ---------------------------------------------------------------------------
# Material & Mesh Helpers
# ---------------------------------------------------------------------------

def _get_or_create_pbr_mat(name: str, base_color, roughness=0.5, metallic=0.0):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)
        bsdf.inputs["Roughness"].default_value = float(roughness)
        bsdf.inputs["Metallic"].default_value = float(metallic)
    return mat


def _finish_bmesh_object(bm, name: str, mat=None, bevel_width: float = 0.0, auto_ground: bool = True):
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    for poly in obj.data.polygons:
        poly.use_smooth = True

    if mat is not None:
        obj.data.materials.append(mat)

    if bevel_width > 0.0:
        bev = obj.modifiers.new("EXCAD_Bevel", 'BEVEL')
        bev.width = float(bevel_width)
        bev.segments = 2
        bev.limit_method = 'ANGLE'
        bev.angle_limit = math.radians(30.0)

    if auto_ground and len(obj.data.vertices) > 0:
        min_z = min(v.co.z for v in obj.data.vertices)
        obj.location.z -= min_z

    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return obj


def _add_box(bm, center, size):
    ret = bmesh.ops.create_cube(bm, size=1.0)
    mat = Matrix.Translation(Vector(center)) @ Matrix.Diagonal((size[0], size[1], size[2], 1.0))
    bmesh.ops.transform(bm, matrix=mat, verts=ret["verts"])


def _add_cyl(bm, center, radius, depth, segs=24):
    ret = bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=segs, radius1=radius, radius2=radius, depth=depth
    )
    bmesh.ops.transform(bm, matrix=Matrix.Translation(Vector(center)), verts=ret["verts"])


# ---------------------------------------------------------------------------
# 1. Pose Presets (works with game_engine, mixamo, default, EXCAD_AutoRig)
# ---------------------------------------------------------------------------

_BONE_ALIASES = {
    "spine": ["spine_01", "spine01", "mixamorig:Spine", "spine"],
    "chest": ["spine_02", "spine_03", "spine03", "mixamorig:Spine2", "chest"],
    "head": ["head", "mixamorig:Head"],
    "upperarm_l": ["upperarm_l", "upper_arm.L", "upperarm01.L", "mixamorig:LeftArm"],
    "upperarm_r": ["upperarm_r", "upper_arm.R", "upperarm01.R", "mixamorig:RightArm"],
    "forearm_l": ["lowerarm_l", "forearm.L", "lowerarm01.L", "mixamorig:LeftForeArm"],
    "forearm_r": ["lowerarm_r", "forearm.R", "lowerarm01.R", "mixamorig:RightForeArm"],
    "thigh_l": ["thigh_l", "thigh.L", "thigh01.L", "mixamorig:LeftUpLeg"],
    "thigh_r": ["thigh_r", "thigh.R", "thigh01.R", "mixamorig:RightUpLeg"],
    "calf_l": ["calf_l", "shin.L", "lowerleg01.L", "mixamorig:LeftLeg"],
    "calf_r": ["calf_r", "shin.R", "lowerleg01.R", "mixamorig:RightLeg"],
}

_POSE_EULER_DEG = {
    "RESET": {},
    "RELAXED": {
        "upperarm_l": (4.0, 0.0, -8.0),
        "upperarm_r": (4.0, 0.0, 8.0),
        "forearm_l": (-12.0, 0.0, 0.0),
        "forearm_r": (-12.0, 0.0, 0.0),
        "head": (3.0, 0.0, -4.0),
    },
    "HEROIC": {
        "chest": (-6.0, 0.0, 0.0),
        "head": (-4.0, 0.0, 6.0),
        "upperarm_l": (6.0, 0.0, -18.0),
        "upperarm_r": (6.0, 0.0, 18.0),
        "forearm_l": (-22.0, 0.0, 12.0),
        "forearm_r": (-22.0, 0.0, -12.0),
        "thigh_l": (-4.0, 0.0, -5.0),
        "thigh_r": (4.0, 0.0, 5.0),
    },
    "WAVE": {
        "upperarm_l": (5.0, 0.0, -10.0),
        "forearm_l": (-14.0, 0.0, 0.0),
        "head": (-4.0, 0.0, -8.0),
    },
    "WALK": {
        "spine": (5.0, 0.0, 4.0),
        "thigh_l": (-24.0, 0.0, 0.0),
        "calf_l": (20.0, 0.0, 0.0),
        "thigh_r": (20.0, 0.0, 0.0),
        "calf_r": (12.0, 0.0, 0.0),
        "upperarm_l": (22.0, 0.0, -6.0),
        "forearm_l": (-18.0, 0.0, 0.0),
        "upperarm_r": (-18.0, 0.0, 6.0),
        "forearm_r": (-25.0, 0.0, 0.0),
    },
    "SITTING": {
        "thigh_l": (-86.0, 0.0, -4.0),
        "thigh_r": (-86.0, 0.0, 4.0),
        "calf_l": (84.0, 0.0, 0.0),
        "calf_r": (84.0, 0.0, 0.0),
        "upperarm_l": (-18.0, 0.0, -8.0),
        "upperarm_r": (-18.0, 0.0, 8.0),
        "forearm_l": (-45.0, 0.0, 0.0),
        "forearm_r": (-45.0, 0.0, 0.0),
    },
}


def _raise_waving_hand(arm_obj):
    """Aim the elbow and wrist in the character's anatomical frame, independent of bone roll."""

    def find(role):
        return next((arm_obj.pose.bones[n] for n in _BONE_ALIASES[role] if n in arm_obj.pose.bones), None)

    right, left, forearm, head = (find(role) for role in ("upperarm_r", "upperarm_l", "forearm_r", "head"))
    if not all((right, left, forearm, head)):
        return 0
    shoulder = right.bone.head_local
    other = left.bone.head_local
    outward = (shoulder - other).normalized()
    up = head.bone.head_local - (shoulder + other) / 2
    up = (up - outward * up.dot(outward)).normalized()
    forward = up.cross(outward).normalized()
    for bone, direction in (
        (right, outward + 0.40 * up + 0.24 * forward),
        (forearm, up + 0.07 * outward + 0.18 * forward),
    ):
        delta = (bone.tail - bone.head).rotation_difference(direction)
        pivot = Matrix.Translation(bone.head)
        bone.matrix = pivot @ delta.to_matrix().to_4x4() @ pivot.inverted() @ bone.matrix
        bpy.context.view_layer.update()
    return 2


def apply_pose_preset(arm_obj, pose_name: str = "RELAXED") -> int:
    if arm_obj is None or arm_obj.type != "ARMATURE" or not arm_obj.pose:
        return 0
    pose_name = pose_name.upper()
    if pose_name not in _POSE_EULER_DEG:
        raise ValueError(f"Unknown pose preset {pose_name}")
    for pb in arm_obj.pose.bones:
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")
        pb.location = Vector((0.0, 0.0, 0.0))
        if pose_name == "RESET":
            pb.scale = (1.0, 1.0, 1.0)

    if pose_name == "RESET":
        bpy.context.view_layer.update()
        return len(arm_obj.pose.bones)
    spec = _POSE_EULER_DEG[pose_name]
    applied = 0
    for role, deg in spec.items():
        for bname in _BONE_ALIASES.get(role, []):
            pb = arm_obj.pose.bones.get(bname)
            if pb is not None:
                pb.rotation_euler = Euler(tuple(math.radians(a) for a in deg), 'XYZ')
                applied += 1
                break
    bpy.context.view_layer.update()
    if pose_name == 'WAVE':
        applied += _raise_waving_hand(arm_obj)
    return applied


# ---------------------------------------------------------------------------
# 2. Studio Backdrop, 3-Point Lighting & Camera
# ---------------------------------------------------------------------------

def create_studio_backdrop(style: str = 'CYCLORAMA', size: float = 4.5, target_height: float = 1.75):
    created = []
    s = max(2.0, float(size))
    half = 0.5 * s
    studio_mat = _get_or_create_pbr_mat("EXCAD_StudioMat", (0.24, 0.25, 0.28), roughness=0.82, metallic=0.0)

    if style == 'ROOM':
        bm = bmesh.new()
        _add_box(bm, (0.0, 0.0, -0.05), (s, s, 0.1))
        _add_box(bm, (0.0, half, 1.4), (s, 0.1, 2.8))
        _add_box(bm, (-half, 0.0, 1.4), (0.1, s, 2.8))
        _add_box(bm, (half, 0.0, 1.4), (0.1, s, 2.8))
        created.append(_finish_bmesh_object(bm, "EXCAD_StudioRoom", studio_mat, auto_ground=False))
    else:
        bm = bmesh.new()
        # Swept cyclorama profile in YZ, extruded along X
        pts_yz = [(-half, 0.0), (0.2 * half, 0.0)]
        r = 0.35 * half
        for i in range(1, 9):
            ang = -0.5 * math.pi + (0.5 * math.pi * i / 8.0)
            pts_yz.append((0.2 * half + r * math.cos(ang), r + r * math.sin(ang)))
        pts_yz.append((0.2 * half + r, 1.6 * target_height))

        left = [bm.verts.new((-half, y, z)) for y, z in pts_yz]
        right = [bm.verts.new((half, y, z)) for y, z in pts_yz]
        for i in range(len(pts_yz) - 1):
            bm.faces.new((left[i], right[i], right[i + 1], left[i + 1]))
        created.append(_finish_bmesh_object(bm, "EXCAD_Cyclorama", studio_mat, auto_ground=False))

    def add_area_light(lname, loc, target, energy, sz, color=(1.0, 1.0, 1.0)):
        ld = bpy.data.lights.new(name=lname, type='AREA')
        ld.energy = float(energy)
        ld.size = float(sz)
        ld.color = color
        lo = bpy.data.objects.new(lname, ld)
        bpy.context.collection.objects.link(lo)
        lo.location = Vector(loc)
        direction = Vector(target) - lo.location
        lo.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        created.append(lo)

    aim = (0.0, 0.0, target_height * 0.52)
    add_area_light("EXCAD_KeyLight", (2.2, -2.4, 2.6), aim, 220.0, 1.6, (1.0, 0.97, 0.93))
    add_area_light("EXCAD_FillLight", (-2.4, -1.8, 1.8), aim, 90.0, 2.2, (0.88, 0.93, 1.0))
    add_area_light("EXCAD_RimLight", (0.2, 2.2, 2.4), aim, 160.0, 1.4, (0.95, 0.98, 1.0))

    cam_d = bpy.data.cameras.new("EXCAD_Camera")
    cam_d.lens = 50.0
    cam_o = bpy.data.objects.new("EXCAD_Camera", cam_d)
    bpy.context.collection.objects.link(cam_o)
    cam_o.location = Vector((0.35, -3.4, target_height * 0.56))
    cam_o.rotation_euler = (Vector(aim) - cam_o.location).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = cam_o
    created.append(cam_o)
    return created


# ---------------------------------------------------------------------------
# 3. Character + Pose + Studio Stage
# ---------------------------------------------------------------------------

def create_character_stage(
    preset: str = 'ATHLETIC_FEMALE',
    pose: str = 'RELAXED',
    ethnicity: str = 'BALANCED',
    rig: str = 'game_engine',
    backdrop: str = 'CYCLORAMA',
):
    human, rig_obj, engine = human_engine.create_studio_human(
        preset=preset, ethnicity=ethnicity, rig_name=rig, apply_skin=True, subdiv_level=1
    )
    posed_bones = apply_pose_preset(rig_obj, pose) if rig_obj is not None else 0
    stage_objs = create_studio_backdrop(style=backdrop, size=4.5, target_height=1.75)
    return {
        "human": human,
        "rig": rig_obj,
        "engine": engine,
        "posed_bones": posed_bones,
        "stage_objects": stage_objs,
    }


# ---------------------------------------------------------------------------
# 4. Parametric CAD Mechanical Parts
# ---------------------------------------------------------------------------

def create_mechanical_part(part_type: str = 'GEAR', size: float = 0.5, detail: int = 16, name: str = ""):
    s = max(0.05, float(size))
    d = max(4, int(detail))
    steel = _get_or_create_pbr_mat("EXCAD_BrushedSteel", (0.68, 0.71, 0.74), roughness=0.28, metallic=0.92)
    bm = bmesh.new()
    pt = part_type.upper()

    if pt == 'GEAR':
        r_out = 0.5 * s
        r_root = 0.40 * s
        r_bore = 0.12 * s
        thick = 0.16 * s
        steps = d * 4
        bot_outer, top_outer, bot_inner, top_inner = [], [], [], []
        for i in range(steps):
            ang = 2.0 * math.pi * i / steps
            phase = i % 4
            rad = r_out if phase in (1, 2) else r_root
            ca, sa = math.cos(ang), math.sin(ang)
            bot_outer.append(bm.verts.new((rad * ca, rad * sa, 0.0)))
            top_outer.append(bm.verts.new((rad * ca, rad * sa, thick)))
            bot_inner.append(bm.verts.new((r_bore * ca, r_bore * sa, 0.0)))
            top_inner.append(bm.verts.new((r_bore * ca, r_bore * sa, thick)))
        for i in range(steps):
            n = (i + 1) % steps
            bm.faces.new((bot_outer[i], bot_outer[n], top_outer[n], top_outer[i]))
            bm.faces.new((top_inner[i], top_inner[n], bot_inner[n], bot_inner[i]))
            bm.faces.new((bot_inner[i], bot_inner[n], bot_outer[n], bot_outer[i]))
            bm.faces.new((top_outer[i], top_outer[n], top_inner[n], top_inner[i]))
        return _finish_bmesh_object(bm, name or "EXCAD_Gear", steel, bevel_width=s * 0.006)

    if pt == 'FLANGE':
        r_outer = 0.5 * s
        r_hub = 0.24 * s
        r_bore = 0.14 * s
        t_plate = 0.10 * s
        t_hub = 0.22 * s
        segs = max(24, d * 2)
        _add_cyl(bm, (0, 0, t_plate * 0.5), r_outer, t_plate, segs)
        _add_cyl(bm, (0, 0, t_hub * 0.5), r_hub, t_hub, segs)
        obj = _finish_bmesh_object(bm, name or "EXCAD_Flange", steel, bevel_width=s * 0.005)
        # Cut central bore + perimeter bolt holes via fast boolean
        cbm = bmesh.new()
        _add_cyl(cbm, (0, 0, t_hub * 0.5), r_bore, t_hub * 2.5, 24)
        n_holes = max(4, min(12, d))
        r_bolt_ring = 0.37 * s
        for i in range(n_holes):
            ang = 2.0 * math.pi * i / n_holes
            _add_cyl(cbm, (r_bolt_ring * math.cos(ang), r_bolt_ring * math.sin(ang), t_plate * 0.5), s * 0.04, t_plate * 3.0, 12)
        cutter = _finish_bmesh_object(cbm, "EXCAD_FlangeCutter", auto_ground=False)
        for o in bpy.context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        mod = obj.modifiers.new("EXCAD_Holes", 'BOOLEAN')
        mod.operation = 'DIFFERENCE'
        mod.object = cutter
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(cutter, do_unlink=True)
        return obj

    if pt == 'BRACKET':
        t = 0.08 * s
        _add_box(bm, (0.0, 0.0, t * 0.5), (s, 0.6 * s, t))
        _add_box(bm, (-0.5 * s + t * 0.5, 0.0, 0.45 * s), (t, 0.6 * s, 0.9 * s))
        _add_box(bm, (0.0, 0.0, 0.25 * s), (0.5 * s, t * 0.7, 0.4 * s))
        return _finish_bmesh_object(bm, name or "EXCAD_Bracket", steel, bevel_width=s * 0.008)

    if pt == 'ENCLOSURE':
        w, dp, h = s, 0.7 * s, 0.45 * s
        t = 0.05 * s
        _add_box(bm, (0, 0, t * 0.5), (w, dp, t))
        _add_box(bm, (0, -0.5 * dp + 0.5 * t, 0.5 * h), (w, t, h))
        _add_box(bm, (0, 0.5 * dp - 0.5 * t, 0.5 * h), (w, t, h))
        _add_box(bm, (-0.5 * w + 0.5 * t, 0, 0.5 * h), (t, dp - 2 * t, h))
        _add_box(bm, (0.5 * w - 0.5 * t, 0, 0.5 * h), (t, dp - 2 * t, h))
        return _finish_bmesh_object(bm, name or "EXCAD_Enclosure", steel, bevel_width=s * 0.01)

    # 'BOLT'
    _add_cyl(bm, (0, 0, 0.14 * s), 0.30 * s, 0.28 * s, segs=6)
    _add_cyl(bm, (0, 0, 0.64 * s), 0.16 * s, 0.72 * s, segs=24)
    return _finish_bmesh_object(bm, name or "EXCAD_Bolt", steel, bevel_width=s * 0.008)


# ---------------------------------------------------------------------------
# 5. Furniture & Architectural Props
# ---------------------------------------------------------------------------

def create_furniture_prop(prop_type: str = 'TABLE', scale: float = 1.0, name: str = ""):
    sc = max(0.1, float(scale))
    wood = _get_or_create_pbr_mat("EXCAD_OakWood", (0.56, 0.39, 0.23), roughness=0.45, metallic=0.0)
    stone = _get_or_create_pbr_mat("EXCAD_PedestalStone", (0.78, 0.77, 0.75), roughness=0.38, metallic=0.0)
    bm = bmesh.new()
    pt = prop_type.upper()

    if pt == 'TABLE':
        w, d, h = 1.4 * sc, 0.8 * sc, 0.75 * sc
        top_t = 0.05 * sc
        leg_s = 0.06 * sc
        _add_box(bm, (0, 0, h - top_t * 0.5), (w, d, top_t))
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                _add_box(bm, (sx * (0.5 * w - 0.08 * sc), sy * (0.5 * d - 0.08 * sc), 0.5 * (h - top_t)), (leg_s, leg_s, h - top_t))
        return _finish_bmesh_object(bm, name or "EXCAD_Table", wood, bevel_width=0.006 * sc)

    if pt == 'CHAIR':
        w, d, sh = 0.48 * sc, 0.48 * sc, 0.46 * sc
        bh = 0.90 * sc
        t = 0.04 * sc
        _add_box(bm, (0, 0, sh), (w, d, t))
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                _add_box(bm, (sx * (0.5 * w - 0.04 * sc), sy * (0.5 * d - 0.04 * sc), 0.5 * sh), (t, t, sh))
        _add_box(bm, (0, 0.5 * d - 0.5 * t, 0.5 * (sh + bh)), (w, t, bh - sh))
        return _finish_bmesh_object(bm, name or "EXCAD_Chair", wood, bevel_width=0.005 * sc)

    if pt == 'SHELF':
        w, d, h = 1.0 * sc, 0.35 * sc, 1.8 * sc
        t = 0.03 * sc
        _add_box(bm, (-0.5 * w + 0.5 * t, 0, 0.5 * h), (t, d, h))
        _add_box(bm, (0.5 * w - 0.5 * t, 0, 0.5 * h), (t, d, h))
        _add_box(bm, (0, 0.5 * d - 0.5 * t, 0.5 * h), (w, t * 0.6, h))
        for i in range(5):
            z = 0.04 * sc + i * ((h - 0.08 * sc) / 4.0)
            _add_box(bm, (0, 0, z), (w - 2 * t, d - t, t))
        return _finish_bmesh_object(bm, name or "EXCAD_Shelf", wood, bevel_width=0.004 * sc)

    # 'PEDESTAL'
    r, h = 0.42 * sc, 0.78 * sc
    _add_cyl(bm, (0, 0, 0.04 * sc), r * 1.12, 0.08 * sc, 32)
    _add_cyl(bm, (0, 0, 0.5 * h), r, h - 0.14 * sc, 32)
    _add_cyl(bm, (0, 0, h - 0.035 * sc), r * 1.08, 0.07 * sc, 32)
    return _finish_bmesh_object(bm, name or "EXCAD_Pedestal", stone, bevel_width=0.006 * sc)
