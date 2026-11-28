# SPDX-FileCopyrightText: 2026 EX-CAD Authors
# SPDX-License-Identifier: GPL-2.0-or-later

import math
import json
import bpy
import bmesh
from mathutils import Vector


def _pbr_mat(name: str, color, roughness=0.5, metallic=0.0, sheen=0.0):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.diffuse_color = (*color, 1.0)
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = float(roughness)
        bsdf.inputs["Metallic"].default_value = float(metallic)
        if "Sheen Weight" in bsdf.inputs:
            bsdf.inputs["Sheen Weight"].default_value = float(sheen)
    return mat


def _tune_fair_skin(human_obj):
    if not human_obj.data.materials:
        return
    mat = human_obj.data.materials[0]
    if mat:
        mat.diffuse_color = (0.86, 0.70, 0.62, 1.0)
        if mat.use_nodes:
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (0.86, 0.70, 0.62, 1.0)
                bsdf.inputs["Roughness"].default_value = 0.44


def _create_eye_material():
    mat = bpy.data.materials.get("EXCAD_EyeMat")
    if mat is None:
        mat = bpy.data.materials.new(name="EXCAD_EyeMat")
    mat.use_nodes = True
    mat.diffuse_color = (0.04, 0.025, 0.018, 1.0)
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.04
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 1.0

    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])

    px = nt.nodes.new("ShaderNodeMath")
    px.operation = 'POWER'
    px.inputs[1].default_value = 2.0
    pz = nt.nodes.new("ShaderNodeMath")
    pz.operation = 'POWER'
    pz.inputs[1].default_value = 2.0
    add = nt.nodes.new("ShaderNodeMath")
    add.operation = 'ADD'
    sq = nt.nodes.new("ShaderNodeMath")
    sq.operation = 'SQRT'

    nt.links.new(sep.outputs["X"], px.inputs[0])
    nt.links.new(sep.outputs["Z"], pz.inputs[0])
    nt.links.new(px.outputs["Value"], add.inputs[0])
    nt.links.new(pz.outputs["Value"], add.inputs[1])
    nt.links.new(add.outputs["Value"], sq.inputs[0])

    front_lt = nt.nodes.new("ShaderNodeMath")
    front_lt.operation = 'LESS_THAN'
    front_lt.inputs[1].default_value = 0.0
    nt.links.new(sep.outputs["Y"], front_lt.inputs[0])

    iris_lt = nt.nodes.new("ShaderNodeMath")
    iris_lt.operation = 'LESS_THAN'
    iris_lt.inputs[1].default_value = 0.0062
    nt.links.new(sq.outputs["Value"], iris_lt.inputs[0])

    iris_mask = nt.nodes.new("ShaderNodeMath")
    iris_mask.operation = 'MULTIPLY'
    nt.links.new(iris_lt.outputs["Value"], iris_mask.inputs[0])
    nt.links.new(front_lt.outputs["Value"], iris_mask.inputs[1])

    pupil_lt = nt.nodes.new("ShaderNodeMath")
    pupil_lt.operation = 'LESS_THAN'
    pupil_lt.inputs[1].default_value = 0.0028
    nt.links.new(sq.outputs["Value"], pupil_lt.inputs[0])

    pupil_mask = nt.nodes.new("ShaderNodeMath")
    pupil_mask.operation = 'MULTIPLY'
    nt.links.new(pupil_lt.outputs["Value"], pupil_mask.inputs[0])
    nt.links.new(front_lt.outputs["Value"], pupil_mask.inputs[1])

    mix1 = nt.nodes.new("ShaderNodeMix")
    mix1.data_type = 'RGBA'
    mix1.inputs["A"].default_value = (0.95, 0.94, 0.92, 1.0)
    mix1.inputs["B"].default_value = (0.055, 0.030, 0.018, 1.0)
    nt.links.new(iris_mask.outputs["Value"], mix1.inputs["Factor"])

    mix2 = nt.nodes.new("ShaderNodeMix")
    mix2.data_type = 'RGBA'
    mix2.inputs["B"].default_value = (0.008, 0.008, 0.010, 1.0)
    nt.links.new(pupil_mask.outputs["Value"], mix2.inputs["Factor"])
    nt.links.new(mix1.outputs["Result"], mix2.inputs["A"])

    nt.links.new(mix2.outputs["Result"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _vgroup_center(human_obj, eval_mesh, candidates: tuple[str, ...]):
    for vg_name in candidates:
        vg = human_obj.vertex_groups.get(vg_name)
        if vg is None:
            continue
        idx = vg.index
        pts = [
            eval_mesh.vertices[i].co
            for i, v in enumerate(human_obj.data.vertices)
            if any(g.group == idx and g.weight > 0.1 for g in v.groups)
        ]
        if pts:
            acc = Vector((0.0, 0.0, 0.0))
            for p in pts:
                acc += p
            return acc / len(pts)
    return None


def _create_eyes(human_obj, l_pos: Vector, r_pos: Vector):
    eye_mat = _create_eye_material()
    eyes = []
    for name, pos in (("EXCAD_Eye_L", l_pos), ("EXCAD_Eye_R", r_pos)):
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=16, radius=0.0122)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        mesh = bpy.data.meshes.new(f"{name}Mesh")
        bm.to_mesh(mesh)
        bm.free()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        obj.location = pos
        obj.data.materials.append(eye_mat)
        for p in obj.data.polygons:
            p.use_smooth = True
        obj.parent = human_obj
        eyes.append(obj)
    return eyes


def _create_eyebrows(human_obj, l_eye: Vector, r_eye: Vector):
    brow_mat = _pbr_mat("EXCAD_EyebrowMat", (0.04, 0.028, 0.022), roughness=0.55)
    brows = []
    for name, eye, sx in (("EXCAD_Brow_L", l_eye, 1.0), ("EXCAD_Brow_R", r_eye, -1.0)):
        cd = bpy.data.curves.new(f"{name}Curve", type='CURVE')
        cd.dimensions = '3D'
        cd.bevel_depth = 0.0026
        cd.bevel_resolution = 2
        pts = [
            (eye.x - sx * 0.015, eye.y - 0.019, eye.z + 0.018),
            (eye.x + sx * 0.001, eye.y - 0.020, eye.z + 0.021),
            (eye.x + sx * 0.019, eye.y - 0.014, eye.z + 0.019),
        ]
        sp = cd.splines.new('NURBS')
        sp.points.add(len(pts) - 1)
        for i, pco in enumerate(pts):
            sp.points[i].co = (*pco, 1.0)
            sp.points[i].radius = 0.85 if i == 0 else (1.0 if i == 1 else 0.45)
        sp.use_endpoint_u = True
        obj = bpy.data.objects.new(name, cd)
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(brow_mat)
        obj.parent = human_obj
        brows.append(obj)
    return brows


def _create_hair(human_obj, head_center: Vector, head_rx: float, head_ry: float, head_rz: float):
    hair_mat = _pbr_mat("EXCAD_DarkWavyHair", (0.022, 0.015, 0.012), roughness=0.32, metallic=0.04, sheen=0.25)

    bm = bmesh.new()
    ret = bmesh.ops.create_uvsphere(bm, u_segments=28, v_segments=16, radius=1.0)
    for v in list(ret["verts"]):
        if v.co.z < -0.10 or (v.co.y < -0.35 and v.co.z < 0.42):
            bm.verts.remove(v)
        else:
            v.co = Vector((
                head_center.x + v.co.x * head_rx * 1.06,
                head_center.y + v.co.y * head_ry * 1.06,
                head_center.z + v.co.z * head_rz * 1.06 + 0.006,
            ))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    cap_mesh = bpy.data.meshes.new("EXCAD_HairCap")
    bm.to_mesh(cap_mesh)
    bm.free()
    cap_obj = bpy.data.objects.new("EXCAD_HairCap", cap_mesh)
    bpy.context.collection.objects.link(cap_obj)
    cap_obj.data.materials.append(hair_mat)
    for p in cap_obj.data.polygons:
        p.use_smooth = True

    curve_data = bpy.data.curves.new("EXCAD_HairStrands", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = 0.0085
    curve_data.bevel_resolution = 2
    curve_data.resolution_u = 10

    cx, cy, cz = head_center.x, head_center.y, head_center.z
    top_z = cz + head_rz * 1.05
    n_strands = 48

    for i in range(n_strands):
        side = -1.0 if (i % 2 == 0) else 1.0
        t = (i // 2) / max(1, (n_strands // 2) - 1)
        ang = 0.24 * math.pi + t * 0.76 * math.pi
        sx = side * math.sin(ang) * head_rx * 1.05
        sy = -math.cos(ang) * head_ry * 1.05

        drop = 0.44 if (t < 0.35 and side < 0) else (0.40 if t < 0.45 else 0.36)
        wave_amp = 0.018 + 0.008 * (1.0 - t)

        pts = [
            (cx + side * 0.005, cy + 0.01 * t, top_z),
            (cx + sx * 0.94, cy + sy * 0.94, cz + head_rz * 0.58),
            (cx + sx * 1.18, cy + sy * 1.10, cz - head_rz * 0.35),
            (cx + sx * 1.24 + side * wave_amp, cy + sy * 1.00 - (0.035 if t < 0.35 else 0.0), cz - head_rz * 0.35 - drop * 0.45),
            (cx + sx * 1.08 - side * wave_amp * 0.8, cy + sy * 0.80 - (0.060 if t < 0.35 else 0.0), cz - head_rz * 0.35 - drop * 0.85),
            (cx + sx * 0.82 + side * wave_amp * 0.5, cy + sy * 0.55 - (0.070 if t < 0.35 else 0.0), cz - head_rz * 0.35 - drop),
        ]

        sp = curve_data.splines.new('NURBS')
        sp.points.add(len(pts) - 1)
        for k, pco in enumerate(pts):
            sp.points[k].co = (pco[0], pco[1], pco[2], 1.0)
            sp.points[k].radius = 1.0 - 0.55 * (k / (len(pts) - 1))
        sp.use_endpoint_u = True

    strands_obj = bpy.data.objects.new("EXCAD_HairStrands", curve_data)
    bpy.context.collection.objects.link(strands_obj)
    strands_obj.data.materials.append(hair_mat)

    for o in (cap_obj, strands_obj):
        o.parent = human_obj
    return [cap_obj, strands_obj]


def _create_vneck_sweater(human_obj, eval_mesh, neck_z: float, hip_z: float):
    sweater_mat = _pbr_mat("EXCAD_CreamKnitSweater", (0.90, 0.88, 0.84), roughness=0.86, metallic=0.0, sheen=0.45)

    sw_data = human_obj.data.copy()
    sw_data.name = f"{human_obj.name}_SweaterMesh"
    sw_obj = bpy.data.objects.new("EXCAD_Sweater", sw_data)
    bpy.context.collection.objects.link(sw_obj)
    sw_obj.matrix_world = human_obj.matrix_world.copy()

    sw_obj.parent = human_obj.parent
    for m in human_obj.modifiers:
        if m.type == 'ARMATURE':
            am = sw_obj.modifiers.new(m.name, 'ARMATURE')
            am.object = m.object
            am.use_deform_preserve_volume = True

    bm = bmesh.new()
    bm.from_mesh(sw_obj.data)
    collar_top = neck_z + 0.04
    v_bottom_z = neck_z - 0.16
    v_half_w = 0.095

    bg = human_obj.vertex_groups.get("body")
    bg_idx = bg.index if bg is not None else -1

    to_delete = []
    for i, v in enumerate(bm.verts):
        co = eval_mesh.vertices[i].co
        if bg_idx >= 0 and not any(g.group == bg_idx for g in human_obj.data.vertices[i].groups):
            to_delete.append(v)
            continue
        # Keep shoulders & sleeves; cut above trapezius, center neck hole, and below hip
        if co.z > collar_top or co.z < hip_z:
            to_delete.append(v)
            continue
        if abs(co.x) < 0.068 and co.z > neck_z - 0.015:
            to_delete.append(v)
            continue
        if co.y < -0.01 and co.z > v_bottom_z:
            t = (co.z - v_bottom_z) / max(1e-5, (neck_z - 0.015) - v_bottom_z)
            if abs(co.x) < v_half_w * min(1.0, t):
                to_delete.append(v)

    bmesh.ops.delete(bm, geom=to_delete, context='VERTS')
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(sw_obj.data)
    bm.free()

    sw_obj.data.materials.clear()
    sw_obj.data.materials.append(sweater_mat)
    for p in sw_obj.data.polygons:
        p.use_smooth = True

    sol = sw_obj.modifiers.new("EXCAD_SweaterThickness", 'SOLIDIFY')
    sol.thickness = 0.010
    sol.offset = 1.0

    sub = sw_obj.modifiers.new("EXCAD_SweaterSubsurf", 'SUBSURF')
    sub.levels = 1
    sub.render_levels = 2
    return sw_obj


def _create_necklace(human_obj, collar_z: float, front_y: float):
    silver = _pbr_mat("EXCAD_SilverPendant", (0.90, 0.91, 0.93), roughness=0.16, metallic=0.96)
    cd = bpy.data.curves.new("EXCAD_NecklaceChain", type='CURVE')
    cd.dimensions = '3D'
    cd.bevel_depth = 0.0016
    cd.bevel_resolution = 2

    pts = [
        (-0.052, front_y * 0.45, collar_z + 0.018),
        (-0.040, front_y * 0.88, collar_z - 0.025),
        (0.0, front_y * 1.02, collar_z - 0.072),
        (0.040, front_y * 0.88, collar_z - 0.025),
        (0.052, front_y * 0.45, collar_z + 0.018),
    ]
    sp = cd.splines.new('NURBS')
    sp.points.add(len(pts) - 1)
    for i, pco in enumerate(pts):
        sp.points[i].co = (pco[0], pco[1], pco[2], 1.0)
    sp.use_endpoint_u = True

    chain = bpy.data.objects.new("EXCAD_Necklace", cd)
    bpy.context.collection.objects.link(chain)
    chain.data.materials.append(silver)
    chain.parent = human_obj

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=0.005)
    pm = bpy.data.meshes.new("EXCAD_PendantMesh")
    bm.to_mesh(pm)
    bm.free()
    pendant = bpy.data.objects.new("EXCAD_Pendant", pm)
    bpy.context.collection.objects.link(pendant)
    pendant.location = (0.0, front_y * 1.03, collar_z - 0.075)
    pendant.data.materials.append(silver)
    pendant.parent = human_obj
    return [chain, pendant]


def stylize_portrait_character(human_obj, add_hair=True, add_sweater=True, add_necklace=True, add_eyes=True, add_brows=True):
    if human_obj is None or human_obj.type != 'MESH':
        raise ValueError("Target must be a human MESH object")

    bpy.context.view_layer.update()
    _tune_fair_skin(human_obj)

    # Temporarily disable topology-changing modifiers (SUBSURF, MASK) so eval_mesh has 1:1 vertex indices
    toggled_mods = []
    for m in human_obj.modifiers:
        if m.type in ('SUBSURF', 'MASK') and m.show_viewport:
            m.show_viewport = False
            toggled_mods.append(m)

    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = human_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()

    try:
        bg = human_obj.vertex_groups.get("body")
        if bg is not None:
            zs = [eval_mesh.vertices[i].co.z for i, v in enumerate(human_obj.data.vertices)
                  if any(g.group == bg.index for g in v.groups)]
        else:
            zs = [v.co.z for v in eval_mesh.vertices]
        min_z, max_z = min(zs), max(zs)
        h = max(0.5, max_z - min_z)

        l_eye = _vgroup_center(human_obj, eval_mesh, ("helper-l-eye", "l-eye"))
        r_eye = _vgroup_center(human_obj, eval_mesh, ("helper-r-eye", "r-eye"))
        if l_eye is None or r_eye is None:
            l_eye = Vector((0.031, -0.120, max_z - 0.115))
            r_eye = Vector((-0.031, -0.120, max_z - 0.115))

        eye_mid = 0.5 * (l_eye + r_eye)
        head_center = Vector((0.0, eye_mid.y + 0.085, eye_mid.z + 0.012))
        head_rz = max(0.085, max_z - head_center.z)

        rig = human_obj.find_armature()
        if rig is not None and "neck_01" in rig.data.bones:
            neck_z = rig.data.bones["neck_01"].head_local.z + 0.045
        else:
            neck_z = min_z + 0.77 * h
        if rig is not None and "pelvis" in rig.data.bones:
            hip_z = rig.data.bones["pelvis"].head_local.z
        else:
            hip_z = min_z + 0.48 * h

        collar_verts = [
            v.co.y for v in eval_mesh.vertices
            if abs(v.co.z - (neck_z - 0.02)) < 0.03 and abs(v.co.x) < 0.06
        ]
        front_y = min(collar_verts) - 0.008 if collar_verts else -0.095

        created = []
        if add_eyes:
            created.extend(_create_eyes(human_obj, l_eye, r_eye))
            if add_brows:
                created.extend(_create_eyebrows(human_obj, l_eye, r_eye))
        if add_hair:
            created.extend(_create_hair(human_obj, head_center, head_rx=0.086, head_ry=0.102, head_rz=head_rz))
        if add_sweater:
            created.append(_create_vneck_sweater(human_obj, eval_mesh, neck_z=neck_z, hip_z=hip_z))
        if add_necklace:
            created.extend(_create_necklace(human_obj, collar_z=neck_z - 0.02, front_y=front_y))
        return created
    finally:
        eval_obj.to_mesh_clear()
        for m in toggled_mods:
            m.show_viewport = True


def style_character(human, options):
    from . import wardrobe
    from .human_assets import equip_asset

    if human is None or human.type != "MESH":
        raise ValueError("Specify the MPFB basemesh to style")
    from .face_fit import basemesh_services

    basemesh_services(human)
    saved = json.loads(human.get("excad_portrait_settings", "{}"))
    settings = {**saved, **{k: options[k] for k in ("iris_color", "iris_size", "skin", "lip_tint") if k in options}}
    garments = options.get("clothes", [])
    if isinstance(garments, str):
        garments = [garments]
    garment_slots = {
        "TSHIRT": "top",
        "VNECK_SWEATER": "top",
        "HOODIE": "top",
        "JACKET": "outerwear",
        "PANTS": "bottom",
        "SKIRT": "bottom",
        "SHOES": "footwear",
    }
    for value in garments:
        if value not in garment_slots:
            raise ValueError(f"Unknown garment {value}")
    for name, allowed in (
        ("skin", wardrobe.SKINS),
        ("expression", wardrobe.EXPRESSIONS),
        ("hands", ("OPEN", "POINT", "FIST", "RELAXED")),
    ):
        if name in options and options[name] not in allowed:
            raise ValueError(f"Unknown {name} preset")
    hair = options.get("hair")
    if isinstance(hair, str) and hair not in wardrobe.HAIR:
        raise ValueError(f"Unknown hair style {hair}")
    if len({garment_slots[g] for g in garments}) != len(garments):
        raise ValueError("Choose one garment per clothing layer")
    if not 0.2 <= options.get("iris_size", 0.52) <= 0.75:
        raise ValueError("iris_size must be between 0.2 and 0.75")
    for name in ("iris_color", "lip_tint"):
        if name in options and (len(options[name]) != 3 or any(not 0 <= v <= 1 for v in options[name])):
            raise ValueError(f"{name} must be a linear RGB triple")
    previous = set(bpy.context.scene.objects)
    remove = options.get("remove", [])
    if isinstance(remove, str):
        remove = [remove]
    replaced = set(remove)
    if replaced - {*garment_slots.values(), "hair", "eyes", "lashes", "teeth"}:
        raise ValueError("Unknown character slot to remove")
    if "clothes" in options and not garments:
        replaced.update(garment_slots.values())
    replaced.update(k for k in ("eyes", "teeth", "lashes") if options.get(k) is False)
    if "hair" in options and hair is None:
        replaced.add("hair")
    created = []

    def store(slot, items):
        replaced.add(slot)
        for obj in items:
            obj["excad_slot"] = slot
        created.extend(items)

    try:
        if options.get("eyes") or "iris_color" in options or "iris_size" in options:
            store(
                "eyes",
                wardrobe.eyes(human, settings.get("iris_color", (0.10, 0.055, 0.024)), settings.get("iris_size", 0.52)),
            )
        if options.get("lashes"):
            store("lashes", [equip_asset(human, "eyelashes02")])
        if options.get("teeth"):
            store("teeth", [equip_asset(human, "teeth_base")])
        if isinstance(hair, str):
            store("hair", wardrobe.hair_style(human, hair))
        for item in garments:
            store(garment_slots[item], [wardrobe.garment(human, item)])
        if options.get("skin") or "lip_tint" in options:
            wardrobe.skin(human, settings.get("skin", "WARM_BEIGE"), settings.get("lip_tint", (0.40, 0.105, 0.10)))
        if "expression" in options:
            wardrobe.expression(human, options["expression"])
        bones = wardrobe.hand_pose(human, options["hands"]) if "hands" in options else 0
        old = [o for o in previous if o.get("excad_human") == human and o.get("excad_slot") in replaced]
        if "clothes" in options or "hair" in options or replaced & {*garment_slots.values(), "hair"}:
            wardrobe.refit_layers(human, old)
    except Exception:
        wardrobe.remove_wearables(human, set(bpy.context.scene.objects) - previous)
        raise
    wardrobe.remove_wearables(human, old)
    human["excad_portrait_settings"] = json.dumps(settings, separators=(",", ":"))
    human["excad_character"] = True
    return {"object": human.name, "added": [o.name for o in created], "posed_fingers": bones}
