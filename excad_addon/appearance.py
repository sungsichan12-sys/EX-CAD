# SPDX-License-Identifier: GPL-2.0-or-later
"""Semantic finishes and a three-point light rig framed from the subject's evaluated bounds."""

import math

import bpy
from mathutils import Matrix, Vector

from .scene_layout import bounds

FINISHES = {
    "BRUSHED_METAL": ((0.55, 0.59, 0.64), 0.28, 0.92),
    "CERAMIC": ((0.66, 0.76, 0.72), 0.22, 0.0),
    "PLASTIC": ((0.13, 0.23, 0.31), 0.38, 0.0),
    "RUBBER": ((0.025, 0.029, 0.032), 0.74, 0.0),
    "WOOD": ((0.36, 0.20, 0.09), 0.55, 0.0),
    "FABRIC": ((0.22, 0.28, 0.32), 0.86, 0.0),
}


def set_material(obj, preset, color=None, roughness=None, metallic=None):
    obj = bpy.context.scene.objects[obj]
    renderable = {"MESH", "CURVE", "SURFACE", "FONT"}
    targets = (
        [obj] if obj.type in renderable else [child for child in obj.children_recursive if child.type in renderable]
    )
    if not targets:
        raise ValueError("The material target needs renderable geometry")
    base, r, m = FINISHES[preset]
    color = list(color if color is not None else base)
    if len(color) != 3 or any(not 0 <= v <= 1 for v in color):
        raise ValueError("Use three linear RGB channels between 0 and 1")
    roughness = r if roughness is None else roughness
    metallic = m if metallic is None else metallic
    if not 0 <= roughness <= 1 or not 0 <= metallic <= 1:
        raise ValueError("Roughness and metallic must be between 0 and 1")
    assigned = [_finish(target, preset, color, roughness, metallic) for target in targets]
    return {"material": assigned[0]} if len(assigned) == 1 else {"materials": assigned}


def _finish(obj, preset, color, roughness, metallic):
    if obj.data.users > 1:
        obj.data = obj.data.copy()
    slots = obj.data.materials
    mat = slots[0] if slots and slots[0] and slots[0].get("excad_finish") and slots[0].users == 1 else None
    if mat is None:
        mat = bpy.data.materials.new(obj.name + " finish")
    mat["excad_finish"] = preset
    mat.use_nodes = True
    mat.diffuse_color = (*color, 1)
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    slots.clear()
    slots.append(mat)
    obj.material_slots[0].link = "DATA"
    return mat.name


def _head_frame(subject):
    from .wardrobe import _eye_centers, rest_mesh

    eyes = [o for o in bpy.context.scene.objects if o.get("excad_human") == subject and "excad_eye_center" in o]
    if len(eyes) == 2:
        centers = [Vector(o["excad_eye_center"]) for o in eyes]
    else:
        if subject.type != "MESH" or not any(name in subject.vertex_groups for name in ("helper-l-eye", "l-eye")):
            raise ValueError("HEADSHOT requires a character with fitted eyes or MPFB eye landmarks")
        with rest_mesh(subject) as mesh:
            centers = _eye_centers(subject, mesh)
    frame = subject.matrix_world.copy()
    rig = subject.find_armature()
    if rig and rig.data.pose_position != "REST":
        head = next((b for b in rig.pose.bones if b.name.lower().split(":")[-1] in {"head", "mixamorighead"}), None)
        if head:
            frame = (
                rig.matrix_world @ head.matrix @ head.bone.matrix_local.inverted() @ rig.matrix_world.inverted() @ frame
            )
    left, right = [frame @ center for center in centers]
    span = (left - right).length
    if span < 1e-6:
        raise ValueError("Eye landmarks are coincident")
    orient = frame.to_3x3()
    front = (orient @ Vector((0, -1, 0))).normalized()
    up = (orient @ Vector((0, 0, 1))).normalized()
    rightward = front.cross(up) * -1
    rightward.normalize()
    up = front.cross(rightward).normalized()
    basis = Matrix((rightward, -front, up)).transposed()
    return (left + right) / 2, basis, span


def _studio_part(subject, role):
    camera = role == "Camera"
    kind, marker = ("CAMERA", "excad_headshot") if camera else ("LIGHT", "excad_light")
    for obj in bpy.context.scene.objects:
        if obj.type != kind or obj.get(marker) != role:
            continue
        owner = obj.get("excad_studio_target")
        if owner == subject or (owner is None and obj.get("excad_subject") == subject.name):
            break
    else:
        name = subject.name + " " + role
        data = bpy.data.cameras.new(name) if camera else bpy.data.lights.new(name, "AREA")
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
    if obj.data.users > 1:
        obj.data = obj.data.copy()
    obj["excad_subject"], obj["excad_studio_target"], obj[marker] = subject.name, subject, role
    return obj


def light_studio(target, style="PRODUCT", power=1.0):
    if style not in {"PRODUCT", "PORTRAIT", "SOFT", "HEADSHOT"} or not math.isfinite(power) or power <= 0:
        raise ValueError("Use PRODUCT, PORTRAIT, SOFT or HEADSHOT with positive power")
    subject = bpy.context.scene.objects[target]
    if style == "HEADSHOT":
        bpy.context.view_layer.update()
        center, basis, extent = _head_frame(subject)
        setup = (
            ("Key", (-5.5, -7.4, 5.2), 10.5 / 0.06**2, 5.2, (1.0, 0.91, 0.82)),
            ("Fill", (5.6, -7, 2.1), 3.0 / 0.06**2, 6.8, (0.81, 0.9, 1.0)),
            ("Rim", (3.5, 5.8, 5.5), 5.0 / 0.06**2, 4.0, (1.0, 0.92, 0.83)),
        )
    else:
        bb = bounds(subject)
        center, basis = Vector(bb.mean(0)), Matrix.Identity(3)
        extent = max(float(max(bb[1] - bb[0])), 0.01)
        setup = (
            ("Key", (-1.5, -2, 2.4), 420, 1.25, (1, 0.86, 0.74) if style == "PORTRAIT" else (1, 1, 1)),
            ("Fill", (2, -1, 1.3), 220, 1.6, (1, 1, 1)),
            ("Rim", (0.8, 1.5, 1.8), 330, 0.9, (1, 1, 1)),
        )
    lights = []
    for role, direction, energy, diameter, color in setup:
        lamp = _studio_part(subject, role)
        lamp.location = center + basis @ Vector(direction) * extent
        lamp.rotation_euler = (center - lamp.location).to_track_quat("-Z", "Y").to_euler()
        lamp.data.shape = "DISK"
        lamp.data.size = extent * diameter * (1.8 if style == "SOFT" else 1)
        lamp.data.energy = energy * extent**2 * power
        lamp.data.color = color
        lights.append(lamp.name)
    if style == "HEADSHOT":
        focus = center + basis @ Vector((0, 0, -1.40)) * extent
        camera = _studio_part(subject, "Camera")
        camera.location = focus + basis @ Vector((3.6, -21.5, 0.8)) * extent
        back = (camera.location - focus).normalized()
        right = basis.col[2].cross(back).normalized()
        camera.rotation_euler = Matrix((right, back.cross(right), back)).transposed().to_euler()
        camera.data.type, camera.data.lens, camera.data.sensor_fit, camera.data.sensor_height = (
            "PERSP",
            86,
            "VERTICAL",
            32,
        )
        camera.data.clip_start = max(0.0001, extent * 0.08)
        camera.data.clip_end = max(100.0, extent * 100)
        bpy.context.scene.camera = camera
        return {"lights": lights, "camera": camera.name}
    return {"lights": lights}
