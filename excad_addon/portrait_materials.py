# SPDX-License-Identifier: GPL-2.0-or-later
"""Surface detail for fitted MPFB hair and tailored clothes."""

import math

import numpy as np
from mathutils.kdtree import KDTree


def hair_fibers(obj):
    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes:
            continue
        if mat.users > 1:
            slot.material = mat = mat.copy()
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        if bsdf is None or not obj.data.uv_layers.active:
            continue
        bsdf.inputs["Roughness"].default_value = 0.48
        bsdf.inputs["Specular IOR Level"].default_value = 0.17
        bsdf.inputs["Anisotropic"].default_value = 0.55
        tangent = nodes.get("Hair direction") or nodes.new("ShaderNodeTangent")
        tangent.name, tangent.direction_type = "Hair direction", "UV_MAP"
        tangent.uv_map = obj.data.uv_layers.active.name
        mat.node_tree.links.new(tangent.outputs["Tangent"], bsdf.inputs["Tangent"])


def skin_detail(human, material):
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    tex = nodes.get("Skin coordinates") or nodes.new("ShaderNodeTexCoord")
    tex.name = "Skin coordinates"
    size = nodes.get("Skin texture scale") or nodes.new("ShaderNodeVectorMath")
    size.name, size.operation = "Skin texture scale", "MULTIPLY"
    # Generated coordinates follow the skin through poses. Restore their metric
    # aspect so pores keep the same scale across the cheek, nose and forehead.
    size.inputs[1].default_value = [
        2 * abs(s * k) for s, k in zip(human.data.texspace_size, human.matrix_world.to_scale())
    ]
    noise = nodes.get("Skin pores") or nodes.new("ShaderNodeTexNoise")
    noise.name = "Skin pores"
    noise.inputs["Scale"].default_value = 1800
    noise.inputs["Detail"].default_value = 1.6
    noise.inputs["Roughness"].default_value = 0.55
    bump = nodes.get("Skin relief") or nodes.new("ShaderNodeBump")
    bump.name = "Skin relief"
    bump.inputs["Strength"].default_value = 0.15
    bump.inputs["Distance"].default_value = 0.00018
    rough = nodes.get("Skin roughness") or nodes.new("ShaderNodeMapRange")
    rough.name = "Skin roughness"
    rough.inputs["To Min"].default_value, rough.inputs["To Max"].default_value = 0.38, 0.54
    links.new(tex.outputs["Generated"], size.inputs[0])
    links.new(size.outputs[0], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(noise.outputs["Fac"], rough.inputs["Value"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])


def finish_fabric(obj, style, color, height, border_edges):
    mesh = obj.data
    samples = []
    for a, b in border_edges:
        start, end = mesh.vertices[a].co, mesh.vertices[b].co
        steps = max(1, math.ceil((end - start).length / (height * 0.002)))
        samples.extend(start.lerp(end, i / steps) for i in range(steps + 1))
    tree = KDTree(len(samples))
    for i, point in enumerate(samples):
        tree.insert(point, i)
    tree.balance()
    distance = np.fromiter((tree.find(v.co)[2] / height for v in mesh.vertices), np.float32)
    attr = mesh.color_attributes.new("EXCAD Hem distance", "FLOAT_COLOR", "POINT")
    rgba = np.ones((len(distance), 4), np.float32)
    rgba[:, :3] = distance[:, None]
    attr.data.foreach_set("color", rgba.ravel())

    material = mesh.materials[0]
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    seam = nodes.get("Distance to stitching") or nodes.new("ShaderNodeVertexColor")
    seam.name, seam.layer_name = "Distance to stitching", attr.name
    width = nodes.get("Hem width") or nodes.new("ShaderNodeMapRange")
    width.name, width.clamp = "Hem width", True
    width.inputs["From Min"].default_value = 0.004
    width.inputs["From Max"].default_value = 0.006
    width.inputs["To Min"].default_value = 1
    width.inputs["To Max"].default_value = 0
    mix = nodes.get("Hem tint") or nodes.new("ShaderNodeMixRGB")
    mix.name = "Hem tint"
    mix.inputs[1].default_value = (*color, 1)
    mix.inputs[2].default_value = (*(c * 0.62 for c in color), 1)
    links.new(seam.outputs["Color"], width.inputs["Value"])
    links.new(width.outputs["Result"], mix.inputs[0])
    links.new(mix.outputs[0], bsdf.inputs["Base Color"])

    knit = style == "VNECK_SWEATER"
    fiber = nodes.get("Fabric grain") or nodes.new("ShaderNodeTexWave" if knit else "ShaderNodeTexNoise")
    fiber.name = "Fabric grain"
    fiber.inputs["Scale"].default_value = 86 if knit else 820
    fiber.inputs["Detail"].default_value = 2
    if knit:
        fiber.bands_direction = "X"
        fiber.inputs["Distortion"].default_value = 1.5
    bump = nodes.get("Fabric weave") or nodes.new("ShaderNodeBump")
    bump.name = "Fabric weave"
    bump.inputs["Strength"].default_value = 0.16
    bump.inputs["Distance"].default_value = 0.00026
    links.new(fiber.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.24
