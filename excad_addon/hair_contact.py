# SPDX-License-Identifier: GPL-2.0-or-later
"""Keep draped locks outside moving shoulders while preserving the head bone's weights."""

import bpy
from mathutils import Vector

GROUP = "EXCAD hair over shoulders"


def _contact_nodes():
    for tree in bpy.data.node_groups:
        if tree.get("excad_hair_contact") == 1:
            return tree
    tree = bpy.data.node_groups.new(GROUP, "GeometryNodeTree")
    tree["excad_hair_contact"] = 1
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    for name, kind in (
        ("Geometry", "Geometry"),
        ("Top", "Object"),
        ("Outerwear", "Object"),
        ("Back projection", "Vector"),
        ("Clearance", "Float"),
    ):
        tree.interface.new_socket(name=name, in_out="INPUT", socket_type="NodeSocket" + kind)

    def node(kind, x, y):
        item = tree.nodes.new(kind)
        item.location = (x, y)
        return item

    def link(out, target):
        tree.links.new(out, target)

    source = node("NodeGroupInput", 0, 0)
    target = node("NodeGroupOutput", 1650, 250)
    surfaces = node("GeometryNodeJoinGeometry", 450, 650)
    for i, slot in enumerate(("Top", "Outerwear")):
        obj = node("GeometryNodeObjectInfo", 210, 850 - i * 230)
        obj.transform_space = "RELATIVE"
        link(source.outputs[slot], obj.inputs["Object"])
        link(obj.outputs["Geometry"], surfaces.inputs["Geometry"])

    def vector(operation, x, y, first, second=None):
        item = node("ShaderNodeVectorMath", x, y)
        item.operation = operation
        link(first, item.inputs[0])
        if second is not None:
            link(second, item.inputs["Scale"] if operation == "SCALE" else item.inputs[1])
        return item.outputs["Value" if operation == "LENGTH" else "Vector"]

    def scalar(operation, x, y, a, b):
        item = node("ShaderNodeMath", x, y)
        item.operation = operation
        link(a, item.inputs[0])
        if isinstance(b, (int, float)):
            item.inputs[1].default_value = b
        else:
            link(b, item.inputs[1])
        return item.outputs[0]

    position = node("GeometryNodeInputPosition", 180, 300).outputs[0]
    back = source.outputs["Back projection"]
    start = vector("ADD", 445, 350, position, back)
    unit = vector("NORMALIZE", 430, -180, back)
    reach = vector("LENGTH", 430, -400, back)
    reverse = node("ShaderNodeVectorMath", 445, 130)
    reverse.operation = "SCALE"
    link(back, reverse.inputs[0])
    reverse.inputs["Scale"].default_value = -1
    ray = node("GeometryNodeRaycast", 680, 640)
    link(surfaces.outputs[0], ray.inputs["Target Geometry"])
    link(start, ray.inputs["Source Position"])
    link(reverse.outputs[0], ray.inputs["Ray Direction"])
    link(
        scalar("ADD", 660, -150, reach, source.outputs["Clearance"]),
        ray.inputs["Ray Length"],
    )
    shift = scalar("SUBTRACT", 865, 165, reach, ray.outputs["Hit Distance"])
    clearance = scalar("ADD", 1030, 115, shift, source.outputs["Clearance"])
    influence = node("GeometryNodeInputNamedAttribute", 865, -140)
    influence.data_type = "FLOAT"
    influence.inputs["Name"].default_value = GROUP
    distance = scalar("MULTIPLY", 1200, 100, clearance, influence.outputs["Attribute"])
    offset = vector("SCALE", 1380, 40, unit, distance)
    move = node("GeometryNodeSetPosition", 1450, 420)
    link(source.outputs["Geometry"], move.inputs["Geometry"])
    link(ray.outputs["Is Hit"], move.inputs["Selection"])
    link(offset, move.inputs["Offset"])
    link(move.outputs["Geometry"], target.inputs["Geometry"])
    return tree


def fit(hair, obstacles, points, to_human, neck, span):
    tree = _contact_nodes()
    modifier = next((m for m in hair.modifiers if m.type == "NODES" and m.node_group == tree), None)
    if modifier is None:
        modifier = hair.modifiers.new(GROUP, "NODES")
        modifier.node_group = tree
    group = hair.vertex_groups.get(GROUP) or hair.vertex_groups.new(name=GROUP)
    for i, local in enumerate(points):
        p = to_human @ local
        below = min(1, max(0, (neck.z + 0.16 * span - p.z) / (0.08 * span)))
        behind = min(1, max(0, (p.y - neck.y + 0.13 * span) / (0.10 * span)))
        group.add([i], below * behind, "REPLACE")
    projection = to_human.to_3x3().inverted() @ Vector((0, 0.34 * span, 0))
    values = {
        "Top": next((o for o in obstacles if o.get("excad_slot") == "top"), None),
        "Outerwear": next((o for o in obstacles if o.get("excad_slot") == "outerwear"), None),
        "Back projection": projection,
        "Clearance": projection.length * 0.06,
    }
    for socket in tree.interface.items_tree:
        if socket.in_out == "INPUT" and socket.name in values:
            if hasattr(modifier, "properties"):
                getattr(modifier.properties.inputs, socket.identifier).value = values[socket.name]
            else:
                modifier[socket.identifier] = values[socket.name]
    hair.update_tag()


def clear(hair):
    for mod in list(hair.modifiers):
        if mod.type == "NODES" and mod.node_group and mod.node_group.get("excad_hair_contact") == 1:
            hair.modifiers.remove(mod)
