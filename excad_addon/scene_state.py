# SPDX-License-Identifier: GPL-2.0-or-later
"""Compact content fingerprints for evaluated scene geometry, transforms, and materials."""

import hashlib
import json

import bpy
import numpy as np

from .scene_mesh import evaluated_mesh, resolved_materials


def _feed(h, value):
    h.update(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def _pixels(image):
    import os

    path = (
        os.path.normcase(os.path.abspath(bpy.path.abspath(image.filepath, library=image.library)))
        if image.filepath
        else ""
    )
    h = hashlib.blake2s(digest_size=8)
    _feed(h, [path, list(image.size), image.colorspace_settings.name, image.alpha_mode])
    if image.is_dirty or image.source == "GENERATED":
        data = np.empty(len(image.pixels), np.float32)
        image.pixels.foreach_get(data)
        h.update(data.tobytes())
    elif image.packed_file:
        h.update(image.packed_file.data)
    elif path:
        try:
            stat = os.stat(path)
            _feed(h, [stat.st_size, stat.st_mtime_ns])
        except OSError:
            pass
    return h.hexdigest()


def _shader(tree, images, visited):
    h = hashlib.blake2s(digest_size=8)
    if tree in visited:
        return tree.name
    for node in tree.nodes:
        inputs = []
        for inp in node.inputs:
            if inp.is_linked:
                inputs.append([(link.from_node.name, link.from_socket.identifier) for link in inp.links])
            elif hasattr(inp, "default_value"):
                value = inp.default_value
                if isinstance(value, bpy.types.ID):
                    value = value.name
                elif not isinstance(value, (str, int, float, bool, type(None))):
                    value = list(value)
                inputs.append(value)
        settings = {
            key: getattr(node, key)
            for key in (
                "operation",
                "blend_type",
                "interpolation",
                "projection",
                "extension",
                "layer_name",
                "uv_map",
                "attribute_name",
                "distribution",
                "subsurface_method",
            )
            if hasattr(node, key)
        }
        ramp = getattr(node, "color_ramp", None)
        if ramp:
            settings["ramp"] = [ramp.interpolation, [(entry.position, list(entry.color)) for entry in ramp.elements]]
        texture = getattr(node, "image", None)
        if texture:
            if texture not in images:
                images[texture] = _pixels(texture)
            settings["image"] = images[texture]
        group = getattr(node, "node_tree", None)
        if group:
            settings["group"] = _shader(group, images, visited | {tree})
        _feed(h, [node.name, node.bl_idname, node.mute, inputs, settings])
    return h.hexdigest()


def _material(mat, images):
    h = hashlib.blake2s(digest_size=8)
    _feed(h, [list(mat.diffuse_color), mat.roughness, mat.metallic])
    if mat.use_nodes and mat.node_tree:
        _feed(h, _shader(mat.node_tree, images, set()))
    return h.hexdigest()


def rows(include_all=False):
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    result = []
    materials = {}
    images = {}
    for obj in sorted(bpy.context.scene.objects, key=lambda o: o.name):
        if (not include_all and obj.type in {"CAMERA", "LIGHT"}
                and not obj.get("excad_light") and not obj.get("excad_headshot") and "excad_view_index" not in obj):
            continue
        ev = obj.evaluated_get(graph)
        h = hashlib.blake2s(digest_size=8)
        h.update(np.asarray(ev.matrix_world, dtype="<f4").tobytes())
        _feed(
            h,
            [
                obj.parent.name if obj.parent else "",
                obj.hide_viewport,
                obj.hide_render,
                obj.visible_get(),
                obj.get("excad_tree", ""),
                list(obj.color),
            ],
        )
        verts = faces = 0
        if "excad_sdf_recipe" in obj:
            _feed(h, obj["excad_sdf_recipe"])
        if ev.type in {"MESH", "CURVE", "SURFACE", "FONT", "META"}:
            with evaluated_mesh(ev) as mesh:
                if mesh:
                    verts, faces = len(mesh.vertices), len(mesh.polygons)
                    for items, prop, dtype, n in (
                        (mesh.vertices, "co", np.float32, 3),
                        (mesh.edges, "vertices", np.int32, 2),
                        (mesh.loops, "vertex_index", np.int32, 1),
                        (mesh.polygons, "loop_start", np.int32, 1),
                        (mesh.polygons, "material_index", np.int32, 1),
                    ):
                        data = np.empty(len(items) * n, dtype=dtype)
                        items.foreach_get(prop, data)
                        h.update(data.tobytes())
                    for mat in resolved_materials(mesh, ev):
                        if mat:
                            if mat.name not in materials:
                                materials[mat.name] = _material(mat, images)
                            _feed(h, [mat.name, materials[mat.name]])
                    smooth = np.empty(faces, np.bool_)
                    mesh.polygons.foreach_get("use_smooth", smooth)
                    h.update(smooth.tobytes())
                    for layer in mesh.uv_layers:
                        uv = np.empty(len(layer.uv) * 2, np.float32)
                        layer.uv.foreach_get("vector", uv)
                        _feed(h, [layer.name, layer.active_render])
                        h.update(uv.tobytes())
                    for attr in mesh.color_attributes:
                        rgba = np.empty(len(attr.data) * 4, np.float32)
                        attr.data.foreach_get("color", rgba)
                        _feed(h, [attr.name, attr.domain, attr.data_type])
                        h.update(rgba.tobytes())
        elif ev.type == "ARMATURE":
            for b in ev.pose.bones:
                h.update(np.asarray(b.matrix, dtype="<f4").tobytes())
        elif ev.type == "LIGHT":
            light = ev.data
            _feed(h, [light.type, light.energy, list(light.color)])
            if light.type == "AREA":
                _feed(h, [light.shape, light.size, light.size_y, light.spread])
        elif ev.type == "CAMERA":
            camera = ev.data
            _feed(h, [camera.type, camera.lens, camera.ortho_scale, camera.clip_start, camera.clip_end,
                      camera.shift_x, camera.shift_y, camera.sensor_fit, camera.sensor_width, camera.sensor_height,
                      obj == bpy.context.scene.camera])
        result.append(
            [
                obj.name,
                obj.type[0],
                verts,
                faces,
                [round(v, 3) for v in ev.matrix_world.translation],
                [round(v, 3) for v in ev.dimensions],
                h.hexdigest(),
            ]
        )
    scene = bpy.context.scene
    sketches = {
        key: hashlib.blake2s(raw.encode(), digest_size=6).hexdigest()
        for key, raw in scene.get("excad_sketches", {}).items()
    }
    state = {"active": scene.get("excad_active_sketch", "Sketch"), "profiles": sketches}
    return result, state
