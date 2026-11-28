# SPDX-License-Identifier: GPL-2.0-or-later
"""Capture registered RGBA images and 16-bit metric depth with Blender's render engine."""

import json
from pathlib import Path
import tempfile

import bpy

from .polyhedron_engine import _configure_camera, _is_cage, _validate_frame, _view_matrix, get_polyhedron_faces


def write_manifest(directory, poly_type, size, center, faces, depth=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"poly_type": poly_type, "size": float(size), "center": list(center), "views": faces}
    path = directory / "views_manifest.json"
    if depth is not None:
        manifest.update(excad_capture=1, write_depth=bool(depth))
    elif previous := _owned_images(path):
        # Updating camera guides can precede a recapture in the same directory.
        manifest["previous_images"] = sorted(previous)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, mode="w", encoding="utf8", delete=False) as stream:
            pending = Path(stream.name)
            json.dump(manifest, stream, indent=2)
        pending.replace(path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
    return path


def _depth_output(scene, group, layer, size, directory):
    layer.use_pass_z = True
    scene.compositing_node_group = group
    group.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    render = group.nodes.new("CompositorNodeRLayers")
    render.scene, render.layer = scene, layer.name
    output = group.nodes.new("NodeGroupOutput")
    group.links.new(render.outputs["Image"], output.inputs["Image"])
    depth = group.nodes.new("ShaderNodeMath")
    depth.operation = "MULTIPLY_ADD"
    depth.use_clamp = True
    # The camera is 1.15 box widths from the center, so the near face is 0.65 widths from the camera.
    depth.inputs[1].default_value, depth.inputs[2].default_value = 1 / size, -0.65
    group.links.new(render.outputs["Depth"], depth.inputs[0])
    file = group.nodes.new("CompositorNodeOutputFile")
    file.directory = str(directory)
    file.save_as_render = False
    file.format.media_type = "IMAGE"
    file.format.file_format, file.format.color_mode, file.format.color_depth = "PNG", "BW", "16"
    file.file_output_items.new("FLOAT", "_depth")
    group.links.new(depth.outputs[0], file.inputs["_depth"])
    return file


def _render(directory, faces, center, size, resolution, write_depth):
    source = bpy.context.scene
    scene = source.copy()
    camera = data = group = None
    legacy = []
    try:
        scene.name = "EXCAD view capture"
        render = scene.render
        render.resolution_x = render.resolution_y = resolution
        render.resolution_percentage = 100
        render.pixel_aspect_x = render.pixel_aspect_y = 1
        render.use_border = render.use_multiview = render.use_sequencer = render.use_motion_blur = False
        render.film_transparent = render.use_file_extension = True
        render.image_settings.media_type = "IMAGE"
        render.image_settings.file_format, render.image_settings.color_mode = "PNG", "RGBA"
        render.image_settings.color_depth = "8"
        render.use_compositing = bool(write_depth)
        scene.compositing_node_group = None
        layer = scene.view_layers[bpy.context.view_layer.name]
        for candidate in scene.view_layers:
            candidate.use = candidate == layer
        for marker in scene.timeline_markers:
            marker.camera = None
        # Earlier EX-CAD releases left a renderable solid behind their wire guides.
        legacy = [o for o in scene.objects if not o.hide_render and _is_cage(o)]
        for obj in legacy:
            obj.hide_render = True
        data = _configure_camera(bpy.data.cameras.new("EXCAD capture camera"), size)
        camera = bpy.data.objects.new(data.name, data)
        scene.collection.objects.link(camera)
        scene.camera = camera
        file = None
        if write_depth:
            group = bpy.data.node_groups.new("EXCAD capture passes", "CompositorNodeTree")
            file = _depth_output(scene, group, layer, size, directory)
        for face in faces:
            target = directory / face["filename"]
            camera.matrix_world = _view_matrix(face, center, size)
            render.filepath = str(target)
            if file:
                file.file_name = target.stem
            if "FINISHED" not in bpy.ops.render.render(scene=scene.name, write_still=True):
                raise RuntimeError(f"Could not render {target.name}")
            if not target.is_file() or (file and not target.with_name(target.stem + "_depth.png").is_file()):
                raise RuntimeError(f"Blender did not write all requested passes for {target.name}")
    finally:
        for obj in legacy:
            obj.hide_render = False
        bpy.data.scenes.remove(scene)
        if group:
            bpy.data.node_groups.remove(group)
        if camera:
            bpy.data.objects.remove(camera, do_unlink=True)
        if data:
            bpy.data.cameras.remove(data)


def _owned_images(manifest):
    if not manifest.is_file():
        return set()
    try:
        old = json.loads(manifest.read_text(encoding="utf8"))
        if not isinstance(old, dict):
            return set()
        if old.get("excad_capture") != 1:
            allowed = {f"face_{i:02d}{suffix}.png" for i in range(20) for suffix in ("", "_depth")}
            return allowed.intersection(old.get("previous_images", ()))
        faces = get_polyhedron_faces(old["poly_type"])
    except (ValueError, KeyError, TypeError):
        return set()
    names = {face["filename"] for face in faces}
    if old.get("write_depth"):
        names.update(Path(name).stem + "_depth.png" for name in tuple(names))
    return names


def _publish(staging, directory):
    manifest = "views_manifest.json"
    new = {p.name for p in staging.iterdir() if p.is_file()}
    names = sorted((new | _owned_images(directory / manifest)) - {manifest}) + [manifest]
    backup = staging / "previous"
    backup.mkdir()
    saved, installed = [], []
    try:
        for name in names:
            dest = directory / name
            if dest.exists():
                if not dest.is_file():
                    raise IsADirectoryError(dest)
                dest.replace(backup / name)
                saved.append(name)
            if name in new:
                (staging / name).replace(dest)
                installed.append(name)
    except Exception:
        for name in reversed(installed):
            (directory / name).unlink()
        for name in saved:
            (backup / name).replace(directory / name)
        raise


def capture(poly_type, output_dir, size, center, resolution, write_depth=False):
    _validate_frame(size, center)
    if not output_dir or not isinstance(resolution, int) or not 4 <= resolution <= 65536:
        raise ValueError("Provide a views folder and a supported Blender image resolution")
    faces = get_polyhedron_faces(poly_type)
    directory = Path(bpy.path.abspath(output_dir)).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory, prefix=".excad-capture-") as tmp:
        staging = Path(tmp)
        _render(staging, faces, center, size, resolution, write_depth)
        write_manifest(staging, poly_type, size, center, faces, depth=write_depth)
        _publish(staging, directory)
    return [str(directory / face["filename"]) for face in faces], str(directory / "views_manifest.json")
