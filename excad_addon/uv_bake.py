# SPDX-License-Identifier: GPL-2.0-or-later
"""Bake multi-view image detail into a Smart UV atlas, independently of vertex density."""

from pathlib import Path
import tempfile

import bpy
import numpy as np

from .inspection import _png
from .image_colors import linear_to_srgb
from .polyhedron_engine import _load_rgba_and_mask, get_polyhedron_faces
from .view_fusion import image_coordinates


def read_depth(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        image.colorspace_settings.name = "Non-Color"
        w, h = image.size
        pixels = np.empty((h, w, 4), np.float32)
        image.pixels.foreach_get(pixels.ravel())
        return pixels[:, :, 0].copy()
    finally:
        bpy.data.images.remove(image)


def smart_uv(obj):
    layer = obj.data.uv_layers.get("EXCAD Atlas")
    if layer:
        obj.data.uv_layers.active = layer
        return layer
    if bpy.context.mode != "OBJECT":
        raise ValueError("Switch to Object Mode before unwrapping")
    active = bpy.context.view_layer.objects.active
    selected = list(bpy.context.selected_objects)
    geometry = (obj.data.vertices, obj.data.edges, obj.data.polygons)
    selection = [np.empty(len(domain), dtype=np.bool_) for domain in geometry]
    for domain, flags in zip(geometry, selection):
        domain.foreach_get("select", flags)
    layer = obj.data.uv_layers.new(name="EXCAD Atlas")
    if layer is None:
        raise ValueError("The mesh has no free UV layer for the atlas")
    obj.data.uv_layers.active = layer
    try:
        for other in selected:
            other.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        if "FINISHED" not in bpy.ops.uv.smart_project(island_margin=0.012, scale_to_bounds=True):
            raise ValueError("Smart UV projection could not unwrap the mesh")
    finally:
        if obj.mode == "EDIT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for domain, flags in zip((obj.data.vertices, obj.data.edges, obj.data.polygons), selection):
            domain.foreach_set("select", flags)
        obj.select_set(False)
        for other in selected:
            other.select_set(True)
        bpy.context.view_layer.objects.active = active
    return obj.data.uv_layers["EXCAD Atlas"]


def rasterize(uv, loops, size):
    ids = np.full((size, size), -1, np.int32)
    weights = np.zeros((size, size, 2), np.float32)
    for i, (a, b, c) in enumerate(uv[loops] * size - 0.5):
        lo = np.maximum(0, np.ceil(np.minimum(np.minimum(a, b), c)).astype(int))
        hi = np.minimum(size - 1, np.floor(np.maximum(np.maximum(a, b), c)).astype(int))
        x0, y0 = lo
        x1, y1 = hi
        if x1 < x0 or y1 < y0:
            continue
        u, v = b - a, c - a
        det = u[0] * v[1] - v[0] * u[1]
        if abs(det) < 1e-8:
            continue
        x = np.arange(x0, x1 + 1, dtype=np.float32) - a[0]
        y = np.arange(y0, y1 + 1, dtype=np.float32)[:, None] - a[1]
        wb, wc = (x * v[1] - y * v[0]) / det, (u[0] * y - u[1] * x) / det
        inside = (wb >= -1e-5) & (wc >= -1e-5) & (wb + wc <= 1.00001)
        region = np.s_[y0 : y1 + 1, x0 : x1 + 1]
        ids[region][inside] = i
        weights[region][inside] = np.stack((wb[inside], wc[inside]), axis=1)
    return ids, weights


def _pad(image, valid, steps=8):
    """Extend observed colors into unseen seam texels and island gutters without wrapping."""
    for _ in range(steps):
        old = valid.copy()
        for dst, src in (
            (np.s_[1:, :], np.s_[:-1, :]),
            (np.s_[:-1, :], np.s_[1:, :]),
            (np.s_[:, 1:], np.s_[:, :-1]),
            (np.s_[:, :-1], np.s_[:, 1:]),
        ):
            fill = ~valid[dst] & old[src]
            image[dst][fill] = image[src][fill]
            valid[dst][fill] = True


def _load_views(directory, poly_type, threshold, use_depth):
    views = []
    for face in get_polyhedron_faces(poly_type):
        file = directory / face["filename"]
        if file.is_file():
            rgb, mask = _load_rgba_and_mask(str(file), threshold)
            depth_file = file.with_name(file.stem + "_depth.png")
            depth = read_depth(depth_file) if use_depth and depth_file.is_file() else None
            if depth is not None and depth.shape != mask.shape:
                raise ValueError(f"Depth dimensions differ for {file.name}")
            views.append((face, rgb, mask, depth))
    if not views:
        raise ValueError("No multiview images available")
    return views


def _sample(view, at, normals, size):
    face, rgb, mask, depth = view
    h, w = mask.shape
    px, py = image_coordinates(at, face, size, w, h)
    facing = normals @ face["normal"]
    idx = np.flatnonzero((px >= -0.5) & (py >= -0.5) & (px <= w - 0.5) & (py <= h - 0.5) & (facing > 0))
    px, py = np.clip(px, 0, w - 1), np.clip(py, 0, h - 1)
    ix, iy = np.rint(px[idx]).astype(int), np.rint(py[idx]).astype(int)
    idx = idx[mask[iy, ix]]
    x0, y0 = np.floor(px[idx]).astype(int), np.floor(py[idx]).astype(int)
    fx, fy = px[idx] - x0, py[idx] - y0
    distance = 0.5 - at[idx] @ face["normal"] / size if depth is not None else None
    colors = np.zeros((len(idx), 3), np.float32)
    weights = np.zeros(len(idx), np.float32)
    for x, wx in ((x0, 1 - fx), (np.minimum(x0 + 1, w - 1), fx)):
        for y, wy in ((y0, 1 - fy), (np.minimum(y0 + 1, h - 1), fy)):
            valid_tap = mask[y, x]
            if depth is not None:
                valid_tap = valid_tap & (distance <= depth[y, x] + 2 / min(w, h))
            contribution = wx * wy * valid_tap
            colors += rgb[y, x] * contribution[:, None]
            weights += contribution
    hit = weights > 1e-8
    return idx[hit], colors[hit] / weights[hit, None], facing[idx[hit]] ** 3


def _project(obj, layer, views, center, size, texture_size):
    mesh = obj.data
    mesh.calc_loop_triangles()
    uv = np.empty((len(mesh.loops), 2), np.float32)
    layer.uv.foreach_get("vector", uv.ravel())
    loops = np.empty((len(mesh.loop_triangles), 3), np.int32)
    mesh.loop_triangles.foreach_get("loops", loops.ravel())
    triangles = np.empty_like(loops)
    mesh.loop_triangles.foreach_get("vertices", triangles.ravel())
    coords = np.empty((len(mesh.vertices), 3), np.float32)
    mesh.vertices.foreach_get("co", coords.ravel())
    matrix = np.asarray(obj.matrix_world, dtype=np.float32)
    points = coords @ matrix[:3, :3].T + matrix[:3, 3] - np.asarray(center)
    normals = np.empty((len(mesh.corner_normals), 3), np.float32)
    mesh.corner_normals.foreach_get("vector", normals.ravel())
    normals = normals @ np.linalg.inv(matrix[:3, :3])

    ids, bary = rasterize(uv, loops, texture_size)
    atlas = np.full((texture_size, texture_size, 3), 0.5, np.float32)
    valid = np.zeros_like(ids, dtype=bool)
    for row in range(0, texture_size, 32):
        y, x = np.nonzero(ids[row : row + 32] >= 0)
        y += row
        fi = ids[y, x]
        wb, wc = bary[y, x].T
        weights = np.stack((1 - wb - wc, wb, wc), axis=1)
        at = np.einsum("ij,ijk->ik", weights, points[triangles[fi]])
        n = np.einsum("ij,ijk->ik", weights, normals[loops[fi]])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
        colors = np.zeros((len(x), 3), np.float32)
        totals = np.zeros(len(x), np.float32)
        for view in views:
            idx, color, weight = _sample(view, at, n, size)
            totals[idx] += weight
            colors[idx] += color * weight[:, None]

        seen = totals > 1e-8
        atlas[y[seen], x[seen]] = colors[seen] / totals[seen, None]
        valid[y[seen], x[seen]] = True
    if not valid.any():
        raise ValueError("No projected texels hit the UV atlas")
    _pad(atlas, valid)
    srgb = linear_to_srgb(atlas)
    return np.rint(np.clip(srgb[::-1], 0, 1) * 255).astype(np.uint8)


def _material(mat, image, uv_name):
    mat.use_nodes = True
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = image
    uv_node = mat.node_tree.nodes.new("ShaderNodeUVMap")
    uv_node.uv_map = uv_name
    mat.node_tree.links.new(uv_node.outputs["UV"], tex.inputs["Vector"])
    mat.node_tree.links.new(tex.outputs["Color"], mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"])


def bake(
    obj,
    image_dir,
    poly_type="DODECAHEDRON_12",
    center=(0, 0, 0.9),
    size=2,
    texture_size=2048,
    path=None,
    threshold=0.15,
    use_depth=True,
):
    if obj.type != "MESH" or not obj.data.polygons or bpy.context.mode != "OBJECT":
        raise ValueError("Select a mesh with faces in Object Mode before baking")
    if obj.library and not obj.override_library:
        raise ValueError("Make the linked object editable before baking")
    if not isinstance(texture_size, int) or texture_size not in {256, 512, 1024, 2048, 4096}:
        raise ValueError("Use an atlas resolution of 256, 512, 1024, 2048 or 4096")
    if not np.isfinite(size) or size <= 0 or np.shape(center) != (3,) or not np.isfinite(center).all():
        raise ValueError("Provide a finite projection center and positive size")
    directory = Path(bpy.path.abspath(image_dir))
    views = _load_views(directory, poly_type, threshold, use_depth)
    dest = (
        Path(bpy.path.abspath(path)) if path else directory / (bpy.path.clean_name(obj.name) + "_albedo.png")
    ).resolve()
    source = obj.data
    mesh = source.copy() if source.users > 1 or source.library else source
    old_materials = list(source.materials)
    old_slots = [(slot.link, slot.material) for slot in obj.material_slots]
    old_indices = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("material_index", old_indices)
    old_active_slot = obj.active_material_index
    had_atlas = "EXCAD Atlas" in mesh.uv_layers
    active_uv = mesh.uv_layers.active.name if mesh.uv_layers.active else None
    image = material = temp = None
    replacing_slots = False
    try:
        if mesh != source:
            obj.data = mesh
        layer = smart_uv(obj)
        pixels = _project(obj, layer, views, center, size, texture_size)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dest.parent, suffix=".png", delete=False) as pending:
            temp = Path(pending.name)
        _png(temp, pixels)
        image = bpy.data.images.load(str(temp), check_existing=False)
        image.filepath_raw = str(dest)
        material = bpy.data.materials.new(obj.name + " Baked")
        _material(material, image, layer.name)
        material["excad_uv_bake"] = image["excad_uv_bake"] = True
        replacing_slots = True
        mesh.materials.clear()
        mesh.materials.append(material)
        obj.material_slots[0].link = "DATA"
        obj.active_material_index = 0
        temp.replace(dest)
    except Exception:
        if mesh != source:
            obj.data = source
            bpy.data.meshes.remove(mesh)
        else:
            if replacing_slots:
                mesh.materials.clear()
                for old in old_materials:
                    mesh.materials.append(old)
                mesh.polygons.foreach_set("material_index", old_indices)
            if not had_atlas and (atlas := mesh.uv_layers.get("EXCAD Atlas")):
                mesh.uv_layers.remove(atlas)
            if active_uv:
                mesh.uv_layers.active = mesh.uv_layers[active_uv]
        if replacing_slots:
            for slot, (link, old) in zip(obj.material_slots, old_slots):
                slot.link = link
                if link == "OBJECT":
                    slot.material = old
            obj.active_material_index = old_active_slot
        if material:
            bpy.data.materials.remove(material)
        if image:
            bpy.data.images.remove(image)
        raise
    finally:
        if temp:
            temp.unlink(missing_ok=True)

    for old in set(old_materials) | {mat for _, mat in old_slots}:
        if old and old.get("excad_uv_bake") and old.users == 0:
            textures = {node.image for node in old.node_tree.nodes if node.type == "TEX_IMAGE" and node.image}
            bpy.data.materials.remove(old)
            for texture in textures:
                if texture.get("excad_uv_bake") and texture.users == 0:
                    bpy.data.images.remove(texture)
    material.name = obj.name + " Baked"
    image.name = dest.name
    return {"path": str(dest), "resolution": texture_size, "views_used": len(views)}
