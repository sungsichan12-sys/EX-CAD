# SPDX-FileCopyrightText: 2026 EX-CAD Authors
# SPDX-License-Identifier: GPL-2.0-or-later

import os
import subprocess
import urllib.parse
import urllib.request

import bpy
from mathutils import Matrix, Vector


CACHE_DIR = os.path.join(os.path.dirname(__file__), "assets_cache")
SUPPORTED_EXTS = (".glb", ".gltf", ".obj", ".fbx", ".blend", ".mhclo")

EXTERNAL_PRESETS = {
    "KHRONOS_RIGGED_FIGURE": {
        "label": "Rigged Human Figure (Khronos CC0)",
        "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/RiggedFigure/glTF-Binary/RiggedFigure.glb",
        "filename": "RiggedFigure.glb",
    },
    "KHRONOS_CESIUM_MAN": {
        "label": "Cesium Man Animated (Khronos CC0)",
        "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/CesiumMan/glTF-Binary/CesiumMan.glb",
        "filename": "CesiumMan.glb",
    },
    "KHRONOS_BRAINSTEM": {
        "label": "BrainStem Mech Character (Khronos CC0)",
        "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/BrainStem/glTF-Binary/BrainStem.glb",
        "filename": "BrainStem.glb",
    },
}


def ensure_cache_dir() -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


def get_asset_roots(extra_dir: str = "") -> list[tuple[str, str]]:
    ensure_cache_dir()
    roots: list[tuple[str, str]] = [("cache", CACHE_DIR)]
    seen = {os.path.normcase(os.path.abspath(CACHE_DIR))}

    if extra_dir:
        ap = os.path.abspath(bpy.path.abspath(extra_dir))
        if os.path.isdir(ap) and os.path.normcase(ap) not in seen:
            roots.append(("local", ap))
            seen.add(os.path.normcase(ap))

    try:
        for lib in bpy.context.preferences.filepaths.asset_libraries:
            if lib.path and os.path.isdir(lib.path):
                ap = os.path.abspath(lib.path)
                if os.path.normcase(ap) not in seen:
                    roots.append((f"lib:{lib.name}", ap))
                    seen.add(os.path.normcase(ap))
    except Exception:
        pass

    try:
        from bl_ext.blender_org.mpfb.services.locationservice import LocationService
        mpfb_user = LocationService.get_user_data("")
        if mpfb_user and os.path.isdir(mpfb_user):
            ap = os.path.abspath(mpfb_user)
            if os.path.normcase(ap) not in seen:
                roots.append(("mpfb_user", ap))
                seen.add(os.path.normcase(ap))
    except Exception:
        pass

    return roots


def search_assets(query: str = "", extra_dir: str = "", limit: int = 30) -> list[dict]:
    tokens = [t.lower() for t in query.strip().split() if t]
    matches: list[dict] = []
    seen_paths = set()

    for source, root_dir in get_asset_roots(extra_dir):
        for r, _, files in os.walk(root_dir):
            if ".git" in r.split(os.sep):
                continue
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in SUPPORTED_EXTS:
                    continue
                full = os.path.join(r, fn)
                norm = os.path.normcase(os.path.abspath(full))
                if norm in seen_paths:
                    continue
                seen_paths.add(norm)
                rel = os.path.relpath(full, root_dir)
                hay = rel.lower()
                if tokens and not all(tok in hay for tok in tokens):
                    continue
                stem = os.path.splitext(fn)[0].lower()
                exact = 0 if (tokens and stem == " ".join(tokens)) else 1
                matches.append({
                    "name": fn,
                    "rel": rel,
                    "path": full,
                    "ext": ext,
                    "source": source,
                    "_score": (exact, len(rel)),
                })

    matches.sort(key=lambda m: m["_score"])
    out = []
    for m in matches[:max(1, int(limit))]:
        m.pop("_score", None)
        out.append(m)
    return out


def import_asset_file(filepath: str, target_basemesh=None, *, merge_vertices=False) -> list:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Asset file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    before = set(bpy.data.objects.keys())

    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=filepath, merge_vertices=merge_vertices)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=filepath)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif ext == ".blend":
        with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
            data_to.objects = list(data_from.objects)
        for obj in data_to.objects:
            if obj is not None and obj.name not in bpy.context.collection.objects:
                bpy.context.collection.objects.link(obj)
    elif ext == ".mhclo":
        from bl_ext.blender_org.mpfb.services.humanservice import HumanService
        basemesh = target_basemesh or bpy.context.active_object
        if basemesh is None:
            raise RuntimeError("Select a human basemesh before importing .mhclo clothes/hair")
        HumanService.add_mhclo_asset(filepath, basemesh, asset_type="Clothes", subdiv_levels=1)
    else:
        raise ValueError(f"Unsupported asset format: {ext}")

    after = set(bpy.data.objects.keys())
    new_names = sorted(after - before)
    return [bpy.data.objects[n] for n in new_names if n in bpy.data.objects]


def bone_shapes(objects):
    return {bone.custom_shape for obj in objects if obj.type == 'ARMATURE' and obj.pose
            for bone in obj.pose.bones if bone.custom_shape is not None}


def _world_bbox(objs):
    # Bone shapes are viewport controls, independent of a character's dimensions.
    controls = bone_shapes(objs)
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    found = False
    for obj in objs:
        if obj in controls or obj.type != 'MESH' or not obj.data or len(obj.data.vertices) == 0:
            continue
        found = True
        mw = obj.matrix_world
        for corner in obj.bound_box:
            wc = mw @ Vector(corner)
            for k in range(3):
                if wc[k] < mins[k]:
                    mins[k] = wc[k]
                if wc[k] > maxs[k]:
                    maxs[k] = wc[k]
    if not found:
        return None
    return Vector(mins), Vector(maxs)


def _root_objects(objs):
    obj_set = set(objs)
    return [o for o in objs if o.parent is None or o.parent not in obj_set]


def import_asset_naturally(
    query_or_path: str,
    place_mode: str = 'NEXT_TO',
    target_height: float = 0.0,
    attach_to=None,
    attach_bone: str = "",
    extra_dir: str = "",
    auto_smooth: bool = True,
):
    resolved = query_or_path.strip()
    if not os.path.isfile(resolved):
        cached = os.path.join(CACHE_DIR, resolved)
        if os.path.isfile(cached):
            resolved = cached
        else:
            hits = search_assets(resolved, extra_dir=extra_dir, limit=1)
            if not hits:
                raise FileNotFoundError(f"No matching asset found for '{query_or_path}'")
            resolved = hits[0]["path"]

    existing_meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    prev_active = attach_to or bpy.context.active_object

    imported = import_asset_file(resolved, target_basemesh=prev_active)
    if not imported:
        return resolved, [], None

    bpy.context.view_layer.update()
    roots = _root_objects(imported)
    bbox = _world_bbox(imported)

    if bbox is not None:
        mn, mx = bbox
        cur_h = mx.z - mn.z
        if target_height > 0.0 and cur_h > 1e-5:
            s = float(target_height) / cur_h
            for r in roots:
                r.scale = (r.scale.x * s, r.scale.y * s, r.scale.z * s)
            bpy.context.view_layer.update()
            bbox = _world_bbox(imported)

    if bbox is not None:
        mn, mx = bbox
        cx = 0.5 * (mn.x + mx.x)
        cy = 0.5 * (mn.y + mx.y)
        delta = Vector((0.0, 0.0, 0.0))

        if place_mode == 'CURSOR':
            cur = bpy.context.scene.cursor.location
            delta = Vector((cur.x - cx, cur.y - cy, cur.z - mn.z))
        elif place_mode == 'ON_TARGET' and prev_active is not None:
            tb = _world_bbox([prev_active])
            if tb is not None:
                t_mn, t_mx = tb
                tcx = 0.5 * (t_mn.x + t_mx.x)
                tcy = 0.5 * (t_mn.y + t_mx.y)
                delta = Vector((tcx - cx, tcy - cy, t_mx.z - mn.z))
            else:
                delta = Vector((-cx, -cy, -mn.z))
        elif place_mode == 'NEXT_TO' and existing_meshes:
            eb = _world_bbox(existing_meshes)
            if eb is not None:
                ex_mn, ex_mx = eb
                ecy = 0.5 * (ex_mn.y + ex_mx.y)
                delta = Vector(((ex_mx.x + 0.35) - mn.x, ecy - cy, -mn.z))
            else:
                delta = Vector((-cx, -cy, -mn.z))
        else:  # 'ORIGIN' or fallback
            delta = Vector((-cx, -cy, -mn.z))

        for r in roots:
            r.location = r.location + delta
        bpy.context.view_layer.update()
        bbox = _world_bbox(imported)

    if attach_to is not None and (place_mode == 'ON_TARGET' or attach_bone):
        for r in roots:
            mw = r.matrix_world.copy()
            r.parent = attach_to
            if attach_bone and attach_to.type == 'ARMATURE' and attach_bone in attach_to.data.bones:
                r.parent_type = 'BONE'
                r.parent_bone = attach_bone
                r.matrix_parent_inverse = Matrix.Identity(4)
            else:
                r.parent_type = 'OBJECT'
                r.matrix_parent_inverse = attach_to.matrix_world.inverted()
            r.matrix_world = mw

    if auto_smooth:
        for obj in imported:
            if obj.type == 'MESH' and obj.data:
                for poly in obj.data.polygons:
                    poly.use_smooth = True

    for o in bpy.context.selected_objects:
        o.select_set(False)
    for o in imported:
        o.select_set(True)
    bpy.context.view_layer.objects.active = roots[0]

    bbox_out = None
    if bbox is not None:
        mn, mx = bbox
        bbox_out = {
            "min": [round(float(mn.x), 3), round(float(mn.y), 3), round(float(mn.z), 3)],
            "max": [round(float(mx.x), 3), round(float(mx.y), 3), round(float(mx.z), 3)],
        }
    return resolved, imported, bbox_out


def fetch_url_asset(url: str, filename: str | None = None, auto_import: bool = True, target_basemesh=None):
    ensure_cache_dir()
    if not filename:
        parsed = urllib.parse.urlparse(url)
        filename = os.path.basename(parsed.path) or "downloaded_asset.glb"
    dest = os.path.join(CACHE_DIR, filename)

    if not os.path.isfile(dest):
        req = urllib.request.Request(url, headers={"User-Agent": "EX-CAD/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
            out.write(resp.read())

    imported = import_asset_file(dest, target_basemesh=target_basemesh) if auto_import else []
    return dest, imported


def fetch_git_repo(repo_url: str, auto_import: bool = True, target_basemesh=None):
    ensure_cache_dir()
    repo_name = os.path.splitext(os.path.basename(repo_url.rstrip("/")))[0] or "repo"
    dest_dir = os.path.join(CACHE_DIR, f"git_{repo_name}")

    if os.path.isdir(os.path.join(dest_dir, ".git")):
        subprocess.run(["git", "-C", dest_dir, "pull", "--ff-only"], check=True, capture_output=True, text=True)
    else:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, dest_dir],
            check=True,
            capture_output=True,
            text=True,
        )

    found = []
    for root, _, files in os.walk(dest_dir):
        if ".git" in root.split(os.sep):
            continue
        for fn in sorted(files):
            if fn.lower().endswith(SUPPORTED_EXTS):
                found.append(os.path.join(root, fn))

    imported = []
    if auto_import and found:
        imported = import_asset_file(found[0], target_basemesh=target_basemesh)
    return dest_dir, found, imported


def list_cached_assets() -> list[str]:
    ensure_cache_dir()
    results = []
    for root, _, files in os.walk(CACHE_DIR):
        if ".git" in root.split(os.sep):
            continue
        for fn in sorted(files):
            if fn.lower().endswith(SUPPORTED_EXTS):
                results.append(os.path.relpath(os.path.join(root, fn), CACHE_DIR))
    return results
