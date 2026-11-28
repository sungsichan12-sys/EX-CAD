# SPDX-License-Identifier: GPL-2.0-or-later
"""Provider protocols run without bpy. Only the final GLB import needs Blender's main thread."""

import base64
import json
import math
import mimetypes
import os
import tempfile
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

CLOUD = {"MESHY": "https://api.meshy.ai/openapi", "TRIPO": "https://openapi.tripo3d.ai/v3"}


def request(url, data=None, *, token=None, content_type=None, timeout=90):
    headers = {}
    if token:
        headers["Authorization"] = "Bearer " + token
    if isinstance(data, dict):
        data = json.dumps(data).encode()
        content_type = "application/json"
    if content_type:
        headers["Content-Type"] = content_type
    try:
        with urlopen(Request(url, data=data, headers=headers), timeout=timeout) as response:
            return response.read()
    except HTTPError as exc:
        raise RuntimeError(f"Provider HTTP {exc.code}: {exc.reason}") from exc


def multipart(path, field):
    boundary = "excad-" + uuid.uuid4().hex
    kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    safe_name = "input" + path.suffix.lower()
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{safe_name}"\r\n'
            f"Content-Type: {kind}\r\n\r\n"
        ).encode()
        + path.read_bytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return body, "multipart/form-data; boundary=" + boundary


def _remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("3D generation timed out")
    return seconds


def _poll(url, token, provider, deadline):
    while time.monotonic() < deadline:
        state = json.loads(request(url, token=token, timeout=_remaining(deadline)))
        if provider == "TRIPO":
            if state.get("code") != 0:
                raise RuntimeError(f"Tripo task query failed ({state.get('code')})")
            state = state["data"]
        status = state["status"].lower()
        if status in {"succeeded", "success"}:
            if provider == "TRIPO":
                return state["output"]["model_url"]
            return state["model_urls"]["glb"]
        if status not in {"pending", "queued", "running", "processing", "in_progress", "created"}:
            raise RuntimeError(f"{provider} task ended with status {status}")
        time.sleep(min(2.0, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"{provider} generation timed out")


def _cloud(provider, prompt, image, deadline):
    token = os.getenv(provider + "_API_KEY")
    if not token:
        raise ValueError(f"Set {provider}_API_KEY to use this provider")
    base = CLOUD[provider]
    if provider == "MESHY":
        if image:
            mime = mimetypes.guess_type(image.name)[0] or "image/png"
            uri = "data:" + mime + ";base64," + base64.b64encode(image.read_bytes()).decode()
            endpoint, payload = "/v1/image-to-3d", {"image_url": uri}
        else:
            endpoint, payload = "/v2/text-to-3d", {"mode": "preview", "prompt": prompt}
        url = base + endpoint
        task = json.loads(request(url, payload, token=token, timeout=_remaining(deadline)))["result"]
        download = _poll(url + "/" + quote(task, safe=""), token, provider, deadline)
        if not image:
            refine = json.loads(
                request(url, {"mode": "refine", "preview_task_id": task}, token=token, timeout=_remaining(deadline))
            )["result"]
            download = _poll(url + "/" + quote(refine, safe=""), token, provider, deadline)
    else:
        payload = {"model": "v3.1-20260211", "texture": True, "pbr": True}
        if image:
            body, ctype = multipart(image, "file")
            upload = json.loads(
                request(base + "/files", body, token=token, content_type=ctype, timeout=_remaining(deadline))
            )
            if upload.get("code") != 0:
                raise RuntimeError(f"Tripo image upload failed ({upload.get('code')})")
            payload["input"] = upload["data"]["file_token"]
            route = "/generation/image-to-model"
        else:
            payload["prompt"] = prompt
            route = "/generation/text-to-model"
        response = json.loads(request(base + route, payload, token=token, timeout=_remaining(deadline)))
        if response.get("code") != 0:
            raise RuntimeError(f"Tripo submission failed ({response.get('code')})")
        task = response["data"]["task_id"]
        download = _poll(base + "/tasks/" + quote(task, safe=""), token, provider, deadline)
    # Signed result URLs have their own authority. Only the API origin receives the key.
    return request(download, timeout=_remaining(deadline))


def _trellis(base, image, deadline):
    if not image:
        raise ValueError("The upstream TRELLIS image server requires image_path")
    config = json.loads(request(base + "/config", timeout=_remaining(deadline)))
    # Gradio 4 serves its queue at the root. Newer releases publish api_prefix.
    api = base + config.get("api_prefix", "").rstrip("/")
    functions = {f["api_name"]: f for f in config["dependencies"] if f.get("api_name")}
    components = {c["id"]: c["type"] for c in config["components"]}
    session = uuid.uuid4().hex

    def call(name, arguments):
        fn = functions.get(name)
        if not fn:
            raise ValueError(f"The TRELLIS server does not expose {name}")
        fields = [components[c] for c in fn["inputs"]]
        if sum(field != "state" for field in fields) != len(arguments):
            raise ValueError(f"Incompatible TRELLIS inputs for {name}")
        inputs = iter(arguments)
        data = [None if field == "state" else next(inputs) for field in fields]
        payload = {"data": data, "fn_index": fn["id"], "session_hash": session, "event_data": None}
        event = json.loads(request(api + "/queue/join", payload, timeout=_remaining(deadline)))["event_id"]
        with urlopen(api + "/queue/data?session_hash=" + session, timeout=_remaining(deadline)) as stream:
            for line in stream:
                if time.monotonic() > deadline:
                    raise TimeoutError("TRELLIS generation timed out")
                if not line.startswith(b"data:"):
                    continue
                message = json.loads(line[5:].strip())
                if message.get("event_id") != event or message.get("msg") != "process_completed":
                    continue
                result = message["output"]
                if not message.get("success"):
                    raise RuntimeError(f"TRELLIS {name} failed: {result.get('error') or 'model server error'}")
                return result["data"]
        raise RuntimeError(f"TRELLIS {name} ended without a result")

    body, ctype = multipart(image, "files")
    file = json.loads(request(api + "/upload", body, content_type=ctype, timeout=_remaining(deadline)))[0]
    call("start_session", [])
    processed = call("preprocess_image", [{"path": file, "meta": {"_type": "gradio.FileData"}}])[0]
    generation = [processed, [], 0, 7.5, 12, 3.0, 12, "stochastic"]
    extraction = [0.95, 1024]
    merged = functions.get("generate_and_extract_glb")
    if merged and not {"image_to_3d", "extract_glb"}.issubset(functions):
        result = call("generate_and_extract_glb", [*generation, *extraction])
        model = next(
            (value for key, value in zip(merged["outputs"], result) if components[key] == "downloadbutton" and value),
            None,
        )
        if not model:
            raise ValueError("The TRELLIS server did not return a downloadable GLB")
    else:
        call("image_to_3d", generation)
        model = call("extract_glb", extraction)[0]
    url = model.get("url") or api + "/file=" + quote(model["path"], safe="/:")
    return request(urljoin(base + "/", url), timeout=_remaining(deadline))


def download_model(prompt="", image_path=None, provider="HUNYUAN3D-2", timeout=900, texture=False):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Specify a positive generation timeout in seconds")
    provider = provider.upper()
    image = Path(image_path) if image_path else None
    if not prompt and image is None:
        raise ValueError("Provide a prompt or an image")
    if image and not image.is_file():
        raise FileNotFoundError(image)
    deadline = time.monotonic() + timeout
    if provider in CLOUD:
        blob = _cloud(provider, prompt, image, deadline)
    elif provider in {"HUNYUAN3D-2", "HUNYUAN"}:
        base = os.getenv("HUNYUAN3D_URL", "http://127.0.0.1:8080").rstrip("/")
        payload = {"image": base64.b64encode(image.read_bytes()).decode()} if image else {"text": prompt}
        blob = request(base + "/generate", dict(payload, type="glb", texture=texture), timeout=_remaining(deadline))
    elif provider == "TRELLIS":
        blob = _trellis(os.getenv("TRELLIS_URL", "http://127.0.0.1:7860").rstrip("/"), image, deadline)
    else:
        raise ValueError(f"Unknown 3D provider {provider}")
    if blob[:4] != b"glTF" or blob[4:8] != b"\x02\x00\x00\x00" or int.from_bytes(blob[8:12], "little") != len(blob):
        raise ValueError("Provider returned an invalid GLB 2.0 file")
    with tempfile.NamedTemporaryFile(prefix="excad-model-", suffix=".glb", delete=False) as out:
        out.write(blob)
        return Path(out.name)


def _dimensions(meshes):
    import bpy
    from mathutils import Vector

    from .scene_mesh import read_surface

    graph = bpy.context.evaluated_depsgraph_get()
    parts = [s for obj in meshes if (s := read_surface(obj, graph)) is not None]
    if not parts:
        raise ValueError("GLB contains no surface geometry")
    low = Vector(tuple(min(s.bounds[0, i] for s in parts) for i in range(3)))
    high = Vector(tuple(max(s.bounds[1, i] for s in parts) for i in range(3)))
    return low, high


def _normalize(roots, meshes, height, name):
    import bpy
    from mathutils import Matrix, Vector

    low, high = _dimensions(meshes)
    if high.z - low.z <= 1e-8:
        raise ValueError("GLB has no height")
    center = Vector(((low.x + high.x) / 2, (low.y + high.y) / 2, low.z))
    matrix = Matrix.Scale(height / (high.z - low.z), 4) @ Matrix.Translation(-center)
    pivot = None
    if any(root.animation_data for root in roots):
        # Keep animation curves in their authored units. Scaling or translating the
        # animated root itself would be undone on the next frame evaluation.
        pivot = bpy.data.objects.new(name + " placement", None)
        bpy.context.collection.objects.link(pivot)
        for root in roots:
            world = root.matrix_world.copy()
            root.parent = pivot
            root.matrix_world = world
        pivot.matrix_world = matrix
    else:
        for root in roots:
            root.matrix_world = matrix @ root.matrix_world
    bpy.context.view_layer.update()
    return pivot


def _rig_parts(meshes, name):
    import bpy
    from mathutils import Vector

    armature = bpy.data.armatures.new(name + " parts")
    rig = bpy.data.objects.new(name + " rig", armature)
    bpy.context.collection.objects.link(rig)
    low, high = _dimensions(meshes)
    size = max(1e-4, (high - low).length)
    centers = {obj: sum(_dimensions([obj]), Vector((0, 0, 0))) / 2 for obj in meshes}
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bones = {}
    try:
        root = armature.edit_bones.new("Root")
        root.head, root.tail = (low + high) / 2, (low + high) / 2 + Vector((0, 0, size * 0.12))
        for obj in meshes:
            bone = armature.edit_bones.new(obj.name)
            bone.head = centers[obj]
            bone.tail = bone.head + Vector((0, 0, size * 0.08))
            bones[obj] = bone.name
        for obj, bone in bones.items():
            parent = obj.parent
            while parent and parent not in bones:
                parent = parent.parent
            armature.edit_bones[bone].parent = armature.edit_bones[bones[parent]] if parent else root
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj, bone in bones.items():
        if obj.data.users > 1:
            obj.data = obj.data.copy()
        group = obj.vertex_groups.get(bone) or obj.vertex_groups.new(name=bone)
        group.add(list(range(len(obj.data.vertices))), 1, "REPLACE")
        obj.modifiers.new("Assembly pose", "ARMATURE").object = rig
    return rig


def import_model(path, height=1.0, remesh=False, rig=False, name="Generated model"):
    import bpy

    from .asset_fetcher import _root_objects, bone_shapes, import_asset_file
    from .humanoid_skin import fit_imported
    from .inspect_shading import warm_materials

    mode = "PARTS" if rig is True else "NONE" if not rig else str(rig).upper()
    if not math.isfinite(height) or height <= 0 or mode not in {"NONE", "PARTS", "HUMANOID"} or not name.strip():
        raise ValueError("Specify a positive height, a name and a valid rig mode")
    if bpy.context.mode != "OBJECT":
        raise ValueError("Switch to Object Mode before importing a generated model")
    groups = (
        "objects",
        "collections",
        "meshes",
        "armatures",
        "materials",
        "images",
        "actions",
        "cameras",
        "lights",
        "node_groups",
    )
    before = {key: set(getattr(bpy.data, key)) for key in groups}
    selected, active = list(bpy.context.selected_objects), bpy.context.view_layer.objects.active
    try:
        # GLB stores separate render vertices at UV seams. Blender can share
        # their positions while retaining a different UV on each face corner.
        imported = import_asset_file(str(path), merge_vertices=True)
        controls = bone_shapes(imported)
        imported = [obj for obj in imported if obj not in controls and obj.name in bpy.context.scene.objects]
        meshes = [o for o in imported if o.type == "MESH"]
        if not meshes:
            raise ValueError("GLB contains no mesh")
        bpy.context.view_layer.update()
        for obj in meshes:
            if remesh:
                if obj.data.shape_keys:
                    raise ValueError("Voxel remeshing changes topology and cannot preserve GLB morph targets")
                if obj.data.users > 1:
                    obj.data = obj.data.copy()
                corners = obj.bound_box
                extent = max(max(c[i] for c in corners) - min(c[i] for c in corners) for i in range(3))
                obj.data.remesh_voxel_size = extent / 128
                obj.data.use_remesh_preserve_attributes = True
                original = obj.data.copy() if obj.data.uv_layers else None
                try:
                    with bpy.context.temp_override(object=obj, active_object=obj):
                        if "FINISHED" not in bpy.ops.object.voxel_remesh():
                            raise ValueError(f"Could not remesh {obj.name}")
                    if original:
                        from .uv_transfer import transfer_uv

                        transfer_uv(original, obj.data)
                finally:
                    if original:
                        bpy.data.meshes.remove(original)
        bpy.context.view_layer.update()
        placement = _normalize(_root_objects(imported), meshes, height, name)
        if placement is not None:
            imported.append(placement)
        loose = [obj for obj in meshes if not obj.find_armature()]
        rig_obj = None
        if loose and mode == "PARTS":
            rig_obj = _rig_parts(loose, name)
        elif loose and mode == "HUMANOID":
            rig_obj = fit_imported(meshes, loose, name)
        if rig_obj is not None:
            for top in _root_objects(imported):
                matrix = top.matrix_world.copy()
                top.parent = rig_obj
                top.matrix_world = matrix
            imported.append(rig_obj)
        roots = _root_objects(imported)
        if len(roots) == 1:
            model = roots[0]
        else:
            model = bpy.data.objects.new(name, None)
            bpy.context.collection.objects.link(model)
            imported.append(model)
            for root in roots:
                world = root.matrix_world.copy()
                root.parent = model
                root.matrix_world = world
        model.name = name
        bpy.context.view_layer.update()
        for obj in meshes:
            warm_materials(obj)
        return {
            "name": model.name,
            "objects": [o.name for o in imported],
            "meshes": [o.name for o in meshes],
            "height": height,
        }
    except Exception:
        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        added = set().union(*(set(getattr(bpy.data, key)) - ids for key, ids in before.items()))
        if added:
            bpy.data.batch_remove(added)
        for obj in selected:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = active
        raise


def generate_3d(prompt="", image_path=None, provider="HUNYUAN3D-2", **options):
    path = download_model(
        prompt, image_path, provider, timeout=options.pop("timeout", 900), texture=options.pop("texture", False)
    )
    try:
        return import_model(path, **options)
    finally:
        path.unlink(missing_ok=True)
