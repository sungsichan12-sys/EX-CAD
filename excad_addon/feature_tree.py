# SPDX-License-Identifier: GPL-2.0-or-later
"""Replay ordered CAD features in isolation before replacing the visible mesh."""

import copy
import json
import math
import os
from pathlib import Path
import tempfile

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from . import feature_geometry as geo
from .sketch_engine import SketchDocument

FEATURES = {"EXTRUDE", "REVOLVE", "LOFT", "SWEEP", "FILLET", "CHAMFER", "SHELL", "PATTERN", "MIRROR3D"}
SHAPES = {"ARC", "SLOT", "REGULAR_POLYGON", "SPLINE", "OFFSET_LOOP", "MIRROR2D"}


def sketch_from_spec(spec, body, owner_name, model_frame, current_placement):
    if "entities" in spec:
        doc = SketchDocument.deserialize(spec)
    else:
        doc = SketchDocument(spec.get("plane", "XY"))
        doc.frame, doc.support = spec.get("frame"), spec.get("support")
        for shape in spec["shapes"]:
            args = dict(shape)
            kind = args.pop("type").upper()
            if kind == "RECT":
                width, height = args["width"], args["height"]
                center = args.get("center", (0, 0))
                doc.add_rect(center[0] - width / 2, center[1] - height / 2, width, height)
            elif kind == "CIRCLE":
                doc.add_circle(*args.get("center", (0, 0)), args["radius"])
            elif kind == "LINE":
                doc.add_line(*args["start"], *args["end"])
            else:
                doc.add_shape(kind, **args)
        for constraint in spec.get("constraints", []):
            doc.add_constraint(**constraint)
    if doc.support:
        name, side = doc.support["object"], doc.support["face"]
        on_body = name in {"SELF", owner_name}
        target = body if on_body else bpy.context.scene.objects.get(name)
        if target is None:
            raise ValueError(f"Missing support body {name}")
        if on_body:
            reference = doc.support.get("reference_frame")
            placement = Matrix(reference) @ model_frame.inverted() if reference else Matrix.Identity(4)
            frame = placement.inverted() @ Matrix(geo.selected_face(target, side, placement))
        else:
            frame = current_placement.inverted() @ Matrix(geo.selected_face(target, side))
        doc.frame = [list(row) for row in frame]
    if spec.get("offset"):
        frame = (
            Matrix(doc.frame)
            if doc.frame
            else Matrix(
                (
                    Vector(doc.plane_to_world(1, 0, 0)),
                    Vector(doc.plane_to_world(0, 1, 0)),
                    Vector(doc.plane_to_world(0, 0, 1)),
                )
            )
            .transposed()
            .to_4x4()
        )
        frame.translation += float(spec["offset"]) * frame.col[2].xyz
        doc.frame = [list(row) for row in frame]
    return doc


def _mesh_copy(objects, source, transform):
    reflected = transform.to_3x3().determinant() < 0
    return geo.new_mesh(
        objects,
        [transform @ v.co for v in source.vertices],
        [tuple(reversed(p.vertices[:])) if reflected else tuple(p.vertices) for p in source.polygons],
        orient=False,
    )


def _pattern(objects, body, feature):
    kind = feature.get("pattern_type", feature.get("mode", "LINEAR")).upper()
    count = int(feature.get("count", 2))
    if not 2 <= count <= 256:
        raise ValueError("Pattern count must be between 2 and 256, including the original")
    axis = geo.axis_vector(feature.get("axis", "X" if kind == "LINEAR" else "Z"))
    pivot = Vector(feature.get("origin", (0, 0, 0)))
    original = body.data.copy()
    try:
        for i in range(1, count):
            if kind == "LINEAR":
                matrix = Matrix.Translation(axis * (float(feature.get("spacing", 1)) * i))
            elif kind == "CIRCULAR":
                angle = float(feature.get("angle_deg", 360))
                divisor = count if abs(angle) == 360 else count - 1
                matrix = (
                    Matrix.Translation(pivot)
                    @ Matrix.Rotation(math.radians(angle) * i / divisor, 4, axis)
                    @ Matrix.Translation(-pivot)
                )
            else:
                raise ValueError("Pattern type must be LINEAR or CIRCULAR")
            geo.boolean(body, _mesh_copy(objects, original, matrix), "ADD")
    finally:
        bpy.data.meshes.remove(original)


def _solid(mesh):
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        if not bm.faces or any(not e.is_manifold for e in bm.edges) or any(not v.is_manifold for v in bm.verts):
            raise ValueError("Feature produced an empty or non-manifold solid")
        if abs(bm.calc_volume(signed=True)) < 1e-12:
            raise ValueError("Feature produced zero volume")
    finally:
        bm.free()


def _vertex_positions(mesh):
    xyz = np.empty((len(mesh.vertices), 3), np.float32)
    mesh.vertices.foreach_get("co", xyz.ravel())
    return xyz


def _carry_face_materials(source, rebuilt):
    indices = np.empty(len(source.polygons), np.int32)
    source.polygons.foreach_get("material_index", indices)
    if np.all(indices == indices[0]):
        result = np.full(len(rebuilt.polygons), indices[0], np.int32)
    else:
        # A dimensional edit can renumber faces. Match their positions in the
        # two model frames so painted caps follow a change in height or width.
        coords = _vertex_positions(source)
        coords = (coords - coords.min(0)) / np.maximum(np.ptp(coords, axis=0), 1e-8)
        tree = BVHTree.FromPolygons(coords.tolist(), [p.vertices[:] for p in source.polygons])
        coords = _vertex_positions(rebuilt)
        minimum, span = coords.min(0), np.maximum(np.ptp(coords, axis=0), 1e-8)
        centers = np.empty((len(rebuilt.polygons), 3), np.float32)
        rebuilt.polygons.foreach_get("center", centers.ravel())
        nearest = (tree.find_nearest(p)[2] for p in (centers - minimum) / span)
        result = np.fromiter((indices[i] if i is not None else 0 for i in nearest), np.int32, len(centers))
    rebuilt.polygons.foreach_set("material_index", result)


def _bounds_frame(mesh):
    xyz = _vertex_positions(mesh)
    span = np.maximum(np.ptp(xyz, axis=0), 1e-8)
    return Matrix.Diagonal(Vector((*np.reciprocal(span), 1))) @ Matrix.Translation(Vector(-xyz.min(0)))


def _carry_attributes(owner, body):
    from .uv_transfer import transfer_uv

    source, result = owner.data, body.data
    if source.uv_layers:
        transfer_uv(source, result, normalize=True)
    if not owner.vertex_groups and not source.color_attributes:
        return
    # Map pristine local geometry in unit bounds. Modifiers, parents and the
    # object's current pose must not affect the correspondence of its rest mesh.
    proxy = bpy.data.objects.new("EXCAD transfer source", source)
    bpy.context.collection.objects.link(proxy)
    original = body.matrix_world.copy()
    try:
        proxy.matrix_world, body.matrix_world = _bounds_frame(source), _bounds_frame(result)
        bpy.context.view_layer.update()
        types = set()
        for group in owner.vertex_groups:
            copied = body.vertex_groups.get(group.name) or body.vertex_groups.new(name=group.name)
            copied.lock_weight = group.lock_weight
            types.add("VGROUP_WEIGHTS")
        for color in source.color_attributes:
            if color.name not in result.color_attributes:
                result.color_attributes.new(color.name, color.data_type, color.domain)
            types.add("COLOR_VERTEX" if color.domain == "POINT" else "COLOR_CORNER")
        with bpy.context.temp_override(
            object=proxy, active_object=proxy, selected_objects=[proxy, body], selected_editable_objects=[proxy, body]
        ):
            for data_type in sorted(types):
                status = bpy.ops.object.data_transfer(
                    data_type=data_type,
                    use_create=True,
                    layers_select_src="ALL",
                    layers_select_dst="NAME",
                    vert_mapping="POLYINTERP_NEAREST",
                    loop_mapping="POLYINTERP_NEAREST",
                    use_object_transform=True,
                )
                if "FINISHED" not in status:
                    raise ValueError(f"Could not transfer {data_type} onto the rebuilt surface")
        if source.color_attributes:
            active = source.color_attributes.active_color
            if active:
                result.color_attributes.active_color = result.color_attributes[active.name]
            index = source.color_attributes.render_color_index
            if 0 <= index < len(source.color_attributes):
                result.color_attributes.render_color_index = result.color_attributes.find(
                    source.color_attributes[index].name
                )
    finally:
        body.matrix_world = original
        bpy.data.objects.remove(proxy, do_unlink=True)


def replace_mesh(owner, body):
    """Retain the object, links and painted surface when replacing its generated solid."""
    old, result = owner.data, body.data
    overrides = [(s.link, s.material) for s in owner.material_slots]
    active = owner.active_material_index
    active_group = owner.vertex_groups.active.name if owner.vertex_groups.active else None
    result.materials.clear()
    for material in old.materials:
        result.materials.append(material)
    if old.polygons:
        if len(old.materials) > 1:
            _carry_face_materials(old, result)
        _carry_attributes(owner, body)
    try:
        owner.data = result
        for slot, (link, material) in zip(owner.material_slots, overrides):
            slot.link = link
            if link == "OBJECT":
                slot.material = material
        if overrides:
            owner.active_material_index = active
        if active_group and active_group in owner.vertex_groups:
            owner.vertex_groups.active_index = owner.vertex_groups[active_group].index
    except Exception:
        owner.data = old
        raise
    if old.users == 0 and not old.library:
        bpy.data.meshes.remove(old)


def _sweep_path(feature, placement, refresh=True):
    """Keep named guides editable in the scene and retain the last successfully built path."""
    path = feature["path_curve"]
    if not isinstance(path, str):
        feature.pop("path_reference", None)
        return path
    reference = feature.get("path_reference", {})
    bound = reference.get("object") == path
    available = path in bpy.data.objects
    if bound and reference.get("points") and (not refresh or not available):
        return reference["points"]
    if not available:
        raise ValueError(f"Unknown sweep guide {path}")
    # Keep the guide in the feature's construction coordinates as the finished
    # model is placed elsewhere. Capture this transform once when the guide is bound.
    frame = Matrix(reference["world_to_model"]) if bound else placement.inverted()
    points = [list(frame @ point) for point in geo.path_points(path)]
    feature["path_reference"] = {"object": path, "world_to_model": [list(row) for row in frame], "points": points}
    return points


class FeatureTree:
    def __init__(self, name, recipe=None):
        self.name = name
        self.owner = bpy.context.scene.objects.get(name)
        if self.owner and self.owner.type != "MESH":
            raise ValueError("Feature tree owner must be a mesh")
        if self.owner and self.owner.library and not self.owner.override_library:
            raise ValueError("Editing a linked model requires a local object or a library override")
        if recipe is not None:
            self.recipe = copy.deepcopy(recipe)
        elif self.owner and "excad_tree" in self.owner:
            self.recipe = json.loads(self.owner["excad_tree"])
        else:
            self.recipe = {"version": 1, "sketches": {}, "features": []}
            if self.owner:
                mesh, matrix = self.owner.data, self.owner.matrix_world
                reflected = matrix.to_3x3().determinant() < 0
                if "excad_sdf_recipe" in self.owner:
                    self.recipe["base"] = {"sdf": json.loads(self.owner["excad_sdf_recipe"])}
                else:
                    self.recipe["base"] = {
                        "vertices": [list(matrix @ v.co) for v in mesh.vertices],
                        "faces": [
                            list(reversed(p.vertices[:])) if reflected else list(p.vertices) for p in mesh.polygons
                        ],
                    }
                self.recipe["model_frame"] = [list(row) for row in matrix]
        if self.recipe.get("version", 1) != 1:
            raise ValueError("Unsupported feature tree version")

    def append(self, kind, args):
        feature = dict(args, type=kind)
        sketch = bpy.context.scene.get("excad_active_sketch", "Sketch")
        if kind in {"EXTRUDE", "REVOLVE", "SWEEP"}:
            feature.setdefault("sketch", feature.pop("profile_sketch", sketch))
        ids = feature.get("sketch_ids", []) if kind == "LOFT" else [feature["sketch"]] if "sketch" in feature else []
        for name in ids:
            record = bpy.context.scene.get("excad_sketches", {}).get(name)
            spec = json.loads(record) if record else None
            if spec and spec.get("support") and spec["support"]["object"] == self.name:
                spec["support"]["object"] = "SELF"
            if name not in self.recipe["sketches"]:
                if not record:
                    raise ValueError(f"Unknown sketch {name}")
                self.recipe["sketches"][name] = spec
            elif record and name == sketch:
                # A reset of the unnamed active sketch must not redefine an earlier feature.
                if spec != self.recipe["sketches"][name]:
                    new_name = f"{name}@{len(self.recipe['features']) + 1}"
                    self.recipe["sketches"][new_name] = spec
                    if kind == "LOFT":
                        feature["sketch_ids"] = [new_name if n == name else n for n in ids]
                    else:
                        feature["sketch"] = new_name
        feature.setdefault("id", f"F{len(self.recipe['features']) + 1}")
        self.recipe["features"].append(feature)

    def rebuild(self):
        recipe = self.recipe
        sketches = recipe.get("sketches", {})
        model_frame = Matrix(recipe["model_frame"]) if "model_frame" in recipe else Matrix.Identity(4)
        current_placement = self.owner.matrix_world @ model_frame.inverted() if self.owner else Matrix.Identity(4)
        with geo.construction() as objects:
            base = recipe.get("base")
            if base and "sdf" in base:
                from .sdf_engine import build_sdf

                body = build_sdf(objects, base["sdf"])
                body.data.transform(model_frame)
                if model_frame.to_3x3().determinant() < 0:
                    body.data.flip_normals()
            else:
                body = geo.new_mesh(objects, base["vertices"], base["faces"], orient=False) if base else None
            for i, feature in enumerate(recipe["features"]):
                kind = feature["type"].upper()
                try:
                    if kind in {"EXTRUDE", "REVOLVE", "SWEEP", "LOFT"}:
                        ids = feature["sketch_ids"] if kind == "LOFT" else [feature["sketch"]]
                        docs = [
                            sketch_from_spec(sketches[key], body, self.name, model_frame, current_placement)
                            for key in ids
                        ]
                        if kind == "EXTRUDE":
                            # Overlap the supporting face by a few mesh-coordinate ULPs.
                            # A microscopically recessed cutter otherwise leaves a zero-thickness lid.
                            overlap = 0
                            if body and docs[0].support:
                                overlap = max(max(body.dimensions), abs(feature["depth"]), 1e-3) * 2e-6
                            tool = geo.extrude(
                                objects,
                                docs[0],
                                feature["depth"],
                                feature.get("symmetric", False),
                                feature.get("draft_angle", 0),
                                overlap,
                            )
                        elif kind == "REVOLVE":
                            tool = geo.revolve(
                                objects,
                                docs[0],
                                feature.get("axis"),
                                feature.get("angle_deg", 360),
                                feature.get("segments", 64),
                                feature.get("origin"),
                            )
                        elif kind == "LOFT":
                            tool = geo.loft(
                                objects,
                                docs,
                                feature.get("segments", 64),
                                feature.get("interpolation", "SMOOTH"),
                                feature.get("steps", 4),
                            )
                        else:
                            tool = geo.sweep(objects, docs[0], _sweep_path(feature, current_placement))
                        op = feature.get("op", "ADD").upper()
                        if body is None:
                            if op != "ADD":
                                raise ValueError("First feature must add a solid")
                            body = tool
                        else:
                            geo.boolean(body, tool, op)
                    elif body is None:
                        raise ValueError("Feature requires a base solid")
                    elif kind in {"FILLET", "CHAMFER"}:
                        width = feature["radius"] if kind == "FILLET" else feature["width"]
                        geo.finish_edges(
                            body,
                            feature.get("edges", "ALL"),
                            width,
                            feature.get("segments", 5) if kind == "FILLET" else 1,
                        )
                    elif kind == "SHELL":
                        geo.shell(body, feature["thickness"], feature.get("open_faces", ["TOP"]))
                    elif kind == "PATTERN":
                        _pattern(objects, body, feature)
                    elif kind == "MIRROR3D":
                        axis = geo.axis_vector(feature.get("axis", "X"))
                        mirror = Matrix.Identity(3)
                        for row in range(3):
                            for col in range(3):
                                mirror[row][col] -= 2 * axis[row] * axis[col]
                        matrix = mirror.to_4x4()
                        matrix.translation = 2 * float(feature.get("offset", 0)) * axis
                        geo.boolean(body, _mesh_copy(objects, body.data, matrix), "ADD")
                    else:
                        raise ValueError(f"Unknown feature {kind}")
                    _solid(body.data)
                except (KeyError, IndexError, ValueError, RuntimeError) as exc:
                    raise ValueError(f"Feature {feature.get('id', i + 1)} ({kind}): {exc}") from exc
            if body is None:
                raise ValueError("Feature tree is empty")
            raw = json.dumps(recipe, separators=(",", ":"), allow_nan=False)
            # The construction context cleans this mesh on failure. A successful
            # assignment to the visible object keeps it alive when the temporary body is removed.
            result = body.data
            # Blender booleans may allocate an empty first slot on construction meshes.
            result.materials.clear()
            if "model_frame" in recipe:
                result.transform(model_frame.inverted())
                if model_frame.to_3x3().determinant() < 0:
                    result.flip_normals()
            if self.owner is None:
                self.owner = bpy.data.objects.new(self.name, result)
                bpy.context.collection.objects.link(self.owner)
                self.owner.matrix_world = model_frame
            else:
                replace_mesh(self.owner, body)
            result.name = f"{self.owner.name} Solid"
            self.owner["excad_tree"] = raw
            if base and "sdf" in base:
                self.owner["excad_sdf_recipe"] = json.dumps(base["sdf"], separators=(",", ":"), allow_nan=False)
            elif "excad_sdf_recipe" in self.owner:
                del self.owner["excad_sdf_recipe"]
            if self.owner.name in bpy.context.view_layer.objects:
                bpy.context.view_layer.objects.active = self.owner
                self.owner.select_set(True)
        return self.owner


def save_recipe(obj, path):
    path = Path(bpy.path.abspath(path)).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.loads(obj["excad_tree"])
    for sketch in raw.get("sketches", {}).values():
        if sketch.get("support") and sketch["support"]["object"] == obj.name:
            sketch["support"]["object"] = "SELF"
    for feature in raw.get("features", []):
        if feature["type"].upper() == "SWEEP":
            feature["path_curve"] = _sweep_path(feature, Matrix.Identity(4), refresh=False)
            feature.pop("path_reference", None)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", dir=path.parent, delete=False) as f:
        temp = f.name
        try:
            json.dump(raw, f, indent=2, allow_nan=False)
        except Exception:
            f.close()
            os.unlink(temp)
            raise
    try:
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
    return str(path)


def dispatch(cmd, data):
    args = {k: v for k, v in data.items() if k != "cmd"}
    doc = SketchDocument.active()
    if cmd == "sketch_on_face":
        name, face = args["object"], args.get("face", "TOP").upper()
        frame = geo.selected_face(bpy.context.scene.objects[name], face)
        bpy.context.scene["excad_active_sketch"] = args.get("sketch_id", "Sketch")
        doc = SketchDocument.active()
        doc.reset("FACE")
        doc.frame, doc.support = (
            frame,
            {
                "object": name,
                "face": face,
                "reference_frame": [list(row) for row in bpy.context.scene.objects[name].matrix_world],
            },
        )
        doc.save()
        return {"plane": doc.plane, "frame": frame}
    if cmd == "sketch_constraint":
        return {"constraint": doc.add_constraint(**args)}
    if cmd.startswith("sketch_") and cmd[7:].removeprefix("add_").upper() in SHAPES:
        return {"entity": doc.add_shape(cmd[7:].removeprefix("add_").upper(), **args)}
    if cmd in {"feature_tree", "rebuild_feature_tree", "save_feature_tree"} or cmd.upper() in FEATURES:
        name = args.pop("object", None) or args.pop("name", None)
        if not name:
            name = doc.support["object"] if doc.support else "Body1"
        if cmd == "save_feature_tree":
            directory = Path(bpy.data.filepath).parent if bpy.data.filepath else Path(bpy.app.tempdir)
            path = args.get("path") or str(directory / "feature_tree.json")
            return {"path": save_recipe(bpy.context.scene.objects[name], path)}
        recipe = args.pop("recipe", None)
        if "path" in args:
            recipe = json.loads(Path(bpy.path.abspath(args.pop("path"))).read_text(encoding="utf8"))
        tree = FeatureTree(name, recipe)
        if cmd.upper() in FEATURES:
            if cmd == "pattern" and "type" in args:
                args["pattern_type"] = args.pop("type")
            tree.append(cmd.upper(), args)
        for key, value in args.get("parameters", {}).items():
            if key.startswith("sketches."):
                parts = key.split(".")
                if len(parts) < 5 or parts[2] not in {"shapes", "points", "entities", "constraints"}:
                    raise ValueError(f"Invalid sketch dimension {key}")
                entry = tree.recipe["sketches"][parts[1]]
                for field in parts[2:-1]:
                    entry = entry[int(field)] if isinstance(entry, list) else entry[field]
                field = int(parts[-1]) if isinstance(entry, list) else parts[-1]
                previous = entry[field]
                if type(previous) not in {int, float} or type(value) not in {int, float} or not math.isfinite(value):
                    raise ValueError(f"Expected a numeric sketch dimension for {key}")
                entry[field] = value
                continue
            feature_id, field = key.split(".", 1)
            matches = [f for f in tree.recipe["features"] if f.get("id") == feature_id]
            if len(matches) != 1 or field not in matches[0] or field in {"type", "id"}:
                raise ValueError(f"Unknown feature parameter {key}")
            matches[0][field] = value
        obj = tree.rebuild()
        return {
            "name": obj.name,
            "features": len(tree.recipe["features"]),
            "verts": len(obj.data.vertices),
            "faces": len(obj.data.polygons),
        }
    return None
