# SPDX-License-Identifier: GPL-2.0-or-later
"""Signed-distance composition and surface nets on a world-unit sampling grid."""

import json
import math

import bpy
import numpy as np
from mathutils import Euler, Vector

from . import feature_geometry as geo
from .feature_tree import _solid, replace_mesh


def _vec(value):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("Expected a finite 3D vector")
    return result


def _smooth_min(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


def _rotate(p, angle):
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = p
    return (c * x + s * y, -s * x + c * y, z)


def _corners(bounds):
    return np.array([[x, y, z] for x in bounds[:, 0] for y in bounds[:, 1] for z in bounds[:, 2]])


def compile_node(node, depth=0):
    if depth > 40:
        raise ValueError("SDF tree exceeds maximum nesting")
    kind = str(node.get("type", node.get("op", ""))).lower()
    origin = _vec(node.get("center", (0, 0, 0)))
    if kind in {"sphere", "ellipsoid", "round_box", "torus", "cone", "capsule"}:
        if kind == "sphere":
            radius = float(node.get("radius", 1))
            extent = _vec([radius] * 3)

            def sample(p):
                return np.sqrt(sum(q * q for q in p)) - radius
        elif kind == "ellipsoid":
            extent = _vec(node["radii"])

            def sample(p):
                k0 = np.sqrt(sum((q / r) ** 2 for q, r in zip(p, extent)))
                k1 = np.sqrt(sum((q / r**2) ** 2 for q, r in zip(p, extent)))
                return np.where(k1 > 1e-8, k0 * (k0 - 1) / np.maximum(k1, 1e-8), -min(extent))
        elif kind == "round_box":
            extent = _vec(node["size"]) / 2
            radius = float(node.get("radius", min(extent) * 0.15))
            if not 0 <= radius <= min(extent):
                raise ValueError("Box radius exceeds half the smallest dimension")

            def sample(p):
                q = [np.abs(x) - s + radius for x, s in zip(p, extent)]
                return np.sqrt(sum(np.maximum(x, 0) ** 2 for x in q)) + np.minimum(np.maximum.reduce(q), 0) - radius
        elif kind == "torus":
            major, minor = node["major_radius"], node["minor_radius"]
            if not 0 < minor < major:
                raise ValueError("Torus needs 0 < minor_radius < major_radius")
            extent = _vec((major + minor, major + minor, minor))

            def sample(p):
                return np.hypot(np.hypot(p[0], p[1]) - major, p[2]) - minor
        elif kind == "capsule":
            a, b = _vec(node["a"]), _vec(node["b"])
            segment = b - a
            length2 = segment.dot(segment)
            radius = float(node["radius"])
            extent = _vec([radius] * 3)

            def sample(p):
                q = [x - c for x, c in zip(p, a)]
                t = np.clip(sum(x * y for x, y in zip(q, segment)) / max(length2, 1e-12), 0, 1)
                return np.sqrt(sum((x - t * s) ** 2 for x, s in zip(q, segment))) - radius
        else:
            height, lower = float(node["height"]), float(node["radius"])
            upper = float(node.get("top_radius", 0))
            if lower <= 0 or upper < 0:
                raise ValueError("Invalid cone radii")
            extent = _vec((max(lower, upper), max(lower, upper), height / 2))

            def sample(p):
                r, z = np.hypot(p[0], p[1]), p[2]
                cap_r, cap_z = r - np.minimum(r, np.where(z < 0, lower, upper)), np.abs(z) - height / 2
                dx, dz = upper - lower, height
                t = np.clip(((upper - r) * dx + (height / 2 - z) * dz) / (dx * dx + dz * dz), 0, 1)
                side_r, side_z = r - upper + dx * t, z - height / 2 + dz * t
                sign = np.where((side_r < 0) & (cap_z < 0), -1, 1)
                return sign * np.sqrt(np.minimum(cap_r**2 + cap_z**2, side_r**2 + side_z**2))

        if np.min(extent) <= 0:
            raise ValueError("SDF dimensions must be positive")
        bounds = np.stack((-extent, extent))
        if kind == "capsule":
            bounds += np.stack((np.minimum(a, b), np.maximum(a, b)))
    elif kind in {"smooth_union", "smooth_subtract", "smooth_intersect"}:
        children = [compile_node(c, depth + 1) for c in node["children"]]
        if len(children) < 2:
            raise ValueError(f"{kind} needs at least two operands")
        k = float(node.get("k", 0.1))
        if k < 0:
            raise ValueError("Blend distance cannot be negative")

        def sample(p):
            val = children[0][0](p)
            for child, _ in children[1:]:
                b = child(p)
                if kind == "smooth_union":
                    val = _smooth_min(val, b, k)
                else:
                    val = -_smooth_min(-val, b if kind == "smooth_subtract" else -b, k)
            return val

        bounds = children[0][1].copy()
        if kind == "smooth_union":
            for _, bb in children[1:]:
                bounds = np.stack((np.minimum(bounds[0], bb[0]) - k / 4, np.maximum(bounds[1], bb[1]) + k / 4))
    elif kind in {"mirror_x", "radial_array", "twist", "bend"}:
        child, bounds = compile_node(node["child"], depth + 1)
        corners = _corners(bounds)
        radius = float(np.max(np.hypot(corners[:, 0], corners[:, 1])))
        if kind == "mirror_x":

            def sample(p):
                return np.minimum(child(p), child((-p[0], p[1], p[2])))

            width = max(abs(bounds[:, 0]))
            bounds[:, 0] = (-width, width)
        elif kind == "radial_array":
            count = int(node["count"])
            if not 1 <= count <= 128:
                raise ValueError("Radial array requires 1 to 128 copies")

            def sample(p):
                value = child(p)
                for i in range(1, count):
                    value = np.minimum(value, child(_rotate(p, math.tau * i / count)))
                return value

            bounds[:, :2] = [[-radius, -radius], [radius, radius]]
        elif kind == "twist":
            rate = math.radians(node["degrees_per_unit"])

            def sample(p):
                return child(_rotate(p, rate * p[2]))

            bounds[:, :2] = [[-radius, -radius], [radius, radius]]
        else:
            bend_radius = float(node["radius"])
            if bend_radius <= max(0, -bounds[0, 0]) or max(abs(bounds[:, 2])) >= math.pi * bend_radius:
                raise ValueError("Bend radius is too small for this section")

            def sample(p):
                x, y, z = p
                return child(
                    (np.hypot(x + bend_radius, z) - bend_radius, y, np.arctan2(z, x + bend_radius) * bend_radius)
                )

            start, end = bounds[:, 2] / bend_radius
            angles = [start, end] + [i * math.pi / 2 for i in range(-2, 3) if start < i * math.pi / 2 < end]
            coords = [
                ((bend_radius + x) * math.cos(a) - bend_radius, y, (bend_radius + x) * math.sin(a))
                for x in bounds[:, 0]
                for y in bounds[:, 1]
                for a in angles
            ]
            bounds = np.stack((np.min(coords, 0), np.max(coords, 0)))
    else:
        raise ValueError(f"Unknown SDF operation {kind}")

    if "rotation_deg" in node or "scale" in node:
        rotation = Euler(np.radians(_vec(node.get("rotation_deg", (0, 0, 0)))), "XYZ")
        factor = node.get("scale", 1.0)
        scale = _vec([factor] * 3 if np.isscalar(factor) else factor)
        if np.min(scale) <= 0:
            raise ValueError("SDF scale must be positive")
        rotation = np.asarray(rotation.to_matrix(), dtype=np.float32)
        inverse = rotation.T / scale[:, None]
        corners = (_corners(bounds) * scale) @ rotation.T + origin
        # The smallest singular value keeps the field conservative in the
        # parent frame when smoothing a non-uniformly scaled operand.
        distance_scale = float(np.min(scale))

        def transformed(points):
            relative = [x - c for x, c in zip(points, origin)]
            local = tuple(sum(weight * p for weight, p in zip(row, relative) if weight) for row in inverse)
            return distance_scale * sample(local)

        return transformed, np.stack((corners.min(0), corners.max(0)))

    def field(p):
        return sample(tuple(x - c for x, c in zip(p, origin)))

    return field, bounds + origin


def surface_nets(values, origin, step):
    """One averaged intersection per active cell and one quad per sign-changing grid edge."""
    dims = np.array(values.shape) - 1
    corners = np.array([(i, j, k) for i in range(2) for j in range(2) for k in range(2)])
    samples = [values[tuple(slice(int(a), int(a + n)) for a, n in zip(c, dims))] for c in corners]
    mask = values[:-1, :-1, :-1] < 0
    active = np.logical_or.reduce([(v < 0) != mask for v in samples[1:]])
    cells = np.stack(np.nonzero(active), axis=1)
    if not len(cells):
        raise ValueError("SDF surface is empty")
    ids = np.full(dims, -1, np.int32)
    ids[active] = np.arange(len(cells))
    shifts = np.zeros((len(cells), 3))
    counts = np.zeros(len(cells))
    for a, ca in enumerate(corners):
        for axis in range(3):
            if ca[axis]:
                continue
            b = a + (4, 2, 1)[axis]
            va, vb = samples[a][active], samples[b][active]
            hit = (va < 0) != (vb < 0)
            frac = va[hit] / (va[hit] - vb[hit])
            shifts[hit] += ca
            shifts[hit, axis] += frac
            counts[hit] += 1
    points = origin + step * (cells + shifts / counts[:, None])
    faces = []
    for axis in range(3):
        signs = np.diff(values < 0, axis=axis)
        indices = np.stack(np.nonzero(signs), axis=1)
        if not len(indices):
            continue
        u, v = (axis + 1) % 3, (axis + 2) % 3
        ring = []
        for du, dv in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
            j = indices.copy()
            j[:, u] += du
            j[:, v] += dv
            ring.append(ids[tuple(j.T)])
        q = np.stack(ring, axis=1)
        reverse = values[tuple(indices.T)] >= 0
        q[reverse] = q[reverse, ::-1]
        faces.extend(q.tolist())
    return points.tolist(), faces


def _json_vector(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, (Vector, Euler)):
        return list(value)
    raise TypeError(f"Unsupported SDF value {type(value).__name__}")


def _set_dimension(tree, path, value):
    parts = path.split(".")
    entry = tree
    try:
        for field in parts[:-1]:
            entry = entry[int(field)] if isinstance(entry, list) and field.isdecimal() else entry[field]
        last = int(parts[-1]) if isinstance(entry, list) and parts[-1].isdecimal() else parts[-1]
        previous = entry[last]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"Unknown SDF parameter {path}") from exc
    before = previous if isinstance(previous, list) else [previous]
    after = value if isinstance(value, list) else [value]
    if (
        isinstance(previous, list) != isinstance(value, list)
        or len(before) != len(after)
        or any(type(v) not in (float, int) or not math.isfinite(v) for v in (*before, *after))
    ):
        raise ValueError(f"Expected a numeric dimension with the same shape for {path}")
    entry[last] = value


def build_sdf(objects, recipe):
    """Build a temporary solid for both an SDF edit and a dependent CAD history."""
    if recipe.get("version", 1) != 1:
        raise ValueError("Unsupported SDF recipe version")
    tree = recipe["tree"]
    resolution, remesh, smooth = recipe.get("resolution", 96), recipe.get("remesh", "AUTO"), recipe.get("smooth", True)
    if type(resolution) is not int or not 16 <= resolution <= 256 or remesh not in {"AUTO", "QUAD", "VOXEL"}:
        raise ValueError("Use resolution 16..256 and remesh AUTO, QUAD or VOXEL")
    if type(smooth) is not bool:
        raise ValueError("Specify true or false for SDF smoothing")
    field, bounds = compile_node(tree)
    step = float(np.max(bounds[1] - bounds[0])) / resolution
    origin = bounds[0] - 2 * step
    dims = np.ceil((bounds[1] - bounds[0]) / step).astype(int) + 5
    axes = [origin[i] + step * np.arange(n, dtype=np.float32) for i, n in enumerate(dims)]
    values = np.empty(dims, dtype=np.float32)
    for i in range(0, dims[0], 12):
        grid = np.meshgrid(axes[0][i : i + 12], axes[1], axes[2], indexing="ij")
        values[i : i + 12] = field(grid)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite SDF field")
    points, faces = surface_nets(values, origin, step)
    # The field gradient orients enclosed voids inward. Flood-fill orientation
    # would turn each disconnected inner surface outward and add its volume.
    body = geo.new_mesh(objects, points, faces, orient=False)
    voxel = remesh == "VOXEL"
    if not voxel:
        try:
            _solid(body.data)
        except ValueError:
            voxel = True
    if voxel:
        body.data.remesh_voxel_size = step * 0.85
        with bpy.context.temp_override(object=body, active_object=body):
            bpy.ops.object.voxel_remesh()
        _solid(body.data)
    for poly in body.data.polygons:
        poly.use_smooth = smooth
    return body


def generate_sdf(
    tree=None, resolution=None, name="EXCAD SDF", remesh=None, smooth=None, *, target=None, parameters=None
):
    """Create a new solid or rebuild an existing object's stored SDF recipe in its local frame."""
    owner = bpy.context.scene.objects[target] if isinstance(target, str) else target
    if owner and (owner.type != "MESH" or owner.mode != "OBJECT" or owner.library and not owner.override_library):
        raise ValueError("Rebuilding an SDF requires a local mesh object in Object Mode")
    if owner and owner.data.shape_keys:
        raise ValueError("Rebuilding an SDF changes topology and cannot preserve shape keys")
    saved = json.loads(owner.get("excad_sdf_recipe", "{}")) if owner else {}
    if saved.get("version", 1) != 1:
        raise ValueError("Unsupported SDF recipe version")
    tree = tree if tree is not None else saved.get("tree")
    if not isinstance(tree, dict):
        raise ValueError("Supply an SDF tree to create a solid or rebuild a mesh without a saved recipe")
    tree = json.loads(json.dumps(tree, allow_nan=False, default=_json_vector))
    for key, value in (parameters or {}).items():
        _set_dimension(tree, key, value)
    resolution = resolution if resolution is not None else saved.get("resolution", 96)
    remesh = remesh if remesh is not None else saved.get("remesh", "AUTO")
    smooth = smooth if smooth is not None else saved.get("smooth", True)
    recipe = dict(version=1, tree=tree, resolution=resolution, remesh=remesh, smooth=smooth)
    raw = json.dumps(recipe, separators=(",", ":"), allow_nan=False)
    if owner and "excad_tree" in owner:
        from .feature_tree import FeatureTree

        cad = json.loads(owner["excad_tree"])
        base = cad.get("base") or {}
        if "sdf" not in base:
            raise ValueError("This solid uses a CAD recipe. Edit it with rebuild_feature_tree or create a named SDF")
        base["sdf"] = recipe
        return FeatureTree(owner.name, cad).rebuild()
    with geo.construction() as objects:
        body = build_sdf(objects, recipe)
        if owner is None:
            body.name = name
            objects.remove(body)
            owner = body
        else:
            replace_mesh(owner, body)
        owner.data.name = owner.name + " SDF"
        owner["excad_sdf_recipe"] = raw
    if owner.name in bpy.context.view_layer.objects:
        bpy.context.view_layer.objects.active = owner
        owner.select_set(True)
    return owner
