# SPDX-License-Identifier: GPL-2.0-or-later
"""Draw the active parametric sketch in each 3D view; cache solver diagnostics between edits."""

import bpy
import numpy as np
from mathutils import Vector

from .sketch_engine import SketchDocument
from .sketch_profiles import circumcenter, paths_and_loops

_handles = []
_cache_key = None
_cache = None
_gpu_key = _shader = None
_batches = {}


def constrained_variables(doc):
    n = 2 * len(doc.points)
    state = np.array([v for p in doc.points for v in (p.u, p.v)] + [e.radius for e in doc.entities])
    free = [i for i in range(n) if not doc.points[i // 2].fixed]
    free += [n + e.eid for e in doc.entities if e.etype in {"CIRCLE", "ARC"} and "through" not in e.params]
    constrained = set(range(len(state))) - set(free)
    if not free or not doc.constraints:
        return constrained
    r = doc._eval_residuals(state)
    if np.max(np.abs(r), initial=0) > 1e-5:
        return constrained
    jac = np.empty((len(r), len(free)))
    for col, index in enumerate(free):
        delta = state.copy()
        delta[index] += 1e-6
        jac[:, col] = (doc._eval_residuals(delta) - r) / 1e-6
    _, singular, vh = np.linalg.svd(jac, full_matrices=len(r) < len(free))
    nullspace = vh[np.count_nonzero(singular > 1e-5) :]
    return constrained | {index for col, index in enumerate(free) if np.linalg.norm(nullspace[:, col]) < 1e-5}


def drawing():
    global _cache_key, _cache
    doc = SketchDocument.active()
    key = SketchDocument._source
    if key == _cache_key:
        return _cache
    fixed = constrained_variables(doc)
    lines = {False: [], True: []}
    labels = []
    previous_states = []
    dependencies = {}
    radius_constraints = {c.a for c in doc.constraints if c.ctype == "RADIUS"}
    for e, pts, closed in paths_and_loops(doc, dependencies=dependencies)[0]:
        ids = {i for i in (e.p0, e.p1, e.params.get("through", -1), *e.params.get("points", [])) if i >= 0}
        locked = all({2 * i, 2 * i + 1} <= fixed for i in ids)
        if e.etype in {"ARC", "CIRCLE"} and "through" not in e.params:
            locked = locked and 2 * len(doc.points) + e.eid in fixed
        if e.eid in dependencies:
            locked = all(previous_states[i] for i in dependencies[e.eid])
        previous_states.append(locked)
        world = [doc.plane_to_world(*p, 0) for p in pts]
        for a, b in zip(world, world[1:] + ([world[0]] if closed else [])):
            lines[locked].extend((a, b))
        if e.etype == "SLOT":
            a, b = (Vector((doc.points[i].u, doc.points[i].v)) for i in (e.p0, e.p1))
            location = (a + b) / 2
            label = f"{(a - b).length + 2 * e.radius:.4g} × {2 * e.radius:.4g}"
        elif e.etype == "REGULAR_POLYGON":
            location = pts[0]
            label = f"R {e.radius:.4g}"
        elif e.etype == "ARC" and e.eid not in radius_constraints:
            location = pts[len(pts) // 2]
            radius = e.radius
            if "through" in e.params:
                center = circumcenter(pts[0], location, pts[-1])
                radius = (pts[0] - center).length
            label = f"R {radius:.4g}"
        elif e.etype == "OFFSET_LOOP":
            location = (pts[0] + pts[1]) / 2
            label = f"Offset {e.params['distance']:.4g}"
        else:
            continue
        labels.append((Vector(doc.plane_to_world(*location, 0)), label))
    for c in doc.constraints:
        if c.ctype not in {"RADIUS", "DISTANCE", "ANGLE"}:
            continue
        a = doc.points[doc.entities[c.a].p0] if c.ctype == "RADIUS" else doc.points[c.a]
        b = a if c.ctype == "RADIUS" else doc.points[c.b]
        pos = doc.plane_to_world((a.u + b.u) / 2, (a.v + b.v) / 2, 0)
        label = {"RADIUS": "R ", "DISTANCE": "", "ANGLE": "°"}[c.ctype]
        labels.append((Vector(pos), f"{c.value:g}°" if c.ctype == "ANGLE" else f"{label}{c.value:g}"))
    _cache_key, _cache = key, (lines, labels)
    return _cache


def _draw_lines():
    global _gpu_key, _shader, _batches
    import gpu
    from gpu_extras.batch import batch_for_shader

    lines, _ = drawing()
    if _gpu_key != _cache_key:
        _shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        _batches = {
            locked: batch_for_shader(_shader, "LINES", {"pos": verts}) for locked, verts in lines.items() if verts
        }
        _gpu_key = _cache_key
    depth = gpu.state.depth_test_get()
    try:
        gpu.state.depth_test_set("NONE")
        for locked, batch in _batches.items():
            _shader.bind()
            _shader.uniform_float("color", (0.02, 0.02, 0.02, 1) if locked else (0.12, 0.42, 1, 1))
            batch.draw(_shader)
    finally:
        gpu.state.depth_test_set(depth)


def _draw_labels(project):
    import blf

    for pos, label in drawing()[1]:
        xy = project(pos)
        if xy is not None:
            blf.position(0, xy.x + 3, xy.y + 4, 0)
            blf.size(0, 12)
            blf.color(0, 0.9, 0.85, 0.4, 1)
            blf.draw(0, label)


def _draw_text():
    from bpy_extras.view3d_utils import location_3d_to_region_2d

    context = bpy.context
    if context.region_data is not None:
        _draw_labels(lambda pos: location_3d_to_region_2d(context.region, context.region_data, pos))


def register():
    if not bpy.app.background and not _handles:
        for func, stage in ((_draw_lines, "POST_VIEW"), (_draw_text, "POST_PIXEL")):
            _handles.append(bpy.types.SpaceView3D.draw_handler_add(func, (), "WINDOW", stage))


def unregister():
    global _gpu_key, _shader
    _gpu_key = _shader = None
    _batches.clear()
    while _handles:
        bpy.types.SpaceView3D.draw_handler_remove(_handles.pop(), "WINDOW")
