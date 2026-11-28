# SPDX-FileCopyrightText: 2026 EX-CAD Authors
# SPDX-License-Identifier: GPL-2.0-or-later

import math
import json
from dataclasses import asdict, dataclass, field

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import sketch_profiles


@dataclass
class SketchPoint:
    pid: int
    u: float
    v: float
    fixed: bool = False


@dataclass
class SketchEntity:
    eid: int
    etype: str  # 'LINE' | 'CIRCLE'
    p0: int
    p1: int = -1
    radius: float = 0.5
    params: dict = field(default_factory=dict)


@dataclass
class SketchConstraint:
    cid: int
    ctype: str
    a: int
    b: int = -1
    c: int = -1
    d: int = -1
    value: float = 0.0
    value2: float = 0.0


class SketchDocument:
    _active = None
    _source = None

    def __init__(self, plane: str = "XY"):
        self.plane = plane
        self.frame = None
        self.support = None
        self.points: list[SketchPoint] = []
        self.entities: list[SketchEntity] = []
        self.constraints: list[SketchConstraint] = []

    @classmethod
    def active(cls):
        scene = bpy.context.scene
        name = scene.get("excad_active_sketch", "Sketch")
        raw = scene.get("excad_sketches", {}).get(name)
        source = scene.session_uid, name, raw
        if cls._source != source or cls._active is None:
            cls._active = cls.deserialize(json.loads(raw)) if raw else cls()
            cls._source = source
        return cls._active

    def save(self):
        scene = bpy.context.scene
        name = scene.get("excad_active_sketch", "Sketch")
        if "excad_sketches" not in scene:
            scene["excad_sketches"] = {}
        raw = json.dumps(self.serialize(), separators=(",", ":"))
        scene["excad_sketches"][name] = raw
        type(self)._source = (scene.session_uid, name, raw)
        for area in getattr(bpy.context.screen, "areas", ()):
            if area.type == "VIEW_3D":
                area.tag_redraw()

    def reset(self, plane: str = "XY"):
        self.plane = plane
        self.frame = None
        self.support = None
        self.points.clear()
        self.entities.clear()
        self.constraints.clear()

    def serialize(self):
        return {"plane": self.plane, "frame": self.frame, "support": self.support,
                "points": [asdict(p) for p in self.points], "entities": [asdict(e) for e in self.entities],
                "constraints": [asdict(c) for c in self.constraints]}

    @classmethod
    def deserialize(cls, data):
        doc = cls(data.get("plane", "XY"))
        doc.frame, doc.support = data.get("frame"), data.get("support")
        doc.points = [SketchPoint(**p) for p in data["points"]]
        doc.entities = [SketchEntity(**e) for e in data["entities"]]
        doc.constraints = [SketchConstraint(**c) for c in data["constraints"]]
        return doc

    def add_shape(self, kind, **args):
        def point(value):
            u, v = value
            return self.add_point(u, v)

        p0 = p1 = -1
        radius = 0.0
        params = {}
        if kind == "ARC" and "points" in args:
            a, b, c = args["points"]
            sketch_profiles.circumcenter_xy(a, b, c)
            p0, params["through"], p1 = point(a), point(b), point(c)
        elif kind in {"ARC", "REGULAR_POLYGON"}:
            radius = float(args["radius"])
            if radius <= 0:
                raise ValueError("Radius must be positive")
            if kind == "ARC":
                params = {"start_deg": float(args.get("start_deg", 0)), "angle_deg": float(args["angle_deg"])}
                if not 0 < abs(params["angle_deg"]) < 360:
                    raise ValueError("An arc sweep must be between 0 and 360 degrees")
            else:
                params = {"sides": int(args["sides"]), "rotation": float(args.get("rotation", 0))}
                if not 3 <= params["sides"] <= 512:
                    raise ValueError("A regular polygon needs 3 to 512 sides")
            p0 = point(args.get("center", (0, 0)))
        elif kind == "SLOT":
            radius = float(args["width"]) / 2
            if "length" in args:
                # Length is the overall end-to-end slot dimension, as on a drawing.
                span = float(args["length"]) - 2*radius
                if span <= 0:
                    raise ValueError("Slot length must exceed its width")
                angle = math.radians(args.get("angle_deg",0))
                delta = Vector((math.cos(angle),math.sin(angle)))*(span/2)
                center = Vector(args.get("center",(0,0)))
                args["start"],args["end"] = center-delta,center+delta
            if radius <= 0 or (Vector(args["end"]) - Vector(args["start"])).length < 1e-7:
                raise ValueError("Slot needs distinct endpoints and positive width")
            p0, p1 = point(args["start"]), point(args["end"])
        elif kind == "SPLINE":
            pts = args["points"]
            closed = bool(args.get("closed", False))
            if len(pts) < (3 if closed else 2):
                raise ValueError("Too few spline control points")
            params = {"points": [point(p) for p in pts], "closed": closed,
                      "steps": max(2, min(64, int(args.get("steps", 8))))}
            p0 = params["points"][0]
        elif kind in {"OFFSET_LOOP", "MIRROR2D"}:
            source = int(args.get("source", 0))
            if kind=="MIRROR2D" and "source_entity" in args:
                index = int(args["source_entity"])
                if not 0 <= index < len(self.entities):
                    raise ValueError("Unknown entity to mirror")
                params = {"source_entity":index}
            else:
                loops = self.build_2d_loops()
                if not 0 <= source < len(loops):
                    raise ValueError("Source must refer to a closed profile")
                params = {"source": source}
            if kind == "OFFSET_LOOP":
                params["distance"] = float(args["distance"])
                sketch_profiles.offset_loop(loops[source], params["distance"])
            else:
                params.update(axis=args.get("axis", "X").upper(), offset=float(args.get("offset", 0)))
                if params["axis"] not in {"X", "Y"}:
                    raise ValueError("Mirror axis must be X or Y")
        else:
            raise ValueError(f"Unknown sketch shape {kind}")
        eid = len(self.entities)
        self.entities.append(SketchEntity(eid, kind, p0, p1, radius, params))
        return eid

    def add_point(self, u: float, v: float, fixed: bool = False) -> int:
        pid = len(self.points)
        self.points.append(SketchPoint(pid, float(u), float(v), fixed))
        return pid

    def add_line_pts(self, p0: int, p1: int) -> int:
        eid = len(self.entities)
        self.entities.append(SketchEntity(eid, "LINE", p0, p1))
        return eid

    def add_line(self, u0: float, v0: float, u1: float, v1: float) -> int:
        p0 = self.add_point(u0, v0)
        p1 = self.add_point(u1, v1)
        return self.add_line_pts(p0, p1)

    def add_rect(self, u0: float, v0: float, width: float, height: float):
        p0 = self.add_point(u0, v0)
        p1 = self.add_point(u0 + width, v0)
        p2 = self.add_point(u0 + width, v0 + height)
        p3 = self.add_point(u0, v0 + height)
        self.add_line_pts(p0, p1)
        self.add_line_pts(p1, p2)
        self.add_line_pts(p2, p3)
        self.add_line_pts(p3, p0)
        self.add_constraint("HORIZONTAL", p0, p1)
        self.add_constraint("HORIZONTAL", p3, p2)
        self.add_constraint("VERTICAL", p1, p2)
        self.add_constraint("VERTICAL", p0, p3)
        self.add_constraint("DISTANCE", p0, p1, value=abs(width))
        self.add_constraint("DISTANCE", p1, p2, value=abs(height))

    def add_circle(self, cu: float, cv: float, radius: float) -> int:
        center = self.add_point(cu, cv)
        eid = len(self.entities)
        self.entities.append(SketchEntity(eid, "CIRCLE", center, -1, abs(radius)))
        self.add_constraint("RADIUS", eid, value=abs(radius))
        return eid

    def add_constraint(
        self,
        ctype: str,
        a: int,
        b: int = -1,
        c: int = -1,
        d: int = -1,
        value: float | None = None,
        value2: float | None = None,
    ) -> int:
        ctype = ctype.upper()
        if ctype in {"TANGENT", "CONCENTRIC", "RADIUS"}:
            ids = (a,) if ctype == "RADIUS" else (a, b)
            if any(not 0 <= i < len(self.entities) for i in ids):
                raise ValueError("Constraint refers to a missing entity")
            kinds = [self.entities[i].etype for i in ids]
            allowed = {"CIRCLE", "ARC"}
            if ctype == "TANGENT":
                allowed.add("LINE")
            if any(k not in allowed for k in kinds) or kinds == ["LINE", "LINE"]:
                raise ValueError(f"Unsupported entities for {ctype}")
            if ctype == "RADIUS" and "through" in self.entities[a].params:
                raise ValueError("A three-point arc is dimensioned by its control points")
        else:
            arities = {"COINCIDENT": 2, "HORIZONTAL": 2, "VERTICAL": 2, "DISTANCE": 2,
                       "PARALLEL": 4, "PERPENDICULAR": 4, "EQUAL_LENGTH": 4, "FIXED": 1,
                       "SYMMETRIC": 4, "MIDPOINT": 3, "ANGLE": 4}
            if ctype not in arities:
                raise ValueError(f"Unknown constraint {ctype}")
            if any(not 0 <= i < len(self.points) for i in (a, b, c, d)[:arities[ctype]]):
                raise ValueError("Constraint refers to a missing point")
        if value is None:
            value = self.points[a].u if ctype == "FIXED" else 0.0
        if value2 is None:
            value2 = self.points[a].v if ctype == "FIXED" else 0.0
        cid = len(self.constraints)
        self.constraints.append(SketchConstraint(cid, ctype, a, b, c, d, float(value), float(value2)))
        return cid

    def _eval_residuals(self, state: np.ndarray) -> np.ndarray:
        n_pts = len(self.points)
        res = []
        def point(i):
            return state[2*i:2*i+2]

        def round_entity(index):
            e = self.entities[index]
            center = point(e.p0)
            if "through" in e.params:
                cc = np.array(sketch_profiles.circumcenter_xy(center, point(e.params["through"]), point(e.p1)))
                return cc, np.linalg.norm(center - cc)
            return center, state[2*n_pts + index]

        for con in self.constraints:
            t = con.ctype
            if t == "MIDPOINT":
                res.extend(point(con.a) - (point(con.b) + point(con.c)) * 0.5)
            elif t in {"ANGLE", "SYMMETRIC"}:
                a, b, c, d = map(point, (con.a, con.b, con.c, con.d))
                u, v = b - a, d - c
                if np.linalg.norm(v) < 1e-10:
                    raise ValueError(f"Degenerate reference line for {t}")
                if t == "ANGLE":
                    measured = math.atan2(u[0]*v[1] - u[1]*v[0], np.dot(u, v))
                    diff = measured - math.radians(con.value)
                    res.append(math.atan2(math.sin(diff), math.cos(diff)))
                else:
                    unit = v / np.linalg.norm(v)
                    mid = (a + b) * 0.5 - c
                    res.extend((mid[0]*unit[1] - mid[1]*unit[0], np.dot(u, unit)))
            elif t == "CONCENTRIC":
                res.extend(round_entity(con.a)[0] - round_entity(con.b)[0])
            elif t == "TANGENT":
                i, j = con.a, con.b
                if self.entities[j].etype == "LINE":
                    i, j = j, i
                center, radius = round_entity(j)
                e = self.entities[i]
                if e.etype == "LINE":
                    u, v = point(e.p1) - point(e.p0), center - point(e.p0)
                    res.append(abs(u[0]*v[1] - u[1]*v[0]) / max(1e-12, np.linalg.norm(u)) - radius)
                else:
                    c, r = round_entity(i)
                    distance = abs(radius - r) if con.value < 0 else radius + r
                    res.append(np.linalg.norm(center - c) - distance)
            elif t == "COINCIDENT":
                res.append(state[2 * con.a] - state[2 * con.b])
                res.append(state[2 * con.a + 1] - state[2 * con.b + 1])
            elif t == "HORIZONTAL":
                res.append(state[2 * con.a + 1] - state[2 * con.b + 1])
            elif t == "VERTICAL":
                res.append(state[2 * con.a] - state[2 * con.b])
            elif t == "DISTANCE":
                du = state[2 * con.b] - state[2 * con.a]
                dv = state[2 * con.b + 1] - state[2 * con.a + 1]
                res.append(math.sqrt(du * du + dv * dv + 1e-12) - con.value)
            elif t == "RADIUS":
                res.append(state[2 * n_pts + con.a] - con.value)
            elif t == "PARALLEL":
                dx1 = state[2 * con.b] - state[2 * con.a]
                dy1 = state[2 * con.b + 1] - state[2 * con.a + 1]
                dx2 = state[2 * con.d] - state[2 * con.c]
                dy2 = state[2 * con.d + 1] - state[2 * con.c + 1]
                res.append(dx1 * dy2 - dy1 * dx2)
            elif t == "PERPENDICULAR":
                dx1 = state[2 * con.b] - state[2 * con.a]
                dy1 = state[2 * con.b + 1] - state[2 * con.a + 1]
                dx2 = state[2 * con.d] - state[2 * con.c]
                dy2 = state[2 * con.d + 1] - state[2 * con.c + 1]
                res.append(dx1 * dx2 + dy1 * dy2)
            elif t == "EQUAL_LENGTH":
                d1 = math.hypot(state[2 * con.b] - state[2 * con.a], state[2 * con.b + 1] - state[2 * con.a + 1])
                d2 = math.hypot(state[2 * con.d] - state[2 * con.c], state[2 * con.d + 1] - state[2 * con.c + 1])
                res.append(d1 - d2)
            elif t == "FIXED":
                res.append(state[2 * con.a] - con.value)
                res.append(state[2 * con.a + 1] - con.value2)
        return np.array(res, dtype=np.float64)

    def solve(self, max_iters: int = 80, tol: float = 1e-7) -> tuple[bool, int, float]:
        n_pts = len(self.points)
        n_ent = len(self.entities)
        n_vars = 2 * n_pts + n_ent
        if n_vars == 0 or not self.constraints:
            return True, 0, 0.0

        state = np.zeros(n_vars, dtype=np.float64)
        fixed = np.zeros(n_vars, dtype=bool)
        for i, p in enumerate(self.points):
            state[2 * i] = p.u
            state[2 * i + 1] = p.v
            if p.fixed:
                fixed[2 * i] = True
                fixed[2 * i + 1] = True
        for i, e in enumerate(self.entities):
            state[2 * n_pts + i] = e.radius
            if e.etype not in {"CIRCLE", "ARC"} or "through" in e.params:
                fixed[2 * n_pts + i] = True

        eps = 1e-6
        converged = False
        iters = 0
        max_res = 0.0

        for it in range(max_iters):
            r0 = self._eval_residuals(state)
            iters = it
            if len(r0) == 0:
                converged = True
                break
            max_res = float(np.max(np.abs(r0)))
            if max_res < tol:
                converged = True
                break

            J = np.zeros((len(r0), n_vars), dtype=np.float64)
            for col in range(n_vars):
                if fixed[col]:
                    continue
                orig = state[col]
                state[col] = orig + eps
                r1 = self._eval_residuals(state)
                state[col] = orig
                J[:, col] = (r1 - r0) / eps

            A = J.T @ J + 1e-6 * np.eye(n_vars)
            b = -J.T @ r0
            dx = np.linalg.lstsq(A, b, rcond=None)[0]
            dx[fixed] = 0.0
            state += np.clip(dx, -1.0, 1.0)

        for i, p in enumerate(self.points):
            p.u = float(state[2 * i])
            p.v = float(state[2 * i + 1])
        for i, e in enumerate(self.entities):
            if e.etype in {"CIRCLE", "ARC"} and "through" not in e.params:
                e.radius = max(1e-4, float(state[2 * n_pts + i]))
        return converged, iters, max_res

    def plane_to_world(self, u: float, v: float, depth: float) -> tuple[float, float, float]:
        if self.frame is not None:
            return tuple(Matrix(self.frame) @ Vector((u, v, depth)))
        if self.plane == "XZ":
            return (float(u), float(depth), float(v))
        if self.plane == "YZ":
            return (float(depth), float(u), float(v))
        return (float(u), float(v), float(depth))

    def build_2d_loops(self, circle_segments: int = 48):
        return sketch_profiles.paths_and_loops(self, circle_segments)[1]


def _new_feature(doc, name, feature):
    from .feature_tree import FeatureTree
    if not doc.build_2d_loops():
        return None
    base = name
    suffix = 1
    while name in bpy.data.objects:
        name = f"{base}.{suffix:03d}"
        suffix += 1
    recipe = {"version": 1, "sketches": {"profile": doc.serialize()},
              "features": [dict(feature, id="F1", sketch="profile")]}
    return FeatureTree(name, recipe).rebuild()


def extrude_sketch_doc(doc, depth, symmetric, name="EXCAD_Extrude"):
    return _new_feature(doc, name, {"type": "EXTRUDE", "depth": depth, "symmetric": symmetric})


def revolve_sketch_doc(doc, angle_deg, segments, name="EXCAD_Revolve"):
    return _new_feature(doc, name, {"type": "REVOLVE", "angle_deg": angle_deg, "segments": segments})
