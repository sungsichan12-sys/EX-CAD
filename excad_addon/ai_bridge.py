# SPDX-FileCopyrightText: 2026 EX-CAD Authors
# SPDX-License-Identifier: GPL-2.0-or-later

import hashlib
import json
import os
import queue
import shlex
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import bpy
from .protocol import dsl_lines, json_command
from . import appearance, feature_tree, scene_layout, scene_state
from .scene_history import SceneHistory

from . import asset_fetcher, character_stylizer, city_generator, human_engine, image_sketch_engine, oneshot_generators, polyhedron_engine, sketch_engine


def _discard_download(future):
    if not future.cancelled() and future.exception() is None:
        from pathlib import Path
        path = future.result()
        if isinstance(path, Path) and path.suffix == ".glb":
            path.unlink(missing_ok=True)


def _jdump(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


COMPACT_SCHEMA = (
    "JSONL 127.0.0.1:9877. schema{topic?:sketch|cad|character|organic|scene}. batch{steps:[{cmd,...}],return_all?}; "
    "dsl{code:'command {JSON args};command {JSON args}'}; scene_digest{since_hash?}. "
    "Shortcuts: S plane; R u v w h; C u v r; L u v u v; SOLVE; EX depth sym name; "
    "RV deg segments name; H preset; IMG front; POLY faces dir; FIND/USE query; "
    "STAGE; POSE name; PART type size; PROP type scale; STUDIO; CITY; STYLE; DIGEST. "
    "CAD: sketch_new/on_face/constraint, sketch_{arc,slot,regular_polygon,spline,offset_loop,mirror2d}; "
    "extrude/revolve/loft/sweep/fillet/chamfer/shell/pattern/mirror3d; "
    "feature_tree{name,recipe}, save_feature_tree/rebuild_feature_tree. render_inspect{target?,views?}. "
    "Portrait: fit_face_from_image{object,image_path}, character_stylize{object,hair?,clothes?,skin?,expression?,hands?}, "
    "install_asset_pack/equip_asset. SDF: sdf_generate{tree,resolution?}; poly_render/poly_reconstruct/poly_bake; "
    "generate_3d{prompt?,image_path?,provider}. Layout: place_relative/look_at/scatter_on_surface/"
    "check_intersections/resolve_overlaps; set_material/light_studio; checkpoint/rollback{name}, undo, reload_addon."
)

_POLY_MAP = {
    "6": "CUBE_6",
    "8": "OCTAHEDRON_8",
    "12": "DODECAHEDRON_12",
    "20": "ICOSAHEDRON_20",
}


READ_ONLY = {"ping", "schema", "list", "scene_digest", "render_inspect", "check_intersections",
             "human_list_targets", "asset_search", "asset_list_cached", "install_asset_pack", "save_feature_tree", "poly_render"}
LAYOUT = {fn.__name__: fn for fn in (scene_layout.place_relative, scene_layout.look_at,
          scene_layout.scatter_on_surface, scene_layout.check_intersections, scene_layout.resolve_overlaps,
          appearance.set_material, appearance.light_studio)}


class AiBridgeServer:
    _instance = None

    def __init__(self):
        self.running = False
        self.port = 9877
        self._socket = None
        self._poll_callback = lambda: self._timer_poll()
        self._pending = None
        self._pool = None
        self._remembered = False
        self.history = SceneHistory()
        self._sketch_cache = {}
        self._thread = None
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._digest_cache: dict[str, dict[str, list]] = {}

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(self, port: int = 9877):
        if self.running:
            return True
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
            sock.listen(8)
            sock.settimeout(0.2)
        except OSError:
            sock.close()
            raise
        self.port = sock.getsockname()[1]
        self._socket = sock
        self._stop_event = threading.Event()
        self._queue = queue.Queue()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="EXCAD IO")
        self.running = True
        self._thread = threading.Thread(target=self._server_loop, args=(sock, self._queue, self._stop_event), daemon=True)
        self._thread.start()
        if not bpy.app.timers.is_registered(self._poll_callback):
            bpy.app.timers.register(self._poll_callback, first_interval=0.02, persistent=True)
        return True

    def stop(self):
        self.running = False
        if getattr(self, "_stop_event", None) is not None:
            self._stop_event.set()
        if self._pending:
            program, future, holder, event = self._pending
            program.close()
            if not future.cancel():
                future.add_done_callback(_discard_download)
            holder["resp"] = _jdump({"ok":False,"error":"Bridge stopped"})
            event.set()
            self._pending = None
        while True:
            try:
                _, holder, event = self._queue.get_nowait()
            except queue.Empty:
                break
            holder["resp"] = _jdump({"ok": False, "error": "Bridge stopped"})
            event.set()
        if self._pool:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        if self._socket:
            self._socket.close()
            self._socket = None
        if bpy.app.timers.is_registered(self._poll_callback):
            bpy.app.timers.unregister(self._poll_callback)

    def _timer_poll(self):
        if self._pending:
            program, future, holder, event = self._pending
            if not future.done():
                return 0.02
            self._pending = None
        elif not self._queue.empty():
            payload, holder, event = self._queue.get_nowait()
            try:
                data = json.loads(payload)
            except ValueError as exc:
                holder["resp"] = _jdump({"ok": False, "error": str(exc)})
                event.set()
                return 0.02
            self._remembered = False
            program, future = self._steps(data), None
        else:
            return 0.02 if self.running else None
        try:
            if future is None:
                action = next(program)
            elif future.exception() is not None:
                action = program.throw(future.exception())
            else:
                action = program.send(future.result())
            self._pending = program, self._pool.submit(action), holder, event
        except StopIteration as done:
            holder["resp"] = _jdump(done.value)
            event.set()
        except Exception as exc:
            holder["resp"] = _jdump({"ok": False, "error": str(exc)})
            event.set()
        return 0.02 if self.running else None

    def _build_digest(self, include_all: bool = False):
        rows, sketch = scene_state.rows(include_all)
        raw = _jdump({"o": rows, "s": sketch})
        h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        return h, rows, sketch

    def _parse_dsl_line(self, line: str) -> dict:
        command = json_command(line)
        if command is not None:
            return command
        toks = shlex.split(line)
        if not toks:
            return {"ok": True}
        op = toks[0].upper()

        if op == "S":
            return {"cmd": "sketch_new", "plane": toks[1] if len(toks) > 1 else "XY"}
        if op == "R":
            return {
                "cmd": "sketch_add_rect",
                "u0": float(toks[1]), "v0": float(toks[2]),
                "width": float(toks[3]), "height": float(toks[4]),
            }
        if op == "C":
            return {
                "cmd": "sketch_add_circle",
                "cu": float(toks[1]), "cv": float(toks[2]), "radius": float(toks[3]),
            }
        if op == "L":
            return {
                "cmd": "sketch_add_line",
                "u0": float(toks[1]), "v0": float(toks[2]),
                "u1": float(toks[3]), "v1": float(toks[4]),
            }
        if op == "SOLVE":
            return {"cmd": "sketch_solve"}
        if op == "EX":
            return {
                "cmd": "sketch_extrude",
                "depth": float(toks[1]) if len(toks) > 1 else 1.0,
                "symmetric": bool(int(toks[2])) if len(toks) > 2 else False,
                "name": toks[3] if len(toks) > 3 else "EXCAD_Extrude",
            }
        if op == "RV":
            return {
                "cmd": "sketch_revolve",
                "angle_deg": float(toks[1]) if len(toks) > 1 else 360.0,
                "segments": int(toks[2]) if len(toks) > 2 else 48,
                "name": toks[3] if len(toks) > 3 else "EXCAD_Revolve",
            }
        if op == "H":
            return {
                "cmd": "human_generate",
                "preset": toks[1] if len(toks) > 1 else "ATHLETIC_FEMALE",
                "ethnicity": toks[2] if len(toks) > 2 else "BALANCED",
                "rig": toks[3] if len(toks) > 3 else "default",
                "apply_skin": bool(int(toks[4])) if len(toks) > 4 else True,
                "subdiv_level": int(toks[5]) if len(toks) > 5 else 1,
            }
        if op == "IMG":
            side = "" if (len(toks) <= 2 or toks[2] == "-") else toks[2]
            return {
                "cmd": "image_to_mesh",
                "front_image": toks[1],
                "side_image": side,
                "mode": toks[3] if len(toks) > 3 else "MULTIVIEW_CARVE",
                "height": float(toks[4]) if len(toks) > 4 else 1.75,
                "depth": float(toks[5]) if len(toks) > 5 else 0.30,
            }
        if op == "POLY":
            ptype = _POLY_MAP.get(toks[1], toks[1])
            return {
                "cmd": "poly_reconstruct",
                "poly_type": ptype,
                "image_dir": toks[2],
                "resolution": int(toks[3]) if len(toks) > 3 else 56,
                "project_colors": bool(int(toks[4])) if len(toks) > 4 else True,
            }
        if op == "FIND":
            return {"cmd": "asset_search", "query": " ".join(toks[1:])}
        if op == "USE":
            attach = "" if (len(toks) <= 4 or toks[4] == "-") else toks[4]
            return {
                "cmd": "asset_use",
                "query": toks[1],
                "place_mode": toks[2] if len(toks) > 2 else "NEXT_TO",
                "target_height": float(toks[3]) if len(toks) > 3 else 0.0,
                "attach_to": attach,
            }
        if op == "STAGE":
            return {
                "cmd": "oneshot_stage",
                "preset": toks[1] if len(toks) > 1 else "ATHLETIC_FEMALE",
                "pose": toks[2] if len(toks) > 2 else "RELAXED",
                "ethnicity": toks[3] if len(toks) > 3 else "BALANCED",
                "backdrop": toks[4] if len(toks) > 4 else "CYCLORAMA",
            }
        if op == "POSE":
            return {
                "cmd": "oneshot_pose",
                "pose": toks[1] if len(toks) > 1 else "RELAXED",
                "object": toks[2] if len(toks) > 2 else "",
            }
        if op == "PART":
            return {
                "cmd": "oneshot_part",
                "part_type": toks[1] if len(toks) > 1 else "GEAR",
                "size": float(toks[2]) if len(toks) > 2 else 0.5,
                "detail": int(toks[3]) if len(toks) > 3 else 16,
            }
        if op == "PROP":
            return {
                "cmd": "oneshot_prop",
                "prop_type": toks[1] if len(toks) > 1 else "TABLE",
                "scale": float(toks[2]) if len(toks) > 2 else 1.0,
            }
        if op == "STUDIO":
            return {
                "cmd": "oneshot_studio",
                "style": toks[1] if len(toks) > 1 else "CYCLORAMA",
                "size": float(toks[2]) if len(toks) > 2 else 4.5,
            }
        if op == "STYLE":
            return {
                "cmd": "character_stylize",
                "object": toks[1] if len(toks) > 1 else "",
            }
        if op == "CITY":
            return {
                "cmd": "oneshot_city",
                "grid_blocks": int(toks[1]) if len(toks) > 1 else 5,
                "seed": int(toks[2]) if len(toks) > 2 else 42,
                "night_mode": bool(int(toks[3])) if len(toks) > 3 else False,
            }
        if op == "DIGEST":
            return {"cmd": "scene_digest"}
        raise ValueError(f"Unknown DSL op: {op}")

    def _dispatch(self, payload: str) -> str:
        data = json.loads(payload)
        res = self._dispatch_dict(data)
        return _jdump(res)

    def _dispatch_dict(self, data):
        self._remembered = False
        program = self._steps(data)
        try:
            task = next(program)
            while True:
                try:
                    value = task()
                except Exception as exc:
                    task = program.throw(exc)
                else:
                    task = program.send(value)
        except StopIteration as done:
            return done.value

    def _remember(self):
        if not self._remembered:
            self.history.remember()
            self._remembered = True

    def _steps(self, data):
        try:
            if not isinstance(data, dict):
                raise ValueError("Expected a command object")
            cmd = data.get("cmd", "")
            if cmd in {"batch", "dsl"}:
                return (yield from self._sequence(cmd, data))
            if cmd in {"fit_face_from_image", "install_asset_pack", "equip_asset", "character_stylize"}:
                from . import face_fit, human_assets

                if cmd == "install_asset_pack":
                    path = yield partial(
                        human_assets.install_asset_pack, data.get("pack", "system"), face_fit.cache_dir()
                    )
                    return {"ok": True, "path": str(path)}
                name = data.get("human") or data.get("object")
                target = bpy.context.scene.objects[name] if name else bpy.context.active_object
                if target is None:
                    raise ValueError("Specify a character")
                name, identity = target.name, (bpy.context.scene.session_uid, target.session_uid)
                if cmd in {"fit_face_from_image", "equip_asset"}:
                    face_fit.basemesh_services(target)
                if cmd == "fit_face_from_image":
                    ratios = yield partial(
                        face_fit.measure_image, bpy.path.abspath(data["image_path"]), face_fit.cache_dir()
                    )
                else:
                    packs = [data.get("pack", "system")] if cmd == "equip_asset" else human_assets.styling_packs(data)
                    if packs:
                        yield partial(human_assets.prepare_packs, packs, face_fit.cache_dir())
                target = bpy.context.scene.objects.get(name)
                if target is None or identity != (bpy.context.scene.session_uid, target.session_uid):
                    raise ValueError("The selected character changed during preparation")
                self._remember()
                if cmd == "fit_face_from_image":
                    return {"ok": True, **face_fit.apply_measurements(target, ratios)}
                data = dict(data, object=name)
            if cmd == "generate_3d":
                from . import ai_3d_provider as provider

                scene_uid = bpy.context.scene.session_uid
                path = yield partial(
                    provider.download_model,
                    data.get("prompt", ""),
                    bpy.path.abspath(data["image_path"]) if data.get("image_path") else None,
                    data.get("provider", "HUNYUAN3D-2"),
                    timeout=data.get("timeout", 900),
                    texture=data.get("texture", False),
                )
                try:
                    if scene_uid != bpy.context.scene.session_uid:
                        raise ValueError("The active scene changed during generation")
                    self._remember()
                    options = {
                        k: v
                        for k, v in data.items()
                        if k not in {"cmd", "provider", "prompt", "image_path", "texture", "timeout"}
                    }
                    return {"ok": True, **provider.import_model(path, **options)}
                finally:
                    path.unlink(missing_ok=True)
            if cmd not in READ_ONLY and cmd not in {"checkpoint", "undo", "rollback", "reload_addon"}:
                self._remember()
            result = self._execute(data)
            if result.get("ok") and cmd.startswith("sketch_"):
                sketch_engine.SketchDocument.active().save()
            return result
        except (KeyError, TypeError, IndexError, ValueError, RuntimeError, OSError) as exc:
            return {"ok": False, "error": str(exc)}

    def _sequence(self, cmd, data):
        if cmd == "batch":
            steps = data.get("steps", [])
            stop_on_error = bool(data.get("stop_on_error", True))
            return_all = bool(data.get("return_all", False))
            results = []
            last = {"ok": True}
            success = True
            for index, step in enumerate(steps, 1):
                last = yield from self._steps(step)
                success = success and last.get("ok", False)
                if return_all:
                    results.append(last)
                if stop_on_error and not last.get("ok", False):
                    out = {"ok": False, "steps_run": index, "last": last}
                    if return_all:
                        out["results"] = results
                    return out
            out = {"ok": success, "steps_run": len(steps), "last": last}
            if return_all:
                out["results"] = results
            return out

        if cmd == "dsl":
            code = data.get("code", "")
            lines = list(dsl_lines(code))
            last = {"ok": True}
            for idx, ln in enumerate(lines):
                try:
                    last = yield from self._steps(self._parse_dsl_line(ln))
                except (KeyError, TypeError, IndexError, ValueError) as exc:
                    last = {"ok": False, "error": str(exc)}
                if not last.get("ok", False):
                    return {"ok": False, "n": idx + 1, "last": last}
            return {"ok": True, "n": len(lines), "last": last}

    def _execute(self, data: dict) -> dict:
        cmd = data.get("cmd", "")
        doc = sketch_engine.SketchDocument.active()
        if cmd in LAYOUT:
            args = {k:v for k,v in data.items() if k!="cmd"}
            if cmd in {"look_at", "place_relative", "set_material"} and "object" in args:
                args["obj"] = args.pop("object")
            return {"ok":True, **LAYOUT[cmd](**args)}
        if cmd in {"checkpoint", "rollback", "undo"}:
            result = getattr(self.history,cmd)(data["name"]) if cmd!="undo" else self.history.undo()
            self._remembered = False
            return {"ok":True, **result}
        if cmd == "reload_addon":
            from . import reload_addon
            reload_addon(self)
            return {"ok":True, "port":self.port}
        if cmd == "render_inspect":
            from .inspection import render_inspect
            return {"ok": True, **render_inspect(**{k: v for k, v in data.items() if k != "cmd"})}

        if cmd == "schema":
            from .schema import TOPICS
            topic = data.get("topic")
            return {"ok": True, "schema": TOPICS[topic] if topic else COMPACT_SCHEMA}

        if cmd == "ping":
            return {
                "ok": True,
                "service": "EX-CAD Addon",
                "port": self.port,
                "mpfb": human_engine.is_mpfb_available(),
            }

        if cmd == "scene_digest":
            include_all = bool(data.get("include_all", False))
            since_hash = data.get("since_hash", "")
            cur_hash, rows, sketch = self._build_digest(include_all=include_all)
            prev_map = self._digest_cache.pop(since_hash, None)
            prev_sketch = self._sketch_cache.get(since_hash)
            if prev_map is not None:
                self._digest_cache[since_hash] = prev_map
            cur_map = {r[0]: r for r in rows}
            self._digest_cache.pop(cur_hash, None)
            self._digest_cache[cur_hash] = cur_map
            self._sketch_cache[cur_hash] = sketch
            if len(self._digest_cache) > 16:
                oldest = next(iter(self._digest_cache))
                del self._digest_cache[oldest]
                self._sketch_cache.pop(oldest, None)

            if since_hash == cur_hash:
                return {"ok": True, "unchanged": True, "h": cur_hash}
            added_sketch = {"sketch": sketch} if prev_sketch != sketch else {}
            if prev_map is not None:
                added_or_mod = [r for k, r in cur_map.items() if prev_map.get(k) != r]
                removed = [k for k in prev_map if k not in cur_map]
                return {"ok": True, "h": cur_hash, "delta": True, "changed": added_or_mod, "removed": removed, **added_sketch}
            return {"ok": True, "h": cur_hash, "objs": rows, **added_sketch}

        if cmd == "list":
            objs = [
                {
                    "name": o.name,
                    "type": o.type,
                    "verts": len(o.data.vertices) if o.type == 'MESH' else 0,
                    "faces": len(o.data.polygons) if o.type == 'MESH' else 0,
                }
                for o in bpy.data.objects
            ]
            return {"ok": True, "objects": objs}

        if cmd == "human_generate":
            obj, rig, engine = human_engine.create_studio_human(
                preset=data.get("preset", "ATHLETIC_FEMALE"),
                gender=float(data.get("gender", 0.5)),
                age=float(data.get("age", 0.45)),
                muscle=float(data.get("muscle", 0.65)),
                weight=float(data.get("weight", 0.45)),
                height=float(data.get("height", 0.52)),
                proportions=float(data.get("proportions", 0.60)),
                ethnicity=data.get("ethnicity", "BALANCED"),
                rig_name=data.get("rig", "default"),
                apply_skin=bool(data.get("apply_skin", True)),
                subdiv_level=int(data.get("subdiv_level", 1)),
                name=data.get("name", "EXCAD_Human"),
            )
            return {
                "ok": True,
                "engine": engine,
                "name": obj.name,
                "rig": rig.name if rig else None,
                "verts": len(obj.data.vertices),
                "faces": len(obj.data.polygons),
            }

        if cmd == "human_list_targets":
            cat = data.get("category", "")
            if not cat:
                return {"ok": True, "categories": human_engine.list_target_categories()}
            return {"ok": True, "category": cat, "targets": human_engine.list_targets_in_category(cat)}

        if cmd == "human_apply_target":
            obj_name = data.get("object", "")
            obj = bpy.data.objects.get(obj_name) or bpy.context.active_object
            if obj is None:
                return {"ok": False, "error": "No target human object"}
            tname = human_engine.apply_morph_target(
                obj, data["category"], data["target"], float(data.get("weight", 0.5))
            )
            return {"ok": True, "object": obj.name, "target": tname}

        if cmd == "human_refit_rig":
            obj = bpy.data.objects.get(data.get("object", "")) or bpy.context.active_object
            ok = human_engine.refit_human_rig(obj)
            return {"ok": ok, "object": obj.name if obj else None}

        if cmd == "auto_rig_mesh":
            obj = bpy.data.objects.get(data.get("object", "")) or bpy.context.active_object
            arm = human_engine.auto_rig_any_human_mesh(obj, name=data.get("name", "EXCAD_AutoRig"))
            return {"ok": True, "object": obj.name, "rig": arm.name, "bones": len(arm.data.bones)}

        if cmd == "oneshot_stage":
            res = oneshot_generators.create_character_stage(
                preset=data.get("preset", "ATHLETIC_FEMALE"),
                pose=data.get("pose", "RELAXED"),
                ethnicity=data.get("ethnicity", "BALANCED"),
                rig=data.get("rig", "game_engine"),
                backdrop=data.get("backdrop", "CYCLORAMA"),
            )
            return {
                "ok": True,
                "human": res["human"].name,
                "rig": res["rig"].name if res["rig"] else None,
                "posed_bones": res["posed_bones"],
                "stage_objects": [o.name for o in res["stage_objects"]],
            }

        if cmd == "oneshot_pose":
            name = data.get("object")
            obj = bpy.context.scene.objects.get(name) if name else bpy.context.active_object
            if obj is not None and obj.type == 'MESH':
                obj = obj.find_armature()
            n = oneshot_generators.apply_pose_preset(obj, data.get("pose", "RELAXED"))
            return {"ok": n > 0, "rig": obj.name if obj else None, "posed_bones": n}

        if cmd == "oneshot_part":
            obj = oneshot_generators.create_mechanical_part(
                part_type=data.get("part_type", "GEAR"),
                size=float(data.get("size", 0.5)),
                detail=int(data.get("detail", 16)),
                name=data.get("name", ""),
            )
            return {"ok": True, "name": obj.name, "verts": len(obj.data.vertices), "faces": len(obj.data.polygons)}

        if cmd == "oneshot_prop":
            obj = oneshot_generators.create_furniture_prop(
                prop_type=data.get("prop_type", "TABLE"),
                scale=float(data.get("scale", 1.0)),
                name=data.get("name", ""),
            )
            return {"ok": True, "name": obj.name, "verts": len(obj.data.vertices), "faces": len(obj.data.polygons)}

        if cmd == "oneshot_studio":
            objs = oneshot_generators.create_studio_backdrop(
                style=data.get("style", "CYCLORAMA"),
                size=float(data.get("size", 4.5)),
            )
            return {"ok": True, "objects": [o.name for o in objs]}

        if cmd == "character_stylize":
            obj = bpy.data.objects.get(data.get("object", "")) or bpy.context.active_object
            if ("hair" in data and data["hair"] is None) or isinstance(data.get("hair"), str) or any(k in data for k in (
                    "clothes", "remove", "skin", "lip_tint", "eyes", "iris_color", "iris_size", "lashes", "teeth", "hands", "expression")):
                return {"ok": True, **character_stylizer.style_character(obj, data)}
            objs = character_stylizer.stylize_portrait_character(
                obj,
                add_hair=bool(data.get("hair", True)),
                add_sweater=bool(data.get("sweater", True)),
                add_necklace=bool(data.get("necklace", True)),
            )
            return {"ok": True, "object": obj.name, "added": [o.name for o in objs]}

        if cmd == "equip_asset":
            from .human_assets import equip_asset
            human = bpy.context.scene.objects[data.get("human") or data["object"]]
            obj = equip_asset(human, data["name"], data.get("pack", "system"), data.get("subdiv", 0))
            return {"ok": True, "name": obj.name}
        if cmd == "oneshot_city":
            objs = city_generator.generate_city(
                grid_blocks=int(data.get("grid_blocks", 5)),
                block_size=float(data.get("block_size", 14.0)),
                road_width=float(data.get("road_width", 5.0)),
                seed=int(data.get("seed", 42)),
                night_mode=bool(data.get("night_mode", False)),
            )
            return {"ok": True, "objects": [o.name for o in objs]}

        if cmd == "asset_fetch_url":
            dest, objs = asset_fetcher.fetch_url_asset(
                data["url"], data.get("filename"), bool(data.get("import", True))
            )
            return {"ok": True, "file": dest, "imported": [o.name for o in objs]}

        if cmd == "asset_fetch_git":
            dest_dir, found, objs = asset_fetcher.fetch_git_repo(
                data["repo_url"], bool(data.get("import", True))
            )
            return {"ok": True, "dir": dest_dir, "found": found[:20], "imported": [o.name for o in objs]}

        if cmd == "asset_list_cached":
            return {"ok": True, "assets": asset_fetcher.list_cached_assets()}

        if cmd == "asset_search":
            hits = asset_fetcher.search_assets(
                query=data.get("query", ""),
                extra_dir=data.get("extra_dir", ""),
                limit=int(data.get("limit", 20)),
            )
            return {"ok": True, "assets": [[h["name"], h["path"], h["ext"], h["source"]] for h in hits]}

        if cmd == "asset_use":
            attach_name = data.get("attach_to", "")
            attach_obj = bpy.data.objects.get(attach_name) if attach_name else None
            resolved, objs, bbox = asset_fetcher.import_asset_naturally(
                query_or_path=data.get("query", data.get("path", "")),
                place_mode=data.get("place_mode", "NEXT_TO"),
                target_height=float(data.get("target_height", 0.0)),
                attach_to=attach_obj,
                attach_bone=data.get("attach_bone", ""),
                extra_dir=data.get("extra_dir", ""),
            )
            return {
                "ok": True,
                "file": os.path.basename(resolved),
                "imported": [o.name for o in objs],
                "bbox": bbox,
            }

        if cmd == "asset_import":
            rel = data["path"]
            resolved, objs, bbox = asset_fetcher.import_asset_naturally(
                query_or_path=rel,
                place_mode=data.get("place_mode", "NEXT_TO"),
                target_height=float(data.get("target_height", 0.0)),
            )
            return {"ok": True, "file": os.path.basename(resolved), "imported": [o.name for o in objs], "bbox": bbox}

        if cmd == "image_trace_sketch":
            loop_2d, curve_obj = image_sketch_engine.trace_image_to_sketch(
                image_path=data["image_path"],
                plane=data.get("plane", "XZ"),
                target_height_m=float(data.get("height", 1.75)),
                threshold=float(data.get("threshold", 0.15)),
                simplify_eps=float(data.get("simplify_eps", 0.008)),
            )
            return {"ok": True, "curve": curve_obj.name, "points": len(loop_2d)}

        if cmd == "image_to_mesh":
            obj = image_sketch_engine.synthesize_mesh_from_images(
                front_image=data["front_image"],
                side_image=data.get("side_image", ""),
                top_image=data.get("top_image", ""),
                mode=data.get("mode", "MULTIVIEW_CARVE"),
                target_height_m=float(data.get("height", 1.75)),
                depth=float(data.get("depth", 0.30)),
                threshold=float(data.get("threshold", 0.15)),
                simplify_eps=float(data.get("simplify_eps", 0.008)),
                voxel_size=float(data.get("voxel_size", 0.02)),
                smooth_iters=int(data.get("smooth_iters", 4)),
                name=data.get("name", "EXCAD_ImageMesh"),
            )
            return {"ok": True, "name": obj.name, "verts": len(obj.data.vertices), "faces": len(obj.data.polygons)}

        if cmd == "poly_create_rig":
            center = tuple(data.get("center", [0.0, 0.0, 0.9]))
            faces, cams, manifest = polyhedron_engine.create_polyhedron_rig(
                poly_type=data.get("poly_type", "DODECAHEDRON_12"),
                size=float(data.get("size", 2.0)),
                center=center,
                output_dir=data.get("output_dir", ""),
            )
            return {"ok": True, "cameras": len(cams), "manifest": manifest, "views": faces}

        if cmd == "poly_render":
            images, manifest = polyhedron_engine.render_polyhedron_views(
                data.get("poly_type", "DODECAHEDRON_12"), data["output_dir"], size=data.get("size", 2.0),
                center=data.get("center", [0, 0, 0.9]), img_res=data.get("img_res", 256),
                write_depth=data.get("write_depth", False),
            )
            return {"ok": True, "views": len(images), "manifest": manifest}
        if cmd == "sdf_generate":
            from .sdf_engine import generate_sdf
            args = {k:v for k,v in data.items() if k not in {"cmd", "object", "target"}}
            obj = generate_sdf(target=data.get("object", data.get("target")), **args)
            return {"ok": True, "name": obj.name, "verts": len(obj.data.vertices), "faces": len(obj.data.polygons)}
        if cmd == "poly_bake":
            from .uv_bake import bake
            obj = bpy.context.scene.objects[data["object"]]
            return {"ok": True, **bake(obj, **{k:v for k,v in data.items() if k not in {"cmd","object"}})}
        if cmd == "poly_reconstruct":
            center = tuple(data.get("center", [0.0, 0.0, 0.9]))
            obj, n_views = polyhedron_engine.reconstruct_from_polyhedron_views(
                poly_type=data.get("poly_type", "DODECAHEDRON_12"),
                image_dir=data["image_dir"],
                size=float(data.get("size", 2.0)),
                center=center,
                resolution=int(data.get("resolution", 56)),
                threshold=float(data.get("threshold", 0.15)),
                smooth_iters=int(data.get("smooth_iters", 4)),
                project_colors=bool(data.get("project_colors", True)),
                name=data.get("name", "EXCAD_PolyMesh"),
                use_depth=data.get("use_depth", True),
                texture_mode=data.get("texture_mode", "UV"),
                texture_size=data.get("texture_size", 2048),
            )
            return {
                "ok": True,
                "name": obj.name,
                "views_used": n_views,
                "verts": len(obj.data.vertices),
                "faces": len(obj.data.polygons),
            }

        if cmd == "poly_project_colors":
            obj = bpy.data.objects.get(data.get("object", "")) or bpy.context.active_object
            if obj is None or obj.type != 'MESH':
                return {"ok": False, "error": "No target mesh object"}
            center = tuple(data.get("center", [0.0, 0.0, 0.9]))
            n_views = polyhedron_engine.project_colors_to_mesh(
                obj,
                poly_type=data.get("poly_type", "DODECAHEDRON_12"),
                image_dir=data["image_dir"],
                size=float(data.get("size", 2.0)),
                center=center,
                threshold=float(data.get("threshold", 0.15)),
            )
            return {"ok": True, "object": obj.name, "views_used": n_views}

        if cmd == "sketch_new":
            plane = data.get("plane", "XY")
            if plane not in {"XY", "XZ", "YZ"}:
                raise ValueError("Sketch plane must be XY, XZ or YZ")
            bpy.context.scene["excad_active_sketch"] = data.get("sketch_id", "Sketch")
            doc = sketch_engine.SketchDocument.active()
            doc.reset(plane)
            if "offset" in data:
                from mathutils import Matrix, Vector
                frame = Matrix(tuple(Vector(doc.plane_to_world(*axes))
                                     for axes in ((1,0,0), (0,1,0), (0,0,1)))).transposed().to_4x4()
                frame.translation = frame.col[2].xyz * float(data["offset"])
                doc.frame = [list(row) for row in frame]
            return {"ok": True, "plane": plane}
        if cmd == "sketch_add_rect":
            doc.add_rect(
                data.get("u0", -0.5), data.get("v0", -0.5),
                data.get("width", 1.0), data.get("height", 1.0),
            )
            return {"ok": True, "points": len(doc.points)}
        if cmd == "sketch_add_circle":
            doc.add_circle(data.get("cu", 0.0), data.get("cv", 0.0), data.get("radius", 0.5))
            return {"ok": True, "entities": len(doc.entities)}
        if cmd == "sketch_add_line":
            doc.add_line(
                data.get("u0", 0.0), data.get("v0", 0.0),
                data.get("u1", 1.0), data.get("v1", 0.0),
            )
            return {"ok": True, "entities": len(doc.entities)}
        if cmd == "sketch_solve":
            conv, iters, res = doc.solve()
            return {"ok": True, "converged": conv, "iterations": iters, "max_residual": res}
        if cmd in {"sketch_extrude", "sketch_revolve"} and (
                doc.support or any(k in data for k in ("object", "draft_angle", "axis", "op", "sketch"))):
            return {"ok": True, **feature_tree.dispatch(cmd[7:], data)}
        if cmd == "sketch_extrude":
            doc.solve()
            obj = sketch_engine.extrude_sketch_doc(
                doc, float(data.get("depth", 1.0)), bool(data.get("symmetric", False)),
                data.get("name", "EXCAD_Extrude"),
            )
            if obj is None:
                return {"ok": False, "error": "No closed sketch profile to extrude"}
            return {"ok": True, "verts": len(obj.data.vertices), "faces": len(obj.data.polygons)}
        if cmd == "sketch_revolve":
            doc.solve()
            obj = sketch_engine.revolve_sketch_doc(
                doc, float(data.get("angle_deg", 360.0)), int(data.get("segments", 48)),
                data.get("name", "EXCAD_Revolve"),
            )
            if obj is None:
                return {"ok": False, "error": "No closed sketch profile to revolve"}
            return {"ok": True, "verts": len(obj.data.vertices), "faces": len(obj.data.polygons)}

        result = feature_tree.dispatch(cmd, data)
        if result is not None:
            return {"ok": True, **result}
        return {"ok": False, "error": f"Unknown command: {cmd}"}

    def _server_loop(self, sock, requests, stopped):
        while not stopped.is_set():
            try:
                client, _ = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with client:
                client.settimeout(0.2)
                chunks = []
                deadline = time.monotonic() + 15
                while not stopped.is_set() and time.monotonic() < deadline:
                    try:
                        chunk = client.recv(4096)
                    except socket.timeout:
                        continue
                    except OSError:
                        chunks.clear()
                        break
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if b"\n" in chunk:
                        break
                payload = b"".join(chunks).decode("utf-8", errors="replace").strip()
                if payload and not stopped.is_set():
                    holder = {}
                    evt = threading.Event()
                    requests.put((payload, holder, evt))
                    deadline = time.monotonic() + 1200
                    while not evt.wait(0.1) and not stopped.is_set() and time.monotonic() < deadline:
                        pass
                    fallback = "Bridge stopped" if stopped.is_set() else "Timeout"
                    reply = holder.get("resp") or _jdump({"ok": False, "error": fallback})
                    try:
                        client.sendall((reply + "\n").encode("utf-8"))
                    except OSError:
                        pass
        sock.close()
