# SPDX-FileCopyrightText: 2026 EX-CAD Authors
# SPDX-License-Identifier: GPL-2.0-or-later

bl_info = {
    "name": "EX-CAD Studio",
    "author": "EX-CAD Authors",
    "version": (1, 1, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > EX-CAD",
    "description": "Studio Human Generator (MPFB2), External Git/URL Asset Fetcher, Parametric Sketch, and AI Direct Bridge",
    "category": "3D View",
}

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup

from . import ai_bridge, asset_fetcher, character_stylizer, city_generator, human_engine, image_sketch_engine, oneshot_generators, polyhedron_engine, sketch_engine


class EXCADSceneProps(PropertyGroup):
    human_preset: EnumProperty(
        name="Preset",
        items=[
            ('ATHLETIC_FEMALE', "Athletic Female", ""),
            ('ATHLETIC_MALE', "Athletic Male", ""),
            ('REALISTIC_FEMALE', "Realistic Female", ""),
            ('REALISTIC_MALE', "Realistic Male", ""),
            ('HEROIC_SCULPT', "Heroic Sculpt", ""),
            ('TEEN', "Teen", ""),
            ('CUSTOM', "Custom Sliders", ""),
        ],
        default='ATHLETIC_FEMALE',
    )
    human_gender: FloatProperty(name="Gender (F->M)", default=0.5, min=0.0, max=1.0)
    human_age: FloatProperty(name="Age", default=0.45, min=0.0, max=1.0)
    human_muscle: FloatProperty(name="Muscle", default=0.65, min=0.0, max=1.0)
    human_weight: FloatProperty(name="Weight", default=0.45, min=0.0, max=1.0)
    human_height: FloatProperty(name="Height", default=0.52, min=0.0, max=1.0)
    human_proportions: FloatProperty(name="Proportions", default=0.60, min=0.0, max=1.0)
    human_ethnicity: EnumProperty(
        name="Ethnicity",
        items=[
            ('BALANCED', "Balanced", ""),
            ('ASIAN', "Asian", ""),
            ('CAUCASIAN', "Caucasian", ""),
            ('AFRICAN', "African", ""),
        ],
        default='BALANCED',
    )
    human_rig: EnumProperty(
        name="Rig",
        items=[
            ('game_engine', "Game Engine (Clean 53 Bones)", ""),
            ('mixamo', "Mixamo (52 Bones)", ""),
            ('default', "Default Full Skeleton (163 Bones)", ""),
            ('rigify.human', "Rigify Human", ""),
            ('NONE', "No Rig", ""),
        ],
        default='game_engine',
    )
    human_skin: BoolProperty(name="SSS Skin Material", default=True)
    human_subdiv: IntProperty(name="Subdiv", default=1, min=0, max=3)

    asset_preset: EnumProperty(
        name="Sample",
        items=[
            (k, v["label"], v["url"])
            for k, v in asset_fetcher.EXTERNAL_PRESETS.items()
        ],
        default='KHRONOS_RIGGED_FIGURE',
    )
    asset_url: StringProperty(
        name="Asset URL",
        default="https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/RiggedFigure/glTF-Binary/RiggedFigure.glb",
    )
    asset_git_url: StringProperty(
        name="Git Repo",
        default="https://github.com/KhronosGroup/glTF-Sample-Assets.git",
    )
    asset_local_dir: StringProperty(name="Local Assets Dir", subtype='DIR_PATH', default="")
    asset_search_query: StringProperty(name="Search / File", default="Rigged")
    asset_place_mode: EnumProperty(
        name="Placement",
        items=[
            ('NEXT_TO', "Next To Scene (No Overlap)", ""),
            ('ON_TARGET', "On Active Object", ""),
            ('CURSOR', "At 3D Cursor", ""),
            ('ORIGIN', "At Origin (Grounded)", ""),
        ],
        default='NEXT_TO',
    )
    asset_target_height: FloatProperty(name="Target Height (0=Auto)", default=0.0, min=0.0, max=20.0)
    asset_attach_to_active: BoolProperty(name="Parent to Active", default=False)

    img_front: StringProperty(name="Front Image", subtype='FILE_PATH', default="")
    img_side: StringProperty(name="Side Image (Optional)", subtype='FILE_PATH', default="")
    img_top: StringProperty(name="Top Image (Optional)", subtype='FILE_PATH', default="")
    img_mode: EnumProperty(
        name="Synthesis Mode",
        items=[
            ('MULTIVIEW_CARVE', "Multi-View Carve (Front+Side)", ""),
            ('INFLATE', "2D Inflate (Single Image)", ""),
            ('EXTRUDE', "CAD Extrude", ""),
        ],
        default='MULTIVIEW_CARVE',
    )
    img_height: FloatProperty(name="Target Height (m)", default=1.75, min=0.05, max=20.0)
    img_depth: FloatProperty(name="Depth / Thickness", default=0.30, min=0.01, max=10.0)
    img_threshold: FloatProperty(name="BG Threshold", default=0.15, min=0.01, max=0.95)
    img_simplify: FloatProperty(name="Simplify", default=0.008, min=0.001, max=0.1)
    img_voxel_size: FloatProperty(name="Voxel Size", default=0.02, min=0.005, max=0.2)
    img_smooth_iters: IntProperty(name="Smooth Iters", default=4, min=0, max=20)

    poly_type: EnumProperty(
        name="Polyhedron",
        items=[
            ('CUBE_6', "Cube (6 Faces)", ""),
            ('OCTAHEDRON_8', "Octahedron (8 Faces)", ""),
            ('DODECAHEDRON_12', "Dodecahedron (12 Faces)", ""),
            ('ICOSAHEDRON_20', "Icosahedron (20 Faces)", ""),
        ],
        default='DODECAHEDRON_12',
    )
    poly_size: FloatProperty(name="Cage Size (m)", default=2.0, min=0.2, max=20.0)
    poly_dir: StringProperty(name="Views Folder", subtype='DIR_PATH', default="")
    poly_resolution: IntProperty(name="Voxel Res", default=56, min=16, max=112)
    poly_project_color: BoolProperty(name="Project 360° Colors", default=True)
    poly_write_depth: BoolProperty(name="Capture Depth Maps", default=False)

    oneshot_pose: EnumProperty(
        name="Pose",
        items=[
            ('RELAXED', "Relaxed", ""),
            ('HEROIC', "Heroic", ""),
            ('WAVE', "Wave", ""),
            ('WALK', "Walk", ""),
            ('SITTING', "Sitting", ""),
            ('RESET', "Reset T/A-Pose", ""),
        ],
        default='RELAXED',
    )
    oneshot_studio_style: EnumProperty(
        name="Studio",
        items=[('CYCLORAMA', "Cyclorama", ""), ('ROOM', "Room (3-Wall)", "")],
        default='CYCLORAMA',
    )
    oneshot_part: EnumProperty(
        name="CAD Part",
        items=[
            ('GEAR', "Spur Gear", ""),
            ('FLANGE', "Bolt Flange", ""),
            ('BRACKET', "L-Bracket", ""),
            ('ENCLOSURE', "Enclosure Box", ""),
            ('BOLT', "Hex Bolt", ""),
        ],
        default='GEAR',
    )
    oneshot_part_size: FloatProperty(name="Size (m)", default=0.5, min=0.05, max=10.0)
    oneshot_part_detail: IntProperty(name="Teeth/Holes", default=16, min=4, max=64)
    oneshot_prop: EnumProperty(
        name="Prop",
        items=[
            ('TABLE', "Table", ""),
            ('CHAIR', "Chair", ""),
            ('SHELF', "Bookshelf", ""),
            ('PEDESTAL', "Pedestal", ""),
        ],
        default='TABLE',
    )
    oneshot_prop_scale: FloatProperty(name="Scale", default=1.0, min=0.1, max=10.0)
    city_blocks: IntProperty(name="Blocks", default=5, min=3, max=9)
    city_seed: IntProperty(name="Seed", default=42, min=0, max=9999)
    city_night: BoolProperty(name="Night Mode", default=False)


# ---------------------------------------------------------------------------
# Human Operators
# ---------------------------------------------------------------------------

class EXCAD_OT_install_cc0_assets(Operator):
    bl_idname = "excad.install_cc0_assets"
    bl_label = "Install CC0 character assets"
    _job = None

    @classmethod
    def poll(cls, context):
        return cls._job is None

    def execute(self, context):
        from .human_assets import install_asset_pack
        from .face_fit import cache_dir

        if bpy.app.background:
            try:
                install_asset_pack("all")
            except (OSError, RuntimeError, ValueError) as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}
            return {"FINISHED"}
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="EXCAD assets")
        type(self)._job = pool.submit(install_asset_pack, "all", cache_dir())
        pool.shutdown(wait=False)
        self._timer = context.window_manager.event_timer_add(0.25, window=context.window)
        context.window_manager.modal_handler_add(self)
        self.report({"INFO"}, "Downloading the MakeHuman CC0 character packs")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        job = type(self)._job
        if event.type != "TIMER" or not job.done():
            return {"PASS_THROUGH"}
        context.window_manager.event_timer_remove(self._timer)
        type(self)._job = None
        try:
            path = job.result()
        except (OSError, RuntimeError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"MakeHuman eyes, teeth, hair and wardrobe cached in {path}")
        return {"FINISHED"}

    def cancel(self, context):
        context.window_manager.event_timer_remove(self._timer)
        job = type(self)._job
        if job is not None:
            job.cancel()
            type(self)._job = None


class EXCAD_OT_human_generate(Operator):
    bl_idname = "excad.human_generate"
    bl_label = "Generate Studio Human"
    bl_options = {'REGISTER', 'UNDO'}

    preset: StringProperty(default="")
    rig: StringProperty(default="")
    apply_skin: BoolProperty(default=True)
    subdiv_level: IntProperty(default=-1)

    def execute(self, context):
        p = context.scene.excad
        preset = self.preset if self.preset else p.human_preset
        rig = self.rig if self.rig else p.human_rig
        subdiv = self.subdiv_level if self.subdiv_level >= 0 else p.human_subdiv
        skin = self.apply_skin if self.preset else p.human_skin

        obj, rig_obj, engine = human_engine.create_studio_human(
            preset=preset,
            gender=p.human_gender,
            age=p.human_age,
            muscle=p.human_muscle,
            weight=p.human_weight,
            height=p.human_height,
            proportions=p.human_proportions,
            ethnicity=p.human_ethnicity,
            rig_name=rig,
            apply_skin=skin,
            subdiv_level=subdiv,
        )
        rig_info = f", rig={rig_obj.name}" if rig_obj else ""
        self.report(
            {'INFO'},
            f"EX-CAD [{engine}]: {obj.name} ({len(obj.data.vertices)} verts, {len(obj.data.polygons)} faces{rig_info})",
        )
        return {'FINISHED'}


class EXCAD_OT_human_apply_target(Operator):
    bl_idname = "excad.human_apply_target"
    bl_label = "Apply Morph Target"
    bl_options = {'REGISTER', 'UNDO'}

    category: StringProperty(default="expression")
    target: StringProperty(default="")
    weight: FloatProperty(default=0.5, min=-1.0, max=2.0)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a human mesh first")
            return {'CANCELLED'}
        if not self.target:
            targets = human_engine.list_targets_in_category(self.category)
            if not targets:
                self.report({'ERROR'}, f"No targets in category '{self.category}'")
                return {'CANCELLED'}
            self.target = targets[0]
        tname = human_engine.apply_morph_target(obj, self.category, self.target, self.weight)
        self.report({'INFO'}, f"Applied morph target: {tname} ({self.weight:.2f})")
        return {'FINISHED'}


class EXCAD_OT_human_refit_rig(Operator):
    bl_idname = "excad.human_refit_rig"
    bl_label = "Refit Rig to Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH' or obj.find_armature() is None:
            self.report({'ERROR'}, "Select a rigged MPFB2 human mesh first")
            return {'CANCELLED'}
        if not human_engine.refit_human_rig(obj):
            self.report({'ERROR'}, "Refit requires MPFB2 extension")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Refitted rig to {obj.name}")
        return {'FINISHED'}


class EXCAD_OT_auto_rig_mesh(Operator):
    bl_idname = "excad.auto_rig_mesh"
    bl_label = "Auto-Rig Selected Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a human/character mesh first")
            return {'CANCELLED'}
        try:
            arm = human_engine.auto_rig_any_human_mesh(obj)
        except Exception as exc:
            self.report({'ERROR'}, f"Auto-rig failed: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Auto-rigged {obj.name} -> {arm.name} ({len(arm.data.bones)} bones)")
        return {'FINISHED'}


class EXCAD_OT_character_stylize(Operator):
    bl_idname = "excad.character_stylize"
    bl_label = "Add Hair + V-Neck Sweater + Necklace"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a human mesh first")
            return {'CANCELLED'}
        objs = character_stylizer.stylize_portrait_character(obj)
        self.report({'INFO'}, f"Stylized {obj.name}: added {len(objs)} object(s)")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# External Git / URL Asset Operators
# ---------------------------------------------------------------------------

class EXCAD_OT_asset_use_existing(Operator):
    bl_idname = "excad.asset_use_existing"
    bl_label = "Use Existing Asset (Auto-Place)"
    bl_options = {'REGISTER', 'UNDO'}

    query: StringProperty(default="")
    place_mode: StringProperty(default="")
    target_height: FloatProperty(default=-1.0)

    def execute(self, context):
        p = context.scene.excad
        q = self.query if self.query else p.asset_search_query.strip()
        mode = self.place_mode if self.place_mode else p.asset_place_mode
        h = self.target_height if self.target_height >= 0.0 else p.asset_target_height
        attach = context.active_object if (p.asset_attach_to_active or mode == 'ON_TARGET') else None
        if not q:
            self.report({'ERROR'}, "Enter an asset keyword or file path")
            return {'CANCELLED'}
        try:
            resolved, objs, _ = asset_fetcher.import_asset_naturally(
                query_or_path=q,
                place_mode=mode,
                target_height=h,
                attach_to=attach,
                extra_dir=p.asset_local_dir,
            )
        except Exception as exc:
            self.report({'ERROR'}, f"Asset use failed: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Placed {len(objs)} object(s) from {resolved} ({mode})")
        return {'FINISHED'}


class EXCAD_OT_asset_fetch_preset(Operator):
    bl_idname = "excad.asset_fetch_preset"
    bl_label = "Fetch Sample Asset"
    bl_options = {'REGISTER', 'UNDO'}

    preset: StringProperty(default="")

    def execute(self, context):
        key = self.preset if self.preset else context.scene.excad.asset_preset
        info = asset_fetcher.EXTERNAL_PRESETS.get(key)
        if not info:
            self.report({'ERROR'}, f"Unknown preset: {key}")
            return {'CANCELLED'}
        try:
            dest, objs = asset_fetcher.fetch_url_asset(info["url"], info["filename"], auto_import=True)
        except Exception as exc:
            self.report({'ERROR'}, f"Asset fetch failed: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Imported {len(objs)} object(s) from {info['filename']}")
        return {'FINISHED'}


class EXCAD_OT_asset_fetch_url(Operator):
    bl_idname = "excad.asset_fetch_url"
    bl_label = "Fetch & Import URL"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        url = context.scene.excad.asset_url.strip()
        if not url:
            self.report({'ERROR'}, "Enter an asset URL (.glb/.gltf/.obj/.fbx/.mhclo)")
            return {'CANCELLED'}
        try:
            dest, objs = asset_fetcher.fetch_url_asset(url, auto_import=True)
        except Exception as exc:
            self.report({'ERROR'}, f"URL fetch failed: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Imported {len(objs)} object(s) from {dest}")
        return {'FINISHED'}


class EXCAD_OT_asset_fetch_git(Operator):
    bl_idname = "excad.asset_fetch_git"
    bl_label = "Clone & Import Git Repo"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        repo = context.scene.excad.asset_git_url.strip()
        if not repo:
            self.report({'ERROR'}, "Enter a Git repository URL")
            return {'CANCELLED'}
        try:
            dest_dir, found, objs = asset_fetcher.fetch_git_repo(repo, auto_import=True)
        except Exception as exc:
            self.report({'ERROR'}, f"Git clone failed: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Cloned {dest_dir}: found {len(found)} asset(s), imported {len(objs)}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Image -> Sketch -> 3D Mesh Operators
# ---------------------------------------------------------------------------

class EXCAD_OT_image_trace_sketch(Operator):
    bl_idname = "excad.image_trace_sketch"
    bl_label = "Trace Image to Sketch"
    bl_options = {'REGISTER', 'UNDO'}

    image_path: StringProperty(default="")
    plane: StringProperty(default="XZ")

    def execute(self, context):
        p = context.scene.excad
        path = self.image_path if self.image_path else bpy.path.abspath(p.img_front)
        if not path:
            self.report({'ERROR'}, "Select a Front image first")
            return {'CANCELLED'}
        try:
            loop_2d, curve_obj = image_sketch_engine.trace_image_to_sketch(
                path, plane=self.plane, target_height_m=p.img_height,
                threshold=p.img_threshold, simplify_eps=p.img_simplify,
            )
        except Exception as exc:
            self.report({'ERROR'}, f"Trace failed: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Traced {len(loop_2d)} sketch pts -> {curve_obj.name}")
        return {'FINISHED'}


class EXCAD_OT_image_to_mesh(Operator):
    bl_idname = "excad.image_to_mesh"
    bl_label = "Synthesize 3D Mesh from Images"
    bl_options = {'REGISTER', 'UNDO'}

    front_image: StringProperty(default="")
    side_image: StringProperty(default="")
    top_image: StringProperty(default="")
    mode: StringProperty(default="")

    def execute(self, context):
        p = context.scene.excad
        front = self.front_image if self.front_image else bpy.path.abspath(p.img_front)
        side = self.side_image if self.side_image else bpy.path.abspath(p.img_side)
        top = self.top_image if self.top_image else bpy.path.abspath(p.img_top)
        mode = self.mode if self.mode else p.img_mode

        if not front:
            self.report({'ERROR'}, "Select a Front image first")
            return {'CANCELLED'}
        try:
            obj = image_sketch_engine.synthesize_mesh_from_images(
                front_image=front,
                side_image=side,
                top_image=top,
                mode=mode,
                target_height_m=p.img_height,
                depth=p.img_depth,
                threshold=p.img_threshold,
                simplify_eps=p.img_simplify,
                voxel_size=p.img_voxel_size,
                smooth_iters=p.img_smooth_iters,
            )
        except Exception as exc:
            self.report({'ERROR'}, f"Synthesis failed: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Synthesized {obj.name} ({len(obj.data.vertices)} verts, {len(obj.data.polygons)} faces)")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Polyhedron Multi-View Projection Operators
# ---------------------------------------------------------------------------

class EXCAD_OT_poly_create_rig(Operator):
    bl_idname = "excad.poly_create_rig"
    bl_label = "Create Polyhedron Camera Rig"
    bl_options = {'REGISTER', 'UNDO'}

    poly_type: StringProperty(default="")
    output_dir: StringProperty(default="")

    def execute(self, context):
        p = context.scene.excad
        ptype = self.poly_type if self.poly_type else p.poly_type
        out_dir = self.output_dir if self.output_dir else bpy.path.abspath(p.poly_dir)
        faces, cams, manifest = polyhedron_engine.create_polyhedron_rig(
            poly_type=ptype, size=p.poly_size, center=(0.0, 0.0, p.poly_size * 0.45), output_dir=out_dir
        )
        msg = f"Created {len(cams)} cameras ({ptype})"
        if manifest:
            msg += f" + manifest"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class EXCAD_OT_poly_render_views(Operator):
    bl_idname = "excad.poly_render_views"
    bl_label = "Render Polyhedron Views"
    bl_options = {'REGISTER'}

    def execute(self, context):
        p = context.scene.excad
        out_dir = bpy.path.abspath(p.poly_dir)
        if not out_dir:
            self.report({'ERROR'}, "Select a Views Folder first")
            return {'CANCELLED'}
        try:
            rendered, _ = polyhedron_engine.render_polyhedron_views(
                poly_type=p.poly_type, output_dir=out_dir, size=p.poly_size, center=(0.0, 0.0, p.poly_size * 0.45),
                write_depth=p.poly_write_depth,
            )
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f"Rendered {len(rendered)} views to {out_dir}")
        return {'FINISHED'}


class EXCAD_OT_poly_reconstruct(Operator):
    bl_idname = "excad.poly_reconstruct"
    bl_label = "Reconstruct 3D from Polyhedron Views"
    bl_options = {'REGISTER', 'UNDO'}

    poly_type: StringProperty(default="")
    image_dir: StringProperty(default="")
    resolution: IntProperty(default=-1)
    project_colors: BoolProperty(default=True)

    def execute(self, context):
        p = context.scene.excad
        ptype = self.poly_type if self.poly_type else p.poly_type
        img_dir = self.image_dir if self.image_dir else bpy.path.abspath(p.poly_dir)
        res = self.resolution if self.resolution > 0 else p.poly_resolution
        if not img_dir:
            self.report({'ERROR'}, "Select a Views Folder containing face_00.png.. first")
            return {'CANCELLED'}
        try:
            obj, n_views = polyhedron_engine.reconstruct_from_polyhedron_views(
                poly_type=ptype,
                image_dir=img_dir,
                size=p.poly_size,
                center=(0.0, 0.0, p.poly_size * 0.45),
                resolution=res,
                threshold=p.img_threshold,
                smooth_iters=p.img_smooth_iters,
                project_colors=self.project_colors if self.image_dir else p.poly_project_color,
            )
        except Exception as exc:
            self.report({'ERROR'}, f"Polyhedron reconstruction failed: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Reconstructed {obj.name} from {n_views} views ({len(obj.data.vertices)} verts)")
        return {'FINISHED'}


class EXCAD_OT_poly_project_colors(Operator):
    bl_idname = "excad.poly_project_colors"
    bl_label = "Project Polyhedron Colors to Active Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.excad
        obj = context.active_object
        img_dir = bpy.path.abspath(p.poly_dir)
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object first")
            return {'CANCELLED'}
        if not img_dir:
            self.report({'ERROR'}, "Select a Views Folder first")
            return {'CANCELLED'}
        try:
            n_views = polyhedron_engine.project_colors_to_mesh(
                obj, poly_type=p.poly_type, image_dir=img_dir,
                size=p.poly_size, center=(0.0, 0.0, p.poly_size * 0.45), threshold=p.img_threshold,
            )
        except Exception as exc:
            self.report({'ERROR'}, f"Color projection failed: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Projected {n_views} views onto {obj.name}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# One-Shot AI Generators
# ---------------------------------------------------------------------------

class EXCAD_OT_oneshot_stage(Operator):
    bl_idname = "excad.oneshot_stage"
    bl_label = "Character + Pose + Studio"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.excad
        res = oneshot_generators.create_character_stage(
            preset=p.human_preset, pose=p.oneshot_pose, ethnicity=p.human_ethnicity,
            rig=p.human_rig, backdrop=p.oneshot_studio_style,
        )
        self.report({'INFO'}, f"Created stage: {res['human'].name} (pose={p.oneshot_pose})")
        return {'FINISHED'}


class EXCAD_OT_oneshot_pose(Operator):
    bl_idname = "excad.oneshot_pose"
    bl_label = "Apply Pose Preset"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.excad
        obj = context.active_object
        if obj is not None and obj.type == 'MESH':
            obj = obj.find_armature()
        n = oneshot_generators.apply_pose_preset(obj, p.oneshot_pose)
        if n == 0:
            self.report({'ERROR'}, "Select a rigged character or armature")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Applied {p.oneshot_pose} pose ({n} bones)")
        return {'FINISHED'}


class EXCAD_OT_oneshot_part(Operator):
    bl_idname = "excad.oneshot_part"
    bl_label = "Generate CAD Part"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.excad
        obj = oneshot_generators.create_mechanical_part(p.oneshot_part, p.oneshot_part_size, p.oneshot_part_detail)
        self.report({'INFO'}, f"Created {obj.name} ({len(obj.data.vertices)} verts)")
        return {'FINISHED'}


class EXCAD_OT_oneshot_prop(Operator):
    bl_idname = "excad.oneshot_prop"
    bl_label = "Generate Furniture Prop"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.excad
        obj = oneshot_generators.create_furniture_prop(p.oneshot_prop, p.oneshot_prop_scale)
        self.report({'INFO'}, f"Created {obj.name} ({len(obj.data.vertices)} verts)")
        return {'FINISHED'}


class EXCAD_OT_oneshot_studio(Operator):
    bl_idname = "excad.oneshot_studio"
    bl_label = "Create Studio Lighting & Backdrop"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.excad
        objs = oneshot_generators.create_studio_backdrop(p.oneshot_studio_style)
        self.report({'INFO'}, f"Created studio ({len(objs)} objects)")
        return {'FINISHED'}


class EXCAD_OT_oneshot_city(Operator):
    bl_idname = "excad.oneshot_city"
    bl_label = "Generate Procedural City"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.excad
        objs = city_generator.generate_city(
            grid_blocks=p.city_blocks, seed=p.city_seed, night_mode=p.city_night
        )
        self.report({'INFO'}, f"Generated {p.city_blocks}x{p.city_blocks} city ({len(objs)} objects)")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Sketch & Bridge Operators
# ---------------------------------------------------------------------------

class EXCAD_OT_sketch_new(Operator):
    bl_idname = "excad.sketch_new"
    bl_label = "New Parametric Sketch"
    bl_options = {'REGISTER', 'UNDO'}

    plane: EnumProperty(
        name="Plane",
        items=[('XY', "XY", ""), ('XZ', "XZ", ""), ('YZ', "YZ", "")],
        default='XY',
    )

    def execute(self, context):
        sketch_engine.SketchDocument.active().reset(self.plane)
        sketch_engine.SketchDocument.active().save()
        self.report({'INFO'}, f"EX-CAD: Started new {self.plane} sketch")
        return {'FINISHED'}


class EXCAD_OT_sketch_add_rect(Operator):
    bl_idname = "excad.sketch_add_rect"
    bl_label = "Sketch Rectangle"
    bl_options = {'REGISTER', 'UNDO'}

    u0: FloatProperty(name="U Origin", default=-0.5)
    v0: FloatProperty(name="V Origin", default=-0.5)
    width: FloatProperty(name="Width", default=1.0, min=0.001)
    height: FloatProperty(name="Height", default=1.0, min=0.001)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        sketch_engine.SketchDocument.active().add_rect(self.u0, self.v0, self.width, self.height)
        sketch_engine.SketchDocument.active().save()
        return {'FINISHED'}


class EXCAD_OT_sketch_add_circle(Operator):
    bl_idname = "excad.sketch_add_circle"
    bl_label = "Sketch Circle"
    bl_options = {'REGISTER', 'UNDO'}

    cu: FloatProperty(name="Center U", default=0.0)
    cv: FloatProperty(name="Center V", default=0.0)
    radius: FloatProperty(name="Radius", default=0.5, min=0.001)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        sketch_engine.SketchDocument.active().add_circle(self.cu, self.cv, self.radius)
        sketch_engine.SketchDocument.active().save()
        return {'FINISHED'}


class EXCAD_OT_sketch_add_line(Operator):
    bl_idname = "excad.sketch_add_line"
    bl_label = "Sketch Line"
    bl_options = {'REGISTER', 'UNDO'}

    u0: FloatProperty(name="U0", default=0.0)
    v0: FloatProperty(name="V0", default=0.0)
    u1: FloatProperty(name="U1", default=1.0)
    v1: FloatProperty(name="V1", default=0.0)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        sketch_engine.SketchDocument.active().add_line(self.u0, self.v0, self.u1, self.v1)
        sketch_engine.SketchDocument.active().save()
        return {'FINISHED'}


class EXCAD_OT_sketch_solve(Operator):
    bl_idname = "excad.sketch_solve"
    bl_label = "Solve Sketch"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        conv, iters, res = sketch_engine.SketchDocument.active().solve()
        sketch_engine.SketchDocument.active().save()
        self.report({'INFO'}, f"EX-CAD Solve: {'converged' if conv else 'max iters'} ({iters} iters, res={res:.2e})")
        return {'FINISHED'}


class EXCAD_OT_sketch_extrude(Operator):
    bl_idname = "excad.sketch_extrude"
    bl_label = "Extrude Sketch"
    bl_options = {'REGISTER', 'UNDO'}

    depth: FloatProperty(name="Depth", default=1.0, min=0.001)
    symmetric: BoolProperty(name="Symmetric", default=False)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        doc = sketch_engine.SketchDocument.active()
        doc.solve()
        obj = sketch_engine.extrude_sketch_doc(doc, self.depth, self.symmetric)
        if obj is None:
            self.report({'WARNING'}, "Add a closed rectangle, circle, or loop first")
            return {'CANCELLED'}
        return {'FINISHED'}


class EXCAD_OT_sketch_revolve(Operator):
    bl_idname = "excad.sketch_revolve"
    bl_label = "Revolve Sketch"
    bl_options = {'REGISTER', 'UNDO'}

    angle_deg: FloatProperty(name="Angle (deg)", default=360.0, min=1.0, max=360.0)
    segments: IntProperty(name="Segments", default=48, min=8, max=256)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        doc = sketch_engine.SketchDocument.active()
        doc.solve()
        obj = sketch_engine.revolve_sketch_doc(doc, self.angle_deg, self.segments)
        if obj is None:
            self.report({'WARNING'}, "Add a closed profile first")
            return {'CANCELLED'}
        return {'FINISHED'}


class EXCAD_OT_bridge_start(Operator):
    bl_idname = "excad.bridge_start"
    bl_label = "Start AI Direct Bridge"

    port: IntProperty(name="Port", default=9877, min=1024, max=65535)

    def execute(self, context):
        if ai_bridge.AiBridgeServer.instance().start(self.port):
            self.report({'INFO'}, f"EX-CAD: AI Bridge listening on 127.0.0.1:{self.port}")
            return {'FINISHED'}
        self.report({'ERROR'}, "Failed to start AI Bridge")
        return {'CANCELLED'}


class EXCAD_OT_bridge_stop(Operator):
    bl_idname = "excad.bridge_stop"
    bl_label = "Stop AI Direct Bridge"

    def execute(self, context):
        ai_bridge.AiBridgeServer.instance().stop()
        self.report({'INFO'}, "EX-CAD: AI Bridge stopped")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# 3D Viewport Sidebar Panels
# ---------------------------------------------------------------------------

class VIEW3D_PT_excad_human(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "EX-CAD"
    bl_label = "Studio Human Generator"

    def draw(self, context):
        layout = self.layout
        p = context.scene.excad
        engine_txt = "MPFB2 (MakeHuman Engine)" if human_engine.is_mpfb_available() else "Fallback SDF"
        layout.label(text=f"Engine: {engine_txt}", icon='OUTLINER_OB_ARMATURE')
        layout.operator("excad.install_cc0_assets", icon="IMPORT")

        col = layout.column(align=True)
        col.prop(p, "human_preset")
        col.prop(p, "human_ethnicity")
        if p.human_preset == 'CUSTOM':
            col.separator()
            col.prop(p, "human_gender", slider=True)
            col.prop(p, "human_age", slider=True)
            col.prop(p, "human_muscle", slider=True)
            col.prop(p, "human_weight", slider=True)
            col.prop(p, "human_height", slider=True)
            col.prop(p, "human_proportions", slider=True)

        col.separator()
        col.prop(p, "human_rig")
        row = col.row(align=True)
        row.prop(p, "human_skin")
        row.prop(p, "human_subdiv")

        col.separator()
        col.operator("excad.human_generate", text="Generate Studio Human", icon='USER')
        col.operator("excad.human_apply_target", text="Apply Morph Target...", icon='SHAPEKEY_DATA')
        row = col.row(align=True)
        row.operator("excad.human_refit_rig", text="Refit Rig", icon='BONE_DATA')
        row.operator("excad.auto_rig_mesh", text="Auto-Rig Mesh", icon='ARMATURE_DATA')
        col.operator("excad.character_stylize", text="Add Hair + V-Neck Sweater + Necklace", icon='MATCLOTH')


class VIEW3D_PT_excad_assets(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "EX-CAD"
    bl_label = "Assets (Reuse / Git / Web)"

    def draw(self, context):
        layout = self.layout
        p = context.scene.excad

        box = layout.box()
        box.label(text="Use Existing Asset (Auto-Place & Fit):", icon='ASSET_MANAGER')
        box.prop(p, "asset_local_dir", text="Dir")
        box.prop(p, "asset_search_query", text="Find")
        box.prop(p, "asset_place_mode", text="Place")
        row = box.row(align=True)
        row.prop(p, "asset_target_height", text="Height")
        row.prop(p, "asset_attach_to_active", text="Attach")
        box.operator("excad.asset_use_existing", text="Import & Auto-Place", icon='IMPORT')

        col = layout.column(align=True)
        col.separator()
        col.prop(p, "asset_preset", text="Sample")
        col.operator("excad.asset_fetch_preset", text="Download & Import Sample", icon='IMPORT')

        col.separator()
        col.prop(p, "asset_url", text="URL")
        col.operator("excad.asset_fetch_url", text="Fetch URL (.glb/.obj/.fbx)", icon='URL')

        col.separator()
        col.prop(p, "asset_git_url", text="Git")
        col.operator("excad.asset_fetch_git", text="Clone Git & Import", icon='FILE_FOLDER')


class VIEW3D_PT_excad_image2mesh(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "EX-CAD"
    bl_label = "Image -> Sketch -> 3D Mesh"

    def draw(self, context):
        layout = self.layout
        p = context.scene.excad

        col = layout.column(align=True)
        col.prop(p, "img_front", text="Front")
        col.prop(p, "img_side", text="Side")
        col.prop(p, "img_top", text="Top")
        col.separator()
        col.prop(p, "img_mode", text="Mode")
        row = col.row(align=True)
        row.prop(p, "img_height")
        row.prop(p, "img_depth")
        row = col.row(align=True)
        row.prop(p, "img_threshold")
        row.prop(p, "img_voxel_size")
        col.separator()
        col.operator("excad.image_trace_sketch", text="1. Trace Image to Sketch", icon='CURVE_BEZCURVE')
        col.operator("excad.image_to_mesh", text="2. Synthesize 3D Mesh", icon='MOD_REMESH')


class VIEW3D_PT_excad_polyhedron(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "EX-CAD"
    bl_label = "Polyhedron Multi-View Projection"

    def draw(self, context):
        layout = self.layout
        p = context.scene.excad

        col = layout.column(align=True)
        col.prop(p, "poly_type")
        row = col.row(align=True)
        row.prop(p, "poly_size")
        row.prop(p, "poly_resolution")
        col.prop(p, "poly_dir", text="Folder")
        col.prop(p, "poly_project_color")
        col.prop(p, "poly_write_depth")
        col.separator()
        row = col.row(align=True)
        row.operator("excad.poly_create_rig", text="1. Camera Rig", icon='OUTLINER_OB_CAMERA')
        row.operator("excad.poly_render_views", text="Render Views", icon='RENDER_STILL')
        col.operator("excad.poly_reconstruct", text="2. Visual Hull -> 3D Mesh", icon='MESH_ICOSPHERE')
        col.operator("excad.poly_project_colors", text="3. Project Colors to Active", icon='BRUSH_DATA')


class VIEW3D_PT_excad_oneshot(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "EX-CAD"
    bl_label = "One-Shot AI Generators"

    def draw(self, context):
        layout = self.layout
        p = context.scene.excad

        box = layout.box()
        box.label(text="Character + Pose + Studio:", icon='SCENE_DATA')
        row = box.row(align=True)
        row.prop(p, "oneshot_pose", text="")
        row.prop(p, "oneshot_studio_style", text="")
        box.operator("excad.oneshot_stage", text="Generate Full Stage", icon='OUTLINER_OB_ARMATURE')
        row = box.row(align=True)
        row.operator("excad.oneshot_pose", text="Pose Selected", icon='POSE_HLT')
        row.operator("excad.oneshot_studio", text="Studio Only", icon='LIGHT_AREA')

        box2 = layout.box()
        box2.label(text="CAD Mechanical Parts:", icon='MOD_SCREW')
        box2.prop(p, "oneshot_part", text="")
        row = box2.row(align=True)
        row.prop(p, "oneshot_part_size")
        row.prop(p, "oneshot_part_detail")
        box2.operator("excad.oneshot_part", text="Generate CAD Part", icon='MESH_CUBE')

        box3 = layout.box()
        box3.label(text="Furniture & Props:", icon='HOME')
        row = box3.row(align=True)
        row.prop(p, "oneshot_prop", text="")
        row.prop(p, "oneshot_prop_scale")
        box3.operator("excad.oneshot_prop", text="Generate Prop", icon='MESH_GRID')

        box4 = layout.box()
        box4.label(text="Procedural City:", icon='WORLD')
        row = box4.row(align=True)
        row.prop(p, "city_blocks")
        row.prop(p, "city_seed")
        row.prop(p, "city_night")
        box4.operator("excad.oneshot_city", text="Generate City", icon='MOD_BUILD')


class VIEW3D_PT_excad_sketch(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "EX-CAD"
    bl_label = "Parametric Sketch (Fusion Style)"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        row = col.row(align=True)
        for plane in ("XY", "XZ", "YZ"):
            op = row.operator("excad.sketch_new", text=plane)
            op.plane = plane
        col.separator()
        col.operator("excad.sketch_add_rect", text="Rectangle", icon='MESH_PLANE')
        col.operator("excad.sketch_add_circle", text="Circle", icon='MESH_CIRCLE')
        col.operator("excad.sketch_add_line", text="Line", icon='IPO_LINEAR')
        col.separator()
        col.operator("excad.sketch_solve", text="Solve Constraints", icon='CONSTRAINT')
        row = col.row(align=True)
        row.operator("excad.sketch_extrude", text="Extrude", icon='MOD_SOLIDIFY')
        row.operator("excad.sketch_revolve", text="Revolve", icon='MOD_SCREW')


class VIEW3D_PT_excad_bridge(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "EX-CAD"
    bl_label = "AI Direct Bridge (Port 9877)"

    def draw(self, context):
        layout = self.layout
        row = layout.row(align=True)
        row.operator("excad.bridge_start", text="Start Bridge", icon='PLAY')
        row.operator("excad.bridge_stop", text="Stop Bridge", icon='PAUSE')


CLASSES = (
    EXCADSceneProps,
    EXCAD_OT_human_generate,
    EXCAD_OT_install_cc0_assets,
    EXCAD_OT_human_apply_target,
    EXCAD_OT_human_refit_rig,
    EXCAD_OT_auto_rig_mesh,
    EXCAD_OT_character_stylize,
    EXCAD_OT_asset_use_existing,
    EXCAD_OT_asset_fetch_preset,
    EXCAD_OT_asset_fetch_url,
    EXCAD_OT_asset_fetch_git,
    EXCAD_OT_image_trace_sketch,
    EXCAD_OT_image_to_mesh,
    EXCAD_OT_poly_create_rig,
    EXCAD_OT_poly_render_views,
    EXCAD_OT_poly_reconstruct,
    EXCAD_OT_poly_project_colors,
    EXCAD_OT_oneshot_stage,
    EXCAD_OT_oneshot_pose,
    EXCAD_OT_oneshot_part,
    EXCAD_OT_oneshot_prop,
    EXCAD_OT_oneshot_studio,
    EXCAD_OT_oneshot_city,
    EXCAD_OT_sketch_new,
    EXCAD_OT_sketch_add_rect,
    EXCAD_OT_sketch_add_circle,
    EXCAD_OT_sketch_add_line,
    EXCAD_OT_sketch_solve,
    EXCAD_OT_sketch_extrude,
    EXCAD_OT_sketch_revolve,
    EXCAD_OT_bridge_start,
    EXCAD_OT_bridge_stop,
    VIEW3D_PT_excad_human,
    VIEW3D_PT_excad_assets,
    VIEW3D_PT_excad_image2mesh,
    VIEW3D_PT_excad_polyhedron,
    VIEW3D_PT_excad_oneshot,
    VIEW3D_PT_excad_sketch,
    VIEW3D_PT_excad_bridge,
)


def _auto_start_bridge():
    ai_bridge.AiBridgeServer.instance().start(9877)
    return None


def register():
    from . import sketch_overlay
    sketch_overlay.register()
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.excad = PointerProperty(type=EXCADSceneProps)
    if not bpy.app.background:
        bpy.app.timers.register(_auto_start_bridge, first_interval=0.5)


def unregister():
    if bpy.app.timers.is_registered(_auto_start_bridge):
        bpy.app.timers.unregister(_auto_start_bridge)
    from . import sketch_overlay
    sketch_overlay.unregister()
    ai_bridge.AiBridgeServer.instance().stop()
    if hasattr(bpy.types.Scene, "excad"):
        del bpy.types.Scene.excad
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


def reload_addon(server):
    """Replace registered classes and dispatch code while retaining the listening socket."""
    import importlib
    import sys
    from . import sketch_overlay

    modules = (
        "protocol",
        "schema",
        "sketch_profiles",
        "loft_sections",
        "sketch_engine",
        "feature_geometry",
        "uv_transfer",
        "feature_tree",
        "human_engine",
        "asset_fetcher",
        "view_fusion",
        "polyhedron_engine",
        "view_capture",
        "image_sketch_engine",
        "character_stylizer",
        "face_fit",
        "human_assets",
        "hair_contact",
        "portrait_materials",
        "wardrobe",
        "oneshot_generators",
        "city_generator",
        "image_colors",
        "inspect_shading",
        "inspect_rays",
        "scene_mesh",
        "humanoid_skin",
        "inspection",
        "uv_bake",
        "sdf_engine",
        "ai_3d_provider",
        "scatter_support",
        "scene_layout",
        "appearance",
        "scene_history",
        "scene_state",
        "sketch_overlay",
        "ai_bridge",
    )
    active = hasattr(bpy.types.Scene, "excad")
    package_name = __name__
    loaded = [sys.modules[package_name + "." + n] for n in modules if package_name + "." + n in sys.modules]
    loaded.append(sys.modules[package_name])
    # Catch syntax errors before touching active operators or RNA properties.
    from pathlib import Path

    for module in loaded:
        compile(Path(module.__file__).read_bytes(), module.__file__, "exec")
    previous = [(module, module.__dict__.copy()) for module in loaded]
    old_classes, old_properties = CLASSES, EXCADSceneProps
    removed, added = [], []
    importlib.invalidate_caches()
    try:
        if active:
            sketch_overlay.unregister()
            del bpy.types.Scene.excad
            for cls in reversed(old_classes):
                bpy.utils.unregister_class(cls)
                removed.append(cls)
        for module in loaded:
            importlib.reload(module)
        package = sys.modules[package_name]
        if active:
            for cls in package.CLASSES:
                bpy.utils.register_class(cls)
                added.append(cls)
            bpy.types.Scene.excad = PointerProperty(type=package.EXCADSceneProps)
            sketch_overlay.register()
        server.__class__ = package.ai_bridge.AiBridgeServer
        package.ai_bridge.AiBridgeServer._instance = server
        server.history.__class__ = package.ai_bridge.SceneHistory
    except Exception:
        if active and hasattr(bpy.types.Scene, "excad"):
            del bpy.types.Scene.excad
        for cls in reversed(added):
            bpy.utils.unregister_class(cls)
        for module, state in previous:
            module.__dict__.clear()
            module.__dict__.update(state)
        for cls in reversed(removed):
            bpy.utils.register_class(cls)
        if active:
            bpy.types.Scene.excad = PointerProperty(type=old_properties)
            sketch_overlay.register()
        raise
