# SPDX-License-Identifier: GPL-2.0-or-later
"""Optional command groups for agents that need fields beyond the default short schema."""

TOPICS = {
    "sketch": (
        "sketch_new{sketch_id,plane?,offset?}; sketch_on_face{object,face,sketch_id}; "
        "sketch_add_rect{u0,v0,width,height}; sketch_add_circle{cu,cv,radius}; "
        "sketch_arc{points:[start,through,end]} or {center,radius,start_deg,angle_deg}; "
        "sketch_slot{length,width,center?,angle_deg?} or {start,end,width}; "
        "sketch_regular_polygon{radius,sides,center?,rotation?}; sketch_spline{points,closed?,steps?}; "
        "sketch_offset_loop{source,distance}; sketch_mirror2d{source|source_entity,axis,offset?}; "
        "sketch_constraint{ctype,a,b?,c?,d?,value?}. TANGENT/CONCENTRIC take entity IDs. "
        "MIDPOINT(a of b,c), SYMMETRIC(a,b around c,d), ANGLE(a,b to c,d in degrees) use point IDs."
    ),
    "cad": (
        "extrude{object,sketch?,depth,op?:ADD|CUT|INTERSECT,symmetric?,draft_angle?}; "
        "revolve{object,sketch?,axis,origin?,angle_deg?,segments?}; loft{object,sketch_ids,interpolation?,steps?}; "
        "sweep{object,profile_sketch,path_curve}; fillet{object,edges,radius}; chamfer{object,edges,width}; "
        "shell{object,thickness,open_faces}; pattern{object,type:LINEAR|CIRCULAR,count,spacing?,axis?,origin?}; "
        "mirror3d{object,axis,offset?}; feature_tree{name,recipe:{version:1,sketches,features}}; "
        "save_feature_tree{object,path?}; rebuild_feature_tree{object,path?|recipe?,parameters?}. "
        'Example parameters {"pad.depth":0.8,"sketches.base.shapes.0.width":3}.'
    ),
    "character": (
        "human_generate{name,preset?,rig?}; fit_face_from_image{object,image_path}; "
        "character_stylize{object,hair?,clothes?,remove?:[slot],skin?,eyes?,iris_color?,iris_size?,lip_tint?,lashes?,teeth?,expression?,hands?}. "
        "hair SHORT|BOB|PONYTAIL|LONG_STRAIGHT|LONG_WAVY|CURTAIN_BANGS. clothes list of TSHIRT|VNECK_SWEATER|HOODIE|JACKET|PANTS|SKIRT|SHOES. "
        "skin FAIR_IVORY|WARM_BEIGE|TAN|DEEP. expression SMILE|SERIOUS|SURPRISED|WINK. hands RELAXED|FIST|POINT|OPEN. "
        "install_asset_pack{pack:all|system|shirts|pants|skirts}; equip_asset{human,name,pack?}."
    ),
    "organic": (
        "sdf_generate{name,tree,resolution?:16..256,remesh?:AUTO|QUAD|VOXEL}; nodes accept center, rotation_deg XYZ, scale. "
        'Edit with sdf_generate{object,parameters:{"children.1.radius":0.4}}. '
        "sphere{radius},capsule{a,b,radius},ellipsoid{radii},round_box{size,radius},torus{major_radius,minor_radius},cone{height,radius,top_radius?}. "
        "smooth_union/subtract/intersect{children,k},mirror_x{child},radial_array{child,count},twist{child,degrees_per_unit},bend{child,radius}. "
        "poly_render{output_dir,poly_type?,center?,size?,img_res?,write_depth?}; poly_reconstruct{image_dir,poly_type,center,size,use_depth?,texture_mode?:UV|VERTEX}; poly_bake{object,image_dir,poly_type,center,size,texture_size?,use_depth?,path?}. "
        "generate_3d{provider:MESHY|TRIPO|HUNYUAN3D-2|TRELLIS,prompt?,image_path?,height?,remesh?,rig?:PARTS|HUMANOID,texture?,timeout?,name?}."
    ),
    "scene": (
        "place_relative{obj,target?,relation,gap?} ON_TOP_OF|LEFT_OF|RIGHT_OF|IN_FRONT_OF|BEHIND|INSIDE|ALIGN_FLOOR. "
        "look_at{obj,target}; scatter_on_surface{asset,surface_obj,count,min_dist,align_normal?,seed?,max_slope?}; "
        "check_intersections{objects?}; resolve_overlaps{objects?,fixed?,gap?}; set_material{object,preset,color?,roughness?,metallic?}; "
        "light_studio{target,style?:PRODUCT|PORTRAIT|SOFT|HEADSHOT,power?}; HEADSHOT frames the posed face with a camera. render_inspect{target?,views?,path?,size?}; "
        "checkpoint{name}; rollback{name}; undo{}; reload_addon{}; scene_digest{since_hash?,include_all?}."
    ),
}
