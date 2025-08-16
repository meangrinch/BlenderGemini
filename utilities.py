import base64
import heapq
import math
import os
import re
import tempfile
import time

import bmesh
import bpy
import mathutils
import requests


def get_api_key(context, addon_name):
    preferences = context.preferences
    addon_prefs = preferences.addons[addon_name].preferences
    return addon_prefs.api_key


def _ensure_object_mode_for_screenshot(preserve_edit_mode=True):
    """
    Prepare mode for screenshot safety.
    - If preserve_edit_mode is True and active object is in Edit Mode, keep Edit Mode.
    - Otherwise, switch to Object Mode for capture.
    Returns a callable to restore the previous mode, which is a no-op if nothing needs restoring.
    """
    try:
        obj = bpy.context.active_object
        if obj is None:
            return lambda: None
        prev_mode = obj.mode
        if prev_mode != 'OBJECT':
            if preserve_edit_mode and prev_mode == 'EDIT':
                # Keep Edit Mode; no switch, but still provide restore closure
                return lambda: None
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                # If we cannot switch, best effort: do not attempt restore
                return lambda: None

        def _restore():
            try:
                if obj and prev_mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode=prev_mode)
            except Exception:
                pass
        return _restore
    except Exception:
        return lambda: None


def init_props():
    bpy.types.Scene.gemini_chat_history = bpy.props.CollectionProperty(type=bpy.types.PropertyGroup)
    bpy.types.Scene.gemini_model = bpy.props.EnumProperty(
        name="Gemini Model",
        description="Select the Gemini model to use",
        items=[
            ("gemini-2.5-pro", "Gemini 2.5 Pro", "Use Gemini 2.5 Pro"),
            ("gemini-2.5-flash", "Gemini 2.5 Flash", "Use Gemini 2.5 Flash"),
            ("gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite", "Use Gemini 2.5 Flash Lite"),
            ("gemini-2.0-flash", "Gemini 2.0 Flash", "Use Gemini 2.0 Flash"),
            ("gemini-2.0-flash-lite", "Gemini 2.0 Flash Lite", "Use Gemini 2.0 Flash Lite"),
        ],
        default="gemini-2.5-flash",
    )
    bpy.types.Scene.gemini_chat_input = bpy.props.StringProperty(
        name="Message",
        description="Enter your message",
        default="",
    )
    bpy.types.Scene.gemini_button_pressed = bpy.props.BoolProperty(default=False)
    bpy.types.PropertyGroup.type = bpy.props.StringProperty()
    bpy.types.PropertyGroup.content = bpy.props.StringProperty()
    bpy.types.Scene.gemini_enable_thinking = bpy.props.BoolProperty(
        name="Enable Thinking",
        description="Enable model's thinking capabilities in the response (only for compatible models)",
        default=True,
    )
    bpy.types.Scene.gemini_enable_grounding = bpy.props.BoolProperty(
        name="Enable Grounding",
        description="Allow Gemini to use Google Search grounding for responses",
        default=False,
    )


def clear_props():
    del bpy.types.Scene.gemini_chat_history
    del bpy.types.Scene.gemini_chat_input
    del bpy.types.Scene.gemini_button_pressed
    del bpy.types.Scene.gemini_enable_thinking
    del bpy.types.Scene.gemini_enable_grounding


def make_gemini_api_request(url, headers, data):
    """Makes a request to the Gemini API with retry logic for handling errors."""
    max_retries = 5
    wait_time = 1
    max_wait_time = 16

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()

            return response.json()["candidates"][0]["content"]["parts"][0]["text"]

        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed (attempt {attempt + 1}/{max_retries}): {str(e)}"
            print(error_msg)

            if attempt < max_retries - 1:
                current_wait = min(wait_time * (2**attempt), max_wait_time)
                print(f"Retrying in {current_wait} seconds...")
                time.sleep(current_wait)
            else:
                print("Maximum retry attempts reached. Giving up.")
                return None
        except (KeyError, IndexError) as e:
            print(f"Error parsing API response: {str(e)}")
            return None


def capture_viewport_screenshot_base64(context, max_size=1024):
    """
    Capture a Viewport Render Image (3D view only, no UI) of the first available
    3D View and return it as a base64-encoded PNG string.

    If the captured image exceeds max_size in either dimension, it is downscaled
    to fit within max_size while preserving aspect ratio.

    Falls back to region-only screenshot if viewport render is unavailable.
    Returns None on failure.
    """
    try:
        # Preserve Edit Mode if active; otherwise enforce Object Mode
        restore_mode = _ensure_object_mode_for_screenshot(preserve_edit_mode=True)
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    for region in area.regions:
                        if region.type == "WINDOW":
                            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                            tmp_path = tmp_file.name
                            tmp_file.close()
                            loaded_img = None
                            try:
                                # Use a lightweight region-only screenshot to avoid heavy draw-engine paths
                                with context.temp_override(window=window, area=area, region=region):
                                    bpy.ops.screen.screenshot(filepath=tmp_path, full=False)

                                # Optionally downscale if larger than max_size
                                try:
                                    loaded_img = bpy.data.images.load(tmp_path)
                                    width, height = int(loaded_img.size[0]), int(loaded_img.size[1])
                                    largest = max(width, height)
                                    if max_size and largest > int(max_size):
                                        scale = float(max_size) / float(largest)
                                        new_w = max(1, int(round(width * scale)))
                                        new_h = max(1, int(round(height * scale)))
                                        loaded_img.scale(new_w, new_h)
                                        loaded_img.filepath_raw = tmp_path
                                        loaded_img.file_format = 'PNG'
                                        try:
                                            loaded_img.save()
                                        except Exception:
                                            pass
                                finally:
                                    if loaded_img is not None:
                                        try:
                                            bpy.data.images.remove(loaded_img)
                                        except Exception:
                                            pass

                                with open(tmp_path, "rb") as f:
                                    img_bytes = f.read()
                                return base64.b64encode(img_bytes).decode("ascii")
                            finally:
                                try:
                                    os.remove(tmp_path)
                                except Exception:
                                    pass
        return None
    except Exception:
        return None
    finally:
        try:
            restore_mode()
        except Exception:
            pass


def get_scene_objects_as_text(context):
    """
    Scans the current Blender scene and returns a text summary of the visible objects.
    This helps the AI understand the current state of the scene.
    """
    objects = [obj for obj in context.scene.objects if obj.visible_get()]
    if not objects:
        return "The current scene contains no visible objects."

    scene_summary = "Visible Scene Objects:\n"
    for obj in objects:
        scene_summary += f"- Object Name: `{obj.name}`, Type: `{obj.type}`"
        if obj.type == "MESH":
            scene_summary += f", Vertices: {len(obj.data.vertices)}, Faces: {len(obj.data.polygons)}"
        scene_summary += f", Location: {obj.location}\n"
    return scene_summary


def get_blender_version_string():
    """
    Return a concise Blender version string like "4.2.0". Falls back to tuple if needed.
    """
    try:
        vs = getattr(bpy.app, "version_string", None)
        if isinstance(vs, str) and vs.strip():
            return vs.strip()
        ver_tuple = getattr(bpy.app, "version", None)
        if isinstance(ver_tuple, (tuple, list)) and len(ver_tuple) >= 2:
            return ".".join(str(p) for p in ver_tuple[:3])
    except Exception:
        pass
    return "Unknown"


def _summarize_objects_near_cursor(context, max_count=3):
    """
    Return a short text list of the nearest visible objects to the 3D cursor.
    Uses object world positions and bounding box corners for a simple proximity metric.
    """
    try:
        cursor_loc = context.scene.cursor.location
    except Exception:
        return "None"

    candidates = []
    for obj in context.scene.objects:
        try:
            if not obj.visible_get():
                continue
        except Exception:
            continue

        try:
            center = obj.matrix_world.to_translation()
            center_dist = (center - cursor_loc).length

            # Prefer a closer bounding-box corner if available (for large objects)
            min_corner_dist = center_dist
            if getattr(obj, "bound_box", None):
                try:
                    min_corner_dist = min(
                        (obj.matrix_world @ mathutils.Vector(corner) - cursor_loc).length
                        for corner in obj.bound_box
                    )
                except Exception:
                    pass

            proximity = min(center_dist, min_corner_dist)
            candidates.append((proximity, obj))
        except Exception:
            continue

    if not candidates:
        return "None"

    candidates.sort(key=lambda item: item[0])
    lines = []
    for dist, obj in candidates[: max_count if max_count and max_count > 0 else 3]:
        lines.append(f"- `{obj.name}` (Type: {obj.type}) — Distance: {dist:.4f}")
    return "\n".join(lines)


def _build_3d_cursor_context_block(context, include_nearest=True, nearest_count=3):
    """
    Compose a rich context block describing the 3D Cursor's transform and nearby objects.
    """
    try:
        cursor = context.scene.cursor
        loc = cursor.location
        rot_euler = cursor.rotation_euler
        rot_deg = tuple(math.degrees(a) for a in rot_euler)
        mat = cursor.matrix.copy()
        basis = mat.to_3x3()
        right = (basis @ mathutils.Vector((1.0, 0.0, 0.0))).normalized()
        up = (basis @ mathutils.Vector((0.0, 1.0, 0.0))).normalized()
        forward = (basis @ mathutils.Vector((0.0, 0.0, 1.0))).normalized()

        block = (
            "**[3D CURSOR (World Space)]:**\n"
            f"- Location: ({loc.x:.4f}, {loc.y:.4f}, {loc.z:.4f})\n"
            f"- Rotation (Euler, radians): ({rot_euler.x:.5f}, {rot_euler.y:.5f}, {rot_euler.z:.5f})\n"
            f"- Rotation (Euler, degrees): ({rot_deg[0]:.2f}, {rot_deg[1]:.2f}, {rot_deg[2]:.2f})\n"
            "- Orientation axes (unit vectors):\n"
            f"  - Right +X: ({right.x:.4f}, {right.y:.4f}, {right.z:.4f})\n"
            f"  - Up    +Y: ({up.x:.4f}, {up.y:.4f}, {up.z:.4f})\n"
            f"  - Fwd   +Z: ({forward.x:.4f}, {forward.y:.4f}, {forward.z:.4f})\n"
        )

        if include_nearest:
            nearby = _summarize_objects_near_cursor(context, max_count=nearest_count)
            block += "- Objects nearest to cursor:\n" + nearby + "\n"

        return block
    except Exception:
        return ""


def _choose_cursor_target_object(context):
    """
    Choose a target object for cursor-based operations:
    - Prefer the active selected visible object if present.
    - Otherwise, choose the nearest visible MESH object to the 3D cursor.
    Returns (obj, reason, distance).
    """
    try:
        cursor_loc = context.scene.cursor.location
    except Exception:
        return (None, "No cursor in scene", None)

    active = getattr(context, "active_object", None)
    if active is not None:
        try:
            if active.visible_get():
                return (active, "active selection", (active.matrix_world.to_translation() - cursor_loc).length)
        except Exception:
            pass

    # Fallback to nearest visible MESH
    nearest = None
    nearest_dist = None
    for obj in context.scene.objects:
        try:
            if not obj.visible_get() or obj.type != "MESH":
                continue
            dist = (obj.matrix_world.to_translation() - cursor_loc).length
            if nearest is None or dist < nearest_dist:
                nearest = obj
                nearest_dist = dist
        except Exception:
            continue

    if nearest is not None:
        return (nearest, "nearest visible mesh", nearest_dist)

    return (None, "No visible candidates", None)


def _build_cursor_target_object_block(context):
    """
    If a reasonable object candidate exists, provide a block describing it along with
    the cursor expressed in that object's local space.
    """
    try:
        obj, reason, dist = _choose_cursor_target_object(context)
        if obj is None:
            return "**[CURSOR TARGET OBJECT]:** None\n"

        cursor_ws = context.scene.cursor.location.copy()
        local_cursor = obj.matrix_world.inverted() @ cursor_ws
        block = (
            "**[CURSOR TARGET OBJECT]:**\n"
            f"- Name: `{obj.name}` (Type: {obj.type}) — Reason: {reason}"
        )
        if dist is not None:
            block += f" — Distance: {dist:.4f}\n"
        else:
            block += "\n"
        block += (
            f"- Cursor in `{obj.name}` local space: ("
            f"{local_cursor.x:.4f}, {local_cursor.y:.4f}, {local_cursor.z:.4f}"
            ")\n"
        )
        return block
    except Exception:
        return ""


def get_detailed_object_data(obj):
    """
    Serializes the geometry of a single Blender object into a detailed text format.
    Includes a limit to avoid excessively long outputs for high-poly meshes.
    """
    if not obj or obj.type != "MESH":
        return "No mesh object selected for detailed analysis."

    data = obj.data
    vertex_limit = 500
    face_limit = 1000

    summary = f"Detailed Geometry for Object: `{obj.name}`\n"
    summary += "- Type: MESH\n"
    summary += f"- Vertex count: {len(data.vertices)}\n"
    summary += f"- Face count: {len(data.polygons)}\n"

    if len(data.vertices) > vertex_limit or len(data.polygons) > face_limit:
        summary += (
            f"- NOTE: Geometry is too dense to display full details "
            f"(limit: {vertex_limit} vertices, {face_limit} faces).\\n"
        )
        return summary

    summary += "Vertices (x, y, z):\\n"
    for v in data.vertices:
        summary += f"  - ({v.co.x:.4f}, {v.co.y:.4f}, {v.co.z:.4f})\n"

    summary += "Faces (vertex indices):\n"
    for f in data.polygons:
        summary += f"  - {list(f.vertices)}\n"

    return summary


def _compute_falloff_weight(t, mode="SMOOTH"):
    """
    Compute a [0,1] falloff weight given normalized distance t in [0,1].
    mode="SMOOTH" uses a smoothstep curve; mode="LINEAR" uses linear.
    """
    if t <= 0.0:
        return 1.0
    if t >= 1.0:
        return 0.0
    mode_upper = str(mode).upper()
    if mode_upper == "LINEAR":
        return 1.0 - t
    if mode_upper == "GAUSSIAN":
        # Zero-mean Gaussian with sigma=0.35 over t in [0,1]
        sigma = 0.35
        x = min(max(t, 0.0), 1.0)
        return math.exp(-(x * x) / (2.0 * sigma * sigma))
    if mode_upper == "PLATEAU":
        # Flat center up to 0.5, then smooth falloff to 0
        if t <= 0.5:
            return 1.0
        # Remap [0.5,1] -> [0,1]
        x = (t - 0.5) / 0.5
        return 1.0 - (3.0 * x * x - 2.0 * x * x * x)
    # Default SMOOTHSTEP: 1 - (3t^2 - 2t^3)
    return 1.0 - (3.0 * t * t - 2.0 * t * t * t)


def get_vertices_in_radius(obj, center_local, radius):
    """
    Return a list of vertex indices in `obj` whose local-space positions are within
    `radius` of `center_local` (Vector in object local space).
    """
    try:
        mesh = obj.data
        bm = None
        is_edit_mode = False
        try:
            is_edit_mode = getattr(obj, "mode", "OBJECT") == 'EDIT'
        except Exception:
            is_edit_mode = False

        if is_edit_mode:
            bm = bmesh.from_edit_mesh(mesh)
        else:
            bm = bmesh.new()
            bm.from_mesh(mesh)

        bm.verts.ensure_lookup_table()

        center_vec = (
            center_local if isinstance(center_local, mathutils.Vector)
            else mathutils.Vector((center_local.x, center_local.y, center_local.z))
        )
        in_radius = []
        r2 = radius * radius
        for v in bm.verts:
            if (v.co - center_vec).length_squared <= r2:
                in_radius.append(v.index)

        if not is_edit_mode and bm is not None:
            bm.free()
        return in_radius
    except Exception:
        return []


def get_vertices_in_geodesic_radius(obj, center_world, radius):
    """
    Approximate geodesic selection: returns vertex indices whose shortest path
    distance (along mesh edges) from the nearest vertex to `center_world` is <= radius.
    """
    try:
        mesh = obj.data
        bm = None
        is_edit_mode = False
        try:
            is_edit_mode = getattr(obj, "mode", "OBJECT") == 'EDIT'
        except Exception:
            is_edit_mode = False

        if is_edit_mode:
            bm = bmesh.from_edit_mesh(mesh)
        else:
            bm = bmesh.new()
            bm.from_mesh(mesh)

        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()

        # Find start vertex nearest to center
        local_center = obj.matrix_world.inverted() @ center_world
        start_idx = 0
        min_d2 = float("inf")
        for v in bm.verts:
            d2 = (v.co - local_center).length_squared
            if d2 < min_d2:
                min_d2 = d2
                start_idx = v.index

        # Dijkstra/BFS with a min-heap
        distances = {start_idx: 0.0}
        heap = [(0.0, start_idx)]
        in_radius = set()
        while heap:
            dist_u, idx_u = heapq.heappop(heap)
            if dist_u > radius:
                break
            if idx_u in in_radius:
                continue
            in_radius.add(idx_u)
            v = bm.verts[idx_u]
            for e in v.link_edges:
                w = e.other_vert(v)
                cand = dist_u + (w.co - v.co).length
                if cand <= radius and (w.index not in distances or cand < distances[w.index]):
                    distances[w.index] = cand
                    heapq.heappush(heap, (cand, w.index))

        if not is_edit_mode and bm is not None:
            bm.free()
        return sorted(in_radius)
    except Exception:
        return []


def _world_bbox_stats(obj):
    """Return (size_vec, diag_len) of object's world-space AABB."""
    try:
        corners = [obj.matrix_world @ mathutils.Vector(c) for c in getattr(obj, "bound_box", [])]
        if not corners:
            return (mathutils.Vector((1.0, 1.0, 1.0)), 1.0)
        min_x = min(v.x for v in corners)
        max_x = max(v.x for v in corners)
        min_y = min(v.y for v in corners)
        max_y = max(v.y for v in corners)
        min_z = min(v.z for v in corners)
        max_z = max(v.z for v in corners)
        size = mathutils.Vector((max_x - min_x, max_y - min_y, max_z - min_z))
        return (size, size.length)
    except Exception:
        return (mathutils.Vector((1.0, 1.0, 1.0)), 1.0)


def _build_bvh_from_object(obj):
    """
    Build and return a BVHTree for `obj` in its local space. Returns None on failure.
    """
    try:
        from mathutils.bvhtree import BVHTree
    except Exception:
        return None

    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        try:
            verts = [v.co[:] for v in mesh.vertices]
            polys = [p.vertices[:] for p in mesh.polygons]
            if not verts or not polys:
                return None
            bvh = BVHTree.FromPolygons(verts, polys)
            return bvh
        finally:
            try:
                eval_obj.to_mesh_clear()
            except Exception:
                pass
    except Exception:
        return None


def raycast_surface(obj, origin_world, direction_world, max_distance=1000.0):
    """
    Raycast against `obj` surface.
    Returns (hit_world, normal_world, index, distance) or (None, None, None, None).
    """
    try:
        bvh = _build_bvh_from_object(obj)
        if bvh is None:
            return (None, None, None, None)

        inv = obj.matrix_world.inverted()
        o_local = inv @ origin_world
        # Transform direction as a vector
        d_local = inv.to_3x3() @ direction_world
        if d_local.length <= 1e-12:
            return (None, None, None, None)
        d_local.normalize()

        hit = bvh.ray_cast(o_local, d_local, max_distance)
        if hit is None or hit[0] is None:
            return (None, None, None, None)
        loc_l, normal_l, index, dist = hit
        loc_w = obj.matrix_world @ loc_l
        n_w = (obj.matrix_world.to_3x3() @ normal_l).normalized()
        return (loc_w, n_w, index, dist)
    except Exception:
        return (None, None, None, None)


def project_point_to_surface_near(obj, guess_world, max_distance=1000.0):
    """
    Find the nearest surface point on `obj` to the given world-space guess.
    Returns (hit_world, normal_world) or (None, None).
    """
    try:
        bvh = _build_bvh_from_object(obj)
        if bvh is None:
            return (None, None)
        inv = obj.matrix_world.inverted()
        guess_local = inv @ guess_world
        res = bvh.find_nearest(guess_local, max_distance)
        if not res or res[0] is None:
            return (None, None)
        loc_l, normal_l, _index, _dist = res
        loc_w = obj.matrix_world @ loc_l
        n_w = (obj.matrix_world.to_3x3() @ normal_l).normalized()
        return (loc_w, n_w)
    except Exception:
        return (None, None)


def ensure_subsurf_for_local_detail(obj, target_levels=2):
    """
    Ensure the object has at least a Subdivision Surface modifier with the desired levels.
    Non-destructive; increases levels if lower.
    """
    try:
        mod = None
        for m in obj.modifiers:
            if m.type == 'SUBSURF':
                mod = m
                break
        if mod is None:
            mod = obj.modifiers.new(name="GeminiDetailSubdivision", type='SUBSURF')
        mod.levels = max(mod.levels, target_levels)
        mod.render_levels = max(mod.render_levels, target_levels)
        return mod
    except Exception:
        return None


def get_local_geometry_patch_text(obj, center_world, radius, vertex_limit=2000, face_limit=4000):
    """
    Returns a compact text of vertices and faces within a geodesic radius around center_world.
    Limits to keep the prompt bounded.
    """
    try:
        indices = get_vertices_in_geodesic_radius(obj, center_world, radius)
        if not indices:
            return "No local geometry found near the cursor."
        index_set = set(indices)

        data = obj.data
        verts_local = [v.co for v in data.vertices]
        # Map old idx -> new sequential idx within patch
        remap = {idx: i for i, idx in enumerate(indices)}

        # Collect faces whose all vertices are in index_set
        faces = []
        for poly in data.polygons:
            v_idx = list(poly.vertices)
            if all(i in index_set for i in v_idx):
                faces.append([remap[i] for i in v_idx])
                if len(faces) >= face_limit:
                    break

        # Build text
        out = []
        out.append(f"Local Geometry Patch (radius={radius:.4f}m) for `{obj.name}`")
        out.append(f"- Vertices: {min(len(indices), vertex_limit)} (capped)")
        out.append(f"- Faces: {len(faces)}")
        out.append("Vertices (local space x, y, z):")
        for i, idx in enumerate(indices[:vertex_limit]):
            co = verts_local[idx]
            out.append(f"  - ({co.x:.4f}, {co.y:.4f}, {co.z:.4f})")
        out.append("Faces (vertex indices within patch):")
        for f in faces:
            out.append(f"  - {f}")
        return "\n".join(out)
    except Exception:
        return "Local geometry patch unavailable due to an error."


def apply_six_pack(
    context,
    obj=None,
    cursor_world=None,
    row_spacing=None,
    col_spacing=None,
    radius=None,
    ridge_strength=None,
    valley_strength=None,
    falloff="GAUSSIAN",
    add_detail=True,
):
    """
    Stamp a stylized abdominal six-pack near the 3D cursor.
    Heuristically determines spacing, radius, and strengths from the object's size.

    Returns True on success, False otherwise.
    """
    try:
        if obj is None:
            obj, _reason, _dist = _choose_cursor_target_object(context)
        if obj is None or obj.type != 'MESH':
            return False

        cursor = context.scene.cursor
        if cursor_world is None:
            cursor_world = cursor.location.copy()
        basis = cursor.matrix.copy().to_3x3()
        right = (basis @ mathutils.Vector((1.0, 0.0, 0.0))).normalized()
        up = (basis @ mathutils.Vector((0.0, 1.0, 0.0))).normalized()

        # Heuristic sizing from world AABB
        _size, diag = _world_bbox_stats(obj)
        diag = max(diag, 1e-3)
        default_radius = max(0.02, min(0.08, 0.05 * diag))
        default_row = 1.4 * default_radius
        default_col = 1.6 * default_radius
        default_ridge = -0.75 * default_radius
        default_valley = +0.35 * default_radius

        radius = float(radius) if radius is not None else default_radius
        row_spacing = float(row_spacing) if row_spacing is not None else default_row
        col_spacing = float(col_spacing) if col_spacing is not None else default_col
        ridge_strength = float(ridge_strength) if ridge_strength is not None else default_ridge
        valley_strength = float(valley_strength) if valley_strength is not None else default_valley

        if add_detail:
            ensure_subsurf_for_local_detail(obj, target_levels=2)

        # Compute six centers (3 rows x 2 cols) around cursor
        centers_guess = []
        for row_k in (-row_spacing, 0.0, row_spacing):
            # left and right columns
            centers_guess.append(cursor_world + up * row_k - right * (0.5 * col_spacing))
            centers_guess.append(cursor_world + up * row_k + right * (0.5 * col_spacing))

        # Project centers to the mesh surface
        surface_centers = []
        for g in centers_guess:
            hit, n = project_point_to_surface_near(obj, g)
            if hit is None:
                # fall back to guess point
                hit = g
                n = up
            surface_centers.append((hit, n))

        # Apply ridges (fatten outward)
        for hit, _n in surface_centers:
            local_center = obj.matrix_world.inverted() @ hit
            apply_radial_shrink_fatten(
                obj,
                local_center,
                radius,
                ridge_strength,
                falloff=falloff,
                mirror=False,
            )

        # Apply vertical valleys between rows at the midline
        midline_points = [cursor_world + up * k for k in (-row_spacing, 0.0, row_spacing)]
        for p in midline_points:
            hit, _n = project_point_to_surface_near(obj, p)
            if hit is None:
                hit = p
            local_center = obj.matrix_world.inverted() @ hit
            apply_radial_shrink_fatten(
                obj,
                local_center,
                radius * 0.6,
                valley_strength,
                falloff=falloff,
                mirror=False,
            )

        # Light separation between left/right columns along three rows
        row_points = [cursor_world + up * k for k in (-row_spacing, 0.0, row_spacing)]
        for rp in row_points:
            # place a small valley at the center between left and right
            hit, _n = project_point_to_surface_near(obj, rp)
            if hit is None:
                hit = rp
            local_center = obj.matrix_world.inverted() @ hit
            apply_radial_shrink_fatten(
                obj,
                local_center,
                radius * 0.45,
                valley_strength * 0.8,
                falloff=falloff,
                mirror=False,
            )

        # Update the mesh after modifications
        try:
            obj.data.update()
        except Exception:
            pass
        return True
    except Exception:
        return False


def apply_radial_shrink_fatten(
    obj,
    center_local,
    radius,
    strength,
    falloff="SMOOTH",
    mirror=False,
    mirror_axis="X",
):
    """
    Non-destructively shrink/fatten vertices within a radius around `center_local`
    (in object local space) by displacing them along their vertex normals.

    - `strength` > 0 shrinks (moves inward), < 0 fattens (moves outward).
    - `falloff`: "SMOOTH" | "LINEAR" for radial weighting.
    - If `mirror` is True, applies the same operation around the mirrored center
      across the given local axis ("X"|"Y"|"Z").
    """
    try:
        mesh = obj.data
        is_edit_mode = False
        try:
            is_edit_mode = getattr(obj, "mode", "OBJECT") == 'EDIT'
        except Exception:
            is_edit_mode = False

        if is_edit_mode:
            bm = bmesh.from_edit_mesh(mesh)
        else:
            bm = bmesh.new()
            bm.from_mesh(mesh)

        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        bm.normal_update()

        def _apply(center_vec):
            r = float(radius)
            if r <= 0.0:
                return
            r2 = r * r
            for v in bm.verts:
                offset = v.co - center_vec
                d2 = offset.length_squared
                if d2 > r2:
                    continue
                d = math.sqrt(d2)
                w = _compute_falloff_weight(d / r, falloff)
                n = v.normal.normalized()
                v.co = v.co - n * (strength * w)

        # Ensure Vector type
        center_vec = center_local if isinstance(center_local, mathutils.Vector) else mathutils.Vector(center_local)
        _apply(center_vec)

        if mirror:
            axis = str(mirror_axis).upper()
            if axis in {"X", "Y", "Z"}:
                mirrored_center = center_vec.copy()
                if axis == "X":
                    mirrored_center.x = -mirrored_center.x
                elif axis == "Y":
                    mirrored_center.y = -mirrored_center.y
                else:
                    mirrored_center.z = -mirrored_center.z
                if (mirrored_center - center_vec).length > 1e-8:
                    _apply(mirrored_center)

        if is_edit_mode:
            bmesh.update_edit_mesh(mesh)
        else:
            bm.to_mesh(mesh)
            bm.free()
            mesh.update()
        return True
    except Exception:
        return False


def generate_blender_code(
    prompt,
    chat_history,
    context,
    system_prompt,
    detailed_geometry=None,
    use_3d_cursor=False,
    include_viewport_screenshot=False,
):
    api_key = get_api_key(context, "BlenderGemini")

    preferences = context.preferences
    addon_prefs = preferences.addons["BlenderGemini"].preferences

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + context.scene.gemini_model
        + ":generateContent"
    )
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    contents = []
    for message in chat_history[-10:]:  # Keep last 10 messages for context
        role = "user" if message.type == "user" else "model"
        content = message.content if message.type == "user" else "```\n" + message.content + "\n```"
        contents.append({"role": role, "parts": [{"text": content}]})

    scene_context = get_scene_objects_as_text(context)
    full_prompt = ""
    blender_version = get_blender_version_string()
    # Subtle environment hint
    full_prompt += f"(Blender {blender_version})\n\n"
    if detailed_geometry:
        full_prompt += "**Detailed Object Geometry:**\n" + detailed_geometry + "\n\n"

    if use_3d_cursor:
        full_prompt += _build_3d_cursor_context_block(context)
        full_prompt += _build_cursor_target_object_block(context)
        # Advertise built-in helpers for precise localized edits
        full_prompt += (
            "**[AVAILABLE HELPERS]:**\n"
            "- apply_radial_shrink_fatten(obj, center_local, radius, strength,\n"
            "  falloff=\"SMOOTH\", mirror=False, mirror_axis=\"X\")\n"
            "- get_vertices_in_radius(obj, center_local, radius)\n"
            "- get_vertices_in_geodesic_radius(obj, center_world, radius)\n"
            "- raycast_surface(obj, origin_world, direction_world, max_distance=1000.0)\n"
            "- project_point_to_surface_near(obj, guess_world, max_distance=1000.0)\n"
            "- ensure_subsurf_for_local_detail(obj, target_levels=2)\n"
            "- get_local_geometry_patch_text(obj, center_world, radius,\n"
            "  vertex_limit=2000, face_limit=4000)\n"
            "- apply_six_pack(context, obj=None, cursor_world=None, row_spacing=None,\n"
            "  col_spacing=None, radius=None, ridge_strength=None, valley_strength=None,\n"
            "  falloff=\"GAUSSIAN\", add_detail=True)\n\n"
        )
        full_prompt += "\n"

        # Automatically include a compact local geometry patch around the cursor
        try:
            target_obj, _reason, _dist = _choose_cursor_target_object(context)
            if target_obj is not None:
                cursor_ws = context.scene.cursor.location.copy()
                r = getattr(context.scene, "gemini_edit_radius", 0.12) * 2.0
                patch_text = get_local_geometry_patch_text(target_obj, cursor_ws, r)
                full_prompt += "**[LOCAL GEOMETRY PATCH]:**\n" + patch_text + "\n\n"
        except Exception:
            pass

    if include_viewport_screenshot:
        full_prompt += (
            "**Viewport Screenshot:** Attached below. Use it to understand view "
            "orientation and selection state.\n\n"
        )

    full_prompt += "**Scene Summary:**\n" + scene_context + "\n\nUser Request: " + prompt

    user_parts = [{"text": full_prompt}]
    if include_viewport_screenshot:
        image_b64 = capture_viewport_screenshot_base64(context)
        if image_b64:
            user_parts.append(
                {"inlineData": {"mimeType": "image/png", "data": image_b64}}
            )

    contents.append({"role": "user", "parts": user_parts})

    safety_settings_config = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    data = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {
            "temperature": addon_prefs.temperature,
            "topP": addon_prefs.top_p,
            "topK": addon_prefs.top_k,
        },
        "safetySettings": safety_settings_config,
    }

    # Optionally enable Google Search grounding
    try:
        if getattr(context.scene, "gemini_enable_grounding", False):
            data["tools"] = [{"googleSearch": {}}]
    except Exception:
        pass

    # Configure thinking budget for Gemini 2.5 series models
    model_name = context.scene.gemini_model or ""
    if model_name.startswith("gemini-2.5-"):
        if "pro" in model_name:
            # Gemini 2.5 Pro always thinks
            data["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 32768}
        else:
            # Flash / Flash Lite can be toggled
            budget = 0 if not context.scene.gemini_enable_thinking else 24576
            data["generationConfig"]["thinkingConfig"] = {"thinkingBudget": budget}

    response_text = make_gemini_api_request(url, headers, data)
    if response_text:
        # Extract code between ```python and ``` markers
        code_match = re.search(r"```(?:python)?(.*?)```", response_text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return response_text.strip()
    return None


def fix_blender_code(
    original_code,
    error_message,
    context,
    system_prompt,
    detailed_geometry=None,
    use_3d_cursor=False,
    include_viewport_screenshot=False,
):
    """Generate fixed Blender code based on the error message."""
    api_key = get_api_key(context, "BlenderGemini")

    preferences = context.preferences
    addon_prefs = preferences.addons["BlenderGemini"].preferences

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + context.scene.gemini_model
        + ":generateContent"
    )
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    scene_context = get_scene_objects_as_text(context)

    detailed_geo_block = ""
    if detailed_geometry:
        detailed_geo_block = f"""
**[DETAILED GEOMETRY OF SELECTED OBJECT]:**
```
{detailed_geometry}
```
"""

    cursor_block = ""
    if use_3d_cursor:
        cursor_block = _build_3d_cursor_context_block(context) + _build_cursor_target_object_block(context)
        cursor_block += (
            "**[AVAILABLE HELPERS]:**\n"
            "- apply_radial_shrink_fatten(obj, center_local, radius, strength, falloff=\"SMOOTH\",\n"
            "  mirror=False, mirror_axis=\"X\")\n"
            "- get_vertices_in_radius(obj, center_local, radius)\n"
            "- get_vertices_in_geodesic_radius(obj, center_world, radius)\n"
            "- raycast_surface(obj, origin_world, direction_world, max_distance=1000.0)\n"
            "- project_point_to_surface_near(obj, guess_world, max_distance=1000.0)\n"
            "- ensure_subsurf_for_local_detail(obj, target_levels=2)\n"
            "- get_local_geometry_patch_text(obj, center_world, radius, vertex_limit=2000,\n"
            "  face_limit=4000)\n"
            "- apply_six_pack(context, obj=None, cursor_world=None, row_spacing=None, col_spacing=None,\n"
            "  radius=None, ridge_strength=None, valley_strength=None, falloff=\"GAUSSIAN\", add_detail=True)\n"
        )

    screenshot_block = ""
    if include_viewport_screenshot:
        screenshot_block = """
**[VIEWPORT SCREENSHOT]:**
Attached below. Use it to infer view orientation and selection state, and to understand the scene context.
"""

    # Auto include a local geometry patch when cursor targeting is enabled
    patch_block = ""
    try:
        if use_3d_cursor:
            target_obj, _reason, _dist = _choose_cursor_target_object(context)
            if target_obj is not None:
                cursor_ws = context.scene.cursor.location.copy()
                r = getattr(context.scene, "gemini_edit_radius", 0.12) * 2.0
                patch_text = get_local_geometry_patch_text(target_obj, cursor_ws, r)
                patch_block = f"""
**[LOCAL GEOMETRY PATCH]:**
```
{patch_text}
```
"""
    except Exception:
        pass

    blender_version = get_blender_version_string()

    fix_prompt = f"""### Persona
You are a `bpy` Debugging Specialist. Your sole function is to analyze the provided faulty Python script and its corresponding error message, and then generate a corrected, fully functional version.

### Task Context
Environment: Blender {blender_version}
You will be given a script that failed, its error, and a scene summary. Use all of this information to provide a fix.
{detailed_geo_block}
{cursor_block}
{screenshot_block}
{patch_block}
**[SCENE SUMMARY]:**
```
{scene_context}
```

**[FAULTY SCRIPT]:**
```python
{original_code}
```

**[ERROR TRACEBACK]:**
```
{error_message}
```

### Core Directives for Correction

1.  **Root Cause Analysis:** Your first step is to perform a root cause analysis. Meticulously trace the error from the `[ERROR TRACEBACK]` to the specific line and function call in the `[FAULTY SCRIPT]`. Understand *why* the error occurred (e.g., incorrect parameter, wrong object type, context issue).

2.  **Surgical Correction:** The goal is precision. Make the minimum necessary changes to the code to resolve the error. Avoid refactoring or altering code that is unrelated to the bug.

3.  **Preserve Original Intent:** The corrected script **must** achieve the exact same outcome that the `[FAULTY SCRIPT]` was intended for. Do not remove or comment out functionality to bypass the error; fix the underlying issue.

4.  **Maintain Coding Standards:** The fix must adhere to `bpy` best practices.
    -   **API Preference:** Use the Data API (`bpy.data`) over the Operator API (`bpy.ops`) for property modifications.
    -   **Parameter Integrity:** Ensure all function/operator parameters are valid and exist in the API. Do not invent arguments. This is a common source of errors.

5.  **Strict Output Mandate:**
    -   Your response **MUST** be only the complete, corrected, and executable Python script.
    -   Enclose the entire script in a single Python code block.
    -   **DO NOT** include any conversational text, explanations, summaries of changes, or apologies. Your output will be executed directly."""  # noqa

    parts = [{"text": fix_prompt}]
    if include_viewport_screenshot:
        image_b64 = capture_viewport_screenshot_base64(context)
        if image_b64:
            parts.append(
                {"inlineData": {"mimeType": "image/png", "data": image_b64}}
            )

    data = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": addon_prefs.temperature,
            "topP": addon_prefs.top_p,
            "topK": addon_prefs.top_k,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    }

    # Optionally enable Google Search grounding
    try:
        if getattr(context.scene, "gemini_enable_grounding", False):
            data["tools"] = [{"googleSearch": {}}]
    except Exception:
        pass

    # Configure thinking budget for Gemini 2.5 series models
    model_name = context.scene.gemini_model or ""
    if model_name.startswith("gemini-2.5-"):
        if "pro" in model_name:
            # Gemini 2.5 Pro always thinks
            data["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 32768}
        else:
            # Flash / Flash Lite can be toggled
            budget = 0 if not context.scene.gemini_enable_thinking else 32768
            data["generationConfig"]["thinkingConfig"] = {"thinkingBudget": budget}

    response_text = make_gemini_api_request(url, headers, data)
    if response_text:
        # Extract code between ```python and ``` markers
        code_match = re.search(r"```(?:python)?(.*?)```", response_text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return response_text.strip()
    return None


def split_area_to_text_editor(context):
    area = context.area
    for region in area.regions:
        if region.type == "WINDOW":
            with context.temp_override(area=area, region=region):
                bpy.ops.screen.area_split(direction="VERTICAL", factor=0.5)
            break

    new_area = context.screen.areas[-1]
    new_area.type = "TEXT_EDITOR"
    return new_area
