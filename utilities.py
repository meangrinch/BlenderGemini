import base64
import contextlib
import heapq
import inspect
import json
import math
import os
import re
import tempfile

import bmesh
import bpy
import mathutils
import requests

GEMINI_INTERACTIONS_URL = (
    "https://generativelanguage.googleapis.com/v1beta/interactions"
)
API_REQUEST_TIMEOUT = 300
LOCAL_PATCH_RADIUS = 0.24
MAX_SUMMARY_OBJECTS = 200
# A rejected previous_interaction_id (expired or unknown) comes back as one of these.
STALE_INTERACTION_STATUSES = (400, 404)
# Reused so status polls keep one TLS connection open instead of reconnecting each time.
_http_session = requests.Session()


def get_api_key(context, addon_name):
    preferences = context.preferences
    addon_prefs = preferences.addons[addon_name].preferences
    return addon_prefs.api_key


def _ensure_object_mode_for_screenshot():
    """
    Keep Object or Edit Mode as is; switch any other mode to Object Mode for capture.
    Returns a callable that restores the previous mode.
    """
    try:
        obj = bpy.context.active_object
        if obj is None or obj.mode in {"OBJECT", "EDIT"}:
            return lambda: None
        prev_mode = obj.mode
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        return lambda: None

    def _restore():
        with contextlib.suppress(Exception):
            bpy.ops.object.mode_set(mode=prev_mode)

    return _restore


class GEMINI_ChatMessage(bpy.types.PropertyGroup):
    """One chat-history entry (user or assistant)."""

    type: bpy.props.StringProperty(
        name="Type",
        description="Message role: user or assistant",
        default="",
    )
    content: bpy.props.StringProperty(
        name="Content",
        description="Message text or generated code",
        default="",
    )
    interaction_id: bpy.props.StringProperty(
        name="Interaction ID",
        description="Gemini Interactions API id for multi-turn continuity",
        default="",
    )


XHIGH_THINKING_MODELS = ("gemini-4-hydrogen",)
THINKING_LEVEL_MODEL_PREFIXES = ("gemini-3", "gemini-4")
# Models that use background long decode, where the server chains output hops past 64K.
LONG_DECODE_MODELS = ("gemini-4-carbon", "gemini-4-hydrogen")
LONG_DECODE_MAX_OUTPUT_TOKENS = 262144
INTERACTION_POLL_INTERVAL = 2


def get_thinking_level_items(self, context):
    # Explicit enum numbers keep "high" (0) as the default even when "xhigh" is prepended,
    # and skip 3 so old scenes saved with "minimal" (index 3) don't map to "xhigh".
    items = [
        ("high", "High", "Deeper thinking for difficult tasks", "", 0),
        ("medium", "Medium", "Balanced thinking for most tasks", "", 1),
        ("low", "Low", "Lighter thinking for faster responses", "", 2),
    ]
    if context is None or getattr(context, "scene", None) is None:
        return items

    model_name = getattr(context.scene, "gemini_model", "")
    if model_name in XHIGH_THINKING_MODELS:
        items.insert(
            0,
            ("xhigh", "Extra High", "Maximum thinking for the hardest tasks", "", 4),
        )
    return items


def _on_gemini_model_update(self, context):
    if not self.gemini_thinking_level:
        self.gemini_thinking_level = "high"


def init_props():
    bpy.types.Scene.gemini_chat_history = bpy.props.CollectionProperty(
        type=GEMINI_ChatMessage
    )
    bpy.types.Scene.gemini_model = bpy.props.EnumProperty(
        name="Gemini Model",
        description="Select the Gemini model to use",
        items=[
            (
                "gemini-4-carbon",
                "Gemini 4 Carbon",
                "Use Gemini 4 Carbon",
            ),
            (
                "gemini-4-hydrogen",
                "Gemini 4 Hydrogen",
                "Use Gemini 4 Hydrogen",
            ),
            (
                "gemini-3.8-flash",
                "Gemini 3.8 Flash",
                "Use Gemini 3.8 Flash",
            ),
        ],
        default="gemini-4-hydrogen",
        update=_on_gemini_model_update,
    )
    bpy.types.Scene.gemini_chat_input = bpy.props.StringProperty(
        name="Message",
        description="Enter your message",
        default="",
    )
    bpy.types.Scene.gemini_previous_interaction_id = bpy.props.StringProperty(
        name="Previous Interaction ID",
        default="",
    )
    bpy.types.Scene.gemini_thinking_level = bpy.props.EnumProperty(
        name="Thinking Level",
        description="Select the thinking level for Gemini",
        items=get_thinking_level_items,
    )
    bpy.types.Scene.gemini_enable_grounding = bpy.props.BoolProperty(
        name="Enable Grounding",
        description="Allow Gemini to use Google Search grounding for responses",
        default=False,
    )


def clear_props():
    del bpy.types.Scene.gemini_chat_history
    del bpy.types.Scene.gemini_model
    del bpy.types.Scene.gemini_chat_input
    del bpy.types.Scene.gemini_previous_interaction_id
    del bpy.types.Scene.gemini_thinking_level
    del bpy.types.Scene.gemini_enable_grounding


def _build_interaction_input_parts(text, image_b64=None):
    parts = [{"type": "text", "text": text}]
    if image_b64:
        parts.append({"type": "image", "mime_type": "image/png", "data": image_b64})
    return parts


def _build_interaction_generation_config(context):
    generation_config = {}
    model_name = getattr(context.scene, "gemini_model", "") or ""
    if model_name.startswith(THINKING_LEVEL_MODEL_PREFIXES):
        # Empty when the stored level isn't offered by the current model (e.g. "xhigh").
        generation_config["thinking_level"] = (
            context.scene.gemini_thinking_level or "high"
        )
    if model_name in LONG_DECODE_MODELS:
        generation_config["max_output_tokens"] = LONG_DECODE_MAX_OUTPUT_TOKENS

    return generation_config


def _build_interaction_request_data(
    model,
    input_parts,
    system_instruction,
    generation_config=None,
    previous_interaction_id="",
    enable_grounding=False,
):
    data = {
        "model": model,
        "input": input_parts,
        "system_instruction": system_instruction,
    }
    if generation_config:
        data["generation_config"] = generation_config
    if previous_interaction_id:
        data["previous_interaction_id"] = previous_interaction_id
    if enable_grounding:
        data["tools"] = [{"type": "google_search"}]
    if model in LONG_DECODE_MODELS:
        # Long decode only runs in background mode, which requires a stored interaction.
        data["background"] = True
        data["store"] = True
    return data


def _extract_text_blocks(content):
    if not isinstance(content, list):
        return []

    return [
        item["text"]
        for item in content
        if isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
    ]


def _extract_interaction_output_text_from_steps(steps):
    if not isinstance(steps, list):
        raise TypeError("Interaction steps field is not a list.")

    # Long decode appends one model_output step per hop, so join them all.
    output_steps = [
        step
        for step in steps
        if isinstance(step, dict) and step.get("type") == "model_output"
    ]
    if not output_steps:
        raise ValueError("Interaction response is missing a model output step.")

    texts = [
        text
        for step in output_steps
        for text in _extract_text_blocks(step.get("content"))
    ]
    if texts:
        return "".join(texts)
    raise ValueError("Model output step contained no text.")


def _extract_interaction_output_text_from_outputs(outputs):
    if not isinstance(outputs, list):
        raise TypeError("Interaction outputs field is not a list.")

    for output in reversed(outputs):
        if not isinstance(output, dict):
            continue
        if output.get("type") == "text" and isinstance(output.get("text"), str):
            return output["text"]

        texts = _extract_text_blocks(output.get("content"))
        if texts:
            return "".join(texts)

    raise ValueError("Interaction response is missing a text output.")


def _extract_interaction_output_text(response_data):
    status = response_data.get("status")
    if status == "requires_action":
        raise ValueError(
            "Interaction requires tool action, but no tools are configured."
        )
    if status and status != "completed":
        raise ValueError(f"Interaction did not complete: {status}")

    if "outputs" in response_data:
        return _extract_interaction_output_text_from_outputs(
            response_data.get("outputs")
        )
    if isinstance(response_data.get("output_text"), str):
        return response_data["output_text"]
    if "steps" in response_data:
        return _extract_interaction_output_text_from_steps(response_data.get("steps"))

    keys = ", ".join(sorted(response_data.keys()))
    raise ValueError(f"Interaction response has no text output field. Keys: {keys}")


def _extract_code_from_response_text(response_text):
    code_match = re.search(r"```(?:[\w+-]*\n)?(.*?)```", response_text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    return response_text.strip()


def make_gemini_api_request(url, headers, data, cancel):
    """
    Makes a request to the Gemini API with retry logic for handling errors.
    Returns (result, None) on success, or (None, http_status_or_None) on failure.
    Stops between polls and retries once the `cancel` threading.Event is set.
    """
    max_retries = 5
    wait_time = 1
    max_wait_time = 16

    interaction_id = None
    for attempt in range(max_retries):
        try:
            # After a background interaction is created, retries poll it instead of re-creating it.
            if interaction_id:
                response = _http_session.get(
                    f"{url}/{interaction_id}",
                    headers=headers,
                    timeout=API_REQUEST_TIMEOUT,
                )
            else:
                response = _http_session.post(
                    url, headers=headers, json=data, timeout=API_REQUEST_TIMEOUT
                )
            response.raise_for_status()

            response_data = response.json()
            interaction_id = response_data.get("id")
            if not interaction_id:
                raise ValueError("Interaction response is missing an id.")

            while response_data.get("status") in ("in_progress", "queued"):
                if cancel.wait(INTERACTION_POLL_INTERVAL):
                    return None, None
                response = _http_session.get(
                    f"{url}/{interaction_id}",
                    headers=headers,
                    timeout=API_REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                response_data = response.json()

            return {
                "text": _extract_interaction_output_text(response_data),
                "interaction_id": interaction_id,
            }, None

        except requests.exceptions.RequestException as e:
            details = ""
            if (
                hasattr(e, "response")
                and e.response is not None
                and getattr(e.response, "text", None)
            ):
                details = f"\nAPI Error details: {e.response.text}"
            error_msg = f"API request failed (attempt {attempt + 1}/{max_retries}): {e!s}{details}"
            print(error_msg)

            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if (
                status_code is not None
                and 400 <= status_code < 500
                and status_code != 429
            ):
                print("Non-retryable client error. Giving up.")
                return None, status_code

            if attempt < max_retries - 1:
                current_wait = min(wait_time * (2**attempt), max_wait_time)
                print(f"Retrying in {current_wait} seconds...")
                if cancel.wait(current_wait):
                    return None, None
            else:
                print("Maximum retry attempts reached. Giving up.")
                return None, status_code
        except (KeyError, IndexError, TypeError, ValueError) as e:
            print(f"Error parsing API response: {e!s}")
            return None, None


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
        restore_mode = _ensure_object_mode_for_screenshot()
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    for region in area.regions:
                        if region.type == "WINDOW":
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=".png"
                            ) as tmp_file:
                                tmp_path = tmp_file.name
                            loaded_img = None
                            try:
                                # Use a lightweight region-only screenshot to avoid heavy draw-engine paths
                                with context.temp_override(
                                    window=window, area=area, region=region
                                ):
                                    bpy.ops.screen.screenshot(
                                        filepath=tmp_path, full=False
                                    )

                                # Optionally downscale if larger than max_size
                                try:
                                    loaded_img = bpy.data.images.load(tmp_path)
                                    width, height = (
                                        int(loaded_img.size[0]),
                                        int(loaded_img.size[1]),
                                    )
                                    largest = max(width, height)
                                    if max_size and largest > int(max_size):
                                        scale = float(max_size) / float(largest)
                                        new_w = max(1, round(width * scale))
                                        new_h = max(1, round(height * scale))
                                        loaded_img.scale(new_w, new_h)
                                        loaded_img.filepath_raw = tmp_path
                                        loaded_img.file_format = "PNG"
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
                                with contextlib.suppress(OSError):
                                    os.remove(tmp_path)
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
    Scans the current Blender scene and returns a compact context summary.
    This helps the AI understand the current state of the scene.
    """

    def _format_vec(vec):
        try:
            return f"({vec.x:.4f}, {vec.y:.4f}, {vec.z:.4f})"
        except Exception:
            return "unknown"

    def _visible_label(obj):
        try:
            return str(bool(obj.visible_get()))
        except Exception:
            return "unknown"

    def _selected_label(obj):
        try:
            return str(bool(obj.select_get()))
        except Exception:
            return "unknown"

    active_obj = getattr(context.view_layer.objects, "active", None)
    selected_objects = list(getattr(context, "selected_objects", []))
    selected_names = [obj.name for obj in selected_objects]
    selected_set = set(selected_objects)
    # Stable sort puts the active and selected objects first, so the cap keeps them.
    objects = sorted(
        context.scene.objects,
        key=lambda obj: obj != active_obj and obj not in selected_set,
    )

    lines = ["Scene State:"]
    if active_obj is not None:
        active_mode = getattr(active_obj, "mode", "UNKNOWN")
        lines.append(f"- Active Object: `{active_obj.name}` (mode={active_mode})")
    else:
        lines.append("- Active Object: None")
    if selected_names:
        selected_text = ", ".join(f"`{name}`" for name in selected_names)
        lines.append("- Selected Objects: " + selected_text)
    else:
        lines.append("- Selected Objects: None")

    lines.append("")
    lines.append("Scene Objects:")
    if not objects:
        lines.append("- None")
        return "\n".join(lines)

    for obj in objects[:MAX_SUMMARY_OBJECTS]:
        mode = getattr(obj, "mode", "UNKNOWN")
        parts = [
            f"- `{obj.name}`",
            f"type={obj.type}",
            f"visible={_visible_label(obj)}",
            f"selected={_selected_label(obj)}",
            f"active={obj == active_obj}",
            f"mode={mode}",
            f"location={_format_vec(obj.location)}",
        ]
        if obj.type == "MESH":
            parts.extend(
                [
                    f"vertices={len(obj.data.vertices)}",
                    f"faces={len(obj.data.polygons)}",
                ]
            )
        lines.append(", ".join(parts))
    if len(objects) > MAX_SUMMARY_OBJECTS:
        lines.append(
            f"- ...and {len(objects) - MAX_SUMMARY_OBJECTS} more objects not listed"
        )
    return "\n".join(lines)


def _get_nearest_objects_json(context, max_count=3):
    """Return a list of dicts with nearest visible objects to the 3D cursor."""
    try:
        cursor_loc = context.scene.cursor.location
    except Exception:
        return []

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

            min_corner_dist = center_dist
            if getattr(obj, "bound_box", None):
                try:
                    min_corner_dist = min(
                        (
                            obj.matrix_world @ mathutils.Vector(corner) - cursor_loc
                        ).length
                        for corner in obj.bound_box
                    )
                except Exception:
                    pass

            proximity = min(center_dist, min_corner_dist)
            candidates.append((proximity, obj))
        except Exception:
            continue

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0])
    return [
        {"name": obj.name, "type": obj.type, "distance": round(dist, 4)}
        for dist, obj in candidates[:max_count]
    ]


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
                return (
                    active,
                    "active selection",
                    (active.matrix_world.to_translation() - cursor_loc).length,
                )
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


def _build_3d_cursor_context_json(context):
    """Build JSON structure for 3D cursor state."""
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

        cursor_state = {
            "location": {
                "x": round(loc.x, 4),
                "y": round(loc.y, 4),
                "z": round(loc.z, 4),
            },
            "rotation": {
                "euler_radians": {
                    "x": round(rot_euler.x, 5),
                    "y": round(rot_euler.y, 5),
                    "z": round(rot_euler.z, 5),
                },
                "euler_degrees": {
                    "x": round(rot_deg[0], 2),
                    "y": round(rot_deg[1], 2),
                    "z": round(rot_deg[2], 2),
                },
            },
            "orientation_axes": {
                "right_x": [round(right.x, 4), round(right.y, 4), round(right.z, 4)],
                "up_y": [round(up.x, 4), round(up.y, 4), round(up.z, 4)],
                "fwd_z": [
                    round(forward.x, 4),
                    round(forward.y, 4),
                    round(forward.z, 4),
                ],
            },
            "nearest_objects": _get_nearest_objects_json(context),
        }
        return cursor_state
    except Exception:
        return {
            "location": {"x": 0.0000, "y": 0.0000, "z": 0.0000},
            "rotation": {
                "euler_radians": {"x": 0.00000, "y": 0.00000, "z": 0.00000},
                "euler_degrees": {"x": 0.00, "y": 0.00, "z": 0.00},
            },
            "orientation_axes": {
                "right_x": [1.0000, 0.0000, 0.0000],
                "up_y": [0.0000, 1.0000, 0.0000],
                "fwd_z": [0.0000, 0.0000, 1.0000],
            },
            "nearest_objects": [],
        }


def _build_cursor_target_object_json(context):
    """Build JSON structure for cursor target object."""
    try:
        obj, reason, dist = _choose_cursor_target_object(context)
        if obj is None:
            return None

        cursor_ws = context.scene.cursor.location.copy()
        local_cursor = obj.matrix_world.inverted() @ cursor_ws

        return {
            "name": obj.name,
            "type": obj.type,
            "reason": reason,
            "distance": round(dist, 4) if dist is not None else None,
            "cursor_local_space": {
                "x": round(local_cursor.x, 4),
                "y": round(local_cursor.y, 4),
                "z": round(local_cursor.z, 4),
            },
        }
    except Exception:
        return None


def _format_cursor_json_compact(obj):
    """Format JSON in compact/minified format (no indentation, minimal whitespace)."""
    return json.dumps(obj, separators=(",", ":"))


def _build_complete_cursor_json(context):
    """Build complete JSON structure combining cursor_state and cursor_target_object."""
    cursor_state = _build_3d_cursor_context_json(context)
    cursor_target_object = _build_cursor_target_object_json(context)

    return {"cursor_state": cursor_state, "cursor_target_object": cursor_target_object}


def get_detailed_object_data(obj):
    """
    Serializes the geometry of a single Blender object into a JSON format.
    Includes a limit to avoid excessively long outputs for high-poly meshes.
    Returns a JSON string with object name, vertex count, face count, edge count,
    vertices (with normals), edges (only those part of faces), and faces (with normals and UVs).
    """
    if not obj or obj.type != "MESH":
        error_dict = {"error": "No mesh object selected for detailed analysis."}
        return _format_cursor_json_compact(error_dict)

    data = obj.data
    vertex_limit = 500
    face_limit = 1000
    edge_limit = 1500

    vertex_count = len(data.vertices)
    face_count = len(data.polygons)

    # Skip the per-face edge walk on dense meshes; the summary reports all edges instead.
    too_dense = vertex_count > vertex_limit or face_count > face_limit
    if too_dense:
        edge_count = len(data.edges)
    else:
        # edge_keys are already ordered (min, max) tuples.
        edges_in_faces_set = {key for face in data.polygons for key in face.edge_keys}
        edge_count = len(edges_in_faces_set)
        too_dense = edge_count > edge_limit

    if too_dense:
        simplified_dict = {
            "objectName": obj.name,
            "meshData": {
                "vertexCount": vertex_count,
                "faceCount": face_count,
                "edgeCount": edge_count,
                "note": (
                    f"Geometry is too dense to display full details "
                    f"(limit: {vertex_limit} vertices, {face_limit} faces, {edge_limit} edges). "
                    f"Only summary information is included."
                ),
                "simplified": True,
            },
        }
        return _format_cursor_json_compact(simplified_dict)

    # Get UV layer if available
    uv_layer = None
    try:
        if data.uv_layers and len(data.uv_layers) > 0:
            uv_layer = data.uv_layers[0]
    except Exception:
        pass

    # Build vertices as list of dicts (includes normals)
    vertices_list = []
    for idx, v in enumerate(data.vertices):
        pos = [round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4)]
        # Get vertex normal (ensure it's calculated)
        normal = v.normal
        normal_vec = [round(normal.x, 4), round(normal.y, 4), round(normal.z, 4)]
        vertices_list.append({"id": idx, "position": pos, "normal": normal_vec})

    # Collect edges that are part of faces
    edges_list = []
    edge_index_map = {}  # Map (sorted edge_key) -> edge index
    edge_counter = 0

    for edge in data.edges:
        edge_key = edge.key
        if edge_key in edges_in_faces_set and edge_key not in edge_index_map:
            edge_index_map[edge_key] = edge_counter
            edges_list.append({"id": edge_counter, "vertices": list(edge_key)})
            edge_counter += 1

    # Build faces as list of dicts (includes normals and UVs)
    faces_list = []
    for idx, f in enumerate(data.polygons):
        face_verts = list(f.vertices)
        # Get face normal
        face_normal = f.normal
        normal_vec = [
            round(face_normal.x, 4),
            round(face_normal.y, 4),
            round(face_normal.z, 4),
        ]

        # Get UV coordinates per face corner
        uv_coords = []
        if uv_layer:
            try:
                for loop_idx in f.loop_indices:
                    uv = uv_layer.data[loop_idx].uv
                    uv_coords.append([round(uv.x, 4), round(uv.y, 4)])
            except Exception:
                pass

        face_dict = {"id": idx, "vertices": face_verts, "normal": normal_vec}
        if uv_coords:
            face_dict["uvs"] = uv_coords
        faces_list.append(face_dict)

    # Build result dict structure
    result_dict = {
        "objectName": obj.name,
        "meshData": {
            "vertexCount": vertex_count,
            "faceCount": face_count,
            "edgeCount": len(edges_list),
            "vertices": vertices_list,
            "edges": edges_list,
            "faces": faces_list,
        },
    }

    return _format_cursor_json_compact(result_dict)


def _compute_falloff_weight(t, mode="SMOOTH"):
    """
    Compute a [0,1] falloff weight given normalized distance t in [0,1].
    mode="SMOOTH" uses a smoothstep curve; mode="LINEAR" uses linear.
    `mode` must already be uppercase.
    """
    if t <= 0.0:
        return 1.0
    if t >= 1.0:
        return 0.0
    if mode == "LINEAR":
        return 1.0 - t
    # Default SMOOTHSTEP: 1 - (3t^2 - 2t^3)
    return 1.0 - (3.0 * t * t - 2.0 * t * t * t)


def get_vertices_in_radius(obj, center_local, radius):
    """
    Return a list of vertex indices in `obj` whose local-space positions are within
    `radius` of `center_local` (Vector in object local space).
    """
    try:
        mesh = obj.data
        is_edit_mode = False
        try:
            is_edit_mode = getattr(obj, "mode", "OBJECT") == "EDIT"
        except Exception:
            is_edit_mode = False

        # Edit Mode changes live only in the BMesh; otherwise read mesh data without a BMesh copy.
        verts = bmesh.from_edit_mesh(mesh).verts if is_edit_mode else mesh.vertices

        center_vec = (
            center_local
            if isinstance(center_local, mathutils.Vector)
            else mathutils.Vector((center_local.x, center_local.y, center_local.z))
        )
        r2 = radius * radius
        return [v.index for v in verts if (v.co - center_vec).length_squared <= r2]
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
            is_edit_mode = getattr(obj, "mode", "OBJECT") == "EDIT"
        except Exception:
            is_edit_mode = False

        if is_edit_mode:
            bm = bmesh.from_edit_mesh(mesh)
        else:
            bm = bmesh.new()
            bm.from_mesh(mesh)

        bm.verts.ensure_lookup_table()

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
                if cand <= radius and (
                    w.index not in distances or cand < distances[w.index]
                ):
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
        corners = [
            obj.matrix_world @ mathutils.Vector(c)
            for c in getattr(obj, "bound_box", [])
        ]
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
        return BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
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
            if m.type == "SUBSURF":
                mod = m
                break
        if mod is None:
            mod = obj.modifiers.new(name="GeminiDetailSubdivision", type="SUBSURF")
        mod.levels = max(mod.levels, target_levels)
        mod.render_levels = max(mod.render_levels, target_levels)
        return mod
    except Exception:
        return None


def get_local_geometry_patch_text(
    obj, center_world, radius, vertex_limit=2000, face_limit=4000
):
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
            co = data.vertices[idx].co
            out.append(f"  - ({co.x:.4f}, {co.y:.4f}, {co.z:.4f})")
        out.append("Faces (vertex indices within patch):")
        for f in faces:
            out.append(f"  - {f}")
        return "\n".join(out)
    except Exception:
        return "Local geometry patch unavailable due to an error."


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
            is_edit_mode = getattr(obj, "mode", "OBJECT") == "EDIT"
        except Exception:
            is_edit_mode = False

        if is_edit_mode:
            bm = bmesh.from_edit_mesh(mesh)
        else:
            bm = bmesh.new()
            bm.from_mesh(mesh)

        bm.normal_update()
        falloff_mode = str(falloff).upper()

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
                w = _compute_falloff_weight(d / r, falloff_mode)
                n = v.normal.normalized()
                v.co = v.co - n * (strength * w)

        # Ensure Vector type
        center_vec = (
            center_local
            if isinstance(center_local, mathutils.Vector)
            else mathutils.Vector(center_local)
        )
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


# Exposed in the exec namespace and advertised to the model by signature.
CODE_HELPERS = {
    fn.__name__: fn
    for fn in (
        apply_radial_shrink_fatten,
        get_vertices_in_radius,
        get_vertices_in_geodesic_radius,
        raycast_surface,
        project_point_to_surface_near,
        ensure_subsurf_for_local_detail,
        get_local_geometry_patch_text,
    )
}
_HELPERS_TEXT = "**[AVAILABLE HELPERS]:**\n" + "".join(
    f"- {name}{inspect.signature(fn)}\n" for name, fn in CODE_HELPERS.items()
)


def _cursor_context_text(context):
    """3D cursor data, helper signatures, and the local geometry patch near the cursor."""
    cursor_json_str = _format_cursor_json_compact(_build_complete_cursor_json(context))
    text = (
        "**3D Cursor Data:**\n```json\n"
        + cursor_json_str
        + "\n```\n\n"
        + _HELPERS_TEXT
        + "\n"
    )
    try:
        target_obj, _reason, _dist = _choose_cursor_target_object(context)
        if target_obj is not None:
            patch_text = get_local_geometry_patch_text(
                target_obj, context.scene.cursor.location.copy(), LOCAL_PATCH_RADIUS
            )
            text += "**[LOCAL GEOMETRY PATCH]:**\n" + patch_text + "\n\n"
    except Exception:
        pass
    return text


def _prepare_request(
    context,
    prompt_text,
    system_prompt,
    include_viewport_screenshot,
    previous_interaction_id,
):
    """
    Gather everything that needs bpy on the main thread, and return send(cancel),
    which does only network I/O so it can run on a worker thread.
    send returns {"code", "interaction_id"} or None.
    """
    api_key = get_api_key(context, __package__) or os.getenv("GEMINI_API_KEY")
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    image_b64 = (
        capture_viewport_screenshot_base64(context)
        if include_viewport_screenshot
        else None
    )
    request_args = (
        context.scene.gemini_model,
        _build_interaction_input_parts(prompt_text, image_b64),
        system_prompt,
    )
    request_kwargs = {
        "generation_config": _build_interaction_generation_config(context),
        "enable_grounding": context.scene.gemini_enable_grounding,
    }

    data = _build_interaction_request_data(
        *request_args,
        previous_interaction_id=previous_interaction_id,
        **request_kwargs,
    )
    fresh_data = _build_interaction_request_data(*request_args, **request_kwargs)

    def send(cancel):
        response, status = make_gemini_api_request(
            GEMINI_INTERACTIONS_URL, headers, data, cancel
        )
        if (
            not response
            and previous_interaction_id
            and status in STALE_INTERACTION_STATUSES
        ):
            print(
                f"previous_interaction_id '{previous_interaction_id}' was rejected "
                f"(HTTP {status}). Retrying as a fresh interaction..."
            )
            response, _status = make_gemini_api_request(
                GEMINI_INTERACTIONS_URL, headers, fresh_data, cancel
            )

        if response:
            return {
                "code": _extract_code_from_response_text(response["text"]),
                "interaction_id": response["interaction_id"],
            }
        return None

    return send


def prepare_generation_request(
    prompt,
    context,
    generation_system_prompt,
    detailed_geometry=None,
    use_3d_cursor=False,
    include_viewport_screenshot=False,
):
    full_prompt = f"(Blender {bpy.app.version_string})\n\n"
    if detailed_geometry:
        full_prompt += (
            "**Detailed Geometry of Active Object:**\n```json\n"
            + detailed_geometry
            + "\n```\n\n"
        )
    if use_3d_cursor:
        full_prompt += _cursor_context_text(context)
    if include_viewport_screenshot:
        full_prompt += (
            "**Viewport Screenshot:** Attached below. Use it to understand view "
            "orientation and selection state.\n\n"
        )
    full_prompt += (
        "**Scene Summary:**\n"
        + get_scene_objects_as_text(context)
        + "\n\nUser Request: "
        + prompt
    )

    return _prepare_request(
        context,
        full_prompt,
        generation_system_prompt,
        include_viewport_screenshot,
        context.scene.gemini_previous_interaction_id,
    )


def prepare_fix_request(
    original_code,
    error_message,
    context,
    repair_system_prompt,
    detailed_geometry=None,
    use_3d_cursor=False,
    include_viewport_screenshot=False,
    previous_interaction_id="",
):
    """Prepare a request for fixed Blender code based on the error message."""
    detailed_geo_block = ""
    if detailed_geometry:
        detailed_geo_block = f"""
**[DETAILED GEOMETRY OF ACTIVE OBJECT]:**
```json
{detailed_geometry}
```
"""

    cursor_block = _cursor_context_text(context) if use_3d_cursor else ""

    screenshot_block = ""
    if include_viewport_screenshot:
        screenshot_block = """
**[VIEWPORT SCREENSHOT]:**
Attached below. Use it to infer view orientation and selection state, and to understand the scene context.
"""

    fix_prompt = f"""### Repair Context
Environment: Blender {bpy.app.version_string}
Use the failed script, traceback, scene summary, and optional context below to produce the complete corrected script.
{detailed_geo_block}
{cursor_block}
{screenshot_block}
**[SCENE SUMMARY]:**
```
{get_scene_objects_as_text(context)}
```

**[FAULTY SCRIPT]:**
```python
{original_code}
```

**[ERROR TRACEBACK]:**
```
{error_message}
```"""

    return _prepare_request(
        context,
        fix_prompt,
        repair_system_prompt,
        include_viewport_screenshot,
        previous_interaction_id or context.scene.gemini_previous_interaction_id,
    )


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
