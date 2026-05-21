import os
import sys

import bpy
import bpy.props

libs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "lib")
if libs_path not in sys.path:
    sys.path.append(libs_path)

from .utilities import (  # noqa: E402
    apply_radial_shrink_fatten,
    clear_props,
    ensure_subsurf_for_local_detail,
    fix_blender_code,
    generate_blender_code,
    get_api_key,
    get_detailed_object_data,
    get_local_geometry_patch_text,
    get_vertices_in_geodesic_radius,
    get_vertices_in_radius,
    init_props,
    project_point_to_surface_near,
    raycast_surface,
    split_area_to_text_editor,
)

bl_info = {
    "name": "Gemini Blender Assistant",
    "blender": (3, 1, 0),
    "category": "Object",
    "author": "grinnch (@meangrinch)",
    "version": (1, 7, 0),
    "location": "3D View > UI > Gemini Blender Assistant",
    "description": "Generate Blender Python code using Google's Gemini.",
    "wiki_url": "",
    "tracker_url": "",
}


def _ensure_object_mode():
    """Best-effort switch to Object Mode to avoid dangling BMesh contexts."""
    try:
        # bpy.context.mode is e.g. 'OBJECT', 'EDIT_MESH', etc.
        if getattr(bpy.context, "mode", "OBJECT") != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass
    except Exception:
        pass


def _make_namespace(context):
    """Create a fresh namespace for script execution to avoid stale references."""
    return {
        "bpy": bpy,
        "context": context,
        "__name__": "__main__",
        # Expose helpers so generated code can call them directly
        "apply_radial_shrink_fatten": apply_radial_shrink_fatten,
        "get_vertices_in_radius": get_vertices_in_radius,
        "get_vertices_in_geodesic_radius": get_vertices_in_geodesic_radius,
        "raycast_surface": raycast_surface,
        "project_point_to_surface_near": project_point_to_surface_near,
        "ensure_subsurf_for_local_detail": ensure_subsurf_for_local_detail,
        "get_local_geometry_patch_text": get_local_geometry_patch_text,
    }


generation_system_prompt = """### Persona
You are `BlenderGemini`, a specialized AI assistant integrated directly into Blender's scripting environment. Your purpose is to translate user requests into clean, efficient, robust `bpy` Python scripts that operate on the current Blender scene.

### Context
- You operate in a persistent, session-based Blender environment.
- You will be given a compact scene summary with active object, selected objects, visibility, object type, object mode, location, and mesh counts where available.
- Optionally, you may be given detailed geometry for the currently selected object. Use it for high-precision mesh edits.
- Optionally, a screenshot of the 3D Viewport may be attached. Use it for selection state, viewport orientation, and visible context, but do not rely on it for exact coordinates.
- The current scene is the cumulative result of previously executed scripts in this conversation.
- You have access to recent chat history, not necessarily the full project history.

### Output Contract
1. Your response must be a single executable Python script enclosed in one markdown `python` code block.
2. Do not include explanations, apologies, summaries, or conversation outside the code block.
3. The script must be self-contained and runnable, starting with `import bpy`.
4. Reason internally. Do not reveal analysis or chain-of-thought in comments or text.

### Local Execution Boundary
- Generated code may manipulate the Blender scene through `bpy`, standard Blender modules such as `mathutils` and `bmesh`, and helpers explicitly listed in the request context.
- Do not access the filesystem, network, subprocesses, environment variables, API keys, add-on preferences, or external services unless the user explicitly asks for that exact capability.
- Do not install packages, launch applications, run shell commands, or inspect local files.
- If a request is outside Blender scene automation or asks for unsafe local access, output a safe no-op Blender script that prints a concise refusal and does not perform the requested access.
- If Google Search grounding is enabled, it may inform Blender/API facts or user-requested factual support, but the script you produce must not perform web access.

### Scene Interaction Strategy
First analyze the scene summary and recent chat history, then choose the appropriate action.
- Modification: If the request refines existing objects, reference those objects and modify them directly. Do not recreate them.
- Replacement: If the request replaces objects from a previous step, explicitly delete the old objects before creating the replacement.
- Addition: If the request adds new objects, create them without altering existing objects unless specified.

### Selection and Context Rules
- At the start of the script, capture `active_object = bpy.context.view_layer.objects.active` and `selected_objects = list(bpy.context.selected_objects)` before changing mode or selection.
- If the user says "selected", "active", "this object", or does not name an object for a modification, prefer the captured active object, then captured selected objects, then the best matching object from the scene summary.
- If you must use operators, switch to Object Mode if needed, then explicitly set the active object and selection state before the operator call.
- Deselect objects only after preserving references needed by the request.

### API and Workflow Principles
- Prefer the direct Data API (`bpy.data`) for property changes. Use `bpy.ops` mainly for object creation, mode switching, and operations with no direct data equivalent.
- Prefer non-destructive methods like modifiers. Avoid Edit Mode unless specific mesh components must be edited.
- Assign descriptive names to new objects and materials.
- For Principled BSDF inputs, check that each input exists before setting version-sensitive fields such as `IOR`, `Specular IOR Level`, `Metallic`, or `Roughness`.
- When preserving a child's world transform during parenting, assign `child.parent = parent`, then set `child.matrix_parent_inverse = parent.matrix_world.inverted()`.
- Apply visual polishing like `shade_smooth` only when adding geometric detail or when the user's request implies a final visual touch-up.

### Targeting with 3D Cursor
- If "Target with 3D Cursor" is enabled, the request includes the 3D cursor's world-space location and orientation.
- Treat the cursor as a read-only target indicator; do not move it unless explicitly requested.
- When acting on an existing object, transform the world-space cursor location into the object's local space with `center_local = obj.matrix_world.inverted() @ bpy.context.scene.cursor.location`.
- For orientation-aware tasks, use `cursor.matrix.to_3x3()` and derive axes such as `right = basis @ Vector((1, 0, 0))`, `up = basis @ Vector((0, 1, 0))`, and `forward = basis @ Vector((0, 0, 1))`.
- If no object is named near the cursor, prefer the captured selected object nearest to the cursor. If nothing is selected, prefer the nearest visible mesh object.
- When adding geometry and no location is specified, place or center it at the cursor and align to cursor orientation when appropriate.

### Localized Organic Edits Near the Cursor
- If the user requests organic shaping near the cursor, use cursor-local coordinates and these Scene controls:
    - `context.scene.gemini_edit_radius` in meters
    - `context.scene.gemini_edit_strength` where positive shrinks and negative fattens
    - `context.scene.gemini_falloff` as `"SMOOTH"` or `"LINEAR"`
    - `context.scene.gemini_mirror_edit` with `context.scene.gemini_mirror_axis`
- Prefer the listed helper functions for precise localized edits. They are available directly in the execution namespace; do not import them from the add-on package.

### BMesh Workflows for Edit Mode Operations
When modifying specific vertices, edges, or faces, use one of these workflows.

Pattern A: Pure BMesh operation
1. Enter Edit Mode.
2. Create `bm = bmesh.from_edit_mesh(obj.data)`.
3. Find and select target geometry inside `bm`.
4. Use `bmesh.ops` for the operation.
5. Commit with `bmesh.update_edit_mesh(obj.data)`.
6. Free with `bm.free()`.
7. Return to Object Mode.

Pattern B: BMesh selection, `bpy.ops` operation
1. Enter Edit Mode.
2. Create `bm = bmesh.from_edit_mesh(obj.data)`.
3. Find and select geometry inside `bm`.
4. Commit and release before calling an operator:
```python
bmesh.update_edit_mesh(obj.data)
bm.free()
```
5. Call the required `bpy.ops` operator.
6. Return to Object Mode.

Do not keep an active `bmesh` instance when calling a `bpy.ops` operator.

### Example

<user_request>
Create a red metallic sphere. Then add a smaller green cube with slightly rounded edges and parent it to the sphere, positioned 2 units directly above the sphere's center.
</user_request>
<assistant_response>
```python
import bpy

active_object = bpy.context.view_layer.objects.active
selected_objects = list(bpy.context.selected_objects)

if active_object and active_object.mode != "OBJECT":
    bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.object.select_all(action="DESELECT")

def set_node_input(node, name, value):
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value

sphere_obj = bpy.data.objects.get("MetallicRedSphere")
if sphere_obj is None:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))
    sphere_obj = bpy.context.object
    sphere_obj.name = "MetallicRedSphere"

mat_sphere = bpy.data.materials.new(name="RedMetallicMaterial")
mat_sphere.use_nodes = True
bsdf_sphere = mat_sphere.node_tree.nodes.get("Principled BSDF")
if bsdf_sphere:
    set_node_input(bsdf_sphere, "Base Color", (0.8, 0.0, 0.0, 1.0))
    set_node_input(bsdf_sphere, "Metallic", 1.0)
    set_node_input(bsdf_sphere, "Roughness", 0.2)
if sphere_obj.data.materials:
    sphere_obj.data.materials[0] = mat_sphere
else:
    sphere_obj.data.materials.append(mat_sphere)

cube_location = (
    sphere_obj.location.x,
    sphere_obj.location.y,
    sphere_obj.location.z + 2.0,
)
bpy.ops.mesh.primitive_cube_add(size=0.5, location=cube_location)
cube_obj = bpy.context.object
cube_obj.name = "GreenBeveledChildCube"

mat_cube = bpy.data.materials.new(name="GreenPlasticMaterial")
mat_cube.use_nodes = True
bsdf_cube = mat_cube.node_tree.nodes.get("Principled BSDF")
if bsdf_cube:
    set_node_input(bsdf_cube, "Base Color", (0.0, 0.8, 0.0, 1.0))
    set_node_input(bsdf_cube, "IOR", 1.45)
    set_node_input(bsdf_cube, "Specular IOR Level", 0.5)
    set_node_input(bsdf_cube, "Roughness", 0.5)
cube_obj.data.materials.append(mat_cube)

bevel_mod = cube_obj.modifiers.new(name="RoundedEdges", type="BEVEL")
bevel_mod.width = 0.05
bevel_mod.segments = 3

bpy.context.view_layer.objects.active = cube_obj
cube_obj.select_set(True)
bpy.ops.object.shade_smooth()

cube_obj.parent = sphere_obj
cube_obj.matrix_parent_inverse = sphere_obj.matrix_world.inverted()

bpy.ops.object.select_all(action="DESELECT")
```
</assistant_response>"""  # noqa


repair_system_prompt = """### Persona
You are `BlenderGemini Repair`, a focused `bpy` debugging specialist integrated into Blender's scripting environment. Your purpose is to correct a failed Blender Python script while preserving the user's original scene-editing intent.

### Output Contract
1. Your response must be a single complete corrected Python script enclosed in one markdown `python` code block.
2. Do not include explanations, apologies, summaries, diffs, root-cause text, or conversation outside the code block.
3. The script must be executable as-is and start with `import bpy`.
4. Analyze the traceback and root cause internally, then output only the corrected script.

### Repair Rules
- Make the smallest valid change that fixes the failure.
- Preserve the original script's intended Blender scene outcome.
- Do not remove functionality just to bypass the error.
- Keep the same scene interaction strategy unless the traceback proves it is invalid.
- Prefer `bpy.data` for property changes and `bpy.ops` only where operator context is required.
- Check that operator parameters and node inputs exist before using version-sensitive API fields.
- If selection or active-object context matters, capture it before changing mode or selection.
- When preserving a child's world transform during parenting, assign `child.parent = parent`, then set `child.matrix_parent_inverse = parent.matrix_world.inverted()`.

### Local Execution Boundary
- Corrected code may manipulate the Blender scene through `bpy`, standard Blender modules such as `mathutils` and `bmesh`, and helpers explicitly listed in the request context.
- Do not access the filesystem, network, subprocesses, environment variables, API keys, add-on preferences, or external services unless the original user request explicitly asked for that exact capability.
- Do not install packages, launch applications, run shell commands, or inspect local files.
- If the original script attempted unsafe local access unrelated to Blender scene automation, replace that behavior with a safe no-op Blender script that prints a concise refusal.
- If Google Search grounding is enabled, it may inform Blender/API facts, but the corrected script must not perform web access.
"""  # noqa


class GEMINI_OT_DeleteMessage(bpy.types.Operator):
    bl_idname = "gemini.delete_message"
    bl_label = "Delete Message"
    bl_options = {"REGISTER", "UNDO"}

    message_index: bpy.props.IntProperty()

    def execute(self, context):
        context.scene.gemini_chat_history.remove(self.message_index)
        return {"FINISHED"}


class GEMINI_OT_ShowCode(bpy.types.Operator):
    bl_idname = "gemini.show_code"
    bl_label = "Show Code"
    bl_options = {"REGISTER", "UNDO"}

    code: bpy.props.StringProperty(
        name="Code",
        description="The generated code",
        default="",
    )

    def execute(self, context):
        text_name = "Gemini_Generated_Code.py"
        text = bpy.data.texts.get(text_name)
        if text is None:
            text = bpy.data.texts.new(text_name)

        text.clear()
        text.write(self.code)

        text_editor_area = None
        for area in context.screen.areas:
            if area.type == "TEXT_EDITOR":
                text_editor_area = area
                break

        if text_editor_area is None:
            text_editor_area = split_area_to_text_editor(context)

        text_editor_area.spaces.active.text = text

        return {"FINISHED"}


class GEMINI_OT_CopyGeometry(bpy.types.Operator):
    bl_idname = "gemini.copy_geometry"
    bl_label = "Copy Geometry"
    bl_description = "Copy detailed geometry data of selected object to clipboard"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (
            context.active_object is not None and context.active_object.type == "MESH"
        )

    def execute(self, context):
        active_obj = context.active_object
        if not active_obj:
            self.report({"ERROR"}, "No object selected")
            return {"CANCELLED"}

        if active_obj.type != "MESH":
            self.report({"ERROR"}, "Selected object is not a mesh")
            return {"CANCELLED"}

        try:
            geometry_text = get_detailed_object_data(active_obj)
            if geometry_text:
                context.window_manager.clipboard = geometry_text
                self.report(
                    {"INFO"},
                    f"Geometry data copied to clipboard ({len(geometry_text)} characters)",
                )
                return {"FINISHED"}
            else:
                self.report({"ERROR"}, "No geometry data available")
                return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, f"Failed to copy geometry: {str(e)}")
            return {"CANCELLED"}


class GEMINI_OT_CopyCursor(bpy.types.Operator):
    bl_idname = "gemini.copy_cursor"
    bl_label = "Copy Cursor"
    bl_description = "Copy 3D cursor location and orientation data to clipboard"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            from .utilities import (
                _build_complete_cursor_json,
                _format_cursor_json_compact,
            )

            cursor_json = _build_complete_cursor_json(context)
            cursor_json_str = _format_cursor_json_compact(cursor_json)

            if cursor_json_str:
                context.window_manager.clipboard = cursor_json_str
                self.report(
                    {"INFO"},
                    f"Cursor data copied to clipboard ({len(cursor_json_str)} characters)",
                )
                return {"FINISHED"}
            else:
                self.report({"ERROR"}, "No cursor data available")
                return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, f"Failed to copy cursor data: {str(e)}")
            return {"CANCELLED"}


class GEMINI_PT_Panel(bpy.types.Panel):
    bl_label = "Gemini Blender Assistant"
    bl_idname = "GEMINI_PT_Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Gemini Assistant"

    def draw(self, context):
        layout = self.layout
        column = layout.column(align=True)

        column.label(text="Chat history:")
        box = column.box()
        for index, message in enumerate(context.scene.gemini_chat_history):
            if message.type == "assistant":
                row = box.row()
                row.label(text="Assistant: ")
                show_code_op = row.operator("gemini.show_code", text="Show Code")
                show_code_op.code = message.content
                delete_message_op = row.operator(
                    "gemini.delete_message", text="", icon="TRASH", emboss=False
                )
                delete_message_op.message_index = index
            else:
                row = box.row()
                row.label(text=f"User: {message.content}")
                delete_message_op = row.operator(
                    "gemini.delete_message", text="", icon="TRASH", emboss=False
                )
                delete_message_op.message_index = index

        column.separator()

        column.label(text="Gemini Model:")
        column.prop(context.scene, "gemini_model", text="")

        model_name = context.scene.gemini_model
        if model_name.startswith("gemini-3"):
            column.prop(context.scene, "gemini_thinking_level")
        elif "gemini-2.5-flash" in model_name:
            column.prop(context.scene, "gemini_enable_thinking")

        column.label(text="Enter your message:")
        column.prop(context.scene, "gemini_chat_input", text="")
        button_label = (
            "Please wait...(this might take some time)"
            if context.scene.gemini_button_pressed
            else "Execute"
        )
        # Minimal toggles retained per user request
        row = column.row(align=True)
        row.prop(
            context.scene, "gemini_include_geometry", text="Include Selected Geometry"
        )
        row.operator("gemini.copy_geometry", text="", icon="COPY_ID", emboss=False)
        row = column.row(align=True)
        row.prop(context.scene, "gemini_use_3d_cursor", text="Target with 3D Cursor")
        row.operator("gemini.copy_cursor", text="", icon="COPY_ID", emboss=False)
        if context.scene.gemini_use_3d_cursor:
            cursor = context.scene.cursor
            box = column.box()
            box.label(
                text=f"Location: ({cursor.location.x:.4f}, {cursor.location.y:.4f}, {cursor.location.z:.4f})"
            )
        row = column.row(align=True)
        row.prop(
            context.scene,
            "gemini_include_viewport_screenshot",
            text="Attach Viewport Screenshot",
        )
        row = column.row(align=True)
        row.prop(context.scene, "gemini_enable_grounding", text="Enable Grounding")
        row = column.row(align=True)
        row.operator("gemini.send_message", text=button_label)
        row.operator("gemini.clear_chat", text="Clear Chat")

        column.separator()


class GEMINI_OT_ClearChat(bpy.types.Operator):
    bl_idname = "gemini.clear_chat"
    bl_label = "Clear Chat"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        context.scene.gemini_chat_history.clear()
        return {"FINISHED"}


class GEMINI_OT_Execute(bpy.types.Operator):
    bl_idname = "gemini.send_message"
    bl_label = "Send Message"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        api_key = get_api_key(context, __name__)
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            self.report(
                {"ERROR"},
                "No API key detected. Please set your Gemini API key in the addon preferences.",
            )
            return {"CANCELLED"}

        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        max_fix_attempts = addon_prefs.max_fix_attempts

        context.scene.gemini_button_pressed = True
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)

        # Respect user toggles for geometry inclusion and 3D cursor targeting
        detailed_geometry = None
        if context.scene.gemini_include_geometry:
            active_obj = context.active_object
            if active_obj:
                detailed_geometry = get_detailed_object_data(active_obj)
        use_3d_cursor = context.scene.gemini_use_3d_cursor

        blender_code = generate_blender_code(
            context.scene.gemini_chat_input,
            context.scene.gemini_chat_history,
            context,
            generation_system_prompt,
            detailed_geometry=detailed_geometry,
            use_3d_cursor=use_3d_cursor,
            include_viewport_screenshot=context.scene.gemini_include_viewport_screenshot,
        )

        message = context.scene.gemini_chat_history.add()
        message.type = "user"
        message.content = context.scene.gemini_chat_input

        context.scene.gemini_chat_input = ""

        if blender_code:
            objects_before = set(bpy.data.objects)
            materials_before = set(bpy.data.materials)

            history_index = len(context.scene.gemini_chat_history)
            message = context.scene.gemini_chat_history.add()
            message.type = "assistant"
            message.content = blender_code

            # Ensure a neutral mode before executing arbitrary code
            _ensure_object_mode()
            namespace = _make_namespace(context)

            try:
                exec(blender_code, namespace)
            except Exception as e:
                if max_fix_attempts <= 0:
                    self.report(
                        {"ERROR"},
                        f"Error executing code and fixes are disabled: {str(e)}",
                    )
                    context.scene.gemini_button_pressed = False
                    return {"CANCELLED"}

                error_message = f"Error executing generated code: {str(e)}"
                self.report(
                    {"WARNING"},
                    f"Original code had an error. Attempting to fix (1/{max_fix_attempts})...",
                )

                context.scene.gemini_chat_history.remove(history_index)

                objects_after = set(bpy.data.objects)
                materials_after = set(bpy.data.materials)

                # Ensure we're not in Edit Mode before cleanup
                _ensure_object_mode()
                for obj in objects_after - objects_before:
                    bpy.data.objects.remove(obj, do_unlink=True)

                for mat in materials_after - materials_before:
                    bpy.data.materials.remove(mat)

                current_code = blender_code
                current_error = error_message
                current_detailed_geometry = detailed_geometry

                for attempt in range(1, max_fix_attempts + 1):
                    fixed_code = fix_blender_code(
                        current_code,
                        current_error,
                        context,
                        repair_system_prompt,
                        detailed_geometry=current_detailed_geometry,
                        use_3d_cursor=use_3d_cursor,
                        include_viewport_screenshot=context.scene.gemini_include_viewport_screenshot,
                    )

                    if not fixed_code:
                        self.report(
                            {"ERROR"},
                            f"Could not fix the code on attempt {attempt}: {current_error}",
                        )
                        context.scene.gemini_button_pressed = False
                        return {"CANCELLED"}

                    fix_history_index = len(context.scene.gemini_chat_history)
                    message = context.scene.gemini_chat_history.add()
                    message.type = "assistant"
                    message.content = fixed_code

                    try:
                        _ensure_object_mode()
                        namespace = _make_namespace(context)
                        exec(fixed_code, namespace)
                        self.report(
                            {"INFO"},
                            f"Code fixed and executed successfully on attempt {attempt}!",
                        )
                        break
                    except Exception as e2:
                        current_error = f"Error executing fixed code: {str(e2)}"

                        if attempt < max_fix_attempts:
                            self.report(
                                {"WARNING"},
                                (
                                    f"Fix attempt {attempt} had an error. Attempting to fix again "
                                    f"({attempt + 1}/{max_fix_attempts})..."
                                ),
                            )

                            context.scene.gemini_chat_history.remove(fix_history_index)

                            objects_after_fix = set(bpy.data.objects)
                            materials_after_fix = set(bpy.data.materials)

                            _ensure_object_mode()
                            for obj in objects_after_fix - objects_before:
                                bpy.data.objects.remove(obj, do_unlink=True)

                            for mat in materials_after_fix - materials_before:
                                bpy.data.materials.remove(mat)

                            current_code = fixed_code
                        else:
                            self.report(
                                {"ERROR"},
                                f"Error executing code after {max_fix_attempts} fix attempts: {e2}",
                            )
                            context.scene.gemini_button_pressed = False
                            return {"CANCELLED"}
        else:
            self.report(
                {"ERROR"},
                "Failed to generate code from Gemini API. Please check the console for details.",
            )
            context.scene.gemini_button_pressed = False
            return {"CANCELLED"}

        # Always return to Object Mode at the end to keep viewport/draw stable
        _ensure_object_mode()
        context.scene.gemini_button_pressed = False
        return {"FINISHED"}


def menu_func(self, context):
    self.layout.operator(GEMINI_OT_Execute.bl_idname)


class GEMINI_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    api_key: bpy.props.StringProperty(
        name="API Key",
        description="Enter your Google Gemini API Key",
        default="",
        subtype="PASSWORD",
    )

    enable_custom_sampling_parameters: bpy.props.BoolProperty(
        name="Enable custom sampling parameters",
        description="Send temperature, Top P, and Top K with Gemini requests",
        default=False,
    )

    temperature: bpy.props.FloatProperty(
        name="Temperature",
        description="Controls randomness: Lower values are more deterministic, higher values more creative",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=2,
        step=10,
    )

    top_p: bpy.props.FloatProperty(
        name="Top P",
        description="Controls diversity of output via nucleus sampling",
        default=0.95,
        min=0.0,
        max=1.0,
        precision=2,
        step=5,
    )

    top_k: bpy.props.IntProperty(
        name="Top K",
        description="Limits token selection to the K most likely tokens",
        default=64,
        min=0,
        max=64,
    )

    max_fix_attempts: bpy.props.IntProperty(
        name="Max Fix Attempts",
        description="Maximum number of times to attempt fixing code errors (0 = don't attempt fixes)",
        default=1,
        min=0,
        max=5,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "api_key")

        layout.separator()

        layout.label(text="Generation Parameters:")
        layout.prop(self, "enable_custom_sampling_parameters")
        sampling_box = layout.box()
        sampling_box.enabled = self.enable_custom_sampling_parameters
        sampling_box.prop(self, "temperature")
        sampling_box.prop(self, "top_p")
        sampling_box.prop(self, "top_k")

        layout.separator()

        layout.label(text="Error Handling:")
        layout.prop(self, "max_fix_attempts")


classes = [
    GEMINI_AddonPreferences,
    GEMINI_OT_DeleteMessage,
    GEMINI_OT_Execute,
    GEMINI_PT_Panel,
    GEMINI_OT_ClearChat,
    GEMINI_OT_ShowCode,
    GEMINI_OT_CopyGeometry,
    GEMINI_OT_CopyCursor,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.gemini_include_geometry = bpy.props.BoolProperty(
        name="Include Geometry",
        description="Include detailed geometry of the selected object in the request",
        default=False,
    )

    bpy.types.Scene.gemini_use_3d_cursor = bpy.props.BoolProperty(
        name="Target with 3D Cursor",
        description="Use the 3D cursor's location and orientation as a target for operations",
        default=False,
    )

    bpy.types.Scene.gemini_include_viewport_screenshot = bpy.props.BoolProperty(
        name="Attach Viewport Screenshot",
        description="Send a screenshot of the 3D Viewport with your request",
        default=False,
    )

    bpy.types.VIEW3D_MT_mesh_add.append(menu_func)
    init_props()


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.gemini_include_geometry
    del bpy.types.Scene.gemini_use_3d_cursor
    del bpy.types.Scene.gemini_include_viewport_screenshot
    bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)
    clear_props()


if __name__ == "__main__":
    register()
