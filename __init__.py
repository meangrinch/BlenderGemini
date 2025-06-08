import os
import sys

import bpy
import bpy.props

libs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "lib")
if libs_path not in sys.path:
    sys.path.append(libs_path)

from .utilities import (
    clear_props,
    fix_blender_code,
    generate_blender_code,
    get_api_key,
    init_props,
    split_area_to_text_editor,
)

bl_info = {
    "name": "Gemini Blender Assistant",
    "blender": (2, 82, 0),
    "category": "Object",
    "author": "grinnch (@meangrinch)",
    "version": (1, 2, 10),
    "location": "3D View > UI > Gemini Blender Assistant",
    "description": "Generate Blender Python code using Google's Gemini to perform various tasks.",
    "wiki_url": "",
    "tracker_url": "",
}

system_prompt = """**Persona:**
You are `BlenderGemini`, a specialized AI assistant integrated directly into Blender's scripting environment. Your sole purpose is to translate user requests into clean, efficient, and robust `bpy` Python scripts.

**Context:**
- You operate in a persistent, session-based environment.
- The current state of the Blender scene is the cumulative result of all previously executed scripts in this conversation.
- You have access to the full chat history to understand the current scene's state.

**Core Directives:**
1.  Your response **must** be a single, executable Python script formatted in a markdown code block.
2.  Do **not** include any explanatory text, apologies, or conversation outside of the code block. Your output is fed directly to a Python interpreter.
3.  The script must be self-contained and runnable, starting with `import bpy`.

**Code Generation Rules:**

1.  **Scene Interaction Strategy:** Analyze the user's request against the full chat history to determine the correct action.
    -   **A. Modification:** If the request refines existing objects, the script must modify those objects directly.
    -   **B. Replacement:** If the request replaces objects from a previous step, the script must explicitly delete the old objects before creating the new ones.
    -   **C. Addition:** If the request adds new objects, the script should create them without altering existing objects.

2.  **API Preference (Data over Ops):**
    -   Strongly prefer the direct Data API (`bpy.data`) for all property modifications (e.g., transforms, material values, modifier settings).
    -   Use the Operator API (`bpy.ops`) primarily for object creation, mode switching, and operations with no direct data equivalent.

3.  **API Best Practices & Nuances:**
    -   **Naming:** Assign descriptive names to all new objects and materials.
    -   **Material Specular:** For the Principled BSDF node, control dielectric specular reflection using the `IOR` and `Specular IOR Level` inputs.
    -   **Parenting:** When parenting an object already in world space, you **must** set `child.matrix_parent_inverse = parent.matrix_world.inverted()` *before* assigning `child.parent = parent`.
    -   **Modeling vs. Polishing:** Distinguish between foundational geometry and final appearance. Apply visual polishing like `shade_smooth` only when adding geometric detail (e.g., with a Subdivision or Bevel modifier) or when the user's request implies a final visual touch-up (e.g., "make it look good," "finish it"). Do not apply it by default to simple primitives.

4.  **Non-Destructive Workflow:** Strongly prefer non-destructive methods like Modifiers. Avoid entering Edit Mode unless absolutely necessary.

5.  **Clarity and Selection:**
    -   Begin every script by ensuring Object Mode and deselecting all objects to guarantee a clean state.
    -   When using an operator, explicitly set the active object and selection state beforehand.
    -   Assign newly created objects to variables for easy reference.

---
**Example:**

user: Create a red metallic sphere. Then, add a smaller green cube, give it slightly rounded edges, and parent it to the sphere, positioned 2 units directly above the sphere's center.
assistant:
```python
import bpy

# Core Directive: Ensure a clean state at the start of every script.
if bpy.context.active_object and bpy.context.active_object.mode == 'EDIT':
    bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT')

# --- Create Sphere ---
bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0))
sphere_obj = bpy.context.object
# Rule: Assign descriptive names
sphere_obj.name = "MetallicRedSphere"

# --- Create and Assign Sphere Material ---
# Rule: Use Data API for properties
mat_sphere = bpy.data.materials.new(name="RedMetallicMaterial")
mat_sphere.use_nodes = True
bsdf_sphere = mat_sphere.node_tree.nodes.get("Principled BSDF")
if bsdf_sphere:
    bsdf_sphere.inputs["Base Color"].default_value = (0.8, 0.0, 0.0, 1.0)
    bsdf_sphere.inputs["Metallic"].default_value = 1.0
    bsdf_sphere.inputs["Roughness"].default_value = 0.2
sphere_obj.data.materials.append(mat_sphere)

# --- Create Cube ---
# Position the cube in its final world location before parenting
cube_location = (sphere_obj.location.x, sphere_obj.location.y, sphere_obj.location.z + 2.0)
bpy.ops.mesh.primitive_cube_add(size=0.5, location=cube_location)
cube_obj = bpy.context.object
cube_obj.name = "GreenBeveledChildCube"

# --- Create and Assign Cube Material ---
mat_cube = bpy.data.materials.new(name="GreenPlasticMaterial")
mat_cube.use_nodes = True
bsdf_cube = mat_cube.node_tree.nodes.get("Principled BSDF")
if bsdf_cube:
    bsdf_cube.inputs["Base Color"].default_value = (0.0, 0.8, 0.0, 1.0)
    # Rule: Correctly set material properties for dielectrics
    bsdf_cube.inputs["IOR"].default_value = 1.450
    bsdf_cube.inputs["Roughness"].default_value = 0.5
cube_obj.data.materials.append(mat_cube)

# --- Add Modifier and Smooth Shading ---
bevel_mod = cube_obj.modifiers.new(name="BevelEdges", type='BEVEL')
bevel_mod.width = 0.05
bevel_mod.segments = 3

# Rule: Manage selection state before using operators
bpy.context.view_layer.objects.active = cube_obj
cube_obj.select_set(True)
bpy.ops.object.shade_smooth()

# --- Parent Cube to Sphere ---
# Rule: Use matrix_parent_inverse for objects already in world space
cube_obj.parent = sphere_obj
cube_obj.matrix_parent_inverse = sphere_obj.matrix_world.inverted()

# Deselect all at the end to clean up the user's view
bpy.ops.object.select_all(action='DESELECT')
```"""  # noqa


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
                delete_message_op = row.operator("gemini.delete_message", text="", icon="TRASH", emboss=False)
                delete_message_op.message_index = index
            else:
                row = box.row()
                row.label(text=f"User: {message.content}")
                delete_message_op = row.operator("gemini.delete_message", text="", icon="TRASH", emboss=False)
                delete_message_op.message_index = index

        column.separator()

        column.label(text="Gemini Model:")
        column.prop(context.scene, "gemini_model", text="")

        # Conditionally show the 'Enable Thinking' toggle
        if "gemini-2.5-flash" in context.scene.gemini_model:
            column.prop(context.scene, "gemini_enable_thinking")

        column.label(text="Enter your message:")
        column.prop(context.scene, "gemini_chat_input", text="")
        button_label = "Please wait...(this might take some time)" if context.scene.gemini_button_pressed else "Execute"
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

    natural_language_input: bpy.props.StringProperty(
        name="Command",
        description="Enter the natural language command",
        default="",
    )

    def execute(self, context):
        api_key = get_api_key(context, __name__)
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            self.report({"ERROR"}, "No API key detected. Please set your Gemini API key in the addon preferences.")
            return {"CANCELLED"}

        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        max_fix_attempts = addon_prefs.max_fix_attempts

        context.scene.gemini_button_pressed = True
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)

        blender_code = generate_blender_code(
            context.scene.gemini_chat_input, context.scene.gemini_chat_history, context, system_prompt
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

            namespace = {"bpy": bpy, "context": context, "__name__": "__main__"}

            try:
                exec(blender_code, namespace)
            except Exception as e:
                if max_fix_attempts <= 0:
                    self.report({"ERROR"}, f"Error executing code and fixes are disabled: {str(e)}")
                    context.scene.gemini_button_pressed = False
                    return {"CANCELLED"}

                error_message = f"Error executing generated code: {str(e)}"
                self.report({"WARNING"}, f"Original code had an error. Attempting to fix (1/{max_fix_attempts})...")

                context.scene.gemini_chat_history.remove(history_index)

                objects_after = set(bpy.data.objects)
                materials_after = set(bpy.data.materials)

                for obj in objects_after - objects_before:
                    bpy.data.objects.remove(obj, do_unlink=True)

                for mat in materials_after - materials_before:
                    bpy.data.materials.remove(mat)

                current_code = blender_code
                current_error = error_message

                for attempt in range(1, max_fix_attempts + 1):
                    fixed_code = fix_blender_code(current_code, current_error, context, system_prompt)

                    if not fixed_code:
                        self.report({"ERROR"}, f"Could not fix the code on attempt {attempt}: {current_error}")
                        context.scene.gemini_button_pressed = False
                        return {"CANCELLED"}

                    fix_history_index = len(context.scene.gemini_chat_history)
                    message = context.scene.gemini_chat_history.add()
                    message.type = "assistant"
                    message.content = fixed_code

                    try:
                        exec(fixed_code, namespace)
                        self.report({"INFO"}, f"Code fixed and executed successfully on attempt {attempt}!")
                        break
                    except Exception as e2:
                        current_error = f"Error executing fixed code: {str(e2)}"

                        if attempt < max_fix_attempts:
                            self.report(
                                {"WARNING"},
                                f"Fix attempt {attempt} had an error. Attempting to fix again ({attempt+1}/{max_fix_attempts})...",
                            )

                            context.scene.gemini_chat_history.remove(fix_history_index)

                            objects_after_fix = set(bpy.data.objects)
                            materials_after_fix = set(bpy.data.materials)

                            for obj in objects_after_fix - objects_before:
                                bpy.data.objects.remove(obj, do_unlink=True)

                            for mat in materials_after_fix - materials_before:
                                bpy.data.materials.remove(mat)

                            current_code = fixed_code
                        else:
                            self.report({"ERROR"}, f"Error executing code after {max_fix_attempts} fix attempts: {e2}")
                            context.scene.gemini_button_pressed = False
                            return {"CANCELLED"}
        else:
            self.report({"ERROR"}, "Failed to generate code from Gemini API. Please check the console for details.")
            context.scene.gemini_button_pressed = False
            return {"CANCELLED"}

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

    temperature: bpy.props.FloatProperty(
        name="Temperature",
        description="Controls randomness: Lower values are more deterministic, higher values more creative",
        default=0.70,
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
        layout.prop(self, "temperature")
        layout.prop(self, "top_p")
        layout.prop(self, "top_k")

        layout.separator()

        layout.label(text="Error Handling:")
        layout.prop(self, "max_fix_attempts")


def register():
    bpy.utils.register_class(GEMINI_AddonPreferences)
    bpy.utils.register_class(GEMINI_OT_DeleteMessage)
    bpy.utils.register_class(GEMINI_OT_Execute)
    bpy.utils.register_class(GEMINI_PT_Panel)
    bpy.utils.register_class(GEMINI_OT_ClearChat)
    bpy.utils.register_class(GEMINI_OT_ShowCode)

    bpy.types.VIEW3D_MT_mesh_add.append(menu_func)
    init_props()


def unregister():
    bpy.utils.unregister_class(GEMINI_AddonPreferences)
    bpy.utils.unregister_class(GEMINI_OT_DeleteMessage)
    bpy.utils.unregister_class(GEMINI_OT_Execute)
    bpy.utils.unregister_class(GEMINI_PT_Panel)
    bpy.utils.unregister_class(GEMINI_OT_ClearChat)
    bpy.utils.unregister_class(GEMINI_OT_ShowCode)

    bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)
    clear_props()


if __name__ == "__main__":
    register()
