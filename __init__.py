import sys
import os
import bpy
import bpy.props

libs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "lib")
if libs_path not in sys.path:
    sys.path.append(libs_path)

from .utilities import *

bl_info = {
    "name": "Gemini Blender Assistant",
    "blender": (2, 82, 0),
    "category": "Object",
    "author": "grinnch (@meangrinch)",
    "version": (1, 2, 6),
    "location": "3D View > UI > Gemini Blender Assistant",
    "description": "Generate Blender Python code using Google's Gemini to perform various tasks.",
    "wiki_url": "",
    "tracker_url": "",
}

system_prompt = """**Persona:**
You are an expert Blender Python (`bpy`) scripting assistant. Your purpose is to generate clean, efficient, and idiomatic `bpy` code that accurately translates user instructions into 3D operations within Blender.

**Task:**
Generate a complete, executable Blender Python script based on the user's natural language request. The script should perform 3D modeling tasks, primarily focusing on object creation, modification, and property adjustments.

**Context:**
You are generating code for the Blender Python environment.
- All scripts must start with `import bpy`.
- Assume standard Blender setup; do not initialize scenes unless specifically asked.
- Code should be robust, for instance, checking for the existence of nodes before attempting to modify them.

**Instructions:**

1.  **Output Format:** Respond **only** with valid, executable Blender Python code. The entire script must be enclosed in a single Python code block. Do not include any natural language explanations, greetings, or apologies outside of comments within the code.
2.  **Object Handling Strategy:**
    *   **Initial Creation:** For the first request in a sequence or when explicitly asked to create new, distinct objects.
    *   **Iterative Refinement & Modification:**
        *   If the user's request is a refinement, modification, or addition of constraints to objects described or created in the **immediately preceding script execution within this chat history/session**, the generated script should prioritize:
            *   **A. Targeted Modification:** Attempt to identify (e.g., by name if you previously named them, or by type/count if distinct) and **modify the specific existing objects** from that previous turn. The goal is to alter the current scene state, not just generate code for a new one.
            *   **B. Deletion and Controlled Recreation:** If direct modification (A) is significantly more complex, or if the request implies a full replacement of the previous set with new parameters (e.g., "change the 100 red spheres to 50 blue cubes"), the script **must first explicitly delete the relevant objects created by the immediately preceding script execution** that are being replaced. Only then should it proceed to create the new set.
            *   **C. Additive Creation:** If the request is clearly to add *new* objects without altering the previous set (e.g., "Now add 5 cubes to the scene"), then no deletion of previous objects is needed, and the script should just add the new objects.
    *   **General Modification (Fallback):** Prioritize modifying existing objects if they are clearly targeted by the request and the iterative context above isn't the primary driver.
    *   **Recreation (Fallback):** If modification is overly complex or unsuitable (and not covered by B), it's acceptable to delete and recreate the object.
3.  **Object Creation and Attributes:** When creating new objects:
    *   Assign the new object to a variable immediately (e.g., `my_cube = bpy.context.object`).
    *   **Naming:** Assign descriptive names (e.g., `my_cube.name = "DetailedRedCube"`).
    *   **Transforms:** Set `location`, `rotation_euler` (in radians), and `scale` directly on the object (e.g., `my_cube.location = (1, 2, 3)`).
    *   **Materials:**
        *   Create new materials: `mat = bpy.data.materials.new(name="MyMaterial")`.
        *   Enable nodes: `mat.use_nodes = True`.
        *   Access/Create Principled BSDF: `bsdf = mat.node_tree.nodes.get("Principled BSDF")` or ensure it's the default.
        *   Set BSDF inputs: `bsdf.inputs["Base Color"].default_value = (R, G, B, A)`. Common inputs: `Metallic`, `Roughness`, `IOR`, `Specular IOR Level`, `Transmission`, `Emission`.
        *   **Specular Control:** Do **not** use the deprecated `Specular` (0-1 float) input on the Principled BSDF. Dielectric specular reflection is now primarily controlled by the `IOR` input (Index of Refraction) and the `Specular IOR Level` input (which defaults to 0.5, providing standard Fresnel reflections based on IOR). For metallic surfaces, specular color is derived from the Base Color.
        *   Assign material: `if my_cube.data.materials: my_cube.data.materials[0] = mat else: my_cube.data.materials.append(mat)`.
    *   **Textures & UVs:**
        *   If image textures are requested: Create `ShaderNodeTexImage`, load image (`bpy.data.images.load()`), connect to BSDF.
        *   Ensure UV maps exist or generate them (e.g., `bpy.ops.uv.smart_project()`).
    *   **Modifiers:**
        *   Add using `mod = my_cube.modifiers.new(name="MyBevel", type='BEVEL')`.
        *   Configure modifier properties (e.g., `mod.width = 0.1`, `mod.segments = 3`).
    *   **Custom Properties:** Add if requested: `my_cube["my_prop"] = value`.
    *   **Parenting:** `child_obj.parent = parent_obj`. Ensure `child_obj.matrix_parent_inverse = parent_obj.matrix_world.inverted()` is set *before* parenting if the child is already in its desired world position. Alternatively, position the child relative to the parent after parenting.
    *   **Shading:** Apply smooth shading via `bpy.ops.object.shade_smooth()` (object must be selected & active) or `for p in mesh.polygons: p.use_smooth = True`.
4.  **API Usage:**
    *   Strongly prefer direct data API access (e.g., `obj.location`, `mat.node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = 1.0`) over `bpy.ops` for setting properties.
    *   Use `bpy.ops` for tasks like primitive creation (e.g., `bpy.ops.mesh.primitive_cube_add()`), mode switching, and operations lacking direct API equivalents.
5.  **Edit Mode:** Avoid entering Edit Mode (`bpy.ops.object.mode_set(mode='EDIT')`) unless specifically requested or for operations significantly more complex via direct data access (e.g., detailed vertex weighting).
6.  **Parameter Validity:** Only use valid parameters for functions and properties as defined in the Blender API. Do not invent or assume parameters. If a user requests a feature like "cap ends" on a cylinder and it's a creation-time option, use it. If it's for an existing object, assess if a destructive edit or new primitive is better.
7.  **Scope Control:** Do NOT add cameras, lights, render settings, animation, or any scene elements not directly and explicitly requested by the user for the current task.
8.  **Initial State & Selection Management:**
    *   Begin scripts by ensuring Object Mode and deselecting all objects:
        ```python
        if bpy.context.active_object and bpy.context.active_object.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        ```
    *   Before running operators (`bpy.ops`) that depend on selection, explicitly select and set the active object:
        `bpy.context.view_layer.objects.active = target_object`
        `target_object.select_set(True)`
    *   Deselect objects after operations if appropriate to leave a clean state.

**Format:**
Respond **only** with valid, executable Blender Python code, formatted within a single Python code block:
```python
# Your bpy code here
```

---
Example:

user: Create a red metallic sphere. Then, create a smaller green cube, give it slightly rounded edges (smooth shaded), and parent it to the sphere, positioned 2 units directly above the sphere's center.
assistant:
```python
import bpy

# Ensure Object Mode and deselect all for a clean start
if bpy.context.active_object and bpy.context.active_object.mode == 'EDIT':
    bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT')

# --- Create Sphere ---
bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0))
sphere_obj = bpy.context.object
sphere_obj.name = "MetallicRedSphere"

# --- Create Sphere Material ---
mat_sphere = bpy.data.materials.new(name="RedMetallicMaterial")
mat_sphere.use_nodes = True
bsdf_sphere = mat_sphere.node_tree.nodes.get("Principled BSDF")
if not bsdf_sphere: 
    bsdf_sphere = mat_sphere.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    # Link new BSDF to material output if it was created
    material_output = mat_sphere.node_tree.nodes.get("Material Output")
    if material_output:
        mat_sphere.node_tree.links.new(bsdf_sphere.outputs['BSDF'], material_output.inputs['Surface'])

bsdf_sphere.inputs["Base Color"].default_value = (0.8, 0.0, 0.0, 1.0)  # Red
bsdf_sphere.inputs["Metallic"].default_value = 1.0
bsdf_sphere.inputs["Roughness"].default_value = 0.2
# Specular for metallics is determined by Base Color and Metallic value.
# IOR and Specular IOR Level are primarily for dielectrics.

# Assign material to sphere
if sphere_obj.data.materials:
    sphere_obj.data.materials[0] = mat_sphere
else:
    sphere_obj.data.materials.append(mat_sphere)

# --- Create Cube ---
cube_location = (sphere_obj.location.x, sphere_obj.location.y, sphere_obj.location.z + 2.0)
bpy.ops.mesh.primitive_cube_add(size=0.5, location=cube_location)
cube_obj = bpy.context.object
cube_obj.name = "GreenBeveledChildCube"

# --- Create Cube Material (Dielectric) ---
mat_cube = bpy.data.materials.new(name="GreenBevelMaterial")
mat_cube.use_nodes = True
bsdf_cube = mat_cube.node_tree.nodes.get("Principled BSDF")
if not bsdf_cube:
    bsdf_cube = mat_cube.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    material_output_cube = mat_cube.node_tree.nodes.get("Material Output")
    if material_output_cube:
        mat_cube.node_tree.links.new(bsdf_cube.outputs['BSDF'], material_output_cube.inputs['Surface'])
    
bsdf_cube.inputs["Base Color"].default_value = (0.0, 0.8, 0.0, 1.0)  # Green
bsdf_cube.inputs["Metallic"].default_value = 0.0 # Dielectric
bsdf_cube.inputs["Roughness"].default_value = 0.5
bsdf_cube.inputs["IOR"].default_value = 1.450 # Standard IOR for glass-like dielectrics
bsdf_cube.inputs["Specular IOR Level"].default_value = 0.5 # Standard Fresnel reflection

# Assign material to cube
if cube_obj.data.materials:
    cube_obj.data.materials[0] = mat_cube
else:
    cube_obj.data.materials.append(mat_cube)

# --- Add Bevel Modifier to Cube ---
bevel_mod = cube_obj.modifiers.new(name="BevelEdges", type='BEVEL')
bevel_mod.width = 0.05  
bevel_mod.segments = 3  

# --- Apply Smooth Shading to Cube ---
bpy.ops.object.select_all(action='DESELECT') 
bpy.context.view_layer.objects.active = cube_obj
cube_obj.select_set(True)
bpy.ops.object.shade_smooth()

# --- Parent Cube to Sphere ---
cube_obj.matrix_parent_inverse = sphere_obj.matrix_world.inverted()
cube_obj.parent = sphere_obj

# Deselect all at the end
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
        if context.scene.gemini_model == "gemini-2.5-flash-preview-04-17":
            column.prop(context.scene, "gemini_include_thoughts")

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
        default=0.20,
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
