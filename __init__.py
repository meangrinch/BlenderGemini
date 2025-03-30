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
    "version": (1, 2, 3),
    "location": "3D View > UI > Gemini Blender Assistant",
    "description": "Generate Blender Python code using Google's Gemini to perform various tasks.",
    "wiki_url": "",
    "tracker_url": "",
}

system_prompt = """You are a Blender Python code assistant:

1. Respond with your answers in markdown (```).

2. When creating or modifying objects:
  - Change existing objects when possible
  - Delete and recreate objects if modification isn't feasible
  
3. When creating objects, consider and implement where appropriate:
  - Materials: Use nodes. Set properties (color, metallic, roughness, etc.)
  - Textures/UVs: Apply textures, set up UV mapping
  - Transforms: Set location, rotation, scale
  - Modifiers: Use for non-destructive effects
  - Custom Properties: Add when needed
  - Parenting: Establish parent-child relationships
  
4. Do not perform destructive operations on the meshes.

5. Do not invent non-existent parameters like 'segments', 'cap_ends', or 'specular'.

6. Do not do more than what is asked (setting up render settings, adding cameras, etc.,).

7. Do not respond with anything that is not Python code.

Example:

user: Create a sphere and make it metallic red. Add a cube, make it smaller, color it green, position it 2 units above the sphere, and parent the cube to the sphere.
assistant:
```
import bpy

# --- Create Sphere ---
bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))
sphere_obj = bpy.context.object
sphere_obj.name = "MetallicRedSphere"

# --- Create Sphere Material ---
mat_sphere = bpy.data.materials.new(name="RedMetallic")
mat_sphere.use_nodes = True
principled_bsdf_sphere = mat_sphere.node_tree.nodes.get("Principled BSDF")
if principled_bsdf_sphere:
    principled_bsdf_sphere.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)
    principled_bsdf_sphere.inputs["Metallic"].default_value = 1.0
    principled_bsdf_sphere.inputs["Roughness"].default_value = 0.2

if sphere_obj.data.materials:
    sphere_obj.data.materials[0] = mat_sphere
else:
    sphere_obj.data.materials.append(mat_sphere)

# --- Create Cube ---
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 2.0)) # Position above sphere
cube_obj = bpy.context.object
cube_obj.name = "GreenChildCube"

# --- Set Cube Transforms ---
cube_obj.scale = (0.5, 0.5, 0.5)

# --- Create Cube Material ---
mat_cube = bpy.data.materials.new(name="GreenDiffuse")
mat_cube.use_nodes = True
principled_bsdf_cube = mat_cube.node_tree.nodes.get("Principled BSDF")
if principled_bsdf_cube:
    principled_bsdf_cube.inputs["Base Color"].default_value = (0.0, 1.0, 0.0, 1.0)
    principled_bsdf_cube.inputs["Metallic"].default_value = 0.0
    principled_bsdf_cube.inputs["Roughness"].default_value = 0.5

# Assign material to cube
if cube_obj.data.materials:
    cube_obj.data.materials[0] = mat_cube
else:
    cube_obj.data.materials.append(mat_cube)

# --- Parent Cube to Sphere ---
if cube_obj and sphere_obj:
    bpy.ops.object.select_all(action='DESELECT')
    cube_obj.select_set(True)
    sphere_obj.select_set(True)
    bpy.context.view_layer.objects.active = sphere_obj
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)

bpy.ops.object.select_all(action='DESELECT')
```"""

class GEMINI_OT_DeleteMessage(bpy.types.Operator):
    bl_idname = "gemini.delete_message"
    bl_label = "Delete Message"
    bl_options = {'REGISTER', 'UNDO'}

    message_index: bpy.props.IntProperty()

    def execute(self, context):
        context.scene.gemini_chat_history.remove(self.message_index)
        return {'FINISHED'}

class GEMINI_OT_ShowCode(bpy.types.Operator):
    bl_idname = "gemini.show_code"
    bl_label = "Show Code"
    bl_options = {'REGISTER', 'UNDO'}

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
            if area.type == 'TEXT_EDITOR':
                text_editor_area = area
                break

        if text_editor_area is None:
            text_editor_area = split_area_to_text_editor(context)
        
        text_editor_area.spaces.active.text = text

        return {'FINISHED'}

class GEMINI_PT_Panel(bpy.types.Panel):
    bl_label = "Gemini Blender Assistant"
    bl_idname = "GEMINI_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Gemini Assistant'

    def draw(self, context):
        layout = self.layout
        column = layout.column(align=True)

        column.label(text="Chat history:")
        box = column.box()
        for index, message in enumerate(context.scene.gemini_chat_history):
            if message.type == 'assistant':
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
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        context.scene.gemini_chat_history.clear()
        return {'FINISHED'}

class GEMINI_OT_Execute(bpy.types.Operator):
    bl_idname = "gemini.send_message"
    bl_label = "Send Message"
    bl_options = {'REGISTER', 'UNDO'}

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
            self.report({'ERROR'}, "No API key detected. Please set your Gemini API key in the addon preferences.")
            return {'CANCELLED'}

        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        max_fix_attempts = addon_prefs.max_fix_attempts

        context.scene.gemini_button_pressed = True
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        
        blender_code = generate_blender_code(context.scene.gemini_chat_input, context.scene.gemini_chat_history, context, system_prompt)

        message = context.scene.gemini_chat_history.add()
        message.type = 'user'
        message.content = context.scene.gemini_chat_input

        context.scene.gemini_chat_input = ""
    
        if blender_code:
            objects_before = set(bpy.data.objects)
            materials_before = set(bpy.data.materials)
            
            history_index = len(context.scene.gemini_chat_history)
            message = context.scene.gemini_chat_history.add()
            message.type = 'assistant'
            message.content = blender_code

            namespace = {
                'bpy': bpy,
                'context': context,
                '__name__': '__main__'
            }
            
            try:
                exec(blender_code, namespace)
            except Exception as e:
                if max_fix_attempts <= 0:
                    self.report({'ERROR'}, f"Error executing code and fixes are disabled: {str(e)}")
                    context.scene.gemini_button_pressed = False
                    return {'CANCELLED'}
                    
                error_message = f"Error executing generated code: {str(e)}"
                self.report({'WARNING'}, f"Original code had an error. Attempting to fix (1/{max_fix_attempts})...")
                
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
                        self.report({'ERROR'}, f"Could not fix the code on attempt {attempt}: {current_error}")
                        context.scene.gemini_button_pressed = False
                        return {'CANCELLED'}
                    
                    fix_history_index = len(context.scene.gemini_chat_history)
                    message = context.scene.gemini_chat_history.add()
                    message.type = 'assistant'
                    message.content = fixed_code
                    
                    try:
                        exec(fixed_code, namespace)
                        self.report({'INFO'}, f"Code fixed and executed successfully on attempt {attempt}!")
                        break
                    except Exception as e2:
                        current_error = f"Error executing fixed code: {str(e2)}"
                        
                        if attempt < max_fix_attempts:
                            self.report({'WARNING'}, f"Fix attempt {attempt} had an error. Attempting to fix again ({attempt+1}/{max_fix_attempts})...")
                            
                            context.scene.gemini_chat_history.remove(fix_history_index)
                            
                            objects_after_fix = set(bpy.data.objects)
                            materials_after_fix = set(bpy.data.materials)
                            
                            for obj in objects_after_fix - objects_before:
                                bpy.data.objects.remove(obj, do_unlink=True)
                            
                            for mat in materials_after_fix - materials_before:
                                bpy.data.materials.remove(mat)
                            
                            current_code = fixed_code
                        else:
                            self.report({'ERROR'}, f"Error executing code after {max_fix_attempts} fix attempts: {e2}")
                            context.scene.gemini_button_pressed = False
                            return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "Failed to generate code from Gemini API. Please check the console for details.")
            context.scene.gemini_button_pressed = False
            return {'CANCELLED'}

        context.scene.gemini_button_pressed = False
        return {'FINISHED'}

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
        default=1.0,
        min=0.0,
        max=1.0,
        precision=2,
        step=10
    )
    
    top_p: bpy.props.FloatProperty(
        name="Top P",
        description="Controls diversity of output via nucleus sampling",
        default=0.95,
        min=0.0,
        max=1.0,
        precision=2,
        step=5
    )
    
    top_k: bpy.props.IntProperty(
        name="Top K",
        description="Limits token selection to the K most likely tokens",
        default=1,
        min=1,
        max=64
    )
    
    max_fix_attempts: bpy.props.IntProperty(
        name="Max Fix Attempts",
        description="Maximum number of times to attempt fixing code errors (0 = don't attempt fixes)",
        default=1,
        min=0,
        max=5
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
