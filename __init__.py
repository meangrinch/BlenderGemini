import sys
import os
import bpy
import bpy.props
import re

# Add the 'libs' folder to the Python path
libs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "lib")
if libs_path not in sys.path:
    sys.path.append(libs_path)

from .utilities import *
bl_info = {
    "name": "Gemini Blender Assistant",
    "blender": (2, 82, 0),
    "category": "Object",
    "author": "grinnch (@meangrinch)",
    "version": (1, 1, 0),
    "location": "3D View > UI > Gemini Blender Assistant",
    "description": "Generate Blender Python code using Google's Gemini to perform various tasks.",
    "wiki_url": "",
    "tracker_url": "",
}

system_prompt = """You are a Blender Python code assistant. Generate concise Python code snippets for Blender, the 3D software.
- Respond with your answers in markdown (```).
- When modifying objects:
  * Change existing objects when possible
  * Delete and recreate objects if modification isn't feasible
- When creating objects, consider and implement where appropriate:
  * Materials: Use nodes. Set properties (color, metallic, roughness, etc.)
  * Textures/UVs: Apply textures, set up UV mapping
  * Transforms: Set location, rotation, scale
  * Modifiers: Use for non-destructive effects
  * Custom Properties: Add when needed
  * Parenting: Establish parent-child relationships
- When creating materials, use nodes for better control and flexibility.
- Do not perform destructive operations on the meshes.
- Do not invent non-existent parameters like 'segments', 'cap_ends', or 'specular'.
- Do not do more than what is asked (setting up render settings, adding cameras, etc).
- Do not respond with anything that is not Python code.

Example:

user: create a red metallic cube at position (0,0,0)
assistant:
```
import bpy

# Create a new cube
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))

# Get the reference to the created cube
cube = bpy.context.object

# Create a new material
material = bpy.data.materials.new(name="RedMetallicMaterial")
material.use_nodes = True
nodes = material.node_tree.nodes

# Clear default nodes
for node in nodes:
    nodes.remove(node)

# Create Principled BSDF node
bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
bsdf.location = (0, 0)
bsdf.inputs['Base Color'].default_value = (1, 0, 0, 1)  # Red color
bsdf.inputs['Metallic'].default_value = 1  # Metallic
bsdf.inputs['Roughness'].default_value = 0.25  # Slightly shiny

# Create Material Output node
material_output = nodes.new(type='ShaderNodeOutputMaterial')
material_output.location = (200, 0)

# Link nodes
material.node_tree.links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

# Assign material to the cube
if cube.data.materials:
    cube.data.materials[0] = material
else:
    cube.data.materials.append(material)
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

        context.scene.gemini_button_pressed = True
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        
        blender_code = generate_blender_code(context.scene.gemini_chat_input, context.scene.gemini_chat_history, context, system_prompt)

        message = context.scene.gemini_chat_history.add()
        message.type = 'user'
        message.content = context.scene.gemini_chat_input

        context.scene.gemini_chat_input = ""
    
        if blender_code:
            message = context.scene.gemini_chat_history.add()
            message.type = 'assistant'
            message.content = blender_code

            # Create a new namespace dictionary with globals
            namespace = {
                'bpy': bpy,
                'context': context,
                '__name__': '__main__'
            }
            
            try:
                exec(blender_code, namespace)
            except Exception as e:
                self.report({'ERROR'}, f"Error executing generated code: {e}")
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

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "api_key")

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
