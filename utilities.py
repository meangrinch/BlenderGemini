import bpy
import requests
import re
import os
import sys
import json


def get_api_key(context, addon_name):
    preferences = context.preferences
    addon_prefs = preferences.addons[addon_name].preferences
    return addon_prefs.api_key


def init_props():
    bpy.types.Scene.gemini_chat_history = bpy.props.CollectionProperty(type=bpy.types.PropertyGroup)
    bpy.types.Scene.gemini_model = bpy.props.EnumProperty(
        name="Gemini Model",
        description="Select the Gemini model to use",
        items=[
            ("gemini-2.0-flash-exp", "Gemini 2.0 Flash Experimental", "Use Gemini 2.0 Flash"),
            ("gemini-exp-1206", "Gemini Experimental 1206", "Use Gemini Experimental 1206"),
        ],
        default="gemini-2.0-flash-exp",
    )
    bpy.types.Scene.gemini_chat_input = bpy.props.StringProperty(
        name="Message",
        description="Enter your message",
        default="",
    )
    bpy.types.Scene.gemini_button_pressed = bpy.props.BoolProperty(default=False)
    bpy.types.PropertyGroup.type = bpy.props.StringProperty()
    bpy.types.PropertyGroup.content = bpy.props.StringProperty()

def clear_props():
    del bpy.types.Scene.gemini_chat_history
    del bpy.types.Scene.gemini_chat_input
    del bpy.types.Scene.gemini_button_pressed

def generate_blender_code(prompt, chat_history, context, system_prompt):
    api_key = get_api_key(context, "BlenderGemini")
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/" + context.scene.gemini_model + ":generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    messages = [{"text": system_prompt}]
    for message in chat_history[-10:]:
        if message.type == "assistant":
            messages.append({"text": "```\n" + message.content + "\n```"})
        else:
            messages.append({"text": message.content})

    messages.append({"text": "Can you please write Blender code for me that accomplishes the following task: " + prompt + "? Do not respond with anything that is not Python code. Do not provide explanations"})

    data = {
        "contents": [{"parts": messages}],
        "generationConfig": {
            "temperature": 1.0,
            "topP": 0.95,
            "topK": 40,
            "maxOutputTokens": 8192,
        }
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        
        # Extract code between ```python and ``` markers
        code_match = re.search(r'```(?:python)?(.*?)```', response_text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return response_text.strip()
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def split_area_to_text_editor(context):
    area = context.area
    for region in area.regions:
        if region.type == 'WINDOW':
            with context.temp_override(area=area, region=region):
                bpy.ops.screen.area_split(direction='VERTICAL', factor=0.5)
            break

    new_area = context.screen.areas[-1]
    new_area.type = 'TEXT_EDITOR'
    return new_area