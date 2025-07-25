import re
import time

import bpy
import requests


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


def clear_props():
    del bpy.types.Scene.gemini_chat_history
    del bpy.types.Scene.gemini_chat_input
    del bpy.types.Scene.gemini_button_pressed
    del bpy.types.Scene.gemini_enable_thinking


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


def generate_blender_code(prompt, chat_history, context, system_prompt, detailed_geometry=None, use_3d_cursor=False):
    api_key = get_api_key(context, "BlenderGemini")

    preferences = context.preferences
    addon_prefs = preferences.addons["BlenderGemini"].preferences

    url = "https://generativelanguage.googleapis.com/v1beta/models/" + context.scene.gemini_model + ":generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    contents = []
    for message in chat_history[-10:]:  # Keep last 10 messages for context
        role = "user" if message.type == "user" else "model"
        content = message.content if message.type == "user" else "```\n" + message.content + "\n```"
        contents.append({"role": role, "parts": [{"text": content}]})

    scene_context = get_scene_objects_as_text(context)
    full_prompt = ""
    if detailed_geometry:
        full_prompt += "**Detailed Object Geometry:**\n" + detailed_geometry + "\n\n"

    if use_3d_cursor:
        cursor_loc = context.scene.cursor.location
        full_prompt += (
            f"**3D Cursor Location (World Space):**\n"
            f"({cursor_loc.x:.4f}, {cursor_loc.y:.4f}, {cursor_loc.z:.4f})\n\n"
        )

    full_prompt += "**Scene Summary:**\n" + scene_context + "\n\nUser Request: " + prompt

    contents.append({"role": "user", "parts": [{"text": full_prompt}]})

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

    # Conditionally add thinkingConfig for specific model
    if "gemini-2.5-flash" in context.scene.gemini_model and not context.scene.gemini_enable_thinking:
        data["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}

    response_text = make_gemini_api_request(url, headers, data)
    if response_text:
        # Extract code between ```python and ``` markers
        code_match = re.search(r"```(?:python)?(.*?)```", response_text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return response_text.strip()
    return None


def fix_blender_code(
    original_code, error_message, context, system_prompt, detailed_geometry=None, use_3d_cursor=False
):
    """Generate fixed Blender code based on the error message."""
    api_key = get_api_key(context, "BlenderGemini")

    preferences = context.preferences
    addon_prefs = preferences.addons["BlenderGemini"].preferences

    url = "https://generativelanguage.googleapis.com/v1beta/models/" + context.scene.gemini_model + ":generateContent"
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
        cursor_loc = context.scene.cursor.location
        cursor_block = f"""
**[3D CURSOR LOCATION (World Space)]:**
({cursor_loc.x:.4f}, {cursor_loc.y:.4f}, {cursor_loc.z:.4f})
"""

    fix_prompt = f"""## Persona
You are a `bpy` Debugging Specialist. Your sole function is to analyze the provided faulty Python script and its corresponding error message, and then generate a corrected, fully functional version.

## Task Context
You will be given a script that failed, its error, and a scene summary. Use all of this information to provide a fix.
{detailed_geo_block}
{cursor_block}
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

## Core Directives for Correction

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

    data = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": fix_prompt}]}],
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

    # Conditionally add thinkingConfig for specific model
    if "gemini-2.5-flash" in context.scene.gemini_model and not context.scene.gemini_enable_thinking:
        data["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}

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
