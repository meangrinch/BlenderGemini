import bpy
import requests
import re
import time


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
            ("gemini-2.5-pro-preview-05-06", "Gemini 2.5 Pro Preview 05-06", "Use Gemini 2.5 Pro Preview 05-06"),
            ("gemini-2.5-pro-preview-03-25", "Gemini 2.5 Pro Preview 03-25", "Use Gemini 2.5 Pro Preview 03-25"),
            ("gemini-2.5-pro-exp-03-25", "Gemini 2.5 Pro Exp 03-25", "Use Gemini 2.5 Pro Experimental"),
            ("gemini-2.5-flash-preview-04-17", "Gemini 2.5 Flash Preview 04-17", "Use Gemini 2.5 Flash Preview"),
            ("gemini-2.0-flash", "Gemini 2.0 Flash", "Use Gemini 2.0 Flash"),
            ("gemini-2.0-flash-lite", "Gemini 2.0 Flash Lite", "Use Gemini 2.0 Flash Lite"),
        ],
        default="gemini-2.5-pro-exp-03-25",
    )
    bpy.types.Scene.gemini_chat_input = bpy.props.StringProperty(
        name="Message",
        description="Enter your message",
        default="",
    )
    bpy.types.Scene.gemini_button_pressed = bpy.props.BoolProperty(default=False)
    bpy.types.PropertyGroup.type = bpy.props.StringProperty()
    bpy.types.PropertyGroup.content = bpy.props.StringProperty()
    bpy.types.Scene.gemini_include_thoughts = bpy.props.BoolProperty(
        name="Enable Thinking",
        description="Enable model's thinking capabilities in the response (only for compatible models)",
        default=True,
    )


def clear_props():
    del bpy.types.Scene.gemini_chat_history
    del bpy.types.Scene.gemini_chat_input
    del bpy.types.Scene.gemini_button_pressed
    del bpy.types.Scene.gemini_include_thoughts


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
            error_msg = f"API request failed (attempt {attempt+1}/{max_retries}): {str(e)}"
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


def generate_blender_code(prompt, chat_history, context, system_prompt):
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

    contents.append({"role": "user", "parts": [{"text": prompt}]})

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
    if context.scene.gemini_model == "gemini-2.5-flash-preview-04-17" and context.scene.gemini_include_thoughts:
        data["generationConfig"]["thinkingConfig"] = {"includeThoughts": True}

    response_text = make_gemini_api_request(url, headers, data)
    if response_text:
        # Extract code between ```python and ``` markers
        code_match = re.search(r"```(?:python)?(.*?)```", response_text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return response_text.strip()
    return None


def fix_blender_code(original_code, error_message, context, system_prompt):
    """Generate fixed Blender code based on the error message."""
    api_key = get_api_key(context, "BlenderGemini")

    preferences = context.preferences
    addon_prefs = preferences.addons["BlenderGemini"].preferences

    url = "https://generativelanguage.googleapis.com/v1beta/models/" + context.scene.gemini_model + ":generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    fix_prompt = f"""You previously generated this Blender Python code:

```
{original_code}
```

But it produced this error:

```
{error_message}
```

Please fix the code to resolve this error. Only provide the corrected code without any explanation or additional text. The fixed code should accomplish the same task as the original code but without errors."""  # noqa

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
    if context.scene.gemini_model == "gemini-2.5-flash-preview-04-17" and context.scene.gemini_include_thoughts:
        data["generationConfig"]["thinkingConfig"] = {"includeThoughts": True}


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
