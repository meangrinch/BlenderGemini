# BlenderGemini

Blender can be controlled using program scripts written in Python. This plugin provides an easy to use interface that integrates Google's Gemini right in the UI, allowing you to use natural language commands to control Blender.

## Installation

1. Clone this repository by clicking `Code > Download ZIP` on GitHub
2. Open Blender, go to `Edit > Preferences > Add-ons > Install from Disk`
3. Select the downloaded ZIP file and click `Install from Disk`
4. Enable the add-on by checking the checkbox next to `Gemini Blender Assistant`
5. Paste your Gemini API key in the Addon preferences menu
6. To view the code generations in realtime, go to `Window > Toggle System Console`

## Usage

1. In the 3D View, open the sidebar (press `N` if not visible) and locate the `Gemini Blender Assistant` tab
2. Type a natural language command in the input field, e.g., "create a cube at the origin"
3. Click the `Execute` button to generate and execute the Blender Python code

Note: If the generated code produced an error, Gemini will automatically attempt to fix it (up to 1 time by default).

## Requirements

- Blender 3.1 or later
- Gemini API key

## Models Available

- Gemini 2.5 Pro Preview 05-06
- Gemini 2.5 Pro Preview 03-25
- Gemini 2.5 Pro Experimental 03-25
- Gemini 2.5 Flash Preview 04-17
- Gemini 2.0 Flash
- Gemini 2.0 Flash Lite
