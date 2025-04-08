"""
Speed Dial Module for Kokoro-TTS-Local
--------------------------------------
Manages speed dial presets for quick access to frequently used voice and text combinations.

This module provides functions to:
- Load speed dial presets from a JSON file
- Save new presets to the JSON file
- Delete presets from the JSON file
- Validate preset data
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

# Define the path for the speed dial presets file
SPEED_DIAL_FILE = Path("speed_dial.json")

def load_presets() -> Dict[str, Dict[str, Any]]:
    """
    Load speed dial presets from the JSON file.
    
    Returns:
        Dictionary of presets where keys are preset names and values are preset data
    """
    if not SPEED_DIAL_FILE.exists():
        # If file doesn't exist, return an empty dictionary
        return {}
    
    try:
        with open(SPEED_DIAL_FILE, 'r', encoding='utf-8') as f:
            presets = json.load(f)
        
        # Validate the loaded presets
        validated_presets = {}
        for name, preset in presets.items():
            if validate_preset(preset):
                validated_presets[name] = preset
        
        return validated_presets
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading speed dial presets: {e}")
        return {}

def save_preset(name: str, voice: str, text: str, format: str = "wav", speed: float = 1.0) -> bool:
    """
    Save a new speed dial preset.
    
    Args:
        name: Name of the preset
        voice: Voice to use
        text: Text to convert to speech
        format: Output format (default: "wav")
        speed: Speech speed (default: 1.0)
        
    Returns:
        True if successful, False otherwise
    """
    # Create preset data
    preset = {
        "voice": voice,
        "text": text,
        "format": format,
        "speed": speed
    }
    
    # Validate preset data
    if not validate_preset(preset):
        return False
    
    # Load existing presets
    presets = load_presets()
    
    # Add or update the preset
    presets[name] = preset
    
    # Save presets to file
    try:
        with open(SPEED_DIAL_FILE, 'w', encoding='utf-8') as f:
            json.dump(presets, f, indent=2, ensure_ascii=False)
        return True
    except IOError as e:
        print(f"Error saving speed dial preset: {e}")
        return False

def delete_preset(name: str) -> bool:
    """
    Delete a speed dial preset.
    
    Args:
        name: Name of the preset to delete
        
    Returns:
        True if successful, False otherwise
    """
    # Load existing presets
    presets = load_presets()
    
    # Check if preset exists
    if name not in presets:
        return False
    
    # Remove the preset
    del presets[name]
    
    # Save presets to file
    try:
        with open(SPEED_DIAL_FILE, 'w', encoding='utf-8') as f:
            json.dump(presets, f, indent=2, ensure_ascii=False)
        return True
    except IOError as e:
        print(f"Error deleting speed dial preset: {e}")
        return False

def validate_preset(preset: Dict[str, Any]) -> bool:
    """
    Validate a preset's data structure.
    
    Args:
        preset: Preset data to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Check required fields
    required_fields = ["voice", "text"]
    for field in required_fields:
        if field not in preset:
            print(f"Preset missing required field: {field}")
            return False
    
    # Check field types
    if not isinstance(preset.get("voice"), str):
        print("Preset voice must be a string")
        return False
    
    if not isinstance(preset.get("text"), str):
        print("Preset text must be a string")
        return False
    
    # Optional fields with defaults
    if "format" not in preset:
        preset["format"] = "wav"
    elif not isinstance(preset["format"], str):
        print("Preset format must be a string")
        return False
    
    if "speed" not in preset:
        preset["speed"] = 1.0
    elif not isinstance(preset["speed"], (int, float)):
        print("Preset speed must be a number")
        return False
    
    return True

def get_preset_names() -> List[str]:
    """
    Get a list of all preset names.
    
    Returns:
        List of preset names
    """
    presets = load_presets()
    return list(presets.keys())

def get_preset(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific preset by name.
    
    Args:
        name: Name of the preset to get
        
    Returns:
        Preset data or None if not found
    """
    presets = load_presets()
    return presets.get(name)
