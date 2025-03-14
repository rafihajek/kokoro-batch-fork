"""
Kokoro-TTS Local Generator
-------------------------
A Gradio interface for the Kokoro-TTS-Local text-to-speech system.
Supports multiple voices and audio formats, with cross-platform compatibility.

Key Features:
- Multiple voice models support (26+ voices)
- Real-time generation with progress logging
- WAV, MP3, and AAC output formats
- Network sharing capabilities
- Cross-platform compatibility (Windows, macOS, Linux)

Dependencies:
- kokoro: Official Kokoro TTS library
- gradio: Web interface framework
- soundfile: Audio file handling
- pydub: Audio format conversion
"""

import gradio as gr
import os
import sys
import platform
from datetime import datetime
import shutil
from pathlib import Path
import soundfile as sf
from pydub import AudioSegment
import torch
import numpy as np
from models import (
    list_available_voices, build_model,
    generate_speech, download_voice_files
)

# Global configuration
CONFIG_FILE = "tts_config.json"  # Stores user preferences and paths
DEFAULT_OUTPUT_DIR = "outputs"    # Directory for generated audio files
SAMPLE_RATE = 24000  # Updated from 22050 to match new model

# Initialize model globally
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = None

def get_available_voices():
    """Get list of available voice models."""
    try:
        # Initialize model to trigger voice downloads
        global model
        if model is None:
            print("Initializing model and downloading voices...")
            model = build_model(None, device)
        
        voices = list_available_voices()
        if not voices:
            print("No voices found after initialization. Attempting to download...")
            download_voice_files()  # Try downloading again
            voices = list_available_voices()
            
        print("Available voices:", voices)
        return voices
    except Exception as e:
        print(f"Error getting voices: {e}")
        return []

def convert_audio(input_path: str, output_path: str, format: str):
    """Convert audio to specified format."""
    try:
        if format == "wav":
            return input_path
        audio = AudioSegment.from_wav(input_path)
        if format == "mp3":
            audio.export(output_path, format="mp3", bitrate="192k")
        elif format == "aac":
            audio.export(output_path, format="aac", bitrate="192k")
        return output_path
    except Exception as e:
        print(f"Error converting audio: {e}")
        return input_path

def generate_tts_with_logs(voice_name, text, format):
    """Generate TTS audio with progress logging."""
    global model
    
    try:
        # Initialize model if needed
        if model is None:
            print("Initializing model...")
            model = build_model(None, device)
        
        # Create output directory
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        
        # Validate input text
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")
            
        # Limit extremely long texts to prevent memory issues
        MAX_CHARS = 5000
        if len(text) > MAX_CHARS:
            print(f"Warning: Text exceeds {MAX_CHARS} characters. Truncating to prevent memory issues.")
            text = text[:MAX_CHARS] + "..."
        
        # Generate base filename from text
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"tts_{timestamp}"
        wav_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{base_name}.wav")
        
        # Generate speech
        print(f"\nGenerating speech for: '{text}'")
        print(f"Using voice: {voice_name}")
        
        # Validate voice path
        voice_path = os.path.abspath(os.path.join("voices", f"{voice_name}.pt"))
        if not os.path.exists(voice_path):
            raise FileNotFoundError(f"Voice file not found: {voice_path}")
            
        try:
            generator = model(text, voice=voice_path, speed=1.0, split_pattern=r'\n+')
            
            all_audio = []
            max_segments = 100  # Safety limit for very long texts
            segment_count = 0
            
            for gs, ps, audio in generator:
                segment_count += 1
                if segment_count > max_segments:
                    print(f"Warning: Reached maximum segment limit ({max_segments})")
                    break
                    
                if audio is not None:
                    if isinstance(audio, np.ndarray):
                        audio = torch.from_numpy(audio).float()
                    all_audio.append(audio)
                    print(f"Generated segment: {gs}")
                    if ps:  # Only print phonemes if available
                        print(f"Phonemes: {ps}")
            
            if not all_audio:
                raise Exception("No audio generated")
        except Exception as e:
            raise Exception(f"Error in speech generation: {e}")
            
        # Combine audio segments and save
        if not all_audio:
            raise Exception("No audio segments were generated")
            
        # Handle single segment case without concatenation
        if len(all_audio) == 1:
            final_audio = all_audio[0]
        else:
            try:
                final_audio = torch.cat(all_audio, dim=0)
            except RuntimeError as e:
                raise Exception(f"Failed to concatenate audio segments: {e}")
                
        # Save audio file
        try:
            sf.write(wav_path, final_audio.numpy(), SAMPLE_RATE)
        except Exception as e:
            raise Exception(f"Failed to save audio file: {e}")
        
        # Convert to requested format if needed
        if format != "wav":
            output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{base_name}.{format}")
            return convert_audio(wav_path, output_path, format)
        
        return wav_path
        
    except Exception as e:
        print(f"Error generating speech: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_interface(server_name="0.0.0.0", server_port=7860):
    """Create and launch the Gradio interface."""
    
    # Get available voices
    voices = get_available_voices()
    if not voices:
        print("No voices found! Please check the voices directory.")
        return
        
    # Create interface
    with gr.Blocks(title="Kokoro TTS Generator") as interface:
        gr.Markdown("# Kokoro TTS Generator")
        
        with gr.Row():
            with gr.Column():
                voice = gr.Dropdown(
                    choices=voices,
                    value=voices[0] if voices else None,
                    label="Voice"
                )
                text = gr.Textbox(
                    lines=3,
                    placeholder="Enter text to convert to speech...",
                    label="Text"
                )
                format = gr.Radio(
                    choices=["wav", "mp3", "aac"],
                    value="wav",
                    label="Output Format"
                )
                generate = gr.Button("Generate Speech")
            
            with gr.Column():
                output = gr.Audio(label="Generated Audio")
                
        generate.click(
            fn=generate_tts_with_logs,
            inputs=[voice, text, format],
            outputs=output
        )
        
    # Launch interface
    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=True
    )

def cleanup_resources():
    """Properly clean up resources when the application exits"""
    global model
    
    try:
        print("Cleaning up resources...")
        
        # Clean up model resources
        if model is not None:
            print("Releasing model resources...")
            # Clear voice dictionary to release memory
            if hasattr(model, 'voices'):
                model.voices.clear()
            # Delete model reference
            del model
            model = None
            
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            print("Clearing CUDA cache...")
            torch.cuda.empty_cache()
            
        # Restore original functions
        from models import _cleanup_monkey_patches
        _cleanup_monkey_patches()
        
        print("Cleanup completed")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        import traceback
        traceback.print_exc()

# Register cleanup for normal exit
import atexit
atexit.register(cleanup_resources)

# Register cleanup for signals
import signal
import sys

def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}, shutting down...")
    cleanup_resources()
    sys.exit(0)

# Register for common signals
for sig in [signal.SIGINT, signal.SIGTERM]:
    try:
        signal.signal(sig, signal_handler)
    except (ValueError, AttributeError):
        # Some signals might not be available on all platforms
        pass

if __name__ == "__main__":
    try:
        create_interface()
    finally:
        # Ensure cleanup even if Gradio encounters an error
        cleanup_resources()
