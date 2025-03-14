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
from typing import Union, List, Optional, Tuple
from models import (
    list_available_voices, build_model,
    generate_speech, download_voice_files
)

# Define path type for consistent handling
PathLike = Union[str, Path]

# Configuration validation
def validate_sample_rate(rate: int) -> int:
    """Validate sample rate is within acceptable range"""
    valid_rates = [16000, 22050, 24000, 44100, 48000]
    if rate not in valid_rates:
        print(f"Warning: Unusual sample rate {rate}. Valid rates are {valid_rates}")
        return 24000  # Default to safe value
    return rate

# Global configuration
CONFIG_FILE = Path("tts_config.json")  # Stores user preferences and paths
DEFAULT_OUTPUT_DIR = Path("outputs")    # Directory for generated audio files
SAMPLE_RATE = validate_sample_rate(24000)  # Validated sample rate

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

def convert_audio(input_path: PathLike, output_path: PathLike, format: str) -> Optional[PathLike]:
    """Convert audio to specified format.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        format: Output format ('wav', 'mp3', or 'aac')
        
    Returns:
        Path to output file or None on error
    """
    try:
        # Normalize paths
        input_path = Path(input_path).absolute()
        output_path = Path(output_path).absolute()
        
        # Validate input file
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        # For WAV format, just return the input path
        if format.lower() == "wav":
            return input_path
            
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert format
        audio = AudioSegment.from_wav(str(input_path))
        
        # Select proper format and options
        if format.lower() == "mp3":
            audio.export(str(output_path), format="mp3", bitrate="192k")
        elif format.lower() == "aac":
            audio.export(str(output_path), format="aac", bitrate="192k")
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        # Verify file was created
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise IOError(f"Failed to create {format} file")
            
        return output_path
        
    except (IOError, FileNotFoundError, ValueError) as e:
        print(f"Error converting audio: {type(e).__name__}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error converting audio: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_tts_with_logs(voice_name: str, text: str, format: str) -> Optional[PathLike]:
    """Generate TTS audio with progress logging.
    
    Args:
        voice_name: Name of the voice to use
        text: Text to convert to speech
        format: Output format ('wav', 'mp3', 'aac')
        
    Returns:
        Path to generated audio file or None on error
    """
    global model
    
    try:
        # Initialize model if needed
        if model is None:
            print("Initializing model...")
            model = build_model(None, device)
        
        # Create output directory
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
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
        wav_path = DEFAULT_OUTPUT_DIR / f"{base_name}.wav"
        
        # Generate speech
        print(f"\nGenerating speech for: '{text}'")
        print(f"Using voice: {voice_name}")
        
        # Validate voice path using Path for consistent handling
        voice_path = Path("voices").absolute() / f"{voice_name}.pt"
        if not voice_path.exists():
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
        if format.lower() != "wav":
            output_path = DEFAULT_OUTPUT_DIR / f"{base_name}.{format.lower()}"
            return convert_audio(wav_path, output_path, format.lower())
        
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
            if hasattr(model, 'voices') and model.voices is not None:
                try:
                    voice_count = len(model.voices)
                    for voice_name in list(model.voices.keys()):
                        try:
                            # Release each voice explicitly
                            model.voices[voice_name] = None
                        except:
                            pass
                    model.voices.clear()
                    print(f"Cleared {voice_count} voice references")
                except Exception as ve:
                    print(f"Error clearing voices: {type(ve).__name__}: {ve}")
            
            # Clear model attributes that might hold tensors
            for attr_name in dir(model):
                if not attr_name.startswith('__') and hasattr(model, attr_name):
                    try:
                        attr = getattr(model, attr_name)
                        # Handle specific tensor attributes
                        if isinstance(attr, torch.Tensor):
                            if attr.is_cuda:
                                print(f"Releasing CUDA tensor: {attr_name}")
                                setattr(model, attr_name, None)
                        elif hasattr(attr, 'to'):  # Module or Tensor-like object
                            setattr(model, attr_name, None)
                    except:
                        pass
            
            # Delete model reference
            try:
                del model
                model = None
                print("Model reference deleted")
            except Exception as me:
                print(f"Error deleting model: {type(me).__name__}: {me}")
        
        # Clear CUDA memory explicitly
        if torch.cuda.is_available():
            try:
                # Get initial memory usage 
                try:
                    initial = torch.cuda.memory_allocated()
                    initial_mb = initial / (1024 * 1024)
                    print(f"CUDA memory before cleanup: {initial_mb:.2f} MB")
                except:
                    pass
                
                # Free memory  
                print("Clearing CUDA cache...")
                torch.cuda.empty_cache()
                
                # Force synchronization
                try:
                    torch.cuda.synchronize()
                except:
                    pass
                
                # Get final memory usage
                try:
                    final = torch.cuda.memory_allocated()
                    final_mb = final / (1024 * 1024)
                    freed_mb = (initial - final) / (1024 * 1024)
                    print(f"CUDA memory after cleanup: {final_mb:.2f} MB (freed {freed_mb:.2f} MB)")
                except:
                    pass
            except Exception as ce:
                print(f"Error clearing CUDA memory: {type(ce).__name__}: {ce}")
        
        # Restore original functions
        try:
            from models import _cleanup_monkey_patches
            _cleanup_monkey_patches()
            print("Monkey patches restored")
        except Exception as pe:
            print(f"Error restoring monkey patches: {type(pe).__name__}: {pe}")
        
        # Final garbage collection
        try:
            import gc
            collected = gc.collect()
            print(f"Garbage collection completed: {collected} objects collected")
        except Exception as gce:
            print(f"Error during garbage collection: {type(gce).__name__}: {gce}")
            
        print("Cleanup completed")
        
    except Exception as e:
        print(f"Error during cleanup: {type(e).__name__}: {e}")
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
