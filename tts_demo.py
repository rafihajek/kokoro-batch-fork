import torch
from typing import Optional, Tuple, List
from models import build_model, generate_speech, list_available_voices
from tqdm.auto import tqdm
import soundfile as sf
from pathlib import Path
import numpy as np
import time
import os

# Constants
SAMPLE_RATE = 24000
DEFAULT_MODEL_PATH = 'kokoro-v1_0.pth'
DEFAULT_OUTPUT_FILE = 'output.wav'
DEFAULT_LANGUAGE = 'a'  # 'a' for American English, 'b' for British English
DEFAULT_TEXT = "Hello, welcome to this text-to-speech test."

# Configure tqdm for better Windows console support
tqdm.monitor_interval = 0

def print_menu():
    """Print the main menu options."""
    print("\n=== Kokoro TTS Menu ===")
    print("1. List available voices")
    print("2. Generate speech")
    print("3. Exit")
    return input("Select an option (1-3): ").strip()

def select_voice(voices: List[str]) -> str:
    """Interactive voice selection."""
    print("\nAvailable voices:")
    for i, voice in enumerate(voices, 1):
        print(f"{i}. {voice}")
    
    while True:
        try:
            choice = input("\nSelect a voice number (or press Enter for default 'af_bella'): ").strip()
            if not choice:
                return "af_bella"
            choice = int(choice)
            if 1 <= choice <= len(voices):
                return voices[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def get_text_input() -> str:
    """Get text input from user."""
    print("\nEnter the text you want to convert to speech")
    print("(or press Enter for default text)")
    text = input("> ").strip()
    return text if text else DEFAULT_TEXT

def get_speed() -> float:
    """Get speech speed from user."""
    while True:
        try:
            speed = input("\nEnter speech speed (0.5-2.0, default 1.0): ").strip()
            if not speed:
                return 1.0
            speed = float(speed)
            if 0.5 <= speed <= 2.0:
                return speed
            print("Speed must be between 0.5 and 2.0")
        except ValueError:
            print("Please enter a valid number.")

def save_audio_with_retry(audio_data: np.ndarray, sample_rate: int, output_path: Path, max_retries: int = 3, retry_delay: float = 1.0) -> bool:
    """
    Attempt to save audio data to file with retry logic.
    Returns True if successful, False otherwise.
    """
    for attempt in range(max_retries):
        try:
            sf.write(output_path, audio_data, sample_rate)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\nFailed to save audio (attempt {attempt + 1}/{max_retries})")
                print("The output file might be in use by another program (e.g., media player).")
                print(f"Please close any programs that might be using '{output_path}'")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"\nError: Could not save audio after {max_retries} attempts.")
                print(f"Please ensure '{output_path}' is not open in any other program and try again.")
                return False
    return False

def main() -> None:
    try:
        # Set up device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Build model
        print("\nInitializing model...")
        with tqdm(total=1, desc="Building model") as pbar:
            model = build_model(DEFAULT_MODEL_PATH, device)
            pbar.update(1)
        
        while True:
            choice = print_menu()
            
            if choice == "1":
                # List voices
                voices = list_available_voices()
                print("\nAvailable voices:")
                for voice in voices:
                    print(f"- {voice}")
                    
            elif choice == "2":
                # Generate speech
                voices = list_available_voices()
                if not voices:
                    print("No voices found! Please check the voices directory.")
                    continue
                
                # Get user inputs
                voice = select_voice(voices)
                text = get_text_input()
                speed = get_speed()
                
                print(f"\nGenerating speech for: '{text}'")
                print(f"Using voice: {voice}")
                print(f"Speed: {speed}x")
                
                # Generate speech
                all_audio = []
                
                # Use the improved generate_speech function that handles unusual words
                print(f"\nProcessing text for TTS generation...")
                audio, phonemes = generate_speech(model, text, voice, DEFAULT_LANGUAGE, device, speed)
                
                if audio is not None:
                    all_audio.append(audio)
                    print(f"\nGenerated speech successfully")
                    print(f"Phonemes: {phonemes}")
                else:
                    print("\nFalling back to direct model call...")
                    generator = model(text, voice=f"voices/{voice}.pt", speed=speed, split_pattern=r'\n+')
                    
                    with tqdm(desc="Generating speech") as pbar:
                        for gs, ps, audio in generator:
                            if audio is not None:
                                if isinstance(audio, np.ndarray):
                                    audio = torch.from_numpy(audio).float()
                                all_audio.append(audio)
                                print(f"\nGenerated segment: {gs}")
                                print(f"Phonemes: {ps}")
                                pbar.update(1)
                
                # Save audio
                if all_audio:
                    final_audio = torch.cat(all_audio, dim=0)
                    output_path = Path(DEFAULT_OUTPUT_FILE)
                    if save_audio_with_retry(final_audio.numpy(), SAMPLE_RATE, output_path):
                        print(f"\nAudio saved to {output_path.absolute()}")
                else:
                    print("Error: Failed to generate audio")
                    
            elif choice == "3":
                print("\nGoodbye!")
                break
                
            else:
                print("\nInvalid choice. Please try again.")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 