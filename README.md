# Kokoro TTS Local

A local implementation of the Kokoro Text-to-Speech model, featuring dynamic module loading, automatic dependency management, and a web interface.

## Features

- Local text-to-speech synthesis using the Kokoro-82M model
- Multiple voice support with easy voice selection (31 voices available)
- Automatic model and voice downloading from Hugging Face
- Phoneme output support and visualization
- Interactive CLI and web interface
- Voice listing functionality
- Cross-platform support (Windows, Linux, macOS)
- Real-time generation progress display
- Multiple output formats (WAV, MP3, AAC)

## Prerequisites

- Python 3.8 or higher
- FFmpeg (optional, for MP3/AAC conversion)
- CUDA-compatible GPU (optional, for faster generation)
- Git (for version control and package management)

## Installation

1. Clone the repository and create a Python virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For GPU acceleration, install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.8 (for RTX 50-series cards)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

You can verify CUDA support is enabled with:
```python
import torch
print(torch.cuda.is_available())  # Should print True if CUDA is available
```

The system will automatically download required models and voice files on first run.

## Usage

You can use either the command-line interface or the web interface:

### Command Line Interface

Run the interactive CLI:
```bash
python tts_demo.py
```

The CLI provides an interactive menu with the following options:
1. List available voices - Shows all available voice options
2. Generate speech - Interactive process to:
   - Select a voice from the numbered list
   - Enter text to convert to speech
   - Adjust speech speed (0.5-2.0)
3. Exit - Quit the program

Example session:
```
=== Kokoro TTS Menu ===
1. List available voices
2. Generate speech
3. Exit
Select an option (1-3): 2

Available voices:
1. af_alloy
2. af_aoede
3. af_bella
...

Select a voice number (or press Enter for default 'af_bella'): 3

Enter the text you want to convert to speech
(or press Enter for default text)
> Hello, world!

Enter speech speed (0.5-2.0, default 1.0): 1.2

Generating speech for: 'Hello, world!'
Using voice: af_bella
Speed: 1.2x
...
```

### Web Interface

For a more user-friendly experience, launch the web interface:

```bash
python gradio_interface.py
```

Then open your browser to the URL shown in the console (typically http://localhost:7860).

The web interface provides:
- Easy voice selection from a dropdown menu
- Text input field with examples
- Real-time generation progress
- Audio playback in the browser
- Multiple output format options (WAV, MP3, AAC)
- Download options for generated audio

## Available Voices

The system includes 31 different voices across various categories:

### American English Voices
- Female (af_*):
  - af_alloy: Alloy - Clear and professional
  - af_aoede: Aoede - Smooth and melodic
  - af_bella: Bella - Warm and friendly
  - af_jessica: Jessica - Natural and engaging
  - af_kore: Kore - Bright and energetic
  - af_nicole: Nicole - Professional and articulate
  - af_nova: Nova - Modern and dynamic
  - af_river: River - Soft and flowing
  - af_sarah: Sarah - Casual and approachable
  - af_sky: Sky - Light and airy

- Male (am_*):
  - am_adam: Adam - Strong and confident
  - am_echo: Echo - Resonant and clear
  - am_eric: Eric - Professional and authoritative
  - am_fenrir: Fenrir - Deep and powerful
  - am_liam: Liam - Friendly and conversational
  - am_michael: Michael - Warm and trustworthy
  - am_onyx: Onyx - Rich and sophisticated
  - am_puck: Puck - Playful and energetic

### British English Voices
- Female (bf_*):
  - bf_alice: Alice - Refined and elegant
  - bf_emma: Emma - Warm and professional
  - bf_isabella: Isabella - Sophisticated and clear
  - bf_lily: Lily - Sweet and gentle

- Male (bm_*):
  - bm_daniel: Daniel - Polished and professional
  - bm_fable: Fable - Storytelling and engaging
  - bm_george: George - Classic British accent
  - bm_lewis: Lewis - Modern British accent

### Special Voices
- French Female (ff_*):
  - ff_siwis: Siwis - French accent

- High-pitched Voices:
  - Female (hf_*):
    - hf_alpha: Alpha - Higher female pitch
    - hf_beta: Beta - Alternative high female pitch
  - Male (hm_*):
    - hm_omega: Omega - Higher male pitch
    - hm_psi: Psi - Alternative high male pitch

## Project Structure

```
.
├── .cache/                 # Cache directory for downloaded models
│   └── huggingface/       # Hugging Face model cache
├── .git/                   # Git repository data
├── .gitignore             # Git ignore rules
├── __pycache__/           # Python cache files
├── voices/                # Voice model files (downloaded on demand)
│   └── *.pt              # Individual voice files
├── venv/                  # Python virtual environment
├── outputs/               # Generated audio files directory
├── LICENSE                # Apache 2.0 License file
├── README.md             # Project documentation
├── models.py             # Core TTS model implementation
├── gradio_interface.py   # Web interface implementation
├── config.json           # Model configuration file
├── requirements.txt      # Python dependencies
└── tts_demo.py          # CLI implementation
```

## Model Information

The project uses the latest Kokoro model from Hugging Face:
- Repository: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- Model file: `kokoro-v1_0.pth` (downloaded automatically)
- Sample rate: 24kHz
- Voice files: Located in the `voices/` directory (downloaded automatically)
- Available voices: 31 voices across multiple categories
- Languages: American English ('a'), British English ('b')
- Model size: 82M parameters

## Troubleshooting

Common issues and solutions:

1. **Model Download Issues**
   - Ensure stable internet connection
   - Check Hugging Face is accessible
   - Verify sufficient disk space
   - Try clearing the `.cache/huggingface` directory

2. **CUDA/GPU Issues**
   - Verify CUDA installation with `nvidia-smi`
   - Update GPU drivers
   - Install PyTorch with CUDA support using the appropriate command:
     ```bash
     # For CUDA 11.8
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

     # For CUDA 12.1
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

     # For CUDA 12.8 (for RTX 50-series cards)
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
     ```
   - Verify CUDA is available in PyTorch:
     ```python
     import torch
     print(torch.cuda.is_available())  # Should print True
     ```
   - Fall back to CPU if needed

3. **Audio Output Issues**
   - Check system audio settings
   - Verify output directory permissions
   - Install FFmpeg for MP3/AAC support
   - Try different output formats

4. **Voice File Issues**
   - Delete and let system redownload voice files
   - Check `voices/` directory permissions
   - Verify voice file integrity
   - Try using a different voice

5. **Web Interface Issues**
   - Check port 7860 availability
   - Try different browser
   - Clear browser cache
   - Check network firewall settings

For any other issues:
1. Check the console output for error messages
2. Verify all prerequisites are installed
3. Ensure virtual environment is activated
4. Check system resource usage
5. Try reinstalling dependencies

## Contributing

Feel free to contribute by:
1. Opening issues for bugs or feature requests
2. Submitting pull requests with improvements
3. Helping with documentation
4. Testing different voices and reporting issues
5. Suggesting new features or optimizations
6. Testing on different platforms and reporting results

## License

Apache 2.0 - See LICENSE file for details