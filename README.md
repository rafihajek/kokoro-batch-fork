# Kokoro TTS Local

A local implementation of the Kokoro Text-to-Speech model, featuring dynamic module loading, automatic dependency management, and a web interface.

## Features

- Local text-to-speech synthesis using the Kokoro-82M model
- Multiple voice support with easy voice selection (54 voices available across 8 languages)
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

**Alternative Installation (Simplified):**
For a simpler setup, you can also install the official Kokoro package directly:
```bash
pip install kokoro>=0.9.2 soundfile
apt-get install espeak-ng  # On Linux
# or brew install espeak  # On macOS
```

3. (Optional) For GPU acceleration, install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

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

The system includes 54 different voices across 8 languages:

### 🇺🇸 American English (20 voices)
**Language code: 'a'**

**Female voices (af_*):**
- af_heart: ❤️ Premium quality voice (Grade A)
- af_alloy: Clear and professional (Grade C)
- af_aoede: Smooth and melodic (Grade C+)
- af_bella: 🔥 Warm and friendly (Grade A-)
- af_jessica: Natural and engaging (Grade D)
- af_kore: Bright and energetic (Grade C+)
- af_nicole: 🎧 Professional and articulate (Grade B-)
- af_nova: Modern and dynamic (Grade C)
- af_river: Soft and flowing (Grade D)
- af_sarah: Casual and approachable (Grade C+)
- af_sky: Light and airy (Grade C-)

**Male voices (am_*):**
- am_adam: Strong and confident (Grade F+)
- am_echo: Resonant and clear (Grade D)
- am_eric: Professional and authoritative (Grade D)
- am_fenrir: Deep and powerful (Grade C+)
- am_liam: Friendly and conversational (Grade D)
- am_michael: Warm and trustworthy (Grade C+)
- am_onyx: Rich and sophisticated (Grade D)
- am_puck: Playful and energetic (Grade C+)
- am_santa: Holiday-themed voice (Grade D-)

### 🇬🇧 British English (8 voices)
**Language code: 'b'**

**Female voices (bf_*):**
- bf_alice: Refined and elegant (Grade D)
- bf_emma: Warm and professional (Grade B-)
- bf_isabella: Sophisticated and clear (Grade C)
- bf_lily: Sweet and gentle (Grade D)

**Male voices (bm_*):**
- bm_daniel: Polished and professional (Grade D)
- bm_fable: Storytelling and engaging (Grade C)
- bm_george: Classic British accent (Grade C)
- bm_lewis: Modern British accent (Grade D+)

### 🇯🇵 Japanese (5 voices)
**Language code: 'j'**

**Female voices (jf_*):**
- jf_alpha: Standard Japanese female (Grade C+)
- jf_gongitsune: Based on classic tale (Grade C)
- jf_nezumi: Mouse bride tale voice (Grade C-)
- jf_tebukuro: Glove story voice (Grade C)

**Male voices (jm_*):**
- jm_kumo: Spider thread tale voice (Grade C-)

### 🇨🇳 Mandarin Chinese (8 voices)
**Language code: 'z'**

**Female voices (zf_*):**
- zf_xiaobei: Chinese female voice (Grade D)
- zf_xiaoni: Chinese female voice (Grade D)
- zf_xiaoxiao: Chinese female voice (Grade D)
- zf_xiaoyi: Chinese female voice (Grade D)

**Male voices (zm_*):**
- zm_yunjian: Chinese male voice (Grade D)
- zm_yunxi: Chinese male voice (Grade D)
- zm_yunxia: Chinese male voice (Grade D)
- zm_yunyang: Chinese male voice (Grade D)

### 🇪🇸 Spanish (3 voices)
**Language code: 'e'**

**Female voices (ef_*):**
- ef_dora: Spanish female voice

**Male voices (em_*):**
- em_alex: Spanish male voice
- em_santa: Spanish holiday voice

### 🇫🇷 French (1 voice)
**Language code: 'f'**

**Female voices (ff_*):**
- ff_siwis: French female voice (Grade B-)

### 🇮🇳 Hindi (4 voices)
**Language code: 'h'**

**Female voices (hf_*):**
- hf_alpha: Hindi female voice (Grade C)
- hf_beta: Hindi female voice (Grade C)

**Male voices (hm_*):**
- hm_omega: Hindi male voice (Grade C)
- hm_psi: Hindi male voice (Grade C)

### 🇮🇹 Italian (2 voices)
**Language code: 'i'**

**Female voices (if_*):**
- if_sara: Italian female voice (Grade C)

**Male voices (im_*):**
- im_nicola: Italian male voice (Grade C)

### 🇧🇷 Brazilian Portuguese (3 voices)
**Language code: 'p'**

**Female voices (pf_*):**
- pf_dora: Portuguese female voice

**Male voices (pm_*):**
- pm_alex: Portuguese male voice
- pm_santa: Portuguese holiday voice

**Note:** Quality grades (A to F) indicate the overall quality based on training data quality and duration. Higher grades generally produce better speech quality.

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
- Available voices: 54 voices across 8 languages
- Languages: American English ('a'), British English ('b'), Japanese ('j'), Mandarin Chinese ('z'), Spanish ('e'), French ('f'), Hindi ('h'), Italian ('i'), Brazilian Portuguese ('p')
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

     # For CUDA 12.6
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

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