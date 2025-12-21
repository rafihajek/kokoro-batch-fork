# Kokoro TTS Local

A local text-to-speech system using the Kokoro-82M model. Runs entirely offline after initial setup.

## Quick Start

```bash
# Install
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Generate audio from a PDF
python batch_tts.py --file book.pdf --pages 1-20 --voice af_heart --name "Ch1-Introduction"
```

---

## CLI Usage (Recommended)

**`batch_tts.py` is the intended way to use this tool.** It handles long documents, tracks progress, and can resume interrupted sessions.

### Basic Commands

```bash
# Convert a PDF (specific pages)
python batch_tts.py --file document.pdf --pages 50-75 --voice af_heart

# Convert a text file
python batch_tts.py --file notes.txt --voice af_bella

# Convert from clipboard
python batch_tts.py --clipboard --voice af_alloy

# Resume an interrupted session
python batch_tts.py --resume

# List all sessions
python batch_tts.py --list
```

### Custom Session Names

Use `--name` to give sessions descriptive names (used for output folder):

```bash
python batch_tts.py --file book.pdf --pages 92-113 --name "Ch7-The-Godfather" --voice af_heart
python batch_tts.py --file book.pdf --pages 114-127 --name "Ch8-The-Omnivore" --voice af_heart
```

Output will be saved to: `outputs/batch/Ch7-The-Godfather/final.wav`

Without `--name`, output uses a content hash: `outputs/batch/abc123def/final.wav`

### Running in Background / Non-Interactive Mode

When running from scripts or automation, always specify `--voice` to avoid the interactive voice selection prompt:

```bash
# Good - explicit voice
python batch_tts.py --file book.pdf --pages 1-20 --voice af_heart

# Bad - will prompt for voice selection
python batch_tts.py --file book.pdf --pages 1-20
```

### All CLI Options

| Option | Description |
|--------|-------------|
| `--file`, `-f` | Input file (PDF, txt, md) |
| `--pages`, `-p` | Page range for PDFs (e.g., `9-22` or `5`) |
| `--name` | Custom session name for output folder |
| `--voice`, `-v` | Voice to use (see voices below) |
| `--speed`, `-s` | Speech speed (0.5-2.0, default 1.0) |
| `--chunks`, `-n` | Max chunks to process this run |
| `--clipboard`, `-c` | Read text from clipboard |
| `--resume`, `-r` | Resume a saved session |
| `--list`, `-l` | List all sessions |
| `--merge`, `-m` | Merge chunks for a session |

---

## How It Works

1. **Text Extraction** - PDFs are parsed, soft line-breaks rejoined, encoding artifacts cleaned
2. **Chunking** - Text is split into ~2500 character chunks at sentence boundaries
3. **Generation** - Each chunk is converted to audio using Kokoro
4. **Merging** - All chunks are concatenated into a single `final.wav`

### Session Management

Long documents are processed in chunks. Progress is saved after each chunk, so you can:
- **Interrupt anytime** with Ctrl+C (progress is saved)
- **Resume later** with `--resume`
- **Run multiple sessions** in parallel (each gets its own output folder)

Output structure:
```
outputs/batch/
├── Ch7-The-Godfather/      # Named session
│   ├── progress.json       # Session state
│   ├── chunk_0001.wav
│   ├── chunk_0002.wav
│   └── final.wav           # Merged output
└── abc123def/              # Hash-named session
    ├── progress.json
    └── final.wav
```

---

## Available Voices

### American English (Recommended)
| Voice | Quality | Notes |
|-------|---------|-------|
| `af_heart` | A | Best overall quality |
| `af_bella` | A- | Warm and friendly |
| `af_nicole` | B- | Professional, articulate |
| `af_alloy` | C | Clear, professional |
| `am_fenrir` | C+ | Deep male voice |
| `am_michael` | C+ | Warm male voice |

### British English
| Voice | Quality |
|-------|---------|
| `bf_emma` | B- |
| `bm_george` | C |

### Other Languages
- Japanese: `jf_alpha`, `jm_kumo`
- Chinese: `zf_xiaoxiao`, `zm_yunxi`
- Spanish: `ef_dora`, `em_alex`
- French: `ff_siwis`

Full list: 54 voices across 8 languages. Run `python batch_tts.py` without arguments to see all.

---

## Installation

### Prerequisites
- Python 3.8+
- ~2GB disk space for models
- FFmpeg (optional, for MP3 conversion)

### Setup

```bash
# Clone the repository
git clone https://github.com/lostmyalias/kokoro-batch-fork.git
cd kokoro-batch-fork

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Acceleration (Optional)

For faster generation, install PyTorch with CUDA:

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is available:
```python
import torch
print(torch.cuda.is_available())  # Should print True
```

### First Run

On first run, the system will automatically download:
- Model file (~500MB): `kokoro-v1_0.pth`
- Voice files (~1MB each): stored in `voices/`

After initial download, the tool works **completely offline**.

---

## Text Cleaning

`batch_tts.py` automatically cleans text for natural speech:

| Issue | Before | After |
|-------|--------|-------|
| Smart quotes | `"Hello"` | `"Hello"` |
| Em-dashes | `word—word` | `word - word` |
| Figure refs | `Figure 1: The...` | `The...` |
| Word spacing | `proposeVIBE` | `propose VIBE` |
| Hyphenation | `sophis-ticated` | `sophisticated` |
| Citations | `as shown [1,2]` | `as shown` |

---

## Performance

| Hardware | Speed | Notes |
|----------|-------|-------|
| CPU only | ~15x realtime | 24 min audio in ~90 sec |
| CUDA GPU | ~50-100x realtime | Much faster |

Running multiple sessions in parallel on CPU will share resources (not 2x faster, more like 1.3x).

---

## Other Interfaces

While `batch_tts.py` is recommended, other interfaces exist:

- **`tts_demo.py`** - Simple interactive CLI for single text inputs
- **`gradio_interface.py`** - Web interface at http://localhost:7860

---

## Troubleshooting

### "EOF when reading a line" error
You're running in non-interactive mode without specifying `--voice`. Always use `--voice af_heart` (or another voice) when running from scripts.

### Model download fails
- Check internet connection
- Try deleting `.cache/huggingface` and rerunning
- Ensure sufficient disk space (~2GB)

### Audio sounds robotic/choppy
- Try a different voice (some are higher quality than others)
- Check that PDF text extraction is clean with `--pages 1-2` first

### CUDA not detected
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## License

Apache 2.0 - See LICENSE file for details.

Based on [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M).
