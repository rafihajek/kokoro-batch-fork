"""
Batch TTS Script for Kokoro-TTS-Local
-------------------------------------
Processes long text in chunks with progress tracking and graceful interruption.

Usage:
    python batch_tts.py --file input.pdf          # New session from PDF
    python batch_tts.py --file input.txt          # New session from text
    python batch_tts.py --clipboard               # New session from clipboard
    python batch_tts.py --resume                  # Pick session to continue
    python batch_tts.py --list                    # List all sessions
    python batch_tts.py --chunks 5                # Limit to 5 chunks this run
    python batch_tts.py --voice af_bella          # Specify voice
    python batch_tts.py --speed 1.2               # Specify speed
    python batch_tts.py --merge SESSION_ID        # Manually merge a session

Custom Session Names:
    Use --name to give sessions descriptive names (used for output folder AND filename):

    python batch_tts.py --file book.pdf --pages 92-113 --name "Ch7-The-Godfather" --voice af_heart
    python batch_tts.py --file book.pdf --pages 114-127 --name "Ch8-The-Omnivore" --voice af_heart

    Output will be saved to: outputs/batch/Ch7-The-Godfather/Ch7-The-Godfather.wav
    Without --name, output uses a content hash: outputs/batch/abc123def/abc123def.wav

    IMPORTANT: When running in background/non-interactive mode, always specify --voice
    to avoid the interactive voice selection prompt.
"""

import argparse
import hashlib
import json
import re
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

from models import build_model, list_available_voices

# Paths relative to script location (not cwd)
SCRIPT_DIR = Path(__file__).parent.resolve()

# CRITICAL: Change to script directory so Kokoro downloads go here, not cwd
import os
os.chdir(SCRIPT_DIR)

# Constants
CHUNK_SIZE = 2500  # Characters per chunk (safe for Kokoro's limits)
SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"
DEFAULT_SPEED = 1.0

OUTPUT_BASE = SCRIPT_DIR / "outputs" / "batch"
MODEL_PATH = SCRIPT_DIR / "kokoro-v1_0.pth"
VOICES_DIR = SCRIPT_DIR / "voices"

# Graceful shutdown flag
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    if shutdown_requested:
        print("\n\nForce quit. Progress saved up to last completed chunk.")
        sys.exit(1)
    print("\n\nShutdown requested. Finishing current chunk then saving...")
    shutdown_requested = True


# Register signal handler
signal.signal(signal.SIGINT, signal_handler)


# =============================================================================
# Text Loading
# =============================================================================

def get_text_hash(text: str) -> str:
    """Generate a short hash to identify text content."""
    return hashlib.md5(text.encode()).hexdigest()[:8]


def read_text_file(file_path: Path) -> str:
    """Read text from .txt or .md file."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Apply TTS cleaning for all text input
    return clean_for_tts(text)


def normalize_pdf_text(text: str) -> str:
    """
    Clean up PDF text by joining soft line breaks while preserving paragraphs.

    PDF extractors create line breaks at page margins, not sentence ends.
    This joins those mid-sentence breaks while keeping real paragraph breaks.
    """
    # First, normalize different line ending styles
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Protect real paragraph breaks (2+ newlines) with a placeholder
    text = re.sub(r'\n{2,}', '\n\n<<PARA>>\n\n', text)

    # Now process line by line to join soft breaks
    lines = text.split('\n')
    result = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        if line == '<<PARA>>':
            result.append('\n\n')
            continue

        # Check if previous content exists and doesn't end a sentence
        if result and result[-1] not in ['\n\n', '']:
            prev = result[-1].rstrip()
            # Join if previous line doesn't end with sentence-ending punctuation
            # or if current line starts with lowercase (continuation)
            if prev and not prev[-1] in '.!?:"\'"' and not line[0].isupper():
                # Mid-sentence line break - join with space
                result[-1] = prev + ' ' + line
                continue
            elif prev and prev[-1] not in '.!?:"\'"':
                # Previous line doesn't end sentence but this starts with caps
                # Could be a new sentence or continuation - join anyway
                result[-1] = prev + ' ' + line
                continue

        result.append(line)

    # Rebuild text
    final = ' '.join(result)

    # Clean up: normalize whitespace around paragraph breaks
    final = re.sub(r'\s*\n\n\s*', '\n\n', final)
    final = re.sub(r'  +', ' ', final)  # Multiple spaces to single

    return final.strip()


def clean_for_tts(text: str) -> str:
    """
    Clean text for natural TTS output.

    Handles:
    - PDF encoding artifacts (smart quotes, em-dashes, etc.)
    - Page headers/footers
    - Figure/table references
    - Awkward formatting that causes unnatural pauses
    """
    # =========================================================================
    # 1. Fix encoding artifacts (common PDF extraction issues)
    # =========================================================================

    # Smart quotes → straight quotes (using Unicode escapes for reliability)
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # " "
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # ' '

    # Em-dash variants → spoken pause (comma or dash)
    text = text.replace('\u2014', ' - ')  # em-dash —
    text = text.replace('\u2013', ' - ')  # en-dash –
    text = text.replace('\ufffd', '-')    # U+FFFD replacement character

    # Ellipsis
    text = text.replace('\u2026', '...')  # …

    # Other common artifacts (ligatures)
    text = text.replace('\ufb01', 'fi')   # ﬁ
    text = text.replace('\ufb02', 'fl')   # ﬂ
    text = text.replace('\ufb00', 'ff')   # ﬀ
    text = text.replace('\ufb03', 'ffi')  # ﬃ
    text = text.replace('\ufb04', 'ffl')  # ﬄ

    # Symbols that don't speak well
    text = text.replace('\u2122', '')     # ™
    text = text.replace('\u00ae', '')     # ®
    text = text.replace('\u00a9', '')     # ©
    text = text.replace('\u2022', ',')    # • bullet → pause

    # =========================================================================
    # 2. Remove page headers/footers
    # =========================================================================

    # Pattern: "Title: Subtitle N" or "Title N" at start of line where N is page number
    # Common patterns like "VIBE OS: A Sovereign Multi-Agent Runtime 2"
    text = re.sub(
        r'\n*[A-Z][A-Za-z\s:&\-]+\s+\d{1,3}\s*(?=\n|[a-z])',
        '\n',
        text
    )

    # Also catch standalone page numbers
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)

    # =========================================================================
    # 3. Handle figure/table references
    # =========================================================================

    # Remove "Figure N:" prefix but keep the caption
    text = re.sub(r'Figure\s+\d+[.:]\s*', '', text)

    # Remove "Table N:" prefix but keep the caption
    text = re.sub(r'Table\s+\d+[.:]\s*', '', text)

    # Remove inline references like "(see Figure 3)" or "(Table 2)"
    text = re.sub(r'\((?:see\s+)?(?:Figure|Table|Fig\.)\s+\d+\)', '', text, flags=re.IGNORECASE)

    # =========================================================================
    # 4. Clean up table data for speech
    # =========================================================================

    # Tables often become "Col1 Col2 Col3 Val1 Val2 Val3" - hard to fix perfectly
    # but we can add pauses after common table header patterns

    # Add pause after "Note:" which often precedes table footnotes
    text = re.sub(r'(Note:)', r'\1 ', text)

    # =========================================================================
    # 5. General TTS cleanup
    # =========================================================================

    # Hyphenated words at line breaks (from PDF) - rejoin them
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

    # Multiple spaces → single space
    text = re.sub(r'  +', ' ', text)

    # Multiple newlines → single paragraph break
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Section numbers like "2.1" at start of sentence - add a pause
    text = re.sub(r'(\d+\.\d+)\s+([A-Z])', r'\1. \2', text)

    # Fix missing spaces between words (common PDF extraction issue)
    # Pattern: lowercase followed by uppercase = likely missing space
    # e.g., "proposeVIBE" → "propose VIBE", "singlePrimary" → "single Primary"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Numbered lists - add slight pause
    text = re.sub(r'^(\d+\.)\s*', r'\1 ', text, flags=re.MULTILINE)

    # Academic citations like [1], [2,3], etc. - remove for speech
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)

    # URLs - they sound terrible, summarize or remove
    text = re.sub(r'https?://\S+', '', text)

    # Email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # =========================================================================
    # 6. Markdown formatting cleanup
    # =========================================================================

    # Tables - remove entirely (header rows, separator rows, data rows)
    text = re.sub(r'^\|.+\|$', '', text, flags=re.MULTILINE)

    # Headers - remove # symbols, keep text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Bold/italic - keep text, remove markers
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)  # *italic* (not **)
    text = re.sub(r'__(.+?)__', r'\1', text)      # __bold__
    text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'\1', text)  # _italic_ (not __)

    # Links - keep text, drop URL
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Code blocks - remove entirely
    text = re.sub(r'```[\s\S]*?```', '', text)

    # Inline code - keep text
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Horizontal rules - convert to pause
    text = re.sub(r'^-{3,}$', '.', text, flags=re.MULTILINE)
    text = re.sub(r'^\*{3,}$', '.', text, flags=re.MULTILINE)

    # Bullet points - remove markers
    text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)

    # Blockquotes - remove > marker
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

    # Task lists - remove checkbox markers
    text = re.sub(r'^\s*[-*+]\s*\[[x ]\]\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # =========================================================================
    # 7. Final cleanup
    # =========================================================================

    # Clean up any double spaces we created
    text = re.sub(r'  +', ' ', text)

    return text.strip()


def read_pdf(file_path: Path, page_range: Optional[Tuple[int, int]] = None) -> str:
    """
    Read text from PDF file.

    Args:
        file_path: Path to PDF
        page_range: Optional (start, end) page numbers (1-indexed, inclusive)
    """
    try:
        import pypdf
    except ImportError:
        print("ERROR: PDF support requires pypdf")
        print("Install with: pip install pypdf")
        sys.exit(1)

    reader = pypdf.PdfReader(file_path)
    total_pages = len(reader.pages)

    # Determine page range (convert to 0-indexed)
    if page_range:
        start_page = max(0, page_range[0] - 1)  # Convert to 0-indexed
        end_page = min(total_pages, page_range[1])  # end is exclusive in slice
        print(f"Reading pages {page_range[0]}-{page_range[1]} of {total_pages}")
    else:
        start_page = 0
        end_page = total_pages

    text_parts = []
    for i, page in enumerate(reader.pages[start_page:end_page], start=start_page + 1):
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    raw_text = "\n\n".join(text_parts)

    # Normalize: join soft line breaks, keep paragraph breaks
    normalized = normalize_pdf_text(raw_text)

    # Apply TTS-specific cleaning
    return clean_for_tts(normalized)


def read_clipboard() -> str:
    """Read text from clipboard."""
    try:
        import pyperclip
    except ImportError:
        print("ERROR: Clipboard support requires pyperclip")
        print("Install with: pip install pyperclip")
        sys.exit(1)

    text = pyperclip.paste()
    if not text or not text.strip():
        print("ERROR: Clipboard is empty")
        sys.exit(1)
    # Apply TTS cleaning for clipboard input
    return clean_for_tts(text)


def parse_page_range(page_str: str) -> Optional[Tuple[int, int]]:
    """Parse page range string like '9-22' into (9, 22)."""
    if not page_str:
        return None
    match = re.match(r"(\d+)-(\d+)", page_str)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    # Single page
    if page_str.isdigit():
        page = int(page_str)
        return (page, page)
    print(f"Invalid page range: {page_str} (use format: 9-22)")
    return None


def load_text(source: str, page_range: Optional[Tuple[int, int]] = None) -> Tuple[str, str]:
    """
    Load text from specified source.
    Returns (text, source_name).
    """
    if source == "clipboard":
        return read_clipboard(), "clipboard"

    path = Path(source)
    if not path.exists():
        print(f"ERROR: File not found: {source}")
        sys.exit(1)

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path, page_range), path.name
    elif suffix in [".txt", ".md"]:
        return read_text_file(path), path.name
    else:
        # Try as text file
        return read_text_file(path), path.name


# =============================================================================
# Text Chunking
# =============================================================================

def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Normalize whitespace, preserve paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Split on sentence-ending punctuation followed by space
    pattern = r"(?<=[.!?])\s+"
    sentences = re.split(pattern, text)

    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, max_chars: int = CHUNK_SIZE) -> Tuple[List[str], List[str]]:
    """
    Split text into chunks respecting sentence boundaries.

    Returns:
        (chunks, warnings) - list of chunks and any warning messages
    """
    sentences = split_sentences(text)
    chunks = []
    warnings = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # Check if single sentence exceeds limit
        if sentence_len > max_chars:
            warnings.append(
                f"Long sentence ({sentence_len} chars): '{sentence[:50]}...'"
            )

            # Save current chunk first
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            # Split long sentence by commas/semicolons as fallback
            parts = re.split(r"(?<=[,;])\s+", sentence)
            for part in parts:
                if current_length + len(part) + 1 > max_chars and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [part]
                    current_length = len(part)
                else:
                    current_chunk.append(part)
                    current_length += len(part) + 1
            continue

        # Normal case: add sentence to chunk
        if current_length + sentence_len + 1 > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len
        else:
            current_chunk.append(sentence)
            current_length += sentence_len + 1

    # Don't forget last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks, warnings


# =============================================================================
# Session Management
# =============================================================================

def get_session_dir(session_id: str) -> Path:
    """Get the output directory for a session."""
    return OUTPUT_BASE / session_id


def load_session(session_id: str) -> Optional[dict]:
    """Load session state from progress.json."""
    progress_file = get_session_dir(session_id) / "progress.json"
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_session(session_id: str, data: dict):
    """Save session state to progress.json."""
    session_dir = get_session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    data["updated"] = datetime.now().isoformat()
    progress_file = session_dir / "progress.json"

    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def create_session(text: str, source_name: str, voice: str, speed: float, custom_name: str = None) -> Tuple[str, dict]:
    """
    Create a new session for the given text.
    Returns (session_id, session_data).

    Args:
        custom_name: Optional name for the session folder (e.g., "Ch13-The-Hunt")
    """
    text_hash = get_text_hash(text)
    chunks, warnings = chunk_text(text)

    # Use custom name if provided, otherwise use hash
    session_id = custom_name if custom_name else text_hash

    # Show warnings for long sentences
    if warnings:
        print("\n⚠️  Long sentence warnings:")
        for w in warnings[:5]:  # Show first 5
            print(f"   {w}")
        if len(warnings) > 5:
            print(f"   ... and {len(warnings) - 5} more")

    session_data = {
        "source_hash": text_hash,
        "source_name": source_name,
        "custom_name": custom_name,
        "chunks_total": len(chunks),
        "chunks_done": 0,
        "voice": voice,
        "speed": speed,
        "created": datetime.now().isoformat(),
    }

    save_session(session_id, session_data)

    return session_id, session_data


def list_all_sessions() -> List[Tuple[str, dict]]:
    """List all saved sessions."""
    sessions = []
    if OUTPUT_BASE.exists():
        for session_dir in OUTPUT_BASE.iterdir():
            if session_dir.is_dir():
                data = load_session(session_dir.name)
                if data:
                    sessions.append((session_dir.name, data))

    # Sort by last updated
    sessions.sort(key=lambda x: x[1].get("updated", ""), reverse=True)
    return sessions


def print_sessions(sessions: List[Tuple[str, dict]]):
    """Display sessions in a formatted table."""
    if not sessions:
        print("\nNo saved sessions found.")
        return

    print("\n=== Saved Sessions ===")
    print(f"{'#':<3} {'ID':<10} {'Progress':<12} {'Voice':<15} {'Source':<20}")
    print("-" * 65)

    for i, (sid, data) in enumerate(sessions, 1):
        done = data.get("chunks_done", 0)
        total = data.get("chunks_total", 0)
        pct = (done / total * 100) if total > 0 else 0
        status = "COMPLETE" if done >= total else f"{done}/{total} ({pct:.0f}%)"
        voice = data.get("voice", "?")
        source = data.get("source_name", "?")[:20]

        print(f"{i:<3} {sid:<10} {status:<12} {voice:<15} {source:<20}")


# =============================================================================
# Audio Generation
# =============================================================================

def select_voice(voices: List[str], default: str = DEFAULT_VOICE) -> str:
    """Interactive voice selection."""
    print("\nAvailable voices:")
    for i, v in enumerate(voices, 1):
        marker = " (default)" if v == default else ""
        print(f"  {i}. {v}{marker}")

    while True:
        choice = input(f"\nSelect voice [1-{len(voices)}] or Enter for '{default}': ").strip()
        if not choice:
            return default
        try:
            idx = int(choice)
            if 1 <= idx <= len(voices):
                return voices[idx - 1]
        except ValueError:
            pass
        print("Invalid choice.")


def generate_chunk(model, text: str, voice_path: Path, speed: float) -> Optional[np.ndarray]:
    """Generate audio for a single chunk."""
    all_audio = []

    try:
        generator = model(text, voice=voice_path, speed=speed, split_pattern=r"\n+")
        for gs, ps, audio in generator:
            if audio is not None:
                if isinstance(audio, torch.Tensor):
                    all_audio.append(audio)
                else:
                    all_audio.append(torch.from_numpy(audio).float())

        if all_audio:
            if len(all_audio) == 1:
                return all_audio[0].numpy()
            return torch.cat(all_audio, dim=0).numpy()

    except Exception as e:
        print(f"\nError generating audio: {e}")

    return None


def process_chunks(
    model,
    chunks: List[str],
    session_id: str,
    session_data: dict,
    max_chunks: Optional[int] = None,
) -> int:
    """
    Process chunks with progress tracking.
    Returns number of chunks processed this session.
    """
    global shutdown_requested

    voice = session_data["voice"]
    speed = session_data["speed"]
    start_idx = session_data["chunks_done"]
    total = session_data["chunks_total"]

    voice_path = VOICES_DIR / f"{voice}.pt"
    if not voice_path.exists():
        print(f"ERROR: Voice file not found: {voice_path}")
        return 0

    session_dir = get_session_dir(session_id)

    # Calculate how many to process
    remaining = total - start_idx
    to_process = min(max_chunks, remaining) if max_chunks else remaining

    print(f"\nProcessing {to_process} chunks (starting from {start_idx + 1}/{total})")
    print(f"Voice: {voice} | Speed: {speed}x")
    print("-" * 50)

    processed = 0
    for i in range(start_idx, start_idx + to_process):
        if shutdown_requested:
            print("\nStopping after this chunk...")
            break

        chunk_num = i + 1
        chunk_text = chunks[i]

        # Progress indicator
        print(f"\n[{chunk_num}/{total}] Generating...")
        if len(chunk_text) > 80:
            print(f"   Preview: {chunk_text[:80]}...")
        else:
            print(f"   Text: {chunk_text}")

        # Generate audio
        audio = generate_chunk(model, chunk_text, voice_path, speed)

        if audio is None:
            print(f"   FAILED - stopping")
            break

        # Save immediately
        output_file = session_dir / f"chunk_{chunk_num:04d}.wav"
        sf.write(str(output_file), audio, SAMPLE_RATE)
        print(f"   Saved: {output_file.name}")

        # Update progress
        session_data["chunks_done"] = chunk_num
        save_session(session_id, session_data)
        processed += 1

    return processed


# =============================================================================
# Merge Functionality
# =============================================================================

def merge_chunks(session_id: str) -> bool:
    """Merge all chunks into final.wav."""
    session_dir = get_session_dir(session_id)
    chunk_files = sorted(session_dir.glob("chunk_*.wav"))

    if not chunk_files:
        print("No chunk files found to merge.")
        return False

    print(f"\nMerging {len(chunk_files)} chunks...")

    all_audio = []
    for chunk_file in tqdm(chunk_files, desc="Reading chunks"):
        audio, sr = sf.read(str(chunk_file))
        if sr != SAMPLE_RATE:
            print(f"Warning: {chunk_file.name} has different sample rate ({sr})")
        all_audio.append(audio)

    # Concatenate
    final_audio = np.concatenate(all_audio)

    # Name the final file after the session (uses --name if provided, else hash)
    final_path = session_dir / f"{session_id}.wav"

    sf.write(str(final_path), final_audio, SAMPLE_RATE)
    duration_mins = len(final_audio) / SAMPLE_RATE / 60

    print(f"\n[OK] Merged to: {final_path}")
    print(f"  Duration: {duration_mins:.1f} minutes")

    return True


# =============================================================================
# Main Entry Points
# =============================================================================

def cmd_new_session(args):
    """Start a new session from file or clipboard."""
    # Parse page range if provided
    page_range = None
    if hasattr(args, 'pages') and args.pages:
        page_range = parse_page_range(args.pages)

    # Load text
    if args.clipboard:
        text, source_name = load_text("clipboard")
    elif args.file:
        text, source_name = load_text(args.file, page_range)
    else:
        print("Specify --file or --clipboard")
        return

    # Get custom name if provided
    custom_name = getattr(args, 'name', None)

    print(f"\nLoaded: {source_name}")
    print(f"Characters: {len(text):,}")

    # Check if session already exists (by hash or custom name)
    text_hash = get_text_hash(text)
    session_key = custom_name if custom_name else text_hash
    existing = load_session(session_key)

    if existing:
        done = existing.get("chunks_done", 0)
        total = existing.get("chunks_total", 0)
        print(f"\n⚠️  Session already exists: [{session_key}]")
        print(f"   Progress: {done}/{total} chunks")

        choice = input("Resume existing (r) or start fresh (f)? [r/f]: ").strip().lower()
        if choice == "r":
            return cmd_resume_specific(session_key, text, args)
        elif choice != "f":
            print("Cancelled.")
            return

    # Chunk the text
    chunks, warnings = chunk_text(text)
    print(f"Chunks: {len(chunks)}")

    # Select voice
    if args.voice:
        voice = args.voice
    else:
        voices = list_available_voices()
        voice = select_voice(voices)

    speed = args.speed

    # Create session with optional custom name
    session_id, session_data = create_session(text, source_name, voice, speed, custom_name)
    print(f"\nSession created: [{session_id}]")

    # Initialize model
    print("\nInitializing model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = build_model(MODEL_PATH.resolve(), device)

    # Process
    processed = process_chunks(model, chunks, session_id, session_data, args.chunks)

    # Summary
    done = session_data["chunks_done"]
    total = session_data["chunks_total"]

    print("\n" + "=" * 50)
    print(f"Session [{session_id}]: {done}/{total} chunks complete")

    if done >= total:
        print("\n[OK] All chunks complete! Merging...")
        merge_chunks(session_id)
    else:
        print(f"\nRemaining: {total - done} chunks")
        print(f"Resume with: python batch_tts.py --resume")


def cmd_resume_specific(session_id: str, text: str, args):
    """Resume a specific session with provided text."""
    session_data = load_session(session_id)
    if not session_data:
        print(f"Session not found: {session_id}")
        return

    # Verify hash
    new_hash = get_text_hash(text)
    if new_hash != session_data.get("source_hash"):
        print("\n⚠️  WARNING: Text content has changed!")
        print("   This may cause chunk misalignment.")
        if input("   Continue anyway? [y/n]: ").strip().lower() != "y":
            return

    # Re-chunk
    chunks, _ = chunk_text(text)

    if len(chunks) != session_data.get("chunks_total"):
        print(f"\n⚠️  Chunk count changed: {session_data['chunks_total']} -> {len(chunks)}")
        if input("   Continue anyway? [y/n]: ").strip().lower() != "y":
            return
        session_data["chunks_total"] = len(chunks)

    # Override voice/speed if specified
    if args.voice:
        session_data["voice"] = args.voice
    if args.speed != DEFAULT_SPEED:
        session_data["speed"] = args.speed

    # Initialize model
    print("\nInitializing model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = build_model(MODEL_PATH.resolve(), device)

    # Process
    processed = process_chunks(model, chunks, session_id, session_data, args.chunks)

    # Summary
    done = session_data["chunks_done"]
    total = session_data["chunks_total"]

    print("\n" + "=" * 50)
    print(f"Session [{session_id}]: {done}/{total} chunks complete")

    if done >= total:
        print("\n[OK] All chunks complete! Merging...")
        merge_chunks(session_id)
    else:
        print(f"\nRemaining: {total - done} chunks")


def cmd_resume(args):
    """Interactive resume from saved sessions."""
    sessions = list_all_sessions()
    print_sessions(sessions)

    if not sessions:
        return

    choice = input("\nEnter session # to resume (or 'q' to quit): ").strip()
    if choice.lower() == "q":
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(sessions):
            session_id, session_data = sessions[idx]

            # Need to re-load the source text
            print(f"\nResuming session [{session_id}]")
            print(f"Original source: {session_data.get('source_name', 'unknown')}")
            source = input("Enter file path (or 'clipboard'): ").strip()

            text, _ = load_text(source)
            cmd_resume_specific(session_id, text, args)
        else:
            print("Invalid selection.")
    except ValueError:
        print("Invalid input.")


def cmd_list(args):
    """List all sessions."""
    sessions = list_all_sessions()
    print_sessions(sessions)


def cmd_merge(args):
    """Manually merge a session."""
    if args.merge:
        merge_chunks(args.merge)
    else:
        # Interactive selection
        sessions = list_all_sessions()
        print_sessions(sessions)

        if not sessions:
            return

        choice = input("\nEnter session # to merge: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                merge_chunks(sessions[idx][0])
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input.")


# =============================================================================
# CLI Parser
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch TTS with progress tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_tts.py --file book.pdf --pages 9-22   # Pages 9-22 only
  python batch_tts.py --file book.pdf --chunks 10    # Process 10 chunks from PDF
  python batch_tts.py --clipboard                    # From clipboard
  python batch_tts.py --resume                       # Resume a session
  python batch_tts.py --list                         # List all sessions
  python batch_tts.py --merge abc123                 # Merge session abc123
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--file", "-f", help="Input file (PDF, txt, md)")
    input_group.add_argument(
        "--clipboard", "-c", action="store_true", help="Read from clipboard"
    )
    input_group.add_argument(
        "--resume", "-r", action="store_true", help="Resume a saved session"
    )
    input_group.add_argument(
        "--list", "-l", action="store_true", help="List all sessions"
    )
    input_group.add_argument("--merge", "-m", nargs="?", const="", help="Merge chunks for a session")

    # Processing options
    parser.add_argument(
        "--chunks", "-n", type=int, help="Max chunks to process this session"
    )
    parser.add_argument("--voice", "-v", help="Voice to use")
    parser.add_argument(
        "--speed", "-s", type=float, default=DEFAULT_SPEED, help="Speech speed (0.1-3.0)"
    )
    parser.add_argument(
        "--pages", "-p", help="Page range for PDFs (e.g., '9-22' or '5')"
    )
    parser.add_argument(
        "--name", help="Custom name for the session (used in output folder)"
    )

    args = parser.parse_args()

    # Route to appropriate command
    if args.list:
        cmd_list(args)
    elif args.resume:
        cmd_resume(args)
    elif args.merge is not None:
        cmd_merge(args)
    elif args.file or args.clipboard:
        cmd_new_session(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
