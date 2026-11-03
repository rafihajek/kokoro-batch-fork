"""
audiobook.py - Turn an EPUB into a chaptered M4B audiobook with Kokoro TTS.
---------------------------------------------------------------------------

Pipeline (each stage can be re-run; finished work is reused):

  1. EXTRACT    EPUB -> list of chapters -> cleaned text you can review
                (outputs/audiobooks/<book>/text/*.txt, plan.json)
  2. SYNTHESIZE text -> one WAV per chapter (resumable; Ctrl+C is safe)
  3. BUILD      WAVs -> single .m4b with chapter markers, cover and tags,
                then verify it with ffprobe

Usage:
    python audiobook.py book.epub --extract-only      # stage 1 only: review text first
    python audiobook.py book.epub                     # all stages (resumes automatically)
    python audiobook.py book.epub --only 2,3          # sample build: just chapters #2 and #3
    python audiobook.py book.epub --lexicon lexicons/rwanda.tsv

Editing the chapter plan:
    After --extract-only, edit outputs/audiobooks/<book>/plan.json to rename
    chapters ("title") or drop them ("include": false). Your edits are kept on
    the next run. Delete plan.json to regenerate it from scratch.

Why the M4B works in both BookPlayer (iOS) and Voice (Android):
    ffmpeg's ipod/mp4 muxer writes chapters twice - a QuickTime chapter track
    (read by BookPlayer/AVFoundation) and a Nero 'chpl' atom (the only format
    Voice reads). Do NOT pass -movflags +disable_chpl or Voice loses chapters.
"""

import argparse
import hashlib
import json
import os
import posixpath
import re
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, Comment, NavigableString, ProcessingInstruction, Tag

SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_BASE = SCRIPT_DIR / "outputs" / "audiobooks"

SAMPLE_RATE = 24000          # Kokoro's native output rate
DEFAULT_VOICE = "af_heart"   # Kokoro's highest-graded English voice

# Silence (seconds) inserted between pieces of text. Tuned for audiobook pacing.
PAUSE = {
    "line": 0.25,        # between lines of verse / dialogue split by <br>
    "paragraph": 0.45,   # between normal paragraphs
    "block": 0.8,        # before/after block quotes, epigraphs, notes
    "section": 1.4,      # scene break inside a chapter
    "after_title": 1.0,  # after the spoken chapter title
    "chapter_end": 1.8,  # tail of each chapter, before the next one starts
}

# =============================================================================
# Data types
# =============================================================================

@dataclass
class Para:
    """One unit of spoken text plus the pause that should FOLLOW it."""
    text: str
    kind: str = "paragraph"   # paragraph | quote | attribution | note | line
    pause_after: float = PAUSE["paragraph"]


@dataclass
class Chapter:
    num: int                  # 1-based position in the plan
    title: str                # shown in the player's chapter list
    spoken_title: str         # read aloud at the start ("" = none)
    files: List[str]          # EPUB document hrefs that make up this chapter
    include: bool = True
    paras: List[Para] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return f"{self.num:02d}_" + re.sub(r"[^A-Za-z0-9]+", "-", self.title).strip("-")[:40]


# =============================================================================
# Stage 1a: read the EPUB container (stdlib zipfile + BeautifulSoup)
# =============================================================================

class Epub:
    """Minimal EPUB 2/3 reader: metadata, spine order, TOC labels, cover, documents."""

    def __init__(self, path: Path):
        self.path = path
        self.zip = zipfile.ZipFile(path)
        # META-INF/container.xml tells us where the OPF package file lives
        container = self._soup("META-INF/container.xml", "xml")
        self.opf_path = container.find("rootfile")["full-path"]
        self.base = posixpath.dirname(self.opf_path)
        opf = self._soup(self.opf_path, "xml")

        # Manifest: id -> (href relative to zip root, media-type, properties)
        self.manifest: Dict[str, Tuple[str, str, str]] = {}
        for item in opf.find("manifest").find_all("item"):
            href = self._join(item["href"])
            self.manifest[item["id"]] = (href, item.get("media-type", ""), item.get("properties", ""))

        # Spine: the reading order (this is the source of truth, NOT the TOC -
        # some books leave real content like a prologue out of the TOC)
        self.spine = [self.manifest[ref["idref"]][0]
                      for ref in opf.find("spine").find_all("itemref")
                      if ref["idref"] in self.manifest]

        # Metadata (Dublin Core)
        def dc(name):
            el = opf.find(name) or opf.find("dc:" + name)
            return el.get_text(strip=True) if el else ""
        self.title = dc("title")
        self.author = dc("creator")
        self.publisher = dc("publisher")
        date = dc("date")
        self.year = date[:4] if date[:4].isdigit() else ""

        self.cover_href = self._find_cover(opf)
        self.toc = self._read_toc(opf)   # document href -> TOC label

    # -- helpers --------------------------------------------------------------
    def _join(self, href: str) -> str:
        href = href.split("#")[0]
        return posixpath.normpath(posixpath.join(self.base, href)) if self.base else href

    def read(self, href: str) -> bytes:
        return self.zip.read(href)

    def _soup(self, href: str, parser: str = "lxml") -> BeautifulSoup:
        return BeautifulSoup(self.read(href), parser)

    def _find_cover(self, opf) -> Optional[str]:
        # EPUB3: <item properties="cover-image">
        for href, mtype, props in self.manifest.values():
            if "cover-image" in props.split():
                return href
        # EPUB2: <meta name="cover" content="ITEM_ID">
        meta = opf.find("meta", attrs={"name": "cover"})
        if meta and meta.get("content") in self.manifest:
            href, mtype, _ = self.manifest[meta["content"]]
            if mtype.startswith("image/"):
                return href
        # Fallback: any image with "cover" in its name
        for href, mtype, _ in self.manifest.values():
            if mtype.startswith("image/") and "cover" in href.lower():
                return href
        return None

    def _read_toc(self, opf) -> Dict[str, str]:
        """Map document href -> first TOC label pointing at it (NCX or EPUB3 nav)."""
        toc: Dict[str, str] = {}
        ncx_id = opf.find("spine").get("toc")
        ncx = self.manifest.get(ncx_id) if ncx_id else None
        if not ncx:
            ncx = next((v for v in self.manifest.values() if v[1] == "application/x-dtbncx+xml"), None)
        if ncx:
            soup = self._soup(ncx[0], "xml")
            ncx_base = posixpath.dirname(ncx[0])
            for nav in soup.find_all("navPoint"):
                label = nav.find("navLabel").get_text(" ", strip=True)
                src = posixpath.normpath(posixpath.join(ncx_base, nav.find("content")["src"].split("#")[0]))
                toc.setdefault(src, label)
            return toc
        # EPUB3 navigation document
        nav_doc = next((v for v in self.manifest.values() if "nav" in v[2].split()), None)
        if nav_doc:
            soup = self._soup(nav_doc[0])
            nav_base = posixpath.dirname(nav_doc[0])
            for a in soup.select("nav a[href]"):
                src = posixpath.normpath(posixpath.join(nav_base, a["href"].split("#")[0]))
                toc.setdefault(src, a.get_text(" ", strip=True))
        return toc


# =============================================================================
# Stage 1b: HTML -> paragraphs
# =============================================================================

BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "section"}
HEADING_CLASSES = {"title-chapter", "title-part", "title", "chapter-title", "chapter-number", "part-title"}
QUOTE_CLASSES = {"verse", "blocktext", "p-blocktext", "p-indent-blocktext", "p-br-blocktext",
                 "epigraph", "extract", "poem", "poetry"}
ATTRIBUTION_CLASSES = {"source", "attribution", "epigraph-source", "author"}
BREAK_CLASSES = {"cb", "page-break", "scene-break", "space-break", "ornament", "dinkus"}

# Labels / filenames of documents that should never be read aloud
SKIP_LABEL_RE = re.compile(
    r"^(cover|title ?page|half ?title|praise|copyright( page| notice)?|"
    r"(table of )?contents|notes?|endnotes|footnotes|index|also by|about the (author|publisher)|"
    r"newsletter|sign ?up|other books?)\b", re.I)
SKIP_FILE_RE = re.compile(r"(cov|cover|tp|title|toc|cop|copyright|notice|pra|praise|ftn|notes|index|adcard)\d*\.", re.I)

NUMBER_WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
                "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen",
                "Eighteen", "Nineteen", "Twenty"]


def classes(el: Tag) -> set:
    return set(el.get("class") or [])


def is_leaf_block(el: Tag) -> bool:
    """A block element with no block-level children = one paragraph."""
    return el.name in BLOCK_TAGS and not any(
        isinstance(c, Tag) and c.name in BLOCK_TAGS for c in el.descendants)


def collect_footnotes(epub: Epub) -> Dict[str, str]:
    """
    Find footnote bodies anywhere in the book.
    Returns {"<href>#<id>": note text} so in-text links can be resolved.
    """
    notes = {}
    for href in epub.spine:
        soup = epub._soup(href)
        for fn in soup.select(".footnote, aside[epub\\:type~=footnote], aside[epub\\:type~=endnote], "
                              "li[epub\\:type~=endnote], li[epub\\:type~=footnote]"):
            # The note's anchor id is on the element itself or its first <a id=...>
            anchor = fn if fn.get("id") else fn.find(id=True)
            if not anchor:
                continue
            body = fn.__copy__()
            # Drop the back-link number ("1") at the start of the note
            for a in body.find_all("a"):
                if a.get_text(strip=True).isdigit():
                    a.decompose()
            text = norm_ws(body.get_text(" "))
            if text:
                notes[f"{href}#{anchor['id']}"] = text
    return notes


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def doc_to_paras(epub: Epub, href: str, footnotes: Dict[str, str]) -> Tuple[List[Para], List[str], int]:
    """
    Convert one XHTML document's <body> to Para objects.
    Returns (paras, warnings, source_word_count) - the word count lets us
    verify nothing was silently dropped.
    """
    soup = epub._soup(href)
    body = soup.body or soup
    warnings = []

    # Remove things that must never be spoken: page-number processing
    # instructions (<?dp n="22" folio="16"?>), scripts/styles, images
    # (find_all(string=...) skips PIs in recent bs4, so walk the tree directly)
    for node in [n for n in body.descendants if isinstance(n, (ProcessingInstruction, Comment))]:
        node.extract()
    for tag in body.find_all(["script", "style", "img", "svg", "sup"]):
        # <sup> is almost always a footnote marker; footnote links handled below
        if tag.name == "sup" and tag.find("a"):
            continue
        if tag.name == "sup" and not tag.get_text(strip=True).isdigit():
            continue  # keep real superscripts like "1st"
        tag.decompose()

    source_words = len(body.get_text(" ").split())

    # Replace footnote links with nothing, remembering which note to read
    # after the paragraph that contains them
    pending_notes: Dict[int, List[str]] = {}
    for a in body.find_all("a", href=True):
        target = a["href"]
        full = posixpath.normpath(posixpath.join(posixpath.dirname(href), target.split("#")[0])) \
            + ("#" + target.split("#")[1] if "#" in target else "")
        if full in footnotes:
            block = a.find_parent(lambda t: is_leaf_block(t))
            if block is not None:
                pending_notes.setdefault(id(block), []).append(footnotes[full])
            (a.find_parent("sup") or a).decompose()

    paras: List[Para] = []
    for el in body.find_all(lambda t: isinstance(t, Tag) and is_leaf_block(t)):
        cls = classes(el)
        if cls & HEADING_CLASSES or el.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            continue  # chapter headings are replaced by our own spoken title
        if cls & BREAK_CLASSES:
            if paras:
                paras[-1].pause_after = max(paras[-1].pause_after, PAUSE["section"])
            continue

        # Inside-paragraph <br> = separate lines (verse, dialogue, addresses)
        for br in el.find_all("br"):
            br.replace_with("\n")
        lines = [norm_ws(x) for x in el.get_text().split("\n")]
        lines = [x for x in lines if x]
        if not lines:
            continue

        in_quote = bool(cls & QUOTE_CLASSES) or el.find_parent(class_=lambda c: c in QUOTE_CLASSES) is not None
        if cls & ATTRIBUTION_CLASSES:
            # "—RALPH ELLISON / Invisible Man" -> "Ralph Ellison, Invisible Man."
            text = ", ".join(lines)
            paras.append(Para(text, "attribution", PAUSE["block"]))
            if len(paras) >= 2:
                paras[-2].pause_after = PAUSE["line"]
            continue

        kind = "quote" if in_quote else "paragraph"
        for i, line in enumerate(lines):
            last = i == len(lines) - 1
            pause = (PAUSE["block"] if in_quote else PAUSE["paragraph"]) if last else PAUSE["line"]
            paras.append(Para(line, kind if last else "line", pause))

        # Footnote: read it right after the paragraph it belongs to
        for note in pending_notes.get(id(el), []):
            paras[-1].pause_after = PAUSE["block"]
            paras.append(Para("Note: " + note, "note", PAUSE["block"]))

    # Warn about text sitting directly inside non-leaf containers (would be lost)
    for el in body.find_all(lambda t: isinstance(t, Tag) and t.name in BLOCK_TAGS and not is_leaf_block(t)):
        stray = "".join(s for s in el.find_all(string=True, recursive=False) if isinstance(s, NavigableString)).strip()
        if stray:
            warnings.append(f"{href}: text outside a paragraph was skipped: {stray[:60]!r}")
    return paras, warnings, source_words


# =============================================================================
# Stage 1c: text normalization for natural speech
# =============================================================================

MONTHS = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
# Words that stay ALL CAPS because they are spoken as letters (or are fine as-is)
KEEP_CAPS = {"I", "A", "OK", "TV", "US", "UN", "USA", "UK", "FBI", "CIA", "BBC", "CNN", "NATO", "AIDS", "II", "III", "IV"}


def ordinal(n: int) -> str:
    """15 -> '15th', 22 -> '22nd' (misaki reads '15th' as 'fifteenth')."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def smart_case(word: str) -> str:
    """Title-case an ALL-CAPS word unless it's a short acronym we should keep."""
    core = re.sub(r"[^A-Za-z]", "", word)
    if core in KEEP_CAPS or len(core) <= 1:
        return word
    return word[0] + word[1:].lower() if word[:1].isalpha() else word[:2] + word[2:].lower()


def normalize(text: str, kind: str, first_in_chapter: bool) -> str:
    """Make printed text read naturally by a TTS model."""
    t = text

    # Leading dashes on attributions ("—RALPH ELLISON") and stray brackets
    if kind == "attribution":
        t = t.lstrip("—–- ")
        # ALL CAPS names are spelled out letter by letter by the G2P -> fix
        t = " ".join(smart_case(w) if w.isupper() else w for w in t.split())

    # Book-style small-caps lead-in: "IN THE PROVINCE of Kibungo" -> "In the province of Kibungo"
    m = re.match(r"^((?:[A-Z][A-Z'’.\-]*\s+)+)(?=[a-z])", t)
    if m and (first_in_chapter or len(m.group(1).split()) >= 2):
        lead = m.group(1).split()
        fixed = [lead[0] if lead[0] in KEEP_CAPS else lead[0][0] + lead[0][1:].lower()]
        fixed += [w if w in KEEP_CAPS else w.lower() for w in lead[1:]]
        t = " ".join(fixed) + " " + t[m.end():]

    # Dates: "April 15" -> "April 15th"; also "December 24 and 28" -> "24th and 28th"
    def _day(mm):
        return f"{mm.group(1)} {ordinal(int(mm.group(2)))}"
    # (?!:|,\d) avoids times ("May 5:30") and thousands ("May 1,000")
    t = re.sub(MONTHS + r" (\d{1,2})\b(?!:|,\d)", _day, t)
    t = re.sub(r"(\d{1,2}(?:st|nd|rd|th)) (and|or|to|through|until) (\d{1,2})\b(?!:|,\d|\s?(?:percent|%))",
               lambda mm: f"{mm.group(1)} {mm.group(2)} {ordinal(int(mm.group(3)))}", t)

    # Year ranges: "1959–62" -> "1959 to 1962" (the dash version is read as
    # "nineteen hundred fifty-nine sixty-two"); handles century rollover 1999–00
    def _years(mm):
        start, end2 = mm.group(1), mm.group(2)
        century = int(start[:2]) + (1 if int(end2) < int(start[2:]) else 0)
        return f"{start} to {century}{end2}"
    t = re.sub(r"\b(1[5-9]\d\d|20\d\d)[–\-](\d{2})\b", _years, t)
    t = re.sub(r"\b(1[5-9]\d\d|20\d\d)[–\-](1[5-9]\d\d|20\d\d)\b", r"\1 to \2", t)

    # Ligatures and odd spaces from typesetting
    for a, b in {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
                 " ": " ", " ": " ", " ": " ", "​": "", "­": ""}.items():
        t = t.replace(a, b)

    # "!!!!!!" or "! ! ! !" -> "!" (the model makes odd noises on long runs)
    t = re.sub(r"(?:\s*!){2,}", "!", t)
    t = re.sub(r"(?:\s*\?){2,}", "?", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)

    t = norm_ws(t)
    if t[:1].islower():                      # "for my parents" -> "For my parents"
        t = t[0].upper() + t[1:]
    # Every spoken unit must end in punctuation, or Kokoro may clip the last word
    if t and t[-1] not in ".!?…\"'”’)":
        t += "."
    return t


def load_lexicon(path: Optional[Path]) -> List[Tuple[re.Pattern, str]]:
    """
    Pronunciation overrides: TSV lines of  word<TAB>phonemes  (misaki/Kokoro IPA).
    Lines starting with # are comments. Matching is case-sensitive, whole word.
    """
    rules = []
    if not path:
        return rules
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        word, ipa = [x.strip() for x in line.split("\t", 1)]
        rules.append((re.compile(rf"(?<![\w\[/]){re.escape(word)}(?![\w\]])"), f"[{word}](/{ipa}/)"))
    # Longest words first so "Interahamwe" wins over a shorter overlapping entry
    rules.sort(key=lambda r: -len(r[1]))
    return rules


def apply_lexicon(text: str, rules) -> str:
    for pat, repl in rules:
        text = pat.sub(lambda _m: repl, text)
    return text


# =============================================================================
# Stage 1d: build the chapter plan
# =============================================================================

def chapter_title(label: str, first_text: str) -> Tuple[str, str]:
    """TOC label -> (display title, spoken title)."""
    label = norm_ws(label)
    if label.isdigit():                       # "7" -> "Chapter 7"
        n = int(label)
        spoken = f"Chapter {NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else n}."
        return f"Chapter {n}", spoken
    if label.isupper():                       # "ACKNOWLEDGMENTS" -> "Acknowledgments"
        label = label.title()
    return label, label.rstrip(".") + "."


def build_plan(epub: Epub) -> List[Chapter]:
    """Decide which spine documents become which chapters."""
    chapters: List[Chapter] = []
    for href in epub.spine:
        label = epub.toc.get(href, "")
        fname = posixpath.basename(href)
        text = norm_ws(epub._soup(href).body.get_text(" ")) if epub._soup(href).body else ""
        if not text:
            continue
        if (label and SKIP_LABEL_RE.match(label)) or (not label and SKIP_FILE_RE.search(fname)):
            continue
        words = len(text.split())
        if not label and words < 60 and chapters:
            # Tiny untitled page (dedication etc.) -> attach to the previous chapter
            chapters[-1].files.append(href)
            continue
        if not label:
            if words < 60:
                # Tiny untitled page before any chapter (e.g. dedication): goes into the opening
                if not chapters or chapters[-1].title != "Opening Credits":
                    chapters.append(Chapter(0, "Opening Credits", "", []))
                chapters[-1].files.append(href)
                continue
            title, spoken = "Untitled Section", ""   # rename in plan.json
        else:
            title, spoken = chapter_title(label, text)
        chapters.append(Chapter(0, title, spoken, [href]))

    # Always start with spoken opening credits (title + author)
    if not chapters or chapters[0].title != "Opening Credits":
        chapters.insert(0, Chapter(0, "Opening Credits", "", []))
    for i, ch in enumerate(chapters, 1):
        ch.num = i
    return chapters


def plan_to_json(chapters: List[Chapter]) -> list:
    return [{"num": c.num, "title": c.title, "spoken_title": c.spoken_title,
             "files": c.files, "include": c.include} for c in chapters]


def load_or_create_plan(epub: Epub, work: Path) -> List[Chapter]:
    plan_file = work / "plan.json"
    if plan_file.exists():
        data = json.loads(plan_file.read_text(encoding="utf-8"))
        chapters = [Chapter(d["num"], d["title"], d.get("spoken_title", ""), d["files"], d.get("include", True))
                    for d in data["chapters"]]
        print(f"Using existing chapter plan: {plan_file}")
    else:
        chapters = build_plan(epub)
        plan_file.write_text(json.dumps({"book": epub.title, "chapters": plan_to_json(chapters)},
                                        indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Created chapter plan: {plan_file}")
    return chapters


def extract(epub: Epub, chapters: List[Chapter], lexicon, work: Path) -> List[str]:
    """Fill chapter.paras with normalized text and write review files."""
    footnotes = collect_footnotes(epub)
    warnings: List[str] = []
    text_dir = work / "text"
    if text_dir.exists():
        shutil.rmtree(text_dir)
    text_dir.mkdir(parents=True)

    for ch in chapters:
        raw: List[Para] = []
        if ch.title == "Opening Credits":
            by = f" By {epub.author}." if epub.author else ""
            raw.append(Para(f"{epub.title}.{by}", "paragraph", PAUSE["section"]))
        for href in ch.files:
            paras, warns, src_words = doc_to_paras(epub, href, footnotes)
            warnings += warns
            out_words = sum(len(p.text.split()) for p in paras)
            # Headings are intentionally dropped, so allow a small difference
            if src_words - out_words > 25:
                warnings.append(f"{href}: {src_words - out_words} source words not in output - check it")
            raw += paras
        if raw:
            raw[-1].pause_after = PAUSE["chapter_end"]

        ch.paras = []
        for i, p in enumerate(raw):
            t = normalize(p.text, p.kind, first_in_chapter=(i == 0))
            if t:
                ch.paras.append(Para(apply_lexicon(t, lexicon), p.kind, p.pause_after))

        # Review file: exactly what will be spoken, one unit per line
        lines = [f"# {ch.num}. {ch.title}   ({'INCLUDED' if ch.include else 'SKIPPED'})"]
        if ch.spoken_title:
            lines.append(f"[title] {ch.spoken_title}")
        lines += [f"[{p.kind}] {p.text}" for p in ch.paras]
        (text_dir / f"{ch.slug}.txt").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return warnings


# =============================================================================
# Stage 2: synthesis
# =============================================================================

stop_requested = False


def _on_sigint(signum, frame):
    global stop_requested
    if stop_requested:
        print("\nForce quit. Finished chapters are kept; the current one restarts next run.")
        sys.exit(1)
    stop_requested = True
    print("\nStopping after the current paragraph... (Ctrl+C again to quit now)")


# Words ending in "." that do NOT end a sentence
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "mt", "ft", "gen", "col", "lt", "sgt", "capt", "maj", "cmdr",
                 "adm", "gov", "sen", "rep", "pres", "rev", "fr", "msgr", "prof", "jr", "sr", "vs", "etc",
                 "no", "vol", "ch", "pp", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
                 "oct", "nov", "dec", "inc", "co", "corp", "ltd", "dept", "approx", "e.g", "i.e"}
_BOUNDARY = re.compile(r"[.!?…][\"'”’)\]]*\s+(?=[\"'“‘(\[]?[A-Z0-9])")


def split_sentences(text: str) -> List[str]:
    """
    Split a paragraph into sentences without breaking on "Mr.", "St.",
    initials ("J. Smith") or dotted acronyms ("U.N. troops").
    Closing quotes stay attached to their sentence.
    """
    out, start = [], 0
    for m in _BOUNDARY.finditer(text):
        before = text[start:m.start() + 1]
        last_word = re.findall(r"([A-Za-z.]+)\.$", before)
        if last_word:
            w = last_word[0].lower()
            # abbreviation, single initial ("J."), or dotted acronym ("U.N.")
            if w in ABBREVIATIONS or len(w) == 1 or re.fullmatch(r"(?:[a-z]\.)+[a-z]", w):
                continue
        end = m.end()
        out.append(text[start:end].strip())
        start = end
    if text[start:].strip():
        out.append(text[start:].strip())
    return out


def chapter_hash(ch: Chapter, voice: str, speed: float) -> str:
    """Fingerprint of everything that affects a chapter's audio."""
    blob = json.dumps([ch.spoken_title, [(p.text, p.pause_after) for p in ch.paras], voice, speed])
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def synthesize(chapters: List[Chapter], work: Path, voice: str, speed: float, device: str) -> bool:
    """Generate one WAV per chapter. Returns False if interrupted."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    wav_dir = work / "wav"
    wav_dir.mkdir(exist_ok=True)
    state_file = work / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}

    todo = [c for c in chapters if c.include and
            not ((wav_dir / f"{c.slug}.wav").exists() and state.get(c.slug) == chapter_hash(c, voice, speed))]
    if not todo:
        print("All chapters already synthesized.")
        return True

    total_words = sum(len(p.text.split()) for c in todo for p in c.paras)
    print(f"\nSynthesizing {len(todo)} chapter(s), {total_words:,} words | voice={voice} speed={speed} device={device}")
    pipeline = KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M", device=device)
    signal.signal(signal.SIGINT, _on_sigint)

    def silence(sec: float):
        return np.zeros(int(sec * SAMPLE_RATE), dtype=np.float32)

    def speak(text: str):
        # One Kokoro call per sentence: keeps each call well under the
        # 510-token limit and avoids clipped endings on long passages.
        # (Kokoro still auto-splits a single very long sentence at commas.)
        parts = []
        for sentence in split_sentences(text):
            for r in pipeline(sentence, voice=voice, speed=speed, split_pattern=None):
                if r.audio is not None:
                    parts.append(r.audio.cpu().numpy())
        return np.concatenate(parts) if parts else None

    words_done, t0 = 0, time.time()
    for ch in todo:
        out = wav_dir / f"{ch.slug}.wav"
        tmp = out.with_suffix(".partial.wav")
        n_words = sum(len(p.text.split()) for p in ch.paras)
        print(f"\n[{ch.num:02d}] {ch.title}  ({n_words:,} words)")
        failed = 0
        with sf.SoundFile(tmp, "w", SAMPLE_RATE, 1, "PCM_16") as f:   # stream to disk, low RAM
            f.write(silence(0.4))
            if ch.spoken_title:
                a = speak(ch.spoken_title)
                if a is not None:
                    f.write(a)
                f.write(silence(PAUSE["after_title"]))
            for i, p in enumerate(ch.paras, 1):
                if stop_requested:
                    break
                a = speak(p.text)
                if a is None or len(a) == 0:
                    failed += 1
                    print(f"   !! no audio for: {p.text[:70]}")
                else:
                    f.write(a)
                f.write(silence(p.pause_after))
                words_done += len(p.text.split())
                if i % 10 == 0 or i == len(ch.paras):
                    rate = words_done / max(time.time() - t0, 1e-6)
                    eta = (total_words - words_done) / max(rate, 1e-6)
                    print(f"   {i}/{len(ch.paras)} paragraphs | {rate * 60:,.0f} words/min | ETA {eta / 60:.0f} min",
                          end="\r", flush=True)
        if stop_requested:
            tmp.unlink(missing_ok=True)
            print("\nInterrupted. Run the same command again to resume.")
            return False
        tmp.replace(out)                                  # atomic: never leaves half a chapter
        state[ch.slug] = chapter_hash(ch, voice, speed)
        state_file.write_text(json.dumps(state, indent=2))
        dur = sf.info(str(out)).duration
        print(f"\n   saved {out.name}  {dur / 60:.1f} min" + (f"  ({failed} empty paragraphs!)" if failed else ""))
    return True


# =============================================================================
# Stage 3: build + verify the M4B
# =============================================================================

def ffmeta_escape(s: str) -> str:
    return re.sub(r"([=;#\\\n])", r"\\\1", s)


def build_m4b(epub: Epub, chapters: List[Chapter], work: Path, out_path: Path,
              voice: str, bitrate: str) -> Path:
    import soundfile as sf

    wav_dir = work / "wav"
    chosen = [c for c in chapters if c.include]
    missing = [c.slug for c in chosen if not (wav_dir / f"{c.slug}.wav").exists()]
    if missing:
        sys.exit(f"Missing audio for: {missing}. Run synthesis first.")

    # Chapter start/end in ms from exact sample counts (no ffprobe rounding drift)
    meta = [";FFMETADATA1",
            f"title={ffmeta_escape(epub.title)}", f"album={ffmeta_escape(epub.title)}",
            f"artist={ffmeta_escape(epub.author)}", f"album_artist={ffmeta_escape(epub.author)}",
            f"composer={ffmeta_escape('Kokoro TTS (' + voice + ')')}",   # narrator field in many players
            "genre=Audiobook", f"date={epub.year}",
            f"comment={ffmeta_escape('Generated from EPUB with kokoro-batch-fork audiobook.py')}"]
    concat = []
    pos = 0
    for c in chosen:
        wav = wav_dir / f"{c.slug}.wav"
        frames = sf.info(str(wav)).frames
        start, end = round(pos * 1000 / SAMPLE_RATE), round((pos + frames) * 1000 / SAMPLE_RATE)
        meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={end}", f"title={ffmeta_escape(c.title)}"]
        # concat demuxer quoting: wrap in single quotes, escape embedded quotes
        concat.append("file '" + str(wav.resolve()).replace("'", "'\\''") + "'")
        pos += frames
    (work / "chapters.ffmeta").write_text("\n".join(meta) + "\n", encoding="utf-8")
    (work / "concat.txt").write_text("\n".join(concat) + "\n", encoding="utf-8")

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "concat", "-safe", "0", "-i", str(work / "concat.txt"),
           "-i", str(work / "chapters.ffmeta")]
    cover = None
    if epub.cover_href:
        cover = work / ("cover" + Path(epub.cover_href).suffix.lower())
        cover.write_bytes(epub.read(epub.cover_href))
        cmd += ["-i", str(cover)]
    # One AAC encode over the whole book = no gaps/clicks at chapter joins.
    # aac_at = Apple's AudioToolbox encoder (best AAC available on macOS);
    # fall back to ffmpeg's built-in aac elsewhere.
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    codec = "aac_at" if " aac_at " in encoders else "aac"
    cmd += ["-map", "0:a", "-c:a", codec, "-b:a", bitrate, "-ac", "1",
            "-map_metadata", "1", "-map_chapters", "1"]
    if cover:
        cmd += ["-map", "2:v", "-c:v", "copy", "-disposition:v", "attached_pic"]
    cmd += ["-movflags", "+faststart", "-f", "ipod", str(out_path)]

    print(f"\nEncoding {out_path.name} ({codec} {bitrate} mono, {pos / SAMPLE_RATE / 3600:.2f} h)...")
    subprocess.run(cmd, check=True)
    verify_m4b(out_path, chosen, pos / SAMPLE_RATE)
    return out_path


def verify_m4b(path: Path, chosen: List[Chapter], expected_sec: float):
    """Sanity-check the result the way a player would see it."""
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_chapters", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout)
    chaps = probe.get("chapters", [])
    dur = float(probe["format"]["duration"])
    has_cover = any(s.get("disposition", {}).get("attached_pic") for s in probe["streams"])
    raw = path.read_bytes()
    problems = []
    if len(chaps) != len(chosen):
        problems.append(f"chapter count {len(chaps)} != planned {len(chosen)}")
    if abs(dur - expected_sec) > 2:
        problems.append(f"duration {dur:.0f}s differs from expected {expected_sec:.0f}s")
    if b"chpl" not in raw:
        problems.append("no Nero 'chpl' chapter atom (Voice on Android will show no chapters)")
    if b"chap" not in raw:
        problems.append("no QuickTime chapter track (BookPlayer may show no chapters)")

    print(f"\n=== {path.name} ===")
    print(f"Size {path.stat().st_size / 1e6:.0f} MB | {dur / 3600:.2f} h | {len(chaps)} chapters | cover: {'yes' if has_cover else 'no'}")
    for c, pc in zip(chosen, chaps):
        secs = float(pc["end_time"]) - float(pc["start_time"])
        words = sum(len(p.text.split()) for p in c.paras)
        wpm = words / (secs / 60) if secs else 0
        flag = "  <-- check: unusual pace" if words > 50 and not 110 <= wpm <= 210 else ""
        print(f"  {pc['tags'].get('title', '?'):28} {float(pc['start_time']) / 60:7.1f} min  {secs / 60:6.1f} min  {wpm:4.0f} wpm{flag}")
    if problems:
        print("PROBLEMS:\n  - " + "\n  - ".join(problems))
    else:
        print("OK: chapters (QuickTime + Nero), duration and cover all check out.")


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="EPUB -> chaptered M4B audiobook with Kokoro TTS",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("epub", type=Path)
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--device", default="cpu", help="cpu (default; as fast as mps on M1) | mps | cuda")
    ap.add_argument("--bitrate", default="64k", help="AAC bitrate; 64k mono is transparent for speech")
    ap.add_argument("--lexicon", type=Path, help="TSV of word<TAB>phonemes pronunciation overrides")
    ap.add_argument("--work", type=Path, help="work dir (default outputs/audiobooks/<book>)")
    ap.add_argument("--extract-only", action="store_true", help="stop after writing review text")
    ap.add_argument("--only", help="comma-separated chapter numbers for a sample build, e.g. 2,4")
    ap.add_argument("-o", "--output", type=Path, help="output .m4b path")
    args = ap.parse_args()

    epub = Epub(args.epub)
    # Short, lowercase work-dir name from the first 5 title words: "we-wish-to-inform-you"
    words = re.findall(r"[A-Za-z0-9]+", epub.title or args.epub.stem)
    slug = "-".join(words[:5]).lower() or "book"
    work = args.work or OUTPUT_BASE / slug
    work.mkdir(parents=True, exist_ok=True)
    print(f"Book: {epub.title} - {epub.author} ({epub.year})\nWork dir: {work}")

    chapters = load_or_create_plan(epub, work)
    lexicon = load_lexicon(args.lexicon)
    warnings = extract(epub, chapters, lexicon, work)

    print(f"\n{'#':>3}  {'Chapter':30} {'Words':>7}  Source files")
    for c in chapters:
        words = sum(len(p.text.split()) for p in c.paras)
        mark = "" if c.include else "  (skipped)"
        print(f"{c.num:>3}  {c.title[:30]:30} {words:>7,}  {', '.join(posixpath.basename(f) for f in c.files)}{mark}")
    total = sum(len(p.text.split()) for c in chapters if c.include for p in c.paras)
    print(f"Total: {total:,} words (~{total / 155 / 60:.1f} h of audio)")
    for w in warnings:
        print("WARNING:", w)
    print(f"Review text in: {work / 'text'}")
    if args.extract_only:
        return

    if args.only:
        keep = {int(x) for x in args.only.split(",")}
        for c in chapters:
            c.include = c.include and c.num in keep

    if not synthesize(chapters, work, args.voice, args.speed, args.device):
        sys.exit(130)

    # Readable filename (shown in the Files app / file managers): "Title - Author.m4b"
    nice = re.sub(r'[\\/:*?"<>|]+', "", f"{epub.title} - {epub.author}" if epub.author else epub.title).strip()
    suffix = " (sample).m4b" if args.only else ".m4b"
    out = args.output or work / f"{nice}{suffix}"
    build_m4b(epub, chapters, work, out, args.voice, args.bitrate)
    print(f"\nDone: {out}")


if __name__ == "__main__":
    main()
