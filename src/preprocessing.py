from __future__ import annotations

import re
import unicodedata


SECTION_MARKER_RE = re.compile(r"\[[^\]]+\]")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_label(value: str) -> str:
    """Normalize labels for comparisons without changing saved accents."""
    value = (value or "").strip().lower()
    decomposed = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return " ".join(value.split())


def clean_lyrics(
    text: str,
    *,
    lowercase: bool = False,
    remove_section_markers: bool = True,
) -> str:
    """Clean lyrics conservatively, preserving meaningful repetitions."""
    text = text or ""
    text = text.replace("\r", " ").replace("\n", " ")
    if remove_section_markers:
        text = SECTION_MARKER_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def word_count(text: str) -> int:
    return len((text or "").split())

