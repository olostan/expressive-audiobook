"""Word counts, sentence splits, ligatures, and duration estimates."""

from __future__ import annotations

import math
import re
import unicodedata

_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_LATEX_RE = re.compile(r"\$\$.*?\$\$|\$[^$]+\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)", re.DOTALL)
_ANCHOR_RE = re.compile(r"\[\[[^\]]+\]\]")
_ABBREV = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "fig",
    "eq",
    "vs",
    "etc",
}


def normalize_text(text: str) -> str:
    """NFC, drop soft hyphens, expand common ligatures, and collapse NULs."""
    text = unicodedata.normalize("NFC", text.replace("\x00", ""))
    text = text.replace("\u00ad", "")
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    return text


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def strip_unspoken_markup(text: str) -> str:
    """Remove LaTeX spans and [[anchor]] markers before a word estimate."""
    cleaned = _LATEX_RE.sub(" ", text or "")
    cleaned = _ANCHOR_RE.sub(" ", cleaned)
    return cleaned


def spoken_source_words(text: str) -> int:
    return word_count(strip_unspoken_markup(text))


def estimate_seconds(words: int, wpm: float) -> float:
    if wpm <= 0:
        raise ValueError("wpm must be positive")
    return words / wpm * 60.0


def max_words_under(seconds: float, wpm: float) -> int:
    """Largest word count whose duration is strictly below `seconds`."""
    if seconds <= 0 or wpm <= 0:
        return 0
    return max(0, math.floor(seconds * wpm / 60.0 - 1e-9))


def min_words_at(seconds: float, wpm: float) -> int:
    """Smallest word count whose duration is at least `seconds`."""
    if seconds <= 0 or wpm <= 0:
        return 0
    return max(0, math.ceil(seconds * wpm / 60.0 - 1e-9))


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation. Never returns a partial token."""
    raw = (text or "").strip()
    if not raw:
        return []
    tokens = raw.split()
    sentences: list[str] = []
    current: list[str] = []
    for tok in tokens:
        current.append(tok)
        if _ends_sentence(tok):
            sentences.append(" ".join(current))
            current = []
    if current:
        sentences.append(" ".join(current))
    return sentences


def _ends_sentence(token: str) -> bool:
    if not token.endswith((".", "!", "?")):
        return False
    bare = token[:-1].rstrip("\"'”’)]")
    if bare.lower() in _ABBREV:
        return False
    if re.fullmatch(r"[A-Z]", bare):
        return False
    if bare.lower() in {"e.g", "i.e"}:
        return False
    return True


def join_hyphenated_lines(lines: list[str]) -> str:
    """Rejoin line-break hyphenation. Keeps real compound hyphens."""
    parts: list[str] = []
    for raw in lines:
        line = normalize_text(raw).strip()
        if not line:
            continue
        if not parts:
            parts.append(line)
            continue
        prev = parts[-1]
        if _is_line_break_hyphen(prev) and line[:1].islower():
            parts[-1] = prev[:-1] + line
        else:
            parts.append(line)
    return " ".join(parts)


def _is_line_break_hyphen(text: str) -> bool:
    if text.endswith("--") or text.endswith("—"):
        return False
    return len(text) >= 2 and text.endswith("-") and text[-2].isalpha()


def count_sentences(text: str) -> int:
    content = (text or "").strip()
    if not content:
        return 0
    parts = [p for p in re.split(r"[.!?]+", content) if p.strip()]
    return max(1, len(parts))
