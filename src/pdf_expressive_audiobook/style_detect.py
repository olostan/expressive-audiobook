"""Understand a book from its opening pages and propose a narration style."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .config import AppConfig
from .ingest import load_chapter, load_manifest
from .tts import PREBUILT_VOICES
from .tuning import EXPRESSIVENESS, PACE, PAUSES

# Short descriptors from the prebuilt voice table, to help match characters to voices.
VOICE_TRAITS = {
    "Puck": "upbeat", "Charon": "informative", "Kore": "firm", "Fenrir": "excitable",
    "Leda": "youthful", "Orus": "firm", "Aoede": "breezy", "Callirrhoe": "easy-going",
    "Autonoe": "bright", "Enceladus": "breathy", "Iapetus": "clear", "Umbriel": "easy-going",
    "Algieba": "smooth", "Despina": "smooth", "Erinome": "clear", "Algenib": "gravelly",
    "Rasalgethi": "informative", "Laomedeon": "upbeat", "Achernar": "soft", "Alnilam": "firm",
    "Schedar": "even", "Gacrux": "mature", "Pulcherrima": "forward", "Achird": "friendly",
    "Zubenelgenubi": "casual", "Vindemiatrix": "gentle", "Sadachbia": "lively",
    "Sadaltager": "knowledgeable", "Sulafat": "warm", "Zephyr": "bright",
}
MAX_CHARACTER_VOICES = 6
MIN_SPEAKER_LINES = 5

STYLE_PROMPT = """You are preparing an audiobook of the book excerpted below. The listener owns the book.
Read the excerpt (the opening pages of the first chapters) and the detected features, then decide how a
skilled human narrator should perform the whole book.

Detected features of the whole book:
{features}

Return JSON only, with these keys:
- "title": the book's title if it is evident, else null
- "genre": a short phrase
- "summary": one or two sentences on what the book is and how it is written
- "tone": a few words (for example "playful, erudite, warm")
- "narrator_persona": two or three sentences describing the narrator's voice and attitude
- "default_delivery_style": a short free-form delivery phrase for ordinary passages (not empty)
- "literary_traits": 3 to 8 short items, what the narration must preserve (wordplay, dialogue, humor, rigor, ...)
- "speakers": names of recurring characters in dialogue, if any (else [])
- "pronunciation_lexicon": up to 10 hard or invented names as [{{"word": "...", "ipa": "/.../"}}]
- "verbalize": object with "diagrams", "formulas", "footnotes", "citations": one sentence each on how to speak them
- "speaker_voices": object mapping each recurring speaker from the detected features (at most {max_voices}) to
  one distinct prebuilt voice that suits the character. The narrator already uses {narrator}; do not reuse it.
  Choose from: {voices}
- "recommended": object with "expressiveness" ({expressiveness}), "pace" ({pace}), "pause_density" ({pauses})
- "notes": markdown, 2 to 6 short paragraphs or bullets of concrete direction: where to smile, where to slow
  down and let a hard idea land, how to treat dialogue and asides. Do not quote the excerpt.

Excerpt:
{excerpt}
"""


def book_features(ingest_dir: Path) -> dict:
    manifest = load_manifest(ingest_dir)
    kinds: Counter = Counter()
    words = 0
    speakers: Counter = Counter()
    chapters = manifest.get("chapters") or []
    for item in chapters:
        chapter = load_chapter(Path(ingest_dir) / item["path"])
        for element in chapter.elements:
            kinds[element.kind] += 1
        for block in chapter.blocks:
            words += len(block.text.split())
            if block.speaker:
                speakers[block.speaker] += 1
    return {
        "chapters": len(chapters),
        "pages": manifest.get("page_count"),
        "words": words,
        "minutes_estimate": round(words / 150),
        "figures": kinds["figure"],
        "formulas": kinds["formula"],
        "footnotes": kinds["footnote"],
        "citations": kinds["citation"],
        "dialogue_speakers": [name for name, count in speakers.most_common(8) if count >= MIN_SPEAKER_LINES],
        "speaker_counts": _merge_case(speakers),
        "skipped_pages": len(manifest.get("skipped_pages") or []),
        "chapter_detection": manifest.get("detection"),
    }


def _merge_case(speakers: Counter) -> dict[str, int]:
    """ACHILLES and Achilles are one speaker. Keep the most common spelling."""
    best: dict[str, tuple[int, str]] = {}
    totals: Counter = Counter()
    for name, count in speakers.items():
        key = name.casefold()
        totals[key] += count
        if key not in best or count > best[key][0]:
            best[key] = (count, name)
    return {best[key][1]: total for key, total in totals.most_common() if total >= MIN_SPEAKER_LINES}


def assign_character_voices(narrator: str, speakers: list[str], suggested: dict | None = None) -> dict[str, str]:
    """Distinct prebuilt voices for recurring speakers. Suggestions are used when valid."""
    taken = {narrator}
    result: dict[str, str] = {}
    suggested = {str(k).casefold(): str(v) for k, v in (suggested or {}).items()}
    for name in speakers[:MAX_CHARACTER_VOICES]:
        voice = suggested.get(name.casefold())
        if voice not in PREBUILT_VOICES or voice in taken:
            voice = next((v for v in sorted(VOICE_TRAITS) if v in PREBUILT_VOICES and v not in taken), None)
        if voice is None:
            break
        taken.add(voice)
        result[name] = voice
    return result


def sample_excerpt(ingest_dir: Path, pages: int = 5, max_chars: int = 9000) -> str:
    """Text of the first ``pages`` pages that have narratable content."""
    manifest = load_manifest(ingest_dir)
    collected: list[str] = []
    first_page: int | None = None
    for item in manifest.get("chapters") or []:
        chapter = load_chapter(Path(ingest_dir) / item["path"])
        for block in chapter.blocks:
            if not block.text.strip() or block.kind == "scanned_page":
                continue
            if first_page is None:
                first_page = block.page_start
            if block.page_start >= first_page + pages:
                return _clip("\n\n".join(collected), max_chars)
            collected.append(block.text.strip())
    return _clip("\n\n".join(collected), max_chars)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def detect_style(ingest_dir: Path, config: AppConfig, client, pages: int = 5, narrator: str = "Schedar") -> dict:
    """Ask the text model for a style description. Returns its parsed JSON."""
    features = book_features(ingest_dir)
    excerpt = sample_excerpt(ingest_dir, pages)
    if not excerpt.strip():
        raise RuntimeError("No narratable text found in the first pages to analyse")
    prompt = STYLE_PROMPT.format(
        features=json.dumps(features, ensure_ascii=False, indent=2),
        excerpt=excerpt,
        expressiveness=" | ".join(EXPRESSIVENESS),
        pace=" | ".join(PACE),
        pauses=" | ".join(PAUSES),
        max_voices=MAX_CHARACTER_VOICES,
        narrator=narrator,
        voices=", ".join(f"{name} ({VOICE_TRAITS[name]})" for name in sorted(VOICE_TRAITS) if name != narrator),
    )
    request = {
        "model": config.prep.model,
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3},
    }
    return client.generate(request)


def fallback_style(features: dict) -> dict:
    """Used when no API key is available. Neutral, but marks what was detected."""
    traits = ["Preserve the author's wording.", "Let the sentences carry the meaning; do not over-act."]
    if features.get("dialogue_speakers"):
        traits.append("Give each speaker in dialogue a distinct manner.")
    return {
        "genre": None,
        "summary": "",
        "tone": "",
        "narrator_persona": "A clear, warm, unhurried narrator who trusts the text.",
        "default_delivery_style": "warm and unhurried",
        "literary_traits": traits,
        "speakers": features.get("dialogue_speakers") or [],
        "pronunciation_lexicon": [],
        "verbalize": {
            "diagrams": "One or two sentences on what the figure shows and why it is here.",
            "formulas": "Speak the expression so a listener can reconstruct it.",
            "footnotes": "Mention a short note; read a long one only if the sentence needs it.",
            "citations": "Read the name and speak the year as words.",
        },
        "recommended": {},
        "notes": "No style analysis was run (no API key). Edit this file by hand, or re-run "
        "`audiobook import --restyle` once GEMINI_API_KEY is set.",
    }
