"""Understand a book from its opening pages and propose a narration style."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .config import AppConfig
from .ingest import load_chapter, load_manifest
from .tts import PREBUILT_VOICES

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

STYLE_PROMPT = """You are a seasoned audiobook director writing the narration brief for the book below. The
listener owns the book. A text-to-speech model will perform it and obeys written direction, so the brief must be
concrete and specific to THIS book. Generic advice ("be warm and clear") is a failure.

Step 1 - identify the book. Use the metadata, the book's own table of contents, the chapter list and the excerpts.
If you recognise the book, draw on what you actually know about it: its structure, recurring characters and how
they differ, running jokes, signature devices, famous set pieces, specialised vocabulary, and where a narrator
would struggle. If you do not recognise it, say so and work only from the material given. Never invent characters,
chapters or facts you are unsure of; leave a field empty instead.

Step 2 - decide how a skilled human narrator should perform the whole book, and write that down.

Book metadata:
{metadata}

Detected features of the whole book:
{features}

The book's own table of contents (it is skipped in the audio):
{contents}

Chapter list as detected (titles can be imperfect):
{chapters}

Return JSON only, with these keys:
- "identification": object {{"title", "author", "recognised": true/false, "notes": what you know that matters for narration}}
- "genre": a short phrase
- "summary": two or three sentences on what the book is and how it is written
- "tone": a few words
- "narrator_persona": three to five vivid sentences: who is speaking and with what attitude
- "default_delivery_style": a short free-form delivery phrase for ordinary passages (not empty)
- "literary_traits": 4 to 10 specific items the narration must preserve
- "speaker_profiles": up to {max_voices} recurring speakers, each {{"name", "description", "vocal_cue", "voice"}}.
  "name" must be exactly the label from the detected features when one matches. "description" is personality in
  one or two sentences. "vocal_cue" is how the voice should sound. "voice" is one distinct prebuilt voice that suits
  the character, chosen from: {voices}. The narrator uses {narrator}; do not reuse it. [] if the book has no dialogue.
- "performance_directives": 4 to 10 objects {{"title", "when", "how"}}. Each is a directive specific to this book:
  a structural device and how to make the listener feel it, a tonal register and when it applies, how to move between
  exposition and dialogue, where to let an idea land in silence, how to treat jokes, asides, quotations and
  emotional peaks. "when" says where in the book it applies (name chapters or recurring situations). "how" is two
  to four sentences of concrete delivery instruction.
- "verbalize": object with "diagrams", "formulas", "footnotes", "citations": two to four sentences each, specific
  to the kind of content this book has
- "pronunciation_lexicon": up to 25 hard, foreign or invented names and terms as [{{"word": "...", "ipa": "/.../"}}];
  omit any whose IPA you are not sure of
- "recommended": object of numbers from 0 to 1: "expressiveness" (0 plain reading, 1 fully performed),
  "pace" (0 slowest, 0.5 conversational, 1 fastest), "pause_density" (0 only punctuation pauses, 1 frequent
  room to think)
- "audience": who the book is for: "kids", "young_adult", "general" or "technical"
- "soundscape": object {{"music": {{"recommended": true/false, "style": "a few words on fitting background music", "reason": "one sentence"}},
  "sfx": {{"recommended": true/false, "reason": "one sentence"}}}}. Recommend music and sound effects for fiction and
  stories, most of all for children. Do not recommend them for scientific, technical, academic or reference books.
- "notes": optional markdown with anything else a narrator must know. Do not quote the excerpts.

Excerpts:
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


def _speaker_key(name) -> str:
    """'The Tortoise (Mr. T)' and 'Tortoise' are the same speaker."""
    text = re.sub(r"\(.*?\)", "", str(name)).strip().casefold()
    return re.sub(r"^(?:the|a|an)\s+", "", text)


def assign_character_voices(narrator: str, speakers: list[str], suggested: dict | None = None) -> dict[str, str]:
    """Distinct prebuilt voices for recurring speakers. Suggestions are used when valid."""
    taken = {narrator}
    result: dict[str, str] = {}
    suggested = {_speaker_key(k): str(v) for k, v in (suggested or {}).items()}
    for name in speakers[:MAX_CHARACTER_VOICES]:
        voice = suggested.get(_speaker_key(name))
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


def chapter_titles(ingest_dir: Path, limit: int = 80) -> str:
    manifest = load_manifest(ingest_dir)
    rows = [f"{item['index']}. {item['title'][:70]}" for item in (manifest.get("chapters") or [])[:limit]]
    return "\n".join(rows) or "(none)"


def representative_excerpts(ingest_dir: Path, per_chapter: int = 2200) -> str:
    """Short passages from other parts of the book: a dialogue-heavy chapter, a technical one, a middle one."""
    manifest = load_manifest(ingest_dir)
    rows = []
    for item in manifest.get("chapters") or []:
        chapter = load_chapter(Path(ingest_dir) / item["path"])
        prose = [b for b in chapter.blocks if b.text.strip() and b.kind != "scanned_page"]
        if len(prose) < 3:
            continue
        rows.append(
            {
                "item": item,
                "blocks": prose,
                "dialogue": sum(1 for b in prose if b.speaker),
                "technical": sum(1 for e in chapter.elements if e.kind in {"formula", "figure"}),
            }
        )
    if len(rows) < 2:
        return ""
    chosen: list[tuple[str, dict]] = []
    rest = rows[1:]
    for label, key in (("dialogue-heavy", "dialogue"), ("technical", "technical")):
        best = max(rest, key=lambda r: r[key], default=None)
        if best and best[key] > 0 and all(best is not c[1] for c in chosen):
            chosen.append((label, best))
    middle = rows[int(len(rows) * 0.6)]
    if all(middle is not c[1] for c in chosen):
        chosen.append(("later in the book", middle))
    parts = []
    for label, row in chosen:
        text, size = [], 0
        for block in row["blocks"]:
            line = f"[{block.speaker}] {block.text.strip()}" if block.speaker else block.text.strip()
            text.append(line)
            size += len(line)
            if size >= per_chapter:
                break
        parts.append(f"--- {label}: chapter {row['item']['index']} \"{row['item']['title'][:60]}\" ---\n" + "\n".join(text))
    return "\n\n".join(parts)


def detect_style(
    ingest_dir: Path,
    config: AppConfig,
    client,
    pages: int = 5,
    narrator: str = "Schedar",
    model: str | None = None,
) -> dict:
    """Ask the text model for a narration brief. Returns its parsed JSON."""
    features = book_features(ingest_dir)
    manifest = load_manifest(ingest_dir)
    opening = sample_excerpt(ingest_dir, pages, max_chars=6000)
    if not opening.strip():
        raise RuntimeError("No narratable text found in the first pages to analyse")
    excerpts = f"--- the opening pages ---\n{opening}"
    more = representative_excerpts(ingest_dir)
    if more:
        excerpts += "\n\n" + more
    metadata = {k: manifest.get(k) for k in ("title", "author") if manifest.get(k)}
    source = manifest.get("source_file") or manifest.get("source_pdf")
    if source:
        metadata["file_name"] = Path(source).name
    brief = {k: v for k, v in features.items() if k != "speaker_counts"}
    prompt = STYLE_PROMPT.format(
        metadata=json.dumps(metadata, ensure_ascii=False),
        features=json.dumps({**brief, "speakers_by_lines": features["speaker_counts"]}, ensure_ascii=False, indent=2),
        contents=(manifest.get("contents_text") or "(not available)")[:3500],
        chapters=chapter_titles(ingest_dir),
        excerpt=excerpts,
        max_voices=MAX_CHARACTER_VOICES,
        narrator=narrator,
        voices=", ".join(f"{name} ({VOICE_TRAITS[name]})" for name in sorted(VOICE_TRAITS) if name != narrator),
    )
    request = {
        "model": model or config.prep.model,
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
