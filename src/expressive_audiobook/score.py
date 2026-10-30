"""The score: which music plays under which chapter, and which sound effects land where.

Two text-model "directors" write plans as small JSON files under ``score/``
(editable by hand; a plan whose inputs have not changed is never rewritten).
Sounds are then taken from the shared library, or generated and added to it.

    score/music.json         palette + one entry per chapter
    score/sfx-NNN.json       cues for one chapter
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from .config import AppConfig
from .ingest import load_chapter, load_manifest
from .library import SFX_CATEGORIES, Asset, Library, asset_id
from .project import Project, atomic_write, fingerprint
from .soundgen import SoundGenError, SoundRequest, make_provider
from .textutil import word_count

BED_SUFFIX = (
    "Instrumental background music. Seamless, even and steady, with no melodic hook, "
    "no vocals, no drums and no sudden changes."
)
KIDS_NEGATIVE = "scary, harsh, loud, distorted, screaming, sudden bang"
MIN_CUE_WORDS = 150  # intensity 1.0: about one cue per this many words
MAX_CUE_WORDS = 1200  # intensity near 0: about one cue per this many words

MUSIC_PROMPT = """You are the music supervisor for an audiobook. Choose quiet background music for each chapter.
The music sits far below the narrator and must never compete with the words.

Audience: {audience}
Listener's wish for the music: {wish}

About the book:
{about}

Chapters, each with short excerpts:
{chapters}

Music already in the shared library (reuse an id when it truly fits the chapter):
{catalog}

Return JSON only:
{{
  "palette": "one or two sentences naming the instruments and feel shared by the whole book, so chapters sound like one score",
  "chapters": [
    {{"chapter": <number>,
      "music": true or false (false for a chapter that is better left bare),
      "reuse": "<library id>" or null,
      "title": "2-4 words",
      "mood": "a few words",
      "prompt": "what a music generator should make for this chapter: setting, mood, tempo, instruments (8-30 words). No lyrics, no story names."}}
  ]
}}
Include every chapter number exactly once. Keep chapters near each other in mood close in sound; change the music when the story's mood really changes."""

SFX_PROMPT = """You are the sound designer for an audiobook. Below is one chapter, already split into numbered turns.
Choose where a quiet sound effect would add to the listening without distracting from the story.

Audience: {audience}
Intensity: {intensity:.2f} on a scale from 0 (none) to 1 (very noticeable).
Use at most {budget} cues in this chapter. Fewer is better; skip anything uncertain.

Rules:
- Only concrete, audible things the text itself describes or clearly implies: a door, rain arriving, footsteps in snow, a bell, a magic chime. Never music, never voices, never narration.
- "spot" is a short one-off sound at a turn boundary ("before" the turn starts or "after" it ends).
- "ambience" is a quiet loop (rain, wind, a crowd, a forest) under turns "at" through "through".
- Prefer a sound from the library below. Write a new one only when nothing fits.
{audience_rule}
Sounds already in the shared library:
{catalog}

Chapter "{title}":
{turns}

Return JSON only:
{{"cues": [
  {{"at": <turn number>, "when": "before" or "after", "type": "spot" or "ambience", "through": <turn number or null>,
    "library_id": "<id from the library>" or null,
    "title": "2-4 words",
    "prompt": "what a sound generator should make (6-20 words, one sound only), when library_id is null",
    "seconds": <1-6 for a spot, 8-12 for an ambience>,
    "category": "one of {categories}",
    "reason": "one short sentence"}}
]}}"""


@dataclass
class ChapterMusic:
    chapter: int
    enabled: bool
    prompt: str = ""
    title: str = ""
    mood: str = ""
    reuse: str | None = None
    tracks: list[Asset] = field(default_factory=list)


@dataclass
class Cue:
    at: int
    when: str
    kind: str  # "spot" or "ambience"
    through: int | None
    asset: Asset
    reason: str = ""


@dataclass
class SoundSet:
    music: ChapterMusic
    cues: list[Cue] = field(default_factory=list)


def level_db(level: float, config: AppConfig) -> float:
    """Music loudness under speech for a 0-1 level (0 is silent)."""
    s = config.sound
    return s.music_db_floor + (s.music_db_ceiling - s.music_db_floor) * max(0.0, min(1.0, level))


def cue_budget(words: int, intensity: float) -> int:
    if intensity <= 0.02 or words <= 0:
        return 0
    per_cue = MAX_CUE_WORDS - (MAX_CUE_WORDS - MIN_CUE_WORDS) * min(1.0, intensity)
    return max(1, math.ceil(words / per_cue))


def soundscape(project: Project) -> dict:
    return project.data.get("soundscape") or {}


def music_wanted(project: Project) -> float:
    music = soundscape(project).get("music") or {}
    return float(music.get("level") or 0) if music.get("enabled") else 0.0


def sfx_wanted(project: Project) -> float:
    sfx = soundscape(project).get("sfx") or {}
    return float(sfx.get("intensity") or 0) if sfx.get("enabled") else 0.0


def audience_of(project: Project) -> str:
    return str(soundscape(project).get("audience") or "general")


def _model(config: AppConfig) -> str:
    return config.sound.director_model or config.prep.model


def _ask(client, config: AppConfig, prompt: str) -> dict:
    request = {
        "model": _model(config),
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.5},
    }
    reply = client.generate(request)
    if not isinstance(reply, dict):
        raise ValueError("the sound director did not return a JSON object")
    return reply


# ---------------------------------------------------------------- music


def _excerpt(chapter, words: int) -> str:
    texts = [b.text for b in chapter.blocks if b.kind != "heading" and b.text.strip()]
    if not texts:
        return ""
    joined = " ".join(texts).split()
    if len(joined) <= words:
        return " ".join(joined)
    head, mid, tail = words // 2, words // 4, words // 4
    middle = len(joined) // 2
    return " ".join(joined[:head]) + " […] " + " ".join(joined[middle : middle + mid]) + " […] " + " ".join(joined[-tail:])


def _chapter_digest(project: Project, per_chapter: int = 160, limit: int = 40) -> str:
    manifest = load_manifest(project.ingest_dir)
    rows = []
    for item in (manifest.get("chapters") or [])[:limit]:
        chapter = load_chapter(project.ingest_dir / item["path"])
        rows.append(f"[{item['index']}] {item['title'][:70]}\n{_excerpt(chapter, per_chapter)}")
    return "\n\n".join(rows)


def music_plan_path(project: Project) -> Path:
    return project.root / "score" / "music.json"


def plan_music(project: Project, config: AppConfig, client, library: Library, *, force: bool = False) -> dict:
    """Ask the director for a palette and one entry per chapter. Reuses the saved plan when nothing changed."""
    soundscape_cfg = soundscape(project)
    about = (project.style_path.read_text(encoding="utf-8")[:1800] if project.style_path.exists() else "") or (
        project.data.get("title") or project.name
    )
    digest = _chapter_digest(project)
    catalog = library.catalog("music")
    wish = (soundscape_cfg.get("music") or {}).get("style_hint") or "none given"
    key = fingerprint(digest, about, wish, audience_of(project), _model(config))
    path = music_plan_path(project)
    if path.exists() and not force:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("fingerprint") == key:
            return saved
    reply = _ask(
        client, config,
        MUSIC_PROMPT.format(audience=audience_of(project), wish=wish, about=about, chapters=digest, catalog=catalog),
    )
    chapters: dict[str, dict] = {}
    for entry in reply.get("chapters") or []:
        try:
            number = int(entry["chapter"])
        except (KeyError, TypeError, ValueError):
            continue
        reuse = entry.get("reuse")
        chapters[str(number)] = {
            "music": bool(entry.get("music", True)),
            "reuse": reuse if reuse and library.get(str(reuse)) else None,
            "title": str(entry.get("title") or "").strip(),
            "mood": str(entry.get("mood") or "").strip(),
            "prompt": " ".join(str(entry.get("prompt") or "").split()),
        }
    if not chapters:
        raise ValueError("the music director returned no chapters")
    plan = {"fingerprint": key, "palette": " ".join(str(reply.get("palette") or "").split()), "chapters": chapters}
    atomic_write(path, json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    return plan


def music_prompt(plan: dict, entry: dict) -> str:
    parts = [entry["prompt"], plan.get("palette") or "", BED_SUFFIX]
    return " ".join(p.strip() for p in parts if p and p.strip())


def ensure_music(
    project: Project, config: AppConfig, plan: dict, library: Library, chapter: int, *, ui=None, provider=None
) -> ChapterMusic:
    entry = (plan.get("chapters") or {}).get(str(chapter))
    if not entry or not entry.get("music") or not (entry.get("prompt") or entry.get("reuse")):
        return ChapterMusic(chapter, False)
    s = config.sound
    result = ChapterMusic(chapter, True, entry.get("prompt", ""), entry.get("title", ""), entry.get("mood", ""), entry.get("reuse"))
    base = library.get(entry["reuse"]) if entry.get("reuse") else None
    prompt = base.prompt if base else music_prompt(plan, entry)
    if base:
        result.tracks.append(base)
    provider = provider or make_provider("music", s)
    for variant in range(1, s.music_variants + 1):
        if len(result.tracks) >= s.music_variants:
            break
        seed = variant
        ident = asset_id("music", prompt, provider.name, provider.model, seed, s.music_seconds, True)
        found = library.get(ident)
        if found is None:
            if ui:
                ui.info(f"🎼 chapter {chapter}: composing '{entry.get('title') or entry.get('mood') or 'music'}' (variant {variant})")
            found = _generate(
                library, provider,
                SoundRequest("music", prompt, s.music_seconds, seed, True, s.music_negative_prompt),
                Asset(ident, "music", prompt, entry.get("title", ""), mood=entry.get("mood", ""), seconds=s.music_seconds,
                      loop=True, provider=provider.name, model=provider.model, seed=seed, origin=project.name),
            )
        if found.status != "rejected" and found.id not in {t.id for t in result.tracks}:
            result.tracks.append(found)
    return result


def _generate(library: Library, provider, request: SoundRequest, asset: Asset) -> Asset:
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        raw = Path(scratch) / "raw.wav"
        provider.generate(request, raw)
        return library.add(raw, asset)


# ---------------------------------------------------------------- sound effects


def sfx_plan_path(project: Project, chapter: int) -> Path:
    return project.root / "score" / f"sfx-{chapter:03d}.json"


def numbered_turns(chunks: list, outputs: dict) -> list[str]:
    turns: list[str] = []
    for chunk in sorted(chunks, key=lambda c: c.order):
        output = outputs.get(chunk.id)
        if output is None:
            continue
        turns.extend(str(t.get("text") or "") for t in output.get("turns") or [])
    return turns


def plan_sfx(
    project: Project, config: AppConfig, client, library: Library, chapter: int, title: str, turns: list[str], *, force: bool = False
) -> list[dict]:
    intensity = sfx_wanted(project)
    words = sum(word_count(t) for t in turns)
    budget = cue_budget(words, intensity)
    if not budget or not turns:
        return []
    key = fingerprint(turns, round(intensity, 2), audience_of(project), _model(config))
    path = sfx_plan_path(project, chapter)
    if path.exists() and not force:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("fingerprint") == key:
            return saved.get("cues") or []
    audience = audience_of(project)
    rule = (
        "- The listeners are young children: no frightening, sudden, or loud sounds. Keep everything gentle.\n"
        if audience in {"kids", "children"}
        else ""
    )
    numbered = "\n".join(f"{i}. {t[:260]}" for i, t in enumerate(turns, start=1))
    reply = _ask(
        client, config,
        SFX_PROMPT.format(
            audience=audience, intensity=intensity, budget=budget, audience_rule=rule, catalog=library.catalog("sfx"),
            title=title, turns=numbered, categories=", ".join(SFX_CATEGORIES),
        ),
    )
    cues = validate_cues(reply.get("cues") or [], len(turns), budget, library)
    atomic_write(path, json.dumps({"fingerprint": key, "cues": cues}, indent=2, ensure_ascii=False) + "\n")
    return cues


def validate_cues(raw: list, turn_count: int, budget: int, library: Library) -> list[dict]:
    """Keep well-formed cues, drop overlaps, and respect the budget."""
    kept: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            at = int(item["at"])
        except (KeyError, TypeError, ValueError):
            continue
        kind = item.get("type") if item.get("type") in {"spot", "ambience"} else "spot"
        when = item.get("when") if item.get("when") in {"before", "after"} else "before"
        through = item.get("through")
        try:
            through = int(through) if through is not None else None
        except (TypeError, ValueError):
            through = None
        if not 1 <= at <= turn_count:
            continue
        if kind == "ambience":
            through = min(turn_count, max(at, through or at))
        else:
            through = None
        ident = item.get("library_id")
        if ident and library.get(str(ident)) is None:
            ident = None
        prompt = " ".join(str(item.get("prompt") or "").split())
        if not ident and not prompt:
            continue
        category = item.get("category") if item.get("category") in SFX_CATEGORIES else ("ambience" if kind == "ambience" else "other")
        kept.append(
            {
                "at": at, "when": when, "type": kind, "through": through, "library_id": ident,
                "title": str(item.get("title") or "").strip(), "prompt": prompt,
                "seconds": float(item.get("seconds") or (10 if kind == "ambience" else 3)),
                "category": category, "reason": str(item.get("reason") or "").strip(),
            }
        )
    kept.sort(key=lambda c: (c["at"], c["when"] != "before"))
    spaced: list[dict] = []
    for cue in kept:
        if spaced and cue["type"] == "spot" and spaced[-1]["type"] == "spot" and cue["at"] - spaced[-1]["at"] < 2:
            continue
        spaced.append(cue)
    return spaced[:budget]


def ensure_cues(project: Project, config: AppConfig, cues: list[dict], library: Library, *, ui=None, provider=None) -> list[Cue]:
    s = config.sound
    out: list[Cue] = []
    kids = audience_of(project) in {"kids", "children"}
    for cue in cues:
        asset = library.get(cue["library_id"]) if cue.get("library_id") else None
        if asset is None:
            provider = provider or make_provider("sfx", s)
            ambience = cue["type"] == "ambience"
            seconds = max(0.5, min(cue["seconds"], s.ambience_seconds if ambience else s.sfx_max_seconds))
            prompt = cue["prompt"]
            ident = asset_id("sfx", prompt, provider.name, provider.model, 1, seconds, ambience)
            asset = library.get(ident)
            if asset is None:
                if ui:
                    ui.info(f"🔔 making sound: {cue.get('title') or prompt[:40]}")
                try:
                    asset = _generate(
                        library, provider,
                        SoundRequest("sfx", prompt, seconds, 1, ambience, KIDS_NEGATIVE if kids else ""),
                        Asset(ident, "sfx", prompt, cue.get("title", ""), category=cue.get("category", ""), seconds=seconds,
                              loop=ambience, provider=provider.name, model=provider.model, seed=1, origin=project.name),
                    )
                except SoundGenError as exc:
                    if ui:
                        ui.warn(f"skipped a sound effect: {exc}")
                    continue
        if asset.status == "rejected" or (s.require_approval and asset.status != "approved"):
            continue
        out.append(Cue(cue["at"], cue["when"], cue["type"], cue.get("through"), asset, cue.get("reason", "")))
    return out


def chapter_seed(project: Project, chapter: int) -> int:
    return random.Random(f"{project.name}:{chapter}").randrange(1 << 30)
