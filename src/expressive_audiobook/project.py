"""A project is one book: its settings, its style, and every file generated from it.

Layout, all under ``projects/<name>/``::

    project.json   settings you can edit (voice, pace, expressiveness, ...)
    style.md       narration style written at import, editable
    ingest/        manifest.json and chapters/NNN.json: the book as data
    chunks/        speech-sized pieces of each chapter
    prep/          narration text, styles, and tags per chunk, plus review/
    tts/           the exact speech requests
    audio/         clips, chapter WAVs, book.wav
    samples/       short WAVs from ``generate --sample``
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .home import default_projects_dir
from .config import AppConfig, StyleProfile, VoiceMapping, _merge_config, style_from_dict
from .tts import classify_voice_name
from .tuning import DEFAULTS as TUNING_DEFAULTS
from .tuning import chapter_tail_ms, gap_ms, level, tag_floor

PROJECT_FILE = "project.json"
STYLE_FILE = "style.md"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

DEFAULTS: dict = {
    "version": 1,
    "name": "",
    "title": None,
    "source_pdf": "",
    "source_sha256": "",
    "created_at": "",
    "language": "en",
    "narration": {
        "voice": "Schedar",
        "lite": False,
        "expressiveness": TUNING_DEFAULTS["expressiveness"],
        "pace": TUNING_DEFAULTS["pace"],
        "pause_density": TUNING_DEFAULTS["pause_density"],
    },
    "audio": {"gap_ms": None, "chapter_tail_ms": None, "format": "mp3", "keep_wav": False},
    "style": {"pages": 5, "model": None},
    "character_voices": {},
    # Optional extras. Both cost generation time or money, so both start off.
    "soundscape": {
        "music": {"enabled": False, "level": 0.4, "style_hint": ""},
        "sfx": {"enabled": False, "intensity": 0.4},
        "audience": None,
    },
    "overrides": {},
}


class ProjectError(RuntimeError):
    pass


def projects_dir(explicit: str | Path | None = None) -> Path:
    return Path(explicit or os.environ.get("AUDIOBOOK_PROJECTS") or default_projects_dir())


def valid_name(name: str) -> str:
    if not _NAME_RE.match(name or ""):
        raise ProjectError("--name may use letters, digits, '.', '_' and '-' and must not start with a symbol")
    return name


def list_projects(base: Path) -> list[str]:
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if (p / PROJECT_FILE).exists())


def resolve_project(name: str | None, base: Path) -> "Project":
    if name:
        return Project.load(base / name)
    found = list_projects(base)
    if len(found) == 1:
        return Project.load(base / found[0])
    if not found:
        raise ProjectError(f"No projects in {base}. Run: audiobook import --pdf book.pdf --name <name>")
    raise ProjectError(f"Several projects in {base} ({', '.join(found)}). Pick one with --project NAME")


@dataclass
class Project:
    root: Path
    data: dict

    @property
    def name(self) -> str:
        return self.data["name"]

    @property
    def ingest_dir(self) -> Path:
        return self.root / "ingest"

    @property
    def style_path(self) -> Path:
        return self.root / STYLE_FILE

    @property
    def samples_dir(self) -> Path:
        return self.root / "samples"

    @classmethod
    def create(cls, base: Path, name: str, pdf: Path) -> "Project":
        valid_name(name)
        data = copy.deepcopy(DEFAULTS)
        data.update(
            name=name,
            source_pdf=str(pdf.resolve()),
            source_sha256=file_sha256(pdf),
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        project = cls(base / name, data)
        project.root.mkdir(parents=True, exist_ok=True)
        return project

    @classmethod
    def load(cls, root: Path) -> "Project":
        path = root / PROJECT_FILE
        if not path.exists():
            raise ProjectError(f"{root} is not a project (no {PROJECT_FILE}). Run audiobook import first.")
        stored = json.loads(path.read_text(encoding="utf-8"))
        return cls(root, _merge(copy.deepcopy(DEFAULTS), stored))

    def save(self) -> None:
        atomic_write(self.root / PROJECT_FILE, json.dumps(self.data, indent=2, ensure_ascii=False) + "\n")

    def tuning(self) -> dict:
        """Validated delivery settings with gaps filled in from pause density."""
        narration = self.data["narration"]
        expressiveness = level("expressiveness", narration["expressiveness"])
        pace = level("pace", narration["pace"])
        pauses = level("pause_density", narration["pause_density"])
        audio = self.data["audio"]
        return {
            "expressiveness": expressiveness,
            "pace": pace,
            "pause_density": pauses,
            "gap_ms": audio.get("gap_ms") if audio.get("gap_ms") is not None else gap_ms(pauses),
            "chapter_tail_ms": audio.get("chapter_tail_ms")
            if audio.get("chapter_tail_ms") is not None
            else chapter_tail_ms(pauses),
        }

    def config(self, base: AppConfig) -> AppConfig:
        """The base config with this project's paths, voice, and tuning applied."""
        config = copy.deepcopy(base)
        root = self.root
        config.paths.artifacts = str(root)
        for key in ("ingest", "chunks", "prep", "tts", "audio"):
            setattr(config.paths, key, str(root / key))
        tuning = self.tuning()
        narration = self.data["narration"]
        voice = narration.get("voice") or config.tts.default_voice
        config.tts.default_voice = voice
        config.tts.default_voice_type = classify_voice_name(voice)
        if narration.get("lite"):
            config.tts.model = config.tts.lite_model
        config.prep.language = self.data.get("language") or config.prep.language
        config.prep.expressiveness = tuning["expressiveness"]
        config.prep.pace = tuning["pace"]
        config.prep.pause_density = tuning["pause_density"]
        config.prep.min_tags_per_100_words = tag_floor(tuning["expressiveness"])
        config.assemble.gap_ms = tuning["gap_ms"]
        config.assemble.chapter_tail_ms = tuning["chapter_tail_ms"]
        config.assemble.format = str(self.data["audio"].get("format") or "wav").lower()
        config.assemble.keep_wav = bool(self.data["audio"].get("keep_wav", False))
        config.ingest.skip_contents = True
        overrides = self.data.get("overrides") or {}
        if overrides:
            _merge_config(config, overrides)
        return config

    def profile(self) -> StyleProfile:
        profile = load_style_md(self.style_path)
        voices = dict(profile.character_voices)
        narration_voice = self.data["narration"].get("voice")
        if narration_voice:
            voices["narrator"] = VoiceMapping(narration_voice, classify_voice_name(narration_voice))
        for speaker, spec in (self.data.get("character_voices") or {}).items():
            voice = spec if isinstance(spec, str) else spec.get("voice", "")
            kind = "" if isinstance(spec, str) else spec.get("voice_type", "")
            voices[speaker] = VoiceMapping(voice, classify_voice_name(voice, kind))
        return replace(profile, character_voices=voices)


def load_style_md(path: Path) -> StyleProfile:
    if not path.exists():
        return StyleProfile(id="default")
    front, body = split_front_matter(path.read_text(encoding="utf-8"))
    profile = style_from_dict(front)
    notes = body.strip()
    return replace(profile, free_text=notes)


def render_style_md(front: dict, notes: str) -> str:
    header = yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=88).strip()
    return f"---\n{header}\n---\n\n{notes.strip()}\n"


def split_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("\n---", 1)
    if len(parts) != 2:
        return {}, text
    head = parts[0][3:]
    body = parts[1].lstrip("-").lstrip("\n") if parts[1].startswith("\n") else parts[1]
    data = yaml.safe_load(head) or {}
    return (data if isinstance(data, dict) else {}), body


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(*parts: object) -> str:
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def atomic_write(path: Path, text: str | bytes) -> None:
    """Write beside the target, then rename, so an interrupt never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(text if isinstance(text, bytes) else text.encode("utf-8"))
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def _merge(base: dict, stored: dict) -> dict:
    for key, value in stored.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict) and key not in {"character_voices", "overrides"}:
            _merge(base[key], value)
        else:
            base[key] = value
    return base
