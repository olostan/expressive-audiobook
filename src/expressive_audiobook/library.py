"""A shared library of generated music and sound effects.

Lives in ``~/.audiobook/library`` so every book can reuse it::

    music/<id>.flac  <id>.json
    sfx/<id>.flac    <id>.json

An id is a slug plus a hash of what produced the sound, so asking for the same
prompt, provider, and seed again is a cache hit and costs nothing. Anything you
generate starts as a ``candidate``; ``approved`` and ``rejected`` are yours to set.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .export import ffmpeg_path
from .home import library_dir
from .project import atomic_write, fingerprint

KINDS = ("music", "sfx")
STATUSES = ("candidate", "approved", "rejected")
SFX_CATEGORIES = ("ambience", "foley", "impact", "creature", "magic", "weather", "household", "vehicle", "other")


class LibraryError(RuntimeError):
    pass


@dataclass
class Asset:
    id: str
    kind: str
    prompt: str
    title: str = ""
    tags: list[str] = field(default_factory=list)
    category: str = ""  # sfx only
    mood: str = ""  # music only
    seconds: float = 0.0
    loop: bool = False
    provider: str = ""
    model: str = ""
    seed: int | None = None
    status: str = "candidate"
    origin: str = ""  # project that first asked for it
    created: str = ""
    file: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Asset":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def line(self) -> str:
        """One line for a model prompt or a listing."""
        bits = [self.id, f"{self.seconds:.0f}s"]
        if self.category:
            bits.append(self.category)
        if self.mood:
            bits.append(self.mood)
        if self.loop:
            bits.append("loops")
        return f"{' | '.join(bits)} | {self.title or self.prompt}"[:160]


def slugify(text: str, words: int = 4) -> str:
    parts = re.findall(r"[a-z0-9]+", text.lower())[:words]
    return "-".join(parts) or "sound"


def asset_id(kind: str, prompt: str, provider: str, model: str, seed: int | None, seconds: float, loop: bool) -> str:
    stamp = fingerprint(kind, prompt, provider, model, seed, round(seconds, 1), loop)[:8]
    return f"{kind}-{slugify(prompt)}-{stamp}"


class Library:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else library_dir()

    def _dir(self, kind: str) -> Path:
        if kind not in KINDS:
            raise LibraryError(f"kind must be one of {', '.join(KINDS)}")
        return self.root / kind

    def path(self, asset: Asset) -> Path:
        return self._dir(asset.kind) / asset.file

    def get(self, ident: str) -> Asset | None:
        kind = ident.split("-", 1)[0]
        if kind not in KINDS:
            return None
        sidecar = self._dir(kind) / f"{ident}.json"
        if not sidecar.exists():
            return None
        return Asset.from_dict(json.loads(sidecar.read_text(encoding="utf-8")))

    def all(self, kind: str | None = None, *, usable: bool = False) -> list[Asset]:
        found: list[Asset] = []
        for k in ([kind] if kind else KINDS):
            folder = self._dir(k)
            if not folder.exists():
                continue
            for sidecar in sorted(folder.glob("*.json")):
                try:
                    asset = Asset.from_dict(json.loads(sidecar.read_text(encoding="utf-8")))
                except (OSError, ValueError, TypeError):
                    continue
                if (folder / asset.file).exists() and (not usable or asset.status != "rejected"):
                    found.append(asset)
        return found

    def add(self, source: Path | str, asset: Asset) -> Asset:
        """Copy ``source`` into the library as a FLAC and write its sidecar. Replaces an item with the same id."""
        folder = self._dir(asset.kind)
        folder.mkdir(parents=True, exist_ok=True)
        asset.file = f"{asset.id}.flac"
        asset.created = asset.created or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        destination = folder / asset.file
        temp = destination.with_suffix(".tmp.flac")
        to_flac(Path(source), temp)
        temp.replace(destination)
        atomic_write(folder / f"{asset.id}.json", json.dumps(asdict(asset), indent=2, ensure_ascii=False) + "\n")
        return asset

    def set_status(self, ident: str, status: str) -> Asset:
        if status not in STATUSES:
            raise LibraryError(f"status must be one of {', '.join(STATUSES)}")
        asset = self.get(ident)
        if asset is None:
            raise LibraryError(f"No library item {ident!r}")
        asset.status = status
        atomic_write(self._dir(asset.kind) / f"{asset.id}.json", json.dumps(asdict(asset), indent=2, ensure_ascii=False) + "\n")
        return asset

    def remove(self, ident: str) -> None:
        asset = self.get(ident)
        if asset is None:
            raise LibraryError(f"No library item {ident!r}")
        folder = self._dir(asset.kind)
        (folder / asset.file).unlink(missing_ok=True)
        (folder / f"{asset.id}.json").unlink(missing_ok=True)

    def catalog(self, kind: str, limit: int = 120) -> str:
        """Text listing for a model prompt: every usable item, newest last."""
        rows = [asset.line() for asset in self.all(kind, usable=True)][-limit:]
        return "\n".join(rows) or "(empty)"


def to_flac(source: Path, destination: Path, sample_rate: int = 44100) -> None:
    """Normalize any audio file to 44.1 kHz stereo FLAC with ffmpeg."""
    ffmpeg = ffmpeg_path()
    if ffmpeg is None:
        raise LibraryError("ffmpeg is required for music and sound effects (brew install ffmpeg)")
    if not Path(source).exists():
        raise LibraryError(f"Audio file not found: {source}")
    command = [ffmpeg, "-y", "-v", "error", "-i", str(source), "-ar", str(sample_rate), "-ac", "2", "-c:a", "flac", "-f", "flac", str(destination)]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise LibraryError(f"ffmpeg could not read {source}: {done.stderr.strip()[-300:]}")


def copy_out(asset: Asset, library: Library, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(library.path(asset), destination)
    return destination
