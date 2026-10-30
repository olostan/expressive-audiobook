"""Providers that turn a text prompt into music or a sound effect.

Every provider writes an audio file and returns its path; the library converts
it to FLAC. ``sa3`` runs Stability's Stable Audio 3 locally through MLX (free, a
few seconds per clip on Apple Silicon). ``elevenlabs`` calls the ElevenLabs
sound-effects API (a paid account, ``ELEVENLABS_API_KEY``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from .config import SoundConfig, load_env_file
from .home import models_dir


class SoundGenError(RuntimeError):
    pass


@dataclass(frozen=True)
class SoundRequest:
    kind: str  # "music" or "sfx"
    prompt: str
    seconds: float
    seed: int = 0
    loop: bool = False
    negative: str = ""


class SoundProvider(Protocol):
    name: str
    model: str

    def generate(self, request: SoundRequest, destination: Path) -> Path: ...


def sa3_dir(config: SoundConfig) -> Path:
    return Path(config.sa3_dir).expanduser() if config.sa3_dir else models_dir() / "stable-audio-3" / "optimized" / "mlx"


class Sa3Provider:
    """Stable Audio 3, run as a subprocess in its own virtual environment."""

    name = "sa3"

    def __init__(self, config: SoundConfig, *, runner=subprocess.run):
        self.config = config
        self.root = sa3_dir(config)
        self._runner = runner
        self.model = f"sa3:{config.sa3_music_dit}/{config.sa3_sfx_dit}"

    def installed(self) -> bool:
        return (self.root / ".venv" / "bin" / "python").exists() and (self.root / "scripts" / "sa3_mlx.py").exists()

    def generate(self, request: SoundRequest, destination: Path) -> Path:
        if not self.installed():
            raise SoundGenError(
                f"Stable Audio 3 is not installed at {self.root}. Run: audiobook models install sa3"
            )
        dit, decoder = (
            (self.config.sa3_music_dit, self.config.sa3_music_decoder)
            if request.kind == "music"
            else (self.config.sa3_sfx_dit, self.config.sa3_sfx_decoder)
        )
        command = [
            str(self.root / ".venv" / "bin" / "python"),
            str(self.root / "scripts" / "sa3_mlx.py"),
            "--prompt", request.prompt,
            "--dit", dit,
            "--decoder", decoder,
            "--seconds", f"{request.seconds:g}",
            "--seed", str(request.seed),
            "--out", str(destination),
        ]
        if request.negative:
            command += ["--negative-prompt", request.negative]
        destination.parent.mkdir(parents=True, exist_ok=True)
        done = self._runner(command, capture_output=True, text=True, cwd=str(self.root))
        if done.returncode != 0 or not destination.exists():
            tail = (done.stderr or done.stdout or "").strip()[-400:]
            raise SoundGenError(f"Stable Audio 3 failed ({done.returncode}): {tail}")
        return destination


class ElevenLabsProvider:
    """Sound effects only. Music is not offered by this provider."""

    name = "elevenlabs"
    URL = "https://api.elevenlabs.io/v1/sound-generation"

    def __init__(self, config: SoundConfig, *, send=None):
        self.config = config
        self.model = config.elevenlabs_model
        self._send = send or self._post

    def generate(self, request: SoundRequest, destination: Path) -> Path:
        if request.kind != "sfx":
            raise SoundGenError("ElevenLabs is configured for sound effects only")
        load_env_file()
        key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not key:
            raise SoundGenError("ELEVENLABS_API_KEY is not set")
        body = {
            "text": request.prompt,
            "duration_seconds": min(30.0, max(0.5, request.seconds)),
            "prompt_influence": 0.5,
            "model_id": self.model,
            "loop": bool(request.loop),
        }
        data = self._send(key, body)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return destination

    def _post(self, key: str, body: dict) -> bytes:
        try:
            reply = httpx.post(
                self.URL,
                params={"output_format": "mp3_44100_128"},
                headers={"xi-api-key": key},
                json=body,
                timeout=120,
            )
        except httpx.HTTPError as exc:
            raise SoundGenError(f"ElevenLabs request failed: {exc}") from exc
        if reply.status_code != 200:
            raise SoundGenError(f"ElevenLabs returned {reply.status_code}: {reply.text[:300]}")
        return reply.content


def make_provider(kind: str, config: SoundConfig):
    name = config.music_provider if kind == "music" else config.sfx_provider
    if name == "sa3":
        return Sa3Provider(config)
    if name == "elevenlabs":
        return ElevenLabsProvider(config)
    raise SoundGenError(f"Unknown {kind} provider {name!r}. Use sa3" + (" or elevenlabs" if kind == "sfx" else ""))


def install_sa3(config: SoundConfig, *, bundles: str = "sm-music,sm-sfx,medium", out=sys.stdout) -> Path:
    """Clone Stability's repo and run its installer (a venv with MLX plus the weights)."""
    root = sa3_dir(config)
    repo = root.parents[1]
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/Stability-AI/stable-audio-3.git", str(repo)], check=True)
    subprocess.run([str(root / "install.sh"), "-y", "--download", bundles], check=True, cwd=str(root))
    return root
