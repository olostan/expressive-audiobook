"""Speech engines that are not Gemini: Orpheus run locally, and any OpenAI-style speech server.

Both engines speak one turn at a time. The ``<short pause>`` tag is turned into real silence here, so
a pause is exact and the model never sees it; every other tag (``<laugh>``, ``<sigh>``, ...) goes to
the model as written. ``style`` is not used: Orpheus has no style control.
"""

from __future__ import annotations

import atexit
from abc import ABC, abstractmethod
import base64
import hashlib
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from .audio import load_pcm
from .config import AppConfig, OrpheusConfig, SpeechHttpConfig, env_value
from .gemini_api import RetryableHttpError, call_with_backoff
from .home import models_dir
from .models import TtsRequest

_PAUSE = re.compile(r"\s*<\s*short pause\s*>\s*", re.IGNORECASE)


class SpeechError(RuntimeError):
    pass


def split_pauses(text: str) -> list[str | None]:
    """Spoken pieces in order, with None where a pause goes. Empty pieces are dropped."""
    pieces: list[str | None] = []
    for index, part in enumerate(_PAUSE.split(text)):
        if index:
            pieces.append(None)
        if part.strip():
            pieces.append(part.strip())
    while pieces and pieces[0] is None:
        pieces.pop(0)
    while pieces and pieces[-1] is None:
        pieces.pop()
    return pieces


def resolve_voice(voice: str, speaker: str, voices: list[str], default: str) -> str:
    """A voice this engine has. Gemini names from a project map to the narrator voice, or a stable pick for a character."""
    if voice and voice.lower() in voices:
        return voice.lower()
    if speaker.lower() == "narrator":
        return default
    others = [name for name in voices if name != default] or voices
    pick = int(hashlib.sha1(speaker.lower().encode("utf-8")).hexdigest(), 16) % len(others)
    return others[pick]


def silence_pcm(milliseconds: float, sample_rate: int) -> bytes:
    return b"\x00" * (int(round(sample_rate * milliseconds / 1000.0)) * 2)


class SegmentedSpeech(ABC):
    """Shared turn handling; subclasses provide ``_speak(voice, text) -> 16-bit mono PCM``."""

    voices: list[str]
    default_voice: str
    pause_ms: float
    sample_rate: int

    def synthesize(self, request: TtsRequest) -> bytes:
        item = request.speakers[0]
        voice = self.voice_for(str(item.get("voice") or ""), str(item.get("speaker") or "narrator"))
        out = bytearray()
        for turn in request.turns:
            for piece in split_pauses(str(turn.get("text") or "")):
                out += silence_pcm(self.pause_ms, self.sample_rate) if piece is None else self._speak(voice, piece)
        if not out:
            raise SpeechError("nothing to speak in this turn")
        return bytes(out)

    def voice_for(self, voice: str, speaker: str) -> str:
        return resolve_voice(voice, speaker, self.voices, self.default_voice)

    def list_voices(self, **filters) -> list[dict]:
        search = (filters.get("search") or "").lower()
        return [
            {"id": name, "display_name": name, "type": "prebuilt", "gender": "", "language_code": "en-US"}
            for name in self.voices
            if search in name
        ]

    def design_voice(self, *args, **kwargs) -> str:
        raise SpeechError("Designing voices is a Gemini feature. This speech engine has a fixed set of voices.")

    @abstractmethod
    def _speak(self, voice: str, text: str) -> bytes:
        """16-bit mono PCM at ``sample_rate`` for one piece of text."""


def orpheus_dir(config: OrpheusConfig) -> Path:
    return Path(config.dir).expanduser() if config.dir else models_dir() / "orpheus"


class OrpheusWorker:
    """The model, loaded once in a child process inside the Orpheus virtual environment."""

    def __init__(self, config: OrpheusConfig, *, popen=subprocess.Popen):
        self.config = config
        self.root = orpheus_dir(config)
        self._popen = popen
        self._process = None
        self._lock = threading.Lock()
        self.sample_rate = 24000

    def installed(self) -> bool:
        return (self.root / ".venv" / "bin" / "python").exists()

    def _start(self):
        if not self.installed():
            raise SpeechError(f"Orpheus is not installed at {self.root}. Run: audiobook models install orpheus")
        script = Path(__file__).resolve().with_name("orpheus_worker.py")
        self._process = self._popen(
            [str(self.root / ".venv" / "bin" / "python"), str(script), "--model", self.config.model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
        )
        atexit.register(self.close)
        hello = self._read()
        if "error" in hello:
            self.close()
            raise SpeechError(hello["error"])
        self.sample_rate = int(hello.get("sample_rate") or self.sample_rate)

    def _read(self) -> dict:
        line = self._process.stdout.readline()
        if not line:
            raise SpeechError("The Orpheus process stopped unexpectedly. Run the command again to restart it.")
        return json.loads(line)

    def speak(self, voice: str, text: str) -> bytes:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._start()
            job = {
                "text": text,
                "voice": voice,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "repetition_penalty": self.config.repetition_penalty,
                "max_tokens": self.config.max_tokens,
            }
            self._process.stdin.write(json.dumps(job) + "\n")
            self._process.stdin.flush()
            answer = self._read()
        if not answer.get("ok"):
            raise SpeechError(f"Orpheus failed: {answer.get('error')}")
        return base64.b64decode(answer["pcm"])

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            try:
                process.stdin.close()
                process.wait(timeout=10)
            except Exception:
                process.kill()


def install_orpheus(config: OrpheusConfig, *, run=subprocess.run) -> Path:
    """Create a virtual environment with mlx-audio and download the model (about 7 GB)."""
    root = orpheus_dir(config)
    root.mkdir(parents=True, exist_ok=True)
    python = root / ".venv" / "bin" / "python"
    if not python.exists():
        run([sys.executable, "-m", "venv", str(root / ".venv")], check=True)
    run([str(python), "-m", "pip", "install", "--upgrade", "pip", "mlx-audio"], check=True)
    run(
        [str(python), "-c", "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])", config.model],
        check=True,
    )
    return root


class OrpheusTtsClient(SegmentedSpeech):
    def __init__(self, config: AppConfig, worker: OrpheusWorker | None = None):
        settings = config.tts.orpheus
        self.worker = worker or OrpheusWorker(settings)
        self.voices = [name.lower() for name in settings.voices]
        self.default_voice = settings.default_voice.lower()
        self.pause_ms = settings.pause_ms
        self.sample_rate = config.tts.sample_rate

    def _speak(self, voice: str, text: str) -> bytes:
        return self.worker.speak(voice, text)


class SpeechHttpClient(SegmentedSpeech):
    """POST {base_url}/audio/speech, the OpenAI shape that Orpheus-FastAPI, Kokoro-FastAPI and others serve."""

    def __init__(self, config: AppConfig, sender=None, sleep=None):
        self.settings: SpeechHttpConfig = config.tts.http
        self.voices = [self.settings.default_voice.lower()]
        self.default_voice = self.settings.default_voice.lower()
        self.pause_ms = self.settings.pause_ms
        self.sample_rate = config.tts.sample_rate
        self.retries = config.tts.max_http_retries
        self.sender = sender
        self.sleep = sleep

    def voice_for(self, voice: str, speaker: str) -> str:
        # The server knows its own voices, so pass the project's name through unless it is a Gemini voice.
        from .tts import PREBUILT_VOICES

        return voice if voice and voice not in PREBUILT_VOICES else self.default_voice

    def _speak(self, voice: str, text: str) -> bytes:
        url = self.settings.base_url.rstrip("/") + "/audio/speech"
        headers = {"Content-Type": "application/json"}
        key = env_value(self.settings.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = {"model": self.settings.model, "input": text, "voice": voice, "response_format": self.settings.response_format}
        send = self.sender or self._send

        def once() -> bytes:
            status, raw = send("POST", url, headers, body)
            if status == 429 or 500 <= status <= 599:
                raise RetryableHttpError(status, raw[:300].decode("utf-8", "replace"))
            if status >= 400:
                raise SpeechError(f"Speech server error {status}: {raw[:300].decode('utf-8', 'replace')}")
            return load_pcm(raw, expected_rate=self.sample_rate).pcm

        return call_with_backoff(
            once, max_retries=self.retries, initial=1.0, multiplier=2.0, sleep=self.sleep or time.sleep
        )

    def _send(self, method: str, url: str, headers: dict, body: dict) -> tuple[int, bytes]:
        import httpx

        try:
            with httpx.Client(timeout=httpx.Timeout(self.settings.timeout_seconds, connect=10.0)) as client:
                response = client.request(method, url, headers=headers, json=body)
                return response.status_code, response.content
        except httpx.ConnectError as exc:
            raise SpeechError(f"Could not reach the speech server at {url}. Is it running? ({exc})") from exc
