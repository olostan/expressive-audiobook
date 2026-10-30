"""``audiobook doctor``: check the machine and the configuration, and say how to fix what is missing.

Only the engines you selected are required. A missing Gemini key is a problem when Gemini is used and a
note when it is not. Keys are never printed. Nothing is sent to a paid API; ``--online`` makes one
free request to confirm a Gemini key is accepted.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .config import AppConfig, env_value
from .home import audiobook_home, library_dir, models_dir
from .project import Project, list_projects
from .ui import Ui

GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1"


@dataclass
class Finding:
    level: str  # ok, info, warn, fail
    label: str
    detail: str = ""
    fix: str = ""


def fetch(url: str, headers: dict | None = None, timeout: float = 3.0) -> tuple[int, bytes]:
    """GET a URL. Raises OSError when nothing answers; HTTP error statuses are returned, not raised."""
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        raise OSError(str(exc.reason)) from exc


def _json_ids(raw: bytes) -> list[str]:
    import json

    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError:
        return []
    items = data.get("data") if isinstance(data, dict) else None
    if items is None and isinstance(data, dict):
        items = data.get("models")
    ids = []
    for item in items or []:
        if isinstance(item, dict):
            ids.append(str(item.get("id") or item.get("name") or item.get("model") or ""))
    return [name for name in ids if name]


class Doctor:
    def __init__(self, config: AppConfig, *, project: Project | None = None, online: bool = False,
                 projects_base: Path | None = None, http=fetch, which=shutil.which, run=subprocess.run):
        self.config = config
        self.project = project
        self.online = online
        self.projects_base = projects_base
        self.http = http
        self.which = which
        self.run = run
        self.findings: list[tuple[str, Finding]] = []
        self._section = ""

    # ------------------------------------------------------------------ helpers
    def section(self, title: str) -> None:
        self._section = title

    def add(self, level: str, label: str, detail: str = "", fix: str = "") -> None:
        self.findings.append((self._section, Finding(level, label, detail, fix)))

    def uses_gemini(self) -> bool:
        return self.config.text.provider == "gemini" or self.config.tts.provider == "gemini"

    # ------------------------------------------------------------------ checks
    def system(self) -> None:
        self.section("System")
        version = ".".join(str(part) for part in sys.version_info[:3])
        if sys.version_info >= (3, 11):
            self.add("ok", "Python", version)
        else:
            self.add("fail", "Python", f"{version}, need 3.11 or newer", "install Python 3.11+ and reinstall the package")
        self.add("ok", "audiobook", f"{__version__} on {platform.system()} {platform.machine()}")

    def keys(self) -> None:
        self.section("API keys")
        cfg = self.config
        env_file = Path(".env")
        self.add("info", ".env", f"found in {Path.cwd()}" if env_file.exists() else "none here (the environment is used)")
        needed = self.uses_gemini()
        used_by = " and ".join(
            name for name, provider in (("reading", cfg.text.provider), ("speech", cfg.tts.provider)) if provider == "gemini"
        )
        if env_value("GEMINI_API_KEY"):
            self.add("ok", "GEMINI_API_KEY", f"set (used for {used_by})" if needed else "set (not used by your current engines)")
            if self.online and needed:
                self._check_gemini_online()
        elif needed:
            self.add("fail", "GEMINI_API_KEY", f"missing, needed for {used_by}",
                     "put it in .env (see .env.example), or run locally: text.provider = \"openai\" and tts.provider = \"orpheus\"")
        else:
            self.add("info", "GEMINI_API_KEY", "not set, and not needed with your engines")
        sfx_uses_eleven = cfg.sound.sfx_provider == "elevenlabs"
        if env_value("ELEVENLABS_API_KEY"):
            self.add("ok", "ELEVENLABS_API_KEY", "set" if sfx_uses_eleven else "set (not used: sound effects are made locally)")
        elif sfx_uses_eleven:
            self.add("fail", "ELEVENLABS_API_KEY", "missing, needed because sound.sfx_provider is \"elevenlabs\"",
                     "put it in .env, or set sound.sfx_provider back to \"sa3\"")
        else:
            self.add("info", "ELEVENLABS_API_KEY", "not set, and not needed")

    def _check_gemini_online(self) -> None:
        key = env_value("GEMINI_API_KEY") or ""
        try:
            status, _ = self.http(GEMINI_MODELS_URL, {"x-goog-api-key": key})
        except OSError as exc:
            self.add("warn", "Gemini API", f"could not be reached ({exc})", "check your internet connection")
            return
        if status == 200:
            self.add("ok", "Gemini API", "reachable, key accepted")
        elif status in (400, 401, 403):
            self.add("fail", "Gemini API", f"key rejected (HTTP {status})", "create a new key at https://aistudio.google.com/apikey")
        else:
            self.add("warn", "Gemini API", f"answered HTTP {status}")

    def text_model(self) -> None:
        self.section("Reading the book")
        settings = self.config.text
        if settings.provider == "gemini":
            if env_value("GEMINI_API_KEY"):
                self.add("ok", "text.provider = gemini", f"model {self.config.prep.model}")
            else:  # the missing key is already reported once, under API keys
                self.add("warn", "text.provider = gemini", "waiting for GEMINI_API_KEY")
            return
        if settings.provider != "openai":
            self.add("fail", f"text.provider = {settings.provider}", "unknown", "use \"gemini\" or \"openai\"")
            return
        self.add("info", "text.provider = openai", f"{settings.base_url}, model {settings.model or '(not set)'}")
        if not settings.model:
            self.add("fail", "text.model", "not set", "name the model your server runs, for example text.model = \"qwen3:14b\"")
        headers = {}
        key = env_value(settings.api_key_env)
        if settings.api_key_env and not key:
            self.add("fail", settings.api_key_env, "named by text.api_key_env but not set", "put it in .env")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            status, raw = self.http(settings.base_url.rstrip("/") + "/models", headers)
        except OSError as exc:
            self.add("fail", "text server", f"not reachable at {settings.base_url} ({exc})",
                     "start it (Ollama: `ollama serve`, or open the app), or fix text.base_url")
            return
        if status >= 400:
            self.add("fail", "text server", f"answered HTTP {status}", "check text.base_url and the API key")
            return
        self.add("ok", "text server", "reachable")
        names = _json_ids(raw)
        if settings.model and names:
            if any(settings.model == n or n.startswith(settings.model + ":") or settings.model.startswith(n + ":") for n in names):
                self.add("ok", "text.model", f"{settings.model} is available")
            else:
                self.add("fail", "text.model", f"{settings.model} is not installed on the server",
                         f"Ollama: `ollama pull {settings.model}`. Installed: {', '.join(names[:6]) or 'none'}")
        elif settings.model:
            self.add("warn", "text.model", "the server did not list its models, so the name was not checked")

    def speech(self) -> None:
        self.section("Speaking")
        tts = self.config.tts
        if tts.provider == "gemini":
            if env_value("GEMINI_API_KEY"):
                self.add("ok", "tts.provider = gemini", f"model {tts.model}")
            else:
                self.add("warn", "tts.provider = gemini", "waiting for GEMINI_API_KEY")
        elif tts.provider == "orpheus":
            self._orpheus()
        elif tts.provider == "http":
            self._speech_http()
        else:
            self.add("fail", f"tts.provider = {tts.provider}", "unknown", "use \"gemini\", \"orpheus\" or \"http\"")

    def _orpheus(self) -> None:
        from .speech_local import orpheus_dir

        settings = self.config.tts.orpheus
        self.add("info", "tts.provider = orpheus", f"voice {settings.default_voice}, model {settings.model}")
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            self.add("fail", "Apple Silicon", f"this is {platform.system()} {platform.machine()}; the local Orpheus engine needs an Apple Silicon Mac",
                     "use tts.provider = \"http\" with an Orpheus server, or \"gemini\"")
        root = orpheus_dir(settings)
        python = root / ".venv" / "bin" / "python"
        if not python.exists():
            self.add("fail", "Orpheus install", f"not found at {root}", "run: audiobook models install orpheus")
            return
        done = self.run([str(python), "-c", "import mlx_audio"], capture_output=True, text=True)
        if done.returncode == 0:
            self.add("ok", "Orpheus install", f"mlx-audio ready in {root}")
        else:
            self.add("fail", "Orpheus install", "the environment exists but mlx-audio does not import", "run: audiobook models install orpheus")
        cache = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub" / ("models--" + settings.model.replace("/", "--"))
        if cache.exists():
            self.add("ok", "Orpheus model", "downloaded")
        else:
            self.add("warn", "Orpheus model", "not downloaded yet (about 7 GB)", "run: audiobook models install orpheus")

    def _speech_http(self) -> None:
        settings = self.config.tts.http
        self.add("info", "tts.provider = http", f"{settings.base_url}, model {settings.model}")
        try:
            status, _ = self.http(settings.base_url.rstrip("/") + "/models", {})
        except OSError as exc:
            self.add("fail", "speech server", f"not reachable at {settings.base_url} ({exc})", "start the server, or fix tts.http.base_url")
            return
        self.add("ok" if status < 500 else "warn", "speech server", f"answered HTTP {status}")

    def tools(self) -> None:
        self.section("Tools")
        want_sound = False
        if self.project is not None:
            scape = self.project.data.get("soundscape") or {}
            want_sound = bool((scape.get("music") or {}).get("enabled") or (scape.get("sfx") or {}).get("enabled"))
        if self.which("ffmpeg"):
            self.add("ok", "ffmpeg", "found (MP3/M4A output and sound mixing)")
        else:
            self.add("warn" if want_sound else "info", "ffmpeg",
                     "not found" + (": music and effects cannot be mixed" if want_sound else ": audio is written as WAV only"),
                     "brew install ffmpeg")
        if importlib.util.find_spec("numpy"):
            self.add("ok", "numpy", "found (sound mixing)")
        else:
            self.add("warn" if want_sound else "info", "numpy", "not installed: needed to mix music and effects", 'pip install -e ".[sound]"')
        if self.which("tesseract"):
            self.add("ok", "tesseract", "found (OCR for scanned pages)")
        else:
            self.add("info", "tesseract", "not found: only needed for scanned pages")
        from .soundgen import sa3_dir

        sa3 = sa3_dir(self.config.sound)
        if (sa3 / ".venv" / "bin" / "python").exists():
            self.add("ok", "Stable Audio 3", "installed (local music and sound effects)")
        else:
            self.add("warn" if want_sound else "info", "Stable Audio 3",
                     "not installed" + (": this project wants music or effects" if want_sound else ": only needed for background music and effects"),
                     "audiobook models install sa3")

    def places(self) -> None:
        self.section("Where things live")
        self.add("info", "home", str(audiobook_home()))
        self.add("info", "models", str(models_dir()))
        self.add("info", "library", str(library_dir()))
        if self.projects_base is not None:
            names = list_projects(self.projects_base)
            self.add("info", "projects", f"{self.projects_base} ({len(names)}: {', '.join(names) or 'none'})")
        if self.project is not None:
            overrides = self.project.data.get("overrides") or {}
            self.add("info", "project", f"{self.project.name}" + (f", overrides: {', '.join(overrides)}" if overrides else ""))

    def run_all(self) -> list[tuple[str, Finding]]:
        self.system()
        self.keys()
        self.text_model()
        self.speech()
        self.tools()
        self.places()
        return self.findings


def run_doctor(config: AppConfig, ui: Ui, *, project: Project | None = None, online: bool = False,
               projects_base: Path | None = None, **probes) -> int:
    """Print the report. Returns 1 when something required is missing, otherwise 0."""
    doctor = Doctor(config, project=project, online=online, projects_base=projects_base, **probes)
    findings = doctor.run_all()
    ui.banner("🩺", "audiobook doctor", f"project {project.name}" if project else "")
    section = None
    for name, item in findings:
        if name != section:
            section = name
            ui._emit(f"\n   {ui.paint(name, 'bold')}")
        mark = {"ok": ui.paint("✓", "green", "bold"), "info": ui.paint("·", "dim"),
                "warn": ui.paint("⚠", "yellow", "bold"), "fail": ui.paint("✗", "red", "bold")}[item.level]
        detail = f"  {ui.paint(item.detail, 'dim')}" if item.detail else ""
        ui._emit(f"   {mark} {item.label}{detail}")
        if item.fix and item.level in ("warn", "fail"):
            ui.hint(item.fix)
    failures = sum(1 for _, item in findings if item.level == "fail")
    warnings = sum(1 for _, item in findings if item.level == "warn")
    ui._emit("")
    if failures:
        ui.error(f"{failures} problem{'s' if failures != 1 else ''} to fix before you can generate audio"
                 + (f", {warnings} warning{'s' if warnings != 1 else ''}" if warnings else ""))
        return 1
    ui.ok("Ready to go" + (f" ({warnings} warning{'s' if warnings != 1 else ''})" if warnings else ""))
    return 0
