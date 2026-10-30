"""Every audiobook command in the README must run."""

import shlex
from pathlib import Path

import pytest

from expressive_audiobook.cli import main
from pdf_factory import build_outline_pdf
from test_live_pipeline import _valid_from_request, _write_config

ROOT = Path(__file__).resolve().parents[1]


class Prep:
    def generate(self, request):
        from expressive_audiobook.config import load_config

        return _valid_from_request(request, load_config(ROOT / "config" / "default.toml"))


class Speech:
    def __init__(self):
        self.seen = []

    def synthesize(self, request):
        self.seen.append(request.speakers[0]["voice"])
        return b"\x00\x00" * 2400

    def list_voices(self, **filters):
        return [
            {
                "id": "Schedar",
                "display_name": "Schedar",
                "type": "prebuilt",
                "gender": "neutral",
                "language_code": "en-US",
            },
            {
                "id": "Charon",
                "display_name": "Charon",
                "type": "prebuilt",
                "gender": "male",
            },
        ]

    def design_voice(self, description, **kwargs):
        assert description.strip()
        return "voice_readme123"


def _commands(readme: str) -> list[str]:
    commands = []
    fence = None
    for line in readme.splitlines():
        if line.startswith("```"):
            fence = line[3:].strip() if fence is None else None
            continue
        if fence == "bash" and line.startswith("audiobook "):
            commands.append(line.strip())
    return commands


def test_readme_audiobook_commands_run(tmp_path: Path):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    commands = _commands(readme)
    assert len(commands) >= 15
    pdf = tmp_path / "owned-book.pdf"
    build_outline_pdf(pdf)
    config = _write_config(tmp_path)
    prep = Prep()
    speech = Speech()
    for line in commands:
        argv = shlex.split(line.replace("/path/to/book.pdf", str(pdf)))
        assert argv[0] == "audiobook"
        if argv[1:] == ["--version"]:
            with pytest.raises(SystemExit) as caught:
                main(["--config", str(config), "--version"], prep_client=prep, tts_client=speech)
            assert caught.value.code in (None, 0)
            continue
        code = main(["--config", str(config), *argv[1:]], prep_client=prep, tts_client=speech)
        assert code == 0, line
    assert "Schedar" in speech.seen
    assert "Charon" in speech.seen
    assert "Sulafat" in speech.seen
    samples = list((tmp_path / "artifacts").glob("owned-book/samples/*.wav"))
    assert samples
    assert not any(path.name.startswith("chapter-") for path in samples)
    chapter = tmp_path / "artifacts" / "audio" / "chapter-001.wav"
    assert chapter.exists()
    review = tmp_path / "artifacts" / "prep" / "review" / "chapter-001.md"
    assert review.exists()
    assert "Ordinary narration needs little color." in review.read_text(encoding="utf-8")
