"""Background music and sound effects through the project workflow, with fake model clients."""

import json
from pathlib import Path

import numpy as np

from expressive_audiobook import score
from expressive_audiobook.cli import main
from expressive_audiobook.config import load_config
from expressive_audiobook.library import Library
from expressive_audiobook.mixer import decode, write_wav
from test_live_pipeline import _valid_from_request
from test_project_workflow import STYLE

ROOT = Path(__file__).resolve().parents[1]
RATE = 44100


def story(chapters: int = 2) -> str:
    sentence = "The little fox crept to the old wooden door and listened to the rain. "
    parts = ["# The Fox\n"]
    for number in range(1, chapters + 1):
        parts.append(f"## Chapter {number}\n\n" + (sentence * 14 + "\n\n") * 3)
    return "\n".join(parts)


class Director:
    """Plays the text model: style, music supervisor, sound designer, and text prep."""

    def __init__(self):
        self.music_calls = 0
        self.sfx_calls = 0
        self.prep_calls = 0
        self.sfx_prompts = []

    def generate(self, request):
        text = request["contents"][0]["parts"][0]["text"]
        if "audiobook director" in text:
            return STYLE
        if "music supervisor" in text:
            self.music_calls += 1
            return {
                "palette": "warm felt piano and soft strings",
                "chapters": [
                    {"chapter": 1, "music": True, "reuse": None, "title": "Quiet night", "mood": "calm",
                     "prompt": "slow calm night music with felt piano"},
                    {"chapter": 2, "music": False, "reuse": None, "title": "", "mood": "", "prompt": ""},
                ],
            }
        if "sound designer" in text:
            self.sfx_calls += 1
            self.sfx_prompts.append(text)
            return {
                "cues": [
                    {"at": 1, "when": "before", "type": "spot", "through": None, "library_id": None, "title": "Door creak",
                     "prompt": "an old wooden door creaking open", "seconds": 2, "category": "household", "reason": "door"},
                    {"at": 2, "when": "before", "type": "ambience", "through": 3, "library_id": None, "title": "Soft rain",
                     "prompt": "gentle steady rain falling", "seconds": 8, "category": "weather", "reason": "rain"},
                    {"at": 99, "when": "before", "type": "spot", "through": None, "library_id": None, "title": "bad",
                     "prompt": "never", "seconds": 2, "category": "other", "reason": "out of range"},
                ]
            }
        self.prep_calls += 1
        return _valid_from_request(request, load_config(ROOT / "src" / "expressive_audiobook" / "default.toml"))


class Voice:
    def synthesize(self, request):
        t = np.arange(24000 * 2) / 24000
        return (np.sin(2 * np.pi * 180 * t) * 8000).astype("<i2").tobytes()


class FakeProvider:
    name = "fake"
    model = "fake-1"

    def __init__(self):
        self.requests = []

    def generate(self, request, destination):
        self.requests.append(request)
        seconds = min(request.seconds, 12.0)
        t = np.arange(int(seconds * RATE)) / RATE
        hz = 330 if request.kind == "music" else 900
        wave = (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
        write_wav(destination, np.stack([wave, wave], axis=1), RATE)
        return destination


def setup(tmp_path, monkeypatch, *, music=0.5, sfx=0.8, chapters=2):
    monkeypatch.setenv("AUDIOBOOK_LIBRARY", str(tmp_path / "library"))
    fake = FakeProvider()
    monkeypatch.setattr(score, "make_provider", lambda kind, cfg: fake)
    monkeypatch.setattr(score, "MIN_CUE_WORDS", 10)  # the fake narration is only a few words long
    monkeypatch.setattr(score, "MAX_CUE_WORDS", 20)
    source = tmp_path / "fox.md"
    source.write_text(story(chapters), encoding="utf-8")
    director = Director()
    projects = str(tmp_path / "projects")
    assert main(["import", "--source", str(source), "--name", "fox", "--projects-dir", projects], prep_client=director) == 0
    path = tmp_path / "projects" / "fox" / "project.json"
    data = json.loads(path.read_text())
    data["audio"].update(format="wav", keep_wav=True)
    data["soundscape"] = {"music": {"enabled": music > 0, "level": music, "style_hint": ""},
                          "sfx": {"enabled": sfx > 0, "intensity": sfx}, "audience": "kids"}
    data["overrides"] = {"sound": {"music_seconds": 12, "music_variants": 2, "bed_crossfade_s": 2, "preroll_s": 1.5, "tail_s": 2}}
    path.write_text(json.dumps(data))
    return director, fake, projects, tmp_path / "projects" / "fox"


def run(projects, director, *argv):
    return main([*argv, "--projects-dir", projects, "--project", "fox"], prep_client=director, tts_client=Voice())


def test_generate_mixes_music_and_effects_and_fills_the_library(tmp_path, monkeypatch):
    director, fake, projects, root = setup(tmp_path, monkeypatch)
    assert run(projects, director, "generate", "--all") == 0
    library = Library(tmp_path / "library")
    music = library.all("music")
    effects = library.all("sfx")
    assert len(music) == 2 and all(a.status == "candidate" for a in music)  # two variants for chapter 1 only
    assert {a.category for a in effects} == {"household", "weather"}
    plan = json.loads((root / "score" / "music.json").read_text())
    assert plan["chapters"]["1"]["music"] and not plan["chapters"]["2"]["music"]
    cues = json.loads((root / "score" / "sfx-001.json").read_text())["cues"]
    assert [c["at"] for c in cues] == [1, 2]  # the out-of-range cue is dropped
    assert any("no frightening" in p for p in director.sfx_prompts)  # kids rule reached the prompt

    chapter1 = decode(root / "audio" / "chapter-001.wav", RATE)
    chapter2 = decode(root / "audio" / "chapter-002.wav", RATE)
    book = decode(root / "audio" / "book.wav", RATE)
    assert np.isfinite(chapter1).all() and np.max(np.abs(chapter1)) <= 0.98
    assert abs(len(book) - len(chapter1) - len(chapter2)) < RATE

    # Lead-in: chapter 1 has music before the first word; chapter 2 has none.
    lead = chapter1[: int(1.0 * RATE)]
    assert np.sqrt(np.mean(lead**2)) > 1e-4
    assert np.abs(chapter2[: int(0.2 * RATE)]).max() > 0  # narration starts immediately


def test_second_run_reuses_plans_library_and_audio(tmp_path, monkeypatch):
    director, fake, projects, root = setup(tmp_path, monkeypatch)
    assert run(projects, director, "generate", "--all") == 0
    calls = (director.music_calls, director.sfx_calls, len(fake.requests))
    stamp = (root / "audio" / "chapter-001.wav").stat().st_mtime_ns
    assert run(projects, director, "generate", "--all") == 0
    assert (director.music_calls, director.sfx_calls, len(fake.requests)) == calls
    assert (root / "audio" / "chapter-001.wav").stat().st_mtime_ns == stamp


def test_changing_the_music_level_remixes_without_new_sounds(tmp_path, monkeypatch):
    director, fake, projects, root = setup(tmp_path, monkeypatch)
    assert run(projects, director, "generate", "--chapter", "1") == 0
    before = len(fake.requests)
    path = root / "project.json"
    data = json.loads(path.read_text())
    data["soundscape"]["music"]["level"] = 1.0
    path.write_text(json.dumps(data))
    stamp = (root / "audio" / "chapter-001.wav").stat().st_mtime_ns
    assert run(projects, director, "generate", "--chapter", "1") == 0
    assert len(fake.requests) == before
    assert (root / "audio" / "chapter-001.wav").stat().st_mtime_ns != stamp


def test_everything_is_off_by_default_and_a_dry_run_makes_nothing(tmp_path, monkeypatch, capsys):
    director, fake, projects, root = setup(tmp_path, monkeypatch, music=0, sfx=0)
    assert run(projects, director, "generate", "--chapter", "1") == 0
    assert director.music_calls == 0 and director.sfx_calls == 0 and not fake.requests
    assert not (root / "score").exists()

    (tmp_path / "b").mkdir()
    director, fake, projects, root = setup(tmp_path / "b", monkeypatch)
    assert run(projects, director, "generate", "--all", "--dry-run") == 0
    assert not fake.requests and director.music_calls == 0
    assert "music" in capsys.readouterr().out


def test_rejected_and_unapproved_sounds_are_not_mixed(tmp_path, monkeypatch):
    director, fake, projects, root = setup(tmp_path, monkeypatch, sfx=0.8)
    assert run(projects, director, "generate", "--chapter", "1") == 0
    library = Library(tmp_path / "library")
    for asset in library.all("music"):
        library.set_status(asset.id, "rejected")
    path = root / "project.json"
    data = json.loads(path.read_text())
    data["soundscape"]["music"]["level"] = 0.51  # force a remix
    path.write_text(json.dumps(data))
    assert run(projects, director, "generate", "--chapter", "1") == 0
    chapter = decode(root / "audio" / "chapter-001.wav", RATE)
    voice_only_seconds = len(chapter) / RATE
    assert voice_only_seconds < 80  # no music lead-in or tail beyond the narration


def _ui(*answers):
    import io

    from expressive_audiobook.ui import Ui

    queue = list(answers)
    return Ui(io.StringIO(), interactive=True, input_fn=lambda prompt: queue.pop(0))


def _project(tmp_path):
    from expressive_audiobook.project import Project

    return Project.create(tmp_path, "p", tmp_path / "x.md") if (tmp_path / "x.md").write_text("x") else None


def test_interactive_import_asks_and_non_interactive_stays_off(tmp_path):
    from expressive_audiobook.workflow import _choose_soundscape

    detected = {"audience": "kids", "soundscape": {"music": {"recommended": True, "style": "soft piano", "reason": "a bedtime story"},
                                                    "sfx": {"recommended": True, "reason": "has doors and rain"}}}
    project = _project(tmp_path)
    _choose_soundscape(_ui("", "y", "", "0.3", "y", "0.6"), project, detected, music=None, sfx=None, audience=None)
    scape = project.data["soundscape"]
    assert scape["audience"] == "kids"
    assert scape["music"] == {"enabled": True, "level": 0.3, "style_hint": "soft piano"}
    assert scape["sfx"] == {"enabled": True, "intensity": 0.6}

    quiet = _project(tmp_path / "q") if (tmp_path / "q").mkdir() is None else None
    from expressive_audiobook.ui import Ui
    import io

    _choose_soundscape(Ui(io.StringIO(), interactive=False), quiet, detected, music=None, sfx=None, audience=None)
    assert not quiet.data["soundscape"]["music"]["enabled"] and not quiet.data["soundscape"]["sfx"]["enabled"]
    flagged = _project(tmp_path / "f") if (tmp_path / "f").mkdir() is None else None
    _choose_soundscape(Ui(io.StringIO(), interactive=False), flagged, detected, music=0.5, sfx=0, audience="general")
    assert flagged.data["soundscape"]["music"]["enabled"] and not flagged.data["soundscape"]["sfx"]["enabled"]
    assert flagged.data["soundscape"]["audience"] == "general"


def test_sound_command_changes_settings(tmp_path, monkeypatch):
    director, fake, projects, root = setup(tmp_path, monkeypatch, music=0, sfx=0)
    assert run(projects, director, "sound", "--music", "0.7", "--audience", "kids", "--music-style", "harp") == 0
    scape = json.loads((root / "project.json").read_text())["soundscape"]
    assert scape["music"] == {"enabled": True, "level": 0.7, "style_hint": "harp"} and scape["audience"] == "kids"


def test_library_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AUDIOBOOK_LIBRARY", str(tmp_path / "library"))
    wav = tmp_path / "bell.wav"
    write_wav(wav, np.zeros((RATE, 2), dtype=np.float32) + 0.1, RATE)
    assert main(["library", "add", str(wav), "--kind", "sfx", "--title", "Bell", "--prompt", "a small bell"]) == 0
    library = Library(tmp_path / "library")
    (asset,) = library.all("sfx")
    assert asset.status == "approved" and abs(asset.seconds - 1.0) < 0.1
    assert main(["library", "list"]) == 0
    assert "Bell" in capsys.readouterr().out
    assert main(["library", "reject", asset.id]) == 0
    assert library.all("sfx", usable=True) == []
    assert main(["library", "rm", asset.id]) == 0 and library.all() == []
