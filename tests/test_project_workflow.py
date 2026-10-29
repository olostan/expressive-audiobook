"""import -> prepare -> generate on a project, with mocked Gemini clients."""

import json
from pathlib import Path

from pdf_expressive_audiobook.cli import main
from pdf_expressive_audiobook.config import load_config
from pdf_expressive_audiobook.prep import flat_delivery_errors
from pdf_expressive_audiobook.project import Project, load_style_md
from pdf_factory import build_outline_pdf
from test_live_pipeline import _valid_from_request

ROOT = Path(__file__).resolve().parents[1]

STYLE = {
    "genre": "test fiction",
    "tone": "dry",
    "summary": "A synthetic fixture.",
    "narrator_persona": "A dry, careful narrator.",
    "default_delivery_style": "dry and careful",
    "literary_traits": ["Keep the asides."],
    "speakers": ["Guide"],
    "pronunciation_lexicon": [{"word": "Quillan", "ipa": "/ˈkwɪlən/"}],
    "verbalize": {"formulas": "Say the math."},
    "recommended": {"expressiveness": "high", "pace": "slow", "pause_density": "generous"},
    "notes": "Slow down for the gate scene.",
}


class Prep:
    def __init__(self):
        self.calls = 0
        self.styled = 0
        self.requests = []

    def generate(self, request):
        text = request["contents"][0]["parts"][0]["text"]
        if "preparing an audiobook of the book excerpted" in text:
            self.styled += 1
            return STYLE
        self.calls += 1
        self.requests.append(text)
        return _valid_from_request(request, load_config(ROOT / "config" / "default.toml"))


class Speech:
    def __init__(self):
        self.calls = 0

    def synthesize(self, request):
        self.calls += 1
        return b"\x01\x00" * 2400


def _run(tmp_path, *argv, prep, speech):
    return main(
        [*argv, "--projects-dir", str(tmp_path / "projects"), "--project", "owned"],
        prep_client=prep,
        tts_client=speech,
    )


def _import(tmp_path, prep):
    pdf = tmp_path / "book.pdf"
    if not pdf.exists():
        build_outline_pdf(pdf)
    code = main(
        ["import", "--pdf", str(pdf), "--name", "owned", "--projects-dir", str(tmp_path / "projects")],
        prep_client=prep,
    )
    assert code == 0
    return tmp_path / "projects" / "owned"


def test_import_writes_project_style_and_chapters(tmp_path):
    prep = Prep()
    root = _import(tmp_path, prep)
    project = Project.load(root)
    assert project.data["source_sha256"] and project.data["title"]
    assert project.data["narration"]["expressiveness"] == "high"
    assert project.data["narration"]["pace"] == "slow"
    assert (root / "ingest" / "manifest.json").exists()
    assert len(list((root / "ingest" / "chapters").glob("*.json"))) == 3
    manifest = json.loads((root / "ingest" / "manifest.json").read_text())
    assert manifest["skipped_pages"] == [{"page": 1, "reason": "contents"}]
    profile = load_style_md(root / "style.md")
    assert profile.narrator_persona.startswith("A dry")
    assert profile.pronunciation_lexicon[0].word == "Quillan"
    assert "Slow down for the gate scene." in profile.free_text
    # Importing the same PDF again keeps edits and does not call Gemini again.
    (root / "style.md").write_text((root / "style.md").read_text() + "\nMY EDIT\n")
    _import(tmp_path, prep)
    assert prep.styled == 1
    assert "MY EDIT" in (root / "style.md").read_text()


def test_prepare_generate_sample_is_resumable_and_tracks_changes(tmp_path):
    prep, speech = Prep(), Speech()
    root = _import(tmp_path, prep)

    assert _run(tmp_path, "prepare", "--chapter", "1", prep=prep, speech=speech) == 0
    assert prep.calls >= 1 and speech.calls == 0
    assert list((root / "prep" / "review").glob("chapter-001.md"))
    first_prep_calls = prep.calls

    assert _run(tmp_path, "generate", "--chapter", "1", "--sample", "5", prep=prep, speech=speech) == 0
    assert prep.calls == first_prep_calls  # prep was cached
    assert speech.calls >= 1
    assert list((root / "samples").glob("chapter-001-5s.wav"))
    spoken = speech.calls

    # Same command again: nothing is redone.
    assert _run(tmp_path, "gen", "--chapter", "1", "--sample", "5", prep=prep, speech=speech) == 0
    assert (prep.calls, speech.calls) == (first_prep_calls, spoken)

    # Full chapter reuses the sample's clips, then builds the chapter file with a closing silence.
    assert _run(tmp_path, "generate", "--chapter", "1", prep=prep, speech=speech) == 0
    assert (root / "audio" / "chapter-001.wav").exists()

    # Editing the tuning invalidates prep and speech for that chapter only.
    data = json.loads((root / "project.json").read_text())
    data["narration"]["pace"] = "brisk"
    (root / "project.json").write_text(json.dumps(data))
    before = prep.calls
    assert _run(tmp_path, "prepare", "--chapter", "1", prep=prep, speech=speech) == 0
    assert prep.calls > before
    assert "Pace: brisk" in prep.requests[-1]

    assert _run(tmp_path, "status", prep=prep, speech=speech) == 0


def test_generate_needs_a_selection_and_a_dry_run_calls_nothing(tmp_path):
    prep, speech = Prep(), Speech()
    _import(tmp_path, prep)
    assert _run(tmp_path, "generate", prep=prep, speech=speech) == 1
    assert _run(tmp_path, "generate", "--all", "--dry-run", prep=prep, speech=speech) == 0
    assert prep.calls == 0 and speech.calls == 0


def test_flat_delivery_is_flagged_only_for_long_untagged_text():
    config = load_config(ROOT / "config" / "default.toml")
    config.prep.min_tags_per_100_words = 1.0
    flat = {"turns": [{"text": "word " * 120, "style": ""}]}
    assert flat_delivery_errors(flat, config)
    lively = {"turns": [{"text": "word " * 120 + "<chuckle> <short pause>", "style": "warm"}]}
    assert not flat_delivery_errors(lively, config)
    short = {"turns": [{"text": "word " * 10, "style": ""}]}
    assert not flat_delivery_errors(short, config)
