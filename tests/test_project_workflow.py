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


def test_long_pause_is_rewritten_and_no_longer_allowed():
    from pdf_expressive_audiobook.tts import speakable_text

    config = load_config(ROOT / "config" / "default.toml")
    assert "<long pause>" not in config.prep.allowed_tags
    assert speakable_text("Wait. <long pause> Go.") == "Wait. <short pause> <short pause> Go."


def _chunk(text, speakers=("Achilles",)):
    from pdf_expressive_audiobook.models import Chunk

    return Chunk(
        id="c001-0000", chapter_id="ch-001", chapter_index=1, chapter_title="t", order=0, kind="dialogue",
        text=text, speakers=list(speakers), source_pages=[1], estimated_duration_sec=5, word_count=8,
        block_ids=["b-1"], segments=[{"speaker": "Achilles", "text": text, "block_ids": ["b-1"]}],
    )


def _output(text, speaker="Achilles"):
    return {
        "prompt_version": "prep-v1", "chunk_id": "c001-0000",
        "turns": [{"text": text, "style": "", "speaker": speaker, "rationale": "r", "block_ids": ["b-1"],
                   "inserted_descriptions": []}],
    }


def test_validator_rejects_speaker_labels_and_typographic_art():
    from pdf_expressive_audiobook.prep import validate_prep_output

    config = load_config(ROOT / "config" / "default.toml")
    chunk = _chunk("Hello there.")
    assert not validate_prep_output(_output("Hello there."), chunk, config)
    assert any("speaker label" in e for e in validate_prep_output(_output("Achilles: Hello there."), chunk, config))
    assert any("typographic art" in e for e in validate_prep_output(_output("C---- four"), chunk, config))
    assert not validate_prep_output(_output("C, dash dash dash dash, four"), chunk, config)


def test_chunks_keep_who_says_what():
    from pdf_expressive_audiobook.chunking import chunk_chapter
    from pdf_expressive_audiobook.models import Block, Chapter

    def block(n, speaker, text):
        return Block(id=f"b-{n}", kind="dialogue_turn" if speaker else "paragraph", text=text,
                     page_start=1, page_end=1, stream_index=n, speaker=speaker)

    chapter = Chapter(id="ch-001", index=1, title="t", kind="prose", page_start=1, page_end=1, detection="x",
                      blocks=[block(1, None, "The phone rings."), block(2, "Achilles", "Hello."),
                              block(3, "Tortoise", "Hi there.")])
    chunk = chunk_chapter(chapter)[0]
    assert [s["speaker"] for s in chunk.segments] == ["narrator", "Achilles", "Tortoise"]


def test_character_voices_are_distinct_valid_and_skip_the_narrator():
    from pdf_expressive_audiobook.style_detect import assign_character_voices

    voices = assign_character_voices("Schedar", ["A", "B", "C"], {"A": "Puck", "B": "Puck", "C": "Schedar", "D": "x"})
    assert voices["A"] == "Puck"
    assert len(set(voices.values())) == 3 and "Schedar" not in voices.values()
