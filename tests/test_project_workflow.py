"""import -> prepare -> generate on a project, with mocked Gemini clients."""

import json
from pathlib import Path

from expressive_audiobook.cli import main
from expressive_audiobook.config import load_config
from expressive_audiobook.prep import flat_delivery_errors
from expressive_audiobook.project import Project, load_style_md
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
    "speaker_profiles": [
        {"name": "Guide", "description": "Patient and amused.", "vocal_cue": "Warm, unhurried.", "voice": "Puck"}
    ],
    "performance_directives": [
        {"title": "The gate scene", "when": "Chapter 2", "how": "Slow down and let each reply land."}
    ],
    "pronunciation_lexicon": [{"word": "Quillan", "ipa": "/ˈkwɪlən/"}],
    "verbalize": {"formulas": "Say the math."},
    "recommended": {"expressiveness": 0.9, "pace": 0.1, "pause_density": 0.9},
    "notes": "Slow down for the gate scene.",
}


class Prep:
    def __init__(self):
        self.calls = 0
        self.styled = 0
        self.style_prompts = []
        self.requests = []

    def generate(self, request):
        text = request["contents"][0]["parts"][0]["text"]
        if "audiobook director" in text:
            self.style_prompts.append(text)
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
    assert project.data["narration"]["expressiveness"] == 0.9
    assert project.data["narration"]["pace"] == 0.1
    assert (root / "ingest" / "manifest.json").exists()
    assert len(list((root / "ingest" / "chapters").glob("*.json"))) == 3
    manifest = json.loads((root / "ingest" / "manifest.json").read_text())
    assert manifest["skipped_pages"] == [{"page": 1, "reason": "contents"}]
    profile = load_style_md(root / "style.md")
    assert profile.narrator_persona.startswith("A dry")
    assert profile.pronunciation_lexicon[0].word == "Quillan"
    assert "Slow down for the gate scene." in profile.free_text
    assert "**Guide:** Patient and amused." in profile.free_text
    assert "1. **The gate scene** *(Chapter 2)*" in profile.free_text
    prompt = prep.style_prompts[0]
    assert "1. Chapter 1. The Open Road" in prompt  # chapter list
    assert "--- the opening pages ---" in prompt
    assert "Chapter 1. The Open Road" in prompt and "Contents" in prompt  # the book's own contents page
    # Restyling keeps the previous file.
    _import_again = main(
        ["import", "--pdf", str(tmp_path / "book.pdf"), "--name", "owned", "--projects-dir", str(tmp_path / "projects"),
         "--restyle"],
        prep_client=prep,
    )
    assert _import_again == 0 and (root / "style.previous.md").exists()
    # Importing the same PDF again keeps edits and does not call Gemini again.
    (root / "style.md").write_text((root / "style.md").read_text() + "\nMY EDIT\n")
    _import(tmp_path, prep)
    assert prep.styled == 2  # first import and the explicit --restyle only
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
    assert list((root / "samples").glob("chapter-001-5s.*"))
    spoken = speech.calls

    # Same command again: nothing is redone.
    assert _run(tmp_path, "gen", "--chapter", "1", "--sample", "5", prep=prep, speech=speech) == 0
    assert (prep.calls, speech.calls) == (first_prep_calls, spoken)

    # Full chapter reuses the sample's clips, then builds the chapter file with a closing silence.
    assert _run(tmp_path, "generate", "--chapter", "1", prep=prep, speech=speech) == 0
    assert list((root / "audio").glob("chapter-001.*"))

    # Editing the tuning invalidates prep and speech for that chapter only.
    data = json.loads((root / "project.json").read_text())
    data["narration"]["pace"] = 0.9
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
    from expressive_audiobook.tts import speakable_text

    config = load_config(ROOT / "config" / "default.toml")
    assert "<long pause>" not in config.prep.allowed_tags
    assert speakable_text("Wait. <long pause> Go.") == "Wait. <short pause> <short pause> Go."


def test_pause_tags_at_the_edges_of_a_turn_are_dropped():
    from expressive_audiobook.tts import speakable_text

    assert speakable_text("They danced. <short pause> <short pause>") == "They danced."
    assert speakable_text("<short pause> Once upon a time. <short pause> Then, <short pause> rain.") == (
        "Once upon a time. <short pause> Then, <short pause> rain."
    )
    assert speakable_text("<short pause>") == "<short pause>"  # nothing else to say: left alone


def _chunk(text, speakers=("Achilles",)):
    from expressive_audiobook.models import Chunk

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
    from expressive_audiobook.prep import validate_prep_output

    config = load_config(ROOT / "config" / "default.toml")
    chunk = _chunk("Hello there.")
    assert not validate_prep_output(_output("Hello there."), chunk, config)
    assert any("speaker label" in e for e in validate_prep_output(_output("Achilles: Hello there."), chunk, config))
    assert any("typographic art" in e for e in validate_prep_output(_output("C---- four"), chunk, config))
    assert not validate_prep_output(_output("C, dash dash dash dash, four"), chunk, config)


def test_chunks_keep_who_says_what():
    from expressive_audiobook.chunking import chunk_chapter
    from expressive_audiobook.models import Block, Chapter

    def block(n, speaker, text):
        return Block(id=f"b-{n}", kind="dialogue_turn" if speaker else "paragraph", text=text,
                     page_start=1, page_end=1, stream_index=n, speaker=speaker)

    chapter = Chapter(id="ch-001", index=1, title="t", kind="prose", page_start=1, page_end=1, detection="x",
                      blocks=[block(1, None, "The phone rings."), block(2, "Achilles", "Hello."),
                              block(3, "Tortoise", "Hi there.")])
    chunk = chunk_chapter(chapter)[0]
    assert [s["speaker"] for s in chunk.segments] == ["narrator", "Achilles", "Tortoise"]


def test_character_voices_are_distinct_valid_and_skip_the_narrator():
    from expressive_audiobook.style_detect import assign_character_voices

    voices = assign_character_voices("Schedar", ["A", "B", "C"], {"A": "Puck", "B": "Puck", "C": "Schedar", "D": "x"})
    assert voices["A"] == "Puck"
    assert len(set(voices.values())) == 3 and "Schedar" not in voices.values()


def test_cache_follows_settings_that_matter(tmp_path):
    prep, speech = Prep(), Speech()
    root = _import(tmp_path, prep)

    def run():
        return _run(tmp_path, "generate", "--chapter", "1", prep=prep, speech=speech) == 0

    def edit(fn):
        data = json.loads((root / "project.json").read_text())
        fn(data)
        (root / "project.json").write_text(json.dumps(data))

    def counts():
        return prep.calls, speech.calls

    assert run()
    base = counts()
    assert run() and counts() == base  # nothing changed: all cached

    # Assembly-only settings redo nothing that is paid for.
    edit(lambda d: d["audio"].update(gap_ms=1234))
    assert run() and counts() == base

    # Voice affects speech but not the narration text.
    edit(lambda d: d["narration"].update(voice="Charon"))
    assert run()
    assert prep.calls == base[0] and speech.calls > base[1]
    after_voice = counts()

    # A character voice change must not redo prep either.
    edit(lambda d: d["character_voices"].update({"Guide": "Puck"}))
    assert run() and prep.calls == after_voice[0]

    # Tuning changes the prompt, so prep and speech are redone.
    edit(lambda d: d["narration"].update(expressiveness=0.2))
    assert run() and prep.calls > after_voice[0]
    after_tuning = counts()

    # Editing style.md redoes prep.
    (root / "style.md").write_text((root / "style.md").read_text() + "\nNew direction: whisper everything.\n")
    assert run() and prep.calls > after_tuning[0]
    assert "whisper everything" in prep.requests[-1]


def test_rejected_retry_request_falls_back_to_the_plain_request():
    from expressive_audiobook.prep import prepare_chunk_output
    from expressive_audiobook.config import StyleProfile

    class Rejected(Exception):
        code = 400

    config = load_config(ROOT / "config" / "default.toml")
    chunk = _chunk("Hello there.")
    calls = []

    class Client:
        def generate(self, request):
            calls.append(len(request["contents"]))
            if len(calls) == 1:
                return _output("Achilles: Hello there.")  # fails validation, triggers a feedback retry
            if len(request["contents"]) > 1:
                raise Rejected("INVALID_ARGUMENT")
            return _output("Hello there.")

    out = prepare_chunk_output(chunk, StyleProfile(), config, Client())
    assert out["turns"][0]["text"] == "Hello there."
    assert calls == [1, 2, 1]


def test_api_errors_print_a_short_message_not_a_traceback(tmp_path, capsys):
    prep, speech = Prep(), Speech()
    _import(tmp_path, prep)

    class Boom(Exception):
        pass

    class Failing(Prep):
        def generate(self, request):
            raise Boom("400 INVALID_ARGUMENT")

    assert _run(tmp_path, "prepare", "--chapter", "1", prep=Failing(), speech=speech) == 1
    err = capsys.readouterr().err
    assert "Boom: 400 INVALID_ARGUMENT" in err and "Traceback" not in err
    assert list((tmp_path / "projects" / "owned" / "prep" / "errors").glob("*.txt"))


def test_tuning_levels_are_numbers_with_old_words_as_aliases():
    import pytest

    from expressive_audiobook.tuning import chapter_tail_ms, delivery_guidance, gap_ms, level

    assert level("pace", 0.3) == 0.3 and level("pace", "slow") == 0.1 and level("expressiveness", "high") == 0.85
    for bad in ("normal", 1.5, -0.1, None, True):
        with pytest.raises(ValueError, match="number from 0 to 1"):
            level("expressiveness", bad)
    assert gap_ms(0) == 250 and gap_ms(1) == 900 and chapter_tail_ms(1) == 2500
    low = delivery_guidance(0.1, 0.1, 0.0)
    high = delivery_guidance(0.9, 0.9, 1.0)
    assert "minimal" in low and "slow" in low and "only where punctuation" in low
    assert "Perform the text" in high and "brisk" in high and "two in a row" in high


def _need_ffmpeg():
    import shutil

    import pytest

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")


def test_mp3_replaces_the_wav_by_default_and_is_not_rebuilt(tmp_path):
    _need_ffmpeg()
    prep, speech = Prep(), Speech()
    root = _import(tmp_path, prep)
    assert _run(tmp_path, "generate", "--all", prep=prep, speech=speech) == 0
    audio = root / "audio"
    mp3 = audio / "chapter-001.mp3"
    assert mp3.exists() and mp3.read_bytes()[:3] == b"ID3"
    assert not list(audio.glob("chapter-*.wav")) and not (audio / "book.wav").exists()
    assert (audio / "book.mp3").exists()
    assert list((audio / "clips" / "001").glob("*.wav"))  # the speech cache stays
    stamps = {p.name: p.stat().st_mtime_ns for p in audio.glob("*.mp3")}
    assert _run(tmp_path, "generate", "--all", prep=prep, speech=speech) == 0
    assert {p.name: p.stat().st_mtime_ns for p in audio.glob("*.mp3")} == stamps
    assert not list(audio.glob("chapter-*.wav"))  # nothing rebuilt


def test_keep_wav_true_keeps_both_and_changed_inputs_rebuild(tmp_path):
    _need_ffmpeg()
    prep, speech = Prep(), Speech()
    root = _import(tmp_path, prep)
    data = json.loads((root / "project.json").read_text())
    data["audio"]["keep_wav"] = True
    (root / "project.json").write_text(json.dumps(data))
    assert _run(tmp_path, "generate", "--chapter", "1", prep=prep, speech=speech) == 0
    wav, mp3 = root / "audio" / "chapter-001.wav", root / "audio" / "chapter-001.mp3"
    assert wav.exists() and mp3.exists()
    stamp = mp3.stat().st_mtime_ns
    assert _run(tmp_path, "generate", "--chapter", "1", prep=prep, speech=speech) == 0
    assert mp3.stat().st_mtime_ns == stamp
    data["audio"]["keep_wav"] = False
    data["audio"]["gap_ms"] = 1500  # different assembly: rebuilt, converted, WAV dropped
    (root / "project.json").write_text(json.dumps(data))
    assert _run(tmp_path, "generate", "--chapter", "1", prep=prep, speech=speech) == 0
    assert mp3.exists() and not wav.exists()
    assert mp3.stat().st_mtime_ns != stamp


def test_missing_ffmpeg_keeps_the_wav_and_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("expressive_audiobook.export.shutil.which", lambda name: None)
    prep, speech = Prep(), Speech()
    root = _import(tmp_path, prep)
    assert _run(tmp_path, "generate", "--chapter", "1", prep=prep, speech=speech) == 0
    assert (root / "audio" / "chapter-001.wav").exists()
    assert not list((root / "audio").glob("*.mp3"))
    assert "ffmpeg not found" in capsys.readouterr().out
