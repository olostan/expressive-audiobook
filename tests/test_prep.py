import base64
import io
from pathlib import Path

from PIL import Image

from pdf_expressive_audiobook.config import load_config, load_style_profile
from pdf_expressive_audiobook.models import AssociatedRef, Chunk
from pdf_expressive_audiobook.prep import (
    build_prep_request,
    downscale_png,
    render_prompt,
    split_turn_to_fit,
    tts_turns_from_output,
    validate_prep_output,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config" / "default.toml")
PROFILE = load_style_profile(ROOT / "profiles" / "examples" / "neutral-narration.yaml")


def _png(color: tuple[int, int, int], size=(40, 30)) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _chunk(associated=None, text="The guide laughed once, short and surprised.") -> Chunk:
    return Chunk(
        id="c001-0000",
        chapter_id="ch-001",
        chapter_index=1,
        chapter_title="A Short Chapter",
        order=0,
        kind="prose",
        text=text,
        speakers=["narrator"],
        source_pages=[1],
        estimated_duration_sec=2,
        word_count=6,
        associated=associated or [],
        block_ids=["b-1"],
    )


def _output(text="The guide laughed once, short and surprised. <chuckle>", **extra) -> dict:
    turn = {
        "text": text,
        "style": extra.get("style", "warm, amused"),
        "speaker": "narrator",
        "rationale": extra.get("rationale", "The laugh is in the action."),
        "block_ids": extra.get("block_ids", ["b-1"]),
        "inserted_descriptions": extra.get("descriptions", []),
    }
    return {
        "prompt_version": CONFIG.prep.prompt_version,
        "chunk_id": "c001-0000",
        "turns": extra.get("turns", [turn]),
    }


def test_prompt_injects_profile_allowlist_and_inference():
    chunk = _chunk()
    profile = PROFILE.with_style_text("keep the jokes intact")
    prompt = render_prompt(
        chunk,
        profile,
        CONFIG.prep,
        tts_model=CONFIG.tts.model,
        wpm=CONFIG.chunking.wpm,
        hard_cap_seconds=CONFIG.chunking.hard_cap_seconds,
    )
    assert CONFIG.tts.model in prompt
    assert "<short pause>" in prompt
    assert "|mhm|" in prompt
    assert "keep the jokes intact" in prompt
    assert "Infer the emotion" in prompt
    assert "{{" not in prompt
    assert "under-tagged" in prompt and "over-tagged" in prompt


def test_validator_accepts_tags_pipes_and_a_composed_style():
    chunk = _chunk()
    output = _output(
        text="Wait... <short pause> did you hear that? |mhm|",
        style="a private, wondering tone, as if the thought just arrived",
    )
    assert validate_prep_output(output, chunk, CONFIG) == []


def test_validator_rejects_stage_directions_sound_effects_and_unknown_tags():
    chunk = _chunk()
    samples = [
        "She spoke [laughs] and left.",
        "She spoke (softly) and left.",
        "She spoke *pauses* and left.",
        "She spoke <applause> and left.",
        "She spoke <guffaw> and left.",
        "She spoke |applause| and left.",
    ]
    for text in samples:
        errors = validate_prep_output(_output(text=text, style=""), chunk, CONFIG)
        assert errors, text


def test_validator_does_not_treat_style_as_an_enum_but_rejects_a_transcript_copy():
    chunk = _chunk()
    novel = _output(style="high and bright, telling one friend a secret")
    assert validate_prep_output(novel, chunk, CONFIG) == []
    copied = _output(style=chunk.text)
    assert any("style" in error for error in validate_prep_output(copied, chunk, CONFIG))


def test_validator_catches_dropped_elements_latex_and_long_descriptions():
    note = AssociatedRef(
        id="fn-1",
        kind="footnote",
        anchor="b-1",
        stream_index=0,
        text="Only about the weather.",
        estimated_spoken_words=4,
        speak=True,
        placement="inline",
    )
    formula = AssociatedRef(
        id="formula-1",
        kind="formula",
        anchor="b-1",
        stream_index=0,
        text=r"$$x = \frac{-b}{2a}$$",
        latex=r"$$x = \frac{-b}{2a}$$",
        estimated_spoken_words=20,
        speak=True,
        placement="inline",
    )
    chunk = _chunk(associated=[note, formula], text="See [[footnote:fn-1]] and the formula.")
    missing = _output(
        text="See the note.",
        descriptions=[
            {
                "element_id": "fn-1",
                "kind": "footnote",
                "anchor": "b-1",
                "spoken_text": "footnote: only about the weather.",
                "decision": "read-full",
                "reason": "",
            }
        ],
    )
    assert any("formula-1" in error for error in validate_prep_output(missing, chunk, CONFIG))

    latex = _output(
        text=r"x = \frac{-b}{2a}",
        descriptions=[
            {
                "element_id": "fn-1",
                "kind": "footnote",
                "anchor": "b-1",
                "spoken_text": "footnote: only about the weather.",
                "decision": "mention",
                "reason": "",
            },
            {
                "element_id": "formula-1",
                "kind": "formula",
                "anchor": "b-1",
                "spoken_text": r"x equals \frac{-b}{2a}",
                "decision": "read-full",
                "reason": "",
            },
        ],
    )
    assert any("math" in error or "LaTeX" in error for error in validate_prep_output(latex, chunk, CONFIG))

    long_spoken = " ".join(["word"] * 90) + "."
    too_long = _output(
        text="The formula is spoken in full beside the note.",
        descriptions=[
            {
                "element_id": "fn-1",
                "kind": "footnote",
                "anchor": "b-1",
                "spoken_text": "footnote: only about the weather.",
                "decision": "mention",
                "reason": "",
            },
            {
                "element_id": "formula-1",
                "kind": "formula",
                "anchor": "b-1",
                "spoken_text": long_spoken,
                "decision": "read-full",
                "reason": "",
            },
        ],
    )
    assert any("cap" in error for error in validate_prep_output(too_long, chunk, CONFIG))


def test_subturn_keeps_the_footnote_on_the_anchor_turn():
    note = AssociatedRef(
        id="fn-1",
        kind="footnote",
        anchor="b-2",
        stream_index=1,
        text="A side note.",
        estimated_spoken_words=3,
        speak=True,
        placement="inline",
    )
    chunk = _chunk(associated=[note], text="First thought. Second thought.")
    good = _output(
        turns=[
            {
                "text": "First thought.",
                "style": "hopeful, unhurried",
                "speaker": "narrator",
                "rationale": "The opening is light.",
                "block_ids": ["b-1"],
                "inserted_descriptions": [],
            },
            {
                "text": "Second thought. footnote: a side note.",
                "style": "quiet, heavier",
                "speaker": "narrator",
                "rationale": "The cost arrives here.",
                "block_ids": ["b-2"],
                "inserted_descriptions": [
                    {
                        "element_id": "fn-1",
                        "kind": "footnote",
                        "anchor": "b-2",
                        "spoken_text": "footnote: a side note.",
                        "decision": "read-full",
                        "reason": "",
                    }
                ],
            },
        ]
    )
    assert validate_prep_output(good, chunk, CONFIG) == []
    misplaced = _output(
        turns=[
            {
                "text": "First thought. footnote: a side note.",
                "style": "",
                "speaker": "narrator",
                "rationale": "Wrong turn.",
                "block_ids": ["b-1"],
                "inserted_descriptions": [
                    {
                        "element_id": "fn-1",
                        "kind": "footnote",
                        "anchor": "b-2",
                        "spoken_text": "footnote: a side note.",
                        "decision": "read-full",
                        "reason": "",
                    }
                ],
            },
            {
                "text": "Second thought.",
                "style": "quiet, heavier",
                "speaker": "narrator",
                "rationale": "The shift.",
                "block_ids": ["b-2"],
                "inserted_descriptions": [],
            },
        ]
    )
    assert any("anchor" in error for error in validate_prep_output(misplaced, chunk, CONFIG))


def test_split_turn_to_fit_copies_style_and_moves_the_description():
    spoken = "x equals negative b, all over two a."
    turn = {
        "text": "The plan seemed kind. " + spoken + " Then the cost arrived and the room felt smaller today.",
        "style": "quiet, heavier",
        "speaker": "narrator",
        "rationale": "A shift.",
        "block_ids": ["b-1"],
        "inserted_descriptions": [
            {
                "element_id": "formula-1",
                "kind": "formula",
                "anchor": "b-1",
                "spoken_text": spoken,
                "decision": "read-full",
                "reason": "",
            }
        ],
    }
    parts = split_turn_to_fit(turn, max_words=8)
    assert len(parts) >= 2
    assert all(part["style"] == "quiet, heavier" for part in parts)
    owners = [part for part in parts if part["inserted_descriptions"]]
    assert len(owners) == 1
    assert spoken in owners[0]["text"]


def test_rationale_is_stripped_from_tts_turns():
    output = _output(rationale="RATIONALE_SENTINEL_do_not_speak")
    turns = tts_turns_from_output(output)
    blob = str(turns)
    assert "RATIONALE_SENTINEL_do_not_speak" not in blob
    assert "inserted_descriptions" not in blob
    assert set(turns[0]) == {"text", "style", "speaker"}


def test_request_includes_images_in_reading_order_and_downscales(tmp_path: Path):
    red = tmp_path / "red.png"
    blue = tmp_path / "blue.png"
    wide = tmp_path / "wide.png"
    red.write_bytes(_png((200, 20, 20)))
    blue.write_bytes(_png((20, 20, 200)))
    wide.write_bytes(_png((10, 180, 10), size=(2000, 800)))
    associated = [
        AssociatedRef(
            id="formula-1",
            kind="formula",
            anchor="b-1",
            stream_index=2,
            text="$E = mc^2$",
            latex="$E = mc^2$",
            estimated_spoken_words=10,
            speak=True,
            placement="inline",
            image_path="blue.png",
        ),
        AssociatedRef(
            id="fig-1",
            kind="figure",
            anchor="b-1",
            stream_index=1,
            text="Figure 1. A square.",
            caption="Figure 1. A square.",
            estimated_spoken_words=20,
            speak=True,
            placement="inline",
            image_path="red.png",
        ),
        AssociatedRef(
            id="fig-2",
            kind="figure",
            anchor="b-1",
            stream_index=3,
            text="Figure 2. A wide field.",
            caption="Figure 2. A wide field.",
            estimated_spoken_words=20,
            speak=True,
            placement="inline",
            image_path="wide.png",
        ),
    ]
    config = load_config(ROOT / "config" / "default.toml")
    config.prep.max_images_per_request = 2
    config.prep.max_image_edge_px = 1024
    request = build_prep_request(_chunk(associated), PROFILE, config, image_root=tmp_path)
    parts = request["contents"][0]["parts"]
    images = [part["inlineData"] for part in parts if "inlineData" in part]
    assert len(images) == 2
    first = Image.open(io.BytesIO(base64.b64decode(images[0]["data"])))
    second = Image.open(io.BytesIO(base64.b64decode(images[1]["data"])))
    assert first.getpixel((0, 0))[0] > 150
    assert second.getpixel((0, 0))[2] > 150
    intros = [part["text"] for part in parts if part.get("text", "").startswith("Embedded image")]
    assert "fig-1" in intros[0]
    assert "formula-1" in intros[1]
    assert "fig-2" in parts[0]["text"]
    assert request["model"] == config.prep.model
    shrunk = downscale_png(wide.read_bytes(), 1024)
    assert max(Image.open(io.BytesIO(shrunk)).size) <= 1024
