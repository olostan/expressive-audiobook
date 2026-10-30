import json
from pathlib import Path

from expressive_audiobook.chunking import load_chapter_jsonl
from expressive_audiobook.cli import main
from expressive_audiobook.config import load_config
from expressive_audiobook.models import Chunk
from expressive_audiobook.pipeline import run_chunk, run_extract
from expressive_audiobook.review import render_review_markdown
from pdf_factory import build_outline_pdf

ROOT = Path(__file__).resolve().parents[1]


def test_review_shows_source_style_rationale_and_descriptions():
    chunk = Chunk(
        id="c001-0000",
        chapter_id="ch-001",
        chapter_index=1,
        chapter_title="A Short Chapter",
        order=0,
        kind="prose",
        text="The guide laughed once.",
        speakers=["narrator"],
        source_pages=[2],
        estimated_duration_sec=1,
        word_count=4,
        block_ids=["b-1"],
    )
    output = {
        "c001-0000": {
            "prompt_version": "prep-v1",
            "chunk_id": "c001-0000",
            "turns": [
                {
                    "text": "The guide laughed once. <chuckle>",
                    "style": "warm, amused",
                    "speaker": "narrator",
                    "rationale": "The laugh is shown by the action.",
                    "inserted_descriptions": [
                        {
                            "element_id": "fig-1",
                            "kind": "figure",
                            "anchor": "b-1",
                            "spoken_text": "The figure shows two paths crossing.",
                            "decision": "read-full",
                            "reason": "",
                        }
                    ],
                }
            ],
        }
    }
    report = render_review_markdown([chunk], output)
    assert "The guide laughed once." in report
    assert "warm, amused" in report
    assert "The laugh is shown by the action." in report
    assert "two paths crossing" in report


def test_cli_extract_and_chunk(tmp_path: Path):
    pdf = tmp_path / "book.pdf"
    build_outline_pdf(pdf)
    config_path = tmp_path / "audiobook.toml"
    config_path.write_text(
        "\n".join(
            [
                "[paths]",
                f'artifacts = "{tmp_path / "artifacts"}"',
                f'ingest = "{tmp_path / "artifacts" / "ingest"}"',
                f'chunks = "{tmp_path / "artifacts" / "chunks"}"',
                f'prep = "{tmp_path / "artifacts" / "prep"}"',
                f'tts = "{tmp_path / "artifacts" / "tts"}"',
                f'audio = "{tmp_path / "artifacts" / "audio"}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert main(["--config", str(config_path), "extract", "--pdf", str(pdf)]) == 0
    assert main(["--config", str(config_path), "chunk"]) == 0
    chunks = list((tmp_path / "artifacts" / "chunks").glob("*.jsonl"))
    assert chunks
    config = load_config(config_path)
    written = run_chunk(config)
    assert written
    chapter_files = list((tmp_path / "artifacts" / "ingest" / "chapters").glob("*.json"))
    assert chapter_files
    prose_index = next(
        json.loads(path.read_text(encoding="utf-8"))["index"]
        for path in chapter_files
        if "Open Road" in json.loads(path.read_text(encoding="utf-8"))["title"]
    )
    loaded = load_chapter_jsonl(tmp_path / "artifacts" / "chunks" / f"{prose_index:03d}.jsonl")
    out = tmp_path / "artifacts" / "prep" / "outputs" / f"{prose_index:03d}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "prompt_version": "prep-v1",
                "chunk_id": loaded[0].id,
                "turns": [
                    {
                        "text": "Prepared line.",
                        "style": "",
                        "speaker": "narrator",
                        "rationale": "Nothing to heighten.",
                        "inserted_descriptions": [],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["--config", str(config_path), "review", "--chapter", str(prose_index)]) == 0
    report = (tmp_path / "artifacts" / "prep" / "review" / f"chapter-{prose_index:03d}.md").read_text(
        encoding="utf-8"
    )
    assert "Prepared line." in report
    assert "Nothing to heighten." in report


def test_extract_library_round_trip_is_idempotent(tmp_path: Path):
    pdf = tmp_path / "book.pdf"
    build_outline_pdf(pdf)
    config = load_config(ROOT / "config" / "default.toml")
    config.paths.ingest = str(tmp_path / "ingest")
    config.paths.chunks = str(tmp_path / "chunks")
    run_extract(pdf, config)
    first = run_chunk(config)
    second = run_chunk(config)
    assert [path.read_text(encoding="utf-8") for path in first] == [
        path.read_text(encoding="utf-8") for path in second
    ]
