"""Markdown and plain-text import."""

import json
from pathlib import Path

import pytest

from expressive_audiobook.chunking import chunk_chapter
from expressive_audiobook.cli import main
from expressive_audiobook.config import IngestConfig
from expressive_audiobook.ingest import load_chapter, load_manifest
from expressive_audiobook.ingest_text import extract_text, is_text_source

SENTENCES = "The fox walked slowly along the river. It was a quiet morning and nobody was awake. "


def prose(sentences: int = 6) -> str:
    return (SENTENCES * sentences).strip()


MARKDOWN = f"""---
title: The Fox and the Lantern
author: A. Writer
---

# The Fox and the Lantern

{prose()}

## Chapter 1: The River

{prose()} Read the [map](http://example.com) with *great* care and **no** fear.

![a picture](fox.png)

```python
print("never spoken")
```

* * *

- first thing to pack
- second thing to pack

> A quoted line from the old fox.

## Chapter 2: The Hill

### Climbing

{prose()}

<!-- hidden note -->
"""


def test_markdown_splits_on_headings_and_strips_markup(tmp_path: Path):
    source = tmp_path / "fox.md"
    source.write_text(MARKDOWN, encoding="utf-8")
    document = extract_text(source, tmp_path / "ingest", IngestConfig())
    assert document.detection == "markdown-headings"
    assert document.title == "The Fox and the Lantern"
    assert [chapter.title for chapter in document.chapters] == [
        "The Fox and the Lantern",  # text between the book title and the first chapter
        "Chapter 1: The River",
        "Chapter 2: The Hill",
    ]
    first = document.chapters[1]
    text = "\n".join(block.text for block in first.blocks)
    assert "Read the map with great care and no fear." in text
    assert "never spoken" not in text and "fox.png" not in text and "http" not in text
    assert "first thing to pack" in text and "A quoted line from the old fox." in text
    assert "*" not in text and "hidden" not in "".join(b.text for b in document.chapters[2].blocks)
    assert first.blocks[0].kind == "heading" and first.blocks[0].text == "Chapter 1: The River"
    second = document.chapters[2]
    assert [b.text for b in second.blocks if b.kind == "heading"] == ["Chapter 2: The Hill", "Climbing"]
    manifest = load_manifest(tmp_path / "ingest")
    assert manifest["author"] == "A. Writer" and manifest["source_kind"] == "markdown"
    assert load_chapter(tmp_path / "ingest" / "chapters" / "003.json").title == "Chapter 2: The Hill"


def test_chapters_chunk_like_pdf_chapters(tmp_path: Path):
    source = tmp_path / "fox.md"
    source.write_text(MARKDOWN, encoding="utf-8")
    extract_text(source, tmp_path / "ingest")
    chapter = load_chapter(tmp_path / "ingest" / "chapters" / "002.json")
    chunks = chunk_chapter(chapter)
    assert chunks and chunks[0].text.startswith("Chapter 1: The River")


def test_markdown_front_text_before_first_chapter_is_kept(tmp_path: Path):
    source = tmp_path / "a.md"
    source.write_text(f"# Title\n\n{prose(8)}\n\n## One\n\n{prose()}\n\n## Two\n\n{prose()}\n", encoding="utf-8")
    document = extract_text(source, tmp_path / "ingest")
    assert [c.title for c in document.chapters] == ["Title", "One", "Two"]


def test_text_chapter_markers(tmp_path: Path):
    source = tmp_path / "tale.txt"
    source.write_text(
        f"CHAPTER I\n\nDown the Hole\n\n{prose()}\n\nCHAPTER II\n\nThe Pool of Tears\n\n{prose()}\n"
        f"\n***\n\n{prose(2)}\n",
        encoding="utf-8",
    )
    document = extract_text(source, tmp_path / "ingest")
    assert document.detection == "text-headings"
    assert [c.title for c in document.chapters] == ["CHAPTER I. Down the Hole", "CHAPTER II. The Pool of Tears"]
    assert not any("***" in b.text for c in document.chapters for b in c.blocks)


def test_text_joins_wrapped_lines_and_numbered_headings(tmp_path: Path):
    source = tmp_path / "wrapped.txt"
    body = "The fox walked slowly along the\nriver and the morning was very quiet\nand nobody was awake yet."
    source.write_text(f"1. The River\n\n{body}\n\n2. The Hill\n\n{body}\n", encoding="utf-8")
    document = extract_text(source, tmp_path / "ingest")
    assert [c.title for c in document.chapters] == ["1. The River", "2. The Hill"]
    assert document.chapters[0].blocks[1].text.startswith("The fox walked slowly along the river and")


def test_text_without_markers_is_split_by_length(tmp_path: Path):
    source = tmp_path / "long.txt"
    paragraph = prose(10)
    source.write_text("\n\n".join([paragraph] * 80), encoding="utf-8")
    document = extract_text(source, tmp_path / "ingest")
    assert document.detection == "size"
    assert len(document.chapters) >= 3
    assert document.chapters[0].title == "Part 1"
    assert sum(len(c.blocks) for c in document.chapters) == 80 + len(document.chapters)


def test_short_text_is_one_chapter_titled_from_the_file_name(tmp_path: Path):
    source = tmp_path / "little_fox.txt"
    source.write_text(prose(), encoding="utf-8")
    document = extract_text(source, tmp_path / "ingest")
    assert [c.title for c in document.chapters] == ["Little Fox"]


def test_script_style_speakers_become_dialogue_turns(tmp_path: Path):
    source = tmp_path / "play.txt"
    source.write_text(
        "Chapter 1\n\nFOX: Hello there.\n\nOWL: Good evening.\n\nFOX: What a night.\n\nOWL: Indeed it is.\n",
        encoding="utf-8",
    )
    document = extract_text(source, tmp_path / "ingest")
    assert [b.speaker for b in document.chapters[0].blocks if b.speaker] == ["FOX", "OWL", "FOX", "OWL"]
    assert document.chapters[0].kind == "dialogue"


def test_empty_file_is_rejected(tmp_path: Path):
    source = tmp_path / "empty.md"
    source.write_text("# Nothing\n", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_text(source, tmp_path / "ingest")


def test_is_text_source():
    assert is_text_source("a.MD") and is_text_source("b.txt") and not is_text_source("c.pdf")


def test_import_command_accepts_markdown(tmp_path: Path):
    source = tmp_path / "fox.md"
    source.write_text(MARKDOWN, encoding="utf-8")
    code = main(
        ["import", "--source", str(source), "--name", "fox", "--projects-dir", str(tmp_path / "projects")],
        prep_client=None,
    )
    assert code == 0
    root = tmp_path / "projects" / "fox"
    project = json.loads((root / "project.json").read_text())
    assert project["title"] == "The Fox and the Lantern"
    assert len(list((root / "ingest" / "chapters").glob("*.json"))) == 3
