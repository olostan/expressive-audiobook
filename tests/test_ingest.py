from pathlib import Path

from PIL import Image

from pdf_expressive_audiobook.config import IngestConfig
from pdf_expressive_audiobook.ingest import extract_pdf, load_chapter
from pdf_factory import build_heuristic_pdf, build_outline_pdf, build_toc_pdf


def _by_title(document, fragment: str):
    matches = [chapter for chapter in document.chapters if fragment in chapter.title]
    assert matches, [chapter.title for chapter in document.chapters]
    return matches[0]


def test_outline_pdf_preserves_structure_and_linked_elements(tmp_path: Path):
    pdf = tmp_path / "outline.pdf"
    build_outline_pdf(pdf)
    dest = tmp_path / "ingest"
    document = extract_pdf(pdf, dest, IngestConfig())
    assert document.outline_used
    assert document.detection == "outline"
    prose = _by_title(document, "Open Road")
    dialogue = _by_title(document, "Gate Said")
    marks = _by_title(document, "Marks on the Page")
    assert any(chapter.kind == "front_matter" for chapter in document.chapters)

    prose_text = "\n".join(block.text for block in prose.blocks)
    assert "reconsider" in prose_text
    assert "Sample Volume" not in prose_text
    assert "\n2\n" not in f"\n{prose_text}\n"
    footnote = next(element for element in prose.elements if element.kind == "footnote")
    anchor = next(block for block in prose.blocks if block.id == footnote.anchor_block_id)
    assert "weather" in footnote.text
    assert "rain" in anchor.text
    assert f"[[footnote:{footnote.id}]]" in anchor.text
    assert any(element.citation_kind == "author_year" and "Quillan" in element.text for element in prose.elements)
    assert any(element.citation_kind == "cross_reference" for element in prose.elements)
    inline = next(element for element in prose.elements if element.kind == "formula" and not element.display)
    assert "mc^2" in (inline.latex or "")
    assert inline.image_path
    assert (dest / "chapters" / inline.image_path).stat().st_size > 0

    speakers = [block.speaker for block in dialogue.blocks if block.speaker]
    assert speakers == ["GUIDE", "WANDERER", "RIVER", "GATE"]
    assert dialogue.kind == "dialogue"
    assert any("splits after the stone" in block.text for block in dialogue.blocks)
    assert any(block.kind == "aside" for block in prose.blocks)

    figures = [element for element in marks.elements if element.kind == "figure"]
    assert {element.figure_source for element in figures} == {"raster", "vector"}
    captions = {element.caption for element in figures}
    assert any(caption and "red square" in caption for caption in captions)
    assert any(caption and "crossing a circle" in caption for caption in captions)
    for figure in figures:
        image = dest / "chapters" / figure.image_path
        assert image.exists()
        assert figure.dpi == 144
        assert figure.bbox is not None
        with Image.open(image) as handle:
            assert handle.format == "PNG"
            assert handle.size[0] > 10
    display = next(element for element in marks.elements if element.kind == "formula" and element.display)
    assert r"\frac" in (display.latex or "")
    assert (dest / "chapters" / display.image_path).exists()
    assert any(element.citation_kind == "numeric" and element.text == "[12]" for element in marks.elements)
    assert marks.scanned_pages == [5]
    assert any("no text layer" in block.text for block in marks.blocks)
    saved = load_chapter(dest / "chapters" / f"{prose.index:03d}.json")
    assert saved.title == prose.title


def test_scanned_page_can_take_an_ocr_hook(tmp_path: Path):
    pdf = tmp_path / "outline.pdf"
    build_outline_pdf(pdf)
    dest = tmp_path / "ocr"
    document = extract_pdf(
        pdf,
        dest,
        IngestConfig(ocr_enabled=True),
        ocr_fn=lambda png: "Recovered line from the scan.",
    )
    marks = _by_title(document, "Marks on the Page")
    assert 5 in marks.scanned_pages
    assert any("Recovered line from the scan." in block.text for block in marks.blocks)


def test_heuristic_and_toc_detection(tmp_path: Path):
    heuristic = tmp_path / "heads.pdf"
    build_heuristic_pdf(heuristic)
    heads = extract_pdf(heuristic, tmp_path / "heads", IngestConfig())
    assert heads.detection == "heuristic"
    assert [chapter.title for chapter in heads.chapters] == [
        "Chapter 1. The Quiet Harbor",
        "Chapter 2. The Evening Gate",
    ]
    assert all("Sample Volume" not in " ".join(block.text for block in chapter.blocks) for chapter in heads.chapters)

    contents = tmp_path / "toc.pdf"
    build_toc_pdf(contents)
    toc = extract_pdf(contents, tmp_path / "toc", IngestConfig())
    assert toc.detection == "toc"
    titles = [chapter.title for chapter in toc.chapters if chapter.kind != "front_matter"]
    assert titles == ["Chapter 1. Alpha", "Chapter 2. Beta"]
