"""Stage 1: read a PDF into one JSON document model per chapter.

Chapter boundaries come from the PDF outline when it has any, otherwise from
a table-of-contents page, otherwise from heading size and chapter regexes.
Running headers, footers, and page numbers are removed. Line-break hyphens
are rejoined. Footnotes, figures, formulas, and citations stay attached to
an anchor in reading order. Pages with no text layer are flagged, and an
optional OCR callable can fill them in.
"""

from __future__ import annotations

import io
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from .config import IngestConfig
from .models import BBox, Block, Chapter, Document, LinkedElement, to_jsonable
from .textutil import join_hyphenated_lines, normalize_text

_CHAPTER_RE = re.compile(
    r"^(?:chapter|part)\s+([0-9]+|[ivxlcdm]+)\b[.:)\-\s]*(.*)$",
    re.IGNORECASE,
)
_DIALOGUE_RE = re.compile(r"^dialogue\b[.:)\-\s]*(.*)$", re.IGNORECASE)
_SPEAKER_RE = re.compile(
    r"^(?P<speaker>[A-Z][A-Za-z.'’\-]*(?:\s+[A-Z][A-Za-z.'’\-]*){0,3}):\s*(?P<rest>.*)$"
)
_NON_SPEAKERS = {
    "aside",
    "note",
    "figure",
    "fig",
    "chapter",
    "contents",
    "part",
    "section",
    "table",
}
_CAPTION_RE = re.compile(r"^(?:figure|fig\.?)\s+\d+\b", re.IGNORECASE)
_FOOTNOTE_LINE_RE = re.compile(r"^([0-9]+|[*†‡])\s+(\S.*)$")
_MARKER_RE = re.compile(r"^[0-9*†‡]+$")
_SENTINEL_RE = re.compile(r"\x01([0-9*†‡]+)\x01")
_DISPLAY_LATEX_RE = re.compile(r"^\$\$[\s\S]+\$\$$")
_INLINE_LATEX_RE = re.compile(r"\$\$[\s\S]+?\$\$|\$[^$\n]+\$")
_AUTHOR_YEAR_RE = re.compile(
    r"\(([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?),?\s+((?:1[89]|20)\d{2}[a-z]?)\)"
    r"|\b([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?),?\s+\(((?:1[89]|20)\d{2}[a-z]?)\)"
)
_NUMERIC_CITE_RE = re.compile(r"\[(\d{1,3})\]")
_XREF_RE = re.compile(r"\b((?:see\s+)?(?:chapter|section)\s+\d+)\b", re.IGNORECASE)
_TOC_LINE_RE = re.compile(r"^(?P<title>.+?)(?:\s*\.{2,}|\s{2,})\s*(?P<page>\d+)\s*$")
_CONTENTS_RE = re.compile(r"^(contents|table of contents)$", re.IGNORECASE)
_MATH_CHARS = set("=+-*/^_{}\\<>≤≥≠∞±×÷∑∫√")


@dataclass
class _Span:
    text: str
    size: float
    font: str
    flags: int
    bbox: tuple[float, float, float, float]


@dataclass
class _Line:
    spans: list[_Span]
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    size: float
    flags: int


@dataclass
class _Page:
    number: int
    width: float
    height: float
    lines: list[_Line]
    images: list[pymupdf.Rect]
    drawings: list[pymupdf.Rect]
    source: object


@dataclass
class _Range:
    page_start: int
    page_end: int
    title: str
    detection: str
    kind: str = "prose"


def extract_pdf(
    path: Path | str,
    dest_dir: Path | str,
    config: IngestConfig | None = None,
    ocr_fn=None,
) -> Document:
    """Extract `path` and write per-chapter JSON plus a manifest under `dest_dir`."""
    config = config or IngestConfig()
    pdf_path = Path(path)
    dest = Path(dest_dir)
    chapters_dir = dest / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(pdf_path)
    try:
        pages = [_read_page(document[index], index + 1) for index in range(document.page_count)]
        _strip_running_matter(pages, config)
        ranges, detection, outline_used = _detect_ranges(document, pages)
        title = (document.metadata or {}).get("title") or None
        if title is not None and not str(title).strip():
            title = None
        chapters: list[Chapter] = []
        warnings: list[str] = []
        for index, spec in enumerate(ranges, start=1):
            chapter = _build_chapter(
                document,
                pages,
                spec,
                index,
                chapters_dir,
                config,
                ocr_fn,
            )
            chapters.append(chapter)
            warnings.extend(chapter.warnings)
        result = Document(
            source_pdf=str(pdf_path),
            title=str(title) if title else None,
            chapters=chapters,
            outline_used=outline_used,
            detection=detection,
            warnings=warnings,
        )
    finally:
        document.close()

    manifest = {
        "source_pdf": result.source_pdf,
        "title": result.title,
        "outline_used": result.outline_used,
        "detection": result.detection,
        "warnings": result.warnings,
        "chapters": [
            {
                "index": chapter.index,
                "id": chapter.id,
                "title": chapter.title,
                "kind": chapter.kind,
                "page_start": chapter.page_start,
                "page_end": chapter.page_end,
                "detection": chapter.detection,
                "path": f"chapters/{chapter.index:03d}.json",
            }
            for chapter in chapters
        ],
    }
    for chapter in chapters:
        target = chapters_dir / f"{chapter.index:03d}.json"
        target.write_text(
            json.dumps(to_jsonable(chapter), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def load_manifest(ingest_dir: Path | str) -> dict:
    return json.loads((Path(ingest_dir) / "manifest.json").read_text(encoding="utf-8"))


def load_chapter(path: Path | str) -> Chapter:
    return Chapter.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _read_page(page: pymupdf.Page, number: int) -> _Page:
    payload = page.get_text("dict")
    spans: list[_Span] = []
    images: list[pymupdf.Rect] = []
    for block in payload.get("blocks", []):
        if block.get("type") == 1:
            images.append(pymupdf.Rect(block["bbox"]))
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text") or ""
                if text == "":
                    continue
                spans.append(
                    _Span(
                        text=text,
                        size=float(span.get("size") or 0),
                        font=str(span.get("font") or ""),
                        flags=int(span.get("flags") or 0),
                        bbox=tuple(span["bbox"]),
                    )
                )
    drawings: list[pymupdf.Rect] = []
    page_area = float(page.rect.width * page.rect.height) or 1.0
    for drawing in page.get_drawings():
        rect = pymupdf.Rect(drawing.get("rect"))
        if rect.is_empty:
            continue
        if rect.height < 1.6 and rect.width > 30:
            continue
        if rect.width < 1.6 and rect.height > 30:
            continue
        if rect.get_area() > page_area * 0.75:
            continue
        drawings.append(rect)
    return _Page(
        number=number,
        width=float(page.rect.width),
        height=float(page.rect.height),
        lines=_cluster_lines(spans, number),
        images=images,
        drawings=drawings,
        source=page,
    )


def _cluster_lines(spans: list[_Span], page_number: int) -> list[_Line]:
    ordered = sorted(spans, key=lambda span: (span.bbox[1], span.bbox[0]))
    groups: list[list[_Span]] = []
    for span in ordered:
        if not span.text:
            continue
        placed = False
        for group in groups:
            if any(abs(span.bbox[1] - other.bbox[1]) <= 5 for other in group):
                group.append(span)
                placed = True
                break
        if not placed:
            groups.append([span])
    lines: list[_Line] = []
    for group in groups:
        group.sort(key=lambda span: span.bbox[0])
        lines.append(_line_from_spans(group, page_number))
    lines.sort(key=lambda line: (line.bbox[1], line.bbox[0]))
    return lines


def _line_from_spans(spans: list[_Span], page_number: int) -> _Line:
    sizes = [span.size for span in spans if span.text.strip()]
    main = statistics.median(sizes) if sizes else 0
    pieces: list[str] = []
    flags = 0
    for span in spans:
        token = normalize_text(span.text)
        stripped = token.strip()
        if stripped and _MARKER_RE.fullmatch(stripped) and main and span.size <= main * 0.75:
            pieces.append(f"\x01{stripped}\x01")
            continue
        pieces.append(token)
        flags |= span.flags
    text = re.sub(r"[ \t]+", " ", "".join(pieces)).strip()
    bbox = (
        min(span.bbox[0] for span in spans),
        min(span.bbox[1] for span in spans),
        max(span.bbox[2] for span in spans),
        max(span.bbox[3] for span in spans),
    )
    body_sizes = [
        span.size
        for span in spans
        if span.text.strip()
        and not (main and span.size <= main * 0.75 and _MARKER_RE.fullmatch(span.text.strip()))
    ]
    size = statistics.median(body_sizes) if body_sizes else main
    return _Line(spans=spans, page=page_number, bbox=bbox, text=text, size=float(size or 0), flags=flags)


def _strip_running_matter(pages: list[_Page], config: IngestConfig) -> None:
    ratio = config.header_footer_band_ratio
    top_counts: dict[str, int] = {}
    bottom_counts: dict[str, int] = {}
    for page in pages:
        seen_top: set[str] = set()
        seen_bottom: set[str] = set()
        for line in page.lines:
            key = _norm_key(line.text)
            if not key:
                continue
            if line.bbox[1] < page.height * ratio:
                seen_top.add(key)
            if line.bbox[1] > page.height * (1 - ratio) and not _FOOTNOTE_LINE_RE.match(line.text):
                seen_bottom.add(key)
        for key in seen_top:
            top_counts[key] = top_counts.get(key, 0) + 1
        for key in seen_bottom:
            bottom_counts[key] = bottom_counts.get(key, 0) + 1
    repeated_top = {key for key, count in top_counts.items() if count >= 2}
    repeated_bottom = {key for key, count in bottom_counts.items() if count >= 2 and not key.isdigit()}
    for page in pages:
        kept: list[_Line] = []
        for line in page.lines:
            key = _norm_key(line.text)
            in_top = line.bbox[1] < page.height * ratio
            in_bottom = line.bbox[1] > page.height * (1 - ratio)
            if in_top and key in repeated_top:
                continue
            if in_bottom and key.isdigit():
                continue
            if in_bottom and key in repeated_bottom:
                continue
            kept.append(line)
        page.lines = kept


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _detect_ranges(document: pymupdf.Document, pages: list[_Page]) -> tuple[list[_Range], str, bool]:
    outline = _outline_entries(document)
    if outline:
        ranges = _with_front_matter(_ranges_from_entries(outline, len(pages), "outline"), len(pages))
        return ranges, "outline", True
    toc = _toc_entries(pages)
    if toc:
        ranges = _with_front_matter(_ranges_from_entries(toc, len(pages), "toc"), len(pages))
        return ranges, "toc", False
    heuristic = _heuristic_entries(pages)
    if heuristic:
        ranges = _with_front_matter(_ranges_from_entries(heuristic, len(pages), "heuristic"), len(pages))
        return ranges, "heuristic", False
    title = pages[0].lines[0].text if pages and pages[0].lines else "Document"
    return [_Range(1, len(pages), title, "heuristic")], "heuristic", False


def _outline_entries(document: pymupdf.Document) -> list[tuple[str, int]]:
    try:
        toc = document.get_toc() or []
    except Exception:
        return []
    entries = []
    for item in toc:
        if len(item) < 3 or int(item[0]) != 1:
            continue
        title = str(item[1]).strip()
        page = int(item[2])
        if page < 1 or _CONTENTS_RE.match(title):
            continue
        entries.append((title, page))
    return entries


def _toc_entries(pages: list[_Page]) -> list[tuple[str, int]]:
    for page in pages:
        texts = [line.text.strip() for line in page.lines]
        if not any(_CONTENTS_RE.match(text) for text in texts):
            continue
        found: list[tuple[str, int]] = []
        for text in texts:
            match = _TOC_LINE_RE.match(text)
            if not match:
                continue
            title = match.group("title").strip(" .")
            if _CONTENTS_RE.match(title):
                continue
            found.append((title, int(match.group("page"))))
        if len(found) >= 2:
            return found
    return []


def _heuristic_entries(pages: list[_Page]) -> list[tuple[str, int]]:
    sizes = [line.size for page in pages for line in page.lines if line.size]
    body = statistics.median(sizes) if sizes else 11
    entries: list[tuple[str, int]] = []
    seen_pages: set[int] = set()
    for page in pages:
        for line in page.lines:
            if page.number in seen_pages:
                break
            title = _chapter_title(line, body)
            if title:
                entries.append((title, page.number))
                seen_pages.add(page.number)
    return entries


def _chapter_title(line: _Line, body: float) -> str | None:
    text = line.text.strip()
    if not text or _CAPTION_RE.match(text):
        return None
    if _CHAPTER_RE.match(text) or _DIALOGUE_RE.match(text):
        return text
    if line.size >= max(body * 1.5, 16) and len(text) < 80 and len(text.split()) <= 14:
        if text.endswith(".") and not _CHAPTER_RE.match(text):
            return None
        return text
    return None


def _ranges_from_entries(entries: list[tuple[str, int]], page_count: int, detection: str) -> list[_Range]:
    ordered = sorted(entries, key=lambda item: item[1])
    ranges: list[_Range] = []
    for index, (title, start) in enumerate(ordered):
        end = ordered[index + 1][1] - 1 if index + 1 < len(ordered) else page_count
        end = max(start, min(page_count, end))
        start = max(1, min(page_count, start))
        ranges.append(_Range(start, end, title, detection))
    return ranges


def _with_front_matter(ranges: list[_Range], page_count: int) -> list[_Range]:
    del page_count
    if not ranges or ranges[0].page_start <= 1:
        return ranges
    front = _Range(1, ranges[0].page_start - 1, "Front matter", ranges[0].detection, kind="front_matter")
    return [front, *ranges]


def _build_chapter(
    document: pymupdf.Document,
    pages: list[_Page],
    spec: _Range,
    index: int,
    chapters_dir: Path,
    config: IngestConfig,
    ocr_fn,
) -> Chapter:
    blocks: list[Block] = []
    elements: list[LinkedElement] = []
    warnings: list[str] = []
    scanned: list[int] = []
    body_sizes = [
        line.size
        for page in pages
        if spec.page_start <= page.number <= spec.page_end
        for line in page.lines
        if line.size
    ]
    body_size = statistics.median(body_sizes) if body_sizes else 11
    for page in pages:
        if not (spec.page_start <= page.number <= spec.page_end):
            continue
        footnotes = _take_footnotes(page, config)
        captions, formulas, body = _split_special_lines(page.lines)
        if _is_scanned(page, body, captions, formulas):
            scanned.append(page.number)
            ocr_text = _run_ocr(document[page.number - 1], config, ocr_fn, warnings)
            if ocr_text.strip():
                blocks.append(
                    _make_block(index, len(blocks) + 1, "paragraph", ocr_text.strip(), page, None)
                )
            else:
                warnings.append(f"scanned page {page.number} has no text layer")
                blocks.append(
                    _make_block(
                        index,
                        len(blocks) + 1,
                        "scanned_page",
                        f"Scanned page with no text layer (page {page.number}).",
                        page,
                        None,
                    )
                )
            continue
        page_blocks = _blocks_from_lines(body, body_size, index, page, len(blocks))
        _link_inline_notes(page_blocks, footnotes, page.number, elements, warnings)
        _link_inline_formulas(page_blocks, page, elements, chapters_dir, index, config, document)
        for line in formulas:
            block = _make_block(
                index,
                len(blocks) + len(page_blocks) + 1,
                "formula",
                line.text,
                page,
                line.bbox,
            )
            page_blocks.append(block)
            _add_formula_element(
                block,
                line.text,
                line.bbox,
                page,
                document,
                chapters_dir,
                index,
                config,
                elements,
                display=True,
            )
        blocks.extend(page_blocks)
        _add_figures(
            document[page.number - 1],
            page,
            captions,
            blocks,
            elements,
            chapters_dir,
            index,
            config,
        )
    by_id = {block.id: block for block in blocks}
    for number, block in enumerate(blocks):
        block.stream_index = number
    for element in elements:
        anchor = by_id.get(element.anchor_block_id)
        if anchor is not None:
            element.stream_index = anchor.stream_index
    dialogue = sum(1 for block in blocks if block.kind == "dialogue_turn")
    prose = sum(1 for block in blocks if block.kind in {"paragraph", "aside"})
    if spec.kind == "front_matter":
        kind = "front_matter"
    elif dialogue and dialogue >= prose:
        kind = "dialogue"
    else:
        kind = "prose"
    chapter = Chapter(
        id=f"ch-{index:03d}",
        index=index,
        title=spec.title,
        kind=kind,
        page_start=spec.page_start,
        page_end=spec.page_end,
        detection=spec.detection,
        blocks=blocks,
        elements=elements,
        scanned_pages=scanned,
        warnings=warnings,
    )
    _attach_citations(chapter)
    return chapter


def _make_block(chapter_index: int, number: int, kind: str, text: str, page: _Page, bbox) -> Block:
    box = None
    if bbox is not None:
        box = BBox(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return Block(
        id=f"b-{chapter_index:03d}-{number - 1:04d}",
        kind=kind,
        text=text,
        page_start=page.number,
        page_end=page.number,
        stream_index=number - 1,
        bbox=box,
    )


def _take_footnotes(page: _Page, config: IngestConfig) -> dict[str, str]:
    notes: dict[str, str] = {}
    kept: list[_Line] = []
    threshold = page.height * config.footnote_region_ratio
    for line in page.lines:
        match = _FOOTNOTE_LINE_RE.match(line.text.strip())
        if match and line.bbox[1] >= threshold and line.size <= 10:
            marker, text = match.group(1), match.group(2).strip()
            previous = notes.get(marker, "")
            notes[marker] = f"{previous} {text}".strip() if previous else text
            continue
        kept.append(line)
    page.lines = kept
    return notes


def _split_special_lines(lines: list[_Line]) -> tuple[list[_Line], list[_Line], list[_Line]]:
    captions: list[_Line] = []
    formulas: list[_Line] = []
    body: list[_Line] = []
    for line in lines:
        if _CAPTION_RE.match(line.text.strip()):
            captions.append(line)
        elif _is_display_formula(line.text):
            formulas.append(line)
        else:
            body.append(line)
    return captions, formulas, body


def _is_display_formula(text: str) -> bool:
    stripped = text.strip()
    if _DISPLAY_LATEX_RE.match(stripped):
        return True
    compact = [char for char in stripped if not char.isspace()]
    if len(compact) < 4 or len(stripped) > 180:
        return False
    ratio = sum(1 for char in compact if char in _MATH_CHARS or char == "$") / len(compact)
    return ratio >= 0.45 and not stripped.endswith(".")


def _is_scanned(page: _Page, body: list[_Line], captions: list[_Line], formulas: list[_Line]) -> bool:
    if body or captions or formulas:
        return False
    page_area = page.width * page.height or 1
    return any(rect.get_area() >= page_area * 0.45 for rect in page.images)


def _run_ocr(page: pymupdf.Page, config: IngestConfig, ocr_fn, warnings: list[str]) -> str:
    if ocr_fn is None and not config.ocr_enabled:
        return ""
    pixmap = page.get_pixmap(dpi=config.image_dpi, alpha=False)
    png = pixmap.tobytes("png")
    if ocr_fn is not None:
        return str(ocr_fn(png) or "")
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        warnings.append("ocr_enabled is set but pytesseract or pillow is not installed")
        return ""
    try:
        return pytesseract.image_to_string(Image.open(io.BytesIO(png))) or ""
    except Exception as exc:  # pragma: no cover - depends on a local binary
        warnings.append(f"ocr failed: {exc}")
        return ""


def _blocks_from_lines(
    lines: list[_Line],
    body_size: float,
    chapter_index: int,
    page: _Page,
    already: int,
) -> list[Block]:
    groups: list[list[_Line]] = []
    current: list[_Line] = []
    for line in lines:
        if not current:
            current = [line]
            continue
        if _starts_new_block(current[-1], line, body_size):
            groups.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        groups.append(current)
    blocks: list[Block] = []
    pending_speaker: str | None = None
    for group in groups:
        text = join_hyphenated_lines([line.text for line in group])
        bbox = (
            min(line.bbox[0] for line in group),
            min(line.bbox[1] for line in group),
            max(line.bbox[2] for line in group),
            max(line.bbox[3] for line in group),
        )
        speaker = _speaker_name(text)
        if speaker and not text.split(":", 1)[1].strip():
            pending_speaker = speaker
            continue
        if speaker:
            rest = text.split(":", 1)[1].strip()
            block = _make_block(chapter_index, already + len(blocks) + 1, "dialogue_turn", rest, page, bbox)
            block.speaker = speaker
            pending_speaker = None
        elif pending_speaker:
            block = _make_block(chapter_index, already + len(blocks) + 1, "dialogue_turn", text, page, bbox)
            block.speaker = pending_speaker
            pending_speaker = None
        elif _is_heading_text(text, group[-1], body_size):
            block = _make_block(chapter_index, already + len(blocks) + 1, "heading", text, page, bbox)
            block.level = 1 if _CHAPTER_RE.match(text) or _DIALOGUE_RE.match(text) else 2
        elif _is_aside(text, group[-1]):
            block = _make_block(chapter_index, already + len(blocks) + 1, "aside", text, page, bbox)
        else:
            block = _make_block(chapter_index, already + len(blocks) + 1, "paragraph", text, page, bbox)
        blocks.append(block)
    return blocks


def _starts_new_block(prev: _Line, line: _Line, body_size: float) -> bool:
    if _is_heading_line(prev, body_size) or _is_heading_line(line, body_size):
        return True
    if _speaker_name(prev.text) or _speaker_name(line.text):
        return True
    if abs(line.size - prev.size) > 2:
        return True
    gap = line.bbox[1] - prev.bbox[3]
    return gap > max(prev.size * 0.65, 6)


def _is_heading_line(line: _Line, body_size: float) -> bool:
    return _is_heading_text(line.text, line, body_size)


def _is_heading_text(text: str, line: _Line, body_size: float) -> bool:
    if _speaker_name(text) or _CAPTION_RE.match(text):
        return False
    if len(text) > 90:
        return False
    if _CHAPTER_RE.match(text) or _DIALOGUE_RE.match(text):
        return True
    if line.size >= max(body_size * 1.25, body_size + 1.5):
        return True
    if (line.flags & 16) and line.size >= body_size * 1.05 and len(text) < 80:
        return True
    return False


def _speaker_name(text: str) -> str | None:
    match = _SPEAKER_RE.match(text.strip())
    if not match:
        return None
    speaker = match.group("speaker").strip()
    if speaker.lower() in _NON_SPEAKERS or speaker.lower().startswith("figure"):
        return None
    return speaker


def _is_aside(text: str, line: _Line) -> bool:
    if text.lower().startswith("aside:"):
        return True
    return bool(line.flags & 2) and not bool(line.flags & 16) and not _speaker_name(text)


def _link_inline_notes(blocks, footnotes, page_number, elements, warnings) -> None:
    for block in blocks:
        def replace(match: re.Match, block=block) -> str:
            marker = match.group(1)
            element_id = f"fn-p{page_number}-{marker}"
            text = footnotes.get(marker, "")
            if marker not in footnotes:
                warnings.append(f"footnote marker {marker} on page {page_number} has no note text")
            elements.append(
                LinkedElement(
                    id=element_id,
                    kind="footnote",
                    anchor_block_id=block.id,
                    anchor_offset=match.start(),
                    stream_index=block.stream_index,
                    page=page_number,
                    text=text,
                    marker=marker,
                )
            )
            return f"[[footnote:{element_id}]]"

        block.text = _SENTINEL_RE.sub(replace, block.text)


def _link_inline_formulas(blocks, page, elements, chapters_dir, chapter_index, config, document) -> None:
    for block in list(blocks):
        if block.kind == "formula":
            continue
        for match in _INLINE_LATEX_RE.finditer(block.text):
            latex = match.group(0)
            bbox = _span_bbox_containing(block, page, "$") or (
                (block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1) if block.bbox else None
            )
            _add_formula_element(
                block,
                latex,
                bbox,
                page,
                document,
                chapters_dir,
                chapter_index,
                config,
                elements,
                display=False,
                offset=match.start(),
            )


def _span_bbox_containing(block: Block, page: _Page, needle: str):
    hits = []
    for line in page.lines:
        for span in line.spans:
            if needle in span.text and block.bbox:
                if span.bbox[1] >= block.bbox.y0 - 2 and span.bbox[3] <= block.bbox.y1 + 2:
                    hits.append(span.bbox)
    if not hits:
        return None
    return (
        min(item[0] for item in hits),
        min(item[1] for item in hits),
        max(item[2] for item in hits),
        max(item[3] for item in hits),
    )


def _add_formula_element(
    block,
    raw,
    bbox,
    page,
    document,
    chapters_dir,
    chapter_index,
    config,
    elements,
    *,
    display: bool,
    offset: int = 0,
) -> None:
    element_id = f"formula-p{page.number}-{sum(1 for item in elements if item.kind == 'formula') + 1}"
    image_path = None
    box = None
    if bbox is not None:
        padded = (bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2)
        image_path, box = _render_clip(
            document[page.number - 1],
            padded,
            chapters_dir / f"{chapter_index:03d}" / "formulas" / f"{element_id}.png",
            chapters_dir,
            config.image_dpi,
        )
    latex = raw.strip() if ("\\" in raw or "$" in raw) else None
    elements.append(
        LinkedElement(
            id=element_id,
            kind="formula",
            anchor_block_id=block.id,
            anchor_offset=offset,
            stream_index=block.stream_index,
            page=page.number,
            text=raw.strip(),
            image_path=image_path,
            bbox=box,
            dpi=config.image_dpi if image_path else None,
            latex=latex,
            display=display,
        )
    )


def _add_figures(page_obj, page, captions, blocks, elements, chapters_dir, chapter_index, config) -> None:
    page_area = page.width * page.height or 1
    regions: list[dict] = []
    for rect in page.images:
        if rect.get_area() >= page_area * config.full_page_image_ratio:
            continue
        if rect.width < 8 or rect.height < 8:
            continue
        regions.append({"source": "raster", "rect": pymupdf.Rect(rect)})
    for rect in _cluster_rects(page.drawings, config.vector_cluster_gap):
        if rect.get_area() < config.min_vector_area:
            continue
        regions.append({"source": "vector", "rect": rect})
    used: set[int] = set()
    for region in regions:
        caption_index = _match_caption(region["rect"], captions, used)
        caption_text = captions[caption_index].text.strip() if caption_index is not None else None
        if caption_index is not None:
            used.add(caption_index)
        anchor = _nearest_block(blocks, region["rect"].y0, page.number)
        if anchor is None:
            continue
        element_id = f"fig-p{page.number}-{sum(1 for item in elements if item.kind == 'figure') + 1}"
        clip = pymupdf.Rect(region["rect"])
        image_path, box = _render_clip(
            page_obj,
            (clip.x0, clip.y0, clip.x1, clip.y1),
            chapters_dir / f"{chapter_index:03d}" / "figures" / f"{element_id}.png",
            chapters_dir,
            config.image_dpi,
        )
        elements.append(
            LinkedElement(
                id=element_id,
                kind="figure",
                anchor_block_id=anchor.id,
                anchor_offset=0,
                stream_index=anchor.stream_index,
                page=page.number,
                text=caption_text or "",
                image_path=image_path,
                bbox=box,
                dpi=config.image_dpi if image_path else None,
                caption=caption_text,
                figure_source=region["source"],
            )
        )


def _match_caption(rect: pymupdf.Rect, captions: list[_Line], used: set[int]) -> int | None:
    best = None
    best_gap = 1e9
    for index, caption in enumerate(captions):
        if index in used:
            continue
        horizontal = min(rect.x1, caption.bbox[2]) - max(rect.x0, caption.bbox[0])
        centers = abs(((rect.x0 + rect.x1) / 2) - ((caption.bbox[0] + caption.bbox[2]) / 2))
        if horizontal <= 0 and centers > 80:
            continue
        gap = caption.bbox[1] - rect.y1
        if -30 <= gap <= 90 and abs(gap) < best_gap:
            best = index
            best_gap = abs(gap)
    return best


def _nearest_block(blocks: list[Block], y: float, page_number: int) -> Block | None:
    on_page = [block for block in blocks if block.page_start == page_number and block.kind != "caption"]
    if not on_page:
        return None
    preceding = [block for block in on_page if block.bbox and block.bbox.y1 <= y + 8]
    if preceding:
        return preceding[-1]
    return on_page[0]


def _cluster_rects(rects: list[pymupdf.Rect], gap: float) -> list[pymupdf.Rect]:
    clusters: list[pymupdf.Rect] = []
    for rect in rects:
        hits: list[pymupdf.Rect] = []
        rest: list[pymupdf.Rect] = []
        for cluster in clusters:
            grown = pymupdf.Rect(cluster.x0 - gap, cluster.y0 - gap, cluster.x1 + gap, cluster.y1 + gap)
            if grown.intersects(rect):
                hits.append(cluster)
            else:
                rest.append(cluster)
        if not hits:
            clusters.append(pymupdf.Rect(rect))
            continue
        merged = pymupdf.Rect(rect)
        for hit in hits:
            merged.include_rect(hit)
        clusters = rest + [merged]
    return clusters


def _render_clip(page, bbox, dest: Path, chapters_dir: Path, dpi: int) -> tuple[str | None, BBox | None]:
    clip = pymupdf.Rect(bbox) & page.rect
    if clip.is_empty or clip.width < 2 or clip.height < 2:
        return None, None
    dest.parent.mkdir(parents=True, exist_ok=True)
    pixmap = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
    pixmap.save(dest)
    relative = dest.relative_to(chapters_dir).as_posix()
    box = BBox(float(clip.x0), float(clip.y0), float(clip.x1), float(clip.y1))
    return relative, box


def _attach_citations(chapter: Chapter) -> None:
    """Author-year, numeric, and cross-reference citations, anchored in body blocks."""
    for block in chapter.blocks:
        if block.kind in {"heading", "formula", "scanned_page"}:
            continue
        for match in _NUMERIC_CITE_RE.finditer(block.text):
            raw = match.group(0)
            chapter.elements.append(
                LinkedElement(
                    id=f"cite-p{block.page_start}-n{match.group(1)}",
                    kind="citation",
                    anchor_block_id=block.id,
                    anchor_offset=match.start(),
                    stream_index=block.stream_index,
                    page=block.page_start,
                    text=raw,
                    citation_kind="numeric",
                )
            )
        for match in _AUTHOR_YEAR_RE.finditer(block.text):
            raw = match.group(0)
            chapter.elements.append(
                LinkedElement(
                    id=f"cite-p{block.page_start}-a{match.start()}",
                    kind="citation",
                    anchor_block_id=block.id,
                    anchor_offset=match.start(),
                    stream_index=block.stream_index,
                    page=block.page_start,
                    text=raw,
                    citation_kind="author_year",
                )
            )
        for match in _XREF_RE.finditer(block.text):
            raw = match.group(1)
            if match.start() == 0 and _CHAPTER_RE.match(block.text):
                continue
            chapter.elements.append(
                LinkedElement(
                    id=f"cite-p{block.page_start}-x{match.start()}",
                    kind="citation",
                    anchor_block_id=block.id,
                    anchor_offset=match.start(),
                    stream_index=block.stream_index,
                    page=block.page_start,
                    text=raw,
                    citation_kind="cross_reference",
                )
            )
