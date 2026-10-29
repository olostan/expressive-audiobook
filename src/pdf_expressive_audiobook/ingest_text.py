"""Stage 1 for plain text and Markdown: one JSON document model per chapter.

The output is the same manifest and chapter files the PDF extractor writes, so
chunking, preparation and speech work unchanged. Chapters come from Markdown
headings, or from chapter-style lines in plain text, or, when a file has
neither, from a split by length at scene breaks and paragraph boundaries.
There are no pages, so a "page" is a stretch of about ``WORDS_PER_PAGE`` words.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import IngestConfig
from .ingest import _speaker_name
from .models import Block, Chapter, Document, to_jsonable
from .textutil import join_hyphenated_lines, normalize_text, word_count

MARKDOWN_SUFFIXES = {".md", ".markdown"}
TEXT_SUFFIXES = {".txt", ".text"}
WORDS_PER_PAGE = 300
SPLIT_TARGET_WORDS = 3000
SPLIT_MIN_WORDS = 4500
MIN_SPEAKER_TURNS = 3

_NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    "sixteen|seventeen|eighteen|nineteen|twenty"
)
_STRONG_HEADING_RE = re.compile(
    rf"^(?:chapter|book|part|section|letter|story|tale|adventure)\s+(?:[0-9]+|[ivxlcdm]+|{_NUMBER_WORDS})\b.*$",
    re.IGNORECASE,
)
_SPECIAL_HEADING_RE = re.compile(
    r"^(?:prologue|epilogue|preface|foreword|introduction|interlude|afterword)\b[\s\S]{0,40}$",
    re.IGNORECASE,
)
_BARE_NUMERAL_RE = re.compile(r"^(?:[IVXLCDM]{1,7}|\d{1,3})[.)]?$")
_NUMBERED_TITLE_RE = re.compile(r"^(?:[IVXLCDM]{1,7}|\d{1,3})[.)]\s+\S.{0,70}$")
_NUMBER_ONLY_TITLE_RE = re.compile(
    rf"^(?:(?:chapter|book|part|section|letter)\s+)?(?:[0-9]+|[IVXLCDM]+|{_NUMBER_WORDS})[.:)]?$",
    re.IGNORECASE,
)
_SCENE_BREAK_RE = re.compile(r"^(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,}|(?:~\s*){3,}|(?:#\s*){3,}|•\s*•\s*•)$")
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_MD_FENCE_RE = re.compile(r"^(```|~~~)")
_MD_LIST_RE = re.compile(r"^(?:[-*+]|\d+[.)])\s+(.*)$")
_MD_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)


@dataclass
class _Item:
    kind: str  # "heading", "para" or "break"
    text: str = ""
    level: int = 0


def is_text_source(path: Path | str) -> bool:
    return Path(path).suffix.lower() in MARKDOWN_SUFFIXES | TEXT_SUFFIXES


def extract_text(path: Path | str, dest_dir: Path | str, config: IngestConfig | None = None) -> Document:
    """Read a ``.md`` or ``.txt`` file and write per-chapter JSON plus a manifest under ``dest_dir``."""
    source = Path(path)
    dest = Path(dest_dir)
    chapters_dir = dest / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)

    text = normalize_text(_read(source).replace("\r\n", "\n").replace("\r", "\n"))
    markdown = source.suffix.lower() in MARKDOWN_SUFFIXES
    meta: dict = {}
    if markdown:
        meta, text = _front_matter(text)
        items = _markdown_items(text)
    else:
        items = _text_items(text)
    title = _clean(str(meta.get("title") or "")) or None
    author = _clean(str(meta.get("author") or "")) or None

    sections, detection, book_title = _sections(items, markdown, title, source)
    title = title or book_title
    if len(sections) == 1:
        split = _split_by_length(sections[0][1], sections[0][0])
        if len(split) > 1:
            sections, detection = split, "size"
    if not any(word_count(item.text) for _, body in sections for item in body if item.kind == "para"):
        raise ValueError(f"{source.name} has no readable text")

    chapters = _build_chapters(sections, detection)
    manifest = {
        "source_file": str(source),
        "source_pdf": str(source),
        "source_kind": "markdown" if markdown else "text",
        "title": title,
        "outline_used": detection in {"markdown-headings", "text-headings"},
        "detection": detection,
        "warnings": [],
        "page_count": chapters[-1].page_end if chapters else 0,
        "author": author,
        "contents_text": "",
        "skipped_pages": [],
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
        (chapters_dir / f"{chapter.index:03d}.json").write_text(
            json.dumps(to_jsonable(chapter), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return Document(
        source_pdf=str(source),
        title=title,
        chapters=chapters,
        outline_used=manifest["outline_used"],
        detection=detection,
    )


def _read(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _front_matter(text: str) -> tuple[dict, str]:
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    try:
        data = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}, text
    return (data if isinstance(data, dict) else {}), text[match.end():]


def _clean(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------- parsing


def _markdown_items(text: str) -> list[_Item]:
    items: list[_Item] = []
    para: list[str] = []
    fence: str | None = None

    def flush() -> None:
        if para:
            joined = _inline(" ".join(para))
            if joined:
                items.append(_Item("para", joined))
            para.clear()

    for raw in _MD_COMMENT_RE.sub("", text).split("\n"):
        line = raw.strip()
        if fence:
            if line.startswith(fence):
                fence = None
            continue
        opening = _MD_FENCE_RE.match(line)
        if opening:
            flush()
            fence = opening.group(1)
            continue
        if not line:
            flush()
            continue
        heading = _MD_HEADING_RE.match(line)
        if heading:
            flush()
            title = _inline(heading.group(2))
            if title:
                items.append(_Item("heading", title, len(heading.group(1))))
            continue
        if para and len(para) == 1 and re.fullmatch(r"=+|-{2,}", line):
            items.append(_Item("heading", _inline(para.pop()), 1 if line[0] == "=" else 2))
            continue
        if _SCENE_BREAK_RE.match(line):
            flush()
            items.append(_Item("break"))
            continue
        if line.startswith("|"):
            flush()
            continue
        while line.startswith(">"):
            line = line[1:].strip()
        listed = _MD_LIST_RE.match(line)
        if listed:
            flush()
            line = listed.group(1)
        if line:
            para.append(line)
    flush()
    return items


_INLINE_RULES = [
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), ""),
    (re.compile(r"\[\^[^\]]+\]"), ""),
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),
    (re.compile(r"\[([^\]]+)\]\[[^\]]*\]"), r"\1"),
    (re.compile(r"<https?://[^>]+>"), ""),
    (re.compile(r"</?[A-Za-z][^>]*>"), ""),
    (re.compile(r"`+([^`]*)`+"), r"\1"),
    (re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1"), r"\2"),
    (re.compile(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])"), r"\1"),
    (re.compile(r"(?<![\w_])_(?=\S)(.+?)(?<=\S)_(?![\w_])"), r"\1"),
    (re.compile(r"~~(.+?)~~"), r"\1"),
    (re.compile(r"\\([\\`*_{}\[\]()#+\-.!>])"), r"\1"),
]


def _inline(text: str) -> str:
    for pattern, replacement in _INLINE_RULES:
        text = pattern.sub(replacement, text)
    return _clean(text)


def _text_items(text: str) -> list[_Item]:
    stripped = text.strip("\n")
    if re.search(r"\n[ \t]*\n", stripped):
        raw = [part for part in re.split(r"\n[ \t]*\n", stripped)]
        paragraphs = [join_hyphenated_lines(part.split("\n")) for part in raw]
    else:  # no blank lines at all: one paragraph per line
        paragraphs = [_clean(line) for line in stripped.split("\n")]
    items: list[_Item] = []
    for paragraph in paragraphs:
        paragraph = _clean(paragraph)
        if not paragraph:
            continue
        items.append(_Item("break") if _SCENE_BREAK_RE.match(paragraph) else _Item("para", paragraph))
    return items


# ---------------------------------------------------------------- chapters

_Section = tuple[str, list[_Item]]


def _sections(items: list[_Item], markdown: bool, title: str | None, source: Path) -> tuple[list[_Section], str, str | None]:
    """Split items into ``(chapter title, items)`` sections. Returns the sections, how, and a book title."""
    fallback = title or _prettify(source.stem)
    if markdown:
        found = _markdown_sections(items, fallback)
        if found:
            return found
    else:
        found = _text_sections(items, fallback)
        if found:
            return found
    return [(fallback, items)], "none", None


def _prettify(stem: str) -> str:
    return _clean(re.sub(r"[_\-]+", " ", stem)).title() or "Untitled"


def _markdown_sections(items: list[_Item], fallback: str) -> tuple[list[_Section], str, str | None] | None:
    headings = [item for item in items if item.kind == "heading"]
    if not headings:
        return None
    book_title: str | None = None
    levels = sorted({item.level for item in headings})
    counts = {level: sum(1 for item in headings if item.level == level) for level in levels}
    chapter_level = next((level for level in levels if counts[level] >= 2), None)
    first_heading = headings[0]
    if chapter_level is not None and chapter_level > first_heading.level and counts[first_heading.level] == 1:
        book_title = first_heading.text  # a lone top heading names the book
    if chapter_level is None:
        top = levels[0]
        if counts[top] == 1 and len(levels) == 1:
            return [(first_heading.text, [i for i in items if i is not first_heading])], "markdown-headings", first_heading.text
        chapter_level = levels[-1] if len(levels) > 1 else top
        if len(levels) > 1:
            book_title = first_heading.text
    sections = _group(items, lambda item: item.kind == "heading" and item.level == chapter_level, book_title or fallback)
    if not sections:
        return None
    # A part-level split that gives huge chapters yields to the next level down.
    deeper = [level for level in levels if level > chapter_level and counts[level] >= 2 * len(sections)]
    if deeper and sum(word_count(i.text) for _, body in sections for i in body) / len(sections) > 6000:
        sections = _group(items, lambda item: item.kind == "heading" and item.level == deeper[0], book_title or fallback)
    return sections, "markdown-headings", book_title


def _group(items: list[_Item], is_start, front_title: str) -> list[_Section]:
    sections: list[_Section] = []
    current: tuple[str, list[_Item]] | None = None
    front: list[_Item] = []
    for item in items:
        if is_start(item):
            current = (item.text, [])
            sections.append(current)
        elif current is None:
            if item.kind != "heading" or not front:
                front.append(item)
        else:
            current[1].append(item)
    readable = sum(word_count(i.text) for i in front if i.kind == "para")
    if front and readable >= 40:
        sections.insert(0, (front_title if front_title else "Opening", front))
    return sections


def _text_sections(items: list[_Item], fallback: str) -> tuple[list[_Section], str, str | None] | None:
    for tier, minimum in ((_is_strong_heading, 1), (_is_numbered_heading, 2), (_is_caps_heading, 3)):
        hits = [index for index, item in enumerate(items) if item.kind == "para" and tier(item.text)]
        if len(hits) >= minimum:
            return _text_group(items, hits, fallback), "text-headings", None
    return None


def _is_strong_heading(text: str) -> bool:
    if "\n" in text or len(text) > 80:
        return False
    if _STRONG_HEADING_RE.match(text):
        return not text.rstrip().endswith((",",)) and len(text.split()) <= 12
    return bool(_SPECIAL_HEADING_RE.match(text)) and len(text.split()) <= 6


def _is_numbered_heading(text: str) -> bool:
    return len(text) <= 80 and bool(_BARE_NUMERAL_RE.match(text) or _NUMBERED_TITLE_RE.match(text))


def _is_caps_heading(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return (
        3 <= len(letters)
        and len(text) <= 60
        and len(text.split()) <= 8
        and text == text.upper()
        and not text.rstrip().endswith((".", "!", "?", ",", ":", ";"))
    )


def _text_group(items: list[_Item], hits: list[int], fallback: str) -> list[_Section]:
    sections: list[_Section] = []
    if hits[0] > 0:
        front = items[: hits[0]]
        if sum(word_count(i.text) for i in front if i.kind == "para") >= 40:
            sections.append((fallback, front))
    for position, start in enumerate(hits):
        end = hits[position + 1] if position + 1 < len(hits) else len(items)
        title = items[start].text
        body = items[start + 1 : end]
        if _NUMBER_ONLY_TITLE_RE.match(title) and body and body[0].kind == "para" and _looks_like_subtitle(body[0].text):
            title = f"{title.rstrip('.:)')}. {body[0].text}"
            body = body[1:]
        sections.append((title, body))
    return sections


def _looks_like_subtitle(text: str) -> bool:
    return len(text) <= 60 and len(text.split()) <= 8 and not text.rstrip().endswith((".", ",", ";", ":"))


def _split_by_length(items: list[_Item], title: str) -> list[_Section]:
    total = sum(word_count(i.text) for i in items if i.kind == "para")
    if total < SPLIT_MIN_WORDS:
        return [(title, items)]
    sections: list[list[_Item]] = [[]]
    words = consumed = 0
    for item in items:
        size = word_count(item.text) if item.kind == "para" else 0
        cut = words >= SPLIT_TARGET_WORDS or (words >= SPLIT_TARGET_WORDS * 0.6 and item.kind == "break")
        if cut and total - consumed >= SPLIT_TARGET_WORDS * 0.3:
            sections.append([])
            words = 0
            if item.kind == "break":
                continue
        sections[-1].append(item)
        words += size
        consumed += size
    return [(f"Part {number}", part) for number, part in enumerate(sections, start=1)]


# ---------------------------------------------------------------- blocks


def _build_chapters(sections: list[_Section], detection: str) -> list[Chapter]:
    chapters: list[Chapter] = []
    words_before = 0
    for index, (title, body) in enumerate(sections, start=1):
        blocks: list[Block] = []
        first = words_before // WORDS_PER_PAGE + 1

        def add(kind: str, text: str, level: int | None = None, speaker: str | None = None) -> None:
            nonlocal words_before
            page = words_before // WORDS_PER_PAGE + 1
            number = len(blocks)
            blocks.append(
                Block(
                    id=f"b-{index:03d}-{number:04d}",
                    kind=kind,
                    text=text,
                    page_start=page,
                    page_end=page,
                    stream_index=number,
                    speaker=speaker,
                    level=level,
                )
            )
            words_before += word_count(text)

        add("heading", title, 1)
        scripted = sum(1 for item in body if item.kind == "para" and _speaker_name(item.text)) >= MIN_SPEAKER_TURNS
        for item in body:
            if item.kind == "heading":
                add("heading", item.text, 2)
            elif item.kind == "para":
                speaker = _speaker_name(item.text) if scripted else None
                if speaker:
                    add("dialogue_turn", item.text.split(":", 1)[1].strip() or item.text, speaker=speaker)
                else:
                    add("paragraph", item.text)
        dialogue = sum(1 for block in blocks if block.kind == "dialogue_turn")
        prose = sum(1 for block in blocks if block.kind == "paragraph")
        chapters.append(
            Chapter(
                id=f"ch-{index:03d}",
                index=index,
                title=title,
                kind="dialogue" if dialogue and dialogue >= prose else "prose",
                page_start=first,
                page_end=max(first, (words_before - 1) // WORDS_PER_PAGE + 1),
                detection=detection,
                blocks=blocks,
            )
        )
    return chapters
