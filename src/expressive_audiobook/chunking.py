"""Stage 2: split a chapter into TTS chunks at sentence and turn boundaries.

Footnotes, figures, formulas, and citations are never dropped. Each one is
attached to the chunk that contains its anchor when the spoken-word budget
fits, otherwise the next chunk is responsible for speaking it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import ChunkingConfig
from .models import AssociatedRef, Chapter, Chunk, LinkedElement, to_jsonable
from .textutil import estimate_seconds, spoken_source_words, split_sentences


@dataclass
class _Piece:
    text: str
    block_ids: list[str]
    kind: str
    speaker: str | None
    pages: list[int]
    elements: list[tuple[LinkedElement, bool]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def estimate(self, config: ChunkingConfig) -> int:
        """Spoken-word estimate, including budgets for elements this piece speaks."""
        total = spoken_source_words(self.text)
        for element, speak in self.elements:
            if speak:
                total += _budget(element, config)
        return total


def chunk_chapter(chapter: Chapter, config: ChunkingConfig | None = None) -> list[Chunk]:
    """Return ordered chunks for one chapter."""
    config = config or ChunkingConfig()
    pieces = _pieces(chapter, config)
    groups = _pack(pieces, config)
    return [_chunk_from_group(chapter, order, group, config) for order, group in enumerate(groups)]


def write_chapter_jsonl(chunks: list[Chunk], path: Path | str) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(to_jsonable(chunk), ensure_ascii=False) for chunk in chunks]
    dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_chapter_jsonl(path: Path | str) -> list[Chunk]:
    chunks: list[Chunk] = []
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip():
            chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks


def _pieces(chapter: Chapter, config: ChunkingConfig) -> list[_Piece]:
    by_anchor: dict[str, list[LinkedElement]] = {}
    for element in chapter.elements:
        by_anchor.setdefault(element.anchor_block_id, []).append(element)
    pieces: list[_Piece] = []
    pending_headings: list = []
    for block in chapter.blocks:
        if block.kind == "heading":
            pending_headings.append(block)
            continue
        if block.kind == "caption":
            continue
        elements = list(by_anchor.get(block.id, []))
        heading_ids = [item.id for item in pending_headings]
        heading_text = "\n".join(item.text for item in pending_headings)
        pending_headings = []
        pages = [block.page_start]
        if block.kind == "scanned_page":
            pieces.append(
                _Piece(
                    text=block.text,
                    block_ids=heading_ids + [block.id],
                    kind="paragraph",
                    speaker=None,
                    pages=pages,
                    warnings=["scanned_page"],
                )
            )
            continue
        body = block.text.strip()
        text = f"{heading_text}\n{body}".strip() if heading_text else body
        slices = _slice_text(text, config.hard_words)
        unsplittable = spoken_source_words(text) > config.hard_words and len(slices) == 1
        if len(slices) == 1:
            piece = _Piece(
                text=slices[0],
                block_ids=heading_ids + [block.id],
                kind=block.kind,
                speaker=block.speaker,
                pages=pages,
                elements=[(element, True) for element in elements],
                warnings=["sentence_exceeds_hard_cap"] if unsplittable else [],
            )
            pieces.extend(_peel_until_fit(piece, config))
            continue
        buckets: list[list[LinkedElement]] = [[] for _ in slices]
        for element in elements:
            buckets[_slice_index(element, slices)].append(element)
        for index, slice_text in enumerate(slices):
            block_ids = heading_ids + [block.id] if index == 0 else [block.id]
            piece = _Piece(
                text=slice_text,
                block_ids=block_ids,
                kind=block.kind,
                speaker=block.speaker,
                pages=pages,
                elements=[(element, True) for element in buckets[index]],
            )
            pieces.extend(_peel_until_fit(piece, config))
    if pending_headings:
        pieces.append(
            _Piece(
                text="\n".join(item.text for item in pending_headings),
                block_ids=[item.id for item in pending_headings],
                kind="heading",
                speaker=None,
                pages=[pending_headings[-1].page_start],
            )
        )
    return pieces


def _slice_text(text: str, hard_words: int) -> list[str]:
    if spoken_source_words(text) <= hard_words:
        return [text]
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text]
    slices: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        words = spoken_source_words(sentence)
        if current and current_words + words > hard_words:
            slices.append(" ".join(current))
            current = [sentence]
            current_words = words
        else:
            current.append(sentence)
            current_words += words
    if current:
        slices.append(" ".join(current))
    return slices


def _slice_index(element: LinkedElement, slices: list[str]) -> int:
    for index, text in enumerate(slices):
        if element.kind == "footnote" and f"[[footnote:{element.id}]]" in text:
            return index
        if element.latex and element.latex in text:
            return index
        if element.kind == "citation" and element.text and element.text in text:
            return index
        if element.kind == "formula" and element.text and element.text in text:
            return index
    return 0


def _peel_until_fit(piece: _Piece, config: ChunkingConfig) -> list[_Piece]:
    """Move the largest speakable side elements onto following pieces when over the cap."""
    guard = 0
    peeled: list[_Piece] = []
    while piece.estimate(config) > config.hard_words and guard < 12:
        guard += 1
        movable = [
            (index, element)
            for index, (element, speak) in enumerate(piece.elements)
            if speak and element.kind in {"footnote", "figure", "formula"} and _budget(element, config) > 0
        ]
        if not movable:
            break
        index, element = max(movable, key=lambda item: _budget(item[1], config))
        piece.elements[index] = (element, False)
        peeled.append(_adjacent_piece(element, piece.pages))
    return [piece, *peeled]


def _adjacent_piece(element: LinkedElement, pages: list[int]) -> _Piece:
    if element.kind == "footnote":
        text = element.text
        kind = "footnote"
    elif element.kind == "figure":
        text = element.caption or element.text or element.id
        kind = "figure"
    else:
        text = element.text
        kind = "formula"
    return _Piece(
        text=text,
        block_ids=[element.anchor_block_id] if element.anchor_block_id else [],
        kind=kind,
        speaker=None,
        pages=pages or ([element.page] if element.page else []),
        elements=[(element, True)],
    )


def _pack(pieces: list[_Piece], config: ChunkingConfig) -> list[list[_Piece]]:
    groups: list[list[_Piece]] = []
    current: list[_Piece] = []

    def total(group: list[_Piece]) -> int:
        return sum(piece.estimate(config) for piece in group)

    def flush() -> None:
        if current:
            groups.append(list(current))
            current.clear()

    for piece in pieces:
        if not current:
            current.append(piece)
            continue
        projected = total(current) + piece.estimate(config)
        if total(current) >= config.min_words and projected > config.target_words:
            flush()
            current.append(piece)
            continue
        if projected > config.hard_words:
            flush()
            current.append(piece)
            continue
        current.append(piece)
    flush()
    return groups


def _chunk_from_group(
    chapter: Chapter,
    order: int,
    group: list[_Piece],
    config: ChunkingConfig,
) -> Chunk:
    text = "\n\n".join(piece.text for piece in group if piece.text).strip()
    words = sum(piece.estimate(config) for piece in group)
    speakers: list[str] = []
    pages: list[int] = []
    block_ids: list[str] = []
    warnings: list[str] = []
    associated: list[AssociatedRef] = []
    seen: set[str] = set()
    for piece in group:
        if piece.speaker and piece.speaker not in speakers:
            speakers.append(piece.speaker)
        for page in piece.pages:
            if page not in pages:
                pages.append(page)
        for block_id in piece.block_ids:
            if block_id not in block_ids:
                block_ids.append(block_id)
        warnings.extend(piece.warnings)
        for element, speak in piece.elements:
            if element.id in seen and speak:
                for ref in associated:
                    if ref.id == element.id:
                        ref.speak = True
                        ref.placement = "adjacent"
                        ref.estimated_spoken_words = _budget(element, config)
                continue
            if element.id in seen:
                continue
            seen.add(element.id)
            associated.append(
                AssociatedRef(
                    id=element.id,
                    kind=element.kind,
                    anchor=element.anchor_block_id,
                    stream_index=element.stream_index,
                    text=element.text,
                    estimated_spoken_words=_budget(element, config) if speak else 0,
                    speak=speak,
                    placement="inline" if speak else "adjacent",
                    image_path=element.image_path,
                    page=element.page,
                    caption=element.caption,
                    latex=element.latex,
                )
            )
    return Chunk(
        id=f"c{chapter.index:03d}-{order:04d}",
        chapter_id=chapter.id,
        chapter_index=chapter.index,
        chapter_title=chapter.title,
        order=order,
        kind=_chunk_kind(group),
        text=text,
        speakers=speakers,
        source_pages=pages,
        estimated_duration_sec=round(estimate_seconds(words, config.wpm), 3),
        word_count=words,
        associated=associated,
        block_ids=block_ids,
        warnings=warnings,
        segments=[
            {"speaker": piece.speaker or "narrator", "text": piece.text, "block_ids": list(piece.block_ids)}
            for piece in group
            if piece.text
        ],
    )


def _chunk_kind(group: list[_Piece]) -> str:
    kinds = {piece.kind for piece in group}
    if kinds <= {"footnote"}:
        return "footnote"
    if kinds <= {"figure"}:
        return "figure"
    if kinds <= {"formula"}:
        return "formula"
    if kinds <= {"dialogue_turn"}:
        return "dialogue"
    if kinds <= {"aside"}:
        return "aside"
    return "prose"


def _budget(element: LinkedElement, config: ChunkingConfig) -> int:
    if element.kind == "figure":
        return config.figure_description_sentences * config.words_per_spoken_sentence
    if element.kind == "formula":
        return config.formula_description_words
    if element.kind == "footnote":
        return spoken_source_words(element.text) + config.footnote_lead_in_words
    return 0
