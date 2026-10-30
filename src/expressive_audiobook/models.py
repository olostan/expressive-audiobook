"""Dataclasses shared across stages. JSON on disk is the stage boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {key: to_jsonable(value) for key, value in asdict(obj).items()}
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    return obj


@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_dict(cls, data: dict | None) -> BBox | None:
        if not data:
            return None
        return cls(float(data["x0"]), float(data["y0"]), float(data["x1"]), float(data["y1"]))


@dataclass
class Block:
    id: str
    kind: str
    text: str
    page_start: int
    page_end: int
    stream_index: int
    speaker: str | None = None
    level: int | None = None
    bbox: BBox | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Block:
        return cls(
            id=data["id"],
            kind=data["kind"],
            text=data.get("text", ""),
            page_start=int(data["page_start"]),
            page_end=int(data["page_end"]),
            stream_index=int(data.get("stream_index", 0)),
            speaker=data.get("speaker"),
            level=data.get("level"),
            bbox=BBox.from_dict(data.get("bbox")),
        )


@dataclass
class LinkedElement:
    """A footnote, figure, formula, or citation anchored in the paragraph stream."""

    id: str
    kind: str
    anchor_block_id: str
    anchor_offset: int
    stream_index: int
    page: int
    text: str
    marker: str | None = None
    image_path: str | None = None
    bbox: BBox | None = None
    dpi: int | None = None
    caption: str | None = None
    latex: str | None = None
    display: bool = False
    citation_kind: str | None = None
    figure_source: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> LinkedElement:
        return cls(
            id=data["id"],
            kind=data["kind"],
            anchor_block_id=data.get("anchor_block_id", ""),
            anchor_offset=int(data.get("anchor_offset", 0)),
            stream_index=int(data.get("stream_index", 0)),
            page=int(data.get("page", 0)),
            text=data.get("text", ""),
            marker=data.get("marker"),
            image_path=data.get("image_path"),
            bbox=BBox.from_dict(data.get("bbox")),
            dpi=data.get("dpi"),
            caption=data.get("caption"),
            latex=data.get("latex"),
            display=bool(data.get("display", False)),
            citation_kind=data.get("citation_kind"),
            figure_source=data.get("figure_source"),
        )


@dataclass
class Chapter:
    id: str
    index: int
    title: str
    kind: str
    page_start: int
    page_end: int
    detection: str
    blocks: list[Block] = field(default_factory=list)
    elements: list[LinkedElement] = field(default_factory=list)
    scanned_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> Chapter:
        return cls(
            id=data["id"],
            index=int(data["index"]),
            title=data.get("title", ""),
            kind=data.get("kind", "prose"),
            page_start=int(data["page_start"]),
            page_end=int(data["page_end"]),
            detection=data.get("detection", "heuristic"),
            blocks=[Block.from_dict(item) for item in data.get("blocks", [])],
            elements=[LinkedElement.from_dict(item) for item in data.get("elements", [])],
            scanned_pages=[int(n) for n in data.get("scanned_pages", [])],
            warnings=list(data.get("warnings", [])),
        )


@dataclass
class Document:
    source_pdf: str
    title: str | None
    chapters: list[Chapter]
    outline_used: bool
    detection: str
    warnings: list[str] = field(default_factory=list)
    skipped_pages: list[dict] = field(default_factory=list)


@dataclass
class AssociatedRef:
    id: str
    kind: str
    anchor: str
    stream_index: int
    text: str
    estimated_spoken_words: int
    speak: bool
    placement: str
    image_path: str | None = None
    page: int | None = None
    caption: str | None = None
    latex: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> AssociatedRef:
        return cls(
            id=data["id"],
            kind=data["kind"],
            anchor=data.get("anchor", ""),
            stream_index=int(data.get("stream_index", 0)),
            text=data.get("text", ""),
            estimated_spoken_words=int(data.get("estimated_spoken_words", 0)),
            speak=bool(data.get("speak", True)),
            placement=data.get("placement", "inline"),
            image_path=data.get("image_path"),
            page=data.get("page"),
            caption=data.get("caption"),
            latex=data.get("latex"),
        )


@dataclass
class Chunk:
    id: str
    chapter_id: str
    chapter_index: int
    chapter_title: str
    order: int
    kind: str
    text: str
    speakers: list[str]
    source_pages: list[int]
    estimated_duration_sec: float
    word_count: int
    associated: list[AssociatedRef] = field(default_factory=list)
    block_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    segments: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(
            id=data["id"],
            chapter_id=data.get("chapter_id", ""),
            chapter_index=int(data.get("chapter_index", 0)),
            chapter_title=data.get("chapter_title", ""),
            order=int(data.get("order", 0)),
            kind=data.get("kind", "prose"),
            text=data.get("text", ""),
            speakers=list(data.get("speakers", [])),
            source_pages=[int(n) for n in data.get("source_pages", [])],
            estimated_duration_sec=float(data.get("estimated_duration_sec", 0)),
            word_count=int(data.get("word_count", 0)),
            associated=[AssociatedRef.from_dict(item) for item in data.get("associated", [])],
            block_ids=list(data.get("block_ids", [])),
            warnings=list(data.get("warnings", [])),
            segments=[dict(item) for item in data.get("segments", [])],
        )


@dataclass
class InsertedDescription:
    element_id: str
    kind: str
    anchor: str
    spoken_text: str
    decision: str
    reason: str = ""


@dataclass
class PreparedTurn:
    text: str
    style: str
    speaker: str
    rationale: str
    block_ids: list[str] = field(default_factory=list)
    inserted_descriptions: list[InsertedDescription] = field(default_factory=list)


@dataclass
class PrepOutput:
    prompt_version: str
    chunk_id: str
    turns: list[PreparedTurn]


@dataclass
class TtsRequest:
    """One TTS HTTP call. `turns` never carries rationale or descriptions."""

    api: str
    model: str
    audio_mime: str
    sample_rate: int
    speakers: list[dict]
    turns: list[dict]
    payload: dict
