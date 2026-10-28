"""Stage runners. Each one reads the previous stage's files and writes its own."""

from __future__ import annotations

import json
from pathlib import Path

from .audio import assemble_clips
from .chunking import chunk_chapter, load_chapter_jsonl, write_chapter_jsonl
from .config import AppConfig, StyleProfile, load_config
from .ingest import extract_pdf, load_chapter, load_manifest
from .models import Chunk, to_jsonable
from .prep import build_prep_request
from .review import render_review_html, render_review_markdown, write_review
from .tts import plan_tts_requests


def run_extract(pdf: Path, config: AppConfig, *, chapter: int | None = None, ocr_fn=None) -> list[Path]:
    document = extract_pdf(pdf, config.paths.ingest, config.ingest, ocr_fn=ocr_fn)
    paths = []
    for item in document.chapters:
        if chapter is not None and item.index != chapter:
            continue
        paths.append(Path(config.paths.ingest) / "chapters" / f"{item.index:03d}.json")
    if chapter is not None and not paths:
        raise SystemExit(f"No chapter with index {chapter}")
    return paths


def run_chunk(config: AppConfig, *, chapter: int | None = None) -> list[Path]:
    manifest = load_manifest(config.paths.ingest)
    written: list[Path] = []
    for item in manifest["chapters"]:
        if chapter is not None and int(item["index"]) != chapter:
            continue
        source = Path(config.paths.ingest) / item["path"]
        model = load_chapter(source)
        chunks = chunk_chapter(model, config.chunking)
        dest = Path(config.paths.chunks) / f"{model.index:03d}.jsonl"
        write_chapter_jsonl(chunks, dest)
        written.append(dest)
    if chapter is not None and not written:
        raise SystemExit(f"No chapter with index {chapter}")
    return written


def run_prep(
    config: AppConfig,
    profile: StyleProfile,
    *,
    chapter: int | None = None,
) -> list[Path]:
    """Write multimodal request JSON. Does not call the text model."""
    paths = _chunk_paths(config, chapter)
    written: list[Path] = []
    request_dir = Path(config.paths.prep) / "requests"
    for path in paths:
        chunks = load_chapter_jsonl(path)
        image_root = _image_root(config)
        neighbor_count = max(0, int(config.prep.neighbor_chunks))
        for index, chunk in enumerate(chunks):
            previous_text = _neighbor_text(chunks, max(0, index - neighbor_count), index)
            following_text = _neighbor_text(chunks, index + 1, index + 1 + neighbor_count)
            request = build_prep_request(
                chunk,
                profile,
                config,
                previous_text=previous_text,
                following_text=following_text,
                image_root=image_root,
            )
            dest = request_dir / f"{chunk.id}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            written.append(dest)
    return written


def run_tts_plan(config: AppConfig, profile: StyleProfile, *, chapter: int | None = None) -> list[Path]:
    """Plan TTS calls from prep outputs. Does not synthesize audio."""
    outputs = _load_prep_outputs(config, chapter)
    if not outputs:
        return []
    written: list[Path] = []
    plan_dir = Path(config.paths.tts) / "plans"
    for chunk_id, output in outputs.items():
        requests = plan_tts_requests(output.get("turns") or [], profile, config)
        dest = plan_dir / f"{chunk_id}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunk_id": chunk_id,
            "prompt_version": output.get("prompt_version"),
            "requests": [to_jsonable(request) for request in requests],
        }
        dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(dest)
    return written


def run_assemble(config: AppConfig, *, chapter: int | None = None) -> list[Path]:
    """Concatenate clip files that already exist. Synthesis is a separate stage."""
    audio_root = Path(config.paths.audio)
    clip_root = audio_root / "clips"
    if not clip_root.exists():
        return []
    written: list[Path] = []
    chapter_dirs = sorted(path for path in clip_root.iterdir() if path.is_dir())
    for folder in chapter_dirs:
        index = _chapter_index_from_name(folder.name)
        if chapter is not None and index != chapter:
            continue
        clips = sorted(path for path in folder.iterdir() if path.suffix.lower() in {".wav", ".pcm"})
        if not clips:
            continue
        dest = audio_root / f"chapter-{index:03d}.wav"
        assemble_clips(
            clips,
            dest,
            sample_rate=config.tts.sample_rate,
            gap_ms=config.assemble.gap_ms,
            crossfade_ms=config.assemble.crossfade_ms,
        )
        written.append(dest)
    if config.assemble.write_full_book and chapter is None and len(written) > 1:
        book = audio_root / "book.wav"
        assemble_clips(
            written,
            book,
            sample_rate=config.tts.sample_rate,
            gap_ms=config.assemble.gap_ms,
            crossfade_ms=config.assemble.crossfade_ms,
        )
        written.append(book)
    return written


def run_review(config: AppConfig, *, chapter: int, fmt: str = "md") -> Path:
    chunks = _chunks_for_chapter(config, chapter)
    outputs = _load_prep_outputs(config, chapter)
    title = chunks[0].chapter_title if chunks else f"Chapter {chapter}"
    if fmt == "html":
        text = render_review_html(chunks, outputs, title)
        dest = Path(config.paths.prep) / "review" / f"chapter-{chapter:03d}.html"
    else:
        text = render_review_markdown(chunks, outputs)
        dest = Path(config.paths.prep) / "review" / f"chapter-{chapter:03d}.md"
    write_review(dest, text)
    return dest


def _chunk_paths(config: AppConfig, chapter: int | None) -> list[Path]:
    root = Path(config.paths.chunks)
    if not root.exists():
        raise SystemExit(f"No chunks at {root}. Run the chunk stage first.")
    paths = sorted(root.glob("*.jsonl"))
    if chapter is None:
        return paths
    matched = [path for path in paths if path.stem == f"{chapter:03d}"]
    if not matched:
        raise SystemExit(f"No chunk file for chapter {chapter}")
    return matched


def _chunks_for_chapter(config: AppConfig, chapter: int) -> list[Chunk]:
    path = Path(config.paths.chunks) / f"{chapter:03d}.jsonl"
    if not path.exists():
        raise SystemExit(f"No chunk file for chapter {chapter}")
    return load_chapter_jsonl(path)


def _load_prep_outputs(config: AppConfig, chapter: int | None) -> dict[str, dict]:
    root = Path(config.paths.prep) / "outputs"
    if not root.exists():
        return {}
    found: dict[str, dict] = {}
    for path in sorted(root.glob("*.jsonl")):
        if chapter is not None and path.stem != f"{chapter:03d}":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            found[str(item["chunk_id"])] = item
    return found


def _image_root(config: AppConfig) -> Path:
    return Path(config.paths.ingest) / "chapters"


def _neighbor_text(chunks: list[Chunk], start: int, end: int) -> str:
    return "\n\n".join(chunk.text for chunk in chunks[start:end] if chunk.text)


def _chapter_index_from_name(name: str) -> int:
    digits = "".join(char for char in name if char.isdigit())
    if not digits:
        raise SystemExit(f"Clip folder {name} does not contain a chapter index")
    return int(digits)


def resolve_config(path: str | None) -> AppConfig:
    return load_config(path)
