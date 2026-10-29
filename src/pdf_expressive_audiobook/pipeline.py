"""Stage runners. Each one reads the previous stage's files and writes its own."""

from __future__ import annotations

import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .audio import PcmClip, assemble_clips, load_pcm, write_wav
from .chunking import chunk_chapter, load_chapter_jsonl, write_chapter_jsonl
from .config import AppConfig, StyleProfile, load_config
from .ingest import extract_pdf, load_chapter, load_manifest
from .models import Chunk, to_jsonable
from .project import atomic_write, fingerprint
from .prep import (
    build_prep_request,
    prepare_chunk_output,
    schema_errors,
    validate_prep_output,
)
from .review import render_review_html, render_review_markdown, write_review
from .tts import estimate_tts_cost, plan_tts_requests


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
            tail_ms=config.assemble.chapter_tail_ms,
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


def select_chunks(
    chunks: list[Chunk],
    *,
    start: int = 0,
    seconds: float | None = None,
    max_chunks: int | None = None,
) -> list[Chunk]:
    """Take consecutive chunks, optionally until a duration or a count is reached."""
    ordered = sorted(chunks, key=lambda chunk: chunk.order)
    if start < 0 or (ordered and start >= len(ordered)) or (not ordered and start > 0):
        raise SystemExit(
            f"sample-start {start} is outside this chapter "
            f"({len(ordered)} chunk{'s' if len(ordered) != 1 else ''}, indexes start at 0)"
        )
    chosen = ordered[start:]
    if seconds is not None:
        if seconds <= 0:
            raise SystemExit("--sample-seconds must be greater than 0")
        picked: list[Chunk] = []
        total = 0.0
        for chunk in chosen:
            picked.append(chunk)
            total += chunk.estimated_duration_sec
            if total >= seconds:
                break
        chosen = picked
    if max_chunks is not None:
        if max_chunks < 1:
            raise SystemExit("--max-chunks must be at least 1")
        chosen = chosen[:max_chunks]
    return chosen


def format_estimate(chunks: list[Chunk], config: AppConfig) -> str:
    seconds = sum(chunk.estimated_duration_sec for chunk in chunks)
    tokens = int(round(sum(chunk.word_count for chunk in chunks) * config.cost.tokens_per_word))
    cost = estimate_tts_cost(
        audio_seconds=seconds,
        tts_input_tokens=tokens,
        tts_output_price_per_million=config.cost.tts_output_price_per_million,
        tts_input_price_per_million=config.cost.tts_input_price_per_million,
        audio_tokens_per_second=config.cost.audio_tokens_per_second,
    )
    return (
        f"chunks: {len(chunks)}\n"
        f"audio_minutes: {seconds / 60.0:.2f}\n"
        f"estimated_cost_usd: {cost['total_cost']:.4f}\n"
    )


def book_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._").lower()
    return slug or "book"


def utc_stamp(moment: datetime | None = None) -> str:
    current = moment or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def config_with_paths(config: AppConfig, **paths: str) -> AppConfig:
    return replace(config, paths=replace(config.paths, **paths))


def run_prep_model(
    config: AppConfig,
    profile: StyleProfile,
    client,
    *,
    chapter: int | None = None,
    only_ids: set[str] | None = None,
    chapters: list[int] | None = None,
    force: bool = False,
    progress=None,
) -> list[Path]:
    """Call the text model for each chunk. Chunks whose inputs are unchanged are skipped.

    ``progress(status, chunk_id)`` is called with "cached", "done", or "failed".
    """
    paths = _chunk_paths(config, chapter)
    if chapters is not None:
        wanted = {f"{index:03d}" for index in chapters}
        paths = [path for path in paths if path.stem in wanted]
    jobs: list[tuple[Path, list[Chunk]]] = []
    for path in paths:
        chunks = load_chapter_jsonl(path)
        if only_ids is not None:
            chunks = [chunk for chunk in chunks if chunk.id in only_ids]
        if chunks:
            jobs.append((path, chunks))
    written: list[Path] = []
    workers = max(1, int(config.prep.concurrency))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for path, chunks in jobs:
            chapter_chunks = load_chapter_jsonl(path)
            positions = {chunk.id: index for index, chunk in enumerate(chapter_chunks)}
            for chunk in chunks:
                futures.append(
                    pool.submit(
                        _prep_one,
                        config,
                        profile,
                        client,
                        chapter_chunks,
                        positions[chunk.id],
                        chunk,
                        force,
                        progress,
                    )
                )
        results, failures = _gather(pool, futures)
        written.extend(results)
        if failures:
            for index in sorted({chunk.chapter_index for _, chunks in jobs for chunk in chunks}):
                _rewrite_prep_jsonl(config, index)
            raise failures[0]
    chapters = {chunk.chapter_index for _, chunks in jobs for chunk in chunks}
    for index in sorted(chapters):
        _rewrite_prep_jsonl(config, index)
    return written


def run_tts_model(
    config: AppConfig,
    profile: StyleProfile,
    client,
    *,
    chapter: int | None = None,
    only_ids: set[str] | None = None,
    chapters: list[int] | None = None,
    force: bool = False,
    progress=None,
) -> list[Path]:
    """Synthesize planned requests. Clips whose request is unchanged are skipped.

    ``progress(status, name)`` is called with "cached", "done", or "failed".
    """
    outputs = _load_prep_outputs(config, chapter)
    if chapters is not None:
        wanted_prefix = tuple(f"c{index:03d}-" for index in chapters)
        outputs = {key: value for key, value in outputs.items() if key.startswith(wanted_prefix)}
    if only_ids is not None:
        outputs = {key: value for key, value in outputs.items() if key in only_ids}
    if not outputs:
        return []
    chunks_by_id = {
        chunk.id: chunk
        for path in _chunk_paths(config, chapter)
        for chunk in load_chapter_jsonl(path)
    }
    written: list[Path] = []
    jobs = []
    for chunk_id, output in outputs.items():
        chunk = chunks_by_id.get(chunk_id)
        order = chunk.order if chunk else 0
        chapter_index = chunk.chapter_index if chunk else _chapter_index_from_chunk_id(chunk_id)
        requests = plan_tts_requests(output.get("turns") or [], profile, config)
        plan_dir = Path(config.paths.tts) / "plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = plan_dir / f"{chunk_id}.json"
        atomic_write(
            plan_path,
            json.dumps(
                {
                    "chunk_id": chunk_id,
                    "prompt_version": output.get("prompt_version"),
                    "requests": [to_jsonable(request) for request in requests],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
        expected = set()
        for request_index, request in enumerate(requests):
            dest = _clip_path(config, chapter_index, order, chunk_id, request_index)
            expected.add(dest.name)
            jobs.append((request, dest))
        _remove_stale_clips(config, chapter_index, chunk_id, expected)
    workers = max(1, int(config.tts.concurrency))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_synthesize_one, client, config, request, dest, force, progress)
            for request, dest in jobs
        ]
        results, failures = _gather(pool, futures)
        written.extend(results)
        if failures:
            raise failures[0]
    return written


def run_sample(
    config: AppConfig,
    profile: StyleProfile,
    *,
    pdf: Path | None,
    chapter: int,
    prep_client,
    tts_client,
    seconds: float = 60,
    start: int = 0,
    max_chunks: int | None = None,
    dry_run: bool = False,
    stamp: str | None = None,
) -> dict:
    """Run prep, speech, and assembly for a short span, outside the chapter cache."""
    if pdf is not None:
        run_extract(pdf, config, chapter=chapter)
    run_chunk(config, chapter=chapter)
    chunks = select_chunks(
        _chunks_for_chapter(config, chapter),
        start=start,
        seconds=seconds,
        max_chunks=max_chunks,
    )
    if dry_run:
        return {"dry_run": True, "chunks": chunks, "estimate": format_estimate(chunks, config)}
    if prep_client is None or tts_client is None:
        raise RuntimeError("GEMINI_API_KEY is not set")
    slug = book_slug(Path(pdf).stem if pdf else _manifest_title(config))
    token = stamp or utc_stamp()
    public_dir = Path(config.paths.artifacts) / slug / "samples"
    public_dir.mkdir(parents=True, exist_ok=True)
    work = public_dir / "work" / f"{chapter}-{token}"
    if work.exists():
        shutil.rmtree(work)
    isolated = config_with_paths(
        config,
        prep=str(work / "prep"),
        tts=str(work / "tts"),
        audio=str(work / "audio"),
    )
    only_ids = {chunk.id for chunk in chunks}
    run_prep_model(isolated, profile, prep_client, chapter=chapter, only_ids=only_ids)
    run_tts_model(isolated, profile, tts_client, chapter=chapter, only_ids=only_ids)
    clip_root = Path(isolated.paths.audio) / "clips" / f"{chapter:03d}"
    if not clip_root.exists():
        raise RuntimeError(f"Sample for chapter {chapter} produced no audio clips")
    clips = sorted(path for path in clip_root.iterdir() if path.suffix.lower() in {".wav", ".pcm"})
    if not clips:
        raise RuntimeError(f"Sample for chapter {chapter} produced no audio clips")
    wav_path = public_dir / f"{chapter}-{token}.wav"
    assemble_clips(
        clips,
        wav_path,
        sample_rate=config.tts.sample_rate,
        gap_ms=config.assemble.gap_ms,
        crossfade_ms=config.assemble.crossfade_ms,
    )
    outputs = _load_prep_outputs(isolated, chapter)
    review_text = render_review_markdown(chunks, outputs)
    review_path = public_dir / f"{chapter}-{token}.md"
    write_review(review_path, review_text)
    return {
        "dry_run": False,
        "chunks": chunks,
        "wav": wav_path,
        "review": review_path,
        "estimate": format_estimate(chunks, config),
    }


def run_chapter_pipeline(
    config: AppConfig,
    profile: StyleProfile,
    *,
    pdf: Path | None,
    chapter: int | None,
    prep_client,
    tts_client,
    max_chunks: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Extract, chunk, prep, synthesize, and assemble one chapter or the whole book."""
    if pdf is not None:
        run_extract(pdf, config, chapter=chapter)
    run_chunk(config, chapter=chapter)
    selected: list[Chunk] = []
    indexes = _chapter_indexes(config, chapter)
    for index in indexes:
        chapter_chunks = _chunks_for_chapter(config, index)
        if max_chunks is not None:
            chapter_chunks = select_chunks(chapter_chunks, max_chunks=max_chunks)
        selected.extend(chapter_chunks)
    if dry_run:
        return {"dry_run": True, "chunks": selected, "estimate": format_estimate(selected, config)}
    if prep_client is None or tts_client is None:
        raise RuntimeError("GEMINI_API_KEY is not set")
    only_ids = {chunk.id for chunk in selected} if max_chunks is not None else None
    run_prep_model(config, profile, prep_client, chapter=chapter, only_ids=only_ids)
    run_tts_model(config, profile, tts_client, chapter=chapter, only_ids=only_ids)
    audio = run_assemble(config, chapter=chapter)
    return {
        "dry_run": False,
        "chunks": selected,
        "audio": audio,
        "estimate": format_estimate(selected, config),
    }


def chapter_rows(config: AppConfig) -> list[dict]:
    manifest = load_manifest(config.paths.ingest)
    return list(manifest.get("chapters") or [])


def _prep_one(
    config,
    profile,
    client,
    chunks: list[Chunk],
    index: int,
    chunk: Chunk,
    force: bool = False,
    progress=None,
) -> Path:
    dest = _prep_chunk_path(config, chunk.id)
    neighbor_count = max(0, int(config.prep.neighbor_chunks))
    previous_text = _neighbor_text(chunks, max(0, index - neighbor_count), index)
    following_text = _neighbor_text(chunks, index + 1, index + 1 + neighbor_count)
    image_root = _image_root(config)
    request = build_prep_request(
        chunk,
        profile,
        config,
        previous_text=previous_text,
        following_text=following_text,
        image_root=image_root,
    )
    stamp = fingerprint(request)
    if not force and dest.exists() and _fingerprint_matches(dest, stamp):
        try:
            existing = json.loads(dest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        if isinstance(existing, dict) and not schema_errors(existing) and not validate_prep_output(existing, chunk, config):
            if progress:
                progress("cached", chunk.id)
            return dest
    request_path = Path(config.paths.prep) / "requests" / f"{chunk.id}.json"
    atomic_write(request_path, json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    try:
        output = prepare_chunk_output(
            chunk,
            profile,
            config,
            client,
            previous_text=previous_text,
            following_text=following_text,
            image_root=image_root,
            request=request,
        )
    except Exception as exc:
        if progress:
            progress("failed", chunk.id)
        _record_failure(config, chunk.id, exc)
        raise
    atomic_write(dest, json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    _write_fingerprint(dest, stamp)
    if progress:
        progress("done", chunk.id)
    return dest


def _gather(pool, futures) -> tuple[list, list]:
    """Wait for every job. Ctrl-C cancels the queue instead of finishing it."""
    results, failures = [], []
    try:
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:  # finish the rest so one bad chunk does not waste the others
                failures.append(exc)
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    return results, failures


def _record_failure(config: AppConfig, chunk_id: str, exc: Exception) -> None:
    """Keep the reason for a failed chunk beside its request so it can be inspected later."""
    import traceback

    try:
        text = "".join(traceback.format_exception(exc))
        atomic_write(Path(config.paths.prep) / "errors" / f"{chunk_id}.txt", text)
    except OSError:
        pass


def _fingerprint_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".fp")


def _fingerprint_matches(dest: Path, stamp: str) -> bool:
    """A file with no recorded fingerprint predates tracking and is kept."""
    marker = _fingerprint_path(dest)
    if not marker.exists():
        return True
    return marker.read_text(encoding="utf-8").strip() == stamp


def _write_fingerprint(dest: Path, stamp: str) -> None:
    atomic_write(_fingerprint_path(dest), stamp + "\n")


def _synthesize_one(client, config: AppConfig, request, dest: Path, force: bool = False, progress=None) -> Path:
    stamp = fingerprint(request.payload)
    if not force and dest.exists() and _fingerprint_matches(dest, stamp):
        try:
            load_pcm(dest.read_bytes(), expected_rate=config.tts.sample_rate)
            if progress:
                progress("cached", dest.name)
            return dest
        except Exception:
            pass
    try:
        pcm = client.synthesize(request)
        clip = load_pcm(pcm, expected_rate=config.tts.sample_rate)
        if not isinstance(clip, PcmClip):
            raise RuntimeError(f"Speech response for {dest.name} was not PCM")
        write_wav(dest, clip)
    except Exception:
        if progress:
            progress("failed", dest.name)
        raise
    _write_fingerprint(dest, stamp)
    if progress:
        progress("done", dest.name)
    return dest


def _remove_stale_clips(config: AppConfig, chapter_index: int, chunk_id: str, expected: set[str]) -> None:
    folder = Path(config.paths.audio) / "clips" / f"{chapter_index:03d}"
    if not folder.exists():
        return
    for path in folder.glob(f"*-{chunk_id}-r*.wav"):
        if path.name not in expected:
            path.unlink(missing_ok=True)
            _fingerprint_path(path).unlink(missing_ok=True)


def _prep_chunk_path(config: AppConfig, chunk_id: str) -> Path:
    return Path(config.paths.prep) / "outputs" / "chunks" / f"{chunk_id}.json"


def _rewrite_prep_jsonl(config: AppConfig, chapter: int) -> Path:
    root = Path(config.paths.prep) / "outputs" / "chunks"
    dest = Path(config.paths.prep) / "outputs" / f"{chapter:03d}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if root.exists():
        prefix = f"c{chapter:03d}-"
        for path in sorted(root.glob("*.json")):
            if not path.stem.startswith(prefix):
                continue
            lines.append(json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False))
    dest.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")
    return dest


def _clip_path(config: AppConfig, chapter_index: int, order: int, chunk_id: str, request_index: int) -> Path:
    return (
        Path(config.paths.audio)
        / "clips"
        / f"{chapter_index:03d}"
        / f"{order:04d}-{chunk_id}-r{request_index:02d}.wav"
    )


def _chapter_index_from_chunk_id(chunk_id: str) -> int:
    match = re.match(r"c(\d+)-", chunk_id)
    if not match:
        raise SystemExit(f"Chunk id {chunk_id} does not encode a chapter index")
    return int(match.group(1))


def _chapter_indexes(config: AppConfig, chapter: int | None) -> list[int]:
    if chapter is not None:
        return [chapter]
    return [int(item["index"]) for item in chapter_rows(config)]


def _manifest_title(config: AppConfig) -> str:
    try:
        manifest = load_manifest(config.paths.ingest)
    except FileNotFoundError:
        return "book"
    return str(manifest.get("title") or "book")
