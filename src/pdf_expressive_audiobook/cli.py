"""Command line for the staged audiobook pipeline."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from . import __version__
from .audio import AudioFormatError
from .config import StyleProfile, VoiceMapping, gemini_api_key, load_style_profile
from .pipeline import (
    chapter_rows,
    format_estimate,
    resolve_config,
    run_assemble,
    run_chapter_pipeline,
    run_chunk,
    run_extract,
    run_prep_model,
    run_review,
    run_sample,
    run_tts_model,
    select_chunks,
)
from .prep import ChunkPrepFailed, GeminiTextPrepClient
from .tts import GeminiTtsClient, classify_voice_name


def main(argv: list[str] | None = None, *, prep_client=None, tts_client=None) -> int:
    parser = argparse.ArgumentParser(
        prog="audiobook",
        description="Turn a PDF you own into a multi-chapter expressive audiobook.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", help="Path to a TOML config file")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="Stage 1: PDF to per-chapter JSON")
    _add_chapter(extract, required=False)
    extract.add_argument("--pdf", required=True, help="Path to the book PDF")

    chunk = sub.add_parser("chunk", help="Stage 2: chapter JSON to TTS chunk JSONL")
    _add_chapter(chunk, required=False)

    chapters = sub.add_parser("chapters", help="List detected chapter index, title, and pages")
    chapters.add_argument("--pdf", help="PDF to extract before listing")

    chunks = sub.add_parser("chunks", help="List chunk index, duration, speakers, and opening words")
    _add_chapter(chunks, required=True)
    chunks.add_argument("--pdf", help="PDF to extract and chunk before listing")

    prep = sub.add_parser("prep", help="Stage 3: prepare chunk text with the Gemini text model")
    _add_chapter(prep, required=False)
    _add_style(prep)
    _add_voice(prep)

    tts = sub.add_parser("tts", help="Stage 4: synthesize speech from prep outputs")
    _add_chapter(tts, required=False)
    _add_style(tts)
    _add_voice(tts)

    assemble = sub.add_parser("assemble", help="Stage 5: concatenate audio clips into a chapter WAV")
    _add_chapter(assemble, required=False)

    estimate = sub.add_parser("estimate", help="Print chunk count, audio minutes, and estimated cost")
    estimate.add_argument("--pdf", help="PDF to extract and chunk before estimating")
    _add_chapter(estimate, required=False)
    _add_sample_flags(estimate, seconds_default=None)

    run = sub.add_parser("run", help="Run the pipeline through a chapter WAV")
    run.add_argument("--pdf", required=True, help="Path to the book PDF")
    _add_chapter(run, required=False)
    _add_style(run)
    _add_voice(run)
    _add_sample_flags(run, seconds_default=None)
    run.add_argument("--dry-run", action="store_true", help="Estimate cost and do not call Gemini")

    sample = sub.add_parser("sample", help="Prepare, speak, and assemble about a minute of one chapter")
    sample.add_argument("--pdf", required=True, help="Path to the book PDF")
    _add_chapter(sample, required=True)
    _add_style(sample)
    _add_voice(sample)
    _add_sample_flags(sample, seconds_default=60)
    sample.add_argument("--dry-run", action="store_true", help="Estimate the sample and do not call Gemini")

    review = sub.add_parser("review", help="Write a review report for one chapter")
    review.add_argument("--chapter", type=int, required=True)
    review.add_argument("--format", choices=("md", "html"), default="md")

    voices = sub.add_parser("voices", help="List catalog voices or design one")
    voices_sub = voices.add_subparsers(dest="voices_command", required=True)
    voices_list = voices_sub.add_parser("list", help="List prebuilt and library voices")
    voices_list.add_argument("--search", default=None)
    voices_list.add_argument("--language", default=None)
    voices_list.add_argument("--gender", default=None, help="female, male, or neutral")
    voices_list.add_argument("--type", dest="voice_type", default=None, help="prebuilt, library, or prompted")
    voices_list.add_argument("--page-size", type=int, default=50)
    voices_design = voices_sub.add_parser("design", help="Create a designed voice and print its id")
    voices_design.add_argument("--description", required=True, help="How the voice should sound")
    voices_design.add_argument("--display-name", default="Designed narrator")
    voices_design.add_argument("--language", default=None, help="BCP-47 tag such as en-US")
    voices_design.add_argument("--gender", default=None)

    args = parser.parse_args(argv)
    try:
        return _dispatch(args, prep_client=prep_client, tts_client=tts_client)
    except (FileNotFoundError, RuntimeError, ValueError, AudioFormatError, ChunkPrepFailed) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except SystemExit as exc:
        if exc.code in (None, 0):
            raise
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            return 1
        raise


def _dispatch(args: argparse.Namespace, *, prep_client, tts_client) -> int:
    config = resolve_config(args.config)
    if args.command == "extract":
        paths = run_extract(args.pdf, config, chapter=args.chapter)
        _print_paths("extracted", paths)
        return 0
    if args.command == "chunk":
        paths = run_chunk(config, chapter=args.chapter)
        _print_paths("chunked", paths)
        return 0
    if args.command == "chapters":
        if args.pdf:
            run_extract(args.pdf, config)
        rows = chapter_rows(config)
        if not rows:
            print("No chapters found. Run audiobook extract --pdf <file> first.")
            return 1
        for row in rows:
            print(f"{row['index']}\t{row.get('title') or '(untitled)'}\t{row['page_start']}-{row['page_end']}")
        return 0
    if args.command == "chunks":
        if args.pdf:
            run_extract(args.pdf, config, chapter=args.chapter)
            run_chunk(config, chapter=args.chapter)
        from .pipeline import _chunks_for_chapter

        for chunk in _chunks_for_chapter(config, args.chapter):
            speakers = ",".join(chunk.speakers) or "narrator"
            opening = " ".join((chunk.text or "").split())[:80]
            print(f"{chunk.order}\t{chunk.estimated_duration_sec:.1f}s\t{speakers}\t{opening}")
        return 0
    if args.command == "prep":
        config, profile = _runtime(config, args)
        client = _prep_client(config, prep_client)
        paths = run_prep_model(config, profile, client, chapter=args.chapter)
        _print_paths("prepared", paths)
        return 0
    if args.command == "tts":
        config, profile = _runtime(config, args)
        client = _tts_client(config, tts_client)
        paths = run_tts_model(config, profile, client, chapter=args.chapter)
        if not paths:
            print("No prep outputs to synthesize. Run audiobook prep first.")
            return 1
        _print_paths("synthesized", paths)
        return 0
    if args.command == "assemble":
        paths = run_assemble(config, chapter=args.chapter)
        if not paths:
            print("No audio clips to assemble. Run audiobook tts first.")
            return 1
        _print_paths("assembled", paths)
        return 0
    if args.command == "estimate":
        return _estimate(config, args)
    if args.command == "review":
        dest = run_review(config, chapter=args.chapter, fmt=args.format)
        print(dest)
        return 0
    if args.command == "voices":
        client = _tts_client(config, tts_client)
        if args.voices_command == "list":
            voices = client.list_voices(
                search=args.search,
                language_code=args.language,
                page_size=args.page_size,
                gender=args.gender,
                voice_type=args.voice_type,
            )
            if not voices:
                print("No voices returned.")
                return 0
            for voice in voices:
                print(_format_voice(voice))
            return 0
        voice_id = client.design_voice(
            args.description,
            display_name=args.display_name,
            language_code=args.language,
            gender=args.gender,
        )
        print(voice_id)
        return 0
    if args.command in {"run", "sample"}:
        return _run(config, args, prep_client=prep_client, tts_client=tts_client)
    print(f"Unknown command {args.command}", file=sys.stderr)
    return 2


def _run(config, args, *, prep_client, tts_client) -> int:
    config, profile = _runtime(config, args)
    sample_seconds = args.sample_seconds
    if args.command == "sample" or sample_seconds is not None:
        if args.chapter is None:
            raise SystemExit("A sample needs --chapter N")
        result = run_sample(
            config,
            profile,
            pdf=Path(args.pdf),
            chapter=args.chapter,
            prep_client=None if args.dry_run else _prep_client(config, prep_client),
            tts_client=None if args.dry_run else _tts_client(config, tts_client),
            seconds=60 if sample_seconds is None else sample_seconds,
            start=args.sample_start,
            max_chunks=args.max_chunks,
            dry_run=args.dry_run,
        )
        print(result["estimate"], end="")
        if result["dry_run"]:
            print("dry_run: sample")
            return 0
        print(f"sample_wav: {result['wav']}")
        print(f"sample_review: {result['review']}")
        return 0
    result = run_chapter_pipeline(
        config,
        profile,
        pdf=Path(args.pdf),
        chapter=args.chapter,
        prep_client=None if args.dry_run else _prep_client(config, prep_client),
        tts_client=None if args.dry_run else _tts_client(config, tts_client),
        max_chunks=args.max_chunks,
        dry_run=args.dry_run,
    )
    print(result["estimate"], end="")
    if result["dry_run"]:
        print("dry_run: chapter")
        return 0
    _print_paths("audio", result["audio"])
    return 0


def _estimate(config, args) -> int:
    if args.pdf:
        run_extract(args.pdf, config, chapter=args.chapter)
        run_chunk(config, chapter=args.chapter)
    from .pipeline import _chapter_indexes, _chunks_for_chapter

    selected = []
    for index in _chapter_indexes(config, args.chapter):
        chunks = _chunks_for_chapter(config, index)
        if args.sample_seconds is not None or args.max_chunks is not None or args.sample_start:
            chunks = select_chunks(
                chunks,
                start=args.sample_start,
                seconds=args.sample_seconds,
                max_chunks=args.max_chunks,
            )
        selected.extend(chunks)
    print(format_estimate(selected, config), end="")
    return 0


def _runtime(config, args):
    profile = _profile(args)
    if getattr(args, "lite", False):
        config = replace(config, tts=replace(config.tts, model=config.tts.lite_model))
    voice = getattr(args, "voice", None)
    if voice:
        kind = classify_voice_name(voice)
        voices = dict(profile.character_voices)
        voices["narrator"] = VoiceMapping(voice=voice, voice_type=kind)
        profile = replace(profile, character_voices=voices)
        config = replace(
            config,
            tts=replace(config.tts, default_voice=voice, default_voice_type=kind),
        )
    return config, profile


def _prep_client(config, prep_client):
    if prep_client is not None:
        return prep_client
    return GeminiTextPrepClient(gemini_api_key(), config)


def _tts_client(config, tts_client):
    if tts_client is not None:
        return tts_client
    return GeminiTtsClient(gemini_api_key(), config)


def _add_chapter(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument("--chapter", type=int, required=required, help="1-based chapter index")


def _add_style(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--style",
        default=None,
        help="Free-text direction, or a path to a YAML/TOML style profile",
    )
    parser.add_argument("--style-profile", default=None, help="YAML or TOML style profile")


def _add_voice(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--voice", default=None, help="Prebuilt name, library id, or voice_ design id")
    parser.add_argument("--lite", action="store_true", help="Use the lite TTS model from config")


def _add_sample_flags(parser: argparse.ArgumentParser, *, seconds_default: int | None) -> None:
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=seconds_default,
        help="Speak consecutive chunks until this many estimated seconds",
    )
    parser.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="0-based chunk index to start from (see audiobook chunks)",
    )
    parser.add_argument("--max-chunks", type=int, default=None, help="Cap how many chunks are included")


def _profile(args: argparse.Namespace) -> StyleProfile:
    style = getattr(args, "style", None)
    profile_path = getattr(args, "style_profile", None)
    extra = style
    if profile_path is None and style and _is_profile_path(style):
        profile_path = style
        extra = None
    profile = load_style_profile(profile_path)
    return profile.with_style_text(extra)


def _is_profile_path(value: str) -> bool:
    path = Path(value)
    if path.suffix.lower() not in {".yaml", ".yml", ".toml"}:
        return False
    if not path.exists():
        raise SystemExit(f"Style profile not found: {value}")
    return True


def _format_voice(voice: dict) -> str:
    identifier = str(voice.get("id") or voice.get("name") or "")
    display = str(voice.get("display_name") or voice.get("displayName") or "")
    hidden = {
        "id",
        "name",
        "display_name",
        "displayName",
        "sample_audio",
        "sampleAudio",
        "prompted",
    }
    meta = []
    for key in sorted(voice):
        if key in hidden:
            continue
        value = voice[key]
        if value in (None, "") or isinstance(value, (dict, list)):
            continue
        meta.append(f"{key}={value}")
    return "\t".join(part for part in (identifier, display, " ".join(meta)) if part)


def _print_paths(label: str, paths: list) -> None:
    print(f"{label}: {len(paths)}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    raise SystemExit(main())
