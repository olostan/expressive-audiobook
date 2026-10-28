"""Command line for the staged audiobook pipeline."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import gemini_api_key, load_style_profile
from .pipeline import (
    resolve_config,
    run_assemble,
    run_chunk,
    run_extract,
    run_prep,
    run_review,
    run_tts_plan,
)
from .tts import GeminiTtsClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audiobook",
        description="Turn a PDF you own into a multi-chapter expressive audiobook.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", help="Path to a TOML config file")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="Stage 1: PDF to per-chapter JSON")
    _add_chapter(extract)
    extract.add_argument("--pdf", required=True, help="Path to the book PDF")

    chunk = sub.add_parser("chunk", help="Stage 2: chapter JSON to TTS chunk JSONL")
    _add_chapter(chunk)

    prep = sub.add_parser("prep", help="Stage 3: write multimodal text-prep requests")
    _add_chapter(prep)
    _add_style(prep)

    tts = sub.add_parser("tts", help="Stage 4: plan TTS requests from prep outputs")
    _add_chapter(tts)
    _add_style(tts)

    assemble = sub.add_parser("assemble", help="Stage 5: concatenate existing audio clips")
    _add_chapter(assemble)

    run = sub.add_parser("run", help="Run extract and chunk, then write prep requests")
    _add_chapter(run)
    run.add_argument("--pdf", required=True, help="Path to the book PDF")
    _add_style(run)

    review = sub.add_parser("review", help="Write a review report for one chapter")
    review.add_argument("--chapter", type=int, required=True)
    review.add_argument("--format", choices=("md", "html"), default="md")

    voices = sub.add_parser("voices", help="Voice catalog")
    voices_sub = voices.add_subparsers(dest="voices_command", required=True)
    voices_list = voices_sub.add_parser("list", help="GET /v1beta/voices")
    voices_list.add_argument("--search", default=None)
    voices_list.add_argument("--language", default=None)
    voices_list.add_argument("--page-size", type=int, default=50)

    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    config = resolve_config(args.config)
    if args.command == "extract":
        paths = run_extract(args.pdf, config, chapter=args.chapter)
        _print_paths("extracted", paths)
        return 0
    if args.command == "chunk":
        paths = run_chunk(config, chapter=args.chapter)
        _print_paths("chunked", paths)
        return 0
    if args.command == "prep":
        profile = _profile(args)
        paths = run_prep(config, profile, chapter=args.chapter)
        _print_paths("wrote prep requests (model call not run)", paths)
        return 0
    if args.command == "tts":
        profile = _profile(args)
        paths = run_tts_plan(config, profile, chapter=args.chapter)
        if not paths:
            print("No prep outputs to plan. Text-prep results belong in artifacts/prep/outputs/.")
            print("TTS synthesis is not implemented.")
            return 0
        _print_paths("planned TTS requests (synthesis not run)", paths)
        return 0
    if args.command == "assemble":
        paths = run_assemble(config, chapter=args.chapter)
        if not paths:
            print("No audio clips to assemble. Put WAV or PCM files in artifacts/audio/clips/<chapter>/.")
            return 0
        _print_paths("assembled", paths)
        return 0
    if args.command == "run":
        extracted = run_extract(args.pdf, config, chapter=args.chapter)
        chunked = run_chunk(config, chapter=args.chapter)
        prepared = run_prep(config, _profile(args), chapter=args.chapter)
        _print_paths("extracted", extracted)
        _print_paths("chunked", chunked)
        _print_paths("wrote prep requests (model call not run)", prepared)
        print("TTS synthesis is not implemented. Assemble reads clips after synthesis exists.")
        return 0
    if args.command == "review":
        dest = run_review(config, chapter=args.chapter, fmt=args.format)
        print(dest)
        return 0
    if args.command == "voices":
        client = GeminiTtsClient(gemini_api_key(), config)
        voices = client.list_voices(
            search=args.search,
            language_code=args.language,
            page_size=args.page_size,
        )
        if not voices:
            print("No voices returned.")
            return 0
        for voice in voices:
            name = voice.get("display_name") or voice.get("name") or ""
            identifier = voice.get("id") or voice.get("name") or ""
            print(f"{identifier}\t{name}")
        return 0
    print(f"Unknown command {args.command}", file=sys.stderr)
    return 2


def _add_chapter(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chapter", type=int, default=None, help="1-based chapter index")


def _add_style(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--style", default=None, help="Free-text style direction appended to the profile")
    parser.add_argument("--style-profile", default=None, help="YAML or TOML style profile")


def _profile(args: argparse.Namespace):
    profile = load_style_profile(args.style_profile)
    return profile.with_style_text(args.style)


def _print_paths(label: str, paths: list) -> None:
    print(f"{label}: {len(paths)}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    raise SystemExit(main())
