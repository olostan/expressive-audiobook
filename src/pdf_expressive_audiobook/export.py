"""Turn assembled WAV files into MP3 or M4A with ffmpeg, when it is installed."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

FORMATS = ("wav", "mp3", "m4a")


class ExportError(RuntimeError):
    pass


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def target_path(wav: Path, fmt: str) -> Path:
    return wav.with_suffix(f".{fmt}")


def convert(
    wav: Path,
    fmt: str,
    *,
    quality: int = 4,
    tags: dict[str, str] | None = None,
    run=subprocess.run,
) -> Path | None:
    """Write ``wav`` as ``fmt`` beside it. Returns None when nothing is needed or possible.

    An up-to-date output (newer than the WAV) is left alone. The file is written under a
    temporary name and renamed, so an interrupted run never leaves a broken MP3.
    """
    if fmt == "wav":
        return None
    if fmt not in FORMATS:
        raise ExportError(f"audio format must be one of {', '.join(FORMATS)} (got {fmt!r})")
    exe = ffmpeg_path()
    if exe is None:
        return None
    dest = target_path(wav, fmt)
    if dest.exists() and dest.stat().st_mtime >= wav.stat().st_mtime:
        return dest
    part = dest.with_name(f".{dest.stem}.part{dest.suffix}")
    codec = (
        ["-c:a", "libmp3lame", "-q:a", str(quality), "-id3v2_version", "3"]
        if fmt == "mp3"
        else ["-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart"]
    )
    meta = [item for key, value in (tags or {}).items() if value for item in ("-metadata", f"{key}={value}")]
    command = [exe, "-y", "-loglevel", "error", "-i", str(wav), "-map_metadata", "-1", *meta, *codec, str(part)]
    result = run(command, capture_output=True, text=True)
    if result.returncode != 0:
        part.unlink(missing_ok=True)
        raise ExportError(f"ffmpeg failed for {wav.name}: {(result.stderr or '').strip()[:300]}")
    part.replace(dest)
    return dest
