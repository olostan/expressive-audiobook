"""Where each spoken clip lands in a chapter, in seconds.

Sound effects and ducking both need this. Turns are numbered 1.. through a
chapter's prepared chunks in reading order; a clip may carry several turns, in
which case a turn's position inside it is estimated from its share of the text.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path

from .audio import clip_offsets
from .config import AppConfig, StyleProfile
from .tts import plan_tts_requests


@dataclass
class ClipSpan:
    path: Path
    start: float
    end: float
    first_turn: int
    turn_chars: list[int] = field(default_factory=list)


@dataclass
class Timeline:
    clips: list[ClipSpan]
    turns: int  # numbered across every chunk given, spoken or not

    @property
    def spans(self) -> list[tuple[float, float]]:
        return [(clip.start, clip.end) for clip in self.clips]

    def turn_span(self, number: int) -> tuple[float, float] | None:
        for clip in self.clips:
            if clip.first_turn <= number < clip.first_turn + len(clip.turn_chars):
                total = max(1, sum(clip.turn_chars))
                before = sum(clip.turn_chars[: number - clip.first_turn])
                length = clip.end - clip.start
                start = clip.start + length * before / total
                end = clip.start + length * (before + clip.turn_chars[number - clip.first_turn]) / total
                return start, end
        return None


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def chapter_timeline(
    config: AppConfig,
    profile: StyleProfile,
    chunks: list,
    outputs: dict,
    clip_path,
    spoken_ids: set[str] | None = None,
) -> Timeline:
    """``chunks`` is the whole chapter in order. ``spoken_ids`` limits which are in the audio (samples)."""
    entries: list[tuple[Path, int, list[int]]] = []  # path, first turn, chars per turn
    number = 1
    for chunk in sorted(chunks, key=lambda c: c.order):
        output = outputs.get(chunk.id)
        if output is None:
            continue
        turns = output.get("turns") or []
        requests = plan_tts_requests(turns, profile, config)
        cursor = 0
        for request_index, request in enumerate(requests):
            count = len(request.turns)
            if spoken_ids is None or chunk.id in spoken_ids:
                chars = [max(1, len(str(t.get("text") or ""))) for t in turns[cursor : cursor + count]]
                entries.append((clip_path(chunk, request_index), number + cursor, chars))
            cursor += count
        number += len(turns)
    rate = config.tts.sample_rate
    lengths = [int(round(wav_seconds(path) * rate)) for path, _, _ in entries]
    starts = clip_offsets(lengths, gap_ms=config.assemble.gap_ms, crossfade_ms=config.assemble.crossfade_ms, rate=rate)
    clips = [
        ClipSpan(path, start / rate, (start + length) / rate, first, chars)
        for (path, first, chars), start, length in zip(entries, starts, lengths)
    ]
    return Timeline(clips, number - 1)
