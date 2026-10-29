"""PCM and WAV helpers for 24 kHz mono 16-bit audio.

WAV inputs are parsed as RIFF containers. A 44-byte header is the common
case for a minimal PCM file, but fmt and data chunks can sit later when
JUNK, fact, or LIST chunks are present. Callers should use these helpers
instead of dropping a fixed number of bytes.
"""

from __future__ import annotations

import array
import struct
import wave
from dataclasses import dataclass
from pathlib import Path


class AudioFormatError(ValueError):
    """Raised when a clip is not mono 16-bit PCM at the expected rate."""


@dataclass(frozen=True)
class PcmClip:
    pcm: bytes
    sample_rate: int
    channels: int = 1
    sample_width: int = 2

    def __post_init__(self) -> None:
        if self.channels != 1 or self.sample_width != 2:
            raise AudioFormatError(
                f"expected mono 16-bit PCM, got channels={self.channels} width={self.sample_width}"
            )
        if len(self.pcm) % 2 != 0:
            raise AudioFormatError("PCM byte length is not a multiple of the 16-bit sample width")

    @property
    def frame_count(self) -> int:
        return len(self.pcm) // 2


def parse_wav(data: bytes) -> PcmClip:
    """Return the PCM payload of a RIFF/WAVE file.

    Walks chunks instead of assuming a 44-byte header. Requires format 1
    (linear PCM), one channel, and 16-bit samples.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioFormatError("not a RIFF WAVE file")
    offset = 12
    fmt: dict | None = None
    payload: bytes | None = None
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        size = struct.unpack_from("<I", data, offset + 4)[0]
        start = offset + 8
        end = start + size
        if end > len(data):
            raise AudioFormatError(f"chunk {chunk_id!r} extends past the end of the file")
        body = data[start:end]
        if chunk_id == b"fmt ":
            if len(body) < 16:
                raise AudioFormatError("fmt chunk is shorter than 16 bytes")
            audio_format, channels, rate, _byte_rate, _align, bits = struct.unpack_from(
                "<HHIIHH", body
            )
            fmt = {
                "audio_format": audio_format,
                "channels": channels,
                "sample_rate": rate,
                "bits": bits,
            }
        elif chunk_id == b"data":
            payload = body
        offset = end + (size % 2)
    if fmt is None or payload is None:
        raise AudioFormatError("WAVE file is missing a fmt or data chunk")
    if fmt["audio_format"] != 1:
        raise AudioFormatError(f"WAVE audio format {fmt['audio_format']} is not linear PCM")
    if fmt["channels"] != 1 or fmt["bits"] != 16:
        raise AudioFormatError(
            f"expected mono 16-bit PCM, got channels={fmt['channels']} bits={fmt['bits']}"
        )
    return PcmClip(payload, int(fmt["sample_rate"]), 1, 2)


def load_pcm(data: bytes, *, expected_rate: int) -> PcmClip:
    """Load a WAV container or headerless PCM and require `expected_rate`."""
    if data.startswith(b"RIFF"):
        clip = parse_wav(data)
    else:
        clip = PcmClip(data, expected_rate, 1, 2)
    if clip.sample_rate != expected_rate:
        raise AudioFormatError(
            f"sample rate {clip.sample_rate} does not match expected {expected_rate}"
        )
    return clip


def silence(duration_sec: float, sample_rate: int) -> PcmClip:
    if duration_sec < 0:
        raise AudioFormatError("silence duration must be non-negative")
    frames = int(round(duration_sec * sample_rate))
    return PcmClip(b"\x00" * (frames * 2), sample_rate, 1, 2)


def write_wav(path: Path | str, clip: PcmClip) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(clip.sample_rate)
        handle.writeframes(clip.pcm)


def concatenate(
    clips: list[PcmClip],
    *,
    gap_ms: float = 0,
    crossfade_ms: float = 0,
) -> PcmClip:
    """Join clips at one sample rate.

    With no gap, a positive crossfade overlap-adds the join.
    With a gap, each interior edge is faded and silence is inserted.
    The first sample of the first clip and the last sample of the last
    clip are left unchanged.
    """
    if not clips:
        raise AudioFormatError("no clips to concatenate")
    rate = clips[0].sample_rate
    for clip in clips[1:]:
        if clip.sample_rate != rate:
            raise AudioFormatError(
                f"sample rate {clip.sample_rate} does not match {rate}"
            )
    if gap_ms < 0 or crossfade_ms < 0:
        raise AudioFormatError("gap and crossfade must be non-negative")
    fade_frames = int(round(rate * crossfade_ms / 1000.0))
    gap_frames = int(round(rate * gap_ms / 1000.0))
    samples = [_to_samples(clip.pcm) for clip in clips]
    if gap_frames > 0 and fade_frames > 0:
        faded = []
        for index, frames in enumerate(samples):
            item = frames
            if index > 0:
                item = _fade_in(item, fade_frames)
            if index < len(samples) - 1:
                item = _fade_out(item, fade_frames)
            faded.append(item)
        samples = faded
        pieces = []
        gap = array.array("h", [0] * gap_frames)
        for index, frames in enumerate(samples):
            if index:
                pieces.append(gap)
            pieces.append(frames)
        merged = _join(pieces)
    elif fade_frames > 0:
        merged = samples[0]
        for nxt in samples[1:]:
            merged = _crossfade(merged, nxt, fade_frames)
    elif gap_frames > 0:
        gap = array.array("h", [0] * gap_frames)
        pieces = []
        for index, frames in enumerate(samples):
            if index:
                pieces.append(gap)
            pieces.append(frames)
        merged = _join(pieces)
    else:
        merged = _join(samples)
    return PcmClip(_from_samples(merged), rate, 1, 2)


def clip_offsets(lengths: list[int], *, gap_ms: float, crossfade_ms: float, rate: int) -> list[int]:
    """Start frame of each clip as ``concatenate`` lays them out."""
    gap = int(round(rate * gap_ms / 1000.0))
    fade = int(round(rate * crossfade_ms / 1000.0))
    starts: list[int] = []
    position = 0
    for index, length in enumerate(lengths):
        if index:
            if gap > 0:
                position += gap
            elif fade > 0:
                position -= min(fade, lengths[index - 1], length)
        starts.append(position)
        position += length
    return starts


def assemble_clips(
    paths: list[Path | str],
    dest: Path | str,
    *,
    sample_rate: int,
    gap_ms: float = 0,
    crossfade_ms: float = 0,
    tail_ms: float = 0,
) -> PcmClip:
    """Load WAV or raw PCM files in order and write one WAV, optionally ending in silence."""
    clips = [load_pcm(Path(path).read_bytes(), expected_rate=sample_rate) for path in paths]
    mixed = concatenate(clips, gap_ms=gap_ms, crossfade_ms=crossfade_ms)
    if tail_ms > 0:
        silence = b"\x00\x00" * int(round(mixed.sample_rate * tail_ms / 1000.0))
        mixed = PcmClip(mixed.pcm + silence, mixed.sample_rate)
    write_wav(dest, mixed)
    return mixed


def _to_samples(pcm: bytes) -> array.array:
    samples = array.array("h")
    samples.frombytes(pcm)
    if samples.itemsize != 2:
        raise AudioFormatError("int16 array is not 2 bytes on this platform")
    if _native_is_big():
        samples.byteswap()
    return samples


def _from_samples(samples: array.array) -> bytes:
    if _native_is_big():
        samples = array.array("h", samples)
        samples.byteswap()
    return samples.tobytes()


def _native_is_big() -> bool:
    return array.array("h", [1]).tobytes()[0] == 0


def _join(parts: list[array.array]) -> array.array:
    out = array.array("h")
    for part in parts:
        out.extend(part)
    return out


def _crossfade(left: array.array, right: array.array, fade: int) -> array.array:
    fade = min(fade, len(left), len(right))
    if fade <= 0:
        return _join([left, right])
    out = array.array("h", left[:-fade])
    for index in range(fade):
        weight = (index + 0.5) / fade
        mixed = int(round(left[-fade + index] * (1.0 - weight) + right[index] * weight))
        out.append(_clamp(mixed))
    out.extend(right[fade:])
    return out


def _fade_out(frames: array.array, fade: int) -> array.array:
    fade = min(fade, len(frames))
    if fade <= 0:
        return frames
    out = array.array("h", frames)
    for index in range(fade):
        weight = 1.0 - ((index + 0.5) / fade)
        pos = len(out) - fade + index
        out[pos] = _clamp(int(round(out[pos] * weight)))
    return out


def _fade_in(frames: array.array, fade: int) -> array.array:
    fade = min(fade, len(frames))
    if fade <= 0:
        return frames
    out = array.array("h", frames)
    for index in range(fade):
        weight = (index + 0.5) / fade
        out[index] = _clamp(int(round(out[index] * weight)))
    return out


def _clamp(value: int) -> int:
    return max(-32768, min(32767, value))
