"""Mix narration with background music and sound effects.

ffmpeg decodes and resamples everything to 44.1 kHz stereo; numpy does the mix.
Because the narration timeline is known (every clip's start and length), ducking
is computed from it directly instead of detecting speech.

Music is built as a *bed*: several variants of the same mood are joined with
long equal-power crossfades in a shuffled order that never repeats the same
track twice in a row, starting at a random point, with a slow level drift, so a
short loop does not read as a loop.
"""

from __future__ import annotations

import random
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import SoundConfig
from .export import ffmpeg_path

RATE = 44100
ENV_RATE = 50
TARGET_RMS_DB = -20.0
PEAK_LIMIT = 0.97


class MixError(RuntimeError):
    pass


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def decode(path: Path | str, rate: int = RATE) -> np.ndarray:
    """Any audio file to float32 stereo, shape ``(frames, 2)``."""
    ffmpeg = ffmpeg_path()
    if ffmpeg is None:
        raise MixError("ffmpeg is required to mix music and sound effects (brew install ffmpeg)")
    done = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(rate), "-"],
        capture_output=True,
    )
    if done.returncode != 0:
        raise MixError(f"ffmpeg could not decode {path}: {done.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(done.stdout, dtype="<f4").reshape(-1, 2).copy()


def write_wav(path: Path | str, audio: np.ndarray, rate: int = RATE) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0


def level_to(audio: np.ndarray, target_db: float = TARGET_RMS_DB) -> np.ndarray:
    level = rms(audio)
    return audio if level < 1e-6 else audio * (db_to_gain(target_db) / level)


def _fades(count: int) -> tuple[np.ndarray, np.ndarray]:
    angle = (np.arange(count, dtype=np.float32) + 0.5) / max(1, count) * (np.pi / 2)
    return np.cos(angle)[:, None], np.sin(angle)[:, None]


def build_bed(
    tracks: list[np.ndarray],
    seconds: float,
    *,
    rate: int = RATE,
    crossfade_s: float = 8.0,
    seed: int = 0,
    random_start: bool = True,
    drift: bool = True,
) -> np.ndarray:
    """A bed of exactly ``seconds`` made from ``tracks`` without an obvious loop point."""
    usable = [level_to(t) for t in tracks if len(t) > rate]
    if not usable:
        raise MixError("No music track is long enough to build a background bed")
    target = int(seconds * rate)
    rng = random.Random(seed)
    cf = max(1, min(int(crossfade_s * rate), min(len(t) for t in usable) // 3))
    order: list[int] = []
    deck: list[int] = []
    produced = 0
    first = True
    segments: list[np.ndarray] = []
    while produced < target + cf:
        if not deck:
            deck = list(range(len(usable)))
            rng.shuffle(deck)
            if len(deck) > 1 and order and deck[0] == order[-1]:
                deck.append(deck.pop(0))
        index = deck.pop(0)
        order.append(index)
        track = usable[index]
        if first and random_start:
            track = track[rng.randrange(0, max(1, len(track) // 4)):]
            if len(track) < 3 * cf:
                track = usable[index]
        first = False
        segments.append(track)
        produced += len(track) - cf
    fade_out, fade_in = _fades(cf)
    pieces: list[np.ndarray] = []
    tail: np.ndarray | None = None
    for segment in segments:
        if tail is None:
            pieces.append(segment[: len(segment) - cf])
        else:
            pieces.append(tail * fade_out + segment[:cf] * fade_in)
            pieces.append(segment[cf : len(segment) - cf])
        tail = segment[len(segment) - cf :]
    bed = np.concatenate(pieces)[:target]
    if drift and len(bed):
        t = np.arange(len(bed), dtype=np.float32) / rate
        bed = bed * (1.0 + 0.10 * np.sin(2 * np.pi * t / 37.0 + rng.uniform(0, 6.28)))[:, None]
    return bed.astype(np.float32)


def duck_curve(
    spans: list[tuple[float, float]],
    frames: int,
    *,
    rate: int = RATE,
    duck_db: float = -7.0,
    attack_ms: float = 400.0,
    release_ms: float = 1200.0,
) -> np.ndarray:
    """Gain (1.0 = untouched) that dips under speech and eases back during pauses."""
    length = frames // max(1, rate // ENV_RATE) + 2
    target = np.ones(length, dtype=np.float64)
    floor = db_to_gain(duck_db)
    for start, end in spans:
        target[max(0, int(start * ENV_RATE)) : int(end * ENV_RATE) + 1] = floor
    attack = np.exp(-1.0 / max(1e-3, attack_ms / 1000.0 * ENV_RATE))
    release = np.exp(-1.0 / max(1e-3, release_ms / 1000.0 * ENV_RATE))
    smooth = np.empty(length)
    y = 1.0
    for i in range(length):
        goal = target[i]
        k = attack if goal < y else release
        y = goal + (y - goal) * k
        smooth[i] = y
    positions = np.arange(frames, dtype=np.float64) / rate * ENV_RATE
    return np.interp(positions, np.arange(length), smooth).astype(np.float32)


def _ramp(frames: int, rising: bool) -> np.ndarray:
    ramp = np.sin(np.linspace(0, np.pi / 2, max(1, frames), dtype=np.float32)) ** 2
    return ramp if rising else ramp[::-1]


@dataclass
class Spot:
    at: float  # seconds into the narration
    audio: np.ndarray
    db: float


@dataclass
class Ambience:
    start: float
    end: float
    audio: np.ndarray
    db: float


@dataclass
class MixPlan:
    spans: list[tuple[float, float]]  # speech spans, seconds into the narration
    music: list[np.ndarray] = field(default_factory=list)
    music_db: float = -24.0
    spots: list[Spot] = field(default_factory=list)
    ambiences: list[Ambience] = field(default_factory=list)
    seed: int = 0


def speech_level(dry: np.ndarray, spans: list[tuple[float, float]], rate: int) -> float:
    pieces = [dry[int(s * rate) : int(e * rate)] for s, e in spans if e > s]
    joined = np.concatenate(pieces) if pieces else dry
    return rms(joined[:: 4]) or 1e-3


def mix_chapter(dry: np.ndarray, plan: MixPlan, config: SoundConfig, *, rate: int = RATE) -> np.ndarray:
    """Narration plus music and effects. The result is longer by the music's lead-in and tail."""
    has_music = bool(plan.music)
    pre = int(config.preroll_s * rate) if has_music else 0
    post = int(config.tail_s * rate) if has_music else 0
    total = pre + len(dry) + post
    out = np.zeros((total, 2), dtype=np.float32)
    out[pre : pre + len(dry)] += dry
    reference = speech_level(dry, plan.spans, rate)
    shifted = [(s + pre / rate, e + pre / rate) for s, e in plan.spans]

    if has_music:
        bed = build_bed(plan.music, total / rate, rate=rate, crossfade_s=config.bed_crossfade_s, seed=plan.seed)
        if len(bed) < total:
            bed = np.pad(bed, ((0, total - len(bed)), (0, 0)), mode="edge")
        bed = bed * (reference * db_to_gain(plan.music_db) / db_to_gain(TARGET_RMS_DB))
        bed *= duck_curve(
            shifted, total, rate=rate, duck_db=config.duck_db,
            attack_ms=config.duck_attack_ms, release_ms=config.duck_release_ms,
        )[:, None]
        bed[:pre] *= _ramp(pre, True)[:, None]
        bed[total - post :] *= _ramp(post, False)[:, None]
        out += bed

    for spot in plan.spots:
        clip = level_to(spot.audio) * (reference * db_to_gain(spot.db) / db_to_gain(TARGET_RMS_DB))
        edge = min(len(clip) // 4, int(0.08 * rate))
        if edge:
            clip = clip.copy()
            clip[:edge] *= _ramp(edge, True)[:, None]
            clip[-edge:] *= _ramp(edge, False)[:, None]
        _add(out, clip, pre + int(spot.at * rate))

    for amb in plan.ambiences:
        frames = int((amb.end - amb.start) * rate)
        if frames < rate:
            continue
        loop = build_bed([amb.audio], frames / rate, rate=rate,
                         crossfade_s=min(1.5, len(amb.audio) / rate / 4), seed=plan.seed, random_start=False)
        loop = level_to(loop) * (reference * db_to_gain(amb.db) / db_to_gain(TARGET_RMS_DB))
        edge = min(len(loop) // 3, int(1.5 * rate))
        loop[:edge] *= _ramp(edge, True)[:, None]
        loop[-edge:] *= _ramp(edge, False)[:, None]
        _add(out, loop, pre + int(amb.start * rate))

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > PEAK_LIMIT:
        out *= PEAK_LIMIT / peak
    return out


def _add(out: np.ndarray, clip: np.ndarray, start: int) -> None:
    start = max(0, start)
    end = min(len(out), start + len(clip))
    if end > start:
        out[start:end] += clip[: end - start]


def join_chapters(chapters: list[np.ndarray], gap_s: float = 0.0, rate: int = RATE) -> np.ndarray:
    gap = np.zeros((int(gap_s * rate), 2), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for index, audio in enumerate(chapters):
        if index and len(gap):
            pieces.append(gap)
        pieces.append(audio)
    return np.concatenate(pieces)
