"""Music beds, ducking, and the chapter mix, with synthetic tones."""

import numpy as np
import pytest

from pdf_expressive_audiobook.config import SoundConfig
from pdf_expressive_audiobook.mixer import (
    MixPlan,
    Spot,
    build_bed,
    db_to_gain,
    decode,
    duck_curve,
    mix_chapter,
    rms,
    write_wav,
)

RATE = 8000  # a low rate keeps the tests fast; the mixer takes the rate as a parameter


def tone(seconds: float, hz: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    wave = (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    return np.stack([wave, wave], axis=1)


def test_bed_has_exact_length_and_never_repeats_a_track_back_to_back():
    tracks = [tone(10, 220), tone(10, 330), tone(10, 440)]
    bed = build_bed(tracks, 95, rate=RATE, crossfade_s=2, seed=7)
    assert len(bed) == 95 * RATE
    assert np.isfinite(bed).all()
    # No gaps: the bed never drops to silence at a join.
    window = RATE // 2
    levels = [rms(bed[i : i + window]) for i in range(0, len(bed) - window, window)]
    assert min(levels) > 0.5 * max(levels)


def test_bed_is_deterministic_per_seed_and_differs_between_seeds():
    tracks = [tone(10, 220), tone(10, 330)]
    a = build_bed(tracks, 40, rate=RATE, crossfade_s=2, seed=1)
    b = build_bed(tracks, 40, rate=RATE, crossfade_s=2, seed=1)
    c = build_bed(tracks, 40, rate=RATE, crossfade_s=2, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_single_track_bed_loops_with_a_crossfade():
    bed = build_bed([tone(6, 220)], 30, rate=RATE, crossfade_s=2, seed=3)
    assert len(bed) == 30 * RATE


def test_duck_curve_dips_under_speech_and_recovers():
    curve = duck_curve([(2.0, 4.0)], 10 * RATE, rate=RATE, duck_db=-8, attack_ms=100, release_ms=500)
    assert curve[RATE // 2] == pytest.approx(1.0, abs=0.01)
    assert curve[3 * RATE] == pytest.approx(db_to_gain(-8), abs=0.02)
    assert curve[9 * RATE] == pytest.approx(1.0, abs=0.02)


def test_mix_adds_lead_in_tail_and_keeps_music_under_the_voice():
    config = SoundConfig(preroll_s=2, tail_s=3, bed_crossfade_s=2, duck_db=-6)
    voice = tone(20, 200, amp=0.4)
    plan = MixPlan(spans=[(0.0, 20.0)], music=[tone(12, 330), tone(12, 440)], music_db=-20, seed=5)
    mixed = mix_chapter(voice, plan, config, rate=RATE)
    assert len(mixed) == (2 + 20 + 3) * RATE
    assert rms(mixed[: RATE // 2]) < rms(mixed[10 * RATE : 11 * RATE])  # fades in
    assert rms(mixed[-RATE // 2 :]) < 0.2 * rms(mixed[10 * RATE : 11 * RATE])  # fades out
    # Music alone sits well under the narration level.
    assert rms(mixed[10 * RATE : 11 * RATE] - np.pad(voice, ((2 * RATE, 3 * RATE), (0, 0)))[10 * RATE : 11 * RATE]) < 0.3 * rms(voice)


def test_spot_effect_lands_at_its_time():
    config = SoundConfig()
    voice = tone(10, 200, amp=0.2)
    click = tone(0.5, 1000, amp=0.5)
    plan = MixPlan(spans=[(0.0, 10.0)], spots=[Spot(at=6.0, audio=click, db=-6)])
    mixed = mix_chapter(voice, plan, config, rate=RATE)
    assert len(mixed) == len(voice)  # no music: no lead-in or tail
    before = rms(mixed[4 * RATE : 5 * RATE] - voice[4 * RATE : 5 * RATE])
    after = rms(mixed[int(6.1 * RATE) : int(6.4 * RATE)] - voice[int(6.1 * RATE) : int(6.4 * RATE)])
    assert before < 1e-6 < after


def test_peak_is_limited():
    config = SoundConfig()
    plan = MixPlan(spans=[(0, 5)], spots=[Spot(at=1.0, audio=tone(1, 500, amp=0.9), db=12)])
    assert np.max(np.abs(mix_chapter(tone(5, 200, amp=0.9), plan, config, rate=RATE))) <= 0.97 + 1e-6


def test_decode_round_trip(tmp_path):
    path = tmp_path / "t.wav"
    write_wav(path, tone(1, 440), rate=RATE)
    audio = decode(path, rate=RATE)
    assert audio.shape[1] == 2 and abs(len(audio) - RATE) < 5
    assert decode(path, rate=16000).shape[0] == pytest.approx(16000, abs=10)
