import array
import struct
import wave
from pathlib import Path

import pytest

from pdf_expressive_audiobook.audio import (
    AudioFormatError,
    PcmClip,
    assemble_clips,
    concatenate,
    load_pcm,
    parse_wav,
    silence,
    write_wav,
)


def _tone(frames: int, rate: int = 24000, amplitude: int = 8000) -> PcmClip:
    samples = array.array("h", [amplitude if index % 2 == 0 else -amplitude for index in range(frames)])
    return PcmClip(samples.tobytes(), rate, 1, 2)


def _wav_with_junk(pcm: bytes, rate: int = 24000) -> bytes:
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)

    def chunk(name: bytes, payload: bytes) -> bytes:
        pad = b"\x00" if len(payload) % 2 else b""
        return name + struct.pack("<I", len(payload)) + payload + pad

    body = chunk(b"JUNK", b"abcd") + chunk(b"fmt ", fmt) + chunk(b"data", pcm)
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def test_parse_standard_wav_and_junk_chunk_before_data(tmp_path: Path):
    clip = _tone(100)
    dest = tmp_path / "plain.wav"
    write_wav(dest, clip)
    plain = dest.read_bytes()
    assert plain[8:12] == b"WAVE"
    loaded = load_pcm(plain, expected_rate=24000)
    assert loaded.pcm == clip.pcm

    junked = _wav_with_junk(clip.pcm)
    assert junked.find(b"data") != 36
    parsed = parse_wav(junked)
    assert parsed.pcm == clip.pcm
    assert parsed.sample_rate == 24000


def test_rejects_wrong_rate_and_stereo():
    other = _tone(10, rate=16000)
    blob = _wav_with_junk(other.pcm, rate=16000)
    with pytest.raises(AudioFormatError):
        load_pcm(blob, expected_rate=24000)
    stereo_fmt = struct.pack("<HHIIHH", 1, 2, 24000, 96000, 4, 16)
    body = b"fmt " + struct.pack("<I", len(stereo_fmt)) + stereo_fmt
    body += b"data" + struct.pack("<I", 4) + b"\x00\x00\x00\x00"
    stereo = b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body
    with pytest.raises(AudioFormatError):
        parse_wav(stereo)


def test_silence_duration():
    clip = silence(0.5, 24000)
    assert clip.frame_count == 12000
    assert clip.pcm == b"\x00" * 24000


def test_crossfade_shortens_and_gap_inserts_silence():
    left = _tone(1000)
    right = _tone(1000, amplitude=2000)
    faded = concatenate([left, right], crossfade_ms=1000 / 24)
    assert faded.frame_count == 1000

    gapped = concatenate([left, right], gap_ms=100)
    assert gapped.frame_count == 2000 + int(round(24000 * 0.1))


def test_gap_and_fade_keep_length_and_soften_the_join():
    left = _tone(400, amplitude=10000)
    right = _tone(400, amplitude=10000)
    mixed = concatenate([left, right], gap_ms=10, crossfade_ms=5)
    gap_frames = int(round(24000 * 0.01))
    assert mixed.frame_count == 800 + gap_frames
    samples = array.array("h")
    samples.frombytes(mixed.pcm)
    join = 400
    assert abs(samples[join - 1]) < 10000


def test_assemble_writes_a_readable_wav(tmp_path: Path):
    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    write_wav(first, _tone(50))
    write_wav(second, _tone(70))
    dest = tmp_path / "out.wav"
    assemble_clips([first, second], dest, sample_rate=24000, gap_ms=0, crossfade_ms=0)
    with wave.open(str(dest)) as handle:
        assert handle.getframerate() == 24000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getnframes() == 120
