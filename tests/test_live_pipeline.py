"""Live stage behavior with mocked Gemini clients."""

import base64
import json
import struct
import threading
from pathlib import Path

import pytest

from expressive_audiobook.audio import parse_wav
from expressive_audiobook.cli import main
from expressive_audiobook.config import AppConfig, StyleProfile, load_config
from expressive_audiobook.gemini_api import design_voice_sdk, generate_prep_sdk
from expressive_audiobook.models import Chunk, to_jsonable
from expressive_audiobook.pipeline import (
    format_estimate,
    run_prep_model,
    run_sample,
    select_chunks,
)
from expressive_audiobook.prep import ChunkPrepFailed, prepare_chunk_output
from expressive_audiobook.tts import (
    PREBUILT_VOICES,
    GeminiTtsClient,
    classify_voice_name,
    plan_tts_requests,
)
from pdf_factory import build_outline_pdf

ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path) -> AppConfig:
    config = load_config(ROOT / "config" / "default.toml")
    config.paths.artifacts = str(tmp_path / "artifacts")
    config.paths.ingest = str(tmp_path / "artifacts" / "ingest")
    config.paths.chunks = str(tmp_path / "artifacts" / "chunks")
    config.paths.prep = str(tmp_path / "artifacts" / "prep")
    config.paths.tts = str(tmp_path / "artifacts" / "tts")
    config.paths.audio = str(tmp_path / "artifacts" / "audio")
    config.prep.max_validation_retries = 1
    config.prep.max_http_retries = 2
    config.prep.backoff_initial_seconds = 0.01
    config.tts.max_http_retries = 2
    config.tts.backoff_initial_seconds = 0.01
    return config


def _chunk(chunk_id="c001-0000", text="The road stayed empty.", order=0, duration=30, words=6) -> Chunk:
    return Chunk(
        id=chunk_id,
        chapter_id="ch-001",
        chapter_index=1,
        chapter_title="A Short Chapter",
        order=order,
        kind="prose",
        text=text,
        speakers=["narrator"],
        source_pages=[1],
        estimated_duration_sec=duration,
        word_count=words,
        block_ids=["b-1"],
    )


def _valid(chunk: Chunk, config: AppConfig, style="even, unhurried") -> dict:
    descriptions = []
    anchors = []
    for item in chunk.associated:
        if not item.speak:
            continue
        anchors.append(item.anchor)
        spoken = {
            "figure": "The figure shows the relationship in the caption.",
            "formula": "The expression relates the quantities on the page.",
            "footnote": "A brief note.",
            "citation": "A source is cited.",
        }.get(item.kind, "A related note.")
        descriptions.append(
            {
                "element_id": item.id,
                "kind": item.kind,
                "anchor": item.anchor,
                "spoken_text": spoken,
                "decision": "mention",
                "reason": "",
            }
        )
    return {
        "prompt_version": config.prep.prompt_version,
        "chunk_id": chunk.id,
        "turns": [
            {
                "text": "The road stayed empty and the pace stayed even.",
                "style": style,
                "speaker": "narrator",
                "rationale": "Ordinary narration needs little color.",
                "block_ids": anchors or ["b-1"],
                "inserted_descriptions": descriptions,
            }
        ],
    }


def _chunk_from_request(request: dict) -> dict:
    text = request["contents"][0]["parts"][0]["text"]
    marker = "## Chunk\n\n"
    start = text.rfind(marker)
    rest = text[start + len(marker) :]
    end = rest.find("\n## Read-only")
    return json.loads(rest[:end].strip())


def _valid_from_request(request: dict, config: AppConfig) -> dict:
    raw = _chunk_from_request(request)
    return _valid(Chunk.from_dict(raw), config)


class _ScriptedPrep:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def test_validation_errors_are_fed_back_and_then_accepted():
    config = load_config(ROOT / "config" / "default.toml")
    config.prep.max_validation_retries = 1
    chunk = _chunk()
    bad = {
        "prompt_version": config.prep.prompt_version,
        "chunk_id": "c999-0000",
        "turns": [
            {
                "text": "The road stayed empty.",
                "style": "",
                "speaker": "narrator",
                "rationale": "flat",
                "inserted_descriptions": [],
            }
        ],
    }
    client = _ScriptedPrep([bad, _valid(chunk, config)])
    output = prepare_chunk_output(chunk, StyleProfile(), config, client)
    assert output["chunk_id"] == chunk.id
    feedback = client.requests[1]["contents"][-1]["parts"][0]["text"]
    assert "does not match" in feedback


def test_schema_errors_are_retried():
    config = load_config(ROOT / "config" / "default.toml")
    config.prep.max_validation_retries = 1
    chunk = _chunk()
    client = _ScriptedPrep([{"chunk_id": chunk.id}, _valid(chunk, config)])
    output = prepare_chunk_output(chunk, StyleProfile(), config, client)
    assert output["turns"]
    assert "validation" in client.requests[1]["contents"][-1]["parts"][0]["text"].lower()


def test_prep_failure_names_the_chunk():
    config = load_config(ROOT / "config" / "default.toml")
    config.prep.max_validation_retries = 1
    chunk = _chunk()
    bad = {"prompt_version": config.prep.prompt_version, "chunk_id": "other", "turns": []}
    client = _ScriptedPrep([bad, bad])
    with pytest.raises(ChunkPrepFailed) as caught:
        prepare_chunk_output(chunk, StyleProfile(), config, client)
    assert chunk.id in str(caught.value)


def test_completed_chunks_are_skipped_on_rerun(tmp_path: Path):
    config = _config(tmp_path)
    chunk = _chunk()
    path = Path(config.paths.chunks) / "001.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(to_jsonable(chunk)) + "\n", encoding="utf-8")
    client = _ScriptedPrep([_valid(chunk, config)])
    run_prep_model(config, StyleProfile(), client, chapter=1)
    assert client.requests
    again = _ScriptedPrep([_valid(chunk, config)])
    run_prep_model(config, StyleProfile(), again, chapter=1)
    assert again.requests == []
    saved = json.loads((Path(config.paths.prep) / "outputs" / "chunks" / f"{chunk.id}.json").read_text())
    assert saved["chunk_id"] == chunk.id


def test_prep_concurrency_stays_within_the_limit(tmp_path: Path):
    config = _config(tmp_path)
    config.prep.concurrency = 2
    chunks = [_chunk(f"c001-{index:04d}", order=index, text=f"Sentence {index}.") for index in range(4)]
    path = Path(config.paths.chunks) / "001.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(to_jsonable(chunk)) + "\n" for chunk in chunks), encoding="utf-8")
    release = threading.Event()
    started = threading.Event()
    lock = threading.Lock()
    state = {"current": 0, "peak": 0}

    class Client:
        def generate(self, request):
            with lock:
                state["current"] += 1
                state["peak"] = max(state["peak"], state["current"])
                if state["current"] >= 2:
                    started.set()
            assert release.wait(5)
            with lock:
                state["current"] -= 1
            raw = _chunk_from_request(request)
            return _valid(Chunk.from_dict(raw), config)

    try:
        worker = threading.Thread(
            target=lambda: run_prep_model(config, StyleProfile(), Client(), chapter=1),
            daemon=True,
        )
        worker.start()
        assert started.wait(5)
        threading.Event().wait(0.05)
        assert state["peak"] <= 2
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()


def test_select_chunks_reaches_the_duration_and_can_start_later():
    chunks = [
        _chunk("c001-0000", order=0, duration=30),
        _chunk("c001-0001", order=1, duration=25),
        _chunk("c001-0002", order=2, duration=40, text="GUIDE: You brought the map."),
    ]
    selected = select_chunks(chunks, seconds=60)
    assert [chunk.order for chunk in selected] == [0, 1, 2]
    later = select_chunks(chunks, start=2, seconds=60)
    assert [chunk.order for chunk in later] == [2]
    capped = select_chunks(chunks, seconds=60, max_chunks=1)
    assert [chunk.order for chunk in capped] == [0]


def test_estimate_uses_config_pricing():
    config = load_config(ROOT / "config" / "default.toml")
    text = format_estimate([_chunk(duration=10, words=100)], config)
    assert "chunks: 1\n" in text
    assert "audio_minutes: 0.17\n" in text
    tokens = round(100 * config.cost.tokens_per_word)
    output = 10 * config.cost.audio_tokens_per_second / 1_000_000 * config.cost.tts_output_price_per_million
    input_cost = tokens / 1_000_000 * config.cost.tts_input_price_per_million
    assert f"estimated_cost_usd: {output + input_cost:.4f}" in text


def _wav_with_junk(pcm: bytes, rate: int = 24000) -> bytes:
    def chunk(tag: bytes, payload: bytes) -> bytes:
        pad = b"\x00" if len(payload) % 2 else b""
        return tag + struct.pack("<I", len(payload)) + payload + pad

    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    body = b"WAVE" + chunk(b"fmt ", fmt) + chunk(b"JUNK", b"skip-me") + chunk(b"data", pcm)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _audio_response(blob: bytes, mime: str = "audio/l16") -> bytes:
    payload = {
        "candidates": [
            {"content": {"parts": [{"inlineData": {"mimeType": mime, "data": base64.b64encode(blob).decode()}}]}}
        ]
    }
    return json.dumps(payload).encode()


def _one_request(config, voice="Schedar", voice_type="prebuilt"):
    from expressive_audiobook.config import VoiceMapping

    profile = StyleProfile(character_voices={"narrator": VoiceMapping(voice, voice_type)})
    return plan_tts_requests(
        [{"text": "The road stayed empty.", "style": "even", "speaker": "narrator"}],
        profile,
        config,
    )[0]


def test_synthesize_accepts_pcm_and_wav_and_retries_429():
    config = load_config(ROOT / "config" / "default.toml")
    config.tts.max_http_retries = 2
    config.tts.backoff_initial_seconds = 0.01
    pcm = b"\x01\x00\x02\x00\x03\x00\x04\x00"
    wav = _wav_with_junk(pcm)
    calls = {"n": 0}
    seen = {}

    def sender(method, url, headers, body):
        calls["n"] += 1
        seen["body"] = body
        seen["url"] = url
        if calls["n"] == 1:
            return 429, b'{"error":"slow"}'
        return 200, _audio_response(wav, "audio/wav")

    slept = []
    client = GeminiTtsClient("test-key", config, sender=sender, sleep=slept.append)
    assert client.synthesize(_one_request(config)) == pcm
    assert slept == [0.01]
    assert seen["body"]["generationConfig"]["responseFormat"]["audio"]["mimeType"] == "AUDIO_L16"
    assert seen["body"]["generationConfig"]["speechConfig"]["voiceConfig"]["voice"] == "Schedar"
    assert "model" not in seen["body"]

    def pcm_sender(method, url, headers, body):
        return 200, _audio_response(pcm)

    assert GeminiTtsClient("test-key", config, sender=pcm_sender).synthesize(_one_request(config)) == pcm


def test_synthesize_posts_design_and_library_ids():
    config = load_config(ROOT / "config" / "default.toml")
    seen = {}

    def sender(method, url, headers, body):
        seen["voice"] = body["generationConfig"]["speechConfig"]["voiceConfig"]["voice"]
        return 200, _audio_response(b"\x00\x00" * 8)

    client = GeminiTtsClient("test-key", config, sender=sender)
    client.synthesize(_one_request(config, "voice_example", "design"))
    assert seen["voice"] == "voice_example"
    client.synthesize(_one_request(config, "library-narrator-01", "library"))
    assert seen["voice"] == "library-narrator-01"


def test_existing_clip_is_not_synthesized_again(tmp_path: Path):
    from expressive_audiobook.audio import PcmClip, write_wav
    from expressive_audiobook.pipeline import run_tts_model

    config = _config(tmp_path)
    chunk = _chunk()
    output = _valid(chunk, config)
    out = Path(config.paths.prep) / "outputs" / "001.jsonl"
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(output) + "\n", encoding="utf-8")
    chunk_path = Path(config.paths.chunks) / "001.jsonl"
    chunk_path.parent.mkdir(parents=True)
    chunk_path.write_text(json.dumps(to_jsonable(chunk)) + "\n", encoding="utf-8")
    dest = Path(config.paths.audio) / "clips" / "001" / f"0000-{chunk.id}-r00.wav"
    write_wav(dest, PcmClip(b"\x00\x00" * 16, 24000))
    calls = {"n": 0}

    class Client:
        def synthesize(self, request):
            calls["n"] += 1
            return b"\x00\x00" * 16

    run_tts_model(config, StyleProfile(character_voices={}), Client(), chapter=1)
    assert calls["n"] == 0


def test_design_voice_posts_prompted_body_and_prints_id():
    config = load_config(ROOT / "config" / "default.toml")
    seen = {}

    def sender(method, url, headers, body):
        seen["method"] = method
        seen["url"] = url
        seen["body"] = body
        return 200, b'{"id":"voice_abc123"}'

    client = GeminiTtsClient("test-key", config, sender=sender)
    assert client.design_voice("A calm even narrator") == "voice_abc123"
    assert seen["method"] == "POST"
    assert seen["url"].endswith("/v1beta/voices")
    assert seen["body"]["store"] is True
    assert seen["body"]["voice"]["type"] == "prompted"
    assert seen["body"]["voice"]["prompted"]["input"] == "A calm even narrator"


def test_sdk_prep_and_voice_design_use_the_client_methods():
    class Models:
        def generate_content(self, *, model, contents, config):
            assert model == "gemini-3.8-flash"
            assert config.response_mime_type == "application/json"
            assert contents[0].parts
            return type("Response", (), {"text": '{"chunk_id": "c001-0000"}'})()

    class Voices:
        def create(self, *, store, voice):
            assert store is True
            assert voice["type"] == "prompted"
            assert voice["prompted"]["input"] == "A calm even narrator"
            return type("Voice", (), {"id": "voice_from_sdk"})()

    prep_client = type("Client", (), {"models": Models()})()
    voice_client = type("Client", (), {"voices": Voices()})()
    parsed = generate_prep_sdk(
        "key",
        {
            "model": "gemini-3.8-flash",
            "contents": [{"role": "user", "parts": [{"text": "Return JSON"}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
        },
        client=prep_client,
    )
    assert parsed["chunk_id"] == "c001-0000"
    assert design_voice_sdk(
        "key",
        {"type": "prompted", "prompted": {"input": "A calm even narrator"}},
        client=voice_client,
    ) == "voice_from_sdk"


def test_prebuilt_names_are_checked_and_other_ids_pass():
    config = load_config(ROOT / "config" / "default.toml")
    assert config.tts.default_voice == "Schedar"
    assert config.tts.default_voice in PREBUILT_VOICES
    assert classify_voice_name("Schedar") == "prebuilt"
    assert classify_voice_name("Charon") == "prebuilt"
    assert classify_voice_name("Sulafat") == "prebuilt"
    assert classify_voice_name("voice_abc") == "design"
    assert classify_voice_name("voicekey_abc") == "replication"
    assert classify_voice_name("library-narrator-01") == "library"
    with pytest.raises(ValueError):
        classify_voice_name("Nope")


def test_sample_writes_wav_and_review_outside_the_chapter_cache(tmp_path: Path):
    config = _config(tmp_path)
    pdf = tmp_path / "owned-book.pdf"
    build_outline_pdf(pdf)

    class Prep:
        def generate(self, request):
            return _valid_from_request(request, config)

    class Speech:
        def __init__(self):
            self.voices = []

        def synthesize(self, request):
            self.voices.append(request.speakers[0]["voice"])
            return b"\x00\x00" * 2400

    speech = Speech()
    result = run_sample(
        config,
        StyleProfile(),
        pdf=pdf,
        chapter=1,
        prep_client=Prep(),
        tts_client=speech,
        seconds=60,
        stamp="20261028T120000Z",
    )
    wav = result["wav"]
    review = result["review"]
    assert wav == tmp_path / "artifacts" / "owned-book" / "samples" / "1-20261028T120000Z.wav"
    assert review == wav.with_suffix(".md")
    clip = parse_wav(wav.read_bytes())
    assert clip.sample_rate == 24000
    assert clip.channels == 1 and clip.sample_width == 2
    assert clip.frame_count > 0
    report = review.read_text(encoding="utf-8")
    source = result["chunks"][0].text.split()[0]
    assert source in report
    assert "The road stayed empty and the pace stayed even." in report
    assert "even, unhurried" in report
    assert "Ordinary narration needs little color." in report
    assert "narrator" in report
    assert "### Source" in report
    assert not (tmp_path / "artifacts" / "prep").exists()
    assert not (tmp_path / "artifacts" / "audio").exists()
    assert speech.voices
    assert set(speech.voices) == {"Schedar"}


def test_cli_sample_dry_run_estimate_and_voice_override(tmp_path: Path, capsys):
    config_path = _write_config(tmp_path)
    pdf = tmp_path / "owned-book.pdf"
    build_outline_pdf(pdf)
    assert (
        main(
            [
                "--config",
                str(config_path),
                "sample",
                "--pdf",
                str(pdf),
                "--chapter",
                "1",
                "--style",
                "even and clear",
                "--sample-seconds",
                "60",
                "--dry-run",
            ]
        )
        == 0
    )
    dry = capsys.readouterr().out
    assert "chunks:" in dry
    assert "audio_minutes:" in dry
    assert "estimated_cost_usd:" in dry
    assert "dry_run: sample" in dry
    assert main(["--config", str(config_path), "run", "--pdf", str(pdf), "--chapter", "1", "--voice", "Nope"]) == 1
    assert "not a documented prebuilt voice" in capsys.readouterr().err


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "audiobook.toml"
    art = tmp_path / "artifacts"
    config_path.write_text(
        "\n".join(
            [
                "[paths]",
                f'artifacts = "{art}"',
                f'ingest = "{art / "ingest"}"',
                f'chunks = "{art / "chunks"}"',
                f'prep = "{art / "prep"}"',
                f'tts = "{art / "tts"}"',
                f'audio = "{art / "audio"}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path
