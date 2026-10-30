"""Gemini is optional: another text model and a local speech engine take its place."""

import copy
import io
import json
import wave
from pathlib import Path

import pytest

from expressive_audiobook.config import StyleProfile, VoiceMapping, _merge_config, load_config
from expressive_audiobook.engines import EngineError, make_text_client, make_tts_client
from expressive_audiobook.llm import OpenAICompatTextClient, chat_messages
from expressive_audiobook.speech_local import (
    OrpheusTtsClient,
    SpeechHttpClient,
    resolve_voice,
    split_pauses,
)
from expressive_audiobook.tts import plan_tts_requests

ROOT = Path(__file__).resolve().parents[1]


def _config(**sections):
    config = load_config(ROOT / "config" / "default.toml")
    return _merge_config(copy.deepcopy(config), sections)


def _reply(text: str):
    body = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
    return lambda method, url, headers, payload: (200, body)


def test_chat_messages_turn_parts_into_text_and_optional_images():
    request = {
        "contents": [
            {"role": "user", "parts": [{"text": "A"}, {"inlineData": {"mimeType": "image/png", "data": "QQ=="}}]},
            {"role": "model", "parts": [{"text": "B"}]},
        ]
    }
    assert chat_messages(request, vision=False) == [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
    ]
    seen = chat_messages(request, vision=True)
    assert seen[0]["content"][1]["image_url"]["url"] == "data:image/png;base64,QQ=="


def test_openai_client_sends_json_mode_and_parses_the_reply():
    sent = {}

    def sender(method, url, headers, body):
        sent.update(url=url, body=body, headers=headers)
        return _reply('<think>hmm</think>```json\n{"ok": 1}\n```')(method, url, headers, body)

    config = _config(text={"provider": "openai", "model": "qwen3:14b"})
    client = OpenAICompatTextClient(config, sender=sender)
    request = {
        "model": "gemini-ignored",
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3},
    }
    assert client.generate(request) == {"ok": 1}
    assert sent["url"] == "http://localhost:11434/v1/chat/completions"
    assert sent["body"]["model"] == "qwen3:14b"
    assert sent["body"]["response_format"] == {"type": "json_object"}
    assert sent["body"]["temperature"] == 0.3
    assert "Authorization" not in sent["headers"]


def test_openai_client_needs_a_model_name():
    client = OpenAICompatTextClient(_config(text={"provider": "openai"}), sender=_reply("{}"))
    with pytest.raises(RuntimeError, match="text.model"):
        client.generate({"contents": []})


def test_factories_pick_the_engine_and_only_gemini_needs_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(ROOT / "tests")  # no .env here
    gemini = _config()
    assert make_text_client(gemini) is None
    with pytest.raises(EngineError, match="GEMINI_API_KEY"):
        make_text_client(gemini, required=True)
    local = _config(text={"provider": "openai", "model": "m"}, tts={"provider": "orpheus"})
    assert isinstance(make_text_client(local, required=True), OpenAICompatTextClient)
    assert isinstance(make_tts_client(local, required=True), OrpheusTtsClient)
    assert isinstance(make_tts_client(_config(tts={"provider": "http"})), SpeechHttpClient)


def test_provider_defaults_follow_the_chosen_engines():
    config = _config(text={"provider": "openai", "model": "qwen3:14b"}, tts={"provider": "orpheus"})
    assert config.prep.model == "qwen3:14b"
    assert "<chuckle>" in config.prep.allowed_tags
    assert "<whispers>" not in config.prep.allowed_tags
    assert config.prep.backchannels == []
    assert config.tts.concurrency == 1
    assert "no `style` control" in config.prep.engine_notes
    # An explicit setting in the same file wins.
    explicit = _config(tts={"provider": "orpheus"}, prep={"allowed_tags": ["<sigh>"]})
    assert explicit.prep.allowed_tags == ["<sigh>"]


def test_gemini_prompt_is_unchanged_by_the_engine_hook():
    from expressive_audiobook.prep import load_prompt_template

    assert "{{engine_notes}}" in load_prompt_template()
    config = _config()
    assert config.prep.engine_notes == ""


def test_pauses_become_silence_and_edges_are_dropped():
    assert split_pauses("<short pause> One <short pause> two <short pause><short pause> three <short pause>") == [
        "One",
        None,
        "two",
        None,
        None,
        "three",
    ]
    assert split_pauses("plain") == ["plain"]


def test_voices_map_to_the_engine():
    voices = ["tara", "leah", "leo"]
    assert resolve_voice("Leo", "narrator", voices, "tara") == "leo"
    assert resolve_voice("Schedar", "narrator", voices, "tara") == "tara"
    first = resolve_voice("Puck", "Achilles", voices, "tara")
    assert first in {"leah", "leo"}
    assert resolve_voice("Puck", "Achilles", voices, "tara") == first


def test_local_engine_plans_one_request_per_turn_and_speaks_with_real_pauses():
    config = _config(tts={"provider": "orpheus"})
    profile = StyleProfile(character_voices={"narrator": VoiceMapping("tara"), "Hero": VoiceMapping("leo")})
    turns = [
        {"text": "Hello <chuckle> there. <short pause> Bye.", "style": "warm", "speaker": "narrator"},
        {"text": "Right.", "style": "", "speaker": "Hero"},
    ]
    requests = plan_tts_requests(turns, profile, config)
    assert [r.api for r in requests] == ["orpheus", "orpheus"]
    assert requests[1].payload["voice"] == "leo"

    spoken = []

    class Worker:
        def speak(self, voice, text):
            spoken.append((voice, text))
            return b"\x01\x00" * 240

    client = OrpheusTtsClient(config, worker=Worker())
    pcm = client.synthesize(requests[0])
    assert spoken == [("tara", "Hello <chuckle> there."), ("tara", "Bye.")]
    pause = int(round(24000 * config.tts.orpheus.pause_ms / 1000)) * 2
    assert len(pcm) == 240 * 2 * 2 + pause
    assert client.synthesize(requests[1]) and spoken[-1] == ("leo", "Right.")
    assert client.list_voices()[0]["id"] == "tara"
    with pytest.raises(RuntimeError, match="Gemini"):
        client.design_voice("x")


def test_request_stamp_changes_with_voice_and_engine_settings():
    base = _config(tts={"provider": "orpheus"})
    other = _config(tts={"provider": "orpheus", "orpheus": {"temperature": 0.9}})
    turn = [{"text": "Hi.", "style": "", "speaker": "narrator"}]
    a = plan_tts_requests(turn, StyleProfile(), base)[0].payload
    b = plan_tts_requests(turn, StyleProfile(), other)[0].payload
    assert a != b


def test_speech_server_client_reads_wav_replies():
    config = _config(tts={"provider": "http"})
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\x02\x00" * 100)
    seen = {}

    def sender(method, url, headers, body):
        seen.update(url=url, body=body)
        return 200, buffer.getvalue()

    client = SpeechHttpClient(config, sender=sender)
    request = plan_tts_requests(
        [{"text": "Hello", "style": "", "speaker": "narrator"}], StyleProfile(), config
    )[0]
    assert client.synthesize(request) == b"\x02\x00" * 100
    assert seen["url"].endswith("/audio/speech")
    assert seen["body"]["voice"] == "tara"
