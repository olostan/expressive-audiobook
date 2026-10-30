import json
from pathlib import Path

import pytest

from expressive_audiobook.config import StyleProfile, VoiceMapping, load_config
from expressive_audiobook.models import to_jsonable
from expressive_audiobook.tts import (
    GeminiTtsClient,
    estimate_latency,
    estimate_tts_cost,
    plan_tts_requests,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "src" / "expressive_audiobook" / "default.toml")


def _profile() -> StyleProfile:
    return StyleProfile(
        id="test",
        character_voices={
            "narrator": VoiceMapping("Kore", "prebuilt"),
            "Guide": VoiceMapping("Charon", "prebuilt"),
            "Wanderer": VoiceMapping("Puck", "prebuilt"),
            "Guest": VoiceMapping("Orus", "prebuilt"),
            "Custom": VoiceMapping("voice_example", "design"),
            "Library": VoiceMapping("library-narrator-01", "library"),
        },
    )


def _turn(text, speaker, style=""):
    return {
        "text": text,
        "style": style,
        "speaker": speaker,
        "rationale": "RATIONALE_SENTINEL",
        "inserted_descriptions": [{"element_id": "fn-1", "spoken_text": "secret note"}],
    }


def test_two_prebuilt_speakers_share_one_request_and_a_third_splits():
    profile = _profile()
    paired = plan_tts_requests(
        [
            _turn("You brought the map.", "Guide", "lightly teasing"),
            _turn("I almost left it.", "Wanderer", "apologetic, warm"),
        ],
        profile,
        CONFIG,
    )
    assert len(paired) == 1
    assert paired[0].api == "generate_content"
    payload = paired[0].payload
    speech = payload["generationConfig"]["speechConfig"]
    assert "multiSpeakerVoiceConfig" in speech
    names = {item["speaker"] for item in speech["multiSpeakerVoiceConfig"]["speakerVoiceConfigs"]}
    assert names == {"Guide", "Wanderer"}
    parts = payload["contents"][0]["parts"]
    assert parts[0]["speech_metadata"]["speaker"] == "Guide"
    assert parts[0]["speech_metadata"]["style"] == "lightly teasing"
    assert payload["generationConfig"]["responseFormat"]["audio"]["mimeType"] == "AUDIO_L16"
    assert payload["generationConfig"]["responseFormat"]["audio"]["sampleRate"] == 24000
    blob = json.dumps(to_jsonable(paired[0]))
    assert "RATIONALE_SENTINEL" not in blob
    assert "secret note" not in blob

    split = plan_tts_requests(
        [
            _turn("One.", "Guide"),
            _turn("Two.", "Wanderer"),
            _turn("Three.", "Guest"),
        ],
        profile,
        CONFIG,
    )
    assert len(split) == 2
    assert len(split[0].speakers) == 2
    assert split[1].speakers[0]["speaker"] == "Guest"


def test_custom_and_library_voices_are_per_turn():
    profile = _profile()
    requests = plan_tts_requests(
        [_turn("Hello.", "Custom"), _turn("Again.", "Custom")],
        profile,
        CONFIG,
    )
    assert len(requests) == 2
    assert requests[0].payload["generationConfig"]["speechConfig"]["voiceConfig"]["voice"] == "voice_example"
    library = plan_tts_requests(
        [_turn("Hello.", "Library"), _turn("There.", "Guide")],
        profile,
        CONFIG,
    )
    assert len(library) == 2


def test_single_prebuilt_speaker_stays_one_request_per_turn():
    requests = plan_tts_requests(
        [_turn("First.", "narrator", ""), _turn("Second.", "narrator", "speaking slowly")],
        _profile(),
        CONFIG,
    )
    assert len(requests) == 2
    assert "voiceConfig" in requests[0].payload["generationConfig"]["speechConfig"]
    assert requests[0].payload["generationConfig"]["speechConfig"]["voiceConfig"]["voice"] == "Kore"


def test_interactions_payload_uses_annotations():
    config = load_config(ROOT / "src" / "expressive_audiobook" / "default.toml")
    config.tts.api = "interactions"
    requests = plan_tts_requests(
        [
            _turn("You brought the map.", "Guide", "lightly teasing"),
            _turn("I almost left it.", "Wanderer"),
        ],
        _profile(),
        config,
    )
    payload = requests[0].payload
    assert payload["response_format"]["mime_type"] == "audio/l16"
    annotation = payload["input"][0]["content"][0]["annotations"][0]
    assert annotation["type"] == "speech_metadata"
    assert annotation["speaker"] == "Guide"
    assert annotation["style"] == "lightly teasing"
    assert "RATIONALE_SENTINEL" not in json.dumps(payload)


def test_cost_matches_documented_ten_second_equivalent_and_latency_is_parametric():
    cost = estimate_tts_cost(
        audio_seconds=10,
        tts_input_tokens=1000,
        tts_output_price_per_million=9.0,
        tts_input_price_per_million=0.5,
        audio_tokens_per_second=25,
    )
    assert cost["audio_tokens"] == 250
    assert cost["output_cost"] == pytest.approx(0.00225)
    latency = estimate_latency(
        prep_requests=2,
        tts_requests=3,
        seconds_per_prep_request=6,
        seconds_per_tts_request=8,
    )
    assert latency["total_seconds"] == 36


class _Response:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_list_voices_calls_the_catalog_endpoint():
    seen = {}

    def opener(request):
        seen["url"] = request.full_url
        seen["key"] = request.get_header("X-goog-api-key")
        return _Response(b'{"voices":[{"id":"Kore","display_name":"Kore"}]}')

    client = GeminiTtsClient("test-key", CONFIG, opener=opener)
    voices = client.list_voices(search="warm", language_code="en-US")
    assert voices[0]["id"] == "Kore"
    assert "/v1beta/voices" in seen["url"]
    assert "search=warm" in seen["url"]
    assert seen["key"] == "test-key"
    with pytest.raises(RuntimeError):
        GeminiTtsClient(None, CONFIG).list_voices()
