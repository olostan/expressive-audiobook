"""Stage 4: plan TTS requests and list voices.

`synthesize` is stubbed. `list_voices` performs GET /v1beta/voices.
Multi-speaker requests are used only when a run of turns uses exactly two
prebuilt voices. Every other turn is its own request.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol

from .config import AppConfig, StyleProfile, TtsConfig, VoiceMapping
from .models import TtsRequest

# Prebuilt studio voices named in the public Gemini 3.8 TTS docs.
PREBUILT_VOICES = frozenset(
    {
        "Zephyr",
        "Puck",
        "Charon",
        "Kore",
        "Fenrir",
        "Leda",
        "Orus",
        "Aoede",
        "Callirrhoe",
        "Autonoe",
        "Enceladus",
        "Iapetus",
        "Umbriel",
        "Algieba",
        "Despina",
        "Erinome",
        "Algenib",
        "Rasalgethi",
        "Laomedeia",
        "Achernar",
        "Alnilam",
        "Schedar",
        "Gacrux",
        "Pulcherrima",
        "Achird",
        "Zubenelgenubi",
        "Vindemiatrix",
        "Sadachbia",
        "Sadaltager",
        "Sulafat",
    }
)


class SpeechClient(Protocol):
    def synthesize(self, request: TtsRequest) -> bytes:
        """Return audio bytes for one planned request."""

    def list_voices(self, **filters) -> list[dict]:
        """Return voice records from the voice catalog."""


@dataclass
class ResolvedTurn:
    text: str
    style: str
    speaker: str
    voice: str
    voice_type: str


class GeminiTtsClient:
    """HTTP voice catalog plus a stubbed synthesize call."""

    def __init__(self, api_key: str | None, config: AppConfig, opener: Callable | None = None):
        self.api_key = api_key
        self.config = config
        self.opener = opener or urllib.request.urlopen

    def synthesize(self, request: TtsRequest) -> bytes:
        raise NotImplementedError(
            "TTS synthesis is not implemented. plan_tts_requests() builds the payload."
        )

    def list_voices(
        self,
        *,
        search: str | None = None,
        language_code: str | None = None,
        page_size: int = 50,
    ) -> list[dict]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        params: dict[str, str | int] = {"page_size": page_size}
        if search:
            params["search"] = search
        if language_code:
            params["language_code"] = language_code
        endpoint = self.config.tts.endpoints.voices
        url = endpoint + "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={"x-goog-api-key": self.api_key})
        with self.opener(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, list):
            return payload
        return list(payload.get("voices") or [])


def plan_tts_requests(turns: list[dict], profile: StyleProfile, config: AppConfig) -> list[TtsRequest]:
    """Group a chunk's prepared turns into TTS calls.

    A consecutive run that uses exactly two prebuilt voices becomes one
    multi-speaker request. Any other turn, including a single prebuilt
    narrator and any designed, replicated, or library voice, is its own
    request. Rationale and description fields are not copied.
    """
    resolved = [_resolve(turn, profile, config.tts) for turn in turns]
    groups: list[list[ResolvedTurn]] = []
    index = 0
    while index < len(resolved):
        item = resolved[index]
        if item.voice_type != "prebuilt":
            groups.append([item])
            index += 1
            continue
        window = [item]
        speakers = {item.speaker}
        cursor = index + 1
        while cursor < len(resolved):
            nxt = resolved[cursor]
            if nxt.voice_type != "prebuilt":
                break
            trial = speakers | {nxt.speaker}
            if len(trial) > 2:
                break
            window.append(nxt)
            speakers = trial
            cursor += 1
        if len(speakers) == 2:
            groups.append(window)
            index = cursor
        else:
            groups.append([item])
            index += 1
    return [_build_request(group, config) for group in groups]


def estimate_tts_cost(
    *,
    audio_seconds: float,
    tts_input_tokens: int,
    tts_output_price_per_million: float,
    tts_input_price_per_million: float,
    audio_tokens_per_second: float = 25.0,
) -> dict:
    """Parametric TTS cost. Audio tokens default to the documented 25 per second."""
    audio_tokens = audio_seconds * audio_tokens_per_second
    output_cost = audio_tokens / 1_000_000 * tts_output_price_per_million
    input_cost = tts_input_tokens / 1_000_000 * tts_input_price_per_million
    return {
        "audio_seconds": audio_seconds,
        "audio_tokens": audio_tokens,
        "output_cost": output_cost,
        "input_cost": input_cost,
        "total_cost": output_cost + input_cost,
    }


def estimate_latency(
    *,
    prep_requests: int,
    tts_requests: int,
    seconds_per_prep_request: float,
    seconds_per_tts_request: float,
) -> dict:
    """Parametric latency. Per-request seconds are not an API guarantee."""
    prep = prep_requests * seconds_per_prep_request
    tts = tts_requests * seconds_per_tts_request
    return {"prep_seconds": prep, "tts_seconds": tts, "total_seconds": prep + tts}


def _resolve(turn: dict, profile: StyleProfile, config: TtsConfig) -> ResolvedTurn:
    speaker = str(turn.get("speaker") or "narrator")
    mapping = _voice_for(speaker, profile, config)
    return ResolvedTurn(
        text=str(turn.get("text") or ""),
        style=str(turn.get("style") or ""),
        speaker=speaker,
        voice=mapping.voice,
        voice_type=_classify(mapping),
    )


def _voice_for(speaker: str, profile: StyleProfile, config: TtsConfig) -> VoiceMapping:
    voices = profile.character_voices
    if speaker in voices:
        return voices[speaker]
    folded = {name.lower(): mapping for name, mapping in voices.items()}
    if speaker.lower() in folded:
        return folded[speaker.lower()]
    if "narrator" in folded:
        return folded["narrator"]
    return VoiceMapping(voice=config.default_voice, voice_type=config.default_voice_type)


def _classify(mapping: VoiceMapping) -> str:
    declared = (mapping.voice_type or "").strip().lower()
    if declared in {"prebuilt", "library", "design", "replication"}:
        return declared
    voice = mapping.voice
    if voice.startswith("voicekey_"):
        return "replication"
    if voice.startswith("voice_"):
        return "design"
    if voice in PREBUILT_VOICES:
        return "prebuilt"
    return "library"


def _build_request(group: list[ResolvedTurn], config: AppConfig) -> TtsRequest:
    speakers = []
    seen = set()
    for item in group:
        if item.speaker in seen:
            continue
        seen.add(item.speaker)
        speakers.append({"speaker": item.speaker, "voice": item.voice, "voice_type": item.voice_type})
    turns = [{"text": item.text, "style": item.style, "speaker": item.speaker} for item in group]
    api = config.tts.api
    if api == "interactions":
        payload = _interactions_payload(turns, speakers, config)
    else:
        payload = _generate_content_payload(turns, speakers, config)
    return TtsRequest(
        api=api if api == "interactions" else "generate_content",
        model=config.tts.model,
        audio_mime=config.tts.audio_mime,
        sample_rate=config.tts.sample_rate,
        speakers=speakers,
        turns=turns,
        payload=payload,
    )


def _generate_content_payload(turns: list[dict], speakers: list[dict], config: AppConfig) -> dict:
    fields = config.tts.fields
    parts = []
    multi = len(speakers) > 1
    for turn in turns:
        metadata = {fields.style: turn["style"]}
        if multi:
            metadata[fields.speaker] = turn["speaker"]
        parts.append({"text": turn["text"], fields.speech_metadata: metadata})
    if multi:
        speech = {
            fields.multi_speaker_voice_config: {
                fields.speaker_voice_configs: [
                    {
                        fields.speaker: item["speaker"],
                        fields.voice_config: {
                            fields.prebuilt_voice_config: {fields.voice_name: item["voice"]}
                        },
                    }
                    for item in speakers
                ]
            }
        }
    else:
        speech = {fields.voice_config: {fields.voice: speakers[0]["voice"]}}
    generation = {
        fields.response_modalities: ["AUDIO"],
        fields.response_format: {
            "audio": {
                fields.mime_type: config.tts.audio_mime,
                fields.sample_rate: config.tts.sample_rate,
            }
        },
        fields.speech_config: speech,
    }
    return {
        "model": config.tts.model,
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation,
    }


def _interactions_payload(turns: list[dict], speakers: list[dict], config: AppConfig) -> dict:
    content = []
    for turn in turns:
        annotation = {"type": config.tts.fields.annotation_type, "style": turn["style"]}
        if len(speakers) > 1:
            annotation["speaker"] = turn["speaker"]
        content.append({"type": "text", "text": turn["text"], "annotations": [annotation]})
    if len(speakers) > 1:
        speech_config = {
            "speakers": [{"speaker": item["speaker"], "voice": item["voice"]} for item in speakers]
        }
    else:
        speech_config = [{"voice": speakers[0]["voice"]}]
    return {
        "model": config.tts.model,
        "input": [{"type": "user_input", "content": content}],
        "response_format": {
            "type": "audio",
            "mime_type": config.tts.interactions_audio_mime,
            "sample_rate": config.tts.sample_rate,
        },
        "generation_config": {"speech_config": speech_config},
    }
