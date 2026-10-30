"""Stage 4: plan TTS requests, synthesize audio, and manage voices.

Multi-speaker requests are used only when a run of turns uses exactly two
prebuilt voices. Every other turn is its own request.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol

from .config import AppConfig, StyleProfile, TtsConfig, VoiceMapping
from .gemini_api import (
    call_with_backoff,
    design_voice_body,
    design_voice_rest,
    design_voice_sdk,
    httpx_send,
    list_voices_sdk,
    synthesize_rest,
)
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


_PREBUILT_SHAPE = re.compile(r"^[A-Z][A-Za-z]+$")


def classify_voice_name(voice: str, declared: str = "") -> str:
    """Return prebuilt, library, design, or replication.

    A single Capitalized word is treated as a prebuilt name and must be in
    the documented list. Hyphenated ids are Extended Voice Library ids.
    ``voice_...`` is a designed voice and ``voicekey_...`` is a replication key.
    """
    name = (voice or "").strip()
    if not name:
        raise ValueError("A voice name or id is required")
    kind = (declared or "").strip().lower()
    if kind in {"prebuilt", "library", "design", "replication"}:
        if kind == "prebuilt" and name not in PREBUILT_VOICES:
            raise ValueError(_unknown_prebuilt(name))
        return kind
    if name.startswith("voicekey_"):
        return "replication"
    if name.startswith("voice_"):
        return "design"
    if name in PREBUILT_VOICES:
        return "prebuilt"
    if _PREBUILT_SHAPE.match(name):
        raise ValueError(_unknown_prebuilt(name))
    return "library"


def _unknown_prebuilt(name: str) -> str:
    known = ", ".join(sorted(PREBUILT_VOICES))
    return (
        f"{name} is not a documented prebuilt voice. "
        f"Prebuilt names: {known}. "
        "Extended Voice Library ids and voice_ design ids are also accepted."
    )


class GeminiTtsClient:
    """Live synthesis, voice catalog, and voice design."""

    def __init__(
        self,
        api_key: str | None,
        config: AppConfig,
        opener: Callable | None = None,
        sender: Callable | None = None,
        sleep=None,
    ):
        self.api_key = api_key
        self.config = config
        self.opener = opener or urllib.request.urlopen
        self.sender = sender
        self.sleep = sleep or time.sleep

    def synthesize(self, request: TtsRequest) -> bytes:
        if not self.api_key and self.sender is None:
            raise RuntimeError("GEMINI_API_KEY is not set")
        if request.api == "interactions":
            url = self.config.tts.endpoints.interactions
            body = dict(request.payload)
        else:
            url = self.config.tts.endpoints.generate_content.format(model=request.model)
            body = {key: value for key, value in request.payload.items() if key != "model"}
        sender = self.sender or httpx_send

        def once() -> bytes:
            return synthesize_rest(
                self.api_key or "",
                url=url,
                body=body,
                sample_rate=request.sample_rate,
                sender=sender,
            )

        return call_with_backoff(
            once,
            max_retries=self.config.tts.max_http_retries,
            initial=self.config.tts.backoff_initial_seconds,
            multiplier=self.config.tts.backoff_multiplier,
            sleep=self.sleep,
        )

    def list_voices(
        self,
        *,
        search: str | None = None,
        language_code: str | None = None,
        page_size: int = 50,
        gender: str | None = None,
        voice_type: str | None = None,
    ) -> list[dict]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        filters = {
            "search": search,
            "language_code": language_code,
            "page_size": page_size,
            "gender": gender,
            "voice_type": voice_type,
        }
        if self.opener is not urllib.request.urlopen:
            return self._list_voices_urllib(filters)
        try:
            return list_voices_sdk(self.api_key, **filters)
        except ImportError:
            return self._list_voices_http(filters)

    def design_voice(
        self,
        description: str,
        *,
        display_name: str = "Designed narrator",
        language_code: str | None = None,
        gender: str | None = None,
    ) -> str:
        """Create a prompted voice and return its ``voice_...`` id."""
        if not description or not description.strip():
            raise ValueError("A voice description is required")
        if not self.api_key and self.sender is None:
            raise RuntimeError("GEMINI_API_KEY is not set")
        body = design_voice_body(
            description.strip(),
            model=self.config.tts.model,
            display_name=display_name,
            language_code=language_code,
            gender=gender,
        )
        url = self.config.tts.endpoints.voices

        def once() -> str:
            if self.sender is not None:
                return design_voice_rest(self.api_key or "", url, body, self.sender)
            try:
                return design_voice_sdk(self.api_key or "", body["voice"])
            except ImportError:
                return design_voice_rest(self.api_key or "", url, body, httpx_send)

        return call_with_backoff(
            once,
            max_retries=self.config.tts.max_http_retries,
            initial=self.config.tts.backoff_initial_seconds,
            multiplier=self.config.tts.backoff_multiplier,
            sleep=self.sleep,
        )

    def _list_voices_urllib(self, filters: dict) -> list[dict]:
        params = _voice_query(filters)
        url = self.config.tts.endpoints.voices + "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={"x-goog-api-key": self.api_key or ""})
        with self.opener(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, list):
            return payload
        return list(payload.get("voices") or [])

    def _list_voices_http(self, filters: dict) -> list[dict]:
        params = _voice_query(filters)
        url = self.config.tts.endpoints.voices + "?" + urllib.parse.urlencode(params)
        status, raw = httpx_send("GET", url, {"x-goog-api-key": self.api_key or ""}, None)
        if status == 429 or 500 <= status <= 599:
            from .gemini_api import RetryableHttpError

            raise RetryableHttpError(status, raw[:400].decode("utf-8", "replace"))
        if status >= 400:
            raise RuntimeError(f"Gemini API error {status}: {raw[:400].decode('utf-8', 'replace')}")
        payload = json.loads(raw.decode("utf-8"))
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


_LONG_PAUSE_RE = re.compile(r"<\s*long pause\s*>", re.IGNORECASE)


_EDGE_PAUSES_RE = re.compile(r"^(?:\s*<\s*short pause\s*>)+\s*|\s*(?:<\s*short pause\s*>\s*)+$", re.IGNORECASE)


def speakable_text(text: str) -> str:
    """Make a turn safe to speak.

    The model sometimes says the word "long" for <long pause>; two short pauses sound the same.
    A pause tag with nothing after it (or before it) is sometimes read aloud as "short pause", and the
    silence between clips already covers that moment, so tags at the edges of a turn are dropped.
    """
    text = _LONG_PAUSE_RE.sub("<short pause> <short pause>", text)
    trimmed = _EDGE_PAUSES_RE.sub("", text).strip()
    return trimmed or text


def _resolve(turn: dict, profile: StyleProfile, config: TtsConfig) -> ResolvedTurn:
    speaker = str(turn.get("speaker") or "narrator")
    mapping = _voice_for(speaker, profile, config)
    return ResolvedTurn(
        text=speakable_text(str(turn.get("text") or "")),
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
    return classify_voice_name(mapping.voice, mapping.voice_type)


def _voice_query(filters: dict) -> dict[str, str | int]:
    params: dict[str, str | int] = {}
    page_size = filters.get("page_size")
    if page_size:
        params["page_size"] = page_size
    if filters.get("search"):
        params["search"] = filters["search"]
    if filters.get("language_code"):
        params["language_code"] = filters["language_code"]
    if filters.get("gender"):
        params["gender"] = filters["gender"]
    if filters.get("voice_type"):
        params["type"] = filters["voice_type"]
    return params


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
