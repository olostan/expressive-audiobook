"""Gemini calls.

Text prep and voice design use the ``google-genai`` SDK. Speech synthesis
uses REST through ``httpx`` because ``GenerateContentConfig`` does not
expose ``response_format``, which is how raw ``AUDIO_L16`` is requested.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Callable

from .audio import concatenate, load_pcm

Sender = Callable[[str, str, dict, dict | None], tuple[int, bytes]]


class RetryableHttpError(RuntimeError):
    def __init__(self, status: int, detail: str):
        self.status = status
        super().__init__(f"Gemini API status {status}: {detail}")


class ModelOutputError(ValueError):
    def __init__(self, message: str, raw: str):
        self.raw = raw
        super().__init__(message)


def call_with_backoff(fn, *, max_retries: int, initial: float, multiplier: float, sleep):
    """Run ``fn`` and retry when it raises ``RetryableHttpError``."""
    attempt = 0
    while True:
        try:
            return fn()
        except RetryableHttpError as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Gemini API still failing after {attempt} retries "
                    f"(last status {exc.status})"
                ) from exc
            sleep(initial * (multiplier ** attempt))
            attempt += 1


def api_headers(api_key: str) -> dict[str, str]:
    return {"x-goog-api-key": api_key, "Content-Type": "application/json"}


def decode_json_response(status: int, raw: bytes) -> dict:
    detail = raw[:500].decode("utf-8", "replace")
    if status == 429 or 500 <= status <= 599:
        raise RetryableHttpError(status, detail)
    if status >= 400:
        raise RuntimeError(f"Gemini API error {status}: {detail}")
    if not raw:
        return {}
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ModelOutputError("Gemini response JSON was not an object", raw.decode("utf-8", "replace"))
    return value


def httpx_send(method: str, url: str, headers: dict, body: dict | None) -> tuple[int, bytes]:
    import httpx

    timeout = httpx.Timeout(180.0, connect=30.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.request(method, url, headers=headers, json=body)
        return response.status_code, response.content


def raise_retryable(exc: Exception) -> None:
    status = getattr(exc, "code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status <= 599):
        raise RetryableHttpError(status, str(exc)) from exc


def parse_model_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ModelOutputError(f"model did not return JSON ({exc})", text) from exc
    if not isinstance(value, dict):
        raise ModelOutputError("model JSON was not an object", text)
    return value


def _sdk_contents(request_contents: list[dict]):
    from google.genai import types

    contents = []
    for item in request_contents:
        parts = []
        for part in item.get("parts") or []:
            if part.get("text"):
                parts.append(types.Part.from_text(text=part["text"]))
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict) and inline.get("data"):
                mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
                parts.append(
                    types.Part.from_bytes(data=base64.b64decode(inline["data"]), mime_type=mime)
                )
        contents.append(types.Content(role=item.get("role") or "user", parts=parts))
    return contents


def generate_prep_sdk(api_key: str, request: dict, *, client=None) -> dict:
    """Call the text model through ``google-genai``."""
    from google.genai import types

    if client is None:
        from google import genai

        client = genai.Client(api_key=api_key)
    generation = request.get("generationConfig") or {}
    try:
        response = client.models.generate_content(
            model=request["model"],
            contents=_sdk_contents(request.get("contents") or []),
            config=types.GenerateContentConfig(
                temperature=generation.get("temperature"),
                response_mime_type=generation.get("responseMimeType") or "application/json",
            ),
        )
    except Exception as exc:
        raise_retryable(exc)
        raise
    return parse_model_json(getattr(response, "text", None) or "")


def generate_prep_rest(
    api_key: str,
    request: dict,
    *,
    endpoint_template: str,
    sender: Sender,
) -> dict:
    url = endpoint_template.format(model=request["model"])
    body = {
        "contents": request.get("contents") or [],
        "generationConfig": request.get("generationConfig") or {},
    }
    status, raw = sender("POST", url, api_headers(api_key), body)
    payload = decode_json_response(status, raw)
    text = _text_from_generate_content(payload)
    return parse_model_json(text)


def generate_prep(
    api_key: str,
    request: dict,
    *,
    endpoint_template: str,
    sender: Sender | None = None,
    sdk=generate_prep_sdk,
) -> dict:
    if sender is not None:
        return generate_prep_rest(
            api_key, request, endpoint_template=endpoint_template, sender=sender
        )
    try:
        return sdk(api_key, request)
    except ImportError:
        return generate_prep_rest(
            api_key, request, endpoint_template=endpoint_template, sender=httpx_send
        )


def _text_from_generate_content(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise ModelOutputError("response had no candidates", json.dumps(payload)[:2000])
    parts = (candidates[0].get("content") or {}).get("parts") or []
    texts = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")]
    if not texts:
        raise ModelOutputError("response had no text", json.dumps(payload)[:2000])
    return "\n".join(texts)


def extract_audio_blobs(payload: dict) -> list[bytes]:
    """Pull base64 audio out of a generateContent or interactions response."""
    blobs: list[bytes] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            inline = node.get("inlineData") or node.get("inline_data")
            if isinstance(inline, dict) and isinstance(inline.get("data"), str):
                blobs.append(base64.b64decode(inline["data"]))
                return
            mime = str(node.get("mimeType") or node.get("mime_type") or "")
            data = node.get("data")
            if mime.startswith("audio") and isinstance(data, str):
                blobs.append(base64.b64decode(data))
                return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    if not blobs:
        raise RuntimeError("Gemini response did not include audio bytes")
    return blobs


def audio_blobs_to_pcm(blobs: list[bytes], sample_rate: int) -> bytes:
    """Accept headerless PCM or RIFF WAV and return 16-bit mono PCM."""
    clips = [load_pcm(blob, expected_rate=sample_rate) for blob in blobs]
    if len(clips) == 1:
        return clips[0].pcm
    return concatenate(clips).pcm


def synthesize_rest(
    api_key: str,
    *,
    url: str,
    body: dict,
    sample_rate: int,
    sender: Sender,
) -> bytes:
    status, raw = sender("POST", url, api_headers(api_key), body)
    payload = decode_json_response(status, raw)
    return audio_blobs_to_pcm(extract_audio_blobs(payload), sample_rate)


def voice_to_dict(voice) -> dict:
    if isinstance(voice, dict):
        data = dict(voice)
    elif hasattr(voice, "model_dump"):
        data = voice.model_dump(exclude_none=True)
    else:
        data = {
            "id": getattr(voice, "id", "") or "",
            "name": getattr(voice, "name", "") or "",
            "display_name": getattr(voice, "display_name", "") or "",
        }
    data.pop("sample_audio", None)
    data.pop("sampleAudio", None)
    return data


def list_voices_sdk(api_key: str, **filters) -> list[dict]:
    from google import genai

    client = genai.Client(api_key=api_key)
    kwargs = {}
    for key in ("search", "language_code", "gender", "page_size", "accent", "voice_type"):
        value = filters.get(key)
        if value not in (None, "", []):
            kwargs["type" if key == "voice_type" else key] = value
    try:
        result = client.voices.list(**kwargs)
    except Exception as exc:
        raise_retryable(exc)
        raise
    if isinstance(result, list):
        records = result
    elif hasattr(result, "voices"):
        records = list(result.voices or [])
    else:
        records = list(result)
    voices = [voice_to_dict(item) for item in records]
    page_size = filters.get("page_size")
    if page_size:
        return voices[: int(page_size)]
    return voices


def design_voice_sdk(api_key: str, voice: dict, *, client=None) -> str:
    if client is None:
        from google import genai

        client = genai.Client(api_key=api_key)
    try:
        created = client.voices.create(store=True, voice=voice)
    except Exception as exc:
        raise_retryable(exc)
        raise
    voice_id = created.get("id") if isinstance(created, dict) else getattr(created, "id", None)
    if not voice_id:
        raise RuntimeError("Voice design response did not include an id")
    return str(voice_id)


def design_voice_body(
    description: str,
    *,
    model: str,
    display_name: str,
    language_code: str | None,
    gender: str | None,
) -> dict:
    voice: dict = {
        "model": model,
        "type": "prompted",
        "display_name": display_name,
        "prompted": {"input": description},
    }
    if language_code:
        voice["language_code"] = language_code
    if gender:
        voice["gender"] = gender
    return {"store": True, "voice": voice}


def design_voice_rest(api_key: str, url: str, body: dict, sender: Sender) -> str:
    status, raw = sender("POST", url, api_headers(api_key), body)
    payload = decode_json_response(status, raw)
    voice_id = payload.get("id") or (payload.get("voice") or {}).get("id")
    if not voice_id:
        raise RuntimeError("Voice design response did not include an id")
    return str(voice_id)
