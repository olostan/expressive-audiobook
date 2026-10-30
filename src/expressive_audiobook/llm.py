"""A text model behind any OpenAI-compatible chat endpoint (Ollama, LM Studio, llama.cpp, vLLM, hosted APIs).

Prep, style detection and the sound planners all call ``client.generate(request)`` with a request in
the shape the Gemini client takes (``contents`` -> ``parts``, ``generationConfig``). This client
translates that shape into a chat-completions call and returns the parsed JSON object, so no caller
needs to know which model answered.
"""

from __future__ import annotations

import json
import re
import time

from .config import AppConfig, env_value
from .gemini_api import ModelOutputError, RetryableHttpError, call_with_backoff, parse_model_json

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def chat_messages(request: dict, *, vision: bool) -> list[dict]:
    """Turn ``contents`` into chat messages. Images become data URLs when the model can see them."""
    messages: list[dict] = []
    for content in request.get("contents") or []:
        role = "assistant" if content.get("role") == "model" else "user"
        blocks: list[dict] = []
        for part in content.get("parts") or []:
            if part.get("text"):
                blocks.append({"type": "text", "text": part["text"]})
                continue
            blob = part.get("inlineData") or part.get("inline_data")
            if blob and vision:
                mime = blob.get("mimeType") or blob.get("mime_type") or "image/png"
                blocks.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{blob['data']}"}})
        if not blocks:
            continue
        if all(block["type"] == "text" for block in blocks):
            messages.append({"role": role, "content": "\n\n".join(block["text"] for block in blocks)})
        else:
            messages.append({"role": role, "content": blocks})
    return messages


def reply_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ModelOutputError("response had no choices", json.dumps(payload)[:2000])
    text = ((choices[0].get("message") or {}).get("content")) or ""
    text = _THINK.sub("", text).strip()
    if not text:
        raise ModelOutputError("response had no text", json.dumps(payload)[:2000])
    return text


class OpenAICompatTextClient:
    """``generate(request) -> dict`` against ``{base_url}/chat/completions``."""

    def __init__(self, config: AppConfig, sender=None, sleep=None):
        self.config = config
        self.settings = config.text
        self.sender = sender
        self.sleep = sleep or time.sleep

    def generate(self, request: dict) -> dict:
        settings = self.settings
        if not settings.model:
            raise RuntimeError("text.model is not set. Name the model your server runs, for example text.model = \"qwen3:14b\"")
        generation = request.get("generationConfig") or {}
        body: dict = {
            "model": settings.model,
            "messages": chat_messages(request, vision=settings.vision),
            "stream": False,
        }
        if "temperature" in generation:
            body["temperature"] = generation["temperature"]
        if settings.json_mode == "json_object" and generation.get("responseMimeType") == "application/json":
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        key = env_value(settings.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        url = settings.base_url.rstrip("/") + "/chat/completions"
        send = self.sender or (lambda m, u, h, b: _send(m, u, h, b, settings.timeout_seconds))

        def once() -> dict:
            status, raw = send("POST", url, headers, body)
            detail = raw[:500].decode("utf-8", "replace")
            if status == 429 or 500 <= status <= 599:
                raise RetryableHttpError(status, detail)
            if status >= 400:
                raise RuntimeError(f"Text model error {status} from {url}: {detail}")
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ModelOutputError(f"text model reply was not JSON ({exc})", detail) from exc
            return parse_model_json(reply_text(payload))

        return call_with_backoff(
            once,
            max_retries=settings.max_http_retries,
            initial=settings.backoff_initial_seconds,
            multiplier=settings.backoff_multiplier,
            sleep=self.sleep,
        )


def _send(method: str, url: str, headers: dict, body: dict | None, timeout_seconds: float) -> tuple[int, bytes]:
    import httpx

    timeout = httpx.Timeout(timeout_seconds, connect=10.0)
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.request(method, url, headers=headers, json=body)
            return response.status_code, response.content
    except httpx.ConnectError as exc:
        raise RuntimeError(f"Could not reach the text model at {url}. Is the server running? ({exc})") from exc
