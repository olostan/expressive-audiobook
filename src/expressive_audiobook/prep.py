"""Stage 3: prompt, multimodal request, validator, and the live text call."""

from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Protocol

from PIL import Image
from jsonschema import Draft202012Validator

from .config import AppConfig, PrepConfig, StyleProfile
from .gemini_api import ModelOutputError, call_with_backoff, generate_prep
from .models import Chunk, to_jsonable
from .tuning import delivery_guidance
from .textutil import count_sentences, estimate_seconds, spoken_source_words, split_sentences, word_count

_TAG_RE = re.compile(r"<([^<>]+)>")
_BACKCHANNEL_RE = re.compile(r"\|([^|]*)\|")
_BRACKET_RE = re.compile(r"\[[^\[\]]+\]")
_STAR_RE = re.compile(r"(?<!\*)\*([^*]{1,60})\*(?!\*)")
_PAREN_STAGE_RE = re.compile(
    r"\((?:softly|quietly|loudly|whisper(?:ing|s)?|shouting|pause|pauses|beat|"
    r"laughs?|sighs?|crying|sarcastically|angrily|gently|slowly|quickly|aside)\)",
    re.IGNORECASE,
)
_LATEX_RE = re.compile(r"\$\$[\s\S]+?\$\$|\$[^$\n]+\$|\\[a-zA-Z]+|\\\(|\\\)|\\\[|\\\]")
_MATH_SYMBOLS = set("∑∫√≤≥≠∞±∈∉⊂⊆∀∃∂∇≈≡∝×÷")
_ANCHOR_LEAK_RE = re.compile(r"\[\[")
_ART_RE = re.compile(r"-{3,}|_{3,}|={3,}|~{3,}|\.{4,}")
_NON_VOCAL = (
    "applause",
    "clap",
    "clapping",
    "thunder",
    "thud",
    "bang",
    "music",
    "footsteps",
    "door",
    "crash",
    "ringtone",
    "gunshot",
)


class PrepValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


class TextPrepClient(Protocol):
    def generate(self, request: dict) -> dict:
        """Return parsed JSON prep output for one multimodal request."""


class ChunkPrepFailed(RuntimeError):
    def __init__(self, chunk_id: str, errors: list[str]):
        self.chunk_id = chunk_id
        self.errors = errors
        detail = "; ".join(errors)
        super().__init__(f"Text prep failed for chunk {chunk_id}: {detail}")


class GeminiTextPrepClient:
    """Live text-prep client. HTTP 429 and 5xx responses are retried with backoff."""

    def __init__(self, api_key: str | None, config: AppConfig, sender=None, sleep=None):
        self.api_key = api_key
        self.config = config
        self.sender = sender
        self.sleep = sleep or time.sleep

    def generate(self, request: dict) -> dict:
        if not self.api_key and self.sender is None:
            raise RuntimeError("GEMINI_API_KEY is not set")

        def once() -> dict:
            return generate_prep(
                self.api_key or "",
                request,
                endpoint_template=self.config.tts.endpoints.generate_content,
                sender=self.sender,
            )

        return call_with_backoff(
            once,
            max_retries=self.config.prep.max_http_retries,
            initial=self.config.prep.backoff_initial_seconds,
            multiplier=self.config.prep.backoff_multiplier,
            sleep=self.sleep,
        )


def schema_path() -> Path:
    return Path(__file__).resolve().parent / "schemas" / "prep_output.json"


def load_prep_schema() -> dict:
    return json.loads(schema_path().read_text(encoding="utf-8"))


def schema_errors(output: dict) -> list[str]:
    validator = Draft202012Validator(load_prep_schema())
    found = []
    for error in sorted(validator.iter_errors(output), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.path) or "(root)"
        found.append(f"{location}: {error.message}")
    return found


def feedback_request(request: dict, errors: list[str], previous) -> dict:
    """Return a copy of ``request`` with validator errors appended for a retry."""
    lines = "\n".join(f"- {error}" for error in errors)
    previous_text = previous if isinstance(previous, str) else json.dumps(previous, ensure_ascii=False)
    if len(previous_text) > 6000:
        previous_text = previous_text[:6000] + "\n…"
    note = (
        "The previous JSON failed validation. Fix every error and return JSON only.\n"
        f"{lines}\n"
        f"Previous JSON:\n{previous_text}"
    )
    updated = dict(request)
    updated["contents"] = list(request.get("contents") or []) + [
        {"role": "user", "parts": [{"text": note}]}
    ]
    return updated


def prepare_chunk_output(
    chunk: Chunk,
    profile: StyleProfile,
    config: AppConfig,
    client: TextPrepClient,
    *,
    previous_text: str = "",
    following_text: str = "",
    image_root: Path | None = None,
    request: dict | None = None,
) -> dict:
    """Call the text model, validate, and retry with the errors fed back."""
    if request is None:
        request = build_prep_request(
            chunk,
            profile,
            config,
            previous_text=previous_text,
            following_text=following_text,
            image_root=image_root,
        )
    attempts = config.prep.max_validation_retries + 1
    errors: list[str] = []
    previous = None
    use_feedback = True
    for attempt in range(attempts):
        payload = request if attempt == 0 or not use_feedback else feedback_request(request, errors, previous)
        try:
            output = client.generate(payload)
        except ModelOutputError as exc:
            errors = [str(exc)]
            previous = exc.raw
            continue
        except Exception as exc:
            # The API rejected the retry-with-feedback form of the request. Ask again with the
            # original request instead of losing the chunk.
            code = getattr(exc, "code", None)
            if attempt > 0 and use_feedback and isinstance(code, int) and 400 <= code < 500 and code != 429:
                use_feedback = False
                errors = [f"the retry request was rejected by the API ({code}); asking again without feedback"]
                continue
            raise
        errors = schema_errors(output)
        if not errors:
            errors = validate_prep_output(output, chunk, config)
        if not errors:
            flat = flat_delivery_errors(output, config)
            if not flat or attempt == attempts - 1:
                return output
            errors = flat
        previous = output
    raise ChunkPrepFailed(chunk.id, errors or ["no prep output"])


def prompt_template_path() -> Path:
    return Path(__file__).resolve().parent / "prompts" / "text_prep.md"


def load_prompt_template() -> str:
    return prompt_template_path().read_text(encoding="utf-8")


def render_prompt(
    chunk: Chunk,
    profile: StyleProfile,
    config: PrepConfig,
    *,
    tts_model: str,
    wpm: float,
    hard_cap_seconds: float,
    previous_text: str = "",
    following_text: str = "",
    omitted_images: str = "",
    template: str | None = None,
) -> str:
    body = template if template is not None else load_prompt_template()
    values = {
        "tts_model": tts_model,
        "language": config.language,
        "hard_cap_seconds": str(hard_cap_seconds),
        "wpm": str(wpm),
        "allowed_tags": ", ".join(config.allowed_tags),
        "backchannels": ", ".join(config.backchannels),
        "style_examples": ", ".join(f'"{item}"' for item in config.style_examples),
        "engine_notes": config.engine_notes,
        "style_profile": profile.render(),
        "delivery_guidance": delivery_guidance(config.expressiveness, config.pace, config.pause_density),
        "prompt_version": config.prompt_version,
        "chunk_id": chunk.id,
        "chunk_json": json.dumps(to_jsonable(chunk), ensure_ascii=False, indent=2),
        "previous_text": previous_text.strip() or "(none)",
        "following_text": following_text.strip() or "(none)",
        "omitted_images": omitted_images.strip() or "(none)",
    }
    rendered = body
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


def build_prep_request(
    chunk: Chunk,
    profile: StyleProfile,
    config: AppConfig,
    *,
    previous: Chunk | None = None,
    following: Chunk | None = None,
    previous_text: str | None = None,
    following_text: str | None = None,
    image_root: Path | None = None,
    template: str | None = None,
) -> dict:
    """Build a GenerateContent request with text plus images in reading order.

    Images are downscaled to `max_image_edge_px`. Only the first
    `max_images_per_request` are inlined; the rest are named in the prompt
    so they are not silently dropped.
    """
    images = _image_elements(chunk)
    limit = config.prep.max_images_per_request
    inlined = images[:limit]
    omitted = images[limit:]
    omitted_text = "\n".join(
        f"- {item.id} ({item.kind}) caption: {item.caption or item.text or '(no caption)'}"
        for item in omitted
    )
    prompt = render_prompt(
        chunk,
        profile,
        config.prep,
        tts_model=config.tts.model,
        wpm=config.chunking.wpm,
        hard_cap_seconds=config.chunking.hard_cap_seconds,
        previous_text=previous_text if previous_text is not None else (previous.text if previous else ""),
        following_text=following_text if following_text is not None else (following.text if following else ""),
        omitted_images=omitted_text,
        template=template,
    )
    fields = config.prep.fields
    parts: list[dict] = [{"text": prompt}]
    for item in inlined:
        intro = f"Embedded image for element {item.id} ({item.kind})."
        if item.caption:
            intro += f" Caption: {item.caption}"
        elif item.latex or item.text:
            intro += f" Source text: {item.latex or item.text}"
        parts.append({"text": intro})
        png = _load_png(item.image_path or "", image_root)
        png = downscale_png(png, config.prep.max_image_edge_px)
        parts.append(
            {
                fields.inline_data: {
                    fields.mime_type: "image/png",
                    "data": base64.b64encode(png).decode("ascii"),
                }
            }
        )
    return {
        "model": config.prep.model,
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": config.prep.temperature,
        },
        "prompt_version": config.prep.prompt_version,
    }


def downscale_png(data: bytes, max_edge: int) -> bytes:
    if max_edge <= 0:
        return data
    image = Image.open(io.BytesIO(data))
    image.load()
    if max(image.size) <= max_edge:
        return data
    image = image.convert("RGB")
    image.thumbnail((max_edge, max_edge))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def validate_prep_output(output: dict, chunk: Chunk, config: AppConfig) -> list[str]:
    """Return a list of problems. Empty means the output is acceptable."""
    errors: list[str] = []
    if output.get("chunk_id") != chunk.id:
        errors.append(f"chunk_id {output.get('chunk_id')!r} does not match {chunk.id}")
    if output.get("prompt_version") != config.prep.prompt_version:
        errors.append("prompt_version does not match the active config")
    turns = output.get("turns")
    if not isinstance(turns, list) or not turns:
        return errors + ["turns must be a non-empty list"]
    hard_words = config.chunking.hard_words
    speakers = {name.casefold() for name in chunk.speakers} | {"narrator"}
    descriptions = []
    for index, turn in enumerate(turns):
        errors.extend(_validate_turn(turn, index, config, hard_words, speakers))
        for item in turn.get("inserted_descriptions") or []:
            descriptions.append((index, turn, item))
    errors.extend(_validate_descriptions(descriptions, turns, chunk, config))
    return errors


def flat_delivery_errors(output: dict, config: AppConfig) -> list[str]:
    """Soft check: a long passage with no tags and no style is under-performed."""
    floor = config.prep.min_tags_per_100_words
    if floor <= 0:
        return []
    turns = output.get("turns") or []
    words = sum(spoken_source_words(turn.get("text") or "") for turn in turns)
    if words < 40:
        return []
    marks = sum(len(_TAG_RE.findall(turn.get("text") or "")) for turn in turns)
    marks += sum(len(_BACKCHANNEL_RE.findall(turn.get("text") or "")) for turn in turns)
    styled = sum(1 for turn in turns if (turn.get("style") or "").strip())
    needed = int(words * floor / 100)
    if marks >= needed and (styled or marks):
        return []
    return [
        f"the delivery is too flat: {marks} inline tags and {styled} styled turns for {words} words "
        f"(expected at least {max(needed, 1)} tags). Add fitting inline tags, pauses, and a style. "
        "Keep the transcript wording otherwise unchanged."
    ]


def assert_valid(output: dict, chunk: Chunk, config: AppConfig) -> None:
    errors = validate_prep_output(output, chunk, config)
    if errors:
        raise PrepValidationError(errors)


def tts_turns_from_output(output: dict) -> list[dict]:
    """Turns safe to send to TTS: text, style, and speaker only."""
    safe = []
    for turn in output.get("turns") or []:
        safe.append(
            {
                "text": turn.get("text", ""),
                "style": turn.get("style", ""),
                "speaker": turn.get("speaker", "narrator"),
            }
        )
    return safe


def split_turn_to_fit(turn: dict, *, max_words: int) -> list[dict]:
    """Split one prepared turn on sentence boundaries until each fits.

    Style and speaker are copied. An inserted description stays on the
    sub-turn whose text contains its spoken_text, or on the first sub-turn
    when the spoken text is not in the transcript yet. Nothing is dropped.
    """
    text = turn.get("text") or ""
    if spoken_source_words(text) <= max_words:
        return [dict(turn)]
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [dict(turn)]
    slices: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        words = spoken_source_words(sentence)
        if current and current_words + words > max_words:
            slices.append(" ".join(current))
            current = [sentence]
            current_words = words
        else:
            current.append(sentence)
            current_words += words
    if current:
        slices.append(" ".join(current))
    descriptions = list(turn.get("inserted_descriptions") or [])
    buckets: list[list[dict]] = [[] for _ in slices]
    for item in descriptions:
        spoken = (item.get("spoken_text") or "").strip()
        placed = False
        if spoken:
            for index, slice_text in enumerate(slices):
                if spoken in slice_text:
                    buckets[index].append(item)
                    placed = True
                    break
        if not placed:
            buckets[0].append(item)
    results = []
    for index, slice_text in enumerate(slices):
        results.append(
            {
                "text": slice_text,
                "style": turn.get("style", ""),
                "speaker": turn.get("speaker", "narrator"),
                "rationale": turn.get("rationale", ""),
                "block_ids": list(turn.get("block_ids") or []),
                "inserted_descriptions": buckets[index],
            }
        )
    return results


def _label_prefix(text: str, speakers: set[str]) -> bool:
    head = text.lstrip().split(":", 1)
    return len(head) == 2 and head[0].strip().casefold() in speakers


def _validate_turn(
    turn: dict, index: int, config: AppConfig, hard_words: int, known_speakers: set[str] | None = None
) -> list[str]:
    errors = []
    prefix = f"turn {index}"
    if not isinstance(turn, dict):
        return [f"{prefix} is not an object"]
    text = turn.get("text")
    style = turn.get("style")
    speaker = turn.get("speaker")
    if not isinstance(text, str):
        errors.append(f"{prefix} text must be a string")
        text = ""
    if not isinstance(style, str):
        errors.append(f"{prefix} style must be a string")
        style = ""
    if not isinstance(speaker, str) or not speaker.strip():
        errors.append(f"{prefix} speaker must be a non-empty string")
    if "rationale" not in turn:
        errors.append(f"{prefix} is missing rationale")
    if isinstance(style, str) and len(style) > config.prep.max_style_chars:
        errors.append(f"{prefix} style is longer than {config.prep.max_style_chars} characters")
    if isinstance(style, str) and text and _style_quotes_transcript(style, text):
        errors.append(f"{prefix} style contains transcript text")
    errors.extend(_transcript_errors(text or "", config.prep, prefix))
    if known_speakers and _label_prefix(text or "", known_speakers):
        errors.append(f"{prefix} starts with a speaker label; the voice says who is talking, so drop it")
    words = spoken_source_words(text or "")
    if words > hard_words:
        errors.append(
            f"{prefix} is {estimate_seconds(words, config.chunking.wpm):.1f}s, "
            f"over the {config.chunking.hard_cap_seconds}s cap"
        )
    return errors


def _transcript_errors(text: str, config: PrepConfig, prefix: str) -> list[str]:
    errors = []
    allowed = set(config.allowed_tags)
    for match in _TAG_RE.finditer(text):
        tag = f"<{match.group(1)}>"
        if tag not in allowed:
            errors.append(f"{prefix} has an angle-bracket tag that is not allowed: {tag}")
    allowed_channels = {item.strip("|") for item in config.backchannels}
    for match in _BACKCHANNEL_RE.finditer(text):
        if match.group(1).strip() not in allowed_channels:
            errors.append(f"{prefix} has a pipe backchannel that is not allowed: |{match.group(1)}|")
    if _BRACKET_RE.search(text):
        errors.append(f"{prefix} has square-bracket stage direction or marker text")
    if _STAR_RE.search(text):
        errors.append(f"{prefix} has asterisk stage direction")
    if _PAREN_STAGE_RE.search(text):
        errors.append(f"{prefix} has a parenthetical stage direction")
    if _LATEX_RE.search(text) or any(symbol in text for symbol in _MATH_SYMBOLS):
        errors.append(f"{prefix} still contains raw math or LaTeX")
    if _ANCHOR_LEAK_RE.search(text):
        errors.append(f"{prefix} still contains an anchor marker")
    if _ART_RE.search(text):
        errors.append(
            f"{prefix} contains a run of dashes or other typographic art; "
            "say it in words (for example 'five dashes') instead"
        )
    lowered = text.lower()
    for word in _NON_VOCAL:
        if f"<{word}" in lowered or f"[{word}" in lowered or f"*{word}" in lowered:
            errors.append(f"{prefix} contains a non-vocal sound effect ({word})")
    return errors


def _style_quotes_transcript(style: str, text: str) -> bool:
    if style.strip() and style.strip() == text.strip():
        return True
    for sentence in split_sentences(text):
        snippet = sentence.strip()
        if len(snippet) >= 24 and snippet in style:
            return True
    return False


def _validate_descriptions(descriptions, turns, chunk: Chunk, config: AppConfig) -> list[str]:
    del turns
    errors = []
    required = [item for item in chunk.associated if item.speak]
    seen: dict[str, tuple] = {}
    for index, turn, item in descriptions:
        if not isinstance(item, dict):
            errors.append(f"turn {index} has a description that is not an object")
            continue
        element_id = item.get("element_id")
        if not element_id:
            errors.append(f"turn {index} has a description without element_id")
            continue
        if element_id in seen:
            errors.append(f"element {element_id} is described more than once")
        seen[element_id] = (index, turn, item)
        errors.extend(_one_description(index, turn, item, chunk, config))
    for element in required:
        if element.id not in seen:
            errors.append(f"associated element {element.id} was dropped")
            continue
        index, turn, item = seen[element.id]
        block_ids = turn.get("block_ids") or []
        if block_ids and element.anchor and element.anchor not in block_ids:
            errors.append(
                f"element {element.id} is on turn {index}, which does not cover anchor {element.anchor}"
            )
    return errors


def _one_description(index: int, turn: dict, item: dict, chunk: Chunk, config: AppConfig) -> list[str]:
    del turn
    errors = []
    prefix = f"turn {index} description {item.get('element_id')}"
    kind = item.get("kind")
    if kind not in {"figure", "formula", "footnote", "citation"}:
        errors.append(f"{prefix} has kind {kind!r}")
    decision = item.get("decision")
    if decision not in {"read-full", "mention", "skip-with-reason"}:
        errors.append(f"{prefix} has decision {decision!r}")
        return errors
    spoken = item.get("spoken_text") or ""
    if decision == "skip-with-reason":
        if not (item.get("reason") or "").strip():
            errors.append(f"{prefix} skip is missing a reason")
        if spoken.strip():
            errors.append(f"{prefix} skip should not include spoken text")
        return errors
    if _LATEX_RE.search(spoken) or any(symbol in spoken for symbol in _MATH_SYMBOLS):
        errors.append(f"{prefix} spoken text still contains raw math or LaTeX")
    sentences = count_sentences(spoken)
    words = word_count(spoken)
    source = _source_element(chunk, item.get("element_id"))
    if kind in {"figure", "formula"}:
        if sentences > config.prep.max_description_sentences:
            errors.append(f"{prefix} is {sentences} sentences, over the description cap")
        if words > config.prep.max_description_words:
            errors.append(f"{prefix} is {words} words, over the description cap")
    elif kind == "footnote" and decision == "mention" and words > config.prep.footnote_mention_max_words:
        errors.append(f"{prefix} mention is longer than the footnote mention cap")
    elif kind == "footnote" and decision == "read-full" and source is not None:
        limit = word_count(source.text) + config.prep.footnote_read_full_extra_words
        if words > max(limit, 1):
            errors.append(f"{prefix} read-full is longer than the source note plus a short lead-in")
    elif kind == "citation" and words > config.prep.citation_max_words:
        errors.append(f"{prefix} citation is longer than the citation cap")
    return errors


def _source_element(chunk: Chunk, element_id: str):
    for item in chunk.associated:
        if item.id == element_id:
            return item
    return None


def _image_elements(chunk: Chunk) -> list:
    items = [
        item
        for item in chunk.associated
        if item.speak and item.image_path and item.kind in {"figure", "formula"}
    ]
    return sorted(items, key=lambda item: (item.stream_index, item.id))


def _load_png(image_path: str, image_root: Path | None) -> bytes:
    path = Path(image_path)
    if not path.is_absolute():
        root = image_root or Path(".")
        path = root / image_path
    return path.read_bytes()
