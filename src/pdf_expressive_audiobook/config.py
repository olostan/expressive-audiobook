"""Load TOML config, style profiles, and GEMINI_API_KEY."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import tomllib
import yaml

from .textutil import max_words_under, min_words_at


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    cwd = Path.cwd() / "config" / "default.toml"
    if cwd.exists():
        return cwd
    packaged = repo_root() / "config" / "default.toml"
    return packaged


@dataclass
class PathsConfig:
    artifacts: str = "artifacts"
    ingest: str = "artifacts/ingest"
    chunks: str = "artifacts/chunks"
    prep: str = "artifacts/prep"
    tts: str = "artifacts/tts"
    audio: str = "artifacts/audio"


@dataclass
class IngestConfig:
    ocr_enabled: bool = False
    header_footer_band_ratio: float = 0.09
    footnote_region_ratio: float = 0.60
    image_dpi: int = 144
    vector_cluster_gap: float = 18.0
    min_vector_area: float = 800.0
    full_page_image_ratio: float = 0.70
    skip_contents: bool = True


@dataclass
class ChunkingConfig:
    wpm: float = 150
    target_min_seconds: float = 30
    target_max_seconds: float = 50
    hard_cap_seconds: float = 60
    figure_description_sentences: int = 2
    words_per_spoken_sentence: int = 18
    formula_description_words: int = 40
    footnote_lead_in_words: int = 2

    @property
    def min_words(self) -> int:
        return min_words_at(self.target_min_seconds, self.wpm)

    @property
    def target_words(self) -> int:
        return max_words_under(self.target_max_seconds, self.wpm)

    @property
    def hard_words(self) -> int:
        return max_words_under(self.hard_cap_seconds, self.wpm)


@dataclass
class PrepFields:
    inline_data: str = "inlineData"
    mime_type: str = "mimeType"


@dataclass
class PrepConfig:
    model: str = "gemini-3.8-flash"
    prompt_version: str = "prep-v1"
    language: str = "en"
    max_images_per_request: int = 4
    max_image_edge_px: int = 1024
    max_style_chars: int = 400
    max_description_sentences: int = 3
    max_description_words: int = 80
    footnote_mention_max_words: int = 40
    citation_max_words: int = 40
    footnote_read_full_extra_words: int = 16
    neighbor_chunks: int = 1
    temperature: float = 0.4
    max_validation_retries: int = 2
    max_http_retries: int = 5
    backoff_initial_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    concurrency: int = 4
    # Delivery tuning. A project's tuning block sets these; see tuning.py.
    expressiveness: float = 0.5
    pace: float = 0.5
    pause_density: float = 0.4
    min_tags_per_100_words: float = 0.0
    allowed_tags: list[str] = field(default_factory=list)
    backchannels: list[str] = field(default_factory=lambda: ["|mhm|", "|yeah|"])
    style_examples: list[str] = field(default_factory=list)
    fields: PrepFields = field(default_factory=PrepFields)


@dataclass
class TtsFields:
    response_modalities: str = "responseModalities"
    response_format: str = "responseFormat"
    speech_config: str = "speechConfig"
    voice_config: str = "voiceConfig"
    voice: str = "voice"
    multi_speaker_voice_config: str = "multiSpeakerVoiceConfig"
    speaker_voice_configs: str = "speakerVoiceConfigs"
    prebuilt_voice_config: str = "prebuiltVoiceConfig"
    voice_name: str = "voiceName"
    speech_metadata: str = "speech_metadata"
    style: str = "style"
    speaker: str = "speaker"
    mime_type: str = "mimeType"
    sample_rate: str = "sampleRate"
    annotation_type: str = "speech_metadata"
    response_mime_type: str = "mime_type"


@dataclass
class TtsEndpoints:
    generate_content: str = (
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )
    interactions: str = "https://generativelanguage.googleapis.com/v1beta/interactions"
    voices: str = "https://generativelanguage.googleapis.com/v1beta/voices"


@dataclass
class TtsConfig:
    api: str = "generate_content"
    model: str = "gemini-3.8-flash-tts"
    lite_model: str = "gemini-3.8-flash-lite-tts"
    language: str = "en"
    default_voice: str = "Schedar"
    default_voice_type: str = "prebuilt"
    audio_mime: str = "AUDIO_L16"
    interactions_audio_mime: str = "audio/l16"
    sample_rate: int = 24000
    channels: int = 1
    sample_width_bits: int = 16
    max_http_retries: int = 5
    backoff_initial_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    concurrency: int = 2
    endpoints: TtsEndpoints = field(default_factory=TtsEndpoints)
    fields: TtsFields = field(default_factory=TtsFields)


@dataclass
class AssembleConfig:
    gap_ms: float = 200
    crossfade_ms: float = 12
    chapter_tail_ms: float = 0
    write_full_book: bool = True
    format: str = "wav"  # also write mp3 or m4a next to each WAV when ffmpeg is installed
    keep_wav: bool = True  # projects default to False: the WAV is removed once the MP3/M4A exists
    mp3_quality: int = 4  # libmp3lame VBR quality, 0 best to 9 smallest; 4 is about 130 kbps


@dataclass
class CostConfig:
    tts_input_price_per_million: float = 0.50
    tts_output_price_per_million: float = 9.00
    audio_tokens_per_second: float = 25
    seconds_per_tts_request: float = 8.0
    seconds_per_prep_request: float = 6.0
    tokens_per_word: float = 1.3


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    prep: PrepConfig = field(default_factory=PrepConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    assemble: AssembleConfig = field(default_factory=AssembleConfig)
    cost: CostConfig = field(default_factory=CostConfig)


@dataclass
class VoiceMapping:
    voice: str
    voice_type: str = ""


@dataclass
class LexiconEntry:
    word: str
    ipa: str
    note: str = ""


@dataclass
class StyleProfile:
    id: str = "default"
    narrator_persona: str = ""
    default_delivery_style: str = ""
    literary_traits: list[str] = field(default_factory=list)
    pronunciation_lexicon: list[LexiconEntry] = field(default_factory=list)
    character_voices: dict[str, VoiceMapping] = field(default_factory=dict)
    verbalize: dict[str, str] = field(default_factory=dict)
    free_text: str = ""

    def with_style_text(self, extra: str | None) -> StyleProfile:
        if not extra:
            return self
        merged = (self.free_text or "").strip()
        piece = extra.strip()
        free = f"{merged}\n{piece}".strip() if merged else piece
        return StyleProfile(
            id=self.id,
            narrator_persona=self.narrator_persona,
            default_delivery_style=self.default_delivery_style,
            literary_traits=list(self.literary_traits),
            pronunciation_lexicon=list(self.pronunciation_lexicon),
            character_voices=dict(self.character_voices),
            verbalize=dict(self.verbalize),
            free_text=free,
        )

    def render(self) -> str:
        """Text for the prep prompt. Voices are left out: they change speech, not the narration text."""
        lines = [
            f"id: {self.id}",
            f"narrator_persona: {self.narrator_persona.strip()}",
            f"default_delivery_style: {self.default_delivery_style!r}",
            "literary_traits:",
        ]
        for trait in self.literary_traits:
            lines.append(f"  - {trait.strip()}")
        lines.append("pronunciation_lexicon:")
        for entry in self.pronunciation_lexicon:
            lines.append(f"  - {entry.word} {entry.ipa} {entry.note}".rstrip())
        lines.append("verbalize:")
        for key, value in self.verbalize.items():
            lines.append(f"  {key}: {value.strip()}")
        if self.free_text.strip():
            lines.append("free_text:")
            lines.append(self.free_text.strip())
        return "\n".join(lines)


def load_config(path: Path | str | None = None) -> AppConfig:
    config_path = Path(path) if path else default_config_path()
    if not config_path.exists():
        return AppConfig()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    return _merge_config(AppConfig(), raw)


def load_style_profile(path: Path | str | None) -> StyleProfile:
    if path is None:
        return StyleProfile()
    profile_path = Path(path)
    text = profile_path.read_text(encoding="utf-8")
    if profile_path.suffix.lower() == ".toml":
        data = tomllib.loads(text)
    else:
        data = yaml.safe_load(text) or {}
    return style_from_dict(data)


def style_from_dict(data: dict) -> StyleProfile:
    lexicon = [
        LexiconEntry(
            word=str(item.get("word", "")),
            ipa=str(item.get("ipa", "")),
            note=str(item.get("note", "")),
        )
        for item in data.get("pronunciation_lexicon") or []
    ]
    voices: dict[str, VoiceMapping] = {}
    for name, spec in (data.get("character_voices") or {}).items():
        if isinstance(spec, str):
            voices[str(name)] = VoiceMapping(voice=spec)
        else:
            voices[str(name)] = VoiceMapping(
                voice=str(spec.get("voice", "")),
                voice_type=str(spec.get("voice_type", "")),
            )
    verbalize = {str(key): str(value) for key, value in (data.get("verbalize") or {}).items()}
    return StyleProfile(
        id=str(data.get("id") or "default"),
        narrator_persona=str(data.get("narrator_persona") or ""),
        default_delivery_style=str(data.get("default_delivery_style") or ""),
        literary_traits=[str(item) for item in data.get("literary_traits") or []],
        pronunciation_lexicon=lexicon,
        character_voices=voices,
        verbalize=verbalize,
        free_text=str(data.get("free_text") or ""),
    )


def load_env_file(path: Path | str = ".env") -> None:
    """Load KEY=VALUE lines into the environment if the file exists.

    Existing environment variables win. This never prints values.
    """
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def gemini_api_key() -> str | None:
    load_env_file()
    value = os.environ.get("GEMINI_API_KEY", "").strip()
    return value or None


def _merge_config(config: AppConfig, raw: dict) -> AppConfig:
    mapping = {
        "paths": config.paths,
        "ingest": config.ingest,
        "chunking": config.chunking,
        "prep": config.prep,
        "tts": config.tts,
        "assemble": config.assemble,
        "cost": config.cost,
    }
    for name, target in mapping.items():
        section = raw.get(name)
        if isinstance(section, dict):
            _apply_section(target, section)
    return config


def _apply_section(target: object, section: dict) -> None:
    known = {item.name for item in fields(target)}
    for key, value in section.items():
        if key not in known:
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_section(current, value)
        else:
            setattr(target, key, value)
