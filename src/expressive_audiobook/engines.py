"""Choose the text model and the speech engine from the config.

Gemini is one choice among several. Its key is only required when a Gemini engine is selected.
"""

from __future__ import annotations

from .config import AppConfig, gemini_api_key
from .llm import OpenAICompatTextClient
from .prep import GeminiTextPrepClient
from .tts import GeminiTtsClient

GEMINI_KEY_HINT = (
    "GEMINI_API_KEY is not set. Put it in .env or the environment (see .env.example), "
    "or choose another engine: text.provider = \"openai\" (for example Ollama) and tts.provider = \"orpheus\"."
)


class EngineError(RuntimeError):
    pass


def text_needs_gemini(config: AppConfig) -> bool:
    return config.text.provider == "gemini"


def tts_needs_gemini(config: AppConfig) -> bool:
    return config.tts.provider == "gemini"


def make_text_client(config: AppConfig, *, required: bool = False):
    """The model that reads the book, or None when it is Gemini and there is no key (unless ``required``)."""
    provider = config.text.provider
    if provider == "gemini":
        key = gemini_api_key()
        if not key:
            if required:
                raise EngineError(GEMINI_KEY_HINT)
            return None
        return GeminiTextPrepClient(key, config)
    if provider == "openai":
        return OpenAICompatTextClient(config)
    raise EngineError(f"Unknown text.provider {provider!r}. Use \"gemini\" or \"openai\".")


def make_tts_client(config: AppConfig, *, required: bool = False):
    provider = config.tts.provider
    if provider == "gemini":
        key = gemini_api_key()
        if required and not key:
            raise EngineError(GEMINI_KEY_HINT)
        return GeminiTtsClient(key, config)
    if provider == "orpheus":
        from .speech_local import OrpheusTtsClient

        return OrpheusTtsClient(config)
    if provider == "http":
        from .speech_local import SpeechHttpClient

        return SpeechHttpClient(config)
    raise EngineError(f"Unknown tts.provider {provider!r}. Use \"gemini\", \"orpheus\" or \"http\".")
