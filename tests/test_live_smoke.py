"""Opt-in Gemini smoke test. Skip unless GEMINI_API_KEY is set."""

import pytest

from pdf_expressive_audiobook.audio import load_pcm
from pdf_expressive_audiobook.config import StyleProfile, VoiceMapping, gemini_api_key, load_config
from pdf_expressive_audiobook.models import Chunk
from pdf_expressive_audiobook.prep import GeminiTextPrepClient, prepare_chunk_output
from pdf_expressive_audiobook.tts import GeminiTtsClient, plan_tts_requests

pytestmark = pytest.mark.live


def test_live_prep_and_short_tts():
    api_key = gemini_api_key()
    if not api_key:
        pytest.skip("GEMINI_API_KEY is not set")
    config = load_config()
    config.prep.max_validation_retries = 1
    chunk = Chunk(
        id="c900-0000",
        chapter_id="ch-900",
        chapter_index=900,
        chapter_title="Smoke",
        order=0,
        kind="prose",
        text="The morning was clear and the road was empty.",
        speakers=["narrator"],
        source_pages=[1],
        estimated_duration_sec=4,
        word_count=9,
        block_ids=["b-1"],
    )
    prepared = prepare_chunk_output(chunk, StyleProfile(), config, GeminiTextPrepClient(api_key, config))
    assert prepared["chunk_id"] == chunk.id
    assert prepared["turns"]
    profile = StyleProfile(
        character_voices={"narrator": VoiceMapping(config.tts.default_voice, "prebuilt")}
    )
    request = plan_tts_requests(
        [
            {
                "text": "The morning was clear and the road was empty.",
                "style": "even",
                "speaker": "narrator",
            }
        ],
        profile,
        config,
    )[0]
    pcm = GeminiTtsClient(api_key, config).synthesize(request)
    clip = load_pcm(pcm, expected_rate=24000)
    assert clip.sample_rate == 24000
    assert clip.channels == 1
    assert clip.sample_width == 2
    duration = clip.frame_count / clip.sample_rate
    assert 0.3 < duration < 20
