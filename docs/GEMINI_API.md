# Gemini API notes

Checked against public documentation on 2026-10-24. Nothing here was confirmed by a live API call. Model ids and field names used by this repo live in `config/default.toml`, not as scattered literals.

Sources:

- [Speech generation (Interactions)](https://ai.google.dev/gemini-api/docs/speech-generation)
- [Speech generation (Generate Content, legacy)](https://ai.google.dev/gemini-api/docs/generate-content/speech-generation)
- [gemini-3.8-flash-tts model card](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash-tts)
- [gemini-3.8-flash model card](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)
- [Models index](https://ai.google.dev/gemini-api/docs/models)
- [Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Vertex / Gemini Enterprise speech overview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/text-to-speech/overview)
- [Prompting guide (Cloud)](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/text-to-speech/prompting-guide)

## Verified in those pages

| Item | What the docs say |
| --- | --- |
| TTS models | `gemini-3.8-flash-tts` and `gemini-3.8-flash-lite-tts` |
| Text-prep model | `gemini-3.8-flash` accepts text, image, video, audio, and PDF, and returns text. Audio generation is not a capability of that model. |
| Verbatim text | The TTS `text` field is spoken aloud. Sustained direction goes in `speech_metadata.style`, not in the transcript. |
| GenerateContent style | `parts[].speech_metadata.style` (and `speaker` on multi-speaker turns) |
| Interactions style | An annotation with `"type": "speech_metadata"`, plus `speaker` and `style` |
| Single-speaker voice | `speechConfig.voiceConfig.voice` on GenerateContent. The value may be a prebuilt name, an Extended Voice Library id, `voice_...`, or `voicekey_...`. |
| Interactions voice | `generation_config.speech_config` is `[{"voice": "Kore"}]` for one speaker and `{"speakers": [...]}` for two. |
| Multi-speaker limit | Up to two speakers per request, and only with prebuilt voices. Designed or replicated voices are synthesized one turn at a time. |
| GenerateContent multi-speaker shape | `speechConfig.multiSpeakerVoiceConfig.speakerVoiceConfigs[]` with `voiceConfig.prebuiltVoiceConfig.voiceName` |
| Inline tags | Angle brackets for point-in-time vocal events. The allowlist in config matches the recommended tags in the speech-generation markdown, including `<short pause>`, `<long pause>`, `<breath>`, `<laugh>`, and the rest listed there. |
| Backchannels | Pipe-wrapped listener reactions such as `\|mhm\|`. This repo's default allowlist is `\|mhm\|` and `\|yeah\|` and is configurable. |
| Emphasis and hesitation | Capitalized words, punctuation, and ellipses. |
| Empty style | Docs recommend starting with an empty style and adding a short style only when a turn needs it. |
| Unary audio | GenerateContent unary responses default to WAV (`audio/wav` / `AUDIO_WAV`), 24 kHz, mono, 16-bit PCM, with a RIFF header. The Interactions limitations section describes that header as 44 bytes. |
| Raw PCM | Request `AUDIO_L16` on GenerateContent, or `audio/l16` on Interactions. Streaming defaults to headerless 24 kHz mono 16-bit PCM. |
| Sample rate on the Gemini API | The Gemini API speech docs say `sampleRate` / `sample_rate` may be set (examples include 24000, 16000, 8000). |
| Languages | The models index describes Flash TTS as covering 130 languages and Flash-Lite TTS as covering 101. Language is auto-detected. This repo does not send an unverified language field on the TTS request. `tts.language` is a hint for the prep prompt. |
| Voice catalog | `GET /v1beta/voices`, also `client.voices.list()`. Filters documented for the Gemini API include language, region, accent, gender, pitch, persona, context, type, and search. |
| Pricing, standard tier, through 2026-12-31 | `gemini-3.8-flash-tts`: $0.50 / 1M text input tokens, $9.00 / 1M audio output tokens, 25 audio tokens per second, described as $0.00225 per 10 seconds. From 2027-01-01 the same page lists $1.00 and $18.00. Flash-Lite TTS is cheaper on that page ($6.00 / 1M audio tokens through 2026-12-31). Batch, flex, and priority tiers differ. |
| SDK gap | The GenerateContent docs say custom `responseFormat` overrides are supported on REST until the Python and JavaScript SDKs expose `response_format` on `GenerateContentConfig`. Interactions is the SDK path that already shows `response_format`. |

## Not verified, or contradicted

| Item | Note |
| --- | --- |
| Live call | No request was sent. Field names above are from the published pages only. |
| `sampleRate` on Vertex | The Cloud speech overview says the models ignore `sampleRate` and always emit 24 kHz PCM/WAV (8 kHz for μ-law and A-law). That disagrees with the Gemini API docs. This pipeline requests 24000 and still checks the WAV `fmt` chunk before concatenating. |
| Always-44-byte headers | One limitations sentence says unary WAV has a 44-byte RIFF header. A `fmt` chunk can be longer than 16 bytes, and `JUNK`, `fact`, or `LIST` chunks move `data`. Assembly parses RIFF chunks instead of slicing 44 bytes. |
| IPA spelling | The AI Studio developer guide describes following a hard word with `/ipa/`. That convention is used in the prep prompt. It was not restated as a formal API field in the speech-generation page read for this note. |
| Exact latency | No published per-request latency was found. `docs/NOTES.md` keeps seconds-per-request as a parameter. |
| Text-prep token price | Not copied into config. Image tokens depend on resolution. Treat prep cost as parametric. |
| Empty string versus omitted `style` | Docs say to start empty. Whether `""` and an omitted field behave the same was not A/B tested. The planner sends `"style": ""` when the prep output is empty. |
| Python package `google-genai` | Not installed here, so SDK attribute names were not executed. |

## How this repo uses that

- Default TTS API is GenerateContent. `tts.api = "interactions"` switches the payload builder.
- Default audio request is `AUDIO_L16` at 24000 Hz, mono, 16-bit. Interactions uses `audio/l16` from `tts.interactions_audio_mime`.
- `audiobook voices list` calls `GET {tts.endpoints.voices}` with `x-goog-api-key`.
- `GeminiTextPrepClient.generate` and `GeminiTtsClient.synthesize` raise `NotImplementedError`. Planning and validation do not need a key.
