# Style profile

A style profile is the only literary input besides the PDF. Pass it with `--style-profile path.yaml` (or `.toml`). `--style "free text"` is appended to `free_text` for that run and is not written back to the file.

The JSON Schema is `profiles/schema.json`. Two generic examples live in `profiles/examples/`:

- `neutral-narration.yaml` — empty default style, one narrator voice
- `playful-philosophical.yaml` — traits for wordplay, self-reference, and dialogue, plus two invented speakers

## Fields

| Field | Role |
| --- | --- |
| `id` | Name stored with the run. Required. |
| `narrator_persona` | Who is speaking the prose. Injected into the text-prep prompt. |
| `default_delivery_style` | Hint only. Empty is the recommended start. |
| `literary_traits` | What to preserve: wordplay, self-reference, dialogue shape, and so on. |
| `pronunciation_lexicon` | `{word, ipa, note}`. The prep prompt is told to write `Word /ipa/` when a listed word appears. |
| `character_voices` | Speaker label to `{voice, voice_type}`. Include `narrator`. |
| `verbalize` | How to speak `diagrams`, `formulas`, `footnotes`, and `citations`. |
| `free_text` | Extra direction. The CLI `--style` string is appended here. |

`voice` may be a prebuilt studio name (`Kore`), an Extended Voice Library id, a Voice design id (`voice_...`), or a replication id (`voice_...` or stateless `voicekey_...`).

`voice_type` is `prebuilt`, `library`, `design`, or `replication`. The TTS planner pairs turns into one multi-speaker request only when the voices are prebuilt and there are exactly two speakers. Every other voice is synthesized one turn at a time, which matches the public limit of two prebuilt speakers per request and the rule that designed or replicated voices are not combined in one call.

Unknown speakers fall back to the `narrator` entry, then to `tts.default_voice` in the TOML config.
