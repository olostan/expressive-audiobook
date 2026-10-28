# pdf-expressive-audiobook

A staged pipeline that turns a PDF you own into a multi-chapter expressive audiobook. You supply the PDF and a style profile. The profile is the only place literary judgment lives: narrator persona, delivery, pronunciation, character voices, and how diagrams and formulas should be spoken.

Stages write files under `artifacts/` so you can rerun or resume one stage without repeating the others.

| Stage | Command | Status |
| --- | --- | --- |
| 1. Ingest | `audiobook extract` | Implemented |
| 2. Chunk | `audiobook chunk` | Implemented |
| 3. Text prep | `audiobook prep` | Prompt, schema, validator, and multimodal request builder. The model call is stubbed. |
| 4. TTS | `audiobook tts` | Request planner and `voices list`. Synthesis is stubbed. |
| 5. Assemble | `audiobook assemble` | PCM/WAV helpers are implemented. The command concatenates clips you already have. |

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Put `GEMINI_API_KEY` in `.env` when you want `audiobook voices list`. The key is never read from a committed file. Text-prep and synthesis do not call the network yet.

PyMuPDF is used to read page structure and render figure and formula crops. Its license is separate from this project's license.

Optional OCR, off by default:

```bash
pip install -e ".[ocr]"
```

Set `ingest.ocr_enabled` in the TOML config. Without `pytesseract` and a `tesseract` binary, scanned pages are flagged and left as placeholders.

## Usage

```bash
audiobook extract --pdf /path/to/book.pdf
audiobook chunk
audiobook prep --style-profile profiles/examples/neutral-narration.yaml \
  --style "keep the pace measured"
audiobook review --chapter 1
audiobook tts --style-profile profiles/examples/neutral-narration.yaml
audiobook voices list --search narrator --language en-US
audiobook assemble
audiobook run --pdf /path/to/book.pdf --chapter 1 \
  --style-profile profiles/examples/playful-philosophical.yaml
```

`--chapter N` is the 1-based index in the ingest manifest. Front matter is its own chapter when pages precede the first outline or contents entry.

`--style` is free text appended to the profile at runtime. `--style-profile` is a YAML or TOML file. See [docs/STYLE_PROFILE.md](docs/STYLE_PROFILE.md) and `profiles/schema.json`. The example profiles are generic. They do not describe a particular book.

## What each stage writes

```
artifacts/ingest/manifest.json
artifacts/ingest/chapters/001.json
artifacts/ingest/chapters/001/figures/*.png
artifacts/ingest/chapters/001/formulas/*.png
artifacts/chunks/001.jsonl
artifacts/prep/requests/<chunk-id>.json
artifacts/prep/outputs/001.jsonl          # you add this after a text-prep run
artifacts/prep/review/chapter-001.md
artifacts/tts/plans/<chunk-id>.json
artifacts/audio/clips/001/*.wav           # you add synthesized clips
artifacts/audio/chapter-001.wav
artifacts/audio/book.wav
```

`audiobook prep` writes the request JSON only. When a later run fills `artifacts/prep/outputs/`, `audiobook tts` plans speech requests and `audiobook review` shows source text, prepared text, style, rationale, and inserted descriptions.

## Configuration

`config/default.toml` holds paths, chunk timing, the text-prep model, the TTS model, voice defaults, audio format, and the allowlist of inline vocal tags. Model ids and API field names stay in that file.

Chunking defaults to about 150 words per minute, a target of 30–50 seconds, and a hard cap strictly under 60 seconds. Splits happen on paragraph, sentence, or dialogue-turn boundaries. A single sentence that is already over the cap is kept whole and marked with a warning.

## Tests

```bash
pytest
```

Fixtures are synthetic PDFs built in the test run. One dialogue fixture quotes a short public-domain passage from Lewis Carroll's 1895 "What the Tortoise Said to Achilles."

## Further reading

- [docs/NOTES.md](docs/NOTES.md) — failure modes and a parametric cost and latency estimate
- [docs/GEMINI_API.md](docs/GEMINI_API.md) — which Gemini TTS and text-model details were checked against public docs
- [docs/STYLE_PROFILE.md](docs/STYLE_PROFILE.md) — profile fields
