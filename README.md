# pdf-expressive-audiobook

Turn a PDF you own into a spoken chapter. You pick the PDF and the delivery. A style profile (or one line of direction) is the only place literary judgment lives: pace, vocal tags, character voices, and how diagrams and formulas should be spoken.

The recommended first step is a one-minute sample. It runs text preparation, speech synthesis, and assembly on a short stretch of one chapter, writes a WAV you can play, and writes a review of the emotion and pace choices. That sample is stored apart from the full-chapter cache, so it does not count as progress on the chapter.

With no voice setup, speech uses the prebuilt Gemini voice **Schedar** (labeled "Even" in the Gemini voice list), which suits long-form narration.

## What you need

- macOS or Linux
- Python 3.11 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

Optional:

- Tesseract, only for scanned pages. Install the `tesseract` binary and `pip install -e ".[ocr]"`, then set `ingest.ocr_enabled` to `true` in a copy of `config/default.toml`.
- ffmpeg is not required. Chapter and sample audio are ordinary WAV files written by Python.

## Install

```bash
git clone https://github.com/olostan/pdf-expressive-audiobook.git
cd pdf-expressive-audiobook
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Open `.env` and set `GEMINI_API_KEY` to the key you created in AI Studio. The process environment wins over `.env`. The key is never read from a committed file.

Check the install:

```bash
audiobook --version
```

## Choose a voice

You can skip this section and stay on Schedar.

Prebuilt names, Extended Voice Library ids, and designed `voice_...` ids all work. Pass any of them with `--voice`. A single Capitalized word is checked against the documented prebuilt list, so a typo is rejected before a paid call. Library ids usually contain a hyphen or digit. Designed ids start with `voice_`.

List voices (the catalog returns prebuilt voices and any library or designed voices on your key, plus metadata such as language and gender when the API sends them):

```bash
audiobook voices list --language en-US
audiobook voices list --search narrator --language en-US --gender neutral
```

Hear three prebuilt voices on the same minute of text. Schedar is even, Charon is informative, and Sulafat is warm. Use the chapter index you got from `audiobook chapters`.

```bash
audiobook sample --pdf /path/to/book.pdf --chapter 1 --style "even and clear" --voice Schedar --sample-seconds 60
audiobook sample --pdf /path/to/book.pdf --chapter 1 --style "even and clear" --voice Charon --sample-seconds 60
audiobook sample --pdf /path/to/book.pdf --chapter 1 --style "even and clear" --voice Sulafat --sample-seconds 60
```

Create a designed voice from a description. The command calls `POST /v1beta/voices` and prints a `voice_...` id:

```bash
audiobook voices design --description "A calm adult narrator with an even pace and a warm tone"
```

Use that id as `--voice voice_...`, or put it in a style profile under `character_voices.narrator`. You can also design a voice in the AI Studio Voice Design page and copy the same kind of id. Temporary emotion belongs in the style string for a turn, not in the voice description. Describe age, timbre, accent, and baseline pace.

`--lite` switches speech to `gemini-3.8-flash-lite-tts` from the config. Text preparation stays on the text model.

## Write a style profile

Copy an example and edit it:

```bash
cp profiles/examples/neutral-narration.yaml my-style.yaml
```

`neutral-narration.yaml` keeps ordinary sentences quiet. `playful-philosophical.yaml` allows a little more color and names two invented speakers. Field notes are in [docs/STYLE_PROFILE.md](docs/STYLE_PROFILE.md).

You can skip the file and pass one line. `--style` is either that line or a path ending in `.yaml`, `.yml`, or `.toml`.

```bash
audiobook sample --pdf /path/to/book.pdf --chapter 1 --style "even, unhurried, almost no vocal tags" --sample-seconds 60
audiobook sample --pdf /path/to/book.pdf --chapter 1 --style profiles/examples/neutral-narration.yaml --sample-seconds 60
```

`--style-profile` is the same file form. When both are set, the file is the profile and `--style` is extra direction for that run. Nothing is written back to the file.

## Find the chapter number

```bash
audiobook extract --pdf /path/to/book.pdf
audiobook chapters
audiobook chunks --pdf /path/to/book.pdf --chapter 1
```

`audiobook chapters` prints the 1-based index, the title, and the page range. `audiobook chunks` prints the 0-based chunk index, estimated seconds, speakers, and the opening words. Use a chunk index whose speakers are not only the narrator when you want the sample to include dialogue.

## Estimate before you spend

```bash
audiobook estimate --pdf /path/to/book.pdf --chapter 1 --sample-seconds 60
audiobook run --pdf /path/to/book.pdf --chapter 1 --style "even and clear" --dry-run
audiobook run --pdf /path/to/book.pdf --chapter 1 --style "even and clear" --max-chunks 5 --dry-run
audiobook sample --pdf /path/to/book.pdf --chapter 1 --style "even and clear" --sample-seconds 60 --dry-run
```

The lines are chunk count, estimated audio minutes, and estimated TTS cost in USD from `config/default.toml`. Prices there are a dated copy of the public Gemini price for `gemini-3.8-flash-tts`. Change them when the price page changes. Text-prep image tokens are not included. `--sample-seconds` and `--max-chunks` limit the estimate to the chunks that would actually be spoken.

## One-minute sample (do this first)

```bash
audiobook sample --pdf /path/to/book.pdf --chapter 1 --style "even and clear" --sample-seconds 60
```

The same flags work on `run`:

```bash
audiobook run --pdf /path/to/book.pdf --chapter 1 --style "even and clear" --sample-seconds 60
```

Chunks are taken in order from the start of the chapter until their estimated duration reaches about 60 seconds. The chunk that crosses the limit is included. `--sample-start` is the 0-based chunk index from `audiobook chunks` (use the index that shows dialogue when you want to hear speakers).

```bash
audiobook sample --pdf /path/to/book.pdf --chapter 1 --style "let each speaker stay distinct" --sample-start 0 --sample-seconds 60
```

The command runs the full path on only those chunks: Gemini text prep (emotion, inline tags, style strings, figure, formula, and footnote handling, then the validator), Gemini speech, and assembly. It also writes the review for exactly those chunks.

Outputs (the PDF file name, lowercased, is the book folder):

```text
artifacts/<book>/samples/<chapter>-<timestamp>.wav
artifacts/<book>/samples/<chapter>-<timestamp>.md
```

Play the WAV. Open the Markdown review beside it. Each chunk shows the source text, and each turn shows speaker, style, prepared text, and the rationale. Tags such as `<chuckle>` or `<short pause>` appear in the prepared text. An empty style is normal for a plain sentence.

Sample files are not the chapter cache. A later full chapter still prepares and speaks every chunk, including the ones you already sampled.

## Full chapter

```bash
audiobook run --pdf /path/to/book.pdf --chapter 1 --style profiles/examples/neutral-narration.yaml
audiobook review --chapter 1
```

The chapter WAV is `artifacts/audio/chapter-001.wav` (the number matches the chapter index). `audiobook review` reads the full-chapter prep outputs, not the sample review.

`--max-chunks 5` on `run` (without `--sample-seconds`) caps a chapter run and stores those chunks in the chapter cache. A later `run` of the same chapter continues with the chunks that are not finished yet. Use `--sample-seconds` when you want a listen that must not become part of that cache.

## Where files land

```text
artifacts/ingest/manifest.json
artifacts/ingest/chapters/001.json
artifacts/chunks/001.jsonl
artifacts/prep/requests/<chunk-id>.json
artifacts/prep/outputs/chunks/<chunk-id>.json
artifacts/prep/outputs/001.jsonl
artifacts/prep/review/chapter-001.md
artifacts/tts/plans/<chunk-id>.json
artifacts/audio/clips/001/*.wav
artifacts/audio/chapter-001.wav
artifacts/<book>/samples/<chapter>-<timestamp>.wav
artifacts/<book>/samples/<chapter>-<timestamp>.md
```

## Run one stage, or resume

Each stage reads the previous stage's files. A chunk whose prep file already validates is not sent to the text model again. A clip that already plays as 24 kHz mono 16-bit WAV is not synthesized again. Delete that chunk's prep file and its clips when you want them redone.

```bash
audiobook prep --chapter 1 --style profiles/examples/neutral-narration.yaml
audiobook tts --chapter 1 --style profiles/examples/neutral-narration.yaml
audiobook assemble --chapter 1
```

Text prep and speech retry HTTP 429 and 5xx with exponential backoff (`prep.max_http_retries`, `tts.max_http_retries`, `backoff_initial_seconds`, `backoff_multiplier` in the TOML config). Several chunks run at once, capped by `prep.concurrency` and `tts.concurrency`. If a prep response fails the JSON schema or the validator, the errors go back to the model up to `prep.max_validation_retries` times. The chunk then stops with its id and the errors in the message.

## Scanned pages

A page with no text layer is kept as a spoken placeholder and listed on the chapter. To OCR it:

```bash
pip install -e ".[ocr]"
```

Install the `tesseract` binary from your OS packages, set `ingest.ocr_enabled = true`, and run `audiobook extract` again. Dense math and tiny notes will still need a look in the review.

## Troubleshooting

**Chapter breaks look wrong.** Run `audiobook chapters` and compare titles with the pages in the PDF. The extractor uses the PDF outline when there is one, otherwise headings. Front matter is its own chapter when pages come before the first chapter heading. Details are in [docs/NOTES.md](docs/NOTES.md).

**Scanned or blank pages.** You will see a warning and a placeholder line. Turn on OCR as above, or skip that chapter.

**Prep failed for chunk …** Read the errors. The usual causes are an inline tag that is not on the allowlist, a style string that quotes the transcript, raw math left in the spoken text, or a figure or note the model dropped. Tighten the style profile, delete that chunk's file under `artifacts/prep/outputs/chunks/`, and run prep again. Finished chunks stay finished.

**Rate limits.** The client already backs off on 429 and 5xx. If it still stops, lower `prep.concurrency` and `tts.concurrency` in the config and run the same command again. Finished chunks and clips are skipped.

**The voice sounds robotic.** Try another prebuilt voice (`Charon`, `Sulafat`, `Schedar`) with the 60-second sample. Add a short `--style` such as `warm, unhurried, a little smile in the voice`. Do not pile tags onto every sentence.

**The voice sounds over-acted.** Put `Prefer an empty style on ordinary sentences` in the profile's `literary_traits`. Keep `--style` to a few words. Empty style is what the speech model expects for a plain sentence. Design a calmer `voice_...` id if the prebuilt voice itself is too bright.

**Dialogue is all one voice.** Set `character_voices` in the profile for the speaker labels `audiobook chunks` prints, or pass a profile that already names them. Only two prebuilt voices can share one request. A library id or a `voice_...` id is spoken one turn at a time.

## Tests

```bash
pytest
pytest -m live
```

`pytest` does not call Gemini. `pytest -m live` runs one short text-prep call and one short speech call with the default prebuilt voice, and checks that the audio is 24 kHz mono 16-bit WAV-compatible PCM. It is skipped unless `GEMINI_API_KEY` is set, and it spends a small amount of quota.

## Further reading

- [docs/STYLE_PROFILE.md](docs/STYLE_PROFILE.md) — profile fields
- [docs/NOTES.md](docs/NOTES.md) — extraction failure modes and the cost formula
- [docs/GEMINI_API.md](docs/GEMINI_API.md) — model ids, speech fields, and voice design
- `config/default.toml` — models, voices, retries, and prices
