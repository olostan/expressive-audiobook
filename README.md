# expressive-audiobook

Turn a PDF, Markdown or text file you own into a spoken chapter. You pick the book and the delivery. A style profile (or one line of direction) is the only place literary judgment lives: pace, vocal tags, character voices, and how diagrams and formulas should be spoken.

Work happens in a **project**: one folder per book under `projects/<name>/` that holds the extracted chapters, your settings, the narration style, and every generated file. Import once, tune, then generate as often as you like. Every step is resumable.

**Google is optional.** Gemini is the zero-setup default, but you can run everything locally instead: a local language model such as Ollama reads the book, and Orpheus (on Apple Silicon) speaks it with laughs, sighs and pauses, with no API key at all. You can also mix: for example Ollama for reading and Gemini for speech. See [Run without Google](#run-without-google). ElevenLabs is optional too, and only used if you pick it for sound effects.

With Gemini and no voice setup, speech uses the prebuilt voice **Schedar** (labeled "Even" in the Gemini voice list), which suits long-form narration.

## What you need

- macOS or Linux
- Python 3.11 or newer
- A way to read and speak the book, either one of:
  - a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) (the default, nothing else to install), or
  - local models, with no Google account: Ollama (or another OpenAI-compatible server) for reading and Orpheus for speech (see [Run without Google](#run-without-google)).

Optional:

- An ElevenLabs API key, only if you choose ElevenLabs for sound effects. Music and effects are made locally by default, so nothing needs ElevenLabs.
- ffmpeg, to also write MP3 or M4A copies of the audio (`brew install ffmpeg`). Without it, chapter and sample audio are ordinary WAV files written by Python.
- ffmpeg and `numpy` (`pip install -e ".[sound]"`) for mixing background music and sound effects; a local Stable Audio model makes the sounds (see below).
- Tesseract, only for scanned pages. Install the `tesseract` binary and `pip install -e ".[ocr]"`, then set `"overrides": {"ingest": {"ocr_enabled": true}}` in the project's `project.json`.

## Install

```bash
git clone https://github.com/olostan/expressive-audiobook.git
cd expressive-audiobook
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Open `.env` and set `GEMINI_API_KEY` to the key you created in AI Studio. `ELEVENLABS_API_KEY` can stay empty, and so can `GEMINI_API_KEY` if you run local models. The process environment wins over `.env`. The key is never read from a committed file.

Check the install, and your whole setup:

```bash
audiobook --version
audiobook doctor
```

`audiobook doctor` checks Python, which API keys you need for the engines you chose (and which you do not), whether Ollama or an Orpheus server is reachable and has your model, whether Orpheus is installed, and the optional tools (ffmpeg, numpy, Tesseract, Stable Audio). Each problem comes with the command that fixes it, and the exit code is 1 when something required is missing. Keys are never printed. Add `--online` to confirm a Gemini key is accepted (one free request), and `--project NAME` to check against a project's own settings.

## Hear an example

[`examples/pip-and-the-lantern`](examples/pip-and-the-lantern) holds a short children's story, the [narrated MP3](examples/pip-and-the-lantern/pip-and-the-lantern.mp3) (about four minutes, with background music and sound effects), the style `import` wrote for it, and the settings used. Open the folder to see how to make the same thing from the source text.

## Quick start: import, tune, prepare, generate

```text
audiobook import --source ~/Downloads/book.pdf --name mybook
audiobook import --source ~/stories/pip.md --name pip        # Markdown or plain text works too
```

1. **Import.** Reads the book into chapters (JSON), skips Contents pages, then asks Gemini to read the opening and write a narration style. It prints what it found and what you can change, and (in a terminal) asks you to confirm the suggested delivery and whether you want background music and sound effects. No speech is generated and almost nothing is spent.
2. **Tune (optional).** Edit `projects/mybook/project.json` (voice, expressiveness, pace, pauses) and `projects/mybook/style.md` (narrator persona, traits, pronunciation, how to speak formulas and figures, free notes).
3. **Prepare (optional).** Writes the narration for chapters: the spoken text, delivery style, and inline tags such as `<chuckle>` and `<short pause>`. Review it in `prep/review/chapter-NNN.md`, or edit `prep/outputs/chunks/*.json` by hand.

   ```text
   audiobook prepare --project mybook --chapter 4
   audiobook prepare --project mybook --chapter 4-6
   audiobook prepare --project mybook --all
   ```
4. **Generate.** Prepares anything not yet prepared, speaks it, and assembles audio.

   ```text
   audiobook generate --project mybook --chapter 4 --sample 30   # first 30 seconds of chapter 4
   audiobook generate --project mybook --chapter 4               # the whole chapter
   audiobook generate --project mybook --all                     # the book (also writes the whole-book file)
   audiobook status --project mybook                             # progress per chapter
   ```

`--project` can be left out when `projects/` holds only one project. `gen` is short for `generate`. Add `--dry-run` to see the estimate without any API call, `--force` to redo cached results, `--quiet` (before the command) for plain output. Colors turn off automatically when output is not a terminal or `NO_COLOR` is set.

### Markdown and text files

`--source` takes a `.pdf`, `.md` or `.txt` file (`--pdf` still works).

- **Markdown:** chapters are the headings. A single top heading names the book; the next level down becomes the chapters. YAML front matter (`title`, `author`) is read. Links, emphasis, images, code blocks, tables and comments are stripped; list items and block quotes are read as paragraphs.
- **Plain text:** chapters are lines like `Chapter 3`, `Prologue`, `IV`, `2. The Hill` or ALL-CAPS titles, with a short line after a bare `Chapter 3` taken as its title. `* * *` scene breaks are dropped.
- **No headings at all:** a long file is cut into parts of about 3,000 words at scene breaks and paragraph boundaries; a short one is a single chapter.
- A `NAME: line` script format is read as dialogue with speakers. Quoted dialogue in prose is left to the narrator for now.

Add `--non-interactive` to skip every question (the suggestions and flags are used), or pass the answers directly: `--music 0.4 --sfx 0.3 --audience kids`.

### Background music and sound effects

Both are optional and **off by default**: they cost generation time (or money) and suit stories more than technical books. Import suggests them for fiction and asks; you can also set them any time:

```text
audiobook sound --project pip --music 0.4 --sfx 0.3 --audience kids
audiobook sound --project pip --music-style "soft piano and strings, like a bedtime story"
```

- `--music` is 0 (off) to 1 (very noticeable). It sets how far under the voice the music sits; it also ducks while the narrator speaks.
- `--sfx` is 0 (off) to 1 (very noticeable). It sets how many sound effects are suggested (about one per 1,200 words at 0.1 and one per 150 at 1).
- `--audience kids` asks for gentler music and no frightening or sudden sounds.

On `generate`, a "music supervisor" pass (one text-model call) writes `score/music.json`: one shared palette for the book and a prompt for each chapter. Each chapter gets a few variants of its music, and the mixer joins them with long crossfades in a shuffled order, from a random start, with a slow level drift, so a short loop is not noticeable. Each chapter fades its music in before the first word and out after the last. A "sound designer" pass reads each chapter's prepared text and writes `score/sfx-NNN.json`: one-off sounds at turn boundaries and quiet ambience loops (rain, wind) under a range of turns. Both plans are plain JSON you can edit; a plan whose inputs have not changed is not rewritten. Changing a level only remixes; speech is never redone.

Sounds are made locally with Stable Audio 3 (free, a few seconds each on Apple Silicon) and kept in a library shared by every book, so a rain loop made for one book is reused by the next; the sound designer is shown what the library already holds and prefers it.

```text
audiobook models install sa3                    # one time, about 8 GB, into ~/.audiobook/models
audiobook library list                          # everything made so far
audiobook library play music-warm-felt-piano-73a3b5d1
audiobook library reject sfx-a-small-soft-splash-1fe6bfa9   # never mix this one again
audiobook library generate --kind sfx --prompt "a wooden door creaking open" --seconds 3
audiobook library add ~/Music/rain.flac --kind sfx --loop --title "Rain"
```

New sounds start as `candidate`. Everything except `rejected` is mixed; set `sound.require_approval = true` (in `overrides`) to mix only items you have `approve`d. ElevenLabs can make the effects instead (`"overrides": {"sound": {"sfx_provider": "elevenlabs"}}`, with `ELEVENLABS_API_KEY` in `.env`). Mixing needs `ffmpeg` and `numpy` (`pip install -e ".[sound]"`). With music or effects on, chapters are written as 44.1 kHz stereo, and a `--dry-run` shows how many tracks and cues are planned.

### Run without Google

Gemini is the default, not a requirement. Two settings pick the models, and each can be changed on its own:

| What | Setting | Choices |
| --- | --- | --- |
| Reading the book (narration prep, style detection, music and effects planning) | `text.provider` | `gemini`, or `openai` for any OpenAI-compatible server: **Ollama**, LM Studio, llama.cpp, vLLM or a hosted API |
| Speaking | `tts.provider` | `gemini`, `orpheus` (runs on your Mac), or `http` (an OpenAI-style `/v1/audio/speech` server, for example Orpheus-FastAPI) |

With both set to local engines there is no API key anywhere.

```text
ollama pull qwen3:14b                           # any chat model your machine can run
audiobook models install orpheus                # one time, about 7 GB, Apple Silicon (uses mlx-audio)
```

Put the choice in a config file (used by `import`, which runs before a project exists) or in one project's `project.json`:

```json
{ "overrides": { "text": { "provider": "openai", "model": "qwen3:14b" },
                 "tts":  { "provider": "orpheus" } } }
```

```toml
# my-config.toml, used with: audiobook --config my-config.toml import --source book.pdf --name mybook
[text]
provider = "openai"
model = "qwen3:14b"
base_url = "http://localhost:11434/v1"   # Ollama; LM Studio is :1234
[tts]
provider = "orpheus"
```

What changes with a local engine:

- **Orpheus is expressive through tags and punctuation, not a style line.** It performs `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>` and `<gasp>`. The prep step is told this and uses only those tags, CAPITALIZED emphasis and punctuation. The per-turn `style` is kept for your review but is not spoken. `<short pause>` becomes real silence of `tts.orpheus.pause_ms`.
- **Voices** are Orpheus's eight (`tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac`, `zoe`). Set `narration.voice` to one, for example `"leo"`. Gemini names left in a project fall back to `tara` for the narrator, and each character gets a stable different voice. Designing voices (`voices design`) is Gemini-only.
- **Speech runs one clip at a time** and costs nothing; `--dry-run` shows a cost of 0.
- **A smaller text model follows the instructions less closely.** Prep checks every result (allowed tags, no dropped figures or notes) and asks the model to fix mistakes, so weak output costs time, not quality. Try a larger model if chunks keep failing. Page images are only sent when `text.vision = true` and the model can see.
- Changing the engine changes the cache fingerprints, so only the affected work is redone.

### Where things live

`~/.audiobook/` (override with `AUDIOBOOK_HOME`) holds `projects/`, `library/` and `models/`. A `./projects` folder keeps being used while it exists and `~/.audiobook/projects` does not.

### Resuming and changing your mind

Stop with Ctrl-C at any time and run the same command again. Each chunk of narration and each speech clip is saved atomically with a fingerprint of everything that produced it: the text, `style.md`, the tuning, the voice, the models, and the prompt. Unchanged work is skipped (shown as `✓ cached`). Change a setting and only the affected chunks are redone, and stale clips are removed. A sample reuses the same cache as the full chapter, so listening to 30 seconds first costs nothing extra later.

### project.json

```json
{
  "narration": {
    "voice": "Schedar",
    "expressiveness": 0.6,
    "pace": 0.3,
    "pause_density": 0.8
  },
  "audio": { "gap_ms": null, "chapter_tail_ms": null },
  "character_voices": { "Achilles": "Puck" },
  "overrides": { "prep": { "temperature": 0.5 } }
}
```

| Setting | Values | Effect |
| --- | --- | --- |
| `expressiveness` | 0 to 1 | 0 reads plainly; 1 is fully performed. Sets how often turns get a style and inline tags (about 1.5 tags per 100 words at 0.6). A passage that comes back too flat is sent back once for more performance. |
| `pace` | 0 to 1 | 0 is slowest, about 0.5 is conversational, 1 is fastest. Below 0.4 the narrator is asked for an unhurried style. |
| `pause_density` | 0 to 1 | 0 means only punctuation pauses; 1 means frequent `<short pause>` tags (two in a row for a longer pause; the model sometimes speaks `<long pause>` aloud). Also sets the default silence between passages (250 ms at 0, 900 ms at 1) and at the end of a chapter. |
| `audio.format` | `mp3`, `m4a`, `wav` | With `mp3` or `m4a` and `ffmpeg` installed, each chapter, sample and `book` is written as a compressed, tagged file (title, track, album), about 6 times smaller than WAV (WAV is 24 kHz mono, about 170 MB per hour). Without ffmpeg you get a note and WAV only. Default `mp3`. |
| `audio.keep_wav` | true/false | Default `false`: the WAV is deleted once its MP3/M4A exists, and is not rebuilt while the compressed file is current. Set `true` to keep both. The per-passage speech clips in `audio/clips/` are always kept, so nothing is paid for twice. |
| `gap_ms`, `chapter_tail_ms` | number or `null` | Silence between passages and after a chapter. `null` follows `pause_density`. |
| `voice`, `lite` | a Gemini voice, true/false | Narrator voice; `lite` uses the cheaper speech model. |
| `character_voices` | speaker to voice | Voices for characters in dialogue. |
| `overrides` | any section of `default.toml` | Per-book changes to models, chunk length, retries, and so on. |

The old words (`low`/`medium`/`high`, `slow`/`measured`/`normal`/`brisk`, `minimal`/`normal`/`generous`) still work and map to numbers.

Import suggests expressiveness, pace, and pauses from the book itself. `audiobook import --restyle` asks again.

### Layout

```text
projects/mybook/
  project.json   style.md
  ingest/        manifest.json, chapters/NNN.json, page images
  chunks/        prep/ (outputs, review/)   tts/
  audio/         clips/, chapter-NNN.mp3 (or .m4a/.wav), book.mp3
  samples/       chapter-004-30s.mp3
  score/         music.json, sfx-NNN.json (when music or effects are on)
```

`projects/` is ignored by git. Importing the same file again keeps your edits and does nothing; a different file under the same name needs `--force` and deletes the generated files. `audiobook projects` lists your projects.

## Choose a voice

You can skip this section and stay on Schedar.

Prebuilt names, Extended Voice Library ids, and designed `voice_...` ids all work. Use any of them as `narration.voice` in `project.json`, or pass them with `--voice` to the low-level commands. With a local engine the voices are different; see [Run without Google](#run-without-google). A single Capitalized word is checked against the documented prebuilt list, so a typo is rejected before a paid call. Library ids usually contain a hyphen or digit. Designed ids start with `voice_`.

List voices (the catalog returns prebuilt voices and any library or designed voices on your key, plus metadata such as language and gender when the API sends them):

```bash
audiobook voices list --language en-US
audiobook voices list --search narrator --language en-US --gender neutral
```

Hear three prebuilt voices on the same minute of text. Schedar is even, Charon is informative, and Sulafat is warm. Set `narration.voice` in `project.json`, then sample a chapter. Only the speech is redone, and the sample is overwritten, so copy it aside before trying the next voice.

```text
audiobook generate --project mybook --chapter 1 --sample 60     # with "voice": "Schedar"
audiobook generate --project mybook --chapter 1 --sample 60     # after changing it to "Charon"
```

Create a designed voice from a description. The command calls `POST /v1beta/voices` and prints a `voice_...` id:

```bash
audiobook voices design --description "A calm adult narrator with an even pace and a warm tone"
```

Use that id as `narration.voice` in `project.json` (or `--voice voice_...` with the low-level commands). You can also design a voice in the AI Studio Voice Design page and copy the same kind of id. Temporary emotion belongs in the style string for a turn, not in the voice description. Describe age, timbre, accent, and baseline pace.

`"lite": true` in `project.json` (`--lite` on the low-level commands) switches speech to `gemini-3.8-flash-lite-tts` from the config. Text preparation stays on the text model.

## Write a style profile

In a project the narration style is `projects/<name>/style.md`, written by `import` and yours to edit. For a reusable, structured file, or for the low-level commands, copy an example and edit it:

```bash
cp profiles/examples/neutral-narration.yaml my-style.yaml
```

`neutral-narration.yaml` keeps ordinary sentences quiet. `playful-philosophical.yaml` allows a little more color and names two invented speakers. Field notes are in [docs/STYLE_PROFILE.md](docs/STYLE_PROFILE.md).

The low-level commands also take one line instead of a file. `--style` is either that line or a path ending in `.yaml`, `.yml`, or `.toml`; `--style-profile` is the same file form. When both are set, the file is the profile and `--style` is extra direction for that run. Nothing is written back to the file.

```bash
audiobook sample --pdf /path/to/book.pdf --chapter 1 --style "even, unhurried, almost no vocal tags" --sample-seconds 60
audiobook sample --pdf /path/to/book.pdf --chapter 1 --style profiles/examples/neutral-narration.yaml --sample-seconds 60
```

## Low-level commands (without a project)

Everything above is the normal way to work. These older commands take a PDF directly and write to `artifacts/` (or into a project when you pass `--project`). They are handy for one-off experiments and for running a single stage.

### Find the chapter number

```bash
audiobook extract --pdf /path/to/book.pdf
audiobook chapters
audiobook chunks --pdf /path/to/book.pdf --chapter 1
```

`audiobook chapters` prints the 1-based index, the title, and the page range. `audiobook chunks` prints the 0-based chunk index, estimated seconds, speakers, and the opening words. Use a chunk index whose speakers are not only the narrator when you want the sample to include dialogue.

### Estimate before you spend

```bash
audiobook estimate --pdf /path/to/book.pdf --chapter 1 --sample-seconds 60
audiobook run --pdf /path/to/book.pdf --chapter 1 --style "even and clear" --dry-run
audiobook run --pdf /path/to/book.pdf --chapter 1 --style "even and clear" --max-chunks 5 --dry-run
audiobook sample --pdf /path/to/book.pdf --chapter 1 --style "even and clear" --sample-seconds 60 --dry-run
```

The lines are chunk count, estimated audio minutes, and estimated TTS cost in USD from `default.toml`. Prices there are a dated copy of the public Gemini price for `gemini-3.8-flash-tts`. Change them when the price page changes. Text-prep image tokens are not included. `--sample-seconds` and `--max-chunks` limit the estimate to the chunks that would actually be spoken.

### One-minute sample

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

### Full chapter

```bash
audiobook run --pdf /path/to/book.pdf --chapter 1 --style profiles/examples/neutral-narration.yaml
audiobook review --chapter 1
```

The chapter WAV is `artifacts/audio/chapter-001.wav` (the number matches the chapter index). `audiobook review` reads the full-chapter prep outputs, not the sample review.

`--max-chunks 5` on `run` (without `--sample-seconds`) caps a chapter run and stores those chunks in the chapter cache. A later `run` of the same chapter continues with the chunks that are not finished yet. Use `--sample-seconds` when you want a listen that must not become part of that cache.

### Where files land

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

### Run one stage, or resume

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

**Chapter breaks look wrong.** Run `audiobook chapters --project mybook` and compare titles with the pages in the PDF. The extractor uses the PDF outline when there is one, otherwise headings. Front matter is its own chapter when pages come before the first chapter heading. Details are in [docs/NOTES.md](docs/NOTES.md).

**Scanned or blank pages.** You will see a warning and a placeholder line. Turn on OCR as above, or skip that chapter.

**Prep failed for chunk …** Read the errors. The usual causes are an inline tag that is not on the allowlist, a style string that quotes the transcript, raw math left in the spoken text, or a figure or note the model dropped. Tighten the style profile, delete that chunk's file under `projects/<name>/prep/outputs/chunks/`, and run `prepare` again. Finished chunks stay finished.

**Rate limits.** The client already backs off on 429 and 5xx. If it still stops, lower `prep.concurrency` and `tts.concurrency` (in `overrides` in `project.json`, or `default.toml`) and run the same command again. Finished chunks and clips are skipped.

**The voice sounds robotic.** Try another prebuilt voice (`Charon`, `Sulafat`, `Schedar`) with a 60-second sample. Add a short line to `style.md` such as `warm, unhurried, a little smile in the voice`. Do not pile tags onto every sentence.

**The voice sounds over-acted.** Put `Prefer an empty style on ordinary sentences` in `style.md`, or lower `expressiveness`. Keep the style notes short. Empty style is what the speech model expects for a plain sentence. Design a calmer `voice_...` id if the prebuilt voice itself is too bright.

**Dialogue is all one voice.** Set `character_voices` in `project.json` for the speaker labels `audiobook chunks --project mybook --chapter N` prints. Only two prebuilt voices can share one request. A library id or a `voice_...` id is spoken one turn at a time.

## Tests

```bash
pytest
pytest -m live
```

`pytest` does not call Gemini. `pytest -m live` runs one short text-prep call and one short speech call with the default prebuilt voice, and checks that the audio is 24 kHz mono 16-bit WAV-compatible PCM. It is skipped unless `GEMINI_API_KEY` is set, and it spends a small amount of quota.

## License

Copyright 2026 Valentyn Shybanov. Licensed under the [Apache License, Version 2.0](LICENSE).

## Further reading

- [docs/STYLE_PROFILE.md](docs/STYLE_PROFILE.md) — profile fields
- [docs/NOTES.md](docs/NOTES.md) — extraction failure modes and the cost formula
- [docs/GEMINI_API.md](docs/GEMINI_API.md) — model ids, speech fields, and voice design
- `src/expressive_audiobook/default.toml` — models, voices, retries, and prices (to change it for every run, copy it to `config/default.toml` in your working folder, or pass `--config`)
