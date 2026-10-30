# Notes

Failure modes to expect on a real PDF, and a way to estimate cost before a live run. Prices that were copied from a public page are dated in [GEMINI_API.md](GEMINI_API.md). Anything else stays a parameter.

## Scanned pages

A page with no text layer and a large image is recorded on `chapter.scanned_pages`. With `ingest.ocr_enabled` false, the chapter gets a placeholder block (`Scanned page with no text layer (page N).`) and a warning. The placeholder is chunked so the page is not dropped silently.

OCR is a hook. Pass a callable in tests, or enable the config flag and install `pytesseract` plus a `tesseract` binary. If the import fails, the warning says so and the placeholder remains. OCR quality on dense notation, small footnotes, and rotated pages will be uneven. Review those chunks before synthesis.

## Ligatures and odd encodings

Extraction normalizes to NFC, strips NUL bytes and soft hyphens, and expands the common Latin ligatures (ff, fi, fl, ffi, ffl). A font that never encoded the ligature as a Unicode character will not be repaired by that table. Custom encodings, missing ToUnicode maps, and math fonts can still yield garbage. When a formula's text layer looks wrong, the cropped PNG is what the multimodal prep request sends.

Line-break hyphens (`recon-` at the end of a line, continued by a lowercase letter) are rejoined. Hyphens that are part of a compound (`well-known`) are kept. A discretionary hyphen that the extractor already turned into a real hyphen plus a line break can be mis-joined. Spot-check words that were hyphenated across lines.

## Formulas and typographic notation

Display math is detected from `$$...$$` or from a short line whose characters are mostly operators and TeX commands. Inline math is `$...$` inside a paragraph. Both keep the raw text and a cropped image.

The prep prompt tells the model to replace that with spoken math a listener can rebuild, and the validator rejects leftover `$...$`, backslash commands, and a set of Unicode math operators in the transcript. A formula that is only a picture, with no text layer and no nearby caption, can be missed. A line of prose that is unusually full of symbols can be classified as display math.

Typographic games that depend on letter shape, mirrored words, or layout will not survive a plain transcript. The style profile is the place to say how those moments should be handled. The pipeline does not special-case any book.

## Figure-heavy pages

Raster images come from image blocks. Vector figures are clusters of drawings, ignoring hairline rules and full-page borders. A caption matching `Figure N` or `Fig. N` is attached by horizontal overlap and vertical distance. Each region is rendered to PNG at `ingest.image_dpi` (default 144) and stored next to the chapter.

A full-page image on a textless page is a scan, not a figure. A decorative rule is dropped. A diagram with no caption is still kept if the drawing cluster is large enough, anchored to the nearest preceding text block. The prep model is told to say one to three sentences, starting from the caption. The validator enforces the sentence and word caps in config. An image past `prep.max_images_per_request` is named in the prompt text instead of inlined, so it is not silent, but the model does not see its pixels.

## Chunk boundaries in dialogue

Chunks target 30–50 seconds at the configured words per minute and must stay strictly under 60 seconds, except for one unsplittable sentence, which is kept intact and warned. Splits are on paragraph, sentence, or whole dialogue turn. A turn that fits under the cap is never cut mid-sentence. Alternating speakers can therefore land in different chunks, and a short reply can sit alone at the end of a chapter.

Footnotes, figures, and formulas that would blow the cap move to the next chunk. The original chunk keeps a non-speaking reference so the anchor is visible. The speaking chunk is the one the validator requires a description for. A huge footnote read in full will sound like an interruption. The profile's `verbalize.footnotes` guidance and the model's `read-full` / `mention` / `skip-with-reason` decision are the control. Review those decisions before paying for audio.

Neighbor chunks are passed into the prep prompt as read-only context. They are not transcribed. `prep.neighbor_chunks` (default 1) is how many chunks on each side are included. A joke whose setup sits further away than that window can still be missed.

## Cost and latency

Use your own word count from the chunk JSONL (`word_count` already includes description budgets).

```
audio_seconds ≈ sum(chunk.estimated_duration_sec)
audio_tokens  ≈ audio_seconds * audio_tokens_per_second
tts_output    ≈ audio_tokens / 1e6 * tts_output_price_per_million
tts_input     ≈ tts_input_tokens / 1e6 * tts_input_price_per_million
```

`estimate_tts_cost` and `estimate_latency` in `expressive_audiobook.tts` implement that. Defaults in config, copied from the Gemini pricing page on 2026-10-24 for `gemini-3.8-flash-tts` standard tier through 2026-12-31:

- 25 audio tokens per second
- $0.50 per 1M text input tokens
- $9.00 per 1M audio output tokens
- equivalent, on that page, to $0.00225 per 10 seconds of audio

From 2027-01-01 the same page lists $1.00 and $18.00. Batch, flex, and priority prices differ. Override the config rather than trusting a stale default.

Text-prep cost is separate: prompt tokens plus image tokens. Image-token accounting was not pinned down for `gemini-3.8-flash` in this repo, so leave `prep_price_per_million` and tokens-per-image as parameters you measure from one real response.

Latency is `prep_requests * seconds_per_prep_request + tts_requests * seconds_per_tts_request`. The default 6s and 8s figures are placeholders, not a measured API latency. A two-speaker prebuilt dialogue is one TTS request. A narrator turn, or any library, design, or replication voice, is one request per turn. That is the main driver of call count.

Run `audiobook review` on a chapter before a full synthesis pass. Rationales and description decisions are the cheapest place to retune the profile.
