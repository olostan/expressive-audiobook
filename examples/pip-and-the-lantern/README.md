# Pip and the Lantern

A short bedtime story (two chapters, about four minutes) made with expressive-audiobook, with background music and sound effects.

▶️ [Listen to audio sample (MP3)](https://github.com/olostan/expressive-audiobook/raw/main/examples/pip-and-the-lantern/pip-and-the-lantern.mp3)

| File | What it is |
| --- | --- |
| [pip-and-the-lantern.mp3](pip-and-the-lantern.mp3) | The result: the whole story, narrated, with music and effects |
| [pip-and-the-lantern.md](pip-and-the-lantern.md) | The source text (a Markdown file; the headings become chapters) |
| [style.md](style.md) | The narration style that `import` wrote for it, unedited |
| [settings.json](settings.json) | The delivery settings and soundscape used (the `narration`, `audio` and `soundscape` parts of `project.json`) |

It was made with the Gemini voice Schedar for speech and Stable Audio 3, run locally, for the music and sounds. Your result will sound a little different: speech and sounds are generated fresh each time.

To make your own copy:

```text
audiobook import --source examples/pip-and-the-lantern/pip-and-the-lantern.md --name pip --music 0.4 --sfx 0.5 --audience kids
audiobook generate --project pip --all
```

The story was written for this project and is covered by the repository's Apache-2.0 license.
