"""The project workflow: import, prepare, generate, status.

Every step is safe to repeat. Extraction is skipped when the PDF is unchanged,
preparation and speech are skipped per chunk when their inputs are unchanged, and
chapter audio is rebuilt only from a complete set of clips.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .audio import assemble_clips
from .chunking import load_chapter_jsonl
from .config import AppConfig
from .ingest import load_manifest
from .pipeline import (
    _clip_path,
    _load_prep_outputs,
    format_estimate,
    run_chunk,
    run_extract,
    run_prep_model,
    run_review,
    run_tts_model,
    select_chunks,
)
from .project import (
    PROJECT_FILE,
    Project,
    ProjectError,
    atomic_write,
    file_sha256,
    projects_dir,
    render_style_md,
)
from .style_detect import assign_character_voices, book_features, detect_style, fallback_style
from .tts import plan_tts_requests
from .tuning import EXPRESSIVENESS, PACE, PAUSES
from .ui import Ui

GENERATED = ("ingest", "chunks", "prep", "tts", "audio", "samples")


def parse_chapters(spec: str | None) -> list[int] | None:
    """``None``/``all`` selects everything. Otherwise ``3``, ``3-5``, ``1,4,7-8``."""
    if spec is None or spec.strip().lower() == "all":
        return None
    found: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                low, high = (int(x) for x in part.split("-", 1))
                found.extend(range(low, high + 1))
            else:
                found.append(int(part))
        except ValueError:
            raise ProjectError(f"Bad chapter selection {spec!r}. Use 3, 3-5, 1,4,7 or all") from None
    return sorted(set(found))


# ---------------------------------------------------------------- import


def import_book(
    pdf: Path,
    name: str,
    base_config: AppConfig,
    *,
    ui: Ui,
    client=None,
    base_dir: Path | None = None,
    force: bool = False,
    restyle: bool = False,
    style_pages: int | None = None,
) -> Project:
    pdf = Path(pdf).expanduser()
    if not pdf.exists():
        raise ProjectError(f"PDF not found: {pdf}")
    base = projects_dir(base_dir)
    root = base / name
    fresh = not (root / PROJECT_FILE).exists()
    ui.banner("📖", f"Importing {pdf.name}", f"→ {root}")
    if fresh:
        project = Project.create(base, name, pdf)
    else:
        project = Project.load(root)
        digest = file_sha256(pdf)
        if digest != project.data.get("source_sha256"):
            if not force:
                raise ProjectError(
                    f"Project '{name}' was imported from a different PDF. "
                    "Use --force to replace it (generated files are deleted), or pick another --name."
                )
            for folder in GENERATED:
                shutil.rmtree(root / folder, ignore_errors=True)
            project.data.update(source_pdf=str(pdf.resolve()), source_sha256=digest)
            restyle = True
    config = project.config(base_config)

    manifest_path = project.ingest_dir / "manifest.json"
    if manifest_path.exists() and not force:
        ui.ok("book already extracted (same PDF); use --force to extract again")
    else:
        ui.info("🔎 reading pages, chapters, figures, formulas, footnotes…")
        run_extract(pdf, config)
        ui.ok("extracted")
    manifest = load_manifest(project.ingest_dir)
    meta_title = str(manifest.get("title") or "")
    if not meta_title or "\\" in meta_title or meta_title.lower().endswith((".obd", ".doc", ".docx", ".indd", ".qxd")):
        meta_title = ""
    project.data["title"] = meta_title or project.data.get("title") or name
    features = book_features(project.ingest_dir)
    project.data["detected"] = features

    first_time_style = fresh or restyle or not project.style_path.exists()
    ai_used = False
    if first_time_style:
        ui.banner("🎨", "Detecting narration style", f"from the first {style_pages or project.data['style']['pages']} pages")
        pages = style_pages or int(project.data["style"]["pages"])
        project.data["style"]["pages"] = pages
        detected = None
        if client is not None:
            try:
                detected = detect_style(
                    project.ingest_dir, config, client, pages, project.data["narration"]["voice"]
                )
                ai_used = True
            except Exception as exc:
                ui.warn(f"style analysis failed ({exc}); writing a neutral style instead")
        else:
            ui.warn("GEMINI_API_KEY is not set; writing a neutral style you can edit")
        detected = detected or fallback_style(features)
        _write_style(project, detected)
        _apply_recommendations(project, detected.get("recommended") or {})
        if not project.data.get("character_voices"):
            project.data["character_voices"] = assign_character_voices(
                project.data["narration"]["voice"],
                list(features.get("speaker_counts") or {}),
                detected.get("speaker_voices"),
            )
        ui.ok(f"wrote {project.style_path}")
    project.save()
    _report_import(ui, project, manifest, features, first_time_style, ai_used)
    return project


def _write_style(project: Project, detected: dict) -> None:
    front = {
        "id": project.name,
        "narrator_persona": str(detected.get("narrator_persona") or "").strip(),
        "default_delivery_style": str(detected.get("default_delivery_style") or "").strip(),
        "literary_traits": [str(x) for x in detected.get("literary_traits") or []],
        "pronunciation_lexicon": [
            {"word": str(x.get("word", "")), "ipa": str(x.get("ipa", ""))}
            for x in detected.get("pronunciation_lexicon") or []
            if isinstance(x, dict) and x.get("word")
        ],
        "verbalize": {str(k): str(v) for k, v in (detected.get("verbalize") or {}).items()},
    }
    head = []
    if detected.get("genre"):
        head.append(f"**Genre:** {detected['genre']}")
    if detected.get("tone"):
        head.append(f"**Tone:** {detected['tone']}")
    if detected.get("summary"):
        head.append(str(detected["summary"]).strip())
    speakers = [str(x) for x in detected.get("speakers") or []]
    if speakers:
        head.append("**Recurring speakers:** " + ", ".join(speakers))
    notes = "\n\n".join(head + [str(detected.get("notes") or "").strip()]).strip()
    atomic_write(project.style_path, render_style_md(front, notes))


def _apply_recommendations(project: Project, recommended: dict) -> None:
    narration = project.data["narration"]
    for key, table in (("expressiveness", EXPRESSIVENESS), ("pace", PACE), ("pause_density", PAUSES)):
        value = recommended.get(key)
        if value in table:
            narration[key] = value


def _report_import(ui: Ui, project: Project, manifest: dict, features: dict, styled: bool, ai: bool) -> None:
    chapters = manifest.get("chapters") or []
    ui.banner("✅", "Import complete", project.data.get("title") or project.name)
    ui.kv("chapters", f"{len(chapters)} ({manifest.get('detection')} detection)")
    ui.kv("pages", manifest.get("page_count") or "?")
    skipped = manifest.get("skipped_pages") or []
    if skipped:
        pages = ", ".join(str(item["page"]) for item in skipped[:12])
        ui.kv("skipped", f"{len(skipped)} contents page(s): {pages}")
    ui.kv("narration", f"~{features['minutes_estimate']} min · {features['words']:,} words")
    ui.kv("content", f"{features['figures']} figures · {features['formulas']} formulas · {features['footnotes']} footnotes")
    voices = project.data.get("character_voices") or {}
    if voices:
        ui.kv("speakers", ", ".join(f"{name} → {voice}" for name, voice in voices.items()))
    elif features["dialogue_speakers"]:
        ui.kv("speakers", ", ".join(features["dialogue_speakers"]))
    if manifest.get("warnings"):
        ui.warn(f"{len(manifest['warnings'])} extraction warnings (see ingest/manifest.json)")
    ui.info("")
    for item in chapters[:10]:
        pages = ui.paint("p.%s-%s" % (item["page_start"], item["page_end"]), "dim")
        ui.info(f"{ui.paint(str(item['index']).rjust(3), 'cyan')}  {item['title'][:70]}  {pages}")
    if len(chapters) > 10:
        ui.info(ui.paint(f"… and {len(chapters) - 10} more (audiobook chapters)", "dim"))
    tuning = project.tuning()
    ui.banner("🎛️", "What you can change")
    ui.kv("project.json", f"{project.root / PROJECT_FILE}")
    ui.kv("", f"voice={project.data['narration']['voice']}  expressiveness={tuning['expressiveness']}  "
              f"pace={tuning['pace']}  pause_density={tuning['pause_density']}")
    ui.kv("", f"gap between passages={tuning['gap_ms']} ms  chapter ending silence={tuning['chapter_tail_ms']} ms")
    ui.kv("style.md", f"{project.style_path}" + (" (written by Gemini)" if ai else " (neutral, edit me)" if styled else " (kept)"))
    ui.info("")
    ui.hint(f"Review/tune the two files, then: audiobook prepare --project {project.name} --chapter N")
    ui.hint(f"Or hear something now:      audiobook generate --project {project.name} --chapter N --sample 30")


# ---------------------------------------------------------------- shared


def _selected_chapters(project: Project, config: AppConfig, spec: list[int] | None) -> list[int]:
    manifest = load_manifest(project.ingest_dir)
    known = [int(item["index"]) for item in manifest["chapters"]]
    if spec is not None:
        missing = [n for n in spec if n not in known]
        if missing:
            raise ProjectError(f"No chapter {', '.join(map(str, missing))}. Run: audiobook chapters --project {project.name}")
        known = spec
    run_chunk_all(config, known)
    return [n for n in known if _chunks(config, n)]


def run_chunk_all(config: AppConfig, indexes: list[int]) -> None:
    for index in indexes:
        run_chunk(config, chapter=index)


def _chunks(config: AppConfig, index: int):
    path = Path(config.paths.chunks) / f"{index:03d}.jsonl"
    return load_chapter_jsonl(path) if path.exists() else []


def _plan(config, profile, chapters, sample):
    """Chunks to work on, per chapter, honoring a sample length in seconds."""
    plan: dict[int, list] = {}
    for index in chapters:
        chunks = _chunks(config, index)
        if sample:
            chunks = select_chunks(chunks, seconds=sample)
        plan[index] = chunks
    return plan


# ---------------------------------------------------------------- prepare


def prepare_book(
    project: Project,
    base_config: AppConfig,
    prep_client,
    *,
    ui: Ui,
    chapters: list[int] | None,
    sample: float | None = None,
    force: bool = False,
    dry_run: bool = False,
    hint: bool = True,
) -> dict:
    config = project.config(base_config)
    profile = project.profile()
    selected = _selected_chapters(project, config, chapters)
    if not selected:
        raise ProjectError("Nothing to prepare: the selected chapters have no narratable text")
    plan = _plan(config, profile, selected, sample)
    every = [chunk for chunks in plan.values() for chunk in chunks]
    label = f"{len(selected)} chapter(s)" + (f", first {sample:g}s of each" if sample else "")
    ui.banner("✍️", "Preparing narration", label)
    ui.kv("tuning", _tuning_line(project))
    ui.kv("estimate", format_estimate(every, config).replace("\n", "  ").strip())
    if dry_run:
        ui.info("dry run: no API calls")
        return {"chapters": selected, "reviews": [], "chunks": every}
    progress = ui.progress("prep", len(every), "✍️")
    try:
        run_prep_model(
            config,
            profile,
            prep_client,
            chapters=selected,
            only_ids={chunk.id for chunk in every},
            force=force,
            progress=lambda status, name: progress.advance(status, name),
        )
    finally:
        progress.close()
    reviews = [run_review(config, chapter=index) for index in selected]
    ui.banner("🔍", "Review the narration")
    for path in reviews[:8]:
        ui.kv("", path)
    if len(reviews) > 8:
        ui.info(ui.paint(f"… {len(reviews) - 8} more in {Path(config.paths.prep) / 'review'}", "dim"))
    ui.info("Edit anything in prep/outputs/chunks/*.json to keep your own wording (unchanged inputs are never overwritten).")
    if hint:
        ui.hint(f"Next: audiobook generate --project {project.name} " + _chapter_flag(chapters))
    return {"chapters": selected, "reviews": reviews, "chunks": every}


# ---------------------------------------------------------------- generate


def generate_book(
    project: Project,
    base_config: AppConfig,
    prep_client,
    tts_client,
    *,
    ui: Ui,
    chapters: list[int] | None,
    sample: float | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    config = project.config(base_config)
    profile = project.profile()
    result = prepare_book(
        project, base_config, prep_client, ui=ui, chapters=chapters, sample=sample, force=force, dry_run=dry_run, hint=False
    )
    if dry_run:
        return result
    selected = result["chapters"]
    plan = _plan(config, profile, selected, sample)
    only_ids = {chunk.id for chunks in plan.values() for chunk in chunks}
    outputs = _load_prep_outputs(config, None)
    total = sum(
        len(plan_tts_requests(outputs[cid].get("turns") or [], profile, config))
        for cid in only_ids
        if cid in outputs
    )
    ui.banner("🎙️", "Synthesizing speech", f"voice {project.data['narration']['voice']}")
    progress = ui.progress("speech", total, "🎙️")
    try:
        run_tts_model(
            config,
            profile,
            tts_client,
            chapters=selected,
            only_ids=only_ids,
            force=force,
            progress=lambda status, name: progress.advance(status, name),
        )
    finally:
        progress.close()

    ui.banner("🎧", "Assembling audio")
    outputs = _load_prep_outputs(config, None)
    audio: list[Path] = []
    for index in selected:
        chunks = plan[index]
        clips = _expected_clips(config, profile, chunks, outputs)
        if sample:
            dest = project.samples_dir / f"chapter-{index:03d}-{sample:g}s.wav"
            tail = 0
        else:
            dest = Path(config.paths.audio) / f"chapter-{index:03d}.wav"
            tail = config.assemble.chapter_tail_ms
        missing = [path for path in clips if not path.exists()]
        if not clips or missing:
            ui.warn(f"chapter {index}: {len(missing)} clip(s) missing; not assembled")
            continue
        assemble_clips(
            clips,
            dest,
            sample_rate=config.tts.sample_rate,
            gap_ms=config.assemble.gap_ms,
            crossfade_ms=config.assemble.crossfade_ms,
            tail_ms=tail,
        )
        audio.append(dest)
        ui.ok(f"chapter {index}: {dest}")
    if not sample and chapters is None and len(audio) > 1 and config.assemble.write_full_book:
        book = Path(config.paths.audio) / "book.wav"
        assemble_clips(
            audio, book, sample_rate=config.tts.sample_rate, gap_ms=config.assemble.gap_ms,
            crossfade_ms=config.assemble.crossfade_ms,
        )
        audio.append(book)
        ui.ok(f"whole book: {book}")
    ui.banner("🎉", "Done", f"{len(audio)} file(s)")
    if sample:
        ui.hint("Like it? Drop --sample to generate the full chapter. Reuses everything already spoken.")
    else:
        ui.hint("Change project.json or style.md and run again: only the affected chunks are redone.")
    result["audio"] = audio
    return result


def _expected_clips(config, profile, chunks, outputs) -> list[Path]:
    paths: list[Path] = []
    for chunk in sorted(chunks, key=lambda c: c.order):
        output = outputs.get(chunk.id)
        if output is None:
            continue
        requests = plan_tts_requests(output.get("turns") or [], profile, config)
        for request_index in range(len(requests)):
            paths.append(_clip_path(config, chunk.chapter_index, chunk.order, chunk.id, request_index))
    return paths


# ---------------------------------------------------------------- status


def project_status(project: Project, base_config: AppConfig, *, ui: Ui) -> list[dict]:
    config = project.config(base_config)
    profile = project.profile()
    manifest = load_manifest(project.ingest_dir)
    outputs = _load_prep_outputs(config, None)
    rows = []
    for item in manifest["chapters"]:
        index = int(item["index"])
        chunks = _chunks(config, index)
        prepared = sum(1 for chunk in chunks if chunk.id in outputs)
        clips = _expected_clips(config, profile, chunks, outputs)
        have = sum(1 for path in clips if path.exists())
        done = (Path(config.paths.audio) / f"chapter-{index:03d}.wav").exists()
        rows.append(
            {
                "index": index,
                "title": item["title"],
                "chunks": len(chunks),
                "prepared": prepared,
                "clips": have,
                "clips_expected": len(clips),
                "audio": done,
            }
        )
    ui.banner("📊", f"Status of {project.name}", f"{len(rows)} chapters")
    table = []
    for row in rows:
        if not row["chunks"]:
            continue
        prep = _meter(ui, row["prepared"], row["chunks"])
        speech = _meter(ui, row["clips"], row["clips_expected"]) if row["clips_expected"] else ui.paint("—", "dim")
        table.append([str(row["index"]), row["title"][:44], prep, speech, "🎧" if row["audio"] else ""])
    if table:
        ui.table(["#", "chapter", "prepared", "spoken", ""], table)
    else:
        ui.info("No chunks yet. Run: audiobook prepare")
    return rows


def _meter(ui: Ui, have: int, total: int) -> str:
    text = f"{have}/{total}"
    return ui.paint(text, "green" if have == total and total else "yellow" if have else "dim")


def _tuning_line(project: Project) -> str:
    t = project.tuning()
    return f"expressiveness={t['expressiveness']} pace={t['pace']} pauses={t['pause_density']} voice={project.data['narration']['voice']}"


def _chapter_flag(chapters: list[int] | None) -> str:
    return "--all" if chapters is None else "--chapter " + ",".join(map(str, chapters))
