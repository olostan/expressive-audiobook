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
from .export import ExportError, convert, ffmpeg_path, target_path
from .ingest import load_manifest
from .ingest_text import is_text_source
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
    fingerprint,
    projects_dir,
    render_style_md,
)
from .library import Library
from .score import (
    ChapterMusic,
    SoundSet,
    audience_of,
    chapter_seed,
    cue_budget,
    ensure_cues,
    ensure_music,
    level_db,
    music_wanted,
    numbered_turns,
    plan_music,
    plan_sfx,
    sfx_wanted,
)
from .soundgen import SoundGenError
from .style_detect import assign_character_voices, book_features, detect_style, fallback_style
from .timeline import chapter_timeline
from .tts import plan_tts_requests
from .tuning import level
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
    music: float | None = None,
    sfx: float | None = None,
    audience: str | None = None,
) -> Project:
    pdf = Path(pdf).expanduser()
    if not pdf.exists():
        raise ProjectError(f"File not found: {pdf}")
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
                    f"Project '{name}' was imported from a different file. "
                    "Use --force to replace it (generated files are deleted), or pick another --name."
                )
            for folder in GENERATED:
                shutil.rmtree(root / folder, ignore_errors=True)
            project.data.update(source_pdf=str(pdf.resolve()), source_sha256=digest)
            restyle = True
    config = project.config(base_config)

    manifest_path = project.ingest_dir / "manifest.json"
    if manifest_path.exists() and not force:
        ui.ok("book already extracted (same file); use --force to extract again")
    else:
        ui.info("🔎 reading chapters…" if is_text_source(pdf) else "🔎 reading pages, chapters, figures, formulas, footnotes…")
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
                    project.ingest_dir,
                    config,
                    client,
                    pages,
                    project.data["narration"]["voice"],
                    model=project.data["style"].get("model"),
                )
                ai_used = True
            except Exception as exc:
                ui.warn(f"style analysis failed ({exc}); writing a neutral style instead")
        else:
            ui.warn("GEMINI_API_KEY is not set; writing a neutral style you can edit")
        detected = detected or fallback_style(features)
        _write_style(project, detected)
        _apply_recommendations(project, detected.get("recommended") or {})
        _confirm_tuning(ui, project)
        _choose_soundscape(ui, project, detected, music=music, sfx=sfx, audience=audience)
        if not project.data.get("character_voices"):
            project.data["character_voices"] = assign_character_voices(
                project.data["narration"]["voice"],
                list(features.get("speaker_counts") or {}),
                {p.get("name"): p.get("voice") for p in detected.get("speaker_profiles") or [] if isinstance(p, dict)}
                or detected.get("speaker_voices"),
            )
        ui.ok(f"wrote {project.style_path}")
    project.save()
    _report_import(ui, project, manifest, features, first_time_style, ai_used)
    return project


def _write_style(project: Project, detected: dict) -> None:
    front = {
        "id": project.name,
        "narrator_persona": " ".join(str(detected.get("narrator_persona") or "").split()),
        "default_delivery_style": " ".join(str(detected.get("default_delivery_style") or "").split()),
        "literary_traits": [str(x) for x in detected.get("literary_traits") or []],
        "pronunciation_lexicon": [
            {"word": str(x.get("word", "")), "ipa": str(x.get("ipa", ""))}
            for x in detected.get("pronunciation_lexicon") or []
            if isinstance(x, dict) and x.get("word")
        ],
        "verbalize": {str(k): " ".join(str(v).split()) for k, v in (detected.get("verbalize") or {}).items()},
    }
    if project.style_path.exists():
        shutil.copyfile(project.style_path, project.style_path.with_name("style.previous.md"))
    atomic_write(project.style_path, render_style_md(front, _style_body(detected)))


def _style_body(detected: dict) -> str:
    sections = []
    head = []
    if detected.get("genre"):
        head.append(f"**Genre:** {detected['genre']}")
    if detected.get("tone"):
        head.append(f"**Tone:** {detected['tone']}")
    if detected.get("summary"):
        head.append(str(detected["summary"]).strip())
    sections.append("\n\n".join(head))
    profiles = [p for p in detected.get("speaker_profiles") or [] if isinstance(p, dict) and p.get("name")]
    if profiles:
        lines = ["**Recurring Speakers & Vocal Profiles:**", ""]
        for p in profiles:
            cue = f" *Vocal cue: {p['vocal_cue'].strip()}*" if p.get("vocal_cue") else ""
            lines.append(f"- **{p['name']}:** {str(p.get('description') or '').strip()}{cue}")
        sections.append("\n".join(lines))
    elif detected.get("speakers"):
        sections.append("**Recurring speakers:** " + ", ".join(str(x) for x in detected["speakers"]))
    directives = [d for d in detected.get("performance_directives") or [] if isinstance(d, dict) and d.get("how")]
    if directives:
        lines = ["---", "", "### Performance & Pacing Directives", ""]
        for number, d in enumerate(directives, start=1):
            title = str(d.get("title") or "").strip() or "Direction"
            when = f" *({str(d['when']).strip()})*" if d.get("when") else ""
            lines.append(f"{number}. **{title}**{when}")
            lines.append(f"   {str(d['how']).strip()}")
            lines.append("")
        sections.append("\n".join(lines).rstrip())
    if detected.get("notes"):
        sections.append(str(detected["notes"]).strip())
    return "\n\n".join(s for s in sections if s.strip())


def _apply_recommendations(project: Project, recommended: dict) -> None:
    narration = project.data["narration"]
    for key in ("expressiveness", "pace", "pause_density"):
        try:
            narration[key] = level(key, recommended[key])
        except (KeyError, ValueError):
            pass


def _confirm_tuning(ui: Ui, project: Project) -> None:
    """Show the suggested delivery and let the listener adjust it. Silent when nobody is there to ask."""
    if not ui.interactive:
        return
    narration = project.data["narration"]
    ui.banner("🎛️", "Suggested delivery")
    ui.kv("", f"expressiveness={narration['expressiveness']}  pace={narration['pace']}  pause_density={narration['pause_density']}")
    if ui.confirm("Use these?", True):
        return
    for key, hint in (("expressiveness", "0 plain reading, 1 fully performed"), ("pace", "0 slowest, 0.5 conversational, 1 fastest"),
                      ("pause_density", "0 punctuation only, 1 frequent pauses")):
        narration[key] = level(key, ui.ask_level(f"{key} ({hint})", float(narration[key])))


AUDIENCES = ("kids", "young_adult", "general", "technical")


def _choose_soundscape(ui: Ui, project: Project, detected: dict, *, music, sfx, audience) -> None:
    """Music and effects cost generation time (or money) and not every book wants them, so they start off.

    Flags set them directly. Otherwise, when someone is at the keyboard, show what the book suggests and ask.
    """
    scape = project.data["soundscape"]
    suggested = detected.get("soundscape") if isinstance(detected.get("soundscape"), dict) else {}
    want_music = suggested.get("music") if isinstance(suggested.get("music"), dict) else {}
    want_sfx = suggested.get("sfx") if isinstance(suggested.get("sfx"), dict) else {}
    found = str(detected.get("audience") or "").strip().lower()
    scape["audience"] = audience or (found if found in AUDIENCES else scape.get("audience") or "general")
    if music is not None:
        scape["music"].update(enabled=music > 0, level=music)
    if sfx is not None:
        scape["sfx"].update(enabled=sfx > 0, intensity=sfx)
    if music is not None and sfx is not None:
        return
    if not ui.interactive:
        if want_music.get("recommended") or want_sfx.get("recommended"):
            ui.hint("This book suits background music/effects. Turn them on with: "
                    f"audiobook sound --project {project.name} --music 0.4 --sfx 0.3")
        return
    ui.banner("🎼", "Music and sound effects", f"audience: {scape['audience']}")
    scape["audience"] = ui.ask("Who is this for? (kids, young_adult, general, technical)", scape["audience"]) or scape["audience"]
    if music is None:
        reason = want_music.get("reason") or ""
        if reason:
            ui.info(f"Suggestion: {reason}")
        if ui.confirm("Add soft background music? (generated locally; costs time, not money)", bool(want_music.get("recommended"))):
            hint = ui.ask("Style of music", str(want_music.get("style") or ""))
            level = ui.ask_level("How noticeable, from 0 (barely there) to 1 (very noticeable)?", 0.4)
            scape["music"].update(enabled=level > 0, level=level, style_hint=hint)
    if sfx is None:
        reason = want_sfx.get("reason") or ""
        if reason:
            ui.info(f"Suggestion: {reason}")
        if ui.confirm("Add occasional sound effects?", bool(want_sfx.get("recommended"))):
            level = ui.ask_level("How often and how noticeable, from 0 (none) to 1 (very noticeable)?", 0.4)
            scape["sfx"].update(enabled=level > 0, intensity=level)


def _report_import(ui: Ui, project: Project, manifest: dict, features: dict, styled: bool, ai: bool) -> None:
    chapters = manifest.get("chapters") or []
    ui.banner("✅", "Import complete", project.data.get("title") or project.name)
    ui.kv("chapters", f"{len(chapters)} ({manifest.get('detection')} detection)")
    if manifest.get("source_kind") in {"markdown", "text"}:
        ui.kv("source", f"{manifest['source_kind']} file")
    else:
        ui.kv("pages", manifest.get("page_count") or "?")
    skipped = manifest.get("skipped_pages") or []
    if skipped:
        pages = ", ".join(str(item["page"]) for item in skipped[:12])
        ui.kv("skipped", f"{len(skipped)} contents page(s): {pages}")
    ui.kv("narration", f"~{features['minutes_estimate']} min · {features['words']:,} words")
    if features["figures"] or features["formulas"] or features["footnotes"]:
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
    scape = project.data["soundscape"]
    ui.kv("soundscape", f"music {scape['music']['level'] if scape['music']['enabled'] else 'off'} · "
                        f"effects {scape['sfx']['intensity'] if scape['sfx']['enabled'] else 'off'} · audience {scape.get('audience') or 'general'}")
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
        _sound_estimate(project, config, result["chapters"], result["chunks"], ui)
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

    outputs = _load_prep_outputs(config, None)
    sounds = _prepare_sounds(project, config, profile, prep_client, selected, plan, outputs, ui, force=force)

    ui.banner("🎧", "Assembling audio")
    files: list[Path] = []
    manifest = load_manifest(project.ingest_dir)
    titles = {int(item["index"]): item["title"][:100] for item in manifest["chapters"]}
    base_tags = {"album": project.data.get("title") or project.name, "artist": manifest.get("author") or ""}
    base_tags = {k: v for k, v in base_tags.items() if v}
    a = config.assemble
    jobs: dict[int, tuple[Path, list[Path], float, str, object]] = {}
    for index in selected:
        clips = _expected_clips(config, profile, plan[index], outputs)
        if sample:
            dest = project.samples_dir / f"chapter-{index:03d}-{sample:g}s.wav"
            tail = 0
        else:
            dest = Path(config.paths.audio) / f"chapter-{index:03d}.wav"
            tail = a.chapter_tail_ms
        missing = [path for path in clips if not path.exists()]
        if not clips or missing:
            ui.warn(f"chapter {index}: {len(missing)} clip(s) missing; not assembled")
            continue
        stamp = _stamp(config, [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in clips], tail)
        if sounds:
            stamp = fingerprint(stamp, _sound_signature(project, config, sounds.get(index)))
            build = lambda d=dest, c=clips, t=tail, i=index: _build_mixed(
                project, config, profile, plan[i], outputs, c, d, t, sounds.get(i), i
            )
        else:
            build = lambda d=dest, c=clips, t=tail: _build(config, c, d, t)
        jobs[index] = (dest, clips, tail, stamp, build)
        tags = {"title": titles.get(index, f"Chapter {index}"), "track": str(index), **base_tags}
        made = _publish(ui, config, dest, stamp, build, tags)
        files.extend(made)
        ui.ok(f"chapter {index}: {made[-1]}")
    if not sample and chapters is None and len(jobs) > 1 and a.write_full_book:
        book = Path(config.paths.audio) / "book.wav"
        stamp = fingerprint([job[3] for job in jobs.values()], a.gap_ms, a.crossfade_ms)

        def build_book() -> None:
            for dest, clips, tail, _, build in jobs.values():  # a dropped chapter WAV is rebuilt from its clips
                if not dest.exists():
                    build()
            if sounds:
                from .mixer import decode, join_chapters, write_wav as write_stereo

                pieces = [decode(job[0], config.sound.sample_rate) for job in jobs.values()]
                write_stereo(book, join_chapters(pieces), config.sound.sample_rate)
            else:
                _build(config, [job[0] for job in jobs.values()], book, 0)

        made = _publish(ui, config, book, stamp, build_book, {"title": base_tags.get("album", project.name), **base_tags})
        files.extend(made)
        for dest, *_ in jobs.values():  # chapter WAVs rebuilt only for the book are not kept either
            if not a.keep_wav and dest.exists() and convertible(config.assemble.format) and target_path(dest, a.format).exists():
                dest.unlink()
        ui.ok(f"whole book: {made[-1]}")
    ui.banner("🎉", "Done", f"{len(files)} file(s)")
    if sample:
        ui.hint("Like it? Drop --sample to generate the full chapter. Reuses everything already spoken.")
    else:
        ui.hint("Change project.json or style.md and run again: only the affected chunks are redone.")
    result["audio"] = files
    return result


def _stamp(config: AppConfig, sources: list, tail_ms: float) -> str:
    a = config.assemble
    return fingerprint(sources, a.gap_ms, a.crossfade_ms, tail_ms, config.tts.sample_rate)


def _build(config: AppConfig, sources: list[Path], dest: Path, tail_ms: float) -> None:
    a = config.assemble
    assemble_clips(
        sources, dest, sample_rate=config.tts.sample_rate, gap_ms=a.gap_ms,
        crossfade_ms=a.crossfade_ms, tail_ms=tail_ms,
    )


def convertible(fmt: str) -> bool:
    return fmt != "wav" and ffmpeg_path() is not None


def _publish(ui: Ui, config: AppConfig, wav: Path, stamp: str, build, tags: dict) -> list[Path]:
    """Make ``wav`` (if its inputs changed), write the MP3/M4A, and drop the WAV unless keep_wav.

    Returns the files that exist afterwards. A WAV removed earlier is not rebuilt while the
    compressed copy is still current.
    """
    a = config.assemble
    marker = wav.with_name(wav.name + ".fp")
    fresh = marker.exists() and marker.read_text(encoding="utf-8").strip() == stamp
    fmt = a.format
    packed = target_path(wav, fmt) if convertible(fmt) else None
    if packed and not a.keep_wav and fresh and packed.exists() and not wav.exists():
        return [packed]
    if not (fresh and wav.exists()):
        build()
        atomic_write(marker, stamp + "\n")
    out = _compress(ui, config, wav, tags)
    if out is None:
        return [wav]
    if not a.keep_wav:
        wav.unlink()
        return [out]
    return [wav, out]


def _compress(ui: Ui, config: AppConfig, wav: Path, tags: dict) -> Path | None:
    """Also write MP3/M4A when the project asks for it and ffmpeg exists."""
    fmt = config.assemble.format
    if fmt == "wav":
        return None
    if ffmpeg_path() is None:
        if not getattr(ui, "ffmpeg_warned", False):
            ui.warn(f"ffmpeg not found, so only WAV files were written. Install it (brew install ffmpeg) to get {fmt}.")
            ui.ffmpeg_warned = True
        return None
    try:
        out = convert(wav, fmt, quality=config.assemble.mp3_quality, tags=tags)
    except ExportError as exc:
        ui.warn(str(exc))
        return None
    if out and out.stat().st_mtime >= wav.stat().st_mtime:
        ui.info(f"🎵 {out.name}  {wav.stat().st_size / 1e6:.1f} MB → {out.stat().st_size / 1e6:.1f} MB")
    return out


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


# ---------------------------------------------------------------- music and sound effects

SA3_MUSIC_SPEED = 6.0  # seconds of music made per second on Apple Silicon (measured)


def _sound_estimate(project: Project, config: AppConfig, chapters: list[int], chunks: list, ui: Ui) -> None:
    music, sfx = music_wanted(project), sfx_wanted(project)
    if not (music or sfx):
        return
    s = config.sound
    if music:
        tracks = len(chapters) * s.music_variants
        minutes = tracks * s.music_seconds / SA3_MUSIC_SPEED / 60
        ui.kv("music", f"up to {tracks} tracks of {s.music_seconds:g}s ({s.music_provider}; ~{minutes:.0f} min to generate, reused from the library when possible)")
    if sfx:
        words = sum(c.word_count for c in chunks)
        ui.kv("effects", f"up to ~{cue_budget(words, sfx)} cues at intensity {sfx} ({s.sfx_provider})")


def _prepare_sounds(project, config, profile, client, selected, plan, outputs, ui, *, force=False, library=None, providers=None):
    """Plan and generate the music and effects for the chapters. Failures only cost the extras, never the audio."""
    music_level, sfx_level = music_wanted(project), sfx_wanted(project)
    if not (music_level or sfx_level):
        return {}
    if client is None:
        ui.warn("music and sound effects need the Gemini key to plan them; continuing without")
        return {}
    library = library or Library()
    music_provider, sfx_provider = (providers or (None, None))
    ui.banner("🎼", "Soundscape", f"music {music_level or 'off'} · effects {sfx_level or 'off'} · audience {audience_of(project)}")
    manifest = load_manifest(project.ingest_dir)
    titles = {int(item["index"]): item["title"] for item in manifest["chapters"]}
    music_plan = None
    if music_level:
        try:
            music_plan = plan_music(project, config, client, library, force=force)
        except Exception as exc:
            ui.warn(f"could not plan the music ({exc}); continuing without it")
    result = {}
    for index in selected:
        music = ChapterMusic(index, False)
        cues = []
        if music_plan:
            try:
                music = ensure_music(project, config, music_plan, library, index, ui=ui, provider=music_provider)
            except (SoundGenError, RuntimeError, ValueError) as exc:
                ui.warn(f"chapter {index}: no music ({exc})")
        if sfx_level:
            try:
                turns = numbered_turns(plan[index], outputs)
                raw = plan_sfx(project, config, client, library, index, titles.get(index, f"Chapter {index}"), turns, force=force)
                cues = ensure_cues(project, config, raw, library, ui=ui, provider=sfx_provider)
            except (SoundGenError, RuntimeError, ValueError) as exc:
                ui.warn(f"chapter {index}: no sound effects ({exc})")
        result[index] = SoundSet(music, cues)
        ui.ok(f"chapter {index}: {len(music.tracks)} music track(s), {len(cues)} effect(s)")
    return result


def _sound_signature(project: Project, config: AppConfig, sounds) -> str:
    if sounds is None:
        return "none"
    return fingerprint(
        [t.id for t in sounds.music.tracks],
        music_wanted(project),
        [(c.asset.id, c.at, c.when, c.kind, c.through) for c in sounds.cues],
        config.sound.__dict__,
    )


def _build_mixed(project, config, profile, chunks, outputs, clips, dest, tail_ms, sounds, chapter) -> None:
    """Assemble the narration, then lay music and effects under it and write a 44.1 kHz stereo WAV."""
    import tempfile

    from .mixer import Ambience, MixPlan, Spot, decode, mix_chapter, write_wav as write_stereo

    s = config.sound
    library = Library()
    with tempfile.TemporaryDirectory() as scratch:
        dry_path = Path(scratch) / "dry.wav"
        _build(config, clips, dry_path, tail_ms)
        dry = decode(dry_path, s.sample_rate)
    timeline = chapter_timeline(
        config, profile, chunks, outputs,
        lambda chunk, i: _clip_path(config, chunk.chapter_index, chunk.order, chunk.id, i),
    )
    plan = MixPlan(spans=timeline.spans, seed=chapter_seed(project, chapter))
    if sounds is not None and sounds.music.tracks and music_wanted(project):
        plan.music = [decode(library.path(track), s.sample_rate) for track in sounds.music.tracks]
        plan.music_db = level_db(music_wanted(project), config)
    for cue in (sounds.cues if sounds is not None else []):
        first = timeline.turn_span(cue.at)
        if first is None:
            continue
        audio = decode(library.path(cue.asset), s.sample_rate)
        if cue.kind == "ambience":
            last = timeline.turn_span(cue.through or cue.at)
            plan.ambiences.append(Ambience(first[0], (last or first)[1], audio, s.ambience_db))
        else:
            at = max(0.0, first[0] - 0.3) if cue.when == "before" else first[1] + 0.15
            plan.spots.append(Spot(at, audio, s.sfx_db))
    write_stereo(dest, mix_chapter(dry, plan, s, rate=s.sample_rate), s.sample_rate)
