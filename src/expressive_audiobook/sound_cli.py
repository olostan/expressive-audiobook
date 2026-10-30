"""Commands for the sound library, sound settings, and local model setup."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from .config import AppConfig
from .library import KINDS, STATUSES, Asset, Library, LibraryError, asset_id
from .project import Project, ProjectError
from .soundgen import SoundGenError, SoundRequest, install_sa3, make_provider
from .ui import Ui
from .workflow import AUDIENCES


def add_parsers(sub) -> None:
    sound = sub.add_parser("sound", help="Show or change a project's background music and sound-effect settings")
    sound.add_argument("--music", type=float, default=None, metavar="0-1", help="Music level; 0 turns music off")
    sound.add_argument("--sfx", type=float, default=None, metavar="0-1", help="Sound-effect intensity; 0 turns effects off")
    sound.add_argument("--audience", choices=AUDIENCES, default=None)
    sound.add_argument("--music-style", default=None, help="Describe the music you would like, in your own words")

    library = sub.add_parser("library", help="The shared library of generated music and sound effects")
    lib = library.add_subparsers(dest="library_command", required=True)
    listing = lib.add_parser("list", help="List items")
    listing.add_argument("--kind", choices=KINDS, default=None)
    listing.add_argument("--status", choices=STATUSES, default=None)
    for name, helptext in (("approve", "Mark an item approved"), ("reject", "Never mix this item"), ("rm", "Delete an item"),
                           ("show", "Show an item's details"), ("play", "Play an item (macOS afplay)")):
        item = lib.add_parser(name, help=helptext)
        item.add_argument("id")
    add = lib.add_parser("add", help="Add your own audio file")
    add.add_argument("file")
    add.add_argument("--kind", choices=KINDS, required=True)
    add.add_argument("--title", default="")
    add.add_argument("--prompt", default="", help="Describe the sound; used when choosing it for a book")
    add.add_argument("--mood", default="")
    add.add_argument("--category", default="")
    add.add_argument("--loop", action="store_true")
    make = lib.add_parser("generate", help="Make a new item from a description")
    make.add_argument("--kind", choices=KINDS, required=True)
    make.add_argument("--prompt", required=True)
    make.add_argument("--seconds", type=float, default=None)
    make.add_argument("--seed", type=int, default=1)
    make.add_argument("--loop", action="store_true")

    models = sub.add_parser("models", help="Install local models")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    install = models_sub.add_parser(
        "install", help="Install Stable Audio 3 (music and sound effects, about 8 GB) or Orpheus (local speech, about 7 GB)"
    )
    install.add_argument("name", choices=["sa3", "orpheus"])
    install.add_argument("--bundles", default="sm-music,sm-sfx,medium")


def run_sound(args, project: Project, ui: Ui) -> int:
    scape = project.data["soundscape"]
    changed = False
    if args.music is not None:
        scape["music"].update(enabled=args.music > 0, level=args.music)
        changed = True
    if args.sfx is not None:
        scape["sfx"].update(enabled=args.sfx > 0, intensity=args.sfx)
        changed = True
    if args.audience:
        scape["audience"] = args.audience
        changed = True
    if args.music_style is not None:
        scape["music"]["style_hint"] = args.music_style
        changed = True
    for value in (args.music, args.sfx):
        if value is not None and not 0 <= value <= 1:
            raise ProjectError("--music and --sfx are numbers from 0 to 1")
    if changed:
        project.save()
    music, sfx = scape["music"], scape["sfx"]
    ui.banner("🎼", f"Soundscape of {project.name}")
    ui.kv("music", f"{music['level']}" if music["enabled"] else "off")
    if music.get("style_hint"):
        ui.kv("", f"style: {music['style_hint']}")
    ui.kv("effects", f"{sfx['intensity']}" if sfx["enabled"] else "off")
    ui.kv("audience", scape.get("audience") or "general")
    if changed:
        ui.hint(f"Applies next time you run: audiobook generate --project {project.name} --chapter N")
    return 0


def run_library(args, config: AppConfig, ui: Ui, library: Library | None = None) -> int:
    library = library or Library()
    command = args.library_command
    if command == "list":
        rows = [
            [a.id, a.kind, a.status, f"{a.seconds:.0f}s", (a.title or a.prompt)[:48]]
            for a in library.all(args.kind)
            if not args.status or a.status == args.status
        ]
        if not rows:
            print(f"The library at {library.root} has nothing matching.")
            return 0
        ui.table(["id", "kind", "status", "length", "description"], rows)
        return 0
    if command in {"approve", "reject"}:
        asset = library.set_status(args.id, "approved" if command == "approve" else "rejected")
        ui.ok(f"{asset.id}: {asset.status}")
        return 0
    if command == "rm":
        library.remove(args.id)
        ui.ok(f"removed {args.id}")
        return 0
    if command in {"show", "play"}:
        asset = library.get(args.id)
        if asset is None:
            raise LibraryError(f"No library item {args.id!r}")
        if command == "show":
            for key, value in vars(asset).items():
                ui.kv(key, value)
            ui.kv("path", library.path(asset))
            return 0
        player = "afplay" if sys.platform == "darwin" else "ffplay"
        subprocess.run([player, str(library.path(asset))])
        return 0
    if command == "add":
        source = Path(args.file).expanduser()
        ident = asset_id(args.kind, args.prompt or args.title or source.stem, "user", source.name, None, 0, args.loop)
        asset = Asset(ident, args.kind, args.prompt or args.title or source.stem, args.title or source.stem, mood=args.mood,
                      category=args.category, loop=args.loop, provider="user", model=source.name, status="approved")
        from .mixer import decode

        asset.seconds = len(decode(source)) / 44100
        library.add(source, asset)
        ui.ok(f"added {asset.id}")
        return 0
    if command == "generate":
        s = config.sound
        seconds = args.seconds or (s.music_seconds if args.kind == "music" else s.sfx_max_seconds)
        provider = make_provider(args.kind, s)
        ident = asset_id(args.kind, args.prompt, provider.name, provider.model, args.seed, seconds, args.loop)
        existing = library.get(ident)
        if existing:
            ui.ok(f"already in the library: {existing.id}")
            return 0
        negative = s.music_negative_prompt if args.kind == "music" else ""
        with tempfile.TemporaryDirectory() as scratch:
            raw = Path(scratch) / "raw.wav"
            provider.generate(SoundRequest(args.kind, args.prompt, seconds, args.seed, args.loop, negative), raw)
            asset = library.add(raw, Asset(ident, args.kind, args.prompt, seconds=seconds, loop=args.loop, provider=provider.name,
                                            model=provider.model, seed=args.seed))
        ui.ok(f"{asset.id}  (candidate; listen with: audiobook library play {asset.id})")
        return 0
    raise SoundGenError(f"Unknown library command {command}")


def run_models(args, config: AppConfig, ui: Ui) -> int:
    if args.name == "orpheus":
        from .speech_local import install_orpheus

        ui.banner("📦", "Installing Orpheus", "about 7 GB of model files; needs Apple Silicon")
        root = install_orpheus(config.tts.orpheus)
        ui.ok(f"installed at {root}")
        ui.kv("use it", 'set "overrides": {"tts": {"provider": "orpheus"}} in project.json')
        return 0
    ui.banner("📦", "Installing Stable Audio 3", "about 8 GB of model files")
    root = install_sa3(config.sound, bundles=args.bundles)
    ui.ok(f"installed at {root}")
    return 0
