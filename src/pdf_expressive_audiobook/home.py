"""Where user-wide files live: ``~/.audiobook`` (override with ``$AUDIOBOOK_HOME``).

    projects/   one folder per book
    library/    generated music and sound effects shared by every book
    models/     local model installs (Stable Audio 3, ...)
"""

from __future__ import annotations

import os
from pathlib import Path


def audiobook_home() -> Path:
    return Path(os.environ.get("AUDIOBOOK_HOME") or Path.home() / ".audiobook").expanduser()


def models_dir() -> Path:
    return audiobook_home() / "models"


def library_dir() -> Path:
    return Path(os.environ.get("AUDIOBOOK_LIBRARY") or audiobook_home() / "library").expanduser()


def default_projects_dir() -> Path:
    """``./projects`` keeps working while it exists and the home folder has none yet."""
    home = audiobook_home() / "projects"
    legacy = Path("projects")
    if legacy.is_dir() and not home.exists():
        return legacy
    return home
