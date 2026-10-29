"""Terminal output: colors, emoji, and a thread-safe progress bar.

Colors and the live bar turn off when the stream is not a TTY, when NO_COLOR is
set, or when quiet is requested, so logs and tests stay plain.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time

RESET = "\033[0m"
CODES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}


class Ui:
    def __init__(
        self, stream=None, *, color: bool | None = None, quiet: bool = False,
        interactive: bool | None = None, input_fn=None,
    ):
        self.stream = stream if stream is not None else sys.stdout
        tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.tty = tty
        self.color = (tty and "NO_COLOR" not in os.environ) if color is None else color
        self.quiet = quiet
        self._lock = threading.Lock()
        self._input = input_fn or input
        # Questions are asked only when someone is there to answer them.
        self.interactive = (tty and sys.stdin.isatty()) if interactive is None else interactive

    def ask(self, question: str, default: str = "") -> str:
        """Ask for a line of text. Returns the default when not interactive or the answer is empty."""
        if not self.interactive:
            return default
        shown = f" [{default}]" if default else ""
        try:
            answer = self._input(f"   {self.paint('?', 'magenta', 'bold')} {question}{shown} ").strip()
        except EOFError:
            return default
        return answer or default

    def confirm(self, question: str, default: bool = False) -> bool:
        answer = self.ask(question + (" [Y/n]" if default else " [y/N]"), "").lower()
        if not answer:
            return default
        return answer.startswith("y")

    def ask_level(self, question: str, default: float) -> float:
        """A number from 0 to 1."""
        while True:
            answer = self.ask(question, f"{default:g}")
            try:
                value = float(answer)
            except ValueError:
                value = -1.0
            if 0.0 <= value <= 1.0:
                return value
            if not self.interactive:
                return default
            self.warn("Please enter a number from 0 to 1")

    def paint(self, text: str, *styles: str) -> str:
        if not self.color or not styles:
            return text
        return "".join(CODES[name] for name in styles) + text + RESET

    def _emit(self, line: str) -> None:
        if self.quiet:
            return
        with self._lock:
            self.stream.write(line + "\n")
            self.stream.flush()

    def banner(self, emoji: str, title: str, detail: str = "") -> None:
        rule = self.paint("─" * 3, "dim")
        text = f"\n{rule} {emoji}  {self.paint(title, 'bold', 'cyan')}"
        if detail:
            text += f"  {self.paint(detail, 'dim')}"
        self._emit(text)

    def info(self, message: str) -> None:
        self._emit(f"   {message}")

    def ok(self, message: str) -> None:
        self._emit(f"   {self.paint('✓', 'green', 'bold')} {message}")

    def warn(self, message: str) -> None:
        self._emit(f"   {self.paint('⚠', 'yellow', 'bold')}  {self.paint(message, 'yellow')}")

    def error(self, message: str) -> None:
        with self._lock:
            self.stream.write(f"{self.paint('✗', 'red', 'bold')} {self.paint(message, 'red')}\n")
            self.stream.flush()

    def hint(self, message: str) -> None:
        self._emit(f"   {self.paint('→', 'magenta')} {message}")

    def kv(self, key: str, value: object, width: int = 14) -> None:
        self._emit(f"   {self.paint(key.ljust(width), 'dim')} {value}")

    def table(self, headers: list[str], rows: list[list[str]]) -> None:
        widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
        head = "  ".join(self.paint(h.ljust(w), "bold") for h, w in zip(headers, widths))
        self._emit(f"   {head}")
        for row in rows:
            self._emit("   " + "  ".join(str(c).ljust(w) for c, w in zip(row, widths)))

    def progress(self, label: str, total: int, emoji: str = "▸") -> "Progress":
        return Progress(self, label, total, emoji)


class Progress:
    """Counts cached, done, and failed items. Safe to call from worker threads."""

    def __init__(self, ui: Ui, label: str, total: int, emoji: str):
        self.ui = ui
        self.label = label
        self.total = total
        self.emoji = emoji
        self.cached = 0
        self.done = 0
        self.failed = 0
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self._live = ui.tty and not ui.quiet
        if total == 0:
            ui.info(ui.paint("nothing to do", "dim"))
        else:
            self._draw("")

    @property
    def finished(self) -> int:
        return self.cached + self.done + self.failed

    def advance(self, status: str, name: str = "") -> None:
        with self._lock:
            if status == "cached":
                self.cached += 1
            elif status == "failed":
                self.failed += 1
            else:
                self.done += 1
            if self._live:
                self._draw(name)
            else:
                mark = {"cached": "✓ cached", "failed": "✗ failed"}.get(status, "↻ done")
                self.ui._emit(f"   [{self.finished}/{self.total}] {mark} {name}")

    def eta(self) -> float | None:
        worked = self.done + self.failed
        if worked == 0:
            return None
        remaining = self.total - self.finished
        return (time.monotonic() - self.started) / worked * remaining

    def _draw(self, name: str) -> None:
        if not self._live:
            return
        ui = self.ui
        width = max(10, min(28, shutil.get_terminal_size((80, 20)).columns // 4))
        fraction = self.finished / self.total if self.total else 1.0
        filled = int(width * fraction)
        bar = ui.paint("█" * filled, "green") + ui.paint("░" * (width - filled), "dim")
        eta = self.eta()
        tail = f" eta {_clock(eta)}" if eta is not None and self.finished < self.total else ""
        counts = f"{self.finished}/{self.total}"
        extras = []
        if self.cached:
            extras.append(ui.paint(f"{self.cached} cached", "dim"))
        if self.failed:
            extras.append(ui.paint(f"{self.failed} failed", "red"))
        line = f"\r\033[K   {self.emoji} {self.label} {bar} {counts}{tail}"
        if extras:
            line += "  " + " ".join(extras)
        if name:
            room = shutil.get_terminal_size((80, 20)).columns - len(_plain(line)) - 2
            if room > 8:
                line += "  " + ui.paint(name[:room], "dim")
        ui.stream.write(line)
        ui.stream.flush()

    def close(self) -> None:
        if self._live and self.total:
            self.ui.stream.write("\n")
            self.ui.stream.flush()
        took = _clock(time.monotonic() - self.started)
        if self.total:
            parts = [f"{self.done} new", f"{self.cached} cached"]
            if self.failed:
                parts.append(f"{self.failed} failed")
            mark = self.ui.warn if self.failed else self.ui.ok
            mark(f"{self.label}: " + ", ".join(parts) + f" in {took}")


def _clock(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    seconds = int(seconds)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _plain(text: str) -> str:
    import re

    return re.sub(r"\033\[[0-9;]*[A-Za-z]|\r", "", text)
