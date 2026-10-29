"""Delivery tuning as numbers from 0 to 1.

expressiveness  0 = plain reading            1 = fully performed
pace            0 = slowest                  1 = fastest (about 0.5 is conversational)
pause_density   0 = only punctuation pauses  1 = frequent room to think

The older words (low/medium/high, slow/measured/normal/brisk, minimal/normal/generous)
are still accepted and mapped to numbers.
"""

from __future__ import annotations

ALIASES = {
    "expressiveness": {"low": 0.25, "medium": 0.5, "high": 0.85},
    "pace": {"slow": 0.1, "measured": 0.3, "normal": 0.5, "brisk": 0.8},
    "pause_density": {"minimal": 0.1, "normal": 0.4, "generous": 0.85},
}
DEFAULTS = {"expressiveness": 0.6, "pace": 0.3, "pause_density": 0.8}


def level(name: str, value) -> float:
    """A number from 0 to 1, or one of the older words for it."""
    if isinstance(value, str) and value.strip().lower() in ALIASES[name]:
        return ALIASES[name][value.strip().lower()]
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = None
    if number is None or isinstance(value, bool) or not 0.0 <= number <= 1.0:
        words = ", ".join(ALIASES[name])
        raise ValueError(f"{name} must be a number from 0 to 1 (got {value!r}); the words {words} also work")
    return round(number, 3)


def tags_per_100_words(expressiveness: float) -> float:
    return round(2.5 * expressiveness, 2)


def tag_floor(expressiveness: float) -> float:
    """Minimum vocal tags per 100 words before a retry is requested."""
    return round(tags_per_100_words(expressiveness) / 2, 2)


def _every(per_100: float) -> int:
    return max(20, int(round(100 / per_100 / 10.0)) * 10)


def expressiveness_guidance(x: float) -> str:
    if x < 0.15:
        return "Expressiveness: minimal. Read plainly. Use a style or a tag only for a clear emotional turn."
    per_100 = tags_per_100_words(x)
    cadence = f"roughly one inline tag (a chuckle, a sigh, a breath) per {_every(per_100)} words wherever the writing invites it"
    if x < 0.4:
        return f"Expressiveness: low. Mark clear emotional beats with a style, and use {cadence}."
    if x < 0.7:
        return f"Expressiveness: medium. Give most paragraphs a fitting style, and use {cadence}."
    return (
        "Expressiveness: high. Perform the text. Give nearly every turn a specific style, split turns "
        f"whenever the mood moves, and use {cadence}."
    )


def pace_guidance(x: float) -> str:
    if x < 0.2:
        return "Pace: slow. Put 'slow, deliberate pace' in the style of nearly every turn. Leave room between ideas."
    if x < 0.4:
        return (
            "Pace: measured. Put 'unhurried' or 'measured pace' in the style of most turns. Speak "
            "slightly slower than conversation."
        )
    if x < 0.65:
        return "Pace: normal conversational pace."
    return "Pace: brisk. Speak a little faster than conversation, with short pauses."


def pause_guidance(x: float) -> str:
    if x < 0.15:
        return "Pauses: only where punctuation already implies one."
    every = max(30, int(round((120 - 70 * x) / 10.0)) * 10)
    if x < 0.5:
        return f"Pauses: add <short pause> at natural beats, such as before a punchline, about one per {every} words."
    return (
        "Pauses: the listener wants time to think. Add <short pause> after a sentence that carries an "
        "idea, and two in a row (<short pause> <short pause>) after a dense claim, a question, or the "
        f"end of a paragraph. Aim for about one pause per {every} words. Never put a pause on every "
        "sentence, and never inside a short phrase."
    )


def delivery_guidance(expressiveness, pace, pause_density) -> str:
    return "\n".join(
        [
            expressiveness_guidance(level("expressiveness", expressiveness)),
            pace_guidance(level("pace", pace)),
            pause_guidance(level("pause_density", pause_density)),
        ]
    )


def gap_ms(pause_density: float) -> int:
    """Silence between consecutive passages: 250 ms at 0, 900 ms at 1."""
    return int(round(250 + 650 * pause_density))


def chapter_tail_ms(pause_density: float) -> int:
    return int(round(800 + 1700 * pause_density))
