"""Turn the human-friendly tuning words in a project into prompt and audio settings."""

from __future__ import annotations

EXPRESSIVENESS = {
    "low": (
        0.4,
        "Expressiveness: low. Mark only clear emotional beats with a style. Use tags rarely, "
        "roughly one per 250 words.",
    ),
    "medium": (
        1.0,
        "Expressiveness: medium. Give most paragraphs a fitting style, and add an inline tag "
        "(a chuckle, a sigh, a breath) roughly once per 100 words wherever the writing invites it.",
    ),
    "high": (
        2.0,
        "Expressiveness: high. Perform the text. Give nearly every turn a specific style, and "
        "use inline vocal tags roughly two or three times per 100 words. Split turns whenever "
        "the mood moves.",
    ),
}

PACE = {
    "brisk": "Pace: brisk. Speak a little faster than conversation, with short pauses.",
    "normal": "Pace: normal conversational pace.",
    "measured": (
        "Pace: measured. Put 'unhurried' or 'measured pace' in the style of most turns. Speak "
        "slightly slower than conversation."
    ),
    "slow": (
        "Pace: slow. Put 'slow, deliberate pace' in the style of nearly every turn. Leave "
        "room between ideas."
    ),
}

PAUSES = {
    "minimal": "Pauses: only where punctuation already implies one.",
    "normal": "Pauses: add <short pause> at natural beats, such as before a punchline.",
    "generous": (
        "Pauses: the listener wants time to think. Add <short pause> after a sentence that carries an "
        "idea, and two in a row (<short pause> <short pause>) after a dense claim, a question, or "
        "the end of a paragraph. Aim for about one pause per 40 to 60 words. Never put a pause on "
        "every sentence, and never inside a short phrase."
    ),
}

# Silence between consecutive speech clips (one clip is roughly one passage).
GAP_MS = {"minimal": 250, "normal": 500, "generous": 900}
CHAPTER_TAIL_MS = {"minimal": 800, "normal": 1500, "generous": 2500}


def validate(name: str, value: str, table: dict) -> str:
    if value not in table:
        raise ValueError(f"{name} must be one of {', '.join(table)} (got {value!r})")
    return value


def delivery_guidance(expressiveness: str, pace: str, pause_density: str) -> str:
    return "\n".join(
        [
            EXPRESSIVENESS[validate("expressiveness", expressiveness, EXPRESSIVENESS)][1],
            PACE[validate("pace", pace, PACE)],
            PAUSES[validate("pause_density", pause_density, PAUSES)],
        ]
    )


def tag_floor(expressiveness: str) -> float:
    """Minimum vocal tags per 100 words before a retry is requested."""
    return EXPRESSIVENESS[validate("expressiveness", expressiveness, EXPRESSIVENESS)][0] * 0.5
