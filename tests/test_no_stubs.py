"""Shipped source must call real implementations."""

from pathlib import Path

BANNED = (
    "NotImplementedError",
    "TODO",
    "FIXME",
    "not implemented",
    "stubbed",
)


def test_src_has_no_stub_markers():
    root = Path(__file__).resolve().parents[1] / "src"
    hits = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for marker in BANNED:
            if marker.lower() in text:
                hits.append(f"{path.relative_to(root)} contains {marker}")
    assert hits == []
