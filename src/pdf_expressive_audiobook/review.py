"""Human review of text-prep choices. Rationales stay in this report."""

from __future__ import annotations

import html
from pathlib import Path

from .models import Chunk


def render_review_markdown(chunks: list[Chunk], outputs: dict[str, dict]) -> str:
    lines = ["# Text-prep review", ""]
    if not chunks:
        lines.append("No chunks in this chapter.")
        return "\n".join(lines) + "\n"
    for chunk in chunks:
        output = outputs.get(chunk.id)
        lines.append(f"## {chunk.id}")
        lines.append("")
        lines.append(f"- Kind: {chunk.kind}")
        lines.append(f"- Pages: {', '.join(str(page) for page in chunk.source_pages) or 'n/a'}")
        lines.append(f"- Estimated duration: {chunk.estimated_duration_sec:.1f}s")
        lines.append("")
        lines.append("### Source")
        lines.append("")
        lines.append(_fence(chunk.text or "(empty)"))
        if output is None:
            lines.append("No prep output for this chunk yet.")
            lines.append("")
            continue
        lines.append(f"Prompt version: `{output.get('prompt_version', '')}`")
        lines.append("")
        lines.append(
            "<table><thead><tr><th>Speaker</th><th>Style</th><th>Prepared text</th>"
            "<th>Rationale</th><th>Descriptions</th></tr></thead><tbody>"
        )
        for turn in output.get("turns") or []:
            descriptions = _format_descriptions(turn.get("inserted_descriptions") or [])
            lines.append(
                "<tr>"
                f"<td>{_cell(turn.get('speaker', ''))}</td>"
                f"<td>{_cell(turn.get('style', ''))}</td>"
                f"<td>{_cell(turn.get('text', ''))}</td>"
                f"<td>{_cell(turn.get('rationale', ''))}</td>"
                f"<td>{_cell(descriptions)}</td>"
                "</tr>"
            )
        lines.append("</tbody></table>")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_review_html(chunks: list[Chunk], outputs: dict[str, dict], title: str) -> str:
    body = render_review_markdown(chunks, outputs)
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:Georgia,serif;margin:2rem;line-height:1.45}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;"
        "padding:.4rem;vertical-align:top}pre{white-space:pre-wrap}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"{_markdown_body_as_html(body)}</body></html>\n"
    )


def write_review(path: Path | str, text: str) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")


def _format_descriptions(items: list[dict]) -> str:
    if not items:
        return ""
    rows = []
    for item in items:
        decision = item.get("decision", "")
        reason = item.get("reason") or ""
        extra = f" ({reason})" if reason else ""
        rows.append(
            f"{item.get('element_id')} [{item.get('kind')}] {decision}{extra}: {item.get('spoken_text', '')}"
        )
    return "\n".join(rows)


def _cell(value: str) -> str:
    return html.escape(str(value)).replace("\n", "<br>")


def _fence(text: str) -> str:
    return "```\n" + text.replace("```", "'''") + "\n```\n"


def _markdown_body_as_html(markdown: str) -> str:
    """Enough HTML for the review file: headings, paragraphs, and raw tables."""
    parts: list[str] = []
    in_code = False
    code: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("```"):
            if in_code:
                parts.append("<pre>" + html.escape("\n".join(code)) + "</pre>")
                code = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code.append(line)
            continue
        if line.startswith("<table") or line.startswith("<tr") or line.startswith("</"):
            parts.append(line)
            continue
        if line.startswith("## "):
            parts.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("# "):
            continue
        elif line.startswith("- "):
            parts.append(f"<p>{html.escape(line[2:])}</p>")
        elif line.strip():
            parts.append(f"<p>{html.escape(line)}</p>")
    return "\n".join(parts)
