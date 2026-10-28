"""Build synthetic PDFs that exercise structure, not a particular book."""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf
from PIL import Image


def build_outline_pdf(path: Path | str) -> None:
    """Multi-page fixture with an outline, notes, math, figures, and a scan."""
    doc = pymupdf.open()
    try:
        _contents_page(doc)
        _prose_page(doc)
        _dialogue_page(doc)
        _figure_page(doc)
        _scanned_page(doc)
        doc.set_toc(
            [
                [1, "Chapter 1. The Open Road", 2],
                [1, "Chapter 2. What the Gate Said", 3],
                [1, "Chapter 3. Marks on the Page", 4],
            ]
        )
        doc.set_metadata({"title": "Synthetic Structure Fixture"})
        doc.save(path)
    finally:
        doc.close()


def build_heuristic_pdf(path: Path | str) -> None:
    """Two chapters and no outline, so detection falls back to headings."""
    doc = pymupdf.open()
    try:
        for number, title, body in (
            (1, "Chapter 1. The Quiet Harbor", "The harbor kept its own slow clock."),
            (2, "Chapter 2. The Evening Gate", "The gate closed after the last cart."),
        ):
            page = doc.new_page(width=612, height=792)
            _header(page, number)
            page.insert_text((72, 130), title, fontsize=20, fontname="hebo")
            page.insert_text((72, 180), body, fontsize=11, fontname="helv")
        doc.set_metadata({"title": "Synthetic Heading Fixture"})
        doc.save(path)
    finally:
        doc.close()


def build_toc_pdf(path: Path | str) -> None:
    """No outline. A contents page supplies the chapter starts."""
    doc = pymupdf.open()
    try:
        contents = doc.new_page(width=612, height=792)
        _header(contents, 1)
        contents.insert_text((72, 120), "Contents", fontsize=18, fontname="hebo")
        contents.insert_text((72, 170), "Chapter 1. Alpha .......... 2", fontsize=12, fontname="helv")
        contents.insert_text((72, 194), "Chapter 2. Beta ........... 3", fontsize=12, fontname="helv")
        for number, title, body in (
            (2, "Chapter 1. Alpha", "Alpha begins with a plain sentence."),
            (3, "Chapter 2. Beta", "Beta ends the short contents fixture."),
        ):
            page = doc.new_page(width=612, height=792)
            _header(page, number)
            page.insert_text((72, 130), title, fontsize=18, fontname="hebo")
            page.insert_text((72, 180), body, fontsize=11, fontname="helv")
        doc.set_metadata({"title": "Synthetic Contents Fixture"})
        doc.save(path)
    finally:
        doc.close()


def _header(page: pymupdf.Page, number: int) -> None:
    page.insert_text((72, 40), "Sample Volume", fontsize=9, fontname="helv")
    page.insert_text((300, 770), str(number), fontsize=9, fontname="helv")


def _contents_page(doc: pymupdf.Document) -> None:
    page = doc.new_page(width=612, height=792)
    _header(page, 1)
    page.insert_text((72, 120), "Contents", fontsize=18, fontname="hebo")
    page.insert_text((72, 170), "Chapter 1. The Open Road .......... 2", fontsize=12, fontname="helv")
    page.insert_text((72, 194), "Chapter 2. What the Gate Said ...... 3", fontsize=12, fontname="helv")
    page.insert_text((72, 218), "Chapter 3. Marks on the Page ....... 4", fontsize=12, fontname="helv")


def _prose_page(doc: pymupdf.Document) -> None:
    page = doc.new_page(width=612, height=792)
    _header(page, 2)
    page.insert_text((72, 130), "Chapter 1. The Open Road", fontsize=22, fontname="hebo")
    page.insert_text((72, 170), "A Small Beginning", fontsize=14, fontname="hebo")
    page.insert_text((72, 210), "The travelers agreed to recon-", fontsize=11, fontname="helv")
    left = "sider the route once the rain"
    page.insert_text((72, 226), left, fontsize=11, fontname="helv")
    width = pymupdf.get_text_length(left, fontsize=11, fontname="helv")
    marker = pymupdf.get_text_length("1", fontsize=7, fontname="helv")
    page.insert_text((72 + width + 1, 226), "1", fontsize=7, fontname="helv")
    page.insert_text((72 + width + 1 + marker + 2, 226), " eased.", fontsize=11, fontname="helv")
    _pieces(
        page,
        72,
        268,
        [
            ("They wrote the toy identity ", "helv"),
            ("$E = mc^2$", "cour"),
            (" in the margin.", "helv"),
        ],
    )
    page.insert_text((72, 304), "A margin note cites (Okada, 2004) after the storm.", fontsize=11, fontname="helv")
    page.insert_text((72, 336), "The map says see Chapter 2 before the ridge.", fontsize=11, fontname="helv")
    page.insert_text((90, 372), "Aside: the margin is not the argument.", fontsize=11, fontname="tiit")
    page.draw_line((72, 680), (180, 680), width=0.4)
    page.insert_text((72, 706), "1 This rain note is only about the weather.", fontsize=8, fontname="helv")


def _dialogue_page(doc: pymupdf.Document) -> None:
    page = doc.new_page(width=612, height=792)
    _header(page, 3)
    page.insert_text((72, 130), "Chapter 2. What the Gate Said", fontsize=22, fontname="hebo")
    page.insert_text((72, 180), "A public-domain exchange follows.", fontsize=11, fontname="helv")
    page.insert_text((72, 214), "GUIDE: You brought the map.", fontsize=11, fontname="helv")
    page.insert_text((72, 240), "WANDERER: I almost left it on the table.", fontsize=11, fontname="helv")
    # Lewis Carroll, What the Tortoise Said to Achilles (1895). Public domain.
    page.insert_text(
        (72, 280),
        "ACHILLES: So you've got to the end of our race-course?",
        fontsize=11,
        fontname="helv",
    )
    page.insert_text(
        (72, 306),
        "TORTOISE: Even though it does consist of an infinite series of distances?",
        fontsize=11,
        fontname="helv",
    )


def _figure_page(doc: pymupdf.Document) -> None:
    page = doc.new_page(width=612, height=792)
    _header(page, 4)
    page.insert_text((72, 120), "Chapter 3. Marks on the Page", fontsize=22, fontname="hebo")
    page.insert_text((72, 160), "A later note points to the same result [12].", fontsize=11, fontname="helv")
    image = Image.new("RGB", (160, 90), (210, 40, 40))
    for x in range(110, 150):
        for y in range(20, 70):
            image.putpixel((x, y), (30, 60, 180))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    page.insert_image(pymupdf.Rect(72, 190, 232, 280), stream=buffer.getvalue())
    page.insert_text((72, 304), "Figure 1. A red square beside a blue bar.", fontsize=10, fontname="helv")
    page.draw_circle((430, 230), 36, color=(0, 0, 0), width=1.4)
    page.draw_line((390, 210), (470, 260), color=(0, 0, 0.8), width=1.2)
    page.draw_line((390, 260), (470, 210), color=(0.1, 0.4, 0.1), width=1.2)
    page.insert_text((340, 304), "Figure 2. Two paths crossing a circle.", fontsize=10, fontname="helv")
    page.insert_text(
        (72, 370),
        r"$$x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}$$",
        fontsize=12,
        fontname="cour",
    )


def _scanned_page(doc: pymupdf.Document) -> None:
    pixmap = doc[1].get_pixmap(dpi=72, alpha=False)
    page = doc.new_page(width=612, height=792)
    page.insert_image(page.rect, pixmap=pixmap)


def _pieces(page: pymupdf.Page, x: float, y: float, parts: list[tuple[str, str]]) -> None:
    cursor = x
    for text, font in parts:
        page.insert_text((cursor, y), text, fontsize=11, fontname=font)
        cursor += pymupdf.get_text_length(text, fontsize=11, fontname=font)
