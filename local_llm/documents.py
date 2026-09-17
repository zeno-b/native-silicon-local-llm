"""Document generation: Markdown in, a real .pdf/.docx/.xlsx/.pptx out.

Written against the file formats directly, with nothing but the standard
library. That is a deliberate trade against pulling in reportlab/python-docx/
openpyxl: this app is meant to run on a laptop that may be offline, the rest of
it has no third-party document dependency, and a capability that only exists
when a wheel happened to install is the "I can't do that" failure the agent is
supposed to have stopped giving. A built-in writer always works.

The pipeline is two stages, so a format is a renderer and nothing else:

    markdown text  ->  parse_markdown()  ->  [Block, ...]  ->  one writer

PDF is generated as PDF 1.4 with the base-14 fonts (no font file to embed, so
every viewer can draw it). The OOXML formats (.docx/.xlsx/.pptx) are zip
archives of XML parts, written from the minimal part set Word, Excel and
PowerPoint actually require.
"""

from __future__ import annotations

import csv
import html
import io
import json
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# Every extension this module can produce, mapped to the human name used in tool
# descriptions and error messages. The tool layer reads this, so adding a writer
# here is all it takes to make the format offerable to the model.
FORMATS = {
    ".pdf": "PDF",
    ".docx": "Word document",
    ".xlsx": "Excel workbook",
    ".pptx": "PowerPoint deck",
    ".html": "HTML page",
    ".htm": "HTML page",
    ".md": "Markdown",
    ".txt": "plain text",
    ".csv": "CSV",
    ".json": "JSON",
}


class DocumentError(ValueError):
    """A document could not be produced. The message is shown to the user."""


# --------------------------------------------------------------------------- #
# Intermediate representation
# --------------------------------------------------------------------------- #

@dataclass
class Run:
    """A span of text with inline styling. The unit every writer draws."""
    text: str
    bold: bool = False
    italic: bool = False
    code: bool = False


@dataclass
class Block:
    """One paragraph-level thing.

    kind is one of: heading, para, bullet, number, code, table, rule.
    `level` carries the heading level or the list nesting depth; `rows` carries a
    table (first row is the header when `header` is set).
    """
    kind: str
    runs: list[Run] = field(default_factory=list)
    level: int = 0
    text: str = ""
    rows: list[list[str]] = field(default_factory=list)
    header: bool = True

    def plain(self) -> str:
        return "".join(r.text for r in self.runs) or self.text


# --------------------------------------------------------------------------- #
# Markdown subset parser
# --------------------------------------------------------------------------- #

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_NUMBER = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
_RULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_FENCE = re.compile(r"^\s*(?:```|~~~)(.*)$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
# Inline styling, longest marker first so ** is never read as two *.
_INLINE = re.compile(
    r"(\*\*[^*]+\*\*|__[^_]+__|\*[^*\n]+\*|(?<![\w_])_[^_\n]+_(?![\w_])|`[^`\n]+`|"
    r"\[[^\]\n]+\]\([^)\s]+\))"
)
_LINK = re.compile(r"^\[([^\]]+)\]\(([^)\s]+)\)$")


def parse_runs(text: str) -> list[Run]:
    """Split one line into styled runs.

    Deliberately small: bold, italic, inline code and links. A model asked for a
    document writes ordinary prose with the occasional emphasis, and every
    construct understood here has to be drawn by four different writers.
    """
    runs: list[Run] = []
    for piece in _INLINE.split(text):
        if not piece:
            continue
        link = _LINK.match(piece)
        if link:
            label, url = link.group(1), link.group(2)
            # No clickable annotation: a link that renders as its label alone
            # loses the address entirely once the document is printed.
            runs.append(Run(label if label == url else f"{label} ({url})"))
        elif piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            runs.append(Run(piece[2:-2], bold=True))
        elif piece.startswith("__") and piece.endswith("__") and len(piece) > 4:
            runs.append(Run(piece[2:-2], bold=True))
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            runs.append(Run(piece[1:-1], code=True))
        elif (piece.startswith("*") and piece.endswith("*") and len(piece) > 2) or \
             (piece.startswith("_") and piece.endswith("_") and len(piece) > 2):
            runs.append(Run(piece[1:-1], italic=True))
        else:
            runs.append(Run(piece))
    return runs or [Run("")]


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


# A model that loses its place emits the same token until the budget runs out.
# One turn produced a document whose content was several thousand consecutive
# newlines; rendered literally that is hundreds of blank pages. Markdown needs at
# most one blank line between blocks, so collapsing runs costs nothing real and
# turns a runaway into a short document.
_RUNAWAY_BLANKS = re.compile(r"\n{3,}")
_RUNAWAY_REPEAT = re.compile(r"(.{1,40}?)\1{9,}", re.S)


def _tame_runaway(text: str) -> str:
    text = _RUNAWAY_BLANKS.sub("\n\n", text or "")
    # Any short fragment repeated ten times or more in a row is a decode loop,
    # not content. Keep two copies so a legitimate repeated row survives.
    return _RUNAWAY_REPEAT.sub(lambda m: m.group(1) * 2, text)


def parse_markdown(text: str) -> list[Block]:
    """Turn Markdown into the block list every writer consumes."""
    text = _tame_runaway(text)
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[Block] = []
    para: list[str] = []
    i = 0

    def flush() -> None:
        if para:
            joined = " ".join(s.strip() for s in para).strip()
            if joined:
                blocks.append(Block("para", runs=parse_runs(joined)))
            para.clear()

    while i < len(lines):
        line = lines[i]

        fence = _FENCE.match(line)
        if fence:
            flush()
            body: list[str] = []
            i += 1
            while i < len(lines) and not _FENCE.match(lines[i]):
                body.append(lines[i])
                i += 1
            i += 1  # closing fence
            blocks.append(Block("code", text="\n".join(body), level=0))
            continue

        # A table is a header row followed by a |---|---| separator. Checked
        # before the paragraph accumulator sees either, or the header line is
        # swallowed as prose and the table loses its first row.
        if "|" in line and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            flush()
            rows = [_split_row(line)]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(_split_row(lines[i]))
                i += 1
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            blocks.append(Block("table", rows=rows, header=True))
            continue

        if not line.strip():
            flush()
            i += 1
            continue

        if _RULE.match(line):
            flush()
            blocks.append(Block("rule"))
            i += 1
            continue

        heading = _HEADING.match(line)
        if heading:
            flush()
            blocks.append(Block("heading", level=len(heading.group(1)),
                                runs=parse_runs(heading.group(2).strip())))
            i += 1
            continue

        bullet = _BULLET.match(line)
        if bullet:
            flush()
            blocks.append(Block("bullet", level=len(bullet.group(1)) // 2,
                                runs=parse_runs(bullet.group(2).strip())))
            i += 1
            continue

        number = _NUMBER.match(line)
        if number:
            flush()
            blocks.append(Block("number", level=len(number.group(1)) // 2,
                                runs=parse_runs(number.group(3).strip()),
                                text=number.group(2)))
            i += 1
            continue

        para.append(line)
        i += 1

    flush()
    return blocks


def document_title(blocks: list[Block], fallback: str = "") -> str:
    """The title to put in document properties: the first heading, else fallback."""
    for block in blocks:
        if block.kind == "heading":
            return block.plain()[:200]
    return fallback[:200]


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #

# Character widths (1/1000 em) for the base-14 fonts, codes 32..126. Without
# these every line would have to be wrapped by character count, which on a
# proportional font is wrong by up to 40% and produces either a ragged half-empty
# column or text running off the page. Helvetica-Oblique shares Helvetica's
# widths; Courier is fixed-pitch at 600.
_W_HELV = (
    "278 278 355 556 556 889 667 191 333 333 389 584 278 333 278 278 "
    "556 556 556 556 556 556 556 556 556 556 278 278 584 584 584 556 "
    "1015 667 667 722 722 667 611 778 722 278 500 667 556 833 722 778 "
    "667 778 722 667 611 722 667 944 667 667 611 278 278 278 469 556 "
    "333 556 556 500 556 556 278 556 556 222 222 500 222 833 556 556 "
    "556 556 333 500 278 556 500 722 500 500 500 334 260 334 584")
_W_HELV_BOLD = (
    "278 333 474 556 556 889 722 238 333 333 389 584 278 333 278 278 "
    "556 556 556 556 556 556 556 556 556 556 333 333 584 584 584 611 "
    "975 722 722 722 722 667 611 778 722 278 556 722 611 833 722 778 "
    "667 778 722 667 611 722 667 944 667 667 611 333 278 333 584 556 "
    "333 556 611 556 611 556 333 611 611 278 278 556 278 889 611 611 "
    "611 611 389 556 333 611 556 778 556 556 500 389 280 389 584")


def _width_table(spec: str) -> list[int]:
    return [int(n) for n in spec.split()]


_PDF_FONTS = {
    # name -> (PDF base font, widths for 32..126, default width for the rest)
    "regular": ("Helvetica", _width_table(_W_HELV), 556),
    "bold": ("Helvetica-Bold", _width_table(_W_HELV_BOLD), 556),
    "italic": ("Helvetica-Oblique", _width_table(_W_HELV), 556),
    "bolditalic": ("Helvetica-BoldOblique", _width_table(_W_HELV_BOLD), 556),
    "mono": ("Courier", [600] * 95, 600),
}
_PDF_FONT_ORDER = ["regular", "bold", "italic", "bolditalic", "mono"]


# Non-ASCII characters the base-14 fonts can draw and that a document actually
# contains. Everything else falls back to the accent-stripped base letter (in the
# standard fonts an accented letter is exactly as wide as its base), then to the
# font default. Getting this wrong is visible: measuring "i" as 556 instead of
# 222 pushes the word after "naive" three points to the right.
_WIDE_CHARS = {
    "\u2013": 556, "\u2014": 1000, "\u2018": 222, "\u2019": 222,
    "\u201c": 333, "\u201d": 333, "\u2022": 350, "\u2026": 1000,
    "\u20ac": 556, "\u00a0": 278, "\u00ab": 556, "\u00bb": 556,
    "\u00b0": 400, "\u00a3": 556, "\u00a9": 737, "\u00ae": 737,
    "\u00d7": 584, "\u00f7": 584, "\u00b1": 584, "\u2192": 1000,
    "\u2713": 556, "\u00df": 556, "\u00e6": 889, "\u0153": 944,
}


def _char_width(ch: str, widths: list[int], default: int) -> int:
    code = ord(ch)
    if 32 <= code <= 126:
        return widths[code - 32]
    if ch in _WIDE_CHARS:
        return _WIDE_CHARS[ch]
    base = unicodedata.normalize("NFD", ch)[:1]
    if base and 32 <= ord(base) <= 126:
        return widths[ord(base) - 32]
    return default


def _text_width(text: str, font: str, size: float) -> float:
    _, widths, default = _PDF_FONTS[font]
    total = sum(_char_width(ch, widths, default) for ch in text)
    return total * size / 1000.0


def _run_font(run: Run) -> str:
    if run.code:
        return "mono"
    if run.bold and run.italic:
        return "bolditalic"
    if run.bold:
        return "bold"
    if run.italic:
        return "italic"
    return "regular"


def _pdf_escape(text: str) -> bytes:
    """Encode for a PDF literal string in WinAnsi.

    Characters outside cp1252 become "?" rather than raising: a document with one
    unrepresentable glyph should still be produced, and the base-14 fonts have no
    way to draw it whatever we do here.
    """
    raw = text.encode("cp1252", "replace")
    out = bytearray()
    for byte in raw:
        if byte in (0x28, 0x29, 0x5C):    # ( ) \
            out += b"\\" + bytes([byte])
        elif byte < 32 or byte > 126:
            out += f"\\{byte:03o}".encode("ascii")
        else:
            out.append(byte)
    return bytes(out)


@dataclass
class _Piece:
    text: str
    font: str
    size: float


@dataclass
class _Line:
    """One laid-out line, ready to be placed on a page."""
    pieces: list[_Piece]
    indent: float = 0.0
    height: float = 0.0
    space_before: float = 0.0
    space_after: float = 0.0
    # A heading must not be the last thing on a page with its paragraph over the
    # break. Set on headings and on a table's header row.
    keep_with_next: bool = False
    rule: bool = False
    shade: bool = False


_PAGE_W, _PAGE_H = 595.28, 841.89          # A4 portrait, in points
_MARGIN = 56.0                              # ~2cm
_BODY_SIZE = 10.5
_BODY_LEAD = 15.0
_HEADING_SIZES = {1: 19.0, 2: 15.0, 3: 12.5, 4: 11.5, 5: 11.0, 6: 11.0}


def _wrap_runs(runs: list[Run], size: float, width: float,
               default_font: str = "regular") -> list[list[_Piece]]:
    """Greedy word wrap across styled runs. Returns one piece-list per line."""
    lines: list[list[_Piece]] = [[]]
    used = 0.0
    for run in runs:
        font = _run_font(run) if default_font == "regular" else default_font
        if run.code and default_font != "regular":
            font = "mono"
        # Keep the spaces: splitting on whitespace and re-joining loses the run
        # boundary spacing ("**bold** text" would come out "boldtext").
        for word in re.split(r"(\s+)", run.text):
            if not word:
                continue
            if word.isspace():
                if used > 0:
                    lines[-1].append(_Piece(" ", font, size))
                    used += _text_width(" ", font, size)
                continue
            word_width = _text_width(word, font, size)
            if used + word_width > width and used > 0:
                # Trailing space before a wrap is invisible but shifts the next
                # line's measurement, so drop it.
                while lines[-1] and lines[-1][-1].text == " ":
                    lines[-1].pop()
                lines.append([])
                used = 0.0
            if word_width > width:
                # One unbreakable token longer than the column (a URL, a hash).
                # Break it on character boundaries rather than letting it run off
                # the page edge.
                for chunk in _break_long(word, font, size, width):
                    if used > 0:
                        lines.append([])
                        used = 0.0
                    lines[-1].append(_Piece(chunk, font, size))
                    used = _text_width(chunk, font, size)
                continue
            lines[-1].append(_Piece(word, font, size))
            used += word_width
    return [ln for ln in lines if ln] or [[]]


def _break_long(word: str, font: str, size: float, width: float) -> list[str]:
    chunks, current = [], ""
    for ch in word:
        if _text_width(current + ch, font, size) > width and current:
            chunks.append(current)
            current = ch
        else:
            current += ch
    if current:
        chunks.append(current)
    return chunks


def _layout(blocks: list[Block], title: str) -> list[_Line]:
    """Blocks to lines, at the page's text width."""
    width = _PAGE_W - 2 * _MARGIN
    lines: list[_Line] = []

    if title:
        for pieces in _wrap_runs([Run(title, bold=True)], 20.0, width):
            lines.append(_Line(pieces, height=26.0, space_before=0.0,
                               keep_with_next=True))
        lines.append(_Line([], height=8.0, space_after=6.0, rule=True))

    for block in blocks:
        if block.kind == "rule":
            lines.append(_Line([], height=10.0, space_before=6.0, rule=True))
            continue

        if block.kind == "heading":
            size = _HEADING_SIZES.get(block.level, 11.0)
            runs = [Run(r.text, bold=True, italic=r.italic, code=r.code)
                    for r in block.runs]
            wrapped = _wrap_runs(runs, size, width)
            for n, pieces in enumerate(wrapped):
                last = n == len(wrapped) - 1
                lines.append(_Line(pieces, height=size * 1.35,
                                   space_before=(16.0 if n == 0 else 0.0),
                                   space_after=(size * 0.45 if last else 0.0),
                                   keep_with_next=True))
            continue

        if block.kind == "code":
            for raw in (block.text or "").split("\n"):
                for chunk in (_break_long(raw, "mono", 9.0, width - 12) or [""]):
                    lines.append(_Line([_Piece(chunk, "mono", 9.0)], indent=8.0,
                                       height=12.0, shade=True))
            lines.append(_Line([], height=6.0))
            continue

        if block.kind == "table":
            lines.extend(_layout_table(block, width))
            continue

        if block.kind in ("bullet", "number"):
            marker = "• " if block.kind == "bullet" else f"{block.text or '1'}. "
            indent = 12.0 + block.level * 14.0
            marker_w = _text_width(marker, "regular", _BODY_SIZE)
            wrapped = _wrap_runs(block.runs, _BODY_SIZE, width - indent - marker_w)
            for n, pieces in enumerate(wrapped):
                if n == 0:
                    pieces = [_Piece(marker, "regular", _BODY_SIZE)] + pieces
                    lines.append(_Line(pieces, indent=indent, height=_BODY_LEAD,
                                       space_before=2.0))
                else:
                    lines.append(_Line(pieces, indent=indent + marker_w,
                                       height=_BODY_LEAD))
            continue

        wrapped = _wrap_runs(block.runs, _BODY_SIZE, width)
        for n, pieces in enumerate(wrapped):
            lines.append(_Line(pieces, height=_BODY_LEAD,
                               space_before=(7.0 if n == 0 else 0.0)))

    return lines


def _layout_table(block: Block, width: float) -> list[_Line]:
    """A table as indented lines, columns sized by their widest cell.

    Not a drawn grid: ruled boxes need per-cell vertical alignment, and a wrapped
    cell would then have to push every neighbouring cell's box down. Column
    alignment plus a rule under the header reads correctly and cannot desynchronise.
    """
    rows = block.rows
    if not rows:
        return []
    columns = len(rows[0])
    natural = []
    for col in range(columns):
        widest = max(_text_width(row[col], "bold" if i == 0 else "regular", _BODY_SIZE)
                     for i, row in enumerate(rows))
        natural.append(max(widest, 24.0))
    gap = 10.0
    total = sum(natural) + gap * (columns - 1)
    if total > width:
        scale = (width - gap * (columns - 1)) / max(1.0, sum(natural))
        natural = [w * scale for w in natural]

    lines: list[_Line] = []
    for index, row in enumerate(rows):
        font = "bold" if (index == 0 and block.header) else "regular"
        # Wrap every cell, then emit as many physical lines as the tallest cell.
        cells = [_wrap_runs([Run(cell)], _BODY_SIZE, natural[col], default_font=font)
                 for col, cell in enumerate(row)]
        tallest = max(len(c) for c in cells)
        for depth in range(tallest):
            pieces: list[_Piece] = []
            x = 0.0
            for col in range(columns):
                part = cells[col][depth] if depth < len(cells[col]) else []
                pieces.append(_Piece("", font, _BODY_SIZE))   # column origin marker
                pieces[-1].text = ""
                for piece in part:
                    piece.font = font
                pieces.extend(part)
                x += natural[col] + gap
            lines.append(_Line(pieces, height=_BODY_LEAD,
                               space_before=(6.0 if index == 0 and depth == 0 else 0.0),
                               keep_with_next=(index == 0)))
            # Column positions are needed at draw time; carry them on the line.
            lines[-1].columns = [sum(natural[:c]) + gap * c for c in range(columns)]  # type: ignore[attr-defined]
            lines[-1].cells = [cells[c][depth] if depth < len(cells[c]) else []       # type: ignore[attr-defined]
                               for c in range(columns)]
        if index == 0 and block.header:
            lines.append(_Line([], height=5.0, rule=True))
    lines.append(_Line([], height=6.0))
    return lines


def _paginate(lines: list[_Line]) -> list[list[tuple[float, _Line]]]:
    """Assign every line a y coordinate, breaking pages at the bottom margin."""
    pages: list[list[tuple[float, _Line]]] = []
    current: list[tuple[float, _Line]] = []
    y = _PAGE_H - _MARGIN
    bottom = _MARGIN + 24.0          # room for the page number
    index = 0
    while index < len(lines):
        line = lines[index]
        top = y - line.space_before
        if top - line.height < bottom and current:
            pages.append(current)
            current = []
            y = _PAGE_H - _MARGIN
            continue
        y = top - line.height
        current.append((y, line))
        y -= line.space_after
        # A keep_with_next line reserved the follower's height so the break would
        # happen before it, not inside it. That reservation must not also be
        # drawn as blank space, so the cursor gets it straight back: without
        # this, every heading is followed by a second heading's worth of gap.
        y += getattr(line, "claim", 0.0)
        index += 1
    if current:
        pages.append(current)
    return pages or [[]]


def _pdf_bytes(blocks: list[Block], title: str) -> bytes:
    lines = _layout(blocks, title)
    # Honour keep_with_next by moving a trailing run of such lines forward.
    pages = _paginate(_with_keeps(lines))

    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font_ids: dict[str, int] = {}
    for key in _PDF_FONT_ORDER:
        base = _PDF_FONTS[key][0]
        font_ids[key] = add(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /" + base.encode("ascii")
            + b" /Encoding /WinAnsiEncoding >>")

    resources = (b"<< /Font << "
                 + b" ".join(f"/F{i} {font_ids[k]} 0 R".encode("ascii")
                             for i, k in enumerate(_PDF_FONT_ORDER))
                 + b" >> >>")

    pages_id = len(objects) + 1 + 2 * len(pages)      # reserved below
    page_ids: list[int] = []
    for number, page in enumerate(pages, 1):
        stream = _page_stream(page, number, len(pages))
        content_id = add(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
                         + stream + b"\nendstream")
        page_ids.append(add(
            b"<< /Type /Page /Parent " + str(pages_id).encode("ascii")
            + b" 0 R /MediaBox [0 0 "
            + f"{_PAGE_W:.2f} {_PAGE_H:.2f}".encode("ascii") + b"] /Resources "
            + resources + b" /Contents " + str(content_id).encode("ascii") + b" 0 R >>"))

    kids = b" ".join(f"{pid} 0 R".encode("ascii") for pid in page_ids)
    pages_obj = add(b"<< /Type /Pages /Count " + str(len(page_ids)).encode("ascii")
                    + b" /Kids [" + kids + b"] >>")
    assert pages_obj == pages_id, "page tree id reservation drifted"
    info = add(b"<< /Title (" + _pdf_escape(title or "Document") + b") /Producer "
               b"(local-llm) /CreationDate (D:"
               + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S").encode("ascii")
               + b"Z) >>")
    catalog = add(b"<< /Type /Catalog /Pages " + str(pages_id).encode("ascii") + b" 0 R >>")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += str(number).encode("ascii") + b" 0 obj\n" + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode("ascii") + b"\n"
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (b"trailer\n<< /Size " + str(len(objects) + 1).encode("ascii")
            + b" /Root " + str(catalog).encode("ascii") + b" 0 R /Info "
            + str(info).encode("ascii") + b" 0 R >>\nstartxref\n"
            + str(xref_at).encode("ascii") + b"\n%%EOF\n")
    return bytes(out)


def _with_keeps(lines: list[_Line]) -> list[_Line]:
    """Mark each keep_with_next line with the height it must reserve.

    A heading at the very bottom of a page with its first body line overleaf is
    the one pagination fault a reader always notices. Rather than backtracking
    after the fact, a heading simply claims the height of the line that follows
    it, so the break happens before it instead.
    """
    out: list[_Line] = []
    for index, line in enumerate(lines):
        if line.keep_with_next and index + 1 < len(lines):
            follower = lines[index + 1]
            line = _Line(line.pieces, line.indent,
                         line.height + follower.space_before + follower.height,
                         line.space_before, line.space_after, False,
                         line.rule, line.shade)
            # The reserved height must not also be drawn as blank space, so the
            # follower's own advance is what actually moves the cursor: record
            # the claim and give the height straight back on the next line.
            line.claim = follower.space_before + follower.height  # type: ignore[attr-defined]
            for attr in ("columns", "cells"):
                if hasattr(lines[index], attr):
                    setattr(line, attr, getattr(lines[index], attr))
        out.append(line)
    return out


def _page_stream(page: list[tuple[float, _Line]], number: int, total: int) -> bytes:
    out = bytearray()
    for y, line in page:
        claim = getattr(line, "claim", 0.0)
        draw_y = y + claim          # the claim reserved space below, not above
        if line.rule:
            out += (f"0.75 w 0.7 0.7 0.7 RG {_MARGIN:.2f} {draw_y + 4:.2f} m "
                    f"{_PAGE_W - _MARGIN:.2f} {draw_y + 4:.2f} l S\n").encode("ascii")
            continue
        if not line.pieces:
            continue
        if line.shade:
            out += (f"0.96 0.96 0.96 rg {_MARGIN:.2f} {draw_y - 2.5:.2f} "
                    f"{_PAGE_W - 2 * _MARGIN:.2f} {line.height:.2f} re f 0 0 0 rg\n"
                    ).encode("ascii")
        columns = getattr(line, "columns", None)
        cells = getattr(line, "cells", None)
        if columns is not None and cells is not None:
            for column_x, pieces in zip(columns, cells):
                out += _draw_pieces(pieces, _MARGIN + line.indent + column_x, draw_y)
            continue
        out += _draw_pieces(line.pieces, _MARGIN + line.indent, draw_y)

    if total > 1:
        label = f"{number} / {total}"
        width = _text_width(label, "regular", 8.5)
        out += (f"BT /F0 8.5 Tf 0.45 0.45 0.45 rg "
                f"{(_PAGE_W - width) / 2:.2f} {_MARGIN - 12:.2f} Td "
                ).encode("ascii") + b"(" + _pdf_escape(label) + b") Tj ET\n"
    return bytes(out)


def _draw_pieces(pieces: list[_Piece], x: float, y: float) -> bytes:
    out = bytearray()
    cursor = x
    for piece in pieces:
        if not piece.text:
            continue
        index = _PDF_FONT_ORDER.index(piece.font)
        out += (f"BT /F{index} {piece.size:.2f} Tf 0 0 0 rg "
                f"{cursor:.2f} {y:.2f} Td ").encode("ascii")
        out += b"(" + _pdf_escape(piece.text) + b") Tj ET\n"
        cursor += _text_width(piece.text, piece.font, piece.size)
    return bytes(out)


# --------------------------------------------------------------------------- #
# OOXML helpers (.docx / .xlsx / .pptx are zips of XML parts)
# --------------------------------------------------------------------------- #

def _xml(text: str) -> str:
    """Escape for XML character data, dropping what XML 1.0 cannot carry.

    A control character in the model's output would otherwise produce a file that
    Word refuses to open with "unreadable content", which reads as a bug in this
    app rather than as one stray byte.
    """
    cleaned = "".join(ch for ch in (text or "")
                      if ch in "\t\n" or 0x20 <= ord(ch) < 0xFFFE)
    return (cleaned.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _zip(parts: dict[str, str | bytes]) -> bytes:
    """Pack the parts into an OOXML package.

    Deflated and with a fixed timestamp, so the same document rendered twice is
    byte-identical -- which is what makes a regenerated attachment comparable.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, body in parts.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, body if isinstance(body, bytes)
                             else body.encode("utf-8"))
    return buffer.getvalue()


_RELS_ROOT = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '{items}</Relationships>'
)


def _core_props(title: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties '
        'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f'<dc:title>{_xml(title)}</dc:title>'
        '<dc:creator>local-llm</dc:creator>'
        '<cp:lastModifiedBy>local-llm</cp:lastModifiedBy>'
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{stamp}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{stamp}</dcterms:modified>'
        '</cp:coreProperties>'
    )


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #

_DOCX_STYLES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    '<w:docDefaults><w:rPrDefault><w:rPr>'
    '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>'
    '<w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:rPrDefault>'
    '<w:pPrDefault><w:pPr><w:spacing w:after="140" w:line="276" w:lineRule="auto"/>'
    '</w:pPr></w:pPrDefault></w:docDefaults>'
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
    '<w:name w:val="Normal"/><w:qFormat/></w:style>'
    '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>'
    '<w:basedOn w:val="Normal"/><w:qFormat/>'
    '<w:pPr><w:spacing w:after="240"/></w:pPr>'
    '<w:rPr><w:b/><w:sz w:val="52"/><w:szCs w:val="52"/></w:rPr></w:style>'
    + "".join(
        f'<w:style w:type="paragraph" w:styleId="Heading{n}">'
        f'<w:name w:val="heading {n}"/><w:basedOn w:val="Normal"/><w:qFormat/>'
        f'<w:pPr><w:keepNext/><w:outlineLvl w:val="{n - 1}"/>'
        f'<w:spacing w:before="{280 - n * 20}" w:after="120"/></w:pPr>'
        f'<w:rPr><w:b/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
        f'<w:color w:val="1F2A37"/></w:rPr></w:style>'
        for n, size in ((1, 36), (2, 30), (3, 26), (4, 24), (5, 22), (6, 22)))
    + '<w:style w:type="paragraph" w:styleId="ListParagraph">'
    '<w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/>'
    '<w:pPr><w:spacing w:after="60"/></w:pPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/>'
    '<w:basedOn w:val="Normal"/>'
    '<w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/>'
    '<w:ind w:left="284"/><w:shd w:val="clear" w:fill="F4F4F5"/></w:pPr>'
    '<w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="18"/>'
    '</w:rPr></w:style>'
    '<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/>'
    '<w:tblPr><w:tblBorders>'
    + "".join(f'<w:{edge} w:val="single" w:sz="4" w:space="0" w:color="C9CDD3"/>'
              for edge in ("top", "left", "bottom", "right", "insideH", "insideV"))
    + '</w:tblBorders></w:tblPr></w:style>'
    '</w:styles>'
)

_DOCX_NUMBERING = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    '<w:abstractNum w:abstractNumId="0"><w:multiLevelType w:val="hybridMultilevel"/>'
    + "".join(
        f'<w:lvl w:ilvl="{lvl}"><w:start w:val="1"/><w:numFmt w:val="bullet"/>'
        f'<w:lvlText w:val="{mark}"/><w:lvlJc w:val="left"/>'
        f'<w:pPr><w:ind w:left="{720 + lvl * 360}" w:hanging="360"/></w:pPr>'
        f'<w:rPr><w:rFonts w:ascii="Symbol" w:hAnsi="Symbol" w:hint="default"/>'
        f'</w:rPr></w:lvl>'
        for lvl, mark in enumerate(("", "o", "", "", "o",
                                    "", "", "o", "")))
    + '</w:abstractNum>'
    '<w:abstractNum w:abstractNumId="1"><w:multiLevelType w:val="hybridMultilevel"/>'
    + "".join(
        f'<w:lvl w:ilvl="{lvl}"><w:start w:val="1"/><w:numFmt w:val="{fmt}"/>'
        f'<w:lvlText w:val="%{lvl + 1}."/><w:lvlJc w:val="left"/>'
        f'<w:pPr><w:ind w:left="{720 + lvl * 360}" w:hanging="360"/></w:pPr></w:lvl>'
        for lvl, fmt in enumerate(("decimal", "lowerLetter", "lowerRoman") * 3))
    + '</w:abstractNum>'
    '<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>'
    '<w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num>'
    '</w:numbering>'
)


def _docx_runs(runs: list[Run]) -> str:
    out = []
    for run in runs:
        if not run.text:
            continue
        props = []
        if run.bold:
            props.append("<w:b/>")
        if run.italic:
            props.append("<w:i/>")
        if run.code:
            props.append('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>'
                         '<w:shd w:val="clear" w:fill="F4F4F5"/>')
        rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
        out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{_xml(run.text)}</w:t></w:r>')
    return "".join(out) or '<w:r><w:t xml:space="preserve"></w:t></w:r>'


def _docx_body(blocks: list[Block], title: str) -> str:
    out: list[str] = []
    if title:
        out.append(f'<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr>'
                   f'{_docx_runs([Run(title)])}</w:p>')
    for block in blocks:
        if block.kind == "heading":
            style = f"Heading{min(6, max(1, block.level))}"
            out.append(f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
                       f'{_docx_runs(block.runs)}</w:p>')
        elif block.kind in ("bullet", "number"):
            num = 1 if block.kind == "bullet" else 2
            level = min(8, block.level)
            out.append(
                '<w:p><w:pPr><w:pStyle w:val="ListParagraph"/><w:numPr>'
                f'<w:ilvl w:val="{level}"/><w:numId w:val="{num}"/></w:numPr></w:pPr>'
                f'{_docx_runs(block.runs)}</w:p>')
        elif block.kind == "code":
            for line in (block.text or "").split("\n"):
                out.append('<w:p><w:pPr><w:pStyle w:val="Code"/></w:pPr>'
                           f'{_docx_runs([Run(line)])}</w:p>')
            out.append('<w:p/>')
        elif block.kind == "rule":
            out.append('<w:p><w:pPr><w:pBdr><w:bottom w:val="single" w:sz="6" '
                       'w:space="1" w:color="C9CDD3"/></w:pBdr></w:pPr></w:p>')
        elif block.kind == "table":
            out.append(_docx_table(block))
        else:
            out.append(f'<w:p>{_docx_runs(block.runs)}</w:p>')
    # Word wants a sectPr at the end of the body; without it the page size and
    # margins fall back to the application default rather than A4.
    out.append('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
               '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" '
               'w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>')
    return "".join(out)


def _docx_table(block: Block) -> str:
    rows = []
    for index, row in enumerate(block.rows):
        cells = []
        for cell in row:
            runs = parse_runs(cell)
            if index == 0 and block.header:
                runs = [Run(r.text, bold=True, italic=r.italic, code=r.code) for r in runs]
            shade = ('<w:shd w:val="clear" w:fill="F1F3F5"/>'
                     if index == 0 and block.header else "")
            cells.append(f'<w:tc><w:tcPr>{shade}</w:tcPr>'
                         f'<w:p><w:pPr><w:spacing w:after="40"/></w:pPr>'
                         f'{_docx_runs(runs)}</w:p></w:tc>')
        header = ('<w:trPr><w:tblHeader/></w:trPr>'
                  if index == 0 and block.header else "")
        rows.append(f"<w:tr>{header}{''.join(cells)}</w:tr>")
    # <w:tblGrid> is not optional: it is a required child in the schema, and
    # without it Word reports the document as containing unreadable content and
    # python-docx refuses the table outright. Columns are sized by their widest
    # cell rather than split evenly, so a narrow "Q2" column does not get the
    # same width as a sentence.
    columns = len(block.rows[0]) if block.rows else 1
    content_twips = 9638            # A4 minus the 2cm margins set in the sectPr
    weights = [max(4, max(len(r[c]) for r in block.rows)) for c in range(columns)]
    total = sum(weights)
    grid = "".join(f'<w:gridCol w:w="{max(600, int(content_twips * w / total))}"/>'
                   for w in weights)
    return ('<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
            '<w:tblW w:w="5000" w:type="pct"/><w:tblLayout w:type="fixed"/>'
            '<w:tblBorders>'
            + "".join(f'<w:{edge} w:val="single" w:sz="4" w:space="0" w:color="C9CDD3"/>'
                      for edge in ("top", "left", "bottom", "right", "insideH", "insideV"))
            + '</w:tblBorders></w:tblPr>'
            + f'<w:tblGrid>{grid}</w:tblGrid>' + "".join(rows) + '</w:tbl>')


def _docx_bytes(blocks: list[Block], title: str) -> bytes:
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f'<w:body>{_docx_body(blocks, title)}</w:body></w:document>'
    )
    return _zip({
        "[Content_Types].xml":
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
            '<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '</Types>',
        "_rels/.rels": _RELS_ROOT.format(items=(
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>')),
        "word/_rels/document.xml.rels": _RELS_ROOT.format(items=(
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>')),
        "word/document.xml": document,
        "word/styles.xml": _DOCX_STYLES,
        "word/numbering.xml": _DOCX_NUMBERING,
        "docProps/core.xml": _core_props(title),
    })


# --------------------------------------------------------------------------- #
# XLSX
# --------------------------------------------------------------------------- #

def _column_name(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def _sheet_rows(blocks: list[Block]) -> list[list[list[str]]]:
    """One grid per table, plus a leading grid of everything that is not a table.

    A spreadsheet the model was asked for is almost always the tables; the prose
    around them is kept rather than dropped, on its own sheet, because silently
    losing half a document is worse than an extra tab.
    """
    tables = [b.rows for b in blocks if b.kind == "table" and b.rows]
    prose: list[list[str]] = []
    for block in blocks:
        if block.kind == "table":
            continue
        if block.kind == "heading":
            prose.append([block.plain()])
        elif block.kind in ("bullet", "number"):
            prose.append([("- " if block.kind == "bullet" else f"{block.text}. ")
                          + block.plain()])
        elif block.kind == "code":
            prose.extend([[line] for line in (block.text or "").split("\n")])
        elif block.kind == "rule":
            prose.append([""])
        else:
            prose.append([block.plain()])
    grids = list(tables)
    if prose and not grids:
        grids = [prose]
    elif prose:
        grids.append(prose)
    return grids or [[[""]]]


def _xlsx_sheet(rows: list[list[str]], header: bool) -> str:
    body = []
    for r, row in enumerate(rows, 1):
        cells = []
        for c, value in enumerate(row):
            ref = f"{_column_name(c)}{r}"
            text = "" if value is None else str(value)
            number = _as_number(text)
            if number is not None:
                cells.append(f'<c r="{ref}"><v>{number}</v></c>')
            else:
                style = ' s="1"' if (header and r == 1) else ""
                cells.append(f'<c r="{ref}" t="inlineStr"{style}>'
                             f'<is><t xml:space="preserve">{_xml(text)}</t></is></c>')
        body.append(f'<row r="{r}">{"".join(cells)}</row>')
    widths = _xlsx_widths(rows)
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'{widths}<sheetData>{"".join(body)}</sheetData></worksheet>')


def _xlsx_widths(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    columns = max(len(r) for r in rows)
    out = []
    for c in range(columns):
        widest = max((len(str(row[c])) for row in rows if c < len(row)), default=8)
        out.append(f'<col min="{c + 1}" max="{c + 1}" '
                   f'width="{min(60, max(9, widest + 2))}" customWidth="1"/>')
    return f'<cols>{"".join(out)}</cols>'


def _as_number(text: str) -> str | None:
    """The cell value as a number, or None to store it as text.

    Only unambiguous forms: a spreadsheet that silently turns "1-2" into a date
    or an order reference into 1.0E+15 is the classic way to corrupt data, so
    anything with a separator, a leading zero or a currency symbol stays text.
    """
    candidate = text.strip()
    if not candidate or len(candidate) > 15:
        return None
    if not re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?", candidate):
        return None
    return candidate


def _xlsx_bytes(blocks: list[Block], title: str) -> bytes:
    grids = _sheet_rows(blocks)
    names = []
    for index in range(len(grids)):
        base = "Data" if index == 0 else f"Data{index + 1}"
        if len(grids) > 1 and index == len(grids) - 1 and any(
                b.kind == "table" for b in blocks):
            base = "Notes"
        names.append(base)
    sheets = "".join(
        f'<sheet name="{_xml(name)}" sheetId="{i + 1}" r:id="rId{i + 1}"/>'
        for i, name in enumerate(names))
    rels = "".join(
        f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/'
        f'officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i + 1}.xml"/>'
        for i in range(len(grids)))
    styles_id = len(grids) + 1
    parts: dict[str, str | bytes] = {
        "[Content_Types].xml":
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(f'<Override PartName="/xl/worksheets/sheet{i + 1}.xml" '
                      'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                      for i in range(len(grids)))
            + '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '</Types>',
        "_rels/.rels": _RELS_ROOT.format(items=(
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>')),
        "xl/workbook.xml":
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets>{sheets}</sheets></workbook>',
        "xl/_rels/workbook.xml.rels": _RELS_ROOT.format(items=(
            rels + f'<Relationship Id="rId{styles_id}" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/styles" Target="styles.xml"/>')),
        "xl/styles.xml":
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
            '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="3"><fill><patternFill patternType="none"/></fill>'
            '<fill><patternFill patternType="gray125"/></fill>'
            '<fill><patternFill patternType="solid"><fgColor rgb="FFF1F3F5"/>'
            '<bgColor indexed="64"/></patternFill></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
            '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>'
            '</cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
            '</styleSheet>',
        "docProps/core.xml": _core_props(title),
    }
    table_count = sum(1 for b in blocks if b.kind == "table" and b.rows)
    for index, grid in enumerate(grids):
        parts[f"xl/worksheets/sheet{index + 1}.xml"] = _xlsx_sheet(
            grid, header=index < table_count)
    return _zip(parts)


# --------------------------------------------------------------------------- #
# PPTX
# --------------------------------------------------------------------------- #

_SLIDE_W, _SLIDE_H = 12192000, 6858000      # 16:9, in EMU


def _slides_from_blocks(blocks: list[Block], title: str) -> list[dict]:
    """Group blocks into slides: a heading opens one, its content fills it.

    Capped at a readable number of lines per slide; the overflow continues on a
    "(cont.)" slide rather than being drawn off the bottom edge.
    """
    slides: list[dict] = []
    current: dict | None = None

    def start(heading: str) -> None:
        nonlocal current
        current = {"title": heading, "bullets": []}
        slides.append(current)

    cover = bool(title)
    if cover:
        slides.append({"title": title, "bullets": [], "cover": True})
    for block in blocks:
        if block.kind == "heading" and block.level <= 2:
            start(block.plain())
            continue
        if current is None:
            # Content before the first heading. With a cover slide already
            # carrying the title, reusing it here produced two slides with the
            # same heading, the second one holding the intro paragraph.
            start(block.plain() if block.kind == "heading"
                  else ("Overview" if cover else (title or "Overview")))
            if block.kind == "heading":
                continue
        if block.kind == "heading":
            current["bullets"].append((0, block.plain(), True))
        elif block.kind in ("bullet", "number"):
            current["bullets"].append((min(4, block.level), block.plain(), False))
        elif block.kind == "table":
            for index, row in enumerate(block.rows):
                current["bullets"].append((0, "   ".join(row), index == 0))
        elif block.kind == "code":
            for line in (block.text or "").split("\n"):
                current["bullets"].append((1, line, False))
        elif block.kind == "rule":
            continue
        else:
            current["bullets"].append((0, block.plain(), False))

    out: list[dict] = []
    for slide in slides:
        bullets = slide["bullets"]
        if len(bullets) <= 9:
            out.append(slide)
            continue
        for offset in range(0, len(bullets), 9):
            chunk = dict(slide)
            chunk["bullets"] = bullets[offset:offset + 9]
            if offset:
                chunk["title"] = f"{slide['title']} (cont.)"
            out.append(chunk)
    return out or [{"title": title or "Document", "bullets": [], "cover": True}]


def _pptx_slide(slide: dict) -> str:
    cover = slide.get("cover")
    title_y = 2400000 if cover else 620000
    shapes = [
        '<p:sp><p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr><a:spLocks '
        'noGrp="1"/></p:cNvSpPr><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm>'
        f'<a:off x="838200" y="{title_y}"/>'
        f'<a:ext cx="{_SLIDE_W - 1676400}" cy="1000000"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>'
        '<p:txBody><a:bodyPr anchor="b"><a:normAutofit/></a:bodyPr><a:lstStyle/>'
        f'<a:p><a:pPr algn="{"ctr" if cover else "l"}"/>'
        f'<a:r><a:rPr lang="en-US" sz="{4000 if cover else 2800}" b="1" dirty="0">'
        '<a:solidFill><a:srgbClr val="1F2A37"/></a:solidFill></a:rPr>'
        f'<a:t>{_xml(slide["title"])}</a:t></a:r></a:p></p:txBody></p:sp>'
    ]
    if slide["bullets"]:
        paragraphs = []
        for level, text, strong in slide["bullets"]:
            paragraphs.append(
                f'<a:p><a:pPr lvl="{level}"/><a:r><a:rPr lang="en-US" sz="1800" '
                f'b="{1 if strong else 0}" dirty="0"><a:solidFill>'
                '<a:srgbClr val="333A45"/></a:solidFill></a:rPr>'
                f'<a:t>{_xml(text)}</a:t></a:r></a:p>')
        shapes.append(
            '<p:sp><p:nvSpPr><p:cNvPr id="3" name="Content"/><p:cNvSpPr>'
            '<a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr/></p:nvSpPr><p:spPr>'
            f'<a:xfrm><a:off x="838200" y="1750000"/>'
            f'<a:ext cx="{_SLIDE_W - 1676400}" cy="{_SLIDE_H - 2400000}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>'
            '<p:txBody><a:bodyPr><a:normAutofit/></a:bodyPr><a:lstStyle/>'
            + "".join(paragraphs) + '</p:txBody></p:sp>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/>'
        '<p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/>'
        '<a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/>'
        '</a:xfrm></p:grpSpPr>' + "".join(shapes) +
        '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
    )


_PPTX_THEME = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office">'
    '<a:themeElements><a:clrScheme name="Office">'
    '<a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1>'
    '<a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1>'
    '<a:dk2><a:srgbClr val="1F2A37"/></a:dk2><a:lt2><a:srgbClr val="EEF2F6"/></a:lt2>'
    '<a:accent1><a:srgbClr val="2F6FEB"/></a:accent1><a:accent2><a:srgbClr val="5B8DEF"/></a:accent2>'
    '<a:accent3><a:srgbClr val="16A34A"/></a:accent3><a:accent4><a:srgbClr val="F59E0B"/></a:accent4>'
    '<a:accent5><a:srgbClr val="DC2626"/></a:accent5><a:accent6><a:srgbClr val="7C3AED"/></a:accent6>'
    '<a:hlink><a:srgbClr val="2F6FEB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink>'
    '</a:clrScheme><a:fontScheme name="Office">'
    '<a:majorFont><a:latin typeface="Calibri Light"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>'
    '<a:minorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont>'
    '</a:fontScheme><a:fmtScheme name="Office">'
    '<a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>'
    '<a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
    '<a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
    '<a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>'
    '<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle>'
    '<a:effectStyle><a:effectLst/></a:effectStyle>'
    '<a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>'
    '<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>'
    '</a:fmtScheme></a:themeElements></a:theme>'
)

_PPTX_LAYOUT = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
    'type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>'
    '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
    '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
    '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
    '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
)

_PPTX_MASTER = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
    # An explicit background. Without one the slides inherit whatever the
    # renderer defaults to, which on macOS Quick Look is a flat grey.
    '<p:cSld><p:bg><p:bgPr><a:solidFill><a:schemeClr val="lt1"/></a:solidFill>'
    '<a:effectLst/></p:bgPr></p:bg>'
    '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/>'
    '<p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/>'
    '<a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm>'
    '</p:grpSpPr></p:spTree></p:cSld>'
    '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" '
    'accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" '
    'accent6="accent6" hlink="hlink" folHlink="folHlink"/>'
    '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
    '</p:sldMaster>'
)


def _pptx_bytes(blocks: list[Block], title: str) -> bytes:
    slides = _slides_from_blocks(blocks, title)
    # Slide ids start at 256 by convention; PowerPoint rejects lower ones.
    slide_list = "".join(
        f'<p:sldId id="{256 + i}" r:id="rId{i + 2}"/>' for i in range(len(slides)))
    parts: dict[str, str | bytes] = {
        "[Content_Types].xml":
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
            '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>'
            '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>'
            '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
            + "".join(f'<Override PartName="/ppt/slides/slide{i + 1}.xml" '
                      'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
                      for i in range(len(slides)))
            + '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '</Types>',
        "_rels/.rels": _RELS_ROOT.format(items=(
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>')),
        "ppt/presentation.xml":
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
            f'<p:sldIdLst>{slide_list}</p:sldIdLst>'
            f'<p:sldSz cx="{_SLIDE_W}" cy="{_SLIDE_H}"/>'
            f'<p:notesSz cx="{_SLIDE_H}" cy="{_SLIDE_W}"/></p:presentation>',
        "ppt/_rels/presentation.xml.rels": _RELS_ROOT.format(items=(
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
            + "".join(
                f'<Relationship Id="rId{i + 2}" Type="http://schemas.openxmlformats.org/'
                f'officeDocument/2006/relationships/slide" Target="slides/slide{i + 1}.xml"/>'
                for i in range(len(slides)))
            + f'<Relationship Id="rId{len(slides) + 2}" Type="http://schemas.openxmlformats.org/'
              'officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>')),
        "ppt/slideMasters/slideMaster1.xml": _PPTX_MASTER,
        "ppt/slideMasters/_rels/slideMaster1.xml.rels": _RELS_ROOT.format(items=(
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>')),
        "ppt/slideLayouts/slideLayout1.xml": _PPTX_LAYOUT,
        "ppt/slideLayouts/_rels/slideLayout1.xml.rels": _RELS_ROOT.format(items=(
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>')),
        "ppt/theme/theme1.xml": _PPTX_THEME,
        "docProps/core.xml": _core_props(title),
    }
    for index, slide in enumerate(slides):
        parts[f"ppt/slides/slide{index + 1}.xml"] = _pptx_slide(slide)
        parts[f"ppt/slides/_rels/slide{index + 1}.xml.rels"] = _RELS_ROOT.format(items=(
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'))
    return _zip(parts)


# --------------------------------------------------------------------------- #
# HTML / text / CSV
# --------------------------------------------------------------------------- #

def _html_runs(runs: list[Run]) -> str:
    out = []
    for run in runs:
        text = html.escape(run.text)
        if run.code:
            text = f"<code>{text}</code>"
        if run.bold:
            text = f"<strong>{text}</strong>"
        if run.italic:
            text = f"<em>{text}</em>"
        out.append(text)
    return "".join(out)


def _html_text(blocks: list[Block], title: str) -> str:
    body: list[str] = []
    list_open: str | None = None

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            body.append(f"</{list_open}>")
            list_open = None

    for block in blocks:
        if block.kind in ("bullet", "number"):
            want = "ul" if block.kind == "bullet" else "ol"
            if list_open != want:
                close_list()
                body.append(f"<{want}>")
                list_open = want
            body.append(f"<li>{_html_runs(block.runs)}</li>")
            continue
        close_list()
        if block.kind == "heading":
            level = min(6, max(1, block.level))
            body.append(f"<h{level}>{_html_runs(block.runs)}</h{level}>")
        elif block.kind == "code":
            body.append(f"<pre><code>{html.escape(block.text or '')}</code></pre>")
        elif block.kind == "rule":
            body.append("<hr>")
        elif block.kind == "table":
            rows = []
            for index, row in enumerate(block.rows):
                tag = "th" if index == 0 and block.header else "td"
                cells = "".join(f"<{tag}>{_html_runs(parse_runs(c))}</{tag}>" for c in row)
                rows.append(f"<tr>{cells}</tr>")
            body.append("<table>" + "".join(rows) + "</table>")
        else:
            body.append(f"<p>{_html_runs(block.runs)}</p>")
    close_list()
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{html.escape(title or 'Document')}</title>\n"
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        "<style>\n"
        ":root{color-scheme:light dark}\n"
        "body{max-width:46rem;margin:3rem auto;padding:0 1.25rem;"
        "font:16px/1.65 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
        "color:#1f2a37;background:#fff}\n"
        "h1,h2,h3,h4{line-height:1.25;margin:2rem 0 .6rem}\nh1{font-size:1.9rem}\n"
        "code{background:#f4f4f5;padding:.1em .35em;border-radius:4px;font-size:.9em}\n"
        "pre{background:#f4f4f5;padding:1rem;border-radius:8px;overflow-x:auto}\n"
        "pre code{background:none;padding:0}\n"
        "table{border-collapse:collapse;width:100%;margin:1.2rem 0}\n"
        "th,td{border:1px solid #c9cdd3;padding:.45rem .6rem;text-align:left}\n"
        "th{background:#f1f3f5}\nhr{border:0;border-top:1px solid #c9cdd3;margin:2rem 0}\n"
        "@media(prefers-color-scheme:dark){body{background:#14161a;color:#e6e8eb}\n"
        "code,pre,th{background:#1f2329}th,td{border-color:#333a45}}\n"
        "</style>\n</head>\n<body>\n"
        + (f"<h1>{html.escape(title)}</h1>\n" if title else "")
        + "\n".join(body) + "\n</body>\n</html>\n"
    )


def _plain_text(blocks: list[Block], title: str) -> str:
    out: list[str] = []
    if title:
        out += [title, "=" * len(title), ""]
    for block in blocks:
        if block.kind == "heading":
            out += ["", block.plain(), "-" * len(block.plain()), ""]
        elif block.kind == "bullet":
            out.append("  " * block.level + "- " + block.plain())
        elif block.kind == "number":
            out.append("  " * block.level + f"{block.text or '1'}. " + block.plain())
        elif block.kind == "code":
            out += [""] + ["    " + line for line in (block.text or "").split("\n")] + [""]
        elif block.kind == "rule":
            out += ["", "-" * 60, ""]
        elif block.kind == "table":
            widths = [max(len(r[c]) for r in block.rows)
                      for c in range(len(block.rows[0]))]
            for index, row in enumerate(block.rows):
                out.append("  ".join(cell.ljust(widths[c])
                                     for c, cell in enumerate(row)).rstrip())
                if index == 0 and block.header:
                    out.append("  ".join("-" * w for w in widths))
            out.append("")
        else:
            out += [block.plain(), ""]
    return "\n".join(out).strip() + "\n"


def _csv_text(blocks: list[Block]) -> str:
    rows: list[list[str]] = []
    for block in blocks:
        if block.kind == "table":
            rows.extend(block.rows)
        elif block.kind in ("bullet", "number", "para", "heading"):
            rows.append([block.plain()])
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows or [[""]])
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def render(content: str, suffix: str, title: str = "") -> bytes:
    """Render Markdown `content` into `suffix`'s format. Returns the file bytes.

    Raises DocumentError for an extension this module cannot write, so the caller
    can say which formats it does support rather than producing a .pdf-named text
    file that no reader will open.
    """
    suffix = (suffix or "").lower()
    if suffix not in FORMATS:
        raise DocumentError(
            f"cannot write {suffix or 'a file with no extension'}. Supported: "
            + ", ".join(sorted(FORMATS)))
    if suffix == ".json":
        # Passed through rather than parsed: the caller asked for data, not prose.
        try:
            parsed = json.loads(content)
        except Exception as exc:
            raise DocumentError(f"the content is not valid JSON: {exc}") from None
        return (json.dumps(parsed, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    blocks = parse_markdown(content)
    heading = title or document_title(blocks)
    # The title is drawn separately, so a document opening with the very heading
    # used as its title would print it twice.
    if not title and blocks and blocks[0].kind == "heading" \
            and blocks[0].plain() == heading:
        blocks = blocks[1:]

    if suffix == ".pdf":
        return _pdf_bytes(blocks, heading)
    if suffix == ".docx":
        return _docx_bytes(blocks, heading)
    if suffix == ".xlsx":
        return _xlsx_bytes(blocks, heading)
    if suffix == ".pptx":
        return _pptx_bytes(blocks, heading)
    if suffix in (".html", ".htm"):
        return _html_text(blocks, heading).encode("utf-8")
    if suffix == ".csv":
        return _csv_text(blocks).encode("utf-8")
    if suffix == ".txt":
        return _plain_text(blocks, heading).encode("utf-8")
    # .md keeps the author's own Markdown; re-rendering it from blocks would only
    # lose whatever this parser does not model.
    return (content if content.endswith("\n") else content + "\n").encode("utf-8")


def describe_document(suffix: str) -> str:
    """The human name for an extension, for notices and tool results."""
    return FORMATS.get((suffix or "").lower(), "file")


__all__ = [
    "FORMATS",
    "DocumentError",
    "Run",
    "Block",
    "parse_markdown",
    "_tame_runaway",
    "parse_runs",
    "document_title",
    "render",
    "describe_document",
]
