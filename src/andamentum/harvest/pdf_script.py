"""Raised text and page footnotes in a PDF, written back into its markdown.

Docling's standard PDF pipeline keeps no script information: a superscript
citation "patients¹." comes out as "patients 1 .", with nothing to say the 1
was raised, and a page's footnotes are merged into one paragraph of body text.
Anything reading the markdown afterwards — a reference checker telling a
citation from an exponent, a reader matching a note to its mark — has lost
the evidence it needed.

The PDF still has it. Every character carries a font size and a position, so:

*A raised run* is a run of characters smaller than their line's main size and
sitting above its baseline — a superscript. Citations, footnote marks,
exponents and affiliation marks all qualify; this module does not judge which
is which. It reports formatting.

*A footnote* is found by structure, not by font: the block of lines at the
foot of a page whose entries start with consecutive numbers k, k+1, … each of
which also appears as a raised mark higher on the same page. Word and LaTeX set
footnotes smaller with raised numbers, but not every producer does — one PDF
of the test set sets them at body size with plain numbers — and the pairing of
mark and note holds either way, where a numbered table at the foot of a page
does not.

:func:`mark_markdown` writes both into the markdown docling produced: a raised
run as ``<sup>…</sup>``, and a raised number that has a footnote of that
number on the same page as a Markdown footnote — ``[^n]`` at the mark and
``[^n]: text`` at the end, the footnote's text removed from where docling
merged it. A run is placed by the non-whitespace text just before it, in
document order; one that cannot be placed is counted and logged, never
guessed at.

Off by default (``preserve_superscripts`` on :func:`~andamentum.harvest.extract`):
every existing caller gets exactly the markdown it got before.
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field

logger = logging.getLogger("andamentum.harvest")

_SMALLER = 0.85  # a raised character is at most this share of its line's size
_ABOVE = 0.2  # ... and its bottom sits this share of that size above the baseline
_RAISED_CHARS = re.compile(r"[0-9a-z,\-–—*†‡§]")
_CONTEXT = 24  # non-whitespace characters used to place a run


@dataclass(frozen=True)
class RaisedRun:
    page: int
    text: str
    before: str  # the non-whitespace characters just before it, on its page
    attached_before: bool  # no space between the run and the text before it
    attached_after: bool  # no space between the run and the text after it


@dataclass(frozen=True)
class PdfFootnote:
    page: int
    number: int
    text: str


@dataclass
class PdfScript:
    raised: list[RaisedRun] = field(default_factory=list)
    footnotes: list[PdfFootnote] = field(default_factory=list)


@dataclass
class MarkStats:
    marked: int = 0
    notes: int = 0
    unplaced: int = 0
    moved: int = 0  # placed by context alone: docling had dropped the digit


@dataclass
class _Char:
    ch: str
    size: float
    bottom: float
    top: float


def analyse_pdf(data: bytes) -> PdfScript:
    """The raised runs and page footnotes of a PDF."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(data)
    try:
        pages = _read_pages(document, pdfium)
    finally:
        document.close()
    return _script(pages)


def _read_pages(document, pdfium) -> list[list[list[_Char]]]:
    """Every page's lines of characters, with size and vertical position."""
    pages: list[list[list[_Char]]] = []  # page → lines → chars
    for page_index in range(len(document)):
        page = document[page_index]
        textpage = page.get_textpage()
        lines: list[list[_Char]] = [[]]
        for i in range(textpage.count_chars()):
            ch = textpage.get_text_range(i, 1)
            if ch in ("\r", "\n"):
                if lines[-1]:
                    lines.append([])
                continue
            left, bottom, right, top = textpage.get_charbox(i)
            size = float(pdfium.raw.FPDFText_GetFontSize(textpage, i))
            lines[-1].append(_Char(ch, size, bottom, top))
        pages.append([line for line in lines if line])
        textpage.close()
        page.close()
    return pages


def _script(pages: list[list[list[_Char]]]) -> PdfScript:

    body_sizes = [
        c.size for page in pages for line in page for c in line if c.ch.isalpha()
    ]
    if not body_sizes:
        return PdfScript()

    script = PdfScript()
    for page_number, lines in enumerate(pages, start=1):
        all_runs = _runs(_stream(lines), page=page_number)
        # Marks right after a digit are exponents (10⁵), not note marks.
        marks = {
            int(r.text)
            for r in all_runs
            if r.text.isdigit() and not r.before[-1:].isdigit()
        }
        footnote_start = _footnote_block_start(lines, marks=marks)
        body_lines = lines if footnote_start is None else lines[:footnote_start]
        script.raised += _runs(_stream(body_lines), page=page_number)
        if footnote_start is not None:
            script.footnotes += _footnotes(lines[footnote_start:], page=page_number)
    return script


def _stream(lines: list[list[_Char]]) -> list[tuple[_Char, bool]]:
    """The lines' characters in reading order, each with whether it is raised."""
    stream: list[tuple[_Char, bool]] = []
    for line in lines:
        stream += list(zip(line, _raised_flags(line)))
        stream.append((_Char(" ", 0, 0, 0), False))  # the line break
    return stream


def _line_size(line: list[_Char]) -> float:
    sizes = [round(c.size, 1) for c in line if c.ch.isalnum()]
    return statistics.mode(sizes) if sizes else 0.0


def _raised_flags(line: list[_Char]) -> list[bool]:
    size = _line_size(line)
    main = [c.bottom for c in line if c.ch.isalnum() and round(c.size, 1) == size]
    if not size or not main:
        return [False] * len(line)
    baseline = statistics.median(main)
    return [
        bool(_RAISED_CHARS.fullmatch(c.ch))
        and c.size < _SMALLER * size
        and c.bottom > baseline + _ABOVE * size
        for c in line
    ]


def _leading_number(line: list[_Char]) -> int | None:
    """The number a line starts with — raised or not — or None."""
    text = "".join(c.ch for c in line).lstrip()
    found = re.match(r"(\d{1,3})(?!\d)", text)
    return int(found.group(1)) if found else None


def _footnote_block_start(lines: list[list[_Char]], *, marks: set[int]) -> int | None:
    """The index of the page's first footnote line, or None.

    The earliest line starting with a raised mark's number k such that, from
    it to the foot of the page, lines open with k, k+1, … in order, every one
    of them a raised mark on the page, and the last is the page's highest
    mark: a page's footnotes are all at its foot, so a block that stops short
    of the highest mark is something else — a numbered reference list that
    starts at the foot of the page and runs on to the next. A line opening
    with any other number is a wrapped line of the note above ("…no.↵1,
    Springer…").
    """
    if not marks:
        return None
    highest = max(marks)
    for start, line in enumerate(lines):
        first = _leading_number(line)
        if first is None or first not in marks:
            continue
        expected = first
        for later in lines[start:]:
            if _leading_number(later) == expected and expected in marks:
                expected += 1
        if expected - 1 == highest and all(
            n in marks for n in range(first, highest + 1)
        ):
            return start
    return None


def _runs(stream: list[tuple[_Char, bool]], *, page: int) -> list[RaisedRun]:
    runs: list[RaisedRun] = []
    i = 0
    while i < len(stream):
        if not stream[i][1]:
            i += 1
            continue
        j = i
        while j < len(stream) and stream[j][1]:
            j += 1
        text = "".join(c.ch for c, _ in stream[i:j]).strip(",")
        if any(ch.isalnum() for ch in text):
            before = "".join(c.ch for c, _ in stream[:i] if not c.ch.isspace())[
                -_CONTEXT:
            ]
            runs.append(
                RaisedRun(
                    page=page,
                    text=text,
                    before=before,
                    attached_before=i > 0 and not stream[i - 1][0].ch.isspace(),
                    attached_after=j < len(stream) and not stream[j][0].ch.isspace(),
                )
            )
        i = j
    return runs


def _footnotes(lines: list[list[_Char]], *, page: int) -> list[PdfFootnote]:
    notes: list[PdfFootnote] = []
    for line in lines:
        text = "".join(c.ch for c in line)
        number = _leading_number(line)
        expected = notes[-1].number + 1 if notes else number
        if number is not None and number == expected:
            rest = re.sub(r"^\s*\d{1,3}[.)]?\s*", "", text)
            notes.append(PdfFootnote(page=page, number=number, text=rest.strip()))
        elif notes:
            last = notes[-1]
            notes[-1] = PdfFootnote(
                page=page, number=last.number, text=f"{last.text} {text.strip()}"
            )
    return notes


# ── writing it into the markdown ────────────────────────────────────────────


def mark_markdown(markdown: str, script: PdfScript) -> tuple[str, MarkStats]:
    """*markdown* with raised runs as ``<sup>`` and page footnotes as Markdown
    footnotes. See the module docstring."""
    stats = MarkStats()
    stripped, positions = _strip(markdown)
    edits: list[tuple[int, int, str]] = []  # (start, end, replacement) in markdown
    cursor = 0
    footnotes = {(n.page, n.number): n for n in script.footnotes}
    used_notes: set[tuple[int, int]] = set()

    for run in script.raised:
        needle_digits = re.sub(r"\s+", "", run.text)
        before = _key(run.before)
        found = -1
        for context in (before, before[-12:], before[-6:]):
            found = stripped.find(context + needle_digits, cursor)
            if found >= 0:
                found += len(context)
                break
        if found >= 0:
            start = positions[found]
            end = positions[found + len(needle_digits) - 1] + 1
            # Close the spaces docling put around a run the PDF set tight.
            if run.attached_before:
                while start > 0 and markdown[start - 1] == " ":
                    start -= 1
            if run.attached_after:
                while end < len(markdown) and markdown[end] == " ":
                    end += 1
            cursor = found + len(needle_digits)
        else:
            # Docling sometimes drops the raised digit from its sentence (and
            # sets it down elsewhere): "…remains useful. Further…". The PDF
            # still says where it belongs — right after its preceding text —
            # so the mark goes there. Only the full context is trusted for
            # this; a shortened one could match anywhere.
            at = stripped.find(before, cursor) if len(before) >= 12 else -1
            if at < 0:
                stats.unplaced += 1
                continue
            start = end = positions[at + len(before) - 1] + 1
            cursor = at + len(before)
            stats.moved += 1
        key = (run.page, int(needle_digits)) if needle_digits.isdigit() else None
        if key is not None and key in footnotes and key not in used_notes:
            used_notes.add(key)
            replacement = f"[^{key[0]}-{key[1]}]"
            stats.notes += 1
        else:
            replacement = f"<sup>{needle_digits}</sup>"
            stats.marked += 1
        edits.append((start, end, replacement))

    # The footnotes' own text, where docling merged it into the body.
    for key in sorted(used_notes):
        note = footnotes[key]
        span = _find_note(markdown, stripped, positions, note)
        if span is not None and not any(
            a < span[1] and span[0] < b for a, b, _ in edits
        ):
            edits.append((span[0], span[1], ""))

    out = markdown
    for start, end, replacement in sorted(edits, reverse=True):
        out = out[:start] + replacement + out[end:]
    out = re.sub(r"\n[ \t]+(?=\n)", "\n", out)  # lines a removal left blank
    out = re.sub(r"\n{3,}", "\n\n", out).rstrip()
    if used_notes:
        definitions = "\n".join(
            f"[^{page}-{number}]: {footnotes[(page, number)].text}"
            for page, number in sorted(used_notes)
        )
        out = f"{out}\n\n{definitions}"
    if stats.unplaced:
        logger.info(
            "[harvest.pdf_script] %d raised run(s) marked, %d footnote(s) linked, "
            "%d run(s) could not be placed in the markdown",
            stats.marked,
            stats.notes,
            stats.unplaced,
        )
    return out + "\n", stats


_ENTITIES = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'"}
_SAME = str.maketrans(
    {"‘": "'", "’": "'", "“": "'", "”": "'", '"': "'", "–": "-", "—": "-"}
)


def _strip(text: str) -> tuple[str, list[int]]:
    """A comparison key for *text*, and where each of its characters is.

    Whitespace is dropped, HTML entities read as the character they stand
    for, and quotes and dashes of every kind as one — docling writes
    ``'CADD'`` and ``&amp;`` where the PDF has ``“CADD”`` and ``&``, and a
    run or a note has to be found across the difference.
    """
    chars, positions = [], []
    index = 0
    while index < len(text):
        entity = next((e for e in _ENTITIES if text.startswith(e, index)), None)
        if entity is not None:
            chars.append(_ENTITIES[entity].translate(_SAME))
            positions.append(index)
            index += len(entity)
            continue
        ch = text[index]
        if not ch.isspace():
            chars.append(ch.translate(_SAME))
            positions.append(index)
        index += 1
    return "".join(chars), positions


def _key(text: str) -> str:
    return _strip(text)[0]


def _find_note(
    markdown: str, stripped: str, positions: list[int], note: PdfFootnote
) -> tuple[int, int] | None:
    """Where a footnote's number and text sit in the markdown, or None."""
    body = _key(note.text)
    if not body:
        return None
    head = f"{note.number}{body[:30]}"
    found = stripped.find(head)
    if found < 0:
        return None
    tail = body[-20:]
    end = stripped.find(tail, found + len(head) - 30)
    if end < 0:
        return None
    last = positions[end + len(tail) - 1]
    # The last character may be the start of an entity (&amp;): take it whole.
    entity = next((e for e in _ENTITIES if markdown.startswith(e, last)), "")
    return positions[found], last + max(1, len(entity))
