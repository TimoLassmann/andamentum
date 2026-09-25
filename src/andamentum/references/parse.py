"""Find the reference list and split it into entries.

Deterministic and dependency-free. Everything returned carries offsets into
the text it was given.

An entry is a paragraph (one line of the text) of the list. A numbered list is
recognised by its leading numbers (``1.``, ``1)``, ``[1]``, and the ``- [1]``
list items a PDF converter writes), and an unnumbered line that follows a
numbered one is a wrapped continuation of it. In an unnumbered list — the
author-year styles — every line that carries a year or a DOI is an entry, and
a line with neither (a stray heading, a figure legend after the list) is not.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .models import ReferenceEntry

REFERENCE_HEADINGS = frozenset(
    {
        "references",
        "reference list",
        "bibliography",
        "works cited",
        "literature cited",
        "literature",
        "cited literature",
    }
)

_HEADING_LINE = re.compile(
    r"^[ \t#]*(references|reference list|bibliography|works cited|literature cited)"
    r"[ \t:]*$",
    re.IGNORECASE | re.MULTILINE,
)
# "1. ", "1) ", "[1] ", "- [1] ", and the "1.Author" and "1 Author" that
# Nature- and BMJ-style lists produce with no delimiter or no space.
_ENTRY_NUMBER = re.compile(
    r"^[ \t]*(?:[-*•][ \t]*)?\[?(\d{1,3})(?:[\].)][ \t]*|[ \t]+)(?=[^\d\s])"
)
DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.IGNORECASE)
YEAR = re.compile(r"(?<![\d.])(1[89]\d\d|20\d\d)([a-z]?)(?![\d])")


def find_reference_list(
    text: str, *, sections: Sequence[tuple[str, int, int]] | None = None
) -> tuple[int, int] | None:
    """``(start, end)`` of the reference list's body, or ``None``.

    With *sections* — ``(title, start, end)`` for each section, as a document
    reader already knows them — the LAST section whose title is a reference
    heading is used, and its heading line is skipped. Without them, the last
    line that is only a reference heading starts the list, which runs to the
    end of the text. The last one, because a paper that mentions "References"
    in a methods sentence has its real list at the end.
    """
    if sections:
        for title, start, end in reversed(list(sections)):
            if title.strip().strip(":").lower() in REFERENCE_HEADINGS:
                body_start = start
                heading = _HEADING_LINE.match(text, start)
                if heading is not None:
                    body_start = heading.end()
                return body_start, end
        return None
    last = None
    for heading in _HEADING_LINE.finditer(text):
        last = heading
    if last is None:
        return None
    return last.end(), len(text)


def split_entries(text: str, *, start: int, end: int) -> list[ReferenceEntry]:
    """The entries between *start* and *end*, numbered or not.

    The list counts as numbered when most of its year-carrying lines start
    with a number. Then an unnumbered line continues the entry above it.
    Otherwise every line with a year or a DOI is one entry.
    """
    lines: list[tuple[int, int, str]] = []
    position = start
    for line in text[start:end].splitlines(keepends=True):
        body = line.rstrip("\r\n")
        if body.strip():
            lines.append((position, position + len(body), body))
        position += len(line)

    substantive = [line for line in lines if _looks_like_entry(line[2])]
    numbered_lines = sum(1 for line in substantive if _ENTRY_NUMBER.match(line[2]))
    numbered = bool(substantive) and numbered_lines >= 0.6 * len(substantive)

    raw: list[dict] = []
    for line_start, line_end, body in lines:
        number = _ENTRY_NUMBER.match(body) if numbered else None
        if number is not None:
            raw.append(
                {"number": int(number.group(1)), "start": line_start + number.end(), "end": line_end}
            )
        elif numbered and raw:
            raw[-1]["end"] = line_end  # a wrapped continuation
        elif not numbered and _looks_like_entry(body):
            lead = len(body) - len(body.lstrip())
            raw.append({"number": None, "start": line_start + lead, "end": line_end})
    return [_entry(text, r) for r in raw]


def _looks_like_entry(line: str) -> bool:
    return bool(YEAR.search(line) or DOI.search(line))


def _entry(text: str, raw: dict) -> ReferenceEntry:
    stripped = text[raw["start"] : raw["end"]].rstrip()
    doi = DOI.search(stripped)
    first_year = YEAR.search(stripped)
    lead = stripped[: first_year.start()] if first_year else stripped[:120]
    return ReferenceEntry(
        number=raw["number"],
        text=" ".join(stripped.split()),
        start=raw["start"],
        end=raw["start"] + len(stripped),
        doi=doi.group(1).rstrip(".,;)") if doi else "",
        lead=" ".join(lead.split())[:200],
        years=sorted({int(m.group(1)) for m in YEAR.finditer(stripped)}),
        year=int(first_year.group(1)) if first_year else None,
        year_suffix=first_year.group(2) if first_year else "",
    )
