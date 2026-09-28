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

    The list is read as one stream, not line by line: a converted PDF breaks
    it at arbitrary points ("…s41586-↵020-2434-2. 4.LeCun…") and merges
    several entries into one paragraph, so a line need not start an entry
    and an entry need not start a line.

    *Numbered* when entry numbers — at a line start or inline, "1.", "[1]",
    "1 " — form a rising sequence (steps of one to three: a missing entry
    leaves a gap) and cover most of the list. Each accepted number starts an
    entry; a number inside an entry ("vol. 9, no. 4.") is refused because the
    text since the last cut is not yet a whole entry.

    *Unnumbered* otherwise: an entry starts at a line that opens like one, or
    inline before a new first author once the text since the last cut is a
    whole entry. A line of names only continues the entry above (a long
    author list broken across lines).
    """
    body = text[start:end]
    cuts = _numbered_cuts(body)
    year_lines = sum(1 for line in body.splitlines() if _looks_like_entry(line))
    if len(cuts) >= 2 and len(cuts) >= 0.5 * year_lines:
        raw = []
        for (cut, number, head), following in zip(cuts, cuts[1:] + [(len(body), 0, 0)]):
            segment = body[cut + head : following[0]]
            lead = len(segment) - len(segment.lstrip())
            raw.append({"number": number, "start": start + cut + head + lead, "end": start + following[0]})
        return [e for e in (_entry(text, r) for r in raw) if e.text]
    return _unnumbered(text, start=start, end=end)


# An entry number anywhere in the list: "1.Rentzsch", "[4]LeCun", "3 Turro E,",
# "- 6 Li H". The next character opens a name or a quoted title.
_NUMBER_ANYWHERE = re.compile(
    r"(?:^|(?<=\s))(?:[-*•][ \t]*)?\[?(\d{1,3})(?:[\].)][ \t]*|[ \t]+)(?=[A-Z“\"'‘(])",
    re.MULTILINE,
)
_MIN_ENTRY = 40  # characters; shorter than this is not a whole entry


def _numbered_cuts(body: str) -> list[tuple[int, int, int]]:
    """``(offset, number, length of the number marker)`` of each entry start."""
    cuts: list[tuple[int, int, int]] = []
    for m in _NUMBER_ANYWHERE.finditer(body):
        number = int(m.group(1))
        at_line_start = m.start() == 0 or body[m.start() - 1] == "\n"
        if not cuts:
            if number <= 3 or at_line_start:
                cuts.append((m.start(), number, m.end() - m.start()))
            continue
        last_offset, last, _ = cuts[-1]
        if at_line_start:
            # A line that opens with a number is an entry by layout: accepted
            # unless it goes backwards, so a repeated number (two entries
            # numbered 4) is kept for the check to report.
            ok = number >= last
        else:
            ok = last < number <= last + 3 and _complete(body[last_offset : m.start()])
        if ok:
            cuts.append((m.start(), number, m.end() - m.start()))
    return cuts


# A first author opening an entry: "Rentzsch, P.", "van der Berg, J.",
# and the Vancouver/Springer form "Rentzsch P," / "Gudmundsson S,".
_AUTHOR_START = re.compile(
    r"(?<=\s)(?:(?:van|von|de|der|den|le|la|du|da|di)\s)?[A-Z][\w'’\-]+"
    # "Surname, I." — or "Surname AB" followed by a comma or "(": without
    # that, "Next-generation DNA" inside a title reads as an author.
    r"(?:,\s(?:[A-Z]\.\s?){1,3}(?=[,\s&(])|\s[A-Z]{1,3}(?=,|\s\())"
)
_NAME_TOKEN = re.compile(
    r"(?:(?:van|von|de|der|den|le|la|du|da|di)\s)?[A-Z][\w'’\-]+,?\s(?:[A-Z]\.?\s?){1,3}|\bet al\.?|\band\b|&amp;|&"
)


def _unnumbered(text: str, *, start: int, end: int) -> list[ReferenceEntry]:
    raw: list[dict] = []
    position = start
    for line in text[start:end].splitlines(keepends=True):
        body = line.rstrip("\r\n")
        line_start = position
        position += len(line)
        if not body.strip():
            continue
        cuts = [0]
        for m in _AUTHOR_START.finditer(body):
            if m.start() > 0 and _complete(body[cuts[-1] : m.start()]):
                cuts.append(m.start())
        for piece_index, (a, b) in enumerate(zip(cuts, cuts[1:] + [len(body)])):
            piece = body[a:b]
            if not piece.strip():
                continue
            piece_start = line_start + a + (len(piece) - len(piece.lstrip()))
            piece_end = line_start + a + len(piece.rstrip())
            opens = piece_index > 0 or _opens_entry(piece)
            # An entry above that has no year yet is not finished: a long
            # author list broken across lines ("Turro, E., …↵Syrris, P., …,
            # 2020. Whole-genome…") continues it even though the line opens
            # with a name.
            unfinished = raw and not _looks_like_entry(text[raw[-1]["start"] : raw[-1]["end"]])
            if raw and (not opens or _names_only(piece) or (piece_index == 0 and unfinished)):
                raw[-1]["end"] = piece_end  # continues the entry above
            elif opens or _looks_like_entry(piece):
                raw.append({"number": None, "start": piece_start, "end": piece_end})
    return [_entry(text, r) for r in raw]


def _names_only(piece: str) -> bool:
    """A line of author names and nothing else — a broken author list."""
    if YEAR.search(piece):
        return False
    rest = _NAME_TOKEN.sub(" ", piece)
    return len(re.sub(r"[\W\d_]+", "", rest)) < 12


def _opens_entry(line: str) -> bool:
    """Whether a line starts an author-year entry: a first author, or a
    capitalised name with the year close behind — "World Health Organization
    (2020)"."""
    text = line.lstrip()
    if _AUTHOR_START.match(" " + text, 1):
        return True
    year = YEAR.search(text[:80])
    return bool(text[:1].isupper() and year)


def _complete(piece: str) -> bool:
    return len(piece.strip()) >= _MIN_ENTRY and _looks_like_entry(piece)


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
