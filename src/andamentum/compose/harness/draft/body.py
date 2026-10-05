"""A drafted section body (markdown with ``[[fN]]`` markers) as Word paragraphs.

The compose drafter writes plain paragraphs, ``- `` bullets, ``1. `` numbered
lists and pipe tables, and marks fact use with ``[[f12]]``. A Word document has
paragraphs only, so this module is the one place that turns a body into them:

- blocks are split at blank lines; every list item is its own paragraph (the
  marker text stays on it so a document without list styles still reads right);
- a pipe table becomes one paragraph per row (cells joined by `` | ``): the
  operation set has no table, so the structure is flattened and said so in the
  report, never dropped;
- inline markdown (``**bold**``, ``*italic*``, backticks) is removed, because
  an inserted paragraph carries plain text;
- fact markers are cut out of the text and kept as :class:`Cite` positions:
  the position is where a citation field goes, moved in front of a sentence's
  closing punctuation (``... rate (Smith 2020).``);
- ``[TO FILL: ...]`` stays in the text and is reported as a range.

Pure functions, no model calls. Offsets index the cleaned paragraph text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

MARKER_GROUP = re.compile(r"[ \t]*(?:\[\[(f\d+)\]\][ \t]*)+")
_MARKER_ID = re.compile(r"\[\[(f\d+)\]\]")
TO_FILL = re.compile(r"\[TO FILL:[^\]]*\]")
_BULLET = re.compile(r"^\s*[-*•]\s+(.*)$")
_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_RULE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_BOLD = re.compile(r"(\*\*|__)(.+?)\1")
_ITALIC = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])")
_CODE = re.compile(r"`([^`\n]+)`")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_CLOSING = ".!?"
_CLOSERS_AFTER_PUNCTUATION = frozenset("\"')]”’")

ListKind = Literal["bullet", "number"]


@dataclass(frozen=True)
class Cite:
    """A citation position in the cleaned text and the facts it stands for."""

    at: int
    fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class BodyParagraph:
    text: str
    list_kind: ListKind | None
    list_marker: str  # "" for a plain paragraph, "-" or "3." for a list item
    cites: tuple[Cite, ...]
    table_row: bool = False

    def to_fill(self) -> list[tuple[int, int]]:
        """Ranges of ``[TO FILL: ...]`` markers in the text."""
        return [(m.start(), m.end()) for m in TO_FILL.finditer(self.text)]

    def sentences(self) -> list[tuple[int, int]]:
        """Sentence ranges (start, end) in the text; blanks are skipped."""
        return sentence_spans(self.text)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_BREAK.finditer(text):
        if text[start : match.start()].strip():
            spans.append((start, match.start()))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def strip_inline_markdown(text: str) -> str:
    text = _BOLD.sub(lambda m: m.group(2), text)
    text = _ITALIC.sub(lambda m: m.group(1), text)
    return _CODE.sub(lambda m: m.group(1), text)


def clean_markers(raw: str) -> tuple[str, tuple[Cite, ...]]:
    """Cut ``[[fN]]`` markers out of *raw*; return the text and the cite positions.

    A group of adjacent markers is one cite. Whitespace around a group is
    dropped (one space is put back when words would otherwise join), and a cite
    that follows a sentence's closing punctuation is placed before it.
    """
    text = ""
    cites: list[Cite] = []
    last = 0
    for match in MARKER_GROUP.finditer(raw):
        text += raw[last : match.start()]
        ids = tuple(_MARKER_ID.findall(match.group(0)))
        at = len(text)
        end = len(text)
        while end > 0 and text[end - 1] in _CLOSERS_AFTER_PUNCTUATION:
            end -= 1
        if end > 0 and text[end - 1] in _CLOSING:
            at = end - 1
        cites.append(Cite(at=at, fact_ids=ids))
        after = raw[match.end() :]
        if text and not text[-1].isspace() and after and not after[0].isspace():
            if after[0] not in ".,;:)]":
                text += " "
        last = match.end()
    text += raw[last:]
    return text, tuple(cites)


def _paragraph(
    raw: str, *, kind: ListKind | None, marker: str, table_row: bool = False
) -> BodyParagraph:
    text, cites = clean_markers(strip_inline_markdown(raw.strip()))
    return BodyParagraph(
        text=text.strip(),
        list_kind=kind,
        list_marker=marker,
        cites=cites if text == text.strip() else _shift(text, cites),
        table_row=table_row,
    )


def _shift(text: str, cites: tuple[Cite, ...]) -> tuple[Cite, ...]:
    """Re-base cite offsets after leading whitespace was stripped from *text*."""
    lead = len(text) - len(text.lstrip())
    size = len(text.strip())
    return tuple(
        Cite(at=min(max(c.at - lead, 0), size), fact_ids=c.fact_ids) for c in cites
    )


def parse_body(body: str) -> list[BodyParagraph]:
    """The paragraphs of a drafted section body, in order. Empty parts are dropped."""
    paragraphs: list[BodyParagraph] = []
    for block in re.split(r"\n\s*\n", body.strip()):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if all(_TABLE_ROW.match(line) for line in lines):
            for line in lines:
                if _TABLE_RULE.match(line):
                    continue
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                paragraphs.append(
                    _paragraph(" | ".join(cells), kind=None, marker="", table_row=True)
                )
            continue
        pending: tuple[ListKind | None, str, list[str]] | None = None
        for line in lines:
            bullet = _BULLET.match(line)
            numbered = _NUMBERED.match(line)
            if bullet or numbered:
                if pending is not None:
                    paragraphs.append(_flush(pending))
                if bullet:
                    pending = ("bullet", "-", [bullet.group(1)])
                else:
                    assert numbered is not None
                    pending = ("number", f"{numbered.group(1)}.", [numbered.group(2)])
            elif pending is None:
                pending = (None, "", [line.strip()])
            else:
                pending[2].append(line.strip())
        if pending is not None:
            paragraphs.append(_flush(pending))
    return [p for p in paragraphs if p.text]


def _flush(pending: tuple[ListKind | None, str, list[str]]) -> BodyParagraph:
    kind, marker, parts = pending
    return _paragraph(" ".join(parts), kind=kind, marker=marker)
