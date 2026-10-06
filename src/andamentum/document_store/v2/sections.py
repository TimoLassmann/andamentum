"""Split a markdown document into sections that keep their line ranges.

A section starts at a markdown heading and runs to the next heading of any
level. Text before the first heading is a level-0 section with an empty
heading. A section longer than ``max_chars`` is cut at paragraph boundaries
(blank lines), so the embedding model and the reranker see passages within
their input limits; a single paragraph longer than the cap stays whole.

Line numbers are 1-based and inclusive, so a citation is ``path:line_from-line_to``
and ``read(path, line_from, ...)`` returns the same lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Section:
    heading: str
    level: int
    line_from: int
    line_to: int
    text: str


def _trim(lines: list[str], start: int) -> tuple[int, int] | None:
    """First and last non-blank line indices of ``lines`` offset by ``start``."""
    idx = [i for i, line in enumerate(lines) if line.strip()]
    if not idx:
        return None
    return start + idx[0], start + idx[-1]


def split_sections(text: str, *, max_chars: int) -> list[Section]:
    lines = text.split("\n")
    # (heading, level, first line index) for each heading outside code fences
    starts: list[tuple[str, int, int]] = [("", 0, 0)]
    in_fence = False
    for i, line in enumerate(lines):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        m = None if in_fence else _HEADING.match(line)
        if m:
            starts.append((m.group(2), len(m.group(1)), i))
    if len(starts) > 1 and starts[1][2] == 0:
        starts.pop(0)  # the document opens with a heading

    out: list[Section] = []
    for n, (heading, level, first) in enumerate(starts):
        end = starts[n + 1][2] if n + 1 < len(starts) else len(lines)
        span = _trim(lines[first:end], first)
        if span is None:
            continue
        lo, hi = span
        out.extend(
            _cap(lines, heading=heading, level=level, lo=lo, hi=hi, max_chars=max_chars)
        )
    return out


def _cap(
    lines: list[str], *, heading: str, level: int, lo: int, hi: int, max_chars: int
) -> list[Section]:
    """Pack paragraphs of lines[lo..hi] into pieces of at most ``max_chars``."""

    def section(a: int, b: int) -> Section:
        return Section(heading, level, a + 1, b + 1, "\n".join(lines[a : b + 1]))

    if len("\n".join(lines[lo : hi + 1])) <= max_chars:
        return [section(lo, hi)]
    paragraphs: list[tuple[int, int]] = []
    start = None
    for i in range(lo, hi + 1):
        if lines[i].strip():
            if start is None:
                start = i
        elif start is not None:
            paragraphs.append((start, i - 1))
            start = None
    if start is not None:
        paragraphs.append((start, hi))
    if (
        len(paragraphs) > 1
        and paragraphs[0][0] == paragraphs[0][1]
        and _HEADING.match(lines[lo])
    ):
        # A heading line is never a piece of its own; it opens the first one.
        paragraphs[:2] = [(paragraphs[0][0], paragraphs[1][1])]

    pieces: list[Section] = []
    a, b = paragraphs[0]
    for p_lo, p_hi in paragraphs[1:]:
        if len("\n".join(lines[a : p_hi + 1])) > max_chars:
            pieces.append(section(a, b))
            a = p_lo
        b = p_hi
    pieces.append(section(a, b))
    return pieces
