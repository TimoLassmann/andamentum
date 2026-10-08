"""Deterministic markdown chunking: headings, then paragraphs, then sentences."""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

from .structural import find_headings, fence_line_ranges

_SENT_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")
_CLOSING_HASHES_RE = re.compile(r"[ \t]+#+$")


@dataclass(frozen=True)
class Chunk:
    """One chunk of the source. ``text == source[start:end]``."""

    text: str
    start: int
    end: int
    heading_path: tuple[str, ...]


def chunk(text: str, *, max_chars: int) -> list[Chunk]:
    """Split markdown `text` into chunks of at most `max_chars` characters.

    Headings bound sections; an oversize section is split at paragraph breaks,
    then at sentence ends, then at whitespace. Fenced code blocks, tables and
    YAML front matter are never cut, so a chunk holding one may exceed
    `max_chars`. A single token longer than `max_chars` is cut hard.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be >= 1, got {max_chars}")
    if not text.strip():
        return []

    lines = text.split("\n")
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)

    blocks = _atomic_blocks(lines, offsets)
    headings = [
        h for h in find_headings(text) if not any(a <= h.start < b for a, b in blocks)
    ]
    hstarts = [h.start for h in headings]
    segments = _segments(lines, offsets, blocks, set(hstarts))

    # Group segments by section: 0 is the text before the first heading.
    groups: list[list[list]] = [[] for _ in range(len(headings) + 1)]
    for seg in segments:
        groups[bisect_right(hstarts, seg[0])].append(list(seg))

    paths = _heading_paths(headings)
    out: list[Chunk] = []
    pending: int | None = None  # start of heading-only sections awaiting a body
    for k, segs in enumerate(groups):
        if not segs:
            continue
        path: tuple[str, ...] = paths[k - 1] if k else ()
        if k:
            h = headings[k - 1]
            heading_end = h.start + len(text[h.start : h.line_end].rstrip())
            if len(segs) == 1 and segs[0][1] == heading_end and k < len(headings):
                pending = h.start if pending is None else pending
                continue
            if segs[0][1] == heading_end and len(segs) > 1:
                # A heading line never stands alone: it joins the next block.
                segs[0:2] = [[h.start, segs[1][1], segs[1][2]]]
        if pending is not None:
            segs[0][0] = pending
            pending = None
        for start, end in _pack(text, segs, max_chars):
            out.append(Chunk(text[start:end], start, end, path))
    return out


def _heading_paths(headings: list) -> list[tuple[str, ...]]:
    stack: list[tuple[int, str]] = []
    paths: list[tuple[str, ...]] = []
    for h in headings:
        title = _CLOSING_HASHES_RE.sub("", h.title)
        if title and set(title) == {"#"}:
            title = ""
        while stack and stack[-1][0] >= h.level:
            stack.pop()
        stack.append((h.level, title))
        paths.append(tuple(t for _, t in stack))
    return paths


def _atomic_blocks(lines: list[str], offsets: list[int]) -> list[tuple[int, int]]:
    """Character ranges of front matter, fenced code blocks and tables."""
    n = len(lines)
    line_blocks: list[tuple[int, int]] = []
    i = 0
    # YAML front matter: "---" on the first line, closed by "---" or "..."
    # before any blank line ("#" lines inside are YAML comments). A document
    # that merely opens with a horizontal rule has a blank line before its
    # next rule, so it is not mistaken for one atomic block.
    if lines and lines[0].rstrip() == "---":
        for j in range(1, n):
            line = lines[j].rstrip()
            if line in ("---", "..."):
                if j > 1:
                    line_blocks.append((0, j))
                    i = j + 1
                break
            if not line:
                break
    fences = [(a + i, b + i) for a, b in fence_line_ranges(lines[i:])]
    fi = 0
    while i < n:
        if fi < len(fences) and fences[fi][0] == i:
            line_blocks.append(fences[fi])
            i = fences[fi][1] + 1
            fi += 1
        elif lines[i].lstrip().startswith("|"):
            j = i
            while j + 1 < n and lines[j + 1].lstrip().startswith("|"):
                j += 1
            line_blocks.append((i, j))
            i = j + 1
        else:
            i += 1
    out: list[tuple[int, int]] = []
    for first, last in line_blocks:
        while last > first and not lines[last].strip():
            last -= 1
        out.append((offsets[first], offsets[last] + len(lines[last].rstrip())))
    return out


def _segments(
    lines: list[str],
    offsets: list[int],
    blocks: list[tuple[int, int]],
    heading_starts: set[int],
) -> list[tuple[int, int, bool]]:
    """Paragraphs and atomic blocks as ``(start, end, atomic)``, in source order."""
    out: list[tuple[int, int, bool]] = []
    block_at = {a: b for a, b in blocks}
    para: tuple[int, int] | None = None
    i, n = 0, len(lines)
    while i < n:
        o = offsets[i]
        if o in block_at:
            if para:
                out.append((para[0], para[1], False))
                para = None
            end = block_at[o]
            out.append((o, end, True))
            while i < n and offsets[i] <= end:
                i += 1
            continue
        if not lines[i].strip():
            if para:
                out.append((para[0], para[1], False))
                para = None
        else:
            line_end = o + len(lines[i].rstrip())
            if para and o in heading_starts:
                out.append((para[0], para[1], False))
                para = None
            para = (para[0], line_end) if para else (o, line_end)
        i += 1
    if para:
        out.append((para[0], para[1], False))
    return out


def _pack(text: str, segs: list[list], max_chars: int) -> list[tuple[int, int]]:
    """Greedily pack pieces of one section into spans of at most `max_chars`."""
    pieces: list[tuple[int, int]] = []
    for start, end, atomic in segs:
        if atomic or end - start <= max_chars:
            pieces.append((start, end))
        else:
            pieces.extend(_sentence_pieces(text, start, end, max_chars))
    spans: list[tuple[int, int]] = []
    cs, ce = pieces[0]
    for ps, pe in pieces[1:]:
        if pe - cs <= max_chars:
            ce = pe
        else:
            spans.append((cs, ce))
            cs, ce = ps, pe
    spans.append((cs, ce))
    return spans


def _sentence_pieces(
    text: str, start: int, end: int, max_chars: int
) -> list[tuple[int, int]]:
    body = text[start:end]
    out: list[tuple[int, int]] = []
    prev = 0
    for m in _SENT_END_RE.finditer(body):
        out.extend(_cut(text, start + prev, start + m.start(), max_chars))
        prev = m.end()
    out.extend(_cut(text, start + prev, end, max_chars))
    return out


def _cut(text: str, x: int, y: int, max_chars: int) -> list[tuple[int, int]]:
    """Cut [x, y) at whitespace into pieces of at most `max_chars`."""
    out: list[tuple[int, int]] = []
    while y - x > max_chars:
        k = x + max_chars
        while k > x and not text[k].isspace():
            k -= 1
        if k <= x:  # no whitespace to cut at: hard cut
            out.append((x, x + max_chars))
            x += max_chars
            continue
        out.append((x, x + len(text[x:k].rstrip())))
        x = k
        while x < y and text[x].isspace():
            x += 1
    if x < y:
        out.append((x, y))
    return out
