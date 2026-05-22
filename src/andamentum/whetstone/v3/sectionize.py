"""Deterministic sectioniser — splits the source into reviewable units.

Reuses the chunker's structural layer (heading detection + size-banded
recursive splitting), so units are heading-aware, carry char offsets back to
the source, and are bounded in size (extraction runs per unit, so each must be
small enough for a weak model). No LLM, no embeddings.
"""

from __future__ import annotations

from andamentum.chunker.structural import (
    build_section_tree,
    find_headings,
    split_section_recursively,
)

from .model import Section

_DEFAULT_TARGET_MAX = 6000


def sectionize(source: str, *, target_max: int = _DEFAULT_TARGET_MAX) -> list[Section]:
    """Split *source* into heading-aware, size-banded sections with offsets."""
    headings = find_headings(source)
    pieces: list[tuple[int, int, str]] = []  # (start, end, title)

    if not headings:
        pieces.append((0, len(source), "(document)"))
    else:
        first = headings[0].start
        if source[:first].strip():
            pieces.append((0, first, "(preamble)"))
        for top in build_section_tree(source, headings):
            for piece in split_section_recursively(top, target_max):
                pieces.append((piece.start, piece.end, piece.title))

    out: list[Section] = []
    for i, (start, end, title) in enumerate(pieces, 1):
        text = source[start:end]
        if not text.strip():
            continue
        out.append(
            Section(
                id=f"s{i}",
                title=title.strip() or f"s{i}",
                text=text,
                start=start,
                end=end,
            )
        )
    return out
