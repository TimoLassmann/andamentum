"""Deterministic sectioniser — splits the source into reviewable units.

Reuses the chunker's structural layer (heading detection + size-banded
recursive splitting), so units are heading-aware, carry char offsets back to
the source, and are bounded in size (extraction runs per unit, so each must be
small enough for a weak model). No LLM, no embeddings.
"""

from __future__ import annotations

from andamentum.chunker.structural import (
    Heading,
    build_section_tree,
    find_headings,
    split_section_recursively,
)

from .model import Section

DEFAULT_TARGET_MAX = 6000


def sectionize(source: str, *, target_max: int = DEFAULT_TARGET_MAX) -> list[Section]:
    """Split *source* into heading-aware, size-banded sections with offsets.

    Headings come from markdown ``#`` lines — the right rule for markdown, PDF
    and HTML input, where that is the only structure the text carries. A Word
    document says which paragraphs are headings in their style, and is
    sectionised by :func:`~andamentum.whetstone.pipeline.docx_source.sectionize_docx`
    instead; both funnel into :func:`sectionize_from_headings`, so the banding
    and the ``Section`` contract are shared rather than reimplemented.
    """
    return sectionize_from_headings(
        source, find_headings(source), target_max=target_max
    )


def sectionize_from_headings(
    source: str, headings: list[Heading], *, target_max: int = DEFAULT_TARGET_MAX
) -> list[Section]:
    """Band *source* into sections at the given headings.

    ``headings`` must be in document order with offsets into *source*; where
    they came from is not this function's business.
    """
    pieces: list[tuple[int, int, str, int]] = []  # (start, end, title, level)

    if not headings:
        pieces.append((0, len(source), "(document)", 0))
    else:
        first = headings[0].start
        if source[:first].strip():
            pieces.append((0, first, "(preamble)", 0))
        for top in build_section_tree(source, headings):
            for piece in split_section_recursively(top, target_max):
                pieces.append((piece.start, piece.end, piece.title, piece.level))

    # A parent too large for one section is split into its children plus an
    # intro piece for the text before the first child. When that text is only
    # the heading ("Results" straight into its first subsection), the intro
    # piece is a section of one word, and a reviewer that reads it reports an
    # empty Results section. Such a heading joins the piece after it, when
    # that piece is its descendant, and is named in its ``parents``. A heading
    # whose next piece is a sibling stays: that section is empty for real.
    out: list[Section] = []
    held_start: int | None = None
    held_titles: list[str] = []
    for i, (start, end, title, level) in enumerate(pieces, 1):
        text = source[start:end]
        if not text.strip():
            continue
        next_level = pieces[i][3] if i < len(pieces) else 0
        if next_level > level and _heading_only(text, title):
            if held_start is None:
                held_start = start
            held_titles.append(title.strip())
            continue
        if held_start is not None:
            start, text = held_start, source[held_start:end]
        out.append(
            Section(
                id=f"s{i}",
                title=title.strip() or f"s{i}",
                text=text,
                start=start,
                end=end,
                parents=held_titles,
            )
        )
        held_start, held_titles = None, []
    return out


def _heading_only(text: str, title: str) -> bool:
    """True when *text* is its heading line and nothing more.

    A markdown section opens with ``## Title``; a Word section with the bare
    title, because the projection holds only what a reader sees.
    """
    first, _, rest = text.strip().partition("\n")
    return (
        first.startswith("#") or first.strip() == title.strip()
    ) and not rest.strip()
