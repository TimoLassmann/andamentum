"""A Word document, sectionised in its own coordinates.

The text handed to the reviewer is :func:`~andamentum.whetstone.docx.atoms.projection`
— the atom stream of the document's own paragraphs — and the sections index
into exactly that string. So a quote the reviewer returns is already an address
in the document: no conversion round-trip, no realignment, no similarity
threshold. (The converted-markdown route lost 15 of 43 rewrites on one real
manuscript, every one of them reporting a nearest match at 0.97 or better.)

The markdown route is unchanged and still correct for its inputs — see
:func:`~andamentum.whetstone.pipeline.sectionize.sectionize`. Both produce the same
``Section`` shape, and both leave ``section.text == text[section.start:section.end]``
true, so a consumer never has to know which one ran.

Imported lazily by callers: this module pulls in lxml, and a markdown review
should not pay for it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from andamentum.chunker.structural import Heading

from ..docx.atoms import document_atoms, projection
from ..docx.headings import document_headings
from ..docx.reading import read_docx
from .model import Section
from .sectionize import DEFAULT_TARGET_MAX, sectionize_from_headings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceDocument:
    """The text under review plus its sections, in one coordinate system."""

    text: str
    sections: list[Section]


def sectionize_docx(
    paragraphs: Sequence[Any],
    *,
    styles_root: Any | None,
    target_max: int = DEFAULT_TARGET_MAX,
) -> SourceDocument:
    """Project *paragraphs* to text and split it at the Word heading styles."""
    stream = document_atoms(paragraphs)
    text = projection(stream)
    found = document_headings(paragraphs, stream=stream, styles_root=styles_root)

    # ``line_end`` is the markdown heading's newline, which a Word heading does
    # not have; the field is unread by the banding, so the paragraph's end
    # stands in for it rather than inventing an offset.
    headings = [
        Heading(start=h.start, line_end=h.end, level=h.level, title=h.title)
        for h in found
    ]
    sections = sectionize_from_headings(text, headings, target_max=target_max)

    # A document whose headings are a custom style with no outline level, no
    # ``heading N`` name and no ``basedOn`` chain reads as one giant section.
    # That failure is otherwise silent — it surfaces only as a thin review.
    logger.info(
        "docx sectionise: %d paragraphs, %d chars, %d styled headings, %d sections",
        len(paragraphs),
        len(text),
        len(headings),
        len(sections),
    )
    return SourceDocument(text=text, sections=sections)


def load_docx(
    path: str | Path, *, target_max: int = DEFAULT_TARGET_MAX
) -> SourceDocument:
    """Read a ``.docx`` off disk and sectionise it. No temporary files."""
    parts = read_docx(path)
    return sectionize_docx(
        parts.paragraphs, styles_root=parts.styles_root, target_max=target_max
    )
