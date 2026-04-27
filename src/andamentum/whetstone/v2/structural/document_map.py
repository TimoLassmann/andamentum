"""Build a deterministic DocumentMap from the chunker's section list.

The map is built purely from the section's heading + first paragraph.
It feeds the reflection prompt — gives the senior reviewer a compact
overview of the manuscript so it can name section ids when proposing
investigation tasks. The map is computed once during ChunkAndScan and
remains stable across the reflection loop.
"""

from __future__ import annotations

import re

from ..schemas import SectionCard
from .types import SectionRef

_FIRST_SENTENCE_RE = re.compile(r"^([^.!?]{20,400}[.!?])")


def build_document_map(sections: list[SectionRef]) -> list[SectionCard]:
    """One SectionCard per section, with title and a deterministic gist."""
    return [
        SectionCard(
            section_id=section.id,
            title=section.title or section.id,
            one_line_gist=_extract_first_sentence(section),
        )
        for section in sections
    ]


def _extract_first_sentence(section: SectionRef) -> str:
    """Return the first non-trivial sentence of the section.

    Skips the heading line(s) and any whitespace-only lines, then returns
    the first sentence (20-400 chars) we encounter. Falls back to the
    section's heading if no sentence is found (e.g. a figure-only section).
    """
    text = _strip_leading_headings(section.text)
    m = _FIRST_SENTENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fall back to the first non-empty line, truncated.
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return ""


def _strip_leading_headings(text: str) -> str:
    """Drop leading lines that are markdown headings or blank."""
    lines = text.splitlines()
    out_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out_start = i + 1
            continue
        break
    return "\n".join(lines[out_start:])
