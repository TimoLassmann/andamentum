"""Build a deterministic DocumentMap from the chunker's section list.

In Phase 1 the map is built purely from the section's heading + first
paragraph. The skim_agent (Phase 2) will enrich the ``one_line_gist``
field with an LLM-written summary; the deterministic version here gives
us something useful immediately AND is what skim_agent's output replaces.

This means: the rest of the pipeline can ALWAYS rely on a populated
DocumentMap, regardless of whether the LLM has run yet.
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
