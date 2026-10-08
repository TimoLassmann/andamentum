"""The document model — the compact, verified representation of a draft.

Built once per run. The ONLY thing an LLM extracts is claims (verbatim spans);
everything else (gists, the citation summary) is deterministic. Support /
links / equations are NOT stored — criterion stages read the real source on
demand. Every located item carries a `Span` so the source can be re-read and
findings anchored.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["minor", "moderate", "major"]
"""How much work the author must do. Shared by every mode that emits findings.

  - ``major``    — conclusions, validity, or reproducibility are at stake.
  - ``moderate`` — a real weakness a competent reader will notice.
  - ``minor``    — a local improvement; safe to ignore, nice to fix.
"""


class Span(BaseModel):
    """A located char range in the source, with its origin section."""

    section_id: str
    start: int  # inclusive, source coordinates
    end: int  # exclusive


class Section(BaseModel):
    """One reviewable unit (heading-delimited, size-banded), with offsets."""

    id: str
    title: str
    text: str
    start: int  # offset of the section in the full source
    end: int


class Claim(BaseModel):
    """An assertion the document makes, as a located verbatim span."""

    id: str
    quote: str  # verbatim span from the source
    span: Span


class SectionGist(BaseModel):
    """A one-line deterministic summary of a section (title + first sentence)."""

    section_id: str
    title: str
    gist: str


class CitationSummary(BaseModel):
    """How the document cites, as the reference checker reads it.

    Read by :mod:`andamentum.references` with no model and no lookup — the
    same reader as ``andamentum-references``, so a review and a
    reference check cannot disagree about whether a draft cites anything.
    ``style`` is that checker's style id (``"undecided"`` when the text does
    not settle it), or ``None`` when there is neither a list nor citing notes.
    """

    found_list: bool
    entries: int = 0
    style: str | None = None
    citations: int = 0


class DocumentModel(BaseModel):
    """The shared store every reasoning stage queries.

    Holds the located claims + deterministic gists and citation summary, plus
    the sections and full source so a stage can read the real text on demand.
    ``citations`` is ``None`` when the run did not read them — checklist mode,
    whose prompts never show them.
    """

    source: str
    sections: list[Section] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    gists: list[SectionGist] = Field(default_factory=list)
    citations: CitationSummary | None = None

    def section_by_id(self, section_id: str) -> Section | None:
        return next((s for s in self.sections if s.id == section_id), None)
