"""The v3 document model — the compact, verified representation of a draft.

Built once per run. The ONLY thing an LLM extracts is claims (verbatim spans);
everything else (gists, citations, has_citation) is deterministic. Support /
links / equations are NOT stored — criterion stages read the real source on
demand. Every located item carries a `Span` so the source can be re-read and
findings anchored.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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
    has_citation: bool = False  # deterministic: a citation marker sits in the span


class SectionGist(BaseModel):
    """A one-line deterministic summary of a section (title + first sentence)."""

    section_id: str
    title: str
    gist: str


class Citation(BaseModel):
    """An in-text citation marker found in the source (deterministic)."""

    marker: str  # e.g. "[12]", "[@smith2020]", "(Smith et al., 2020)"
    section_id: str


class DocumentModel(BaseModel):
    """The shared store every reasoning stage queries.

    Holds the located claims + deterministic gists/citations, plus the sections
    and full source so a stage can read the real text on demand.
    """

    source: str
    sections: list[Section] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    gists: list[SectionGist] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)

    def section_by_id(self, section_id: str) -> Section | None:
        return next((s for s in self.sections if s.id == section_id), None)
