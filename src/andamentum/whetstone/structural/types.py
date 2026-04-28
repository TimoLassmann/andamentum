"""Internal types for structural extraction.

Plain dataclasses (not pydantic) because these are internal to the
review pipeline — they don't cross any external API boundary. Pydantic
overhead is unnecessary.

Char offsets in these types are relative to the SECTION's text, not the
whole document. The section's own ``char_start`` / ``char_end`` give
its position in the document; combine the two to get a global offset
when needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ── Sections ────────────────────────────────────────────────────────────


@dataclass
class SectionRef:
    """A section after chunking. Stable id; offsets in the original document."""

    id: str  # e.g. "sec_001"
    title: str
    text: str  # the section's verbatim markdown
    char_start: int  # global offset in the original document
    char_end: int


# ── Citations ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CitationKey:
    """A citation token found in the text.

    For numeric citations like ``[12]``, ``key`` is ``"12"``. For
    pandoc-flavour ``[@smith2020]`` it's ``"smith2020"``.
    """

    raw: str  # what appeared in the text, e.g. "[12]" or "[@smith2020]"
    key: str  # the citation key without delimiters
    style: Literal["numeric", "pandoc"]


@dataclass
class CitationOccurrence:
    """One place where a citation appears in the document."""

    key: CitationKey
    section_id: str
    char_start: int  # within the section
    char_end: int


@dataclass
class CitationGraph:
    """All citation activity in the document."""

    occurrences: list[CitationOccurrence] = field(default_factory=list)
    # Reference entries discovered in the references section(s).
    # key → entry text (one line, normalised whitespace).
    references_defined: dict[str, str] = field(default_factory=dict)
    # The section_ids of all sections that constitute the references list.
    # Real papers often have references split across MULTIPLE chunker units
    # (the chunker can't fit a long bibliography in one section), so this is
    # a list, not a single id. Empty when no references list was found.
    references_section_ids: list[str] = field(default_factory=list)


# ── Terms / acronyms ────────────────────────────────────────────────────


@dataclass
class TermDefinition:
    """An acronym or technical term defined in one section."""

    term: str  # e.g. "MCC"
    expansion: str  # e.g. "Minimal Criterion Coevolution"
    section_id: str
    char_start: int
    char_end: int


@dataclass
class TermUsage:
    """A use of a term (whether or not it's been defined)."""

    term: str
    section_id: str
    char_start: int
    char_end: int


@dataclass
class TermGlossary:
    definitions: list[TermDefinition] = field(default_factory=list)
    usages: list[TermUsage] = field(default_factory=list)


# ── Numeric claims ──────────────────────────────────────────────────────


@dataclass
class NumericClaim:
    """A numeric value mentioned in the text."""

    raw: str  # the literal text matched, e.g. "N=50" or "p<0.05"
    kind: Literal["sample_size", "percentage", "p_value", "count"]
    value: str  # the numeric portion as text, e.g. "50" or "0.05"
    section_id: str
    char_start: int
    char_end: int


# ── Cross-references ────────────────────────────────────────────────────


@dataclass
class CrossReference:
    """A "see Figure 3" / "see Section 2.1" / etc. reference."""

    raw: str  # the literal text matched, e.g. "Figure 3"
    kind: Literal["figure", "table", "section", "equation"]
    target: str  # the target identifier, e.g. "3" or "2.1"
    section_id: str
    char_start: int
    char_end: int


# ── Aggregate ───────────────────────────────────────────────────────────


@dataclass
class StructuralFacts:
    """All deterministic facts extracted from the document.

    Computed once by ChunkAndScan, then read by the deterministic-findings
    synthesiser AND by every later LLM agent (so they don't have to
    re-discover things they could just look up).
    """

    citation_graph: CitationGraph = field(default_factory=CitationGraph)
    term_glossary: TermGlossary = field(default_factory=TermGlossary)
    numeric_claims: list[NumericClaim] = field(default_factory=list)
    cross_references: list[CrossReference] = field(default_factory=list)
