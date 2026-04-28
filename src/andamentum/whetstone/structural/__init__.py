"""Deterministic structural extractors for whetstone v2.

Everything in this package is pure-Python text processing — no LLM calls,
no embeddings, no network. The output is ``StructuralFacts``, which the
deterministic-findings synthesiser turns into actionable Findings.

This is the "free" layer: useful output before any LLM has been spent.
"""

from .types import (
    CitationGraph,
    CitationKey,
    CitationOccurrence,
    CrossReference,
    NumericClaim,
    SectionRef,
    StructuralFacts,
    TermDefinition,
    TermGlossary,
    TermUsage,
)

__all__ = [
    "CitationGraph",
    "CitationKey",
    "CitationOccurrence",
    "CrossReference",
    "NumericClaim",
    "SectionRef",
    "StructuralFacts",
    "TermDefinition",
    "TermGlossary",
    "TermUsage",
]
