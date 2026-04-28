"""Lightweight section-kind classifier.

Maps section heading text to a coarse IMRAD-ish enum so lenses can
target only the section kinds where they're load-bearing. Heuristic,
not LLM — heading-text matching is reliable enough that a deterministic
classifier is the right tool.

Kinds:
  abstract     — Abstract / Summary / Synopsis
  introduction — Introduction / Background / Motivation
  methods      — Methods / Materials and Methods / Experimental
  results      — Results / Findings / Observations
  discussion   — Discussion / Interpretation
  conclusion   — Conclusion / Conclusions / Summary (tail)
  references   — References / Bibliography / Citations
  other        — anything else (e.g. acknowledgements, appendix, COI,
                  data availability, supplementary, unrelated headings)
"""

from __future__ import annotations

import re
from typing import Literal

SectionKind = Literal[
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "references",
    "other",
]

ALL_KINDS: frozenset[SectionKind] = frozenset(
    {
        "abstract",
        "introduction",
        "methods",
        "results",
        "discussion",
        "conclusion",
        "references",
        "other",
    }
)


# Patterns are checked in order. First match wins. Patterns are
# case-insensitive and require the keyword to appear at the start of
# the heading (after optional numbering like "1. " or "1.2 ").
_PATTERNS: list[tuple[SectionKind, re.Pattern[str]]] = [
    ("abstract", re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:abstract|summary|synopsis)\b", re.IGNORECASE)),
    ("introduction", re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:introduction|background|motivation)\b", re.IGNORECASE)),
    ("methods", re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:methods?|materials\s+and\s+methods?|experimental|methodology|procedure)\b", re.IGNORECASE)),
    ("results", re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:results?|findings?|observations?)\b", re.IGNORECASE)),
    ("discussion", re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:discussion|interpretation)\b", re.IGNORECASE)),
    ("conclusion", re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:conclusions?|concluding\s+remarks?)\b", re.IGNORECASE)),
    ("references", re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:references?|bibliography|citations?|works\s+cited)\b", re.IGNORECASE)),
]


def classify_section_kind(title: str) -> SectionKind:
    """Return the kind enum for a section heading.

    Strips the leading hash characters of a markdown heading if present
    so callers can pass either ``"## Methods"`` or ``"Methods"``.
    """
    cleaned = title.lstrip("#").strip()
    for kind, pattern in _PATTERNS:
        if pattern.match(cleaned):
            return kind
    return "other"
