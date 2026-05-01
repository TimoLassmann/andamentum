"""General academic identifier extractor.

Phase 3 of the epistemic efficiency plan. Pulls DOIs, PMIDs, and arXiv
IDs out of arbitrary text (URL, source reference, content body). Used
by the quality-scoring path so more evidence items hit Path 1 (free
bibliometric lookup, e.g. via OpenAlex) instead of Path 2 (LLM-based
quality assessment).

This module is provider-agnostic. DOIs / PMIDs / arXiv IDs are universal
academic infrastructure; recognising them in raw text is a general
capability, not a per-provider rule. The bibliometric resolver (currently
``OpenAlexQualityScorer``) consumes the identifiers via the
``quality_scorer`` Protocol — a future Crossref or Semantic Scholar
implementation would reuse the same extractor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Identifiers:
    """Identifiers extracted from a source. All fields optional — None
    means "not found", not "found but empty"."""

    doi: Optional[str] = None
    pmid: Optional[str] = None
    arxiv: Optional[str] = None

    @property
    def has_any(self) -> bool:
        """True if at least one identifier was extracted."""
        return any((self.doi, self.pmid, self.arxiv))


# DOI: ``10.<registrant>/<suffix>``. The registrant is 4-9 digits per the
# Crossref spec; the suffix is broad — any printable character except
# whitespace, but we exclude trailing punctuation that's likely part of
# the surrounding text (period, comma, semicolon, parens, brackets,
# quote marks). Match is non-greedy on the suffix so consecutive DOIs
# don't merge.
_DOI_PATTERN = re.compile(
    r"\b10\.\d{4,9}/[^\s\)\]\>\"';,]+",
    re.IGNORECASE,
)

# PMID: pure digit string preceded by a recognisable cue. We don't
# match bare digit strings (would have far too many false positives in
# scientific text). Recognised cues: ``PMID:`` / ``pmid:`` / ``pmid=``
# / ``pubmed/`` / ``pubmed.ncbi.nlm.nih.gov/``. PMIDs are typically
# 1-8 digits.
_PMID_PATTERN = re.compile(
    r"(?:"
    r"PMID[:\s=]+"
    r"|pubmed[/.](?:ncbi\.nlm\.nih\.gov/)?"
    r")(\d{1,9})\b",
    re.IGNORECASE,
)

# arXiv: post-2007 format ``YYMM.NNNNN`` (4-5 digits after the dot).
# Pre-2007 IDs were ``archive/YYMMNNN`` but rare for current evidence
# so we ignore them.
_ARXIV_PATTERN = re.compile(
    r"\barXiv[:\s]*(\d{4}\.\d{4,5})\b",
    re.IGNORECASE,
)


def extract_identifiers(*texts: Optional[str]) -> Identifiers:
    """Extract DOI / PMID / arXiv identifiers from one or more text
    inputs. Searches all inputs and returns the first match for each
    identifier type.

    Args:
        *texts: Strings to search. ``None`` and empty strings are
            skipped. Typical usage::

                extract_identifiers(evidence.source_ref,
                                    evidence.extracted_content[:1000])

    Returns:
        ``Identifiers`` with whatever was found. ``has_any`` is True
        when at least one identifier was extracted.
    """
    doi: Optional[str] = None
    pmid: Optional[str] = None
    arxiv: Optional[str] = None

    for text in texts:
        if not text:
            continue
        if doi is None:
            m = _DOI_PATTERN.search(text)
            if m:
                # Strip a trailing period that often clings to DOIs in
                # prose ("...see doi:10.1234/foo. The next sentence").
                doi = m.group(0).rstrip(".")
        if pmid is None:
            m = _PMID_PATTERN.search(text)
            if m:
                pmid = m.group(1)
        if arxiv is None:
            m = _ARXIV_PATTERN.search(text)
            if m:
                arxiv = m.group(1)

        # Early exit if we've found all three.
        if doi and pmid and arxiv:
            break

    return Identifiers(doi=doi, pmid=pmid, arxiv=arxiv)
