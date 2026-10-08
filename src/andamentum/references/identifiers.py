"""General academic identifier extractor.

Pulls DOIs, PMIDs, PMCIDs, arXiv IDs, ISBNs and URLs out of arbitrary text
(URL, source reference, content body). Provider-agnostic: these are universal
bibliographic infrastructure, so recognising them in raw text is a general
capability. Epistemic's quality scorer consumes the DOI / PMID / arXiv
identifiers; the document store's enrichers consume all of them.

Bare digit strings are never matched (far too many false positives in
scientific text): a PMID or ISBN needs a recognisable cue, and an ISBN also
needs a valid checksum.
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
    pmcid: Optional[str] = None
    isbn: Optional[str] = None
    urls: tuple[str, ...] = ()

    @property
    def has_any(self) -> bool:
        """True if a DOI, PMID or arXiv identifier was extracted.

        Deliberately NOT over ``pmcid``, ``isbn`` and ``urls``: this is the
        question "can a bibliometric lookup (OpenAlex) resolve this source?",
        which epistemic's quality scorer asks, and an ISBN or a bare URL does
        not answer it. Use :attr:`has_bibliographic` for "anything at all".
        """
        return any((self.doi, self.pmid, self.arxiv))

    @property
    def has_bibliographic(self) -> bool:
        """True if any identifier at all (including PMCID, ISBN, URLs) was found."""
        return self.has_any or bool(self.pmcid or self.isbn or self.urls)


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


# PMCID: ``PMC`` followed by digits.
_PMCID_PATTERN = re.compile(r"\bPMC(\d{1,10})\b")

# ISBN: needs the cue ``ISBN`` (optionally ``ISBN-10`` / ``ISBN-13``); the
# candidate is digits with optional hyphens/spaces, the last character may be X.
_ISBN_PATTERN = re.compile(
    r"\bISBN(?:-1[03])?[:\s=]*([0-9][0-9\- ]{8,16}[0-9Xx])\b",
    re.IGNORECASE,
)

# URL: http(s) up to whitespace or a closing delimiter; trailing punctuation
# is stripped afterwards.
_URL_PATTERN = re.compile(r"https?://[^\s<>\"\[\]{}|\\^`]+", re.IGNORECASE)
_URL_TRAILING = ".,;:!?'\""


def _valid_isbn(raw: str) -> Optional[str]:
    """The digits of *raw* when it is a valid ISBN-10 or ISBN-13, else None."""
    digits = re.sub(r"[\s-]", "", raw).upper()
    if len(digits) == 10 and re.fullmatch(r"\d{9}[\dX]", digits):
        total = sum(
            (10 if c == "X" else int(c)) * (10 - i) for i, c in enumerate(digits)
        )
        return digits if total % 11 == 0 else None
    if len(digits) == 13 and digits.isdigit():
        total = sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(digits))
        return digits if total % 10 == 0 else None
    return None


def _clean_url(url: str) -> str:
    """Strip trailing punctuation, and a closing ``)`` that has no opener."""
    while url:
        if url[-1] in _URL_TRAILING:
            url = url[:-1]
        elif url[-1] == ")" and url.count(")") > url.count("("):
            url = url[:-1]
        else:
            break
    return url


def extract_identifiers(*texts: Optional[str]) -> Identifiers:
    """Extract DOI / PMID / PMCID / arXiv / ISBN identifiers and URLs from one
    or more text inputs. Searches all inputs and returns the first match for
    each identifier type; ``urls`` collects every distinct URL in order.

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
    pmcid: Optional[str] = None
    isbn: Optional[str] = None
    urls: list[str] = []

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

        if pmcid is None:
            m = _PMCID_PATTERN.search(text)
            if m:
                pmcid = f"PMC{m.group(1)}"
        if isbn is None:
            for m in _ISBN_PATTERN.finditer(text):
                isbn = _valid_isbn(m.group(1))
                if isbn:
                    break
        for m in _URL_PATTERN.finditer(text):
            url = _clean_url(m.group(0))
            if url and url not in urls:
                urls.append(url)

    return Identifiers(
        doi=doi,
        pmid=pmid,
        arxiv=arxiv,
        pmcid=pmcid,
        isbn=isbn,
        urls=tuple(urls),
    )
