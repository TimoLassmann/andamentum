"""Deterministic parts of the document model — no LLM.

Citation detection, `has_citation`, section gists, and assembling the verified
claims + deterministic facets into a `DocumentModel`.
"""

from __future__ import annotations

import re

from .model import Citation, Claim, DocumentModel, Section, SectionGist

# In-text citation markers: numeric [12] / [12, 13] / [12-15], Pandoc [@key],
# and author-year (Smith et al., 2020) / (Smith and Jones, 2019).
_CITATION_RES = [
    re.compile(r"\[\d+(?:\s*[,–\-]\s*\d+)*\]"),
    re.compile(r"\[@[\w:.\-]+\]"),
    re.compile(
        r"\([A-Z][A-Za-z]+(?: et al\.?| (?:and|&) [A-Z][A-Za-z]+)?,?\s*\d{4}[a-z]?\)"
    ),
]


def find_citation_markers(text: str) -> list[str]:
    """All in-text citation markers in *text*, in order of appearance."""
    hits: list[tuple[int, str]] = []
    for rx in _CITATION_RES:
        for m in rx.finditer(text):
            hits.append((m.start(), m.group(0)))
    return [m for _, m in sorted(hits)]


def has_citation(text: str) -> bool:
    """True if a citation marker appears in *text* (e.g. a claim's quote)."""
    return any(rx.search(text) for rx in _CITATION_RES)


def _heading_stripped(section: Section) -> str:
    """Section body with a leading markdown heading line removed."""
    body = section.text.lstrip()
    if body.startswith("#"):
        nl = body.find("\n")
        body = body[nl + 1 :] if nl != -1 else ""
    return body.strip()


def gist_for(section: Section, *, max_chars: int = 160) -> str:
    """Deterministic one-line gist: the first sentence of the section body."""
    body = " ".join(_heading_stripped(section).split())
    if not body:
        return ""
    # First sentence: up to a sentence-ending punctuation, else a char cap.
    m = re.search(r"[.!?](?:\s|$)", body)
    sentence = body[: m.end()].strip() if m else body
    if len(sentence) > max_chars:
        sentence = sentence[: max_chars - 1].rstrip() + "…"
    return sentence


def build_document_model(
    source: str, sections: list[Section], claims: list[Claim]
) -> DocumentModel:
    """Assemble verified claims + deterministic gists/citations into the model.

    Claims arrive already located (Phase-1 verify); here we set `has_citation`
    deterministically and add the deterministic facets.
    """
    for c in claims:
        c.has_citation = has_citation(c.quote)

    gists = [
        SectionGist(section_id=s.id, title=s.title, gist=gist_for(s)) for s in sections
    ]
    citations = [
        Citation(marker=marker, section_id=s.id)
        for s in sections
        for marker in find_citation_markers(s.text)
    ]
    return DocumentModel(
        source=source,
        sections=sections,
        claims=claims,
        gists=gists,
        citations=citations,
    )
