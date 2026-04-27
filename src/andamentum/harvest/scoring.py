"""Score the structural quality of an extracted markdown document.

Used by the HTML race path: when both trafilatura and Docling produced
markdown for the same page, we pick the one that preserved more structure
(headings, paragraph breaks) without drowning in link-decoration noise.
"""

from __future__ import annotations

import re

_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")

# Tunables — chosen empirically from the BBC-homepage / arXiv-PDF / clean-blog
# spectrum. Heading count is the strongest single signal; everything else is
# a tie-breaker.
_HEADING_WEIGHT = 10.0
_PARAGRAPH_WEIGHT = 1.0
_CHAR_WEIGHT = 0.001
_LINK_DENSITY_PENALTY_THRESHOLD = 0.05  # >5% chars in link decoration
_LINK_DENSITY_PENALTY_WEIGHT = 1_000.0
_DISQUALIFYING_PENALTY = -100_000.0


def score_markdown(md: str) -> float:
    """Higher = better.

    Heuristic combining heading count, paragraph density, char count, and a
    penalty for link-decoration spam (the symptom of an extractor that kept
    every `<a>` as `[text](url)` even when it's noise).

    Markdown with zero headings AND zero paragraph breaks is considered
    structurally collapsed (the BBC-homepage soup symptom) and gets a large
    disqualifying penalty so it never wins a race against a structured peer.
    """
    if not md:
        return _DISQUALIFYING_PENALTY

    heading_count = len(_HEADING_RE.findall(md))
    paragraph_count = md.count("\n\n")
    char_count = len(md)
    link_chars = sum(len(m.group(0)) for m in _LINK_RE.finditer(md))
    link_density = link_chars / char_count if char_count else 0.0

    score = (
        heading_count * _HEADING_WEIGHT
        + paragraph_count * _PARAGRAPH_WEIGHT
        + char_count * _CHAR_WEIGHT
    )
    if link_density > _LINK_DENSITY_PENALTY_THRESHOLD:
        score -= link_density * _LINK_DENSITY_PENALTY_WEIGHT
    if heading_count == 0 and paragraph_count == 0:
        score += _DISQUALIFYING_PENALTY
    return score
