"""The claim and window helpers the arms map their answers with.

``best_window`` picks the stretch of a long page that shares most with a claim,
so a reader sees the passage that could support the claim rather than the top
of the page. The same window rule applies to every arm.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

WINDOW_CHARS = 2400
_WORD = re.compile(r"[a-z0-9]{4,}")


class ClaimEvidence(BaseModel):
    """One claim and the text that is supposed to support it."""

    claim: str
    evidence: str


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.casefold()))


def best_window(claim: str, texts: Sequence[str], *, width: int = WINDOW_CHARS) -> str:
    """The *width*-character stretch of *texts* with most of the claim's words.

    Windows start every half width; the earliest best one wins. Returns ``""``
    when there is no text.
    """
    wanted = _words(claim)
    best = ""
    best_score = -1
    for text in texts:
        starts = range(0, max(len(text) - width // 2, 1), max(width // 2, 1))
        for start in starts:
            window = text[start : start + width]
            score = len(wanted & _words(window))
            if score > best_score:
                best, best_score = window, score
    return best
