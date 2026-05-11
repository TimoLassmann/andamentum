"""Weasel-word detector.

Matt Might's "weasel words" list (from his ``weasel-words.el`` Emacs
package), extended slightly with hedge words common in academic writing.
A weasel word is a vague qualifier that weakens a claim without adding
information ("many studies", "fairly accurate", "various reasons").
"""

from __future__ import annotations

import re

from .models import Span, WeaselFinding
from .sentences import Sentence, index_for_offset

# Matt Might's core list, plus common academic hedges. Lowercased.
WEASEL_WORDS: tuple[str, ...] = (
    # Matt Might
    "many",
    "various",
    "very",
    "fairly",
    "several",
    "extremely",
    "exceedingly",
    "quite",
    "remarkably",
    "few",
    "surprisingly",
    "mostly",
    "largely",
    "huge",
    "tiny",
    "excellent",
    "interestingly",
    "significantly",
    "substantially",
    "clearly",
    "vast",
    "relatively",
    "completely",
    # Common academic hedges
    "somewhat",
    "rather",
    "almost",
    "nearly",
    "approximately",
    "roughly",
    "about",
    "often",
    "usually",
    "frequently",
    "occasionally",
    "generally",
    "essentially",
    "basically",
    "arguably",
    "presumably",
    "seemingly",
    "apparently",
    "notably",
    "particularly",
    "considerably",
    "dramatically",
)

_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in WEASEL_WORDS) + r")\b",
    re.IGNORECASE,
)


def scan(text: str, sentences: list[Sentence]) -> list[WeaselFinding]:
    """Return every weasel-word occurrence in ``text`` with offsets and
    sentence index. Matches are case-insensitive but the returned ``word``
    field preserves the source casing."""
    findings: list[WeaselFinding] = []
    for m in _PATTERN.finditer(text):
        findings.append(
            WeaselFinding(
                word=m.group(1),
                span=Span(start=m.start(), end=m.end()),
                sentence_index=index_for_offset(sentences, m.start()),
            )
        )
    return findings
