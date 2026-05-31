"""Extra deterministic prose checks: duplicate words, adverb density,
weak sentence openers."""

from __future__ import annotations

import re

from .models import (
    AdverbStats,
    DuplicateWordFinding,
    Span,
    WeakOpenerFinding,
)
from .sentences import Sentence, index_for_offset

# "had had" (past perfect) and "that that" (occasional grammatical
# construction) are the only idiomatic repeats common enough to allowlist.
_DUPLICATE_ALLOWLIST = {"had", "that"}

_DUPLICATE_PATTERN = re.compile(r"\b(\w+)\b\s+\b\1\b", re.IGNORECASE)


def scan_duplicates(text: str, sentences: list[Sentence]) -> list[DuplicateWordFinding]:
    """Find adjacent identical word pairs ("the the"). Allowlisted idioms
    like "had had" / "that that" are skipped."""
    findings: list[DuplicateWordFinding] = []
    for m in _DUPLICATE_PATTERN.finditer(text):
        word = m.group(1).lower()
        if word in _DUPLICATE_ALLOWLIST:
            continue
        findings.append(
            DuplicateWordFinding(
                word=m.group(1),
                span=Span(start=m.start(), end=m.end()),
                sentence_index=index_for_offset(sentences, m.start()),
            )
        )
    return findings


# Words ending in -ly that aren't adverbs (subset). Adding more here trades
# recall for precision — the goal is to avoid embarrassing false positives,
# not to be exhaustive.
_NON_ADVERB_LY = {
    "only",
    "family",
    "ugly",
    "supply",
    "reply",
    "apply",
    "ally",
    "rally",
    "rely",
    "july",
    "italy",
    "holy",
    "silly",
    "homely",
    "lonely",
    "lovely",
    "lowly",
    "manly",
    "wily",
    "worldly",
    "ghastly",
    "early",
    "fly",
    "imply",
    "comply",
    "multiply",
    "butterfly",
    "monopoly",
    "ply",
    "assembly",
    "anomaly",
}

_WORD_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z'-]*\b")


def adverb_stats(text: str) -> AdverbStats:
    """Count adverbs by the -ly heuristic, minus a curated exclusion list,
    and return both the raw count and the density (adverbs / total words)."""
    words = _WORD_PATTERN.findall(text)
    if not words:
        return AdverbStats(adverb_count=0, adverb_density=0.0)

    adverbs = 0
    for w in words:
        lw = w.lower()
        if len(lw) > 3 and lw.endswith("ly") and lw not in _NON_ADVERB_LY:
            adverbs += 1
    return AdverbStats(
        adverb_count=adverbs,
        adverb_density=adverbs / len(words),
    )


# Vacuous sentence openers — "It is X that Y" ≡ "Y is X"; "There is X" ≡ "X exists".
_WEAK_OPENER = re.compile(
    r"\b(there\s+(?:is|are|was|were)|it\s+(?:is|was|seems|appears))\b",
    re.IGNORECASE,
)


def scan_weak_openers(sentences: list[Sentence]) -> list[WeakOpenerFinding]:
    """Flag sentences that open with "There is/are/was/were" or
    "It is/was/seems/appears"."""
    findings: list[WeakOpenerFinding] = []
    for s in sentences:
        m = _WEAK_OPENER.match(s.text)
        if m is None:
            continue
        findings.append(
            WeakOpenerFinding(
                matched_text=m.group(0),
                span=Span(start=s.start + m.start(), end=s.start + m.end()),
                sentence_index=index_for_offset(sentences, s.start),
            )
        )
    return findings
