"""andamentum.proofread — deterministic, dependency-light proofreading.

Single public API: ``analyze(text) -> ProofreadResult``. Combines five
readability scores (SMOG, Flesch–Kincaid grade, Flesch reading ease,
Gunning Fog, Coleman–Liau, ARI) with classic deterministic style checks
(weasel words, passive voice, duplicate words, weak openers, adverb
density).

No LLM, no I/O. Pure function — same input always produces the same
result. Designed to complement, not replace, the LLM-driven reviews in
``andamentum.whetstone``.
"""

from .api import analyze
from .models import (
    AdverbStats,
    DuplicateWordFinding,
    PassiveFinding,
    ProofreadResult,
    ReadabilityScores,
    Span,
    WeakOpenerFinding,
    WeaselFinding,
)

__all__ = [
    "analyze",
    "AdverbStats",
    "DuplicateWordFinding",
    "PassiveFinding",
    "ProofreadResult",
    "ReadabilityScores",
    "Span",
    "WeakOpenerFinding",
    "WeaselFinding",
]
