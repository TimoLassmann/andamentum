"""andamentum.proofread — deterministic, dependency-light proofreading.

Single public API: ``analyze(text) -> ProofreadResult``. Combines five
readability scores (SMOG, Flesch–Kincaid grade, Flesch reading ease,
Gunning Fog, Coleman–Liau, ARI) with classic deterministic style checks
(weasel words, passive voice, duplicate words, weak openers, adverb
density).
``analyze_slides(slides)`` runs per-slide checks over a deck.

No LLM, no I/O. Pure function — same input always produces the same
result. Designed to complement, not replace, the LLM-driven reviews in
``andamentum.whetstone``.
"""

from .api import analyze
from .slides import Slide, SlideFinding, SlidesResult, analyze_slides
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
    "analyze_slides",
    "Slide",
    "SlideFinding",
    "SlidesResult",
    "AdverbStats",
    "DuplicateWordFinding",
    "PassiveFinding",
    "ProofreadResult",
    "ReadabilityScores",
    "Span",
    "WeakOpenerFinding",
    "WeaselFinding",
]
