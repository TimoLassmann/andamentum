"""andamentum.proofread — deterministic, dependency-light proofreading.

Main function: ``analyze(text) -> ProofreadResult``. Combines six
readability scores (SMOG, Flesch–Kincaid grade, Flesch reading ease,
Gunning Fog, Coleman–Liau, ARI) with classic deterministic style checks
(weasel words, passive voice, duplicate words, weak openers, adverb
density).
``analyze_slides(slides)`` runs per-slide checks over a deck.

No LLM, no I/O. Pure functions — same input always produces the same
result. Designed to complement, not replace, the LLM-driven reviews in
``andamentum.whetstone``.
"""

from .api import analyze
from .errors import DeckReadError, ProofreadError
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
    # main functions
    "analyze",
    "analyze_slides",
    # result types
    "ProofreadResult",
    "SlidesResult",
    "SlideFinding",
    "Slide",
    "ReadabilityScores",
    "AdverbStats",
    "DuplicateWordFinding",
    "PassiveFinding",
    "Span",
    "WeakOpenerFinding",
    "WeaselFinding",
    # errors
    "ProofreadError",
    "DeckReadError",
]
