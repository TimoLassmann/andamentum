"""Deterministic readability metrics.

Thin wrapper over ``textstat``. The grade-level scores all use the U.S.
school-grade convention. Flesch reading-ease is the only score where
higher = easier; the others are grade levels where higher = harder.

SMOG (McLaughlin, 1969) — Simple Measure of Gobbledygook — is the headline
score asked for. It is well-correlated with comprehension and stable
across short and long passages. textstat's implementation matches the
canonical formula:

    SMOG = 1.0430 * sqrt(polysyllables * (30 / sentences)) + 3.1291

where a polysyllable has ≥3 syllables.
"""

from __future__ import annotations

from textstat import textstat as _ts

from .models import ReadabilityScores


def compute(text: str) -> ReadabilityScores:
    """Compute the readability score block. Safe on empty / whitespace-only
    input — returns zeroed counts and 0.0 grade levels rather than raising."""
    stripped = text.strip()
    if not stripped:
        return ReadabilityScores(
            smog_index=0.0,
            flesch_kincaid_grade=0.0,
            flesch_reading_ease=0.0,
            gunning_fog=0.0,
            coleman_liau_index=0.0,
            automated_readability_index=0.0,
            word_count=0,
            sentence_count=0,
            avg_sentence_length=0.0,
            avg_syllables_per_word=0.0,
        )

    word_count = _ts.lexicon_count(text)
    sentence_count = _ts.sentence_count(text)
    avg_sentence_length = (
        float(word_count) / sentence_count if sentence_count else 0.0
    )

    return ReadabilityScores(
        smog_index=float(_ts.smog_index(text)),
        flesch_kincaid_grade=float(_ts.flesch_kincaid_grade(text)),
        flesch_reading_ease=float(_ts.flesch_reading_ease(text)),
        gunning_fog=float(_ts.gunning_fog(text)),
        coleman_liau_index=float(_ts.coleman_liau_index(text)),
        automated_readability_index=float(_ts.automated_readability_index(text)),
        word_count=int(word_count),
        sentence_count=int(sentence_count),
        avg_sentence_length=avg_sentence_length,
        avg_syllables_per_word=float(_ts.avg_syllables_per_word(text)),
    )
