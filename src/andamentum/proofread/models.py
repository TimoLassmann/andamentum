"""Pydantic models returned by ``analyze()``.

Findings carry ``span`` (start/end char offsets into the input text, both
inclusive-exclusive) and ``sentence_index`` (0-based, matching the
segmentation in ``sentences.py``) so callers can map back into the source
or render highlights.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Span(BaseModel):
    """Half-open char range into the input string: ``text[start:end]``."""

    start: int
    end: int


class ReadabilityScores(BaseModel):
    """Standard readability metrics. All grade-level scores use the U.S.
    school-grade convention (12 ≈ end of high school)."""

    smog_index: float
    flesch_kincaid_grade: float
    flesch_reading_ease: float
    gunning_fog: float
    coleman_liau_index: float
    automated_readability_index: float
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    avg_syllables_per_word: float


class WeaselFinding(BaseModel):
    """A weasel word that weakens a claim ("many", "various", "fairly", ...)."""

    word: str
    span: Span
    sentence_index: int


class PassiveFinding(BaseModel):
    """A likely passive-voice construction (be-verb + past participle)."""

    matched_text: str
    span: Span
    sentence_index: int


class DuplicateWordFinding(BaseModel):
    """Adjacent repeated word ("the the"). Common idiomatic repeats are
    excluded (e.g. "had had", "that that")."""

    word: str
    span: Span
    sentence_index: int


class WeakOpenerFinding(BaseModel):
    """A sentence that opens with a vacuous construction
    ("There is/are/was/were", "It is/was")."""

    matched_text: str
    span: Span
    sentence_index: int


class AdverbStats(BaseModel):
    adverb_count: int
    adverb_density: float = Field(
        description="Adverbs (-ly heuristic minus exclusion list) per total word."
    )


class ProofreadResult(BaseModel):
    """Result of ``analyze(text)``. Deterministic; same input → same output."""

    readability: ReadabilityScores
    weasel_words: list[WeaselFinding]
    passive_voice: list[PassiveFinding]
    duplicate_words: list[DuplicateWordFinding]
    weak_openers: list[WeakOpenerFinding]
    adverbs: AdverbStats
    summary: str
