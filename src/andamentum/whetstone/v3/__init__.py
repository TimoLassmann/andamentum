"""Whetstone v3 — whole-document, digest-focused, SPECS-criterion review.

A clean rebuild of the review pipeline, built alongside v2 and swapped in once
validated. See docs/.internal/plans/2026-05-22-whetstone-v3-whole-doc-specs.md.

The core shift: reasoning happens once, over the whole paper, via a compact
verified *document model* (claims as located verbatim spans), instead of
section-by-section lens reads. Only extraction is per-section; everything else
reasons over the model. Every LLM-emitted span is string-match-verified against
the source (the `locate` primitive), so hallucinations cannot survive.
"""

from .criteria import Criterion, criterion_set_for
from .extract_criteria import extract_criteria_from_guidelines
from .graph import review_document_v3, run_review_v3

__all__ = [
    "Criterion",
    "criterion_set_for",
    "extract_criteria_from_guidelines",
    "review_document_v3",
    "run_review_v3",
]
