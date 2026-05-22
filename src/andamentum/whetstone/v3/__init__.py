"""Whetstone v3 — whole-document, digest-focused, SPECS-criterion review.

A clean rebuild of the review pipeline, built alongside v2 and swapped in once
validated. See docs/plans/2026-05-22-whetstone-v3-whole-doc-specs.md.

The core shift: reasoning happens once, over the whole paper, via a compact
verified *document model* (claims as located verbatim spans), instead of
section-by-section lens reads. Only extraction is per-section; everything else
reasons over the model. Every LLM-emitted span is string-match-verified against
the source (the `locate` primitive), so hallucinations cannot survive.
"""

from .graph import review_document_v3, run_review_v3

__all__ = ["run_review_v3", "review_document_v3"]
