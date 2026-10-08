"""Whetstone v3 — whole-document, digest-focused, SPECS-criterion review.

A clean rebuild of the review pipeline, built alongside v2 and swapped in once
validated.

The core shift: reasoning happens once, over the whole paper, via a compact
verified *document model* (claims as located verbatim spans), instead of
section-by-section lens reads. Only extraction is per-section; everything else
reasons over the model. Every LLM-emitted span is string-match-verified against
the source (the `locate` primitive), so hallucinations cannot survive.

Three graphs live here, one per mode: `graph.py` (the criterion cascade —
`review_document` / `run_review`), `panel/` (N simulated expert reviews +
a synthesis — `run_panel`) and `checklist/` (a fixed bank of quota-driven
`Probe`s + a deterministic coverage table — `checklist_document` /
`run_checklist`). All three open with the same deterministic `sectionize`
pass and end in the same `schemas.ReviewResult`, so the three renderers consume
any of them unchanged; review and checklist share the rest of the substrate too
(`extract.build_claims` → `digest.build_document_model`), while panel reasons
from the section list alone. What differs is everything built on top of that,
which is why each mode is a separate graph rather than a flag on one.
"""

from .checklist import Probe, checklist_document, probe_bank_for, run_checklist
from .criteria import Criterion, criterion_set_for
from .extract_criteria import extract_criteria_from_guidelines
from .graph import review_document, run_review
from .reviewers import (
    DEFAULT_PROFILE_NAME,
    PROFILES,
    ReviewerProfile,
    author_for,
    fits,
    resolve,
)

__all__ = [
    "Criterion",
    "criterion_set_for",
    "extract_criteria_from_guidelines",
    "review_document",
    "run_review",
    "PROFILES",
    "DEFAULT_PROFILE_NAME",
    "ReviewerProfile",
    "author_for",
    "fits",
    "resolve",
    "Probe",
    "probe_bank_for",
    "checklist_document",
    "run_checklist",
]
