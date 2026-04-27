"""Whetstone v2 — pydantic-graph driven document review.

Single entry point: ``review_document(source, *, model)``.

Phase 1 (this version) ships only the deterministic substrate:
StructuralScan extracts citations / terms / numerics / cross-references and
emits findings without any LLM call. Later phases add Skim, Investigate,
Challenge, Synthesise.
"""

from .api import review_document
from .renderers import render_docx, render_html, render_markdown
from .schemas import (
    AuthorQuestion,
    Edit,
    Finding,
    Hypothesis,
    Quote,
    ReviewMetrics,
    ReviewResult,
    SectionCard,
)

__all__ = [
    "review_document",
    "render_docx",
    "render_html",
    "render_markdown",
    "AuthorQuestion",
    "Edit",
    "Finding",
    "Hypothesis",
    "Quote",
    "ReviewMetrics",
    "ReviewResult",
    "SectionCard",
]
