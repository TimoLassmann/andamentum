"""Whetstone v2 — pydantic-graph driven document review.

Single entry point: ``review_document(source, *, model)``.

Without a ``model`` only the deterministic substrate runs (chunking +
structural extractors). With a model, the full critical-review pipeline
runs: lens reading → bounded reflection–investigation loop → optional
editor → challenge → author questions → synthesis.
"""

from .api import review_document
from .renderers import render_docx, render_html, render_markdown
from .schemas import (
    AuthorQuestion,
    Edit,
    ExpertProfile,
    ExpertReview,
    Finding,
    PanelSynthesis,
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
    "ExpertProfile",
    "ExpertReview",
    "Finding",
    "PanelSynthesis",
    "Quote",
    "ReviewMetrics",
    "ReviewResult",
    "SectionCard",
]
