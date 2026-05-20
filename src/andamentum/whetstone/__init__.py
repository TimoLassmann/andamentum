"""Whetstone — pydantic-graph driven review of YOUR OWN DRAFTS.

Single entry point: ``review_document(source, *, model)``.

**Not a peer-review tool.** Whetstone is for sharpening drafts you wrote
yourself. Do not call it on manuscripts, grants, or other documents
shared with you in confidence (as a journal reviewer, grant panel member,
or examiner). Most publishers and funders explicitly prohibit this use.
See ``RESPONSIBLE_USE.md`` at the repo root and
``src/andamentum/whetstone/RESPONSIBLE_USE.md`` for the policy landscape
and the in-code protections (confidentiality-marker tripwire, tiered
watermarking, locked AI-author attribution, panel-mode authorship gate).

Without a ``model`` only the deterministic substrate runs (chunking +
structural extractors + surface-style proofread). With a model, the full
critical-review pipeline runs: lens reading → bounded reflection–
investigation loop → optional editor → challenge → author questions →
synthesis.

The proofread pass (weasel words, passive voice, duplicate words, weak
sentence openers) runs by default in every mode — pure deterministic
checks, no LLM, no network, adds milliseconds. Findings flow through the
same renderers as the LLM-driven lenses and appear as Word comments in
the .docx output. Disable with ``proofread=False`` (or ``--no-proofread``
from the CLI) when you want to suppress that layer.
"""

from .api import review_document
from .renderers import render_docx, render_html, render_markdown
from .schemas import (
    AuthorQuestion,
    CheckableItem,
    CustomEvaluation,
    Edit,
    ExpertProfile,
    ExpertReview,
    Finding,
    GuidelineEvaluation,
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
    "CheckableItem",
    "CustomEvaluation",
    "Edit",
    "ExpertProfile",
    "ExpertReview",
    "Finding",
    "GuidelineEvaluation",
    "PanelSynthesis",
    "Quote",
    "ReviewMetrics",
    "ReviewResult",
    "SectionCard",
]
