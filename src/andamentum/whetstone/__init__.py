"""Whetstone — pydantic-graph driven review of YOUR OWN DRAFTS.

Single entry point: ``await review_document(source, *, model)``.

**Not a peer-review tool.** Whetstone is for sharpening drafts you wrote
yourself. Do not call it on manuscripts, grants, or other documents
shared with you in confidence (as a journal reviewer, grant panel
member, or examiner). Most publishers and funders explicitly prohibit
this use. See ``RESPONSIBLE_USE.md`` at the repo root and
``src/andamentum/whetstone/RESPONSIBLE_USE.md`` for the policy
landscape and the in-code protections (confidentiality-marker tripwire,
tiered watermarking, locked AI-author attribution, panel-mode
authorship gate).

Pipeline: deterministic substrate (sectionize + claim extraction +
document model) → criterion cascade (one of six pluggable sets:
academic / external_communication / essay / tutorial / creative /
general) → gap-loop re-grounding → consolidate → gate → synthesise →
optional editor → finalize. Optional novelty check via deep_research
(opt-in with ``check_novelty=True``).

Panel mode (multi-expert simulated peer review) lives in
``whetstone.v3.panel`` and is invoked via the ``andamentum-whetstone
panel`` subcommand or ``await run_panel_v3(markdown, *, model)``.

Deterministic style / readability checks are a separate concern — use
``andamentum.proofread`` (or ``andamentum-whetstone proofread <source>``).
The two pipelines are intentionally disjoint; whetstone reviews IDEAS,
proofread reviews STYLE.
"""

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
from .v3 import (
    Criterion,
    criterion_set_for,
    extract_criteria_from_guidelines,
    review_document_v3 as review_document,
    run_review_v3 as run_review,
)
from .v3.panel import run_panel_v3 as run_panel

__all__ = [
    "review_document",
    "run_review",
    "run_panel",
    "Criterion",
    "criterion_set_for",
    "extract_criteria_from_guidelines",
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
