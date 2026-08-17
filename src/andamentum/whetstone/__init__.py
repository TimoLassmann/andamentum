"""Whetstone — pydantic-graph driven review of YOUR OWN DRAFTS.

Three modes, each its own graph, all returning the same ``ReviewResult``
for the same three renderers:

- **review** — the default. ``await review_document(source, *, model)``
  harvests a path or URL first; ``await run_review(markdown, *, model)``
  takes text you already have.
- **panel** — ``await run_panel(markdown, *, model)``.
- **checklist** — ``await checklist_document(source, *, model)`` harvests
  first; ``await run_checklist(markdown, *, model)`` takes text.

**Not a peer-review tool.** Whetstone is for sharpening drafts you wrote
yourself. Do not call it on manuscripts, grants, or other documents
shared with you in confidence (as a journal reviewer, grant panel
member, or examiner). Most publishers and funders explicitly prohibit
this use. See ``RESPONSIBLE_USE.md`` at the repo root and
``src/andamentum/whetstone/RESPONSIBLE_USE.md`` for the policy
landscape and the in-code protections (confidentiality-marker tripwire,
tiered watermarking, locked AI-author attribution, panel-mode
authorship gate).

Review pipeline: deterministic substrate (sectionize + claim extraction +
document model) → criterion cascade (one of six pluggable sets:
academic / external_communication / essay / tutorial / creative /
general) → gap-loop re-grounding → consolidate → gate → synthesise →
optional editor → finalize. Optional novelty check via deep_research
(opt-in with ``check_novelty=True``).

Panel mode (multi-expert simulated peer review) lives in
``whetstone.v3.panel`` and is invoked via the ``andamentum-whetstone
panel`` subcommand or ``await run_panel(markdown, *, model)``.

Checklist mode lives in ``whetstone.v3.checklist`` and is invoked via
the ``andamentum-whetstone checklist`` subcommand, ``await
checklist_document(source, *, model)`` (path or URL, harvested first)
or ``await run_checklist(markdown, *, model)``. It runs a bank of
``Probe``s — each one defect class, under a quota — instead of the
criterion cascade, and summarises with a deterministic coverage table
rather than synthesised prose. Use it when the question is "run the
standard check" (pre-submission, draft-vs-draft, CI); use review mode
when the question is "help me improve this draft". Same
responsible-use scope: your own drafts only, and the confidentiality
tripwire applies to checklist mode exactly as it does to review.

Deterministic style / readability checks are a separate concern — use
``andamentum.proofread`` (or ``andamentum-whetstone proofread <source>``).
The two pipelines are intentionally disjoint; whetstone reviews IDEAS,
proofread reviews STYLE.
"""

from .renderers import render_docx, render_html, render_markdown
from .reviewed_document import ReviewedDocument, open_source
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
    Probe,
    checklist_document_v3 as checklist_document,
    criterion_set_for,
    extract_criteria_from_guidelines,
    probe_bank_for,
    review_document_v3 as review_document,
    run_checklist_v3 as run_checklist,
    run_review_v3 as run_review,
)
from .v3.panel import run_panel_v3 as run_panel

__all__ = [
    "review_document",
    "run_review",
    "run_panel",
    "checklist_document",
    "run_checklist",
    "Criterion",
    "Probe",
    "criterion_set_for",
    "probe_bank_for",
    "extract_criteria_from_guidelines",
    "ReviewedDocument",
    "open_source",
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
