"""Whetstone — pydantic-graph driven review of YOUR OWN DRAFTS.

Three modes, each its own graph, all returning the same ``ReviewResult``
for the same three renderers:

- **review** — the default. ``await review_document(source, *, model)``
  harvests a path or URL first; ``await run_review(markdown, *, model)``
  takes text you already have.
- **panel** — ``await run_panel(markdown, *, model)``.
- **revise** — ``await revise_document(document, spec=RevisionSpec(...),
  model=...)`` reviews, edits and re-reviews until a stated condition holds (the
  ``andamentum-whetstone revise`` command).
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
``whetstone.pipeline.panel`` and is invoked via the ``andamentum-whetstone
panel`` command or ``await run_panel(markdown, *, model)``.

Checklist mode lives in ``whetstone.pipeline.checklist`` and is invoked via
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
``andamentum.proofread`` (or the ``andamentum-proofread`` command).
The two pipelines are intentionally disjoint; whetstone reviews IDEAS,
proofread reviews STYLE.
"""

from pathlib import Path

from ._confidentiality import ConfidentialityMarkerError, check_confidentiality
from ._watermark import (
    provenance_line,
    stamp_docx_core_properties,
    write_provenance_customxml,
)
from .docx.exceptions import DocumentReviewError
from .errors import WhetstoneError
from .pipeline import (
    Criterion,
    Probe,
    checklist_document,
    criterion_set_for,  # noqa: F401
    extract_criteria_from_guidelines,  # noqa: F401
    probe_bank_for,  # noqa: F401
    review_document,
    run_checklist,
    run_review,
)
from .pipeline.panel import run_panel
from .pipeline.reviewers import PROFILES, ReviewerProfile
from .renderers import render_docx, render_html, render_markdown
from .reviewed_document import ReviewedDocument, open_source
from .revise import FigureRef, RevisionResult, RevisionSpec, revise_document
from .schemas import (  # noqa: F401
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


def stamp_provenance(docx_path: str | Path, *, model: str | None) -> None:
    """Mark a ``.docx`` as AI-assisted: core properties plus the custom XML part.

    The same two steps the review renderers apply to a reviewed document, for
    callers that generate a ``.docx`` themselves.
    """
    stamp_docx_core_properties(docx_path, model=model)
    write_provenance_customxml(docx_path, model=model)


def reviewer_profile(name: str) -> ReviewerProfile:
    """The built-in reviewer profile called *name*; ``ValueError`` if there is none."""
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown reviewer profile {name!r}; valid: {', '.join(sorted(PROFILES))}"
        ) from None


# The docx primitives imported above (Atom, TextMap, apply_edits, ...) stay
# importable from here because ``compose.harness`` reads them from this
# namespace; they are internals, so they are not in ``__all__``.
__all__ = [
    # main functions
    "review_document",
    "run_review",
    "run_panel",
    "checklist_document",
    "run_checklist",
    "revise_document",
    "open_source",
    # renderers
    "render_docx",
    "render_html",
    "render_markdown",
    # inputs
    "Criterion",
    "Probe",
    "ReviewerProfile",
    "reviewer_profile",
    "ReviewedDocument",
    "RevisionSpec",
    "FigureRef",
    # results
    "ReviewResult",
    "Finding",
    "Edit",
    "Quote",
    "AuthorQuestion",
    "RevisionResult",
    # provenance and the confidentiality check
    "check_confidentiality",
    "stamp_provenance",
    "provenance_line",
    # errors
    "WhetstoneError",
    "ConfidentialityMarkerError",
    "DocumentReviewError",
]
