"""Public schemas for whetstone v2.

These are the types the API returns and that downstream consumers (other
agents, CLIs, the user's tooling) read. Pydantic models because:
  • LLM-filled types (Finding-from-investigator) need pydantic
  • External consumers benefit from .model_dump() / JSON serialisation
  • Flat field shapes are reliably filled by small local models

Schemas are intentionally tight: 3-value enums where possible, no nested
optional structures, no fields the agent has to guess about.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# Severity → default priority bucket. Lenses don't need to set priority
# explicitly; it is derived from severity at construction time. Reflection
# (or any later step) may override by setting priority directly.
_DEFAULT_PRIORITY_FROM_SEVERITY = {
    "major": "must_fix",
    "moderate": "should_fix",
    "minor": "consider",
}


# ── Atomic types ────────────────────────────────────────────────────────


class Quote(BaseModel):
    """A verbatim span of source text. Locatable via section_id + char range."""

    section_id: str
    char_start: int  # offset within the section (NOT the whole document)
    char_end: int
    text: str  # the verbatim text at [char_start, char_end)


class Finding(BaseModel):
    """One issue identified with the document.

    Whether emitted by the deterministic substrate or by an LLM
    investigator, every Finding has the same shape so downstream consumers
    treat them uniformly. The ``source`` field tells the consumer how
    confident they should be in the finding's provenance.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str
    severity: Literal["minor", "moderate", "major"]
    confidence: Literal["low", "medium", "high"]
    rationale: str  # 2-3 sentences explaining the issue
    quotes: list[Quote] = Field(default_factory=list)
    sections_involved: list[str] = Field(default_factory=list)  # section_id list
    source: Literal["deterministic", "investigate", "challenged"] = "deterministic"
    perspective: Optional[str] = None  # for panel mode; None for single-perspective
    category: str = ""  # short clustering tag picked by the lens
    priority: Literal["must_fix", "should_fix", "consider"] = "consider"
    # priority is derived from severity by default (see _derive_priority);
    # downstream nodes (reflection, challenge) may override it explicitly.

    @model_validator(mode="before")
    @classmethod
    def _derive_priority(cls, data: Any) -> Any:
        """Fill in priority from severity when caller didn't provide one.

        This runs before field validation, so we operate on the raw input
        dict. If the caller passed priority explicitly, we leave it
        alone; otherwise we map severity → bucket.
        """
        if isinstance(data, dict) and "priority" not in data and "severity" in data:
            data["priority"] = _DEFAULT_PRIORITY_FROM_SEVERITY.get(
                data["severity"], "consider"
            )
        return data


class Edit(BaseModel):
    """A concrete proposed rewrite at a specific span in the source.

    Where ``Finding`` says "this is wrong" without a fix, ``Edit`` says
    "change THIS to THAT". Renderers turn each Edit into:
      • Word: a tracked-change (deletion of original_text + insertion of new_text)
      • Markdown: a unified-diff-style block
      • HTML: a side-by-side card with old → new
    """

    title: str
    severity: Literal["minor", "moderate", "major"] = "minor"
    confidence: Literal["low", "medium", "high"] = "medium"
    rationale: str  # 1-2 sentences explaining why
    section_id: str
    char_start: int  # offset within the section (NOT the whole document)
    char_end: int
    original_text: str  # what's there now (verbatim)
    new_text: str  # what to replace it with
    perspective: Optional[str] = None


class AuthorQuestion(BaseModel):
    """A question only the document's author can answer.

    Generated when an LLM investigator concludes a hypothesis can't be
    resolved from the document text alone. Phase 6 feature; included in
    Phase 1 schema so the result type is stable.
    """

    question: str
    sections_involved: list[str] = Field(default_factory=list)
    why: str  # one sentence: why we couldn't answer it ourselves


# ── Document map ────────────────────────────────────────────────────────


class SectionCard(BaseModel):
    """A compact summary of one section, used as cross-section context."""

    section_id: str
    title: str
    one_line_gist: str = ""  # populated deterministically by document_map


# ── Result types ────────────────────────────────────────────────────────


class ReviewMetrics(BaseModel):
    """Telemetry for the run."""

    llm_calls: int = 0
    wall_seconds: float = 0.0
    deterministic_findings_count: int = 0
    investigated_findings_count: int = 0
    challenged_findings_count: int = 0
    edits_count: int = 0
    sections_processed: int = 0
    reflection_rounds_used: int = 0  # how many of reflection_round_cap got consumed


# ── Panel-mode types ────────────────────────────────────────────────────


class ExpertProfile(BaseModel):
    """A fictional but plausible biosketch of a discipline-matched reviewer.

    Generated by the ``expert_generator`` agent in panel mode. Each field
    is plain prose — nothing nested — so small local models can fill the
    schema reliably.
    """

    name: str = Field(description="Full name of the fictional expert.")
    position: str = Field(
        description="Current academic position and institution."
    )
    education: str = Field(
        description="Educational background (degrees, institutions, years)."
    )
    contributions: str = Field(
        description="3-5 key contributions to the field, as flowing prose."
    )
    research: str = Field(description="Current research focus and interests.")
    discipline: str = Field(description="Primary academic discipline.")


class ExpertReview(BaseModel):
    """One discipline-matched expert's structured review.

    Ports v1's ``ExpertReviewOutput``. Scores stay as integers 1–10
    (rather than three-value enums) because that's what the downstream
    panel synthesiser averages and because v1's calibration was tuned at
    that resolution.
    """

    expert_name: str = Field(
        description="Name of the expert providing the review."
    )
    discipline: str = Field(description="Expert's academic discipline.")
    overall_score: int = Field(
        ge=1, le=10, description="Overall quality score 1-10."
    )
    overall_assessment: str = Field(
        description="Brief overall assessment (2-3 sentences)."
    )
    scientific_rigor_score: int = Field(
        ge=1, le=10, description="Scientific rigor score 1-10."
    )
    scientific_rigor_justification: str = Field(
        description="Justification for scientific rigor score (2-3 sentences)."
    )
    methodology_score: int = Field(
        ge=1, le=10, description="Methodology quality score 1-10."
    )
    methodology_justification: str = Field(
        description="Justification for methodology score (2-3 sentences)."
    )
    novelty_score: int = Field(
        ge=1, le=10, description="Novelty and innovation score 1-10."
    )
    novelty_justification: str = Field(
        description="Justification for novelty score (2-3 sentences)."
    )
    clarity_score: int = Field(
        ge=1, le=10, description="Clarity of presentation score 1-10."
    )
    clarity_justification: str = Field(
        description="Justification for clarity score (2-3 sentences)."
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="3-5 key strengths.",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="3-5 key weaknesses.",
    )
    recommendation: Literal[
        "Accept", "Minor Revisions", "Major Revisions", "Reject"
    ] = Field(
        description=(
            "Final recommendation — Accept, Minor Revisions, "
            "Major Revisions, or Reject."
        )
    )
    recommendation_justification: str = Field(
        description="Justification for recommendation (3-4 sentences)."
    )


class PanelSynthesis(BaseModel):
    """Aggregated meta-review across the expert panel.

    Ports v1's ``PanelSynthesisOutput`` with v2 hygiene: nullable
    ``confidence_level`` is a 3-value Literal, ``overall_recommendation``
    is the same 4-value Literal as ``ExpertReview``.
    """

    average_overall_score: float = Field(
        description="Average overall score across all experts."
    )
    score_range: str = Field(
        description="Range of overall scores (e.g. '7-9')."
    )
    number_of_experts: int = Field(
        description="Total number of expert reviewers."
    )
    consensus_strengths: list[str] = Field(
        default_factory=list,
        description="Strengths identified by multiple experts (3-5 items).",
    )
    consensus_weaknesses: list[str] = Field(
        default_factory=list,
        description="Weaknesses identified by multiple experts (3-5 items).",
    )
    divergent_opinions: list[str] = Field(
        default_factory=list,
        description="Areas where experts disagreed (0-3 items).",
    )
    scientific_rigor_summary: str = Field(
        description="Synthesis of scientific rigor assessments (2-3 sentences)."
    )
    methodology_summary: str = Field(
        description="Synthesis of methodology assessments (2-3 sentences)."
    )
    novelty_summary: str = Field(
        description="Synthesis of novelty assessments (2-3 sentences)."
    )
    clarity_summary: str = Field(
        description="Synthesis of clarity assessments (2-3 sentences)."
    )
    overall_recommendation: Literal[
        "Accept", "Minor Revisions", "Major Revisions", "Reject"
    ] = Field(
        description=(
            "Overall recommendation — Accept, Minor Revisions, "
            "Major Revisions, or Reject."
        )
    )
    recommendation_justification: str = Field(
        description="Justification for overall recommendation (4-5 sentences)."
    )
    confidence_level: Literal["high", "medium", "low"] = Field(
        description=(
            "Confidence in recommendation — high (aligned, similar "
            "scores), medium (general agreement), low (significant "
            "disagreement)."
        )
    )
    key_decision_factors: list[str] = Field(
        default_factory=list,
        description="Key factors that influenced the recommendation (3-5 items).",
    )
    review_summary: str = Field(
        description=(
            "Comprehensive executive summary of the multi-expert review "
            "(5-7 paragraphs)."
        )
    )


# ── Guidelines / custom-criteria types ─────────────────────────────────


class CheckableItem(BaseModel):
    """One checkable rule extracted from baseline / journal-guidelines / custom criteria.

    The ``source`` field tracks provenance:
      • ``baseline`` — built-in submission-readiness checks
      • ``guidelines`` — extracted from a journal's free-text author guidelines
      • ``custom`` — supplied by the caller as ad-hoc criteria text
    """

    name: str = Field(description="Short human-readable rule name.")
    source: Literal["baseline", "guidelines", "custom"] = Field(
        description="Provenance of the item."
    )


class GuidelineEvaluation(BaseModel):
    """Verdict for one journal-guidelines check against the manuscript."""

    item_name: str = Field(description="Verbatim name of the check that was evaluated.")
    status: Literal["pass", "fail", "unclear"] = Field(
        description=(
            "pass = clearly met; fail = clearly not met; unclear = ambiguous "
            "or the rule does not apply to this document."
        )
    )
    notes: str = Field(
        description=(
            "1-2 sentences. For pass: cite evidence (quote a phrase or name "
            "the section). For fail: say what is missing and what should be "
            "added. For unclear: say why."
        )
    )
    category: str = Field(
        default="",
        description="Optional clustering tag (e.g. 'abstract', 'figures').",
    )


class CustomEvaluation(BaseModel):
    """Verdict for one user-supplied criterion against the manuscript.

    Flat counterpart to the runtime-built dynamic schema. Renderers and
    downstream consumers iterate this list — they never see the dynamic
    model the LLM filled.
    """

    criterion: str = Field(description="Original criterion text supplied by the caller.")
    status: Literal["pass", "fail", "unclear"] = Field(
        description="pass = met; fail = not met; unclear = ambiguous or NA."
    )
    notes: str = Field(
        description="1-2 sentences explaining the verdict, grounded in the document."
    )


class ReviewResult(BaseModel):
    """What ``review_document`` returns.

    Designed to be machine-friendly — every consumer (the user, another
    agent, a CLI rendering, a CI gate) reads the same flat structure.

    In ``mode="panel"`` runs, the panel-only fields (``expert_profiles``,
    ``expert_reviews``, ``panel_synthesis``) are populated; the standard
    review-mode fields may also carry deterministic findings + the
    document map from the shared chunk_and_scan substrate.

    In ``mode="guidelines"`` runs, ``checkable_items`` lists what was
    extracted from the supplied guidelines and ``guideline_evaluations``
    holds one verdict per item. In ``mode="custom"`` runs,
    ``custom_evaluations`` is populated.
    """

    summary: str = ""  # filled in Phase 4 (Synthesise)
    findings: list[Finding] = Field(default_factory=list)  # post-LLM, Phase 2+
    deterministic_findings: list[Finding] = Field(default_factory=list)
    edits: list[Edit] = Field(default_factory=list)  # concrete rewrites (Edit phase)
    author_questions: list[AuthorQuestion] = Field(default_factory=list)
    document_map: list[SectionCard] = Field(default_factory=list)
    metrics: ReviewMetrics = Field(default_factory=ReviewMetrics)
    # Panel mode (mode="panel") populates these; otherwise empty.
    expert_profiles: list[ExpertProfile] = Field(default_factory=list)
    expert_reviews: list[ExpertReview] = Field(default_factory=list)
    panel_synthesis: PanelSynthesis | None = None
    # Guidelines mode (mode="guidelines") populates these; otherwise empty.
    checkable_items: list[CheckableItem] = Field(default_factory=list)
    guideline_evaluations: list[GuidelineEvaluation] = Field(default_factory=list)
    # Custom mode (mode="custom") populates this; otherwise empty.
    custom_evaluations: list[CustomEvaluation] = Field(default_factory=list)
