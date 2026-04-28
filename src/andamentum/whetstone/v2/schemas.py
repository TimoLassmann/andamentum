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


class ReviewResult(BaseModel):
    """What ``review_document`` returns.

    Designed to be machine-friendly — every consumer (the user, another
    agent, a CLI rendering, a CI gate) reads the same flat structure.
    """

    summary: str = ""  # filled in Phase 4 (Synthesise)
    findings: list[Finding] = Field(default_factory=list)  # post-LLM, Phase 2+
    deterministic_findings: list[Finding] = Field(default_factory=list)
    edits: list[Edit] = Field(default_factory=list)  # concrete rewrites (Edit phase)
    author_questions: list[AuthorQuestion] = Field(default_factory=list)
    document_map: list[SectionCard] = Field(default_factory=list)
    metrics: ReviewMetrics = Field(default_factory=ReviewMetrics)
