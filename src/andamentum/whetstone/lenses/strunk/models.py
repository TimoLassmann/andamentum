"""Internal pydantic schemas for the Strunk lens.

Two layers:

1. Per-rule report schemas. One report per rule. Each report carries a
   list of ``Violation`` rows — zero-to-many per section. Flat fields,
   ``Literal`` enums, no nested optionals so small local models fill
   them reliably. An empty list IS the "no violations found" answer.
2. Internal pipeline types. ``StrunkFinding`` is the graph's shared
   currency; the ``Aggregate`` node converts it into the public
   ``whetstone.schemas.Finding`` shape on the way out.

A ``StrunkDemand`` is emitted only when a rule's agent call raises or
returns a non-schema-valid result. Phase A leaves the resolve loop
stubbed; Phase 4 will consume demands by retrying on a stronger model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Internal pipeline types ─────────────────────────────────────────────


class StrunkFinding(BaseModel):
    """One rule violation emitted by any node in the sub-graph.

    The ``Aggregate`` node deduplicates these by ``(char_start,
    char_end)``, sorts by char offset, and converts them into the
    public ``whetstone.schemas.Finding`` shape. Char offsets are
    relative to the *section's* text, matching the section-local
    convention used by ``Quote.char_start``/``char_end`` upstream.
    """

    rule_number: int                       # Strunk rule number (1-18; Ch V/VI use 100+)
    rule_name: str                         # short slug, e.g. "omit-needless-words"
    char_start: int                        # within section.text
    char_end: int
    title: str
    rationale: str
    severity: Literal["minor", "moderate", "major"] = "minor"
    confidence: Literal["low", "medium", "high"] = "medium"
    category: str = ""
    span_text: str = ""                    # the offending span, verbatim from source
    suggested_replacement: str = ""        # empty if no concrete fix proposed


class StrunkDemand(BaseModel):
    """Escalation request from an agent node when it cannot commit.

    Phase A: emitted only on schema-validation failure or executor
    exception (per-section calls don't naturally produce per-violation
    abstentions — an empty violations list IS the "nothing found"
    answer). Phase 4 will replay these on a stronger model.
    """

    rule: str                              # "r11", "r13", ...
    reason: Literal[
        "schema_validation_failed",
        "executor_exception",
    ]
    suggested_escalation: Literal["larger_model", "expand_context"] = (
        "larger_model"
    )


# ── Per-rule report schemas (LLM-filled, one report per section) ────────


class ActiveVoiceViolation(BaseModel):
    """One R11 violation found in the section."""

    span: str = Field(
        description=(
            "Exact verbatim substring from the section text that contains "
            "the passive-voice construction. Quote it as-is, not "
            "paraphrased."
        ),
    )
    suggested_active_rewrite: str = Field(
        default="",
        description=(
            "Full active-voice rewrite of the sentence the span lives in. "
            "Empty if no clean rewrite is possible (e.g. the agent of "
            "the passive is unrecoverable from context)."
        ),
    )
    confidence: Literal["low", "medium", "high"] = "medium"


class ActiveVoiceReport(BaseModel):
    """R11 result for one whole section. Empty ``violations`` list
    means the section has no passive-voice issues worth flagging."""

    violations: list[ActiveVoiceViolation] = Field(default_factory=list)


class OmitNeedlessWordsViolation(BaseModel):
    """One R13 violation found in the section. The ``category`` enum is
    deliberately closed so small models pick from a fixed shortlist
    rather than inventing a free-form reason."""

    span: str = Field(
        description=(
            "Exact verbatim substring from the section text containing "
            "the needless words. Quote it as-is."
        ),
    )
    category: Literal[
        "throat-clearing",         # "the fact that ...", "it is the case that ..."
        "redundancy",              # "advance planning", "consensus of opinion"
        "weak-qualifier",          # "rather", "very", "little", "pretty"
        "filler-prepositional",    # "of an X nature" in place of an adjective
        "other",
    ] = "other"
    suggested_deletion: str = Field(
        default="",
        description=(
            "Sentence rewritten with the needless words removed — NOT a "
            "diff. Empty if no clean rewrite is obvious."
        ),
    )
    confidence: Literal["low", "medium", "high"] = "medium"


class OmitNeedlessWordsReport(BaseModel):
    """R13 result for one whole section. Empty ``violations`` list
    means the section reads cleanly under Rule 13."""

    violations: list[OmitNeedlessWordsViolation] = Field(default_factory=list)
