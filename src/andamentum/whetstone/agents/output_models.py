"""Output models for document-review agents.

Editing agents reuse DocumentPatch from the package.
Review agents reuse DocumentIssue from the package.
This module defines NEW models needed by synthesis, multi-expert, and custom agents.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from ..models import DocumentPatch
from ..issues import DocumentIssue


# ---------------------------------------------------------------------------
# Editing agent output
# ---------------------------------------------------------------------------


class EditingOutput(BaseModel):
    """Output from editing agents (grammar, academic_writing, polish)."""

    patches: list[DocumentPatch] = Field(
        default_factory=list,
        description="List of DocumentPatch objects for edits and comments",
    )


# ---------------------------------------------------------------------------
# Review agent output
# ---------------------------------------------------------------------------


class ReviewOutput(BaseModel):
    """Output from review agents (clarity, scientific_merit, methodology, results)."""

    issues: list[DocumentIssue] = Field(
        default_factory=list,
        description="List of DocumentIssue objects (10-15 issues total)",
    )


# ---------------------------------------------------------------------------
# Synthesis: document_review_synthesizer
# ---------------------------------------------------------------------------


class CriticalIssue(BaseModel):
    """A critical issue identified during document review."""

    title: str = Field(description="Brief issue title")
    description: str = Field(description="Detailed description")
    priority: str = Field(default="", description="Priority — high, medium, or low")
    recommendation: str = Field(default="", description="Specific actionable recommendation")


class SynthesisCriticalIssue(CriticalIssue):
    """Critical issue from multi-reviewer synthesis, with attribution."""

    issue_type: str = Field(description="Severity — major, minor, or suggestion")
    category: str = Field(
        description="Category — clarity, methodology, scientific_merit, results, novelty, or cross_cutting"
    )
    source_reviewers: list[str] = Field(
        default_factory=list,
        description="List of reviewer names who identified this issue",
    )


class DocumentReviewSynthesisOutput(BaseModel):
    """Output from the document_review_synthesizer agent."""

    review_summary: str = Field(description="Executive summary of the complete review (3-5 paragraphs)")
    critical_issues: list[SynthesisCriticalIssue] = Field(
        default_factory=list,
        description="10-15 most critical issues identified across all reviews",
    )
    recommendations: str = Field(
        description="Prioritized recommendations organized by urgency (must-fix, should-fix, consider)"
    )
    novelty_findings: Optional[str] = Field(
        default=None,
        description="Summary of novelty check results (if novelty checking was performed)",
    )


# ---------------------------------------------------------------------------
# Synthesis: review_synthesizer (multi-expert panel)
# ---------------------------------------------------------------------------


class PanelSynthesisOutput(BaseModel):
    """Output from the review_synthesizer agent (multi-expert panel)."""

    average_overall_score: float = Field(description="Average overall score across all experts")
    score_range: str = Field(description="Range of overall scores (e.g., '7-9')")
    number_of_experts: int = Field(description="Total number of expert reviewers")
    consensus_strengths: list[str] = Field(description="Strengths identified by multiple experts (3-5 items)")
    consensus_weaknesses: list[str] = Field(description="Weaknesses identified by multiple experts (3-5 items)")
    divergent_opinions: list[str] = Field(
        default_factory=list,
        description="Areas where experts disagreed (0-3 items)",
    )
    scientific_rigor_summary: str = Field(description="Synthesis of scientific rigor assessments (2-3 sentences)")
    methodology_summary: str = Field(description="Synthesis of methodology assessments (2-3 sentences)")
    novelty_summary: str = Field(description="Synthesis of novelty assessments (2-3 sentences)")
    clarity_summary: str = Field(description="Synthesis of clarity assessments (2-3 sentences)")
    overall_recommendation: str = Field(
        description="Overall recommendation — Accept, Minor Revisions, Major Revisions, or Reject"
    )
    recommendation_justification: str = Field(description="Justification for overall recommendation (4-5 sentences)")
    confidence_level: str = Field(description="Confidence in recommendation — High, Medium, or Low")
    key_decision_factors: list[str] = Field(description="Key factors that influenced the recommendation (3-5 items)")
    review_summary: str = Field(
        description="Comprehensive executive summary of the multi-expert review (5-7 paragraphs)"
    )
    critical_issues: list[CriticalIssue] = Field(
        default_factory=list,
        description="Critical issues identified across expert reviews (5-10 most important)",
    )
    novelty_findings: str = Field(
        default="",
        description="Summary of novelty check results (if novelty checking was performed)",
    )


# ---------------------------------------------------------------------------
# Synthesis: results_formatter
# ---------------------------------------------------------------------------


class FormatterOutput(BaseModel):
    """Output from the results_formatter agent."""

    review_summary: str = Field(description="Professional markdown-formatted review report")
    critical_issues: list[CriticalIssue] = Field(
        default_factory=list,
        description="Key issues extracted from the review (3-8 most important)",
    )
    novelty_findings: str = Field(
        default="",
        description="Summary of novelty check results (if novelty checking was performed)",
    )


# ---------------------------------------------------------------------------
# Multi-expert: keyword_extractor
# ---------------------------------------------------------------------------


class KeywordExtractionOutput(BaseModel):
    """Output from the keyword_extractor agent."""

    disciplines: list[str] = Field(description="List of 3-5 academic disciplines relevant to the document")


# ---------------------------------------------------------------------------
# Multi-expert: expert_generator
# ---------------------------------------------------------------------------


class ExpertProfile(BaseModel):
    """Output from the expert_generator agent — a fictional expert biosketch."""

    name: str = Field(description="Full name of the fictional expert")
    position: str = Field(description="Current academic position and institution")
    education: str = Field(description="Educational background (degrees, institutions, years)")
    contributions: str = Field(description="Key contributions to the field (3-5 bullet points)")
    research: str = Field(description="Current research focus and interests")
    discipline: str = Field(description="Primary academic discipline")


# ---------------------------------------------------------------------------
# Multi-expert: expert_reviewer
# ---------------------------------------------------------------------------


class ExpertReviewOutput(BaseModel):
    """Output from the expert_reviewer agent."""

    expert_name: str = Field(description="Name of the expert providing the review")
    discipline: str = Field(description="Expert's academic discipline")
    overall_score: int = Field(ge=1, le=10, description="Overall quality score 1-10")
    overall_assessment: str = Field(description="Brief overall assessment (2-3 sentences)")
    scientific_rigor_score: int = Field(ge=1, le=10, description="Scientific rigor score 1-10")
    scientific_rigor_justification: str = Field(description="Justification for scientific rigor score (2-3 sentences)")
    methodology_score: int = Field(ge=1, le=10, description="Methodology quality score 1-10")
    methodology_justification: str = Field(description="Justification for methodology score (2-3 sentences)")
    novelty_score: int = Field(ge=1, le=10, description="Novelty and innovation score 1-10")
    novelty_justification: str = Field(description="Justification for novelty score (2-3 sentences)")
    clarity_score: int = Field(ge=1, le=10, description="Clarity of presentation score 1-10")
    clarity_justification: str = Field(description="Justification for clarity score (2-3 sentences)")
    strengths: list[str] = Field(description="List of 3-5 key strengths")
    weaknesses: list[str] = Field(description="List of 3-5 key weaknesses")
    recommendation: str = Field(
        description="Final recommendation — Accept, Minor Revisions, Major Revisions, or Reject"
    )
    recommendation_justification: str = Field(description="Justification for recommendation (3-4 sentences)")


# ---------------------------------------------------------------------------
# Custom: schema_generator
# ---------------------------------------------------------------------------


class AnalysisField(BaseModel):
    """A single field specification generated by the schema_generator."""

    name: str = Field(description="Field name in snake_case")
    description: str = Field(description="Clear description of what this field contains")
    field_type: Literal["str", "int", "float", "bool"] = Field(
        description="Data type — str for text, int for whole numbers, float for decimals, bool for yes/no"
    )
    min_value: Optional[int] = Field(default=None, description="Minimum value for int/float fields")
    max_value: Optional[int] = Field(default=None, description="Maximum value for int/float fields")

    @model_validator(mode="after")
    def validate_constraints(self) -> AnalysisField:
        """Ensure min_value/max_value only used with numeric types."""
        if self.field_type not in ("int", "float"):
            if self.min_value is not None or self.max_value is not None:
                raise ValueError(
                    f"Field '{self.name}': min_value and max_value can only be used with "
                    f"int or float types, not {self.field_type}"
                )
        return self


class SchemaGeneratorOutput(BaseModel):
    """Output from the schema_generator agent."""

    fields: list[AnalysisField] = Field(
        description="List of AnalysisField objects defining what to extract from documents"
    )
