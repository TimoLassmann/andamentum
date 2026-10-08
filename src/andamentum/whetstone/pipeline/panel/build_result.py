"""Deterministic assembly of the panel's ``ReviewResult`` (+ aggregate-failure gate)."""

from __future__ import annotations

from ...schemas import (
    ExpertProfile,
    ExpertReview,
    PanelSynthesis,
    ReviewMetrics,
    ReviewResult,
)


def _degradation_reason(
    *,
    profiles_attempted: int,
    profiles_produced: int,
    reviews_attempted: int,
    reviews_produced: int,
    soft_failure_threshold: float,
) -> str:
    """Why this run is degraded — empty string means healthy.

    A panel run is degraded when no expert reviews were even attempted
    (no disciplines), or when the failure rate of either fan-out stage
    (profile generation, expert review) reaches the threshold. A run
    that lost most of its experts is not green.
    """
    if profiles_attempted == 0:
        return (
            "no disciplines were identified — no expert profiles or "
            "reviews were attempted"
        )
    profile_failure_rate = 1 - profiles_produced / profiles_attempted
    if profile_failure_rate >= soft_failure_threshold:
        return (
            f"{profiles_attempted - profiles_produced}/{profiles_attempted} "
            "expert profile generation(s) failed"
        )
    if reviews_attempted:
        review_failure_rate = 1 - reviews_produced / reviews_attempted
        if review_failure_rate >= soft_failure_threshold:
            return (
                f"{reviews_attempted - reviews_produced}/{reviews_attempted} "
                "expert review(s) failed"
            )
    return ""


def build_panel_result(
    *,
    summary: str,
    sections_processed: int,
    expert_profiles: list[ExpertProfile],
    expert_reviews: list[ExpertReview],
    panel_synthesis: PanelSynthesis | None,
    profiles_attempted: int,
    reviews_attempted: int,
    llm_calls: int,
    soft_failure_threshold: float,
) -> ReviewResult:
    """Fill the ``ReviewResult`` contract with the panel-specific fields;
    the criterion-cascade fields stay empty. Sets the ``degraded`` flag
    when the aggregate-failure gate trips (dialect L7: a run that skipped
    most of its work is not green)."""
    degraded_reason = _degradation_reason(
        profiles_attempted=profiles_attempted,
        profiles_produced=len(expert_profiles),
        reviews_attempted=reviews_attempted,
        reviews_produced=len(expert_reviews),
        soft_failure_threshold=soft_failure_threshold,
    )
    return ReviewResult(
        summary=summary,
        findings=[],
        deterministic_findings=[],
        edits=[],
        author_questions=[],
        document_map=[],
        expert_profiles=list(expert_profiles),
        expert_reviews=list(expert_reviews),
        panel_synthesis=panel_synthesis,
        degraded=bool(degraded_reason),
        degraded_reason=degraded_reason,
        metrics=ReviewMetrics(
            llm_calls=llm_calls,
            wall_seconds=0.0,
            sections_processed=sections_processed,
        ),
    )
