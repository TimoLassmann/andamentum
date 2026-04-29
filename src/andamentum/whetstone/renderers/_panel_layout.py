"""Shared layout helpers for panel-mode rendering.

The three renderers (markdown, html, docx) all need:
  • per-criterion aggregated scores (average, range, per-expert breakdown)
  • a way to detect criteria where the panel materially disagreed
  • the prose summary attached to each named criterion

These helpers compute it once. The renderers convert the result to
their respective output formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..schemas import ExpertReview, PanelSynthesis


# A range of 2+ points across reviewers (e.g. 5 vs 7) is the threshold
# at which the per-criterion prose is worth surfacing. Below 2 points
# the panel agreed, and the score table tells the whole story.
DIVERGENCE_THRESHOLD = 2


@dataclass(frozen=True)
class CriterionScores:
    name: str
    average: float
    min_score: int
    max_score: int
    per_expert: tuple[tuple[str, int], ...]

    @property
    def range_str(self) -> str:
        if self.min_score == self.max_score:
            return str(self.min_score)
        return f"{self.min_score}-{self.max_score}"

    @property
    def diverged(self) -> bool:
        return (self.max_score - self.min_score) >= DIVERGENCE_THRESHOLD


_CRITERIA: tuple[tuple[str, Callable[[ExpertReview], int]], ...] = (
    ("Overall", lambda r: r.overall_score),
    ("Scientific rigor", lambda r: r.scientific_rigor_score),
    ("Methodology", lambda r: r.methodology_score),
    ("Novelty", lambda r: r.novelty_score),
    ("Clarity", lambda r: r.clarity_score),
)


def reviewer_scores(reviews: list[ExpertReview]) -> list[CriterionScores]:
    """One CriterionScores row per criterion, aggregated across reviews."""
    if not reviews:
        return []
    rows: list[CriterionScores] = []
    for name, getter in _CRITERIA:
        scores = [getter(r) for r in reviews]
        rows.append(
            CriterionScores(
                name=name,
                average=sum(scores) / len(scores),
                min_score=min(scores),
                max_score=max(scores),
                per_expert=tuple((r.expert_name, getter(r)) for r in reviews),
            )
        )
    return rows


def diverged_criteria(rows: list[CriterionScores]) -> list[CriterionScores]:
    """Subset of rows where the panel showed material disagreement."""
    # "Overall" is excluded because its prose lives in recommendation_justification,
    # not in a criterion-specific summary field.
    return [r for r in rows if r.diverged and r.name != "Overall"]


def criterion_summary(name: str, s: PanelSynthesis) -> str:
    return {
        "Scientific rigor": s.scientific_rigor_summary,
        "Methodology": s.methodology_summary,
        "Novelty": s.novelty_summary,
        "Clarity": s.clarity_summary,
    }.get(name, "")


def headline_line(s: PanelSynthesis) -> str:
    """One-line BLUF: recommendation + confidence + average + range + n."""
    return (
        f"**Recommendation: {s.overall_recommendation}** "
        f"(confidence: {s.confidence_level}) · Average score "
        f"**{s.average_overall_score:.1f}/10** "
        f"(range: {s.score_range}, n={s.number_of_experts})"
    )


def body_markdown(s: PanelSynthesis, reviews: list[ExpertReview]) -> str:
    """Panel-synthesis body as markdown with level-3 (###) subheadings.

    The headline line + recommendation_justification go first (level-3
    headings would be too noisy for a one-liner). Subsequent sections —
    reviewer scores, consensus, divergence, key factors, detailed
    synthesis — each get their own ``###`` heading.

    The caller is expected to wrap this in whatever section heading is
    appropriate (``## Panel synthesis`` for a standalone markdown doc,
    nothing in docx where the content goes inside ``## Executive
    Summary``).
    """
    lines: list[str] = [headline_line(s), ""]

    if s.recommendation_justification.strip():
        lines += [s.recommendation_justification.strip(), ""]

    rows = reviewer_scores(reviews)
    if rows:
        lines += ["### Reviewer scores", ""]
        for row in rows:
            per_expert = " · ".join(f"{name} {score}" for name, score in row.per_expert)
            lines.append(
                f"- **{row.name}:** {row.average:.1f} avg, range "
                f"{row.range_str} · {per_expert}"
            )
        lines.append("")

    if s.consensus_strengths:
        lines += ["### Consensus strengths", ""]
        lines += [f"- {item}" for item in s.consensus_strengths]
        lines.append("")

    if s.consensus_weaknesses:
        lines += ["### Consensus weaknesses", ""]
        lines += [f"- {item}" for item in s.consensus_weaknesses]
        lines.append("")

    if s.divergent_opinions:
        lines += ["### Divergent opinions", ""]
        lines += [f"- {item}" for item in s.divergent_opinions]
        lines.append("")

    diverged = diverged_criteria(rows)
    if diverged:
        lines += ["### Where the panel diverged", ""]
        for row in diverged:
            body = criterion_summary(row.name, s).strip()
            tail = f" {body}" if body else ""
            lines += [f"**{row.name}** (range {row.range_str}).{tail}", ""]

    if s.key_decision_factors:
        lines += ["### Key decision factors", ""]
        lines += [f"- {item}" for item in s.key_decision_factors]
        lines.append("")

    if s.review_summary.strip():
        lines += ["### Detailed synthesis", "", s.review_summary.strip(), ""]

    return "\n".join(lines).rstrip()
