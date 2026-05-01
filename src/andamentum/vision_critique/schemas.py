"""Bounded pydantic schemas for vision critique.

Critique schemas are deliberately tight: enums for any "what's wrong"
field, fixed-set ``Literal`` lists for any "suggested fix" field. This
serves two purposes — small local vision models reliably fill bounded
schemas (validated against ``ollama:gemma4:e4b-it-q4_K_M`` on the
``broken_bar.png`` calibration fixture), and downstream callers get a
predictable action surface to map to render-parameter changes.

The default schema is ``FigureCritique`` (chart-figure layout review).
Callers wanting a different shape (e.g. whetstone reviewing a manuscript
panel for figure-quality issues) pass their own ``BaseModel`` subclass
to ``critique_figure(schema=...)``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


AspectIssue = Literal["too_tall", "too_wide", "ok"]


SuggestedFix = Literal[
    "rotate_x_labels",
    "horizontal_bars",
    "increase_width",
    "truncate_labels",
    "wrap_labels",
    "smaller_font",
    "no_change_needed",
]


class FigureCritique(BaseModel):
    """Bounded layout-and-readability critique of a rendered figure.

    Fields are intentionally minimal and close-set — small vision models
    fill them reliably, and the receiver can mechanically translate each
    flagged issue into a render-parameter change.
    """

    label_overlap: bool = Field(
        description=(
            "Do x-axis or y-axis tick labels visibly overlap or run into each other?"
        )
    )
    labels_legible: bool = Field(
        description=(
            "Are all axis tick labels readable without straining? "
            "False if any are squished, rotated into illegibility, or "
            "cut off."
        )
    )
    legend_blocks_data: bool = Field(
        description=(
            "Does the legend, if present, cover any data points, bars, "
            "or lines? False if there is no legend or it is well-placed."
        )
    )
    aspect_ratio_issue: AspectIssue = Field(
        description=(
            "Is the figure too tall, too wide, or ok-shaped for the data it carries?"
        )
    )
    suggested_fixes: list[SuggestedFix] = Field(
        default_factory=list,
        description=(
            "Concrete fixes from the allowed set. Empty list — or a "
            "single 'no_change_needed' — when the figure is fine."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="0..1 confidence in this critique.",
    )
    one_line_summary: str = Field(
        description=(
            "One short sentence summarising the main issue, or "
            "'figure is fine' when nothing needs changing."
        )
    )

    @property
    def has_issues(self) -> bool:
        """True when at least one flag is set or a real fix is suggested."""
        if self.label_overlap or not self.labels_legible:
            return True
        if self.legend_blocks_data:
            return True
        if self.aspect_ratio_issue != "ok":
            return True
        real_fixes = [f for f in self.suggested_fixes if f != "no_change_needed"]
        return bool(real_fixes)
