"""Figure-vs-text goal check. The one genuinely new, narrowly-scoped piece.

Whetstone has no image-extraction path from a ``.docx`` and no notion of a
figure at all — building a general figure-locator is out of scope here.
This module instead composes two things that already exist: a plain regex
scan for ``"Figure N"`` / ``"Fig. N"`` mentions, and
``andamentum.vision_critique.critique_figure``'s existing ``extra_context``
and ``schema`` parameters — the caption and referencing sentences go in as
context, and the model's verdict comes back typed. Nothing inside
``vision_critique`` changes.

It does NOT discover embedded images in a ``.docx`` on its own, does NOT
associate a caption with an image automatically, and does NOT handle a
figure with no textual label — the caller names each figure explicitly (see
``spec.FigureRef``).
"""

from __future__ import annotations

import bisect
import re

from pydantic import BaseModel, Field

from .spec import FigureRef


class FigureTextMatch(BaseModel):
    """One vision model's verdict on one figure against its caption/text."""

    matches_description: bool
    discrepancies: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class FigureCheckResult(BaseModel):
    label: str
    matched: FigureTextMatch | None  # None on a crashed call or an unmentioned figure
    error: str = ""


class FigureTextGoalReport(BaseModel):
    checks: list[FigureCheckResult] = Field(default_factory=list)

    @property
    def satisfied(self) -> bool:
        return bool(self.checks) and all(
            c.matched is not None and c.matched.matches_description for c in self.checks
        )


_FIGURE_LABEL_ID = re.compile(r"(\S+)\s*$")


def locate_figure_mentions(text: str, label: str) -> tuple[str, list[str]]:
    """The caption line and every OTHER line mentioning *label* in *text*.

    *label*'s trailing token (e.g. ``"1"`` from ``"Figure 1"``) is matched
    against ``Fig``/``Figure``/``Figs.``/``Fig.`` followed by that token, as
    a whole word — so ``"Figure 1"`` does not also match ``"Figure 10"``.
    The FIRST occurrence whose match starts at (or right after whitespace
    following) its line's start is treated as the caption; every other
    occurrence's containing line is a referencing sentence. Returns
    ``("", [])`` when *label* is never mentioned at all.
    """
    id_match = _FIGURE_LABEL_ID.search(label.strip())
    figure_id = id_match.group(1) if id_match else label.strip()
    pattern = re.compile(
        rf"\bFig(?:ure|s\.|\.)?\s*{re.escape(figure_id)}\b", re.IGNORECASE
    )

    lines = text.split("\n")
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    caption = ""
    referencing: list[str] = []
    caption_found = False
    for occurrence in pattern.finditer(text):
        line_index = bisect.bisect_right(offsets, occurrence.start()) - 1
        line = lines[line_index]
        prefix = text[offsets[line_index] : occurrence.start()]
        if not caption_found and not prefix.strip():
            caption = line.strip()
            caption_found = True
            continue
        stripped = line.strip()
        if stripped and stripped not in referencing:
            referencing.append(stripped)
    return caption, referencing


def unmentioned_figures(text: str, figures: tuple[FigureRef, ...]) -> list[str]:
    """Labels among *figures* that never appear in *text* at all."""
    missing = []
    for figure in figures:
        caption, referencing = locate_figure_mentions(text, figure.label)
        if not caption and not referencing:
            missing.append(figure.label)
    return missing


async def check_figure_text_match(
    text: str, figures: tuple[FigureRef, ...], *, model: str
) -> FigureTextGoalReport:
    """Check each of *figures* against the caption/text that names it.

    A figure never mentioned in *text*, or whose vision call crashes, is
    recorded with ``matched=None`` and an ``error`` — never silently
    dropped, and never counted as a pass (see
    :attr:`FigureTextGoalReport.satisfied`).
    """
    from andamentum.vision_critique import critique_figure

    checks: list[FigureCheckResult] = []
    for figure in figures:
        caption, referencing = locate_figure_mentions(text, figure.label)
        if not caption and not referencing:
            checks.append(
                FigureCheckResult(
                    label=figure.label,
                    matched=None,
                    error=(
                        f'"{figure.label}" is never mentioned in the text '
                        '(no "Figure N" / "Fig. N" match).'
                    ),
                )
            )
            continue
        context = "Caption: {}\n\nSentences in the manuscript that reference this figure:\n{}".format(
            caption or "(none found; only referencing sentences below)",
            "\n".join(f"- {s}" for s in referencing) or "(none beyond the caption)",
        )
        try:
            matched = await critique_figure(
                figure.path,
                model=model,
                schema=FigureTextMatch,
                extra_context=context,
            )
        except Exception as exc:
            checks.append(FigureCheckResult(label=figure.label, matched=None, error=str(exc)))
            continue
        checks.append(FigureCheckResult(label=figure.label, matched=matched))
    return FigureTextGoalReport(checks=checks)
