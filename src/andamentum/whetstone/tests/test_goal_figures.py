"""Tests for the figure-vs-text goal check.

``locate_figure_mentions`` is pure — no model call. ``check_figure_text_match``
stubs ``vision_critique.critique_figure`` the way ``test_sequential_rounds.py``
stubs whetstone's own agents: patched at its call site, never a real network
call.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from andamentum.whetstone.goal.figures import (
    FigureTextMatch,
    check_figure_text_match,
    locate_figure_mentions,
    unmentioned_figures,
)
from andamentum.whetstone.goal.spec import FigureRef


def test_caption_and_referencing_sentence_are_split() -> None:
    text = (
        "Figure 1. A bar chart of counts.\n"
        "As shown in Figure 1, counts rise steadily.\n"
    )
    caption, referencing = locate_figure_mentions(text, "Figure 1")
    assert caption == "Figure 1. A bar chart of counts."
    assert referencing == ["As shown in Figure 1, counts rise steadily."]


def test_figure_1_does_not_collide_with_figure_10() -> None:
    text = "Figure 1. The main result.\nFigure 10. A supplementary panel.\n"
    caption, referencing = locate_figure_mentions(text, "Figure 1")
    assert caption == "Figure 1. The main result."
    assert referencing == []


def test_multiple_referencing_sentences_are_all_kept() -> None:
    text = (
        "Figure 2. A scatter plot.\n"
        "Figure 2 shows the correlation.\n"
        "We revisit Figure 2 in the discussion.\n"
    )
    caption, referencing = locate_figure_mentions(text, "Figure 2")
    assert caption == "Figure 2. A scatter plot."
    assert len(referencing) == 2


def test_never_mentioned_returns_empty() -> None:
    caption, referencing = locate_figure_mentions(
        "Nothing about figures here.", "Figure 3"
    )
    assert caption == ""
    assert referencing == []


def test_unmentioned_figures_lists_only_the_missing_ones() -> None:
    text = "Figure 1. Present.\n"
    figures = (
        FigureRef(label="Figure 1", path=Path("fig1.png")),
        FigureRef(label="Figure 2", path=Path("fig2.png")),
    )
    assert unmentioned_figures(text, figures) == ["Figure 2"]


async def test_check_reports_unmentioned_figure_without_a_model_call() -> None:
    figures = (FigureRef(label="Figure 9", path=Path("fig9.png")),)
    with patch("andamentum.vision_critique.critique_figure") as mock_critique:
        report = await check_figure_text_match(
            "No figures mentioned here.", figures, model="stub"
        )
    mock_critique.assert_not_called()
    assert not report.satisfied
    assert report.checks[0].matched is None
    assert "never mentioned" in report.checks[0].error


async def test_check_satisfied_when_the_model_reports_a_match() -> None:
    figures = (FigureRef(label="Figure 1", path=Path("fig1.png")),)
    text = "Figure 1. Counts by group.\nFigure 1 shows four bars.\n"

    async def fake_critique(image, *, model, schema, extra_context=None):
        assert extra_context is not None and "Counts by group" in extra_context
        return FigureTextMatch(
            matches_description=True, discrepancies=[], confidence=0.9
        )

    with patch("andamentum.vision_critique.critique_figure", new=fake_critique):
        report = await check_figure_text_match(text, figures, model="stub")

    assert report.satisfied


async def test_check_not_satisfied_on_a_reported_mismatch() -> None:
    figures = (FigureRef(label="Figure 1", path=Path("fig1.png")),)
    text = "Figure 1. Four bars.\nFigure 1 shows four bars rising.\n"

    async def fake_critique(image, *, model, schema, extra_context=None):
        return FigureTextMatch(
            matches_description=False,
            discrepancies=["the figure shows three bars, not four"],
            confidence=0.8,
        )

    with patch("andamentum.vision_critique.critique_figure", new=fake_critique):
        report = await check_figure_text_match(text, figures, model="stub")

    assert not report.satisfied
    assert report.checks[0].matched is not None
    assert report.checks[0].matched.discrepancies == [
        "the figure shows three bars, not four"
    ]


async def test_a_crashed_call_is_recorded_not_raised() -> None:
    figures = (FigureRef(label="Figure 1", path=Path("fig1.png")),)
    text = "Figure 1. Something.\n"

    async def boom(image, *, model, schema, extra_context=None):
        raise RuntimeError("model unavailable")

    with patch("andamentum.vision_critique.critique_figure", new=boom):
        report = await check_figure_text_match(text, figures, model="stub")

    assert not report.satisfied
    assert report.checks[0].matched is None
    assert "model unavailable" in report.checks[0].error
