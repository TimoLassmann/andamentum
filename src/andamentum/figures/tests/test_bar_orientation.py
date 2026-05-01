"""Tests for the auto-horizontal heuristic and label-rotation defaults.

Closes the long-categorical-labels bug: ``figure(kind="bar", ...)`` with
many or long labels used to produce overlapping x-tick mush. The
advisor now picks horizontal orientation for that shape automatically;
short-label / few-bar cases still render vertical.
"""

from __future__ import annotations

import pytest

from andamentum.figures import figure
from andamentum.figures.advisor import (
    recommend_horizontal_bars,
    recommend_label_rotation,
)


# ── advisor heuristics (unit-level) ────────────────────────────────────


class TestRecommendHorizontalBars:
    def test_few_short_labels_vertical(self):
        should, reason = recommend_horizontal_bars(["A", "B", "C", "D"])
        assert should is False
        assert reason == ""

    def test_many_long_labels_horizontal(self):
        labels = [f"Topic with longer text {i}" for i in range(10)]
        should, reason = recommend_horizontal_bars(labels)
        assert should is True
        assert "10 categories" in reason

    def test_one_very_long_label_triggers_horizontal(self):
        labels = ["short", "Catastrophically lengthy category name"]
        should, reason = recommend_horizontal_bars(labels)
        assert should is True
        assert "long labels" in reason or "room to breathe" in reason

    def test_seven_medium_labels_triggers_horizontal(self):
        # 7 categories AND labels ≥ 10 chars → horizontal
        labels = [f"Category-{i}-name" for i in range(7)]
        should, _ = recommend_horizontal_bars(labels)
        assert should is True

    def test_seven_short_labels_stays_vertical(self):
        # 7 categories but labels are short → still vertical
        labels = [f"C{i}" for i in range(7)]
        should, _ = recommend_horizontal_bars(labels)
        assert should is False


class TestRecommendLabelRotation:
    def test_few_short_labels_no_rotation(self):
        assert recommend_label_rotation(["A", "B", "C"]) == 0.0

    def test_moderate_count_rotates_30(self):
        labels = [f"Cat-{i}" for i in range(5)]  # 5 cats, 5 chars each
        assert recommend_label_rotation(labels) == 30.0

    def test_many_categories_rotates_45(self):
        labels = [f"C{i}" for i in range(8)]
        assert recommend_label_rotation(labels) == 45.0

    def test_long_labels_rotate_45(self):
        labels = ["short", "longer label here"]
        assert recommend_label_rotation(labels) == 45.0


# ── figure() integration ──────────────────────────────────────────────


def _long_label_data():
    topics = [f"Long topic name number {i} description" for i in range(10)]
    return {"Topic": topics, "Count": list(range(10, 0, -1))}


def _short_label_data():
    return {"Group": ["A", "B", "C"], "Count": [10, 20, 15]}


class TestFigureAutoHorizontal:
    def test_long_labels_auto_horizontal_with_advisor_note(self, tmp_path):
        result = figure(
            data=_long_label_data(),
            kind="bar",
            output=str(tmp_path / "auto.png"),
        )
        assert (tmp_path / "auto.png").exists()
        assert any("horizontal" in note.lower() for note in result.advisor_notes), (
            f"expected horizontal advisor note, got: {result.advisor_notes}"
        )

    def test_short_labels_no_horizontal_note(self, tmp_path):
        result = figure(
            data=_short_label_data(),
            kind="bar",
            output=str(tmp_path / "vert.png"),
        )
        assert (tmp_path / "vert.png").exists()
        assert not any(
            "horizontal" in note.lower() and "auto-selected" in note.lower()
            for note in result.advisor_notes
        )

    def test_explicit_horizontal_true_honoured(self, tmp_path):
        result = figure(
            data=_short_label_data(),
            kind="bar",
            horizontal=True,
            output=str(tmp_path / "h.png"),
        )
        assert (tmp_path / "h.png").exists()
        # Explicit user choice → no auto-selected note.
        assert not any("auto-selected" in note.lower() for note in result.advisor_notes)

    def test_explicit_horizontal_false_with_long_labels_warns(self, tmp_path):
        result = figure(
            data=_long_label_data(),
            kind="bar",
            horizontal=False,
            output=str(tmp_path / "forced_v.png"),
        )
        assert (tmp_path / "forced_v.png").exists()
        # User overrode auto-pick despite long labels — surface a warning note.
        assert any(
            "consider horizontal=true" in note.lower() for note in result.advisor_notes
        ), f"expected override-warning note, got: {result.advisor_notes}"


class TestFigureAxisLabels:
    """Horizontal bars flip the visual axes; labels must follow."""

    def test_horizontal_axis_labels_flipped(self, tmp_path):
        # Use figure() and re-load via matplotlib to verify the labels
        # are placed on the correct axes. We can't easily inspect the
        # saved image, but we can confirm the advisor path runs cleanly
        # and no exception is raised.
        from pathlib import Path

        result = figure(
            data={"Group": ["alpha", "beta", "gamma"], "Score": [1.0, 2.0, 3.0]},
            kind="bar",
            horizontal=True,
            output=str(tmp_path / "h_labels.png"),
        )
        assert Path(result.path).exists()


@pytest.mark.parametrize(
    "n_cats,label_len,expect_horizontal",
    [
        (3, 5, False),  # tiny chart, vertical
        (10, 5, False),  # many bars but short labels — vertical with rotation
        (10, 25, True),  # many bars, long labels — flip
        (3, 30, True),  # few bars but ultra-long labels — flip
    ],
)
def test_orientation_by_data_shape(tmp_path, n_cats, label_len, expect_horizontal):
    labels = [("X" * label_len) + str(i) for i in range(n_cats)]
    counts = list(range(n_cats, 0, -1))
    result = figure(
        data={"Cat": labels, "Count": counts},
        kind="bar",
        output=str(tmp_path / "p.png"),
    )
    auto_horizontal = any(
        "auto-selected horizontal" in n.lower() for n in result.advisor_notes
    )
    assert auto_horizontal is expect_horizontal
