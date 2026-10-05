from __future__ import annotations

import pytest

from andamentum.compose.harness.checks.statistics import (
    StatisticsCheck,
    _grim_possible,
)

from ._helpers import CTX, make_view


def run(text):
    return StatisticsCheck().run(make_view((text, None)), CTX)


@pytest.mark.parametrize(
    "text",
    [
        "t(24) = 2.10, p = .046",  # exact p is 0.0463
        "t(24) = 2.10, p = 0.046",
        "t(24) = 2.10, p < .05",
        "F(1, 48) = 4.50, p = .039",
        "χ²(2) = 6.00, p = .050",
        "χ2(2, N = 100) = 6.00, p = .050",
        "r(48) = .35, p = .013",
        "z = 1.96, p = .050",
        "t(30) = 5.00, p < .001",
        "t(30) = −1.50, p = .144",
        "no statistics here, p = .03",
    ],
)
def test_consistent(text):
    assert run(text) == []


def test_decision_flip_is_error():
    (d,) = run("The effect was significant, t(24) = 1.50, p = .03.")
    assert d.severity == "error"
    assert d.data["decision_flips"] is True
    assert d.paragraph_id == "p0"
    assert "t(24) = 1.50, p = .03" == d.evidence[0]


def test_flip_other_direction():
    (d,) = run("F(1, 48) = 4.50, p = .20")
    assert d.severity == "error"


def test_mismatch_without_flip_is_warning():
    (d,) = run("t(24) = 3.50, p = .01")  # exact .0018, both significant
    assert d.severity == "warning"
    assert d.data["decision_flips"] is False


def test_less_than_wrong():
    (d,) = run("t(24) = 1.50, p < .05")
    assert d.severity == "error"


def test_one_tailed_accepted():
    assert run("A one-tailed test: t(24) = 1.80, p = .042") == []
    assert run("t(24) = 1.80, p = .042") != []


def test_rounding_range_forgiven():
    # exact p for t(24)=2.05 is 0.0513, 2.055 gives 0.0508; reported .051 fits
    assert run("t(24) = 2.05, p = .051") == []


def test_multiple_results_in_paragraph():
    out = run("t(10) = 1.00, p = .34 and t(10) = 1.00, p = .001.")
    assert len(out) == 1
    assert out[0].data["reported_p"] == "=.001"


def test_grim_flags_impossible_mean():
    out = run("Scores (M = 3.48, n = 20, 1-item scale) were high.")
    assert len(out) == 1
    assert "GRIM" in out[0].message


def test_grim_accepts_possible_mean():
    assert run("Scores (M = 3.45, n = 20, single-item) were high.") == []


def test_grim_skipped_without_items_or_n():
    assert run("M = 3.48, n = 20") == []
    assert run("M = 3.48 on a 5-item scale") == []
    assert run("M = 3.48, n = 20 and n = 21, 5-item") == []


def test_grim_skipped_when_n_large():
    assert run("M = 3.48, n = 200, 1-item") == []


def test_grim_possible_rounding():
    assert _grim_possible("3.45", 20)
    assert not _grim_possible("3.48", 20)
    assert _grim_possible("2.5", 4)
