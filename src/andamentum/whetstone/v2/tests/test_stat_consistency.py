"""Tests for statistical self-consistency (Step 8 — statcheck-equivalent).

Two surfaces:

1. The regex extractor finds (test_statistic, df, reported_p) triples in
   prose for t / F / chi2 / z / r tests.
2. The consistency check recomputes the implied p-value and emits a
   Finding when reported and implied disagree beyond rounding.

Tests use synthetic prose with known-correct and known-incorrect numbers
to lock in the comparison rules.
"""

from __future__ import annotations

import math

import pytest

from andamentum.whetstone.v2.structural.stat_consistency import (
    check_stat_consistency,
    extract_stat_claims,
    recompute_p_value,
)
from andamentum.whetstone.v2.structural.types import SectionRef


def _section(text: str) -> SectionRef:
    return SectionRef(id="sec_001", title="x", text=text, char_start=0, char_end=len(text))


# ── Extraction ─────────────────────────────────────────────────────────


def test_extracts_t_test():
    claims = extract_stat_claims([_section("We found t(48) = 2.34, p = 0.024.")])
    assert len(claims) == 1
    c = claims[0]
    assert c.test_kind == "t"
    assert c.df == (48,)
    assert c.statistic_value == pytest.approx(2.34)
    assert c.reported_p_op == "="
    assert c.reported_p_value == pytest.approx(0.024)


def test_extracts_F_test():
    claims = extract_stat_claims([_section("F(2, 96) = 4.51, p = 0.013.")])
    assert len(claims) == 1
    c = claims[0]
    assert c.test_kind == "F"
    assert c.df == (2, 96)


def test_extracts_chi2_test_unicode():
    claims = extract_stat_claims([_section("χ²(2) = 5.91, p = 0.052.")])
    assert len(claims) == 1
    assert claims[0].test_kind == "chi2"
    assert claims[0].df == (2,)


def test_extracts_chi2_test_ascii():
    claims = extract_stat_claims(
        [_section("Chi-square(3) = 7.81, p = 0.05.")]
    )
    assert len(claims) == 1
    assert claims[0].test_kind == "chi2"


def test_extracts_z_test():
    claims = extract_stat_claims([_section("z = 1.96, p = 0.05.")])
    assert len(claims) == 1
    assert claims[0].test_kind == "z"
    assert claims[0].df == ()


def test_extracts_r_with_df():
    claims = extract_stat_claims([_section("r(48) = 0.42, p < 0.001.")])
    assert len(claims) == 1
    assert claims[0].test_kind == "r"
    assert claims[0].df == (48,)
    assert claims[0].reported_p_op == "<"


def test_extracts_r_no_df_with_n():
    claims = extract_stat_claims(
        [_section("Effect r = .42, p = .002, n = 50.")]
    )
    assert len(claims) == 1
    assert claims[0].test_kind == "r"
    assert claims[0].df == (48,)  # n - 2


def test_dotleading_p_value_normalised():
    claims = extract_stat_claims([_section("t(48) = 2.34, p = .024.")])
    assert claims[0].reported_p_value == pytest.approx(0.024)


def test_no_match_when_pattern_partial():
    # Missing p-value side
    claims = extract_stat_claims([_section("We found t(48) = 2.34 in our data.")])
    assert claims == []


# ── Recomputation ──────────────────────────────────────────────────────


def test_recompute_t_test():
    claim = extract_stat_claims([_section("t(48) = 2.34, p = 0.024.")])[0]
    p = recompute_p_value(claim)
    assert p is not None
    # Two-tailed t(48), |t|=2.34: p ≈ 0.0235
    assert p == pytest.approx(0.0235, abs=0.001)


def test_recompute_z_test():
    claim = extract_stat_claims([_section("z = 1.96, p = 0.05.")])[0]
    p = recompute_p_value(claim)
    assert p is not None
    # z=1.96 → two-tailed p ≈ 0.05
    assert p == pytest.approx(0.05, abs=0.005)


def test_recompute_F_test_one_tailed():
    claim = extract_stat_claims([_section("F(2, 96) = 4.51, p = 0.013.")])[0]
    p = recompute_p_value(claim)
    assert p is not None
    # F(2,96) = 4.51 → p ≈ 0.0135
    assert p == pytest.approx(0.0135, abs=0.005)


def test_recompute_chi2():
    claim = extract_stat_claims([_section("χ²(2) = 5.99, p = 0.05.")])[0]
    p = recompute_p_value(claim)
    assert p is not None
    # χ²(2)=5.99 → p ≈ 0.05
    assert p == pytest.approx(0.05, abs=0.01)


def test_recompute_r_through_t():
    claim = extract_stat_claims([_section("r(48) = 0.30, p = 0.04.")])[0]
    p = recompute_p_value(claim)
    assert p is not None
    # r=0.30, df=48 → t = 0.30 * sqrt(48 / 0.91) ≈ 2.18; p two-tailed ≈ 0.034
    assert p == pytest.approx(0.034, abs=0.005)


def test_recompute_r_with_invalid_value_returns_none():
    # |r| ≥ 1 is impossible; recompute should refuse rather than NaN
    claim = extract_stat_claims([_section("r(48) = 1.00, p = 0.001.")])[0]
    assert recompute_p_value(claim) is None


# ── Consistency check ──────────────────────────────────────────────────


def test_consistent_t_test_emits_no_finding():
    findings = check_stat_consistency(
        [_section("We found t(48) = 2.34, p = 0.024.")]
    )
    assert findings == []


def test_decision_changing_inconsistency_is_major():
    # Reported as significant (p < 0.05), but t(48)=1.20 → p ≈ 0.236
    findings = check_stat_consistency(
        [_section("Effect significant: t(48) = 1.20, p < 0.05.")]
    )
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "major"
    assert "decision-changing" in f.rationale.lower()
    assert f.category == "statistics"


def test_non_decision_changing_inconsistency_is_moderate():
    # Reported and implied both > 0.05 but materially different
    # t(20) = 1.0 → p ≈ 0.33, but author wrote p = 0.5
    findings = check_stat_consistency(
        [_section("Not significant: t(20) = 1.00, p = 0.50.")]
    )
    assert len(findings) == 1
    assert findings[0].severity == "moderate"


def test_inequality_p_consistent_when_implied_within_bound():
    # t(48) = 5.0 → p << 0.001; reporter said p < 0.001; consistent
    findings = check_stat_consistency(
        [_section("Strong effect: t(48) = 5.00, p < 0.001.")]
    )
    assert findings == []


def test_inequality_p_inconsistent_when_implied_exceeds_bound():
    # t(48) = 1.5 → p ≈ 0.140; reporter said p < 0.05 (not satisfied)
    findings = check_stat_consistency(
        [_section("We claimed: t(48) = 1.50, p < 0.05.")]
    )
    assert len(findings) == 1
    assert findings[0].severity == "major"  # decision-changing


def test_finding_includes_quote_anchored_to_section():
    findings = check_stat_consistency(
        [_section("Effect: t(48) = 1.20, p < 0.05 was found.")]
    )
    assert len(findings) == 1
    quote = findings[0].quotes[0]
    assert quote.section_id == "sec_001"
    assert "t(48)" in quote.text
    assert "p < 0.05" in quote.text


def test_unparseable_combination_skipped_silently():
    # df=0 → cannot compute; should NOT emit a low-confidence finding
    findings = check_stat_consistency(
        [_section("Pathological case: t(0) = 2.00, p = 0.05.")]
    )
    assert findings == []


def test_multiple_claims_in_one_section():
    text = (
        "Across two analyses: t(48) = 2.34, p = 0.024; "
        "and F(2, 96) = 4.51, p = 0.013."
    )
    claims = extract_stat_claims([_section(text)])
    assert len(claims) == 2


def test_recompute_does_not_raise_on_pathological_values():
    # Negative df, NaN-able values — should return None, not crash
    claim = extract_stat_claims([_section("F(2, 96) = -1.00, p = 0.5.")])[0]
    p = recompute_p_value(claim)
    # F can technically accept negative input but produce NaN; recompute
    # should either return a valid number or None — never raise.
    assert p is None or (not math.isnan(p))
