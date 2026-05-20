"""Tests for the Aggregate node's pure conversion helper."""

from __future__ import annotations

from andamentum.whetstone.lenses.strunk.models import StrunkFinding
from andamentum.whetstone.lenses.strunk.nodes.aggregate import (
    _to_whetstone_findings,
)


def _mk_finding(rule_number: int, start: int, end: int, **kw) -> StrunkFinding:
    return StrunkFinding(
        rule_number=rule_number,
        rule_name=kw.get("rule_name", f"rule-{rule_number}"),
        char_start=start,
        char_end=end,
        title=kw.get("title", f"R{rule_number}: violation"),
        rationale=kw.get("rationale", "test rationale"),
        severity=kw.get("severity", "minor"),
        confidence=kw.get("confidence", "high"),
        category=kw.get("category", f"r{rule_number}"),
        span_text=kw.get("span_text", ""),
        suggested_replacement=kw.get("suggested_replacement", ""),
    )


def test_aggregate_empty():
    assert _to_whetstone_findings([], "sec_001", "irrelevant") == []


def test_aggregate_sets_perspective_and_source():
    raw = [_mk_finding(2, 0, 5)]
    out = _to_whetstone_findings(raw, "sec_001", "hello world")
    assert len(out) == 1
    assert out[0].perspective == "strunk"
    assert out[0].source == "investigate"


def test_aggregate_slices_verbatim_quote_from_section():
    text = "Red, white and blue."
    raw = [_mk_finding(2, 0, 19, span_text="ignored")]
    out = _to_whetstone_findings(raw, "sec_001", text)
    q = out[0].quotes[0]
    assert q.section_id == "sec_001"
    assert q.char_start == 0
    assert q.char_end == 19
    assert q.text == "Red, white and blue"


def test_aggregate_appends_suggested_rewrite_to_rationale():
    raw = [_mk_finding(2, 0, 5, suggested_replacement="Red, white, and blue")]
    out = _to_whetstone_findings(raw, "sec_001", "hello")
    assert "Suggested rewrite" in out[0].rationale
    assert "Red, white, and blue" in out[0].rationale


def test_aggregate_sorts_by_char_start_then_rule_number():
    raw = [
        _mk_finding(13, 50, 60),
        _mk_finding(2, 10, 20),
        _mk_finding(11, 10, 20),  # same start as R2, different rule
        _mk_finding(2, 30, 40),
    ]
    out = _to_whetstone_findings(raw, "sec_001", " " * 100)
    starts_and_rules = [
        (q.quotes[0].char_start, int(q.category.lstrip("r-").split("-")[0]))
        for q in out
    ]
    assert starts_and_rules == [(10, 2), (10, 11), (30, 2), (50, 13)]


def test_aggregate_propagates_severity_and_confidence():
    raw = [
        _mk_finding(13, 0, 5, severity="major", confidence="low"),
        _mk_finding(11, 6, 10, severity="moderate", confidence="medium"),
    ]
    out = _to_whetstone_findings(raw, "sec_001", " " * 20)
    assert out[0].severity == "major"
    assert out[0].confidence == "low"
    assert out[1].severity == "moderate"
    assert out[1].confidence == "medium"


def test_aggregate_priority_derived_from_severity():
    raw = [_mk_finding(2, 0, 5, severity="major")]
    out = _to_whetstone_findings(raw, "sec_001", "hello")
    assert out[0].priority == "must_fix"

    raw = [_mk_finding(2, 0, 5, severity="moderate")]
    out = _to_whetstone_findings(raw, "sec_001", "hello")
    assert out[0].priority == "should_fix"

    raw = [_mk_finding(2, 0, 5, severity="minor")]
    out = _to_whetstone_findings(raw, "sec_001", "hello")
    assert out[0].priority == "consider"
