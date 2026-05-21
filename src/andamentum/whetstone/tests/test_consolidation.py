"""Unit tests for the pure consolidation substrate (no LLM, no Ollama)."""

from __future__ import annotations

from typing import Literal

from andamentum.whetstone.schemas import Finding, Quote
from andamentum.whetstone.structural.consolidation import (
    anchor_overlap,
    merge_group,
    rollup_deterministic,
    union_find_groups,
)


def _f(
    *,
    title="x",
    section="s1",
    start=0,
    end=10,
    severity: Literal["minor", "moderate", "major"] = "moderate",
    confidence: Literal["low", "medium", "high"] = "medium",
    source: Literal["deterministic", "investigate", "challenged"] = "investigate",
    perspective=None,
    category="",
    rationale="r",
    quote="quote",
) -> Finding:
    return Finding(
        title=title,
        severity=severity,
        confidence=confidence,
        rationale=rationale,
        quotes=[Quote(section_id=section, char_start=start, char_end=end, text=quote)],
        sections_involved=[section],
        source=source,
        perspective=perspective,
        category=category,
    )


# ── anchor_overlap ──────────────────────────────────────────────────────


def test_anchor_overlap_same_section_overlapping() -> None:
    assert anchor_overlap(_f(start=0, end=10), _f(start=5, end=15)) is True


def test_anchor_overlap_same_section_disjoint() -> None:
    assert anchor_overlap(_f(start=0, end=10), _f(start=10, end=20)) is False


def test_anchor_overlap_different_section() -> None:
    assert anchor_overlap(_f(section="s1"), _f(section="s2")) is False


def test_anchor_overlap_no_quote() -> None:
    no_quote = Finding(title="t", severity="minor", confidence="low", rationale="r")
    assert anchor_overlap(no_quote, _f()) is False


# ── union_find_groups ─────────────────────────────────────────────────────


def test_union_find_transitive() -> None:
    # 0-1 and 1-2 ⇒ {0,1,2}; 3 alone.
    groups = union_find_groups(4, [(0, 1), (1, 2)])
    assert groups == [[0, 1, 2], [3]]


def test_union_find_no_edges_all_singletons() -> None:
    assert union_find_groups(3, []) == [[0], [1], [2]]


# ── rollup_deterministic ──────────────────────────────────────────────────


def test_rollup_collapses_high_volume_category() -> None:
    findings = [
        _f(title="Passive voice", source="deterministic", category="style:passive",
           start=i * 20, end=i * 20 + 5, quote=f"q{i}")
        for i in range(5)
    ]
    out = rollup_deterministic(findings, min_count=3)
    assert len(out) == 1
    assert out[0].title == "5× Passive voice"
    assert "5 instances" in out[0].rationale


def test_rollup_keeps_low_volume_individual() -> None:
    findings = [
        _f(title="Duplicate word", source="deterministic",
           category="style:duplicate_word", start=0, end=5, quote="a"),
        _f(title="Duplicate word", source="deterministic",
           category="style:duplicate_word", start=20, end=25, quote="b"),
    ]
    out = rollup_deterministic(findings, min_count=3)
    assert len(out) == 2  # below threshold → untouched


def test_rollup_separates_by_section() -> None:
    findings = (
        [_f(source="deterministic", category="style:passive", section="s1",
            start=i * 20, end=i * 20 + 5, quote=f"a{i}") for i in range(3)]
        + [_f(source="deterministic", category="style:passive", section="s2",
              start=i * 20, end=i * 20 + 5, quote=f"b{i}") for i in range(3)]
    )
    out = rollup_deterministic(findings, min_count=3)
    assert len(out) == 2  # one summary per section


# ── merge_group ────────────────────────────────────────────────────────────


def test_merge_single_member_unchanged() -> None:
    f = _f()
    assert merge_group([f]) is f


def test_merge_keeps_highest_severity_and_records_perspectives() -> None:
    a = _f(title="A", severity="minor", confidence="low", perspective="rigorous")
    b = _f(title="B", severity="major", confidence="medium", perspective="skeptic")
    merged = merge_group([a, b])
    assert merged.title == "B"  # higher severity is canonical
    assert merged.corroborated_by == ["rigorous", "skeptic"]


def test_merge_bumps_confidence_when_two_perspectives_agree() -> None:
    a = _f(confidence="medium", perspective="rigorous")
    b = _f(confidence="medium", perspective="skeptic")
    merged = merge_group([a, b])
    assert merged.confidence == "high"  # bumped one tier by corroboration


def test_merge_no_bump_when_single_perspective() -> None:
    a = _f(confidence="medium", perspective="rigorous")
    b = _f(confidence="medium", perspective="rigorous")
    merged = merge_group([a, b])
    assert merged.confidence == "medium"  # same perspective → no corroboration bump


def test_merge_source_is_challenged_if_any_llm_member() -> None:
    a = _f(source="deterministic", perspective=None)
    b = _f(source="investigate", perspective="rigorous")
    merged = merge_group([a, b])
    assert merged.source == "challenged"
