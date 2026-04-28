"""Tests for the priority field on Finding (Step 3 — bucketing).

Two surfaces:

1. The Finding schema validator derives priority from severity by
   default but lets callers override.
2. The markdown renderer groups findings by priority (must_fix /
   should_fix / consider), not severity.
"""

from __future__ import annotations

from andamentum.whetstone.v2.renderers.markdown import render_markdown
from andamentum.whetstone.v2.schemas import Finding, ReviewResult


def _f(severity: str, title: str = "x", **kw) -> Finding:
    return Finding(
        title=title,
        severity=severity,  # type: ignore[arg-type]
        confidence="medium",
        rationale="rationale",
        **kw,
    )


# ── Priority derivation ────────────────────────────────────────────────


def test_priority_defaults_from_major():
    f = _f("major")
    assert f.priority == "must_fix"


def test_priority_defaults_from_moderate():
    f = _f("moderate")
    assert f.priority == "should_fix"


def test_priority_defaults_from_minor():
    f = _f("minor")
    assert f.priority == "consider"


def test_priority_explicit_override():
    f = Finding(
        title="x",
        severity="major",
        confidence="medium",
        rationale="r",
        priority="consider",
    )
    assert f.priority == "consider"


def test_priority_serialises_in_model_dump():
    f = _f("major")
    dumped = f.model_dump()
    assert dumped["priority"] == "must_fix"


# ── Markdown renderer bucketing ────────────────────────────────────────


def test_markdown_renders_must_fix_heading_when_present():
    result = ReviewResult(
        deterministic_findings=[_f("major", title="Critical issue")],
    )
    out = render_markdown(result)
    assert "MUST FIX" in out
    assert "Critical issue" in out


def test_markdown_renders_three_buckets_in_order():
    result = ReviewResult(
        deterministic_findings=[
            _f("minor", title="A small thing"),
            _f("major", title="A big thing"),
            _f("moderate", title="A medium thing"),
        ],
    )
    out = render_markdown(result)
    must_idx = out.find("MUST FIX")
    should_idx = out.find("SHOULD FIX")
    consider_idx = out.find("CONSIDER")
    assert 0 < must_idx < should_idx < consider_idx


def test_markdown_omits_empty_buckets():
    result = ReviewResult(
        deterministic_findings=[_f("minor", title="Only a small thing")],
    )
    out = render_markdown(result)
    assert "CONSIDER" in out
    assert "MUST FIX" not in out
    assert "SHOULD FIX" not in out


def test_markdown_keeps_severity_label_alongside_priority():
    result = ReviewResult(
        deterministic_findings=[_f("major", title="Important")],
    )
    out = render_markdown(result)
    # Severity should still appear as a tag alongside the priority bucket
    # so reviewers can see both axes.
    assert "_major_" in out


def test_markdown_clean_document_message():
    result = ReviewResult()
    out = render_markdown(result)
    assert "looks clean" in out
