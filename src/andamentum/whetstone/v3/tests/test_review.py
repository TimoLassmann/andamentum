"""Tests for the criterion set, generic review (mocked), and verify-findings."""

from __future__ import annotations

import logging
import types
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded

from andamentum.whetstone.v3.criteria import SPECS, Criterion, criterion_set_for
from andamentum.whetstone.v3.model import DocumentModel, Section
from andamentum.whetstone.v3.review import (
    Finding,
    _CriterionFindings,
    _RawFinding,
    run_criteria,
    verify_findings,
)


@contextmanager
def _capture_v3_logs() -> Iterator[list[logging.LogRecord]]:
    """Attach a handler directly to the v3 logger. ``caplog`` is unreliable
    here because ``whetstone/cli.py`` flips ``andamentum.whetstone.propagate``
    to False, breaking propagation to the root caplog handler when CLI tests
    run earlier in the same session."""
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    logger = logging.getLogger("andamentum.whetstone.v3")
    prior_level = logger.level
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)


def test_specs_is_the_academic_default() -> None:
    assert criterion_set_for("academic") is SPECS
    assert criterion_set_for("anything-unknown") is SPECS  # falls back
    assert [c.name for c in SPECS] == [
        "Story",
        "Presentation",
        "Evaluations",
        "Correctness",
        "Significance",
    ]
    assert all(c.questions for c in SPECS)


def _model(src: str) -> DocumentModel:
    return DocumentModel(
        source=src,
        sections=[Section(id="s1", title="S", text=src, start=0, end=len(src))],
    )


async def test_run_criteria_tags_findings_by_criterion() -> None:
    out = _CriterionFindings(
        findings=[
            _RawFinding(issue="x", quote="the claim is unsupported", severity="major")
        ]
    )

    class _Agent:
        async def run(self, _prompt, **_kwargs):
            # Accept deps/usage_limits kwargs from review_criterion silently.
            return types.SimpleNamespace(output=out)

    def _build(_criterion, _agent_model):
        return _Agent()

    with patch("andamentum.whetstone.v3.review._build_agent", new=_build):
        findings = await run_criteria(SPECS, _model("x"), agent_model="stub")
    # one finding per criterion, each tagged with its criterion name
    assert {f.criterion for f in findings} == {c.name for c in SPECS}


async def test_run_criteria_logs_unexpected_model_behaviour_body() -> None:
    """Stage 3: UnexpectedModelBehavior.body should land in the log line.

    Today this exception is caught by the trailing ``except Exception``
    and logged with only ``str(exc)`` — the upstream response body
    (e.g. Ollama's null-content HTTP 400 payload) is lost. With the
    typed cascade in place we log the first 500 chars of ``body``.
    """

    class _Agent:
        async def run(self, _prompt, **_kwargs):
            raise UnexpectedModelBehavior(
                "stage-3 boom", body="upstream provider error payload"
            )

    def _build(_criterion, _agent_model):
        return _Agent()

    one_criterion = [Criterion(name="Story", questions=["q?"], facets=[])]
    with patch("andamentum.whetstone.v3.review._build_agent", new=_build):
        with _capture_v3_logs() as records:
            findings = await run_criteria(
                one_criterion, _model("x"), agent_model="stub"
            )

    assert findings == []  # the criterion gracefully skipped
    log_text = "\n".join(r.getMessage() for r in records)
    assert "Story" in log_text
    assert "model behaviour error" in log_text
    assert "upstream provider error payload" in log_text


async def test_run_criteria_logs_usage_limit_exceeded() -> None:
    """Stage 3: UsageLimitExceeded gets its own log line, distinct from
    generic crashes."""

    class _Agent:
        async def run(self, _prompt, **_kwargs):
            raise UsageLimitExceeded("request_limit (18) exceeded")

    def _build(_criterion, _agent_model):
        return _Agent()

    one_criterion = [Criterion(name="Story", questions=["q?"], facets=[])]
    with patch("andamentum.whetstone.v3.review._build_agent", new=_build):
        with _capture_v3_logs() as records:
            findings = await run_criteria(
                one_criterion, _model("x"), agent_model="stub"
            )

    assert findings == []
    log_text = "\n".join(r.getMessage() for r in records)
    assert "Story" in log_text
    assert "usage limit hit" in log_text


async def test_run_criteria_still_catches_generic_exception() -> None:
    """Stage 3: the defence-in-depth ``except Exception`` catch is
    preserved; a non-typed crash still logs + continues."""

    class _Agent:
        async def run(self, _prompt, **_kwargs):
            raise RuntimeError("unexpected boom")

    def _build(_criterion, _agent_model):
        return _Agent()

    one_criterion = [Criterion(name="Story", questions=["q?"], facets=[])]
    with patch("andamentum.whetstone.v3.review._build_agent", new=_build):
        with _capture_v3_logs() as records:
            findings = await run_criteria(
                one_criterion, _model("x"), agent_model="stub"
            )

    assert findings == []
    log_text = "\n".join(r.getMessage() for r in records)
    assert "Story" in log_text
    assert "crashed" in log_text
    assert "unexpected boom" in log_text


def test_verify_findings_drops_hallucinations_and_locates_real_ones() -> None:
    src = "The method is fast. The evaluation lacks a baseline comparison."
    model = _model(src)
    findings = [
        Finding(
            criterion="Evaluations",
            issue="no baseline",
            quote="lacks a baseline comparison",
            severity="major",
        ),
        Finding(
            criterion="Story",
            issue="invented",
            quote="we cured cancer",
            severity="major",
        ),
    ]
    kept = verify_findings(findings, model)
    assert len(kept) == 1
    assert kept[0].issue == "no baseline"
    assert kept[0].span is not None
    assert kept[0].span.section_id == "s1"
    assert src[kept[0].span.start : kept[0].span.end] == "lacks a baseline comparison"
