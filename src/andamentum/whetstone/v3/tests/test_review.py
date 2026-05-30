"""Tests for the criterion set, generic review (mocked), and verify-findings."""

from __future__ import annotations

import logging
import types
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded

from andamentum.whetstone.v3.criteria import (
    CREATIVE,
    ESSAY,
    EXTERNAL_COMMS,
    GENERAL,
    SPECS,
    TUTORIAL,
    Criterion,
    criterion_set_for,
)
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
    assert [c.name for c in SPECS] == [
        "Story",
        "Presentation",
        "Evaluations",
        "Correctness",
        "Significance",
    ]
    assert all(c.questions for c in SPECS)


def test_external_comms_set_is_routed_for_external_communication() -> None:
    """Issue 1: blog posts, articles, op-eds get their own criteria
    (Hook/Argument/Evidence/Voice/Clarity), not SPECS — applying
    Evaluations and Correctness to a non-academic piece produces
    forced findings that don't serve the author."""
    assert criterion_set_for("external_communication") is EXTERNAL_COMMS
    assert [c.name for c in EXTERNAL_COMMS] == [
        "Hook",
        "Argument",
        "Evidence",
        "Voice",
        "Clarity",
    ]
    assert all(c.questions for c in EXTERNAL_COMMS)


def test_general_set_is_routed_for_general() -> None:
    """Issue 1: notes, drafts, books, technical documentation get
    GENERAL (Purpose/Structure/Completeness/Clarity)."""
    assert criterion_set_for("general") is GENERAL
    assert [c.name for c in GENERAL] == [
        "Purpose",
        "Structure",
        "Completeness",
        "Clarity",
    ]
    assert all(c.questions for c in GENERAL)


def test_essay_set_is_routed_for_essay() -> None:
    """Phase A: personal/narrative/opinion essays get ESSAY
    (Thesis/Narrative arc/Specificity/Voice/Fresh observation) — not
    the academic SPECS or the generic GENERAL."""
    assert criterion_set_for("essay") is ESSAY
    assert [c.name for c in ESSAY] == [
        "Thesis",
        "Narrative arc",
        "Specificity",
        "Voice",
        "Fresh observation",
    ]
    assert all(c.questions for c in ESSAY)


def test_tutorial_set_is_routed_for_tutorial() -> None:
    """Phase A: how-tos / walkthroughs / cookbooks get TUTORIAL
    (Goal/Prerequisites/Step ordering/Correctness/Completeness) —
    the reader is trying to accomplish a task."""
    assert criterion_set_for("tutorial") is TUTORIAL
    assert [c.name for c in TUTORIAL] == [
        "Goal",
        "Prerequisites",
        "Step ordering",
        "Correctness",
        "Completeness",
    ]
    assert all(c.questions for c in TUTORIAL)


def test_creative_set_is_routed_for_creative() -> None:
    """Phase A: short fiction / memoir / narrative non-fiction get
    CREATIVE (Premise/Character & voice/Scene & sensory grounding/
    Tension/Prose craft) — story craft is the substance."""
    assert criterion_set_for("creative") is CREATIVE
    assert [c.name for c in CREATIVE] == [
        "Premise",
        "Character & voice",
        "Scene & sensory grounding",
        "Tension",
        "Prose craft",
    ]
    assert all(c.questions for c in CREATIVE)


def test_unknown_document_type_falls_back_to_general() -> None:
    """Issue 1: the previous behaviour silently fell back to SPECS for
    unknown types — meaning Evaluations and Correctness would run on
    essays. GENERAL is the safe neutral default."""
    assert criterion_set_for("anything-unknown") is GENERAL
    assert criterion_set_for("") is GENERAL


def test_six_sets_are_disjoint() -> None:
    """Sanity check that the six criterion sets are distinct objects —
    not aliases of each other (a copy/paste bug would silently regress
    routing)."""
    all_sets = [SPECS, EXTERNAL_COMMS, ESSAY, TUTORIAL, CREATIVE, GENERAL]
    # Every pair is a distinct object
    for i, a in enumerate(all_sets):
        for j, b in enumerate(all_sets):
            if i != j:
                assert a is not b, (
                    f"criterion sets at positions {i} and {j} are the same "
                    f"object — copy/paste bug in v3/criteria.py"
                )


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
        def output_validator(self, fn):  # absorbs the lock-and-refine validator
            return fn

        async def run(self, _prompt, **_kwargs):
            # Accept deps/usage_limits kwargs from review_criterion silently.
            return types.SimpleNamespace(output=out)

    def _build(_criterion, _agent_model):
        return _Agent()

    with patch("andamentum.whetstone.v3.review._build_agent", new=_build):
        findings, failed = await run_criteria(SPECS, _model("x"), agent_model="stub")
    # one finding per criterion, each tagged with its criterion name
    assert {f.criterion for f in findings} == {c.name for c in SPECS}
    assert failed == []  # nothing crashed


async def test_run_criteria_logs_unexpected_model_behaviour_body() -> None:
    """Stage 3: UnexpectedModelBehavior.body should land in the log line.

    Today this exception is caught by the trailing ``except Exception``
    and logged with only ``str(exc)`` — the upstream response body
    (e.g. Ollama's null-content HTTP 400 payload) is lost. With the
    typed cascade in place we log the first 500 chars of ``body``.
    """

    class _Agent:
        def output_validator(self, fn):  # absorbs the lock-and-refine validator
            return fn

        async def run(self, _prompt, **_kwargs):
            raise UnexpectedModelBehavior(
                "stage-3 boom", body="upstream provider error payload"
            )

    def _build(_criterion, _agent_model):
        return _Agent()

    one_criterion = [Criterion(name="Story", questions=["q?"], facets=[])]
    with patch("andamentum.whetstone.v3.review._build_agent", new=_build):
        with _capture_v3_logs() as records:
            findings, failed = await run_criteria(
                one_criterion, _model("x"), agent_model="stub"
            )

    assert findings == []  # the criterion gracefully skipped
    assert failed == ["Story"]  # ...but the failure is recorded, not silent
    log_text = "\n".join(r.getMessage() for r in records)
    assert "Story" in log_text
    assert "model behaviour error" in log_text
    assert "upstream provider error payload" in log_text


async def test_run_criteria_logs_usage_limit_exceeded() -> None:
    """Stage 3: UsageLimitExceeded gets its own log line, distinct from
    generic crashes."""

    class _Agent:
        def output_validator(self, fn):  # absorbs the lock-and-refine validator
            return fn

        async def run(self, _prompt, **_kwargs):
            raise UsageLimitExceeded("request_limit (18) exceeded")

    def _build(_criterion, _agent_model):
        return _Agent()

    one_criterion = [Criterion(name="Story", questions=["q?"], facets=[])]
    with patch("andamentum.whetstone.v3.review._build_agent", new=_build):
        with _capture_v3_logs() as records:
            findings, failed = await run_criteria(
                one_criterion, _model("x"), agent_model="stub"
            )

    assert findings == []
    assert failed == ["Story"]
    log_text = "\n".join(r.getMessage() for r in records)
    assert "Story" in log_text
    assert "usage limit hit" in log_text


async def test_run_criteria_still_catches_generic_exception() -> None:
    """Stage 3: the defence-in-depth ``except Exception`` catch is
    preserved; a non-typed crash still logs + continues."""

    class _Agent:
        def output_validator(self, fn):  # absorbs the lock-and-refine validator
            return fn

        async def run(self, _prompt, **_kwargs):
            raise RuntimeError("unexpected boom")

    def _build(_criterion, _agent_model):
        return _Agent()

    one_criterion = [Criterion(name="Story", questions=["q?"], facets=[])]
    with patch("andamentum.whetstone.v3.review._build_agent", new=_build):
        with _capture_v3_logs() as records:
            findings, failed = await run_criteria(
                one_criterion, _model("x"), agent_model="stub"
            )

    assert findings == []
    assert failed == ["Story"]
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
