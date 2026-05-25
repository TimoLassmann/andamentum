"""Unified criteria-input surface for v3.

Tests Phase C's collapsed custom + guidelines path:
``run_review_v3(criteria=..., guidelines_text=...)``. Both are mutually
exclusive routes to the same V3Deps.criteria slot; the explicit list
wins over the extracted prose wins over the document-type default.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from andamentum.whetstone.v3 import (
    Criterion,
    criterion_set_for,
    run_review_v3,
)
from andamentum.whetstone.v3.criteria import GENERAL


CLEAN_MD = "# Methods\n\nWe trained a small transformer on synthetic data.\n"


async def _captured_run(captured: dict):
    """Mock helper: capture the deps passed to the graph for assertion."""

    async def _run(_node, *, state, deps):
        captured["criteria"] = list(deps.criteria)

        class _Wrapper:
            output = type("Stub", (), {})()

        return _Wrapper()

    return _run


async def test_explicit_criteria_kwarg_replaces_document_type_default() -> None:
    """When the caller passes `criteria=[...]`, the active criterion set
    is exactly that list — not the document-type default."""
    custom = [Criterion(name="Originality", questions=["Is this novel?"])]
    captured: dict = {}
    with patch(
        "andamentum.whetstone.v3.graph.review_graph_v3.run",
        new=AsyncMock(side_effect=await _captured_run(captured)),
    ):
        await run_review_v3(
            CLEAN_MD,
            model="stub",
            document_type="academic",
            criteria=custom,
        )
    assert captured["criteria"] == custom
    # Document-type default would have been SPECS (5 academic criteria).
    assert len(captured["criteria"]) == 1


async def test_guidelines_text_invokes_extractor() -> None:
    """When the caller passes `guidelines_text=...`, the extractor is
    invoked once with that prose and its output becomes the active
    criterion set."""
    extracted = [
        Criterion(name="Reproducibility", questions=["Is the code available?"]),
        Criterion(name="Statistics", questions=["Are p-values reported?"]),
    ]
    captured: dict = {}
    with (
        patch(
            "andamentum.whetstone.v3.graph.review_graph_v3.run",
            new=AsyncMock(side_effect=await _captured_run(captured)),
        ),
        patch(
            "andamentum.whetstone.v3.extract_criteria.extract_criteria_from_guidelines",
            new=AsyncMock(return_value=extracted),
        ) as mock_extract,
    ):
        await run_review_v3(
            CLEAN_MD,
            model="stub",
            document_type="academic",
            guidelines_text="Authors must release code.\nReport all p-values.",
        )
    mock_extract.assert_awaited_once()
    assert captured["criteria"] == extracted


async def test_both_criteria_and_guidelines_raises_value_error() -> None:
    """Passing both kwargs is ambiguous; raise ValueError before doing
    any work (no LLM call, no graph run)."""
    with (
        patch(
            "andamentum.whetstone.v3.graph.review_graph_v3.run",
            new=AsyncMock(),
        ) as mock_run,
        patch(
            "andamentum.whetstone.v3.extract_criteria.extract_criteria_from_guidelines",
            new=AsyncMock(),
        ) as mock_extract,
        patch(
            "andamentum.whetstone._document_type.classify", new=AsyncMock()
        ) as mock_classify,
    ):
        with pytest.raises(ValueError, match="criteria.*guidelines_text"):
            await run_review_v3(
                CLEAN_MD,
                model="stub",
                document_type="academic",
                criteria=[Criterion(name="X", questions=["?"])],
                guidelines_text="prose",
            )
    mock_run.assert_not_awaited()
    mock_extract.assert_not_awaited()
    mock_classify.assert_not_awaited()


async def test_neither_falls_back_to_document_type_default() -> None:
    """Without either kwarg, the existing document-type → criterion-set
    routing applies (criterion_set_for)."""
    captured: dict = {}
    with patch(
        "andamentum.whetstone.v3.graph.review_graph_v3.run",
        new=AsyncMock(side_effect=await _captured_run(captured)),
    ):
        await run_review_v3(CLEAN_MD, model="stub", document_type="general")
    assert captured["criteria"] == criterion_set_for("general")
    assert captured["criteria"] == GENERAL


async def test_classifier_skipped_when_caller_supplies_criteria() -> None:
    """The classifier is only consumed when we fall back to the
    document-type default. When the caller passes `criteria=...` or
    `guidelines_text=...`, we never need a document_type, so the
    classifier MUST NOT fire (it's an LLM call)."""
    custom = [Criterion(name="X", questions=["?"])]
    with (
        patch(
            "andamentum.whetstone.v3.graph.review_graph_v3.run",
            new=AsyncMock(side_effect=await _captured_run({})),
        ),
        patch(
            "andamentum.whetstone._document_type.classify", new=AsyncMock()
        ) as mock_classify,
    ):
        await run_review_v3(
            CLEAN_MD,
            model="stub",
            document_type="auto",  # would normally trigger classify
            criteria=custom,
        )
    mock_classify.assert_not_awaited()


async def test_extract_criteria_caps_at_max_criteria() -> None:
    """The extractor caps its output at max_criteria, even when the LLM
    returns more."""
    from andamentum.whetstone.v3.extract_criteria import (
        _ExtractedCriterion,
        _ExtractionResult,
        extract_criteria_from_guidelines,
    )

    # Stub agent: returns 12 criteria.
    over_count = [
        _ExtractedCriterion(name=f"C{i}", questions=[f"Question {i}?"])
        for i in range(12)
    ]
    stub_result = _ExtractionResult(criteria=over_count)

    class _FakeAgent:
        async def run(self, _prompt: str):  # noqa: ARG002
            class _W:
                output = stub_result

            return _W()

    with patch(
        "andamentum.whetstone.v3.extract_criteria.build_pydantic_ai_agent",
        return_value=_FakeAgent(),
    ):
        with patch(
            "andamentum.whetstone.v3.extract_criteria.resolve_model",
            return_value="stub",
        ):
            out = await extract_criteria_from_guidelines(
                "prose", model="stub", max_criteria=8
            )
    assert len(out) == 8
    # First 8 in order preserved.
    assert [c.name for c in out] == [f"C{i}" for i in range(8)]


async def test_extract_criteria_empty_output_raises_runtime_error() -> None:
    """The extractor refuses to silently return [] — caller must
    explicitly fall back (per the no-silent-failures rule)."""
    from andamentum.whetstone.v3.extract_criteria import (
        _ExtractionResult,
        extract_criteria_from_guidelines,
    )

    stub_result = _ExtractionResult(criteria=[])

    class _FakeAgent:
        async def run(self, _prompt: str):  # noqa: ARG002
            class _W:
                output = stub_result

            return _W()

    with patch(
        "andamentum.whetstone.v3.extract_criteria.build_pydantic_ai_agent",
        return_value=_FakeAgent(),
    ):
        with patch(
            "andamentum.whetstone.v3.extract_criteria.resolve_model",
            return_value="stub",
        ):
            with pytest.raises(RuntimeError, match="no criteria"):
                await extract_criteria_from_guidelines("prose", model="stub")


async def test_extract_criteria_empty_prose_raises_value_error() -> None:
    """Empty / whitespace-only prose is a caller bug — raise immediately
    (no LLM call)."""
    from andamentum.whetstone.v3.extract_criteria import (
        extract_criteria_from_guidelines,
    )

    with patch(
        "andamentum.whetstone.v3.extract_criteria.build_pydantic_ai_agent"
    ) as mock_agent:
        with pytest.raises(ValueError, match="prose is empty"):
            await extract_criteria_from_guidelines("   \n\n  ", model="stub")
    mock_agent.assert_not_called()
