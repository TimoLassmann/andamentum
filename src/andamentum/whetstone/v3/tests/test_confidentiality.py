"""Confidentiality-marker tripwire integration for v3.

The tripwire itself (the marker list, the regex matching, the
exception class) is exercised by the v2 test suite in
`tests/test_confidentiality.py`. These tests cover the v3-specific
wiring: that `run_review_v3` and `review_document_v3` invoke the
tripwire as the very first step (before the classifier, before any
graph node) and that `confirm_own_draft=True` bypasses it cleanly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from andamentum.whetstone._confidentiality import ConfidentialityMarkerError
from andamentum.whetstone.v3.graph import (
    review_document_v3,
    run_review_v3,
)


CLEAN_MD = "# Methods\n\nWe trained a small transformer on synthetic data.\n"
MARKER_MD = "# Methods\n\nManuscript ID: NEJM-2026-001\n\nDetails follow below.\n"


class _StubResult:
    """Sentinel returned by the mocked graph so we can assert it flowed
    through without exercising the real pipeline."""


async def _mock_run(_node, *, state, deps):  # noqa: ARG001 - signature mirrors pydantic-graph
    class _Wrapper:
        output = _StubResult()

    return _Wrapper()


async def test_refuses_markered_input_by_default() -> None:
    """A draft containing a confidentiality marker raises
    ConfidentialityMarkerError on default invocation; the error names
    the matched marker so the user knows what triggered the refusal."""
    with pytest.raises(ConfidentialityMarkerError) as excinfo:
        await run_review_v3(MARKER_MD, model="stub", document_type="academic")
    assert "Manuscript ID:" in str(excinfo.value)
    assert excinfo.value.marker == "Manuscript ID:"


async def test_confirm_own_draft_bypasses_tripwire() -> None:
    """confirm_own_draft=True is the explicit attestation that this is
    the user's own draft; even markered text proceeds into the graph."""
    with patch(
        "andamentum.whetstone.v3.graph.review_graph_v3.run",
        new=AsyncMock(side_effect=_mock_run),
    ) as mock_run:
        result = await run_review_v3(
            MARKER_MD,
            model="stub",
            document_type="academic",
            confirm_own_draft=True,
        )
    assert isinstance(result, _StubResult)
    mock_run.assert_awaited_once()


async def test_clean_input_passes_through_without_flag() -> None:
    """Markdown with no confidentiality markers proceeds normally —
    the user does NOT need to set confirm_own_draft for clean input."""
    with patch(
        "andamentum.whetstone.v3.graph.review_graph_v3.run",
        new=AsyncMock(side_effect=_mock_run),
    ) as mock_run:
        result = await run_review_v3(CLEAN_MD, model="stub", document_type="academic")
    assert isinstance(result, _StubResult)
    mock_run.assert_awaited_once()


async def test_tripwire_fires_before_any_llm_call() -> None:
    """The refusal happens BEFORE the graph runs AND BEFORE the
    document-type classifier — so no LLM ever sees confidential text."""
    with (
        patch(
            "andamentum.whetstone.v3.graph.review_graph_v3.run",
            new=AsyncMock(side_effect=_mock_run),
        ) as mock_run,
        patch(
            "andamentum.whetstone._document_type.classify", new=AsyncMock()
        ) as mock_classify,
    ):
        with pytest.raises(ConfidentialityMarkerError):
            # document_type="auto" would normally invoke the classifier
            await run_review_v3(MARKER_MD, model="stub", document_type="auto")

    mock_run.assert_not_awaited()
    mock_classify.assert_not_awaited()


async def test_review_document_v3_also_tripwires(tmp_path) -> None:
    """The file-entry wrapper (review_document_v3) applies the same
    check after the harvest step — covers the second public entry."""
    draft = tmp_path / "draft.md"
    draft.write_text(MARKER_MD)
    with pytest.raises(ConfidentialityMarkerError):
        await review_document_v3(str(draft), model="stub", document_type="academic")
