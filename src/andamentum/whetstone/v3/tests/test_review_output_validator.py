"""Tests for the verbatim-quote output validator (Stage 2).

The validator runs after the per-criterion agent produces a
``_CriterionFindings``. It locates each finding's quote in the source
and, on any unanchored quote, raises ``ModelRetry`` so pydantic-ai
sends the model back to fix the quotes verbatim. After
``_VALIDATOR_REQUOTE_ATTEMPTS`` re-quote rounds the validator stops
pushing and returns only the anchored subset — ``verify_findings``
remains the deterministic floor.

These tests exercise the validator function directly with a duck-typed
``RunContext`` so we don't need an LLM. The pyright cast at the
boundary mirrors the test fixture pattern already used in
``test_tools.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from pydantic_ai import ModelRetry, RunContext

from andamentum.whetstone.v3.model import DocumentModel, Section
from andamentum.whetstone.v3.review import (
    _CriterionFindings,
    _RawFinding,
    _validate_quotes_anchor,
)
from andamentum.whetstone.v3.tools import DocDeps


def _model() -> DocumentModel:
    """Two-section synthetic paper. Anchored quotes are verbatim slices
    of the source; anything paraphrased won't ``locate``."""
    source = (
        "Adam combines AdaGrad and RMSProp.\n\n"
        "The method exhibits invariance to diagonal rescaling.\n"
    )
    return DocumentModel(
        source=source,
        sections=[
            Section(
                id="abstract",
                title="Abstract",
                text="Adam combines AdaGrad and RMSProp.",
                start=0,
                end=34,
            ),
            Section(
                id="intro",
                title="Introduction",
                text="The method exhibits invariance to diagonal rescaling.",
                start=36,
                end=88,
            ),
        ],
    )


def _ctx(model: DocumentModel, *, retry: int) -> RunContext[DocDeps]:
    """Duck-typed RunContext exposing ``deps``, ``retry``, and
    ``partial_output``. The validator only reads those three attributes."""
    return cast(
        RunContext[DocDeps],
        SimpleNamespace(
            deps=DocDeps(document_model=model),
            retry=retry,
            partial_output=False,
        ),
    )


async def test_validator_passes_when_all_quotes_anchor() -> None:
    """Happy path: every quote is a verbatim slice of the source, so
    the validator returns the output unchanged on the first attempt."""
    model = _model()
    output = _CriterionFindings(
        findings=[
            _RawFinding(
                issue="claim",
                quote="Adam combines AdaGrad and RMSProp.",
                severity="moderate",
            ),
            _RawFinding(
                issue="property",
                quote="invariance to diagonal rescaling",
                severity="minor",
            ),
        ]
    )
    result = await _validate_quotes_anchor(_ctx(model, retry=0), output)
    assert len(result.findings) == 2
    assert {f.quote for f in result.findings} == {f.quote for f in output.findings}


async def test_validator_raises_modelretry_on_first_bad_quote() -> None:
    """First attempt with any unanchored quote: validator raises
    ``ModelRetry`` listing the bad quotes verbatim so the model can
    re-quote on its next turn."""
    model = _model()
    output = _CriterionFindings(
        findings=[
            _RawFinding(
                issue="real claim",
                quote="Adam combines AdaGrad and RMSProp.",
                severity="moderate",
            ),
            _RawFinding(
                issue="paraphrased",
                quote="Adam combines AdaGrad with RMSProp.",  # 'with' vs 'and'
                severity="moderate",
            ),
        ]
    )
    with pytest.raises(ModelRetry) as exc:
        await _validate_quotes_anchor(_ctx(model, retry=0), output)
    assert "verbatim" in str(exc.value).lower()
    assert "Adam combines AdaGrad with RMSProp." in str(exc.value)


async def test_validator_raises_modelretry_on_second_attempt_too() -> None:
    """Second re-quote attempt still pushes the model — we only stop at
    ``ctx.retry >= _VALIDATOR_REQUOTE_ATTEMPTS`` (currently 2)."""
    model = _model()
    output = _CriterionFindings(
        findings=[
            _RawFinding(
                issue="still bad",
                quote="not in the document",
                severity="moderate",
            ),
        ]
    )
    with pytest.raises(ModelRetry):
        await _validate_quotes_anchor(_ctx(model, retry=1), output)


async def test_validator_returns_anchored_only_on_third_attempt() -> None:
    """Attempts exhausted (``ctx.retry == 2``): validator stops raising
    and returns only the anchored findings. Equivalent to today's
    silent-drop floor, just preserved at the agent boundary so
    verify_findings still does its enrichment pass."""
    model = _model()
    output = _CriterionFindings(
        findings=[
            _RawFinding(
                issue="anchored",
                quote="Adam combines AdaGrad and RMSProp.",
                severity="major",
            ),
            _RawFinding(
                issue="bad",
                quote="we cured cancer",
                severity="major",
            ),
        ]
    )
    result = await _validate_quotes_anchor(_ctx(model, retry=2), output)
    assert len(result.findings) == 1
    assert result.findings[0].issue == "anchored"


async def test_validator_caps_preview_at_five_quotes() -> None:
    """Even when many quotes fail, the validator includes at most five
    in the retry prompt — bounds the message size."""
    model = _model()
    output = _CriterionFindings(
        findings=[
            _RawFinding(issue=f"bad{i}", quote=f"bogus quote number {i}", severity="minor")
            for i in range(8)
        ]
    )
    with pytest.raises(ModelRetry) as exc:
        await _validate_quotes_anchor(_ctx(model, retry=0), output)
    message = str(exc.value)
    # All eight bad quotes counted in the lead-in
    assert "8 quote(s)" in message
    # But only the first five appear verbatim in the preview
    assert "bogus quote number 0" in message
    assert "bogus quote number 4" in message
    assert "bogus quote number 5" not in message
    assert "bogus quote number 7" not in message


async def test_validator_is_safe_under_partial_output() -> None:
    """Defensive guard: if the run is streamed (we don't, but a future
    caller might) the validator returns the partial output as-is."""
    model = _model()
    output = _CriterionFindings(
        findings=[
            _RawFinding(issue="bad", quote="not in source", severity="minor"),
        ]
    )
    ctx = cast(
        RunContext[DocDeps],
        SimpleNamespace(
            deps=DocDeps(document_model=model),
            retry=0,
            partial_output=True,
        ),
    )
    result = await _validate_quotes_anchor(ctx, output)
    # No raise, no filtering — partial output passes through untouched
    assert result is output
