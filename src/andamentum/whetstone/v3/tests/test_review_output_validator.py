"""Tests for the lock-and-refine output validator (Issue 2).

The validator is now built by ``make_anchor_validator(source, output_class)``
which returns a closure that:

- Accumulates anchored findings across retry attempts inside one
  ``agent.run`` call (closure state on a dict ``locked: quote → finding``).
- Tells the model which findings are locked and asks it to refine ONLY
  the unanchored ones — instead of regenerating the whole output and
  potentially breaking quotes that were previously fine.
- On retry exhaustion, returns the accumulated locked set (no regression
  from earlier good attempts).

The closure must reset per criterion (no leakage across criteria); each
test below builds a fresh validator and exercises it with a duck-typed
``RunContext`` stub mimicking pydantic-ai's retry-loop semantics.
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
    make_anchor_validator,
)
from andamentum.whetstone.v3.tools import DocDeps


def _model() -> DocumentModel:
    source = (
        "Adam combines AdaGrad and RMSProp.\n\n"
        "The method exhibits invariance to diagonal rescaling.\n"
    )
    return DocumentModel(
        source=source,
        sections=[
            Section(id="abstract", title="A", text=source[:34], start=0, end=34),
            Section(id="intro", title="I", text=source[36:88], start=36, end=88),
        ],
    )


def _ctx(*, retry: int, partial_output: bool = False) -> RunContext[DocDeps]:
    """Duck-typed RunContext stub exposing only the three attributes the
    validator reads — ``retry``, ``partial_output``, and (defensively) the
    rest of pydantic-ai's RunContext shape is irrelevant here because the
    new validator closes over ``source`` instead of going through deps."""
    return cast(
        RunContext[DocDeps],
        SimpleNamespace(retry=retry, partial_output=partial_output),
    )


# ── happy path ─────────────────────────────────────────────────────────


async def test_all_quotes_anchor_first_attempt_no_retry() -> None:
    model = _model()
    validator = make_anchor_validator(model.source, _CriterionFindings)
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
    result = await validator(_ctx(retry=0), output)
    assert len(result.findings) == 2
    assert {f.quote for f in result.findings} == {f.quote for f in output.findings}


# ── lock-and-refine: anchored findings persist across retries ──────────


async def test_anchored_findings_persist_when_retry_breaks_them() -> None:
    """The core bug the lock-and-refine fix addresses. Attempt 0:
    model returns 2 anchored findings + 1 unanchored. Validator raises
    ModelRetry. Attempt 1: model paraphrases the previously-good ones
    (oops) and fixes the bad one. Without lock-and-refine the validator
    would see "1 anchored, 2 unanchored" and either ask again or accept
    only the 1 — net signal loss. With lock-and-refine: the 2 originally
    anchored survive."""
    model = _model()
    validator = make_anchor_validator(model.source, _CriterionFindings)

    # Attempt 0: model returns 2 good + 1 bad
    attempt_0 = _CriterionFindings(
        findings=[
            _RawFinding(
                issue="a",
                quote="Adam combines AdaGrad and RMSProp.",
                severity="major",
            ),
            _RawFinding(
                issue="b",
                quote="invariance to diagonal rescaling",
                severity="moderate",
            ),
            _RawFinding(
                issue="c",
                quote="paraphrased not in source",
                severity="minor",
            ),
        ]
    )
    with pytest.raises(ModelRetry) as exc:
        await validator(_ctx(retry=0), attempt_0)
    # Retry message tells model 2 are locked, 1 needs fixing
    assert "2 already verified verbatim" in str(exc.value)
    assert "1 quote(s) are not present verbatim" in str(exc.value)
    assert "paraphrased not in source" in str(exc.value)

    # Attempt 1: model fixes the bad one BUT paraphrases the good ones
    # (the failure mode lock-and-refine is meant to absorb)
    attempt_1 = _CriterionFindings(
        findings=[
            _RawFinding(
                issue="a-paraphrased",
                quote="Adam combines AdaGrad with RMSProp.",  # 'with' vs 'and'
                severity="major",
            ),
            _RawFinding(
                issue="b-paraphrased",
                quote="invariance to a diagonal rescaling",  # extra 'a'
                severity="moderate",
            ),
            _RawFinding(
                issue="c-fixed",
                quote="The method exhibits invariance to diagonal rescaling.",
                severity="minor",
            ),
        ]
    )
    # No more bad quotes (the previously-paraphrased ones aren't bad —
    # they're not in `locked` so they appear as unanchored, but locked
    # has 2 already + the fixed one. Wait — let me reread).
    # Actually attempt_1's "Adam combines AdaGrad with RMSProp." won't
    # anchor (it's paraphrased), and won't already be in locked. So it
    # appears as unanchored. The original 2 are STILL in locked from
    # attempt 0. So we expect: 3 locked (2 from before + 1 newly fixed),
    # 2 unanchored (the paraphrased variants), retry < max → another
    # ModelRetry.
    with pytest.raises(ModelRetry) as exc:
        await validator(_ctx(retry=1), attempt_1)
    assert "3 already verified verbatim" in str(exc.value)

    # Attempt 2: model gives up on the paraphrased ones (returns empty)
    # The validator should exhaust attempts and return the 3 locked
    final = await validator(_ctx(retry=2), _CriterionFindings(findings=[]))
    assert len(final.findings) == 3
    # All three locked findings survive — the 2 from attempt 0 + the 1
    # successfully fixed in attempt 1
    quotes = {f.quote for f in final.findings}
    assert "Adam combines AdaGrad and RMSProp." in quotes
    assert "invariance to diagonal rescaling" in quotes
    assert "The method exhibits invariance to diagonal rescaling." in quotes


# ── retry budget ───────────────────────────────────────────────────────


async def test_validator_raises_on_first_attempt_with_bad_quote() -> None:
    model = _model()
    validator = make_anchor_validator(model.source, _CriterionFindings)
    output = _CriterionFindings(
        findings=[
            _RawFinding(issue="bad", quote="not in source", severity="minor"),
        ]
    )
    with pytest.raises(ModelRetry) as exc:
        await validator(_ctx(retry=0), output)
    assert "not in source" in str(exc.value)
    # No locked findings yet — retry message reflects that
    assert "0 already verified verbatim" in str(exc.value)


async def test_validator_raises_on_second_attempt_too() -> None:
    """retry=1 still pushes the model (max_attempts default = 2)."""
    model = _model()
    validator = make_anchor_validator(model.source, _CriterionFindings)
    output = _CriterionFindings(
        findings=[_RawFinding(issue="still bad", quote="not in source", severity="minor")]
    )
    with pytest.raises(ModelRetry):
        await validator(_ctx(retry=1), output)


async def test_validator_returns_locked_on_third_attempt() -> None:
    """retry=2 (== max_attempts) stops raising and returns the lock."""
    model = _model()
    validator = make_anchor_validator(model.source, _CriterionFindings)

    # Attempt 0: 1 anchored, 1 unanchored → raises
    with pytest.raises(ModelRetry):
        await validator(
            _ctx(retry=0),
            _CriterionFindings(
                findings=[
                    _RawFinding(
                        issue="good",
                        quote="Adam combines AdaGrad and RMSProp.",
                        severity="major",
                    ),
                    _RawFinding(issue="bad", quote="not in source", severity="minor"),
                ]
            ),
        )
    # Attempts exhausted at retry=2 — return the 1 locked finding,
    # drop the bad one silently (same floor as old behaviour)
    final = await validator(
        _ctx(retry=2),
        _CriterionFindings(
            findings=[_RawFinding(issue="bad", quote="still not in source", severity="minor")]
        ),
    )
    assert len(final.findings) == 1
    assert final.findings[0].quote == "Adam combines AdaGrad and RMSProp."


# ── edge cases ─────────────────────────────────────────────────────────


async def test_partial_output_returns_input_untouched() -> None:
    model = _model()
    validator = make_anchor_validator(model.source, _CriterionFindings)
    output = _CriterionFindings(findings=[])
    result = await validator(_ctx(retry=0, partial_output=True), output)
    assert result is output


async def test_locked_set_dedupes_same_quote_emitted_twice() -> None:
    """The lock is keyed by quote text. If the model re-emits an
    already-locked finding, the second one doesn't double-count."""
    model = _model()
    validator = make_anchor_validator(model.source, _CriterionFindings)
    duplicate = _CriterionFindings(
        findings=[
            _RawFinding(
                issue="a", quote="Adam combines AdaGrad and RMSProp.", severity="major"
            ),
            _RawFinding(
                issue="a-again",
                quote="Adam combines AdaGrad and RMSProp.",  # same quote
                severity="minor",
            ),
        ]
    )
    result = await validator(_ctx(retry=0), duplicate)
    # Only ONE locked entry for that quote
    assert len(result.findings) == 1


async def test_preview_capped_at_five_quotes() -> None:
    model = _model()
    validator = make_anchor_validator(model.source, _CriterionFindings)
    output = _CriterionFindings(
        findings=[
            _RawFinding(issue=f"bad{i}", quote=f"bogus {i}", severity="minor")
            for i in range(8)
        ]
    )
    with pytest.raises(ModelRetry) as exc:
        await validator(_ctx(retry=0), output)
    msg = str(exc.value)
    # Lead-in counts all 8
    assert "8 quote(s) are not present verbatim" in msg
    # Only first 5 appear in the preview block
    assert "bogus 0" in msg
    assert "bogus 4" in msg
    assert "bogus 5" not in msg


async def test_validator_state_is_per_factory_call() -> None:
    """Two validators built from the same source have isolated locked
    sets — one criterion's accumulation must not leak into another's."""
    model = _model()
    v1 = make_anchor_validator(model.source, _CriterionFindings)
    v2 = make_anchor_validator(model.source, _CriterionFindings)

    good = _CriterionFindings(
        findings=[
            _RawFinding(
                issue="x",
                quote="Adam combines AdaGrad and RMSProp.",
                severity="major",
            )
        ]
    )
    await v1(_ctx(retry=0), good)
    # v2 hasn't seen anything — its locked set is empty
    final = await v2(_ctx(retry=2), _CriterionFindings(findings=[]))
    assert len(final.findings) == 0
