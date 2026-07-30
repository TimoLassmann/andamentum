"""Boundary validation on the public entry points — fully offline.

A judge is only useful if its failures are legible. These pin that a
degenerate input is rejected AT THE ENTRY POINT with a message naming what
the caller got wrong, rather than sailing through to die several frames down
in the pure-math layer with ``max() iterable argument is empty``.

They also pin that rejection happens BEFORE any model call — a caller error
must never cost a token, and must never be quietly turned into a default
verdict.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from andamentum.llm_judge import Criterion, judge_compare, judge_score
from andamentum.llm_judge import elicit as elicit_module

CRITERIA = [Criterion(name="correctness", description="Is it correct?")]


@pytest.fixture
def no_model_calls(monkeypatch):
    """Any model call at all fails the test."""
    boom = AsyncMock(side_effect=AssertionError("the model must not be called"))
    monkeypatch.setattr(elicit_module, "elicit_criterion_score", boom)
    monkeypatch.setattr(elicit_module, "elicit_pairwise", boom)
    return boom


# ── empty model sequence ─────────────────────────────────────────────────


async def test_score_rejects_an_empty_model_sequence(no_model_calls):
    with pytest.raises(ValueError, match="at least one judge"):
        await judge_score("x", criteria=CRITERIA, model=[])
    no_model_calls.assert_not_awaited()


async def test_compare_rejects_an_empty_model_sequence(no_model_calls):
    with pytest.raises(ValueError, match="at least one judge"):
        await judge_compare("a", "b", criteria=CRITERIA, model=[])
    no_model_calls.assert_not_awaited()


# ── blank model ids ──────────────────────────────────────────────────────


async def test_score_rejects_a_blank_model_id(no_model_calls):
    with pytest.raises(ValueError, match="non-empty strings"):
        await judge_score("x", criteria=CRITERIA, model="")
    no_model_calls.assert_not_awaited()


async def test_score_rejects_a_blank_model_id_inside_a_panel(no_model_calls):
    with pytest.raises(ValueError, match="non-empty strings"):
        await judge_score("x", criteria=CRITERIA, model=["good:model", "   "])
    no_model_calls.assert_not_awaited()


async def test_compare_rejects_a_blank_model_id(no_model_calls):
    with pytest.raises(ValueError, match="non-empty strings"):
        await judge_compare("a", "b", criteria=CRITERIA, model=[""])
    no_model_calls.assert_not_awaited()


# ── empty criteria ───────────────────────────────────────────────────────


async def test_score_rejects_an_empty_criteria_list(no_model_calls):
    with pytest.raises(ValueError, match="at least one Criterion"):
        await judge_score("x", criteria=[], model="m")
    no_model_calls.assert_not_awaited()


async def test_compare_rejects_an_empty_criteria_list(no_model_calls):
    with pytest.raises(ValueError, match="at least one Criterion"):
        await judge_compare("a", "b", criteria=[], model="m")
    no_model_calls.assert_not_awaited()


async def test_the_error_points_the_caller_at_criteria_none_for_the_defaults(
    no_model_calls,
):
    """`criteria=[]` is almost always a caller meaning `criteria=None` — the
    message must say so, not just complain."""
    with pytest.raises(ValueError, match="criteria=None"):
        await judge_score("x", criteria=[], model="m")


# ── the valid boundary cases stay valid ──────────────────────────────────


async def test_a_one_element_model_list_is_still_a_legal_panel(monkeypatch):
    """A single-model list is a deliberate panel request, NOT a degenerate
    input — the validation must not reject it."""

    async def fake_elicit(output, criterion, context, *, model, temperature):
        assert temperature == elicit_module.PANEL_SAMPLING_TEMPERATURE
        return elicit_module._CriterionDist(
            reasoning="r", meets=70, partial=20, fails=10
        )

    monkeypatch.setattr(
        elicit_module, "elicit_criterion_score", AsyncMock(side_effect=fake_elicit)
    )
    result = await judge_score("x", criteria=CRITERIA, model=["m"])
    assert result.overall == "meets"
    assert result.judges is not None
    assert len(result.judges) == 1
