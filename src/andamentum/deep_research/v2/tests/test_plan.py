"""PLAN: one to four sub-questions; a failed plan is the question itself."""

from __future__ import annotations

import pytest

from ..plan import (
    MAX_SUBQUESTIONS,
    PLAN_INSTRUCTIONS,
    filters_for,
    plan,
    single_subquestion,
)
from ..types import Budget, SubQuestion
from .llm_stub import ScriptedLLM, make_asker

QUESTION = "How fast is the Apollo cache and what does SaltLog cost?"


def planned(*texts: str):
    return {"subquestions": [{"text": t, "kind": "topical"} for t in texts]}


async def test_a_question_that_asks_one_thing_stays_one_subquestion() -> None:
    model = ScriptedLLM(lambda _i, _p: planned("How fast is the Apollo cache?"))
    asker = make_asker(model=model)

    outcome = await plan("How fast is the Apollo cache?", asker=asker)

    assert [(s.id, s.text) for s in outcome.subquestions] == [
        ("Q1", "How fast is the Apollo cache?")
    ]
    assert outcome.degraded == [] and outcome.spent == ""
    assert asker.stats.step_llm_calls == {"Plan": 1}
    assert model.calls[0][1] == "QUESTION: How fast is the Apollo cache?"


async def test_a_question_with_parts_is_split_and_numbered() -> None:
    model = ScriptedLLM(
        lambda _i, _p: planned(
            "How fast is the Apollo cache?", "What does SaltLog cost?"
        )
    )
    outcome = await plan(QUESTION, asker=make_asker(model=model))
    assert [s.id for s in outcome.subquestions] == ["Q1", "Q2"]


async def test_the_kind_comes_from_the_planner_alone() -> None:
    model = ScriptedLLM(
        lambda _i, _p: {
            "subquestions": [
                {"text": "Which papers appeared since 2020?", "kind": "metadata"}
            ]
        }
    )
    outcome = await plan(QUESTION, asker=make_asker(model=model))
    assert outcome.subquestions[0].kind == "metadata"
    assert len(model.calls) == 1  # no decision-model step for the kind


async def test_the_instructions_name_the_cap_of_four() -> None:
    assert MAX_SUBQUESTIONS == 4
    assert "at most 4 sub-questions" in PLAN_INSTRUCTIONS
    assert "Do not answer them" in PLAN_INSTRUCTIONS


async def test_too_many_subquestions_is_a_degraded_single_plan() -> None:
    model = ScriptedLLM(lambda _i, _p: planned(*(f"Part {n}?" for n in range(5))))
    outcome = await plan(QUESTION, asker=make_asker(model=model))
    assert [s.text for s in outcome.subquestions] == [QUESTION]
    assert outcome.degraded and outcome.degraded[0].startswith("planning failed")


async def test_repeating_subquestions_are_asked_again() -> None:
    answers = iter(
        [
            planned("Same?", "same?"),
            planned("How fast is Apollo?", "What does SaltLog cost?"),
        ]
    )
    model = ScriptedLLM(lambda _i, _p: next(answers))
    outcome = await plan(QUESTION, asker=make_asker(model=model))
    assert [s.text for s in outcome.subquestions] == [
        "How fast is Apollo?",
        "What does SaltLog cost?",
    ]
    assert len(model.calls) == 2


async def test_a_failing_model_gives_the_question_as_its_one_subquestion() -> None:
    model = ScriptedLLM(lambda _i, _p: {"subquestions": "not a list"})
    outcome = await plan(f"  {QUESTION}  ", asker=make_asker(model=model))
    assert [s.text for s in outcome.subquestions] == [QUESTION]
    assert any("planning failed" in line for line in outcome.degraded)


async def test_a_spent_budget_gives_the_question_and_names_the_bound() -> None:
    model = ScriptedLLM(lambda _i, _p: planned("x?"))
    asker = make_asker(model=model, budget=Budget(max_llm_calls=1))
    asker.stats.llm_calls = 1
    outcome = await plan(QUESTION, asker=asker)
    assert outcome.spent == "max_llm_calls"
    assert outcome.subquestions == single_subquestion(QUESTION)
    assert model.calls == []


def sub(text: str, kind: str = "metadata") -> SubQuestion:
    return SubQuestion(id="Q1", text=text, kind=kind)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "year_from", "year_to"),
    [
        ("Papers since 2020 on caching", 2020, None),
        ("Papers after 2020 on caching", 2021, None),
        ("Papers before 2015 on caching", None, 2014),
        ("Papers until 2015 on caching", None, 2015),
        ("Papers from 2018-2022 on caching", 2018, 2022),
        ("Papers between 2022 and 2018", 2018, 2022),
        ("Papers published in 2021", 2021, 2021),
    ],
)
def test_filters_read_a_year_range_from_a_metadata_subquestion(
    text: str, year_from: int | None, year_to: int | None
) -> None:
    found = filters_for(sub(text))
    assert found is not None
    assert (found.year_from, found.year_to, found.open_access_only) == (
        year_from,
        year_to,
        False,
    )


def test_filters_read_open_access() -> None:
    found = filters_for(sub("Open-access papers on the Apollo cache"))
    assert found is not None and found.open_access_only


def test_other_kinds_and_unstated_limits_give_no_filters() -> None:
    assert filters_for(sub("Papers since 2020", kind="topical")) is None
    assert filters_for(sub("Papers on caching")) is None
