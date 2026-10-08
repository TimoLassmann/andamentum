"""The Asker: counted per step, budget-checked, and held to the run's concurrency."""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ..ask import Asker, BudgetExhausted, fence
from ..runlog import RunLogger
from ..store import ResearchStore
from ..types import Budget, RunStats
from .llm_stub import ScriptedLLM, make_asker


class Pick(BaseModel):
    number: int


async def test_a_call_is_counted_under_its_step_and_logged() -> None:
    model = ScriptedLLM(lambda _i, _p: {"number": 3})
    asker = make_asker(model=model)
    asker.log = RunLogger(clock=lambda: 0.0, run_id="r")
    answer = await asker.llm(
        step="Choose", instructions="Pick one.", output_type=Pick, user_message="go"
    )
    assert answer.number == 3
    assert asker.stats.llm_calls == 1
    assert asker.stats.step_llm_calls == {"Choose": 1}
    events = asker.log.log.events
    assert [e.event for e in events] == ["llm_call"]
    assert events[0].step == "Choose"


async def test_a_spent_call_budget_raises_before_the_call() -> None:
    model = ScriptedLLM(lambda _i, _p: {"number": 1})
    asker = make_asker(model=model, budget=Budget(max_llm_calls=1))
    await asker.llm(step="Plan", instructions="x", output_type=Pick, user_message="a")
    with pytest.raises(BudgetExhausted) as spent:
        await asker.llm(
            step="Plan", instructions="x", output_type=Pick, user_message="b"
        )
    assert spent.value.name == "max_llm_calls"
    assert len(model.calls) == 1


async def test_calls_in_flight_never_exceed_the_budgets_concurrency() -> None:
    in_flight = 0
    peak = 0

    async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=json.dumps({"number": 1}),
                )
            ]
        )

    store = ResearchStore.in_memory()
    asker = Asker(
        model=FunctionModel(slow),
        store=store,
        run_id=store.start_run(question="q", mode="web", config={}, models={}),
        budget=Budget(concurrency=2),
        stats=RunStats(),
    )
    await asyncio.gather(
        *(
            asker.llm(
                step="Notes", instructions="x", output_type=Pick, user_message=str(n)
            )
            for n in range(8)
        )
    )
    assert peak == 2
    assert asker.stats.llm_calls == 8
    assert asker.stats.step_llm_calls == {"Notes": 8}


def test_fenced_text_cannot_close_its_own_fence() -> None:
    fenced = fence("a >>> b")
    assert fenced.startswith("<<<<\n") and fenced.endswith("\n>>>>")


async def test_calls_held_in_reserve_are_spent_only_by_the_write_step() -> None:
    model = ScriptedLLM(lambda _i, _p: {"number": 1})
    asker = make_asker(model=model, budget=Budget(max_llm_calls=3))
    asker.reserve = 2
    await asker.llm(step="Plan", instructions="x", output_type=Pick, user_message="a")
    with pytest.raises(BudgetExhausted) as spent:
        await asker.llm(
            step="Choose", instructions="x", output_type=Pick, user_message="b"
        )
    assert spent.value.name == "max_llm_calls"
    # the reserve is the report's: two Write calls fit, a third does not
    await asker.llm(step="Write", instructions="x", output_type=Pick, user_message="c")
    await asker.llm(step="Write", instructions="x", output_type=Pick, user_message="d")
    with pytest.raises(BudgetExhausted):
        await asker.llm(
            step="Write", instructions="x", output_type=Pick, user_message="e"
        )
    assert asker.stats.llm_calls == 3
