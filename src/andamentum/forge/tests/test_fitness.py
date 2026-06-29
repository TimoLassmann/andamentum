"""The front fitness gate (dialect L9): a brief is built only if it is a function forge
can build *today*; everything else is refused at the door with a concrete reshape.

The whole forge graph runs design-only (dest=None) through a scripted sink whose fitness
head returns the scenario's rung — no live model, no container.

The corpus (``scenario_corpus.CORPUS``) is the *end-state* acceptance spec: rung-1 and
rung-2 both marked ``build``. This test derives the *current-phase* expectation from
``fitness.BUILDABLE_RUNGS`` instead of ``scenario.expected``, so it stays correct across
the Phase-2 flip (when ``stateful_function`` joins ``BUILDABLE_RUNGS`` the same test then
expects those scenarios to build, with no edit here).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from andamentum.core import AgentDefinition
from andamentum.forge import run_forge
from andamentum.forge.fitness import (
    BUILDABLE_RUNGS,
    assess_fitness,
    is_buildable,
    refusal_message,
)
from andamentum.forge.schemas import Fitness, ForgeWhy, NodeTyping
from andamentum.forge.spec import NodeKind

from .conftest import ScriptedSink
from .scenario_corpus import CORPUS


# --- a per-scenario sink: coherent rung-1 design script + a chosen fitness rung ---


class _FitnessScenarioSink(ScriptedSink):
    """A coherent rung-1 design script whose fitness head returns a scenario-chosen rung
    and reshape, so the gate's proceed/refuse branch is exercised end-to-end."""

    def __init__(self, *, rung: str, reshape: str) -> None:
        super().__init__(
            why=ForgeWhy(
                purpose="Do the thing the brief asks.",
                boundary_in="a natural-language request",
                boundary_out="a text answer",
            ),
            areas=["core"],
            jobs_by_area={"core": ["Parse the request.", "Answer the request."]},
            typings={
                "n1": NodeTyping(
                    kind=NodeKind.SPINE, consumes=["input"], produces=["parsed_request"]
                ),
                "n2": NodeTyping(
                    kind=NodeKind.HEAD, consumes=["parsed_request"], produces=["answer"]
                ),
            },
        )
        self._rung = rung
        self._reshape = reshape

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        if defn.name == "fitness":
            return Fitness(
                realizable_as_function=self._rung in ("function", "stateful_function"),
                rung=self._rung,  # type: ignore[arg-type]
                reason=f"stub verdict: {self._rung}",
                suggested_reshape=self._reshape,
            )
        return await super().run(defn, **kwargs)


# --- the corpus drives the gate -------------------------------------------------


@pytest.mark.parametrize("scn", CORPUS, ids=[s.brief for s in CORPUS])
async def test_corpus_fitness_gate(scn) -> None:
    reshape = f"reshape::{scn.brief}"
    sink = _FitnessScenarioSink(rung=scn.rung, reshape=reshape)

    if scn.rung in BUILDABLE_RUNGS:  # buildable in THIS phase → proceeds
        result = await run_forge(
            scn.brief, model="test", sink=sink
        )  # dest=None → design-only
        assert result.fitness is not None
        assert result.fitness.rung == scn.rung
        assert result.design_only
    else:  # not buildable yet → fail loud with the reshape reaching the user
        with pytest.raises(ValueError) as exc:
            await run_forge(scn.brief, model="test", sink=sink)
        assert reshape in str(exc.value)


# --- the policy, directly --------------------------------------------------------


def _fit(rung: str) -> Fitness:
    return Fitness(
        realizable_as_function=rung in ("function", "stateful_function"),
        rung=rung,  # type: ignore[arg-type]
        reason="r",
        suggested_reshape="do the single function instead",
    )


def test_functions_build_apps_agents_services_refused() -> None:
    assert is_buildable(_fit("function")) is True
    assert is_buildable(_fit("stateful_function")) is True  # rung-2 store now wired
    assert is_buildable(_fit("app")) is False
    assert is_buildable(_fit("agent")) is False
    assert is_buildable(_fit("service")) is False


def test_refusal_message_carries_reason_and_reshape() -> None:
    # Only an app / agent / service is refused now (a stateful function is buildable).
    for rung in ("app", "agent", "service"):
        msg = refusal_message(_fit(rung))
        assert "do the single function instead" in msg  # the concrete reshape
        assert rung in msg


# --- the worker, directly --------------------------------------------------------


class _OneShotFitnessSink:
    """Minimal AgentSink returning a fixed Fitness — the worker passthrough under test."""

    def __init__(self, fitness: Fitness) -> None:
        self._fitness = fitness

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        assert defn.name == "fitness"
        return self._fitness


async def test_assess_fitness_returns_the_verdict() -> None:
    why = ForgeWhy(purpose="p", boundary_in="i", boundary_out="o")
    fit = _fit("function")
    out = await assess_fitness(why, sink=_OneShotFitnessSink(fit))
    assert out is fit


# --- the gate fails loud on a non-function, via run_forge -----------------------


async def test_stateful_function_now_passes_the_gate() -> None:
    # rung-2 is buildable now: the gate lets it through to design (the store is wired).
    sink = _FitnessScenarioSink(rung="stateful_function", reshape="n/a")
    result = await run_forge("remember and update a value", model="test", sink=sink)
    assert result.fitness is not None and result.fitness.rung == "stateful_function"
    assert result.design_only


async def test_app_is_still_refused_with_reshape() -> None:
    sink = _FitnessScenarioSink(rung="app", reshape="reshape::APP")
    with pytest.raises(ValueError) as exc:
        await run_forge("manage my whole reading list", model="test", sink=sink)
    assert "reshape::APP" in str(exc.value)
