"""Offline self-tests for the runner — drives the full run_case path with no model.

A stub ``ScriptedSink`` answers every forge design head, so ``run_case`` exercises the real
forge graph (understand→frame→decompose→compile→review) and the real scoring path with zero
model calls. Two paths are proven end-to-end: a coherent rung-1 design scores ``built`` and
passes a sequence case; a fitness head that returns rung ``app`` makes forge refuse, scoring
``refused`` and passing a refuse case.
"""

from __future__ import annotations

from andamentum.core import AgentDefinition
from andamentum.forge.schemas import Fitness, ForgeWhy
from andamentum.forge.spec import NodeKind
from andamentum.forge.tests.conftest import NodeScript, ScriptedSink

from .runner import run_case
from .types import Case

# The model is unused when a sink is injected, but run_forge requires the keyword.
_STUB_MODEL = "stub:offline"


def _coherent_sink() -> ScriptedSink:
    """A small coherent rung-1 script: parse the request (spine) → answer it (head)."""
    return ScriptedSink(
        why=ForgeWhy(
            purpose="Summarise the input.",
            boundary_in="a natural-language request",
            boundary_out="a text answer",
        ),
        areas=["core"],
        jobs_by_area={"core": ["Parse the request.", "Answer the request."]},
        typings={
            "n1": NodeScript(
                kind=NodeKind.SPINE, consumes=["input"], produces=["parsed_request"]
            ),
            "n2": NodeScript(
                kind=NodeKind.HEAD, consumes=["parsed_request"], produces=["answer"]
            ),
        },
    )


class _RefusingSink(ScriptedSink):
    """A coherent sink whose fitness head refuses the brief as an app (rung ``app``)."""

    async def run(self, defn: AgentDefinition, **kwargs: object):  # type: ignore[override]
        if defn.name == "fitness":
            return Fitness(
                realizable_as_function=False,
                rung="app",
                reason="An external driver owns the control loop — this is an app.",
                suggested_reshape="Reshape to one request → one answer.",
            )
        return await super().run(defn, **kwargs)


async def test_run_case_build_scores_built_and_passes() -> None:
    case = Case(
        brief="Summarise the document into three bullet points.",
        expected="build",
        grammar="sequence",
    )
    score = await run_case(case, model=_STUB_MODEL, runs=2, sink=_coherent_sink())

    assert score.total == 2
    assert all(o.kind == "built" for o in score.runs)
    # The parse→answer design shows no structural feature → a plain sequence.
    assert all(o.features == set() for o in score.runs)
    assert score.passes == 2
    assert score.pass_rate == 1.0


async def test_run_case_tier2_end_to_end_scores_and_populates_signals() -> None:
    """Tier-2 (full=True) drives render→build→audit with a stub sink + FakeSandbox.

    Proves the end-to-end wiring deterministically (no model, no container): the run
    reaches the audit stage, produces a works/incomplete verdict, and populates the
    reliability signals (holes filled/total) the design-shape score cannot see.
    """
    from andamentum.forge.tests.conftest import FakeSandbox

    case = Case(
        brief="Summarise the document into three bullet points.",
        expected="build",
        grammar="sequence",
    )
    score = await run_case(
        case,
        model=_STUB_MODEL,
        runs=1,
        full=True,
        sink=_coherent_sink(),
        sandbox=FakeSandbox(exit_code=0, stdout="2 passed in 0.10s"),
    )

    assert score.total == 1
    o = score.runs[0]
    assert o.kind in {"works", "incomplete"}
    assert o.stage_reached == "audit"
    assert o.holes_total >= 1
    assert o.works is not None


async def test_run_case_refuse_scores_refused_and_passes() -> None:
    sink = _RefusingSink(
        why=ForgeWhy(
            purpose="Manage a personal reading list.",
            boundary_in="a request",
            boundary_out="an answer",
        ),
        areas=["core"],
        jobs_by_area={"core": ["Parse the request.", "Answer the request."]},
    )
    case = Case(
        brief="Manage my personal reading list.",
        expected="refuse",
        grammar="none",
    )
    score = await run_case(case, model=_STUB_MODEL, runs=2, sink=sink)

    assert score.total == 2
    assert all(o.kind == "refused" for o in score.runs)
    assert score.passes == 2
    assert score.pass_rate == 1.0
