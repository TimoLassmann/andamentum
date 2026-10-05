"""The decision-model replacement of the topic verifier (``verify_query``).

Offline: the decision model is a tiny stub with an async ``decide``, the
``topic_verifier`` agent a scripted stub.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic_graph import GraphRunContext

from andamentum.core.decisions import Answer, DecisionModel
from andamentum.deep_research.models import VerifierOutput
from andamentum.deep_research.nodes import GenerateOne, NodeDeps, Verify
from andamentum.deep_research.state import ResearchState
from andamentum.deep_research.verify_query import (
    QUESTION,
    REJECT_FEEDBACK,
    VERIFIER_REQUEST_LIMIT,
    verify_query,
)

from .test_search_loop_plumbing import StubAgent, StubBackend

GOAL = "What is metformin's elimination half-life?"
QUERY = "metformin t1/2 plasma kinetics"
AGENT_VERDICT = VerifierOutput(on_topic=True, reason="agent says ok")


class StubDecisionModel:
    def __init__(self, p: float) -> None:
        self.p = p
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    async def decide(self, state: Any, questions: dict[str, Any]) -> dict[str, Answer]:
        self.calls.append((state, questions))
        return {
            "q": Answer(
                choice="yes" if self.p >= 0.5 else "no",
                probabilities={"yes": self.p, "no": 1 - self.p},
                confidence=max(self.p, 1 - self.p),
                p_true=self.p,
            )
        }


def stub(p: float) -> tuple[StubDecisionModel, DecisionModel]:
    s = StubDecisionModel(p)
    return s, cast(DecisionModel, s)


class RecordingAgent(StubAgent):
    def __init__(self, outputs):
        super().__init__(outputs)
        self.kwargs: list[dict] = []

    async def run(self, prompt, **kwargs):
        self.kwargs.append(kwargs)
        return await super().run(prompt, **kwargs)


async def test_without_a_decision_model_the_agent_decides_as_before() -> None:
    agent = RecordingAgent([AGENT_VERDICT])
    verdict = await verify_query(
        QUERY, goal=GOAL, model="stub", overrides={"topic_verifier": agent}
    )
    assert verdict is AGENT_VERDICT
    assert agent.calls == [f"research_goal: {GOAL}\nquery: {QUERY}"]
    assert agent.kwargs[0]["usage_limits"].request_limit == VERIFIER_REQUEST_LIMIT


async def test_explicit_none_is_the_same_as_leaving_it_out() -> None:
    agent = RecordingAgent([AGENT_VERDICT])
    verdict = await verify_query(
        QUERY,
        goal=GOAL,
        model="stub",
        overrides={"topic_verifier": agent},
        decision_model=None,
    )
    assert verdict is AGENT_VERDICT


async def test_a_yes_accepts_without_the_agent() -> None:
    server, model = stub(0.7)
    agent = RecordingAgent([])
    verdict = await verify_query(
        QUERY,
        goal=GOAL,
        model="stub",
        overrides={"topic_verifier": agent},
        decision_model=model,
    )
    assert verdict.on_topic is True
    assert agent.calls == []
    state, questions = server.calls[0]
    assert state == {"research_goal": GOAL, "query": QUERY}
    assert questions == {"q": QUESTION}


async def test_a_no_rejects_with_the_fixed_feedback() -> None:
    _, model = stub(0.3)
    agent = RecordingAgent([])
    verdict = await verify_query(
        QUERY,
        goal=GOAL,
        model="stub",
        overrides={"topic_verifier": agent},
        decision_model=model,
    )
    assert verdict.on_topic is False
    assert verdict.reason == REJECT_FEEDBACK
    assert agent.calls == []


async def test_verify_node_retries_with_the_fixed_feedback_on_a_no() -> None:
    _, model = stub(0.01)
    agent = StubAgent([])
    deps = NodeDeps(
        backend=StubBackend(),
        model="stub",
        correlation_id="test",
        agent_overrides={"topic_verifier": agent},
        decision_model=model,
    )
    ctx = GraphRunContext(state=ResearchState(query=GOAL), deps=deps)
    nxt = await Verify(query="metformin patents").run(ctx)
    assert isinstance(nxt, GenerateOne)
    assert nxt.feedback == REJECT_FEEDBACK
    assert agent.calls == []


async def test_run_research_threads_the_decision_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from andamentum.deep_research import graph, orchestrator, runner
    from andamentum.deep_research.models import EvidenceReport

    seen: list[NodeDeps] = []

    async def fake_run(*, state, deps):
        seen.append(deps)
        return EvidenceReport(
            evidence_summary="s",
            key_findings=["k"],
            sources=["https://example.test/1"],
            total_searches_performed=0,
            total_pages_fetched=0,
            iterations_required=0,
        )

    monkeypatch.setattr(graph.research_graph, "run", fake_run)
    monkeypatch.setattr(runner, "_resolve_model", lambda name: name)
    _, model = stub(0.5)
    await orchestrator.run_research(
        GOAL, model="stub", backend=StubBackend(), decision_model=model
    )
    await orchestrator.run_research(GOAL, model="stub", backend=StubBackend())
    assert seen[0].decision_model is model
    assert seen[1].decision_model is None


def test_cli_has_the_flag() -> None:
    from andamentum.deep_research import cli

    args = cli._build_parser().parse_args(["q", "--decision-model", "nimble:9b"])
    assert args.decision_model == "nimble:9b"
    assert cli._build_parser().parse_args(["q"]).decision_model is None
