"""Tests for DispatchGatherOperation and the PlanEvidence node wiring.

Description-driven dispatch is the only evidence-gathering path. These
tests cover:

- ``DispatchGatherOperation`` persist / abstain / phase-advance behaviour.
- ``PlanEvidence`` routes objective through DispatchGatherOperation and
  quarantines correctly on operation failure.

Tests use a stub agent runner and stub providers — no live LLM, no HTTP
calls.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from andamentum.epistemic.entities import Objective
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import ExtractEvidence, PlanEvidence
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.operations.base import GatheredEvidence
from andamentum.epistemic.operations.dispatch_gather import DispatchGatherOperation
from andamentum.epistemic.repository import EpistemicRepository


# ──────────────────────────────────────────────────────────────────────────────
# Stubs
# ──────────────────────────────────────────────────────────────────────────────


class _DispatchStubRunner:
    """Two-protocol stub mimicking ``DefaultAgentRunner``.

    - Name-based ``.run(agent_name, **kw)`` for everything BaseOperation
      drives via ``self.run_agent``.
    - ``.core_runner`` exposes a definition-based ``.run(defn, **kw)``
      stub that ``gather_evidence_new`` calls.

    Routes ``epistemic_dispatch_provider`` calls through
    ``dispatch_responses[provider_name]``. Any other agent name returns
    a permissive default.
    """

    def __init__(
        self,
        dispatch_responses: dict[str, dict[str, Any]] | None = None,
        scoring_response: dict[str, Any] | None = None,
    ):
        self._dispatch = dispatch_responses or {}
        self._scoring = scoring_response or {
            "source_credibility": 0.7,
            "relevance": 0.7,
            "specificity": 0.7,
            "recency_appropriate": 0.7,
            "justification": "stub",
        }
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.model = "stub-model"
        self.core_runner = _CoreRunnerStub(parent=self)

    async def run(self, agent_name: str, **kwargs: Any) -> Any:
        self.calls.append((agent_name, kwargs))
        if agent_name == "epistemic_assess_evidence_quality":
            return SimpleNamespace(**self._scoring)
        return SimpleNamespace()


class _CoreRunnerStub:
    """Definition-based runner stub — the shape gather_evidence_new wants."""

    def __init__(self, *, parent: _DispatchStubRunner):
        self._parent = parent
        self.model = parent.model

    async def run(self, defn: Any, **kwargs: Any) -> Any:
        name = getattr(defn, "name", str(defn))
        self._parent.calls.append((name, kwargs))
        provider_name = kwargs.get("provider_name", "")
        spec = self._parent._dispatch.get(
            provider_name,
            {"queries": [], "reasoning": "stub abstain", "confidence": 0.5},
        )
        return SimpleNamespace(
            queries=list(spec["queries"]),
            reasoning=str(spec["reasoning"]),
            confidence=float(spec["confidence"]),
        )


class _StubProvider:
    """Minimal provider implementing the dispatch contract."""

    description = (
        "Stub provider. Strong for stub claims. Weak for irrelevant claims. "
        "Returns canned evidence on gather()."
    )
    query_guidance = "Plain text. Use 'id:' prefix for ID lookups."
    output_kind = "assertion_evidence"
    independence_group = "stub"
    provider_contract_version = 1
    query_examples: list[tuple[str, str | None]] = [
        ("stub claim", "stub query"),
        ("off-topic claim", None),
    ]

    def __init__(self, *, name: str, gathered: list[GatheredEvidence]):
        self._name = name
        self._gathered = gathered
        self.gather_calls: list[str] = []

    async def gather(self, query: str) -> list[GatheredEvidence]:
        self.gather_calls.append(query)
        return list(self._gathered)


# ──────────────────────────────────────────────────────────────────────────────
# DispatchGatherOperation tests
# ──────────────────────────────────────────────────────────────────────────────


class TestDispatchGatherOperation:
    async def _make_objective(self, repo: EpistemicRepository) -> Objective:
        obj = Objective(
            entity_id="obj_test",
            objective_id="obj_test",
            description="Does stub claim hold?",
            phase="analyzed",
        )
        await repo.save(obj)
        return obj

    async def test_persists_one_evidence_per_committed_dispatch(
        self, repo: EpistemicRepository
    ) -> None:
        await self._make_objective(repo)
        gathered = GatheredEvidence(
            content="Stub evidence content.",
            source_ref="doi:10.1/abc",
            source_type="stub_a",
            evidence_kind="literature",
            quality_score=0.8,
        )
        provider = _StubProvider(name="stub_a", gathered=[gathered])
        runner = _DispatchStubRunner(
            dispatch_responses={
                "stub_a": {
                    "queries": ["stub query"],
                    "reasoning": "fits",
                    "confidence": 0.9,
                }
            }
        )

        op = DispatchGatherOperation(
            repo, runner, providers={"stub_a": provider}
        )
        result = await op.execute(
            _make_input(entity_id="obj_test", entity_type="objective")
        )

        assert result.success
        ev_list = await repo.query("evidence", objective_id="obj_test")
        assert len(ev_list) == 1
        ev = ev_list[0]
        assert ev.extracted is True
        assert ev.extracted_content == "Stub evidence content."
        assert ev.source_type == "stub_a"
        assert ev.quality_score is not None
        assert provider.gather_calls == ["stub query"]

    async def test_abstain_means_no_gather_no_evidence(
        self, repo: EpistemicRepository
    ) -> None:
        await self._make_objective(repo)
        provider = _StubProvider(name="stub_a", gathered=[])
        runner = _DispatchStubRunner(
            dispatch_responses={
                "stub_a": {
                    "queries": [],
                    "reasoning": "not in scope",
                    "confidence": 0.9,
                }
            }
        )

        op = DispatchGatherOperation(
            repo, runner, providers={"stub_a": provider}
        )
        result = await op.execute(
            _make_input(entity_id="obj_test", entity_type="objective")
        )

        assert result.success
        assert provider.gather_calls == []
        ev_list = await repo.query("evidence", objective_id="obj_test")
        assert ev_list == []

    async def test_phase_advances_to_planned(
        self, repo: EpistemicRepository
    ) -> None:
        await self._make_objective(repo)
        provider = _StubProvider(name="stub_a", gathered=[])
        runner = _DispatchStubRunner()

        op = DispatchGatherOperation(
            repo, runner, providers={"stub_a": provider}
        )
        await op.execute(
            _make_input(entity_id="obj_test", entity_type="objective")
        )

        obj = await repo.get("objective", "obj_test")
        assert obj.phase == "planned"

    async def test_raises_without_providers(
        self, repo: EpistemicRepository
    ) -> None:
        await self._make_objective(repo)
        runner = _DispatchStubRunner()

        op = DispatchGatherOperation(repo, runner, providers={})
        with pytest.raises(RuntimeError, match="providers"):
            await op.execute(
                _make_input(entity_id="obj_test", entity_type="objective")
            )

    async def test_skip_when_phase_not_analyzed(
        self, repo: EpistemicRepository
    ) -> None:
        obj = Objective(
            entity_id="obj_x",
            objective_id="obj_x",
            description="q",
            phase="new",
        )
        await repo.save(obj)
        provider = _StubProvider(name="stub_a", gathered=[])
        runner = _DispatchStubRunner()

        op = DispatchGatherOperation(
            repo, runner, providers={"stub_a": provider}
        )
        result = await op.execute(
            _make_input(entity_id="obj_x", entity_type="objective")
        )

        assert result.success
        assert "not 'analyzed'" in result.message
        assert provider.gather_calls == []
        ev_list = await repo.query("evidence", objective_id="obj_x")
        assert ev_list == []


# ──────────────────────────────────────────────────────────────────────────────
# PlanEvidence node — wiring tests
# ──────────────────────────────────────────────────────────────────────────────


class TestPlanEvidenceWiring:
    async def test_plan_evidence_dispatches_and_advances(
        self, repo: EpistemicRepository, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PlanEvidence runs DispatchGatherOperation and returns ExtractEvidence."""
        obj = Objective(
            entity_id="obj_dispatch",
            objective_id="obj_dispatch",
            description="q?",
            phase="analyzed",
        )
        await repo.save(obj)

        dispatch_calls: list[str] = []

        from andamentum.epistemic.operations import dispatch_gather as dg_mod

        async def fake_dispatch_execute(self: Any, work: Any) -> Any:
            from andamentum.epistemic.operations.base import OperationResult
            dispatch_calls.append(work.entity_id)
            target = await self.repo.get("objective", work.entity_id)
            target.phase = "planned"
            await self.repo.save(target)
            return OperationResult(
                success=True, entity_id=work.entity_id, message="dispatched"
            )

        monkeypatch.setattr(
            dg_mod.DispatchGatherOperation, "execute", fake_dispatch_execute
        )

        state = EpistemicGraphState(
            objective_id="obj_dispatch",
            question="q?",
        )
        deps = EpistemicDeps(
            repo=repo,
            agent_runner=_DispatchStubRunner(),
            providers={"stub_a": _StubProvider(name="stub_a", gathered=[])},
        )

        node = PlanEvidence()
        next_node = await node.run(_FakeCtx(state, deps))  # type: ignore[arg-type]

        assert dispatch_calls == ["obj_dispatch"]
        assert isinstance(next_node, ExtractEvidence)

    async def test_plan_evidence_quarantines_on_exception(
        self, repo: EpistemicRepository, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If DispatchGatherOperation raises, PlanEvidence records a
        quarantine rather than propagating — matches the _run_op contract
        for other nodes."""
        obj = Objective(
            entity_id="obj_boom",
            objective_id="obj_boom",
            description="q?",
            phase="analyzed",
        )
        await repo.save(obj)

        from andamentum.epistemic.operations import dispatch_gather as dg_mod

        async def boom_execute(self: Any, work: Any) -> Any:
            raise RuntimeError("simulated dispatch failure")

        monkeypatch.setattr(
            dg_mod.DispatchGatherOperation, "execute", boom_execute
        )

        state = EpistemicGraphState(
            objective_id="obj_boom",
            question="q?",
        )
        deps = EpistemicDeps(
            repo=repo,
            agent_runner=_DispatchStubRunner(),
            providers={"stub_a": _StubProvider(name="stub_a", gathered=[])},
        )

        node = PlanEvidence()
        await node.run(_FakeCtx(state, deps))  # type: ignore[arg-type]

        assert any(
            q.entity_id == "obj_boom" and q.operation == "dispatch_gather"
            for q in state.quarantined
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_input(*, entity_id: str, entity_type: str) -> Any:
    from andamentum.epistemic.operations.base import OperationInput

    return OperationInput(
        entity_id=entity_id,
        entity_type=entity_type,
        operation="dispatch_gather",
        metadata={},
    )


class _FakeCtx:
    """Minimal GraphRunContext stand-in: just .state and .deps."""

    def __init__(self, state: EpistemicGraphState, deps: EpistemicDeps):
        self.state = state
        self.deps = deps
