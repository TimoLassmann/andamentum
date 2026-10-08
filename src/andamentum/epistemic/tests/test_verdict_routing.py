"""Routing: the verdict object is never stale and never absent.

``CheckCompletion`` decides the run's terminal. It used to query the
claims itself while ``compute_posterior`` applied the same condition
independently — two conditions kept in sync by convention (commit
761a5d9). They are now literally the same object, and this file pins
the topology property that makes that safe: ``CombineClaimVerdicts``
DOMINATES ``CheckCompletion``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.document_store import Store
from andamentum.epistemic.entities import Claim, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import (
    CheckCompletion,
    CheckSynthesisDemand,
    CombineClaimVerdicts,
    PromoteToSupported,
    Scrutinize,
    Synthesize,
    SynthesizeInsufficient,
)
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.graph.topology import topology
from andamentum.epistemic.repository import EpistemicRepository


class _Ctx:
    def __init__(self, state, deps):
        self.state = state
        self.deps = deps


async def _repo(tmp_path: Path, name: str) -> EpistemicRepository:
    store = await Store.open(tmp_path / f"{name}.db")
    return EpistemicRepository(store)


def test_combine_claim_verdicts_is_the_sole_predecessor_of_check_completion() -> None:
    """THE DOMINATION PROPERTY.

    Every route to CheckCompletion passes through CombineClaimVerdicts,
    so ``state.verdict`` is rebuilt from the live DB immediately before
    every terminal decision — including after a
    ``CheckSynthesisDemand → Scrutinize`` loop-back, where a stale
    verdict would otherwise decide the run.

    Re-adding the old ``PromoteToSupported → CheckCompletion`` edge fails
    here: the silent-dead-zone bug class made structurally detectable
    rather than benchmark-detectable.
    """
    topo = topology()
    predecessors = {n for n, succ in topo.items() if CheckCompletion in succ}
    assert predecessors == {CombineClaimVerdicts}, (
        "CheckCompletion's predecessors are "
        f"{sorted(n.__name__ for n in predecessors)}; it must be exactly "
        "{CombineClaimVerdicts}. Any other edge lets a terminal decision "
        "be made against a stale or absent verdict."
    )


def test_check_synthesis_demand_successors_unchanged() -> None:
    """The gates were re-keyed onto the verdict object; no edge moved."""
    topo = topology()
    successors = next(s for n, s in topo.items() if n is CheckSynthesisDemand)
    assert successors == frozenset({Synthesize, Scrutinize, SynthesizeInsufficient})


async def test_check_completion_raises_without_a_verdict(tmp_path: Path) -> None:
    """The guard behind the domination property. Reaching CheckCompletion
    with no verdict is a routing bug — crash rather than route around it."""
    repo = await _repo(tmp_path, "routing_guard")
    obj = Objective(description="x", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    with pytest.raises(RuntimeError, match="CombineClaimVerdicts"):
        await CheckCompletion().run(_Ctx(state, deps))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("assessment", "expected"),
    [("supports", CheckSynthesisDemand), (None, SynthesizeInsufficient)],
)
async def test_check_completion_routing_agrees_with_the_verdict_object(
    tmp_path: Path, assessment: str | None, expected: type
) -> None:
    """Routing is a pure function of ``provenance.n_adjudicated`` — the
    same count ``compute_posterior``'s no-certified-verdict gate uses. The
    writer/aggregator coherence contract is now a structural property of
    one object rather than two conditions maintained in step."""
    repo = await _repo(tmp_path, f"routing_{assessment}")
    obj = Objective(description="x", claim_to_verify="x", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    claim = Claim(
        objective_id=obj.entity_id,
        statement="x",
        stage=ClaimStage.SUPPORTED,
        integrated_assessment=assessment,
        integrated_confidence=0.8 if assessment else None,
    )
    await repo.save(claim)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    ctx = _Ctx(state, deps)
    await CombineClaimVerdicts().run(ctx)  # type: ignore[arg-type]
    next_node = await CheckCompletion().run(ctx)  # type: ignore[arg-type]

    assert isinstance(next_node, expected)
    assert state.verdict is not None
    if assessment is None:
        assert state.verdict.provenance.n_adjudicated == 0
        # The reason text is derived from the same provenance counts.
        assert state.synthesis_insufficient_reason is not None
        assert "1 non-abandoned claim(s)" in state.synthesis_insufficient_reason
    else:
        assert state.verdict.provenance.n_adjudicated == 1
        assert state.synthesis_insufficient_reason is None


async def test_promote_to_supported_routes_through_combine_claim_verdicts(
    tmp_path: Path, fake_runner
) -> None:
    """The retargeted edge, at the node level as well as in topology."""
    repo = await _repo(tmp_path, "routing_promote")
    obj = Objective(description="x", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    claim = Claim(
        objective_id=obj.entity_id,
        statement="x",
        stage=ClaimStage.SUPPORTED,
        scrutiny_verdict="pass",
        integrated_assessment="supports",
        integrated_confidence=0.8,
    )
    await repo.save(claim)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    state.verification_done.add(claim.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")

    next_node = await PromoteToSupported().run(_Ctx(state, deps))  # type: ignore[arg-type]
    assert isinstance(next_node, CombineClaimVerdicts)


def test_verdict_state_field_is_declared_by_every_node_that_touches_it() -> None:
    """P6 contract for the one state field this change adds.
    ``test_node_contracts.py`` only checks the ClassVars EXIST, so
    contract drift on a specific field is otherwise invisible."""
    assert "verdict" in CombineClaimVerdicts.writes
    for node in (
        CheckCompletion,
        CheckSynthesisDemand,
        Synthesize,
        SynthesizeInsufficient,
    ):
        assert "verdict" in node.reads, f"{node.__name__} reads state.verdict"


async def test_synthesis_stage_entry_rebuilds_the_verdict() -> None:
    """Resuming ``--stage synthesis`` must rebuild the verdict from the
    persisted DB before CheckCompletion's fail-loud guard reads it."""
    from andamentum.epistemic.graph.stages import STAGES

    assert STAGES["synthesis"].entry is CombineClaimVerdicts
