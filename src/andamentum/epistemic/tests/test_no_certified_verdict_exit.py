"""Tests for the no_certified_verdict exit in CheckCompletion.

Background — the case 957 v17 finding: the synthesis writer's prose
("No, the claim is not supported") disagreed with the posterior
aggregator's number (0.85 "supports") on a claim that was
cycle_capped at HYPOTHESIS without an ``integrated_assessment``.
The two disagreed because both were running *ungrounded* — the IBE
chain (the system's certifier) never ran on this claim, so neither
the writer nor the aggregator had a certified verdict to articulate.

The fix routes to ``SynthesizeInsufficient`` (the existing
fallibilism terminal) when no active claim has an
``integrated_assessment``. The principle: if the IBE chain didn't
certify a verdict, the output layer must say so rather than
fabricating one. Same shape as the K3 + K6 ``Maximal B`` exits
(retrieval_failed, no_claims, all-abandoned), now extended with a
fourth exit — no_certified_verdict.

These tests pin:
  1. CheckCompletion routes to SynthesizeInsufficient when claims
     exist but none have integrated_assessment.
  2. CheckCompletion routes to CheckSynthesisDemand when at least
     one claim has integrated_assessment (existing behaviour
     preserved).
  3. The synthesis_insufficient_reason names the structural cause.
  4. Exit ordering: retrieval_failed > no_certified_verdict >
     all-abandoned > no-claims.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import (
    CheckCompletion,
    CheckSynthesisDemand,
    SynthesizeInsufficient,
)
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.repository import EpistemicRepository


class _FakeRunContext:
    def __init__(self, state: EpistemicGraphState, deps: EpistemicDeps) -> None:
        self.state = state
        self.deps = deps


async def _setup(
    tmp_path: Path,
    db_name: str,
) -> tuple[Objective, EpistemicRepository]:
    store = DocumentStore.for_database(db_name, db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)
    obj = Objective(
        description="Test claim X.",
        clarified_question="Test claim X.",
        question_type="verificatory",
        claim_to_verify="Test claim X.",
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    return obj, repo


@pytest.mark.asyncio
async def test_routes_to_synthesize_insufficient_when_no_ia(
    tmp_path: Path,
) -> None:
    """The case 957 shape: cycle-capped at HYPOTHESIS, no
    integrated_assessment. Should route to SynthesizeInsufficient."""
    obj, repo = await _setup(tmp_path, "no_ia_capped")
    claim = Claim(
        objective_id=obj.entity_id,
        statement="Test claim X.",
        scope="scope",
        stage=ClaimStage.HYPOTHESIS,
        cycle_capped=True,
        scrutiny_verdict="pass",
    )
    await repo.save(claim)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    ctx = _FakeRunContext(state, deps)
    next_node = await CheckCompletion().run(ctx)  # type: ignore[arg-type]

    assert isinstance(next_node, SynthesizeInsufficient)
    assert state.synthesis_insufficient_reason is not None
    assert "IBE certification" in state.synthesis_insufficient_reason
    assert "integrated_assessment" in state.synthesis_insufficient_reason


@pytest.mark.asyncio
async def test_routes_to_check_synthesis_demand_when_any_ia_exists(
    tmp_path: Path,
) -> None:
    """When at least one active claim has an integrated_assessment,
    route to CheckSynthesisDemand. Existing behaviour preserved."""
    obj, repo = await _setup(tmp_path, "with_ia")
    claim = Claim(
        objective_id=obj.entity_id,
        statement="Test claim X.",
        scope="scope",
        stage=ClaimStage.SUPPORTED,
        scrutiny_verdict="pass",
        integrated_assessment="contradicts",
        integrated_confidence=0.8,
    )
    await repo.save(claim)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    ctx = _FakeRunContext(state, deps)
    next_node = await CheckCompletion().run(ctx)  # type: ignore[arg-type]

    assert isinstance(next_node, CheckSynthesisDemand)
    # Reason field should NOT be set on this path.
    assert state.synthesis_insufficient_reason is None


@pytest.mark.asyncio
async def test_mixed_some_ia_some_not_routes_to_csd(tmp_path: Path) -> None:
    """Mixed case (one claim with IA, one without) — route to
    CheckSynthesisDemand. Existing CSD logic handles partial coverage
    via Gate 3 (n_no_verdict)."""
    obj, repo = await _setup(tmp_path, "mixed")
    obj.claim_to_verify = None  # Remove single-seed mode for multi-claim test
    await repo.save(obj)

    claim_with_ia = Claim(
        objective_id=obj.entity_id,
        statement="claim A",
        scope="scope",
        stage=ClaimStage.SUPPORTED,
        integrated_assessment="supports",
        integrated_confidence=0.7,
    )
    claim_without_ia = Claim(
        objective_id=obj.entity_id,
        statement="claim B",
        scope="scope",
        stage=ClaimStage.HYPOTHESIS,
        cycle_capped=True,
    )
    await repo.save(claim_with_ia)
    await repo.save(claim_without_ia)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    ctx = _FakeRunContext(state, deps)
    next_node = await CheckCompletion().run(ctx)  # type: ignore[arg-type]

    assert isinstance(next_node, CheckSynthesisDemand), (
        "Mixed case (any IA exists) should still route to CSD; the "
        "no_certified_verdict exit is for when ALL claims lack IA."
    )


@pytest.mark.asyncio
async def test_retrieval_failed_takes_precedence_over_no_ia(
    tmp_path: Path,
) -> None:
    """Exit-ordering invariant: retrieval_failed wins over
    no_certified_verdict. The retrieval-failed reason text is more
    informative about the operational failure mode and should be the
    one surfaced when both conditions hold."""
    obj, repo = await _setup(tmp_path, "retrieval_failed_priority")
    claim = Claim(
        objective_id=obj.entity_id,
        statement="claim",
        scope="scope",
        stage=ClaimStage.HYPOTHESIS,
        cycle_capped=True,
    )
    await repo.save(claim)

    state = EpistemicGraphState(
        objective_id=obj.entity_id,
        retrieval_failed=True,
    )
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    ctx = _FakeRunContext(state, deps)
    next_node = await CheckCompletion().run(ctx)  # type: ignore[arg-type]

    assert isinstance(next_node, SynthesizeInsufficient)
    assert state.synthesis_insufficient_reason is not None
    assert "Retrieval failed" in state.synthesis_insufficient_reason
    # The no_certified_verdict reason should NOT have been written.
    assert "IBE certification" not in state.synthesis_insufficient_reason


@pytest.mark.asyncio
async def test_all_abandoned_still_routes_correctly(tmp_path: Path) -> None:
    """The all-abandoned exit (existing behaviour) still fires when
    all claims are abandoned, regardless of whether any had IA before
    abandonment. Pin: existing K6 behaviour preserved."""
    obj, repo = await _setup(tmp_path, "all_abandoned")
    claim = Claim(
        objective_id=obj.entity_id,
        statement="claim",
        scope="scope",
        stage=ClaimStage.HYPOTHESIS,
        abandoned=True,
    )
    await repo.save(claim)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    ctx = _FakeRunContext(state, deps)
    next_node = await CheckCompletion().run(ctx)  # type: ignore[arg-type]

    assert isinstance(next_node, SynthesizeInsufficient)
    assert state.synthesis_insufficient_reason is not None
    assert "abandoned" in state.synthesis_insufficient_reason


@pytest.mark.asyncio
async def test_no_claims_still_routes_correctly(tmp_path: Path) -> None:
    """The no_claims exit still fires when the objective has zero
    claims. Pin: existing K6 behaviour preserved."""
    obj, repo = await _setup(tmp_path, "no_claims")
    # No claims saved.

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    ctx = _FakeRunContext(state, deps)
    next_node = await CheckCompletion().run(ctx)  # type: ignore[arg-type]

    assert isinstance(next_node, SynthesizeInsufficient)
    assert state.synthesis_insufficient_reason is not None
    assert "No claims were created" in state.synthesis_insufficient_reason


@pytest.mark.asyncio
async def test_retrieval_failed_with_ia_set_routes_to_writer(
    tmp_path: Path,
) -> None:
    """Phase A invariant — IA presence is the canonical certifier.

    Case 957 rep 5 shape: ``state.retrieval_failed`` was True (some
    extractions returned empty content during the run) but the IBE
    chain DID certify on the evidence available, so
    ``integrated_assessment`` is set on at least one active claim.

    Pre-fix: CheckCompletion's first check (``if state.retrieval_failed``)
    routed to SynthesizeInsufficient unconditionally, while
    compute_posterior saw the IA and emitted a directional posterior.
    Result: writer prose said "Insufficient Evidence" while the
    aggregator emitted 0.773. Writer-aggregator disagreement.

    Post-fix: ``any_certified`` is the primary gate. With IA set, route
    to CheckSynthesisDemand even when retrieval_failed=True. The writer
    produces a directional verdict; retrieval_failed is annotated in
    prose (or surfaced via metadata) but does NOT suppress the verdict.
    """
    obj, repo = await _setup(tmp_path, "retrieval_failed_with_ia")
    claim = Claim(
        objective_id=obj.entity_id,
        statement="Test claim",
        scope="scope",
        stage=ClaimStage.SUPPORTED,
        integrated_assessment="supports",
        integrated_confidence=0.78,
    )
    await repo.save(claim)

    state = EpistemicGraphState(
        objective_id=obj.entity_id,
        retrieval_failed=True,
    )
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    ctx = _FakeRunContext(state, deps)
    next_node = await CheckCompletion().run(ctx)  # type: ignore[arg-type]

    # IA wins: route to writer despite retrieval_failed=True.
    assert isinstance(next_node, CheckSynthesisDemand)
    # No insufficient-reason should be set (we're not going to the
    # insufficient terminal).
    assert state.synthesis_insufficient_reason is None
