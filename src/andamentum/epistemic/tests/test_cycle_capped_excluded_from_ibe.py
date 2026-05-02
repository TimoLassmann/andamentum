"""Regression: cycle-capped claims are excluded from verification + IBE.

Found via the post-revert benchmark on 2026-05-02. Background:

* The inquiry loop has a "cycle cap" — when scrutiny ↔ resolve cycles
  too many times, the claim gets ``cycle_capped=True``.
* IBE chain is a separate process that fires on SUPPORTED claims with
  no integration verdict yet.
* ``combine_claim_verdicts`` excludes cycle-capped claims from the
  rule-aware combination (consistent with cycle-cap's "inquiry didn't
  converge" semantics).

Pre-fix: the IBE chain didn't check ``cycle_capped``, so it would
fire on cycle-capped SUPPORTED claims, produce an integrated_assessment
+ confidence, and then the combiner would discard the verdict. We saw
this in a benchmark: 1 claim ended up with
``stage=SUPPORTED, cycle_capped=True, integrated_assessment="contradicts",
confidence=0.74``, but the combined verdict was "no_data" because the
combiner skipped it. ~4 LLM calls wasted per cycle-capped claim that
made it past soft_promote.

Fix (this commit): added ``not c.cycle_capped`` to the verification
gate (PromoteToSupported) and the four IBE chain filters
(EnumerateCandidates, ScoreLoveliness, ScoreLikeliness,
SelectBestExplanation). Cycle-capped claims now stop at PromoteToSupported's
``needs_verification`` check; no LLM cost wasted.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import (
    EnumerateCandidates,
    PromoteToSupported,
    ScoreLikeliness,
    ScoreLoveliness,
    SelectBestExplanation,
)
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.repository import EpistemicRepository


class _FakeRunContext:
    def __init__(self, state, deps):
        self.state = state
        self.deps = deps


async def test_cycle_capped_supported_skipped_by_promote_to_supported(
    tmp_path: Path, fake_runner
) -> None:
    """When the only SUPPORTED claim is cycle-capped,
    PromoteToSupported must NOT route to ClusterEvidence (which would
    eventually run verification + IBE on the claim, only for the
    combiner to discard the result). It should treat the run as having
    no work to do for SUPPORTED claims."""
    from andamentum.epistemic.graph.nodes import CheckCompletion

    store = DocumentStore.for_database("cycle_capped_skip", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    # SUPPORTED claim that's been cycle-capped after promotion.
    # This is the state we observed in the post-revert benchmark.
    claim = Claim(
        objective_id=obj.entity_id,
        statement="cycle-capped claim",
        scope="x",
        stage=ClaimStage.SUPPORTED,
        cycle_capped=True,
        integrated_assessment=None,
        scrutiny_verdict="pass",
    )
    await repo.save(claim)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(
        repo=repo, agent_runner=fake_runner, embedding_model="t"
    )

    next_node = await PromoteToSupported().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    # Should NOT route to ClusterEvidence (which would launch IBE).
    # Should route to CheckCompletion since no SUPPORTED-and-not-
    # cycle-capped-and-not-verified claims exist.
    assert isinstance(next_node, CheckCompletion), (
        f"PromoteToSupported routed to {type(next_node).__name__} for "
        "a cycle-capped SUPPORTED claim. Expected CheckCompletion — "
        "cycle-capped claims should not enter verification or IBE "
        "because combine_claim_verdicts will discard their verdict."
    )


async def test_ibe_filters_skip_cycle_capped(
    tmp_path: Path, fake_runner
) -> None:
    """Defensive: even if a cycle-capped SUPPORTED claim somehow
    reaches one of the IBE chain nodes, the filter should skip it
    (no operation called for that claim).

    This is layered defense — PromoteToSupported's cycle-cap check
    above is the primary gate, but the IBE filters back it up so
    a refactor that bypasses PromoteToSupported (or routes
    differently) doesn't silently re-introduce the wasted-LLM-cost
    bug.
    """
    store = DocumentStore.for_database("ibe_skip_capped", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    # SUPPORTED + cycle_capped + would otherwise be IBE-eligible
    # (no integrated_assessment, no integration_candidates).
    claim = Claim(
        objective_id=obj.entity_id,
        statement="cycle-capped claim",
        scope="x",
        stage=ClaimStage.SUPPORTED,
        cycle_capped=True,
        integrated_assessment=None,
    )
    await repo.save(claim)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(
        repo=repo, agent_runner=fake_runner, embedding_model="t"
    )

    # All four IBE chain nodes should run as no-ops on this claim
    # (and proceed to the next stage).
    next_node = await EnumerateCandidates().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    assert isinstance(next_node, ScoreLoveliness)

    next_node = await ScoreLoveliness().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    assert isinstance(next_node, ScoreLikeliness)

    next_node = await ScoreLikeliness().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    assert isinstance(next_node, SelectBestExplanation)

    # Crucial: the claim's integration state was never populated
    # because the filter excluded it.
    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.integrated_assessment is None
    assert not reloaded.integration_candidates


async def test_non_cycle_capped_supported_still_enters_ibe(
    tmp_path: Path, fake_runner
) -> None:
    """Sanity: the cycle-cap exclusion must not break the normal
    IBE-eligible path. A regular SUPPORTED + no-verdict claim should
    still flow through ClusterEvidence → ... → IBE."""
    from andamentum.epistemic.graph.nodes import ClusterEvidence

    store = DocumentStore.for_database("ibe_normal", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="normal supported claim",
        scope="x",
        stage=ClaimStage.SUPPORTED,
        cycle_capped=False,  # NOT cycle-capped
        integrated_assessment=None,
        scrutiny_verdict="pass",
    )
    await repo.save(claim)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(
        repo=repo, agent_runner=fake_runner, embedding_model="t"
    )

    next_node = await PromoteToSupported().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    # Should route to ClusterEvidence (normal IBE-bound path).
    assert isinstance(next_node, ClusterEvidence), (
        f"Got {type(next_node).__name__}; non-cycle-capped SUPPORTED "
        "claim should route to ClusterEvidence to enter the verification "
        "→ IBE chain. The cycle-cap fix must not break this."
    )
