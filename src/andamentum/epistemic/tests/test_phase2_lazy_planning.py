"""Tests for the K8 Bug #1 fix — provider tournament wiring in PlanTask.

Previously these tests pinned "one Evidence stub per sub-claim in
round 1" (the lazy-escalation Phase 2 behaviour from 2026-05-02).
That behaviour was changed by the K8 Bug #1 fix on 2026-05-05
(see docs/superpowers/plans/2026-05-05-k8-bug1-provider-tournament.md):
PlanTask now picks K=RESEARCH_MODE_PROVIDER_K providers via an
iterative tournament and creates one stub per (sub-claim, provider)
pair. The lazy-escalation principle (round 1 narrow, escalate on
demand) is preserved — round 1 still narrows to a small subset of
providers, just K of them rather than 1, to enable per-sub-claim
cross-domain convergence detection.

These tests pin the post-fix behaviour:
  1. Stubs created in round 1 = K × N (K providers per N sub-claims).
  2. Each sub_id appears K times across stubs, once per picked provider.
  3. The fallback path (no agent_runner) still produces stubs.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Objective
from andamentum.epistemic.entities.decomposition import (
    Decomposition,
    SubInvestigation,
)
from andamentum.epistemic.operations.base import OperationInput
from andamentum.epistemic.operations.preplanning import (
    RESEARCH_MODE_PROVIDER_K,
    PlanTaskOperation,
)
from andamentum.epistemic.repository import EpistemicRepository


async def _make_objective_with_decomposition(
    tmp_path: Path, n_subs: int = 3
) -> tuple[Objective, EpistemicRepository]:
    store = DocumentStore.for_database("phase2_planning", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)
    obj = Objective(
        description="parent question",
        clarified_question="parent question (clarified)",
        question_type="verificatory",
        phase="analyzed",
        decomposition=Decomposition(
            sub_investigations=[
                SubInvestigation(
                    id=chr(ord("A") + i),
                    seed_claim=f"Sub-claim {chr(ord('A') + i)} statement",
                    rationale="r",
                )
                for i in range(n_subs)
            ],
            combination_rule="AND",
            rationale="r",
        ),
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    return obj, repo


async def test_round_1_creates_k_stubs_per_sub_claim(
    tmp_path: Path, fake_runner
) -> None:
    """The defining behaviour of the K8 Bug #1 fix: one Evidence stub
    per (sub-claim, provider) pair, where K providers are picked
    once via iterative tournament for the whole objective.

    Pre-K8-fix (lazy-escalation Phase 2): 4 sub-claims × 1 chosen
    provider = 4 stubs. Same provider for every sub-claim due to the
    ranker collapse bug.

    Post-K8-fix: 4 sub-claims × K (=2 by default) providers = 8 stubs.
    The K providers come from a per-objective tournament; every
    sub-claim queries all of them. This unlocks per-sub-claim
    cross-domain convergence detection.
    """
    n_subs = 3
    obj, repo = await _make_objective_with_decomposition(tmp_path, n_subs=n_subs)

    op = PlanTaskOperation(
        repo=repo,
        agent_runner=fake_runner,
        embedding_model="t",
    )
    work = OperationInput(
        entity_id=obj.entity_id,
        entity_type="objective",
        operation="plan_task",
    )
    result = await op.execute(work)
    assert result.success

    # Exactly K * N evidence stubs total (K providers × N sub-claims).
    all_evidence = await repo.query("evidence", objective_id=obj.entity_id)
    expected = RESEARCH_MODE_PROVIDER_K * n_subs
    assert len(all_evidence) == expected, (
        f"Expected K * N = {RESEARCH_MODE_PROVIDER_K} × {n_subs} = "
        f"{expected} stubs (one per (sub-claim, provider) pair); got "
        f"{len(all_evidence)}. The K8 tournament wiring isn't producing "
        "the right shape."
    )


async def test_each_sub_gets_k_stubs_with_correct_id_and_distinct_providers(
    tmp_path: Path, fake_runner
) -> None:
    """Each sub-claim's K stubs are tagged with the right
    sub_investigation_id (so the per-claim filter in MultiSeedClaim
    still works), and the K providers across them are distinct (the
    tournament guarantee — once a provider is picked, it's removed
    from the pool, so the second pick is necessarily different).
    """
    obj, repo = await _make_objective_with_decomposition(tmp_path, n_subs=3)

    op = PlanTaskOperation(
        repo=repo,
        agent_runner=fake_runner,
        embedding_model="t",
    )
    work = OperationInput(
        entity_id=obj.entity_id,
        entity_type="objective",
        operation="plan_task",
    )
    await op.execute(work)

    all_evidence = await repo.query("evidence", objective_id=obj.entity_id)
    sub_ids = {ev.sub_investigation_id for ev in all_evidence}
    assert sub_ids == {"A", "B", "C"}, (
        f"sub_investigation_ids on stubs should match the decomposition "
        f"({{'A','B','C'}}); got {sub_ids}"
    )

    # Group stubs by sub_id and check each group has K distinct providers.
    by_sub: dict[str, set[str]] = {}
    for ev in all_evidence:
        by_sub.setdefault(ev.sub_investigation_id, set()).add(ev.source_type)
    for sub_id, provider_set in by_sub.items():
        assert len(provider_set) == RESEARCH_MODE_PROVIDER_K, (
            f"Sub-claim {sub_id} has providers {sorted(provider_set)}; "
            f"expected exactly {RESEARCH_MODE_PROVIDER_K} distinct providers "
            f"(tournament should guarantee no duplicates within a sub-claim)."
        )

    # All sub-claims should get the SAME set of K providers (per-objective
    # tournament — every sub queries the same K).
    provider_sets_seen = {frozenset(s) for s in by_sub.values()}
    assert len(provider_sets_seen) == 1, (
        "All sub-claims should query the same K providers; got divergent "
        f"sets: {provider_sets_seen}"
    )


async def test_fallback_when_no_agent_runner(tmp_path: Path) -> None:
    """When agent_runner is None, the operation can't run the
    ranker. The tournament path is skipped; the fallback uses the
    first ``RESEARCH_MODE_PROVIDER_K`` candidates from the providers
    list. Pipeline still alive but the diversification benefit is
    lost for that run.

    Without agent_runner, no select_provider step runs, so the only
    provider that survives is ``web_search`` (the universal
    fallback). picked_providers[:K] then yields just web_search; one
    stub per sub-claim.
    """
    obj, repo = await _make_objective_with_decomposition(tmp_path, n_subs=2)

    op = PlanTaskOperation(
        repo=repo,
        agent_runner=None,  # No runner
        embedding_model="t",
    )
    work = OperationInput(
        entity_id=obj.entity_id,
        entity_type="objective",
        operation="plan_task",
    )
    result = await op.execute(work)
    assert result.success

    all_evidence = await repo.query("evidence", objective_id=obj.entity_id)
    # Without agent_runner only ``web_search`` is in providers; the K=2
    # tournament fallback clips to whatever's available, so all stubs
    # use web_search and we get exactly N (one per sub).
    assert len(all_evidence) == 2
    for ev in all_evidence:
        assert ev.source_type == "web_search"


async def test_verify_mode_skips_per_provider_relevance_check(
    tmp_path: Path, fake_runner
) -> None:
    """Phase B: in verify mode (claim_to_verify set), PlanTaskOperation
    must NOT run the per-provider ``epistemic_select_provider`` LLM
    check. That check was a dominant variance source in 5-rep smokes
    (different runs rolled different yes/no decisions, producing
    wildly different provider mixes for the same claim). Instead use
    all configured providers; let the existing dedup-driven
    ``corroboration_count`` carry vouching signal downstream.

    Test: an Objective with claim_to_verify set goes through PlanTask
    and creates one Evidence stub per registered provider, regardless
    of what the per-provider relevance LLM might have decided.
    """
    from andamentum.epistemic.providers import PROVIDER_REGISTRY

    store = DocumentStore.for_database("verify_mode_planning", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    # Verify-mode objective: claim_to_verify set, no decomposition.
    obj = Objective(
        description="Test claim X.",
        clarified_question="Test claim X.",
        question_type="verificatory",
        claim_to_verify="Test claim X.",
        phase="analyzed",
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    op = PlanTaskOperation(
        repo=repo,
        agent_runner=fake_runner,
        embedding_model="t",
    )
    result = await op.execute(
        OperationInput(
            entity_id=obj.entity_id,
            entity_type="objective",
            operation="plan_task",
        )
    )
    assert result.success

    all_evidence = await repo.query("evidence", objective_id=obj.entity_id)
    # All registered providers should be represented (web_search is
    # always added if not already present, but it should also be in
    # PROVIDER_REGISTRY for this assertion to be meaningful).
    seen_providers = {e.source_type for e in all_evidence}
    expected_providers = set(PROVIDER_REGISTRY) | {"web_search"}
    assert seen_providers == expected_providers, (
        f"Verify mode should query ALL configured providers; got "
        f"{seen_providers}, expected {expected_providers}"
    )

    # Verify the operations log does NOT contain
    # epistemic_select_provider calls (the per-provider relevance LLM
    # check). The fake runner records every agent call; in verify mode
    # this agent should be skipped entirely.
    select_calls = [
        c
        for c in op._agent_calls
        if c.get("agent_name") == "epistemic_select_provider"
    ]
    assert select_calls == [], (
        f"Verify mode must skip epistemic_select_provider (per-provider "
        f"relevance check); found {len(select_calls)} calls"
    )
