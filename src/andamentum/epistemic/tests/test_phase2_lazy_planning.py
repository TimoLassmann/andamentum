"""Tests for Phase 2 of the lazy-escalation plan.

Phase 2 changes ``PlanTaskOperation`` so that in multi-seed-claim
mode, round 1 picks ONE provider per sub-claim (via ``epistemic_rank_providers``)
instead of querying all relevant providers in parallel. Phase 3 (a
follow-up) will then have ``InvestigateClaimOperation`` consume demand
from scrutiny and pick the next-most-promising UNUSED provider.

These tests pin:
  1. Per-sub-claim evidence stubs are produced one-per-sub (not
     N-per-sub × M-providers).
  2. The chosen provider matches what the ranker agent returned.
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
from andamentum.epistemic.operations.preplanning import PlanTaskOperation
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


async def test_phase2_creates_one_stub_per_sub_claim(
    tmp_path: Path, fake_runner
) -> None:
    """The defining behaviour of Phase 2: one Evidence stub per
    sub-claim in round 1, not N stubs (one per relevant provider).

    Pre-Phase-2: 4 sub-claims × ~6 relevant providers = ~24 stubs.
    Post-Phase-2: 4 sub-claims × 1 chosen provider = 4 stubs.

    Cost reduction: query formulation drops from ~24 LLM calls to
    ~4, plus the downstream extraction + scoring + judging chain
    is bounded proportionally.
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
    result = await op.execute(work)
    assert result.success

    # Exactly N evidence stubs total (one per sub-claim).
    all_evidence = await repo.query("evidence", objective_id=obj.entity_id)
    assert len(all_evidence) == 3, (
        f"Expected 1 evidence stub per sub-claim (3 total); got "
        f"{len(all_evidence)}. Phase 2 round-1 narrowing isn't firing."
    )


async def test_phase2_each_sub_gets_its_own_stub_with_correct_id(
    tmp_path: Path, fake_runner
) -> None:
    """Each sub-claim's stub is tagged with the right sub_investigation_id
    (so the per-claim filter in MultiSeedClaim still works) and the
    provider matches what the ranker chose."""
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

    # Conftest fake_runner returns chosen_provider="web_search" by
    # default (see test_phase2 default in conftest._FAKE_DEFAULTS).
    for ev in all_evidence:
        assert ev.source_type == "web_search", (
            f"Stub for sub {ev.sub_investigation_id} has source_type="
            f"{ev.source_type!r}; conftest fake_runner returns "
            "'web_search' for epistemic_rank_providers, so all stubs "
            "should reflect that choice."
        )


async def test_phase2_fallback_when_no_agent_runner(tmp_path: Path) -> None:
    """When agent_runner is None, the operation can't run the
    ranker. It should fall back to the first available provider —
    pipeline still alive but lazy-escalation benefit lost for that
    run. This is the documented fallback path."""
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
    # Without agent_runner, the relevance filter doesn't run either —
    # only "web_search" survives as the universal fallback. So one
    # stub per sub-claim (not multiple).
    assert len(all_evidence) == 2
