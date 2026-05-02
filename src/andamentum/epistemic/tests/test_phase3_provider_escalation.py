"""Tests for Phase 3 of the lazy-escalation plan.

Phase 3 changes ``InvestigateClaimOperation`` so that when the
inquiry loop comes back around (scrutiny said "needs_resolution"),
investigation pulls the NEXT unused provider for the sub-claim
instead of regenerating queries against the same providers.

Round 1 (Phase 2): one provider per sub-claim.
Round 2+ (Phase 3): each round adds a different provider, escalating
breadth on demand.

These tests pin:
  1. When unused providers exist, the investigation operation runs
     the ranker and tags new stubs with the ranker's choice.
  2. When only one unused provider remains, no rank LLM call is
     needed (cheap pre-filter).
  3. When all providers are used, the operation falls back to the
     existing behavior (agent picks source_type freely).
"""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, Evidence, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.operations.base import OperationInput
from andamentum.epistemic.operations.investigation import (
    InvestigateClaimOperation,
)
from andamentum.epistemic.repository import EpistemicRepository
from andamentum.epistemic.tests.conftest import FakeAgentRunner


async def _setup_claim_with_evidence_from_providers(
    tmp_path: Path,
    used_providers: list[str],
) -> tuple[Claim, EpistemicRepository]:
    """Build a Claim with one Evidence item per ``used_provider``,
    simulating "round 1 already happened with these providers"."""
    store = DocumentStore.for_database("phase3_escalation", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)
    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="claim about mortality and intermittent fasting",
        scope="x",
        stage=ClaimStage.HYPOTHESIS,
        scrutiny_verdict="needs_resolution",
        sub_investigation_id="A",
    )
    await repo.save(claim)

    for prov in used_providers:
        ev = Evidence(
            objective_id=obj.entity_id,
            source_type=prov,
            source_ref=f"existing_{prov}_stub",
            extracted=True,
            extracted_content=f"Some content from {prov}",
            sub_investigation_id="A",
        )
        await repo.save(ev)
        claim.evidence_ids.append(ev.entity_id)

    claim.evidence_count = len(claim.evidence_ids)
    await repo.save(claim)
    return claim, repo


async def test_phase3_picks_unused_provider_when_multiple_remain(
    tmp_path: Path,
) -> None:
    """When the claim has been queried against some providers but
    not others, the operation runs the ranker on UNUSED providers
    and tags new stubs with the chosen one — not whatever the
    investigation agent picked."""
    # Round 1 used pubmed and openalex; many other providers unused.
    claim, repo = await _setup_claim_with_evidence_from_providers(
        tmp_path, used_providers=["pubmed", "openalex"]
    )

    # Override the ranker to return "cochrane" — different from
    # what the investigation agent would pick on its own.
    runner = FakeAgentRunner(
        overrides={
            "epistemic_rank_providers": {
                "chosen_provider": "cochrane",
                "reasoning": "Cochrane reviews for clinical RCT questions.",
            },
            "epistemic_investigate_claim": {
                # The investigation agent normally picks source_type
                # per-query (typically web_search). Phase 3 overrides
                # this with the ranker's choice.
                "evidence_queries": [
                    {
                        "source_type": "web_search",  # would be picked
                        "query": "intermittent fasting mortality cochrane",
                    },
                    {
                        "source_type": "web_search",
                        "query": "intermittent fasting RCT all-cause mortality",
                    },
                ],
                "reasoning": "Targeted search for clinical RCT evidence.",
            },
        }
    )

    op = InvestigateClaimOperation(
        repo=repo,
        agent_runner=runner,
        embedding_model="t",
    )
    work = OperationInput(
        entity_id=claim.entity_id,
        entity_type="claim",
        operation="investigate_claim",
    )
    result = await op.execute(work)
    assert result.success

    # The new stubs created by Investigate should be tagged with
    # "cochrane" (the ranker's choice), NOT "web_search" (the
    # investigation agent's per-query default).
    new_stubs = await repo.query(
        "evidence", objective_id=claim.objective_id, extracted=False
    )
    assert len(new_stubs) >= 1, (
        "Investigation should have created at least one new stub."
    )
    for stub in new_stubs:
        assert stub.source_type == "cochrane", (
            f"New stub has source_type={stub.source_type!r}; expected "
            "'cochrane' (the ranker's choice). Phase 3 should override "
            "the investigation agent's per-query source_type with the "
            "ranker's pick."
        )
        assert stub.sub_investigation_id == "A"


async def test_phase3_no_rank_call_when_one_unused_provider(
    tmp_path: Path,
) -> None:
    """When only one provider is unused, no need to run the ranker
    LLM — there's nothing to choose between. The operation should
    pick the single unused provider directly."""
    from andamentum.epistemic.providers import PROVIDER_REGISTRY

    # Use all but one provider, so only that one remains unused.
    all_providers = sorted(PROVIDER_REGISTRY)
    used = all_providers[:-1]
    expected_unused = all_providers[-1]

    claim, repo = await _setup_claim_with_evidence_from_providers(
        tmp_path, used_providers=used
    )

    # Ranker shouldn't fire; if it does, return wrong answer to
    # detect.
    runner = FakeAgentRunner(
        overrides={
            "epistemic_rank_providers": {
                "chosen_provider": "should_not_fire",
                "reasoning": "test sentinel",
            },
            "epistemic_investigate_claim": {
                "evidence_queries": [
                    {"source_type": "web_search", "query": "q"}
                ],
                "reasoning": "r",
            },
        }
    )

    op = InvestigateClaimOperation(
        repo=repo,
        agent_runner=runner,
        embedding_model="t",
    )
    work = OperationInput(
        entity_id=claim.entity_id,
        entity_type="claim",
        operation="investigate_claim",
    )
    await op.execute(work)

    new_stubs = await repo.query(
        "evidence", objective_id=claim.objective_id, extracted=False
    )
    assert len(new_stubs) >= 1
    for stub in new_stubs:
        assert stub.source_type == expected_unused, (
            f"With only one unused provider ({expected_unused!r}) "
            f"available, the operation should pick it directly without "
            f"a rank LLM call. Got source_type={stub.source_type!r}."
        )


async def test_phase3_fallback_when_all_providers_used(
    tmp_path: Path,
) -> None:
    """When the claim has been queried against every provider in
    the registry, no unused providers remain. The operation falls
    back to the existing behavior: agent picks source_type per
    query freely."""
    from andamentum.epistemic.providers import PROVIDER_REGISTRY

    all_providers = sorted(PROVIDER_REGISTRY)
    claim, repo = await _setup_claim_with_evidence_from_providers(
        tmp_path, used_providers=all_providers
    )

    runner = FakeAgentRunner(
        overrides={
            # Ranker shouldn't fire; sentinel value if it does.
            "epistemic_rank_providers": {
                "chosen_provider": "should_not_fire",
                "reasoning": "test sentinel",
            },
            "epistemic_investigate_claim": {
                "evidence_queries": [
                    # Agent's choice — should be honored when no
                    # unused providers remain (fallback path).
                    {"source_type": "web_search", "query": "fallback query"}
                ],
                "reasoning": "Last-ditch web search.",
            },
        }
    )

    op = InvestigateClaimOperation(
        repo=repo,
        agent_runner=runner,
        embedding_model="t",
    )
    work = OperationInput(
        entity_id=claim.entity_id,
        entity_type="claim",
        operation="investigate_claim",
    )
    await op.execute(work)

    new_stubs = await repo.query(
        "evidence", objective_id=claim.objective_id, extracted=False
    )
    assert len(new_stubs) >= 1
    for stub in new_stubs:
        # Agent's choice is honored — its source_type was "web_search".
        assert stub.source_type == "web_search", (
            f"With all providers used, the operation should fall back "
            f"to the agent's source_type choice. Got {stub.source_type!r}."
        )
