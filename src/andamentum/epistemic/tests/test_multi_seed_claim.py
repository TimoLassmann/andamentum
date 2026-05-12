"""Tests for MultiSeedClaimOperation and the per-claim PlanEvidence flow.

Phase 1 of the multi-seed-claim refactor: replace decomposition spawning
with N Claims minted on a single Objective, each with its own evidence
subset (Option 2 from the architectural plan).

Covers:
- PlanTaskOperation in multi-seed mode emits Evidence stubs tagged with
  sub_investigation_id (one stub per provider × sub-investigation).
- MultiSeedClaimOperation mints one Claim per sub-investigation, links
  only the Evidence whose sub_investigation_id matches, judges per-claim.
- Idempotence: re-running on a parent that already has claims for some
  sub-investigations skips them (delta-mint for reflection growth).
- CreateClaims graph node dispatches to MultiSeedClaim when decomposition
  is set.
- Entity round-trip for the new sub_investigation_id fields.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, Evidence, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import CreateClaims
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.operations.base import OperationInput
from andamentum.epistemic.operations.multi_seed_claim import (
    MultiSeedClaimOperation,
)
from andamentum.epistemic.repository import EpistemicRepository


class _FakeRunContext:
    def __init__(self, state: EpistemicGraphState, deps: EpistemicDeps) -> None:
        self.state = state
        self.deps = deps


async def _setup_objective_with_decomposition(
    tmp_path: Path,
    db_name: str,
    *,
    n_subs: int = 3,
) -> tuple[Objective, EpistemicRepository]:
    """Build a parent Objective post-DecomposeQuestion: decomposition is
    set with n_subs sub-investigations, phase=analyzed, no claims yet."""
    store = DocumentStore.for_database(db_name, db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)
    sub_investigations = []
    for i in range(n_subs):
        sub_id = chr(ord("A") + i)
        sub_investigations.append(
            {
                "id": sub_id,
                "seed_claim": f"seed claim for sub {sub_id}",
                "rationale": f"rationale for sub {sub_id}",
                "weight": 1.0,
            }
        )
    obj = Objective(
        description="parent question",
        clarified_question="parent question (clarified)",
        question_type="explanatory",  # intentionally non-verificatory to mirror case 54
        phase="analyzed",
        decomposition={
            "sub_investigations": sub_investigations,
            "combination_rule": "AND",
            "rationale": "all must hold",
        },
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    return obj, repo


# ── Entity round-trip tests ───────────────────────────────────────────


class TestEvidenceSubInvestigationId:
    async def test_field_round_trips(self, tmp_path: Path) -> None:
        store = DocumentStore.for_database("ev_round", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(description="parent")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        ev = Evidence(
            objective_id=obj.entity_id,
            source_type="web_search",
            source_ref="https://example.com/A",
            sub_investigation_id="A",
        )
        await repo.save(ev)

        loaded = await repo.get("evidence", ev.entity_id)
        assert loaded.sub_investigation_id == "A"


class TestClaimSubInvestigationId:
    async def test_field_round_trips(self, tmp_path: Path) -> None:
        store = DocumentStore.for_database("cl_round", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(description="parent")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        claim = Claim(
            objective_id=obj.entity_id,
            statement="claim text",
            scope="scope",
            sub_investigation_id="B",
        )
        await repo.save(claim)

        loaded = await repo.get("claim", claim.entity_id)
        assert loaded.sub_investigation_id == "B"


# ── MultiSeedClaimOperation ───────────────────────────────────────────


class TestMultiSeedClaimMintsOnePerSub:
    async def test_mints_three_claims_per_three_sub_investigations(
        self, tmp_path: Path, fake_runner
    ) -> None:
        obj, repo = await _setup_objective_with_decomposition(
            tmp_path, "multi_three", n_subs=3
        )
        # Pre-populate evidence: each sub_id gets 2 evidence items.
        for sub_id in ("A", "B", "C"):
            for i in range(2):
                ev = Evidence(
                    objective_id=obj.entity_id,
                    source_type="web_search",
                    source_ref=f"https://ex.com/{sub_id}{i}",
                    extracted_content=f"content for {sub_id}-{i}",
                    extracted=True,
                    sub_investigation_id=sub_id,
                )
                await repo.save(ev)

        op = MultiSeedClaimOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test"
        )
        result = await op.execute(
            OperationInput(
                entity_id=obj.entity_id,
                entity_type="objective",
                operation="multi_seed_claim",
            )
        )
        assert result.success
        assert len(result.created_entities) == 3

        # Verify each claim is linked only to its own sub_investigation's
        # evidence — the per-claim evidence pool semantic.
        claims = await repo.query("claim", objective_id=obj.entity_id)
        by_sub: dict[str, Claim] = {
            c.sub_investigation_id: c
            for c in claims
            if c.sub_investigation_id is not None
        }
        assert set(by_sub.keys()) == {"A", "B", "C"}
        for sub_id, claim in by_sub.items():
            assert len(claim.evidence_ids) == 2
            for eid in claim.evidence_ids:
                ev = await repo.get("evidence", eid)
                assert ev.sub_investigation_id == sub_id

    async def test_judges_each_evidence_against_its_claim(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """Per-claim judgment: each Evidence's support_judgment is set
        relative to the specific Claim it was linked to."""
        obj, repo = await _setup_objective_with_decomposition(
            tmp_path, "multi_judge", n_subs=2
        )
        for sub_id in ("A", "B"):
            ev = Evidence(
                objective_id=obj.entity_id,
                source_type="web_search",
                source_ref=f"https://ex.com/{sub_id}",
                extracted_content=f"content for {sub_id}",
                extracted=True,
                sub_investigation_id=sub_id,
            )
            await repo.save(ev)

        op = MultiSeedClaimOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test"
        )
        await op.execute(
            OperationInput(
                entity_id=obj.entity_id,
                entity_type="objective",
                operation="multi_seed_claim",
            )
        )

        all_evidence = await repo.query("evidence", objective_id=obj.entity_id)
        # fake_runner's epistemic_judge_evidence default returns "supports".
        for ev in all_evidence:
            assert ev.support_judgment == "supports"

    async def test_idempotent_skips_existing_sub_investigations(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """Re-running MultiSeedClaim on a parent that already has claims
        for sub A skips A and only mints B, C. Supports reflection
        adding new sub-investigations to a partly-minted decomposition."""
        obj, repo = await _setup_objective_with_decomposition(
            tmp_path, "multi_idempotent", n_subs=3
        )
        # Pre-create a Claim for sub A.
        existing = Claim(
            objective_id=obj.entity_id,
            statement="seed claim for sub A",
            scope="rationale for sub A",
            stage=ClaimStage.HYPOTHESIS,
            sub_investigation_id="A",
        )
        await repo.save(existing)

        op = MultiSeedClaimOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test"
        )
        result = await op.execute(
            OperationInput(
                entity_id=obj.entity_id,
                entity_type="objective",
                operation="multi_seed_claim",
            )
        )
        assert result.success
        # Created 2 new claims (B, C); A was skipped.
        assert len(result.created_entities) == 2

        claims = await repo.query("claim", objective_id=obj.entity_id)
        sub_ids = {c.sub_investigation_id for c in claims}
        assert sub_ids == {"A", "B", "C"}

    async def test_all_existing_returns_did_work_false(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """If every sub-investigation already has a Claim, re-running is
        a did_work=False no-op."""
        obj, repo = await _setup_objective_with_decomposition(
            tmp_path, "multi_all_existing", n_subs=2
        )
        for sub_id in ("A", "B"):
            existing = Claim(
                objective_id=obj.entity_id,
                statement=f"seed claim for sub {sub_id}",
                scope=f"rationale for sub {sub_id}",
                stage=ClaimStage.HYPOTHESIS,
                sub_investigation_id=sub_id,
            )
            await repo.save(existing)

        op = MultiSeedClaimOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test"
        )
        result = await op.execute(
            OperationInput(
                entity_id=obj.entity_id,
                entity_type="objective",
                operation="multi_seed_claim",
            )
        )
        assert result.success
        assert result.did_work is False

    async def test_no_decomposition_fails_cleanly(
        self, tmp_path: Path, fake_runner
    ) -> None:
        store = DocumentStore.for_database("multi_nodecomp", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(description="parent", phase="analyzed")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        op = MultiSeedClaimOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test"
        )
        result = await op.execute(
            OperationInput(
                entity_id=obj.entity_id,
                entity_type="objective",
                operation="multi_seed_claim",
            )
        )
        assert result.success is False
        assert "decomposition" in result.message.lower()


# ── CreateClaims graph node dispatch ──────────────────────────────────


class TestCreateClaimsThirdBranch:
    async def test_decomposition_routes_to_multi_seed_claim(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """When obj has decomposition (no claim_to_verify), CreateClaims
        dispatches to MultiSeedClaim and N claims appear."""
        obj, repo = await _setup_objective_with_decomposition(
            tmp_path, "create_claims_multi", n_subs=2
        )
        # Some extracted evidence per sub-investigation.
        for sub_id in ("A", "B"):
            ev = Evidence(
                objective_id=obj.entity_id,
                source_type="web_search",
                source_ref=f"https://ex.com/{sub_id}",
                extracted_content=f"content for {sub_id}",
                extracted=True,
                sub_investigation_id=sub_id,
            )
            await repo.save(ev)

        state = EpistemicGraphState(objective_id=obj.entity_id)
        deps = EpistemicDeps(
            repo=repo, agent_runner=fake_runner, embedding_model="test"
        )
        ctx = _FakeRunContext(state, deps)
        await CreateClaims().run(ctx)  # type: ignore[arg-type]

        claims = await repo.query("claim", objective_id=obj.entity_id)
        assert len(claims) == 2
        sub_ids = {c.sub_investigation_id for c in claims}
        assert sub_ids == {"A", "B"}
        # Graph state advanced.
        assert state.claims_created is True
        assert len(state.claim_ids) == 2
