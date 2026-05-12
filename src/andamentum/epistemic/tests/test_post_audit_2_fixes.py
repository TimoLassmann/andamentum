"""Regression tests for the post-audit-2 fix queue.

After the post-audit-1 fix queue (commits c1eb6fb..7963bf9) shipped, a
second audit found three more bugs:

* Bug A: Investigation evidence drops sub_investigation_id (parallel to
  the Commit-A gatherer-extras fix; one site got missed).
* Bug C: combination_rule lookup divergence between compute_posterior
  and CombineClaimVerdicts (one read both objective.combination_rule
  and decomposition['combination_rule']; the other read only the field).
* Bug D: Degenerate decomposition silent dead-end (MultiSeedClaim
  mints 0 claims → empty inquiry, silent empty report).

This file pins the regressions for all three.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, Evidence, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.graph.combination import resolve_combination_rule
from andamentum.epistemic.operations.base import OperationInput
from andamentum.epistemic.operations.investigation import (
    InvestigateClaimOperation,
)
from andamentum.epistemic.repository import EpistemicRepository


# ── Bug A: investigation propagates sub_investigation_id ──────────────


class TestInvestigationPropagatesSubInvestigationId:
    """Per-claim tagging invariants survive the routing-unification rewrite.

    Investigation rounds now go through ``dispatch_and_persist_for_text``
    (the shared routing+persistence helper), but the per-claim tags
    (``depends_on_claim_id``, ``sub_investigation_id``) must still be
    set on every Evidence the operation creates. We monkeypatch the
    helper to a thin recorder so the test focuses on tag propagation
    rather than re-testing dispatch.
    """

    @staticmethod
    def _patch_helper(monkeypatch) -> list[dict]:
        """Replace dispatch_and_persist_for_text with a recorder that
        persists one Evidence per call (so the test's invariants run
        against actual saved entities). Returns the call log."""
        from andamentum.epistemic.entities import Evidence
        from andamentum.epistemic.operations import investigation as inv_mod

        calls: list[dict] = []

        async def fake_helper(
            op, text, *, objective_id, providers, core_runner,
            sub_investigation_id=None, depends_on_claim_id=None,
            created_by="dispatch",
        ):
            calls.append({
                "text": text,
                "sub_id": sub_investigation_id,
                "depends_on": depends_on_claim_id,
            })
            ev = Evidence(
                objective_id=objective_id,
                source_type="stub",
                source_ref=f"stub-ref-{len(calls)}",
                extracted=True,
                extracted_content="stub content",
                sub_investigation_id=sub_investigation_id,
                depends_on_claim_id=depends_on_claim_id,
                quality_score=0.5,
                created_by=created_by,
            )
            await op.repo.save(ev)
            return [ev.entity_id]

        monkeypatch.setattr(inv_mod, "dispatch_and_persist_for_text", fake_helper)
        return calls

    async def test_investigation_evidence_inherits_sub_id(
        self, tmp_path: Path, fake_runner, monkeypatch
    ) -> None:
        """When InvestigateClaimOperation runs on a multi-seed-claim
        Claim (carrying sub_investigation_id), the Evidence it creates
        must inherit the same sub_investigation_id."""
        self._patch_helper(monkeypatch)

        store = DocumentStore.for_database("inv_sub_id", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(
            description="parent",
            question_type="verificatory",
        )
        obj.objective_id = obj.entity_id
        await repo.save(obj)
        claim = Claim(
            objective_id=obj.entity_id,
            statement="claim under sub A",
            scope="x",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="needs_resolution",
            sub_investigation_id="A",
        )
        await repo.save(claim)

        op = InvestigateClaimOperation(
            repo,
            fake_runner,
            embedding_model="t",
            providers={"stub": object()},
        )
        result = await op.execute(
            OperationInput(
                entity_id=claim.entity_id,
                entity_type="claim",
                operation="investigate_claim",
            )
        )
        assert result.success
        all_evidence = await repo.query("evidence", objective_id=obj.entity_id)
        assert all_evidence
        for ev in all_evidence:
            assert ev.depends_on_claim_id == claim.entity_id
            assert ev.sub_investigation_id == "A", (
                f"Evidence {ev.entity_id[:8]} created during investigation "
                f"on a sub-A claim has sub_investigation_id="
                f"{ev.sub_investigation_id!r}, expected 'A'"
            )

    async def test_investigation_no_sub_id_when_claim_has_none(
        self, tmp_path: Path, fake_runner, monkeypatch
    ) -> None:
        """Sanity: when the originating Claim has no sub_investigation_id
        (open-research mode), investigation evidence's sub_id stays None."""
        self._patch_helper(monkeypatch)

        store = DocumentStore.for_database("inv_no_sub", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(description="open research", question_type="verificatory")
        obj.objective_id = obj.entity_id
        await repo.save(obj)
        claim = Claim(
            objective_id=obj.entity_id,
            statement="open-research claim",
            scope="x",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="needs_resolution",
            sub_investigation_id=None,
        )
        await repo.save(claim)

        op = InvestigateClaimOperation(
            repo,
            fake_runner,
            embedding_model="t",
            providers={"stub": object()},
        )
        await op.execute(
            OperationInput(
                entity_id=claim.entity_id,
                entity_type="claim",
                operation="investigate_claim",
            )
        )
        all_evidence = await repo.query("evidence", objective_id=obj.entity_id)
        assert all_evidence
        for ev in all_evidence:
            assert ev.sub_investigation_id is None


# ── Bug C: resolve_combination_rule unified lookup ────────────────────


class TestCombinationRuleResolution:
    def test_reads_decomposition_rule(self) -> None:
        obj = Objective(
            description="x",
            decomposition={"combination_rule": "WEIGHTED_AND"},
        )
        assert resolve_combination_rule(obj) == "WEIGHTED_AND"

    def test_returns_none_when_no_decomposition(self) -> None:
        obj = Objective(description="x")
        assert resolve_combination_rule(obj) is None

    def test_returns_none_when_decomposition_lacks_rule(self) -> None:
        obj = Objective(
            description="x",
            decomposition={"sub_investigations": []},
        )
        assert resolve_combination_rule(obj) is None


# ── Bug D: degenerate decomposition falls back to ProposeClaims ───────


class TestDegenerateDecompositionFallback:
    async def test_zero_subs_falls_back_to_propose_claims(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """When DecomposeQuestion produces 0 sub_investigations (or all
        seed_claims are empty), MultiSeedClaim mints 0 claims. CreateClaims
        must detect this and fall back to ProposeClaims so the inquiry
        doesn't end up with zero claims."""
        from andamentum.epistemic.graph.deps import EpistemicDeps
        from andamentum.epistemic.graph.nodes import CreateClaims
        from andamentum.epistemic.graph.state import EpistemicGraphState

        store = DocumentStore.for_database("degenerate_decomp", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        # Pre-extracted evidence so ProposeClaims has something to chew on.
        obj = Objective(
            description="parent",
            question_type="verificatory",
            phase="claims_proposed",
            decomposition={
                "sub_investigations": [],  # Empty — degenerate
                "combination_rule": "AND",
                "rationale": "trivial",
            },
        )
        obj.objective_id = obj.entity_id
        await repo.save(obj)
        # Add some extracted evidence so ProposeClaims (the fallback)
        # has material.
        for i in range(3):
            ev = Evidence(
                objective_id=obj.entity_id,
                source_type="web",
                source_ref=f"https://ex.com/{i}",
                extracted_content=f"content {i}",
                extracted=True,
                support_judgment="supports",
            )
            await repo.save(ev)

        class _FakeRunContext:
            def __init__(self, state, deps):
                self.state = state
                self.deps = deps

        state = EpistemicGraphState(objective_id=obj.entity_id)
        deps = EpistemicDeps(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )
        ctx = _FakeRunContext(state, deps)
        await CreateClaims().run(ctx)  # type: ignore[arg-type]

        # CreateClaims should have run MultiSeedClaim (0 claims), then
        # detected the empty result and run ProposeClaims as fallback.
        ms_calls = [
            op for op in state.operations_log if op["operation"] == "multi_seed_claim"
        ]
        pc_calls = [
            op for op in state.operations_log if op["operation"] == "propose_claims"
        ]
        assert len(ms_calls) == 1
        assert len(pc_calls) == 1, (
            "Expected ProposeClaims fallback after MultiSeedClaim minted "
            "no claims; got "
            f"{len(pc_calls)} ProposeClaims calls"
        )
        # And state.claims_created flips True only after the fallback
        # produced claims.
        assert state.claims_created is True

    async def test_non_degenerate_decomposition_no_fallback(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """Sanity: when MultiSeedClaim DOES mint claims, ProposeClaims
        must NOT run as a fallback."""
        from andamentum.epistemic.graph.deps import EpistemicDeps
        from andamentum.epistemic.graph.nodes import CreateClaims
        from andamentum.epistemic.graph.state import EpistemicGraphState

        store = DocumentStore.for_database("nondegen_decomp", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(
            description="parent",
            question_type="verificatory",
            phase="claims_proposed",
            decomposition={
                "sub_investigations": [
                    {"id": "A", "seed_claim": "alpha", "rationale": "r", "weight": 1.0},
                    {"id": "B", "seed_claim": "beta", "rationale": "r", "weight": 1.0},
                ],
                "combination_rule": "AND",
                "rationale": "both must hold",
            },
        )
        obj.objective_id = obj.entity_id
        await repo.save(obj)
        # Add some sub-tagged evidence so MultiSeedClaim has material per
        # claim.
        for sub_id in ("A", "B"):
            ev = Evidence(
                objective_id=obj.entity_id,
                source_type="web",
                source_ref=f"https://ex.com/{sub_id}",
                extracted_content=f"content {sub_id}",
                extracted=True,
                sub_investigation_id=sub_id,
            )
            await repo.save(ev)

        class _FakeRunContext:
            def __init__(self, state, deps):
                self.state = state
                self.deps = deps

        state = EpistemicGraphState(objective_id=obj.entity_id)
        deps = EpistemicDeps(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )
        ctx = _FakeRunContext(state, deps)
        await CreateClaims().run(ctx)  # type: ignore[arg-type]

        ms_calls = [
            op for op in state.operations_log if op["operation"] == "multi_seed_claim"
        ]
        pc_calls = [
            op for op in state.operations_log if op["operation"] == "propose_claims"
        ]
        assert len(ms_calls) == 1
        assert len(pc_calls) == 0, (
            "ProposeClaims should not fire when MultiSeedClaim minted "
            f"claims; got {len(pc_calls)} unwanted ProposeClaims calls"
        )
