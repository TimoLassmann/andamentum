"""Regression tests for two pre-existing TMS issues exposed by the v0.3
CLI smoke run.

Issue 1 — TMS storm: ``_run_tms_sweep`` invoked ``InvalidateEvidenceOperation``
for every invalidated-but-not-cascaded evidence, even when the evidence
had no effect to cascade (no claim referenced it, no derived evidence
depended on it). Layer-1 dedup invalidations (duplicate URLs) trigger
this constantly. Fix: pre-filter no-op cascades and just flip
``invalidation_cascaded=True`` directly.

Issue 2 — Demote-can't-repromote trap: ``RevalidateClaimOperation``
demotes a claim from SUPPORTED to HYPOTHESIS via ``record_demotion``,
which CLEARS ``scrutiny_verdict`` to None. After the TMS sweep,
the graph proceeded to IBE / PromoteSupported, which both reject
HYPOTHESIS claims with ``verdict=None``. The graph never looped back
to Scrutinize, so the demoted claim sat at HYPOTHESIS forever.

Fix:
  (a) ``_run_tms_sweep`` step 2 adds demoted claims to
      ``state.claims_needing_rescrutiny``.
  (b) ``RunVerification`` A2 short-circuit checks the rescrutiny set
      first; if non-empty, falls through to ResolveUncertainties (which
      already routes to Scrutinize when the rescrutiny set is non-empty).
"""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, Evidence, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import _run_tms_sweep
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.repository import EpistemicRepository


# ── Helpers ───────────────────────────────────────────────────────────


async def _setup_repo(tmp_path: Path, name: str) -> EpistemicRepository:
    store = DocumentStore.for_database(name, db_dir=tmp_path)
    await store.initialize()
    return EpistemicRepository(store)


# ── Issue 1: no-op cascade fast-path ──────────────────────────────────


class TestNoOpCascadeFastPath:
    async def test_orphan_invalidated_evidence_skips_op(self, tmp_path: Path) -> None:
        """An invalidated evidence with no claim references and no
        derived dependencies should NOT trigger
        InvalidateEvidenceOperation — just flip
        invalidation_cascaded=True directly. Saves ~3 DB ops + 1 log
        per orphan."""
        repo = await _setup_repo(tmp_path, "tms_noop_fastpath")
        obj = Objective(description="parent", question_type="verificatory")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        # 5 orphan invalidated evidences (no claim refs them).
        for i in range(5):
            ev = Evidence(
                objective_id=obj.entity_id,
                source_type="web_search",
                source_ref=f"https://ex.com/{i}",
                extracted_content="x",
                extracted=True,
                invalidated=True,
                invalidation_reason="duplicate",
                invalidation_cascaded=False,
            )
            await repo.save(ev)

        state = EpistemicGraphState(objective_id=obj.entity_id)
        deps = EpistemicDeps(repo=repo, agent_runner=None)
        await _run_tms_sweep(deps, state)

        # All 5 evidences should now have invalidation_cascaded=True.
        all_evidence = await repo.query("evidence", objective_id=obj.entity_id)
        for ev in all_evidence:
            assert ev.invalidation_cascaded is True

        # No InvalidateEvidenceOperation calls should appear in
        # operations_log — the fast-path bypassed _run_op.
        invalidate_ops = [
            op
            for op in state.operations_log
            if op["operation"] == "invalidate_evidence"
        ]
        assert invalidate_ops == [], (
            f"Expected 0 invalidate_evidence ops, got {len(invalidate_ops)}"
        )

    async def test_referenced_evidence_still_uses_op(self, tmp_path: Path) -> None:
        """Sanity: when an invalidated evidence IS referenced by a claim,
        the op DOES fire (because there's real cascade work to do —
        removing it from the claim's evidence_ids)."""
        repo = await _setup_repo(tmp_path, "tms_referenced")
        obj = Objective(description="parent", question_type="verificatory")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        ev = Evidence(
            objective_id=obj.entity_id,
            source_type="web_search",
            source_ref="https://ex.com/r",
            extracted_content="content",
            extracted=True,
            invalidated=True,
            invalidation_reason="manually flagged",
            invalidation_cascaded=False,
        )
        await repo.save(ev)
        # Claim references the invalidated evidence.
        claim = Claim(
            objective_id=obj.entity_id,
            statement="referenced",
            scope="s",
            stage=ClaimStage.HYPOTHESIS,
            evidence_ids=[ev.entity_id],
            evidence_count=1,
        )
        await repo.save(claim)

        state = EpistemicGraphState(objective_id=obj.entity_id)
        deps = EpistemicDeps(repo=repo, agent_runner=None)
        await _run_tms_sweep(deps, state)

        # The op DID run (real cascade work needed).
        invalidate_ops = [
            op
            for op in state.operations_log
            if op["operation"] == "invalidate_evidence"
        ]
        assert len(invalidate_ops) == 1
        # And the claim's evidence_ids was properly cleaned.
        reloaded = await repo.get("claim", claim.entity_id)
        assert ev.entity_id not in reloaded.evidence_ids

    async def test_mixed_orphan_and_referenced(self, tmp_path: Path) -> None:
        """A mix: 3 orphan + 1 referenced. Only the referenced one
        triggers the op; orphans are fast-pathed."""
        repo = await _setup_repo(tmp_path, "tms_mixed")
        obj = Objective(description="parent", question_type="verificatory")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        # 3 orphans
        for i in range(3):
            ev = Evidence(
                objective_id=obj.entity_id,
                source_type="web",
                source_ref=f"orphan-{i}",
                extracted=True,
                invalidated=True,
                invalidation_cascaded=False,
            )
            await repo.save(ev)
        # 1 referenced
        ref_ev = Evidence(
            objective_id=obj.entity_id,
            source_type="web",
            source_ref="referenced",
            extracted=True,
            invalidated=True,
            invalidation_cascaded=False,
        )
        await repo.save(ref_ev)
        claim = Claim(
            objective_id=obj.entity_id,
            statement="x",
            scope="s",
            stage=ClaimStage.HYPOTHESIS,
            evidence_ids=[ref_ev.entity_id],
            evidence_count=1,
        )
        await repo.save(claim)

        state = EpistemicGraphState(objective_id=obj.entity_id)
        deps = EpistemicDeps(repo=repo, agent_runner=None)
        await _run_tms_sweep(deps, state)

        invalidate_ops = [
            op
            for op in state.operations_log
            if op["operation"] == "invalidate_evidence"
        ]
        # Only 1 op (for the referenced evidence). Pre-fix: 4 ops.
        assert len(invalidate_ops) == 1


# ── Issue 2: demote → rescrutiny → repromote ─────────────────────────


class TestDemoteAddsToRescrutiny:
    async def test_demote_adds_claim_to_rescrutiny_set(self, tmp_path: Path) -> None:
        """When TMS demotes a claim, it must also be added to
        ``state.claims_needing_rescrutiny`` so ResolveUncertainties
        routes back to Scrutinize before another promotion attempt."""
        repo = await _setup_repo(tmp_path, "tms_demote_rescrutiny")
        obj = Objective(description="parent", question_type="verificatory")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        # Set up a SUPPORTED claim with NO evidence (gate will fail
        # on revalidation because the SUPPORTED gate requires
        # min_evidence_weighted >= 1.0).
        claim = Claim(
            objective_id=obj.entity_id,
            statement="will be demoted",
            scope="s",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            evidence_ids=[],
            evidence_count=0,
        )
        await repo.save(claim)

        # Trigger TMS sweep with at least one invalidated evidence so
        # `had_cascades` flips True and revalidation runs.
        ev = Evidence(
            objective_id=obj.entity_id,
            source_type="web",
            source_ref="dummy",
            extracted=True,
            invalidated=True,
            invalidation_cascaded=False,
        )
        await repo.save(ev)

        state = EpistemicGraphState(objective_id=obj.entity_id)
        # Claim was previously verification-done; TMS should also discard it.
        state.verification_done.add(claim.entity_id)
        deps = EpistemicDeps(repo=repo, agent_runner=None)
        await _run_tms_sweep(deps, state)

        # The fix: TMS demote adds the claim to claims_needing_rescrutiny.
        assert claim.entity_id in state.claims_needing_rescrutiny
        # Sanity: also discarded from verification_done.
        assert claim.entity_id not in state.verification_done
        # Claim should now be HYPOTHESIS with cleared verdict.
        reloaded = await repo.get("claim", claim.entity_id)
        assert reloaded.stage == ClaimStage.HYPOTHESIS
        assert reloaded.scrutiny_verdict is None

    async def test_no_demote_does_not_touch_rescrutiny_set(
        self, tmp_path: Path
    ) -> None:
        """If revalidation finds the gate still passes (no demote), the
        rescrutiny set stays unchanged."""
        repo = await _setup_repo(tmp_path, "tms_no_demote")
        obj = Objective(description="parent", question_type="verificatory")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        # SUPPORTED claim with sufficient evidence to pass the gate.
        evidence_ids = []
        for i in range(3):
            ev = Evidence(
                objective_id=obj.entity_id,
                source_type="web",
                source_ref=f"ev-{i}",
                extracted_content="x",
                extracted=True,
                support_judgment="supports",
                quality_score=0.8,
            )
            await repo.save(ev)
            evidence_ids.append(ev.entity_id)
        claim = Claim(
            objective_id=obj.entity_id,
            statement="will not be demoted",
            scope="s",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            evidence_ids=evidence_ids,
            evidence_count=len(evidence_ids),
            adversarial_balance=0.7,
            adversarial_checked=True,
        )
        await repo.save(claim)

        # Trigger sweep with one orphan invalidated evidence to flip
        # had_cascades — but the claim's evidence is intact.
        orphan = Evidence(
            objective_id=obj.entity_id,
            source_type="web",
            source_ref="orphan",
            extracted=True,
            invalidated=True,
            invalidation_cascaded=False,
        )
        await repo.save(orphan)

        state = EpistemicGraphState(objective_id=obj.entity_id)
        deps = EpistemicDeps(repo=repo, agent_runner=None)
        await _run_tms_sweep(deps, state)

        # No demote → rescrutiny set should remain empty.
        assert state.claims_needing_rescrutiny == set()
