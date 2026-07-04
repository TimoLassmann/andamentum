"""Regression: scrutiny ↔ resolve oscillation per claim is bounded.

Bug context (smoke_v12_decompose, case 54): two of four spawned children
each ran ~125 execution steps stuck in
``scrutinise_claim → resolve_uncertainty (genuine resolution) →
deduplicate_concerns → scrutinise_claim`` cycles. The v6/v7 fixes
addressed *spontaneous* re-entry (A1 fingerprint), in-verification
convergence (A2), and sibling-grouping no-ops (A5) — but a claim whose
seed text reliably produces one *novel* uncertainty per scrutiny pass
still loops, because each genuine resolution legitimately marks the
claim for rescrutiny.

The fix has two parts:

1. *Runtime guardrail* — a per-claim counter
   ``state.scrutiny_resolve_cycles`` capped at
   ``SCRUTINY_RESOLVE_CYCLE_CAP``. Mirrors the existing investigation
   cap. Two enforcement points:
   - ``ResolveUncertainties.run`` refuses to add a claim to
     ``claims_needing_rescrutiny`` once cycles reach the cap
     (primary loop-breaker).
   - ``Scrutinize.run`` discards capped claims from the rescrutiny set
     without rerunning scrutiny (defense in depth).

2. *Cycle-as-signal* (Option 1 of the architectural follow-up) — when
   the cap fires, ``Claim.cycle_capped`` is set and
   ``Claim.persistent_concerns`` snapshots the blocking uncertainties at
   that moment. ``PromoteToSupported`` skips cycle-capped claims, so
   the IBE chain doesn't fabricate a verdict for a claim whose inquiry
   didn't converge. ``compute_posterior`` detects cycle-capped active
   claims and emits ``terminal_state="oscillation_detected"`` with
   posterior=0.5 — a distinct terminal state from
   ``"completed"`` and ``"retrieval_failed"``. The decomposed combiner
   propagates this terminal state.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.confidence import compute_posterior
from andamentum.epistemic.entities import Claim, Objective, Uncertainty
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.entities.uncertainty import UncertaintyType
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import (
    SCRUTINY_RESOLVE_CYCLE_CAP,
    PromoteToSupported,
    ResolveUncertainties,
    Scrutinize,
)
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.repository import EpistemicRepository


class _FakeRunContext:
    """Duck-typed GraphRunContext — Scrutinize / ResolveUncertainties only
    read .state and .deps."""

    def __init__(self, state: EpistemicGraphState, deps: EpistemicDeps) -> None:
        self.state = state
        self.deps = deps


async def _setup_objective_and_claim(
    tmp_path: Path, db_name: str
) -> tuple[Claim, EpistemicRepository]:
    store = DocumentStore.for_database(db_name, db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)
    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    claim = Claim(
        statement="Test claim that triggers the oscillation pattern.",
        scope="test",
        objective_id=obj.entity_id,
        stage=ClaimStage.HYPOTHESIS,
        scrutiny_verdict="pass",
        scrutiny_fingerprint="some-fingerprint",
    )
    await repo.save(claim)
    return claim, repo


# ── Constant + state defaults ─────────────────────────────────────────


def test_cap_constant_is_three() -> None:
    """Mirrors investigation cap of 3 — keep these aligned."""
    assert SCRUTINY_RESOLVE_CYCLE_CAP == 3


def test_state_default_for_scrutiny_resolve_cycles() -> None:
    s = EpistemicGraphState()
    assert s.scrutiny_resolve_cycles == {}


# ── Scrutinize side (defense in depth) ────────────────────────────────


class TestScrutinizeRespectsCap:
    async def test_at_cap_skips_rescrutiny_and_discards_flag(
        self, tmp_path: Path
    ) -> None:
        claim, repo = await _setup_objective_and_claim(tmp_path, "scrutinize_at_cap")

        state = EpistemicGraphState(objective_id=claim.objective_id)
        state.claims_needing_rescrutiny.add(claim.entity_id)
        state.scrutiny_resolve_cycles[claim.entity_id] = SCRUTINY_RESOLVE_CYCLE_CAP

        deps = EpistemicDeps(repo=repo, agent_runner=None)
        ctx = _FakeRunContext(state, deps)
        await Scrutinize().run(ctx)  # type: ignore[arg-type]

        # Cap-discard: claim removed from the rescrutiny set without
        # firing scrutinise_claim. The verdict and fingerprint are
        # preserved (proof scrutiny did NOT run).
        assert claim.entity_id not in state.claims_needing_rescrutiny
        scrutiny_ops = [
            op for op in state.operations_log if op.operation == "scrutinise_claim"
        ]
        assert scrutiny_ops == []
        reloaded = await repo.get("claim", claim.entity_id)
        assert reloaded.scrutiny_verdict == "pass"
        assert reloaded.scrutiny_fingerprint == "some-fingerprint"
        # Option 1: cap-firing marks the claim cycle_capped so downstream
        # (PromoteToSupported, compute_posterior) routes it to a terminal
        # state instead of fabricating a verdict.
        assert reloaded.cycle_capped is True
        # Counter is unchanged at the cap (no further increment after
        # the skip).
        assert (
            state.scrutiny_resolve_cycles[claim.entity_id] == SCRUTINY_RESOLVE_CYCLE_CAP
        )

    async def test_below_cap_increments_and_runs_scrutiny(self, tmp_path: Path) -> None:
        claim, repo = await _setup_objective_and_claim(tmp_path, "scrutinize_below_cap")

        state = EpistemicGraphState(objective_id=claim.objective_id)
        state.claims_needing_rescrutiny.add(claim.entity_id)
        state.scrutiny_resolve_cycles[claim.entity_id] = SCRUTINY_RESOLVE_CYCLE_CAP - 1

        deps = EpistemicDeps(repo=repo, agent_runner=None)
        ctx = _FakeRunContext(state, deps)
        await Scrutinize().run(ctx)  # type: ignore[arg-type]

        # Scrutiny ran (no agent_runner → ScrutiniseClaimOperation falls
        # through to verdict='pass' with no LLM call).
        scrutiny_ops = [
            op for op in state.operations_log if op.operation == "scrutinise_claim"
        ]
        assert len(scrutiny_ops) == 1
        # Counter incremented from CAP-1 to CAP.
        assert (
            state.scrutiny_resolve_cycles[claim.entity_id] == SCRUTINY_RESOLVE_CYCLE_CAP
        )
        # Rescrutiny flag consumed.
        assert claim.entity_id not in state.claims_needing_rescrutiny

    async def test_first_pass_initializes_counter_to_one(self, tmp_path: Path) -> None:
        """Fresh claim that gets marked for rescrutiny: counter starts at 1
        after the first Scrutinize pass."""
        claim, repo = await _setup_objective_and_claim(
            tmp_path, "scrutinize_first_pass"
        )

        state = EpistemicGraphState(objective_id=claim.objective_id)
        state.claims_needing_rescrutiny.add(claim.entity_id)
        # No prior entry in scrutiny_resolve_cycles.
        assert claim.entity_id not in state.scrutiny_resolve_cycles

        deps = EpistemicDeps(repo=repo, agent_runner=None)
        ctx = _FakeRunContext(state, deps)
        await Scrutinize().run(ctx)  # type: ignore[arg-type]

        assert state.scrutiny_resolve_cycles[claim.entity_id] == 1

    async def test_initial_scrutiny_does_not_increment_counter(
        self, tmp_path: Path
    ) -> None:
        """A claim that has no verdict yet (initial scrutiny, not
        rescrutiny) must not count toward the cap. Only rescrutiny passes
        do."""
        store = DocumentStore.for_database("scrutinize_initial", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(description="parent", question_type="verificatory")
        obj.objective_id = obj.entity_id
        await repo.save(obj)
        claim = Claim(
            statement="No verdict yet.",
            scope="test",
            objective_id=obj.entity_id,
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict=None,
        )
        await repo.save(claim)

        state = EpistemicGraphState(objective_id=obj.entity_id)
        # NOT in claims_needing_rescrutiny — this is the initial pass.

        deps = EpistemicDeps(repo=repo, agent_runner=None)
        ctx = _FakeRunContext(state, deps)
        await Scrutinize().run(ctx)  # type: ignore[arg-type]

        scrutiny_ops = [
            op for op in state.operations_log if op.operation == "scrutinise_claim"
        ]
        assert len(scrutiny_ops) == 1
        # Initial scrutiny is unrelated to the oscillation cap.
        assert claim.entity_id not in state.scrutiny_resolve_cycles


# ── ResolveUncertainties side (primary loop-breaker) ──────────────────


class TestResolveUncertaintiesRespectsCap:
    async def test_at_cap_does_not_add_to_rescrutiny(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """When a claim has cycles == cap, a genuine uncertainty resolution
        must NOT add the claim back to claims_needing_rescrutiny, even if
        the uncertainty is fully resolved with did_work=True."""
        claim, repo = await _setup_objective_and_claim(tmp_path, "resolve_at_cap")

        # An unresolved blocking uncertainty whose resolution will trigger
        # the rescrutiny path.
        unc = Uncertainty(
            description="Some blocking concern about the claim.",
            uncertainty_type=UncertaintyType.UNKNOWN,
            objective_id=claim.objective_id,
            affected_claim_ids=[claim.entity_id],
            is_blocking=True,
            resolution=None,
        )
        await repo.save(unc)

        state = EpistemicGraphState(objective_id=claim.objective_id)
        state.scrutiny_resolve_cycles[claim.entity_id] = SCRUTINY_RESOLVE_CYCLE_CAP

        deps = EpistemicDeps(
            repo=repo, agent_runner=fake_runner, embedding_model="test-embed"
        )
        ctx = _FakeRunContext(state, deps)
        await ResolveUncertainties().run(ctx)  # type: ignore[arg-type]

        reloaded_unc = await repo.get("uncertainty", unc.entity_id)
        # The uncertainty should have been resolved (fake_runner default
        # is can_resolve=True with a resolution string).
        assert reloaded_unc.resolution is not None
        # Cap blocks the rescrutiny add — this is the primary loop-breaker.
        assert claim.entity_id not in state.claims_needing_rescrutiny
        # Option 1: claim is marked cycle_capped at the cap-firing site.
        reloaded_claim = await repo.get("claim", claim.entity_id)
        assert reloaded_claim.cycle_capped is True

    async def test_below_cap_adds_to_rescrutiny(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """Sanity: below the cap, a genuine resolution still triggers
        rescrutiny — the existing v7 A5 path is preserved."""
        claim, repo = await _setup_objective_and_claim(tmp_path, "resolve_below_cap")

        unc = Uncertainty(
            description="Different blocking concern.",
            uncertainty_type=UncertaintyType.UNKNOWN,
            objective_id=claim.objective_id,
            affected_claim_ids=[claim.entity_id],
            is_blocking=True,
            resolution=None,
        )
        await repo.save(unc)

        state = EpistemicGraphState(objective_id=claim.objective_id)
        # Cycles below the cap.
        state.scrutiny_resolve_cycles[claim.entity_id] = SCRUTINY_RESOLVE_CYCLE_CAP - 1

        deps = EpistemicDeps(
            repo=repo, agent_runner=fake_runner, embedding_model="test-embed"
        )
        ctx = _FakeRunContext(state, deps)
        await ResolveUncertainties().run(ctx)  # type: ignore[arg-type]

        # Below-cap path: rescrutiny was requested.
        assert claim.entity_id in state.claims_needing_rescrutiny


# ── End-to-end loop termination ───────────────────────────────────────


class TestLoopTermination:
    async def test_simulated_oscillation_terminates_at_cap(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """Drive the loop manually: alternate Scrutinize → ResolveUncertainties
        and verify the system stops requesting rescrutiny after CAP rounds.
        This is the smoke-v12 case 54 reproduction in miniature."""
        claim, repo = await _setup_objective_and_claim(tmp_path, "loop_termination")

        state = EpistemicGraphState(objective_id=claim.objective_id)
        deps = EpistemicDeps(
            repo=repo, agent_runner=fake_runner, embedding_model="test-embed"
        )
        ctx = _FakeRunContext(state, deps)

        rescrutiny_requests = 0
        for round_num in range(SCRUTINY_RESOLVE_CYCLE_CAP + 2):
            # Each round: a fresh blocking uncertainty (the runaway pattern
            # — scrutiny keeps producing one per pass).
            unc = Uncertainty(
                description=f"Round {round_num} concern.",
                uncertainty_type=UncertaintyType.UNKNOWN,
                objective_id=claim.objective_id,
                affected_claim_ids=[claim.entity_id],
                is_blocking=True,
                resolution=None,
            )
            await repo.save(unc)

            await ResolveUncertainties().run(ctx)  # type: ignore[arg-type]

            if claim.entity_id in state.claims_needing_rescrutiny:
                rescrutiny_requests += 1
                # Scrutinize consumes the flag and increments the counter.
                await Scrutinize().run(ctx)  # type: ignore[arg-type]

        # Resolve added rescrutiny on the first CAP-1 rounds, then refused;
        # Scrutinize incremented from 1..CAP across those rounds and
        # would refuse after. Net rescrutiny grants = CAP-1
        # (resolve gates *adding* once cycles reach CAP, and the counter
        # only increments inside Scrutinize when it actually runs).
        # Either way, the bound is tight: rescrutiny stops well before
        # SCRUTINY_RESOLVE_CYCLE_CAP + 2 rounds.
        assert rescrutiny_requests <= SCRUTINY_RESOLVE_CYCLE_CAP
        # Final state: no pending rescrutiny.
        assert claim.entity_id not in state.claims_needing_rescrutiny


# ── Option 1: cycle-as-signal routing ─────────────────────────────────


class TestPromoteToSupportedSkipsCycleCapped:
    async def test_cycle_capped_claim_is_not_promoted(self, tmp_path: Path) -> None:
        """A cycle-capped HYPOTHESIS claim with verdict=pass would normally
        be promoted to SUPPORTED. With Option 1, it stays at HYPOTHESIS
        and is added to terminal_claims so the IBE chain never sees it."""
        claim, repo = await _setup_objective_and_claim(
            tmp_path, "promote_skip_cycle_capped"
        )
        claim.cycle_capped = True
        await repo.save(claim)

        state = EpistemicGraphState(objective_id=claim.objective_id)
        deps = EpistemicDeps(repo=repo, agent_runner=None)
        ctx = _FakeRunContext(state, deps)
        await PromoteToSupported().run(ctx)  # type: ignore[arg-type]

        reloaded = await repo.get("claim", claim.entity_id)
        # Stage stays at HYPOTHESIS — not promoted.
        assert reloaded.stage == ClaimStage.HYPOTHESIS
        # State marks it terminal so CheckCompletion / IBE skip it.
        assert claim.entity_id in state.terminal_claims

    async def test_non_capped_claim_not_added_to_terminal_set(
        self, tmp_path: Path
    ) -> None:
        """Sanity: a non-capped claim is not put into terminal_claims by
        PromoteToSupported (the new branch only triggers on cycle_capped).
        Whether the claim actually promotes depends on gate validation —
        that's covered by the wider PromoteClaimOperation test suite."""
        claim, repo = await _setup_objective_and_claim(
            tmp_path, "promote_no_terminal_for_uncapped"
        )
        # cycle_capped defaults to False.

        state = EpistemicGraphState(objective_id=claim.objective_id)
        deps = EpistemicDeps(repo=repo, agent_runner=None)
        ctx = _FakeRunContext(state, deps)
        await PromoteToSupported().run(ctx)  # type: ignore[arg-type]

        # The Option-1 routing branch did not fire — the claim wasn't
        # marked terminal by the cycle-capped path.
        assert claim.entity_id not in state.terminal_claims


class TestComputePosteriorOscillationDetected:
    async def test_cycle_capped_claim_with_no_ia_suspends_via_no_certified_gate(
        self, tmp_path: Path
    ) -> None:
        """Cycle-capped + no integrated_assessment ⇒ the no-certified-verdict
        gate (added 2026-05-05) suspends the posterior to 0.5 with
        ``terminal_state="oscillation_detected"``. The cycle-capped flag
        is informational provenance — the substantive epistemic state is
        "no IBE certification", which is the gate's general condition."""
        claim, repo = await _setup_objective_and_claim(
            tmp_path, "posterior_oscillation"
        )
        claim.cycle_capped = True
        claim.persistent_concerns = ["unc-1", "unc-2"]
        await repo.save(claim)

        report = await compute_posterior(repo, claim.objective_id)
        assert report is not None
        assert report.terminal_state == "oscillation_detected"
        assert report.posterior == 0.5
        assert report.mode == "counting_only"
        assert "IBE certification" in report.explanation

    async def test_no_cycle_capped_claims_with_ia_means_normal_posterior(
        self, tmp_path: Path
    ) -> None:
        """Sanity: compute_posterior's normal path is unchanged for runs
        with an IBE-certified claim. (Without an integrated_assessment,
        the no-certified-verdict gate would fire even without
        cycle_capped — that's the structurally correct behaviour, but
        not what this test is checking.)"""
        claim, repo = await _setup_objective_and_claim(tmp_path, "posterior_normal")
        # cycle_capped defaults to False. Stamp an IA so the
        # no-certified-verdict gate doesn't preempt the test.
        claim.integrated_assessment = "supports"
        claim.integrated_confidence = 0.7
        await repo.save(claim)

        report = await compute_posterior(repo, claim.objective_id)
        assert report is not None
        assert report.terminal_state == "completed"


# NOTE: Under v0.3 multi-seed-claim, terminal_state propagation
# (retrieval_failed, oscillation_detected) at the combiner level is no
# longer a thing — the v0.2 architecture had per-child PipelineResults
# each with their own terminal_state, which the orchestrator combiner
# propagated. With one graph run per Objective, the terminal_state lives
# on the objective-level PosteriorReport (compute_posterior). The
# claim-level combiner (combine_claim_verdicts) operates over Claims
# which carry cycle_capped/abandoned flags, and it surfaces those as
# n_capped/n_abandoned diagnostics rather than terminal_state values.
# The v0.2 propagation tests for that path were removed when
# decomposed_runner was deleted — see test_combine_claim_verdicts.py
# (TestCombineClaimVerdicts) for the new shape.
