"""Graph-level regression for the self-feeding-loop plan (D1, S2).

docs/epistemic/SELF_FEEDING_LOOP_PLAN.md §0 diagnoses a claim that keeps
failing the SAME scrutiny check: every scrutiny pass raised a brand-new
``Uncertainty`` for the recurring contradiction, so
``ResolveUncertainties`` never saw the blocking set empty and
``PromoteToSupported`` was never reached — the run terminated at
``ConcludeAtCycleCapOperation`` instead.

This test scripts a scrutiny that reports the identical cross-item
contradiction on every pass (a fixed description string, so even the
no-embedding-model exact-text fallback in ``uncertainty_raising`` catches
it) and drives the actual graph nodes — ``Scrutinize`` →
``ResolveUncertainties`` → ``Scrutinize`` (rescrutiny) →
``ResolveUncertainties`` — exactly the oscillation the plan diagnoses.

S2: after the one blocking uncertainty is resolved once, the rescrutiny
that finds the same contradiction again creates NO new blocking
uncertainty (D1 increments ``raise_count`` on the existing, already-
resolved entity instead), and ``ResolveUncertainties(next_on_clear=
"promote")`` proceeds to ``PromoteToSupported`` rather than looping back
to ``Scrutinize`` forever.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, Evidence, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.entities.uncertainty import BLOCKING_TYPES, UncertaintyType
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import (
    PEIRCE_CYCLE_CAP,
    SCRUTINY_RESOLVE_CYCLE_CAP,
    CombineClaimVerdicts,
    Investigate,
    PromoteToSupported,
    ResolveUncertainties,
    Scrutinize,
)
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.repository import EpistemicRepository

from .conftest import FakeAgentRunner

#: The exact wording the scripted scrutiny reports every single pass. Fixed
#: verbatim so the no-embedding-model exact-normalised-text fallback in
#: ``uncertainty_raising.raise_uncertainty`` (used throughout this test,
#: ``embedding_model=None``) recognises it as the same finding each time.
_SAME_CONTRADICTION = "Two sources report incompatible dosages for the same regimen."

#: A directional mass in the "moderate" band (between
#: SCRUTINY_WEAK_MASS_THRESHOLD and SCRUTINY_STRONG_MASS_THRESHOLD) so
#: ``_verdict_from_weight`` maps to "pass" without needing the borderline
#: second-opinion machinery (embedding_model=None keeps this test offline).
_P_SUPPORTS = 0.9
_TARGET_MASS = 0.9


class _RepeatingContradictionRunner(FakeAgentRunner):
    """Reports the identical contradiction on every cross-item check.

    ``epistemic_identify_single_issue`` is called once per evidence item
    (no ``focus`` kwarg) and once for the cross-item contradiction check
    (``focus`` set). Only the contradiction call reports an issue here —
    per-evidence calls fall through to the shared default (``has_issue=
    False``) — and it reports the SAME description every time, exactly
    the "contested claim raises the same contradiction on every pass"
    failure mode the plan diagnoses.
    """

    async def run(self, agent_name: str, **kwargs: Any) -> Any:
        if agent_name == "epistemic_identify_single_issue" and kwargs.get("focus"):
            from types import SimpleNamespace

            self.calls.append((agent_name, kwargs))
            return SimpleNamespace(
                has_issue=True,
                description=_SAME_CONTRADICTION,
                issue_type="contradiction",
                reversal_test=False,
            )
        return await super().run(agent_name, **kwargs)


class _FakeRunContext:
    """Duck-typed GraphRunContext — nodes here only read .state and .deps."""

    def __init__(self, state: EpistemicGraphState, deps: EpistemicDeps) -> None:
        self.state = state
        self.deps = deps


async def _setup(tmp_path: Path) -> tuple[Claim, EpistemicRepository]:
    store = DocumentStore.for_database("self_feeding_loop", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    claim = Claim(
        statement="The regimen is safe at the studied dose.",
        scope="adults",
        objective_id=obj.entity_id,
        stage=ClaimStage.HYPOTHESIS,
    )

    # Two representative evidence items — the cross-item contradiction
    # call only fires with >= 2 evidence summaries — with a graded (not
    # one-hot) directional judgment so neither pre-existing second-opinion
    # site (one-hot direction confirmation, borderline-weight read) fires,
    # and a moderate combined mass so the deterministic weight lands on
    # "pass" without any llm_judge dependency.
    quality = _TARGET_MASS / (_P_SUPPORTS * 2)
    items: list[Evidence] = []
    for i in range(2):
        ev = Evidence(
            objective_id=obj.entity_id,
            source_type="openalex",
            source_ref=f"doi:10.1234/{i}",
            extracted_content=f"Source {i} on the regimen's dosage and safety profile.",
            extracted=True,
            quality_score=quality,
            cluster_status="representative",
            judgment_distribution=[_P_SUPPORTS, 0.05, 0.05],
            support_judgment="supports",
        )
        await repo.save(ev)
        items.append(ev)
    claim.evidence_ids = [ev.entity_id for ev in items]
    claim.evidence_count = len(items)
    await repo.save(claim)
    return claim, repo


class TestSelfFeedingLoopFixed:
    async def test_repeated_contradiction_dedups_and_reaches_promote(
        self, tmp_path: Path
    ) -> None:
        claim, repo = await _setup(tmp_path)
        state = EpistemicGraphState(objective_id=claim.objective_id)
        deps = EpistemicDeps(
            repo=repo,
            agent_runner=_RepeatingContradictionRunner(),
            embedding_model=None,
            judge_model=None,
        )
        ctx = _FakeRunContext(state, deps)

        # ── Pass 1: initial scrutiny raises the blocking contradiction ──
        node = await Scrutinize().run(ctx)  # type: ignore[arg-type]
        assert isinstance(node, ResolveUncertainties)
        assert node.next_on_clear == "promote"

        rows = await repo.query("uncertainty", objective_id=claim.objective_id)
        assert len(rows) == 1
        assert rows[0].uncertainty_type is UncertaintyType.CONTRADICTION
        assert rows[0].uncertainty_type in BLOCKING_TYPES
        assert rows[0].resolution is None
        assert rows[0].raise_count == 1
        first_uncertainty_id = rows[0].entity_id

        # ── Resolve it once; the resolve stub (fake_runner default:
        # can_resolve=True) marks the claim for rescrutiny. ──
        node = await node.run(ctx)  # type: ignore[arg-type]
        assert isinstance(node, Scrutinize)
        assert claim.entity_id in state.claims_needing_rescrutiny
        assert (
            state.scrutiny_resolve_cycles.get(claim.entity_id, 0)
            < SCRUTINY_RESOLVE_CYCLE_CAP
        )

        reloaded = await repo.get("uncertainty", first_uncertainty_id)
        assert reloaded.resolution is not None

        # ── Pass 2 (rescrutiny): the SAME contradiction is reported
        # again. D1 must not create a second blocking uncertainty. ──
        node = await node.run(ctx)  # type: ignore[arg-type]

        rows = await repo.query("uncertainty", objective_id=claim.objective_id)
        assert len(rows) == 1, (
            "rescrutiny re-raising the identical contradiction must "
            "increment the existing uncertainty, not create a second one"
        )
        reloaded = await repo.get("uncertainty", first_uncertainty_id)
        assert reloaded.raise_count == 2
        # Still resolved — D1's increment path does not reopen it.
        assert reloaded.resolution is not None

        # No fresh blocking uncertainty means no rescrutiny loop-back:
        # Scrutinize's own routing sends this straight to
        # ResolveUncertainties(next_on_clear="promote").
        assert isinstance(node, ResolveUncertainties)
        assert node.next_on_clear == "promote"

        # ── ResolveUncertainties finds no unresolved blocking
        # uncertainty left and proceeds to PromoteToSupported — the
        # exact step the plan's diagnosed defect never reached. ──
        node = await node.run(ctx)  # type: ignore[arg-type]
        assert isinstance(node, PromoteToSupported)


async def _setup_min_supporting_sources_refusal(
    tmp_path: Path,
) -> tuple[Claim, EpistemicRepository]:
    """A HYPOTHESIS claim, scrutiny-clean, whose ONLY promotion blocker is
    too few supporting sources: its one evidence item is judged
    (``support_judgment="no_bearing"``, satisfying ``_any_evidence_judged``)
    but not a support, so ``count_supporting_sources`` returns 0 against the
    SUPPORTED gate's ``min_supporting_sources=1`` — the exact shape D4
    checks for via ``refused_only_for_min_supporting_sources``."""
    store = DocumentStore.for_database("self_feeding_loop_d4", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    ev = Evidence(
        objective_id=obj.entity_id,
        source_type="openalex",
        source_ref="doi:10.1234/0",
        extracted_content="Source with no bearing on the claim.",
        extracted=True,
        quality_score=0.5,
        support_judgment="no_bearing",
    )
    await repo.save(ev)

    claim = Claim(
        statement="The regimen is safe at the studied dose.",
        objective_id=obj.entity_id,
        stage=ClaimStage.HYPOTHESIS,
        scrutiny_verdict="pass",
        evidence_ids=[ev.entity_id],
        evidence_count=1,
    )
    await repo.save(claim)
    return claim, repo


class TestPromoteToSupportedRoutesInsufficientSourcesToInvestigate:
    """D4/S5 (self-feeding-loop plan): a HYPOTHESIS claim refused ONLY for
    ``min_supporting_sources``, under its investigation cap, routes to
    ``Investigate`` instead of being left stranded at HYPOTHESIS for the
    run. At the cap, existing behaviour holds — no route change."""

    async def test_under_cap_routes_to_investigate(self, tmp_path: Path) -> None:
        claim, repo = await _setup_min_supporting_sources_refusal(tmp_path)
        state = EpistemicGraphState(objective_id=claim.objective_id)
        deps = EpistemicDeps(
            repo=repo,
            agent_runner=FakeAgentRunner(),
            embedding_model=None,
            judge_model=None,
        )
        ctx = _FakeRunContext(state, deps)

        assert state.investigation_counts.get(claim.entity_id, 0) < PEIRCE_CYCLE_CAP

        node = await PromoteToSupported().run(ctx)  # type: ignore[arg-type]

        assert isinstance(node, Investigate)

        # P3/P4: the demand is graph state, not a fabricated entity
        # field — scrutiny never produced "needs_resolution" for this
        # claim and the persisted verdict must not say otherwise.
        assert claim.entity_id in state.claims_needing_investigation
        reloaded = await repo.get("claim", claim.entity_id)
        assert reloaded.stage == ClaimStage.HYPOTHESIS
        assert reloaded.scrutiny_verdict == "pass"

    async def test_at_cap_does_not_route_to_investigate(self, tmp_path: Path) -> None:
        claim, repo = await _setup_min_supporting_sources_refusal(tmp_path)
        state = EpistemicGraphState(objective_id=claim.objective_id)
        state.investigation_counts[claim.entity_id] = PEIRCE_CYCLE_CAP
        deps = EpistemicDeps(
            repo=repo,
            agent_runner=FakeAgentRunner(),
            embedding_model=None,
            judge_model=None,
        )
        ctx = _FakeRunContext(state, deps)

        node = await PromoteToSupported().run(ctx)  # type: ignore[arg-type]

        assert not isinstance(node, Investigate)
        assert isinstance(node, CombineClaimVerdicts)

        reloaded = await repo.get("claim", claim.entity_id)
        assert reloaded.stage == ClaimStage.HYPOTHESIS
        # Existing behaviour: refusal at the cap leaves scrutiny_verdict
        # untouched — the claim is not marked for a fresh investigation
        # pass it isn't allowed to take.
        assert reloaded.scrutiny_verdict == "pass"

    async def test_at_scrutiny_resolve_cap_does_not_route_to_investigate(
        self, tmp_path: Path
    ) -> None:
        """The investigation-count cap alone is not enough: a claim can
        reach here clean (scrutiny_verdict="pass") with its
        ``scrutiny_resolve_cycles`` already at
        ``SCRUTINY_RESOLVE_CYCLE_CAP`` (three resolve->rescrutiny rounds
        already spent). Routing to Investigate here would only earn a
        rescrutiny request Scrutinize is bound to refuse — the exact
        wasted-round shape the fix for this finding closes. Fall through
        exactly as before: no route change, no cycle_capped stamp, verdict
        untouched.
        """
        claim, repo = await _setup_min_supporting_sources_refusal(tmp_path)
        state = EpistemicGraphState(objective_id=claim.objective_id)
        state.scrutiny_resolve_cycles[claim.entity_id] = SCRUTINY_RESOLVE_CYCLE_CAP
        deps = EpistemicDeps(
            repo=repo,
            agent_runner=FakeAgentRunner(),
            embedding_model=None,
            judge_model=None,
        )
        ctx = _FakeRunContext(state, deps)

        assert state.investigation_counts.get(claim.entity_id, 0) < PEIRCE_CYCLE_CAP

        node = await PromoteToSupported().run(ctx)  # type: ignore[arg-type]

        assert not isinstance(node, Investigate)
        assert isinstance(node, CombineClaimVerdicts)
        assert claim.entity_id not in state.claims_needing_investigation

        reloaded = await repo.get("claim", claim.entity_id)
        assert reloaded.stage == ClaimStage.HYPOTHESIS
        assert reloaded.scrutiny_verdict == "pass"
        assert reloaded.cycle_capped is False
