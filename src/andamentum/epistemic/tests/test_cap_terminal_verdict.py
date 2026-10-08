"""A cycle-capped claim must never report a false "insufficient".

THE DEFECT. Before 2026-08-09, a claim that could not pass a stage gate
retried, hit ``PEIRCE_CYCLE_CAP``, and terminated with
``integrated_assessment=None`` — so ``CheckCompletion``'s
no-certified-verdict exit fired and the pipeline wrote "insufficient
evidence", *even when the evidence already gathered was decisive*.
Measured cost: CONTRADICT recall 0.40 → 0.10 in the 2026-07-08 A/B, with
case 781 the worked example — the refuting studies were present; the
claim just failed to climb the ladder.

THE FIX. ``_mark_cycle_capped`` — the one funnel both cap-firing sites
already go through — dispatches ``ConcludeAtCycleCapOperation``, which
reads the claim's eligible evidence deterministically (no LLM, same
``compute_evidence_weight`` engine scrutiny uses) and, when that reading
is directional, writes the verdict the evidence supports plus a typed
``CapEvaluation`` provenance record. When the reading is NOT directional
it writes no verdict — and the run still terminates "insufficient", but
now as a recorded conclusion about the evidence rather than a scheduling
side effect.

Every test here is offline: ``ConcludeAtCycleCapOperation`` never calls
an agent, and the two cap sites are driven directly through the existing
``_FakeRunContext(state, deps)`` duck type.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from andamentum.document_store import Store
from andamentum.epistemic import gates
from andamentum.epistemic.confidence import compute_posterior
from andamentum.epistemic.entities import (
    CapEvaluation,
    Claim,
    Evidence,
    Objective,
    Uncertainty,
)
from andamentum.epistemic.entities.claim import CandidateRecord, ClaimStage
from andamentum.epistemic.entities.uncertainty import UncertaintyType
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.invariants import capped_claims_carry_a_cap_evaluation
from andamentum.epistemic.graph.nodes import (
    SCRUTINY_RESOLVE_CYCLE_CAP,
    AbandonOrDemote,
    CheckCompletion,
    CheckSynthesisDemand,
    CombineClaimVerdicts,
    ResolveUncertainties,
    Scrutinize,
    Synthesize,
    SynthesizeInsufficient,
)
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.operations.base import OperationInput
from andamentum.epistemic.operations.stage_management import (
    ConcludeAtCycleCapOperation,
    PromoteAsRefutedOperation,
)
from andamentum.epistemic.repository import EpistemicRepository
from andamentum.epistemic.scrutiny_weight import compute_evidence_weight
from andamentum.epistemic.thresholds import (
    CYCLE_CAP_CONFIDENCE_PENALTY,
    MECHANICAL_VERDICT_MAX_CONFIDENCE,
    PEIRCE_CYCLE_CAP,
    POSTERIOR_DECISIVE_THRESHOLD,
    POSTERIOR_DIRECTIONAL_BREAKPOINT,
    SCRUTINY_DEFAULT_QUALITY,
    SCRUTINY_STRONG_MASS_THRESHOLD,
)


class _FakeRunContext:
    """Duck-typed GraphRunContext — the nodes only read .state and .deps."""

    def __init__(self, state: EpistemicGraphState, deps: EpistemicDeps) -> None:
        self.state = state
        self.deps = deps


async def _run_check_completion(state: EpistemicGraphState, deps: EpistemicDeps):
    """Drive CombineClaimVerdicts → CheckCompletion, the real sequence.

    ``CheckCompletion`` is a pure reader over ``state.verdict`` and
    raises when it is absent; ``CombineClaimVerdicts`` is its sole
    predecessor. Seeding through the real node is a strengthening — the
    cap's routing effect is now asserted against a verdict object built
    from the same claim table the cap wrote to.
    """
    ctx = _FakeRunContext(state, deps)
    await CombineClaimVerdicts().run(ctx)  # type: ignore[arg-type]
    return await CheckCompletion().run(ctx)  # type: ignore[arg-type]


# ── Fixtures ─────────────────────────────────────────────────────────


async def _setup(
    tmp_path: Path,
    db_name: str,
    *,
    n_supporting: int = 0,
    n_contradicting: int = 0,
    quality: float = 1.0,
) -> tuple[Objective, Claim, EpistemicRepository]:
    """One verificatory objective, one HYPOTHESIS claim, and N judged
    evidence items linked to it.

    ``quality=1.0`` with one-hot judgments makes each item contribute
    exactly 1.0 of directional mass, so the resulting weight labels are
    easy to reason about against the SCRUTINY_* thresholds.
    """
    store = await Store.open(tmp_path / f"{db_name}.db")
    repo = EpistemicRepository(store)

    obj = Objective(
        description="Interferon-gamma protects against myocarditis.",
        clarified_question="Interferon-gamma protects against myocarditis.",
        question_type="verificatory",
        claim_to_verify="Interferon-gamma protects against myocarditis.",
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="Interferon-gamma protects against myocarditis.",
        scope="test",
        stage=ClaimStage.HYPOTHESIS,
        scrutiny_verdict="pass",
        scrutiny_fingerprint="fp",
    )
    await repo.save(claim)

    evidence_ids: list[str] = []
    for i in range(n_supporting):
        ev = Evidence(
            objective_id=obj.entity_id,
            extracted_content=f"Supporting study {i} with substantive content.",
            extracted=True,
            source_ref=f"doi:sup/{i}",
            source_type="pubmed",
            support_judgment="supports",
            judgment_distribution=[1.0, 0.0, 0.0],
            quality_score=quality,
        )
        await repo.save(ev)
        evidence_ids.append(ev.entity_id)
    for i in range(n_contradicting):
        ev = Evidence(
            objective_id=obj.entity_id,
            extracted_content=f"Refuting study {i} with substantive content.",
            extracted=True,
            source_ref=f"doi:con/{i}",
            source_type="pubmed",
            support_judgment="contradicts",
            judgment_distribution=[0.0, 1.0, 0.0],
            quality_score=quality,
        )
        await repo.save(ev)
        evidence_ids.append(ev.entity_id)

    claim.evidence_ids = evidence_ids
    await repo.save(claim)
    return obj, claim, repo


def _state_at_cap(claim: Claim) -> EpistemicGraphState:
    state = EpistemicGraphState(objective_id=claim.objective_id)
    state.claims_needing_rescrutiny.add(claim.entity_id)
    state.scrutiny_resolve_cycles[claim.entity_id] = SCRUTINY_RESOLVE_CYCLE_CAP
    return state


# ── The case-781 shape: the cap emits the verdict the evidence supports ──


async def test_cap_at_scrutinize_emits_the_verdict_the_evidence_supports(
    tmp_path: Path,
) -> None:
    """THE REGRESSION TEST for the defect this change fixes.

    Three refuting studies and one supporting one — decisive evidence.
    The claim hits the cap at Scrutinize. Before the fix it terminated
    with no verdict at all and the pipeline said "insufficient". Now it
    carries ``contradicts``.

    Also pins that the claim's STAGE is unchanged: promoting it would
    expose the fresh verdict to AbandonOrDemote → record_demotion, which
    nulls integrated_assessment — the fix would erase itself.
    """
    _obj, claim, repo = await _setup(
        tmp_path, "cap_scrutinize", n_supporting=1, n_contradicting=3
    )
    state = _state_at_cap(claim)
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.cycle_capped is True
    assert reloaded.integrated_assessment == "contradicts"
    assert reloaded.integrated_confidence is not None
    assert 0.0 < reloaded.integrated_confidence <= MECHANICAL_VERDICT_MAX_CONFIDENCE
    assert reloaded.integrated_reasoning is not None
    assert reloaded.cap_evaluation is not None
    assert reloaded.cap_evaluation.weight_label == "strong"
    assert reloaded.cap_evaluation.verdict == "contradicts"
    assert reloaded.cap_evaluation.n_evidence == 4
    # Stage is deliberately untouched.
    assert reloaded.stage == ClaimStage.HYPOTHESIS


async def test_cap_at_resolve_uncertainties_emits_the_same_verdict(
    tmp_path: Path, fake_runner
) -> None:
    """Both cap-firing sites route through the one ``_mark_cycle_capped``
    funnel. A fix applied at only one site fails here."""
    _obj, claim, repo = await _setup(
        tmp_path, "cap_resolve", n_supporting=1, n_contradicting=3
    )
    unc = Uncertainty(
        description="A blocking concern about the claim.",
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

    await ResolveUncertainties().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.cycle_capped is True
    assert reloaded.integrated_assessment == "contradicts"
    assert reloaded.cap_evaluation is not None
    assert reloaded.cap_evaluation.verdict == "contradicts"


# ── The cap must never manufacture a direction ───────────────────────


async def test_conflicting_evidence_at_cap_emits_no_verdict(
    tmp_path: Path,
) -> None:
    """Genuine disagreement is not a verdict. Guards against the cap
    inventing a direction out of a balanced pool."""
    _obj, claim, repo = await _setup(
        tmp_path, "cap_conflicting", n_supporting=2, n_contradicting=2
    )
    state = _state_at_cap(claim)
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.cap_evaluation is not None
    assert reloaded.cap_evaluation.weight_label == "conflicting"
    assert reloaded.cap_evaluation.verdict is None
    assert reloaded.integrated_assessment is None


@pytest.mark.parametrize(
    ("n_supporting", "quality", "case"),
    [
        (1, 0.2, "one very low-quality item"),
        (0, 1.0, "no evidence at all"),
    ],
)
async def test_below_the_weak_mass_cut_emits_no_verdict(
    tmp_path: Path, n_supporting: int, quality: float, case: str
) -> None:
    """The WEAK-MASS CUT, specifically.

    Scope note, because this test used to be miscast as the general
    "thin evidence" guard: both fixtures sit below
    ``SCRUTINY_WEAK_MASS_THRESHOLD`` (0.4) — quality 0.2 gives 0.2 of
    mass, and no evidence gives 0.0. They pin that the label algebra
    refuses a direction below the cut. They say NOTHING about a single
    NORMAL-quality item, which clears the cut, reads ``moderate``, and
    does get a verdict; that shape is covered by the volume tests below.
    """
    _obj, claim, repo = await _setup(
        tmp_path,
        f"cap_thin_{n_supporting}_{int(quality * 10)}",
        n_supporting=n_supporting,
        quality=quality,
    )
    state = _state_at_cap(claim)
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.cap_evaluation is not None, case
    assert reloaded.cap_evaluation.weight_label == "weak", case
    assert reloaded.cap_evaluation.verdict is None, case
    assert reloaded.integrated_assessment is None, case


# ── Confidence must depend on how much evidence there is ─────────────


async def test_a_single_normal_quality_item_does_not_report_maximum_confidence(
    tmp_path: Path,
) -> None:
    """THE COMMON REAL SHAPE, and the regression test for a measured defect.

    One unopposed, normal-quality source. The confidence formula was
    directional PURITY alone — ``(dominant - minority) / (dominant +
    minority)`` — which is 1.0 for any one-sided pool regardless of size,
    so a single item saturated ``MECHANICAL_VERDICT_MAX_CONFIDENCE`` and
    produced posterior 0.815, the top of the non-decisive band. That is
    the opposite of the brief's "emit the verdict the evidence supports,
    WITH LOW CONFIDENCE".

    Confidence is now ``purity × mass_adequacy``, where mass_adequacy
    saturates at ``SCRUTINY_STRONG_MASS_THRESHOLD`` — the mass the system
    already calls "strong". No new constant.
    """
    obj, claim, repo = await _setup(
        tmp_path, "cap_one_item", n_supporting=1, quality=SCRUTINY_DEFAULT_QUALITY
    )
    state = _state_at_cap(claim)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    reloaded = await repo.get("claim", claim.entity_id)
    # Clears the weak cut (0.5 > 0.4), so it IS directional — the signal
    # is emitted, not discarded.
    assert reloaded.cap_evaluation is not None
    assert reloaded.cap_evaluation.weight_label == "moderate"
    assert reloaded.integrated_assessment == "supports"
    # ... but nowhere near the ceiling: purity 1.0 × 0.5/1.5.
    assert reloaded.integrated_confidence is not None
    assert reloaded.integrated_confidence == pytest.approx(
        SCRUTINY_DEFAULT_QUALITY / SCRUTINY_STRONG_MASS_THRESHOLD
    )
    assert reloaded.integrated_confidence < MECHANICAL_VERDICT_MAX_CONFIDENCE

    report = await compute_posterior(repo, obj.entity_id)
    assert report is not None
    # Directional but not confident enough to read as a directional label.
    assert 0.5 < report.posterior < POSTERIOR_DIRECTIONAL_BREAKPOINT


async def test_cap_confidence_rises_with_corroborating_volume(
    tmp_path: Path,
) -> None:
    """Ordering, not absolute numbers.

    More unanimous evidence must never report LESS confidence than less
    of it, and a substantial mixed pool must never report less than a
    single unopposed item. The purity-only formula inverted the second
    relation: a single unscored item (0.9) outranked a 5-vs-1 pool
    (0.733). Both relations are asserted as orderings so the underlying
    constants stay tunable.
    """

    async def _cap_confidence(
        name: str, *, n_supporting: int, n_contradicting: int = 0
    ) -> float:
        _obj, claim, repo = await _setup(
            tmp_path,
            name,
            n_supporting=n_supporting,
            n_contradicting=n_contradicting,
            quality=SCRUTINY_DEFAULT_QUALITY,
        )
        state = _state_at_cap(claim)
        deps = EpistemicDeps(repo=repo, agent_runner=None)
        await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
        reloaded = await repo.get("claim", claim.entity_id)
        assert reloaded.integrated_confidence is not None
        return reloaded.integrated_confidence

    one = await _cap_confidence("cap_vol_1", n_supporting=1)
    two = await _cap_confidence("cap_vol_2", n_supporting=2)
    ten = await _cap_confidence("cap_vol_10", n_supporting=10)
    five_vs_one = await _cap_confidence(
        "cap_vol_5v1", n_supporting=5, n_contradicting=1
    )

    assert one < two < ten
    # Saturating: ten unanimous items sit at the ceiling and cannot exceed it.
    assert ten == pytest.approx(MECHANICAL_VERDICT_MAX_CONFIDENCE)
    # The inversion the purity-only formula produced.
    assert one < five_vs_one


# ── Verdict ownership ────────────────────────────────────────────────


async def test_cap_evaluation_never_overwrites_an_ibe_verdict(
    tmp_path: Path,
) -> None:
    """The deterministic path records its reading but never steals a
    verdict the IBE chain (or the refute path) certified."""
    _obj, claim, repo = await _setup(
        tmp_path, "cap_no_steal", n_supporting=0, n_contradicting=3
    )
    claim.integrated_assessment = "supports"
    claim.integrated_confidence = 0.8
    claim.integrated_reasoning = "IBE said so."
    claim.integration_candidates = [
        CandidateRecord(
            candidate_id="A", verdict="supports", description="d", chosen=True
        )
    ]
    await repo.save(claim)

    state = _state_at_cap(claim)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.integrated_assessment == "supports"
    assert reloaded.integrated_confidence == 0.8
    assert reloaded.cap_evaluation is not None
    assert reloaded.cap_evaluation.verdict == "contradicts"


async def test_second_cap_firing_revises_only_its_own_verdict(
    tmp_path: Path,
) -> None:
    """Record and verdict can never disagree across repeated firings.

    Fire once on decisive evidence (verdict written). Then flip the pool
    to balanced and fire again: the cap-owned verdict is CLEARED, not
    left stale beside a ``conflicting`` record.
    """
    _obj, claim, repo = await _setup(
        tmp_path, "cap_second_firing", n_supporting=0, n_contradicting=4
    )
    state = _state_at_cap(claim)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    after_first = await repo.get("claim", claim.entity_id)
    assert after_first.integrated_assessment == "contradicts"

    # Flip half the pool to supporting, so the reading becomes an evenly
    # balanced 2-vs-2 — i.e. conflicting, no direction.
    for i, eid in enumerate(after_first.evidence_ids):
        if i >= 2:
            break
        ev = await repo.get("evidence", eid)
        ev.support_judgment = "supports"
        ev.judgment_distribution = [1.0, 0.0, 0.0]
        await repo.save(ev)

    state.claims_needing_rescrutiny.add(claim.entity_id)
    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    after_second = await repo.get("claim", claim.entity_id)
    assert after_second.cap_evaluation is not None
    assert after_second.cap_evaluation.weight_label == "conflicting"
    assert after_second.cap_evaluation.verdict is None
    assert after_second.integrated_assessment is None
    assert after_second.integrated_confidence is None
    assert after_second.integrated_reasoning is None


async def test_cap_never_clears_a_verdict_certified_after_its_own_record(
    tmp_path: Path,
) -> None:
    """THE UNSAFE ORDERING: cap → certify → cap.

    ``test_cap_evaluation_never_overwrites_an_ibe_verdict`` covers only
    the SAFE ordering (certify, then cap). This covers the one that
    actually broke: the cap fires on a non-directional reading and
    records a ``CapEvaluation``; ``PromoteAsRefutedOperation`` then
    certifies "contradicts" and promotes to SUPPORTED (``AbandonOrDemote``
    dispatches it with no ``cycle_capped`` filter, and it guards only on
    stage); the cap fires again. An ownership predicate inferred from
    "a cap record exists" concluded the cap owned the verdict and NULLED
    it, leaving a SUPPORTED claim with no verdict at all.

    Ownership is now recorded explicitly on the CapEvaluation, so the cap
    can only ever revise or clear a verdict it wrote itself.
    """
    _obj, claim, repo = await _setup(
        tmp_path, "cap_then_certify", n_supporting=0, n_contradicting=3, quality=0.1
    )
    state = _state_at_cap(claim)
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    # Cap #1: mass 0.3 is below the weak cut, so no verdict is written.
    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    after_first = await repo.get("claim", claim.entity_id)
    assert after_first.cap_evaluation is not None
    assert after_first.cap_evaluation.weight_label == "weak"
    assert after_first.cap_evaluation.wrote_live_verdict is False
    assert after_first.integrated_assessment is None

    # The refute path certifies a verdict and promotes. It counts items,
    # not mass, so a low-quality 3-vs-0 pool clears its bar (the deliberate
    # C8 filter divergence) — which is exactly how the two mechanical
    # writers end up disagreeing about one claim.
    refute = PromoteAsRefutedOperation(repo, agent_runner=None)
    result = await refute.execute(
        OperationInput(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="promote_as_refuted",
        )
    )
    assert result.success is True, result.message
    certified = await repo.get("claim", claim.entity_id)
    assert certified.integrated_assessment == "contradicts"
    assert certified.stage == ClaimStage.SUPPORTED

    # Cap #2 on the same unchanged (still weak) evidence.
    state.claims_needing_rescrutiny.add(claim.entity_id)
    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    after_second = await repo.get("claim", claim.entity_id)
    assert after_second.integrated_assessment == "contradicts"
    assert after_second.integrated_confidence == certified.integrated_confidence
    assert after_second.integrated_reasoning == certified.integrated_reasoning
    # The fresh reading is still recorded — the cap keeps observing, it
    # just does not own this verdict.
    assert after_second.cap_evaluation is not None
    assert after_second.cap_evaluation.wrote_live_verdict is False


async def test_a_cap_derived_verdict_survives_promotion_and_a_later_demotion_pass(
    tmp_path: Path,
) -> None:
    """THE ERASURE PATH, driven through the real node sequence.

    Scrutinize → AbandonOrDemote → Scrutinize → AbandonOrDemote on a
    capped-with-verdict claim. ``AbandonOrDemote`` promotes failed-scrutiny
    claims of its own accord (refute or soft), and the next Scrutinize
    visit then sees a SUPPORTED claim with ``scrutiny_verdict="fail"`` and
    routes straight back — where ``DemoteClaimOperation`` →
    ``record_demotion`` nulls ``integrated_assessment``, restoring the
    exact false "insufficient" this change exists to remove and leaving
    ``cap_evaluation`` behind so record and verdict disagree.

    The intended end state, now pinned: a cycle-capped claim is never
    demoted, so its verdict survives the round trip.
    """
    _obj, claim, repo = await _setup(
        tmp_path, "cap_erasure", n_supporting=3, n_contradicting=1
    )
    state = _state_at_cap(claim)
    # Exhausted investigation budget → AbandonOrDemote's promote branch.
    state.investigation_counts[claim.entity_id] = PEIRCE_CYCLE_CAP
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    capped = await repo.get("claim", claim.entity_id)
    assert capped.cycle_capped is True
    assert capped.integrated_assessment == "supports"
    verdict_confidence = capped.integrated_confidence

    # Force the failed-scrutiny shape AbandonOrDemote reacts to.
    capped.scrutiny_verdict = "fail"
    await repo.save(capped)

    for _ in range(2):
        await AbandonOrDemote().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
        current = await repo.get("claim", claim.entity_id)
        current.scrutiny_verdict = "fail"
        await repo.save(current)
        await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    final = await repo.get("claim", claim.entity_id)
    assert final.integrated_assessment == "supports", (
        "the cap-derived verdict was erased — record_demotion nulls it, so "
        "AbandonOrDemote must not demote a cycle-capped claim"
    )
    assert final.integrated_confidence == verdict_confidence
    assert final.integrated_reasoning is not None
    assert final.cap_evaluation is not None
    # Record and verdict agree throughout.
    assert final.cap_evaluation.verdict == final.integrated_assessment

    report = await compute_posterior(repo, final.objective_id)
    assert report is not None
    assert report.terminal_state == "completed"
    assert report.posterior > 0.5


# ── Routing: the false-insufficient exit must not fire ───────────────


async def test_capped_claim_with_verdict_routes_to_check_synthesis_demand(
    tmp_path: Path,
) -> None:
    """The primary defect. A decided capped claim must not take
    CheckCompletion's no-certified-verdict exit."""
    _obj, claim, repo = await _setup(
        tmp_path, "cap_routes_csd", n_supporting=1, n_contradicting=3
    )
    state = _state_at_cap(claim)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    next_node = await _run_check_completion(state, deps)
    assert isinstance(next_node, CheckSynthesisDemand)
    assert state.synthesis_insufficient_reason is None


async def test_capped_claim_with_verdict_yields_a_directional_posterior(
    tmp_path: Path,
) -> None:
    """The aggregator agrees with the router. compute_posterior is
    unchanged by this work — the flip is entirely a consequence of
    integrated_assessment now being set, which is what keeps the
    writer/aggregator coherence contract intact by construction."""
    obj, claim, repo = await _setup(
        tmp_path, "cap_posterior", n_supporting=1, n_contradicting=3
    )
    state = _state_at_cap(claim)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    report = await compute_posterior(repo, obj.entity_id)
    assert report is not None
    assert report.terminal_state == "completed"
    assert report.integration_verdict == "contradicts"
    assert report.posterior < 0.5


async def test_cap_confidence_penalty_is_applied_exactly_once(
    tmp_path: Path,
) -> None:
    """CYCLE_CAP_CONFIDENCE_PENALTY is already applied downstream in
    every posterior path. Pre-applying it at write time too would
    silently recalibrate (0.7² = 0.49). This pins that the stored
    confidence is the RAW margin and the penalty lands exactly once."""
    obj, claim, repo = await _setup(
        tmp_path, "cap_penalty_once", n_supporting=1, n_contradicting=3
    )
    state = _state_at_cap(claim)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    reloaded = await repo.get("claim", claim.entity_id)
    # 3 contradicting vs 1 supporting at quality 1.0 → margin (3-1)/4 = 0.5.
    raw_margin = 0.5
    assert reloaded.integrated_confidence == pytest.approx(raw_margin)

    report = await compute_posterior(repo, obj.entity_id)
    assert report is not None
    assert report.posterior == pytest.approx(
        0.5 - raw_margin * CYCLE_CAP_CONFIDENCE_PENALTY / 2, abs=1e-6
    )


async def test_a_cap_derived_verdict_can_never_be_decisive(
    tmp_path: Path,
) -> None:
    """Structural answer to "isn't a mechanical verdict over-confident?".

    (a) The constants compose so the furthest a cap-derived single-claim
        posterior can sit from neutral is strictly inside the decisive
        band. Raising either constant past the safe product fails here.
    (b) A maximum-margin fixture (all evidence one direction) lands
        strictly inside that band end-to-end.
    """
    assert (
        MECHANICAL_VERDICT_MAX_CONFIDENCE * CYCLE_CAP_CONFIDENCE_PENALTY / 2
        < POSTERIOR_DECISIVE_THRESHOLD - 0.5
    )

    obj, claim, repo = await _setup(
        tmp_path, "cap_not_decisive", n_supporting=0, n_contradicting=6
    )
    state = _state_at_cap(claim)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    report = await compute_posterior(repo, obj.entity_id)
    assert report is not None
    assert 1.0 - POSTERIOR_DECISIVE_THRESHOLD < report.posterior
    assert report.posterior < POSTERIOR_DECISIVE_THRESHOLD


async def test_insufficient_is_still_reached_when_the_evidence_did_not_decide(
    tmp_path: Path,
) -> None:
    """The opposite over-correction. The fallibilism terminal (and
    ``Artefact.artefact_type='insufficient'``) must stay reachable and
    meaningful when the evidence genuinely did not decide."""
    _obj, claim, repo = await _setup(
        tmp_path, "cap_still_insufficient", n_supporting=2, n_contradicting=2
    )
    state = _state_at_cap(claim)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    next_node = await _run_check_completion(state, deps)
    assert isinstance(next_node, SynthesizeInsufficient)
    assert state.synthesis_insufficient_reason is not None
    assert "IBE certification" in state.synthesis_insufficient_reason
    # And the record says the system DID look: reading taken, not directional.
    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.cap_evaluation is not None
    assert reloaded.cap_evaluation.n_evidence == 4
    assert reloaded.cap_evaluation.verdict is None


# ── The second false-insufficient door (CheckSynthesisDemand) ────────


async def _setup_all_capped_demand_state(
    tmp_path: Path,
    db_name: str,
    *,
    integrated_assessment: str,
) -> tuple[EpistemicGraphState, EpistemicRepository, EpistemicDeps]:
    """An objective with no combination rule is Gate 1's short-circuit,
    so use a decomposition-free objective only where that is intended.
    Here we need the LLM-judgment path to run, so give it a decomposition
    with a rule and a claim whose verdict aggregates to an ambiguous
    posterior — the same shape test_phase4_synthesis_loop_back.py uses."""
    from andamentum.epistemic.entities.decomposition import (
        Decomposition,
        SubInvestigation,
    )
    from andamentum.epistemic.tests.conftest import FakeAgentRunner

    store = await Store.open(tmp_path / f"{db_name}.db")
    repo = EpistemicRepository(store)

    obj = Objective(
        description="x",
        decomposition=Decomposition(
            sub_investigations=[
                SubInvestigation(id="A", seed_claim="a", rationale="r")
            ],
            combination_rule="AND",
        ),
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="claim A",
        scope="x",
        stage=ClaimStage.HYPOTHESIS,
        sub_investigation_id="A",
        cycle_capped=True,
        integrated_assessment=integrated_assessment,
        integrated_confidence=0.4,
        cap_evaluation=CapEvaluation(
            weight_label="moderate"
            if integrated_assessment != "insufficient"
            else "weak",
            supporting_mass=0.0,
            contradicting_mass=1.0,
            n_evidence=1,
            verdict=("contradicts" if integrated_assessment == "contradicts" else None),
            confidence=0.4 if integrated_assessment == "contradicts" else None,
            wrote_live_verdict=integrated_assessment == "contradicts",
        ),
    )
    await repo.save(claim)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    runner = FakeAgentRunner(
        overrides={
            "epistemic_check_synthesis_demand": {
                "needs_more": True,
                "justification": "Want more, but nothing can progress.",
                "target_hint": "",
            }
        }
    )
    deps = EpistemicDeps(repo=repo, agent_runner=runner, embedding_model="t")
    # Seed the run's verdict object through its real producer, so the
    # gates under test read the same aggregate production would.
    seed_ctx = _FakeRunContext(state, EpistemicDeps(repo=repo, agent_runner=None))
    await CombineClaimVerdicts().run(seed_ctx)  # type: ignore[arg-type]
    return state, repo, deps


async def test_all_capped_but_directionally_certified_routes_to_synthesize(
    tmp_path: Path,
) -> None:
    """The second false-insufficient door. Without this guard, the
    primary fix is undone at the last hop: CheckSynthesisDemand would
    discard a verdict CheckCompletion just certified."""
    state, _repo, deps = await _setup_all_capped_demand_state(
        tmp_path, "csd_directional", integrated_assessment="contradicts"
    )
    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, Synthesize)
    assert state.synthesis_insufficient_reason is None


async def test_all_capped_and_only_insufficient_verdicts_still_routes_insufficient(
    tmp_path: Path,
) -> None:
    """The guard is DIRECTIONAL-only. A claim whose verdict is literally
    "insufficient" is faithfully rendered by the insufficient terminal —
    turning that into a summary would be the opposite over-correction."""
    state, _repo, deps = await _setup_all_capped_demand_state(
        tmp_path, "csd_insufficient", integrated_assessment="insufficient"
    )
    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, SynthesizeInsufficient)
    assert state.synthesis_insufficient_reason is not None


# ── The invariant ────────────────────────────────────────────────────


def test_capped_claim_without_a_cap_evaluation_violates_the_invariant() -> None:
    state = EpistemicGraphState(objective_id="o")
    capped = Claim(objective_id="o", statement="s", cycle_capped=True)
    message = capped_claims_carry_a_cap_evaluation(state, [capped])
    assert message is not None
    assert capped.entity_id[:12] in message


def test_capped_claim_with_a_cap_evaluation_holds() -> None:
    state = EpistemicGraphState(objective_id="o")
    capped = Claim(
        objective_id="o",
        statement="s",
        cycle_capped=True,
        cap_evaluation=CapEvaluation(
            weight_label="weak",
            supporting_mass=0.0,
            contradicting_mass=0.0,
            n_evidence=0,
        ),
    )
    uncapped = Claim(objective_id="o", statement="t")
    assert capped_claims_carry_a_cap_evaluation(state, [capped, uncapped]) is None
    # An uncapped claim with no record never trips it.
    assert capped_claims_carry_a_cap_evaluation(state, [uncapped]) is None


def test_terminal_nodes_declare_the_cap_evaluation_invariant() -> None:
    """post_invariants are not evaluated at runtime, so without this the
    invariant could be silently dropped from a tuple and nothing would
    notice."""
    for node_cls in (CheckCompletion, Synthesize, SynthesizeInsufficient):
        assert capped_claims_carry_a_cap_evaluation in node_cls.post_invariants, (
            f"{node_cls.__name__} must declare "
            "capped_claims_carry_a_cap_evaluation in post_invariants"
        )


# ── Persistence + provenance validity ────────────────────────────────


def _cap_claim_for_persistence() -> Claim:
    return Claim(
        objective_id="obj-1",
        statement="s",
        cycle_capped=True,
        integrated_assessment="contradicts",
        integrated_confidence=0.5,
        integrated_reasoning="Concluded at the cycle cap: 3 vs 1 contradicting mass.",
        cap_evaluation=CapEvaluation(
            weight_label="strong",
            supporting_mass=1.0,
            contradicting_mass=3.0,
            n_evidence=4,
            verdict="contradicts",
            confidence=0.5,
            doubt=0.37,
            wrote_live_verdict=True,
        ),
    )


async def test_cap_evaluation_round_trips_through_the_repository(
    tmp_path: Path,
) -> None:
    """The live persistence path: ``repo.save`` → ``repo.get``.

    Note this goes through ``EpistemicEntity.to_document``'s full-JSON
    content path, NOT ``_extra_metadata`` — so it pins the field's
    presence on the model, not the metadata projection. The metadata
    projection is pinned separately by the test below.
    """
    store = await Store.open(tmp_path / "cap_roundtrip.db")
    repo = EpistemicRepository(store)

    claim = _cap_claim_for_persistence()
    await repo.save(claim)

    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.integrated_reasoning == claim.integrated_reasoning
    assert reloaded.cap_evaluation is not None
    assert reloaded.cap_evaluation == claim.cap_evaluation


def test_cap_evaluation_and_integrated_reasoning_survive_the_metadata_projection() -> (
    None
):
    """The path the ``integrated_reasoning`` data-loss bug actually lived on.

    ``Claim._extra_metadata`` → ``Claim.from_metadata`` is the legacy
    metadata-only reconstruction (live caller:
    ``cli_handlers.py``'s claim-stats path). ``integrated_reasoning`` was
    READ there but never WRITTEN, so provenance prose reconstructed as
    None. ``repo.save``/``repo.get`` do NOT exercise this — they serialise
    the whole model to JSON — so a repository round-trip cannot pin the
    fix. Deleting either key from ``_extra_metadata`` fails here.
    """
    claim = _cap_claim_for_persistence()

    meta = claim._extra_metadata()
    assert meta["integrated_reasoning"] == claim.integrated_reasoning
    assert meta["cap_evaluation"] is not None

    rebuilt = Claim.from_metadata(meta)
    assert rebuilt.integrated_reasoning == claim.integrated_reasoning
    assert rebuilt.integrated_assessment == "contradicts"
    assert rebuilt.cap_evaluation == claim.cap_evaluation
    assert rebuilt.cap_evaluation is not None
    assert rebuilt.cap_evaluation.wrote_live_verdict is True

    # And the no-record case reconstructs as None rather than raising.
    plain = Claim(objective_id="obj-1", statement="s")
    assert Claim.from_metadata(plain._extra_metadata()).cap_evaluation is None


@pytest.mark.parametrize(
    "kwargs",
    [
        # verdict without confidence
        {"weight_label": "strong", "verdict": "supports"},
        # confidence without verdict
        {"weight_label": "strong", "confidence": 0.4},
        # verdict on a non-directional reading
        {"weight_label": "weak", "verdict": "supports", "confidence": 0.4},
        {"weight_label": "conflicting", "verdict": "contradicts", "confidence": 0.4},
    ],
)
def test_cap_evaluation_rejects_an_inconsistent_record(kwargs: dict) -> None:
    """An inconsistent provenance record is unconstructable, so no
    downstream reader has to defend against one."""
    with pytest.raises(ValidationError):
        CapEvaluation(
            supporting_mass=1.0,
            contradicting_mass=0.0,
            n_evidence=1,
            **kwargs,
        )


# ── One evidence reading on both sides of the cap ────────────────────


async def test_the_cap_reads_the_same_evidence_as_scrutiny(
    tmp_path: Path,
) -> None:
    """The masses recorded at the cap equal ``compute_evidence_weight``
    over ``gates.eligible_representative_evidence`` — the same set and
    the same engine scrutiny uses. A future divergence (someone
    re-derives the filter, or swaps in the unweighted counting masses)
    fails here."""
    _obj, claim, repo = await _setup(
        tmp_path, "cap_same_reading", n_supporting=2, n_contradicting=3, quality=0.8
    )
    state = _state_at_cap(claim)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    reloaded = await repo.get("claim", claim.entity_id)
    expected = compute_evidence_weight(
        await gates.eligible_representative_evidence(reloaded, repo)
    )
    assert reloaded.cap_evaluation is not None
    assert reloaded.cap_evaluation.weight_label == expected.label
    assert reloaded.cap_evaluation.supporting_mass == pytest.approx(
        expected.supporting_mass
    )
    assert reloaded.cap_evaluation.contradicting_mass == pytest.approx(
        expected.contradicting_mass
    )


# ── The cap records the doubt of the judgments it stands on ──────────


async def test_cap_evaluation_records_the_doubt_of_its_judgments(
    tmp_path: Path,
) -> None:
    """The cap is the ONE site that mints a verdict with no LLM
    deliberation, so the doubt of the judgments it stands on is the only
    uncertainty it has. ``compute_evidence_weight`` reports it; discarding
    it here would contradict the premise that a judgment carries its
    doubt. PINS that ``CapEvaluation.doubt`` is the aggregator's number,
    not a re-derivation.
    """
    _obj, claim, repo = await _setup(
        tmp_path, "cap_doubt", n_supporting=1, n_contradicting=3
    )
    state = _state_at_cap(claim)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    reloaded = await repo.get("claim", claim.entity_id)
    expected = compute_evidence_weight(
        await gates.eligible_representative_evidence(reloaded, repo)
    )
    assert reloaded.cap_evaluation is not None
    assert expected.doubt is not None
    assert reloaded.cap_evaluation.doubt == pytest.approx(expected.doubt)
    # One-hot judgments: the reading is directional AND the judgments
    # behind it were individually certain.
    assert reloaded.cap_evaluation.doubt == pytest.approx(0.0, abs=1e-9)


async def test_cap_evaluation_doubt_is_none_when_nothing_was_measured(
    tmp_path: Path,
) -> None:
    """Unmeasured is not zero. Evidence judged before the verbalized-
    distribution instrumentation carries a hard ``support_judgment`` and
    no distribution: it contributes MASS but no doubt, so the cap records
    ``doubt=None`` rather than the maximally-certain-looking 0.0.
    """
    _obj, claim, repo = await _setup(
        tmp_path, "cap_doubt_none", n_supporting=1, n_contradicting=3
    )
    for ev_id in claim.evidence_ids:
        ev = await repo.get("evidence", ev_id)
        ev.judgment_distribution = None
        await repo.save(ev)

    state = _state_at_cap(claim)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.cap_evaluation is not None
    assert reloaded.cap_evaluation.doubt is None
    # The mass is still there — the exemption is about doubt only.
    assert reloaded.cap_evaluation.contradicting_mass > 0.0
    assert reloaded.cap_evaluation.verdict == "contradicts"


async def test_the_cap_does_not_write_the_scrutiny_doubt_field(
    tmp_path: Path,
) -> None:
    """``Claim.scrutiny_doubt`` has exactly one writer,
    ``ScrutiniseClaimOperation``. A second writer would make its
    provenance ambiguous — is this the doubt behind the scrutiny verdict,
    or behind the cap reading? The cap's own measurement lives on its own
    record instead."""
    _obj, claim, repo = await _setup(
        tmp_path, "cap_doubt_owner", n_supporting=1, n_contradicting=3
    )
    state = _state_at_cap(claim)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.cap_evaluation is not None
    assert reloaded.cap_evaluation.doubt is not None
    assert reloaded.scrutiny_doubt is None


async def test_conclude_at_cycle_cap_makes_no_llm_call(tmp_path: Path) -> None:
    """The cap-time verdict is deterministic. Driven with no runner at
    all, and with a runner configured to raise on every agent name."""
    # (a) No agent runner.
    _obj, claim, repo = await _setup(
        tmp_path, "cap_no_llm_a", n_supporting=0, n_contradicting=3
    )
    state = _state_at_cap(claim)
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    assert (await repo.get("claim", claim.entity_id)).cap_evaluation is not None

    # (b) A runner configured to raise on EVERY agent name. If the
    # operation ever reaches for the model, this fails loudly instead of
    # silently degrading.
    _obj2, claim2, repo2 = await _setup(
        tmp_path, "cap_no_llm_b", n_supporting=0, n_contradicting=3
    )

    class _AlwaysRaises:
        async def run(self, agent_name: str, **kwargs: object) -> object:
            raise AssertionError(
                "ConcludeAtCycleCapOperation must not call an agent "
                f"(attempted {agent_name!r})"
            )

    op = ConcludeAtCycleCapOperation(repo2, agent_runner=_AlwaysRaises())
    result = await op.execute(
        OperationInput(
            entity_id=claim2.entity_id,
            entity_type="claim",
            operation="conclude_at_cycle_cap",
        )
    )
    assert result.success is True
    reloaded2 = await repo2.get("claim", claim2.entity_id)
    assert reloaded2.integrated_assessment == "contradicts"
