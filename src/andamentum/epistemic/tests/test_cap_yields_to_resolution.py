"""Self-feeding-loop plan D3: the cap yields to an opposing resolution.

``ConcludeAtCycleCapOperation`` reads the claim's eligible evidence with
no LLM call and, when the reading is directional, writes the verdict it
supports. This suite pins the veto layered on top: before trusting that
mass-derived direction, the cap checks whether any RESOLVED, BLOCKING
``Uncertainty`` affecting the claim carries a bearing (D2) that points
the other way. If so, it downgrades to ``weight_label="conflicting"``
and writes no verdict, rather than a confidently wrong direction.

S4: both branches (agreement/None -> writes; opposition -> abstains),
plus the ownership case where a previous cap-owned verdict must be
CLEARED when a later firing yields.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import Store
from andamentum.epistemic.entities import Claim, Evidence, Objective, Uncertainty
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.entities.uncertainty import UncertaintyType
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import (
    SCRUTINY_RESOLVE_CYCLE_CAP,
    ResolveUncertainties,
)
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.operations.base import OperationInput
from andamentum.epistemic.operations.stage_management import ConcludeAtCycleCapOperation
from andamentum.epistemic.repository import EpistemicRepository

from .conftest import FakeAgentRunner


class _FakeRunContext:
    def __init__(self, state, deps):
        self.state = state
        self.deps = deps


async def _setup(
    tmp_path: Path,
    db_name: str,
    *,
    n_supporting: int,
    n_contradicting: int,
) -> tuple[Objective, Claim, EpistemicRepository]:
    """A HYPOTHESIS claim with N judged, one-hot-quality evidence items —
    same shape as ``test_cap_terminal_verdict._setup``, so the mass
    reading's own direction is unambiguous and this suite exercises only
    the D3 veto layered on top of it."""
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
            quality_score=1.0,
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
            quality_score=1.0,
        )
        await repo.save(ev)
        evidence_ids.append(ev.entity_id)

    claim.evidence_ids = evidence_ids
    await repo.save(claim)
    return obj, claim, repo


async def _run_cap(repo: EpistemicRepository, claim: Claim) -> Claim:
    op = ConcludeAtCycleCapOperation(repo=repo)
    result = await op.execute(
        OperationInput(
            entity_id=claim.entity_id, entity_type="claim", operation="conclude"
        )
    )
    assert result.success, result.message
    reloaded = await repo.get("claim", claim.entity_id)
    assert isinstance(reloaded, Claim)
    return reloaded


# ── Branch: bearings agree (or are None) -> the cap writes ─────────────


async def test_cap_writes_when_no_resolution_opposes(tmp_path: Path) -> None:
    """Baseline: three refuting studies vs one supporting one, no
    resolved uncertainty at all. The cap writes 'contradicts' as before —
    this is the pre-D3 regression test, unaffected by the new veto."""
    _obj, claim, repo = await _setup(
        tmp_path, "cap_yield_none", n_supporting=1, n_contradicting=3
    )
    reloaded = await _run_cap(repo, claim)

    assert reloaded.integrated_assessment == "contradicts"
    assert reloaded.cap_evaluation is not None
    assert reloaded.cap_evaluation.verdict == "contradicts"
    assert reloaded.cap_evaluation.wrote_live_verdict is True
    assert reloaded.cap_evaluation.yielded_to_resolutions == []


async def test_cap_writes_when_resolution_bearing_agrees(tmp_path: Path) -> None:
    """A resolved uncertainty whose bearing AGREES with the mass reading's
    direction never vetoes it."""
    _obj, claim, repo = await _setup(
        tmp_path, "cap_yield_agree", n_supporting=1, n_contradicting=3
    )
    agreeing = Uncertainty(
        objective_id=claim.objective_id,
        description="A concern that, once resolved, still points the same way.",
        uncertainty_type=UncertaintyType.CONTRADICTION,
        affected_claim_ids=[claim.entity_id],
    )
    agreeing.resolve(
        "Confirmed: the effect is real and refutes the claim.", bearing="refutes_claim"
    )
    await repo.save(agreeing)

    reloaded = await _run_cap(repo, claim)

    assert reloaded.integrated_assessment == "contradicts"
    assert reloaded.cap_evaluation is not None
    assert reloaded.cap_evaluation.verdict == "contradicts"
    assert reloaded.cap_evaluation.wrote_live_verdict is True
    assert reloaded.cap_evaluation.yielded_to_resolutions == []


async def test_cap_writes_when_only_neutral_or_unassessed_resolutions_exist(
    tmp_path: Path,
) -> None:
    """A 'neutral' bearing, and an unresolved/unassessed uncertainty
    (resolution_bearing=None), never oppose anything."""
    _obj, claim, repo = await _setup(
        tmp_path, "cap_yield_neutral", n_supporting=1, n_contradicting=3
    )
    neutral = Uncertainty(
        objective_id=claim.objective_id,
        description="A tangential concern.",
        uncertainty_type=UncertaintyType.CONTRADICTION,
        affected_claim_ids=[claim.entity_id],
    )
    neutral.resolve(
        "Addressed, but it doesn't bear on truth either way.", bearing="neutral"
    )
    await repo.save(neutral)

    unassessed = Uncertainty(
        objective_id=claim.objective_id,
        description="Resolved by an older code path with no bearing recorded.",
        uncertainty_type=UncertaintyType.UNKNOWN,
        affected_claim_ids=[claim.entity_id],
    )
    unassessed.resolve("Deduplicated.")  # no bearing kwarg -> None
    await repo.save(unassessed)

    reloaded = await _run_cap(repo, claim)

    assert reloaded.integrated_assessment == "contradicts"
    assert reloaded.cap_evaluation is not None
    assert reloaded.cap_evaluation.wrote_live_verdict is True
    assert reloaded.cap_evaluation.yielded_to_resolutions == []


# ── Branch: an opposing resolution -> the cap abstains ──────────────────


async def test_cap_abstains_when_resolution_bearing_opposes_supports(
    tmp_path: Path,
) -> None:
    """Mass reading says 'supports'; a resolved uncertainty says
    'refutes_claim'. The cap must write no direction."""
    _obj, claim, repo = await _setup(
        tmp_path, "cap_yield_oppose_supports", n_supporting=3, n_contradicting=1
    )
    opposing = Uncertainty(
        objective_id=claim.objective_id,
        description="The apparent support doesn't survive scrutiny.",
        uncertainty_type=UncertaintyType.CONTRADICTION,
        affected_claim_ids=[claim.entity_id],
    )
    opposing.resolve(
        "On inspection, the supporting studies shared a flawed protocol; "
        "the claim is not actually supported.",
        bearing="refutes_claim",
    )
    await repo.save(opposing)

    reloaded = await _run_cap(repo, claim)

    assert reloaded.integrated_assessment is None
    assert reloaded.integrated_confidence is None
    assert reloaded.integrated_reasoning is None
    assert reloaded.cap_evaluation is not None
    assert reloaded.cap_evaluation.weight_label == "conflicting"
    assert reloaded.cap_evaluation.verdict is None
    assert reloaded.cap_evaluation.confidence is None
    assert reloaded.cap_evaluation.wrote_live_verdict is False
    assert reloaded.cap_evaluation.yielded_to_resolutions == [opposing.entity_id]


async def test_cap_abstains_when_resolution_bearing_opposes_contradicts(
    tmp_path: Path,
) -> None:
    """Mirror case: mass reading says 'contradicts'; a resolved
    uncertainty says 'supports_claim'."""
    _obj, claim, repo = await _setup(
        tmp_path, "cap_yield_oppose_contradicts", n_supporting=1, n_contradicting=3
    )
    opposing = Uncertainty(
        objective_id=claim.objective_id,
        description="The apparent refutation doesn't survive scrutiny.",
        uncertainty_type=UncertaintyType.CONTRADICTION,
        affected_claim_ids=[claim.entity_id],
    )
    opposing.resolve(
        "On inspection, the refuting studies used a different endpoint; "
        "the claim is actually supported.",
        bearing="supports_claim",
    )
    await repo.save(opposing)

    reloaded = await _run_cap(repo, claim)

    assert reloaded.integrated_assessment is None
    assert reloaded.cap_evaluation is not None
    assert reloaded.cap_evaluation.weight_label == "conflicting"
    assert reloaded.cap_evaluation.verdict is None
    assert reloaded.cap_evaluation.wrote_live_verdict is False
    assert reloaded.cap_evaluation.yielded_to_resolutions == [opposing.entity_id]


async def test_cap_ignores_opposing_resolution_that_is_non_blocking(
    tmp_path: Path,
) -> None:
    """A resolved uncertainty of a NON-BLOCKING type (a caveat, not a
    promotion-blocker) must never veto the cap's verdict even when its
    bearing opposes the mass reading's direction — only RESOLVED,
    BLOCKING uncertainties are eligible to veto (plan D3's letter)."""
    _obj, claim, repo = await _setup(
        tmp_path, "cap_yield_non_blocking", n_supporting=3, n_contradicting=1
    )
    non_blocking_opposing = Uncertainty(
        objective_id=claim.objective_id,
        description="A caveat that happens to point the other way.",
        uncertainty_type=UncertaintyType.EVIDENCE_GAP,
        affected_claim_ids=[claim.entity_id],
    )
    assert non_blocking_opposing.is_blocking is False
    non_blocking_opposing.resolve(
        "Addressed, but this was always advisory, not a blocker.",
        bearing="refutes_claim",
    )
    await repo.save(non_blocking_opposing)

    reloaded = await _run_cap(repo, claim)

    assert reloaded.integrated_assessment == "supports"
    assert reloaded.cap_evaluation is not None
    assert reloaded.cap_evaluation.verdict == "supports"
    assert reloaded.cap_evaluation.wrote_live_verdict is True
    assert reloaded.cap_evaluation.yielded_to_resolutions == []


async def test_cap_ignores_opposing_resolution_on_an_unaffected_claim(
    tmp_path: Path,
) -> None:
    """An opposing resolution that does NOT name this claim in
    ``affected_claim_ids`` must never veto it — scoping is per-claim."""
    _obj, claim, repo = await _setup(
        tmp_path, "cap_yield_unaffected", n_supporting=1, n_contradicting=3
    )
    other_claim = Claim(
        objective_id=claim.objective_id,
        statement="A different claim entirely.",
        scope="test",
    )
    await repo.save(other_claim)

    opposing_elsewhere = Uncertainty(
        objective_id=claim.objective_id,
        description="Opposes a different claim.",
        uncertainty_type=UncertaintyType.CONTRADICTION,
        affected_claim_ids=[other_claim.entity_id],
    )
    opposing_elsewhere.resolve("Refutes the OTHER claim.", bearing="refutes_claim")
    await repo.save(opposing_elsewhere)

    reloaded = await _run_cap(repo, claim)

    assert reloaded.integrated_assessment == "contradicts"
    assert reloaded.cap_evaluation is not None
    assert reloaded.cap_evaluation.wrote_live_verdict is True
    assert reloaded.cap_evaluation.yielded_to_resolutions == []


# ── Ownership: a yield must CLEAR a previously cap-owned verdict ───────


async def test_yield_clears_a_previously_cap_owned_verdict(tmp_path: Path) -> None:
    """First firing: no opposing resolution -> the cap writes and owns
    the verdict. Between firings, a resolution comes in that opposes that
    direction. Second firing: the cap must yield AND clear what it wrote,
    since the reading it rested on is now contested — never leave a
    written verdict standing behind a 'conflicting' record."""
    _obj, claim, repo = await _setup(
        tmp_path, "cap_yield_clears_own_verdict", n_supporting=1, n_contradicting=3
    )

    after_first = await _run_cap(repo, claim)
    assert after_first.integrated_assessment == "contradicts"
    assert after_first.cap_evaluation is not None
    assert after_first.cap_evaluation.wrote_live_verdict is True

    opposing = Uncertainty(
        objective_id=claim.objective_id,
        description="A new resolution disputes the earlier reading.",
        uncertainty_type=UncertaintyType.CONTRADICTION,
        affected_claim_ids=[claim.entity_id],
    )
    opposing.resolve(
        "The refuting studies don't hold up; claim is supported.",
        bearing="supports_claim",
    )
    await repo.save(opposing)

    after_second = await _run_cap(repo, after_first)

    assert after_second.integrated_assessment is None
    assert after_second.integrated_confidence is None
    assert after_second.integrated_reasoning is None
    assert after_second.cap_evaluation is not None
    assert after_second.cap_evaluation.weight_label == "conflicting"
    assert after_second.cap_evaluation.verdict is None
    assert after_second.cap_evaluation.wrote_live_verdict is False
    assert after_second.cap_evaluation.yielded_to_resolutions == [opposing.entity_id]


async def test_yield_does_not_clear_a_verdict_it_does_not_own(tmp_path: Path) -> None:
    """A verdict certified by another writer (e.g. IBE / refute-promotion)
    must never be cleared by a cap that yields — ownership still gates
    clearing exactly as it gates writing."""
    _obj, claim, repo = await _setup(
        tmp_path, "cap_yield_foreign_verdict", n_supporting=1, n_contradicting=3
    )
    # Simulate a verdict certified by a non-cap writer: integrated_assessment
    # set with no cap_evaluation.wrote_live_verdict backing it.
    claim.integrated_assessment = "supports"
    claim.integrated_confidence = 0.9
    claim.integrated_reasoning = "Certified by the IBE chain."
    await repo.save(claim)

    opposing = Uncertainty(
        objective_id=claim.objective_id,
        description="A resolution disagrees with the certified verdict.",
        uncertainty_type=UncertaintyType.CONTRADICTION,
        affected_claim_ids=[claim.entity_id],
    )
    opposing.resolve(
        "The refuting studies don't hold up; claim is supported.",
        bearing="supports_claim",
    )
    await repo.save(opposing)

    reloaded = await _run_cap(repo, claim)

    # The mass reading here is "contradicts", opposed by a supports_claim
    # resolution, so the cap yields (writes nothing) -- but it must not
    # touch the pre-existing foreign verdict at all.
    assert reloaded.integrated_assessment == "supports"
    assert reloaded.integrated_confidence == 0.9
    assert reloaded.integrated_reasoning == "Certified by the IBE chain."
    assert reloaded.cap_evaluation is not None
    assert reloaded.cap_evaluation.wrote_live_verdict is False
    assert reloaded.cap_evaluation.yielded_to_resolutions == [opposing.entity_id]


# ── Graph-level: the node's repeat-firing skip must not shield a
#    cap-owned verdict from an opposing resolution that arrives after
#    the claim's first cap firing (self-feeding-loop plan D3, node fix
#    to ``_mark_cycle_capped``'s fingerprint comparison) ──────────────


async def test_a_later_opposing_resolution_reruns_the_cap_via_resolve_uncertainties(
    tmp_path: Path,
) -> None:
    """Claim A and claim B share nothing evidentially, but B keeps a
    blocking uncertainty live past A's own cycle cap. A's first cap
    firing (via ``ResolveUncertainties``) writes 'supports' on an
    unopposed reading. Later, a SEPARATE blocking uncertainty that also
    affects A (and B, which stays live) resolves with a bearing that
    opposes 'supports'. ``_mark_cycle_capped`` must re-run the
    mechanical conclusion on this repeat firing — the evidence fingerprint
    alone is unchanged, but the resolution-bearing fingerprint is not —
    and the operation's own veto must then clear A's cap-owned verdict.
    B, never having reached the cycle cap, must stay live throughout."""
    _obj, claim_a, repo = await _setup(
        tmp_path, "cap_yield_graph_a_caps_first", n_supporting=3, n_contradicting=1
    )
    obj = _obj
    claim_b = Claim(
        objective_id=obj.entity_id,
        statement="A different claim that shares the second uncertainty.",
        scope="test",
        stage=ClaimStage.HYPOTHESIS,
    )
    await repo.save(claim_b)

    # Only affects A: unresolved, and its resolution (bearing=None, the
    # fake runner's default) is what drives A's FIRST cap firing.
    unc_a_only = Uncertainty(
        objective_id=obj.entity_id,
        description="A concern specific to claim A.",
        uncertainty_type=UncertaintyType.CONTRADICTION,
        affected_claim_ids=[claim_a.entity_id],
    )
    await repo.save(unc_a_only)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    state.scrutiny_resolve_cycles[claim_a.entity_id] = SCRUTINY_RESOLVE_CYCLE_CAP
    state.scrutiny_resolve_cycles[claim_b.entity_id] = 0
    deps = EpistemicDeps(repo=repo, agent_runner=FakeAgentRunner(), embedding_model="t")

    # ── First firing: A's only live uncertainty resolves with no bearing
    #    (the fake runner's default) -> no opposition -> the cap writes.
    await ResolveUncertainties().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    after_first = await repo.get("claim", claim_a.entity_id)
    assert after_first.cycle_capped is True
    assert after_first.integrated_assessment == "supports"
    assert after_first.cap_evaluation is not None
    assert after_first.cap_evaluation.wrote_live_verdict is True

    # ── A second, still-unresolved blocking uncertainty affects BOTH
    #    claims. B keeps it live (B is nowhere near its own cap), so it
    #    is dispatched for resolution, and this time the fake runner is
    #    overridden to return an opposing bearing.
    unc_shared = Uncertainty(
        objective_id=obj.entity_id,
        description="A concern shared between claim A and claim B.",
        uncertainty_type=UncertaintyType.CONTRADICTION,
        affected_claim_ids=[claim_a.entity_id, claim_b.entity_id],
    )
    await repo.save(unc_shared)
    deps = EpistemicDeps(
        repo=repo,
        agent_runner=FakeAgentRunner(
            {
                "epistemic_resolve_uncertainty": {
                    "resolution": "On inspection, A's apparent support doesn't hold up.",
                    "can_resolve": True,
                    "bearing": "refutes_claim",
                    "remaining_concerns": [],
                }
            }
        ),
        embedding_model="t",
    )

    # ── Second firing: unchanged evidence, but a NEW resolved blocking
    #    bearing opposes what A's cap already wrote. The repeat-firing
    #    skip must not shield that verdict.
    await ResolveUncertainties().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    after_second = await repo.get("claim", claim_a.entity_id)
    assert after_second.integrated_assessment is None, (
        "the cap's earlier 'supports' verdict survived an opposing "
        "resolution that arrived after the first cap firing — the "
        "repeat-firing skip gated only on the evidence fingerprint, "
        "which never changed"
    )
    assert after_second.integrated_confidence is None
    assert after_second.integrated_reasoning is None
    assert after_second.cap_evaluation is not None
    assert after_second.cap_evaluation.weight_label == "conflicting"
    assert after_second.cap_evaluation.wrote_live_verdict is False
    assert after_second.cap_evaluation.yielded_to_resolutions == [unc_shared.entity_id]

    # B never reached its own cycle cap and must remain live.
    after_b = await repo.get("claim", claim_b.entity_id)
    assert after_b.cycle_capped is False
