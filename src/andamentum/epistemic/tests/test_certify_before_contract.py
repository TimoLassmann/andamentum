"""Graph-level regression tests for certify-before-contract
(docs/epistemic/CERTIFY_BEFORE_CONTRACT_PLAN.md, definition-of-done §0).

Covers, at the node level (model on ``test_post_verification_demotion.py``):

* S1 — a claim whose adversarial/support-balance track finds refutation
  still receives its IBE verdict, and the resulting contraction happens
  AFTER that verdict was written, preserving it.
* S2 — a soft-promoted claim receives its IBE verdict and is never
  demoted by ``validate_current_stage`` before IBE runs.
* S3 (graph part) — two passes over a claim with an UNCHANGED evidence
  fingerprint dispatch the adversarial verification track exactly once;
  the second pass's re-promotion attempt is refused by
  ``promotion_blocked_by_contraction`` (D3, Phase A) before it ever
  reaches ``RunVerification`` again. The pure-function coverage of
  ``promotion_blocked_by_contraction`` / ``evidence_fingerprint``
  themselves lives in ``test_evidence_fingerprint.py``.
* S4 — after a contraction, the claim is never routed back through
  ``Scrutinize`` for this run: ``scrutiny_verdict`` and
  ``claims_needing_rescrutiny`` are exactly what ``Claim.contract``
  leaves them (untouched / not added), so ``Scrutinize`` skips it.
* S5 — a cycle-capped claim is terminal: ``ResolveUncertainties`` does
  not dispatch ``resolve_uncertainty`` for a blocking uncertainty whose
  every affected claim is already cycle-capped or abandoned, and
  ``conclude_at_cycle_cap`` (via ``_mark_cycle_capped``) fires once per
  genuine cap firing plus once per evidence change afterwards — never
  once per re-entry on unchanged evidence.
* S6 — ``PromoteSupported``'s ladder follows the verdict (D8): a claim
  with no ``integrated_assessment``, or one certified
  ``contradicts``/``insufficient``, gets no ``promote_claim`` attempt
  and is simply marked ``verification_done``; only a claim certified
  ``supports`` climbs.

No LLM calls anywhere in this file: ``AdversarialSearchOperation``
degrades to a clean structural no-op with ``agent_runner=None`` (marks
``adversarial_checked=True`` without ever reaching the agent-calling
branches — see ``test_post_verification_demotion.py``'s own note on this),
and ``RevalidateClaimOperation`` / ``PromoteClaimOperation`` /
``ScrutiniseClaimOperation``'s dispatch/skip decisions used here are all
deterministic, no-model checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, Evidence, Objective, Uncertainty
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.entities.uncertainty import UncertaintyType
from andamentum.epistemic.gates import claim_evidence_fingerprint
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import (
    SCRUTINY_RESOLVE_CYCLE_CAP,
    EnumerateCandidates,
    PromoteSupported,
    PromoteToSupported,
    ResolveUncertainties,
    RunVerification,
    Scrutinize,
)
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.repository import EpistemicRepository


class _FakeRunContext:
    def __init__(self, state, deps):
        self.state = state
        self.deps = deps


async def _make_repo(tmp_path: Path, name: str) -> EpistemicRepository:
    store = DocumentStore.for_database(name, db_dir=tmp_path)
    await store.initialize()
    return EpistemicRepository(store)


def _ops(state: EpistemicGraphState, operation: str) -> list:
    return [op for op in state.operations_log if op.operation == operation]


# ── S1 ────────────────────────────────────────────────────────────────


async def test_contraction_preserves_the_ibe_verdict_written_before_it(
    tmp_path: Path,
) -> None:
    """A claim whose adversarial track found refutation
    (``adversarial_balance < ADVERSARIAL_REFUTED_THRESHOLD``) still gets
    a real IBE verdict written by ``SelectBestExplanation`` — driven
    through the actual enumerate/score/score/select chain with the fake
    runner, not a fixture field standing in for it — and only THEN is it
    contracted by ``PromoteSupported``. The contraction happens after the
    verdict was written and preserves it (S1, S8).
    """
    from .conftest import FakeAgentRunner

    repo = await _make_repo(tmp_path, "certify_s1")
    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    # Two supporting, no contradicting evidence: isolates the
    # contraction to the ADVERSARIAL-BALANCE branch of
    # validate_current_stage (2-a below), not the support-count branch
    # (which needs >=2 judged items with contradicting >= supporting).
    ev_sup1 = Evidence(
        objective_id=obj.entity_id,
        source_type="web",
        source_ref="https://ex.com/sup1",
        extracted_content="supporting item 1",
        extracted=True,
        support_judgment="supports",
        quality_score=0.5,
    )
    ev_sup2 = Evidence(
        objective_id=obj.entity_id,
        source_type="web",
        source_ref="https://ex.com/sup2",
        extracted_content="supporting item 2",
        extracted=True,
        support_judgment="supports",
        quality_score=0.5,
    )
    for ev in (ev_sup1, ev_sup2):
        await repo.save(ev)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="claim under certify-before-contract test",
        scope="x",
        stage=ClaimStage.SUPPORTED,
        scrutiny_verdict="pass",
        evidence_ids=[ev_sup1.entity_id, ev_sup2.entity_id],
        evidence_count=2,
        integrated_assessment=None,  # IBE has not run yet
        # Adversarial search already ran and found refutation: balance
        # below ADVERSARIAL_REFUTED_THRESHOLD (0.3) is the gate reason
        # this test drives, isolated from every other verification
        # track (all pre-marked checked, same isolation trick as
        # test_post_verification_demotion.py).
        adversarial_checked=True,
        adversarial_balance=0.2,
        convergence_checked=True,
        deductive_checked=True,
        computational_checked=True,
        argument_analyzed=True,
    )
    await repo.save(claim)
    assert claim.promotion_history == []  # nothing has revised this claim yet

    # Bias the enumerator toward a "supports" framing on every slot so
    # enumeration clears the 2-genuine-candidate bar for real (the
    # default fake response is done=True on the first call, which
    # declines — see test_enumerate_declines_with_default_fake_runner).
    # Loveliness/likeliness/select-best-explanation stay on their
    # conftest defaults; this is the same override pattern
    # test_integration_operation.py's balanced-enumeration tests use,
    # not a new fake capability.
    runner = FakeAgentRunner(
        overrides={
            "epistemic_propose_one_candidate": {
                "done": False,
                "verdict": "supports",
                "description": "Mechanism consistent with supporting evidence.",
            },
        }
    )

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=runner, embedding_model="t")

    # Drive RunVerification -> ResolveUncertainties -> EnumerateCandidates
    # -> ScoreLoveliness -> ScoreLikeliness -> SelectBestExplanation ->
    # PromoteSupported, exactly as the graph orders it. No blocking
    # uncertainties and no rescrutiny flag exist, so each node's
    # no-further-work branch chains straight to the next.
    node: Any = RunVerification()
    for _ in range(7):
        node = await node.run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    next_node = node

    reloaded = await repo.get("claim", claim.entity_id)

    # SelectBestExplanation ran for real and wrote a verdict.
    assert reloaded.integrated_assessment == "supports"
    assert reloaded.integrated_confidence is not None

    # The contraction happened after that verdict was written:
    # adversarial_balance=0.2 fails validate_current_stage's
    # adversarial-refutation check.
    assert reloaded.stage == ClaimStage.HYPOTHESIS
    assert reloaded.contraction_count == 1
    assert len(reloaded.promotion_history) == 1
    entry = reloaded.promotion_history[0]
    assert entry.from_stage == ClaimStage.SUPPORTED
    assert entry.to_stage == ClaimStage.HYPOTHESIS
    assert "AGM contraction" in entry.justification
    assert "Adversarial balance" in entry.justification, (
        f"Expected the adversarial-balance gate reason in the "
        f"contraction justification, got: {entry.justification!r}"
    )

    # S1's actual claim: the verdict IBE wrote survives the contraction
    # untouched — contract() is a revision, not a restart.
    verdict_before = reloaded.integrated_assessment
    confidence_before = reloaded.integrated_confidence
    assert verdict_before == "supports"

    # The contraction fingerprint was stamped from the current evidence.
    expected_fp = await claim_evidence_fingerprint(reloaded, repo)
    assert reloaded.contraction_fingerprint == expected_fp

    # select_best_explanation ran BEFORE revalidate_claim contracted it.
    op_names = [op.operation for op in state.operations_log]
    idx_select = op_names.index("select_best_explanation")
    idx_revalidate = op_names.index("revalidate_claim")
    assert idx_select < idx_revalidate, (
        "revalidate_claim ran before select_best_explanation — the "
        "contraction check must run AFTER IBE has certified a verdict."
    )
    revalidate_ops = _ops(state, "revalidate_claim")
    assert len(revalidate_ops) == 1
    assert "Adversarial balance" in revalidate_ops[0].message

    # Verdict fields are unchanged by the contraction: same values
    # before (as loaded right after the graph run) and after re-reading.
    reread = await repo.get("claim", claim.entity_id)
    assert reread.integrated_assessment == verdict_before
    assert reread.integrated_confidence == confidence_before

    # Contracted claims don't climb the ladder this pass, but they are
    # marked done so PromoteToSupported's idempotency check short-circuits.
    assert claim.entity_id in state.verification_done
    assert _ops(state, "promote_claim") == [], (
        "A contracted claim must not have PromoteClaimOperation attempted "
        "on it this pass — it does not climb the ladder."
    )
    from andamentum.epistemic.graph.nodes import CombineClaimVerdicts

    assert isinstance(next_node, CombineClaimVerdicts)


# ── S2 ────────────────────────────────────────────────────────────────


async def test_soft_promoted_claim_reaches_ibe_before_any_revalidation(
    tmp_path: Path,
) -> None:
    """A soft-promoted claim (SUPPORTED, ``integrated_assessment=None``,
    balanced supports/contradicts counts, investigation cap reached —
    the "743 shape") must run through to a real IBE verdict without
    being demoted by ``validate_current_stage`` first.

    Drives ``RunVerification`` on a claim whose support-balance would
    ALREADY fail ``validate_current_stage`` if anything checked it —
    proving nothing in ``RunVerification`` (nor the now-Step-1-only
    ``_run_tms_sweep`` it calls) revalidates the claim — then continues
    the SAME state through the real enumerate/score/score/select chain
    (fake runner, not a fixture write) so ``SelectBestExplanation``
    writes the verdict before ``PromoteSupported``'s revalidation ever
    runs.
    """
    from .conftest import FakeAgentRunner
    from ..thresholds import PEIRCE_CYCLE_CAP

    repo = await _make_repo(tmp_path, "certify_s2")
    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    # Balanced/losing support: 1 supports, 2 contradicts — same pattern
    # validate_current_stage would refuse, set up BEFORE verification
    # runs (as a soft-promoted claim's evidence looks at soft-promote
    # time: directional but not conclusively one-sided).
    ev_sup = Evidence(
        objective_id=obj.entity_id,
        source_type="web",
        source_ref="https://ex.com/sup",
        extracted_content="supporting item",
        extracted=True,
        support_judgment="supports",
        quality_score=0.5,
    )
    ev_con1 = Evidence(
        objective_id=obj.entity_id,
        source_type="web",
        source_ref="https://ex.com/con1",
        extracted_content="counter 1",
        extracted=True,
        support_judgment="contradicts",
        quality_score=0.5,
    )
    ev_con2 = Evidence(
        objective_id=obj.entity_id,
        source_type="web",
        source_ref="https://ex.com/con2",
        extracted_content="counter 2",
        extracted=True,
        support_judgment="contradicts",
        quality_score=0.5,
    )
    for ev in (ev_sup, ev_con1, ev_con2):
        await repo.save(ev)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="soft-promoted claim, investigation cap reached",
        scope="x",
        stage=ClaimStage.SUPPORTED,
        scrutiny_verdict="pass",
        evidence_ids=[ev_sup.entity_id, ev_con1.entity_id, ev_con2.entity_id],
        evidence_count=3,
        integrated_assessment=None,  # soft-promote defers the verdict to IBE
        # Every OTHER verification track pre-marked checked, isolating the
        # routing property under test (whether validate_current_stage runs
        # before IBE) from the tracks themselves — same isolation trick as
        # test_post_verification_demotion.py. adversarial_checked is
        # deliberately left False: pre-G3, the adversarial track was the
        # trigger that flagged claims_needing_tms (since removed — G3);
        # it must still actually dispatch under the fake runner for this
        # test to be evidence that nothing in RunVerification (nor
        # _run_tms_sweep) revalidates the claim.
        adversarial_checked=False,
        convergence_checked=True,
        deductive_checked=True,
        computational_checked=True,
        argument_analyzed=True,
    )
    await repo.save(claim)

    # 743 shape: the investigation and scrutiny-resolve cycle counts are
    # both at PEIRCE_CYCLE_CAP — this claim's inquiry cycle is spent,
    # which is how a real run reaches "soft-promoted, balanced evidence,
    # verdict deferred to IBE" in the first place.
    state = EpistemicGraphState(objective_id=obj.entity_id)
    state.investigation_counts[claim.entity_id] = PEIRCE_CYCLE_CAP
    state.scrutiny_resolve_cycles[claim.entity_id] = PEIRCE_CYCLE_CAP

    runner = FakeAgentRunner(
        overrides={
            "epistemic_propose_one_candidate": {
                "done": False,
                "verdict": "insufficient",
                "description": "Evidence is genuinely mixed.",
            },
        }
    )
    deps = EpistemicDeps(repo=repo, agent_runner=runner, embedding_model="t")

    node = await RunVerification().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    # Nothing revalidated the claim during verification — it is exactly
    # as it was: SUPPORTED, integrated_assessment still None (IBE has not
    # run yet), no rescrutiny flag.
    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.stage == ClaimStage.SUPPORTED, (
        "validate_current_stage was applied before IBE ran — the claim "
        "would fail it (2 contradicts >= 1 supports) if it had been."
    )
    assert reloaded.integrated_assessment is None
    assert reloaded.contraction_count == 0
    assert _ops(state, "revalidate_claim") == []
    assert claim.entity_id not in state.claims_needing_rescrutiny
    assert len(_ops(state, "adversarial_search")) == 1, (
        "The adversarial track must actually dispatch under the fake "
        "runner — otherwise the assertions above are vacuous."
    )

    # Continue the SAME state/deps through the real IBE chain ->
    # PromoteSupported: EnumerateCandidates -> ScoreLoveliness ->
    # ScoreLikeliness -> SelectBestExplanation -> PromoteSupported ->
    # CombineClaimVerdicts.
    for _ in range(6):
        node = await node.run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    op_names = [op.operation for op in state.operations_log]
    idx_select = op_names.index("select_best_explanation")
    idx_revalidate = op_names.index("revalidate_claim")
    assert idx_select < idx_revalidate, (
        "PromoteSupported's revalidate_claim ran before "
        "select_best_explanation wrote the verdict."
    )

    final = await repo.get("claim", claim.entity_id)
    # PromoteSupported's revalidation now applies the gate — for the
    # first time, and only after IBE actually wrote a verdict.
    assert final.integrated_assessment is not None
    assert final.stage == ClaimStage.HYPOTHESIS
    assert final.contraction_count == 1
    assert final.integrated_assessment == "insufficient", (
        "The verdict IBE wrote must survive PromoteSupported's contraction."
    )
    from andamentum.epistemic.graph.nodes import CombineClaimVerdicts

    assert isinstance(node, CombineClaimVerdicts)


# ── S3 (graph part) ─────────────────────────────────────────────────────


async def test_two_passes_on_unchanged_evidence_dispatch_adversarial_once(
    tmp_path: Path,
) -> None:
    """Drive two passes over one claim on an UNCHANGED evidence
    fingerprint: RunVerification -> PromoteSupported (which contracts
    it), then a second pass attempting re-promotion via
    PromoteToSupported. The second pass's re-promotion is refused by
    ``promotion_blocked_by_contraction`` (D3) before the claim can ever
    reach SUPPORTED — and therefore RunVerification's adversarial track
    — again. Across both passes, exactly one adversarial_search row.
    """
    repo = await _make_repo(tmp_path, "certify_s3")
    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    ev_sup = Evidence(
        objective_id=obj.entity_id,
        source_type="web",
        source_ref="https://ex.com/sup",
        extracted_content="supporting item",
        extracted=True,
        support_judgment="supports",
        quality_score=0.5,
    )
    ev_con1 = Evidence(
        objective_id=obj.entity_id,
        source_type="web",
        source_ref="https://ex.com/con1",
        extracted_content="counter 1",
        extracted=True,
        support_judgment="contradicts",
        quality_score=0.5,
    )
    ev_con2 = Evidence(
        objective_id=obj.entity_id,
        source_type="web",
        source_ref="https://ex.com/con2",
        extracted_content="counter 2",
        extracted=True,
        support_judgment="contradicts",
        quality_score=0.5,
    )
    for ev in (ev_sup, ev_con1, ev_con2):
        await repo.save(ev)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="claim exercised across two passes, evidence never changes",
        scope="x",
        stage=ClaimStage.SUPPORTED,
        scrutiny_verdict="pass",
        evidence_ids=[ev_sup.entity_id, ev_con1.entity_id, ev_con2.entity_id],
        evidence_count=3,
        integrated_assessment="supports",
        integrated_confidence=0.8,
        adversarial_checked=False,
        convergence_checked=True,
        deductive_checked=True,
        computational_checked=True,
        argument_analyzed=True,
    )
    await repo.save(claim)

    # ONE state object carried across both passes so operations_log
    # accumulates — the property under test is the CUMULATIVE count.
    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    # ── Pass 1 ──
    await RunVerification().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    assert len(_ops(state, "adversarial_search")) == 1

    await PromoteSupported().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    after_pass_1 = await repo.get("claim", claim.entity_id)
    assert after_pass_1.stage == ClaimStage.HYPOTHESIS
    assert after_pass_1.contraction_count == 1
    assert after_pass_1.scrutiny_verdict == "pass"  # preserved, not reset

    # ── Pass 2: evidence is UNCHANGED. A re-promotion attempt must be
    # refused by promotion_blocked_by_contraction before the claim can
    # ever reach SUPPORTED (and therefore RunVerification's adversarial
    # dispatch) again. ──
    next_node = await PromoteToSupported().run(
        _FakeRunContext(state, deps)  # type: ignore[arg-type]
    )
    still_hypothesis = await repo.get("claim", claim.entity_id)
    assert still_hypothesis.stage == ClaimStage.HYPOTHESIS, (
        "PromoteToSupported re-promoted a contracted claim on unchanged "
        "evidence — promotion_blocked_by_contraction should have refused."
    )
    assert claim.entity_id in state.terminal_claims
    assert _ops(state, "promote_claim") == [], (
        "No promote_claim op should fire for a claim blocked by "
        "promotion_blocked_by_contraction."
    )

    # RunVerification again: the claim is still HYPOTHESIS, so it is
    # filtered out before the per-track loop ever runs.
    await RunVerification().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert len(_ops(state, "adversarial_search")) == 1, (
        "adversarial_search dispatched more than once across two passes "
        "on an unchanged evidence fingerprint (S3)."
    )
    from andamentum.epistemic.graph.nodes import CombineClaimVerdicts

    assert isinstance(next_node, CombineClaimVerdicts)


# ── S4 ────────────────────────────────────────────────────────────────


async def test_contracted_claim_is_terminal_not_routed_back_to_scrutinize(
    tmp_path: Path,
) -> None:
    """After a contraction, ``Scrutinize`` must not re-scrutinise the
    claim: ``Claim.contract`` neither nulls ``scrutiny_verdict`` nor adds
    the claim to ``claims_needing_rescrutiny``, so ``Scrutinize``'s own
    ``needs_scrutiny`` predicate is false for it and no
    ``scrutinise_claim`` op fires.
    """
    repo = await _make_repo(tmp_path, "certify_s4")
    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    ev_sup = Evidence(
        objective_id=obj.entity_id,
        source_type="web",
        source_ref="https://ex.com/sup",
        extracted_content="supporting item",
        extracted=True,
        support_judgment="supports",
        quality_score=0.5,
    )
    ev_con1 = Evidence(
        objective_id=obj.entity_id,
        source_type="web",
        source_ref="https://ex.com/con1",
        extracted_content="counter 1",
        extracted=True,
        support_judgment="contradicts",
        quality_score=0.5,
    )
    ev_con2 = Evidence(
        objective_id=obj.entity_id,
        source_type="web",
        source_ref="https://ex.com/con2",
        extracted_content="counter 2",
        extracted=True,
        support_judgment="contradicts",
        quality_score=0.5,
    )
    for ev in (ev_sup, ev_con1, ev_con2):
        await repo.save(ev)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="claim contracted, must not re-enter scrutiny",
        scope="x",
        stage=ClaimStage.SUPPORTED,
        scrutiny_verdict="pass",
        evidence_ids=[ev_sup.entity_id, ev_con1.entity_id, ev_con2.entity_id],
        evidence_count=3,
        integrated_assessment="supports",
        integrated_confidence=0.8,
    )
    await repo.save(claim)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    await PromoteSupported().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    contracted = await repo.get("claim", claim.entity_id)
    assert contracted.stage == ClaimStage.HYPOTHESIS
    assert contracted.contraction_count == 1
    # The two conditions Scrutinize's needs_scrutiny checks:
    assert contracted.scrutiny_verdict == "pass"  # not nulled by contract()
    assert claim.entity_id not in state.claims_needing_rescrutiny  # not flagged

    next_node = await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert _ops(state, "scrutinise_claim") == [], (
        "Scrutinize dispatched scrutinise_claim for a contracted claim — "
        "it must be terminal for this run, not re-scrutinised."
    )
    # No blocking uncertainties and nothing else queued for scrutiny:
    # Scrutinize's own (unmodified) no-work exit.
    from andamentum.epistemic.graph.nodes import ResolveUncertainties

    assert isinstance(next_node, ResolveUncertainties)


# ── S5 ────────────────────────────────────────────────────────────────


async def _make_claim_and_repo(
    tmp_path: Path, db_name: str, **claim_kwargs: Any
) -> tuple[Claim, EpistemicRepository]:
    repo = await _make_repo(tmp_path, db_name)
    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    claim = Claim(
        objective_id=obj.entity_id,
        statement="claim under S5 test",
        scope="x",
        stage=ClaimStage.HYPOTHESIS,
        **claim_kwargs,
    )
    await repo.save(claim)
    return claim, repo


async def test_resolve_uncertainties_skips_an_uncertainty_whose_claims_are_all_terminal(
    tmp_path: Path, fake_runner
) -> None:
    """A blocking uncertainty whose every affected claim is cycle-capped
    or abandoned gets no ``resolve_uncertainty`` op — dispatching one
    would spend an LLM call informing claims that can no longer act on
    the resolution. The uncertainty itself is left untouched (still
    unresolved), so nothing here manufactures a false resolution either.
    """
    capped, repo = await _make_claim_and_repo(tmp_path, "s5_all_terminal", cycle_capped=True)
    obj_id = capped.objective_id
    abandoned = Claim(
        objective_id=obj_id,
        statement="second claim, abandoned",
        scope="x",
        stage=ClaimStage.HYPOTHESIS,
        abandoned=True,
    )
    await repo.save(abandoned)

    unc = Uncertainty(
        description="Concern affecting only terminal claims.",
        uncertainty_type=UncertaintyType.UNKNOWN,
        objective_id=obj_id,
        affected_claim_ids=[capped.entity_id, abandoned.entity_id],
        is_blocking=True,
        resolution=None,
    )
    await repo.save(unc)

    state = EpistemicGraphState(objective_id=obj_id)
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")
    node = await ResolveUncertainties().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert _ops(state, "resolve_uncertainty") == [], (
        "resolve_uncertainty dispatched for an uncertainty whose every "
        "affected claim is cycle-capped or abandoned."
    )
    reloaded_unc = await repo.get("uncertainty", unc.entity_id)
    assert reloaded_unc.resolution is None
    # A terminal-only uncertainty must not count as still-blocking for
    # routing either, or the graph would loop forever re-entering
    # ResolveUncertainties on an uncertainty it will never resolve. With
    # no blocking uncertainty left and nothing queued for rescrutiny,
    # the node proceeds to its default next_on_clear.
    assert isinstance(node, EnumerateCandidates)


async def test_resolve_uncertainties_terminal_only_uncertainty_routes_to_promote(
    tmp_path: Path, fake_runner
) -> None:
    """A single cycle-capped claim with ``scrutiny_verdict == "pass"`` and
    one blocking uncertainty affecting only it: the uncertainty is
    terminal-only, so it must neither be dispatched nor count as
    still-blocking for routing. With ``next_on_clear="promote"`` the
    node proceeds straight to ``PromoteToSupported`` — the pre-fix
    routing instead recursed ``ResolveUncertainties`` indefinitely
    because the terminal-only uncertainty kept reading as blocking.
    """
    capped, repo = await _make_claim_and_repo(
        tmp_path,
        "s5_terminal_only_promote",
        cycle_capped=True,
        scrutiny_verdict="pass",
    )
    obj_id = capped.objective_id

    unc = Uncertainty(
        description="Concern affecting only the capped claim.",
        uncertainty_type=UncertaintyType.UNKNOWN,
        objective_id=obj_id,
        affected_claim_ids=[capped.entity_id],
        is_blocking=True,
        resolution=None,
    )
    await repo.save(unc)

    state = EpistemicGraphState(objective_id=obj_id)
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")
    node = await ResolveUncertainties(next_on_clear="promote").run(
        _FakeRunContext(state, deps)  # type: ignore[arg-type]
    )

    assert _ops(state, "resolve_uncertainty") == [], (
        "resolve_uncertainty dispatched for a terminal-only uncertainty."
    )
    assert isinstance(node, PromoteToSupported)


async def test_resolve_uncertainties_still_resolves_when_one_affected_claim_is_live(
    tmp_path: Path, fake_runner
) -> None:
    """Sanity: the skip requires ALL affected claims to be terminal. One
    live (non-capped, non-abandoned) affected claim among several is
    enough to resolve normally."""
    capped, repo = await _make_claim_and_repo(tmp_path, "s5_one_live", cycle_capped=True)
    obj_id = capped.objective_id
    live = Claim(
        objective_id=obj_id,
        statement="second claim, still live",
        scope="x",
        stage=ClaimStage.HYPOTHESIS,
    )
    await repo.save(live)

    unc = Uncertainty(
        description="Concern affecting one terminal and one live claim.",
        uncertainty_type=UncertaintyType.UNKNOWN,
        objective_id=obj_id,
        affected_claim_ids=[capped.entity_id, live.entity_id],
        is_blocking=True,
        resolution=None,
    )
    await repo.save(unc)

    state = EpistemicGraphState(objective_id=obj_id)
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")
    await ResolveUncertainties().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert len(_ops(state, "resolve_uncertainty")) == 1


async def test_resolve_uncertainties_dispatches_when_no_affected_claims_recorded(
    tmp_path: Path, fake_runner
) -> None:
    """Sanity: an uncertainty with an empty ``affected_claim_ids`` is
    treated exactly as before the D7 change — there is nothing to check
    the terminal condition against, so it is dispatched normally."""
    repo = await _make_repo(tmp_path, "s5_no_affected")
    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    unc = Uncertainty(
        description="Objective-scope concern, no specific claim.",
        uncertainty_type=UncertaintyType.UNKNOWN,
        objective_id=obj.entity_id,
        affected_claim_ids=[],
        is_blocking=True,
        resolution=None,
    )
    await repo.save(unc)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")
    await ResolveUncertainties().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert len(_ops(state, "resolve_uncertainty")) == 1


async def test_conclude_at_cycle_cap_once_per_firing_not_per_reentry_on_unchanged_evidence(
    tmp_path: Path,
) -> None:
    """``conclude_at_cycle_cap`` fires on the FIRST cap firing, and on a
    later firing only when the claim's evidence fingerprint has moved
    since the last conclusion. Drives the cap from its two real call
    sites: ``ResolveUncertainties`` (primary) for the first firing, then
    ``Scrutinize`` (defense in depth) for two later firings — one on the
    SAME evidence (must add no row) and one after the evidence has
    actually changed (must add exactly one more row).
    """
    repo = await _make_repo(tmp_path, "s5_cap_firing_once")
    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    ev = Evidence(
        objective_id=obj.entity_id,
        source_type="web",
        source_ref="https://ex.com/s5",
        extracted_content="some evidence",
        extracted=True,
        support_judgment="supports",
        quality_score=0.5,
    )
    await repo.save(ev)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="claim driven to the cycle cap three times",
        scope="x",
        stage=ClaimStage.HYPOTHESIS,
        scrutiny_verdict="pass",
        evidence_ids=[ev.entity_id],
        evidence_count=1,
    )
    await repo.save(claim)

    unc = Uncertainty(
        description="Blocking concern, resolved but the claim is already spent.",
        uncertainty_type=UncertaintyType.UNKNOWN,
        objective_id=obj.entity_id,
        affected_claim_ids=[claim.entity_id],
        is_blocking=True,
        resolution=None,
    )
    await repo.save(unc)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    state.scrutiny_resolve_cycles[claim.entity_id] = SCRUTINY_RESOLVE_CYCLE_CAP
    from .conftest import FakeAgentRunner

    deps = EpistemicDeps(
        repo=repo, agent_runner=FakeAgentRunner(), embedding_model="t"
    )

    # ── Firing 1 (ResolveUncertainties, primary loop-breaker) ──
    await ResolveUncertainties().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    assert len(_ops(state, "conclude_at_cycle_cap")) == 1
    after_1 = await repo.get("claim", claim.entity_id)
    assert after_1.cycle_capped is True
    fp_at_firing_1 = await claim_evidence_fingerprint(after_1, repo)
    assert after_1.cap_fingerprint == fp_at_firing_1

    # ── Firing 2 (Scrutinize, defense in depth), SAME evidence ──
    state.claims_needing_rescrutiny.add(claim.entity_id)
    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    assert len(_ops(state, "conclude_at_cycle_cap")) == 1, (
        "conclude_at_cycle_cap re-ran on a re-entry with unchanged "
        "evidence — the cap is terminal and must not redo mechanical "
        "work it already did."
    )

    # ── Evidence actually changes ──
    ev_reloaded = await repo.get("evidence", ev.entity_id)
    ev_reloaded.support_judgment = "contradicts"
    await repo.save(ev_reloaded)

    # ── Firing 3 (Scrutinize again), evidence has moved ──
    state.claims_needing_rescrutiny.add(claim.entity_id)
    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    assert len(_ops(state, "conclude_at_cycle_cap")) == 2, (
        "conclude_at_cycle_cap did not re-run after the evidence set "
        "actually changed."
    )
    after_3 = await repo.get("claim", claim.entity_id)
    fp_at_firing_3 = await claim_evidence_fingerprint(after_3, repo)
    assert after_3.cap_fingerprint == fp_at_firing_3
    assert fp_at_firing_3 != fp_at_firing_1


# ── S6 ────────────────────────────────────────────────────────────────


async def _s6_claim(
    *,
    repo: EpistemicRepository,
    objective_id: str,
    entity_id: str,
    integrated_assessment: str | None,
    n_evidence: int = 1,
) -> Claim:
    """A SUPPORTED claim with clean evidence — the gate
    RevalidateClaimOperation checks (``validate_current_stage``) passes
    unconditionally (no adversarial data, no judged support/contradict
    pair), so this claim is never contracted and the S6 ladder-follows-
    the-verdict check is isolated from D1's contraction check.
    ``n_evidence=2`` also clears the PROVISIONAL gate (min_evidence=2,
    min_quality_sum=0.5), for the one claim this test expects to
    actually climb."""
    evidence_ids = []
    for i in range(n_evidence):
        ev = Evidence(
            objective_id=objective_id,
            source_type="web",
            source_ref=f"https://ex.com/{entity_id}-{i}",
            extracted_content="clean supporting item",
            extracted=True,
            quality_score=0.5,
        )
        await repo.save(ev)
        evidence_ids.append(ev.entity_id)
    claim = Claim(
        entity_id=entity_id,
        objective_id=objective_id,
        statement=f"S6 claim {entity_id}",
        scope="x",
        stage=ClaimStage.SUPPORTED,
        scrutiny_verdict="pass",
        evidence_ids=evidence_ids,
        evidence_count=n_evidence,
        integrated_assessment=integrated_assessment,
        integrated_confidence=0.8 if integrated_assessment else None,
    )
    await repo.save(claim)
    return claim


async def test_promote_supported_ladder_follows_the_verdict(
    tmp_path: Path,
) -> None:
    """``PromoteSupported`` (D8): only a claim IBE-certified ``supports``
    climbs the ladder. A claim with no verdict, or one certified
    ``contradicts``/``insufficient``, gets zero ``promote_claim``
    attempts and is simply marked ``verification_done`` (S6)."""
    repo = await _make_repo(tmp_path, "certify_s6")
    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    supports_claim = await _s6_claim(
        repo=repo,
        objective_id=obj.entity_id,
        entity_id="s6-supports",
        integrated_assessment="supports",
        n_evidence=2,
    )
    contradicts_claim = await _s6_claim(
        repo=repo,
        objective_id=obj.entity_id,
        entity_id="s6-contradicts",
        integrated_assessment="contradicts",
    )
    insufficient_claim = await _s6_claim(
        repo=repo,
        objective_id=obj.entity_id,
        entity_id="s6-insufficient",
        integrated_assessment="insufficient",
    )
    no_verdict_claim = await _s6_claim(
        repo=repo,
        objective_id=obj.entity_id,
        entity_id="s6-no-verdict",
        integrated_assessment=None,
    )

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=None, embedding_model="t")

    next_node = await PromoteSupported().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    from andamentum.epistemic.graph.nodes import CombineClaimVerdicts

    assert isinstance(next_node, CombineClaimVerdicts)

    promote_ops = _ops(state, "promote_claim")
    promoted_ids = {op.entity_id for op in promote_ops}
    assert promoted_ids == {supports_claim.entity_id}, (
        "Only the 'supports'-certified claim should have a promote_claim "
        f"attempt; got attempts for {promoted_ids!r}."
    )

    # Every claim is marked verification_done regardless of verdict —
    # PromoteToSupported's idempotency check must short-circuit on all
    # four next re-entry.
    for claim in (
        supports_claim,
        contradicts_claim,
        insufficient_claim,
        no_verdict_claim,
    ):
        assert claim.entity_id in state.verification_done

    # The non-'supports' claims did not move off SUPPORTED — the ladder
    # never touched them (no promote_claim attempt was made for them at
    # all, per the assertion above; whether an attempted promotion for
    # the 'supports' claim actually succeeds is a gate-validation
    # question PromoteClaimOperation's own test suite covers, not S6's
    # concern here).
    for claim in (contradicts_claim, insufficient_claim, no_verdict_claim):
        reloaded = await repo.get("claim", claim.entity_id)
        assert reloaded.stage == ClaimStage.SUPPORTED


# ── G1 (final alignment review, docs/epistemic/
# CERTIFY_BEFORE_CONTRACT_PLAN.md §6) ────────────────────────────────────
#
# The fingerprint-changed re-promotion branch of PromoteToSupported Step 1
# used to call claim.reset_for_reinquiry() BEFORE attempting
# PromoteClaimOperation. A refused promotion (for any reason other than
# the fingerprint check itself — e.g. a blocking uncertainty) therefore
# left the claim at HYPOTHESIS with its verdict already erased. G1
# reorders: attempt promotion first, reset only on success.


async def _g1_contracted_claim_and_repo(
    tmp_path: Path, db_name: str
) -> tuple[Claim, Evidence, EpistemicRepository]:
    """A HYPOTHESIS claim already contracted once, with its fingerprint
    recorded against a ONE-evidence-item state. The caller adds a second
    evidence item to change the fingerprint before driving the node."""
    repo = await _make_repo(tmp_path, db_name)
    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    ev1 = Evidence(
        objective_id=obj.entity_id,
        source_type="web",
        source_ref="https://ex.com/first",
        extracted_content="first supporting item",
        extracted=True,
        support_judgment="supports",
        quality_score=0.5,
    )
    await repo.save(ev1)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="claim contracted once, evidence about to change",
        scope="x",
        stage=ClaimStage.HYPOTHESIS,
        scrutiny_verdict="pass",
        evidence_ids=[ev1.entity_id],
        evidence_count=1,
        contraction_count=1,
        # A verdict IBE certified before the contraction — this is what
        # must survive a REFUSED re-promotion (G1) and must be nulled by
        # reset_for_reinquiry() only on a SUCCESSFUL one.
        integrated_assessment="insufficient",
        integrated_confidence=0.6,
    )
    old_fingerprint = await claim_evidence_fingerprint(claim, repo)
    claim.contraction_fingerprint = old_fingerprint
    await repo.save(claim)
    return claim, ev1, repo


async def test_g1_changed_fingerprint_promotable_resets_only_on_success(
    tmp_path: Path,
) -> None:
    """(a) Changed fingerprint + promotable: the claim is promoted,
    verification flags/verdict are reset, and routing defaults are set —
    all AFTER the successful promote_claim."""
    claim, ev1, repo = await _g1_contracted_claim_and_repo(
        tmp_path, "g1_promotable"
    )
    obj_id = claim.objective_id

    # New evidence changes the fingerprint and gives the claim a second
    # supporting source (min_supporting_sources=1 is already met by ev1
    # alone, but a second item makes the fingerprint change unambiguous).
    ev2 = Evidence(
        objective_id=obj_id,
        source_type="web",
        source_ref="https://ex.com/second",
        extracted_content="second supporting item",
        extracted=True,
        support_judgment="supports",
        quality_score=0.5,
    )
    await repo.save(ev2)
    claim.evidence_ids = [ev1.entity_id, ev2.entity_id]
    claim.evidence_count = 2
    await repo.save(claim)

    new_fingerprint = await claim_evidence_fingerprint(claim, repo)
    assert new_fingerprint != claim.contraction_fingerprint, (
        "test setup error: the fingerprint must actually change"
    )

    state = EpistemicGraphState(objective_id=obj_id)
    deps = EpistemicDeps(repo=repo, agent_runner=None, embedding_model="t")

    await PromoteToSupported().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert _ops(state, "promote_claim") != [], (
        "the re-promotion attempt must actually be made when the "
        "fingerprint has changed."
    )
    assert _ops(state, "set_routing_defaults") != []

    promoted = await repo.get("claim", claim.entity_id)
    assert promoted.stage == ClaimStage.SUPPORTED
    # reset_for_reinquiry() ran — the pre-contraction verdict is gone,
    # a fresh inquiry cycle owns this claim now.
    assert promoted.integrated_assessment is None
    assert promoted.integrated_confidence is None
    assert promoted.contraction_fingerprint is None
    assert claim.entity_id not in state.verification_done


async def test_g1_changed_fingerprint_refused_promotion_keeps_verdict(
    tmp_path: Path,
) -> None:
    """(b) Changed fingerprint + promotion refused (a blocking
    uncertainty): the claim stays at HYPOTHESIS, its pre-contraction
    verdict is untouched, and contraction_fingerprint is unchanged — the
    reorder must not erase state on a refusal that has nothing to do
    with the fingerprint check itself."""
    claim, ev1, repo = await _g1_contracted_claim_and_repo(
        tmp_path, "g1_refused"
    )
    obj_id = claim.objective_id

    ev2 = Evidence(
        objective_id=obj_id,
        source_type="web",
        source_ref="https://ex.com/second",
        extracted_content="second supporting item",
        extracted=True,
        support_judgment="supports",
        quality_score=0.5,
    )
    await repo.save(ev2)
    claim.evidence_ids = [ev1.entity_id, ev2.entity_id]
    claim.evidence_count = 2
    await repo.save(claim)

    new_fingerprint = await claim_evidence_fingerprint(claim, repo)
    assert new_fingerprint != claim.contraction_fingerprint

    unc = Uncertainty(
        description="Blocking concern raised after the contraction.",
        uncertainty_type=UncertaintyType.UNKNOWN,
        objective_id=obj_id,
        affected_claim_ids=[claim.entity_id],
        is_blocking=True,
        resolution=None,
    )
    await repo.save(unc)

    state = EpistemicGraphState(objective_id=obj_id)
    deps = EpistemicDeps(repo=repo, agent_runner=None, embedding_model="t")

    await PromoteToSupported().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    promote_ops = _ops(state, "promote_claim")
    assert promote_ops != [], "the attempt must still be made"
    assert not promote_ops[-1].success, (
        "test setup error: the blocking uncertainty must actually refuse "
        "the promotion for this test to exercise the refusal branch."
    )

    refused = await repo.get("claim", claim.entity_id)
    assert refused.stage == ClaimStage.HYPOTHESIS, (
        "a refused promotion must not leave the claim mid-transition."
    )
    assert refused.integrated_assessment == "insufficient", (
        "reset_for_reinquiry() ran on a REFUSED promotion — the verdict "
        "that survived the contraction was erased for nothing (the G1 bug)."
    )
    assert refused.integrated_confidence == 0.6
    assert refused.contraction_fingerprint == claim.contraction_fingerprint, (
        "contraction_fingerprint must be untouched by a refused attempt."
    )
    assert claim.entity_id in state.terminal_claims
