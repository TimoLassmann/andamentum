"""The typed ``EpistemicVerdict`` — one decision, derived not duplicated.

Before this object the system decided its verdict in five independent
places (``graph/combination._verdict_label``,
``confidence.compute_posterior``, the writer agent's prose,
``report_generator``'s re-parse of that prose, and
``audit_report._normalised_verdict``) and nothing forced them to agree.
These tests pin the properties that make a sixth derivation impossible
and the fifth one unnecessary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.document_store import DocumentStore
from andamentum.epistemic.confidence import (
    _claim_directional_masses,
    build_verdict,
)
from andamentum.epistemic.entities import Claim, Evidence, Objective, Snapshot
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.entities.decomposition import (
    Decomposition,
    SubInvestigation,
)
from andamentum.epistemic.entities.verdict import (
    display_label,
    verdict_label,
)
from andamentum.epistemic.repository import EpistemicRepository


async def _repo(tmp_path: Path, name: str) -> EpistemicRepository:
    store = DocumentStore.for_database(name, db_dir=tmp_path)
    await store.initialize()
    return EpistemicRepository(store)


# ── direction ↔ label ────────────────────────────────────────────────


_LABEL_FOR_DIRECTION = {
    "supports": "Confirmed",
    "contradicts": "Refuted",
    "insufficient": "Inconclusive",
}


@pytest.mark.parametrize(
    "p", [0.0, 0.30, 0.34, 0.50, 0.66, 0.70, 0.90, 1.0]
)
def test_direction_and_label_never_disagree(p: float) -> None:
    """``display_label`` MAPS a direction; it never re-thresholds.

    This is the exact defect that existed between
    ``graph/combination._verdict_label`` (``>`` / ``<``) and
    ``audit_report._normalised_verdict`` (``>=`` / ``<=``): at p == 0.66
    the combiner said "insufficient" while the renderer printed
    "Confirmed". A future re-thresholding inside a renderer fails here.
    """
    direction = verdict_label(p)
    assert display_label(direction) == _LABEL_FOR_DIRECTION[direction]


def test_terminal_state_override_beats_a_directional_posterior() -> None:
    """An inquiry that did not complete reports "Insufficient evidence"
    whatever the number says — the posterior is not a directional answer
    in that state. Rule 1 of the renderer this function replaces,
    preserved verbatim so the semantics live in one place."""
    assert display_label("supports", "oscillation_detected") == "Insufficient evidence"
    assert display_label("contradicts", "retrieval_failed") == "Insufficient evidence"


def test_supports_refined_gets_its_own_label() -> None:
    assert (
        display_label("supports", has_supports_refined=True)
        == "Confirmed with refinement"
    )


# ── build_verdict over a real DB ─────────────────────────────────────


async def test_build_verdict_covers_verify_mode_without_decomposition(
    tmp_path: Path,
) -> None:
    """THE SCIFACT ARM. A verify-mode objective has no decomposition, so
    the previous ``CombinedVerdictData`` was structurally unreachable —
    the entire benchmark path produced no structured verdict at all and
    the writer's prose was the only verdict that existed. A regression
    that re-adds a decomposition-only early return fails here."""
    repo = await _repo(tmp_path, "verdict_verify")
    obj = Objective(
        description="Aspirin reduces colorectal cancer risk.",
        claim_to_verify="Aspirin reduces colorectal cancer risk.",
        question_type="verificatory",
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    evidence_ids: list[str] = []
    for i in range(2):
        ev = Evidence(
            objective_id=obj.entity_id,
            source_ref=f"doi:sup/{i}",
            extracted=True,
            extracted_content="A supporting study.",
            support_judgment="supports",
            judgment_distribution=[1.0, 0.0, 0.0],
        )
        await repo.save(ev)
        evidence_ids.append(ev.entity_id)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="Aspirin reduces colorectal cancer risk.",
        scope="adults",
        stage=ClaimStage.SUPPORTED,
        integrated_assessment="supports",
        integrated_confidence=0.8,
        evidence_ids=evidence_ids,
    )
    await repo.save(claim)

    verdict = await build_verdict(repo, objective_id=obj.entity_id)

    assert verdict.direction == "supports"
    assert verdict.posterior is not None and verdict.posterior > 0.66
    assert verdict.provenance.combination_rule is None
    assert verdict.provenance.n_adjudicated == 1
    assert len(verdict.claims) == 1
    assert verdict.claims[0].supporting_evidence


async def test_build_verdict_honours_the_combination_rule(tmp_path: Path) -> None:
    """The rule-aware maths still comes from ``combine_claim_verdicts``
    and is reached through exactly one path. A fix that bypasses the
    combiner (e.g. always confidence-weighted) fails here: AND over
    [0.9, 0.7] is min = 0.7, while the weighted average is 0.8."""
    repo = await _repo(tmp_path, "verdict_rule")
    obj = Objective(
        description="parent",
        decomposition=Decomposition(
            sub_investigations=[
                SubInvestigation(id="A", seed_claim="a", rationale="r"),
                SubInvestigation(id="B", seed_claim="b", rationale="r"),
            ],
            combination_rule="AND",
        ),
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    for sub, assessment, conf in (("A", "supports", 0.8), ("B", "contradicts", 0.6)):
        c = Claim(
            objective_id=obj.entity_id,
            statement=f"claim {sub}",
            scope="x",
            stage=ClaimStage.SUPPORTED,
            sub_investigation_id=sub,
            integrated_assessment=assessment,
            integrated_confidence=conf,
        )
        await repo.save(c)

    verdict = await build_verdict(repo, objective_id=obj.entity_id)

    # supports@0.8 → 0.9; contradicts@0.6 → 0.2. AND → min = 0.2.
    assert verdict.posterior == pytest.approx(0.2)
    assert verdict.provenance.combination_rule == "AND"
    assert verdict.direction == "contradicts"


async def test_posterior_is_the_embedded_report_not_a_copy(tmp_path: Path) -> None:
    """No second stored number that can drift from the report.
    Converting either computed field to a stored one fails here."""
    repo = await _repo(tmp_path, "verdict_embedded")
    obj = Objective(
        description="x",
        claim_to_verify="x",
        question_type="verificatory",
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    c = Claim(
        objective_id=obj.entity_id,
        statement="x",
        stage=ClaimStage.SUPPORTED,
        integrated_assessment="supports",
        integrated_confidence=0.9,
    )
    await repo.save(c)

    verdict = await build_verdict(repo, objective_id=obj.entity_id)
    assert verdict.posterior_report is not None
    assert verdict.posterior == verdict.posterior_report.posterior
    assert verdict.confidence == verdict.posterior_report.integration_confidence
    assert verdict.explanation == verdict.posterior_report.explanation

    # No-data shape: nothing to embed, nothing to derive.
    empty_repo = await _repo(tmp_path, "verdict_embedded_empty")
    empty = Objective(description="x")
    empty.objective_id = empty.entity_id
    await empty_repo.save(empty)
    no_data = await build_verdict(empty_repo, objective_id=empty.entity_id)
    assert no_data.posterior_report is None
    assert no_data.posterior is None
    assert no_data.direction == "no_data"
    assert no_data.label == "Insufficient evidence"


async def test_retrieval_failed_is_captured_at_build_time(tmp_path: Path) -> None:
    """``retrieval_failed`` lives only on graph state, so capturing it
    here and persisting the result is what fixes the documented
    live-vs-offline divergence: the offline report used to re-run
    ``compute_posterior`` without it and always claimed "completed"."""
    repo = await _repo(tmp_path, "verdict_retrieval")
    obj = Objective(description="x", claim_to_verify="x", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    c = Claim(
        objective_id=obj.entity_id,
        statement="x",
        stage=ClaimStage.SUPPORTED,
        integrated_assessment="supports",
        integrated_confidence=0.8,
    )
    await repo.save(c)

    verdict = await build_verdict(
        repo, objective_id=obj.entity_id, retrieval_failed=True
    )
    assert verdict.provenance.retrieval_failed is True
    assert verdict.posterior_report is not None
    assert verdict.posterior_report.terminal_state == "retrieval_failed"

    snapshot = Snapshot(objective_id=obj.entity_id, verdict=verdict)
    await repo.save(snapshot)
    reloaded = await repo.get("snapshot", snapshot.entity_id)
    assert reloaded.verdict is not None
    assert reloaded.verdict.terminal_state == "retrieval_failed"
    assert reloaded.verdict.label == "Insufficient evidence"


async def test_evidence_refs_partition_and_sum_to_the_counting_masses(
    tmp_path: Path,
) -> None:
    """The refs ARE the evidence the posterior counted — same filter,
    same weighting — so a reader following the verdict back to its
    sources cannot be shown a different set (CONSTITUTION rule 6)."""
    repo = await _repo(tmp_path, "verdict_refs")
    obj = Objective(description="x", claim_to_verify="x", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    spec = [("supports", [1.0, 0.0, 0.0])] * 2
    spec += [("contradicts", [0.0, 1.0, 0.0])]
    spec += [("no_bearing", [0.0, 0.0, 1.0])]
    eids: list[str] = []
    for i, (judgment, dist) in enumerate(spec):
        ev = Evidence(
            objective_id=obj.entity_id,
            source_ref=f"doi:x/{i}",
            extracted=True,
            extracted_content="content",
            support_judgment=judgment,
            judgment_distribution=dist,
        )
        await repo.save(ev)
        eids.append(ev.entity_id)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="x",
        stage=ClaimStage.SUPPORTED,
        integrated_assessment="supports",
        integrated_confidence=0.5,
        evidence_ids=eids,
    )
    await repo.save(claim)

    verdict = await build_verdict(repo, objective_id=obj.entity_id)
    cv = verdict.claims[0]
    assert len(cv.supporting_evidence) == 2
    assert len(cv.refuting_evidence) == 1
    assert cv.n_no_bearing == 1

    all_evidence = await repo.get_evidence_for_objective(obj.entity_id)
    s_mass, c_mass = _claim_directional_masses(claim, all_evidence)
    assert sum(r.mass for r in cv.supporting_evidence) == pytest.approx(s_mass)
    assert sum(r.mass for r in cv.refuting_evidence) == pytest.approx(c_mass)


async def test_suspended_claim_is_flagged_and_contributes_neutral(
    tmp_path: Path,
) -> None:
    """The concordance tripwire survives the collapse and is VISIBLE on
    the object rather than being an invisible internal of
    ``compute_posterior``. Suspend-only: the direction is withheld, never
    flipped."""
    from andamentum.epistemic.thresholds import ADVERSARIAL_REFUTED_THRESHOLD

    repo = await _repo(tmp_path, "verdict_suspended")
    obj = Objective(description="x", claim_to_verify="x", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    eids: list[str] = []
    for i in range(3):
        ev = Evidence(
            objective_id=obj.entity_id,
            source_ref=f"doi:con/{i}",
            extracted=True,
            extracted_content="content",
            support_judgment="contradicts",
            judgment_distribution=[0.0, 1.0, 0.0],
        )
        await repo.save(ev)
        eids.append(ev.entity_id)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="x",
        stage=ClaimStage.SUPPORTED,
        # IBE says supports, counting mass says contradicts, and the
        # independent adversarial leg says refuted → tripwire fires.
        integrated_assessment="supports",
        integrated_confidence=0.9,
        adversarial_balance=ADVERSARIAL_REFUTED_THRESHOLD - 0.05,
        evidence_ids=eids,
    )
    await repo.save(claim)

    verdict = await build_verdict(repo, objective_id=obj.entity_id)
    cv = verdict.claims[0]
    assert cv.suspended is True
    assert cv.posterior == 0.5
    assert verdict.posterior_report is not None
    assert claim.entity_id in verdict.posterior_report.suspended_claim_ids


async def test_claim_verdict_carries_the_cap_evaluation(tmp_path: Path) -> None:
    """RULING C5. ``CapEvaluation`` is a PERSISTED SNAPSHOT on the Claim
    with an entity lifetime; ``EpistemicVerdict`` is a RUN-LEVEL
    aggregate rebuilt from the live DB every pass. Both survive, and the
    aggregate carries the snapshot so no renderer has to reach back to
    the entity to explain a cap-derived verdict."""
    from andamentum.epistemic.entities import CapEvaluation

    repo = await _repo(tmp_path, "verdict_cap")
    obj = Objective(description="x", claim_to_verify="x", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    claim = Claim(
        objective_id=obj.entity_id,
        statement="x",
        stage=ClaimStage.HYPOTHESIS,
        cycle_capped=True,
        integrated_assessment="contradicts",
        integrated_confidence=0.5,
        cap_evaluation=CapEvaluation(
            weight_label="strong",
            supporting_mass=1.0,
            contradicting_mass=3.0,
            n_evidence=4,
            verdict="contradicts",
            confidence=0.5,
            wrote_live_verdict=True,
        ),
    )
    await repo.save(claim)

    verdict = await build_verdict(repo, objective_id=obj.entity_id)
    cv = verdict.claims[0]
    assert cv.cycle_capped is True
    assert cv.cap_evaluation is not None
    assert cv.cap_evaluation.weight_label == "strong"
    assert cv.cap_evaluation.verdict == "contradicts"
    assert verdict.provenance.n_cycle_capped == 1
