"""SEAM 2 — the doubt reaches the verdict object, and something reads it.

A doubt signal the verdict object cannot express is a signal no consumer can
read. ``CapEvaluation.doubt`` already reaches ``ClaimVerdict`` and is read by
ZERO renderers; these tests exist so the judgment-doubt plumbing does not
repeat that.

``build_verdict`` is deterministic — every field here is a pure copy off an
entity already loaded — so the whole seam is verifiable offline, and one test
pins that it makes no model call at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.document_store import DocumentStore
from andamentum.epistemic.confidence import build_verdict
from andamentum.epistemic.entities import Claim, Evidence, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.entities.judge_opinion import JudgeOpinion
from andamentum.epistemic.entities.verdict import render_verdict_line
from andamentum.epistemic.report_data import ClaimSummary, EvidenceSummary
from andamentum.epistemic.repository import EpistemicRepository
from andamentum.epistemic.thresholds import JUDGMENT_DOUBT_ESCALATION_THRESHOLD

_HIGH = JUDGMENT_DOUBT_ESCALATION_THRESHOLD + 0.2
_LOW = JUDGMENT_DOUBT_ESCALATION_THRESHOLD - 0.2


def _opinion(*, overall: str, reference: str, dissents: bool) -> JudgeOpinion:
    dist = {
        "meets": [0.9, 0.06, 0.04],
        "partial": [0.05, 0.9, 0.05],
        "fails": [0.04, 0.06, 0.9],
    }[overall]
    return JudgeOpinion(
        subject="does the evidence establish the claim?",
        reference=reference,
        distribution=dist,
        overall=overall,  # type: ignore[arg-type]
        expected_score=0.5,
        summary="stub",
        dissents=dissents,
    )


async def _seed(tmp_path: Path, name: str = "seam2"):
    """One objective, two claims: one fully instrumented, one untouched."""
    store = DocumentStore.for_database(name, db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="q")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    # Instrumented claim: a graded supporting item carrying a direction
    # confirmation and a quality second opinion, plus an adversarial-style
    # item with NO distribution at all.
    measured = Claim(
        objective_id=obj.entity_id,
        statement="measured claim",
        scope="x",
        stage=ClaimStage.SUPPORTED,
        integrated_assessment="supports",
        integrated_confidence=0.7,
        scrutiny_verdict="pass",
        scrutiny_doubt=_HIGH,
        scrutiny_opinion=_opinion(overall="fails", reference="moderate", dissents=True),
        judgment_dissent_count=2,
    )
    confirmed = Evidence(
        objective_id=obj.entity_id,
        source_type="openalex",
        source_ref="doi:confirmed",
        extracted_content="A supporting finding.",
        extracted=True,
        quality_score=0.8,
        cluster_status="representative",
        judgment_distribution=[0.6, 0.2, 0.2],
        support_judgment="supports",
        quality_distribution=[0.7, 0.2, 0.1],
        direction_opinion=_opinion(
            overall="fails", reference="supports", dissents=True
        ),
    )
    await repo.save(confirmed)
    distributionless = Evidence(
        objective_id=obj.entity_id,
        source_type="web",
        source_ref="https://example.invalid/adversarial",
        extracted_content="A counter finding.",
        extracted=True,
        quality_score=0.4,
        cluster_status="representative",
        support_judgment="contradicts",
    )
    await repo.save(distributionless)
    measured.evidence_ids = [confirmed.entity_id, distributionless.entity_id]
    measured.evidence_count = 2
    await repo.save(measured)

    unmeasured = Claim(
        objective_id=obj.entity_id,
        statement="unmeasured claim",
        scope="x",
        stage=ClaimStage.SUPPORTED,
        integrated_assessment="supports",
        integrated_confidence=0.7,
    )
    await repo.save(unmeasured)

    return repo, obj, measured, unmeasured, confirmed, distributionless


# ── EvidenceRef ──────────────────────────────────────────────────────


async def test_evidence_ref_carries_the_doubt_of_the_item_it_names(
    tmp_path: Path,
) -> None:
    """The refs on a ClaimVerdict are provably the items that produced its
    number — so the reliability they report must be those items' own."""
    repo, obj, measured, _u, confirmed, _d = await _seed(tmp_path)

    verdict = await build_verdict(repo, objective_id=obj.objective_id)
    cv = next(c for c in verdict.claims if c.claim_id == measured.entity_id)
    ref = next(
        r for r in cv.supporting_evidence if r.evidence_id == confirmed.entity_id
    )

    assert ref.judgment_distribution == pytest.approx(confirmed.judgment_distribution)
    assert ref.judgment_doubt == pytest.approx(confirmed.judgment_entropy)
    assert ref.quality_doubt == pytest.approx(confirmed.quality_doubt)


async def test_an_unmeasured_item_reports_none_and_never_zero(
    tmp_path: Path,
) -> None:
    """NONE MEANS NOT MEASURED. Adversarial counter-evidence is permanently
    distribution-less by construction, so coercing absence to 0.0 would read
    downstream as certainty the system never established.

    PINS: the three-state contract at the verdict boundary.
    """
    repo, obj, measured, _u, _c, distributionless = await _seed(tmp_path)

    verdict = await build_verdict(repo, objective_id=obj.objective_id)
    cv = next(c for c in verdict.claims if c.claim_id == measured.entity_id)
    ref = next(
        r
        for r in cv.refuting_evidence
        if r.evidence_id == distributionless.entity_id
    )

    assert ref.judgment_distribution is None
    assert ref.judgment_doubt is None
    assert ref.quality_doubt is None
    assert ref.direction_opinion is None


async def test_a_confirmed_item_carries_its_whole_opinion_object(
    tmp_path: Path,
) -> None:
    """The whole ``JudgeOpinion``, not a scalar: a consumer needs the dissent
    flag AND the doubt, and they measure different things."""
    repo, obj, measured, _u, confirmed, _d = await _seed(tmp_path)

    verdict = await build_verdict(repo, objective_id=obj.objective_id)
    cv = next(c for c in verdict.claims if c.claim_id == measured.entity_id)
    ref = next(
        r for r in cv.supporting_evidence if r.evidence_id == confirmed.entity_id
    )

    assert ref.direction_opinion is not None
    assert ref.direction_opinion.dissents is True
    assert ref.direction_opinion.reference == "supports"
    assert ref.direction_opinion.doubt < JUDGMENT_DOUBT_ESCALATION_THRESHOLD


# ── ClaimVerdict ─────────────────────────────────────────────────────


async def test_claim_verdict_carries_the_reading_and_its_reliability(
    tmp_path: Path,
) -> None:
    """THE LARGEST HOLE SEAM 2 CLOSES.

    ``scrutiny_verdict`` travels with the doubt deliberately: doubt without
    the reading it is doubt ABOUT is un-interpretable.
    """
    repo, obj, measured, _u, _c, _d = await _seed(tmp_path)

    verdict = await build_verdict(repo, objective_id=obj.objective_id)
    cv = next(c for c in verdict.claims if c.claim_id == measured.entity_id)

    assert cv.scrutiny_verdict == "pass"
    assert cv.scrutiny_doubt == pytest.approx(_HIGH)
    assert cv.scrutiny_opinion is not None
    assert cv.scrutiny_opinion.dissents is True
    assert cv.judgment_dissent_count == 2


async def test_an_unmeasured_claim_reports_none_and_zero(tmp_path: Path) -> None:
    """Every pre-existing fixture in the suite is this claim. None of them
    may start reporting a measurement they never took."""
    repo, obj, _m, unmeasured, _c, _d = await _seed(tmp_path)

    verdict = await build_verdict(repo, objective_id=obj.objective_id)
    cv = next(c for c in verdict.claims if c.claim_id == unmeasured.entity_id)

    assert cv.scrutiny_verdict is None
    assert cv.scrutiny_doubt is None
    assert cv.scrutiny_opinion is None
    assert cv.judgment_dissent_count == 0


# ── VerdictProvenance ────────────────────────────────────────────────


async def test_provenance_counts_high_doubt_and_unmeasured_separately(
    tmp_path: Path,
) -> None:
    """PINS the companion count.

    A bare ``n_high_doubt_claims == 0`` reads as "all measured, all sure".
    On a run where nothing was measured that is a false statement, and it is
    exactly the run shape the pre-Tier-0 and adversarial-only arms produce.
    """
    repo, obj, _m, _u, _c, _d = await _seed(tmp_path)

    prov = (await build_verdict(repo, objective_id=obj.objective_id)).provenance

    assert prov.n_active_claims == 2
    assert prov.n_high_doubt_claims == 1
    assert prov.n_doubt_unmeasured_claims == 1


async def test_provenance_records_the_escalation_flag_it_was_given(
    tmp_path: Path,
) -> None:
    """Gate 0's one-shot flag lives on graph state, not the DB — the same
    shape and the same reason as ``retrieval_failed``."""
    repo, obj, _m, _u, _c, _d = await _seed(tmp_path)

    default = await build_verdict(repo, objective_id=obj.objective_id)
    assert default.provenance.judgment_doubt_escalated is False

    escalated = await build_verdict(
        repo, objective_id=obj.objective_id, judgment_doubt_escalated=True
    )
    assert escalated.provenance.judgment_doubt_escalated is True


# ── The seam is free, and moves no existing string ───────────────────


async def test_build_verdict_makes_no_model_call(tmp_path: Path, monkeypatch) -> None:
    """PINS that SEAM 2 stays free.

    The elicitation seam is made to RAISE, and a full ``build_verdict`` over
    a database containing confirmed evidence and a stored opinion still
    succeeds. A future "just re-elicit it here" fails loudly.
    """
    from andamentum.llm_judge import elicit as _elicit

    async def _explode(*_a, **_k):
        raise AssertionError("build_verdict must not call a model")

    monkeypatch.setattr(_elicit, "elicit_criterion_score", _explode)

    repo, obj, _m, _u, _c, _d = await _seed(tmp_path)
    verdict = await build_verdict(repo, objective_id=obj.objective_id)

    assert verdict.objective_id == obj.objective_id


async def test_the_rendered_verdict_line_is_byte_identical_with_and_without_doubt(
    tmp_path: Path,
) -> None:
    """The artefact's ``> **Verdict:**`` line is what the benchmark harness
    reads. ``_verdict_detail_parts`` is deliberately untouched, so populating
    the new fields must not move a single character of it."""
    repo, obj, measured, _u, _c, _d = await _seed(tmp_path)
    with_doubt = await build_verdict(repo, objective_id=obj.objective_id)

    stripped = with_doubt.model_copy(deep=True)
    for cv in stripped.claims:
        cv.scrutiny_verdict = None
        cv.scrutiny_doubt = None
        cv.scrutiny_opinion = None
        cv.judgment_dissent_count = 0
        for ref in [*cv.supporting_evidence, *cv.refuting_evidence]:
            ref.judgment_distribution = None
            ref.judgment_doubt = None
            ref.quality_doubt = None
            ref.direction_opinion = None

    assert render_verdict_line(with_doubt) == render_verdict_line(stripped)


async def test_the_verdict_object_round_trips_through_json(tmp_path: Path) -> None:
    """``Snapshot.verdict`` is a full-JSON round trip, so every new field
    must survive one — including the embedded opinion objects."""
    from andamentum.epistemic.entities.verdict import EpistemicVerdict

    repo, obj, _m, _u, _c, _d = await _seed(tmp_path)
    original = await build_verdict(
        repo, objective_id=obj.objective_id, judgment_doubt_escalated=True
    )

    revived = EpistemicVerdict.model_validate_json(original.model_dump_json())

    assert revived.model_dump() == original.model_dump()
    revived_measured = next(c for c in revived.claims if c.scrutiny_opinion is not None)
    assert revived_measured.scrutiny_opinion is not None
    assert revived_measured.scrutiny_opinion.dissents is True


# ── The readers (so this does not repeat CapEvaluation.doubt) ────────


def test_a_high_doubt_claim_annotates_its_scrutiny_gate_row() -> None:
    """THE READ SITE. Without it the new fields are dead data on the verdict
    object, exactly like ``CapEvaluation.doubt``."""
    from andamentum.epistemic.report_generator import _build_gate_trace

    claim = Claim(
        objective_id="o",
        statement="c",
        scope="x",
        scrutiny_verdict="pass",
        scrutiny_doubt=_HIGH,
    )
    row = next(r for r in _build_gate_trace(claim, None, None) if r.name == "scrutiny")

    assert row.note is not None
    assert "unsure" in row.note


def test_a_dissent_annotates_the_scrutiny_gate_row_even_at_low_doubt() -> None:
    """A CONFIDENT dissent has near-zero entropy, so a doubt-only note would
    stay silent on precisely the claim most likely to be wrong."""
    from andamentum.epistemic.report_generator import _build_gate_trace

    claim = Claim(
        objective_id="o",
        statement="c",
        scope="x",
        scrutiny_verdict="pass",
        scrutiny_doubt=_LOW,
        judgment_dissent_count=1,
    )
    row = next(r for r in _build_gate_trace(claim, None, None) if r.name == "scrutiny")

    assert row.note is not None
    assert "independent judge" in row.note


def test_an_unmeasured_claim_leaves_the_gate_row_untouched() -> None:
    """Every pre-existing fixture renders byte-identically."""
    from andamentum.epistemic.report_generator import _build_gate_trace

    claim = Claim(objective_id="o", statement="c", scope="x", scrutiny_verdict="pass")
    row = next(r for r in _build_gate_trace(claim, None, None) if r.name == "scrutiny")

    assert row.note is None


def test_a_dissenting_confirmation_annotates_the_evidence_line() -> None:
    from andamentum.epistemic.audit_report import _evidence_line

    base = dict(
        evidence_id="e",
        source_type="openalex",
        source_ref="doi:10.1/x",
        extracted_content="c",
        judgment_reasoning="It reports the outcome directly.",
    )
    unconfirmed = _evidence_line(EvidenceSummary(**base))
    agreed = _evidence_line(EvidenceSummary(**base, direction_dissent=False))
    dissented = _evidence_line(EvidenceSummary(**base, direction_dissent=True))

    # A confirmation that AGREED is deliberately silent: annotating agreement
    # would add noise to every line the budget happened to reach.
    assert agreed == unconfirmed
    assert dissented != unconfirmed
    assert "independent judge disagreed" in dissented


def test_claim_summary_doubt_is_matched_by_claim_id_not_index() -> None:
    """The two lists are built by different code. An index assumption would
    silently attach one claim's doubt to another's badge."""
    from andamentum.epistemic.entities.verdict import ClaimVerdict

    summaries = [
        ClaimSummary(claim_id="c-A", statement="A", scope="", assumptions=[], stage="x"),
        ClaimSummary(claim_id="c-B", statement="B", scope="", assumptions=[], stage="x"),
    ]
    # Deliberately the REVERSE order.
    claim_verdicts = [
        ClaimVerdict(claim_id="c-B", scrutiny_doubt=_HIGH),
        ClaimVerdict(claim_id="c-A", scrutiny_doubt=_LOW),
    ]
    doubt_by_claim = {cv.claim_id: cv.scrutiny_doubt for cv in claim_verdicts}
    for summary in summaries:
        summary.scrutiny_doubt = doubt_by_claim.get(summary.claim_id)

    assert summaries[0].scrutiny_doubt == pytest.approx(_LOW)
    assert summaries[1].scrutiny_doubt == pytest.approx(_HIGH)
