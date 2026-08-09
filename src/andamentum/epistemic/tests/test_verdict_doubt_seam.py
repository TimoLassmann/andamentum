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
from typing import NamedTuple

import pytest

from andamentum.document_store import DocumentStore
from andamentum.epistemic.confidence import build_verdict
from andamentum.epistemic.entities import Claim, Evidence, Objective, Snapshot
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.entities.judge_opinion import JudgeOpinion
from andamentum.epistemic.entities.verdict import render_verdict_line
from andamentum.epistemic.report_data import EvidenceSummary
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


class _Seeded(NamedTuple):
    """Named rather than a bare tuple: seven same-shaped entities whose
    positional unpacking would silently transpose."""

    repo: EpistemicRepository
    objective: Objective
    measured: Claim
    unmeasured: Claim
    confirmed: Evidence
    distributionless: Evidence
    confirmed_refuting: Evidence


async def _seed(tmp_path: Path, name: str = "seam2") -> _Seeded:
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
    # A CONTRADICTING item that IS instrumented. Without it the refuting
    # branch of ``_evidence_refs_for_claim`` is exercised only by the
    # distribution-less item above, so every assertion on it is
    # ``None == None`` and stripping the reliability fields from that
    # branch outright would go unnoticed — on a REFUTED claim those are
    # the load-bearing items.
    confirmed_refuting = Evidence(
        objective_id=obj.entity_id,
        source_type="openalex",
        source_ref="doi:confirmed-refuting",
        extracted_content="A contradicting finding.",
        extracted=True,
        quality_score=0.6,
        cluster_status="representative",
        judgment_distribution=[0.15, 0.75, 0.10],
        support_judgment="contradicts",
        quality_distribution=[0.2, 0.3, 0.5],
        direction_opinion=_opinion(
            overall="meets", reference="contradicts", dissents=False
        ),
    )
    await repo.save(confirmed_refuting)
    measured.evidence_ids = [
        confirmed.entity_id,
        distributionless.entity_id,
        confirmed_refuting.entity_id,
    ]
    measured.evidence_count = 3
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

    return _Seeded(
        repo=repo,
        objective=obj,
        measured=measured,
        unmeasured=unmeasured,
        confirmed=confirmed,
        distributionless=distributionless,
        confirmed_refuting=confirmed_refuting,
    )


# ── EvidenceRef ──────────────────────────────────────────────────────


async def test_evidence_ref_carries_the_doubt_of_the_item_it_names(
    tmp_path: Path,
) -> None:
    """The refs on a ClaimVerdict are provably the items that produced its
    number — so the reliability they report must be those items' own."""
    seeded = await _seed(tmp_path)
    repo = seeded.repo
    obj = seeded.objective
    measured = seeded.measured
    confirmed = seeded.confirmed

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
    seeded = await _seed(tmp_path)
    repo = seeded.repo
    obj = seeded.objective
    measured = seeded.measured
    distributionless = seeded.distributionless

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


async def test_a_refuting_item_carries_the_same_four_fields(
    tmp_path: Path,
) -> None:
    """THE REFUTING BRANCH IS NOT A COPY THAT CAN ROT.

    ``_evidence_refs_for_claim`` fills the reliability fields at TWO
    construction sites. Stripping them from the ``elif c > s:`` branch used
    to leave the suite green — on a REFUTED claim that branch holds the
    load-bearing evidence, so the reliability of the whole verdict would go
    dark exactly where it matters most.
    """
    seeded = await _seed(tmp_path)

    verdict = await build_verdict(
        seeded.repo, objective_id=seeded.objective.objective_id
    )
    cv = next(c for c in verdict.claims if c.claim_id == seeded.measured.entity_id)
    ref = next(
        r
        for r in cv.refuting_evidence
        if r.evidence_id == seeded.confirmed_refuting.entity_id
    )

    assert ref.stance == "contradicts"
    assert ref.judgment_distribution == pytest.approx(
        seeded.confirmed_refuting.judgment_distribution
    )
    assert ref.judgment_doubt == pytest.approx(
        seeded.confirmed_refuting.judgment_entropy
    )
    assert ref.quality_doubt == pytest.approx(seeded.confirmed_refuting.quality_doubt)
    assert ref.direction_opinion is not None
    assert ref.direction_opinion.reference == "contradicts"
    assert ref.direction_opinion.dissents is False


async def test_a_confirmed_item_carries_its_whole_opinion_object(
    tmp_path: Path,
) -> None:
    """The whole ``JudgeOpinion``, not a scalar: a consumer needs the dissent
    flag AND the doubt, and they measure different things."""
    seeded = await _seed(tmp_path)
    repo = seeded.repo
    obj = seeded.objective
    measured = seeded.measured
    confirmed = seeded.confirmed

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
    seeded = await _seed(tmp_path)
    repo = seeded.repo
    obj = seeded.objective
    measured = seeded.measured

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
    seeded = await _seed(tmp_path)
    repo = seeded.repo
    obj = seeded.objective
    unmeasured = seeded.unmeasured

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
    seeded = await _seed(tmp_path)
    repo = seeded.repo
    obj = seeded.objective

    prov = (await build_verdict(repo, objective_id=obj.objective_id)).provenance

    assert prov.n_active_claims == 2
    assert prov.n_high_doubt_claims == 1
    assert prov.n_doubt_unmeasured_claims == 1


async def test_provenance_records_the_escalation_flag_it_was_given(
    tmp_path: Path,
) -> None:
    """Gate 0's one-shot flag lives on graph state, not the DB — the same
    shape and the same reason as ``retrieval_failed``."""
    seeded = await _seed(tmp_path)
    repo = seeded.repo
    obj = seeded.objective

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

    seeded = await _seed(tmp_path)

    repo = seeded.repo

    obj = seeded.objective
    verdict = await build_verdict(repo, objective_id=obj.objective_id)

    assert verdict.objective_id == obj.objective_id


async def test_the_rendered_verdict_line_is_byte_identical_with_and_without_doubt(
    tmp_path: Path,
) -> None:
    """The artefact's ``> **Verdict:**`` line is what the benchmark harness
    reads. ``_verdict_detail_parts`` is deliberately untouched, so populating
    the new fields must not move a single character of it."""
    seeded = await _seed(tmp_path)
    repo = seeded.repo
    obj = seeded.objective
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

    seeded = await _seed(tmp_path)

    repo = seeded.repo

    obj = seeded.objective
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

    def summary(direction_dissent: bool | None) -> EvidenceSummary:
        """Built by call, not by ``**dict`` unpacking: pyright cannot infer
        per-key types through a dict splat, so the splat form reports a
        spurious error on every field of every construction."""
        return EvidenceSummary(
            evidence_id="e",
            source_type="openalex",
            source_ref="doi:10.1/x",
            extracted_content="c",
            judgment_reasoning="It reports the outcome directly.",
            direction_dissent=direction_dissent,
        )

    unconfirmed = _evidence_line(summary(None))
    agreed = _evidence_line(summary(False))
    dissented = _evidence_line(summary(True))

    # A confirmation that AGREED is deliberately silent: annotating agreement
    # would add noise to every line the budget happened to reach.
    assert agreed == unconfirmed
    assert dissented != unconfirmed
    assert "independent judge disagreed" in dissented


async def _report_data(tmp_path: Path, seeded: _Seeded, *, name: str = "seam2"):
    """Freeze the run's verdict onto a Snapshot and run the REAL extractor.

    This is what makes the reader assertions below non-tautological: they go
    through ``ReportGenerator.extract_report_data``, so deleting the backfill
    inside it fails them. Re-implementing the backfill in a test body — the
    shape this file used to have — exercises zero production code and passes
    under a deleted implementation.
    """
    from andamentum.epistemic.report_generator import ReportGenerator

    repo = seeded.repo
    verdict = await build_verdict(
        repo, objective_id=seeded.objective.objective_id, judgment_doubt_escalated=True
    )
    snap = Snapshot(
        objective_id=seeded.objective.objective_id,
        claim_ids=[seeded.measured.entity_id, seeded.unmeasured.entity_id],
        snapshot_type="final",
        verdict=verdict,
    )
    await repo.save(snap)
    objective = await repo.get("objective", seeded.objective.objective_id)
    objective.snapshot_id = snap.entity_id
    await repo.save(objective)

    generator = ReportGenerator(repo.store, name)
    data = await generator.extract_report_data(
        objective_id=seeded.objective.objective_id
    )
    assert data is not None
    return data, verdict


async def test_claim_summary_doubt_is_matched_by_claim_id_not_index(
    tmp_path: Path,
) -> None:
    """THE PRODUCTION PATH, under REVERSED ordering.

    The claim-summary list and the verdict's claim list are built by
    different code. An index assumption would silently attach one claim's
    doubt to another's badge, and the two claims here are seeded so exactly
    one carries a measurement — so a transposition is visible.
    """
    seeded = await _seed(tmp_path, "seam2_by_id")
    data, verdict = await _report_data(tmp_path, seeded, name="seam2_by_id")

    # Reverse the verdict's claim order in place and re-extract, so the two
    # lists provably disagree on index while agreeing on id.
    snap_id = (
        await seeded.repo.get("objective", seeded.objective.objective_id)
    ).snapshot_id
    assert snap_id is not None
    snap = await seeded.repo.get("snapshot", snap_id)
    assert snap.verdict is not None
    snap.verdict.claims = list(reversed(snap.verdict.claims))
    await seeded.repo.save(snap)

    from andamentum.epistemic.report_generator import ReportGenerator

    reversed_data = await ReportGenerator(
        seeded.repo.store, "seam2_by_id"
    ).extract_report_data(objective_id=seeded.objective.objective_id)
    assert reversed_data is not None

    for extracted in (data, reversed_data):
        by_id = {c.claim_id: c.scrutiny_doubt for c in extracted.claims}
        assert by_id[seeded.measured.entity_id] == pytest.approx(_HIGH)
        assert by_id[seeded.unmeasured.entity_id] is None


async def test_evidence_summary_doubt_comes_from_the_production_extractor(
    tmp_path: Path,
) -> None:
    """PINS ``EvidenceSummary.judgment_doubt``.

    Nulling the fill inside ``extract_report_data`` used to leave the whole
    suite green — the field was dead, unpinned data in exactly the shape of
    ``CapEvaluation.doubt``.
    """
    seeded = await _seed(tmp_path, "seam2_evsum")
    data, _verdict = await _report_data(tmp_path, seeded, name="seam2_evsum")

    by_id = {e.evidence_id: e for e in data.evidence}
    assert by_id[seeded.confirmed.entity_id].judgment_doubt == pytest.approx(
        seeded.confirmed.judgment_entropy
    )
    assert by_id[seeded.confirmed.entity_id].direction_dissent is True
    assert by_id[seeded.confirmed_refuting.entity_id].direction_dissent is False
    # Never measured stays None — not 0.0, and not False.
    assert by_id[seeded.distributionless.entity_id].judgment_doubt is None
    assert by_id[seeded.distributionless.entity_id].direction_dissent is None


async def test_the_report_reads_reliability_off_the_verdict_object(
    tmp_path: Path,
) -> None:
    """SEAM 2 IS ONLY CLOSED IF THE VERDICT OBJECT IS THE THING READ.

    The claim entity and the verdict object are deliberately made to
    DISAGREE here. A renderer that reads the entity would report the
    entity's numbers and this test would fail — which is the whole point:
    without it, every one of the eleven new verdict-object fields could be
    deleted from the read path and the suite would stay green.
    """
    seeded = await _seed(tmp_path, "seam2_source")
    data, _verdict = await _report_data(tmp_path, seeded, name="seam2_source")

    # Baseline: the frozen verdict said 2 dissents, and the note says so.
    row = next(
        r
        for c in data.claims
        if c.claim_id == seeded.measured.entity_id
        for r in c.gate_trace
        if r.name == "scrutiny"
    )
    assert row.note is not None
    assert "disagreed with 2 evidence direction(s)" in row.note

    # Now change the LIVE entity only. The report must keep reporting the
    # frozen record.
    live = await seeded.repo.get("claim", seeded.measured.entity_id)
    live.judgment_dissent_count = 7
    live.scrutiny_doubt = _LOW
    live.scrutiny_opinion = None
    await seeded.repo.save(live)

    from andamentum.epistemic.report_generator import ReportGenerator

    again = await ReportGenerator(
        seeded.repo.store, "seam2_source"
    ).extract_report_data(objective_id=seeded.objective.objective_id)
    assert again is not None
    row_again = next(
        r
        for c in again.claims
        if c.claim_id == seeded.measured.entity_id
        for r in c.gate_trace
        if r.name == "scrutiny"
    )
    assert row_again.note == row.note


async def test_the_report_falls_back_to_the_entity_without_a_snapshot(
    tmp_path: Path,
) -> None:
    """A report extracted before any snapshot exists still annotates.

    The fallback is deliberate, and pinning it stops a later "just read the
    verdict object" simplification from silently blanking the gate-trace
    note on every mid-run report.
    """
    from andamentum.epistemic.report_generator import ReportGenerator

    seeded = await _seed(tmp_path, "seam2_fallback")
    data = await ReportGenerator(
        seeded.repo.store, "seam2_fallback"
    ).extract_report_data(objective_id=seeded.objective.objective_id)
    assert data is not None

    row = next(
        r
        for c in data.claims
        if c.claim_id == seeded.measured.entity_id
        for r in c.gate_trace
        if r.name == "scrutiny"
    )
    assert row.note is not None
    assert "disagreed with 2 evidence direction(s)" in row.note
