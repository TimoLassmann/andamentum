"""Prose RENDERS the verdict; it cannot override it.

The load-bearing property of the whole change. Before this, the chain was
posterior → prompt → LLM-written prose → (in evaluation) a second LLM
reads that prose → label: three lossy transformations, and the writer's
sentence was the only machine-readable verdict the artefact carried.
"""

from __future__ import annotations

from andamentum.epistemic.entities import (
    Claim,
    ClaimStage,
    Evidence,
    Objective,
    Snapshot,
)
from andamentum.epistemic.entities.verdict import render_verdict_line
from andamentum.epistemic.operations import (
    SynthesizeInsufficientReportOperation,
    SynthesizeReportOperation,
)
from andamentum.epistemic.operations.base import OperationInput
from andamentum.epistemic.tests.conftest import FakeAgentRunner, make_verdict


async def _seed(repo, *, posterior: float | None, oid: str) -> str:
    obj = Objective(
        entity_id=oid,
        objective_id=oid,
        description="Aspirin reduces colorectal cancer risk.",
        claim_to_verify="Aspirin reduces colorectal cancer risk.",
        question_type="verificatory",
    )
    await repo.save(obj)

    ev = Evidence(
        entity_id=f"{oid}-e1",
        objective_id=oid,
        extracted=True,
        extracted_content="A refuting study.",
        support_judgment="contradicts",
        quality_score=0.7,
    )
    await repo.save(ev)

    claim = Claim(
        entity_id=f"{oid}-c1",
        objective_id=oid,
        statement="Aspirin reduces colorectal cancer risk.",
        stage=ClaimStage.SUPPORTED,
        evidence_ids=[ev.entity_id],
        integrated_assessment="contradicts",
        integrated_confidence=0.8,
    )
    await repo.save(claim)

    snap = Snapshot(
        entity_id=f"{oid}-snap",
        objective_id=oid,
        snapshot_type="final",
        claim_ids=[claim.entity_id],
        evidence_ids=[ev.entity_id],
        verdict=make_verdict(oid, posterior=posterior, claim_ids=(claim.entity_id,)),
    )
    await repo.save(snap)
    return snap.entity_id


async def test_prose_cannot_override_the_verdict(repo) -> None:
    """THE TEST FOR THE WHOLE CHANGE.

    The snapshot's verdict is a low posterior (Refuted). The writer is
    stubbed to say the exact opposite in prose. The artefact's
    machine-readable ``> **Verdict:**`` line must still read Refuted,
    and must carry the object's posterior.
    """
    snap_id = await _seed(repo, posterior=0.115, oid="obj-override")
    runner = FakeAgentRunner(
        overrides={
            "epistemic_write_answer": {
                "title": "Aspirin and colorectal cancer",
                "answer": "Yes. The evidence strongly supports this claim.",
            }
        }
    )
    op = SynthesizeReportOperation(repo=repo, agent_runner=runner)
    result = await op.execute(
        OperationInput(
            entity_id=snap_id, entity_type="snapshot", operation="synthesize_report"
        )
    )
    assert result.success

    artefact = (await repo.query("artefact", objective_id="obj-override"))[0]
    verdict_lines = [
        ln for ln in artefact.content.splitlines() if ln.startswith("> **Verdict:**")
    ]
    assert len(verdict_lines) == 1
    assert "Refuted" in verdict_lines[0]
    assert "0.115" in verdict_lines[0]
    # The writer's prose is still present — it just isn't the verdict.
    assert "strongly supports" in artefact.content


async def test_verdict_line_present_in_both_content_and_content_body(repo) -> None:
    """The benchmark judge reads ``content_body`` (the quality-signal-free
    view). Putting the verdict only in the header block would leave the
    defect exactly where it was measured."""
    snap_id = await _seed(repo, posterior=0.115, oid="obj-body")
    op = SynthesizeReportOperation(repo=repo, agent_runner=FakeAgentRunner())
    await op.execute(
        OperationInput(
            entity_id=snap_id, entity_type="snapshot", operation="synthesize_report"
        )
    )
    artefact = (await repo.query("artefact", objective_id="obj-body"))[0]
    snapshot = await repo.get("snapshot", snap_id)
    line = f"> **Verdict:** {render_verdict_line(snapshot.verdict)}"
    assert line in artefact.content
    assert line in artefact.content_body


async def test_claims_adjudicated_header_replaces_the_stage_counter(repo) -> None:
    """RULING C13. "Claims Established: 0 of 1" printed beside a
    "Refuted" verdict is literally the disagreeing-headline pair the
    2026-07-08 audit read case 781 off — a cap-decided claim is never
    promoted to SUPPORTED, so the stage counter reads 0. The header now
    reports what the verdict object counted."""
    snap_id = await _seed(repo, posterior=0.115, oid="obj-header")
    op = SynthesizeReportOperation(repo=repo, agent_runner=FakeAgentRunner())
    await op.execute(
        OperationInput(
            entity_id=snap_id, entity_type="snapshot", operation="synthesize_report"
        )
    )
    artefact = (await repo.query("artefact", objective_id="obj-header"))[0]
    assert "**Claims Adjudicated:** 1 of 1" in artefact.content
    assert "Claims Established" not in artefact.content


async def test_both_synthesis_operations_raise_without_a_verdict(repo) -> None:
    """Fail loud. A silent "omit the line" fallback would hide the
    routing bug that produced the verdict-less snapshot."""
    import pytest

    obj = Objective(
        entity_id="obj-noverdict",
        objective_id="obj-noverdict",
        description="x",
    )
    await repo.save(obj)
    snap = Snapshot(
        entity_id="snap-noverdict",
        objective_id="obj-noverdict",
        snapshot_type="final",
    )
    await repo.save(snap)

    work = OperationInput(
        entity_id="snap-noverdict",
        entity_type="snapshot",
        operation="synthesize_report",
    )
    with pytest.raises(RuntimeError, match="carries no verdict"):
        await SynthesizeReportOperation(
            repo=repo, agent_runner=FakeAgentRunner()
        ).execute(work)
    with pytest.raises(RuntimeError, match="carries no verdict"):
        await SynthesizeInsufficientReportOperation(repo=repo).execute(work)


async def test_insufficient_terminal_renders_the_same_object_type(repo) -> None:
    """One shape at both terminals, and the fallibilism terminal stays
    code-owned rather than prompt-owned."""
    obj = Objective(
        entity_id="obj-insuf-render",
        objective_id="obj-insuf-render",
        description="x",
    )
    await repo.save(obj)
    snap = Snapshot(
        entity_id="snap-insuf-render",
        objective_id="obj-insuf-render",
        snapshot_type="final",
        verdict=make_verdict("obj-insuf-render", n_active=1, n_adjudicated=0),
    )
    await repo.save(snap)

    runner = FakeAgentRunner()
    op = SynthesizeInsufficientReportOperation(repo=repo, agent_runner=runner)
    await op.execute(
        OperationInput(
            entity_id=snap.entity_id,
            entity_type="snapshot",
            operation="synthesize_insufficient",
        )
    )

    artefact = (await repo.query("artefact", objective_id="obj-insuf-render"))[0]
    line = next(
        ln for ln in artefact.content.splitlines() if ln.startswith("> **Verdict:**")
    )
    assert line[len("> **Verdict:** ") :].startswith("Insufficient evidence")
    assert artefact.artefact_type == "insufficient"
    assert runner.calls == [], "the insufficient terminal must make no LLM call"


async def test_insufficient_terminal_with_a_posterior_still_says_insufficient(
    repo,
) -> None:
    """The insufficient terminal's headline must agree with the document
    it heads.

    Reachable shape, not a hypothetical: ``CheckSynthesisDemand.
    _maybe_loop_back`` falls through to ``SynthesizeInsufficient`` when
    every claim is cycle-capped and every ``integrated_assessment`` is
    "insufficient" — the fixture ``test_phase4_synthesis_loop_back``
    already uses. Those claims ARE adjudicated, so the posterior is 0.5,
    the direction is "insufficient" and the shared renderer printed
    ``Inconclusive — posterior 0.500 · 1 of 1 claim adjudicated`` inside
    a document titled "Insufficient Evidence" whose body claimed the
    investigation reached no integration verdict.

    The earlier version of ``test_insufficient_terminal_renders_the_same
    _object_type`` used ``posterior_report=None``, where the label is
    trivially "Insufficient evidence" — it structurally could not catch
    this.
    """
    obj = Objective(
        entity_id="obj-insuf-posterior",
        objective_id="obj-insuf-posterior",
        description="x",
    )
    await repo.save(obj)
    verdict = make_verdict(
        "obj-insuf-posterior",
        posterior=0.5,
        n_active=1,
        n_adjudicated=1,
        claim_ids=("c1",),
    )
    verdict.provenance.n_cycle_capped = 1
    assert verdict.label == "Inconclusive", (
        "fixture guard: the OBJECT's own label is directionless-but-not-"
        "insufficient; the terminal is what pins the headline"
    )

    snap = Snapshot(
        entity_id="snap-insuf-posterior",
        objective_id="obj-insuf-posterior",
        snapshot_type="final",
        verdict=verdict,
    )
    await repo.save(snap)

    await SynthesizeInsufficientReportOperation(repo=repo).execute(
        OperationInput(
            entity_id=snap.entity_id,
            entity_type="snapshot",
            operation="synthesize_insufficient",
        )
    )
    artefact = (await repo.query("artefact", objective_id="obj-insuf-posterior"))[0]
    line = next(
        ln for ln in artefact.content.splitlines() if ln.startswith("> **Verdict:**")
    )
    assert line[len("> **Verdict:** ") :].startswith("Insufficient evidence")
    assert "Inconclusive" not in line
    # The provenance clauses still come from the same object.
    assert "1 of 1 claim adjudicated" in line
    assert "1 cycle-capped" in line
    assert artefact.artefact_type == "insufficient"
    # ...and the body must not contradict the count on the line above it.
    assert "without reaching an integration verdict" not in artefact.content
    assert "adjudicated 1 of 1 claim(s), but none of them directionally" in (
        artefact.content
    )
