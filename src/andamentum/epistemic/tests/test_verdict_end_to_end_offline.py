"""One verdict object, end to end — offline, on BOTH terminals.

Full graph traversals with the agent layer stubbed (``FakeAgentRunner``
via monkeypatched ``DefaultAgentRunner``, plus a stubbed
``validate_alignment``), no providers, and a stubbed embedding call. No
live model is contacted.

What these pin: the number the CLI prints, the number persisted on the
frozen Snapshot, and the number the offline report renders are literally
the same object — not three computations that happen to agree today.

TWO traversals, deliberately. ``_run_insufficient`` starts from
``PrepareObjective`` with no providers, so it reaches the fallibilism
terminal. That is the ONLY terminal it can reach, and an earlier version
of this file claimed "end to end" while covering nothing else — deleting
the verdict line from the DIRECTIONAL renderer left the whole file green.
``_run_directional`` therefore seeds an adjudicated claim and enters the
graph at ``CombineClaimVerdicts``, which is the real synthesis-stage
entry point, so the directional renderer is genuinely exercised.

``validate_alignment`` is stubbed because it does NOT go through the
agent runner: ``preplanning`` calls it with
``model=agent_runner.model``, which builds a real pydantic-ai ``Agent``
from the fake model id. It fails at model resolution rather than
reaching the network, but leaving it live meant clarification quarantined
on every run and the traversal never got past preplanning.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, ClaimStage, Evidence, Objective
from andamentum.epistemic.graph import run_epistemic_graph
from andamentum.epistemic.repository import EpistemicRepository

pytestmark = pytest.mark.asyncio


def _stub_agents(monkeypatch) -> None:
    """Make the whole agent layer offline and deterministic."""
    import andamentum.epistemic.alignment as alignment_mod
    import andamentum.epistemic.embeddings as embeddings_mod
    import andamentum.epistemic.runner as runner_mod
    from andamentum.epistemic.alignment import AlignmentResult
    from andamentum.epistemic.tests.conftest import FakeAgentRunner

    monkeypatch.setattr(
        runner_mod,
        "DefaultAgentRunner",
        lambda model: FakeAgentRunner(),  # noqa: ARG005 — stub matches real signature
    )

    async def _aligned(**kwargs):  # noqa: ANN003
        return AlignmentResult(aligned=True)

    monkeypatch.setattr(alignment_mod, "validate_alignment", _aligned)

    async def _fake_embed(texts, *, model=None, **kwargs):  # noqa: ARG001
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(embeddings_mod, "embed_texts", _fake_embed)


async def _run_insufficient(tmp_path: Path, monkeypatch, db_name: str):
    """Offline traversal to the FALLIBILISM terminal. Returns (result, repo)."""
    _stub_agents(monkeypatch)

    result = await run_epistemic_graph(
        question="Aspirin reduces colorectal cancer risk in adults.",
        database_name=db_name,
        mode="verify",
        db_dir=str(tmp_path),
        model="fake:test-model",
        embedding_model="fake-embeddings",
        searxng_url="http://searx.invalid",
        provider="none",
        providers={},
    )

    store = DocumentStore.for_database(db_name, db_dir=tmp_path)
    await store.initialize()
    return result, EpistemicRepository(store)


async def _run_directional(tmp_path: Path, monkeypatch, db_name: str):
    """Offline traversal to the DIRECTIONAL terminal. Returns (result, repo).

    Seeds a verify-mode objective with one adjudicated (refuted) claim and
    its evidence, then enters the graph at ``CombineClaimVerdicts`` — the
    synthesis stage's real entry point, and ``CheckCompletion``'s sole
    predecessor.
    """
    _stub_agents(monkeypatch)
    from andamentum.epistemic.graph.nodes import CombineClaimVerdicts

    store = DocumentStore.for_database(db_name, db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    question = "Aspirin reduces colorectal cancer risk in adults."
    obj = Objective(
        description=question,
        claim_to_verify=question,
        question_type="verificatory",
        phase="analyzed",
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    evidence_ids: list[str] = []
    for i in range(3):
        ev = Evidence(
            objective_id=obj.entity_id,
            source_ref=f"doi:refute/{i}",
            extracted=True,
            extracted_content="A refuting randomised trial.",
            support_judgment="contradicts",
            judgment_distribution=[0.0, 1.0, 0.0],
            quality_score=0.8,
        )
        await repo.save(ev)
        evidence_ids.append(ev.entity_id)

    claim = Claim(
        objective_id=obj.entity_id,
        statement=question,
        scope="adults",
        stage=ClaimStage.SUPPORTED,
        evidence_ids=evidence_ids,
        integrated_assessment="contradicts",
        integrated_confidence=0.8,
    )
    await repo.save(claim)

    result = await run_epistemic_graph(
        question=question,
        database_name=db_name,
        mode="verify",
        db_dir=str(tmp_path),
        objective_id=obj.entity_id,
        start_at=CombineClaimVerdicts,
        model="fake:test-model",
        embedding_model="fake-embeddings",
        searxng_url="http://searx.invalid",
        provider="none",
        providers={},
    )
    return result, repo


# ── The directional arm ───────────────────────────────────────────────


async def test_directional_run_reaches_the_directional_terminal(
    tmp_path: Path, monkeypatch
) -> None:
    """Guard on the fixture itself.

    Without this the other directional tests could silently degrade into
    another insufficient-terminal run and keep passing — which is exactly
    how the previous version of this file covered nothing.
    """
    result, repo = await _run_directional(tmp_path, monkeypatch, "verdict_e2e_dir")

    assert result.verdict is not None
    assert result.verdict.direction == "contradicts"
    assert result.verdict.label == "Refuted"
    artefacts = await repo.get_artefacts_for_objective(result.objective_id)
    assert artefacts
    assert artefacts[-1].artefact_type == "summary"


async def test_directional_artefact_verdict_line_matches_the_object(
    tmp_path: Path, monkeypatch
) -> None:
    """The artefact's machine-readable headline is a rendering of the
    same object the CLI reports — no prose in the loop. Deleting the
    ``> **Verdict:**`` line from the directional renderer fails here."""
    from andamentum.epistemic.entities.verdict import render_verdict_line

    result, repo = await _run_directional(
        tmp_path, monkeypatch, "verdict_e2e_dir_artefact"
    )
    artefacts = await repo.get_artefacts_for_objective(result.objective_id)
    expected = f"> **Verdict:** {render_verdict_line(result.verdict)}"
    assert expected in artefacts[-1].content
    assert expected in artefacts[-1].content_body
    assert "Refuted" in expected


async def test_directional_headline_and_claim_badges_agree(
    tmp_path: Path, monkeypatch
) -> None:
    """One document, one story. The objective headline and every
    per-claim badge come from the same object."""
    result, repo = await _run_directional(
        tmp_path, monkeypatch, "verdict_e2e_dir_badge"
    )

    from andamentum.epistemic.report_generator import ReportGenerator

    generator = ReportGenerator(repo.store, "verdict_e2e_dir_badge")
    data = await generator.extract_report_data(objective_id=result.objective_id)
    assert data is not None
    assert data.verdict_object is not None
    assert data.verdict_object.label == "Refuted"
    assert data.claims
    assert all(c.verdict_label == "Refuted" for c in data.claims)


# ── The fallibilism arm ───────────────────────────────────────────────


async def test_pipeline_result_verdict_matches_the_persisted_snapshot_verdict(
    tmp_path: Path, monkeypatch
) -> None:
    """One object end to end, no post-End recompute.

    ``result.posterior`` IS the report embedded in ``result.verdict``;
    restoring the post-End ``compute_posterior`` call fails the identity
    assertion below, because a second computation produces a different
    object even when it produces the same number.
    """
    result, repo = await _run_insufficient(tmp_path, monkeypatch, "verdict_e2e")

    assert result.verdict is not None
    assert result.posterior is result.verdict.posterior_report

    objective = await repo.get("objective", result.objective_id)
    assert objective.snapshot_id, "the run must have frozen a snapshot"
    snapshot = await repo.get("snapshot", objective.snapshot_id)
    assert snapshot.verdict is not None
    assert snapshot.verdict.direction == result.verdict.direction
    assert snapshot.verdict.posterior == result.verdict.posterior
    assert (
        snapshot.verdict.provenance.n_adjudicated
        == result.verdict.provenance.n_adjudicated
    )


async def test_insufficient_artefact_verdict_line_matches_the_object(
    tmp_path: Path, monkeypatch
) -> None:
    """The fallibilism terminal renders the same object through the
    terminal-specific renderer, which pins the label to "Insufficient
    evidence" — the thing the document's title and ``artefact_type``
    already say."""
    from andamentum.epistemic.entities.verdict import render_insufficient_verdict_line

    result, repo = await _run_insufficient(
        tmp_path, monkeypatch, "verdict_e2e_artefact"
    )
    artefacts = await repo.get_artefacts_for_objective(result.objective_id)
    assert artefacts
    assert artefacts[-1].artefact_type == "insufficient"
    expected = f"> **Verdict:** {render_insufficient_verdict_line(result.verdict)}"
    assert expected in artefacts[-1].content
    assert expected in artefacts[-1].content_body


async def test_offline_report_reads_the_persisted_verdict(
    tmp_path: Path, monkeypatch
) -> None:
    """The offline report is a READER, not a second computation.

    ``compute_posterior`` is monkeypatched to raise: the report must
    still succeed, because it loads the verdict the run persisted. This
    is the source of the documented live/offline ``terminal_state``
    divergence, closed.
    """
    result, repo = await _run_directional(tmp_path, monkeypatch, "verdict_e2e_offline")

    import andamentum.epistemic.confidence as confidence_mod

    async def _raising(*args, **kwargs):
        raise AssertionError("the offline report must not recompute the posterior")

    monkeypatch.setattr(confidence_mod, "compute_posterior", _raising)

    from andamentum.epistemic.report_generator import ReportGenerator

    generator = ReportGenerator(repo.store, "verdict_e2e_offline")
    data = await generator.extract_report_data(objective_id=result.objective_id)

    assert data is not None
    assert data.verdict_object is not None
    assert result.verdict is not None
    assert data.verdict_object.direction == result.verdict.direction
    assert data.confidence_scores is not None
    assert data.confidence_scores.posterior == result.verdict.posterior
    # Per-claim badges are populated from the object, not left blank.
    assert data.claims, "the directional run must have produced a claim to badge"
    assert all(c.verdict_label for c in data.claims)
