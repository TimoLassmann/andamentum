"""One verdict object, end to end — offline.

A full graph traversal with the agent layer stubbed (``FakeAgentRunner``
via monkeypatched ``DefaultAgentRunner``), no providers, and a stubbed
embedding call. No live model is contacted.

What these pin: the number the CLI prints, the number persisted on the
frozen Snapshot, and the number the offline report renders are literally
the same object — not three computations that happen to agree today.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.document_store import DocumentStore
from andamentum.epistemic.graph import run_epistemic_graph
from andamentum.epistemic.repository import EpistemicRepository

pytestmark = pytest.mark.asyncio


async def _run(tmp_path: Path, monkeypatch, db_name: str):
    """Offline traversal to a terminal. Returns (result, repo)."""
    import andamentum.epistemic.embeddings as embeddings_mod
    import andamentum.epistemic.runner as runner_mod
    from andamentum.epistemic.tests.conftest import FakeAgentRunner

    monkeypatch.setattr(
        runner_mod,
        "DefaultAgentRunner",
        lambda model: FakeAgentRunner(),  # noqa: ARG005 — stub matches real signature
    )

    async def _fake_embed(texts, *, model=None, **kwargs):  # noqa: ARG001
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(embeddings_mod, "embed_texts", _fake_embed)

    result = await run_epistemic_graph(
        question="Aspirin reduces colorectal cancer risk in adults.",
        database_name=db_name,
        mode="verify",
        db_dir=str(tmp_path),
        model="fake:test-model",
        embedding_model="fake-embeddings",
        provider="none",
        providers={},
    )

    store = DocumentStore.for_database(db_name, db_dir=tmp_path)
    await store.initialize()
    return result, EpistemicRepository(store)


async def test_pipeline_result_verdict_matches_the_persisted_snapshot_verdict(
    tmp_path: Path, monkeypatch
) -> None:
    """One object end to end, no post-End recompute.

    ``result.posterior`` IS the report embedded in ``result.verdict``;
    restoring the post-End ``compute_posterior`` call fails the identity
    assertion below, because a second computation produces a different
    object even when it produces the same number.
    """
    result, repo = await _run(tmp_path, monkeypatch, "verdict_e2e")

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


async def test_artefact_verdict_line_matches_the_object(
    tmp_path: Path, monkeypatch
) -> None:
    """The artefact's machine-readable headline is a rendering of the
    same object the CLI reports — no prose in the loop."""
    from andamentum.epistemic.entities.verdict import render_verdict_line

    result, repo = await _run(tmp_path, monkeypatch, "verdict_e2e_artefact")
    artefacts = await repo.get_artefacts_for_objective(result.objective_id)
    assert artefacts
    expected = f"> **Verdict:** {render_verdict_line(result.verdict)}"
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
    result, repo = await _run(tmp_path, monkeypatch, "verdict_e2e_offline")

    import andamentum.epistemic.confidence as confidence_mod

    async def _raising(*args, **kwargs):
        raise AssertionError(
            "the offline report must not recompute the posterior"
        )

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
    assert all(c.verdict_label for c in data.claims)
