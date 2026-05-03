"""Tests for stage-runner kwargs on ``run_epistemic_graph``.

The stage-runner machinery exposes ``stop_after`` (Phase 1) and
``start_at`` (Phase 2) as the single mechanism for running any prefix
or suffix of the graph against a real DB. The DB is the checkpoint;
callers resume by passing ``start_at`` on a later invocation.

These tests pin the contract of ``stop_after``:

  1. With ``stop_after=None`` the graph runs to completion (no
     behavior change for existing callers).
  2. With ``stop_after=PrepareObjective`` the graph executes only the
     entry node and returns a partial PipelineResult whose ``status``
     names the stop point.
  3. The DB has the state the entry node wrote (``question_type``).
  4. ``posterior`` is None on a stage run — no full traversal, no
     final answer to score.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.epistemic.graph import run_epistemic_graph
from andamentum.epistemic.graph.nodes import PrepareObjective
from andamentum.epistemic.repository import EpistemicRepository
from andamentum.document_store import DocumentStore


pytestmark = pytest.mark.asyncio


async def test_stop_after_prepare_objective_writes_question_type(
    tmp_path: Path, monkeypatch
) -> None:
    """Run with stop_after=PrepareObjective. Assert the entry node
    executed (question_type written), the run reports it stopped
    where we asked, and no posterior was computed."""
    # Use the in-tree FakeAgentRunner via monkey-patching DefaultAgentRunner
    # so we never make real LLM calls.
    from andamentum.epistemic.tests.conftest import FakeAgentRunner
    import andamentum.epistemic.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "DefaultAgentRunner",
        lambda model: FakeAgentRunner(),  # noqa: ARG005 — stub matches real signature
    )

    result = await run_epistemic_graph(
        question="Is exercise good for cardiovascular health?",
        database_name="stage_stop_after",
        db_dir=str(tmp_path),
        model="fake:test-model",
        embedding_model="fake-embeddings",
        decompose=False,
        stop_after=PrepareObjective,
    )

    assert result.status == "stopped_after:PrepareObjective"
    assert result.successful == 0
    assert result.posterior is None
    assert result.objective_id is not None

    store = DocumentStore.for_database("stage_stop_after", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)
    obj = await repo.get("objective", result.objective_id)
    assert obj is not None
    assert obj.question_type is not None, (
        "PrepareObjective is the only writer of question_type; "
        "if this is None, the entry node did not run."
    )


# Note: there is no explicit "stop_after=None preserves existing
# behavior" test. The existing 1848 tests run with stop_after=None
# (the default), so the no-change path is tested by every other
# call site implicitly. Adding a redundant 60s end-to-end run here
# would burn iteration time for no extra signal.
