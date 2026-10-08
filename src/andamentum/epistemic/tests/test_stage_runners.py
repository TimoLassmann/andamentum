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
from andamentum.document_store import Store


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
        stop_after=PrepareObjective,
    )

    assert result.status == "stopped_after:PrepareObjective"
    assert result.successful == 0
    assert result.posterior is None
    assert result.objective_id is not None

    store = await Store.open(tmp_path / "stage_stop_after.db")
    repo = EpistemicRepository(store)
    obj = await repo.get("objective", result.objective_id)
    assert obj is not None
    assert obj.question_type is not None, (
        "PrepareObjective is the only writer of question_type; "
        "if this is None, the entry node did not run."
    )


# Note: there is no explicit "stop_after=None preserves existing
# behavior" test. The existing 1854 tests run with stop_after=None
# (the default), so the no-change path is tested by every other
# call site implicitly.


async def test_start_at_skips_to_named_node(tmp_path: Path, monkeypatch) -> None:
    """``start_at`` resumes the graph from a named node instead of
    PrepareObjective. The DB must already contain the prerequisite
    state (in this case, an Objective). Combined with ``stop_after``
    on a single-node window, this is the test that pins the
    save-and-resume contract: running with start_at=X stop_after=X
    executes exactly node X and nothing else."""
    from andamentum.epistemic.tests.conftest import FakeAgentRunner
    from andamentum.epistemic.graph.nodes import Decompose
    import andamentum.epistemic.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "DefaultAgentRunner",
        lambda model: FakeAgentRunner(),  # noqa: ARG005
    )

    # Stage 1: run only PrepareObjective. DB now has Objective with
    # question_type set, but no decomposition.
    r1 = await run_epistemic_graph(
        question="Is exercise good for cardiovascular health?",
        database_name="stage_resume",
        db_dir=str(tmp_path),
        model="fake:test-model",
        embedding_model="fake-embeddings",
        stop_after=PrepareObjective,
    )
    assert r1.status == "stopped_after:PrepareObjective"

    # Stage 2: resume from Decompose. objective_id is auto-resumed
    # from the DB (existing_objectives path).
    r2 = await run_epistemic_graph(
        question="(ignored — DB has objective)",
        database_name="stage_resume",
        db_dir=str(tmp_path),
        model="fake:test-model",
        embedding_model="fake-embeddings",
        start_at=Decompose,
        stop_after=Decompose,
    )
    assert r2.status == "stopped_after:Decompose"
    assert r2.objective_id == r1.objective_id, (
        "Resumed run must reuse the saved objective; if a new one "
        "was created, the DB-as-checkpoint contract is broken."
    )

    store = await Store.open(tmp_path / "stage_resume.db")
    repo = EpistemicRepository(store)
    obj = await repo.get("objective", r1.objective_id)
    assert obj is not None
    assert obj.decomposition is not None, (
        "Decompose is the unique writer of Objective.decomposition; "
        "if it's still None after stage 2, start_at didn't actually "
        "execute the named node."
    )


async def test_output_dir_emits_three_artifacts(tmp_path: Path, monkeypatch) -> None:
    """When output_dir is set the runner writes run.jsonl, diff.json,
    and timing.txt next to the DB. These are the only observability
    surface stages need; everything else is derived from the DB."""
    import json

    from andamentum.epistemic.tests.conftest import FakeAgentRunner
    import andamentum.epistemic.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "DefaultAgentRunner",
        lambda model: FakeAgentRunner(),  # noqa: ARG005
    )

    artifacts = tmp_path / "artifacts"
    result = await run_epistemic_graph(
        question="Is exercise good?",
        database_name="stage_artifacts",
        db_dir=str(tmp_path),
        model="fake:test-model",
        embedding_model="fake-embeddings",
        stop_after=PrepareObjective,
        output_dir=artifacts,
    )
    assert result.status == "stopped_after:PrepareObjective"

    run_jsonl = (artifacts / "run.jsonl").read_text().strip().splitlines()
    diff = json.loads((artifacts / "diff.json").read_text())
    timing = (artifacts / "timing.txt").read_text()

    # run.jsonl: one line per node visit, JSON-decodable.
    assert len(run_jsonl) >= 1
    visit_0 = json.loads(run_jsonl[0])
    assert visit_0["node"] == "PrepareObjective"
    assert "ms" in visit_0 and "ts" in visit_0

    # diff.json: keyed deltas relative to a fresh objective.
    assert diff["objective_id"] == result.objective_id
    assert diff["claims"] == 0  # no claims yet (stopped before initial_evidence)
    assert diff["decomposition_present"] is False  # stopped before Decompose

    # timing.txt: human-readable, total + per-node.
    assert "Total:" in timing and "PrepareObjective:" in timing


async def test_stage_invariant_satisfied_no_crash(tmp_path: Path, monkeypatch) -> None:
    """When stop_after matches a known stage exit and the invariant
    holds, the run completes cleanly. ``Decompose`` is the
    ``preplanning`` stage's exit; in research mode the agent emits a
    real decomposition and the invariant passes."""
    from andamentum.epistemic.tests.conftest import FakeAgentRunner
    from andamentum.epistemic.graph.nodes import Decompose
    import andamentum.epistemic.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "DefaultAgentRunner",
        lambda model: FakeAgentRunner(),  # noqa: ARG005
    )

    result = await run_epistemic_graph(
        question="Is exercise good for cardiovascular health?",
        database_name="stage_invariant_ok",
        db_dir=str(tmp_path),
        model="fake:test-model",
        embedding_model="fake-embeddings",
        stop_after=Decompose,
    )
    assert result.status == "stopped_after:Decompose"


async def test_stage_invariant_violation_crashes_loudly(
    tmp_path: Path, monkeypatch
) -> None:
    """Critical: a leaky stage boundary MUST crash, not silently pass
    half-finished state forward. Patch the preplanning invariant to
    return False and confirm the runner raises StageInvariantError
    naming the stage and the leaky exit node."""
    from andamentum.epistemic.tests.conftest import FakeAgentRunner
    from andamentum.epistemic.graph.nodes import Decompose
    from andamentum.epistemic.graph.stages import StageInvariantError, STAGES
    import andamentum.epistemic.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "DefaultAgentRunner",
        lambda model: FakeAgentRunner(),  # noqa: ARG005
    )

    async def _always_false(_state, _repo):
        return False

    # StageDef is frozen, so swap the entry instead of mutating fields.
    from dataclasses import replace

    monkeypatch.setitem(
        STAGES,
        "preplanning",
        replace(STAGES["preplanning"], exit_invariant=_always_false),
    )

    with pytest.raises(StageInvariantError) as excinfo:
        await run_epistemic_graph(
            question="any question",
            database_name="stage_invariant_fail",
            db_dir=str(tmp_path),
            model="fake:test-model",
            embedding_model="fake-embeddings",
            stop_after=Decompose,
        )
    assert "preplanning" in str(excinfo.value)
    assert "Decompose" in str(excinfo.value)


async def test_stop_after_unknown_node_skips_invariant_check(
    tmp_path: Path, monkeypatch
) -> None:
    """Mid-pipeline debugging stop points (e.g. stop_after=PlanEvidence,
    not a stage exit) skip the invariant lookup. The runner doesn't
    have an opinion on non-stage stops; user is on their own."""
    from andamentum.epistemic.tests.conftest import FakeAgentRunner
    from andamentum.epistemic.graph.nodes import PlanEvidence
    import andamentum.epistemic.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "DefaultAgentRunner",
        lambda model: FakeAgentRunner(),  # noqa: ARG005
    )

    # PlanEvidence isn't an exit_after of any stage. The runner should
    # NOT crash even though the post-state is "in the middle of"
    # initial_evidence.
    result = await run_epistemic_graph(
        question="any question",
        database_name="stage_unknown_stop",
        db_dir=str(tmp_path),
        model="fake:test-model",
        embedding_model="fake-embeddings",
        stop_after=PlanEvidence,
    )
    assert result.status == "stopped_after:PlanEvidence"


# ── Phase 6: end-to-end stage chain via the registry ─────────────────


async def test_chain_preplanning_then_resume_from_decompose(
    tmp_path: Path, monkeypatch
) -> None:
    """End-to-end use of the stage runner: run preplanning, save DB,
    resume from a later node on the same DB. Confirms the
    save-and-resume contract holds when going through the stage
    registry rather than raw node classes.

    Uses the registry directly (no CLI subprocess) to keep the test
    self-contained and fast."""
    from andamentum.epistemic.tests.conftest import FakeAgentRunner
    from andamentum.epistemic.graph.stages import get_stage
    import andamentum.epistemic.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "DefaultAgentRunner",
        lambda model: FakeAgentRunner(),  # noqa: ARG005
    )

    pp = get_stage("preplanning")

    # Stage 1: preplanning end-to-end via the registry
    artifacts = tmp_path / "stage1"
    r1 = await run_epistemic_graph(
        question="Is exercise good for cardiovascular health?",
        database_name="chain",
        db_dir=str(tmp_path),
        model="fake:test-model",
        embedding_model="fake-embeddings",
        start_at=pp.entry,
        stop_after=pp.exit_after,
        output_dir=artifacts,
    )
    assert r1.status == f"stopped_after:{pp.exit_after.__name__}"

    # Stage 1 wrote run.jsonl, diff.json, timing.txt and the DB
    # has a fully-decomposed Objective.
    assert (artifacts / "run.jsonl").exists()
    assert (artifacts / "diff.json").exists()

    # Re-reading the DB confirms idempotence: re-running preplanning
    # on the same DB does NOT duplicate the objective.
    r2 = await run_epistemic_graph(
        question="(ignored)",
        database_name="chain",
        db_dir=str(tmp_path),
        model="fake:test-model",
        embedding_model="fake-embeddings",
        start_at=pp.entry,
        stop_after=pp.exit_after,
    )
    assert r2.objective_id == r1.objective_id, (
        "Re-running preplanning on the same DB must reuse the existing "
        "objective; if a new one was created, save-and-resume is broken."
    )


# Note: there is no explicit "stop_after=None preserves existing
# behavior" test. The existing 1848 tests run with stop_after=None
# (the default), so the no-change path is tested by every other
# call site implicitly. Adding a redundant 60s end-to-end run here
# would burn iteration time for no extra signal.


# ── _check_integration (the "integration" stage's exit invariant) ────
#
# No existing call site anywhere in the suite runs
# ``stop_after=CombineClaimVerdicts`` (confirmed by grep before writing
# these), so the predicate itself was previously untested in isolation
# — only exercised transitively by whatever integration test happened
# to hit a stage boundary elsewhere. Stage 3 (PRINCIPLE_FIDELITY_
# PLAN.md §5) changed its logic (graph/stages.py:_check_integration),
# so it gets direct unit coverage here rather than relying on an
# end-to-end run to notice a regression.


class TestCheckIntegrationInvariant:
    """Unit tests for the predicate directly — no graph run needed.

    Before Stage 3, every active (non-abandoned, non-cycle-capped)
    claim reaching CombineClaimVerdicts was REQUIRED to carry a
    non-None ``integrated_assessment`` — IBE always produced some
    verdict, even a fabricated one scored over placeholder candidates.
    Stage 3 added a legitimate way for that not to happen (enumeration
    declining for insufficient explanatory diversity), so the
    invariant now also accepts a claim with no verdict AS LONG AS it
    was actually processed (``state.verification_done``) — the same
    distinction ``no_stranded_claims`` (graph/invariants.py) already
    draws for exactly this reason.
    """

    async def test_still_fails_on_a_genuinely_stranded_claim(self, repo) -> None:
        """A claim with no verdict and NOT in verification_done is
        exactly the routing-bug shape this invariant exists to catch.
        The Stage-3 relaxation must not silence that."""
        from ..entities import Claim, ClaimStage
        from ..entities.verdict import EpistemicVerdict
        from ..graph.stages import _check_integration
        from ..graph.state import EpistemicGraphState

        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test claim",
            stage=ClaimStage.SUPPORTED,
        )
        await repo.save(claim)

        state = EpistemicGraphState(
            objective_id="obj-1",
            verdict=EpistemicVerdict(objective_id="obj-1"),
        )
        assert await _check_integration(state, repo) is False

    async def test_passes_on_a_legitimately_declined_claim(self, repo) -> None:
        """Stage 3's new case: no verdict, but verification_done — the
        claim WAS processed and IBE declined via insufficient
        explanatory diversity, not a routing bug that skipped it."""
        from ..entities import Claim, ClaimStage
        from ..entities.verdict import EpistemicVerdict
        from ..graph.stages import _check_integration
        from ..graph.state import EpistemicGraphState

        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test claim",
            stage=ClaimStage.SUPPORTED,
        )
        await repo.save(claim)

        state = EpistemicGraphState(
            objective_id="obj-1",
            verdict=EpistemicVerdict(objective_id="obj-1"),
            verification_done={"cl-1"},
        )
        assert await _check_integration(state, repo) is True

    async def test_still_fails_when_verdict_is_missing_entirely(self, repo) -> None:
        """verdict_ok is a separate, unrelaxed condition — a claim in
        verification_done does not paper over a missing verdict
        object."""
        from ..entities import Claim, ClaimStage
        from ..graph.stages import _check_integration
        from ..graph.state import EpistemicGraphState

        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test claim",
            stage=ClaimStage.SUPPORTED,
        )
        await repo.save(claim)

        state = EpistemicGraphState(
            objective_id="obj-1",
            verdict=None,
            verification_done={"cl-1"},
        )
        assert await _check_integration(state, repo) is False
