"""Regression tests for the source-agnostic judging contract.

Investigation-time Evidence is persisted with ``extracted=True``
directly via the dispatch helper — it does not go through the per-stub
``ExtractEvidenceOperation`` path. The judging step in
``ExtractNewEvidence`` therefore must not be keyed off any specific
creation path; it must judge Evidence by the invariant the system
actually cares about: claim-linked, content-bearing, unjudged.

The fix has two parts:

1. ``ExtractNewEvidence`` judges by predicate: every Evidence with
   ``depends_on_claim_id`` set, ``extracted_content`` non-empty,
   ``support_judgment is None``, and ``invalidated=False`` is judged
   against its claim — regardless of how the Evidence was created.

2. The ``scrutiny_and_investigation`` stage exit invariant enforces
   the same predicate at the stage boundary. A future creation path
   that bypasses judging fails loudly here rather than silently
   regressing posterior calibration.

These tests pin both.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from andamentum.epistemic.entities import (
    Claim,
    Evidence,
    Objective,
)
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import ExtractNewEvidence
from andamentum.epistemic.graph.stages import (
    StageInvariantError,
    _all_active_claim_evidence_judged,
    _check_scrutiny,
)
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.repository import EpistemicRepository


class _JudgingRunner:
    """Stub runner that returns a fixed verdict for every judge call.

    Mirrors the ``judge_evidence`` agent's output shape; the predicate
    -driven judging loop in ExtractNewEvidence calls it via the
    ``runner.run(agent_name, **kwargs)`` protocol shared with
    DefaultAgentRunner.
    """

    def __init__(self, verdict: str = "supports"):
        self.verdict = verdict
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.model = "stub-model"

    async def run(self, agent_name: str, **kwargs: Any) -> Any:
        self.calls.append((agent_name, kwargs))
        return SimpleNamespace(
            verdict=self.verdict,
            reasoning=f"stub {self.verdict} reasoning",
        )


class _FakeCtx:
    """Minimal stand-in for pydantic-graph GraphRunContext."""

    def __init__(self, state: EpistemicGraphState, deps: EpistemicDeps):
        self.state = state
        self.deps = deps


# ──────────────────────────────────────────────────────────────────────────────
# ExtractNewEvidence — predicate-driven judging
# ──────────────────────────────────────────────────────────────────────────────


class TestPredicateDrivenJudging:
    """``ExtractNewEvidence`` finds unjudged Evidence by predicate, not
    by stub-extras list. It catches Evidence regardless of which path
    created it — the source-agnostic contract."""

    async def test_judges_directly_persisted_evidence(
        self, repo: EpistemicRepository
    ) -> None:
        """Evidence persisted with ``extracted=True`` and
        ``depends_on_claim_id`` (the shape produced by the dispatch
        path) gets judged by ExtractNewEvidence even though it never
        went through ExtractEvidenceOperation."""
        obj = Objective(
            entity_id="obj_direct",
            objective_id="obj_direct",
            description="Test",
            phase="planned",
        )
        await repo.save(obj)

        # Build evidence_ids list in memory before first save so we don't
        # need a follow-up save on the claim (which would collide on the
        # documents.file_path uniqueness constraint used by the store).
        evidence_ids: list[str] = []
        # ``file_path`` uses entity_id[:8] so prefixes must differ in the
        # first 8 chars to avoid a UNIQUE collision on documents.file_path.
        for i, eid in enumerate(["ev_alpha", "ev_betta"]):
            ev = Evidence(
                entity_id=eid,
                objective_id="obj_direct",
                source_type="openalex",
                source_ref=f"doi:test/{i}",
                extracted=True,
                extracted_content=f"content {i}",
                depends_on_claim_id="c_direct",
                quality_score=0.7,
            )
            await repo.save(ev)
            evidence_ids.append(ev.entity_id)

        claim = Claim(
            entity_id="c_direct",
            objective_id="obj_direct",
            statement="Test claim",
            scope="any",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="needs_resolution",
            evidence_ids=evidence_ids,
        )
        await repo.save(claim)

        runner = _JudgingRunner(verdict="supports")
        deps = EpistemicDeps(
            repo=repo,
            agent_runner=runner,
            evidence_gatherer=None,
        )
        state = EpistemicGraphState(objective_id="obj_direct")

        node = ExtractNewEvidence()
        await node.run(_FakeCtx(state, deps))  # type: ignore[arg-type]

        # Both Evidence items now have support_judgment populated.
        for eid in evidence_ids:
            ev = await repo.get("evidence", eid)
            assert isinstance(ev, Evidence)
            assert ev.support_judgment == "supports", (
                f"Evidence {ev.entity_id} should have been judged by "
                "the predicate-driven loop"
            )

    async def test_skips_invalidated_evidence(self, repo: EpistemicRepository) -> None:
        """The judging predicate excludes invalidated Evidence."""
        obj = Objective(
            entity_id="obj_skip",
            objective_id="obj_skip",
            description="Test",
            phase="planned",
        )
        await repo.save(obj)

        ev = Evidence(
            entity_id="ev_invalid",
            objective_id="obj_skip",
            source_type="openalex",
            source_ref="doi:test/1",
            extracted=True,
            extracted_content="content",
            depends_on_claim_id="c_skip",
            invalidated=True,
            quality_score=0.7,
        )
        await repo.save(ev)

        claim = Claim(
            entity_id="c_skip",
            objective_id="obj_skip",
            statement="Test claim",
            scope="any",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="needs_resolution",
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        runner = _JudgingRunner()
        deps = EpistemicDeps(
            repo=repo,
            agent_runner=runner,
            evidence_gatherer=None,
        )
        state = EpistemicGraphState(objective_id="obj_skip")

        node = ExtractNewEvidence()
        await node.run(_FakeCtx(state, deps))  # type: ignore[arg-type]

        ev_after = await repo.get("evidence", "ev_invalid")
        assert isinstance(ev_after, Evidence)
        assert ev_after.support_judgment is None, (
            "Invalidated Evidence should not be judged"
        )

    async def test_skips_already_judged_evidence(
        self, repo: EpistemicRepository
    ) -> None:
        """Idempotent: re-running the node does not re-judge."""
        obj = Objective(
            entity_id="obj_idempotent",
            objective_id="obj_idempotent",
            description="Test",
            phase="planned",
        )
        await repo.save(obj)

        ev = Evidence(
            entity_id="ev_pre_judged",
            objective_id="obj_idempotent",
            source_type="openalex",
            source_ref="doi:test/1",
            extracted=True,
            extracted_content="content",
            depends_on_claim_id="c_idempotent",
            support_judgment="contradicts",  # pre-set
            quality_score=0.7,
        )
        await repo.save(ev)

        claim = Claim(
            entity_id="c_idempotent",
            objective_id="obj_idempotent",
            statement="Test claim",
            scope="any",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="needs_resolution",
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        runner = _JudgingRunner(verdict="supports")  # different verdict
        deps = EpistemicDeps(
            repo=repo,
            agent_runner=runner,
            evidence_gatherer=None,
        )
        state = EpistemicGraphState(objective_id="obj_idempotent")

        node = ExtractNewEvidence()
        await node.run(_FakeCtx(state, deps))  # type: ignore[arg-type]

        ev_after = await repo.get("evidence", "ev_pre_judged")
        assert isinstance(ev_after, Evidence)
        assert ev_after.support_judgment == "contradicts", (
            "Pre-judged Evidence must keep its verdict"
        )
        # The runner was called 0 times (skipped the already-judged item).
        assert len(runner.calls) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Stage invariant — the safety net
# ──────────────────────────────────────────────────────────────────────────────


class TestStageInvariant:
    """``_all_active_claim_evidence_judged`` is the explicit invariant
    that codifies what was previously an implicit contract."""

    async def test_invariant_passes_when_all_judged(
        self, repo: EpistemicRepository
    ) -> None:
        obj = Objective(
            entity_id="obj_pass",
            objective_id="obj_pass",
            description="Test",
            phase="planned",
        )
        await repo.save(obj)
        claim = Claim(
            entity_id="c_pass",
            objective_id="obj_pass",
            statement="Test claim",
            scope="any",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="pass",
        )
        await repo.save(claim)
        ev = Evidence(
            entity_id="ev_judged",
            objective_id="obj_pass",
            source_type="openalex",
            source_ref="doi:test/1",
            extracted=True,
            extracted_content="content",
            depends_on_claim_id="c_pass",
            support_judgment="supports",
        )
        await repo.save(ev)

        state = EpistemicGraphState(objective_id="obj_pass")
        assert await _all_active_claim_evidence_judged(state, repo) is True

    async def test_invariant_fails_when_unjudged_evidence_exists(
        self, repo: EpistemicRepository
    ) -> None:
        """If a non-abandoned claim has content-bearing Evidence without
        ``support_judgment``, the invariant returns False — the stage
        runner will raise StageInvariantError."""
        obj = Objective(
            entity_id="obj_fail",
            objective_id="obj_fail",
            description="Test",
            phase="planned",
        )
        await repo.save(obj)
        claim = Claim(
            entity_id="c_fail",
            objective_id="obj_fail",
            statement="Test claim",
            scope="any",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="pass",
        )
        await repo.save(claim)
        # Unjudged Evidence linked to a non-abandoned claim — the
        # invariant must catch this and the stage must refuse to exit.
        ev = Evidence(
            entity_id="ev_unjudged",
            objective_id="obj_fail",
            source_type="openalex",
            source_ref="doi:test/1",
            extracted=True,
            extracted_content="content",
            depends_on_claim_id="c_fail",
            support_judgment=None,
        )
        await repo.save(ev)

        state = EpistemicGraphState(objective_id="obj_fail")
        assert await _all_active_claim_evidence_judged(state, repo) is False

    async def test_invariant_ignores_abandoned_claim_evidence(
        self, repo: EpistemicRepository
    ) -> None:
        """Evidence linked to an abandoned claim is exempt — abandoned
        claims won't be promoted anyway, and forcing extra LLM calls on
        their evidence wastes budget."""
        obj = Objective(
            entity_id="obj_abandoned",
            objective_id="obj_abandoned",
            description="Test",
            phase="planned",
        )
        await repo.save(obj)
        claim = Claim(
            entity_id="c_abandoned",
            objective_id="obj_abandoned",
            statement="Test claim",
            scope="any",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="fail",
            abandoned=True,
        )
        await repo.save(claim)
        ev = Evidence(
            entity_id="ev_for_abandoned",
            objective_id="obj_abandoned",
            source_type="openalex",
            source_ref="doi:test/1",
            extracted=True,
            extracted_content="content",
            depends_on_claim_id="c_abandoned",
            support_judgment=None,
        )
        await repo.save(ev)

        state = EpistemicGraphState(objective_id="obj_abandoned")
        assert await _all_active_claim_evidence_judged(state, repo) is True

    async def test_scrutiny_check_includes_judging_requirement(
        self, repo: EpistemicRepository
    ) -> None:
        """``_check_scrutiny`` requires BOTH (a) terminal verdicts on
        all active claims AND (b) all their content-bearing Evidence
        judged. A claim with a 'pass' verdict but unjudged Evidence
        fails the compound predicate."""
        obj = Objective(
            entity_id="obj_compound",
            objective_id="obj_compound",
            description="Test",
            phase="planned",
        )
        await repo.save(obj)
        claim = Claim(
            entity_id="c_compound",
            objective_id="obj_compound",
            statement="Test claim",
            scope="any",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="pass",  # terminal verdict ✓
        )
        await repo.save(claim)
        # But the Evidence is unjudged → invariant should fail.
        ev = Evidence(
            entity_id="ev_compound",
            objective_id="obj_compound",
            source_type="openalex",
            source_ref="doi:test/1",
            extracted=True,
            extracted_content="content",
            depends_on_claim_id="c_compound",
            support_judgment=None,
        )
        await repo.save(ev)

        state = EpistemicGraphState(objective_id="obj_compound")
        assert await _check_scrutiny(state, repo) is False


# ──────────────────────────────────────────────────────────────────────────────
# Smoke: StageInvariantError is the right exception type
# ──────────────────────────────────────────────────────────────────────────────


def test_stage_invariant_error_is_runtime_error() -> None:
    """StageInvariantError exists and is a RuntimeError subclass — so
    the stage runner can catch it with the conventional handler."""
    assert issubclass(StageInvariantError, RuntimeError)
    with pytest.raises(StageInvariantError):
        raise StageInvariantError("test")
