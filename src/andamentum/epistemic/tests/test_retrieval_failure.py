"""Tests for retrieval-failure detection and terminal state."""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.confidence import PosteriorReport, compute_posterior
from andamentum.epistemic.entities import Evidence, Objective
from andamentum.epistemic.graph.nodes import _update_retrieval_health
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.report_data import (
    ConfidenceScores,
    InvestigationStats,
    ReportData,
)
from andamentum.epistemic.repository import EpistemicRepository
from andamentum.epistemic.typeset_report import build_typeset_report


class TestGraphStateRetrievalFields:
    def test_default_values(self) -> None:
        s = EpistemicGraphState()
        assert s.consecutive_empty_extractions == 0
        assert s.retrieval_failed is False


class TestPosteriorReportTerminalState:
    def test_default_terminal_state_is_completed(self) -> None:
        p = PosteriorReport(
            posterior=0.5,
            log_odds=0,
            supporting_count=0,
            contradicting_count=0,
            counting_posterior=0.5,
            objective_id="x",
            question_type="predictive",
            explanation="test",
        )
        assert p.terminal_state == "completed"

    def test_terminal_state_accepts_retrieval_failed(self) -> None:
        p = PosteriorReport(
            posterior=0.5,
            log_odds=0,
            supporting_count=0,
            contradicting_count=0,
            counting_posterior=0.5,
            objective_id="x",
            question_type="predictive",
            explanation="test",
            terminal_state="retrieval_failed",
        )
        assert p.terminal_state == "retrieval_failed"


async def _make_repo(tmp_path: Path) -> EpistemicRepository:
    store = DocumentStore.for_database("test", db_dir=tmp_path)
    await store.initialize()
    return EpistemicRepository(store)


async def _make_obj(repo: EpistemicRepository) -> Objective:
    obj = Objective(description="x", question_type="predictive")
    obj.objective_id = obj.entity_id  # self-referential convention
    await repo.save(obj)
    return obj


class TestRetrievalHealthUpdater:
    async def test_empty_extractions_increment(self, tmp_path: Path) -> None:
        repo = await _make_repo(tmp_path)
        obj = await _make_obj(repo)
        state = EpistemicGraphState()
        for _ in range(2):
            ev = Evidence(
                source_type="web",
                source_ref="https://ex.com/x",
                extracted_content="",
                objective_id=obj.entity_id,
            )
            await repo.save(ev)
            _update_retrieval_health(state, ev)
        assert state.consecutive_empty_extractions == 2
        assert state.retrieval_failed is False

    async def test_non_empty_resets_counter(self, tmp_path: Path) -> None:
        repo = await _make_repo(tmp_path)
        obj = await _make_obj(repo)
        state = EpistemicGraphState()
        state.consecutive_empty_extractions = 2
        ev = Evidence(
            source_type="web",
            source_ref="https://ex.com/y",
            extracted_content="real content here",
            objective_id=obj.entity_id,
        )
        await repo.save(ev)
        _update_retrieval_health(state, ev)
        assert state.consecutive_empty_extractions == 0
        assert state.retrieval_failed is False

    async def test_threshold_flips_flag(self, tmp_path: Path) -> None:
        repo = await _make_repo(tmp_path)
        obj = await _make_obj(repo)
        state = EpistemicGraphState()
        for _ in range(3):
            ev = Evidence(
                source_type="web",
                source_ref="https://ex.com/z",
                extracted_content="",
                objective_id=obj.entity_id,
            )
            await repo.save(ev)
            _update_retrieval_health(state, ev)
        assert state.consecutive_empty_extractions == 3
        assert state.retrieval_failed is True

    async def test_non_empty_does_not_unflip_retrieval_failed(
        self, tmp_path: Path
    ) -> None:
        # Once retrieval_failed flips True it stays True even if later
        # extractions succeed. The flag represents a terminal health
        # classification for the run, not a real-time gauge.
        repo = await _make_repo(tmp_path)
        obj = await _make_obj(repo)
        state = EpistemicGraphState()
        state.retrieval_failed = True
        state.consecutive_empty_extractions = 3

        ev = Evidence(
            source_type="web",
            source_ref="https://ex.com/late",
            extracted_content="something real",
            objective_id=obj.entity_id,
        )
        await repo.save(ev)
        _update_retrieval_health(state, ev)

        # counter resets but flag stays — a late recovery shouldn't erase
        # the fact that we've already been flagged as a failed-retrieval run.
        assert state.consecutive_empty_extractions == 0
        assert state.retrieval_failed is True


class TestPipelineResultExposesRetrievalFailed:
    def test_default_is_false(self) -> None:
        from andamentum.epistemic.operations_runner import PipelineResult

        r = PipelineResult(
            objective_id="x",
            iterations=0,
            successful=0,
            failed=0,
            status="done",
        )
        assert r.retrieval_failed is False

    def test_constructor_accepts_flag(self) -> None:
        from andamentum.epistemic.operations_runner import PipelineResult

        r = PipelineResult(
            objective_id="x",
            iterations=0,
            successful=0,
            failed=0,
            status="done",
            retrieval_failed=True,
        )
        assert r.retrieval_failed is True


class TestComputePosteriorRetrievalFailed:
    async def test_emits_terminal_state_report(self, tmp_path: Path) -> None:
        repo = await _make_repo(tmp_path)
        obj = await _make_obj(repo)

        posterior = await compute_posterior(
            repo, objective_id=obj.entity_id, retrieval_failed=True
        )
        assert posterior is not None
        assert posterior.terminal_state == "retrieval_failed"
        assert posterior.posterior == 0.5
        assert posterior.supporting_count == 0
        assert posterior.contradicting_count == 0
        assert "Retrieval failed" in posterior.explanation

    async def test_normal_path_unaffected(self, tmp_path: Path) -> None:
        # When retrieval_failed=False (default), compute_posterior runs the
        # normal path. With no claims, it still returns a report (counting
        # posterior defaults to 0.5 from log_odds 0).
        repo = await _make_repo(tmp_path)
        obj = await _make_obj(repo)

        posterior = await compute_posterior(repo, objective_id=obj.entity_id)
        assert posterior is not None
        assert posterior.terminal_state == "completed"

    async def test_retrieval_failed_with_integrated_assessment_uses_verdict(
        self, tmp_path: Path
    ) -> None:
        """SciFact case 54 v14 shape: retrieval_failed=True but the claim
        already has an integrated_assessment="contradicts" at high
        confidence. Pre-fix: posterior=0.5 (verdict discarded). Post-fix:
        posterior reflects the verdict with retrieval-failed penalty."""
        from andamentum.epistemic.entities import Claim
        from andamentum.epistemic.entities.claim import ClaimStage

        repo = await _make_repo(tmp_path)
        obj = Objective(
            description="AMPK activation increases inflammation-related fibrosis.",
            clarified_question="AMPK activation increases inflammation-related fibrosis.",
            question_type="verificatory",
            claim_to_verify="AMPK activation increases inflammation-related fibrosis.",
        )
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        claim = Claim(
            objective_id=obj.entity_id,
            statement="AMPK activation increases inflammation-related fibrosis.",
            scope="lung tissue",
            stage=ClaimStage.SUPPORTED,
            integrated_assessment="contradicts",
            integrated_confidence=0.857,
        )
        await repo.save(claim)

        report = await compute_posterior(
            repo, objective_id=obj.entity_id, retrieval_failed=True
        )
        assert report is not None
        # contradicts at 0.857, no cap, with retrieval-failed pull (0.7):
        # raw posterior = 0.5 - 0.857/2 = 0.0715.
        # Pulled toward neutral: 0.5 + (0.0715 - 0.5)*0.7 = 0.200.
        assert 0.15 < report.posterior < 0.25
        assert report.terminal_state == "retrieval_failed"
        assert report.integration_verdict == "contradicts"
        assert "Retrieval failed" in report.explanation
        assert "0.7" in report.explanation  # penalty value surfaced

    async def test_retrieval_failed_no_ia_suspends_with_retrieval_terminal(
        self, tmp_path: Path
    ) -> None:
        """retrieval_failed=True, claim has no integrated_assessment.
        Under the no-certified-verdict gate (added 2026-05-05), counting
        on uncertified evidence is structurally unsafe — both signals
        (writer prose, posterior number) suspend rather than commit.
        Terminal state is ``retrieval_failed`` (the more specific
        diagnostic) when the retrieval flag is set; would be
        ``oscillation_detected`` otherwise."""
        from andamentum.epistemic.entities import Claim
        from andamentum.epistemic.entities.claim import ClaimStage

        repo = await _make_repo(tmp_path)
        obj = Objective(
            description="Q",
            clarified_question="Q",
            question_type="verificatory",
            claim_to_verify="claim X is true",
        )
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        claim = Claim(
            objective_id=obj.entity_id,
            statement="claim X is true",
            scope="scope",
            stage=ClaimStage.HYPOTHESIS,
        )
        await repo.save(claim)

        ev_ids: list[str] = []
        for i in range(10):
            ev = Evidence(
                objective_id=obj.entity_id,
                source_type="pubmed",
                source_ref=f"https://pubmed/sup_{i}",
                extracted_content="content",
                extracted=True,
                support_judgment="supports",
            )
            await repo.save(ev)
            ev_ids.append(ev.entity_id)
        claim.evidence_ids = ev_ids
        claim.evidence_count = len(ev_ids)
        await repo.save(claim)

        report = await compute_posterior(
            repo, objective_id=obj.entity_id, retrieval_failed=True
        )
        assert report is not None
        # No-certified-verdict gate fires; posterior suspends.
        # retrieval_failed wins precedence over the default
        # oscillation_detected terminal because it's more specific.
        assert report.posterior == 0.5
        assert report.terminal_state == "retrieval_failed"
        assert report.mode == "counting_only"
        # Diagnostic counts still exposed.
        assert report.supporting_count > report.contradicting_count
        assert report.counting_posterior > 0.5
        assert "Retrieval failed" in report.explanation
        assert "IBE certification" in report.explanation


class TestConfidenceScoresTerminalState:
    def test_default_terminal_state_is_completed(self) -> None:
        cs = ConfidenceScores()
        assert cs.terminal_state == "completed"

    def test_accepts_retrieval_failed(self) -> None:
        cs = ConfidenceScores(terminal_state="retrieval_failed")
        assert cs.terminal_state == "retrieval_failed"


def _make_report_data(terminal_state: str = "completed") -> ReportData:
    from datetime import datetime

    return ReportData(
        research_question="Test?",
        clarified_question="Test?",
        investigation_date=datetime.now(),
        model_used="test:model",
        database_name="test_db",
        direct_answer="Answer text.",
        question_type="predictive",
        verdict="Test verdict",
        stats=InvestigationStats(total_claims=0, total_evidence=0),
        confidence_scores=ConfidenceScores(
            posterior=0.5,
            posterior_supporting=0,
            posterior_contradicting=0,
            posterior_question_type="predictive",
            terminal_state=terminal_state,
        ),
    )


class TestTypesetReportRetrievalFailed:
    def test_emits_warning_callout_when_retrieval_failed(self) -> None:
        data = _make_report_data(terminal_state="retrieval_failed")
        atoms = build_typeset_report(data)
        callout_texts = [
            str(a.get("content", "")) for a in atoms if a.get("kind") == "callout"
        ]
        assert any("Retrieval failed" in t for t in callout_texts)
        # Should NOT include the P(YES) interpretation when retrieval failed.
        assert not any("P(YES)" in t for t in callout_texts)

    def test_emits_p_yes_when_normal_completion(self) -> None:
        data = _make_report_data(terminal_state="completed")
        atoms = build_typeset_report(data)
        callout_texts = [
            str(a.get("content", "")) for a in atoms if a.get("kind") == "callout"
        ]
        assert any("P(YES)" in t for t in callout_texts)
        assert not any("Retrieval failed" in t for t in callout_texts)

    def test_emits_warning_callout_when_oscillation_detected(self) -> None:
        # No-certified-verdict gate fires; renderer must surface the
        # suspension and refuse to emit a directional P(YES).
        data = _make_report_data(terminal_state="oscillation_detected")
        atoms = build_typeset_report(data)
        callout_texts = [
            str(a.get("content", "")) for a in atoms if a.get("kind") == "callout"
        ]
        assert any("No certified verdict" in t for t in callout_texts)
        assert not any("P(YES)" in t for t in callout_texts)

    def test_unknown_terminal_state_fails_loud(self) -> None:
        # Defensive: any future terminal_state added to confidence.py
        # without a matching renderer branch must NOT silently fall
        # through to a P(YES) callout. Surfaces the raw state name
        # instead.
        data = _make_report_data(terminal_state="some_future_state")
        atoms = build_typeset_report(data)
        callout_texts = [
            str(a.get("content", "")) for a in atoms if a.get("kind") == "callout"
        ]
        assert any("some_future_state" in t for t in callout_texts)
        assert not any("P(YES)" in t for t in callout_texts)
