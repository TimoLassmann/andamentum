"""Tests for retrieval-failure detection and terminal state."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from andamentum.document_store import DocumentStore
from andamentum.epistemic.confidence import PosteriorReport, compute_posterior
from andamentum.epistemic.entities import Evidence, Objective
from andamentum.epistemic.graph.nodes import _evaluate_retrieval_health
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.report_data import (
    ConfidenceScores,
    InvestigationStats,
    ReportData,
)
from andamentum.epistemic.audit_report import build_audit_report
from andamentum.epistemic.repository import EpistemicRepository


class TestGraphStateRetrievalFields:
    def test_default_values(self) -> None:
        s = EpistemicGraphState()
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


class TestEvaluateRetrievalHealth:
    """End-of-gathering retrieval-health semantics (replaces the old
    consecutive-empties counter on 2026-05-08).

    Principle: retrieval has failed when *zero* evidence pieces with
    content exist for the objective at end-of-gathering. Off-topic
    providers returning empty is expected and routine; only the total
    yield matters. Invariant under provider count and execution order.
    """

    async def test_zero_evidence_flips_flag_true(self, tmp_path: Path) -> None:
        repo = await _make_repo(tmp_path)
        obj = await _make_obj(repo)
        state = EpistemicGraphState(objective_id=obj.entity_id)
        deps = SimpleNamespace(repo=repo)
        await _evaluate_retrieval_health(state, deps)  # type: ignore[arg-type]
        assert state.retrieval_failed is True

    async def test_any_non_empty_evidence_keeps_flag_false(
        self, tmp_path: Path
    ) -> None:
        """A single piece of real content is sufficient — retrieval has
        not failed. Counterexample to the old counter, which could trip
        on transient empty clusters even when plenty of real content
        was eventually found."""
        repo = await _make_repo(tmp_path)
        obj = await _make_obj(repo)
        state = EpistemicGraphState(objective_id=obj.entity_id)
        deps = SimpleNamespace(repo=repo)

        # Many empty stubs + one with content — retrieval did NOT fail.
        for i in range(5):
            await repo.save(
                Evidence(
                    source_type="off_topic_provider",
                    source_ref=f"https://ex.com/empty/{i}",
                    extracted_content="",
                    objective_id=obj.entity_id,
                )
            )
        await repo.save(
            Evidence(
                source_type="pubmed",
                source_ref="PMID:12345",
                extracted_content="Real abstract content from the on-topic provider.",
                objective_id=obj.entity_id,
            )
        )

        await _evaluate_retrieval_health(state, deps)  # type: ignore[arg-type]
        assert state.retrieval_failed is False

    async def test_all_empty_evidence_flips_flag_true(self, tmp_path: Path) -> None:
        repo = await _make_repo(tmp_path)
        obj = await _make_obj(repo)
        state = EpistemicGraphState(objective_id=obj.entity_id)
        deps = SimpleNamespace(repo=repo)

        for i in range(5):
            await repo.save(
                Evidence(
                    source_type="provider",
                    source_ref=f"https://ex.com/empty/{i}",
                    extracted_content="",
                    objective_id=obj.entity_id,
                )
            )

        await _evaluate_retrieval_health(state, deps)  # type: ignore[arg-type]
        assert state.retrieval_failed is True

    async def test_non_sticky_recovery_unflips_flag(self, tmp_path: Path) -> None:
        """The new check is non-sticky: if a late extraction sweep
        produces non-empty evidence, the flag flips back to False.
        Counter to the old "sticky once tripped" behaviour, which kept
        the flag True even after content was eventually found."""
        repo = await _make_repo(tmp_path)
        obj = await _make_obj(repo)
        state = EpistemicGraphState(objective_id=obj.entity_id, retrieval_failed=True)
        deps = SimpleNamespace(repo=repo)

        await repo.save(
            Evidence(
                source_type="pubmed",
                source_ref="PMID:99999",
                extracted_content="Late but real content.",
                objective_id=obj.entity_id,
            )
        )

        await _evaluate_retrieval_health(state, deps)  # type: ignore[arg-type]
        # Real content was found — retrieval did NOT fail, regardless
        # of any prior state.
        assert state.retrieval_failed is False

    async def test_invariant_under_off_topic_provider_count(
        self, tmp_path: Path
    ) -> None:
        """Adding more off-topic providers (each returning empty) does
        NOT trip the flag, because the on-topic providers still
        produced content. This is the regression test for the bug we
        actually fixed: the old counter would trip on any 3 consecutive
        empties regardless of total yield, so adding off-topic
        providers made spurious trips more likely."""
        repo = await _make_repo(tmp_path)
        obj = await _make_obj(repo)
        state = EpistemicGraphState(objective_id=obj.entity_id)
        deps = SimpleNamespace(repo=repo)

        # Simulate verify-mode: many providers, most off-topic, a
        # couple with real content. Order doesn't matter under the
        # new semantics.
        for provider in [
            "chembl",
            "monarch",
            "clinicaltrials",
            "open_targets",
            "arxiv",  # five off-topic for an immunology question
        ]:
            await repo.save(
                Evidence(
                    source_type=provider,
                    source_ref=f"https://ex.com/{provider}/empty",
                    extracted_content="",
                    objective_id=obj.entity_id,
                )
            )
        for provider in ["pubmed", "openalex", "europepmc"]:
            await repo.save(
                Evidence(
                    source_type=provider,
                    source_ref=f"PMID:{provider}_12345",
                    extracted_content=f"Real {provider} abstract content.",
                    objective_id=obj.entity_id,
                )
            )

        await _evaluate_retrieval_health(state, deps)  # type: ignore[arg-type]
        # 5 empties + 3 with content → not a retrieval failure.
        assert state.retrieval_failed is False


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


class TestAuditReportTerminalState:
    """Confidence body for the Q&A panel handles every terminal_state
    branch from ``confidence.py``: completed → directional posterior;
    retrieval_failed / oscillation_detected → named warnings without
    posterior; unknown → defensive surface of the raw state name.

    These tests are the v2 equivalent of the v1 typeset_report tests —
    same invariant (no silent fall-through to a posterior interpretation
    when the inquiry didn't complete), different renderer surface."""

    @staticmethod
    def _confidence_body_text(data: ReportData) -> str:
        """Pull the "How confident are we?" row body out of the rendered
        Q&A items atom."""
        atoms = build_audit_report(data)
        items_atoms = [a for a in atoms if a.get("kind") == "items"]
        assert items_atoms, "Q&A items panel missing"
        entries = items_atoms[0]["entries"]
        confidence_entry = next(
            (e for e in entries if e["label"] == "How confident are we?"),
            None,
        )
        assert confidence_entry is not None, "How confident? row missing"
        return str(confidence_entry["body"])

    def test_completed_terminal_renders_directional_posterior(self) -> None:
        body = self._confidence_body_text(_make_report_data(terminal_state="completed"))
        # Directional probability framing — not "P(YES)".
        assert "Probability the claim is true" in body
        # Verdict label appears.
        assert any(
            v in body
            for v in ("Confirmed", "Refuted", "Inconclusive", "Insufficient evidence")
        )
        # No leftover v1 phrasing.
        assert "P(YES)" not in body
        assert "Retrieval failed" not in body

    def test_retrieval_failed_suppresses_posterior(self) -> None:
        body = self._confidence_body_text(
            _make_report_data(terminal_state="retrieval_failed")
        )
        assert "retrieval failed" in body.lower()
        assert "No posterior" in body
        # Must not show a directional probability interpretation.
        assert "Probability the claim is true" not in body
        assert "P(YES)" not in body

    def test_oscillation_detected_suppresses_posterior(self) -> None:
        body = self._confidence_body_text(
            _make_report_data(terminal_state="oscillation_detected")
        )
        assert "IBE-certified verdict" in body
        assert "No posterior" in body
        assert "Probability the claim is true" not in body
        assert "P(YES)" not in body

    def test_unknown_terminal_state_surfaces_raw_state(self) -> None:
        """Defensive: any future terminal_state added to confidence.py
        without a matching renderer branch must NOT silently fall
        through to a directional posterior — surface the raw state
        name instead so the gap is visible."""
        body = self._confidence_body_text(
            _make_report_data(terminal_state="some_future_state")
        )
        assert "some_future_state" in body
        assert "Probability the claim is true" not in body
        assert "P(YES)" not in body
