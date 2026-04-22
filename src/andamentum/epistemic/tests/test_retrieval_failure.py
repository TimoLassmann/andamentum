"""Tests for retrieval-failure detection and terminal state."""

from __future__ import annotations

from andamentum.epistemic.confidence import PosteriorReport
from andamentum.epistemic.graph.state import EpistemicGraphState


class TestGraphStateRetrievalFields:
    def test_default_values(self) -> None:
        s = EpistemicGraphState()
        assert s.consecutive_empty_extractions == 0
        assert s.retrieval_failed is False


class TestPosteriorReportTerminalState:
    def test_default_terminal_state_is_completed(self) -> None:
        p = PosteriorReport(
            posterior=0.5, log_odds=0, supporting_count=0, contradicting_count=0,
            counting_posterior=0.5, objective_id="x", question_type="predictive",
            explanation="test",
        )
        assert p.terminal_state == "completed"

    def test_terminal_state_accepts_retrieval_failed(self) -> None:
        p = PosteriorReport(
            posterior=0.5, log_odds=0, supporting_count=0, contradicting_count=0,
            counting_posterior=0.5, objective_id="x", question_type="predictive",
            explanation="test", terminal_state="retrieval_failed",
        )
        assert p.terminal_state == "retrieval_failed"
