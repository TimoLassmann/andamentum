"""Tests for retrieval-failure detection and terminal state."""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.confidence import PosteriorReport
from andamentum.epistemic.entities import Evidence, Objective
from andamentum.epistemic.graph.nodes import _update_retrieval_health
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.repository import EpistemicRepository


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
