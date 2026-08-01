"""Tests for the deferred-ingestion queue: defer, drain, pause, resume, crash.

The load-bearing property under test is that every stage transition is committed
as it completes, so stopping the drain — cooperatively, by budget, or by an
exception — never loses more than the single in-flight document's work.

These tests stub the LLM/embedding layer: the queue mechanics are what is being
verified, not the quality of extraction.
"""

from __future__ import annotations

import pytest

from andamentum.document_store import queue as q
from andamentum.document_store import public
from andamentum.document_store.api import DocumentStore


@pytest.fixture()
async def db(tmp_path, monkeypatch):
    """An initialized store, with the module-level store cache pointed at it."""
    store = DocumentStore(database_name="testq", db_dir=str(tmp_path))
    await store.initialize()
    monkeypatch.setattr(public, "_stores", {"testq": store})
    # Preflight would demand a live Ollama + LLM; the queue mechanics don't.
    monkeypatch.setattr(public, "_preflight", _noop_preflight)
    return store


async def _noop_preflight(*_a, **_kw) -> None:
    return None


def _stub_enrichment(monkeypatch, *, fail_on: set[str] | None = None) -> list[str]:
    """Replace phase 2 + doc metadata extraction. Returns the list of titles seen."""
    seen: list[str] = []
    fail_on = fail_on or set()

    async def _fake_phase2(store, doc_id, content, title, model, embedding_model):
        if title in fail_on:
            raise RuntimeError(f"boom on {title}")
        seen.append(title)
        # A real phase 2 writes chunks; write one so _is_incomplete() is happy.
        await store.store_chunk(doc_id, content, [0.0] * 768, chunk_index=0)

    async def _fake_doc_meta(content, model=None, max_content_chars=3000):
        from andamentum.document_store.metadata_models import DocumentMetadataFields

        return DocumentMetadataFields()

    monkeypatch.setattr(public, "_run_phase2", _fake_phase2)
    monkeypatch.setattr(public, "extract_document_metadata", _fake_doc_meta)
    return seen


class TestDeferMarksWorkWithoutDoingIt:
    async def test_defer_queues_and_skips_llm(self, db, monkeypatch):
        async def _explode(*_a, **_kw):
            raise AssertionError("defer must not call the LLM")

        monkeypatch.setattr(public, "extract_document_metadata", _explode)
        monkeypatch.setattr(public, "_run_phase2", _explode)

        doc_id = await public.ingest("testq", "# Title\n\nbody", process="defer")

        counts = await q.count_by_status(str(db.db_path))
        assert counts[q.PENDING_ENRICH] == 1
        assert counts[q.COMPLETE] == 0
        doc = await db.read(doc_id)
        assert doc is not None and doc.content == "# Title\n\nbody"

    async def test_deferred_doc_is_keyword_searchable_immediately(self, db):
        await public.ingest("testq", "unmistakableword here", process="defer")
        from andamentum.document_store.search import search_fts5

        hits = await search_fts5(str(db.db_path), "unmistakableword", 10)
        assert len(hits) == 1

    async def test_defer_uses_deterministic_first_line_title(self, db):
        doc_id = await public.ingest("testq", "## My Heading\n\nrest", process="defer")
        doc = await db.read(doc_id)
        assert doc is not None and doc.metadata.title == "My Heading"

    async def test_now_requires_models(self, db):
        with pytest.raises(ValueError, match="requires both model"):
            await public.ingest("testq", "x", process="now")


class TestDrain:
    async def test_drain_completes_queued_work(self, db, monkeypatch):
        seen = _stub_enrichment(monkeypatch)
        for i in range(3):
            await public.ingest("testq", f"doc number {i}", process="defer")

        report = await public.process_pending(
            "testq", model="m", embedding_model="e"
        )

        assert report.documents_enriched == 3
        assert report.remaining == 0
        assert report.stopped_early is False
        assert len(seen) == 3
        counts = await q.count_by_status(str(db.db_path))
        assert counts[q.COMPLETE] == 3

    async def test_drain_is_idempotent(self, db, monkeypatch):
        _stub_enrichment(monkeypatch)
        await public.ingest("testq", "only doc", process="defer")
        await public.process_pending("testq", model="m", embedding_model="e")

        again = await public.process_pending("testq", model="m", embedding_model="e")
        assert again.documents_enriched == 0
        assert again.remaining == 0


class TestPauseAndResume:
    async def test_max_docs_stops_early_and_leaves_the_rest_queued(
        self, db, monkeypatch
    ):
        _stub_enrichment(monkeypatch)
        for i in range(5):
            await public.ingest("testq", f"doc {i}", process="defer")

        first = await public.process_pending(
            "testq", model="m", embedding_model="e", max_docs=2
        )
        assert first.documents_enriched == 2
        assert first.stopped_early is True
        assert first.remaining == 3

        second = await public.process_pending("testq", model="m", embedding_model="e")
        assert second.documents_enriched == 3
        assert second.remaining == 0

    async def test_should_continue_pauses_between_documents(self, db, monkeypatch):
        _stub_enrichment(monkeypatch)
        for i in range(4):
            await public.ingest("testq", f"doc {i}", process="defer")

        calls = {"n": 0}

        def should_continue() -> bool:
            calls["n"] += 1
            return calls["n"] <= 2  # allow two documents, then pause

        report = await public.process_pending(
            "testq", model="m", embedding_model="e", should_continue=should_continue
        )
        assert report.documents_enriched == 2
        assert report.stopped_early is True
        assert report.remaining == 2

    async def test_progress_callback_reports_each_document(self, db, monkeypatch):
        _stub_enrichment(monkeypatch)
        for i in range(3):
            await public.ingest("testq", f"doc {i}", process="defer")

        seen: list[tuple[int, int]] = []
        await public.process_pending(
            "testq",
            model="m",
            embedding_model="e",
            on_progress=lambda done, total, title: seen.append((done, total)),
        )
        assert seen == [(1, 3), (2, 3), (3, 3)]


class TestFailureHandling:
    async def test_failure_is_recorded_not_swallowed(self, db, monkeypatch):
        _stub_enrichment(monkeypatch, fail_on={"bad doc"})
        await public.ingest("testq", "bad doc", process="defer")
        await public.ingest("testq", "good doc", process="defer")

        report = await public.process_pending("testq", model="m", embedding_model="e")

        assert report.documents_failed == 1
        assert report.documents_enriched == 1
        assert any("boom" in f for f in report.failures)
        counts = await q.count_by_status(str(db.db_path))
        assert counts[q.FAILED] == 1

    async def test_retry_failed_requeues(self, db, monkeypatch):
        _stub_enrichment(monkeypatch, fail_on={"bad doc"})
        await public.ingest("testq", "bad doc", process="defer")
        await public.process_pending("testq", model="m", embedding_model="e")

        assert await public.retry_failed("testq") == 1
        counts = await q.count_by_status(str(db.db_path))
        assert counts[q.PENDING_ENRICH] == 1

    async def test_crash_mid_document_leaves_it_pending(self, db, monkeypatch):
        """A hard failure must not advance status — the doc stays drainable."""
        _stub_enrichment(monkeypatch, fail_on={"doc a"})
        await public.ingest("testq", "doc a", process="defer")
        await public.process_pending("testq", model="m", embedding_model="e")

        # Now "fix" the transient problem and retry: it completes, no data lost.
        _stub_enrichment(monkeypatch)
        await public.retry_failed("testq")
        report = await public.process_pending("testq", model="m", embedding_model="e")
        assert report.documents_enriched == 1
        assert report.remaining == 0


class TestSourceConversionStage:
    async def test_source_is_queued_without_converting(self, db):
        doc_id = await public.ingest_source("testq", "/tmp/paper.pdf")
        counts = await q.count_by_status(str(db.db_path))
        assert counts[q.PENDING_SOURCE] == 1
        pending = await q.list_pending(str(db.db_path))
        assert pending[0]["source"] == "/tmp/paper.pdf"
        assert pending[0]["doc_id"] == doc_id

    async def test_drain_converts_then_enriches(self, db, monkeypatch):
        seen = _stub_enrichment(monkeypatch)
        await public.ingest_source("testq", "/tmp/paper.pdf")

        calls: list[str] = []

        async def convert(source: str) -> str:
            calls.append(source)
            return "# Converted\n\nreal content"

        report = await public.process_pending(
            "testq", model="m", embedding_model="e", convert_fn=convert
        )

        assert calls == ["/tmp/paper.pdf"]
        assert report.documents_converted == 1
        assert report.documents_enriched == 1
        assert seen == ["Converted"]

    async def test_conversion_is_checkpointed_and_not_repeated(self, db, monkeypatch):
        """The core resumability guarantee: a converted source that fails during
        enrichment is NOT re-converted on the next drain."""
        _stub_enrichment(monkeypatch, fail_on={"Converted"})
        await public.ingest_source("testq", "/tmp/paper.pdf")

        calls: list[str] = []

        async def convert(source: str) -> str:
            calls.append(source)
            return "# Converted\n\nreal content"

        await public.process_pending(
            "testq", model="m", embedding_model="e", convert_fn=convert
        )
        assert calls == ["/tmp/paper.pdf"]  # converted once

        # Retry: enrichment succeeds now. Conversion must NOT run again.
        _stub_enrichment(monkeypatch)
        await public.retry_failed("testq")
        report = await public.process_pending(
            "testq", model="m", embedding_model="e", convert_fn=convert
        )
        assert calls == ["/tmp/paper.pdf"]  # still once — checkpoint honoured
        assert report.documents_converted == 0
        assert report.documents_enriched == 1

    async def test_no_convert_fn_skips_sources_without_failing_them(
        self, db, monkeypatch
    ):
        _stub_enrichment(monkeypatch)
        await public.ingest_source("testq", "/tmp/paper.pdf")

        report = await public.process_pending("testq", model="m", embedding_model="e")

        assert report.documents_skipped == 1
        assert report.documents_failed == 0
        counts = await q.count_by_status(str(db.db_path))
        assert counts[q.PENDING_SOURCE] == 1  # still queued, not lost

    async def test_sources_drain_before_markdown(self, db, monkeypatch):
        """Ordering: conversions first, so a partial run leaves cheap work behind."""
        _stub_enrichment(monkeypatch)
        await public.ingest("testq", "markdown doc", process="defer")
        await public.ingest_source("testq", "/tmp/paper.pdf")

        pending = await q.list_pending(str(db.db_path))
        assert pending[0]["status"] == q.PENDING_SOURCE


class TestAutoRepairDoesNotEatTheQueue:
    async def test_repair_ignores_deliberately_pending_documents(self, db, monkeypatch):
        """The landmine: a search after pausing must not drain the backlog."""
        _stub_enrichment(monkeypatch)
        for i in range(3):
            await public.ingest("testq", f"doc {i}", process="defer")

        report = await public.repair("testq", model="m", embedding_model="e")

        assert report.documents_repaired == 0
        assert report.documents_scanned == 0
        counts = await q.count_by_status(str(db.db_path))
        assert counts[q.PENDING_ENRICH] == 3  # untouched


class TestPendingStatus:
    async def test_counts_each_state(self, db, monkeypatch):
        _stub_enrichment(monkeypatch)
        await public.ingest("testq", "one", process="defer")
        await public.ingest("testq", "two", process="defer")
        await public.ingest_source("testq", "/tmp/x.pdf")

        st = await public.pending_status("testq")
        assert st.pending_enrich == 2
        assert st.pending_source == 1
        assert st.pending == 3
        assert st.complete == 0
