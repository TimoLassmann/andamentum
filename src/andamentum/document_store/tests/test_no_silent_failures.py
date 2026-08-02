"""Regression tests: infrastructure failures must be loud, not degraded.

Three sites used to absorb an unreachable backend into a "successful" result:

* ``extraction.py`` returned empty metadata defaults, so an ingest run against a
  misconfigured Ollama reported every document as fully enriched while writing
  no LLM fields at all. Found by the deferred-ingestion validation experiment,
  which had to build a pre-flight gate around it.
* ``_embed_doc_level`` logged at INFO and blamed content size for *every*
  exception, so a broken embedding backend could take one of the four RRF search
  signals dark without a warning.
* the search signals logged at DEBUG, which is how a malformed FTS5 MATCH query
  hid: keyword ranking silently vanished from the fusion.

The rule these pin: a model producing poor *output* may be degraded; a backend
that cannot be reached may not.
"""

from __future__ import annotations

import logging

import pytest

from andamentum.document_store.extraction import (
    ExtractionUnavailable,
    _reraise_if_infrastructure,
    extract_document_metadata,
)


class TestInfrastructureErrorsRaise:
    """Configuration / transport / provider failures must not become defaults."""

    async def test_missing_ollama_base_url_raises(self, monkeypatch):
        # pydantic-ai's Ollama provider raises UserError at agent CONSTRUCTION
        # when OLLAMA_BASE_URL is unset — before any request is made.
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        with pytest.raises(ExtractionUnavailable, match="cannot reach the model"):
            await extract_document_metadata("some text", model="ollama:nope")

    async def test_error_message_names_the_cause_and_the_fix(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        with pytest.raises(ExtractionUnavailable) as exc:
            await extract_document_metadata("x", model="ollama:m")
        msg = str(exc.value)
        assert "NOT degraded into empty metadata" in msg
        assert "OLLAMA_BASE_URL" in msg
        # The original provider error is preserved for debugging.
        assert exc.value.__cause__ is not None

    def test_connection_errors_are_infrastructure(self):
        import httpx

        for err in (
            httpx.ConnectError("refused"),
            httpx.ReadTimeout("slow"),
            ConnectionError("down"),
        ):
            with pytest.raises(ExtractionUnavailable):
                _reraise_if_infrastructure(err, label="Document", model="m")

    def test_model_quality_errors_are_NOT_infrastructure(self):
        """A model producing unusable output is the case the fallback exists for."""
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        # Returns normally => caller proceeds to the prompted-output retry /
        # default-value fallback rather than raising.
        _reraise_if_infrastructure(
            UnexpectedModelBehavior("ignored the tool schema"), label="Document", model="m"
        )
        _reraise_if_infrastructure(ValueError("bad field"), label="Document", model="m")


class TestDegradedSignalsAreVisible:
    """A search signal dropping out must be visible at default log level."""

    async def test_fts5_signal_failure_logs_a_warning(self, caplog):
        from andamentum.document_store.search import _run_fts5_signal

        # A database path that does not exist makes the signal fail.
        with caplog.at_level(logging.WARNING, logger="andamentum.document_store.search"):
            result = await _run_fts5_signal("/nonexistent/nope.db", "query", 5)

        assert result is None  # search still degrades rather than aborting
        assert any(
            "FTS5 keyword signal FAILED" in r.message and r.levelno >= logging.WARNING
            for r in caplog.records
        ), f"expected a WARNING naming the dropped signal, got: {[r.message for r in caplog.records]}"

    async def test_doc_embedding_failure_warns_instead_of_blaming_size(
        self, tmp_path, monkeypatch, caplog
    ):
        """_embed_doc_level used to log INFO 'content too large' for ANY error.

        Drives the REAL _run_phase2 with a document-level embedding that fails
        for a non-size reason, and asserts (a) phase 2 still completes — the
        doc-level signal is genuinely optional — and (b) the failure is visible
        at WARNING rather than mislabelled at INFO.
        """
        from andamentum.document_store import embeddings as embeddings_mod
        from andamentum.document_store import public
        from andamentum.document_store.api import DocumentStore

        store = DocumentStore(database_name="warnprobe", db_dir=str(tmp_path))
        await store.initialize()
        doc_id = await store.register_document(title="t", content="body text here")

        class _PartlyBrokenEmbeddingService:
            """Chunk embeddings work; the doc-level one hits a dead backend."""

            def __init__(self, *a, **k):
                pass

            async def embed_text(self, *a, **k):
                raise ConnectionError("embedding backend down")

            async def embed_batch(self, texts, **k):
                return [[0.0] * 768 for _ in texts]

            async def close(self):
                return None

        monkeypatch.setattr(
            embeddings_mod, "EmbeddingService", _PartlyBrokenEmbeddingService
        )
        monkeypatch.setattr(public, "make_ollama_embedder", lambda **k: None)

        with caplog.at_level(
            logging.WARNING, logger="andamentum.document_store.public"
        ):
            await public._run_phase2(
                store, doc_id, "body text here", "t", "embed-model"
            )

        assert any(
            "Doc-level embedding FAILED" in r.message and r.levelno >= logging.WARNING
            for r in caplog.records
        ), f"expected a WARNING, got: {[(r.levelname, r.message) for r in caplog.records]}"
        # Phase 2 still succeeded: chunks exist, so the document is searchable.
        import sqlite3

        conn = sqlite3.connect(str(store.db_path))
        try:
            n_chunks = conn.execute(
                "SELECT COUNT(*) FROM chunks c JOIN documents d "
                "ON c.document_id = d.id WHERE d.doc_uuid = ?",
                (doc_id,),
            ).fetchone()[0]
        finally:
            conn.close()
        assert n_chunks > 0
