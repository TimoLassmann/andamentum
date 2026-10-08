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

import pytest

from andamentum.document_store import HybridIndex, HybridRerankSearcher, Store
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
            UnexpectedModelBehavior("ignored the tool schema"),
            label="Document",
            model="m",
        )
        _reraise_if_infrastructure(ValueError("bad field"), label="Document", model="m")


class TestIndexFailuresAreNotAbsorbed:
    """A search signal must not drop out silently: a dead backend raises, a bad
    document is reported, and a search refuses to answer from a partial index."""

    async def test_embedding_backend_failure_raises_from_sync_and_search(self, tmp_path):
        async def dead_backend(texts):
            raise ConnectionError("embedding backend down")

        index = HybridIndex(embedding_model="fake", embed_fn=dead_backend)
        searcher = HybridRerankSearcher(index=index, reranker_model="unused")
        async with await Store.open(
            tmp_path / "s.db", indexes=[index], searcher=searcher
        ) as store:
            await store.put("# T\n\nbody text here", key="a")
            with pytest.raises(ConnectionError, match="backend down"):
                await store.sync()
            with pytest.raises(ConnectionError, match="backend down"):
                await store.search("body", limit=3)

    async def test_a_document_the_index_cannot_process_is_reported_and_blocks_search(
        self, tmp_path, monkeypatch
    ):
        from andamentum.document_store import index as index_module

        def broken_chunker(text, *, max_chars):
            raise ValueError("cannot chunk this")

        monkeypatch.setattr(index_module, "chunk", broken_chunker)
        index = HybridIndex(embedding_model=None)
        searcher = HybridRerankSearcher(index=index, reranker_model="unused")
        async with await Store.open(
            tmp_path / "s.db", indexes=[index], searcher=searcher
        ) as store:
            await store.put("body text here", key="a")
            report = await store.sync()
            assert not report.ok
            [(_, error)] = report.failures["hybrid"]
            assert "cannot chunk this" in error
            with pytest.raises(RuntimeError, match="could not be indexed"):
                await store.search("body", limit=3)
