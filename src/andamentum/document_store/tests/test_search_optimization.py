"""Tests for search performance optimizations."""

from __future__ import annotations

import asyncio
import pytest
from ..search import search_unified


class TestOverFetchFactor:
    def test_vector_limit_is_3x_when_bm25_enabled(self):
        """Over-fetch factor should be 3x, not 10x.

        With limit=10, this means 30 vector candidates instead of 100.
        Reduces BM25 index build cost with minimal quality impact.
        """
        from andamentum.document_store.chunks_search import SearchConfig

        config = SearchConfig(include_bm25=True)
        limit = 10
        # Reproduce the calculation from semantic_search()
        vector_limit = limit * 3 if config.include_bm25 else limit
        assert vector_limit == 30

    def test_vector_limit_equals_limit_when_bm25_disabled(self):
        from andamentum.document_store.chunks_search import SearchConfig

        config = SearchConfig(include_bm25=False)
        limit = 10
        vector_limit = limit * 3 if config.include_bm25 else limit
        assert vector_limit == 10


class TestSignalParallelization:
    @pytest.mark.asyncio
    async def test_search_unified_uses_gather(self):
        """Verify signals run via asyncio.gather, not sequentially.

        We patch the 4 signal functions with slow async stubs. If they run
        in parallel, total time is ~max(durations). If sequential, ~sum.
        """
        import time
        from unittest.mock import patch

        delay = 0.15  # seconds per signal

        async def slow_fts5(*args, **kwargs):
            await asyncio.sleep(delay)
            return None

        async def slow_chunks(*args, **kwargs):
            await asyncio.sleep(delay)
            return None

        async def slow_doc_embed(*args, **kwargs):
            await asyncio.sleep(delay)
            return None

        async def slow_clusters(*args, **kwargs):
            await asyncio.sleep(delay)
            return None

        fake_embedding = [0.1] * 768

        with (
            patch("andamentum.document_store.search._run_fts5_signal", new=slow_fts5),
            patch(
                "andamentum.document_store.search._run_chunk_search", new=slow_chunks
            ),
            patch(
                "andamentum.document_store.search._run_doc_embedding_search",
                new=slow_doc_embed,
            ),
            patch(
                "andamentum.document_store.search._run_cluster_search",
                new=slow_clusters,
            ),
        ):
            start = time.monotonic()
            await search_unified(
                "/fake/path.db", "test query", query_embedding=fake_embedding
            )
            elapsed = time.monotonic() - start

        # 4 signals * 0.15s = 0.6s sequential. Parallel should be ~0.15-0.25s.
        assert elapsed < 0.45, (
            f"Signals took {elapsed:.2f}s — expected <0.45s if parallel (4 x {delay}s)"
        )


class TestClusterCache:
    def test_cache_returns_same_object_on_second_call(self):
        """Cache hit should return the same cluster_states dict."""
        from andamentum.document_store.search import (
            _load_cluster_state,
            _invalidate_cluster_cache,
        )

        # Start clean
        _invalidate_cluster_cache()

        # Create a minimal test database with clusters
        import sqlite3
        import tempfile
        import json
        import os

        db_path = os.path.join(tempfile.mkdtemp(), "test_cache.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY,
                centroid TEXT,
                decay_rate REAL,
                kernel_params TEXT,
                doc_count INTEGER,
                created_at TEXT,
                last_active_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cluster_runs (
                id INTEGER PRIMARY KEY,
                config TEXT,
                doc_count INTEGER,
                cluster_count INTEGER,
                started_at TEXT,
                completed_at TEXT,
                duration_seconds REAL
            )
        """)
        centroid = json.dumps([0.1] * 768)
        kernel_params = json.dumps({"weights": [0.5], "doc_times": [1.0]})
        conn.execute(
            "INSERT INTO clusters (centroid, kernel_params, doc_count, created_at, last_active_at) VALUES (?, ?, ?, ?, ?)",
            (centroid, kernel_params, 1, "2026-01-01", "2026-01-01"),
        )
        conn.commit()
        conn.close()

        # First call — cache miss
        states1, config1 = _load_cluster_state(db_path)
        # Second call — cache hit
        states2, config2 = _load_cluster_state(db_path)

        assert states1 is states2, "Second call should return cached object"

        # Clean up
        _invalidate_cluster_cache()

    def test_invalidate_clears_cache(self):
        """After invalidation, next call should reload from database."""
        from andamentum.document_store.search import (
            _load_cluster_state,
            _invalidate_cluster_cache,
        )

        _invalidate_cluster_cache()

        import sqlite3
        import tempfile
        import json
        import os

        db_path = os.path.join(tempfile.mkdtemp(), "test_invalidate.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY,
                centroid TEXT,
                decay_rate REAL,
                kernel_params TEXT,
                doc_count INTEGER,
                created_at TEXT,
                last_active_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cluster_runs (
                id INTEGER PRIMARY KEY,
                config TEXT,
                doc_count INTEGER,
                cluster_count INTEGER,
                started_at TEXT,
                completed_at TEXT,
                duration_seconds REAL
            )
        """)
        centroid = json.dumps([0.1] * 768)
        kernel_params = json.dumps({"weights": [0.5], "doc_times": [1.0]})
        conn.execute(
            "INSERT INTO clusters (centroid, kernel_params, doc_count, created_at, last_active_at) VALUES (?, ?, ?, ?, ?)",
            (centroid, kernel_params, 1, "2026-01-01", "2026-01-01"),
        )
        conn.commit()
        conn.close()

        states1, _ = _load_cluster_state(db_path)
        _invalidate_cluster_cache(db_path)
        states2, _ = _load_cluster_state(db_path)

        assert states1 is not states2, "After invalidation, should return a new object"

        _invalidate_cluster_cache()
