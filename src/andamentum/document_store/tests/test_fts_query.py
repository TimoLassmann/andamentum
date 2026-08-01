"""Regression tests for FTS5 query preparation (identifier discrimination fix).

Pins the bug: a raw query containing notation like ``c.1234G>A`` raised
``fts5: syntax error`` (killing the keyword signal), and could not separate
near-miss identifiers. `prepare_fts_query` phrase-quotes punctuated tokens so
the existing `porter unicode61` index discriminates them, while leaving prose
and deliberate FTS5 power-queries untouched.
"""
from __future__ import annotations

import sqlite3

import pytest

from andamentum.document_store.fts_query import prepare_fts_query


class TestPrepareFtsQuery:
    def test_plain_prose_unchanged(self):
        assert prepare_fts_query("how does mitosis divide") == "how does mitosis divide"

    def test_identifier_phrase_quoted(self):
        assert prepare_fts_query("c.1234G>A") == '"c.1234G>A"'

    def test_mixed_query_quotes_only_identifier(self):
        assert (
            prepare_fts_query("pathogenic variant c.1234G>A in BRCA1")
            == 'pathogenic variant "c.1234G>A" in BRCA1'
        )

    @pytest.mark.parametrize("q", ['"neural networks"', "python AND numpy", "transform*"])
    def test_power_queries_untouched(self, q):
        assert prepare_fts_query(q) == q

    @pytest.mark.parametrize("q", ["NM_000256.3", "10.1038/nature12373", "COVID-19"])
    def test_various_punctuated_tokens_quoted(self, q):
        assert prepare_fts_query(q) == f'"{q}"'

    def test_pure_punctuation_falls_back(self):
        # Degenerate all-punctuation query: no worse than before, not crashing prep.
        assert prepare_fts_query(":: ->") == ":: ->"

    def test_empty(self):
        assert prepare_fts_query("   ") == ""


class TestAgainstRealFts5Index:
    """The prepared query must be syntax-safe AND discriminate near-misses on a
    real porter-unicode61 FTS5 table — the exact configuration the store uses."""

    @pytest.fixture()
    def fts(self):
        conn = sqlite3.connect(":memory:")
        conn.execute(
            'CREATE VIRTUAL TABLE t USING fts5(body, tokenize="porter unicode61")'
        )
        conn.executemany(
            "INSERT INTO t(rowid, body) VALUES (?,?)",
            [
                (1, "A recurrent c.1234G>A substitution confirmed by Sanger sequencing."),
                (2, "A recurrent c.1234G>T substitution confirmed by Sanger sequencing."),
                (3, "A recurrent c.1234G>C substitution confirmed by Sanger sequencing."),
            ],
        )
        return conn

    def _match(self, conn, q):
        return [r[0] for r in conn.execute(
            "SELECT rowid FROM t WHERE t MATCH ? ORDER BY rank", (q,)
        ).fetchall()]

    def test_raw_identifier_crashes(self, fts):
        # Documents the bug: the un-prepared query is not valid FTS5.
        with pytest.raises(sqlite3.OperationalError):
            self._match(fts, "c.1234G>A")

    def test_prepared_identifier_is_syntax_safe_and_exact(self, fts):
        assert self._match(fts, prepare_fts_query("c.1234G>A")) == [1]

    def test_prepared_excludes_near_miss_siblings(self, fts):
        for ident, want in [("c.1234G>A", 1), ("c.1234G>T", 2), ("c.1234G>C", 3)]:
            assert self._match(fts, prepare_fts_query(ident)) == [want]

    def test_prose_query_unaffected(self, fts):
        # prepared == raw for prose, and it still matches all three docs.
        assert self._match(fts, prepare_fts_query("Sanger sequencing")) == [1, 2, 3]
