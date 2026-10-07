"""Lexical scoring of passages: the one scorer the reading window and the
fallback ordering use. Pure functions, no model.

Package b moves ``lexical_terms`` and ``bm25_scores`` here from the old
``strategies.py`` (``git show bbdb010:src/andamentum/deep_research/v2/strategies.py``,
lines ~930-985), with ``BM25_K1`` 1.5, ``BM25_B`` 0.75 and the stop-word list of
the old ``read._STOPWORDS``.
"""

from __future__ import annotations

from collections.abc import Sequence

BM25_K1 = 1.5
BM25_B = 0.75


def lexical_terms(text: str) -> list[str]:
    """The distinct query terms of *text*: casefolded, 3+ letters, stop-words
    out, a plural ``s`` folded."""
    raise NotImplementedError("package b")


def bm25_scores(query: str, documents: Sequence[str]) -> list[float]:
    """Okapi BM25 of *query* against each of *documents* (the collection is
    *documents* itself); 0.0 for a document sharing no term."""
    raise NotImplementedError("package b")
