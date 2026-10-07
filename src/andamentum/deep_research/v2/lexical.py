"""Lexical scoring of passages: the one scorer the reading window and the
fallback ordering use. Pure functions, no model.

Moved from the old ``strategies.py`` (``lexical_terms``, ``bm25_scores``) with
the stop-word list of the old ``read.py``; behaviour unchanged.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

BM25_K1 = 1.5
BM25_B = 0.75

_WORD = re.compile(r"[^\W_]{3,}")
_STOPWORDS = frozenset(
    "the and for are was were with that this from have has had not but can will "
    "its their there which what when where who how why does did into than then "
    "also been being such any all more most other some only over under about "
    "between these those they them our out one two may might could would should".split()
)


def lexical_terms(text: str) -> list[str]:
    """The distinct query terms of *text*: casefolded, 3+ letters, stop-words
    out, a plural ``s`` folded.

    Words of *text* for BM25 (``latencies`` stays, ``measurements`` ->
    ``measurement``); repeats are kept, so a document's term frequencies count.
    """
    terms: list[str] = []
    for word in _WORD.findall(text.casefold()):
        if word in _STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        terms.append(word)
    return terms


def bm25_scores(query: str, documents: Sequence[str]) -> list[float]:
    """Okapi BM25 of *query* against each of *documents* (the collection is
    *documents* itself); 0.0 for a document sharing no term. The idf is the
    non-negative form, ``log(1 + (N - df + 0.5) / (df + 0.5))``."""
    tokenised = [lexical_terms(d) for d in documents]
    if not tokenised:
        return []
    wanted = set(lexical_terms(query))
    count = len(tokenised)
    average = sum(len(t) for t in tokenised) / count or 1.0
    frequency = Counter(term for terms in tokenised for term in set(terms))
    idf = {
        term: math.log(1 + (count - frequency[term] + 0.5) / (frequency[term] + 0.5))
        for term in wanted
    }
    scores: list[float] = []
    for terms in tokenised:
        counts = Counter(terms)
        norm = BM25_K1 * (1 - BM25_B + BM25_B * len(terms) / average)
        scores.append(
            sum(
                idf[term] * counts[term] * (BM25_K1 + 1) / (counts[term] + norm)
                for term in wanted
                if counts[term]
            )
        )
    return scores
