"""Rank works against a query or a claim by words they share.

Deterministic and local: used for a user's library and for the Zotero library,
where the whole collection is in hand. A claim sentence rarely matches a title
word for word, so the score counts distinct content words found in the work's
title (twice), abstract, keywords, container and author names. A work needs at
least two of the query's words (one when the query has only one) to count.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .models import Work

_STOPWORDS = frozenset(
    """a about after all also an and any are as at be been between but by can
    could did do does for from had has have how if in into is it its may more
    most no not of on or our over per than that the their them then there these
    they this those through to under up was we were what when which while who
    will with within would study studies using use used""".split()
)
_WORD = re.compile(r"[a-z0-9][a-z0-9-]{2,}")


def words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


def _field_words(work: Work, *fields: str) -> set[str]:
    found: set[str] = set()
    for name in fields:
        value = work.csl.get(name)
        if isinstance(value, str):
            found |= words(value)
    return found


def score(work: Work, query_words: set[str]) -> int:
    """Twice the title hits plus the hits in the other text fields."""
    title = _field_words(work, "title")
    other = _field_words(work, "abstract", "keyword", "container-title")
    for name in work.authors:
        other |= words(name)
    return sum(2 if w in title else 1 for w in query_words if w in title or w in other)


def rank(works: Sequence[Work], query: str, *, limit: int) -> list[Work]:
    """The best *limit* works for *query*, best first; ties keep library order."""
    query_words = words(query)
    if not query_words:
        return []
    needed = min(2, len(query_words))
    scored: list[tuple[int, int, Work]] = []
    for position, work in enumerate(works):
        text = _field_words(work, "title", "abstract", "keyword", "container-title")
        for name in work.authors:
            text |= words(name)
        if len(query_words & text) >= needed:
            scored.append((score(work, query_words), position, work))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [work for _, _, work in scored[:limit]]
