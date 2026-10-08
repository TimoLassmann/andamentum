"""The value types of the document store's fixed API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: A ``where`` filter: metadata field (dotted for nested fields) -> value.
#: A scalar matches by equality, ``None`` matches a missing or null field, and
#: a list matches any of its values. All entries must hold (AND).
Scalar = str | int | float | bool
Where = dict[str, Scalar | None | list[Scalar]]


class DocumentStoreError(Exception):
    """Base class for document store errors."""


class SearchUnavailable(DocumentStoreError, RuntimeError):
    """``search`` cannot answer: the store was opened without a searcher, or
    some documents could not be indexed so a search would silently miss them."""


class SchemaMismatch(DocumentStoreError, RuntimeError):
    """The file holds a source schema version this code does not read."""


class DocumentNotFound(DocumentStoreError, KeyError):
    """No live document has this id."""


class KeyExists(DocumentStoreError):
    """Another document already holds this key."""


@dataclass(frozen=True)
class Document:
    """One revision of a document, with its full text."""

    doc_id: str
    key: str | None
    rev: int
    created_at: str  # when the document was first put
    updated_at: str  # when this revision was written
    media_type: str
    text: str
    metadata: dict[str, Any]
    original_hash: str | None  # sha256 of the original bytes, if stored


@dataclass(frozen=True)
class DocRef:
    """A document as ``list`` returns it: its newest revision, without the text.

    ``cursor`` is opaque: pass the last one of a page to ``list(cursor=...)``
    for the next page.
    """

    doc_id: str
    key: str | None
    rev: int
    updated_at: str
    media_type: str
    metadata: dict[str, Any]
    cursor: str


@dataclass(frozen=True)
class Hit:
    """One passage of a search result.

    ``start`` and ``end`` are character offsets in the text of revision
    ``rev``: ``store.read(doc_id, rev=rev, start=start, end=end) == text``.
    """

    doc_id: str
    rev: int
    start: int
    end: int
    text: str
    score: float
    heading_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentHit:
    """A document in a grouped result: its best passage scores it (MaxP)."""

    doc_id: str
    score: float
    hits: list[Hit]


@dataclass
class SyncReport:
    """What one ``sync`` did, per index."""

    processed: dict[str, int] = field(default_factory=dict)  # index -> changes applied
    failures: dict[str, list[tuple[str, str]]] = field(
        default_factory=dict
    )  # index -> [(doc_id, error)] still failing after this sync

    @property
    def ok(self) -> bool:
        return not any(self.failures.values())


def group_by_document(hits: list[Hit]) -> list[DocumentHit]:
    """Group passages by document, in the order of each document's first passage.

    ``hits`` is a ranking (searchers return best first), so the first passage
    of a document is its best-ranked one; the document's ``score`` is the
    highest passage score (MaxP). Passages keep their order.
    """
    grouped: dict[str, list[Hit]] = {}
    for h in hits:
        grouped.setdefault(h.doc_id, []).append(h)
    docs = [
        DocumentHit(doc_id=d, score=max(h.score for h in hs), hits=hs)
        for d, hs in grouped.items()
    ]
    return docs


def line_range(text: str, *, start: int, end: int) -> tuple[int, int]:
    """1-based inclusive line numbers that the span ``text[start:end]`` covers."""
    if not 0 <= start <= end <= len(text):
        raise ValueError(f"span {start}-{end} is outside a text of {len(text)} chars")
    first = text.count("\n", 0, start) + 1
    last = first + text.count("\n", start, max(start, end - 1))
    return first, last
