"""Shared pieces of the enrichers: the result type and the metadata write."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..store import Store
from ..types import Document, DocumentNotFound, DocumentStoreError


class EnrichError(DocumentStoreError):
    """An enricher could not produce its value (a lookup found nothing, say)."""


@dataclass(frozen=True)
class EnrichResult:
    """What one enricher did to one document.

    ``rev`` is the revision written, or ``None`` when the enricher skipped the
    document; ``skipped`` then says why.
    """

    doc_id: str
    key: str
    rev: int | None
    skipped: str | None = None


async def load(store: Store, doc_id: str) -> Document:
    """The newest revision of a live document; :class:`DocumentNotFound` if none."""
    doc = await store.get(doc_id)
    if doc is None:
        raise DocumentNotFound(doc_id)
    return doc


async def write_key(
    store: Store, doc: Document, *, key: str, value: dict[str, Any]
) -> EnrichResult:
    """Write ``value`` under metadata ``key``, keeping every other key.

    One new revision. The read-modify-write is not atomic: an enricher run
    while another writer updates the same document can overwrite that
    writer's change, so run enrichers one at a time per store.
    """
    metadata = dict(doc.metadata)
    metadata[key] = value
    rev = await store.update(doc.doc_id, metadata=metadata)
    return EnrichResult(doc_id=doc.doc_id, key=key, rev=rev)


def skipped(doc: Document, *, key: str, reason: str) -> EnrichResult:
    return EnrichResult(doc_id=doc.doc_id, key=key, rev=None, skipped=reason)
