"""Run one enricher over many documents."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ..store import Store
from ..types import Where
from ._base import EnrichResult

Enricher = Callable[..., Awaitable[EnrichResult]]


async def enrich_all(
    store: Store, enricher: Enricher, *, where: Where | None = None, **args: Any
) -> list[EnrichResult]:
    """Run ``enricher(store, doc_id, **args)`` over every live document matching ``where``.

    The ids are listed first, then the documents are enriched one after the
    other, never concurrently: a local model must not run twice at once. The
    first error propagates; documents already enriched keep their new revision.
    A document deleted after listing raises ``DocumentNotFound``.
    """
    refs = await store.list(where=where, limit=None)
    results: list[EnrichResult] = []
    for ref in reversed(refs):  # oldest first
        results.append(await enricher(store, ref.doc_id, **args))
    return results
