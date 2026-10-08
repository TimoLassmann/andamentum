"""andamentum.document_store — one SQLite file of documents, searchable.

Fixed data, flexible index, fixed API:

- The source tables hold every revision of every document in full, plus the
  original bytes when given. They never change shape.
- Indexes are disposable. Each one catches up from the change log on
  ``sync()`` and is rebuilt when its configuration changes.
- The API is :class:`Store`: ``open``, ``open_with_search``, ``put``, ``get``,
  ``resolve``, ``update``, ``delete``, ``restore``, ``purge``, ``list``,
  ``count``, ``search``, ``read``, ``sync``. Searchers and enrichers change
  behind it.

``await Store.open_with_search(path, embedding_model=..., reranker_model=...)``
is the common case: a store that can search. ``await ingest(store, source)``
turns a file or URL into a stored document. Enrichers (``enrich_identifiers``,
``enrich_facts``, ``enrich_bibliographic``, ``enrich_extract``,
``enrich_classify`` and ``enrich_all``) live in
``andamentum.document_store.enrich``.

The default searcher is :class:`HybridRerankSearcher` (BM25 + vectors, fused,
reranked by a cross-encoder); :class:`AgentSearcher` runs a model-driven
search loop on top of it.
"""

from .agent import AgentBudget, AgentSearcher
from .index import HybridIndex, Index
from .pipeline import ingest
from .reranking import (  # noqa: F401  (importable, not exported)
    SUGGESTED_RERANKER_MODEL,
    rerank,
    reset_reranker_cache,
    select_relevant,
)
from .searchers import HybridRerankSearcher, Searcher, rrf  # noqa: F401
from .store import Store
from .types import (
    DocRef,
    Document,
    DocumentHit,
    DocumentNotFound,
    DocumentStoreError,
    Hit,
    KeyExists,
    SchemaMismatch,
    SearchUnavailable,
    SyncReport,
    Where,
    group_by_document,
    line_range,
)

__all__ = [
    "SUGGESTED_RERANKER_MODEL",
    "AgentBudget",
    "AgentSearcher",
    "DocRef",
    "Document",
    "DocumentHit",
    "DocumentNotFound",
    "DocumentStoreError",
    "Hit",
    "HybridIndex",
    "HybridRerankSearcher",
    "Index",
    "KeyExists",
    "SchemaMismatch",
    "SearchUnavailable",
    "Searcher",
    "Store",
    "SyncReport",
    "Where",
    "group_by_document",
    "ingest",
    "line_range",
]
