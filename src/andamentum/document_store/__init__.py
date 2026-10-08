"""andamentum.document_store — one SQLite file of documents, searchable.

Fixed data, flexible index, fixed API:

- The source tables hold every revision of every document in full, plus the
  original bytes when given. They never change shape.
- Indexes are disposable. Each one catches up from the change log on
  ``sync()`` and is rebuilt when its configuration changes.
- The API is :class:`Store`: ``open``, ``put``, ``get``, ``resolve``,
  ``update``, ``delete``, ``restore``, ``purge``, ``list``, ``count``,
  ``search``, ``read``, ``sync``. Searchers and enrichers change behind it.

The default searcher is :class:`HybridRerankSearcher` (BM25 + vectors, fused,
reranked by a cross-encoder); :class:`AgentSearcher` runs a model-driven
search loop on top of it.
"""

from .agent import AgentBudget, AgentSearcher
from .index import HybridIndex, Index
from .reranking import (
    SUGGESTED_RERANKER_MODEL,
    rerank,
    reset_reranker_cache,
    select_relevant,
)
from .searchers import HybridRerankSearcher, Searcher, rrf
from .store import Store
from .types import (
    DocRef,
    Document,
    DocumentHit,
    DocumentNotFound,
    DocumentStoreError,
    Hit,
    KeyExists,
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
    "Searcher",
    "Store",
    "SyncReport",
    "Where",
    "group_by_document",
    "line_range",
    "rerank",
    "reset_reranker_cache",
    "rrf",
    "select_relevant",
]
