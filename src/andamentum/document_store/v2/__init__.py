"""andamentum.document_store.v2 — sections-first SQLite store with round-0 search.

Round 0 of the agentic-retrieval design: BM25 (porter FTS5), exact identifier
matching (trigram FTS5) and sqlite-vec nearest neighbours over sections with
line ranges, fused by reciprocal rank, optionally reranked by a cross-encoder.
No LLM call. Measured against the current store by ``document_store.bench``.
"""

from .sections import Section, split_sections
from .store import Hit, SectionStore, identifiers, rrf

__all__ = ["Hit", "Section", "SectionStore", "identifiers", "rrf", "split_sections"]
