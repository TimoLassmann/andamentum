"""andamentum.document_store.v2 — sections-first SQLite store with agent-run search.

Round 0: BM25 (porter FTS5), exact identifier matching (trigram FTS5) and
sqlite-vec nearest neighbours over sections with line ranges, fused by
reciprocal rank, optionally reranked by a cross-encoder. No LLM call.

``search_agentic`` lets an agent decide where to look and returns search
results, never prose. Answering over passages is separate, in
``v2.answering``, which no search code imports. Measured against the current
store by ``document_store.bench``.
"""

from .agent import (
    AgentBudget,
    AgentSearchResult,
    DocumentResult,
    EmbedFn,
    MatchType,
    search_agentic,
)
from .answering import answer_closed_book, answer_from_hits
from .decompose import Decomposition, decompose, retrieve_decomposed
from .sections import Section, split_sections
from .store import DocEntry, Hit, OutlineEntry, SectionStore, identifiers, rrf
from .types import Answer, Citation, PassageRef, Selection, Sufficiency

__all__ = [
    "AgentBudget",
    "AgentSearchResult",
    "Answer",
    "Citation",
    "Decomposition",
    "DocEntry",
    "DocumentResult",
    "EmbedFn",
    "Hit",
    "MatchType",
    "OutlineEntry",
    "PassageRef",
    "Section",
    "SectionStore",
    "Selection",
    "Sufficiency",
    "answer_closed_book",
    "answer_from_hits",
    "decompose",
    "identifiers",
    "retrieve_decomposed",
    "rrf",
    "search_agentic",
    "split_sections",
]
