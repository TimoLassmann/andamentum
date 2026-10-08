"""Searchers: swappable ranking over an index, chosen when the store is opened.

``HybridRerankSearcher`` is the default: BM25 (OR-joined terms, porter
stemming) and vector nearest neighbours over chunks, fused by reciprocal rank
(k=60) to the top 50, then reordered by a cross-encoder. This is the
configuration measured as ``v2_rerank`` on the retrieval benchmark (MuSiQue
dev, all supporting passages in the top 10 for 0.490 of questions; HotpotQA
dev 0.955). ``AgentSearcher`` (``agent.py``) runs on top of it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from .index import HybridIndex, Index, to_hit
from .types import Hit, Where

if TYPE_CHECKING:
    from .store import Store

#: Reciprocal-rank-fusion constant and the depth handed to the reranker.
RRF_K = 60
FUSED_DEPTH = 50


class Searcher(Protocol):
    """What ``Store.search`` needs from a searcher."""

    def indexes(self) -> Sequence[Index]:
        """The indexes this searcher reads; the store must have been opened with them."""
        ...

    async def search(
        self, store: Store, query: str, *, limit: int, where: Where | None
    ) -> list[Hit]: ...


def rrf(
    rankings: Sequence[Sequence[int]], *, k: int = RRF_K, depth: int = FUSED_DEPTH
) -> list[tuple[int, float]]:
    """Reciprocal-rank fusion; ties break on the smaller id, so the order is fixed."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:depth]


class HybridRerankSearcher:
    """BM25 + vectors → RRF top 50 → cross-encoder rerank.

    ``reranker_model`` has no default: a score's scale belongs to one model
    (``reranking.SUGGESTED_RERANKER_MODEL`` names the measured one). With an
    index that has no embedding model, the vector leg is left out.
    """

    def __init__(
        self, *, index: HybridIndex, reranker_model: str, depth: int = FUSED_DEPTH
    ) -> None:
        self.index = index
        self.reranker_model = reranker_model
        self.depth = depth

    def indexes(self) -> Sequence[Index]:
        return (self.index,)

    async def candidates(
        self,
        store: Store,
        query: str,
        *,
        depth: int,
        allowed: tuple[str, list[object]] | None,
    ) -> list[Hit]:
        """The fused candidates before reranking; ``score`` is the RRF score.

        ``allowed`` is SQL selecting the document ids to search within (see
        ``Store.where_docs``); ``None`` searches every document.
        """
        db = store.db
        rankings = [self.index.bm25(db, query, k=depth, allowed=allowed)]
        if self.index.embedding_model is not None:
            (vector,) = await self.index.embed([query])
            rankings.append(self.index.nearest(db, vector, k=depth, allowed=allowed))
        fused = rrf(rankings, depth=depth)
        chunks = self.index.chunks(db, [cid for cid, _ in fused])
        score = dict(fused)
        return [to_hit(c, score[c.chunk_id]) for c in chunks]

    async def rerank(self, store: Store, query: str, hits: list[Hit]) -> list[Hit]:
        """Reorder ``hits`` by cross-encoder score against ``query``; the score replaces theirs."""
        from .reranking import rerank

        if not hits:
            return []
        prefixes = self._prefixes(store, hits)
        texts = [f"{p}\n\n{h.text}" if p else h.text for h, p in zip(hits, prefixes)]
        scores = await rerank(query, texts, reranker_model=self.reranker_model)
        scored = [
            Hit(h.doc_id, h.rev, h.start, h.end, h.text, float(s), h.heading_path)
            for h, s in zip(hits, scores, strict=True)
        ]
        # Stable sort: equal scores keep the incoming (fused) order.
        return sorted(scored, key=lambda h: -h.score)

    def _prefixes(self, store: Store, hits: list[Hit]) -> list[str]:
        """Each hit's context prefix ("Title > Heading"), as the index stored it."""
        db = store.db
        t = f"ix_{self.index.name}_chunks"
        out: list[str] = []
        for h in hits:
            row = db.execute(
                f"SELECT prefix FROM {t} WHERE doc_id = ? AND start = ? AND end = ?",
                (h.doc_id, h.start, h.end),
            ).fetchone()
            out.append(row[0] if row else "")
        return out

    async def search(
        self, store: Store, query: str, *, limit: int, where: Where | None
    ) -> list[Hit]:
        depth = max(self.depth, limit)
        allowed = store.where_docs(where) if where else None
        hits = await self.candidates(store, query, depth=depth, allowed=allowed)
        return (await self.rerank(store, query, hits))[:limit]
