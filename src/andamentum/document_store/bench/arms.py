"""Arms: one way of building a store from a corpus and ranking it per question.

An arm returns passage ids, best first. Both arms here build the CURRENT
document store the way ``ingest(process="now")`` does — register the document,
chunk it with the same chunker settings, store chunk and document embeddings —
except that vectors come from the :class:`EmbeddingCache` instead of a live
model call. ``tests/test_bench.py`` pins that equivalence against ``ingest``.

Retrieval bypasses the public ``search()``, which always makes an LLM call to
plan the query; this benchmark measures the store, not the planner.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from andamentum.chunker import extract_units

from ..api import DocumentStore
from ..database import get_async_connection
from ..chunker_adapter import Chunk, units_to_chunks
from ..metadata_models import ChunkMetadataFields
from ..search import search_fts5, search_unified
from ..reranking import SUGGESTED_RERANKER_MODEL
from ..v2 import Hit, SectionStore
from .data import Corpus, Passage, Question
from .embedding_cache import EmbeddingCache

#: Ranked list depth every arm returns; the deepest metric cut-off.
DEPTH = 20


def document_content(p: Passage) -> str:
    """A passage as the store would hold it after harvest: a titled markdown page."""
    return f"# {p.title}\n\n{p.text}"


async def _ingest_like(
    store: DocumentStore, p: Passage, emb: EmbeddingCache | None
) -> str:
    """Mirror of ``public.ingest(process="now")`` with cached vectors."""
    content = document_content(p)
    doc_id = await store.register_document(title=p.title, content=content)
    if emb is None:
        return doc_id

    async def embedding_fn(texts: list[str]) -> list[list[float]]:
        return await emb.get_or_compute(texts)

    chunking = await extract_units(
        content, target_min_chars=1500, target_max_chars=4000, embedding_fn=embedding_fn
    )
    chunks = units_to_chunks(content, chunking.units) or [
        Chunk(
            text=content,
            section_path="",
            chunk_index=0,
            start_char=0,
            end_char=len(content),
        )
    ]
    vectors = await emb.get_or_compute([c.text for c in chunks] + [content])
    await store.store_doc_embedding(doc_id, vectors[-1])
    for chunk, vec in zip(chunks, vectors[:-1], strict=True):
        meta = ChunkMetadataFields(
            parent_doc_id=doc_id,
            section_path=chunk.section_path,
            chunk_index=chunk.chunk_index,
        )
        await store.store_chunk(
            doc_id,
            chunk.text,
            vec,
            metadata=meta.model_dump(mode="json"),
            chunk_index=chunk.chunk_index,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
        )
    return doc_id


@dataclass
class ArmResult:
    """What an answering arm returns: the ranking plus the answer and its cost."""

    ranked: list[str]
    answer: str | None = None
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    entered_loop: bool | None = None
    trace: list[dict] = field(default_factory=list)


@dataclass
class BuiltStore:
    store: DocumentStore
    doc_to_pid: dict[str, str]


async def build_current_store(
    corpus: Corpus, *, workdir: Path, emb: EmbeddingCache | None
) -> BuiltStore:
    """Build (or reopen) the current store for a corpus.

    A store is keyed by corpus hash and embedding model, so it is built once and
    reused; a store layout change gets a new name and rebuilds from the cache.
    """
    tag = f"{emb.model.replace(':', '_').replace('/', '_')}" if emb else "noemb"
    name = f"current_{corpus.suite}_{corpus.tier}_{corpus.sha256[:12]}_{tag}"
    store_dir = workdir / "stores"
    mapping_path = store_dir / f"{name}.map.json"
    store = DocumentStore(name, db_dir=store_dir)
    if mapping_path.exists():
        return BuiltStore(store, json.loads(mapping_path.read_text()))
    if store.db_path.exists():
        store.db_path.unlink()  # a half-built store from an interrupted run
    await store.initialize()
    doc_to_pid: dict[str, str] = {}
    for p in corpus.passages:
        doc_id = await _ingest_like(store, p, emb)
        if doc_id in doc_to_pid:
            raise ValueError(f"passages {doc_to_pid[doc_id]} and {p.pid} share content")
        doc_to_pid[doc_id] = p.pid
    mapping_path.write_text(json.dumps(doc_to_pid))
    return BuiltStore(store, doc_to_pid)


class Arm(Protocol):
    name: str
    store_kind: str  # "current" or "v2": which builder the runner uses
    needs_embeddings: bool

    async def rank(
        self,
        built: Any,
        q: Question,
        emb: EmbeddingCache | None,
        *,
        driver_model: str | None = None,
        run_id: str | None = None,
    ) -> list[str] | ArmResult: ...


class Fts5Arm:
    """Signal 1 alone: BM25 over title + content in ``documents_fts``."""

    name = "fts5"
    store_kind = "current"
    needs_embeddings = False

    async def rank(
        self,
        built: BuiltStore,
        q: Question,
        emb: EmbeddingCache | None,
        *,
        driver_model: str | None = None,
        run_id: str | None = None,
    ) -> list[str]:
        hits = await search_fts5(str(built.store.db_path), q.text, limit=DEPTH)
        return [built.doc_to_pid[doc_id] for doc_id, _ in hits]


class Bm25OrArm:
    """Plain BM25 baseline on the store's own ``documents_fts`` index.

    Every word of the question is a quoted term joined by OR, ranked by FTS5's
    ``bm25()``. This is the textbook one-shot BM25 the plan names as a baseline;
    the store's own keyword signal (:class:`Fts5Arm`) joins terms with an
    implicit AND instead, which a multi-hop question almost never satisfies.
    """

    name = "bm25_or"
    store_kind = "current"
    needs_embeddings = False

    async def rank(
        self,
        built: BuiltStore,
        q: Question,
        emb: EmbeddingCache | None,
        *,
        driver_model: str | None = None,
        run_id: str | None = None,
    ) -> list[str]:
        terms = sorted({t.lower() for t in re.findall(r"\w+", q.text)})
        match = " OR ".join(f'"{t}"' for t in terms)
        async with get_async_connection(str(built.store.db_path)) as db:
            async with db.execute(
                "SELECT d.doc_uuid FROM documents_fts JOIN documents d "
                "ON documents_fts.rowid = d.id WHERE documents_fts MATCH ? "
                "AND d.deleted_at IS NULL ORDER BY bm25(documents_fts) LIMIT ?",
                (match, DEPTH),
            ) as cursor:
                rows = await cursor.fetchall()
        return [built.doc_to_pid[row[0]] for row in rows]


class UnifiedArm:
    """The current store's full 4-signal RRF search, query embedded from the cache."""

    name = "unified"
    store_kind = "current"
    needs_embeddings = True

    async def rank(
        self,
        built: BuiltStore,
        q: Question,
        emb: EmbeddingCache | None,
        *,
        driver_model: str | None = None,
        run_id: str | None = None,
    ) -> list[str]:
        if emb is None:
            raise ValueError("the unified arm needs an embedding cache")
        [qvec] = await emb.get_or_compute([q.text])
        results = await search_unified(
            str(built.store.db_path), q.text, limit=DEPTH, query_embedding=qvec
        )
        return [built.doc_to_pid[r.doc_id] for r in results]


# ---------------------------------------------------------------------------
# v2: the sections-first store (document_store.v2), round 0 of the plan
# ---------------------------------------------------------------------------


@dataclass
class BuiltV2:
    store: SectionStore


async def build_v2_store(
    corpus: Corpus, *, workdir: Path, emb: EmbeddingCache | None
) -> BuiltV2:
    """Build (or reopen) the v2 store; one document per passage, path = pid.

    Documents hold the same markdown as in the current store, so a passage that
    fits one section embeds the identical text and its vector comes from the
    cache: the two stores are compared on the same embeddings.
    """
    tag = f"{emb.model.replace(':', '_').replace('/', '_')}" if emb else "noemb"
    name = f"v2_{corpus.suite}_{corpus.tier}_{corpus.sha256[:12]}_{tag}"
    path = workdir / "stores" / f"{name}.db"
    done = path.with_suffix(".done")
    dim = None
    if emb is not None:
        [probe] = await emb.get_or_compute([document_content(corpus.passages[0])])
        dim = len(probe)
    store = SectionStore(path, embedding_dim=dim)
    if done.exists():
        return BuiltV2(store)
    path.unlink(missing_ok=True)  # a half-built store from an interrupted run
    store.create()
    embed_fn = emb.get_or_compute if emb is not None else None
    for p in corpus.passages:
        await store.add_document(
            path=p.pid, title=p.title, text=document_content(p), embed_fn=embed_fn
        )
    done.touch()
    return BuiltV2(store)


def _passages_in_order(hits: list[Hit]) -> list[str]:
    """Section hits -> passage ids, each at the rank of its best section."""
    seen: list[str] = []
    for h in hits:
        if h.path not in seen:
            seen.append(h.path)
    return seen[:DEPTH]


class V2Arm:
    """Round 0 on the v2 store.

    ``vector`` adds the vector leg, ``exact`` the identifier leg, ``vector_weight``
    counts the vector ranking that many times in the fusion, ``reranker`` reorders
    the fused top 50 with a cross-encoder.
    """

    store_kind = "v2"

    def __init__(
        self,
        name: str,
        *,
        vector: bool,
        exact: bool = True,
        vector_weight: int = 1,
        reranker: str | None = None,
    ) -> None:
        self.name = name
        self.needs_embeddings = vector
        self.exact = exact
        self.vector_weight = vector_weight
        self.reranker = reranker

    async def rank(
        self,
        built: BuiltV2,
        q: Question,
        emb: EmbeddingCache | None,
        *,
        driver_model: str | None = None,
        run_id: str | None = None,
    ) -> list[str]:
        qvec = None
        if self.needs_embeddings:
            if emb is None:
                raise ValueError(f"the {self.name} arm needs an embedding cache")
            [qvec] = await emb.get_or_compute([q.text])
        hits = built.store.round0(
            q.text,
            query_embedding=qvec,
            use_exact=self.exact,
            vector_weight=self.vector_weight,
        )
        if self.reranker is not None:
            hits = await built.store.rerank(q.text, hits, reranker_model=self.reranker)
        return _passages_in_order(hits)


def _usage_counts(usage: Any) -> tuple[int, int, int]:
    """(requests, input tokens, output tokens) of a pydantic-ai usage object."""
    return (
        int(getattr(usage, "requests", 0) or 0),
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )


def _require_driver(name: str, driver_model: str | None) -> str:
    if not driver_model:
        raise ValueError(f"the {name} arm needs --driver-model")
    return driver_model


def _require_embeddings(name: str, emb: EmbeddingCache | None) -> EmbeddingCache:
    if emb is None:
        raise ValueError(f"the {name} arm needs an embedding cache")
    return emb


class V2RagArm:
    """The plan's baseline: round 0 + rerank + one synthesis call over the top 10."""

    name = "v2_rag"
    store_kind = "v2"
    needs_embeddings = True
    reranker = SUGGESTED_RERANKER_MODEL
    exact = False
    vector_weight = 1
    top_k = 10

    async def rank(
        self,
        built: BuiltV2,
        q: Question,
        emb: EmbeddingCache | None,
        *,
        driver_model: str | None = None,
        run_id: str | None = None,
    ) -> ArmResult:
        from ..v2.agent import answer_from_hits  # deferred: Part B module

        model = _require_driver(self.name, driver_model)
        cache = _require_embeddings(self.name, emb)
        [qvec] = await cache.get_or_compute([q.text])
        hits = built.store.round0(q.text, query_embedding=qvec, use_exact=False)
        hits = await built.store.rerank(q.text, hits, reranker_model=self.reranker)
        answer, usage = await answer_from_hits(q.text, hits[: self.top_k], model=model)
        calls, tin, tout = _usage_counts(usage)
        return ArmResult(
            ranked=_passages_in_order(hits),
            answer=answer.text,
            llm_calls=calls,
            tokens_in=tin,
            tokens_out=tout,
            entered_loop=False,
        )


class V2AgentArm:
    """The agent loop (``run_agentic``) with the reranker and the default budget."""

    name = "v2_agent"
    store_kind = "v2"
    needs_embeddings = True
    reranker = SUGGESTED_RERANKER_MODEL
    exact = False
    vector_weight = 1

    async def rank(
        self,
        built: BuiltV2,
        q: Question,
        emb: EmbeddingCache | None,
        *,
        driver_model: str | None = None,
        run_id: str | None = None,
    ) -> ArmResult:
        from ..v2.agent import AgentBudget, run_agentic  # deferred: Part B module

        model = _require_driver(self.name, driver_model)
        cache = _require_embeddings(self.name, emb)
        result = await run_agentic(
            q.text,
            store=built.store,
            model=model,
            embed_fn=cache.get_or_compute,
            reranker_model=self.reranker,
            run_id=run_id or q.qid,
            budget=AgentBudget(),
        )
        return ArmResult(
            ranked=list(result.ranked_paths)[:DEPTH],
            answer=result.answer.text,
            llm_calls=result.llm_calls,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            entered_loop=result.entered_loop,
            trace=result.trace,
        )


def _v2_grid() -> list[V2Arm]:
    """The plan as written, the dev-split grid, and each with the reranker."""
    base = [
        V2Arm("v2_lexical", vector=False),
        V2Arm("v2_hybrid", vector=True),  # plan as written
        V2Arm("v2_noexact", vector=True, exact=False),
        V2Arm("v2_vec2", vector=True, vector_weight=2),
        V2Arm("v2_noexact_vec2", vector=True, exact=False, vector_weight=2),
    ]
    reranked = [
        V2Arm(
            f"{a.name}_rerank",
            vector=a.needs_embeddings,
            exact=a.exact,
            vector_weight=a.vector_weight,
            reranker=SUGGESTED_RERANKER_MODEL,
        )
        for a in base[1:]
    ]
    return base + reranked


ARMS: dict[str, Arm] = {
    a.name: a
    for a in (
        Fts5Arm(),
        Bm25OrArm(),
        UnifiedArm(),
        *_v2_grid(),
        V2RagArm(),
        V2AgentArm(),
    )
}
