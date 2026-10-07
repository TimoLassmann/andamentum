"""Arms: one way of building a store from a corpus and ranking it per question.

An arm returns passage ids, best first, and an answering arm also its answer.
The current store (``v1``) is built the way ``ingest(process="now")`` does —
register the document, chunk it with the same chunker settings, store chunk
and document embeddings — except that vectors come from the
:class:`EmbeddingCache` instead of a live model call; ``tests/test_bench.py``
pins that equivalence against ``ingest``. The v2 store is built with
``SectionStore.add_document``.

A built store is reused only when its recorded layout version, embedding
model, embedding digest and corpus hash all equal what the run asks for; any
difference is an error, never a silent reuse. Bump ``*_LAYOUT`` whenever the
code that builds or opens a store changes what the file holds.

Retrieval bypasses the public ``search()``, which always makes an LLM call to
plan the query; this benchmark measures the store, not the planner.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from andamentum.chunker import extract_units
from pydantic_ai.settings import ModelSettings

from ..api import DocumentStore
from ..chunker_adapter import Chunk, units_to_chunks
from ..database import get_async_connection
from ..metadata_models import ChunkMetadataFields
from ..reranking import SUGGESTED_RERANKER_MODEL
from ..search import search_unified
from ..v2 import Hit, SectionStore
from .data import Corpus, Passage, Question
from .embedding_cache import EmbeddingCache
from .provenance import git_state

#: Ranked list depth every arm returns; the deepest metric cut-off.
DEPTH = 20

#: How many fused results a reranking arm hands to the cross-encoder; the v2
#: store reranks its fused top 50, so the current store gets the same pool.
RERANK_POOL = 50

#: Layout versions of the built store files. v2 layout 2 adds the agent tables
#: (collections, evidence, decisions).
CURRENT_LAYOUT = 1
V2_LAYOUT = 2

#: Sampling settings every answering arm sends with every LLM call.
DRIVER_SETTINGS: ModelSettings = {"temperature": 0.0}


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
class StoreInfo:
    """What the run record says about the store a run used."""

    path: str
    layout: int
    reused: bool
    build: dict[str, Any]  # the meta written when the store was built


def _store_name(
    kind: str, layout: int, corpus: Corpus, emb: EmbeddingCache | None
) -> str:
    tag = (
        f"{emb.model.replace(':', '_').replace('/', '_')}_{emb.digest[:12]}"
        if emb
        else "noemb"
    )
    return f"{kind}{layout}_{corpus.suite}_{corpus.tier}_{corpus.sha256[:12]}_{tag}"


def _expected_meta(layout: int, corpus: Corpus, emb: EmbeddingCache | None) -> dict:
    return {
        "layout": layout,
        "corpus_sha256": corpus.sha256,
        "embedding_model": emb.model if emb else None,
        "embedding_digest": emb.digest if emb else None,
    }


def _check_meta(meta_path: Path, expected: dict) -> dict:
    """The recorded build meta; raises when any field differs from ``expected``."""
    meta = json.loads(meta_path.read_text())
    wrong = {k: (meta.get(k), v) for k, v in expected.items() if meta.get(k) != v}
    if wrong:
        raise ValueError(
            f"store {meta_path.name} was built for a different run "
            f"(recorded, wanted): {wrong}; delete it to rebuild"
        )
    return meta


def _write_meta(meta_path: Path, expected: dict) -> dict:
    meta = {
        **expected,
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
    }
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    return meta


@dataclass
class BuiltStore:
    store: DocumentStore
    doc_to_pid: dict[str, str]
    #: Passage id -> the document text the store holds; what a reranker scores.
    pid_text: dict[str, str] = field(default_factory=dict)
    info: StoreInfo | None = None


async def build_current_store(
    corpus: Corpus, *, workdir: Path, emb: EmbeddingCache | None
) -> BuiltStore:
    """Build (or reopen, after checking its meta) the current store for a corpus."""
    name = _store_name("current", CURRENT_LAYOUT, corpus, emb)
    store_dir = workdir / "stores"
    store_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = store_dir / f"{name}.map.json"
    meta_path = store_dir / f"{name}.meta.json"
    expected = _expected_meta(CURRENT_LAYOUT, corpus, emb)
    store = DocumentStore(name, db_dir=store_dir)
    pid_text = {p.pid: document_content(p) for p in corpus.passages}
    if meta_path.exists():
        meta = _check_meta(meta_path, expected)
        if not store.db_path.exists() or not mapping_path.exists():
            raise FileNotFoundError(
                f"{meta_path.name} exists but its store files are missing; "
                "delete it to rebuild"
            )
        info = StoreInfo(str(store.db_path), CURRENT_LAYOUT, True, meta)
        return BuiltStore(store, json.loads(mapping_path.read_text()), pid_text, info)
    # No meta: anything left is a half-built store from an interrupted run.
    store.db_path.unlink(missing_ok=True)
    mapping_path.unlink(missing_ok=True)
    await store.initialize()
    doc_to_pid: dict[str, str] = {}
    for p in corpus.passages:
        doc_id = await _ingest_like(store, p, emb)
        if doc_id in doc_to_pid:
            raise ValueError(f"passages {doc_to_pid[doc_id]} and {p.pid} share content")
        doc_to_pid[doc_id] = p.pid
    mapping_path.write_text(json.dumps(doc_to_pid))
    meta = _write_meta(meta_path, expected)  # last: marks the store complete
    info = StoreInfo(str(store.db_path), CURRENT_LAYOUT, False, meta)
    return BuiltStore(store, doc_to_pid, pid_text, info)


@dataclass
class BuiltV2:
    store: SectionStore
    info: StoreInfo | None = None


async def build_v2_store(
    corpus: Corpus, *, workdir: Path, emb: EmbeddingCache | None
) -> BuiltV2:
    """Build (or reopen, after checking its meta) the v2 store; one document per passage.

    Documents hold the same markdown as in the current store, so a passage that
    fits one section embeds the identical text and its vector comes from the
    cache: the two stores are compared on the same embeddings.
    """
    name = _store_name("v2_", V2_LAYOUT, corpus, emb)
    path = workdir / "stores" / f"{name}.db"
    meta_path = path.with_suffix(".meta.json")
    expected = _expected_meta(V2_LAYOUT, corpus, emb)
    dim = None
    if emb is not None:
        [probe] = await emb.get_or_compute([document_content(corpus.passages[0])])
        dim = len(probe)
    store = SectionStore(path, embedding_dim=dim)
    if meta_path.exists():
        meta = _check_meta(meta_path, expected)
        if not path.exists():
            raise FileNotFoundError(
                f"{meta_path.name} exists but {path.name} is missing; delete it to rebuild"
            )
        return BuiltV2(store, StoreInfo(str(path), V2_LAYOUT, True, meta))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)  # a half-built store from an interrupted run
    store.create()
    embed_fn = emb.get_or_compute if emb is not None else None
    for p in corpus.passages:
        await store.add_document(
            path=p.pid, title=p.title, text=document_content(p), embed_fn=embed_fn
        )
    meta = _write_meta(meta_path, expected)  # last: marks the store complete
    return BuiltV2(store, StoreInfo(str(path), V2_LAYOUT, False, meta))


@dataclass
class BuiltPool:
    """No store: the pooled passage ids, for arms that do not search."""

    pids: list[str]
    info: StoreInfo | None = None


async def build_pool(
    corpus: Corpus, *, workdir: Path, emb: EmbeddingCache | None
) -> BuiltPool:
    return BuiltPool([p.pid for p in corpus.passages])


class Arm(Protocol):
    name: str
    store_kind: str  # "current", "v2" or "none": which builder the runner uses
    needs_embeddings: bool
    answers: bool  # True: returns an ArmResult with an answer for every question

    async def rank(
        self,
        built: Any,
        q: Question,
        emb: EmbeddingCache | None,
        *,
        driver_model: str | None = None,
        run_id: str | None = None,
    ) -> list[str] | ArmResult: ...


class Bm25OrArm:
    """Plain BM25 baseline on the store's own ``documents_fts`` index.

    Every word of the question is a quoted term joined by OR, ranked by FTS5's
    ``bm25()``. This is the textbook one-shot BM25 the plan names as a baseline;
    the store's own keyword signal (:class:`Fts5Arm`) joins terms with an
    implicit AND instead, which a multi-hop question almost never satisfies.
    """

    name = "bm25"
    store_kind = "current"
    answers = False
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
    """The current store's full 4-signal RRF search, query embedded from the cache.

    With ``reranker``, the fused top :data:`RERANK_POOL` documents are reordered
    by the same cross-encoder the v2 arms use, so a v2 gain can be split into
    what the reranker gives and what the store layout gives.
    """

    store_kind = "current"
    answers = False
    needs_embeddings = True

    def __init__(self, name: str, *, reranker: str | None = None) -> None:
        self.name = name
        self.reranker = reranker

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
            raise ValueError(f"the {self.name} arm needs an embedding cache")
        [qvec] = await emb.get_or_compute([q.text])
        limit = DEPTH if self.reranker is None else RERANK_POOL
        results = await search_unified(
            str(built.store.db_path), q.text, limit=limit, query_embedding=qvec
        )
        pids = [built.doc_to_pid[r.doc_id] for r in results]
        if self.reranker is None or not pids:
            return pids
        from ..reranking import rerank

        scores = await rerank(
            q.text, [built.pid_text[pid] for pid in pids], reranker_model=self.reranker
        )
        order = sorted(range(len(pids)), key=lambda i: (-scores[i], i))
        return [pids[i] for i in order][:DEPTH]


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
    answers = False

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


class OracleArm:
    """Positive control: the supporting passages first, then nothing."""

    name = "oracle"
    store_kind = "none"
    answers = False
    needs_embeddings = False

    async def rank(
        self,
        built: BuiltPool,
        q: Question,
        emb: EmbeddingCache | None,
        *,
        driver_model: str | None = None,
        run_id: str | None = None,
    ) -> list[str]:
        return sorted(q.gold)


class ShuffledArm:
    """Negative control: the pool in a random order, seeded by the question id."""

    name = "shuffled"
    store_kind = "none"
    answers = False
    needs_embeddings = False

    async def rank(
        self,
        built: BuiltPool,
        q: Question,
        emb: EmbeddingCache | None,
        *,
        driver_model: str | None = None,
        run_id: str | None = None,
    ) -> list[str]:
        pids = list(built.pids)
        random.Random(q.qid).shuffle(pids)
        return pids[:DEPTH]


class ClosedBookArm:
    """No retrieval: one LLM call answers from the model's own knowledge."""

    name = "closed_book"
    store_kind = "none"
    answers = True
    needs_embeddings = False

    async def rank(
        self,
        built: BuiltPool,
        q: Question,
        emb: EmbeddingCache | None,
        *,
        driver_model: str | None = None,
        run_id: str | None = None,
    ) -> ArmResult:
        from ..v2.agent import answer_closed_book

        model = _require_driver(self.name, driver_model)
        answer, usage = await answer_closed_book(
            q.text, model=model, model_settings=DRIVER_SETTINGS
        )
        calls, tin, tout = _usage_counts(usage)
        return ArmResult(
            ranked=[],
            answer=answer.text,
            llm_calls=calls,
            tokens_in=tin,
            tokens_out=tout,
            entered_loop=False,
        )


class V2RagArm:
    """One search, one answer: round 0 + rerank + one synthesis call over the top 10."""

    name = "v2_rag"
    store_kind = "v2"
    answers = True
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
        from ..v2.agent import answer_from_hits

        model = _require_driver(self.name, driver_model)
        cache = _require_embeddings(self.name, emb)
        [qvec] = await cache.get_or_compute([q.text])
        hits = built.store.round0(q.text, query_embedding=qvec, use_exact=False)
        hits = await built.store.rerank(q.text, hits, reranker_model=self.reranker)
        answer, usage = await answer_from_hits(
            q.text,
            hits[: self.top_k],
            model=model,
            store=built.store,
            model_settings=DRIVER_SETTINGS,
        )
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
    """The agent loop (``run_agentic``) with the reranker and the default budget.

    With ``vector=False`` the agent works without embeddings: lexical round 0
    and no ``similar`` tool.
    """

    store_kind = "v2"
    answers = True
    reranker = SUGGESTED_RERANKER_MODEL
    exact = False
    vector_weight = 1

    def __init__(self, name: str, *, vector: bool) -> None:
        self.name = name
        self.needs_embeddings = vector

    async def rank(
        self,
        built: BuiltV2,
        q: Question,
        emb: EmbeddingCache | None,
        *,
        driver_model: str | None = None,
        run_id: str | None = None,
    ) -> ArmResult:
        from ..v2.agent import AgentBudget, run_agentic

        model = _require_driver(self.name, driver_model)
        embed_fn = (
            _require_embeddings(self.name, emb).get_or_compute
            if self.needs_embeddings
            else None
        )
        result = await run_agentic(
            q.text,
            store=built.store,
            model=model,
            embed_fn=embed_fn,
            reranker_model=self.reranker,
            run_id=run_id or q.qid,
            budget=AgentBudget(),
            model_settings=DRIVER_SETTINGS,
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


ARMS: dict[str, Arm] = {
    a.name: a
    for a in (
        OracleArm(),
        ShuffledArm(),
        Bm25OrArm(),
        UnifiedArm("v1"),
        UnifiedArm("v1_rerank", reranker=SUGGESTED_RERANKER_MODEL),
        V2Arm("v2", vector=True, exact=False),
        V2Arm("v2_rerank", vector=True, exact=False, reranker=SUGGESTED_RERANKER_MODEL),
        V2Arm(
            "v2_lex_rerank",
            vector=False,
            exact=False,
            reranker=SUGGESTED_RERANKER_MODEL,
        ),
        ClosedBookArm(),
        V2RagArm(),
        V2AgentArm("v2_agent", vector=True),
        V2AgentArm("v2_agent_lex", vector=False),
    )
}
