"""Arms: one way of ranking the store per question.

An arm returns passage ids, best first, and an answering arm also its answer.
Every store arm reads the same store file: one document per passage (key =
passage id), with two indexes — ``hybrid`` (chunks, BM25, trigram, vectors
from the :class:`EmbeddingCache`) and ``lexical`` (the same without vectors).

A built store is reused only when its recorded layout version, embedding
model, embedding digest and corpus hash all equal what the run asks for; any
difference is an error, never a silent reuse. Bump ``STORE_LAYOUT`` whenever
the code that builds the store changes what the file holds.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic_ai.settings import ModelSettings

from ..agent import AgentBudget, AgentSearcher
from ..index import HybridIndex
from ..reranking import SUGGESTED_RERANKER_MODEL
from ..searchers import HybridRerankSearcher
from ..store import Store
from ..types import Hit, group_by_document
from .data import Corpus, Passage, Question
from .embedding_cache import EmbeddingCache
from .provenance import git_state

#: Ranked list depth every arm returns; the deepest metric cut-off.
DEPTH = 20

#: Passages a store arm asks for; enough to fill DEPTH documents.
HIT_LIMIT = 50

#: Layout version of the built store files. 3: the fixed-schema store with
#: ``andamentum.chunker`` chunks (the v1 and v2 layouts are gone).
STORE_LAYOUT = 3

#: Sampling settings every answering arm sends with every LLM call.
DRIVER_SETTINGS: ModelSettings = {"temperature": 0.0}


def document_content(p: Passage) -> str:
    """A passage as the store would hold it after harvest: a titled markdown page."""
    return f"# {p.title}\n\n{p.text}"


@dataclass
class ArmResult:
    """What a model-driven arm returns: the ranking, its cost, and an answer if it answers."""

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


def _store_name(corpus: Corpus, emb: EmbeddingCache | None) -> str:
    tag = (
        f"{emb.model.replace(':', '_').replace('/', '_')}_{emb.digest[:12]}"
        if emb
        else "noemb"
    )
    return f"store{STORE_LAYOUT}_{corpus.suite}_{corpus.tier}_{corpus.sha256[:12]}_{tag}"


def _expected_meta(corpus: Corpus, emb: EmbeddingCache | None) -> dict:
    return {
        "layout": STORE_LAYOUT,
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
    store: Store
    hybrid: HybridIndex | None  # None when the run has no embeddings
    lexical: HybridIndex
    doc_to_pid: dict[str, str]
    info: StoreInfo | None = None


async def build_store(
    corpus: Corpus, *, workdir: Path, emb: EmbeddingCache | None
) -> BuiltStore:
    """Build (or reopen, after checking its meta) the store; one document per passage."""
    name = _store_name(corpus, emb)
    path = workdir / "stores" / f"{name}.db"
    meta_path = path.with_suffix(".meta.json")
    expected = _expected_meta(corpus, emb)
    lexical = HybridIndex(embedding_model=None, name="lexical")
    hybrid = (
        HybridIndex(embedding_model=emb.model, name="hybrid", embed_fn=emb.get_or_compute)
        if emb is not None
        else None
    )
    indexes = [lexical] + ([hybrid] if hybrid is not None else [])
    reused = meta_path.exists()
    meta: dict[str, Any] = {}
    if reused:
        meta = _check_meta(meta_path, expected)
        if not path.exists():
            raise FileNotFoundError(
                f"{meta_path.name} exists but {path.name} is missing; delete it to rebuild"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        for leftover in (path, path.with_suffix(".db-wal"), path.with_suffix(".db-shm")):
            leftover.unlink(missing_ok=True)  # a half-built store from an interrupted run
    store = await Store.open(path, indexes=indexes)
    try:
        if not reused:
            for p in corpus.passages:
                await store.put(document_content(p), key=p.pid, metadata={"title": p.title})
            report = await store.sync()
            if not report.ok:
                raise RuntimeError(f"indexing failed: {report.failures}")
            meta = _write_meta(meta_path, expected)  # last: marks the store complete
        doc_to_pid = {r.doc_id: r.key for r in await store.list(limit=None) if r.key}
        if len(doc_to_pid) != len(corpus.passages):
            raise ValueError(
                f"store holds {len(doc_to_pid)} keyed documents for {len(corpus.passages)} passages"
            )
    except BaseException:
        await store.close()
        raise
    return BuiltStore(
        store, hybrid, lexical, doc_to_pid, StoreInfo(str(path), STORE_LAYOUT, reused, meta)
    )


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
    store_kind: str  # "store" or "none": which builder the runner uses
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


def _ranked(built: BuiltStore, hits: list[Hit]) -> list[str]:
    """Passage hits -> passage ids, each at the rank of its best passage."""
    return [built.doc_to_pid[d.doc_id] for d in group_by_document(hits)][:DEPTH]


def _index(built: BuiltStore, *, vector: bool, name: str) -> HybridIndex:
    if not vector:
        return built.lexical
    if built.hybrid is None:
        raise ValueError(f"the {name} arm needs an embedding cache")
    return built.hybrid


class Bm25Arm:
    """Plain BM25 baseline: OR-joined terms over the lexical index's chunks."""

    name = "bm25"
    store_kind = "store"
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
        ids = built.lexical.bm25(built.store.db, q.text, k=HIT_LIMIT)
        chunks = built.lexical.chunks(built.store.db, ids)
        hits = [Hit(c.doc_id, c.rev, c.start, c.end, c.text, 0.0) for c in chunks]
        return _ranked(built, hits)


class FusedArm:
    """BM25 + vectors fused by RRF, no reranking: what the reranker adds to."""

    name = "fused"
    store_kind = "store"
    answers = False
    needs_embeddings = True
    reranker = None

    async def rank(
        self,
        built: BuiltStore,
        q: Question,
        emb: EmbeddingCache | None,
        *,
        driver_model: str | None = None,
        run_id: str | None = None,
    ) -> list[str]:
        index = _index(built, vector=True, name=self.name)
        searcher = HybridRerankSearcher(index=index, reranker_model=SUGGESTED_RERANKER_MODEL)
        hits = await searcher.candidates(built.store, q.text, depth=HIT_LIMIT, allowed=None)
        return _ranked(built, hits)


class SearchArm:
    """The store's default searcher (``vector=True``), or its lexical ablation."""

    store_kind = "store"
    answers = False
    reranker = SUGGESTED_RERANKER_MODEL

    def __init__(self, name: str, *, vector: bool) -> None:
        self.name = name
        self.needs_embeddings = vector

    async def rank(
        self,
        built: BuiltStore,
        q: Question,
        emb: EmbeddingCache | None,
        *,
        driver_model: str | None = None,
        run_id: str | None = None,
    ) -> list[str]:
        index = _index(built, vector=self.needs_embeddings, name=self.name)
        searcher = HybridRerankSearcher(index=index, reranker_model=self.reranker)
        hits = await searcher.search(built.store, q.text, limit=HIT_LIMIT, where=None)
        return _ranked(built, hits)


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
        from .answering import answer_closed_book

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


class RagArm:
    """One search, one answer: the default searcher + one synthesis call over the top 10."""

    name = "rag"
    store_kind = "store"
    answers = True
    needs_embeddings = True
    reranker = SUGGESTED_RERANKER_MODEL
    top_k = 10

    async def rank(
        self,
        built: BuiltStore,
        q: Question,
        emb: EmbeddingCache | None,
        *,
        driver_model: str | None = None,
        run_id: str | None = None,
    ) -> ArmResult:
        from .answering import answer_from_hits

        model = _require_driver(self.name, driver_model)
        index = _index(built, vector=True, name=self.name)
        searcher = HybridRerankSearcher(index=index, reranker_model=self.reranker)
        hits = await searcher.search(built.store, q.text, limit=HIT_LIMIT, where=None)
        answer, usage = await answer_from_hits(
            q.text,
            hits[: self.top_k],
            model=model,
            store=built.store,
            model_settings=DRIVER_SETTINGS,
        )
        calls, tin, tout = _usage_counts(usage)
        return ArmResult(
            ranked=_ranked(built, hits),
            answer=answer.text,
            llm_calls=calls,
            tokens_in=tin,
            tokens_out=tout,
            entered_loop=False,
        )


class AgentArm:
    """``AgentSearcher`` over the default searcher, with the default budget.

    A search arm: it is scored on its ranking like the store arms, and it
    records its model calls and trace. With ``vector=False`` it runs over the
    lexical index.
    """

    store_kind = "store"
    answers = False
    llm_driven = True
    reranker = SUGGESTED_RERANKER_MODEL

    def __init__(self, name: str, *, vector: bool) -> None:
        self.name = name
        self.needs_embeddings = vector

    async def rank(
        self,
        built: BuiltStore,
        q: Question,
        emb: EmbeddingCache | None,
        *,
        driver_model: str | None = None,
        run_id: str | None = None,
    ) -> ArmResult:
        model = _require_driver(self.name, driver_model)
        index = _index(built, vector=self.needs_embeddings, name=self.name)
        searcher = AgentSearcher(
            model=model,
            inner=HybridRerankSearcher(index=index, reranker_model=self.reranker),
            budget=AgentBudget(),
            model_settings=DRIVER_SETTINGS,
        )
        hits, run = await searcher.search_with_run(
            built.store, q.text, limit=HIT_LIMIT, where=None
        )
        return ArmResult(
            ranked=_ranked(built, hits),
            llm_calls=run.llm_calls,
            tokens_in=run.tokens_in,
            tokens_out=run.tokens_out,
            entered_loop=run.entered_loop,
            trace=run.trace,
        )


ARMS: dict[str, Arm] = {
    a.name: a
    for a in (
        OracleArm(),
        ShuffledArm(),
        Bm25Arm(),
        FusedArm(),
        SearchArm("default", vector=True),
        SearchArm("lexical", vector=False),
        ClosedBookArm(),
        RagArm(),
        AgentArm("agent", vector=True),
        AgentArm("agent_lexical", vector=False),
    )
}

__all__ = ["ARMS", "DEPTH", "ArmResult", "build_pool", "build_store"]
