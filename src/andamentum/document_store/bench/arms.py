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
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from andamentum.chunker import extract_units

from ..api import DocumentStore
from ..database import get_async_connection
from ..chunker_adapter import Chunk, units_to_chunks
from ..metadata_models import ChunkMetadataFields
from ..search import search_fts5, search_unified
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
    needs_embeddings: bool

    async def rank(
        self, built: BuiltStore, q: Question, emb: EmbeddingCache | None
    ) -> list[str]: ...


class Fts5Arm:
    """Signal 1 alone: BM25 over title + content in ``documents_fts``."""

    name = "fts5"
    needs_embeddings = False

    async def rank(
        self, built: BuiltStore, q: Question, emb: EmbeddingCache | None
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
    needs_embeddings = False

    async def rank(
        self, built: BuiltStore, q: Question, emb: EmbeddingCache | None
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
    needs_embeddings = True

    async def rank(
        self, built: BuiltStore, q: Question, emb: EmbeddingCache | None
    ) -> list[str]:
        if emb is None:
            raise ValueError("the unified arm needs an embedding cache")
        [qvec] = await emb.get_or_compute([q.text])
        results = await search_unified(
            str(built.store.db_path), q.text, limit=DEPTH, query_embedding=qvec
        )
        return [built.doc_to_pid[r.doc_id] for r in results]


ARMS: dict[str, Arm] = {a.name: a for a in (Fts5Arm(), Bm25OrArm(), UnifiedArm())}
