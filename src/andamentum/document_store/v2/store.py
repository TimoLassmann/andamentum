"""The v2 store: one SQLite file, sections as the index unit, round-0 search.

Round 0 (agentic-retrieval plan, rollout step 1) answers a question with no
LLM call: three rankings over sections — BM25 on a porter-stemmed FTS5 index,
exact matching of identifiers on a trigram FTS5 index, and sqlite-vec nearest
neighbours — fused by reciprocal rank (k=60) to the top 50, then optionally
reordered by a cross-encoder. Every hit carries ``path:line_from-line_to``.

Differences from the current store that this layout tests:

- The keyword leg joins query terms with OR and lets BM25 weigh them. The
  current store's FTS signal joins them with an implicit AND, which a
  natural-language question almost never satisfies.
- Exact identifiers (accessions, versions, quoted titles) get their own leg on
  a trigram index instead of being phrase-quoted into the BM25 query.
- Everything ranks sections; there is no separate document-level signal and
  no temporal clustering.

Embedding is injected as ``embed_fn`` (texts -> vectors), so the store never
names or calls a model itself; the reranker model is a required argument.
"""

from __future__ import annotations

import re
import sqlite3
import struct
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..connection import get_connection
from .sections import split_sections

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]

#: Reciprocal-rank-fusion constant and fused depth (plan section 4).
RRF_K = 60
FUSED_DEPTH = 50
#: Section size cap in characters (plan section 10 leaves it open).
SECTION_MAX_CHARS = 2000

_WORD = re.compile(r"\w+")
# Identifier-like tokens: anything holding a digit, or internal punctuation
# between word characters (NM_000256.3, c.1234G>A, 10.1038/nature12373, v2.1).
_IDENT = re.compile(r"[\w][\w.\-/:>_]*\d[\w.\-/:>_]*|\w+[._/:>\-]\w[\w.\-/:>_]*")
_QUOTED = re.compile(r'"([^"]{3,})"')


@dataclass(frozen=True)
class Hit:
    section_id: int
    path: str
    heading: str
    line_from: int
    line_to: int
    text: str
    score: float  # fused RRF score, or the reranker's score after reranking

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.line_from}-{self.line_to}"


def identifiers(query: str) -> list[str]:
    """Exact strings worth matching literally: quoted phrases and ID-like tokens."""
    found = [m.group(1) for m in _QUOTED.finditer(query)]
    found += [m.group(0).rstrip(".:-/") for m in _IDENT.finditer(query)]
    out: list[str] = []
    for s in found:
        if len(s) >= 3 and s not in out:  # the trigram index needs 3 characters
            out.append(s)
    return out


def _phrase(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


def rrf(
    rankings: Sequence[Sequence[int]], *, k: int = RRF_K, depth: int = FUSED_DEPTH
) -> list[tuple[int, float]]:
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, sid in enumerate(ranking):
            score[sid] = score.get(sid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:depth]


class SectionStore:
    def __init__(self, path: Path, *, embedding_dim: int | None) -> None:
        """``embedding_dim=None`` builds a lexical-only store with no vector index."""
        self.path = path
        self.embedding_dim = embedding_dim

    def _connect(self):  # noqa: ANN202 — context manager from connection.py
        return get_connection(self.path)

    def create(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                  id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, title TEXT,
                  meta TEXT, text TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sections (
                  id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL REFERENCES documents(id),
                  heading TEXT, level INTEGER, line_from INTEGER, line_to INTEGER,
                  text TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS sections_doc ON sections(doc_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
                  heading, text, content='sections', content_rowid='id',
                  tokenize='porter unicode61');
                CREATE VIRTUAL TABLE IF NOT EXISTS sections_tri USING fts5(
                  text, content='sections', content_rowid='id', tokenize='trigram');
                """
            )
            if self.embedding_dim is not None:
                db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS sections_vec USING "
                    f"vec0(embedding float[{self.embedding_dim}])"
                )
            db.commit()

    async def add_document(
        self,
        *,
        path: str,
        title: str,
        text: str,
        meta: str | None = None,
        embed_fn: EmbedFn | None,
    ) -> int:
        if (embed_fn is None) != (self.embedding_dim is None):
            raise ValueError(
                "embed_fn must be given exactly when the store has a vector index"
            )
        sections = split_sections(text, max_chars=SECTION_MAX_CHARS)
        vectors = await embed_fn([s.text for s in sections]) if embed_fn else []
        with self._connect() as db:
            doc_id = db.execute(
                "INSERT INTO documents (path, title, meta, text) VALUES (?, ?, ?, ?)",
                (path, title, meta, text),
            ).lastrowid
            for n, s in enumerate(sections):
                sid = db.execute(
                    "INSERT INTO sections (doc_id, heading, level, line_from, line_to, text) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (doc_id, s.heading, s.level, s.line_from, s.line_to, s.text),
                ).lastrowid
                db.execute(
                    "INSERT INTO sections_fts (rowid, heading, text) VALUES (?, ?, ?)",
                    (sid, s.heading, s.text),
                )
                db.execute(
                    "INSERT INTO sections_tri (rowid, text) VALUES (?, ?)",
                    (sid, s.text),
                )
                if vectors:
                    vec = vectors[n]
                    if len(vec) != self.embedding_dim:
                        raise ValueError(
                            f"vector has {len(vec)} dims, store expects {self.embedding_dim}"
                        )
                    db.execute(
                        "INSERT INTO sections_vec (rowid, embedding) VALUES (?, ?)",
                        (sid, struct.pack(f"{len(vec)}f", *vec)),
                    )
            db.commit()
        assert doc_id is not None
        return doc_id

    # ── the three round-0 legs ────────────────────────────────────────────

    def bm25(self, query: str, *, k: int = FUSED_DEPTH) -> list[int]:
        terms = sorted({t.lower() for t in _WORD.findall(query)})
        if not terms:
            return []
        match = " OR ".join(_phrase(t) for t in terms)
        with self._connect() as db:
            rows = db.execute(
                "SELECT rowid FROM sections_fts WHERE sections_fts MATCH ? "
                "ORDER BY bm25(sections_fts) LIMIT ?",
                (match, k),
            ).fetchall()
        return [r[0] for r in rows]

    def exact(self, query: str, *, k: int = FUSED_DEPTH) -> list[int]:
        ids = identifiers(query)
        if not ids:
            return []
        match = " OR ".join(_phrase(s) for s in ids)
        with self._connect() as db:
            rows = db.execute(
                "SELECT rowid FROM sections_tri WHERE sections_tri MATCH ? "
                "ORDER BY bm25(sections_tri) LIMIT ?",
                (match, k),
            ).fetchall()
        return [r[0] for r in rows]

    def nearest(
        self, query_embedding: Sequence[float], *, k: int = FUSED_DEPTH
    ) -> list[int]:
        if self.embedding_dim is None:
            raise ValueError("this store has no vector index")
        with self._connect() as db:
            rows = db.execute(
                "SELECT rowid FROM sections_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (struct.pack(f"{len(query_embedding)}f", *query_embedding), k),
            ).fetchall()
        return [r[0] for r in rows]

    # ── round 0 ───────────────────────────────────────────────────────────

    def round0(
        self,
        query: str,
        *,
        query_embedding: Sequence[float] | None,
        use_exact: bool = True,
        vector_weight: int = 1,
    ) -> list[Hit]:
        """Fused top sections for ``query``; vector leg only when an embedding is given.

        ``vector_weight`` counts the vector ranking that many times in the
        fusion. The defaults are the plan as written (all three legs, equal
        weight); other settings are chosen on the benchmark's dev split.
        """
        rankings = [self.bm25(query)]
        if use_exact:
            rankings.append(self.exact(query))
        if query_embedding is not None:
            rankings += [self.nearest(query_embedding)] * vector_weight
        fused = rrf(rankings)
        return self._hits(fused)

    async def rerank(
        self, query: str, hits: list[Hit], *, reranker_model: str
    ) -> list[Hit]:
        """Reorder ``hits`` by a cross-encoder score; the score replaces the RRF score."""
        from ..reranking import rerank

        if not hits:
            return []
        scores = await rerank(
            query, [h.text for h in hits], reranker_model=reranker_model
        )
        scored = [
            Hit(h.section_id, h.path, h.heading, h.line_from, h.line_to, h.text, s)
            for h, s in zip(hits, scores, strict=True)
        ]
        return sorted(scored, key=lambda h: (-h.score, h.section_id))

    def _hits(self, fused: list[tuple[int, float]]) -> list[Hit]:
        if not fused:
            return []
        marks = ",".join("?" * len(fused))
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            rows = {
                r["id"]: r
                for r in db.execute(
                    "SELECT s.id, d.path, s.heading, s.line_from, s.line_to, s.text "
                    f"FROM sections s JOIN documents d ON d.id = s.doc_id WHERE s.id IN ({marks})",
                    [sid for sid, _ in fused],
                )
            }
        return [
            Hit(
                sid,
                rows[sid]["path"],
                rows[sid]["heading"],
                rows[sid]["line_from"],
                rows[sid]["line_to"],
                rows[sid]["text"],
                score,
            )
            for sid, score in fused
        ]
