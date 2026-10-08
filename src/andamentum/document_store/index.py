"""Indexes: disposable tables built from the source tables.

An index owns tables named ``ix_<name>_*``. ``Store.sync`` hands it the ids of
the documents that changed; the index brings its tables in line with each
document's newest revision (or removes the document if it is gone). The store
drops and rebuilds an index whose ``config()`` differs from the one recorded,
so a new chunk size or embedding model is a rebuild, never a data migration.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import struct
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol

from ..chunker import chunk
from .types import Hit

if TYPE_CHECKING:
    from .store import Store

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]

_WORD = re.compile(r"\w+")
# Identifier-like tokens: anything holding a digit, or internal punctuation
# between word characters (NM_000256.3, c.1234G>A, 10.1038/nature12373, v2.1).
_IDENT = re.compile(r"[\w][\w.\-/:>_]*\d[\w.\-/:>_]*|\w+[._/:>\-]\w[\w.\-/:>_]*")
_QUOTED = re.compile(r'"([^"]{3,})"')
_ATX = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
_NAME = re.compile(r"[a-z][a-z0-9_]*")


class Index(Protocol):
    """What ``Store`` needs from an index."""

    name: str

    def config(self) -> dict[str, Any]:
        """Everything that shapes the index's content; a change forces a rebuild."""
        ...

    def create_tables(self, db: sqlite3.Connection) -> None: ...

    def drop_tables(self, db: sqlite3.Connection) -> None: ...

    async def apply(self, store: Store, doc_ids: list[str]) -> list[tuple[str, str]]:
        """Index these documents' newest revisions (or remove them if gone).

        Returns ``[(doc_id, error)]`` for documents that failed. Must be
        idempotent: a sync interrupted part-way repeats it.
        """
        ...

    async def close(self) -> None: ...


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


@lru_cache(maxsize=256)
def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.MULTILINE)


def _regexp(pattern: str | None, text: str | None) -> int:
    if pattern is None or text is None:
        return 0
    return 1 if _compiled(pattern).search(text) else 0


_REGEX_BREAKS = set(".^$*+?{}[]()|\\")
_REGEX_OPTIONAL = set("*?{")


def longest_literal(pattern: str) -> str | None:
    """Longest literal run (>= 3 chars) every match of ``pattern`` must contain.

    Conservative: any alternation, or a quantified group, makes no run
    guaranteed, so the answer is None and the caller scans instead. A
    quantifier makes the character before it optional, so it ends the run
    without that character. Escaped punctuation (``\\.``) is a literal.
    """
    if "|" in pattern:
        return None
    runs: list[str] = []
    cur: list[str] = []

    def end_run() -> None:
        if cur:
            runs.append("".join(cur))
            cur.clear()

    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "\\":
            nxt = pattern[i + 1] if i + 1 < n else ""
            if nxt and not nxt.isalnum():
                cur.append(nxt)
                i += 2
                if i < n and pattern[i] in _REGEX_OPTIONAL:
                    cur.pop()
                    end_run()
                continue
            end_run()  # \d, \w, \b ... or a trailing backslash
            i += 2
            continue
        if c == "[":
            end_run()
            j = i + 1
            if j < n and pattern[j] == "^":
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            while j < n and pattern[j] != "]":
                j += 2 if pattern[j] == "\\" else 1
            i = j + 1
            continue
        if c == ")":
            end_run()
            if i + 1 < n and pattern[i + 1] in _REGEX_OPTIONAL:
                return None  # a quantified group may match zero times
            i += 1
            continue
        if c in _REGEX_OPTIONAL:
            if cur:
                cur.pop()
            end_run()
            i += 1
            continue
        if c in _REGEX_BREAKS:
            end_run()
            i += 1
            continue
        cur.append(c)
        i += 1
    end_run()
    runs = [r for r in runs if len(r) >= 3]
    return max(runs, key=len) if runs else None


def document_title(text: str, metadata: dict[str, Any]) -> str:
    """``metadata["title"]`` if it is a non-empty string, else the first heading."""
    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    m = _ATX.search(text)
    return m.group(1).strip() if m else ""


def context_prefix(title: str, heading_path: Sequence[str]) -> str:
    """``"Title > Heading > Subheading"``, leaving out empty parts and a repeat of the title."""
    parts = [title] if title else []
    for h in heading_path:
        if h and not (not parts[1:] and parts and h == parts[0]):
            parts.append(h)
    return " > ".join(parts)


@dataclass(frozen=True)
class IndexedChunk:
    """One chunk row: where it sits, what the reranker and embedder see."""

    chunk_id: int
    doc_id: str
    rev: int
    start: int
    end: int
    heading_path: tuple[str, ...]
    prefix: str
    text: str

    @property
    def scored_text(self) -> str:
        """The text the embedder and the reranker see: context prefix + chunk."""
        return f"{self.prefix}\n\n{self.text}" if self.prefix else self.text


class HybridIndex:
    """Chunks with a porter BM25 index, a trigram index and (optionally) vectors.

    ``embedding_model`` names an Ollama embedding model; ``None`` builds a
    lexical-only index with no vectors. ``embed_fn`` replaces the Ollama
    client (tests, or another embedding service); the model name is still
    recorded in the config, so a change of model rebuilds the index.
    """

    def __init__(
        self,
        *,
        embedding_model: str | None,
        max_chars: int = 2000,
        name: str = "hybrid",
        embed_fn: EmbedFn | None = None,
    ) -> None:
        if not _NAME.fullmatch(name):
            raise ValueError(f"index name {name!r} must match {_NAME.pattern}")
        if embed_fn is not None and embedding_model is None:
            raise ValueError("embed_fn needs an embedding_model name to record")
        self.name = name
        self.embedding_model = embedding_model
        self.max_chars = max_chars
        self._embed_fn = embed_fn
        self._embedder: Any = None  # OllamaEmbedder, created on first use

    def _t(self, suffix: str) -> str:
        return f"ix_{self.name}_{suffix}"

    def config(self) -> dict[str, Any]:
        return {
            "type": "hybrid",
            "version": 1,
            "embedding_model": self.embedding_model,
            "max_chars": self.max_chars,
        }

    def create_tables(self, db: sqlite3.Connection) -> None:
        db.execute(
            f"CREATE TABLE {self._t('docs')} (doc_id TEXT PRIMARY KEY, rev INTEGER NOT NULL, "
            "content_hash TEXT NOT NULL)"
        )
        db.execute(
            f"CREATE TABLE {self._t('chunks')} (id INTEGER PRIMARY KEY, doc_id TEXT NOT NULL, "
            "rev INTEGER NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL, "
            "heading_path TEXT NOT NULL, prefix TEXT NOT NULL, text TEXT NOT NULL)"
        )
        db.execute(f"CREATE INDEX {self._t('chunks_doc')} ON {self._t('chunks')}(doc_id)")
        db.execute(
            f"CREATE VIRTUAL TABLE {self._t('fts')} USING fts5(prefix, text, "
            "tokenize='porter unicode61')"
        )
        db.execute(f"CREATE VIRTUAL TABLE {self._t('tri')} USING fts5(text, tokenize='trigram')")
        db.execute(f"CREATE TABLE {self._t('state')} (name TEXT PRIMARY KEY, value TEXT)")

    def drop_tables(self, db: sqlite3.Connection) -> None:
        for suffix in ("vec", "fts", "tri", "chunks", "docs", "state"):
            db.execute(f"DROP TABLE IF EXISTS {self._t(suffix)}")

    async def close(self) -> None:
        if self._embedder is not None:
            await self._embedder.close()
            self._embedder = None

    # ── embedding ───────────────────────────────────────────────────────

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self.embedding_model is None:
            raise RuntimeError(f"index {self.name!r} has no embedding model")
        if not texts:
            return []
        if self._embed_fn is not None:
            vectors = await self._embed_fn(texts)
        else:
            if self._embedder is None:
                from ..core.embeddings import OllamaEmbedder

                self._embedder = OllamaEmbedder(model=self.embedding_model)
            vectors = await self._embedder.embed_batch(texts)
        if len(vectors) != len(texts):
            raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(texts)} texts")
        return vectors

    def _ensure_vec(self, db: sqlite3.Connection, dim: int) -> None:
        row = db.execute(f"SELECT value FROM {self._t('state')} WHERE name = 'dim'").fetchone()
        if row is None:
            db.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._t('vec')} USING "
                f"vec0(embedding float[{dim}])"
            )
            db.execute(f"INSERT INTO {self._t('state')} VALUES ('dim', ?)", (str(dim),))
        elif int(row[0]) != dim:
            raise RuntimeError(f"embedding has {dim} dimensions, the index holds {row[0]}")

    def _has_vec(self, db: sqlite3.Connection) -> bool:
        return (
            db.execute(f"SELECT 1 FROM {self._t('state')} WHERE name = 'dim'").fetchone()
            is not None
        )

    # ── building ────────────────────────────────────────────────────────

    async def apply(self, store: Store, doc_ids: list[str]) -> list[tuple[str, str]]:
        db = store.db
        failures: list[tuple[str, str]] = []
        # (action, doc_id, rev, content_hash, chunks); action is "remove" (the
        # document is gone), "rev" (same content, only the revision moved) or "index".
        planned: list[tuple[str, str, int, str, list[IndexedChunk]]] = []
        for doc_id in doc_ids:
            doc = await store.get(doc_id)
            if doc is None:  # deleted or purged
                planned.append(("remove", doc_id, 0, "", []))
                continue
            title = document_title(doc.text, doc.metadata)
            content_hash = hashlib.sha256(f"{title}\x00{doc.text}".encode()).hexdigest()
            row = db.execute(
                f"SELECT content_hash FROM {self._t('docs')} WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if row is not None and row[0] == content_hash:
                planned.append(("rev", doc_id, doc.rev, content_hash, []))
                continue
            try:
                pieces = chunk(doc.text, max_chars=self.max_chars)
            except Exception as exc:  # recorded and reported by Store.sync
                failures.append((doc_id, f"{type(exc).__name__}: {exc}"))
                continue
            chunks = [
                IndexedChunk(
                    chunk_id=0,
                    doc_id=doc_id,
                    rev=doc.rev,
                    start=p.start,
                    end=p.end,
                    heading_path=tuple(p.heading_path),
                    prefix=context_prefix(title, p.heading_path),
                    text=p.text,
                )
                for p in pieces
            ]
            planned.append(("index", doc_id, doc.rev, content_hash, chunks))

        to_embed = [c for _, _, _, _, cs in planned for c in cs]
        vectors: list[list[float]] = []
        if self.embedding_model is not None and to_embed:
            # An embedding failure is raised, never recorded per document: it
            # means the service is down or wrong, and the sync keeps its position.
            vectors = await self.embed([c.scored_text for c in to_embed])
        vec_of = {id(c): v for c, v in zip(to_embed, vectors, strict=False)}

        with store._write() as w:
            if vectors:
                self._ensure_vec(w, len(vectors[0]))
            for action, doc_id, rev, content_hash, chunks in planned:
                if action == "rev":
                    w.execute(f"UPDATE {self._t('docs')} SET rev = ? WHERE doc_id = ?", (rev, doc_id))
                    w.execute(f"UPDATE {self._t('chunks')} SET rev = ? WHERE doc_id = ?", (rev, doc_id))
                    continue
                self._remove(w, doc_id)
                if action == "remove":
                    continue
                w.execute(
                    f"INSERT INTO {self._t('docs')} VALUES (?, ?, ?)", (doc_id, rev, content_hash)
                )
                for c in chunks:
                    cid = w.execute(
                        f"INSERT INTO {self._t('chunks')} (doc_id, rev, start, end, heading_path, "
                        "prefix, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (doc_id, rev, c.start, c.end, json.dumps(list(c.heading_path)), c.prefix, c.text),
                    ).lastrowid
                    w.execute(
                        f"INSERT INTO {self._t('fts')} (rowid, prefix, text) VALUES (?, ?, ?)",
                        (cid, c.prefix, c.text),
                    )
                    w.execute(f"INSERT INTO {self._t('tri')} (rowid, text) VALUES (?, ?)", (cid, c.text))
                    vec = vec_of.get(id(c))
                    if vec is not None:
                        w.execute(
                            f"INSERT INTO {self._t('vec')} (rowid, embedding) VALUES (?, ?)",
                            (cid, struct.pack(f"{len(vec)}f", *vec)),
                        )
        return failures

    def _remove(self, db: sqlite3.Connection, doc_id: str) -> None:
        ids = [r[0] for r in db.execute(f"SELECT id FROM {self._t('chunks')} WHERE doc_id = ?", (doc_id,))]
        if ids:
            marks = ",".join("?" * len(ids))
            db.execute(f"DELETE FROM {self._t('fts')} WHERE rowid IN ({marks})", ids)
            db.execute(f"DELETE FROM {self._t('tri')} WHERE rowid IN ({marks})", ids)
            if self._has_vec(db):
                db.execute(f"DELETE FROM {self._t('vec')} WHERE rowid IN ({marks})", ids)
            db.execute(f"DELETE FROM {self._t('chunks')} WHERE doc_id = ?", (doc_id,))
        db.execute(f"DELETE FROM {self._t('docs')} WHERE doc_id = ?", (doc_id,))

    # ── reading (for searchers and agent tools) ─────────────────────────

    @staticmethod
    def _allowed(allowed: tuple[str, list[object]] | None) -> tuple[str, list[object]]:
        """SQL restricting ``c.doc_id`` to the documents a ``where`` selected."""
        if allowed is None:
            return "", []
        sql, params = allowed
        return f" AND c.doc_id IN ({sql})", list(params)

    def bm25(
        self, db: sqlite3.Connection, query: str, *, k: int,
        allowed: tuple[str, list[object]] | None = None,
    ) -> list[int]:
        """Chunk ids by BM25 over the porter index; query terms are OR-joined."""
        terms = sorted({t.lower() for t in _WORD.findall(query)})
        if not terms:
            return []
        extra, params = self._allowed(allowed)
        rows = db.execute(
            f"SELECT f.rowid FROM {self._t('fts')} f JOIN {self._t('chunks')} c ON c.id = f.rowid "
            f"WHERE {self._t('fts')} MATCH ?{extra} ORDER BY bm25({self._t('fts')}) LIMIT ?",
            [" OR ".join(_phrase(t) for t in terms), *params, k],
        ).fetchall()
        return [r[0] for r in rows]

    def exact(
        self, db: sqlite3.Connection, query: str, *, k: int,
        allowed: tuple[str, list[object]] | None = None,
    ) -> list[int]:
        """Chunk ids matching the query's identifiers literally (trigram index)."""
        ids = identifiers(query)
        if not ids:
            return []
        extra, params = self._allowed(allowed)
        rows = db.execute(
            f"SELECT f.rowid FROM {self._t('tri')} f JOIN {self._t('chunks')} c ON c.id = f.rowid "
            f"WHERE {self._t('tri')} MATCH ?{extra} ORDER BY bm25({self._t('tri')}) LIMIT ?",
            [" OR ".join(_phrase(s) for s in ids), *params, k],
        ).fetchall()
        return [r[0] for r in rows]

    def nearest(
        self, db: sqlite3.Connection, embedding: Sequence[float], *, k: int,
        allowed: tuple[str, list[object]] | None = None,
    ) -> list[int]:
        """Chunk ids nearest to ``embedding`` (L2 distance, as vec0 uses).

        With a filter, the distance is computed over exactly the chunks of the
        allowed documents, so a small allowed set is never crowded out by
        nearer chunks of other documents.
        """
        if not self._has_vec(db):
            return []
        blob = struct.pack(f"{len(embedding)}f", *embedding)
        if allowed is None:
            return [
                r[0]
                for r in db.execute(
                    f"SELECT rowid FROM {self._t('vec')} WHERE embedding MATCH ? AND k = ? "
                    "ORDER BY distance",
                    (blob, k),
                )
            ]
        extra, params = self._allowed(allowed)
        return [
            r[0]
            for r in db.execute(
                f"SELECT v.rowid FROM {self._t('vec')} v JOIN {self._t('chunks')} c "
                f"ON c.id = v.rowid WHERE 1{extra} "
                "ORDER BY vec_distance_l2(v.embedding, ?), v.rowid LIMIT ?",
                [*params, blob, k],
            )
        ]

    def chunks(self, db: sqlite3.Connection, chunk_ids: Sequence[int]) -> list[IndexedChunk]:
        """These chunks, in the given order (ids that no longer exist are left out)."""
        if not chunk_ids:
            return []
        marks = ",".join("?" * len(chunk_ids))
        rows = {
            r[0]: r
            for r in db.execute(
                f"SELECT id, doc_id, rev, start, end, heading_path, prefix, text "
                f"FROM {self._t('chunks')} WHERE id IN ({marks})",
                list(chunk_ids),
            )
        }
        return [
            IndexedChunk(r[0], r[1], r[2], r[3], r[4], tuple(json.loads(r[5])), r[6], r[7])
            for r in (rows[i] for i in chunk_ids if i in rows)
        ]

    def document_chunks(self, db: sqlite3.Connection, doc_id: str) -> list[IndexedChunk]:
        """A document's chunks in reading order."""
        ids = [
            r[0]
            for r in db.execute(
                f"SELECT id FROM {self._t('chunks')} WHERE doc_id = ? ORDER BY start", (doc_id,)
            )
        ]
        return self.chunks(db, ids)

    def grep(
        self, db: sqlite3.Connection, pattern: str, *, limit: int,
        allowed: tuple[str, list[object]] | None = None,
    ) -> list[tuple[IndexedChunk, int]]:
        """Chunks whose text matches the regex, with the match count, most matches first.

        The trigram index narrows candidates to chunks holding the regex's
        longest literal run; without one every chunk is scanned. The regex is
        the final filter either way.
        """
        try:
            compiled = _compiled(pattern)
        except re.error as e:
            raise ValueError(f"invalid regex {pattern!r}: {e}") from e
        db.create_function("regexp", 2, _regexp, deterministic=True)
        extra, params = self._allowed(allowed)
        sql = f"SELECT c.id FROM {self._t('chunks')} c WHERE regexp(?, c.text){extra}"
        args: list[object] = [pattern, *params]
        literal = longest_literal(pattern)
        if literal is not None:
            sql += f" AND c.id IN (SELECT rowid FROM {self._t('tri')} WHERE {self._t('tri')} MATCH ?)"
            args.append(_phrase(literal))
        sql += " ORDER BY c.doc_id, c.start"
        found = self.chunks(db, [r[0] for r in db.execute(sql, args)])
        scored = [(c, sum(1 for _ in compiled.finditer(c.text))) for c in found]
        scored.sort(key=lambda cs: -cs[1])  # stable: ties keep document order
        return scored[:limit]


def to_hit(c: IndexedChunk, score: float) -> Hit:
    return Hit(
        doc_id=c.doc_id, rev=c.rev, start=c.start, end=c.end, text=c.text,
        score=score, heading_path=c.heading_path,
    )
