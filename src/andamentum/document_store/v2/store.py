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

import json
import re
import sqlite3
import struct
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
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


@dataclass(frozen=True)
class DocEntry:
    """One document in a ``list_documents`` / ``glob`` result."""

    path: str
    title: str
    n_sections: int


@dataclass(frozen=True)
class OutlineEntry:
    """One section of a document, in reading order."""

    section_id: int
    heading: str
    level: int
    line_from: int
    line_to: int


@lru_cache(maxsize=256)
def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.MULTILINE)


def _regexp(pattern: str | None, text: str | None) -> int:
    if pattern is None or text is None:
        return 0
    return 1 if _compiled(pattern).search(text) else 0


def _like_prefix(prefix: str) -> str:
    """LIKE pattern (ESCAPE '\\') matching paths that start with ``prefix``."""
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped + "%"


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

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with get_connection(self.path) as db:
            db.create_function("regexp", 2, _regexp, deterministic=True)
            yield db

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
                CREATE TABLE IF NOT EXISTS collections (
                  id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, description TEXT);
                CREATE TABLE IF NOT EXISTS evidence (
                  id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, section_id INTEGER NOT NULL,
                  line_from INTEGER NOT NULL, line_to INTEGER NOT NULL,
                  claim TEXT NOT NULL, created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decisions (
                  id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, step INTEGER NOT NULL,
                  kind TEXT NOT NULL, features TEXT NOT NULL, choice TEXT NOT NULL,
                  decided_by TEXT NOT NULL, confidence REAL, outcome TEXT);
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

    def hits(self, section_ids: Sequence[int]) -> list[Hit]:
        """The sections with these ids as hits, in the given order (score 0.0)."""
        return self._hits([(sid, 0.0) for sid in section_ids])

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

    # ── agent tools ───────────────────────────────────────────────────────

    def add_collection(self, *, path: str, description: str) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO collections (path, description) VALUES (?, ?) "
                "ON CONFLICT(path) DO UPDATE SET description = excluded.description",
                (path, description),
            )
            db.commit()

    def collections(self) -> list[tuple[str, str]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT path, description FROM collections ORDER BY path"
            ).fetchall()
        return [(r[0], r[1] or "") for r in rows]

    _DOC_ENTRY = (
        "SELECT d.path, d.title, "
        "(SELECT COUNT(*) FROM sections s WHERE s.doc_id = d.id) FROM documents d "
    )

    def list_documents(self, *, prefix: str = "", limit: int = 100) -> list[DocEntry]:
        with self._connect() as db:
            rows = db.execute(
                self._DOC_ENTRY
                + "WHERE d.path LIKE ? ESCAPE '\\' ORDER BY d.path LIMIT ?",
                (_like_prefix(prefix), limit),
            ).fetchall()
        return [DocEntry(r[0], r[1] or "", r[2]) for r in rows]

    def glob(self, pattern: str, *, limit: int = 100) -> list[DocEntry]:
        with self._connect() as db:
            rows = db.execute(
                self._DOC_ENTRY + "WHERE d.path GLOB ? ORDER BY d.path LIMIT ?",
                (pattern, limit),
            ).fetchall()
        return [DocEntry(r[0], r[1] or "", r[2]) for r in rows]

    def grep(
        self, pattern: str, *, path_prefix: str = "", limit: int = 20
    ) -> list[Hit]:
        """Sections whose text matches the regex; ``score`` is the match count.

        The trigram index narrows candidates to sections containing the
        regex's longest literal run; with no such run (or for alternations)
        every section under ``path_prefix`` is scanned. Either way the regex
        itself is the final filter.
        """
        try:
            compiled = _compiled(pattern)
        except re.error as e:
            raise ValueError(f"invalid regex {pattern!r}: {e}") from e
        literal = longest_literal(pattern)
        sql = (
            "SELECT s.id FROM sections s JOIN documents d ON d.id = s.doc_id "
            "WHERE d.path LIKE ? ESCAPE '\\' AND regexp(?, s.text)"
        )
        params: list[object] = [_like_prefix(path_prefix), pattern]
        if literal is not None:
            sql += " AND s.id IN (SELECT rowid FROM sections_tri WHERE sections_tri MATCH ?)"
            params.append(_phrase(literal))
        sql += " ORDER BY d.path, s.line_from"
        with self._connect() as db:
            ids = [r[0] for r in db.execute(sql, params).fetchall()]
        hits = self._hits([(sid, 0.0) for sid in ids])
        scored = [
            Hit(
                h.section_id,
                h.path,
                h.heading,
                h.line_from,
                h.line_to,
                h.text,
                float(sum(1 for _ in compiled.finditer(h.text))),
            )
            for h in hits
        ]
        scored.sort(key=lambda h: -h.score)  # stable: ties keep path/line order
        return scored[:limit]

    @staticmethod
    def _filter_sql(
        path_prefix: str, meta: dict[str, str] | None
    ) -> tuple[str, list[object]]:
        clauses = ["d.path LIKE ? ESCAPE '\\'"]
        params: list[object] = [_like_prefix(path_prefix)]
        for key, value in (meta or {}).items():
            clauses.append("CAST(json_extract(d.meta, ?) AS TEXT) = ?")
            params += ["$." + json.dumps(key), str(value)]
        return " AND ".join(clauses), params

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        path_prefix: str = "",
        meta: dict[str, str] | None = None,
    ) -> list[Hit]:
        """BM25 over sections (OR-joined terms), restricted by path and metadata."""
        terms = sorted({t.lower() for t in _WORD.findall(query)})
        if not terms:
            return []
        where, params = self._filter_sql(path_prefix, meta)
        match = " OR ".join(_phrase(t) for t in terms)
        with self._connect() as db:
            rows = db.execute(
                "SELECT f.rowid, bm25(sections_fts) FROM sections_fts f "
                "JOIN sections s ON s.id = f.rowid JOIN documents d ON d.id = s.doc_id "
                f"WHERE sections_fts MATCH ? AND {where} ORDER BY bm25(sections_fts) LIMIT ?",
                [match, *params, k],
            ).fetchall()
        return self._hits([(r[0], -float(r[1])) for r in rows])

    def similar(
        self,
        query_embedding: Sequence[float],
        *,
        k: int = 10,
        path_prefix: str = "",
        meta: dict[str, str] | None = None,
    ) -> list[Hit]:
        """Vector leg; with filters, over-fetch (k*10, at most 500) then filter."""
        if self.embedding_dim is None:
            raise ValueError("this store has no vector index")
        filtered = bool(path_prefix) or bool(meta)
        fetch = min(max(k * 10, k), 500) if filtered else k
        blob = struct.pack(f"{len(query_embedding)}f", *query_embedding)
        with self._connect() as db:
            near = db.execute(
                "SELECT rowid, distance FROM sections_vec WHERE embedding MATCH ? "
                "AND k = ? ORDER BY distance",
                (blob, fetch),
            ).fetchall()
            allowed: set[int] | None = None
            if filtered and near:
                where, params = self._filter_sql(path_prefix, meta)
                marks = ",".join("?" * len(near))
                allowed = {
                    r[0]
                    for r in db.execute(
                        "SELECT s.id FROM sections s JOIN documents d ON d.id = s.doc_id "
                        f"WHERE s.id IN ({marks}) AND {where}",
                        [*[r[0] for r in near], *params],
                    )
                }
        kept = [
            (r[0], -float(r[1])) for r in near if allowed is None or r[0] in allowed
        ]
        return self._hits(kept[:k])

    def _doc_id(self, db: sqlite3.Connection, path: str) -> int:
        row = db.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()
        if row is None:
            raise KeyError(path)
        return row[0]

    def outline(self, path: str) -> list[OutlineEntry]:
        with self._connect() as db:
            doc_id = self._doc_id(db, path)
            rows = db.execute(
                "SELECT id, heading, level, line_from, line_to FROM sections "
                "WHERE doc_id = ? ORDER BY line_from, id",
                (doc_id,),
            ).fetchall()
        return [OutlineEntry(r[0], r[1] or "", r[2], r[3], r[4]) for r in rows]

    def read(self, path: str, *, offset: int = 1, limit: int = 60) -> str:
        """Lines ``offset``..``offset+limit-1`` of the document, each as ``N: line``.

        An offset past the last line returns the empty string.
        """
        if offset < 1 or limit < 1:
            raise ValueError("offset and limit must be >= 1")
        with self._connect() as db:
            row = db.execute(
                "SELECT text FROM documents WHERE path = ?", (path,)
            ).fetchone()
        if row is None:
            raise KeyError(path)
        lines = row[0].split("\n")
        chosen = lines[offset - 1 : offset - 1 + limit]
        return "\n".join(f"{offset + i}: {line}" for i, line in enumerate(chosen))

    def note(
        self, *, run_id: str, section_id: int, line_from: int, line_to: int, claim: str
    ) -> int:
        if line_from > line_to:
            raise ValueError("line_from must be <= line_to")
        with self._connect() as db:
            if (
                db.execute(
                    "SELECT 1 FROM sections WHERE id = ?", (section_id,)
                ).fetchone()
                is None
            ):
                raise ValueError(f"unknown section_id {section_id}")
            rid = db.execute(
                "INSERT INTO evidence (run_id, section_id, line_from, line_to, claim, created) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    section_id,
                    line_from,
                    line_to,
                    claim,
                    datetime.now(timezone.utc).isoformat(),
                ),
            ).lastrowid
            db.commit()
        assert rid is not None
        return rid

    def evidence(self, run_id: str) -> list[tuple[int, int, int, str]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT section_id, line_from, line_to, claim FROM evidence "
                "WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def log_decision(
        self,
        *,
        run_id: str,
        step: int,
        kind: str,
        features: dict,
        choice: str,
        decided_by: str,
        confidence: float | None = None,
    ) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO decisions (run_id, step, kind, features, choice, decided_by, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    step,
                    kind,
                    json.dumps(features),
                    choice,
                    decided_by,
                    confidence,
                ),
            )
            db.commit()

    def decisions(self, run_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT step, kind, features, choice, decided_by, confidence, outcome "
                "FROM decisions WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [
            {
                "step": r[0],
                "kind": r[1],
                "features": json.loads(r[2]),
                "choice": r[3],
                "decided_by": r[4],
                "confidence": r[5],
                "outcome": r[6],
            }
            for r in rows
        ]

    def titles(self, paths: Sequence[str]) -> dict[str, str]:
        """``{path: title}`` for the documents among ``paths`` that exist."""
        if not paths:
            return {}
        marks = ",".join("?" * len(paths))
        with self._connect() as db:
            rows = db.execute(
                f"SELECT path, title FROM documents WHERE path IN ({marks})",
                list(paths),
            ).fetchall()
        return {path: title or "" for path, title in rows}

    def section_path(self, section_id: int) -> str:
        with self._connect() as db:
            row = db.execute(
                "SELECT d.path FROM sections s JOIN documents d ON d.id = s.doc_id "
                "WHERE s.id = ?",
                (section_id,),
            ).fetchone()
        if row is None:
            raise KeyError(section_id)
        return row[0]
