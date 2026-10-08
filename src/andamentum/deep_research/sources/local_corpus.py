"""A local corpus: search, citation links and abstracts from a folder, no network.

A *local corpus folder* holds ``corpus.jsonl``, one JSON record per line::

    {"corpus_id": "1001", "title": "...", "abstract": "...",
     "citations": ["1002", "1003"]}

``citations`` lists the corpus ids the record cites (ids that are not in the
corpus are ignored). A title or an abstract may be empty (a record a source holds
without one stays without one; a record with neither is in the corpus and can be
reached only through citation links). :func:`build_index` derives ``corpus.sqlite`` from it once:
an FTS5 index over title and abstract (ranked by BM25, ``porter unicode61``
tokens), the citation edges inside the corpus, and the byte size of the
``corpus.jsonl`` it was built from. The index is derived data and may be deleted
and rebuilt; the source refuses an index whose ``corpus.jsonl`` has changed.

Every record is a ``Work`` with ``local_id = "local:<corpus_id>"`` (the only
identifier it has), its title and abstract, and ``access="abstract"``: the text a
run reads is the title and the abstract. The source never touches the network
and has no rate limit.

A query is turned into BM25 terms the way plain BM25 reads text: its words, each
stemmed by the tokenizer, joined by OR, so a long natural-language question and a
short keyword query are ranked by the same rule. Ties are broken by corpus
order, so a query always gives the same list.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from andamentum.references import Work

from ..types import Candidate
from ._util import snippet_of, squash
from .base import (
    DEFAULT_RESULTS_PER_QUERY,
    FetchedText,
    SearchFilters,
    SourceError,
)

NAME = "local_corpus"
CORPUS_FILE = "corpus.jsonl"
INDEX_FILE = "corpus.sqlite"
LOCAL_PREFIX = "local:"
INDEX_VERSION = "1"
_WORD = re.compile(r"[^\W_]+")


class CorpusRecord(BaseModel):
    """One line of ``corpus.jsonl``."""

    model_config = ConfigDict(extra="forbid")

    corpus_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    title: str = ""
    abstract: str = ""
    citations: list[str] = Field(default_factory=list)


class IndexSummary(BaseModel):
    """What :func:`build_index` built (and what ``corpus.sqlite`` records)."""

    n_records: int
    n_with_abstract: int
    n_citation_edges: int = Field(
        description="Edges whose both ends are in the corpus."
    )
    n_citation_edges_outside: int = Field(
        description="Cited ids that are not in the corpus (not kept)."
    )
    corpus_bytes: int
    sqlite_version: str


def iter_records(corpus_dir: Path) -> Iterator[CorpusRecord]:
    """The records of ``corpus_dir/corpus.jsonl`` in file order; a bad line raises."""
    path = corpus_dir / CORPUS_FILE
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield CorpusRecord.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error


def local_id_of(corpus_id: str) -> str:
    return f"{LOCAL_PREFIX}{corpus_id}"


def work_of(record: CorpusRecord) -> Work:
    """The record as a ``Work`` (title, abstract, ``local:`` id; no licence tag:
    the corpus' own terms are kept beside it, in the folder)."""
    abstract = squash(record.abstract)
    return Work(
        local_id=local_id_of(record.corpus_id),
        title=squash(record.title),
        abstract=abstract,
        access="abstract" if abstract else "metadata",
    )


def match_expression(query: str) -> str | None:
    """The FTS5 expression that ranks by BM25 over the words of *query* (OR of
    each distinct word, case-folded), or None when it has no word."""
    words = list(dict.fromkeys(w.casefold() for w in _WORD.findall(query)))
    if not words:
        return None
    return " OR ".join(f'"{w}"' for w in words)


def build_index(corpus_dir: Path) -> IndexSummary:
    """Derive ``corpus.sqlite`` from ``corpus.jsonl`` (replacing an older index).

    Raises ``FileNotFoundError`` without the corpus file, ``ValueError`` for a
    bad record or a duplicate corpus id, ``RuntimeError`` when this SQLite has no
    FTS5.
    """
    path = corpus_dir / INDEX_FILE
    partial = path.with_name(path.name + ".partial")
    partial.unlink(missing_ok=True)
    corpus_bytes = (corpus_dir / CORPUS_FILE).stat().st_size
    conn = sqlite3.connect(partial)
    try:
        try:
            conn.execute("CREATE VIRTUAL TABLE probe USING fts5(x)")
            conn.execute("DROP TABLE probe")
        except sqlite3.OperationalError as error:
            raise RuntimeError(f"this SQLite has no FTS5: {error}") from error
        conn.executescript(
            """
            CREATE TABLE docs (
                id INTEGER PRIMARY KEY,
                corpus_id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE fts USING fts5(
                text, content='', tokenize='porter unicode61'
            );
            CREATE TABLE cites (
                src TEXT NOT NULL,
                dst TEXT NOT NULL,
                PRIMARY KEY (src, dst)
            ) WITHOUT ROWID;
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        edges: list[tuple[str, str]] = []
        n_records = 0
        n_abstract = 0
        for record in iter_records(corpus_dir):
            title = squash(record.title)
            abstract = squash(record.abstract)
            try:
                cursor = conn.execute(
                    "INSERT INTO docs(corpus_id, title, abstract) VALUES (?, ?, ?)",
                    (record.corpus_id, title, abstract),
                )
            except sqlite3.IntegrityError:
                raise ValueError(f"duplicate corpus id {record.corpus_id}") from None
            conn.execute(
                "INSERT INTO fts(rowid, text) VALUES (?, ?)",
                (cursor.lastrowid, f"{title}\n{abstract}"),
            )
            edges += [(record.corpus_id, cited) for cited in set(record.citations)]
            n_records += 1
            n_abstract += 1 if abstract else 0
        ids = {row[0] for row in conn.execute("SELECT corpus_id FROM docs")}
        inside = [(src, dst) for src, dst in edges if dst in ids and dst != src]
        conn.executemany("INSERT INTO cites(src, dst) VALUES (?, ?)", inside)
        conn.execute("CREATE INDEX cites_dst ON cites(dst)")
        summary = IndexSummary(
            n_records=n_records,
            n_with_abstract=n_abstract,
            n_citation_edges=len(inside),
            n_citation_edges_outside=len(edges) - len(inside),
            corpus_bytes=corpus_bytes,
            sqlite_version=sqlite3.sqlite_version,
        )
        conn.executemany(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            [
                ("index_version", INDEX_VERSION),
                ("summary", summary.model_dump_json()),
            ],
        )
        conn.commit()
    except BaseException:
        conn.close()
        partial.unlink(missing_ok=True)
        raise
    conn.close()
    partial.replace(path)
    return summary


def read_summary(corpus_dir: Path) -> IndexSummary:
    """What the index of *corpus_dir* records of itself; raises ``SourceError`` (kind
    ``config``) when there is no index or it does not belong to ``corpus.jsonl``."""
    corpus = corpus_dir / CORPUS_FILE
    path = corpus_dir / INDEX_FILE
    if not corpus.is_file():
        raise SourceError(NAME, f"{corpus_dir} holds no {CORPUS_FILE}", kind="config")
    if not path.is_file():
        raise SourceError(
            NAME,
            f"{corpus_dir} has no {INDEX_FILE}: build it from {CORPUS_FILE} "
            "(bench download litsearch-local builds it)",
            kind="config",
        )
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        rows = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    except sqlite3.DatabaseError as error:
        raise SourceError(
            NAME, f"{path} is not a corpus index: {error}", kind="config"
        ) from error
    finally:
        conn.close()
    if rows.get("index_version") != INDEX_VERSION or "summary" not in rows:
        raise SourceError(
            NAME, f"{path} is an index of another version: rebuild it", kind="config"
        )
    summary = IndexSummary.model_validate(json.loads(rows["summary"]))
    if summary.corpus_bytes != corpus.stat().st_size:
        raise SourceError(
            NAME,
            f"{path} was built from another {CORPUS_FILE} (size changed): rebuild it",
            kind="config",
        )
    return summary


class LocalCorpusSource:
    """Search, text and citation links over a local corpus folder.

    One object per run; every call opens its own read-only connection (SQLite
    answers in a worker thread, so a slow query does not stall other cells).
    """

    name = NAME

    def __init__(self, *, corpus_dir: Path) -> None:
        self._dir = Path(corpus_dir)
        self._summary = read_summary(self._dir)
        self._uri = f"{(self._dir / INDEX_FILE).as_uri()}?mode=ro"

    @property
    def summary(self) -> IndexSummary:
        return self._summary

    async def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RESULTS_PER_QUERY,
        filters: SearchFilters | None = None,
    ) -> list[Candidate]:
        """The records that rank best for *query* by BM25, best first, at most
        *limit* (*filters* are ignored: a record carries no year or access)."""
        expression = match_expression(query)
        if expression is None or limit < 1:
            return []
        rows = await asyncio.to_thread(self._ranked, expression, limit)
        return [
            self._candidate(row, query=query, rank=position)
            for position, row in enumerate(rows)
        ]

    async def references(
        self, work: Work, *, limit: int = DEFAULT_RESULTS_PER_QUERY
    ) -> list[Candidate]:
        """The records of the corpus that *work* cites."""
        return await self._links(work, "references", limit)

    async def citations(
        self, work: Work, *, limit: int = DEFAULT_RESULTS_PER_QUERY
    ) -> list[Candidate]:
        """The records of the corpus that cite *work*."""
        return await self._links(work, "citations", limit)

    async def fetch_text(self, work: Work) -> FetchedText | None:
        """The title and the abstract of the record, or None for a work that is not
        a record of this corpus or has neither."""
        corpus_id = _corpus_id(work)
        if corpus_id is None:
            return None
        row = await asyncio.to_thread(self._row, corpus_id)
        if row is None:
            return None
        text = f"{row[1]}\n\n{row[2]}".strip()
        if not text:
            return None  # a record the corpus holds with neither title nor abstract
        return FetchedText(
            source=NAME,
            url=local_id_of(corpus_id),
            text=text,
            access="abstract",
        )

    # -- sqlite, in worker threads -----------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._uri, uri=True)

    def _ranked(self, expression: str, limit: int) -> list[tuple[str, str, str]]:
        conn = self._connect()
        try:
            return conn.execute(
                "SELECT d.corpus_id, d.title, d.abstract FROM fts "
                "JOIN docs d ON d.id = fts.rowid WHERE fts MATCH ? "
                "ORDER BY bm25(fts), d.id LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.DatabaseError as error:
            raise SourceError(
                NAME, f"the corpus index could not be read: {error}", kind="parse"
            ) from error
        finally:
            conn.close()

    def _row(self, corpus_id: str) -> tuple[str, str, str] | None:
        conn = self._connect()
        try:
            return conn.execute(
                "SELECT corpus_id, title, abstract FROM docs WHERE corpus_id = ?",
                (corpus_id,),
            ).fetchone()
        finally:
            conn.close()

    def _linked(
        self, corpus_id: str, kind: str, limit: int
    ) -> list[tuple[str, str, str]] | None:
        conn = self._connect()
        try:
            if kind == "references":
                sql = (
                    "SELECT d.corpus_id, d.title, d.abstract FROM cites c "
                    "JOIN docs d ON d.corpus_id = c.dst WHERE c.src = ? "
                    "ORDER BY d.id LIMIT ?"
                )
            else:
                sql = (
                    "SELECT d.corpus_id, d.title, d.abstract FROM cites c "
                    "JOIN docs d ON d.corpus_id = c.src WHERE c.dst = ? "
                    "ORDER BY d.id LIMIT ?"
                )
            if (
                conn.execute(
                    "SELECT 1 FROM docs WHERE corpus_id = ?", (corpus_id,)
                ).fetchone()
                is None
            ):
                return None
            return conn.execute(sql, (corpus_id, limit)).fetchall()
        finally:
            conn.close()

    async def _links(self, work: Work, kind: str, limit: int) -> list[Candidate]:
        corpus_id = _corpus_id(work)
        if corpus_id is None:
            raise SourceError(
                NAME,
                "cannot place the work in the local corpus: it has no local id",
                kind="config",
            )
        rows = await asyncio.to_thread(self._linked, corpus_id, kind, limit)
        if rows is None:
            raise SourceError(
                NAME,
                f"cannot place {local_id_of(corpus_id)} in the local corpus",
                kind="config",
            )
        return [
            self._candidate(row, query=None, rank=position)
            for position, row in enumerate(rows)
        ]

    @staticmethod
    def _candidate(
        row: tuple[str, str, str], *, query: str | None, rank: int
    ) -> Candidate:
        record = CorpusRecord(corpus_id=row[0], title=row[1], abstract=row[2])
        work = work_of(record)
        return Candidate(
            work=work,
            sources=[NAME],
            snippet=snippet_of(work),
            queries=[] if query is None else [query],
            rank=rank,
        )


def _corpus_id(work: Work) -> str | None:
    if work.local_id and work.local_id.startswith(LOCAL_PREFIX):
        return work.local_id[len(LOCAL_PREFIX) :]
    return None
