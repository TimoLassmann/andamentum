"""The research store: one SQLite file and a folder of blobs.

Everything a run learns lives here, so a second run starts from the first
one's answers: the HTTP cache every source reads and writes through, the works
found (merged by any identifier), the verbatim passages, and each run's rows
so a whole run can be inspected afterwards.

``ResearchStore(path)`` is persistent: ``path`` is the database file and the
blobs go in a ``blobs/`` folder beside it. ``ResearchStore.in_memory()`` keeps
everything in the process and leaves nothing behind. There is no default
location; the caller names the path.

Text bigger than ``INLINE_BLOB_LIMIT`` bytes goes to a file named by its
sha256; smaller text sits in an SQLite table under the same name. Either way a
``*_blob`` column holds the sha256 and :meth:`ResearchStore.get_blob` returns
the bytes.

One writer per process: every public method takes the store's lock, so the
event loop and a worker thread cannot interleave a write. All SQL is here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, JsonValue

from andamentum.references import (
    LICENCE_NO_REDISTRIBUTE,
    Work,
    merge_works,
    works_compatible,
)
from andamentum.references.works import ID_FIELDS, title_key

from .errors import ResearchError
from .types import Passage

SCHEMA_VERSION = 7

TTL_SEARCH_S = 7 * 86_400
TTL_METADATA_S = 30 * 86_400
TTL_FULL_TEXT_S = 180 * 86_400

INLINE_BLOB_LIMIT = 16 * 1024

# Hard cap on the bytes of response bodies the HTTP cache holds. Enforced on
# every write: expired rows go first, then the least recently used.
HTTP_CACHE_MAX_BYTES = 200 * 1024 * 1024

RunItemKind = Literal["subquestion", "finding", "flag", "log"]

_DIGEST = re.compile(r"[0-9a-f]{64}")
_WORK_COLUMNS = (
    "work_id",
    "doi",
    "arxiv_id",
    "pmid",
    "pmcid",
    "openalex_id",
    "s2_id",
    "local_id",
    "url",
    "title",
    "authors_json",
    "year",
    "venue",
    "abstract_blob",
    "oa_url",
    "access",
    "retracted",
    "licence",
)
# The columns that identify a work (each has a unique index), by the kind
# `Work.identity_keys` gives them.
_KEY_COLUMNS = (*ID_FIELDS, "url")
_KIND_COLUMN = {
    "doi": "doi",
    "arxiv": "arxiv_id",
    "pmid": "pmid",
    "pmcid": "pmcid",
    "openalex": "openalex_id",
    "s2": "s2_id",
    "local": "local_id",
    "url": "url",
}

_SCHEMA_V1 = """
CREATE TABLE http_cache (
    key TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    url TEXT NOT NULL,
    status INTEGER NOT NULL,
    body_blob TEXT NOT NULL,
    fetched_at REAL NOT NULL,
    ttl_s INTEGER NOT NULL,
    licence TEXT NOT NULL DEFAULT ''
);
CREATE TABLE works (
    work_id TEXT PRIMARY KEY,
    doi TEXT, arxiv_id TEXT, pmid TEXT, pmcid TEXT,
    openalex_id TEXT, s2_id TEXT, url TEXT,
    title TEXT NOT NULL DEFAULT '',
    authors_json TEXT NOT NULL DEFAULT '[]',
    year INTEGER,
    venue TEXT NOT NULL DEFAULT '',
    abstract_blob TEXT,
    oa_url TEXT,
    access TEXT NOT NULL DEFAULT 'metadata'
        CHECK (access IN ('full_text', 'abstract', 'metadata', 'unreadable')),
    retracted TEXT NOT NULL DEFAULT 'unknown'
        CHECK (retracted IN ('no', 'retracted', 'correction', 'concern', 'unknown')),
    licence TEXT NOT NULL DEFAULT '',
    title_key TEXT,
    first_seen REAL NOT NULL,
    last_checked REAL NOT NULL
);
CREATE UNIQUE INDEX works_doi ON works(doi) WHERE doi IS NOT NULL;
CREATE UNIQUE INDEX works_arxiv_id ON works(arxiv_id) WHERE arxiv_id IS NOT NULL;
CREATE UNIQUE INDEX works_pmid ON works(pmid) WHERE pmid IS NOT NULL;
CREATE UNIQUE INDEX works_pmcid ON works(pmcid) WHERE pmcid IS NOT NULL;
CREATE UNIQUE INDEX works_openalex_id ON works(openalex_id) WHERE openalex_id IS NOT NULL;
CREATE UNIQUE INDEX works_s2_id ON works(s2_id) WHERE s2_id IS NOT NULL;
CREATE UNIQUE INDEX works_url ON works(url) WHERE url IS NOT NULL;
CREATE INDEX works_title_key ON works(title_key) WHERE title_key IS NOT NULL;
CREATE TABLE work_aliases (
    alias_id TEXT PRIMARY KEY,
    work_id TEXT NOT NULL
);
CREATE INDEX work_aliases_work ON work_aliases(work_id);
CREATE TABLE passages (
    passage_id TEXT PRIMARY KEY,
    work_id TEXT NOT NULL,
    "start" INTEGER NOT NULL,
    "end" INTEGER NOT NULL,
    text_blob TEXT NOT NULL,
    retrieved_at REAL NOT NULL
);
CREATE INDEX passages_work ON passages(work_id);
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    mode TEXT NOT NULL,
    config_json TEXT NOT NULL,
    models_json TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    stats_json TEXT,
    degraded_json TEXT
);
CREATE TABLE run_items (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    kind TEXT NOT NULL,
    ref_id TEXT NOT NULL,
    data_json TEXT NOT NULL,
    PRIMARY KEY (run_id, kind, ref_id)
);
"""

# The cache's size and recency, for the size cap: ``size`` is the body's bytes
# (backfilled from the blobs on open), ``last_used`` the last read or write.
_SCHEMA_V2 = """
ALTER TABLE http_cache ADD COLUMN size INTEGER NOT NULL DEFAULT 0;
ALTER TABLE http_cache ADD COLUMN last_used REAL NOT NULL DEFAULT 0;
UPDATE http_cache SET last_used = fetched_at;
CREATE INDEX http_cache_used ON http_cache(last_used);
"""

# The decision cache: an answer by (model digest, request hash, state hash), and
# whether a model's answers to identical requests were found identical.
_SCHEMA_V3 = """
CREATE TABLE decision_cache (
    digest TEXT NOT NULL,
    question_hash TEXT NOT NULL,
    state_hash TEXT NOT NULL,
    answers_json TEXT NOT NULL,
    PRIMARY KEY (digest, question_hash, state_hash)
);
CREATE TABLE decision_determinism (
    digest TEXT PRIMARY KEY,
    deterministic INTEGER NOT NULL,
    detail TEXT NOT NULL,
    checked_at REAL NOT NULL
);
"""

# The identifier of a record in a local corpus (``Work.local_id``).
_SCHEMA_V4 = """
ALTER TABLE works ADD COLUMN local_id TEXT;
CREATE UNIQUE INDEX works_local_id ON works(local_id) WHERE local_id IS NOT NULL;
"""

# Embedding vectors by (model digest or name, sha256 of the text), float32 bytes.
_SCHEMA_V5 = """
CREATE TABLE embeddings (
    model TEXT NOT NULL,
    text_sha TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (model, text_sha)
);
"""

# The decision-answer cache, the determinism check and the embedding cache served
# strategies the engine no longer has; their tables go.
_SCHEMA_V6 = """
DROP TABLE IF EXISTS decision_cache;
DROP TABLE IF EXISTS decision_determinism;
DROP TABLE IF EXISTS embeddings;
"""

# The audit trail of a decision model, which no part of the package has any more.
_SCHEMA_V7 = """
DROP TABLE IF EXISTS decisions;
"""

# Migrations by the version they produce. Development: no shims for old files.
_MIGRATIONS: dict[int, str] = {
    1: _SCHEMA_V1,
    2: _SCHEMA_V2,
    3: _SCHEMA_V3,
    4: _SCHEMA_V4,
    5: _SCHEMA_V5,
    6: _SCHEMA_V6,
    7: _SCHEMA_V7,
}


class StoreError(ResearchError):
    """The store cannot do what was asked: a missing blob, an unknown run, a
    database written by a newer version."""


class CachedResponse(BaseModel):
    key: str
    source: str
    url: str
    status: int
    body: bytes
    fetched_at: float
    ttl_s: int
    licence: str = ""


class CacheStats(BaseModel):
    """What the HTTP cache holds: bytes of bodies, rows per source, the age range."""

    bytes: int = 0
    rows: int = 0
    by_source: dict[str, int] = Field(default_factory=dict)
    oldest: float | None = None  # fetched_at of the oldest row
    newest: float | None = None
    max_bytes: int = HTTP_CACHE_MAX_BYTES


class CachePrune(BaseModel):
    """What :meth:`ResearchStore.prune_cache` removed."""

    expired: int = 0
    evicted: int = 0
    freed_bytes: int = 0
    blobs_deleted: int = 0


class CacheImportCounts(BaseModel):
    """What happened to the rows of one source in :meth:`ResearchStore.import_cache`."""

    imported: int = 0
    expired: int = 0
    kept_existing: int = 0
    unreadable: int = 0


class CacheImport(BaseModel):
    """The outcome of importing one database file's HTTP cache."""

    path: str
    by_source: dict[str, CacheImportCounts] = {}
    problem: str = ""

    @property
    def imported(self) -> int:
        return sum(c.imported for c in self.by_source.values())


class RunRecord(BaseModel):
    run_id: str
    question: str
    mode: str
    config: dict[str, JsonValue]
    models: dict[str, JsonValue]
    started_at: float
    finished_at: float | None = None
    stats: dict[str, JsonValue] = {}
    degraded: list[str] = []


class RunItem(BaseModel):
    run_id: str
    kind: str
    ref_id: str
    data: dict[str, JsonValue]


def cache_key(
    source: str, url: str, params: Mapping[str, JsonValue] | None = None
) -> str:
    """The cache key of one request: sha256 over the source, URL and sorted parameters."""
    canonical = json.dumps(
        [source, url, sorted((k, str(v)) for k, v in (params or {}).items())],
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResearchStore:
    """The cache, works, passages and run records of research, in one place."""

    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], float] = time.time,
        shared_cache: ResearchStore | None = None,
        max_cache_bytes: int | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path: Path | None = path
        self._blob_dir: Path | None = path.parent / "blobs"
        self._open(str(path), clock, shared_cache, max_cache_bytes)

    @classmethod
    def in_memory(
        cls,
        *,
        clock: Callable[[], float] = time.time,
        shared_cache: ResearchStore | None = None,
        max_cache_bytes: int | None = None,
    ) -> ResearchStore:
        """A store that lives and dies with the process."""
        store = cls.__new__(cls)
        store.path = None
        store._blob_dir = None
        store._open(":memory:", clock, shared_cache, max_cache_bytes)
        return store

    def _open(
        self,
        database: str,
        clock: Callable[[], float],
        shared_cache: ResearchStore | None = None,
        max_cache_bytes: int | None = None,
    ) -> None:
        if max_cache_bytes is not None and max_cache_bytes < 0:
            raise ValueError(f"max_cache_bytes must be >= 0, got {max_cache_bytes}")
        self.max_cache_bytes = (
            HTTP_CACHE_MAX_BYTES if max_cache_bytes is None else max_cache_bytes
        )
        if shared_cache is self:
            raise ValueError("a store cannot be its own shared cache")
        self._clock = clock
        self._shared = shared_cache
        self.shared_hits = 0
        self._lock = threading.RLock()
        self._memory_blobs: dict[str, bytes] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        # Autocommit: transactions are explicit, in `_transaction`.
        self._conn = sqlite3.connect(
            database, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        if database != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> ResearchStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- schema -----------------------------------------------------------

    def _migrate(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS blobs_inline (sha256 TEXT PRIMARY KEY, data BLOB NOT NULL)"
        )
        current = self.schema_version()
        if current > SCHEMA_VERSION:
            raise StoreError(
                f"store schema is version {current}; this code reads up to {SCHEMA_VERSION}"
            )
        for version in range(current + 1, SCHEMA_VERSION + 1):
            with self._transaction():
                for statement in _split_statements(_MIGRATIONS[version]):
                    self._conn.execute(statement)
                self._conn.execute("INSERT INTO schema_version VALUES (?)", (version,))
        self._backfill_cache_sizes()

    def _backfill_cache_sizes(self) -> None:
        """Give the rows a version-1 file left at size 0 their body's real size."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, body_blob FROM http_cache WHERE size = 0"
            ).fetchall()
            for row in rows:
                try:
                    size = len(self.get_blob(row["body_blob"]))
                except StoreError:
                    continue  # a row whose blob is gone stays at 0; a read reports it
                self._conn.execute(
                    "UPDATE http_cache SET size = ? WHERE key = ?", (size, row["key"])
                )

    def schema_version(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(version) AS v FROM schema_version"
            ).fetchone()
            return int(row["v"] or 0)

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    # -- blobs ------------------------------------------------------------

    def put_blob(self, data: bytes) -> str:
        """Store *data*; return its sha256, the name it is read back by."""
        digest = hashlib.sha256(data).hexdigest()
        with self._lock:
            if len(data) <= INLINE_BLOB_LIMIT:
                self._conn.execute(
                    "INSERT OR IGNORE INTO blobs_inline VALUES (?, ?)", (digest, data)
                )
            elif self._blob_dir is None:
                self._memory_blobs[digest] = data
            else:
                target = self._blob_path(digest)
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = target.with_suffix(f".{uuid.uuid4().hex}.part")
                    partial.write_bytes(data)
                    os.replace(partial, target)
        return digest

    def get_blob(self, digest: str) -> bytes:
        """The bytes stored under *digest*; ``StoreError`` when there are none."""
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM blobs_inline WHERE sha256 = ?", (digest,)
            ).fetchone()
            if row is not None:
                return bytes(row["data"])
            if self._blob_dir is None:
                if digest in self._memory_blobs:
                    return self._memory_blobs[digest]
            else:
                target = self._blob_path(digest)
                if target.exists():
                    return target.read_bytes()
        raise StoreError(f"no blob {digest}")

    def _blob_path(self, digest: str) -> Path:
        if not _DIGEST.fullmatch(digest) or self._blob_dir is None:
            raise StoreError(f"not a blob digest: {digest!r}")
        return self._blob_dir / digest[:2] / digest

    def _text(self, digest: str | None) -> str:
        return self.get_blob(digest).decode("utf-8") if digest else ""

    # -- HTTP cache -------------------------------------------------------

    def cache_get(self, key: str, *, any_age: bool = False) -> CachedResponse | None:
        """The live entry for *key*; None on a miss or once its TTL has run out.

        ``any_age`` returns an expired entry too: an offline run over a
        snapshot reads what the snapshot holds, however old.

        With a shared cache (``shared_cache=``), an own miss reads through to
        it: a live entry there is copied into this store (with the time it was
        fetched, so its TTL keeps counting from then) and returned, so this
        store ends up holding exactly the entries it was asked for. An
        ``any_age`` read never consults the shared cache: an offline run sees
        only its own snapshot.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM http_cache WHERE key = ?", (key,)
            ).fetchone()
            expired = (
                row is not None and row["fetched_at"] + row["ttl_s"] <= self._clock()
            )
            if row is not None and (any_age or not expired):
                self.cache_hits += 1
                self._touch(key)
                return self._cached(row)
            if self._shared is not None and not any_age:
                shared = self._shared._live_entry(key)
                if shared is not None:
                    self._store_entry(shared)
                    self.cache_hits += 1
                    self.shared_hits += 1
                    return shared
            self.cache_misses += 1
            return None

    def _live_entry(self, key: str) -> CachedResponse | None:
        """The unexpired entry for *key*, counted nowhere (a read-through peek)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM http_cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None or row["fetched_at"] + row["ttl_s"] <= self._clock():
                return None
            self._touch(key)
            return self._cached(row)

    def _touch(self, key: str) -> None:
        self._conn.execute(
            "UPDATE http_cache SET last_used = ? WHERE key = ?", (self._clock(), key)
        )

    _CACHE_INSERT = (
        "INSERT OR REPLACE INTO http_cache "
        "(key, source, url, status, body_blob, fetched_at, ttl_s, licence, size, last_used) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )

    def _store_entry(self, entry: CachedResponse) -> None:
        """Write *entry* as it is, keeping its own ``fetched_at``."""
        digest = self.put_blob(entry.body)
        with self._lock:
            self._conn.execute(
                self._CACHE_INSERT,
                (
                    entry.key,
                    entry.source,
                    entry.url,
                    entry.status,
                    digest,
                    entry.fetched_at,
                    entry.ttl_s,
                    entry.licence,
                    len(entry.body),
                    self._clock(),
                ),
            )
            self._enforce_cache_cap()

    def cache_put(
        self,
        key: str,
        *,
        source: str,
        url: str,
        status: int,
        body: bytes,
        ttl_s: int,
        licence: str = "",
    ) -> None:
        """Store a response, replacing any earlier entry for *key*; with a
        shared cache it is written there too, so the next item finds it."""
        if ttl_s <= 0:
            raise ValueError(f"ttl_s must be positive, got {ttl_s}")
        digest = self.put_blob(body)
        with self._lock:
            now = self._clock()
            self._conn.execute(
                self._CACHE_INSERT,
                (key, source, url, status, digest, now, ttl_s, licence, len(body), now),
            )
            self._enforce_cache_cap()
        if self._shared is not None:
            self._shared.cache_put(
                key,
                source=source,
                url=url,
                status=status,
                body=body,
                ttl_s=ttl_s,
                licence=licence,
            )

    def import_cache(self, source_db: Path) -> CacheImport:
        """Merge the ``http_cache`` rows (and their blobs) of another store's
        database file into this store, keeping the freshest row per key.

        The other file is opened read-only and never changed. A row whose TTL
        has run out is not imported; a row older than the one already here is
        not either. The licence tag travels with the row, so a
        ``no_redistribute`` entry stays usable here and out of every export.
        A file that holds no ``http_cache`` table (or is not a database)
        reports ``problem``; a row whose blob is missing or does not match
        its digest is counted ``unreadable`` and skipped. The counts are by
        the row's ``source``.
        """
        report = CacheImport(path=str(source_db))
        try:
            src = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
        except sqlite3.Error as error:
            report.problem = f"cannot open: {error}"
            return report
        src.row_factory = sqlite3.Row
        try:
            try:
                rows = src.execute("SELECT * FROM http_cache").fetchall()
            except sqlite3.Error as error:
                report.problem = f"no http_cache table ({error})"
                return report
            now = self._clock()
            with self._transaction():
                for row in rows:
                    counts = report.by_source.setdefault(
                        row["source"], CacheImportCounts()
                    )
                    if row["fetched_at"] + row["ttl_s"] <= now:
                        counts.expired += 1
                        continue
                    with self._lock:
                        held = self._conn.execute(
                            "SELECT fetched_at FROM http_cache WHERE key = ?",
                            (row["key"],),
                        ).fetchone()
                    if held is not None and held["fetched_at"] >= row["fetched_at"]:
                        counts.kept_existing += 1
                        continue
                    body = _read_foreign_blob(src, source_db, row["body_blob"])
                    if body is None:
                        counts.unreadable += 1
                        continue
                    self._store_entry(
                        CachedResponse(
                            key=row["key"],
                            source=row["source"],
                            url=row["url"],
                            status=row["status"],
                            body=body,
                            fetched_at=row["fetched_at"],
                            ttl_s=row["ttl_s"],
                            licence=row["licence"],
                        )
                    )
                    counts.imported += 1
        finally:
            src.close()
        return report

    def purge_expired(self) -> int:
        """Delete the entries whose TTL has run out (and their blobs); the number removed."""
        with self._lock:
            return self._delete_rows("fetched_at + ttl_s <= ?", (self._clock(),)).rows

    # -- cache size cap ---------------------------------------------------

    def cache_bytes(self) -> int:
        """The bytes of response bodies the HTTP cache holds."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(size), 0) AS n FROM http_cache"
            ).fetchone()
            return int(row["n"])

    def cache_stats(self) -> CacheStats:
        """Size, rows per source and the age range of the HTTP cache."""
        with self._lock:
            by_source = {
                r["source"]: int(r["n"])
                for r in self._conn.execute(
                    "SELECT source, COUNT(*) AS n FROM http_cache GROUP BY source ORDER BY source"
                )
            }
            span = self._conn.execute(
                "SELECT MIN(fetched_at) AS lo, MAX(fetched_at) AS hi FROM http_cache"
            ).fetchone()
        return CacheStats(
            bytes=self.cache_bytes(),
            rows=sum(by_source.values()),
            by_source=by_source,
            oldest=span["lo"],
            newest=span["hi"],
            max_bytes=self.max_cache_bytes,
        )

    def prune_cache(self, max_bytes: int | None = None) -> CachePrune:
        """Bring the HTTP cache under *max_bytes* (default: this store's cap):
        expired rows first, then the least recently used. Blobs nothing else
        refers to are deleted with their rows."""
        with self._lock:
            return self._enforce_cache_cap(max_bytes)

    def _enforce_cache_cap(self, max_bytes: int | None = None) -> CachePrune:
        cap = self.max_cache_bytes if max_bytes is None else max_bytes
        done = CachePrune()
        if self.cache_bytes() <= cap:
            return done
        expired = self._delete_rows("fetched_at + ttl_s <= ?", (self._clock(),))
        done.expired = expired.rows
        done.freed_bytes += expired.freed
        done.blobs_deleted += expired.blobs
        while True:
            over = self.cache_bytes() - cap
            if over <= 0:
                break
            victim = self._conn.execute(
                "SELECT key FROM http_cache ORDER BY last_used, rowid LIMIT 1"
            ).fetchone()
            if victim is None:
                break
            gone = self._delete_rows("key = ?", (victim["key"],))
            done.evicted += gone.rows
            done.freed_bytes += gone.freed
            done.blobs_deleted += gone.blobs
        return done

    def _delete_rows(self, where: str, args: tuple[object, ...]) -> _Deleted:
        rows = self._conn.execute(
            f"SELECT key, body_blob, size FROM http_cache WHERE {where}", args
        ).fetchall()
        deleted = _Deleted()
        for row in rows:
            self._conn.execute("DELETE FROM http_cache WHERE key = ?", (row["key"],))
            deleted.rows += 1
            deleted.freed += int(row["size"])
            if self._delete_blob_if_unused(row["body_blob"]):
                deleted.blobs += 1
        return deleted

    def _delete_blob_if_unused(self, digest: str) -> bool:
        """Delete blob *digest* unless a cache row, a work or a passage still names it."""
        for table, column in (
            ("http_cache", "body_blob"),
            ("works", "abstract_blob"),
            ("passages", "text_blob"),
        ):
            if self._conn.execute(
                f"SELECT 1 FROM {table} WHERE {column} = ? LIMIT 1", (digest,)
            ).fetchone():
                return False
        removed = self._conn.execute(
            "DELETE FROM blobs_inline WHERE sha256 = ?", (digest,)
        ).rowcount
        if removed:
            return True
        if self._blob_dir is None:
            return self._memory_blobs.pop(digest, None) is not None
        target = self._blob_path(digest)
        if target.exists():
            target.unlink()
            return True
        return False

    def iter_cache(self, *, exportable_only: bool = True) -> Iterator[CachedResponse]:
        """Every entry, expired or not. By default the ``no_redistribute`` ones are left out."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM http_cache ORDER BY rowid"
            ).fetchall()
        for row in rows:
            if exportable_only and row["licence"] == LICENCE_NO_REDISTRIBUTE:
                continue
            yield self._cached(row)

    def _cached(self, row: sqlite3.Row) -> CachedResponse:
        return CachedResponse(
            key=row["key"],
            source=row["source"],
            url=row["url"],
            status=row["status"],
            body=self.get_blob(row["body_blob"]),
            fetched_at=row["fetched_at"],
            ttl_s=row["ttl_s"],
            licence=row["licence"],
        )

    def reference_cache(
        self,
        *,
        source: str = "crossref",
        ttl_s: int = TTL_METADATA_S,
        any_age: bool = False,
    ) -> ReferenceCache:
        """This store as the ``references`` HTTP cache, so Crossref lookups share it.
        ``any_age`` serves expired entries too (an offline run over a snapshot)."""
        return ReferenceCache(self, source=source, ttl_s=ttl_s, any_age=any_age)

    # -- works ------------------------------------------------------------

    def upsert_work(self, work: Work) -> Work:
        """Store *work*, merged with every stored record it shares an identifier with.

        Returns the stored record with its ``work_id``. A record that bridges
        two stored works (a DOI on one, a PMID on the other) merges them into
        the older one; the other id stays valid as an alias.
        """
        with self._transaction():
            matches = self._matching_ids(work)
            if not matches:
                work_id = work.work_id or f"w_{uuid.uuid4().hex[:12]}"
                stored = work.model_copy(update={"work_id": work_id})
                self._insert_work(stored)
                return stored
            survivor = matches[0]
            merged = self._load_work(survivor)
            for other in matches[1:]:
                merged = merge_works(merged, self._load_work(other))
            merged = merge_works(merged, work.model_copy(update={"work_id": survivor}))
            for other in matches[1:]:
                self._absorb(other, into=survivor)
            self._update_work(merged)
            return merged

    def find_work(self, work: Work) -> Work | None:
        """The stored work *work* matches by any identifier, or None."""
        with self._lock:
            matches = self._matching_ids(work)
            return self._load_work(matches[0]) if matches else None

    def get_work(self, work_id: str) -> Work | None:
        """The work with this id (an alias resolves to its surviving work)."""
        with self._lock:
            resolved = self.resolve_work_id(work_id)
            return self._load_work(resolved) if resolved else None

    def resolve_work_id(self, work_id: str) -> str | None:
        """The id the work is stored under now, or None if it is unknown."""
        with self._lock:
            if self._conn.execute(
                "SELECT 1 FROM works WHERE work_id = ?", (work_id,)
            ).fetchone():
                return work_id
            row = self._conn.execute(
                "SELECT work_id FROM work_aliases WHERE alias_id = ?", (work_id,)
            ).fetchone()
            return str(row["work_id"]) if row else None

    def iter_works(self) -> Iterator[Work]:
        with self._lock:
            ids = [
                r["work_id"]
                for r in self._conn.execute("SELECT work_id FROM works ORDER BY rowid")
            ]
        for work_id in ids:
            work = self.get_work(work_id)
            if work is not None:
                yield work

    def set_work_state(
        self,
        work_id: str,
        *,
        access: str | None = None,
        retracted: str | None = None,
    ) -> None:
        """Overwrite what is known about a work's readability or retraction.

        A merge only ever raises these (so a stale record cannot lower them);
        this is for a fresh check or fetch that settles the question. It stamps
        ``last_checked``.
        """
        with self._lock:
            resolved = self.resolve_work_id(work_id)
            if resolved is None:
                raise StoreError(f"unknown work {work_id}")
            if access is not None:
                self._conn.execute(
                    "UPDATE works SET access = ? WHERE work_id = ?", (access, resolved)
                )
            if retracted is not None:
                self._conn.execute(
                    "UPDATE works SET retracted = ? WHERE work_id = ?",
                    (retracted, resolved),
                )
            self._conn.execute(
                "UPDATE works SET last_checked = ? WHERE work_id = ?",
                (self._clock(), resolved),
            )

    def _matching_ids(self, work: Work) -> list[str]:
        found: set[str] = set()
        if work.work_id:
            resolved = self.resolve_work_id(work.work_id)
            if resolved:
                found.add(resolved)
        for kind, value in work.hard_keys():
            column = _KIND_COLUMN[kind]
            row = self._conn.execute(
                f"SELECT work_id FROM works WHERE {column} = ?", (value,)
            ).fetchone()
            if row:
                found.add(row["work_id"])
        weak = title_key(work.title, work.year)
        if not found and weak:
            for row in self._conn.execute(
                "SELECT work_id FROM works WHERE title_key = ?", (weak,)
            ):
                if works_compatible(self._load_work(row["work_id"]), work):
                    found.add(row["work_id"])
        if not found:
            return []
        marks = ",".join("?" * len(found))
        rows = self._conn.execute(
            f"SELECT work_id FROM works WHERE work_id IN ({marks}) ORDER BY first_seen, rowid",
            tuple(found),
        )
        return [r["work_id"] for r in rows]

    def _load_work(self, work_id: str) -> Work:
        row = self._conn.execute(
            "SELECT * FROM works WHERE work_id = ?", (work_id,)
        ).fetchone()
        if row is None:
            raise StoreError(f"unknown work {work_id}")
        return Work(
            work_id=row["work_id"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            pmid=row["pmid"],
            pmcid=row["pmcid"],
            openalex_id=row["openalex_id"],
            s2_id=row["s2_id"],
            local_id=row["local_id"],
            url=row["url"],
            title=row["title"],
            authors=json.loads(row["authors_json"]),
            year=row["year"],
            venue=row["venue"],
            abstract=self._text(row["abstract_blob"]),
            oa_url=row["oa_url"],
            access=row["access"],
            retracted=row["retracted"],
            licence=row["licence"],
        )

    def _work_values(self, work: Work) -> dict[str, JsonValue]:
        return {
            "work_id": work.work_id,
            **{field: getattr(work, field) for field in _KEY_COLUMNS},
            "title": work.title,
            "authors_json": json.dumps(work.authors),
            "year": work.year,
            "venue": work.venue,
            "abstract_blob": self.put_blob(work.abstract.encode("utf-8"))
            if work.abstract
            else None,
            "oa_url": work.oa_url,
            "access": work.access,
            "retracted": work.retracted,
            "licence": work.licence,
            "title_key": title_key(work.title, work.year),
        }

    def _insert_work(self, work: Work) -> None:
        values = self._work_values(work)
        now = self._clock()
        columns = [*values, "first_seen", "last_checked"]
        self._conn.execute(
            f"INSERT INTO works ({','.join(columns)}) VALUES ({','.join('?' * len(columns))})",
            (*values.values(), now, now),
        )

    def _update_work(self, work: Work) -> None:
        values = self._work_values(work)
        del values["work_id"]
        assignments = ",".join(f"{column} = ?" for column in values)
        self._conn.execute(
            f"UPDATE works SET {assignments}, last_checked = ? WHERE work_id = ?",
            (*values.values(), self._clock(), work.work_id),
        )

    def _absorb(self, other: str, *, into: str) -> None:
        """Delete work *other*; its id (and ids that pointed at it) now resolve to *into*."""
        self._conn.execute("DELETE FROM works WHERE work_id = ?", (other,))
        self._conn.execute(
            "UPDATE work_aliases SET work_id = ? WHERE work_id = ?", (into, other)
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO work_aliases VALUES (?, ?)", (other, into)
        )

    # -- passages ---------------------------------------------------------

    def add_passages(self, passages: Sequence[Passage]) -> int:
        """Store passages (content-addressed: one already held is kept); the number new."""
        added = 0
        with self._transaction():
            for passage in passages:
                digest = self.put_blob(passage.text.encode("utf-8"))
                cursor = self._conn.execute(
                    "INSERT OR IGNORE INTO passages VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        passage.passage_id,
                        passage.work_id,
                        passage.start,
                        passage.end,
                        digest,
                        passage.retrieved_at,
                    ),
                )
                added += cursor.rowcount
        return added

    def get_passage(self, passage_id: str) -> Passage | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM passages WHERE passage_id = ?", (passage_id,)
            ).fetchone()
            return self._passage(row) if row else None

    def passages_for_work(self, work_id: str) -> list[Passage]:
        """Every passage of a work, including those stored under its aliases.

        A passage keeps the ``work_id`` it was addressed with; read one's work
        through :meth:`resolve_work_id`.
        """
        with self._lock:
            resolved = self.resolve_work_id(work_id) or work_id
            ids = [resolved] + [
                r["alias_id"]
                for r in self._conn.execute(
                    "SELECT alias_id FROM work_aliases WHERE work_id = ?", (resolved,)
                )
            ]
            marks = ",".join("?" * len(ids))
            rows = self._conn.execute(
                f"SELECT * FROM passages WHERE work_id IN ({marks}) ORDER BY rowid",
                tuple(ids),
            ).fetchall()
            return [self._passage(row) for row in rows]

    def _passage(self, row: sqlite3.Row) -> Passage:
        return Passage(
            passage_id=row["passage_id"],
            work_id=row["work_id"],
            start=row["start"],
            end=row["end"],
            text=self._text(row["text_blob"]),
            retrieved_at=row["retrieved_at"],
        )

    # -- runs -------------------------------------------------------------

    def start_run(
        self,
        *,
        question: str,
        mode: str,
        config: Mapping[str, JsonValue],
        models: Mapping[str, JsonValue],
        run_id: str | None = None,
    ) -> str:
        run_id = run_id or f"r_{uuid.uuid4().hex[:12]}"
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO runs (run_id, question, mode, config_json, models_json, started_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        question,
                        mode,
                        json.dumps(config),
                        json.dumps(models),
                        self._clock(),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise StoreError(f"run {run_id} already exists") from error
        return run_id

    def finish_run(
        self, run_id: str, *, stats: Mapping[str, JsonValue], degraded: Sequence[str]
    ) -> None:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE runs SET finished_at = ?, stats_json = ?, degraded_json = ? WHERE run_id = ?",
                (self._clock(), json.dumps(stats), json.dumps(list(degraded)), run_id),
            )
            if cursor.rowcount == 0:
                raise StoreError(f"unknown run {run_id}")

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return _run_record(row) if row else None

    def list_runs(self) -> list[RunRecord]:
        """Every run, newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC, rowid DESC"
            ).fetchall()
        return [_run_record(row) for row in rows]

    def add_run_item(
        self,
        run_id: str,
        kind: RunItemKind,
        ref_id: str,
        data: BaseModel | Mapping[str, JsonValue],
    ) -> None:
        """Record one row of a run; a row with the same kind and ref_id is replaced."""
        if kind not in get_args(RunItemKind):
            raise ValueError(f"unknown run item kind {kind!r}")
        payload = (
            data.model_dump(mode="json") if isinstance(data, BaseModel) else dict(data)
        )
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO run_items VALUES (?, ?, ?, ?)",
                    (run_id, kind, ref_id, json.dumps(payload)),
                )
            except sqlite3.IntegrityError as error:
                raise StoreError(f"unknown run {run_id}") from error

    def run_items(self, run_id: str, kind: RunItemKind | None = None) -> list[RunItem]:
        query = "SELECT * FROM run_items WHERE run_id = ?"
        args: tuple[Any, ...] = (run_id,)
        if kind is not None:
            query += " AND kind = ?"
            args += (kind,)
        with self._lock:
            rows = self._conn.execute(query + " ORDER BY rowid", args).fetchall()
        return [
            RunItem(
                run_id=r["run_id"],
                kind=r["kind"],
                ref_id=r["ref_id"],
                data=json.loads(r["data_json"]),
            )
            for r in rows
        ]


@dataclass
class _Deleted:
    rows: int = 0
    freed: int = 0
    blobs: int = 0


class ReferenceCache:
    """The ``references.http.Cache`` protocol over the store's HTTP cache."""

    def __init__(
        self, store: ResearchStore, *, source: str, ttl_s: int, any_age: bool = False
    ) -> None:
        self._store = store
        self._source = source
        self._ttl_s = ttl_s
        self._any_age = any_age

    def get(self, key: str) -> bytes | None:
        hit = self._store.cache_get(cache_key(self._source, key), any_age=self._any_age)
        return hit.body if hit else None

    def put(self, key: str, value: bytes) -> None:
        self._store.cache_put(
            cache_key(self._source, key),
            source=self._source,
            url=key,
            status=200,
            body=value,
            ttl_s=self._ttl_s,
        )


def _read_foreign_blob(
    conn: sqlite3.Connection, database: Path, digest: str
) -> bytes | None:
    """The bytes of blob *digest* in another store (inline table, else its
    ``blobs/`` folder), or None when missing or not matching the digest."""
    if not _DIGEST.fullmatch(digest):
        return None
    data: bytes | None = None
    try:
        row = conn.execute(
            "SELECT data FROM blobs_inline WHERE sha256 = ?", (digest,)
        ).fetchone()
    except sqlite3.Error:
        row = None
    if row is not None:
        data = bytes(row["data"])
    else:
        target = database.parent / "blobs" / digest[:2] / digest
        if target.is_file():
            data = target.read_bytes()
    if data is None or hashlib.sha256(data).hexdigest() != digest:
        return None
    return data


def _run_record(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        question=row["question"],
        mode=row["mode"],
        config=json.loads(row["config_json"]),
        models=json.loads(row["models_json"]),
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        stats=json.loads(row["stats_json"]) if row["stats_json"] else {},
        degraded=json.loads(row["degraded_json"]) if row["degraded_json"] else [],
    )


def _split_statements(script: str) -> list[str]:
    """The statements of a schema script (no semicolons occur inside any of them)."""
    return [s.strip() for s in script.split(";") if s.strip()]
