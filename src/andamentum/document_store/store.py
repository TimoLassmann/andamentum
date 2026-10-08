"""The document store: fixed source tables, disposable indexes, a fixed API.

The source tables (``documents``, ``revisions``, ``blobs``, ``changes``; see
``db.py``) hold every revision of every document in full and never change
shape. Indexes are disposable: each one catches up from the ``changes`` log
and is rebuilt from scratch when its configuration changes. A searcher, chosen
when the store is opened, ranks passages from an index. Nothing in this module
calls a model; ``sync`` and ``search`` do so only through the indexes and the
searcher the caller passed in.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .db import connect, new_ulid, now
from .types import (
    DocRef,
    Document,
    DocumentNotFound,
    Hit,
    KeyExists,
    SyncReport,
    Where,
)
from .where import where_sql

if TYPE_CHECKING:
    from .index import Index
    from .searchers import Searcher

#: Documents an index processes per batch during ``sync``.
SYNC_BATCH = 64

#: Index names become table-name parts, so they are checked here as well.
_INDEX_NAME = re.compile(r"[a-z][a-z0-9_]*")

# The newest revision of every document, joined to its document row.
_HEAD = (
    "FROM documents d JOIN revisions r ON r.doc_id = d.id "
    "AND r.rev = (SELECT MAX(rev) FROM revisions WHERE doc_id = d.id)"
)


def _metadata_json(metadata: dict[str, Any] | None) -> str:
    if metadata is None:
        return "{}"
    if not isinstance(metadata, dict):
        raise TypeError(f"metadata must be a dict, got {type(metadata).__name__}")
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)


class Store:
    """One SQLite file of documents. Open it with :meth:`Store.open`."""

    def __init__(
        self,
        path: Path,
        db: sqlite3.Connection,
        *,
        indexes: Sequence[Index],
        searcher: Searcher | None,
    ) -> None:
        self.path = path
        self._db = db
        for ix in indexes:
            if not _INDEX_NAME.fullmatch(ix.name):
                raise ValueError(f"index name {ix.name!r} must match {_INDEX_NAME.pattern}")
        self.indexes = {ix.name: ix for ix in indexes}
        if len(self.indexes) != len(indexes):
            raise ValueError("two indexes share a name")
        self.searcher = searcher
        if searcher is not None:
            for ix in searcher.indexes():
                if self.indexes.get(ix.name) is not ix:
                    raise ValueError(
                        f"the searcher reads index {ix.name!r}, which was not "
                        "passed to Store.open(indexes=...)"
                    )

    @classmethod
    async def open(
        cls,
        path: Path | str,
        *,
        indexes: Sequence[Index] = (),
        searcher: Searcher | None = None,
    ) -> Store:
        """Open (or create) the store at ``path``.

        ``indexes`` are kept up to date by :meth:`sync`; ``searcher`` answers
        :meth:`search`. A store opened with neither stores and reads
        documents and never calls a model.
        """
        path = Path(path)
        return cls(path, connect(path), indexes=indexes, searcher=searcher)

    async def close(self) -> None:
        try:
            for ix in self.indexes.values():
                await ix.close()
        finally:
            self._db.close()

    async def __aenter__(self) -> Store:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @property
    def db(self) -> sqlite3.Connection:
        """The open connection, for indexes and searchers. Not for callers."""
        return self._db

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """One write transaction; it takes the write lock at the start."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield self._db
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    # ── writes ──────────────────────────────────────────────────────────

    async def put(
        self,
        text: str,
        *,
        key: str | None = None,
        media_type: str = "text/markdown",
        metadata: dict[str, Any] | None = None,
        original: bytes | None = None,
    ) -> str:
        """Store a new document and return its id.

        Never deduplicates: two puts of the same text are two documents. A
        ``key`` held by another live document raises :class:`KeyExists`; use
        :meth:`resolve` and :meth:`update` instead. A deleted document gives
        its key up.
        ``original`` keeps the source bytes (a PDF, say) beside the text.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        meta = _metadata_json(metadata)
        doc_id, at = new_ulid(), now()
        with self._write() as db:
            blob = self._put_blob(db, original)
            try:
                db.execute(
                    "INSERT INTO documents (id, key, created_at) VALUES (?, ?, ?)",
                    (doc_id, key, at),
                )
            except sqlite3.IntegrityError as exc:
                if key is not None and "documents.key" in str(exc):
                    raise KeyExists(f"a live document already has key {key!r}") from exc
                raise
            db.execute(
                "INSERT INTO revisions (doc_id, rev, created_at, media_type, blob_hash, "
                "text, metadata) VALUES (?, 1, ?, ?, ?, ?, ?)",
                (doc_id, at, media_type, blob, text, meta),
            )
            db.execute(
                "INSERT INTO changes (doc_id, rev, op) VALUES (?, 1, 'put')", (doc_id,)
            )
        return doc_id

    @staticmethod
    def _put_blob(db: sqlite3.Connection, original: bytes | None) -> str | None:
        if original is None:
            return None
        digest = hashlib.sha256(original).hexdigest()
        db.execute(
            "INSERT OR IGNORE INTO blobs (hash, bytes) VALUES (?, ?)", (digest, original)
        )
        return digest

    async def update(
        self,
        doc_id: str,
        *,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
        media_type: str | None = None,
        original: bytes | None = None,
    ) -> int:
        """Write a new revision and return its number.

        What is not given is copied from the current revision. ``metadata``
        replaces the whole object; to change one field, get, change, update.
        """
        if text is None and metadata is None and media_type is None and original is None:
            raise ValueError("update needs text, metadata, media_type or original")
        with self._write() as db:
            row = db.execute(
                f"SELECT r.rev, r.media_type, r.blob_hash, r.text, r.metadata {_HEAD} "
                "WHERE d.id = ? AND d.deleted_at IS NULL",
                (doc_id,),
            ).fetchone()
            if row is None:
                raise DocumentNotFound(doc_id)
            rev = row[0] + 1
            blob = self._put_blob(db, original) if original is not None else row[2]
            db.execute(
                "INSERT INTO revisions (doc_id, rev, created_at, media_type, blob_hash, "
                "text, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id,
                    rev,
                    now(),
                    media_type if media_type is not None else row[1],
                    blob,
                    text if text is not None else row[3],
                    _metadata_json(metadata) if metadata is not None else row[4],
                ),
            )
            db.execute(
                "INSERT INTO changes (doc_id, rev, op) VALUES (?, ?, 'update')",
                (doc_id, rev),
            )
        return rev

    async def delete(self, doc_id: str) -> bool:
        """Soft-delete; ``False`` if there is no live document with this id."""
        return self._set_deleted(doc_id, deleted=True)

    async def restore(self, doc_id: str) -> bool:
        """Undo a delete; ``False`` if there is no deleted document with this id.

        Raises :class:`KeyExists` if a live document took the key meanwhile.
        """
        return self._set_deleted(doc_id, deleted=False)

    def _set_deleted(self, doc_id: str, *, deleted: bool) -> bool:
        with self._write() as db:
            try:
                cur = db.execute(
                    "UPDATE documents SET deleted_at = ? WHERE id = ? AND deleted_at IS "
                    + ("NULL" if deleted else "NOT NULL"),
                    (now() if deleted else None, doc_id),
                )
            except sqlite3.IntegrityError as exc:
                raise KeyExists(
                    f"cannot restore {doc_id}: a live document now has its key"
                ) from exc
            if cur.rowcount == 0:
                return False
            db.execute(
                "INSERT INTO changes (doc_id, op) VALUES (?, ?)",
                (doc_id, "delete" if deleted else "restore"),
            )
        return True

    async def purge(self, doc_id: str) -> bool:
        """Remove a document and all its revisions for good; ``False`` if absent.

        Original bytes no other revision refers to are removed too.
        """
        with self._write() as db:
            if db.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone() is None:
                return False
            hashes = [
                r[0]
                for r in db.execute(
                    "SELECT DISTINCT blob_hash FROM revisions WHERE doc_id = ? "
                    "AND blob_hash IS NOT NULL",
                    (doc_id,),
                )
            ]
            db.execute("DELETE FROM revisions WHERE doc_id = ?", (doc_id,))
            db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            for h in hashes:
                db.execute(
                    "DELETE FROM blobs WHERE hash = ? AND NOT EXISTS "
                    "(SELECT 1 FROM revisions WHERE blob_hash = ?)",
                    (h, h),
                )
            db.execute("INSERT INTO changes (doc_id, op) VALUES (?, 'purge')", (doc_id,))
        return True

    # ── reads ───────────────────────────────────────────────────────────

    async def get(self, doc_id: str, *, rev: int | None = None) -> Document | None:
        """A revision with its full text: the newest, or ``rev``.

        ``None`` if there is no live document with this id (or no such
        revision).
        """
        sql = (
            "SELECT d.id, d.key, d.created_at, r.rev, r.created_at, r.media_type, "
            "r.text, r.metadata, r.blob_hash FROM documents d JOIN revisions r "
            "ON r.doc_id = d.id WHERE d.id = ? AND d.deleted_at IS NULL AND r.rev = "
        )
        if rev is None:
            sql += "(SELECT MAX(rev) FROM revisions WHERE doc_id = d.id)"
            row = self._db.execute(sql, (doc_id,)).fetchone()
        else:
            row = self._db.execute(sql + "?", (doc_id, rev)).fetchone()
        if row is None:
            return None
        return Document(
            doc_id=row[0],
            key=row[1],
            created_at=row[2],
            rev=row[3],
            updated_at=row[4],
            media_type=row[5],
            text=row[6],
            metadata=json.loads(row[7]),
            original_hash=row[8],
        )

    async def original(self, doc_id: str, *, rev: int | None = None) -> bytes | None:
        """The original bytes stored with a revision, or ``None``."""
        doc = await self.get(doc_id, rev=rev)
        if doc is None or doc.original_hash is None:
            return None
        row = self._db.execute(
            "SELECT bytes FROM blobs WHERE hash = ?", (doc.original_hash,)
        ).fetchone()
        return bytes(row[0])

    async def resolve(self, key: str) -> str | None:
        """The id of the live document with this key, or ``None``."""
        row = self._db.execute(
            "SELECT id FROM documents WHERE key = ? AND deleted_at IS NULL", (key,)
        ).fetchone()
        return row[0] if row else None

    async def read(self, doc_id: str, *, rev: int, start: int, end: int) -> str:
        """Characters ``start``..``end`` of revision ``rev``."""
        doc = await self.get(doc_id, rev=rev)
        if doc is None:
            raise DocumentNotFound(f"{doc_id} rev {rev}")
        if not 0 <= start <= end <= len(doc.text):
            raise ValueError(
                f"span {start}-{end} is outside a text of {len(doc.text)} chars"
            )
        return doc.text[start:end]

    def where_docs(self, where: Where | None) -> tuple[str, list[object]]:
        """SQL selecting the ids of live documents whose newest revision matches."""
        cond, params = where_sql(where, column="r.metadata")
        return f"SELECT d.id {_HEAD} WHERE d.deleted_at IS NULL AND {cond}", params

    async def list(
        self, *, where: Where | None = None, limit: int | None, cursor: str | None = None
    ) -> list[DocRef]:
        """Live documents whose newest revision matches ``where``, newest first.

        ``limit`` has no default: pass ``None`` for all of them. For the next
        page, pass the ``cursor`` of the last ``DocRef`` of this one. Paging
        walks the order as it stood: a document updated between pages moves
        to the front and is not returned again on a later page.
        """
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1 or None")
        cond, params = where_sql(where, column="r.metadata")
        sql = (
            "SELECT d.id, d.key, r.rev, r.created_at, r.media_type, r.metadata, r.rowid "
            f"{_HEAD} WHERE d.deleted_at IS NULL AND {cond}"
        )
        if cursor is not None:
            try:
                position = int(cursor)
            except ValueError:
                raise ValueError(f"not a list cursor: {cursor!r}") from None
            sql += " AND r.rowid < ?"
            params.append(position)
        sql += " ORDER BY r.rowid DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [
            DocRef(
                doc_id=r[0],
                key=r[1],
                rev=r[2],
                updated_at=r[3],
                media_type=r[4],
                metadata=json.loads(r[5]),
                cursor=str(r[6]),
            )
            for r in self._db.execute(sql, params)
        ]

    async def count(self, *, where: Where | None = None) -> int:
        """How many live documents match ``where``."""
        sql, params = self.where_docs(where)
        return int(self._db.execute(f"SELECT COUNT(*) FROM ({sql})", params).fetchone()[0])

    # ── indexes and search ──────────────────────────────────────────────

    async def sync(self) -> SyncReport:
        """Bring every index up to date with the change log.

        An index whose configuration changed is dropped and rebuilt. A
        document an index cannot process is recorded in that index's
        failures table, reported here, and retried on the next sync. A
        transport failure (the embedding server is down) is raised, not
        recorded, and the index keeps its position.
        """
        report = SyncReport()
        for ix in self.indexes.values():
            processed, failures = await self._sync_index(ix)
            report.processed[ix.name] = processed
            report.failures[ix.name] = failures
        return report

    async def _sync_index(self, ix: Index) -> tuple[int, list[tuple[str, str]]]:
        db = self._db
        config = json.dumps(ix.config(), sort_keys=True)
        failures_table = f"ix_{ix.name}_failures"
        row = db.execute(
            "SELECT config, last_seq FROM indexes WHERE name = ?", (ix.name,)
        ).fetchone()
        if row is None or row[0] != config:
            with self._write() as w:
                ix.drop_tables(w)
                w.execute(f"DROP TABLE IF EXISTS {failures_table}")
                ix.create_tables(w)
                w.execute(
                    f"CREATE TABLE {failures_table} (doc_id TEXT PRIMARY KEY, "
                    "error TEXT NOT NULL, at TEXT NOT NULL)"
                )
                w.execute(
                    "INSERT OR REPLACE INTO indexes (name, config, last_seq, built_at) "
                    "VALUES (?, ?, 0, NULL)",
                    (ix.name, config),
                )
            last_seq = 0
        else:
            last_seq = row[1]
        target = db.execute("SELECT COALESCE(MAX(seq), 0) FROM changes").fetchone()[0]
        todo = [
            r[0]
            for r in db.execute(
                "SELECT doc_id FROM changes WHERE seq > ? AND seq <= ? "
                "GROUP BY doc_id ORDER BY MIN(seq)",
                (last_seq, target),
            )
        ]
        retry = [r[0] for r in db.execute(f"SELECT doc_id FROM {failures_table}")]
        todo += [d for d in retry if d not in set(todo)]
        for start in range(0, len(todo), SYNC_BATCH):
            batch = todo[start : start + SYNC_BATCH]
            failed = dict(await ix.apply(self, batch))
            with self._write() as w:
                for doc_id in batch:
                    if doc_id in failed:
                        w.execute(
                            f"INSERT OR REPLACE INTO {failures_table} VALUES (?, ?, ?)",
                            (doc_id, failed[doc_id], now()),
                        )
                    else:
                        w.execute(
                            f"DELETE FROM {failures_table} WHERE doc_id = ?", (doc_id,)
                        )
        with self._write() as w:
            w.execute(
                "UPDATE indexes SET last_seq = ?, built_at = ? WHERE name = ?",
                (target, now(), ix.name),
            )
        failures = [
            (r[0], r[1])
            for r in db.execute(f"SELECT doc_id, error FROM {failures_table} ORDER BY doc_id")
        ]
        return len(todo), failures

    async def search(
        self, query: str, *, limit: int, where: Where | None = None
    ) -> list[Hit]:
        """The best passages for ``query``, best first.

        Brings the indexes up to date first, so the results cover every
        document. ``where`` restricts the candidate documents before ranking.
        """
        if self.searcher is None:
            raise RuntimeError("this store was opened without a searcher")
        if limit < 1:
            raise ValueError("limit must be >= 1")
        report = await self.sync()
        if not report.ok:
            raise RuntimeError(
                "some documents could not be indexed, so a search would silently "
                f"miss them: {report.failures}"
            )
        return await self.searcher.search(self, query, limit=limit, where=where)
