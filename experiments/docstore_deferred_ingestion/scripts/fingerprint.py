"""Read-only database observation: the logical fingerprint and the row dump.

WHY A *LOGICAL* FINGERPRINT
---------------------------
The obvious invariant — sha256 of the .db file — is unreliable. WAL checkpoints
and page reuse move bytes without changing meaning, so asserting on the file
hash would manufacture false failures. The honest invariant is a hash over the
*meaning*: for every document, ordered by uuid,

    doc_uuid|ingest_status|file_hash|len(markdown)|n_chunks|has_doc_embedding|
    n_chunk_embeddings|sorted_metadata_keys

That is exactly the evidence H-b (a no-op drain really is a no-op) and H-a2
(nothing was enriched) need. The file hash, page_count and byte size are
recorded too, but as ADVISORY fields that nothing asserts on.

Every connection here is opened read-only (``mode=ro`` URI) so polling a live
drain cannot contend on the write lock. ``sqlite-vec`` is loaded because
``chunk_embeddings`` and ``doc_embeddings`` are vec0 virtual tables and a plain
connection cannot even name them.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class MissingDatabaseError(FileNotFoundError):
    """A named database file does not exist.

    Raised rather than returning empty counts: a fingerprint of zeros looks
    exactly like a clean database, which is how a mis-named store (see
    ``lifecycle.EPHEMERAL_PREFIXES``) turns into a green but meaningless run.
    """


@contextmanager
def readonly_connection(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open a read-only, vec-enabled connection. Raises if the file is absent."""
    path = Path(db_path)
    if not path.exists():
        raise MissingDatabaseError(
            f"{path} does not exist. Refusing to emit a zeroed fingerprint — an "
            "absent database and an empty one are different facts."
        )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)
    try:
        yield conn
    finally:
        conn.close()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ? LIMIT 1", (name,)
    ).fetchone()
    return row is not None


def document_rows(db_path: str | Path) -> list[dict[str, Any]]:
    """One dict per active document: queue state, sizes, chunk/embedding counts.

    Doubles as the per-stage cost breakdown (``ingest_updated_at``) and as the
    post-SIGKILL truth check for the checkpoint claim (``markdown_sha256``).
    """
    with readonly_connection(db_path) as conn:
        has_chunks = _table_exists(conn, "chunks")
        has_chunk_emb = _table_exists(conn, "chunk_embeddings")
        has_doc_emb = _table_exists(conn, "doc_embeddings")

        rows = conn.execute(
            """
            SELECT id, doc_uuid, dc_title, ingest_status, ingest_error,
                   ingest_updated_at, source_file_path, file_hash, metadata,
                   markdown_content, created_date, updated_date
            FROM documents
            WHERE deleted_at IS NULL
            ORDER BY doc_uuid
            """
        ).fetchall()

        out: list[dict[str, Any]] = []
        for row in rows:
            markdown = row["markdown_content"] or ""
            try:
                metadata = json.loads(row["metadata"] or "{}")
            except json.JSONDecodeError:
                metadata = {}

            n_chunks = 0
            n_chunk_embeddings = 0
            if has_chunks:
                n_chunks = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM chunks WHERE document_id = ?",
                        (row["id"],),
                    ).fetchone()[0]
                )
                if has_chunk_emb:
                    n_chunk_embeddings = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM chunk_embeddings WHERE chunk_id IN "
                            "(SELECT id FROM chunks WHERE document_id = ?)",
                            (row["id"],),
                        ).fetchone()[0]
                    )

            has_doc_embedding = False
            if has_doc_emb:
                has_doc_embedding = (
                    int(
                        conn.execute(
                            "SELECT COUNT(*) FROM doc_embeddings WHERE doc_id = ?",
                            (row["id"],),
                        ).fetchone()[0]
                    )
                    > 0
                )

            out.append(
                {
                    "doc_uuid": row["doc_uuid"],
                    "dc_title": row["dc_title"],
                    "ingest_status": row["ingest_status"],
                    "ingest_error": row["ingest_error"],
                    "ingest_updated_at": row["ingest_updated_at"],
                    "source_file_path": row["source_file_path"],
                    "file_hash": row["file_hash"],
                    "markdown_chars": len(markdown),
                    "markdown_sha256": hashlib.sha256(
                        markdown.encode("utf-8")
                    ).hexdigest(),
                    "n_chunks": n_chunks,
                    "n_chunk_embeddings": n_chunk_embeddings,
                    "has_doc_embedding": has_doc_embedding,
                    "metadata_keys": sorted(metadata.keys()),
                    "llm_metadata_populated": bool(
                        metadata.get("topics") or metadata.get("projects")
                        or metadata.get("people")
                    ),
                    "created_date": row["created_date"],
                    "updated_date": row["updated_date"],
                }
            )
        return out


def logical_fingerprint(rows: list[dict[str, Any]]) -> str:
    """sha256 over the meaning-bearing projection of every document row."""
    lines = [
        "|".join(
            [
                r["doc_uuid"],
                str(r["ingest_status"]),
                str(r["file_hash"]),
                str(r["markdown_chars"]),
                str(r["n_chunks"]),
                str(int(r["has_doc_embedding"])),
                str(r["n_chunk_embeddings"]),
                ",".join(r["metadata_keys"]),
            ]
        )
        for r in sorted(rows, key=lambda r: r["doc_uuid"])
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Counts per ingest status, always reporting all four keys."""
    counts = {
        "pending_source": 0,
        "pending_enrich": 0,
        "complete": 0,
        "failed": 0,
    }
    for row in rows:
        status = row["ingest_status"] or "unknown"
        counts[status] = counts.get(status, 0) + 1
    return counts


def file_advisory(db_path: str | Path) -> dict[str, Any]:
    """Byte-level facts — recorded, never asserted on (WAL moves bytes freely)."""
    path = Path(db_path)
    sidecars = {}
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        sidecars[suffix] = sidecar.stat().st_size if sidecar.exists() else None
    with readonly_connection(path) as conn:
        page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
        journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0])
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return {
        "advisory_only": True,
        "file_bytes": path.stat().st_size,
        "file_sha256": digest.hexdigest(),
        "page_count": page_count,
        "journal_mode": journal_mode,
        "sidecars": sidecars,
    }


def snapshot(db_path: str | Path, *, label: str) -> dict[str, Any]:
    """The full observation of one database at one instant."""
    rows = document_rows(db_path)
    return {
        "label": label,
        "db_path": str(db_path),
        "logical_fingerprint": logical_fingerprint(rows),
        "status_counts": status_counts(rows),
        "n_documents": len(rows),
        "documents": rows,
        "file_advisory": file_advisory(db_path),
    }


def integrity_check(db_path: str | Path) -> dict[str, Any]:
    """``PRAGMA integrity_check`` + ``quick_check`` + WAL/SHM listing.

    A durability claim that ignores sqlite is half a claim: if a hard kill can
    wedge the database, resumability is theoretical. A failure here is a genuine
    negative result and must be reported, never worked around by deleting a -wal.
    """
    path = Path(db_path)
    with readonly_connection(path) as conn:
        integrity = [r[0] for r in conn.execute("PRAGMA integrity_check").fetchall()]
        quick = [r[0] for r in conn.execute("PRAGMA quick_check").fetchall()]
    sidecars = []
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists():
            sidecars.append({"file": sidecar.name, "bytes": sidecar.stat().st_size})
    return {
        "integrity_check": integrity,
        "quick_check": quick,
        "integrity_ok": integrity == ["ok"],
        "quick_ok": quick == ["ok"],
        "sidecar_files": sidecars,
    }


def poll_row(db_path: str | Path, doc_uuid: str) -> dict[str, Any] | None:
    """Cheap single-row read used by the kill-gate supervisor."""
    try:
        with readonly_connection(db_path) as conn:
            row = conn.execute(
                "SELECT doc_uuid, ingest_status, markdown_content "
                "FROM documents WHERE doc_uuid = ?",
                (doc_uuid,),
            ).fetchone()
    except (MissingDatabaseError, sqlite3.OperationalError):
        return None
    if row is None:
        return None
    markdown = row["markdown_content"] or ""
    return {
        "doc_uuid": row["doc_uuid"],
        "ingest_status": row["ingest_status"],
        "markdown_chars": len(markdown),
        "markdown_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
    }


def poll_all_rows(db_path: str | Path) -> list[dict[str, Any]]:
    """All (uuid, status, markdown length) triples — the H-c4 window observer."""
    try:
        with readonly_connection(db_path) as conn:
            rows = conn.execute(
                "SELECT doc_uuid, ingest_status, length(markdown_content) AS n "
                "FROM documents WHERE deleted_at IS NULL"
            ).fetchall()
    except (MissingDatabaseError, sqlite3.OperationalError):
        return []
    return [
        {
            "doc_uuid": r["doc_uuid"],
            "ingest_status": r["ingest_status"],
            "markdown_chars": int(r["n"] or 0),
        }
        for r in rows
    ]
