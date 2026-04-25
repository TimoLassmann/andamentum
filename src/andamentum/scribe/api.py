"""Public Document API.

The Document class is the single entry point for callers. All
mutations go through it; direct SQL is internal.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from .database import open_db


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


class Document:
    """A scribe document — a structured, block-based draft."""

    def __init__(
        self,
        *,
        id: str,
        title: str,
        database: str,
        template: Optional[str] = None,
    ):
        self.id = id
        self.title = title
        self.database = database
        self.template = template

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        title: str,
        database: str,
        template: Optional[str] = None,
    ) -> "Document":
        """Create a new document and return its handle."""
        doc_id = _new_id()
        now = _now_iso()
        with open_db(database) as conn:
            conn.execute(
                "INSERT INTO scribe_documents "
                "(id, title, template, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc_id, title, template, now, now),
            )
            conn.commit()
        return cls(id=doc_id, title=title, database=database, template=template)

    @classmethod
    def open(cls, doc_id: str, *, database: str) -> "Document":
        """Open an existing document by id."""
        with open_db(database) as conn:
            row = conn.execute(
                "SELECT id, title, template FROM scribe_documents WHERE id = ?",
                (doc_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Document {doc_id!r} not found in database {database!r}")
        return cls(
            id=row["id"],
            title=row["title"],
            database=database,
            template=row["template"],
        )
