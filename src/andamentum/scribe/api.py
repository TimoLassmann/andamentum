"""Public Document API.

The Document class is the single entry point for callers. All
mutations go through it; direct SQL is internal.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from .database import open_db
from .models import Block


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Block factory helpers — return plain dicts that Document.append consumes.
# ---------------------------------------------------------------------------


def Paragraph(content: str) -> dict:
    return {"type": "paragraph", "content": content, "metadata": {}}


def Heading(content: str, *, level: int) -> dict:
    if not 1 <= level <= 6:
        raise ValueError(f"heading level must be 1..6, got {level}")
    return {"type": "heading", "content": content, "metadata": {"level": level}}


def Figure(
    *,
    path: str,
    caption: str,
    label: str,
    width_in: Optional[float] = None,
) -> dict:
    return {
        "type": "figure",
        "content": "",
        "metadata": {
            "path": path,
            "caption": caption,
            "label": label,
            "width_in": width_in,
        },
    }


def Table(
    *,
    rows: list[list[str]],
    header_row: bool = True,
    caption: str = "",
    label: str = "",
) -> dict:
    return {
        "type": "table",
        "content": "",
        "metadata": {
            "rows": rows,
            "header_row": header_row,
            "caption": caption,
            "label": label,
        },
    }


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

    def append(self, block_spec: dict, *, parent_id: Optional[str] = None) -> str:
        """Append a block to the end of this document. Returns block id."""
        bid = _new_id()
        now = _now_iso()
        with open_db(self.database) as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 AS next_pos "
                "FROM scribe_blocks WHERE doc_id = ?",
                (self.id,),
            ).fetchone()
            next_pos = row["next_pos"]
            conn.execute(
                "INSERT INTO scribe_blocks "
                "(id, doc_id, type, content, position, parent_id, metadata, "
                " revision, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                (
                    bid,
                    self.id,
                    block_spec["type"],
                    block_spec.get("content", ""),
                    next_pos,
                    parent_id,
                    json.dumps(block_spec.get("metadata", {})),
                    now,
                    now,
                ),
            )
            conn.commit()
        return bid

    def query(self, *, type: Optional[str] = None) -> list[Block]:
        """Return blocks for this document, ordered by position."""
        sql = (
            "SELECT id, doc_id, type, content, position, parent_id, "
            "metadata, revision, created_at, updated_at "
            "FROM scribe_blocks WHERE doc_id = ?"
        )
        params: list[Any] = [self.id]
        if type is not None:
            sql += " AND type = ?"
            params.append(type)
        sql += " ORDER BY position"

        with open_db(self.database) as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            Block(
                id=r["id"],
                doc_id=r["doc_id"],
                type=r["type"],
                content=r["content"],
                position=r["position"],
                parent_id=r["parent_id"],
                metadata=json.loads(r["metadata"]),
                revision=r["revision"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]
