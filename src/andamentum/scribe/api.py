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
from .models import Block, Reference, StaleRevisionError, ValidationIssue
from .parser import extract_citation_keys
from .validate import validate_document


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Block factory helpers — return plain dicts that Document.append consumes.
# ---------------------------------------------------------------------------


def Paragraph(content: str) -> dict[str, Any]:
    return {"type": "paragraph", "content": content, "metadata": {}}


def Heading(content: str, *, level: int) -> dict[str, Any]:
    if not 1 <= level <= 6:
        raise ValueError(f"heading level must be 1..6, got {level}")
    return {"type": "heading", "content": content, "metadata": {"level": level}}


def Figure(
    *,
    path: str,
    caption: str,
    label: str,
    width_in: Optional[float] = None,
) -> dict[str, Any]:
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
) -> dict[str, Any]:
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
        scaffold: Optional[str] = None,
    ) -> "Document":
        """Create a new document and return its handle.

        If `scaffold` is given (currently only "article" is supported),
        the document is pre-populated with the corresponding section
        structure. See scaffolds.py for available scaffolds.

        Note: scribe deliberately ships no grant-application scaffold.
        Funder rules on AI-assisted grant drafting are tightening
        (NHMRC, ARC, NIH NOT-OD-25-122 and equivalents); the project
        chooses not to make AI-assisted grant drafting more frictionless.
        """
        from .scaffolds import SCAFFOLDS

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
        doc = cls(id=doc_id, title=title, database=database, template=template)

        if scaffold is not None:
            if scaffold not in SCAFFOLDS:
                raise ValueError(
                    f"Unknown scaffold {scaffold!r}. Available: {sorted(SCAFFOLDS)}"
                )
            for section_name, guide in SCAFFOLDS[scaffold]:
                doc.append(Heading(section_name, level=1))
                # placeholder paragraph carrying the guide for downstream agents
                doc.append(
                    {
                        "type": "paragraph",
                        "content": "",
                        "metadata": {"guide": guide} if guide else {},
                    }
                )

        return doc

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

    def replace(
        self,
        block_id: str,
        new_content: str,
        *,
        expected_revision: int,
        reason: Optional[str] = None,
    ) -> None:
        """Replace a block's content under optimistic locking.

        Raises StaleRevisionError if the block's current revision differs
        from `expected_revision`. Bumps revision on success and writes
        an audit row to scribe_revisions.

        Uses ``BEGIN IMMEDIATE`` to acquire the write lock before reading
        the current revision, closing the TOCTOU window between the
        check and the update.
        """
        now = _now_iso()
        with open_db(self.database) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT content, revision FROM scribe_blocks "
                "WHERE id = ? AND doc_id = ?",
                (block_id, self.id),
            ).fetchone()
            if row is None:
                raise KeyError(f"Block {block_id!r} not found in document {self.id!r}")
            current_rev = row["revision"]
            if current_rev != expected_revision:
                raise StaleRevisionError(
                    block_id=block_id, expected=expected_revision, actual=current_rev
                )
            new_rev = current_rev + 1
            conn.execute(
                "UPDATE scribe_blocks "
                "SET content = ?, revision = ?, updated_at = ? "
                "WHERE id = ?",
                (new_content, new_rev, now, block_id),
            )
            conn.execute(
                "INSERT INTO scribe_revisions "
                "(block_id, revision, previous_content, new_content, reason, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (block_id, new_rev, row["content"], new_content, reason, now),
            )
            conn.commit()

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

    # ------------------------------------------------------------------
    # Section operations (sections are derived from heading blocks).
    # ------------------------------------------------------------------

    def list_sections(self) -> list[dict]:
        """Return one entry per top-level (level 1) heading.

        Each entry: {"name", "block_id", "position", "block_count", "word_count"}.
        Counts include all blocks until the next level-1 heading (excluding
        the heading itself).
        """
        all_blocks = self.query()
        # Indices of top-level (level 1) headings
        boundaries: list[int] = [
            i
            for i, b in enumerate(all_blocks)
            if b.type == "heading" and int(b.metadata.get("level", 1)) == 1
        ]
        sections: list[dict] = []
        for idx, start in enumerate(boundaries):
            end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(all_blocks)
            head = all_blocks[start]
            body = all_blocks[start + 1 : end]
            words = sum(len(b.content.split()) for b in body)
            sections.append(
                {
                    "name": head.content,
                    "block_id": head.id,
                    "position": head.position,
                    "block_count": len(body),
                    "word_count": words,
                }
            )
        return sections

    def section(self, name: str) -> list[Block]:
        """Return the heading + all blocks belonging to the named section.

        The section ends at the next heading whose level is <= the section
        heading's level (defaults to level 1).
        """
        all_blocks = self.query()
        for i, b in enumerate(all_blocks):
            if b.type == "heading" and b.content == name:
                head_level = int(b.metadata.get("level", 1))
                end = len(all_blocks)
                for j in range(i + 1, len(all_blocks)):
                    nb = all_blocks[j]
                    if (
                        nb.type == "heading"
                        and int(nb.metadata.get("level", 1)) <= head_level
                    ):
                        end = j
                        break
                return all_blocks[i:end]
        raise KeyError(f"Section {name!r} not found in document {self.id!r}")

    def replace_section(
        self,
        name: str,
        content: str,
        *,
        reason: Optional[str] = None,
    ) -> None:
        """Replace the body blocks of a named section with content parsed from markdown.

        The heading itself is preserved. Body blocks are deleted (one revision
        row per deleted block) and content is split on blank lines into
        paragraph blocks inserted in their place.

        Uses ``BEGIN IMMEDIATE`` and reads the section under the write lock
        so the section traversal and the subsequent mutations are atomic.
        """
        # Normalise CRLF before splitting so Windows-flavoured input parses correctly.
        content = content.replace("\r\n", "\n")
        new_paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        now = _now_iso()

        with open_db(self.database) as conn:
            conn.execute("BEGIN IMMEDIATE")

            # Read every block under the write lock and find the named section.
            rows = conn.execute(
                "SELECT id, type, content, position, metadata, revision "
                "FROM scribe_blocks WHERE doc_id = ? ORDER BY position",
                (self.id,),
            ).fetchall()

            head_idx: Optional[int] = None
            head_level = 1
            for i, r in enumerate(rows):
                if r["type"] == "heading" and r["content"] == name:
                    head_idx = i
                    head_level = int(json.loads(r["metadata"]).get("level", 1))
                    break
            if head_idx is None:
                raise KeyError(f"Section {name!r} not found in document {self.id!r}")

            end = len(rows)
            for j in range(head_idx + 1, len(rows)):
                nr = rows[j]
                if (
                    nr["type"] == "heading"
                    and int(json.loads(nr["metadata"]).get("level", 1)) <= head_level
                ):
                    end = j
                    break

            head = rows[head_idx]
            body = rows[head_idx + 1 : end]
            body_ids = [b["id"] for b in body]

            # Log deletion of each existing body block as a revision row.
            for b in body:
                new_rev = b["revision"] + 1
                conn.execute(
                    "INSERT INTO scribe_revisions "
                    "(block_id, revision, previous_content, new_content, reason, created_at) "
                    "VALUES (?, ?, ?, '', ?, ?)",
                    (b["id"], new_rev, b["content"], reason, now),
                )
            if body_ids:
                placeholders = ",".join("?" for _ in body_ids)
                conn.execute(
                    f"DELETE FROM scribe_blocks WHERE id IN ({placeholders})",
                    body_ids,
                )

            # Compact remaining positions, then insert new paragraphs after the heading.
            conn.execute(
                "UPDATE scribe_blocks "
                "SET position = position - ? "
                "WHERE doc_id = ? AND position > ?",
                (len(body), self.id, head["position"]),
            )
            insert_pos = head["position"] + 1
            # Make room for new paragraphs by pushing later blocks down again.
            conn.execute(
                "UPDATE scribe_blocks "
                "SET position = position + ? "
                "WHERE doc_id = ? AND position >= ?",
                (len(new_paragraphs), self.id, insert_pos),
            )
            for offset, para_text in enumerate(new_paragraphs):
                conn.execute(
                    "INSERT INTO scribe_blocks "
                    "(id, doc_id, type, content, position, parent_id, metadata, "
                    " revision, created_at, updated_at) "
                    "VALUES (?, ?, 'paragraph', ?, ?, NULL, '{}', 1, ?, ?)",
                    (_new_id(), self.id, para_text, insert_pos + offset, now, now),
                )
            conn.commit()

    def insert_into_section(
        self,
        section_name: str,
        block_spec: dict[str, Any],
        *,
        position: str = "end",
    ) -> str:
        """Insert a block at the end of (or beginning of) a named section.

        `position` is either "end" (append after the section's current last
        block) or "start" (insert immediately after the heading). Returns
        the new block id.

        The section lookup and the position-shift happen under
        ``BEGIN IMMEDIATE`` so the operation is atomic against concurrent
        writers.
        """
        if position not in ("end", "start"):
            raise ValueError(f"position must be 'end' or 'start', got {position!r}")

        bid = _new_id()
        now = _now_iso()

        with open_db(self.database) as conn:
            conn.execute("BEGIN IMMEDIATE")

            rows = conn.execute(
                "SELECT id, type, content, position, metadata "
                "FROM scribe_blocks WHERE doc_id = ? ORDER BY position",
                (self.id,),
            ).fetchall()

            head_idx: Optional[int] = None
            head_level = 1
            for i, r in enumerate(rows):
                if r["type"] == "heading" and r["content"] == section_name:
                    head_idx = i
                    head_level = int(json.loads(r["metadata"]).get("level", 1))
                    break
            if head_idx is None:
                raise KeyError(
                    f"Section {section_name!r} not found in document {self.id!r}"
                )

            end = len(rows)
            for j in range(head_idx + 1, len(rows)):
                nr = rows[j]
                if (
                    nr["type"] == "heading"
                    and int(json.loads(nr["metadata"]).get("level", 1)) <= head_level
                ):
                    end = j
                    break

            head = rows[head_idx]
            section_rows = rows[head_idx:end]

            if position == "end":
                last = section_rows[-1]
                insert_pos = last["position"] + 1
            else:  # "start"
                insert_pos = head["position"] + 1

            conn.execute(
                "UPDATE scribe_blocks "
                "SET position = position + 1 "
                "WHERE doc_id = ? AND position >= ?",
                (self.id, insert_pos),
            )
            conn.execute(
                "INSERT INTO scribe_blocks "
                "(id, doc_id, type, content, position, parent_id, metadata, "
                " revision, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, NULL, ?, 1, ?, ?)",
                (
                    bid,
                    self.id,
                    block_spec["type"],
                    block_spec.get("content", ""),
                    insert_pos,
                    json.dumps(block_spec.get("metadata", {})),
                    now,
                    now,
                ),
            )
            conn.commit()

        return bid

    # ------------------------------------------------------------------
    # Reference management
    # ------------------------------------------------------------------

    def add_reference(
        self,
        *,
        cite_key: str,
        bibtex: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Attach a bibliographic reference to this document."""
        rid = _new_id()
        now = _now_iso()
        with open_db(self.database) as conn:
            conn.execute(
                "INSERT INTO scribe_references "
                "(id, doc_id, cite_key, bibtex_entry, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (rid, self.id, cite_key, bibtex, json.dumps(metadata or {}), now),
            )
            conn.commit()
        return rid

    def references(self) -> list[Reference]:
        """Return all references attached to this document."""
        with open_db(self.database) as conn:
            rows = conn.execute(
                "SELECT id, doc_id, cite_key, bibtex_entry, metadata, created_at "
                "FROM scribe_references WHERE doc_id = ? ORDER BY created_at",
                (self.id,),
            ).fetchall()
        return [
            Reference(
                id=r["id"],
                doc_id=r["doc_id"],
                cite_key=r["cite_key"],
                bibtex_entry=r["bibtex_entry"],
                metadata=json.loads(r["metadata"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def citations(self) -> list[str]:
        """Return all citation keys used in paragraph blocks (deduped)."""
        seen: list[str] = []
        for blk in self.query(type="paragraph"):
            for key in extract_citation_keys(blk.content):
                if key not in seen:
                    seen.append(key)
        return seen

    def validate(self) -> list[ValidationIssue]:
        """Run structural validators. See validate.validate_document."""
        return validate_document(self)

    def render(self, output_path: str, *, format: str = "docx") -> None:
        """Render this document to a file. v1 supports format='docx' only."""
        if format == "docx":
            from .render_docx import render_to_docx

            render_to_docx(self, output_path)
        else:
            raise ValueError(f"Unsupported format {format!r}. v1 supports: 'docx'.")
