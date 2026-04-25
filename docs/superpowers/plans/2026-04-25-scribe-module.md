# Scribe Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained module for drafting structured documents in Andamentum, with markdown blocks as the primitive, SQLite as the source of truth, and one-way `.docx` rendering. Scope explicitly extended so scribe replaces the existing `document-tools:doc-draft` plugin (Word files only — `.pptx` stays out of scope).

**Architecture:** Block-based document model (`documents`, `blocks`, `references`, `revisions` tables). Markdown is the canonical body format; `.docx` is a derived artifact (no round-tripping). Sections are derived from heading blocks (no separate `sections` table). HTML/PDF render path goes through the existing `andamentum.typeset` module by mapping blocks → typeset atoms. Storage uses stdlib `sqlite3` (lifted from document_store's pattern) but lives in its own database file under `~/.local/share/scribe/<name>.db`. The module is fully decoupled from `whetstone`; the interchange format with whetstone is the rendered `.docx` itself.

**Tech Stack:** Python 3.13, SQLite (stdlib `sqlite3`), `python-docx` (already a dep), `python-markdown` (already a dep), `pydantic` v2 (already a dep), `lxml` (already a dep). No new dependencies.

---

## Decisions locked in

These were settled during the design conversation. Anything not listed is out of scope for v1.

1. **Module name: `andamentum.scribe`.** Single word, evocative, matches `whetstone`/`typeset` naming style. If the reviewer wants a different name, change it once in Task 1 and the rest of the plan uses it consistently — replace `scribe` everywhere before starting.
2. **Block types in v1: `paragraph`, `heading`, `figure`, `table`.** The schema uses a flexible `type TEXT` column with a JSON `metadata` blob; adding `equation`, `code`, `callout` later is a renderer change, not a schema migration.
3. **Sections are derived, not stored.** A "section" is a heading block plus all blocks whose position falls between it and the next heading at the same or higher level. `Document.list_sections()`, `Document.section(name)`, and `Document.replace_section(name, content)` are query/transform operations over existing block rows.
4. **Built-in scaffolds.** `Document.create(scaffold="article")` and `scaffold="grant"` pre-populate canonical section structures (heading + placeholder paragraph each). Definitions live in a small `scaffolds.py` module, sourced from the structures in manuscript-tools' `section-guides.md`.
5. **Inline citations** are markdown spans inside paragraph bodies (Pandoc-flavoured `[@smith2023]`), not block types. They're extracted by regex when needed. The parser also recognises `[verify]` and `[citation needed]` markers; `validate()` reports unresolved markers.
6. **References are first-class.** `references` table stores `cite_key` + raw BibTeX. v1 stores BibTeX strings as-is; cite-key parsing/normalisation is v2.
7. **No round-tripping.** The module renders to `.docx` but never reads `.docx`. Whetstone handles review side independently.
8. **Storage:** Separate database directory `~/.local/share/scribe/` (override with `SCRIBE_DIR` env var). Tables prefixed `scribe_` so a future merge with document_store is collision-free.
9. **Optimistic locking:** Each block carries a `revision` counter. `Document.replace()` checks-and-bumps; stale revision raises `StaleRevisionError`.
10. **Audit trail:** Append-only `scribe_revisions` table written on every `replace()` (also written by `replace_section` for each affected block).
11. **HTML/PDF rendering:** Goes through `typeset`. Scribe owns block→atom mapping; typeset owns atoms→HTML/PDF.
12. **Inline markdown in `.docx`:** Bold (`**x**`), italic (`*x*`), and inline code (`` `x` ``) inside paragraph content render as styled runs. More complex inline syntax falls back to plain text.
13. **Whetstone integration: NONE.** No imports either direction. Hand-off via `.docx` file only.
14. **Doc-draft replacement:** Scribe's CLI mirrors `document-tools:doc-draft` subcommands (`list-sections`, `read-section`, `write-section`, `insert-figure`, `insert-table`) so the user can swap mental models 1:1. PPTX stays with doc-draft.

## File structure

| Path | Responsibility |
|---|---|
| `src/andamentum/scribe/__init__.py` | Public API exports |
| `src/andamentum/scribe/schema.py` | SQL DDL + migration runner |
| `src/andamentum/scribe/database.py` | Path resolution + connection helper |
| `src/andamentum/scribe/models.py` | Pydantic models: `Block`, `Reference`, `Revision`, `ValidationIssue`, `StaleRevisionError`, `BlockType` |
| `src/andamentum/scribe/api.py` | `Document` class: `create`, `open`, `append`, `query`, `replace`, `list_sections`, `section`, `replace_section`, `add_reference`, `citations`, `validate`, `render` |
| `src/andamentum/scribe/parser.py` | Markdown body parsing: citation keys + inline-format runs |
| `src/andamentum/scribe/scaffolds.py` | Built-in document templates (`article`, `grant`) |
| `src/andamentum/scribe/render_typeset.py` | Block list → typeset atom dicts (HTML/PDF path) |
| `src/andamentum/scribe/render_docx.py` | Block list → `python-docx` Document; inline runs; template-aware |
| `src/andamentum/scribe/validate.py` | Citation-key resolution, figure-path existence, unresolved-marker reporting |
| `src/andamentum/scribe/cli.py` | `andamentum-scribe` CLI: `init`, `list-sections`, `read-section`, `write-section`, `insert-figure`, `insert-table`, `render` |
| `src/andamentum/scribe/tests/` | Test files mirroring the above |
| `pyproject.toml` | Register `andamentum-scribe` script |

## Test layout

Per CLAUDE.md convention: `src/andamentum/scribe/tests/` next to the code, no top-level `tests/` directory. `asyncio_mode = "auto"` is already set globally — no `@pytest.mark.asyncio` needed if/when async functions appear (this module is synchronous).

---

## Task 1: Package skeleton + schema

**Files:**
- Create: `src/andamentum/scribe/__init__.py`
- Create: `src/andamentum/scribe/schema.py`
- Create: `src/andamentum/scribe/py.typed` (empty file, signals type-checked package)
- Create: `src/andamentum/scribe/tests/__init__.py` (empty)
- Create: `src/andamentum/scribe/tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# src/andamentum/scribe/tests/test_schema.py
"""Schema creation and migration tests."""

import sqlite3

from andamentum.scribe.schema import init_schema, SCHEMA_VERSION


def test_init_schema_creates_all_tables():
    conn = sqlite3.connect(":memory:")
    init_schema(conn)

    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name LIKE 'scribe_%' ORDER BY name"
    )
    tables = [row[0] for row in cur.fetchall()]
    assert tables == [
        "scribe_blocks",
        "scribe_documents",
        "scribe_meta",
        "scribe_references",
        "scribe_revisions",
    ]


def test_init_schema_records_version():
    conn = sqlite3.connect(":memory:")
    init_schema(conn)
    cur = conn.execute("SELECT value FROM scribe_meta WHERE key='schema_version'")
    assert cur.fetchone()[0] == str(SCHEMA_VERSION)


def test_init_schema_is_idempotent():
    conn = sqlite3.connect(":memory:")
    init_schema(conn)
    init_schema(conn)  # must not raise
    cur = conn.execute("SELECT count(*) FROM scribe_meta WHERE key='schema_version'")
    assert cur.fetchone()[0] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/andamentum/scribe/tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'andamentum.scribe'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/andamentum/scribe/__init__.py
"""andamentum.scribe — structured document drafting.

Block-based document authoring. Markdown is the source of truth;
.docx is a derived artifact. See docs/superpowers/plans/2026-04-25-scribe-module.md
for design rationale.
"""

__version__ = "0.1.0"
__all__: list[str] = []  # populated as public surface lands
```

```python
# src/andamentum/scribe/schema.py
"""SQL DDL and schema migration for scribe.

All scribe tables are prefixed `scribe_` so a future merge into the
document_store database remains collision-free. Schema version is
recorded in `scribe_meta` so migrations can be applied conditionally.
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 1

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS scribe_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scribe_documents (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        template TEXT,
        metadata TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scribe_blocks (
        id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL REFERENCES scribe_documents(id) ON DELETE CASCADE,
        type TEXT NOT NULL,
        content TEXT NOT NULL DEFAULT '',
        position INTEGER NOT NULL,
        parent_id TEXT REFERENCES scribe_blocks(id) ON DELETE CASCADE,
        metadata TEXT NOT NULL DEFAULT '{}',
        revision INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS scribe_blocks_doc_position "
    "ON scribe_blocks(doc_id, position)",
    """
    CREATE TABLE IF NOT EXISTS scribe_references (
        id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL REFERENCES scribe_documents(id) ON DELETE CASCADE,
        cite_key TEXT NOT NULL,
        bibtex_entry TEXT,
        metadata TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        UNIQUE(doc_id, cite_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scribe_revisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        block_id TEXT NOT NULL,
        revision INTEGER NOT NULL,
        previous_content TEXT NOT NULL,
        new_content TEXT NOT NULL,
        reason TEXT,
        created_at TEXT NOT NULL
    )
    """,
]


def init_schema(conn: sqlite3.Connection) -> None:
    """Create scribe tables and record the schema version. Idempotent."""
    conn.execute("PRAGMA foreign_keys = ON")
    for stmt in _DDL:
        conn.execute(stmt)
    conn.execute(
        "INSERT OR IGNORE INTO scribe_meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/andamentum/scribe/tests/test_schema.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/scribe/__init__.py src/andamentum/scribe/schema.py \
        src/andamentum/scribe/py.typed src/andamentum/scribe/tests/__init__.py \
        src/andamentum/scribe/tests/test_schema.py
git commit -m "feat(scribe): package skeleton + schema DDL"
```

---

## Task 2: Database connection + path resolution

**Files:**
- Create: `src/andamentum/scribe/database.py`
- Create: `src/andamentum/scribe/tests/test_database.py`

- [ ] **Step 1: Write the failing test**

```python
# src/andamentum/scribe/tests/test_database.py
"""Database path resolution and connection tests."""

from pathlib import Path

from andamentum.scribe.database import (
    get_databases_dir,
    get_database_path,
    open_db,
)


def test_get_databases_dir_default(monkeypatch):
    monkeypatch.delenv("SCRIBE_DIR", raising=False)
    monkeypatch.setenv("HOME", "/tmp/fake_home")
    assert get_databases_dir() == Path("/tmp/fake_home/.local/share/scribe")


def test_get_databases_dir_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    assert get_databases_dir() == tmp_path


def test_get_database_path_appends_extension(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    assert get_database_path("my_paper") == tmp_path / "my_paper.db"


def test_open_db_creates_file_and_initialises_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    with open_db("test_doc") as conn:
        cur = conn.execute(
            "SELECT value FROM scribe_meta WHERE key='schema_version'"
        )
        assert cur.fetchone() is not None
    assert (tmp_path / "test_doc.db").exists()


def test_open_db_in_memory():
    with open_db(":memory:") as conn:
        cur = conn.execute(
            "SELECT value FROM scribe_meta WHERE key='schema_version'"
        )
        assert cur.fetchone() is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/andamentum/scribe/tests/test_database.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'andamentum.scribe.database'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/andamentum/scribe/database.py
"""Database path resolution and connection management.

Reuses sqlite3 directly. We intentionally do NOT load sqlite-vec here —
scribe has no embedding needs in v1. If that changes, swap to
`document_store.connection.get_connection`.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .schema import init_schema

DB_DIR_ENV_VAR = "SCRIBE_DIR"
_DEFAULT_SUBDIR = ".local/share/scribe"


def get_databases_dir() -> Path:
    """Return the directory that holds scribe databases.

    Honours $SCRIBE_DIR; otherwise falls back to ~/.local/share/scribe/.
    """
    override = os.environ.get(DB_DIR_ENV_VAR)
    if override:
        return Path(override)
    return Path(os.environ["HOME"]) / _DEFAULT_SUBDIR


def get_database_path(name: str) -> Path:
    """Return the .db path for a named scribe database."""
    return get_databases_dir() / f"{name}.db"


@contextmanager
def open_db(name_or_path: str) -> Iterator[sqlite3.Connection]:
    """Open (and migrate, if needed) a scribe database.

    `name_or_path` is either a logical database name ("my_paper") or
    the literal ":memory:" for an in-memory store. File-based DBs are
    created on first use.
    """
    if name_or_path == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        path = get_database_path(name_or_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))

    conn.row_factory = sqlite3.Row
    init_schema(conn)
    try:
        yield conn
    finally:
        conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/andamentum/scribe/tests/test_database.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/scribe/database.py src/andamentum/scribe/tests/test_database.py
git commit -m "feat(scribe): database path resolution and connection helper"
```

---

## Task 3: Pydantic models

**Files:**
- Create: `src/andamentum/scribe/models.py`
- Create: `src/andamentum/scribe/tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# src/andamentum/scribe/tests/test_models.py
"""Pydantic model tests."""

import pytest

from andamentum.scribe.models import (
    Block,
    BlockType,
    Reference,
    Revision,
    StaleRevisionError,
    ValidationIssue,
)


def test_block_paragraph_minimal():
    b = Block(
        id="b1", doc_id="d1", type="paragraph", content="Hello.", position=0
    )
    assert b.revision == 1
    assert b.metadata == {}
    assert b.parent_id is None


def test_block_heading_carries_level_in_metadata():
    b = Block(
        id="b2",
        doc_id="d1",
        type="heading",
        content="Introduction",
        position=0,
        metadata={"level": 1},
    )
    assert b.metadata["level"] == 1


def test_block_figure_carries_path_caption_label():
    b = Block(
        id="b3",
        doc_id="d1",
        type="figure",
        content="",
        position=0,
        metadata={"path": "fig1.png", "caption": "Overview", "label": "fig:overview"},
    )
    assert b.metadata["label"] == "fig:overview"


def test_block_table_carries_rows():
    b = Block(
        id="b4",
        doc_id="d1",
        type="table",
        content="",
        position=0,
        metadata={
            "rows": [["a", "b"], ["1", "2"]],
            "header_row": True,
            "caption": "Demo",
            "label": "tab:demo",
        },
    )
    assert b.metadata["rows"][0] == ["a", "b"]


def test_block_rejects_unknown_type():
    with pytest.raises(ValueError):
        Block(id="bx", doc_id="d1", type="bogus", content="", position=0)


def test_reference_minimal():
    r = Reference(id="r1", doc_id="d1", cite_key="smith2023")
    assert r.bibtex_entry is None


def test_validation_issue_severity_constraint():
    ValidationIssue(severity="error", message="missing", location="b1")
    with pytest.raises(ValueError):
        ValidationIssue(severity="catastrophic", message="x", location="b1")


def test_stale_revision_error_carries_context():
    err = StaleRevisionError(block_id="b1", expected=1, actual=3)
    assert "b1" in str(err)
    assert err.expected == 1
    assert err.actual == 3


def test_block_type_literal_values():
    expected = {"paragraph", "heading", "figure", "table"}
    assert set(BlockType.__args__) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/andamentum/scribe/tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'andamentum.scribe.models'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/andamentum/scribe/models.py
"""Pydantic models for scribe documents, blocks, references."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

BlockType = Literal["paragraph", "heading", "figure", "table"]
Severity = Literal["error", "warning", "info"]


class Block(BaseModel):
    """A single block in a scribe document."""

    id: str
    doc_id: str
    type: BlockType
    content: str = ""
    position: int
    parent_id: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    revision: int = 1
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class Reference(BaseModel):
    """A bibliographic reference attached to a document."""

    id: str
    doc_id: str
    cite_key: str
    bibtex_entry: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    created_at: Optional[str] = None


class Revision(BaseModel):
    """An audit-log entry for a single block edit."""

    id: Optional[int] = None
    block_id: str
    revision: int
    previous_content: str
    new_content: str
    reason: Optional[str] = None
    created_at: Optional[str] = None


class ValidationIssue(BaseModel):
    """One structural issue surfaced by Document.validate()."""

    severity: Severity
    message: str
    location: str  # block_id, cite_key, or other identifier


class StaleRevisionError(Exception):
    """Raised when Document.replace sees a revision mismatch."""

    def __init__(self, block_id: str, expected: int, actual: int):
        super().__init__(
            f"Stale revision for block {block_id}: "
            f"expected {expected}, found {actual}"
        )
        self.block_id = block_id
        self.expected = expected
        self.actual = actual
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/andamentum/scribe/tests/test_models.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/scribe/models.py src/andamentum/scribe/tests/test_models.py
git commit -m "feat(scribe): pydantic models for blocks, references, revisions"
```

---

## Task 4: Document.create + Document.open

**Files:**
- Create: `src/andamentum/scribe/api.py`
- Create: `src/andamentum/scribe/tests/test_document_lifecycle.py`

- [ ] **Step 1: Write the failing test**

```python
# src/andamentum/scribe/tests/test_document_lifecycle.py
"""Tests for Document.create and Document.open."""

import pytest

from andamentum.scribe.api import Document


def test_create_returns_document_with_id_and_title(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="My Paper", database="test")
    assert doc.title == "My Paper"
    assert isinstance(doc.id, str)
    assert len(doc.id) >= 8  # uuid hex prefix


def test_open_round_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="Original", database="test")
    doc_id = doc.id

    reopened = Document.open(doc_id, database="test")
    assert reopened.title == "Original"
    assert reopened.id == doc_id


def test_open_unknown_id_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    Document.create(title="exists", database="test")  # ensures DB file exists
    with pytest.raises(KeyError, match="not found"):
        Document.open("does-not-exist", database="test")


def test_create_records_template_when_provided(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="test", template="nature.docx")
    reopened = Document.open(doc.id, database="test")
    assert reopened.template == "nature.docx"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/andamentum/scribe/tests/test_document_lifecycle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'andamentum.scribe.api'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/andamentum/scribe/api.py
"""Public Document API.

The Document class is the single entry point for callers. All
mutations go through it; direct SQL is internal.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from .database import open_db

if TYPE_CHECKING:
    from .models import Block, Reference, ValidationIssue


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/andamentum/scribe/tests/test_document_lifecycle.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/scribe/api.py src/andamentum/scribe/tests/test_document_lifecycle.py
git commit -m "feat(scribe): Document.create and Document.open"
```

---

## Task 5: Document.append + Document.query (paragraph, heading, figure, table)

**Files:**
- Modify: `src/andamentum/scribe/api.py` (add `append`, `query`, factory helpers)
- Create: `src/andamentum/scribe/tests/test_blocks.py`

- [ ] **Step 1: Write the failing test**

```python
# src/andamentum/scribe/tests/test_blocks.py
"""Tests for block append and query."""

from andamentum.scribe.api import Document, Figure, Heading, Paragraph, Table


def test_append_returns_block_id(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    bid = doc.append(Heading("Intro", level=1))
    assert isinstance(bid, str)
    assert len(bid) >= 8


def test_append_increments_position(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Heading("Intro", level=1))
    doc.append(Paragraph("First paragraph."))
    doc.append(Paragraph("Second."))

    blocks = doc.query()
    positions = [b.position for b in blocks]
    assert positions == [0, 1, 2]


def test_query_filters_by_type(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Heading("Intro", level=1))
    doc.append(Paragraph("Body."))
    doc.append(Figure(path="f.png", caption="C", label="fig:c"))

    paragraphs = doc.query(type="paragraph")
    assert len(paragraphs) == 1
    assert paragraphs[0].content == "Body."


def test_query_returns_blocks_in_position_order(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    for i in range(5):
        doc.append(Paragraph(f"P{i}"))
    blocks = doc.query()
    assert [b.content for b in blocks] == [f"P{i}" for i in range(5)]


def test_factory_heading_carries_level():
    blk = Heading("Methods", level=2)
    assert blk["type"] == "heading"
    assert blk["metadata"]["level"] == 2


def test_factory_figure_carries_metadata():
    blk = Figure(path="x.png", caption="cap", label="fig:x")
    assert blk["type"] == "figure"
    assert blk["metadata"] == {
        "path": "x.png",
        "caption": "cap",
        "label": "fig:x",
        "width_in": None,
    }


def test_factory_figure_with_width():
    blk = Figure(path="x.png", caption="c", label="fig:x", width_in=4.5)
    assert blk["metadata"]["width_in"] == 4.5


def test_factory_table_carries_rows_and_caption():
    blk = Table(
        rows=[["a", "b"], ["1", "2"]],
        header_row=True,
        caption="demo",
        label="tab:demo",
    )
    assert blk["type"] == "table"
    assert blk["metadata"]["rows"] == [["a", "b"], ["1", "2"]]
    assert blk["metadata"]["caption"] == "demo"


def test_factory_heading_rejects_invalid_level():
    import pytest

    with pytest.raises(ValueError):
        Heading("X", level=0)
    with pytest.raises(ValueError):
        Heading("X", level=7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/andamentum/scribe/tests/test_blocks.py -v`
Expected: FAIL — `ImportError: cannot import name 'Heading' from 'andamentum.scribe.api'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/andamentum/scribe/api.py`:

```python
from .models import Block

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


# Add as methods on Document (insert before the closing of the class):

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

    def query(self, *, type: Optional[str] = None) -> list["Block"]:
        """Return blocks for this document, ordered by position."""
        from .models import Block

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/andamentum/scribe/tests/test_blocks.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/scribe/api.py src/andamentum/scribe/tests/test_blocks.py
git commit -m "feat(scribe): block append/query with paragraph/heading/figure/table factories"
```

---

## Task 6: Document.replace + revision tracking

**Files:**
- Modify: `src/andamentum/scribe/api.py` (add `replace`)
- Create: `src/andamentum/scribe/tests/test_replace.py`

- [ ] **Step 1: Write the failing test**

```python
# src/andamentum/scribe/tests/test_replace.py
"""Tests for Document.replace and revision audit trail."""

import pytest

from andamentum.scribe.api import Document, Paragraph
from andamentum.scribe.database import open_db
from andamentum.scribe.models import StaleRevisionError


def test_replace_bumps_revision_and_updates_content(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    bid = doc.append(Paragraph("v1"))

    doc.replace(bid, "v2", expected_revision=1, reason="pass-2")

    blk = doc.query()[0]
    assert blk.content == "v2"
    assert blk.revision == 2


def test_replace_writes_revision_row(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    bid = doc.append(Paragraph("v1"))
    doc.replace(bid, "v2", expected_revision=1, reason="pass-2")

    with open_db("t") as conn:
        row = conn.execute(
            "SELECT previous_content, new_content, reason, revision "
            "FROM scribe_revisions WHERE block_id = ?",
            (bid,),
        ).fetchone()
    assert row["previous_content"] == "v1"
    assert row["new_content"] == "v2"
    assert row["reason"] == "pass-2"
    assert row["revision"] == 2


def test_replace_stale_revision_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    bid = doc.append(Paragraph("v1"))
    doc.replace(bid, "v2", expected_revision=1)

    with pytest.raises(StaleRevisionError) as excinfo:
        doc.replace(bid, "v3", expected_revision=1)
    assert excinfo.value.expected == 1
    assert excinfo.value.actual == 2


def test_replace_unknown_block_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    with pytest.raises(KeyError):
        doc.replace("nope", "x", expected_revision=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/andamentum/scribe/tests/test_replace.py -v`
Expected: FAIL — `AttributeError: 'Document' object has no attribute 'replace'`

- [ ] **Step 3: Write minimal implementation**

Add to `Document` class in `src/andamentum/scribe/api.py`:

```python
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
        """
        from .models import StaleRevisionError

        now = _now_iso()
        with open_db(self.database) as conn:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/andamentum/scribe/tests/test_replace.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/scribe/api.py src/andamentum/scribe/tests/test_replace.py
git commit -m "feat(scribe): Document.replace with optimistic locking and audit trail"
```

---

## Task 7: Section abstraction (`list_sections`, `section`, `replace_section`)

A "section" is a heading block plus all subsequent blocks until the next heading at the same or higher level. Sections are derived from `scribe_blocks` rows; no schema change.

**Files:**
- Modify: `src/andamentum/scribe/api.py` (add `list_sections`, `section`, `replace_section`)
- Create: `src/andamentum/scribe/tests/test_sections.py`

- [ ] **Step 1: Write the failing test**

```python
# src/andamentum/scribe/tests/test_sections.py
"""Tests for section query and replacement."""

import pytest

from andamentum.scribe.api import Document, Heading, Paragraph


def _seed(monkeypatch, tmp_path) -> Document:
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Heading("Introduction", level=1))
    doc.append(Paragraph("Intro body 1."))
    doc.append(Paragraph("Intro body 2."))
    doc.append(Heading("Methods", level=1))
    doc.append(Paragraph("Methods body."))
    doc.append(Heading("Sub-method", level=2))
    doc.append(Paragraph("Sub body."))
    doc.append(Heading("Results", level=1))
    doc.append(Paragraph("Results body."))
    return doc


def test_list_sections_returns_top_level_headings_in_order(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    sections = doc.list_sections()
    names = [s["name"] for s in sections]
    assert names == ["Introduction", "Methods", "Results"]


def test_list_sections_reports_block_and_word_counts(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    sections = doc.list_sections()
    intro = next(s for s in sections if s["name"] == "Introduction")
    # 2 paragraph blocks under "Introduction"
    assert intro["block_count"] == 2
    # 4 words: "Intro body 1." + "Intro body 2." → "Intro body 1 Intro body 2" → 6
    assert intro["word_count"] == 6


def test_section_returns_blocks_under_heading(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    blocks = doc.section("Methods")
    types = [b.type for b in blocks]
    # Methods heading + paragraph + sub-heading + sub-paragraph
    assert types == ["heading", "paragraph", "heading", "paragraph"]
    assert blocks[0].content == "Methods"


def test_section_unknown_name_raises(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    with pytest.raises(KeyError, match="Nope"):
        doc.section("Nope")


def test_replace_section_swaps_body_blocks(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    doc.replace_section(
        "Introduction",
        "First new paragraph.\n\nSecond new paragraph.",
        reason="rewrite",
    )

    blocks = doc.section("Introduction")
    # Heading preserved + 2 new paragraphs
    assert [b.type for b in blocks] == ["heading", "paragraph", "paragraph"]
    assert blocks[1].content == "First new paragraph."
    assert blocks[2].content == "Second new paragraph."


def test_replace_section_preserves_following_sections(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    doc.replace_section("Introduction", "New intro.")
    names = [s["name"] for s in doc.list_sections()]
    assert names == ["Introduction", "Methods", "Results"]


def test_replace_section_writes_revisions_for_each_removed_block(monkeypatch, tmp_path):
    from andamentum.scribe.database import open_db

    doc = _seed(monkeypatch, tmp_path)
    doc.replace_section("Introduction", "New intro body.", reason="rewrite")

    with open_db("t") as conn:
        rows = conn.execute(
            "SELECT reason FROM scribe_revisions WHERE reason = ?",
            ("rewrite",),
        ).fetchall()
    # Two original paragraph blocks removed → two revision rows logging the deletion
    assert len(rows) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/andamentum/scribe/tests/test_sections.py -v`
Expected: FAIL — `AttributeError: 'Document' object has no attribute 'list_sections'`

- [ ] **Step 3: Write minimal implementation**

Add to `Document` class in `src/andamentum/scribe/api.py`:

```python
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

    def section(self, name: str) -> list["Block"]:
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
        """
        section_blocks = self.section(name)
        head = section_blocks[0]
        body = section_blocks[1:]
        body_ids = [b.id for b in body]

        new_paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        now = _now_iso()

        with open_db(self.database) as conn:
            # Log deletion of each existing body block as a revision row
            for b in body:
                new_rev = b.revision + 1
                conn.execute(
                    "INSERT INTO scribe_revisions "
                    "(block_id, revision, previous_content, new_content, reason, created_at) "
                    "VALUES (?, ?, ?, '', ?, ?)",
                    (b.id, new_rev, b.content, reason, now),
                )
            if body_ids:
                placeholders = ",".join("?" for _ in body_ids)
                conn.execute(
                    f"DELETE FROM scribe_blocks WHERE id IN ({placeholders})",
                    body_ids,
                )

            # Compact remaining positions, then insert new paragraphs after the heading
            conn.execute(
                "UPDATE scribe_blocks "
                "SET position = position - ? "
                "WHERE doc_id = ? AND position > ?",
                (len(body), self.id, head.position),
            )
            insert_pos = head.position + 1
            # Make room for new paragraphs by pushing later blocks down again
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/andamentum/scribe/tests/test_sections.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/scribe/api.py src/andamentum/scribe/tests/test_sections.py
git commit -m "feat(scribe): section abstraction (list_sections, section, replace_section)"
```

---

## Task 8: Built-in scaffolds (article, grant)

`Document.create(scaffold="article")` pre-populates standard sections. Scaffold definitions live in `scaffolds.py` so they can be extended without touching the API.

**Files:**
- Create: `src/andamentum/scribe/scaffolds.py`
- Modify: `src/andamentum/scribe/api.py` (extend `create` with `scaffold` keyword)
- Create: `src/andamentum/scribe/tests/test_scaffolds.py`

- [ ] **Step 1: Write the failing test**

```python
# src/andamentum/scribe/tests/test_scaffolds.py
"""Tests for built-in scaffolds."""

import pytest

from andamentum.scribe.api import Document
from andamentum.scribe.scaffolds import SCAFFOLDS


def test_article_scaffold_creates_standard_sections(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t", scaffold="article")
    sections = [s["name"] for s in doc.list_sections()]
    assert sections == [
        "Abstract",
        "Introduction",
        "Methods",
        "Results",
        "Discussion",
        "References",
    ]


def test_grant_scaffold_creates_standard_sections(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="G", database="t", scaffold="grant")
    sections = [s["name"] for s in doc.list_sections()]
    assert sections == [
        "Specific Aims",
        "Background and Significance",
        "Innovation",
        "Approach",
        "Timeline and Milestones",
        "References",
    ]


def test_scaffold_includes_guide_metadata_on_placeholder_paragraphs(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t", scaffold="article")
    intro_blocks = doc.section("Introduction")
    # heading + placeholder paragraph
    assert intro_blocks[1].type == "paragraph"
    assert "guide" in intro_blocks[1].metadata
    assert "funnel" in intro_blocks[1].metadata["guide"].lower()


def test_unknown_scaffold_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="Unknown scaffold"):
        Document.create(title="P", database="t", scaffold="bogus")


def test_scaffolds_constant_is_well_formed():
    for name, sections in SCAFFOLDS.items():
        assert isinstance(name, str)
        assert len(sections) >= 2
        for section_name, guide in sections:
            assert isinstance(section_name, str) and section_name
            assert isinstance(guide, str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/andamentum/scribe/tests/test_scaffolds.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'andamentum.scribe.scaffolds'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/andamentum/scribe/scaffolds.py
"""Built-in document scaffolds.

Each scaffold is a list of (section_name, guide_text) tuples. When a
document is created with `scaffold=<name>`, scribe inserts one level-1
heading per section and one placeholder paragraph carrying the guide
text in `metadata.guide` for downstream agents to consume.

Guide text is sourced from the section structures in
manuscript-tools/section-guides.md and is intentionally short.
"""

from __future__ import annotations

ARTICLE: list[tuple[str, str]] = [
    (
        "Abstract",
        "Background → gap → approach → key results → significance. 150-300 words.",
    ),
    (
        "Introduction",
        "Funnel: broad context → narrowing to gap → contribution. 500-1000 words.",
    ),
    (
        "Methods",
        "Reproducibility goal. Specific tools, versions, parameters. Logical order, not chronological.",
    ),
    (
        "Results",
        "Lead each paragraph with the finding, then the evidence. Reference every figure and table.",
    ),
    (
        "Discussion",
        "Restate finding in context. Compare with prior work. Limitations honestly. Concrete future directions.",
    ),
    ("References", ""),
]

GRANT: list[tuple[str, str]] = [
    (
        "Specific Aims",
        "One-page overview. State the long-term goal, the specific aims, and why the work matters.",
    ),
    (
        "Background and Significance",
        "Establish the problem, cite key prior work, identify the gap your work fills.",
    ),
    (
        "Innovation",
        "What is conceptually or methodologically new. Distinguish from incremental work.",
    ),
    (
        "Approach",
        "For each aim: rationale, methods, expected outcomes, alternative strategies, pitfalls.",
    ),
    (
        "Timeline and Milestones",
        "Project schedule with measurable deliverables per period.",
    ),
    ("References", ""),
]

SCAFFOLDS: dict[str, list[tuple[str, str]]] = {
    "article": ARTICLE,
    "grant": GRANT,
}
```

Modify `Document.create` in `src/andamentum/scribe/api.py`:

```python
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

        If `scaffold` is given (e.g. "article", "grant"), the document is
        pre-populated with the corresponding section structure. See
        scaffolds.py for available scaffolds.
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
                    f"Unknown scaffold {scaffold!r}. "
                    f"Available: {sorted(SCAFFOLDS)}"
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/andamentum/scribe/tests/test_scaffolds.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/scribe/scaffolds.py src/andamentum/scribe/api.py \
        src/andamentum/scribe/tests/test_scaffolds.py
git commit -m "feat(scribe): built-in scaffolds for article and grant"
```

---

## Task 9: References + citation extraction

**Files:**
- Modify: `src/andamentum/scribe/api.py` (add `add_reference`, `references`, `citations`)
- Create: `src/andamentum/scribe/parser.py`
- Create: `src/andamentum/scribe/tests/test_references.py`
- Create: `src/andamentum/scribe/tests/test_parser.py`

- [ ] **Step 1: Write the failing test**

```python
# src/andamentum/scribe/tests/test_parser.py
"""Markdown body parser tests (citation spans + verify markers + inline runs)."""

from andamentum.scribe.parser import (
    extract_citation_keys,
    find_unresolved_markers,
    inline_runs,
)


def test_extract_single_citation():
    assert extract_citation_keys("See [@smith2023].") == ["smith2023"]


def test_extract_multiple_citations():
    text = "Both [@smith2023] and [@jones2024] disagree."
    assert extract_citation_keys(text) == ["smith2023", "jones2024"]


def test_extract_handles_grouped_citations():
    text = "Many studies [@smith2023; @jones2024; @lee2022]."
    assert extract_citation_keys(text) == ["smith2023", "jones2024", "lee2022"]


def test_extract_ignores_email_like_atsigns():
    assert extract_citation_keys("Contact me at [me@example.com].") == []


def test_extract_returns_empty_for_no_citations():
    assert extract_citation_keys("plain text") == []


def test_find_unresolved_markers_verify():
    text = "Foundational work [verify] established the field."
    assert find_unresolved_markers(text) == ["verify"]


def test_find_unresolved_markers_citation_needed():
    text = "Some claim [citation needed]."
    assert find_unresolved_markers(text) == ["citation needed"]


def test_find_unresolved_markers_returns_empty_for_clean_text():
    assert find_unresolved_markers("Clean.") == []


def test_inline_runs_plain_text():
    runs = inline_runs("plain text")
    assert runs == [("plain text", set())]


def test_inline_runs_bold():
    runs = inline_runs("normal **bold** more")
    assert runs == [
        ("normal ", set()),
        ("bold", {"bold"}),
        (" more", set()),
    ]


def test_inline_runs_italic():
    runs = inline_runs("normal *italic* more")
    assert runs == [
        ("normal ", set()),
        ("italic", {"italic"}),
        (" more", set()),
    ]


def test_inline_runs_inline_code():
    runs = inline_runs("call `f(x)` now")
    assert runs == [
        ("call ", set()),
        ("f(x)", {"code"}),
        (" now", set()),
    ]
```

```python
# src/andamentum/scribe/tests/test_references.py
"""Reference management and Document.citations() tests."""

import pytest

from andamentum.scribe.api import Document, Paragraph


def test_add_reference_persists(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.add_reference(cite_key="smith2023", bibtex="@article{smith2023, ...}")

    refs = doc.references()
    assert len(refs) == 1
    assert refs[0].cite_key == "smith2023"


def test_add_reference_duplicate_key_raises(monkeypatch, tmp_path):
    import sqlite3

    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.add_reference(cite_key="smith2023")
    with pytest.raises(sqlite3.IntegrityError):
        doc.add_reference(cite_key="smith2023")


def test_citations_returns_keys_used_in_paragraphs(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Paragraph("As shown by [@smith2023] and [@jones2024]."))
    doc.append(Paragraph("Repeated: [@smith2023]."))

    keys = doc.citations()
    assert sorted(keys) == ["jones2024", "smith2023"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/andamentum/scribe/tests/test_parser.py src/andamentum/scribe/tests/test_references.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'andamentum.scribe.parser'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/andamentum/scribe/parser.py
"""Lightweight inline parsers for scribe markdown bodies.

For v1 we cover three concerns:
  - Pandoc-style citation spans: `[@key]` or `[@k1; @k2; @k3]`
  - Unresolved citation markers: `[verify]`, `[citation needed]`
    (compatible with the manuscript-tools/section-draft convention)
  - Inline formatting runs (bold/italic/code) for the docx renderer

We deliberately do NOT build a full markdown AST here — that's
typeset's job (via python-markdown). This module is for structural
extraction and cheap inline-styling tasks only.
"""

from __future__ import annotations

import re

# Match @key inside [...] but reject @keys that are part of an email
# (i.e. preceded by alphanumeric — `me@example.com`).
_CITATION_KEY_RE = re.compile(r"(?<![A-Za-z0-9])@([A-Za-z][A-Za-z0-9_:.-]*)")
_BRACKET_GROUP_RE = re.compile(r"\[(?P<body>[^\[\]]*)\]")
_UNRESOLVED_MARKERS = ("verify", "citation needed")

# Inline-formatting tokenizer: order matters — code spans first (so we
# don't mis-tokenise asterisks inside backticks), then bold, then italic.
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_CODE_RE = re.compile(r"`([^`]+)`")


def extract_citation_keys(markdown: str) -> list[str]:
    """Return citation keys (in order, preserving duplicates) from a body."""
    keys: list[str] = []
    for match in _BRACKET_GROUP_RE.finditer(markdown):
        body = match.group("body")
        for k in _CITATION_KEY_RE.findall(body):
            keys.append(k)
    return keys


def find_unresolved_markers(markdown: str) -> list[str]:
    """Return unresolved-citation markers found in body.

    Recognises `[verify]` and `[citation needed]`, the convention used
    by manuscript-tools/section-draft.
    """
    out: list[str] = []
    for match in _BRACKET_GROUP_RE.finditer(markdown):
        body = match.group("body").strip().lower()
        if body in _UNRESOLVED_MARKERS:
            out.append(body)
    return out


def inline_runs(markdown: str) -> list[tuple[str, set[str]]]:
    """Tokenise paragraph markdown into (text, styles) runs.

    Styles in `{"bold", "italic", "code"}`. Plain text has an empty
    set. Used by the docx renderer to emit styled python-docx runs.
    Unsupported markdown (links, etc.) falls back to plain text.
    """
    # Walk the string and emit runs; we use a state-machine pass that
    # alternates plain-text segments with styled segments.
    spans: list[tuple[int, int, str]] = []  # (start, end, style)
    for m in _CODE_RE.finditer(markdown):
        spans.append((m.start(), m.end(), "code"))
    for m in _BOLD_RE.finditer(markdown):
        # Skip if overlaps a code span
        if any(s <= m.start() < e for s, e, _ in spans):
            continue
        spans.append((m.start(), m.end(), "bold"))
    for m in _ITALIC_RE.finditer(markdown):
        if any(s <= m.start() < e for s, e, _ in spans):
            continue
        spans.append((m.start(), m.end(), "italic"))

    spans.sort(key=lambda t: t[0])

    runs: list[tuple[str, set[str]]] = []
    cursor = 0
    for start, end, style in spans:
        if start > cursor:
            runs.append((markdown[cursor:start], set()))
        # Strip the markup characters
        if style == "code":
            text = markdown[start + 1 : end - 1]
        elif style == "bold":
            text = markdown[start + 2 : end - 2]
        else:  # italic
            text = markdown[start + 1 : end - 1]
        runs.append((text, {style}))
        cursor = end
    if cursor < len(markdown):
        runs.append((markdown[cursor:], set()))
    return runs
```

Add to `Document` class in `src/andamentum/scribe/api.py`:

```python
    def add_reference(
        self,
        *,
        cite_key: str,
        bibtex: Optional[str] = None,
        metadata: Optional[dict] = None,
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

    def references(self) -> list["Reference"]:
        """Return all references attached to this document."""
        from .models import Reference

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
        from .parser import extract_citation_keys

        seen: list[str] = []
        for blk in self.query(type="paragraph"):
            for key in extract_citation_keys(blk.content):
                if key not in seen:
                    seen.append(key)
        return seen
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/andamentum/scribe/tests/test_parser.py src/andamentum/scribe/tests/test_references.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/scribe/api.py src/andamentum/scribe/parser.py \
        src/andamentum/scribe/tests/test_parser.py \
        src/andamentum/scribe/tests/test_references.py
git commit -m "feat(scribe): references, citation extraction, inline runs, unresolved markers"
```

---

## Task 10: Validation

**Files:**
- Create: `src/andamentum/scribe/validate.py`
- Modify: `src/andamentum/scribe/api.py` (add `validate` method)
- Create: `src/andamentum/scribe/tests/test_validate.py`

- [ ] **Step 1: Write the failing test**

```python
# src/andamentum/scribe/tests/test_validate.py
"""Document.validate() tests."""

from andamentum.scribe.api import Document, Figure, Paragraph


def test_validate_clean_document_returns_no_issues(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    fig = tmp_path / "f.png"
    fig.write_bytes(b"")
    doc = Document.create(title="P", database="t")
    doc.add_reference(cite_key="smith2023")
    doc.append(Paragraph("As shown [@smith2023]."))
    doc.append(Figure(path=str(fig), caption="C", label="fig:c"))

    issues = doc.validate()
    assert issues == []


def test_validate_flags_missing_citation_key(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Paragraph("Cited [@unknown2024]."))
    issues = doc.validate()
    assert any(
        i.severity == "error" and "unknown2024" in i.message for i in issues
    )


def test_validate_flags_missing_figure_file(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Figure(path="/no/such/file.png", caption="C", label="fig:x"))
    issues = doc.validate()
    assert any(
        i.severity == "error" and "fig:x" in i.location for i in issues
    )


def test_validate_warns_on_unused_reference(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.add_reference(cite_key="smith2023")
    doc.append(Paragraph("No citations here."))
    issues = doc.validate()
    assert any(
        i.severity == "warning" and "smith2023" in i.message for i in issues
    )


def test_validate_reports_unresolved_markers(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Paragraph("Foundational work [verify] established it."))
    doc.append(Paragraph("Some claim [citation needed]."))

    issues = doc.validate()
    msgs = [i.message for i in issues if i.severity == "info"]
    assert any("verify" in m for m in msgs)
    assert any("citation needed" in m for m in msgs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/andamentum/scribe/tests/test_validate.py -v`
Expected: FAIL — `AttributeError: 'Document' object has no attribute 'validate'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/andamentum/scribe/validate.py
"""Structural validators for scribe documents.

These are deterministic, no-LLM checks: missing citation keys, missing
figure files, unused references, unresolved [verify] / [citation needed]
markers. Surface as `ValidationIssue` records with severity in
{error, warning, info}. No silent failures — callers should display
every issue.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .models import ValidationIssue
from .parser import find_unresolved_markers

if TYPE_CHECKING:  # pragma: no cover
    from .api import Document


def validate_document(doc: "Document") -> list[ValidationIssue]:
    """Run all structural validators against `doc`."""
    issues: list[ValidationIssue] = []
    cite_keys_used = set(doc.citations())
    cite_keys_defined = {r.cite_key for r in doc.references()}

    # Missing citation keys: used but not defined.
    for key in cite_keys_used - cite_keys_defined:
        issues.append(
            ValidationIssue(
                severity="error",
                message=f"Citation [@{key}] referenced but no matching reference defined.",
                location=key,
            )
        )

    # Unused references: defined but not used.
    for key in cite_keys_defined - cite_keys_used:
        issues.append(
            ValidationIssue(
                severity="warning",
                message=f"Reference {key!r} defined but never cited.",
                location=key,
            )
        )

    # Missing figure files.
    for blk in doc.query(type="figure"):
        path = blk.metadata.get("path")
        if path and not Path(path).exists():
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Figure file not found: {path}",
                    location=blk.metadata.get("label", blk.id),
                )
            )

    # Unresolved citation markers in paragraphs.
    for blk in doc.query(type="paragraph"):
        for marker in find_unresolved_markers(blk.content):
            issues.append(
                ValidationIssue(
                    severity="info",
                    message=f"Unresolved citation marker [{marker}] in block.",
                    location=blk.id,
                )
            )

    return issues
```

Add to `Document` class in `src/andamentum/scribe/api.py`:

```python
    def validate(self) -> list["ValidationIssue"]:
        """Run structural validators. See validate.validate_document."""
        from .validate import validate_document

        return validate_document(self)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/andamentum/scribe/tests/test_validate.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/scribe/validate.py src/andamentum/scribe/api.py \
        src/andamentum/scribe/tests/test_validate.py
git commit -m "feat(scribe): structural validation (citations, figures, refs, markers)"
```

---

## Task 11: Render to typeset atoms (HTML/PDF path)

**Files:**
- Create: `src/andamentum/scribe/render_typeset.py`
- Create: `src/andamentum/scribe/tests/test_render_typeset.py`

- [ ] **Step 1: Write the failing test**

```python
# src/andamentum/scribe/tests/test_render_typeset.py
"""Tests for block → typeset atom conversion."""

from andamentum.scribe.api import Document, Figure, Heading, Paragraph, Table
from andamentum.scribe.render_typeset import to_typeset_atoms


def test_heading_becomes_heading_atom(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="T", database="t")
    doc.append(Heading("Intro", level=2))

    atoms = to_typeset_atoms(doc)
    assert len(atoms) == 1
    assert atoms[0]["kind"] == "heading"
    assert atoms[0]["content"] == "Intro"
    assert atoms[0]["level"] == 2


def test_paragraph_becomes_prose_atom(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="T", database="t")
    doc.append(Paragraph("Hello *world*."))

    atoms = to_typeset_atoms(doc)
    assert atoms[0]["kind"] == "prose"
    assert atoms[0]["content"] == "Hello *world*."


def test_figure_becomes_card_atom_with_caption(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="T", database="t")
    doc.append(Figure(path="x.png", caption="An overview.", label="fig:x"))

    atoms = to_typeset_atoms(doc)
    assert atoms[0]["kind"] == "card"
    assert "An overview." in atoms[0]["content"]
    assert "x.png" in atoms[0]["content"]


def test_table_becomes_prose_atom_with_markdown_table(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="T", database="t")
    doc.append(
        Table(
            rows=[["Col A", "Col B"], ["1", "2"]],
            header_row=True,
            caption="demo",
            label="tab:demo",
        )
    )

    atoms = to_typeset_atoms(doc)
    assert atoms[0]["kind"] == "prose"
    # Markdown table syntax — pipe-delimited
    assert "| Col A | Col B |" in atoms[0]["content"]
    assert "| 1 | 2 |" in atoms[0]["content"]


def test_atoms_validate_against_typeset_validator(monkeypatch, tmp_path):
    from andamentum.typeset.atoms import validate_document

    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="T", database="t")
    doc.append(Heading("H", level=1))
    doc.append(Paragraph("P"))

    atoms = to_typeset_atoms(doc)
    validated = validate_document(atoms)  # raises if invalid
    assert len(validated) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/andamentum/scribe/tests/test_render_typeset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'andamentum.scribe.render_typeset'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/andamentum/scribe/render_typeset.py
"""Map scribe blocks to typeset atom dicts.

Typeset is the package's display layer (HTML+PDF). Scribe owns the
authoring schema; this module is the thin adapter that lets typeset
render scribe documents without typeset learning anything new.

For v1 we map:
  paragraph -> prose
  heading   -> heading
  figure    -> card (with embedded <img> + caption)
  table     -> prose with a markdown table body (typeset already loads
               python-markdown's tables extension)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .api import Document


def _table_to_markdown(rows: list[list[str]], header_row: bool) -> str:
    if not rows:
        return ""
    if header_row:
        head, *body = rows
    else:
        head = [""] * len(rows[0])
        body = rows
    out = ["| " + " | ".join(head) + " |"]
    out.append("| " + " | ".join(["---"] * len(head)) + " |")
    for r in body:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def to_typeset_atoms(doc: "Document") -> list[dict]:
    """Convert the document's blocks to typeset atom dicts."""
    atoms: list[dict] = []
    for blk in doc.query():
        if blk.type == "paragraph":
            atoms.append({"kind": "prose", "content": blk.content})
        elif blk.type == "heading":
            atoms.append(
                {
                    "kind": "heading",
                    "content": blk.content,
                    "level": int(blk.metadata.get("level", 1)),
                }
            )
        elif blk.type == "figure":
            path = blk.metadata.get("path", "")
            caption = blk.metadata.get("caption", "")
            label = blk.metadata.get("label", "")
            body = (
                f'<img src="{path}" alt="{label}" />\n\n'
                f"**{caption}**" if caption else f'<img src="{path}" />'
            )
            atoms.append({"kind": "card", "content": body})
        elif blk.type == "table":
            md_table = _table_to_markdown(
                blk.metadata.get("rows", []),
                bool(blk.metadata.get("header_row", True)),
            )
            caption = blk.metadata.get("caption", "")
            content = md_table if not caption else f"{md_table}\n\n*{caption}*"
            atoms.append({"kind": "prose", "content": content})
        else:  # pragma: no cover — schema enforces the type literal
            raise ValueError(f"Unknown block type for typeset render: {blk.type!r}")
    return atoms
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/andamentum/scribe/tests/test_render_typeset.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/scribe/render_typeset.py \
        src/andamentum/scribe/tests/test_render_typeset.py
git commit -m "feat(scribe): block→typeset adapter (paragraph, heading, figure, table)"
```

---

## Task 12: Render to .docx (with inline runs and tables)

**Files:**
- Create: `src/andamentum/scribe/render_docx.py`
- Modify: `src/andamentum/scribe/api.py` (add `render` method)
- Create: `src/andamentum/scribe/tests/test_render_docx.py`

- [ ] **Step 1: Write the failing test**

```python
# src/andamentum/scribe/tests/test_render_docx.py
"""Tests for python-docx rendering."""

import pytest
from docx import Document as DocxDocument

from andamentum.scribe.api import Document, Figure, Heading, Paragraph, Table


def test_render_writes_docx_file(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Heading("Intro", level=1))
    doc.append(Paragraph("Body text."))
    out = tmp_path / "out.docx"

    doc.render(str(out), format="docx")

    assert out.exists()
    docx = DocxDocument(str(out))
    paragraph_texts = [p.text for p in docx.paragraphs]
    assert "Intro" in paragraph_texts
    assert "Body text." in paragraph_texts


def test_render_heading_uses_heading_style(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Heading("Methods", level=2))
    out = tmp_path / "out.docx"
    doc.render(str(out), format="docx")

    docx = DocxDocument(str(out))
    headings = [p for p in docx.paragraphs if p.style.name.startswith("Heading")]
    assert any("Methods" in p.text for p in headings)
    assert any(p.style.name == "Heading 2" for p in headings)


def test_render_inline_bold_emits_bold_run(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Paragraph("normal **bold** text"))
    out = tmp_path / "out.docx"
    doc.render(str(out), format="docx")

    docx = DocxDocument(str(out))
    para = next(p for p in docx.paragraphs if "bold" in p.text)
    bold_runs = [r for r in para.runs if r.bold]
    assert any(r.text == "bold" for r in bold_runs)


def test_render_inline_italic_emits_italic_run(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Paragraph("normal *em* text"))
    out = tmp_path / "out.docx"
    doc.render(str(out), format="docx")

    docx = DocxDocument(str(out))
    para = next(p for p in docx.paragraphs if "em" in p.text)
    italic_runs = [r for r in para.runs if r.italic]
    assert any(r.text == "em" for r in italic_runs)


def test_render_figure_inserts_image_when_file_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    img = tmp_path / "f.png"
    img.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    doc = Document.create(title="P", database="t")
    doc.append(Figure(path=str(img), caption="cap", label="fig:c"))
    out = tmp_path / "out.docx"
    doc.render(str(out), format="docx")

    docx = DocxDocument(str(out))
    assert any("cap" in p.text for p in docx.paragraphs)


def test_render_figure_with_explicit_width(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    img = tmp_path / "f.png"
    img.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    doc = Document.create(title="P", database="t")
    doc.append(Figure(path=str(img), caption="c", label="fig:x", width_in=4.5))
    out = tmp_path / "out.docx"
    doc.render(str(out), format="docx")  # must not raise


def test_render_table_emits_word_table(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(
        Table(
            rows=[["Col A", "Col B"], ["1", "2"], ["3", "4"]],
            header_row=True,
            caption="demo",
            label="tab:demo",
        )
    )
    out = tmp_path / "out.docx"
    doc.render(str(out), format="docx")

    docx = DocxDocument(str(out))
    assert len(docx.tables) == 1
    table = docx.tables[0]
    assert table.rows[0].cells[0].text == "Col A"
    assert table.rows[1].cells[1].text == "2"
    # Caption appears nearby
    assert any("demo" in p.text for p in docx.paragraphs)


def test_render_unknown_format_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    with pytest.raises(ValueError, match="Unsupported format"):
        doc.render(str(tmp_path / "x"), format="pdf")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/andamentum/scribe/tests/test_render_docx.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'andamentum.scribe.render_docx'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/andamentum/scribe/render_docx.py
"""Render scribe blocks to a python-docx Document.

The .docx is a derived artifact: this is a one-way render. We never
read .docx back into scribe. Templates are honoured if provided
via Document.template; otherwise python-docx's default styles are used.

Inline markdown formatting (**bold**, *italic*, `code`) inside paragraph
content is converted to styled runs via parser.inline_runs(). Anything
fancier (links, images-in-prose) falls back to plain text.

For figures: if the image file exists on disk we embed it; if not we
emit a placeholder paragraph (validate() will already have flagged the
missing file separately). Width honours `metadata.width_in` if set,
defaulting to 5.5 inches.

For tables: emitted as a real Word table with bold header row when
`metadata.header_row` is true. The caption is added as a styled
paragraph immediately after the table.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from docx import Document as DocxDocument
from docx.shared import Inches

from .parser import inline_runs

if TYPE_CHECKING:  # pragma: no cover
    from .api import Document
    from docx.document import Document as _DocxDoc


def _emit_paragraph(out: "_DocxDoc", content: str) -> None:
    para = out.add_paragraph()
    for text, styles in inline_runs(content):
        run = para.add_run(text)
        if "bold" in styles:
            run.bold = True
        if "italic" in styles:
            run.italic = True
        if "code" in styles:
            run.font.name = "Courier New"


def _emit_table(out: "_DocxDoc", metadata: dict) -> None:
    rows = metadata.get("rows", [])
    if not rows:
        return
    table = out.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    header_row = bool(metadata.get("header_row", True))
    for r_idx, row in enumerate(rows):
        for c_idx, cell_text in enumerate(row):
            cell = table.rows[r_idx].cells[c_idx]
            cell.text = cell_text
            if header_row and r_idx == 0:
                for run in cell.paragraphs[0].runs:
                    run.bold = True
    caption = metadata.get("caption", "")
    if caption:
        p = out.add_paragraph(caption)
        if "Caption" in out.styles:
            p.style = out.styles["Caption"]


def render_to_docx(doc: "Document", output_path: str) -> None:
    """Render `doc` to `output_path` as a .docx file."""
    if doc.template:
        out = DocxDocument(doc.template)
    else:
        out = DocxDocument()

    for blk in doc.query():
        if blk.type == "heading":
            level = int(blk.metadata.get("level", 1))
            out.add_heading(blk.content, level=level)
        elif blk.type == "paragraph":
            _emit_paragraph(out, blk.content)
        elif blk.type == "figure":
            path = blk.metadata.get("path", "")
            caption = blk.metadata.get("caption", "")
            width_in = blk.metadata.get("width_in") or 5.5
            if path and Path(path).exists():
                out.add_picture(path, width=Inches(float(width_in)))
            else:
                out.add_paragraph(f"[Missing figure: {path}]")
            if caption:
                p = out.add_paragraph(caption)
                if "Caption" in out.styles:
                    p.style = out.styles["Caption"]
        elif blk.type == "table":
            _emit_table(out, blk.metadata)
        else:  # pragma: no cover
            raise ValueError(f"Unknown block type for docx render: {blk.type!r}")

    out.save(output_path)
```

Add to `Document` class in `src/andamentum/scribe/api.py`:

```python
    def render(self, output_path: str, *, format: str = "docx") -> None:
        """Render this document to a file. v1 supports format='docx' only."""
        if format == "docx":
            from .render_docx import render_to_docx

            render_to_docx(self, output_path)
        else:
            raise ValueError(
                f"Unsupported format {format!r}. v1 supports: 'docx'."
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/andamentum/scribe/tests/test_render_docx.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/scribe/render_docx.py src/andamentum/scribe/api.py \
        src/andamentum/scribe/tests/test_render_docx.py
git commit -m "feat(scribe): docx renderer with inline runs, figures, tables"
```

---

## Task 13: Public API surface

**Files:**
- Modify: `src/andamentum/scribe/__init__.py` (populate `__all__`)
- Create: `src/andamentum/scribe/tests/test_public_api.py`

- [ ] **Step 1: Write the failing test**

```python
# src/andamentum/scribe/tests/test_public_api.py
"""Pin the public surface of andamentum.scribe."""

import andamentum.scribe as scribe


def test_public_all():
    expected = {
        "Document",
        "Heading",
        "Paragraph",
        "Figure",
        "Table",
        "Block",
        "Reference",
        "Revision",
        "ValidationIssue",
        "StaleRevisionError",
    }
    assert set(scribe.__all__) == expected


def test_public_imports_are_resolvable():
    for name in scribe.__all__:
        assert hasattr(scribe, name), f"scribe.{name} is missing"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/andamentum/scribe/tests/test_public_api.py -v`
Expected: FAIL — `__all__` is empty.

- [ ] **Step 3: Write minimal implementation**

Replace `src/andamentum/scribe/__init__.py`:

```python
"""andamentum.scribe — structured document drafting.

Block-based document authoring. Markdown is the source of truth;
.docx is a derived artifact. See docs/superpowers/plans/2026-04-25-scribe-module.md
for design rationale.
"""

from .api import Document, Figure, Heading, Paragraph, Table
from .models import (
    Block,
    Reference,
    Revision,
    StaleRevisionError,
    ValidationIssue,
)

__version__ = "0.1.0"

__all__ = [
    "Block",
    "Document",
    "Figure",
    "Heading",
    "Paragraph",
    "Reference",
    "Revision",
    "StaleRevisionError",
    "Table",
    "ValidationIssue",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/andamentum/scribe/tests/test_public_api.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/scribe/__init__.py \
        src/andamentum/scribe/tests/test_public_api.py
git commit -m "feat(scribe): public API surface in __init__"
```

---

## Task 14: CLI (mirrors doc-draft subcommands)

**Files:**
- Create: `src/andamentum/scribe/cli.py`
- Modify: `pyproject.toml` (register `andamentum-scribe` script)
- Create: `src/andamentum/scribe/tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# src/andamentum/scribe/tests/test_cli.py
"""Smoke tests for andamentum-scribe CLI."""

import os
import subprocess
import sys


def _run(args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "andamentum.scribe.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _envwith(tmp_path) -> dict:
    env = os.environ.copy()
    env["SCRIBE_DIR"] = str(tmp_path)
    return env


def test_help_lists_subcommands():
    result = _run(["--help"])
    assert result.returncode == 0
    for sub in (
        "init",
        "list-sections",
        "read-section",
        "write-section",
        "insert-figure",
        "insert-table",
        "render",
    ):
        assert sub in result.stdout


def test_init_with_scaffold(tmp_path):
    env = _envwith(tmp_path)
    result = _run(
        ["init", "--database", "t", "--title", "P", "--scaffold", "article"],
        env=env,
    )
    assert result.returncode == 0, result.stderr
    doc_id = result.stdout.strip()
    assert len(doc_id) >= 8


def test_list_sections_after_scaffold(tmp_path):
    env = _envwith(tmp_path)
    create = _run(
        ["init", "--database", "t", "--title", "P", "--scaffold", "article"],
        env=env,
    )
    doc_id = create.stdout.strip()

    result = _run(
        ["list-sections", "--database", "t", "--id", doc_id], env=env
    )
    assert result.returncode == 0
    assert "Introduction" in result.stdout
    assert "Methods" in result.stdout


def test_write_section_replaces_content(tmp_path):
    env = _envwith(tmp_path)
    create = _run(
        ["init", "--database", "t", "--title", "P", "--scaffold", "article"],
        env=env,
    )
    doc_id = create.stdout.strip()
    content_file = tmp_path / "intro.md"
    content_file.write_text("Brand new intro paragraph.")

    result = _run(
        [
            "write-section",
            "--database", "t",
            "--id", doc_id,
            "--section", "Introduction",
            "--content-file", str(content_file),
        ],
        env=env,
    )
    assert result.returncode == 0, result.stderr

    read = _run(
        ["read-section", "--database", "t", "--id", doc_id, "--section", "Introduction"],
        env=env,
    )
    assert "Brand new intro paragraph." in read.stdout


def test_insert_table_from_csv(tmp_path):
    env = _envwith(tmp_path)
    create = _run(
        ["init", "--database", "t", "--title", "P", "--scaffold", "article"], env=env
    )
    doc_id = create.stdout.strip()
    csv = tmp_path / "data.csv"
    csv.write_text("Col A,Col B\n1,2\n3,4\n")

    result = _run(
        [
            "insert-table",
            "--database", "t",
            "--id", doc_id,
            "--section", "Results",
            "--csv", str(csv),
            "--caption", "demo",
            "--label", "tab:demo",
        ],
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_render_unknown_doc_exits_nonzero(tmp_path):
    env = _envwith(tmp_path)
    _run(["init", "--database", "t", "--title", "x"], env=env)
    result = _run(
        [
            "render",
            "--database", "t",
            "--id", "nope",
            "--output", str(tmp_path / "x.docx"),
        ],
        env=env,
    )
    assert result.returncode != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/andamentum/scribe/tests/test_cli.py -v`
Expected: FAIL — `No module named andamentum.scribe.cli`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/andamentum/scribe/cli.py
"""Command-line entry point: andamentum-scribe.

Subcommands mirror document-tools:doc-draft so users can swap mental
models 1:1:
  init           Create an empty document (optionally from a scaffold).
  list-sections  Print sections with block and word counts.
  read-section   Print the content of a named section.
  write-section  Replace a named section's body with content from a file.
  insert-figure  Append a figure block (or insert into a named section).
  insert-table   Append a table block (or insert into a named section)
                 from a CSV file.
  render         Render the document to .docx.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .api import Document, Figure, Paragraph, Table


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="andamentum-scribe")
    sub = parser.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="Create a new document")
    init.add_argument("--database", required=True)
    init.add_argument("--title", required=True)
    init.add_argument("--template", default=None)
    init.add_argument(
        "--scaffold",
        default=None,
        choices=["article", "grant"],
        help="Pre-populate canonical sections",
    )

    ls = sub.add_parser("list-sections", help="List sections with counts")
    ls.add_argument("--database", required=True)
    ls.add_argument("--id", required=True)

    rs = sub.add_parser("read-section", help="Print a section's content")
    rs.add_argument("--database", required=True)
    rs.add_argument("--id", required=True)
    rs.add_argument("--section", required=True)

    ws = sub.add_parser("write-section", help="Replace a section's body")
    ws.add_argument("--database", required=True)
    ws.add_argument("--id", required=True)
    ws.add_argument("--section", required=True)
    ws.add_argument("--content-file", required=True)
    ws.add_argument("--reason", default=None)

    ifig = sub.add_parser("insert-figure", help="Append a figure block")
    ifig.add_argument("--database", required=True)
    ifig.add_argument("--id", required=True)
    ifig.add_argument("--image", required=True)
    ifig.add_argument("--caption", required=True)
    ifig.add_argument("--label", required=True)
    ifig.add_argument("--width-in", type=float, default=None)
    ifig.add_argument(
        "--section",
        default=None,
        help="If set, append the figure as the last block of this section",
    )

    itab = sub.add_parser("insert-table", help="Append a table block from CSV")
    itab.add_argument("--database", required=True)
    itab.add_argument("--id", required=True)
    itab.add_argument("--csv", required=True, help="Path to CSV file")
    itab.add_argument("--caption", default="")
    itab.add_argument("--label", default="")
    itab.add_argument("--no-header", action="store_true")
    itab.add_argument(
        "--section",
        default=None,
        help="If set, append the table as the last block of this section",
    )

    render = sub.add_parser("render", help="Render document to .docx")
    render.add_argument("--database", required=True)
    render.add_argument("--id", required=True)
    render.add_argument("--output", required=True)

    return parser


def _append_to_section(doc: Document, section_name: str, block_spec: dict) -> str:
    """Insert a block as the last child of a named section.

    Implementation: append at the end of the section by computing the
    section's last position and shifting later blocks down.
    """
    from .api import open_db, _new_id, _now_iso  # local imports to avoid cycle

    section_blocks = doc.section(section_name)
    last = section_blocks[-1]
    insert_pos = last.position + 1
    bid = _new_id()
    now = _now_iso()
    import json as _json

    with open_db(doc.database) as conn:
        conn.execute(
            "UPDATE scribe_blocks "
            "SET position = position + 1 "
            "WHERE doc_id = ? AND position >= ?",
            (doc.id, insert_pos),
        )
        conn.execute(
            "INSERT INTO scribe_blocks "
            "(id, doc_id, type, content, position, parent_id, metadata, "
            " revision, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?, 1, ?, ?)",
            (
                bid,
                doc.id,
                block_spec["type"],
                block_spec.get("content", ""),
                insert_pos,
                _json.dumps(block_spec.get("metadata", {})),
                now,
                now,
            ),
        )
        conn.commit()
    return bid


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "init":
        doc = Document.create(
            title=args.title,
            database=args.database,
            template=args.template,
            scaffold=args.scaffold,
        )
        print(doc.id)
        return 0

    if args.cmd == "list-sections":
        doc = Document.open(args.id, database=args.database)
        for s in doc.list_sections():
            print(
                f"{s['name']:30s}  blocks={s['block_count']:3d}  words={s['word_count']:5d}"
            )
        return 0

    if args.cmd == "read-section":
        doc = Document.open(args.id, database=args.database)
        for blk in doc.section(args.section):
            if blk.type == "heading":
                level = int(blk.metadata.get("level", 1))
                print(f"{'#' * level} {blk.content}\n")
            elif blk.type == "paragraph":
                print(f"{blk.content}\n")
            elif blk.type == "figure":
                m = blk.metadata
                print(f"![{m.get('caption', '')}]({m.get('path', '')}) " f"{{#{m.get('label', '')}}}\n")
            elif blk.type == "table":
                rows = blk.metadata.get("rows", [])
                for row in rows:
                    print(" | ".join(row))
                print()
        return 0

    if args.cmd == "write-section":
        doc = Document.open(args.id, database=args.database)
        content = Path(args.content_file).read_text()
        doc.replace_section(args.section, content, reason=args.reason)
        return 0

    if args.cmd == "insert-figure":
        doc = Document.open(args.id, database=args.database)
        spec = Figure(
            path=args.image,
            caption=args.caption,
            label=args.label,
            width_in=args.width_in,
        )
        if args.section:
            bid = _append_to_section(doc, args.section, spec)
        else:
            bid = doc.append(spec)
        print(bid)
        return 0

    if args.cmd == "insert-table":
        doc = Document.open(args.id, database=args.database)
        with open(args.csv, newline="") as f:
            rows = [row for row in csv.reader(f)]
        spec = Table(
            rows=rows,
            header_row=not args.no_header,
            caption=args.caption,
            label=args.label,
        )
        if args.section:
            bid = _append_to_section(doc, args.section, spec)
        else:
            bid = doc.append(spec)
        print(bid)
        return 0

    if args.cmd == "render":
        doc = Document.open(args.id, database=args.database)
        doc.render(args.output, format="docx")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
```

Modify `pyproject.toml` — add to `[project.scripts]`:

```toml
[project.scripts]
andamentum-epistemic = "andamentum.epistemic.cli:main"
andamentum-research  = "andamentum.deep_research.cli:main"
andamentum-whetstone = "andamentum.whetstone.cli:main"
andamentum-scribe    = "andamentum.scribe.cli:main"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/andamentum/scribe/tests/test_cli.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/scribe/cli.py pyproject.toml \
        src/andamentum/scribe/tests/test_cli.py
git commit -m "feat(scribe): CLI mirroring doc-draft subcommands"
```

---

## Task 15: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (add scribe to module list, layering, descriptions)

- [ ] **Step 1: Update the Project section bullets**

Add this bullet immediately after the `whetstone` bullet:

```markdown
- `andamentum.scribe` — structured document drafting: block-based authoring (paragraph, heading, figure, table), section abstraction, built-in `article`/`grant` scaffolds, SQLite-backed source of truth, one-way render to `.docx`. Replaces the standalone `document-tools:doc-draft` plugin.
```

- [ ] **Step 2: Update Layering rules**

Add to the `**Layering:**` bullet list (after the whetstone entry):

```markdown
- `scribe` depends only on `typeset` (for HTML/PDF rendering) and stdlib `sqlite3`. MUST NOT depend on `epistemic`, `deep_research`, `document_store`, `whetstone`, or `core`.
```

- [ ] **Step 3: Add scribe description paragraph**

After the Whetstone module paragraph, add:

```markdown
**Scribe module** (`andamentum.scribe`) — block-based document authoring. Documents live in SQLite at `~/.local/share/scribe/<name>.db` (override with `SCRIBE_DIR`). Public entry point: `Document.create(title=..., database=..., scaffold="article" | "grant" | None)`; mutate with `append`/`replace`/`replace_section`; render with `render(path, format="docx")`. Section operations (`list_sections`, `section`, `replace_section`) are derived from heading blocks — there is no separate sections table. Each block has an integer revision counter; `replace()` enforces optimistic locking and writes an audit row to `scribe_revisions`. Citations are Pandoc-flavoured `[@key]` spans extracted by regex; references live in their own table; `[verify]` and `[citation needed]` markers are recognised and reported by `validate()`. Inline markdown (bold/italic/code) renders as styled runs in `.docx`. HTML/PDF rendering goes through `typeset` (block→atom mapping in `render_typeset.py`). Scribe replaces the standalone `document-tools:doc-draft` plugin for Word file authoring; `.pptx` stays out of scope.
```

- [ ] **Step 4: Update CLI section**

Change the CLI count in `## CLIs` from "Three scripts" to "Four scripts" and add:

```bash
andamentum-scribe --help
```

Note that `andamentum-scribe` does NOT take `--model` — it has no LLM dependency in v1.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(scribe): document the new scribe module in CLAUDE.md"
```

---

## Task 16: Verification

**Files:** none (running existing checks)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest`
Expected: all scribe tests pass alongside the existing 814+ tests.

- [ ] **Step 2: Run pyright on scribe**

Run: `uv run pyright src/andamentum/scribe`
Expected: 0 errors in scribe. Pre-existing repo errors elsewhere are out of scope for this plan.

- [ ] **Step 3: Run ruff**

Run: `uv run ruff check src/andamentum/scribe && uv run ruff format src/andamentum/scribe`
Expected: clean.

- [ ] **Step 4: Smoke-test the CLI end to end**

Run:
```bash
export SCRIBE_DIR=$(mktemp -d)
DOCID=$(uv run andamentum-scribe init --database demo --title "Demo" --scaffold article)
uv run andamentum-scribe list-sections --database demo --id $DOCID

cat > /tmp/intro.md <<'EOF'
This is a brand-new introduction paragraph with **bold** and *italic*.

Second paragraph references [@smith2023].
EOF
uv run andamentum-scribe write-section --database demo --id $DOCID --section Introduction --content-file /tmp/intro.md

cat > /tmp/table.csv <<'EOF'
Method,Score
Baseline,0.72
Ours,0.91
EOF
uv run andamentum-scribe insert-table --database demo --id $DOCID --section Results --csv /tmp/table.csv --caption "Demo results" --label tab:results

uv run andamentum-scribe render --database demo --id $DOCID --output /tmp/demo.docx
ls -la /tmp/demo.docx
```
Expected: `demo.docx` exists, is non-empty, opens cleanly in Word, and contains the new intro, the table, and the standard scaffold sections.

- [ ] **Step 5: Final commit (only if fixes were needed)**

```bash
git commit -am "chore(scribe): post-verification fixes"
```

---

## What's deliberately out of scope (v2 candidates)

- **PPTX support.** Stays with `document-tools:doc-draft` for now (or a future `andamentum.deck` module). Different shape — slides aren't blocks.
- **Block types beyond paragraph/heading/figure/table:** `equation`, `code`, `callout`, `aside`. Schema accommodates them via the open `type` column; only renderers need updating.
- **`.bib` file import.** v1 takes raw BibTeX strings via `add_reference(bibtex=...)`.
- **BibTeX cite-key parsing/normalisation.** Conventions documented in citation-resolve are useful for v2 (auto-suggest cite keys, collision handling).
- **Word import (one-way: read .docx → blocks).** Worth doing for adoption but not v1.
- **Section nesting with `parent_id`.** Schema has the column; no API for it yet. v1 supports level-1 and level-2 headings via flat positions.
- **Inline citation rendering.** `.docx` render currently emits `[@key]` literally. Proper citation rendering (numbered, author-date) is a separate concern best handled by a citation processor like pandoc.
- **Concurrent writers actually exercising optimistic locking.** The locks exist; no test simulates contention beyond the stale-revision case.
- **Prose-style validators** (em-dash detection, AI-overuse word lists, hedge stacking). Worth pulling from `manuscript-tools/prose-style-guide.md` in v2 — opt-in via `validate(prose=True)`.
- **LLM-driven block generation.** That's the agent layer's job, not scribe's — scribe is the substrate.

## Self-review notes

- **Spec coverage:** every API in the original pseudocode is implemented (`open`, `create`, `append`, `query`, `replace`, `citations`, `validate`, `render`). Sections (`list_sections`, `section`, `replace_section`), scaffolds (`article`/`grant`), tables, and inline-markdown docx runs are added so scribe can replace `document-tools:doc-draft`.
- **Type consistency:** `BlockType` is the literal `"paragraph" | "heading" | "figure" | "table"` everywhere it appears. Factory helpers return plain dicts (`{"type": ..., "content": ..., "metadata": ...}`) consumed by `append`. `Document.id`, `Block.id`, `Reference.id` are all `str`.
- **No placeholders:** every code block contains the actual implementation. Every command has an expected outcome.
- **Layering:** scribe imports only from `andamentum.typeset.atoms` (Task 11 test) and stdlib `sqlite3` — no other intra-package imports.
- **doc-draft replacement check:** init/list-sections/read-section/write-section/insert-figure/insert-table/render subcommands match the doc-draft surface; PNG-only and no-track-changes constraints are inherited from python-docx and aligned with whetstone's domain split.
