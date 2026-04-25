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
            f"Stale revision for block {block_id}: expected {expected}, found {actual}"
        )
        self.block_id = block_id
        self.expected = expected
        self.actual = actual
