"""Data models for Document Store.

Pydantic models for type safety and validation.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Document type classification.

    Retained for backward compatibility with existing databases and consumers.
    No longer affects indexing behavior — all documents go through the same
    storage pipeline. The value is stored in the document_tier column.

    WORKING: Default type for all documents.
    REFERENCE: Legacy type (no different treatment).
    GENERATED: Legacy type (no different treatment).
    """

    REFERENCE = "reference"
    WORKING = "working"
    GENERATED = "generated"


class DocumentMetadata(BaseModel):
    """Metadata for a document in the store."""

    doc_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    document_type: DocumentType = Field(..., description="Document classification")
    file_path: str = Field(..., description="Path to raw file")
    content_hash: str = Field(..., description="SHA-256 hash of content")
    file_format: str = Field(
        ..., description="Original file format (pdf, md, docx, etc.)"
    )
    file_size_bytes: int = Field(..., description="File size in bytes")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )
    indexed_at: Optional[datetime] = Field(None, description="Last indexing timestamp")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class Document(BaseModel):
    """Complete document with content and metadata."""

    metadata: DocumentMetadata = Field(..., description="Document metadata")
    content: str = Field(..., description="Document content (markdown)")
    raw_file_path: Optional[Path] = Field(None, description="Path to original raw file")


class UpdateResult(BaseModel):
    """Result of a document update operation."""

    success: bool = Field(..., description="Whether update succeeded")
    doc_id: str = Field(..., description="Document identifier")
    previous_hash: str = Field(..., description="Content hash before update")
    new_hash: str = Field(..., description="Content hash after update")
    reindexed: bool = Field(..., description="Whether document was reindexed")
    metadata_updated: bool = Field(False, description="Whether metadata was updated")
    message: str = Field(..., description="Human-readable result message")


class ReembedResult(BaseModel):
    """Result of a batch re-embedding operation."""

    n_embedded: int = Field(
        ..., description="Number of documents successfully embedded"
    )
    n_skipped: int = Field(
        ..., description="Number of documents that already had embeddings"
    )
    n_failed: int = Field(..., description="Number of documents that failed to embed")
    duration_seconds: float = Field(..., description="Total time taken in seconds")
