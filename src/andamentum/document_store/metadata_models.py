"""Pydantic metadata models for documents and chunks.

Two levels of metadata, designed around real query patterns:

Document-level: where it came from, who/what it mentions.
Chunk-level: topic tags, who's mentioned, and two boolean flags (decision, action item)
  that map to the two structured queries users actually run.

Every field has a default. Models are valid with zero LLM extraction.
Deterministic fields are filled by the ingestion pipeline; LLM-extracted
fields are filled optionally via extraction.py.

The store imposes no document-type taxonomy — consumers classify documents
with their own fields in the schema-less metadata dict.

Filterable fields (closed-set, used by query planner):
  source, created_at (date), has_decision (bool), has_action_item (bool)

Non-filterable fields (handled by semantic search):
  projects, people, topics, methods — open-ended, LLM can't know valid values
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


# --- Document-level metadata ---


class DocumentMetadataFields(BaseModel):
    """Structured metadata for a document.

    Deterministic fields (filled by ingestion pipeline):
        source, source_file, created_at

    LLM-extracted:
        title, projects, people
    """

    # Deterministic
    source: str = Field(
        default="manual",
        description="Where content came from: manual, slack, claude_code, zotero, voice",
    )
    source_file: str | None = Field(
        default=None,
        description="Original filename or URL if applicable",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When captured or ingested",
    )

    # LLM-extracted
    title: str = Field(
        default="",
        description="One-line summary of the document, max 10 words",
    )
    projects: list[str] = Field(
        default_factory=list,
        description='Project names this document relates to, e.g. ["GROVE", "RCCC"]. Empty if none.',
    )
    people: list[str] = Field(
        default_factory=list,
        description="All people mentioned anywhere in the document. Empty if none.",
    )


class DocumentLLMFields(BaseModel):
    """LLM-extracted fields for a document.

    Used as output_type for PydanticAI agent. Field descriptions are passed
    directly to the LLM as a tool definition schema.
    """

    title: str = Field(
        default="",
        description="One-line summary of the document, max 10 words",
    )
    projects: list[str] = Field(
        default_factory=list,
        description='Project names mentioned, e.g. ["GROVE", "RCCC"]. Empty array if none.',
    )
    people: list[str] = Field(
        default_factory=list,
        description="Names of all people mentioned. Empty array if none.",
    )


# --- Chunk-level metadata ---


class ChunkMetadataFields(BaseModel):
    """Structured metadata for a chunk.

    Deterministic fields (filled by chunking pipeline):
        parent_doc_id, section_path, chunk_index

    LLM-extracted:
        topics, people, has_decision, has_action_item
    """

    # Deterministic
    parent_doc_id: str = Field(
        default="",
        description="Foreign key to the parent document",
    )
    section_path: str = Field(
        default="",
        description='Heading hierarchy, e.g. "Methods > ODE Solver"',
    )
    chunk_index: int = Field(
        default=0,
        description="Position within the document (0-based)",
    )

    # LLM-extracted
    topics: list[str] = Field(
        default_factory=list,
        description="2-3 specific topic tags for this chunk",
    )
    people: list[str] = Field(
        default_factory=list,
        description="People mentioned in this chunk",
    )
    has_decision: bool = Field(
        default=False,
        description="Whether this chunk contains a decision or commitment",
    )
    has_action_item: bool = Field(
        default=False,
        description="Whether this chunk contains a to-do or next step",
    )


class ChunkLLMFields(BaseModel):
    """LLM-extracted fields for a chunk.

    Used as output_type for PydanticAI agent. Field descriptions are passed
    directly to the LLM as a tool definition schema.

    Binary yes/no questions (has_decision, has_action_item) are more reliable
    for local models than multi-way classification.
    """

    topics: list[str] = Field(
        default_factory=list,
        description='2-3 specific topic tags. Be specific — prefer "MAP-Elites selection" over "optimization". Empty array if content is too generic.',
    )
    people: list[str] = Field(
        default_factory=list,
        description="People mentioned in this chunk. Empty array if none.",
    )
    has_decision: bool = Field(
        default=False,
        description="Does this chunk contain a decision, commitment, or resolution? true/false.",
    )
    has_action_item: bool = Field(
        default=False,
        description="Does this chunk contain a to-do, next step, or action item? true/false.",
    )
