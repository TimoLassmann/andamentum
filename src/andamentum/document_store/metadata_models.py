"""Pydantic models for the fields an LLM extracts from a document.

Used by ``extraction.extract_document_metadata`` and the ``extract`` enricher.
Every field has a default, so the models are valid with no extraction at all.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# --- Document-level metadata ---


class DocumentMetadataFields(BaseModel):
    """LLM-extracted metadata for a document: title, projects, people."""

    # LLM-extracted
    title: str = Field(
        default="",
        description="One-line summary of the document, max 10 words",
    )
    projects: list[str] = Field(
        default_factory=list,
        description='Project names this document relates to, e.g. ["Apollo", "Zephyr"]. Empty if none.',
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
        description='Project names mentioned, e.g. ["Apollo", "Zephyr"]. Empty array if none.',
    )
    people: list[str] = Field(
        default_factory=list,
        description="Names of all people mentioned. Empty array if none.",
    )
