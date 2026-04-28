#!/usr/bin/env python3
"""Patch data models for the whetstone DOCX track-changes machinery.

This module is the only surviving slice of the old whetstone v1 ``models``
file. It contains the two types the preserved ``whetstone.docx`` subpackage
operates on:

* ``DocumentPatch`` — a single edit/comment/analysis instruction that the
  finalisation pipeline applies as a Word tracked change or comment. The
  current whetstone ``renderers/docx.py`` adapts each ``Edit`` and
  ``Finding`` into one of these before invoking
  ``finalize_reviewed_document``.
* ``PatchApplicationResult`` — the return type from
  ``finalize_reviewed_document`` (counts of applied / failed patches).

Everything else from the v1 models module (``ChecklistItem``,
``BaselineCheck``) was v1-only and has been removed alongside the rest of
the v1 surface.
"""

import uuid
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator


class DocumentPatch(BaseModel):
    """
    Represents a single edit or comment to be applied to a document.

    For text_edit patches: Requires text_pattern, new_text, and explanation.
    For comment patches: Requires text_pattern, comment_text, and explanation.
    """

    patch_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    patch_type: Literal["text_edit", "comment", "document_analysis"] = Field(
        ...,
        description="REQUIRED: Must be 'text_edit' (to replace text), 'comment' (to add suggestion), or 'document_analysis' (for high-level document analysis with detailed structured evaluation)",
    )

    # Location identification - REQUIRED for text_edit and comment, OPTIONAL for document_analysis
    text_pattern: Optional[str] = Field(
        default=None,
        description="REQUIRED for text_edit/comment: Exact text to find and modify. NOT NEEDED for document_analysis patches.",
    )

    # Location identification - OPTIONAL
    paragraph_index: Optional[int] = Field(
        default=None, description="OPTIONAL: Target paragraph index if known"
    )
    original_text: Optional[str] = Field(
        default=None, description="OPTIONAL: Original text for validation"
    )

    # Content - CONDITIONAL
    new_text: Optional[str] = Field(
        default=None,
        description="REQUIRED for text_edit patches: The improved replacement text. Must NOT be None or empty for text_edit patches.",
    )
    comment_text: Optional[str] = Field(
        default=None,
        description="REQUIRED for comment patches: The suggestion or note text. Must NOT be None or empty for comment patches.",
    )
    analysis_text: Optional[str] = Field(
        default=None,
        description="REQUIRED for document_analysis patches: Detailed structured analysis in markdown format with comprehensive evaluation, scores, recommendations, and specific insights",
    )

    # Position tracking - SYSTEM USE ONLY (not for LLM)
    found_at: Optional[int] = Field(
        default=None,
        description="SYSTEM-POPULATED: Do NOT set this field. The system will automatically populate it after finding the text_pattern.",
    )
    original_start: Optional[int] = Field(
        default=None,
        description="SYSTEM-POPULATED: Do NOT set this field. Automatically calculated by the system.",
    )
    original_end: Optional[int] = Field(
        default=None,
        description="SYSTEM-POPULATED: Do NOT set this field. Automatically calculated by the system.",
    )

    # Metadata - REQUIRED
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="REQUIRED: Confidence score between 0.0 and 1.0. Use 0.9+ for obvious errors, 0.7+ for good improvements, 0.5+ for suggestions.",
    )
    explanation: str = Field(
        ...,
        description="REQUIRED: Clear, specific explanation of why this change improves the text. Must NOT be empty.",
    )

    @model_validator(mode="after")
    def validate_patch_fields(self):
        """Validate that required fields are present based on patch type."""
        errors = []

        # Check required fields based on patch type
        if self.patch_type == "text_edit":
            if not self.text_pattern or self.text_pattern.strip() == "":
                errors.append(
                    "text_pattern is required and cannot be empty for text_edit patches"
                )
            if self.new_text is None or self.new_text.strip() == "":
                errors.append(
                    "new_text is required and cannot be empty for text_edit patches"
                )
            if self.new_text and self.new_text.strip().lower() in [
                "none",
                "null",
                "n/a",
            ]:
                errors.append(
                    f"new_text has invalid placeholder value: '{self.new_text}'"
                )

        elif self.patch_type == "comment":
            if not self.text_pattern or self.text_pattern.strip() == "":
                errors.append(
                    "text_pattern is required and cannot be empty for comment patches"
                )
            if self.comment_text is None or self.comment_text.strip() == "":
                errors.append(
                    "comment_text is required and cannot be empty for comment patches"
                )
            if self.comment_text and self.comment_text.strip().lower() in [
                "none",
                "null",
                "n/a",
            ]:
                errors.append(
                    f"comment_text has invalid placeholder value: '{self.comment_text}'"
                )

        elif self.patch_type == "document_analysis":
            if self.analysis_text is None or self.analysis_text.strip() == "":
                errors.append(
                    "analysis_text is required and cannot be empty for document_analysis patches - provide detailed structured analysis"
                )
            if self.analysis_text and self.analysis_text.strip().lower() in [
                "none",
                "null",
                "n/a",
            ]:
                errors.append(
                    f"analysis_text has invalid placeholder value: '{self.analysis_text}' - provide comprehensive analysis instead"
                )

        # Check common required fields
        if not self.explanation or self.explanation.strip() == "":
            errors.append("explanation is required and cannot be empty")
        if self.explanation and self.explanation.strip().lower() in [
            "none",
            "null",
            "n/a",
        ]:
            errors.append(
                f"explanation has invalid placeholder value: '{self.explanation}'"
            )

        if not (0.0 <= self.confidence <= 1.0):
            errors.append(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

        if errors:
            raise ValueError(f"DocumentPatch validation failed: {'; '.join(errors)}")

        return self

    def __str__(self) -> str:
        """Human-readable representation of the patch."""
        if self.patch_type == "text_edit":
            return f"Edit[{self.patch_id}]: {self.explanation}"
        elif self.patch_type == "comment":
            return f"Comment[{self.patch_id}]: {self.explanation}"
        else:  # document_analysis
            return f"Analysis[{self.patch_id}]: {self.explanation}"


class PatchApplicationResult(BaseModel):
    """
    Result of applying patches to a document.

    Tracks success/failure and provides diagnostics.
    """

    total_patches: int = Field(..., description="Total number of patches attempted")
    applied_patches: int = Field(
        ..., description="Number of successfully applied patches"
    )
    failed_patches: List[DocumentPatch] = Field(
        default_factory=list, description="Patches that failed to apply"
    )
    processing_time: float = Field(
        ..., description="Time taken to apply patches in seconds"
    )

    # Detailed results
    applied_edits: int = Field(default=0, description="Number of text edits applied")
    applied_comments: int = Field(default=0, description="Number of comments applied")
    location_failures: int = Field(
        default=0, description="Patches that failed due to location issues"
    )
    validation_failures: int = Field(
        default=0, description="Patches that failed validation"
    )

    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        if self.total_patches == 0:
            return 100.0
        return (self.applied_patches / self.total_patches) * 100.0

    def __str__(self) -> str:
        """Human-readable summary of application results."""
        return f"Applied {self.applied_patches}/{self.total_patches} patches ({self.success_rate:.1f}% success rate)"
