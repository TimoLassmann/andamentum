"""Pydantic models and errors for bundles."""

from __future__ import annotations

import hashlib
from typing import Literal, Optional

from pydantic import BaseModel

Severity = Literal["error", "warning", "info"]


class BundleError(Exception):
    """A bundle operation failed (missing section, bad layout, bad argument)."""


class StaleRevisionError(BundleError):
    """Raised when a write names a revision that is no longer current."""

    def __init__(self, section_id: str, expected: int, actual: int):
        super().__init__(
            f"Stale revision for section {section_id}: expected {expected}, found {actual}"
        )
        self.section_id = section_id
        self.expected = expected
        self.actual = actual


class Section(BaseModel):
    """One section of a bundle: frontmatter fields plus the markdown body."""

    id: str
    title: str
    order: int
    revision: int
    status: str
    body: str

    @property
    def content_hash(self) -> str:
        """sha256 of the body. Computed on read, never stored."""
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()


class Reference(BaseModel):
    """One entry of ``references.yaml``."""

    key: str
    text: str
    url: Optional[str] = None


class ValidationIssue(BaseModel):
    """One structural issue surfaced by ``Bundle.validate()``."""

    severity: Severity
    message: str
    location: str  # section id, citation key, or image path
