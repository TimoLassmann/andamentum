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
