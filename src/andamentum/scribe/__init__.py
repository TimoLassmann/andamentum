"""andamentum.scribe — structured document drafting.

Block-based document authoring. Markdown is the source of truth;
.docx is a derived artifact. See docs/superpowers/plans/2026-04-25-scribe-module.md
for design rationale.
"""

# === Functions you can wrap as agent tools ===
# `Document` is a class — wrap its methods (`create`, `open`, `append`, `query`,
# `replace`, `replace_section`, `insert_into_section`, `add_reference`,
# `references`, `citations`, `validate`, `render`) as tools.
# `Heading`, `Paragraph`, `Figure`, `Table` are factory helpers — wrap each
# as a tool that returns a block dict, then pass the dict to `Document.append`.
from .api import Document, Figure, Heading, Paragraph, Table

# === Result/data types (returned by the above; not tools themselves) ===
from .models import (
    Block,
    Reference,
    Revision,
    StaleRevisionError,
    ValidationIssue,
)


__all__ = [
    # Functions / callables
    "Document",
    "Figure",
    "Heading",
    "Paragraph",
    "Table",
    # Data types
    "Block",
    "Reference",
    "Revision",
    "StaleRevisionError",
    "ValidationIssue",
]
