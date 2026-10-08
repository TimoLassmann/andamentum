"""andamentum.compose.bundle — a document as a folder bundle.

A bundle is a plain directory: ``bundle.yaml``, one markdown file per section,
``figures/``, ``references.yaml`` and ``out/``. Markdown is the source of
truth; .docx / .md / .html / .txt are derived. No LLM, no database.
"""

# === Functions you can wrap as agent tools ===
# `Bundle` is a class — wrap its methods (`create`, `open`, `sections`,
# `section`, `section_by_title`, `add_section`, `write_section`,
# `rename_section`, `remove_section`, `reorder`, `add_figure`, `references`,
# `add_reference`, `validate`, `render`) as tools.
from .bundle import Bundle

# === Result/data types (returned by the above; not tools themselves) ===
from .models import (
    Reference,
    BundleError,
    ComposeError,
    Section,
    StaleRevisionError,
    ValidationIssue,
)

__all__ = [
    # Functions / callables
    "Bundle",
    # Data types
    "Reference",
    "BundleError",
    "ComposeError",
    "Section",
    "StaleRevisionError",
    "ValidationIssue",
]
