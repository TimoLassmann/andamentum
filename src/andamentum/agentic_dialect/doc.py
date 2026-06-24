"""Access and normalize the prose canon (``DIALECT.md``).

The markdown is authoritative for prose (rationale, examples) and for the
copy-paste skeleton; the structured laws are authoritative for the enforceable
statements. ``normalize`` lets the drift test match a statement stored on one
logical line against the same text wrapped across several lines in the markdown.
"""

from __future__ import annotations

import re
from pathlib import Path

_DOC = Path(__file__).with_name("DIALECT.md")


def doc_path() -> Path:
    """Absolute path to the prose canon shipped with the package."""
    return _DOC


def read_doc() -> str:
    """The prose canon as text."""
    return _DOC.read_text(encoding="utf-8")


def normalize(text: str) -> str:
    """Strip markdown emphasis markers and collapse whitespace.

    Removes ``*`` and backtick markers and collapses every whitespace run to a
    single space, so prose comparisons survive line-wrapping and emphasis.
    """
    text = text.replace("*", "").replace("`", "")
    return re.sub(r"\s+", " ", text).strip()


def skeleton() -> str:
    """The runnable copy-paste skeleton, lifted verbatim from the canon.

    The skeleton has no structured consumer apart from the prose, so the
    markdown is its single source of truth — this returns the fenced ``python``
    block under the 'Copy-paste skeleton' heading.
    """
    doc = read_doc()
    idx = doc.index("## Copy-paste skeleton")
    fence = doc.index("```python", idx) + len("```python")
    end = doc.index("```", fence)
    return doc[fence:end].strip("\n")
