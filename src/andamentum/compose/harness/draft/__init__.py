"""``write``: draft into a new Word file, or into an existing one, with the compose workers.

``draft_document`` is the library entry; the ``write`` verb wires its flags.
Notes: ``CLAUDE.md`` in this folder.
"""

from __future__ import annotations

from .disclosure import HEADING as AI_STATEMENT_HEADING
from .disclosure import ai_use_statement
from .run import (
    DEFAULT_MAX_ROUNDS,
    DraftResult,
    DraftUsageError,
    RoundNote,
    draft_document,
)
from .search import literature_search
from .works import CitableWork, LiteratureFinding, LiteratureSearch

__all__ = [
    "AI_STATEMENT_HEADING",
    "DEFAULT_MAX_ROUNDS",
    "CitableWork",
    "DraftResult",
    "DraftUsageError",
    "LiteratureFinding",
    "LiteratureSearch",
    "RoundNote",
    "ai_use_statement",
    "draft_document",
    "literature_search",
]
