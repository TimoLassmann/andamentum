"""word: work on an existing Word file.

Check it, answer reviewers' comments, bind and refresh the numbers that come
from result files, find and insert verified citations, or draft a task into it
as tracked changes. The input file is never edited: every job writes an edited
copy, a report and what it keeps between runs into the folder ``out``, and
nothing anywhere else.

``await check_document("paper.docx", out="checked")`` is the main call; the
command line is ``andamentum-word``.

Public names resolve lazily so that importing the package stays cheap.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from andamentum.compose.harness.bindings.bind import BindRun
    from andamentum.compose.harness.bindings.refresh import RefreshRun
    from andamentum.compose.harness.check import CheckRun
    from andamentum.compose.harness.draft.run import DraftResult
    from andamentum.compose.harness.errors import (
        WordError,
        WordInputError,
        WordUsageError,
    )
    from andamentum.compose.harness.respond.run import RespondRun

    from .api import (
        CitationInsertion,
        CitationSearch,
        FoundWork,
        UnverifiedWork,
        bind_numbers,
        check_document,
        draft_document,
        find_citation,
        insert_citation,
        refresh_numbers,
        respond_to_reviews,
    )

__all__ = [
    "check_document",
    "respond_to_reviews",
    "refresh_numbers",
    "bind_numbers",
    "draft_document",
    "find_citation",
    "insert_citation",
    "CheckRun",
    "RespondRun",
    "RefreshRun",
    "BindRun",
    "DraftResult",
    "CitationSearch",
    "FoundWork",
    "UnverifiedWork",
    "CitationInsertion",
    "WordError",
    "WordUsageError",
    "WordInputError",
]

_H = "andamentum.compose.harness"
_HOME = {
    "check_document": f"{__name__}.api",
    "respond_to_reviews": f"{__name__}.api",
    "refresh_numbers": f"{__name__}.api",
    "bind_numbers": f"{__name__}.api",
    "draft_document": f"{__name__}.api",
    "find_citation": f"{__name__}.api",
    "insert_citation": f"{__name__}.api",
    "CitationSearch": f"{__name__}.api",
    "FoundWork": f"{__name__}.api",
    "UnverifiedWork": f"{__name__}.api",
    "CitationInsertion": f"{__name__}.api",
    "CheckRun": f"{_H}.check",
    "RespondRun": f"{_H}.respond.run",
    "RefreshRun": f"{_H}.bindings.refresh",
    "BindRun": f"{_H}.bindings.bind",
    "DraftResult": f"{_H}.draft.run",
    "WordError": f"{_H}.errors",
    "WordUsageError": f"{_H}.errors",
    "WordInputError": f"{_H}.errors",
}


def __getattr__(name: str) -> Any:
    home = _HOME.get(name)
    if home is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(home), name)
