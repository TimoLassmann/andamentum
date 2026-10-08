"""compose — a task in, a finished document out.

``await write_document("<task>", out=..., roles=Roles.all(model))`` compiles a
brief, extracts evidence from the sources, plans, drafts section by section and
revises against independent checks until the goal holds twice running. The
result is a plain document folder: markdown sections, the evidence, a report
and the rendered files in ``out/``. ``continue_document`` picks a folder up
again. (The folder is a ``andamentum.compose.bundle.Bundle``, model-free.)

Working on an existing Word file is the separate ``andamentum.word`` tool.

compose never sends anything. Public names resolve lazily so that importing the
package stays cheap (whetstone and pydantic-graph load on first use).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .api import continue_document, write_document
    from .errors import ComposeError, ComposeInputError, ComposeUsageError
    from .research import ResearchSetup
    from .schemas import ComposeResult, Roles

__all__ = [
    "write_document",
    "continue_document",
    "ComposeResult",
    "Roles",
    "ResearchSetup",
    "ComposeError",
    "ComposeUsageError",
    "ComposeInputError",
]

# Not in __all__ but still importable from here: internals other modules use.
_HOME = {
    "write_document": "api",
    "continue_document": "api",
    "compile_brief": "brief",
    "ComposeResult": "schemas",
    "Brief": "schemas",
    "Roles": "schemas",
    "ResearchSetup": "research",
    "ComposeError": "errors",
    "ComposeUsageError": "errors",
    "ComposeInputError": "errors",
    "StopReason": "schemas",
    "Genre": "genres",
    "GENRES": "genres",
    "genre_for": "genres",
    "Bundle": "bundle",
    "Section": "bundle",
}


def __getattr__(name: str) -> Any:
    home = _HOME.get(name)
    if home is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{home}"), name)
