"""compose — a task in, a finished document out.

``await write_document("<task>", out=..., roles=Roles.all(model))`` compiles a
brief, extracts evidence from the sources, plans, drafts section by section and
revises against independent checks until the goal holds twice running. Every
artefact lives in a plain folder bundle (``andamentum.compose.bundle.Bundle``, model-free).

compose never sends anything. Public names resolve lazily so that importing the
package stays cheap (whetstone and pydantic-graph load on first use).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .api import compose_bundle, write_document
    from .brief import compile_brief
    from .bundle import Bundle, Section
    from .genres import GENRES, Genre, genre_for
    from .research import ResearchSetup
    from .schemas import Brief, ComposeResult, Roles, StopReason

__all__ = [
    "write_document",
    "compose_bundle",
    "compile_brief",
    "ComposeResult",
    "Brief",
    "Roles",
    "ResearchSetup",
    "Genre",
    "GENRES",
    "genre_for",
    "StopReason",
    "Bundle",
    "Section",
]

_HOME = {
    "write_document": "api",
    "compose_bundle": "api",
    "compile_brief": "brief",
    "ComposeResult": "schemas",
    "Brief": "schemas",
    "Roles": "schemas",
    "ResearchSetup": "research",
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
