"""andamentum.serve: a local HTTP server over the library, review and research.

``andamentum-serve --out DIR`` listens on 127.0.0.1 only. Routes:
``POST /library/search``, ``POST /library/read`` (the document store, read only),
``POST /procedure/research``, ``POST /procedure/review_document``,
``POST /procedure/accept_edits`` and ``GET /health``. Each call is one
synchronous request with a JSON body; every error answer is
``{"error": "..."}``. The model is named in each procedure's body; calls on a
local Ollama model run one at a time.

Needs the ``serve`` extra (FastAPI and uvicorn). The names below are imported
on first use, so the command can report a missing extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import ServeError, create_app

__all__ = ["create_app", "ServeError"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import app

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
