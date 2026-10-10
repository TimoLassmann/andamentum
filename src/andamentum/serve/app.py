"""The HTTP routes: a thin layer over the Python functions, with no logic of its own.

Every request body is a pydantic model that refuses an unknown field. Every
error answer is ``{"error": "<one sentence>"}``, the shape the caller passes on
unchanged. A procedure call is written to ``out/<call-id>/`` as
``request.json`` and ``result.json`` (or ``error.json``), so a result outlives
a client that stopped waiting.

No route takes a path, a store or a URL to read: a note is reviewed as the
text sent, never opened as a file.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from .queue import ModelQueue

if TYPE_CHECKING:
    from andamentum.document_store import Store

logger = logging.getLogger(__name__)

#: Headers the home service adds to a forwarded call; kept in request.json.
CALLER_HEADERS = ("x-agent", "x-run-session", "x-scope")


class ServeError(Exception):
    """The base error of ``andamentum.serve``: the server cannot start as asked."""


class Refused(Exception):
    """A request the server answers with an error status and one sentence."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchBody(_Body):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=50)


class ReadBody(_Body):
    doc_id: str = Field(min_length=1)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)


class ResearchBody(_Body):
    question: str = Field(min_length=1)
    mode: Literal["web", "scholarly", "both"]
    depth: Literal["quick", "standard", "deep"] = "standard"
    model: str = Field(min_length=1)


class ReviewBody(_Body):
    note: str = Field(min_length=1)  # markdown text, never a path
    model: str = Field(min_length=1)
    reviewer: str | None = None


class AcceptBody(_Body):
    note: str
    edits: list[dict[str, Any]]  # the accepted edits, as review_document returned them


def create_app(
    *,
    out: Path | str,
    store: Path | str | None = None,
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    searxng_url: str | None = None,
    browser: bool = False,
) -> FastAPI:
    """The andamentum HTTP server as an ASGI app; ``andamentum-serve`` runs it.

    *out* is the folder for procedure records; it is created. *store* opens the
    document store for the library routes, with *embedding_model* and
    *reranker_model* (the models its index was built with; no defaults);
    without it the library routes answer 503. *searxng_url* or *browser* lets
    research search the web. Raises :class:`ServeError` for a store given
    without both models.
    """
    if store is not None and not (embedding_model and reranker_model):
        raise ServeError("a store needs both --embedding-model and --reranker-model")
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    queue = ModelQueue()
    library_models = tuple(m for m in (embedding_model, reranker_model) if m)
    state: dict[str, Any] = {"store": None}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if store is not None:
            state["store"] = await _open_store(
                Path(store),
                embedding_model=embedding_model or "",
                reranker_model=reranker_model or "",
            )
        try:
            yield
        finally:
            if state["store"] is not None:
                await state["store"].close()

    app = FastAPI(title="andamentum", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.exception_handler(Refused)
    async def _refused(_request: Request, exc: Refused) -> JSONResponse:
        return JSONResponse({"error": exc.message}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse({"error": _describe(exc.errors())}, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = "no such route" if exc.status_code == 404 else str(exc.detail)
        return JSONResponse({"error": message}, status_code=exc.status_code)

    def _library() -> Store:
        if state["store"] is None:
            raise Refused(503, "library is not configured: start the server with --store")
        return state["store"]

    @app.get("/health")
    async def health() -> dict[str, Any]:
        from andamentum import __version__

        return {
            "version": __version__,
            "library": state["store"] is not None,
            "queue": queue.state(),
        }

    @app.post("/library/search")
    async def library_search(body: SearchBody) -> dict[str, Any]:
        library = _library()
        # The query is embedded by a local Ollama model: one at a time.
        async with queue.hold(local=True):
            hits = await library.search(body.query, limit=body.limit)
        titles: dict[str, dict[str, Any]] = {}
        for hit in hits:
            if hit.doc_id not in titles:
                titles[hit.doc_id] = await _names(library, hit.doc_id)
        return {
            "hits": [
                {
                    "doc_id": h.doc_id,
                    **titles[h.doc_id],
                    "rev": h.rev,
                    "start": h.start,
                    "end": h.end,
                    "heading_path": list(h.heading_path),
                    "text": h.text,
                    "score": h.score,
                }
                for h in hits
            ]
        }

    @app.post("/library/read")
    async def library_read(body: ReadBody) -> dict[str, Any]:
        library = _library()
        doc = await library.get(body.doc_id)
        if doc is None:
            raise Refused(404, f"no document {body.doc_id}")
        start = 0 if body.start is None else body.start
        end = len(doc.text) if body.end is None else body.end
        if not start <= end <= len(doc.text):
            raise Refused(
                422, f"the span {start}-{end} is outside a text of {len(doc.text)} characters"
            )
        return {
            "doc_id": doc.doc_id,
            "key": doc.key,
            "title": doc.metadata.get("title"),
            "rev": doc.rev,
            "length": len(doc.text),
            "start": start,
            "end": end,
            "text": doc.text[start:end],
        }

    async def _record(
        request: Request, name: str, body: BaseModel, run: Callable[[], Awaitable[Any]]
    ) -> Any:
        """Run one procedure call and keep its request and its result under *out*."""
        call_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
        folder = out_dir / call_id
        folder.mkdir()
        _write(
            folder / "request.json",
            {
                "procedure": name,
                "body": body.model_dump(mode="json"),
                "caller": {h: request.headers[h] for h in CALLER_HEADERS if h in request.headers},
            },
        )
        try:
            result = await run()
        except Refused as exc:
            _write(folder / "error.json", {"status": exc.status, "error": exc.message})
            raise
        except Exception as exc:
            logger.exception("procedure %s failed (%s)", name, call_id)
            message = f"{name} failed: {type(exc).__name__}: {exc}"
            _write(folder / "error.json", {"status": 500, "error": message})
            raise Refused(500, message) from exc
        _write(folder / "result.json", result)
        return result

    @app.post("/procedure/research")
    async def procedure_research(request: Request, body: ResearchBody) -> Any:
        if body.mode in ("web", "both") and not (searxng_url or browser):
            raise Refused(
                422,
                f"mode {body.mode} needs web search: start the server with --searxng-url or "
                "--browser, or ask with mode scholarly",
            )

        async def run() -> Any:
            async with queue.hold_for([body.model]):
                return await _research(body, searxng_url=searxng_url, browser=browser)

        return await _record(request, "research", body, run)

    @app.post("/procedure/review_document")
    async def procedure_review(request: Request, body: ReviewBody) -> Any:
        async def run() -> Any:
            async with queue.hold_for([body.model]):
                return await _review(body)

        return await _record(request, "review_document", body, run)

    @app.post("/procedure/accept_edits")
    async def procedure_accept(request: Request, body: AcceptBody) -> Any:
        async def run() -> Any:
            return _accept(body)  # no model

        return await _record(request, "accept_edits", body, run)

    app.state.queue = queue
    app.state.library_models = library_models
    return app


# --- the calls into the library, one per route; tests replace these ----------------


async def _open_store(path: Path, *, embedding_model: str, reranker_model: str) -> Store:
    from andamentum.document_store import Store

    if not path.exists():
        raise ServeError(f"no store at {path}")
    return await Store.open_with_search(
        path, embedding_model=embedding_model, reranker_model=reranker_model
    )


async def _names(store: Store, doc_id: str) -> dict[str, Any]:
    doc = await store.get(doc_id)
    if doc is None:
        return {"key": None, "title": None}
    return {"key": doc.key, "title": doc.metadata.get("title")}


async def _research(body: ResearchBody, *, searxng_url: str | None, browser: bool) -> Any:
    from andamentum.deep_research import Budget, SourceConfig, research

    sources = SourceConfig(
        mailto=os.environ.get("OPENALEX_EMAIL") or None,
        openalex_api_key=os.environ.get("OPENALEX_API_KEY") or None,
        semantic_scholar_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None,
        searxng_url=searxng_url,
        browser=browser,
    )
    result = await research(
        body.question,
        mode=body.mode,
        model=body.model,
        budget=Budget.for_depth(body.depth),
        sources=sources,
    )
    return result.model_dump(mode="json")


async def _review(body: ReviewBody) -> Any:
    from andamentum.whetstone import (
        ConfidentialityMarkerError,
        ReviewedDocument,
        review_document,
        reviewer_profile,
    )

    try:
        profile = reviewer_profile(body.reviewer) if body.reviewer else None
    except ValueError as exc:
        raise Refused(422, str(exc)) from exc
    document = ReviewedDocument.from_markdown(body.note)
    try:
        result = await review_document(document, model=body.model, reviewer=profile)
    except ConfidentialityMarkerError as exc:
        raise Refused(422, str(exc)) from exc
    finally:
        document.cleanup()
    return result.model_dump(mode="json")


def _accept(body: AcceptBody) -> Any:
    from pydantic import ValidationError

    from andamentum.whetstone import Edit, accept_edits

    try:
        edits = [Edit.model_validate(e) for e in body.edits]
    except ValidationError as exc:
        raise Refused(422, "edits: " + _describe(exc.errors())) from exc
    try:
        return accept_edits(body.note, edits).model_dump(mode="json")
    except ValueError as exc:
        raise Refused(422, str(exc)) from exc


def _describe(errors: Any) -> str:
    """Pydantic's validation errors as one sentence a caller can act on."""
    parts = []
    for error in errors:
        where = ".".join(str(p) for p in error.get("loc", ()) if p != "body")
        parts.append(f"{where}: {error.get('msg', 'invalid')}" if where else error.get("msg", "invalid"))
    return "; ".join(parts) or "the request body is not valid"


def _write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
