"""Bootstrap helper FastAPI app for chunker benchmark case authoring.

Run::

    uv run python -m benchmarks.chunker.app
    # → http://127.0.0.1:8765

Paste source text in the UI, pick a domain + model, click "Chunk it",
see the result with each unit highlighted in a different colour. Click
"Download truth.json" to get a draft annotation file you can edit and
save into benchmarks/chunker/cases/.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from andamentum.chunker.extractor import (
    ExecutorFn,
    extract_units,
    make_runner_executor,
)
from andamentum.core.agents import AgentRunner

STATIC_DIR = Path(__file__).parent / "static"


# ---------- Executor factory (overridable for tests) ----------------------


def _default_executor_factory(model: str) -> ExecutorFn:
    runner = AgentRunner(model=model)
    return make_runner_executor(runner)


_executor_factory: Callable[[str], ExecutorFn] = _default_executor_factory


def _set_executor_factory(factory: Callable[[str], ExecutorFn]) -> None:
    """Test hook: swap the executor factory for fakes."""
    global _executor_factory
    _executor_factory = factory


# ---------- Request / response models -------------------------------------


class ChunkRequest(BaseModel):
    text: str = Field(..., min_length=1)
    domain: str = "general"
    model: str = Field(..., min_length=1)
    window_size: int = 10_000
    lookahead: int = 4_000


class UnitOut(BaseModel):
    id: str
    title: str
    text: str
    kind: str
    source_start: int
    source_end: int
    complete: bool
    anchor_match_method: str


class GapOut(BaseModel):
    source_start: int
    source_end: int
    text: str


class ChunkResponse(BaseModel):
    units: list[UnitOut]
    gaps: list[GapOut]
    coverage: float
    gap_fraction: float
    total_chars: int
    model_calls: int


# ---------- App + endpoints -----------------------------------------------

app = FastAPI(title="andamentum-chunker explorer", version="0.1.0")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# Mount static files (for app.js, future css)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.post("/api/chunk", response_model=ChunkResponse)
async def chunk(req: ChunkRequest) -> ChunkResponse:
    executor = _executor_factory(req.model)
    try:
        result = await extract_units(
            req.text,
            primary_executor=executor,
            window_size=req.window_size,
            lookahead=req.lookahead,
            domain=req.domain,
        )
    except Exception as exc:  # ChunkingFailedError or anything else — show to UI
        raise HTTPException(
            status_code=500,
            detail=f"chunker failed: {exc}",
        )

    return ChunkResponse(
        units=[
            UnitOut(
                id=u.id,
                title=u.title,
                text=u.text,
                kind=u.kind,
                source_start=u.source_start,
                source_end=u.source_end,
                complete=u.complete,
                anchor_match_method=u.anchor_match_method,
            )
            for u in result.units
        ],
        gaps=[
            GapOut(
                source_start=g.source_start,
                source_end=g.source_end,
                text=g.text,
            )
            for g in result.gaps
        ],
        coverage=result.coverage,
        gap_fraction=result.gap_fraction,
        total_chars=result.total_chars,
        model_calls=result.model_calls,
    )


# ---------- Standalone launcher -------------------------------------------


def _run() -> None:
    import uvicorn

    load_dotenv()
    uvicorn.run(
        "benchmarks.chunker.app.main:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_run() or 0)
