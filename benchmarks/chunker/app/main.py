"""Helper FastAPI app for chunker benchmark case authoring.

Run::

    uv run python -m benchmarks.chunker.app
    # → http://127.0.0.1:8765

Workflow:
  1. Paste source text + pick model/domain. Click "Chunk it".
  2. Inspect highlighted preview (left) + editable unit list (right).
  3. Edit titles/anchors; live anchor validation runs on every change.
  4. Set case-level metadata (name, convention, F1 floor, tolerance).
  5. Click "Save case" — writes <name>.input.<ext> + <name>.truth.json
     into benchmarks/chunker/cases/, ready for the benchmark to discover.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urljoin, urlparse

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
from andamentum.chunker.validation import find_anchor
from andamentum.core.agents import AgentRunner
from andamentum.deep_research.backends import HttpxSearchBackend, SearchBackend
from andamentum.deep_research.searxng import SearxngManager

STATIC_DIR = Path(__file__).parent / "static"
CASES_DIR = Path(__file__).resolve().parents[1] / "cases"

# Restrict case names so they cannot escape CASES_DIR or shadow fixtures (`_*`).
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_ALLOWED_EXTS = {"md", "txt", "html", "py", "rst", "json"}


# ---------- Executor factory (overridable for tests) ----------------------


def _default_executor_factory(model: str) -> ExecutorFn:
    runner = AgentRunner(model=model)
    return make_runner_executor(runner)


_executor_factory: Callable[[str], ExecutorFn] = _default_executor_factory


def _set_executor_factory(factory: Callable[[str], ExecutorFn]) -> None:
    """Test hook: swap the executor factory for fakes."""
    global _executor_factory
    _executor_factory = factory


_cases_dir_override: Path | None = None


def _set_cases_dir(path: Path | None) -> None:
    """Test hook: redirect saves to a temp dir."""
    global _cases_dir_override
    _cases_dir_override = path


def _cases_dir() -> Path:
    return _cases_dir_override if _cases_dir_override is not None else CASES_DIR


# ---------- Search backend + SearXNG manager (overridable for tests) ------

SEARXNG_PORT = 4070
SEARXNG_URL = f"http://127.0.0.1:{SEARXNG_PORT}"


def _default_search_backend_factory() -> SearchBackend:
    return HttpxSearchBackend(searxng_url=SEARXNG_URL)


def _default_searxng_manager_factory() -> SearxngManager:
    return SearxngManager(host_port=SEARXNG_PORT)


_search_backend_factory: Callable[[], SearchBackend] = _default_search_backend_factory
_searxng_manager_factory: Callable[[], SearxngManager] = (
    _default_searxng_manager_factory
)


def _set_search_backend_factory(factory: Callable[[], SearchBackend]) -> None:
    """Test hook: swap the search-backend factory for fakes."""
    global _search_backend_factory
    _search_backend_factory = factory


def _set_searxng_manager_factory(factory: Callable[[], SearxngManager]) -> None:
    """Test hook: swap the SearxngManager factory for fakes."""
    global _searxng_manager_factory
    _searxng_manager_factory = factory


# ---------- /api/chunk models ---------------------------------------------


class ChunkRequest(BaseModel):
    text: str = Field(..., min_length=1)
    domain: str = "general"  # accepted for compat; currently no-op
    # Model is now optional — used only as the LLM judge for grey-zone
    # boundaries. Most academic papers finish at the structural stage with
    # zero LLM calls.
    model: str | None = None
    target_min_chars: int = 2_000
    target_max_chars: int = 10_000
    # Legacy fields (window_size, lookahead) accepted but ignored.
    window_size: int | None = None
    lookahead: int | None = None


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


# ---------- /api/match-anchor models --------------------------------------


class MatchAnchorRequest(BaseModel):
    text: str
    anchor: str
    search_from: int = 0


class MatchAnchorResponse(BaseModel):
    found: bool
    start: int | None = None
    end: int | None = None
    method: str | None = None


# ---------- /api/save-case models -----------------------------------------


class SaveUnit(BaseModel):
    title: str = Field(..., min_length=1)
    start_anchor: str = Field(..., min_length=1)
    end_anchor: str = Field(..., min_length=1)
    kind: str | None = None  # optional — most truth files don't include it


class SaveCaseRequest(BaseModel):
    name: str
    extension: Literal["md", "txt", "html", "py", "rst", "json"]
    text: str = Field(..., min_length=1)
    convention: str = Field(..., min_length=1)
    expected_f1_floor: float = Field(..., ge=0.0, le=1.0)
    boundary_tolerance_chars: int = Field(50, ge=0, le=10_000)
    domain: Literal["general", "academic", "web", "code", "transcript"]
    units: list[SaveUnit] = Field(..., min_length=1)
    overwrite: bool = False


class SaveCaseResponse(BaseModel):
    name: str
    input_path: str
    truth_path: str
    units_written: int


# ---------- App + endpoints -----------------------------------------------

app = FastAPI(title="andamentum-chunker case editor", version="0.2.0")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# Mount static files (for app.js, future css)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.post("/api/chunk", response_model=ChunkResponse)
async def chunk(req: ChunkRequest) -> ChunkResponse:
    judge_executor = _executor_factory(req.model) if req.model else None
    try:
        result = await extract_units(
            req.text,
            target_min_chars=req.target_min_chars,
            target_max_chars=req.target_max_chars,
            judge_executor=judge_executor,
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


@app.post("/api/match-anchor", response_model=MatchAnchorResponse)
async def match_anchor(req: MatchAnchorRequest) -> MatchAnchorResponse:
    """Resolve an anchor against source text using the chunker's tiered matcher.

    Mirrors the same logic the runner applies to truth-file anchors so what
    the editor shows is exactly what the benchmark will see.
    """
    if not req.anchor:
        return MatchAnchorResponse(found=False)
    match = find_anchor(req.anchor, req.text, search_from=req.search_from)
    if match is None:
        return MatchAnchorResponse(found=False)
    return MatchAnchorResponse(
        found=True, start=match.start, end=match.end, method=match.method
    )


@app.post("/api/save-case", response_model=SaveCaseResponse)
async def save_case(req: SaveCaseRequest) -> SaveCaseResponse:
    name = req.name.strip().lower()
    if not _NAME_RE.fullmatch(name):
        raise HTTPException(
            status_code=400,
            detail=(
                "case name must be lowercase letters/digits/_/-, "
                "1-64 chars, not starting with '_'"
            ),
        )
    if req.extension not in _ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"extension must be one of {sorted(_ALLOWED_EXTS)}",
        )

    # Verify every anchor resolves before writing anything to disk.
    cursor = 0
    for i, unit in enumerate(req.units):
        start = find_anchor(unit.start_anchor, req.text, search_from=cursor)
        if start is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unit {i} ({unit.title!r}): start_anchor "
                    f"{unit.start_anchor!r} not found in text after cursor {cursor}"
                ),
            )
        end = find_anchor(unit.end_anchor, req.text, search_from=start.end)
        if end is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unit {i} ({unit.title!r}): end_anchor "
                    f"{unit.end_anchor!r} not found after start_anchor"
                ),
            )
        cursor = end.end

    cases_dir = _cases_dir()
    cases_dir.mkdir(parents=True, exist_ok=True)
    input_path = cases_dir / f"{name}.input.{req.extension}"
    truth_path = cases_dir / f"{name}.truth.json"

    # Belt-and-braces: confirm both resolved paths still live inside cases_dir.
    cases_dir_resolved = cases_dir.resolve()
    for p in (input_path, truth_path):
        if cases_dir_resolved not in p.resolve().parents:
            raise HTTPException(
                status_code=400, detail=f"resolved path escapes cases dir: {p}"
            )

    if not req.overwrite and (input_path.exists() or truth_path.exists()):
        raise HTTPException(
            status_code=409,
            detail=(f"case {name!r} already exists (pass overwrite=true to replace)"),
        )

    truth_doc = {
        "convention": req.convention,
        "expected_f1_floor": req.expected_f1_floor,
        "boundary_tolerance_chars": req.boundary_tolerance_chars,
        "domain": req.domain,
        "units": [
            {
                "title": u.title,
                "start_anchor": u.start_anchor,
                "end_anchor": u.end_anchor,
                **({"kind": u.kind} if u.kind else {}),
            }
            for u in req.units
        ],
    }

    # Atomic writes: temp file + rename so a partial write can't poison the dir.
    _atomic_write_text(input_path, req.text)
    _atomic_write_text(
        truth_path, json.dumps(truth_doc, indent=2, ensure_ascii=False) + "\n"
    )

    return SaveCaseResponse(
        name=name,
        input_path=str(input_path),
        truth_path=str(truth_path),
        units_written=len(req.units),
    )


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


# ---------- PDF helpers ---------------------------------------------------

# Maps known abstract/landing pages to their PDF equivalents. Each entry is
# (compiled_pattern, format_string_using_match_groups). New venues should be
# added here rather than scattered through the fetch endpoint.
_ARXIV_ABS_RE = re.compile(
    r"^(https?://(?:www\.)?arxiv\.org)/abs/([\w./+-]+?)(?:v\d+)?/?$",
    re.IGNORECASE,
)
_BIORXIV_ABS_RE = re.compile(
    r"^(https?://(?:www\.)?(?:bio|med)rxiv\.org/content/[\w./+-]+v\d+)/?$",
    re.IGNORECASE,
)


def _canonicalise_pdf_url(url: str) -> str:
    """Rewrite known abstract/landing URLs to their PDF equivalents.

    No-op for any URL not matched. Currently handles arXiv and bio/medRxiv —
    the venues most likely to appear in chunker training cases.
    """
    m = _ARXIV_ABS_RE.match(url)
    if m:
        return f"{m.group(1)}/pdf/{m.group(2)}"
    m = _BIORXIV_ABS_RE.match(url)
    if m:
        return f"{m.group(1)}.full.pdf"
    return url


def _is_pdf_url(url: str) -> bool:
    """Heuristic: does this URL serve a PDF?

    Matches both the obvious case (path ends in `.pdf`) and arXiv's
    extension-less convention (`arxiv.org/pdf/<id>`).
    """
    parsed = urlparse(url)
    path = parsed.path.lower()
    if path.endswith(".pdf"):
        return True
    if "arxiv.org" in parsed.netloc.lower() and path.startswith("/pdf/"):
        return True
    return False


# Markdown link `[text](url)` — captures text in g1, url in g2. URLs may
# contain everything except whitespace and unescaped parens; we accept any
# non-whitespace, non-`)` run, which is fine for trafilatura's output.
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((\S+?)\)")
# Words that strongly imply "this link is the PDF version of the page".
_PDF_TEXT_RE = re.compile(
    r"\b(?:view|download|full[\s-]?text|original|paper)\s*pdf\b|^pdf$",
    re.IGNORECASE,
)


def _extract_pdf_links(markdown: str, base_url: str) -> list[dict[str, str]]:
    """Find PDF links in extracted markdown.

    Returns up to 5 unique entries, each `{url, label}`. A link qualifies if:
      - its href ends in `.pdf` (with optional ?query/#fragment), OR
      - its anchor text matches a "view PDF"-style phrase.

    Relative hrefs are resolved against `base_url`.
    """
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for match in _MD_LINK_RE.finditer(markdown):
        text, raw_url = match.group(1).strip(), match.group(2).strip()
        # Strip a trailing markdown punctuation that the regex may have grabbed
        # (commas, full stops). PDFs themselves never end in those characters.
        raw_url = raw_url.rstrip(".,;:)")
        try:
            absolute = urljoin(base_url, raw_url)
        except Exception:
            continue
        if not absolute.lower().startswith(("http://", "https://")):
            continue

        path = urlparse(absolute).path.lower()
        looks_pdf = path.endswith(".pdf") or _PDF_TEXT_RE.search(text) is not None
        if not looks_pdf:
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        out.append({"url": absolute, "label": text or "PDF"})
        if len(out) >= 5:
            break
    return out


# ---------- /api/searxng-* + /api/search + /api/fetch models --------------


class SearxngStatusResponse(BaseModel):
    state: Literal["running", "stopped", "podman-missing", "error"]
    message: str
    url: str = SEARXNG_URL


class SearchHit(BaseModel):
    title: str
    url: str
    snippet: str
    domain: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_results: int = Field(10, ge=1, le=30)


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class FetchRequest(BaseModel):
    url: str = Field(..., min_length=1)


class PdfLink(BaseModel):
    url: str
    label: str


class FetchResponse(BaseModel):
    requested_url: str
    final_url: str
    title: str
    markdown: str
    word_count: int
    original_length: int
    truncated: bool
    pdf_links: list[PdfLink]
    is_pdf: bool


# ---------- /api/searxng-* + /api/search + /api/fetch endpoints -----------


@app.get("/api/searxng-status", response_model=SearxngStatusResponse)
async def searxng_status() -> SearxngStatusResponse:
    """Read-only status check for the local SearXNG container."""
    try:
        manager = _searxng_manager_factory()
    except Exception as exc:
        return SearxngStatusResponse(state="error", message=str(exc))
    try:
        running = manager.is_running()
    except RuntimeError as exc:
        # SearxngManager raises RuntimeError when podman is not on PATH.
        msg = str(exc)
        if "podman" in msg.lower():
            return SearxngStatusResponse(state="podman-missing", message=msg)
        return SearxngStatusResponse(state="error", message=msg)
    except Exception as exc:
        return SearxngStatusResponse(state="error", message=str(exc))
    if running:
        return SearxngStatusResponse(state="running", message="container up")
    return SearxngStatusResponse(state="stopped", message="container not running")


@app.post("/api/searxng-start", response_model=SearxngStatusResponse)
async def searxng_start() -> SearxngStatusResponse:
    """Pull image (if needed) and start the SearXNG container.

    Slow on first invocation — pulls the image (~30-60s).
    """
    import asyncio

    try:
        manager = _searxng_manager_factory()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"manager init failed: {exc}")

    try:
        # ensure_running is blocking (shells out to podman) — run in a thread
        # so the event loop stays responsive for the status pill.
        await asyncio.to_thread(manager.ensure_running)
    except RuntimeError as exc:
        msg = str(exc)
        if "podman" in msg.lower():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"podman is not installed or not on PATH — install it first "
                    f"(`brew install podman`) and run `podman machine init && "
                    f"podman machine start`. Error: {msg}"
                ),
            )
        raise HTTPException(status_code=500, detail=msg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return SearxngStatusResponse(state="running", message="started")


@app.post("/api/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    """Query SearXNG via the HttpxSearchBackend.

    Returns 503 with a clear message if SearXNG isn't running, instead of
    silently returning an empty list (the backend's default behaviour).
    """
    backend = _search_backend_factory()
    try:
        hits = await backend.search(req.query, max_results=req.max_results)
    finally:
        await _maybe_close(backend)

    if not hits:
        # Distinguish empty results from "SearXNG isn't reachable" by checking
        # status now — a stopped container is the most common cause.
        try:
            manager = _searxng_manager_factory()
            if not manager.is_running():
                raise HTTPException(
                    status_code=503,
                    detail="SearXNG is not running — start it first.",
                )
        except RuntimeError:
            # podman missing → just return empty hits, surface the truth via
            # /api/searxng-status which the UI already polls.
            pass

    return SearchResponse(hits=[SearchHit(**h.model_dump()) for h in hits])


@app.post("/api/fetch", response_model=FetchResponse)
async def fetch(req: FetchRequest) -> FetchResponse:
    """Fetch a URL and extract clean markdown via the harvest pipeline.

    For known abstract pages (arXiv, bio/medRxiv) the URL is rewritten to its
    PDF equivalent before fetching. Harvest dispatches to the right extractor
    by sniffing format + page metadata (article-tagged HTML → trafilatura,
    listing/index → Docling, ambiguous → race both and pick the
    higher-scoring output). After extraction the markdown is scanned for any
    other PDF links so the UI can offer them as one-click upgrades.
    """
    from urllib.parse import urlparse

    from andamentum.harvest import HarvestError, extract as harvest_extract

    final_url = _canonicalise_pdf_url(req.url)
    try:
        markdown = await harvest_extract(final_url)
    except HarvestError as exc:
        raise HTTPException(status_code=502, detail=f"fetch failed: {exc}")

    is_pdf = _is_pdf_url(final_url)
    pdf_links = [] if is_pdf else _extract_pdf_links(markdown, base_url=final_url)

    # Title: first heading line, falling back to the URL's hostname.
    title = _infer_title(markdown) or urlparse(final_url).netloc or final_url

    # Harvest doesn't truncate, so original_length == len(markdown).
    word_count = len(markdown.split())

    return FetchResponse(
        requested_url=req.url,
        final_url=final_url,
        title=title,
        markdown=markdown,
        word_count=word_count,
        original_length=len(markdown),
        truncated=False,
        pdf_links=[PdfLink(**link) for link in pdf_links],
        is_pdf=is_pdf,
    )


def _infer_title(markdown: str) -> str:
    """Pick the first markdown heading line as the title, or empty string."""
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


async def _maybe_close(backend: Any) -> None:
    """Close the backend if it owns its HTTP client (HttpxSearchBackend)."""
    close = getattr(backend, "close", None)
    if callable(close):
        try:
            await close()
        except Exception:  # nosec  — best effort
            pass


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
