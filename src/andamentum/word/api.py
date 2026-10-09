"""The Word tools as Python functions: one async function per job.

Each function that writes takes ``out=``, a folder, and writes only into it:
the edited copy of the document (the input is never edited), the report, and
``state/`` (the ledger, event log and snapshots a later run into the same
folder picks up). Nothing is written next to the input.

The work is done in ``andamentum.compose.harness``; this module names the jobs,
turns ``out`` into the file names the harness writes, and checks the input
file first. Imports are deferred so that importing ``andamentum.word`` is cheap.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from andamentum.compose.harness.errors import WordInputError

if TYPE_CHECKING:
    from andamentum.compose.harness.bindings.bind import BindRun
    from andamentum.compose.harness.bindings.refresh import RefreshRun
    from andamentum.compose.harness.check import CheckRun
    from andamentum.compose.harness.draft.run import DraftResult
    from andamentum.compose.harness.respond.run import RespondRun

__all__ = [
    "CitationInsertion",
    "CitationSearch",
    "FoundWork",
    "ParagraphInfo",
    "UnverifiedWork",
    "bind_numbers",
    "check_document",
    "draft_document",
    "find_citation",
    "insert_citation",
    "list_paragraphs",
    "refresh_numbers",
    "respond_to_reviews",
]

DRAFT_NAME = "draft.docx"


def _require_docx(path: str | Path) -> Path:
    """The input as a Path, or ``WordInputError`` when it cannot be read as .docx."""
    file = Path(path)
    if not file.is_file():
        raise WordInputError(f"{file} does not exist or is not a file")
    if file.suffix.lower() != ".docx":
        raise WordInputError(f"{file} is not a .docx file")
    from andamentum.compose.harness.docx.reader import DocxReadError
    from andamentum.compose.harness.docx.writer import DocxAdapter

    try:
        DocxAdapter().open(file)
    except DocxReadError as exc:
        raise WordInputError(str(exc)) from exc
    return file


def _copy_in(out: str | Path, file: Path, suffix: str) -> Path:
    """Where the edited copy of *file* goes: ``<out>/<stem>.<suffix>.docx``."""
    return Path(out) / f"{file.stem}.{suffix}.docx"


async def check_document(
    path: str | Path,
    *,
    out: str | Path,
    results: str | Path | None = None,
    checks: Sequence[str] = (),
    model: str | None = None,
    decision_model: str | None = None,
    mailto: str | None = None,
    offline: bool = False,
) -> CheckRun:
    """Check a .docx (citations, cross-references, statistics, abstract numbers, structure).

    Writes into *out*: ``<stem>.checked.docx`` (a copy with each located finding
    as a comment), ``<stem>.checked.md`` and ``.json`` (the report) and
    ``state/``. *results* is a folder of result files and enables the numbers
    check (which also needs *model*); *decision_model* enables the citation
    pre-gate; *offline* makes no network call. The result carries the report;
    ``result.report.exit_code`` is 1 when the document has errors.
    """
    from andamentum.compose.harness.check import CheckParams, run_check

    file = _require_docx(path)
    params = CheckParams(
        file=file,
        out=_copy_in(out, file, "checked"),
        results=Path(results) if results is not None else None,
        checks=list(checks),
        model=model,
        decision_model=decision_model,
        mailto=mailto,
        offline=offline,
    )
    return await run_check(params)


async def respond_to_reviews(
    path: str | Path,
    *,
    out: str | Path,
    reviews: str | Path | None = None,
    model: str | None = None,
    decision_model: str | None = None,
    results: str | Path | None = None,
    mailto: str | None = None,
    offline: bool = False,
) -> RespondRun:
    """Answer the comments in a .docx, and a reviewers' report when given.

    Writes into *out*: ``<stem>.responded.docx`` (tracked edits and replies),
    the report, a response letter when *reviews* is given, and ``state/``.
    *model* triages, edits and answers; *decision_model* (a local model)
    triages instead when it is sure. An item that needs the authors is left to
    them: ``result.report.exit_code`` is 1 then.
    """
    from andamentum.compose.harness.respond.run import RespondParams, run_respond

    file = _require_docx(path)
    params = RespondParams(
        file=file,
        out=_copy_in(out, file, "responded"),
        reviews=Path(reviews) if reviews is not None else None,
        model=model,
        decision_model=decision_model,
        results=Path(results) if results is not None else None,
        mailto=mailto,
        offline=offline,
    )
    return await run_respond(params)


async def refresh_numbers(
    path: str | Path, *, out: str | Path, results: str | Path
) -> RefreshRun:
    """Recompute the numbers bound to result files, as tracked changes.

    Writes into *out*: ``<stem>.refreshed.docx``, the report and ``state/``.
    The bindings are read from the ``state/`` of *out*, so use the folder that
    ``bind_numbers`` (or ``draft_document``) wrote them to.
    """
    from andamentum.compose.harness.bindings.refresh import RefreshParams, run_refresh

    file = _require_docx(path)
    params = RefreshParams(
        file=file, out=_copy_in(out, file, "refreshed"), results=Path(results)
    )
    return run_refresh(params)


async def bind_numbers(
    path: str | Path, *, out: str | Path, results: str | Path, model: str
) -> BindRun:
    """Tie the numbers written in a .docx to the result files they come from.

    *model* proposes the matches; each one is verified against the file before
    it is bound. Writes into *out*: ``<stem>.bound.docx``, the report and
    ``state/`` (the ledger that ``refresh_numbers`` reads).
    """
    from andamentum.compose.harness.bindings.bind import BindParams, run_bind

    file = _require_docx(path)
    params = BindParams(
        file=file, out=_copy_in(out, file, "bound"), results=Path(results), model=model
    )
    return await run_bind(params)


async def draft_document(
    task: str,
    *,
    out: str | Path,
    model: str,
    drafter: str | None = None,
    judge: str | None = None,
    into: str | Path | None = None,
    genre: str | None = None,
    sources: Sequence[str] = (),
    results: str | Path | None = None,
    library: str | Path | None = None,
    zotero: bool = False,
    store: str | Path | None = None,
    template: str | Path | None = None,
    max_rounds: int | None = None,
    offline: bool = False,
    confirm_own_sources: bool = False,
    mailto: str | None = None,
    research: bool = False,
    research_mode: Literal["web", "scholarly", "both"] | None = None,
    searxng_url: str | None = None,
    browser: bool = False,
    progress: Callable[[str], None] | None = None,
) -> DraftResult:
    """Draft a document from a task into Word.

    The text arrives as tracked changes, comments, verified citations and
    numbers bound to *results*. With *into*, the drafting extends a copy of
    that .docx (never edited). Writes into *out*: ``draft.docx``, ``draft.report.md``
    and ``.json``, the claims ledger and ``state/``. After drafting, sentences
    that cite nothing are sent through the cite loop (library first, then the
    scholarly finder; a paper is cited only when a passage of it supports the
    sentence); every paper it reads goes into the document store *store* (an
    existing store is reused, a new path makes one; default
    ``state/papers.db``). *model* serves every job;
    *drafter* and *judge* override it for that job. ``result.exit_code`` is 1
    while anything is left for the authors.
    """
    from andamentum.compose.harness.draft.verb import DraftParams, run_draft

    if into is not None:
        _require_docx(into)
    params = DraftParams(
        task=task,
        out=Path(out) / DRAFT_NAME,
        into=Path(into) if into is not None else None,
        genre=genre,
        source=list(sources),
        results=Path(results) if results is not None else None,
        library=Path(library) if library is not None else None,
        zotero=zotero,
        store=Path(store) if store is not None else None,
        template=Path(template) if template is not None else None,
        model=model,
        drafter=drafter,
        judge=judge,
        max_rounds=max_rounds,
        offline=offline,
        confirm_own_sources=confirm_own_sources,
        mailto=mailto,
        research=research,
        research_mode=research_mode,
        searxng_url=searxng_url,
        browser=browser,
    )
    return await run_draft(params, progress=progress)


class FoundWork(BaseModel):
    """A work that exists, whose details agree, and that is not retracted."""

    csl: dict[str, object]
    source: str
    uri: str
    notices: list[dict[str, object]] = Field(default_factory=list)


class UnverifiedWork(BaseModel):
    """A candidate that may not be cited, with the reason."""

    title: str
    doi: str | None = None
    kind: str
    reason: str


class CitationSearch(BaseModel):
    query: str
    verified: list[FoundWork]
    unverified: list[UnverifiedWork]


async def find_citation(
    query: str,
    *,
    library: str | Path | None = None,
    zotero: bool = False,
    limit: int = 5,
    mailto: str | None = None,
) -> CitationSearch:
    """Find works for a title, topic or claim and verify each one.

    Looks in *library* (CSL JSON or BibTeX), Zotero (read only, with *zotero*)
    and Crossref. Only ``verified`` works may be cited. Writes nothing.
    """
    from andamentum.compose.harness.literature.verbs import FindParams, run_find

    params = FindParams(
        query=query,
        library=Path(library) if library is not None else None,
        zotero=zotero,
        limit=limit,
        mailto=mailto,
    )
    found = await run_find(params)
    return CitationSearch.model_validate(found)


class CitationInsertion(BaseModel):
    """What ``insert_citation`` did."""

    cited: bool
    output: Path | None
    message: str


async def insert_citation(
    path: str | Path,
    *,
    out: str | Path,
    paragraph_id: str,
    at: int,
    doi: Sequence[str] = (),
    key: Sequence[str] = (),
    library: str | Path | None = None,
    zotero: bool = False,
    locator: str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    mailto: str | None = None,
) -> CitationInsertion:
    """Cite verified works at a place in a .docx, as tracked changes in a copy.

    The citation goes in verbatim at character offset *at* of the paragraph
    *paragraph_id*. A work that does not verify refuses the whole call and
    nothing is written (``cited`` is False). Writes into *out*:
    ``<stem>.cited.docx`` and ``state/``.
    """
    from andamentum.compose.harness.literature.verbs import InsertParams, run_insert

    file = _require_docx(path)
    output = _copy_in(out, file, "cited")
    params = InsertParams(
        file=file,
        out=output,
        paragraph_id=paragraph_id,
        at=at,
        doi=list(doi),
        key=list(key),
        library=Path(library) if library is not None else None,
        zotero=zotero,
        locator=locator,
        prefix=prefix,
        suffix=suffix,
        mailto=mailto,
    )
    code, message = await run_insert(params)
    return CitationInsertion(
        cited=code == 0, output=output if code == 0 else None, message=message
    )


class ParagraphInfo(BaseModel):
    """One body paragraph of a .docx, as ``insert_citation`` addresses it."""

    id: str
    style: str | None
    heading_level: int | None
    text: str


async def list_paragraphs(path: str | Path) -> list[ParagraphInfo]:
    """The paragraphs of a .docx with the ids that ``insert_citation`` takes.

    Reads the file and changes nothing; no model is called. Raises
    ``WordInputError`` when the file is missing or not a readable .docx.
    """
    from andamentum.compose.harness.docx.writer import DocxAdapter

    file = _require_docx(path)
    view = DocxAdapter().open(file)
    return [
        ParagraphInfo(
            id=p.id, style=p.style, heading_level=p.heading_level, text=p.text
        )
        for p in view.paragraphs
    ]
