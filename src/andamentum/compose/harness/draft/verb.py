"""The ``draft`` verb: a task drafted into Word.

The document is drafted as tracked changes, comments, citations and bound
numbers (``draft_document``); the file named by ``--into`` is only read.
The folder-document pipeline is a different tool (``andamentum-compose``).

Exit codes: 0 done, 1 written but open issues remain, 2 usage, 3 crash (the CLI
and the MCP server turn an uncaught exception into 3 and a tool error).
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
import sqlite3
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from andamentum.compose.research import ResearchSetup, research_setup
from andamentum.core.secrets import key_status
from andamentum.compose.schemas import ROLE_NAMES, Roles

from ..kernel import Verb
from ..literature.access import open_literature
from ..project import project_dir
from .run import DraftResult, DraftUsageError, draft_document
from .cite import CiteTools
from .search import cite_tools, research_literature
from .works import ResearchRun

EXIT_OK, EXIT_USAGE, EXIT_OPEN = 0, 2, 1
WORD_ROUNDS = 2
WORD_ROLES = ("drafter", "judge")  # the critic and the reader are not used


class DraftParams(BaseModel):
    task: str = Field(
        json_schema_extra={"positional": True},
        description="what to write, in your own words",
    )
    out: Path = Field(description="the new .docx to write (a path, never the input)")
    into: Path | None = Field(
        default=None,
        description="draft into a copy of this .docx as tracked changes; it is never edited",
    )
    genre: str | None = Field(default=None, description="pack name or genre .yaml")
    source: list[str] = Field(
        default=[], description="a file or URL to draw on (repeatable)"
    )
    results: Path | None = Field(
        default=None,
        description="folder of result files (csv, tsv, json): numbers are bound to them",
    )
    library: Path | None = Field(
        default=None, description="your references: CSL JSON (.json) or BibTeX (.bib)"
    )
    zotero: bool = Field(
        default=False, description="also cite from your Zotero library (read only)"
    )
    store: Path | None = Field(
        default=None,
        description="document store that keeps the papers read to cite the draft: "
        "an existing store is reused, a new path makes a new one (default: "
        "papers.db in the state folder)",
    )
    template: Path | None = Field(
        default=None, description="a .docx or .dotx whose styles the new document uses"
    )
    model: str | None = Field(
        default=None, description="model id for every role not set below"
    )
    drafter: str | None = Field(default=None, description="model for the drafter")
    critic: str | None = Field(default=None, description="model for the critic")
    judge: str | None = Field(default=None, description="model for the judge")
    reader: str | None = Field(default=None, description="model for the reader")
    max_rounds: int | None = Field(
        default=None,
        description=f"most revise rounds (default {WORD_ROUNDS})",
    )
    offline: bool = Field(
        default=False,
        description="make no network call: no literature, no URL or DOI checks",
    )
    confirm_own_sources: bool = Field(
        default=False, description="proceed although a source looks confidential"
    )
    project: Path | None = Field(
        default=None,
        description="folder that keeps the ledger and logs (default: state/ beside the output)",
    )
    mailto: str | None = Field(
        default=None,
        description="your email, for Crossref's and OpenAlex's polite pools",
    )
    research: bool = Field(
        default=False,
        description="research the task once (deep_research) and cite what it "
        "finds: claims quoted from real passages; the genre picks the sources",
    )
    research_mode: Literal["web", "scholarly", "both"] | None = Field(
        default=None,
        description="with --research: search the web, scholarly databases or "
        "both, instead of the genre's choice",
    )
    searxng_url: str | None = Field(
        default=None,
        description="with --research: your SearxNG address (web search needs "
        "this or --browser)",
    )
    browser: bool = Field(
        default=False,
        description="with --research: search the web and read pages through an "
        "isolated Chrome in a podman container (started for the research, "
        "removed after) instead of SearxNG",
    )


def _progress(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _roles(params: DraftParams, *, needed: tuple[str, ...] = ROLE_NAMES) -> Roles:
    """Roles from the flags. *needed* must have a model; the others (unused by the
    Word path) take the drafter's."""
    chosen: dict[str, str] = {}
    missing: list[str] = []
    for role in ROLE_NAMES:
        value = getattr(params, role) or params.model
        if value:
            chosen[role] = value
        elif role in needed:
            missing.append(role)
    if missing:
        raise DraftUsageError(
            f"no model for: {', '.join(missing)}. Pass --model MODEL to use one "
            "model for every role, or --" + ", --".join(missing) + " individually."
        )
    for role in ROLE_NAMES:
        chosen.setdefault(role, chosen["drafter"])
    return Roles(**chosen)


def _research_setup(params: DraftParams) -> ResearchSetup | None:
    if not params.research:
        return None
    return research_setup(
        mailto=params.mailto,
        searxng_url=params.searxng_url,
        browser=params.browser,
        mode=params.research_mode,
    )


def _check_store_file(path: Path) -> None:
    """An existing *path* must be a document store (SQLite with a ``documents``
    table, or a SQLite file with no tables); anything else is a usage error."""
    if not path.exists():
        return
    try:
        with path.open("rb") as fh:
            header = fh.read(16)
    except OSError as exc:
        raise DraftUsageError(f"--store {path} cannot be read: {exc}") from exc
    if header == b"":  # an empty file is a valid fresh SQLite database
        return
    if header != b"SQLite format 3\x00":
        raise DraftUsageError(f"--store {path} is not a document store file")
    try:
        con = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
        try:
            tables = {
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        finally:
            con.close()
    except sqlite3.Error as exc:
        raise DraftUsageError(f"--store {path} is not a document store: {exc}") from exc
    if tables and "documents" not in tables:
        raise DraftUsageError(
            f"--store {path} is a database of another kind, not a document store"
        )


PAPERS_STORE = "papers.db"  # the document store of the papers read, in the state folder
CITE_CACHE = "cite-http.db"  # the HTTP cache the cite loop's finder and reader share


def _refuse_idle_flags(params: DraftParams) -> None:
    if not params.research:
        idle = [
            flag
            for flag, on in (
                ("research_mode", params.research_mode is not None),
                ("--searxng-url", params.searxng_url is not None),
                ("--browser", params.browser),
            )
            if on
        ]
        if idle:
            raise DraftUsageError(f"{', '.join(idle)} only works with --research")
    if params.research and params.offline:
        raise DraftUsageError("--research searches the network; drop --offline")
    if params.store is not None and params.offline:
        raise DraftUsageError("--store keeps papers read online; drop --offline")


async def run_draft(
    params: DraftParams,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    url_check: Callable[[str], tuple[bool, str]] | None = None,
    progress: Callable[[str], None] | None = None,
    open_cite: Callable[..., AbstractAsyncContextManager[CiteTools]] = cite_tools,
) -> DraftResult:
    """Draft into Word; the seams (*transport*, *url_check*, *open_cite*, which
    opens the cite loop's tools with ``cite_tools``' arguments) are for tests."""
    _refuse_idle_flags(params)
    say = progress or _progress
    roles = _roles(params, needed=WORD_ROLES)
    project = project_dir(params.out, params.project)

    async def draft(
        research: ResearchRun | None = None, cite: CiteTools | None = None
    ) -> DraftResult:
        return await draft_document(
            params.task,
            out=params.out,
            roles=roles,
            into=params.into,
            genre=params.genre,
            sources=params.source,
            results=params.results,
            research=research,
            cite=cite,
            template=params.template,
            max_rounds=_rounds(params, WORD_ROUNDS),
            offline=params.offline,
            confirm_own_sources=params.confirm_own_sources,
            mailto=params.mailto,
            project=project,
            progress=say,
            transport=transport,
            url_check=url_check,
        )

    if params.offline:  # nothing is looked up, so nothing can be verified or cited
        return await draft()
    papers = params.store or project / PAPERS_STORE
    if papers.is_dir():
        raise DraftUsageError(f"--store {papers} is a folder; name a store file")
    _check_store_file(papers)
    async with open_literature(
        library=params.library,
        zotero=params.zotero,
        cache_dir=project / "cache" / "http",
        mailto=params.mailto,
        transport=transport,
        url_check=url_check,
    ) as (sources, http):
        setup = _research_setup(params)
        if setup is None:  # research_setup prints the same line
            from andamentum.deep_research import API_KEY_NAMES

            say(key_status(API_KEY_NAMES))
        async with open_cite(
            sources=sources,
            http=http,
            model=roles.judge,
            store=papers,
            cache=project / CITE_CACHE,
            mailto=params.mailto,
        ) as cite:
            # Literature enters through the cite loop alone: the library and
            # Zotero first, then the scholarly finder; every work it cites is
            # verified through Crossref. Research adds its findings as evidence.
            research = (
                None
                if setup is None
                else research_literature(
                    setup=setup, model=roles.drafter, http=http, progress=say
                )
            )
            return await draft(research, cite)


def _rounds(params: DraftParams, default: int) -> int:
    return default if params.max_rounds is None else params.max_rounds


def _print_word(result: DraftResult) -> None:
    print(f"document: {result.output}")
    print(f"report: {result.report}")
    print(f"project: {result.project}")
    print(f"claims ledger: {result.ledger}")
    print(
        f"{len(result.sections)} section(s), {result.citations} citation(s), "
        f"{result.numbers_bound} number(s) bound"
    )
    for line in result.open_errors:
        print(f"open: {line}")
    for line in result.refused:
        print(f"refused: {line}")
    for line in result.author_questions:
        print(f"question: {line}")


def _handle(params: DraftParams) -> int:
    try:
        result = asyncio.run(run_draft(params))
        _print_word(result)
        return result.exit_code
    except DraftUsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


DRAFT_VERB = Verb(
    name="draft",
    summary="draft a document from a task into Word as tracked changes, comments, "
    "citations and bound numbers. "
    'Example: draft "Results section" --out paper.docx --into draft.docx '
    "--results results --library refs.json --model M",
    params=DraftParams,
    handler=_handle,
)

VERBS: tuple[Verb, ...] = (DRAFT_VERB,)
