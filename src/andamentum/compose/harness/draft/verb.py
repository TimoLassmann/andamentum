"""The ``write`` verb: a task in, a finished document out.

One verb, two destinations. When ``--out`` ends in ``.docx`` or ``--into`` is
given, the document is drafted into Word (``draft_document``: tracked changes,
comments, citations, bound numbers; the file named by ``--into`` is only read).
Otherwise the compose bundle path runs: a folder bundle plus rendered outputs
(``write_document``). Flags that only one destination understands are refused
for the other instead of being ignored.

Exit codes: 0 done, 4 written but open issues remain, 2 usage, 3 crash (the CLI
and the MCP server turn an uncaught exception into 3 and a tool error).
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from andamentum.compose.api import VALID_FORMATS, write_document
from andamentum.compose.research import ResearchSetup
from andamentum.compose.schemas import ROLE_NAMES, Roles

from ..kernel import Verb
from ..literature.access import open_literature
from .run import DraftResult, DraftUsageError, draft_document
from .search import literature_search, research_literature
from .works import LiteratureSearch, ResearchRun

EXIT_OK, EXIT_USAGE, EXIT_OPEN = 0, 2, 4
HARNESS_ROUNDS, BUNDLE_ROUNDS = 2, 4
WORD_ROLES = ("drafter", "judge")  # the critic and the reader are not used


class WriteParams(BaseModel):
    task: str = Field(
        json_schema_extra={"positional": True},
        description="what to write, in your own words",
    )
    out: Path = Field(
        description="the new .docx to write (a folder for a bundle: a --out "
        "without .docx and no --into)"
    )
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
        description=f"most revise rounds (default {HARNESS_ROUNDS} for Word, "
        f"{BUNDLE_ROUNDS} for a bundle)",
    )
    offline: bool = Field(
        default=False,
        description="make no network call: no literature, no URL or DOI checks",
    )
    confirm_own_sources: bool = Field(
        default=False, description="proceed although a source looks confidential"
    )
    mailto: str | None = Field(
        default=None,
        description="your email, for Crossref's and OpenAlex's polite pools",
    )
    research: bool = Field(
        default=False,
        description="research the task once (deep_research v2) and cite what it "
        "finds: claims quoted from real passages; the genre picks the sources",
    )
    research_mode: Literal["web", "scholarly", "both"] | None = Field(
        default=None,
        description="with --research: search the web, scholarly databases or "
        "both, instead of the genre's choice",
    )
    decision_model: str | None = Field(
        default=None,
        description="with --research: a local Ollama decision model (for "
        "example nimble:9b) that screens the works and judges the passages",
    )
    searxng_url: str | None = Field(
        default=None,
        description="with --research: your SearxNG address (web search needs one)",
    )
    format: list[str] = Field(
        default=[], description="bundle only: output format (repeatable)"
    )
    no_provenance: bool = Field(
        default=False, description="bundle only: leave out provenance markers"
    )


def _progress(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def to_word(params: WriteParams) -> bool:
    """True when the document goes into Word, False for the bundle path."""
    return params.into is not None or params.out.suffix.lower() == ".docx"


def _roles(params: WriteParams, *, needed: tuple[str, ...] = ROLE_NAMES) -> Roles:
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


def research_setup(
    *,
    mailto: str | None,
    searxng_url: str | None,
    decision_model: str | None,
    mode: Literal["web", "scholarly", "both"] | None,
) -> ResearchSetup:
    """The research settings from CLI flags; credentials come from the
    environment (``.env`` is loaded by the CLI), never from flags."""
    from andamentum.deep_research.v2 import SourceConfig

    return ResearchSetup(
        sources=SourceConfig(
            mailto=mailto or os.environ.get("OPENALEX_EMAIL") or None,
            openalex_api_key=os.environ.get("OPENALEX_API_KEY") or None,
            semantic_scholar_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None,
            searxng_url=searxng_url,
        ),
        decision_model=decision_model,
        mode=mode,
    )


def _research_setup(params: WriteParams) -> ResearchSetup | None:
    if not params.research:
        return None
    return research_setup(
        mailto=params.mailto,
        searxng_url=params.searxng_url,
        decision_model=params.decision_model,
        mode=params.research_mode,
    )


def _refuse_foreign_flags(params: WriteParams, *, word: bool) -> None:
    if not params.research:
        idle = [
            flag
            for flag, on in (
                ("--research-mode", params.research_mode is not None),
                ("--decision-model", params.decision_model is not None),
                ("--searxng-url", params.searxng_url is not None),
            )
            if on
        ]
        if idle:
            raise DraftUsageError(f"{', '.join(idle)} need --research")
    if params.research and params.offline:
        raise DraftUsageError("--research searches the network; drop --offline")
    if word:
        used = [
            flag
            for flag, on in (
                ("--format", bool(params.format)),
                ("--no-provenance", params.no_provenance),
            )
            if on
        ]
        if used:
            raise DraftUsageError(
                f"{', '.join(used)} only apply to a bundle (an --out without .docx)"
            )
        if params.offline and (params.library or params.zotero):
            raise DraftUsageError(
                "--library and --zotero need the network to verify works; "
                "drop --offline"
            )
        return
    used = [
        flag
        for flag, on in (
            ("--results", params.results is not None),
            ("--library", params.library is not None),
            ("--zotero", params.zotero),
            ("--template", params.template is not None),
            ("--offline", params.offline),
            ("--mailto", params.mailto is not None and not params.research),
        )
        if on
    ]
    if used:
        raise DraftUsageError(
            f"{', '.join(used)} only apply when writing into Word "
            "(--out FILE.docx or --into)"
        )


async def run_write(
    params: WriteParams,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    url_check: Callable[[str], tuple[bool, str]] | None = None,
) -> DraftResult:
    """The Word path; the seams (*transport*, *url_check*) are for tests."""
    roles = _roles(params, needed=WORD_ROLES)
    project = params.out.with_name(f"{params.out.stem}.harness")

    async def draft(
        literature: LiteratureSearch | None, research: ResearchRun | None = None
    ) -> DraftResult:
        return await draft_document(
            params.task,
            out=params.out,
            roles=roles,
            into=params.into,
            genre=params.genre,
            sources=params.source,
            results=params.results,
            literature=literature,
            research=research,
            template=params.template,
            max_rounds=_rounds(params, HARNESS_ROUNDS),
            offline=params.offline,
            confirm_own_sources=params.confirm_own_sources,
            mailto=params.mailto,
            progress=_progress,
            transport=transport,
            url_check=url_check,
        )

    if params.offline:  # nothing is looked up, so nothing can be verified or cited
        return await draft(None)
    async with open_literature(
        library=params.library,
        zotero=params.zotero,
        cache_dir=project / "cache" / "http",
        mailto=params.mailto,
        transport=transport,
        url_check=url_check,
    ) as (sources, http):
        setup = _research_setup(params)
        if setup is None:
            return await draft(literature_search(sources=sources, http=http))
        # Research replaces Crossref as the place new works are found; the
        # user's library and Zotero are still searched, and every work, found
        # either way, is verified through the same Crossref lookup.
        discovery = replace(sources, crossref=None)
        own = discovery.library is not None or discovery.zotero is not None
        return await draft(
            literature_search(sources=discovery, http=http) if own else None,
            research_literature(
                setup=setup, model=roles.drafter, http=http, progress=_progress
            ),
        )


def _rounds(params: WriteParams, default: int) -> int:
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


async def _write_bundle(params: WriteParams) -> int:
    formats = list(params.format)
    bad = [f for f in formats if f not in VALID_FORMATS]
    if bad:
        raise DraftUsageError(
            f"unknown --format {bad[0]!r}; valid: {', '.join(VALID_FORMATS)}"
        )
    result = await write_document(
        params.task,
        out=params.out,
        roles=_roles(params),
        genre=params.genre,
        sources=params.source,
        max_rounds=_rounds(params, BUNDLE_ROUNDS),
        research=_research_setup(params),
        confirm_own_sources=params.confirm_own_sources,
        provenance=not params.no_provenance,
        formats=formats or None,
        progress=_progress,
    )
    for line in result.summary_lines():
        print(line)
    return EXIT_OK if result.goal_met else EXIT_OPEN


def _handle(params: WriteParams) -> int:
    word = to_word(params)
    try:
        _refuse_foreign_flags(params, word=word)
        if word:
            result = asyncio.run(run_write(params))
            _print_word(result)
            return result.exit_code
        return asyncio.run(_write_bundle(params))
    except DraftUsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


WRITE_VERB = Verb(
    name="write",
    summary="write a document from a task: into Word as tracked changes (--out FILE.docx, "
    "optionally --into) or as a bundle folder. "
    'Example: write "Results section" --out paper.docx --into draft.docx '
    "--results results --library refs.json --model M",
    params=WriteParams,
    handler=_handle,
)

VERBS: tuple[Verb, ...] = (WRITE_VERB,)
