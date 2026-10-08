"""``find-citation`` and ``insert-citation``: the literature layer as verbs.

``find-citation QUERY`` looks in the user's library file, Zotero (read only,
``--zotero``) and Crossref, verifies every candidate (it exists, its details
agree, it is not retracted) and prints JSON: the verified works, which may be
cited, and the unverified ones with the reason, which may not.

``insert-citation FILE.docx --paragraph-id ID --at N --doi DOI`` (or ``--key K``
for an item of ``--library`` / ``--zotero``) resolves each work and, only when
every one is verified, writes the citation and the bibliography entry as
tracked changes to a copy (``<stem>.cited.docx``). It never edits FILE. A work
that does not verify refuses the whole call (exit 1) and nothing is written.
The citation is inserted verbatim at character offset ``--at`` of the paragraph
text: the caller chooses the spacing and the full stop.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from andamentum.references import ReferenceHttp

from ..bindings.common import (
    EXIT_OK,
    EXIT_USAGE,
    UsageError,
    output_path,
)
from ..docx.reader import DocxReadError
from ..docx.writer import DocxAdapter
from ..project import project_dir, write_copy
from ..kernel import (
    InsertCitation,
    ProjectStore,
    Transaction,
    UpdateBibliography,
    Verb,
)
from . import (
    LiteratureError,
    Sources,
    Unverified,
    VerifiedWork,
    Work,
    fetch_work,
    find_works,
    read_library,
    resolve,
)
from .access import open_literature

EXIT_OPEN = 1
AUTHOR = "andamentum insert-citation"


class FindParams(BaseModel):
    query: str = Field(
        json_schema_extra={"positional": True},
        description="a title, a topic or a claim to find works for",
    )
    library: Path | None = Field(
        default=None, description="your references: CSL JSON (.json) or BibTeX (.bib)"
    )
    zotero: bool = Field(
        default=False,
        description="also read your Zotero library (local API, read only)",
    )
    limit: int = Field(default=5, description="most works to look up")
    mailto: str | None = Field(
        default=None, description="your email, for Crossref's polite pool"
    )
    cache: Path | None = Field(
        default=None, description="folder that keeps Crossref answers between runs"
    )


class InsertParams(BaseModel):
    file: Path = Field(
        json_schema_extra={"positional": True}, description="the .docx to cite in"
    )
    paragraph_id: str = Field(
        description="id of the paragraph (list them with 'andamentum-word paragraphs FILE.docx')"
    )
    at: int = Field(description="character offset in the paragraph text")
    doi: list[str] = Field(default=[], description="DOI of a work to cite (repeatable)")
    key: list[str] = Field(
        default=[],
        description="id of an item of --library or --zotero to cite (repeatable)",
    )
    library: Path | None = Field(
        default=None, description="your references: CSL JSON (.json) or BibTeX (.bib)"
    )
    zotero: bool = Field(default=False, description="read items from Zotero too")
    locator: str | None = Field(default=None, description="for example 'p. 12'")
    prefix: str | None = Field(default=None, description="text before the citation")
    suffix: str | None = Field(default=None, description="text after the citation")
    out: Path = Field(description="the cited copy to write (a .docx path)")
    project: Path | None = Field(
        default=None, description="folder that keeps the ledger and logs (default: state/ beside the output)"
    )
    mailto: str | None = Field(
        default=None, description="your email, for Crossref's polite pool"
    )


def _work_json(work: Work) -> dict[str, object]:
    return {"csl": work.csl, "source": work.source, "uri": work.uri}


async def run_find(
    params: FindParams, *, transport: httpx.AsyncBaseTransport | None = None
) -> dict[str, object]:
    if params.limit < 1:
        raise UsageError("--limit must be at least 1")
    if params.library is not None and not params.library.is_file():
        raise UsageError(f"--library {params.library} is not a file")
    async with open_literature(
        library=params.library,
        zotero=params.zotero,
        cache_dir=params.cache,
        mailto=params.mailto,
        transport=transport,
    ) as (sources, http):
        found = await find_works(params.query, sources=sources, limit=params.limit)
        verified: list[dict[str, object]] = []
        unverified: list[dict[str, object]] = []
        for work in found:
            verdict = await resolve(work, http=http)
            if isinstance(verdict, Unverified):
                unverified.append(
                    {
                        "title": verdict.work.title,
                        "doi": verdict.work.doi,
                        "kind": verdict.kind,
                        "reason": verdict.reason,
                    }
                )
            else:
                verified.append(
                    {
                        **_work_json(verdict.work),
                        "notices": [n.model_dump() for n in verdict.notices],
                    }
                )
    return {"query": params.query, "verified": verified, "unverified": unverified}


def _handle_find(params: FindParams) -> int:
    try:
        result = asyncio.run(run_find(params))
    except (UsageError, LiteratureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return EXIT_OK


async def _chosen_works(
    params: InsertParams, *, sources: Sources, http: ReferenceHttp
) -> tuple[list[Work], list[str]]:
    """The works the call names, from the library, Zotero or Crossref (by DOI), and
    one line for each DOI that could not be fetched (those are never cited)."""
    if not params.doi and not params.key:
        raise UsageError("name at least one work: --doi DOI or --key KEY")
    pool: list[Work] = []
    if params.key:
        if sources.library is not None:
            pool += read_library(sources.library)
        if sources.zotero is not None:
            pool += await sources.zotero.works()
        if not pool:
            raise UsageError("--key needs --library or --zotero")
    works: list[Work] = []
    for key in params.key:
        match = [w for w in pool if str(w.csl.get("id", "")) == key]
        if not match:
            raise UsageError(f"no item with id {key!r} in the library")
        works.append(match[0])
    failures: list[str] = []
    for doi in params.doi:
        try:
            works.append(await fetch_work(doi, http=http))
        except LiteratureError as exc:
            failures.append(f"{doi}: {exc}")
    return works, failures


async def run_insert(
    params: InsertParams, *, transport: httpx.AsyncBaseTransport | None = None
) -> tuple[int, str]:
    """``(exit code, message)``; a copy is written only when every work verifies."""
    file = params.file
    if not file.is_file() or file.suffix.lower() != ".docx":
        raise UsageError(f"{file} is not a .docx file")
    if params.library is not None and not params.library.is_file():
        raise UsageError(f"--library {params.library} is not a file")
    adapter = DocxAdapter()
    try:
        view = adapter.open(file)
    except DocxReadError as exc:
        raise UsageError(str(exc)) from exc
    paragraph = next((p for p in view.paragraphs if p.id == params.paragraph_id), None)
    if paragraph is None:
        raise UsageError(f"{file} has no paragraph {params.paragraph_id!r}")
    if not 0 <= params.at <= len(paragraph.text):
        raise UsageError(
            f"--at {params.at} is outside the paragraph (0 to {len(paragraph.text)})"
        )
    out = output_path(file, params.out)
    store = ProjectStore(project_dir(out, params.project))
    try:
        project_id = store.init(file, project_id=view.project_id).id
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    async with open_literature(
        library=params.library,
        zotero=params.zotero,
        cache_dir=store.cache_dir / "http",
        mailto=params.mailto,
        transport=transport,
    ) as (sources, http):
        works, failures = await _chosen_works(params, sources=sources, http=http)
        verdicts = [await resolve(work, http=http) for work in works]
    refused = [v for v in verdicts if isinstance(v, Unverified)]
    if refused or failures:
        lines = [f"{v.work.title or v.work.doi}: {v.reason}" for v in refused]
        return EXIT_OPEN, "not cited, nothing written:\n" + "\n".join(
            [*failures, *lines]
        )
    checked = [v for v in verdicts if isinstance(v, VerifiedWork)]
    items = [
        v.work.citation_item(
            locator=params.locator, prefix=params.prefix, suffix=params.suffix
        )
        for v in checked
    ]
    transaction = Transaction(
        ops=[
            InsertCitation(paragraph_id=params.paragraph_id, at=params.at, items=items),
            UpdateBibliography(),
        ],
        author=AUTHOR,
        reason="cite verified works",
        source="insert-citation",
    )
    started = time.monotonic()
    result = write_copy(adapter, view, transaction, out=out, project_id=project_id)
    store.append_event(
        {
            "type": "transaction",
            "source": "insert-citation",
            "author": AUTHOR,
            "ops": len(transaction.ops),
            "applied": result.applied,
            "refused": result.refused,
            "output": str(out),
            "seconds": round(time.monotonic() - started, 2),
        }
    )
    if result.refused:
        out.unlink(missing_ok=True)
        return EXIT_OPEN, "the writer refused, nothing written: " + "; ".join(
            result.refused
        )
    return EXIT_OK, f"cited copy: {out}"


def _handle_insert(params: InsertParams) -> int:
    try:
        code, message = asyncio.run(run_insert(params))
    except (UsageError, LiteratureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    print(message, file=sys.stderr if code else sys.stdout)
    return code


FIND_VERB = Verb(
    name="find-citation",
    summary="find works for a title, topic or claim and verify them; prints JSON "
    "(example: find-citation 'daily salt logging' --library refs.json)",
    params=FindParams,
    handler=_handle_find,
)

INSERT_VERB = Verb(
    name="insert-citation",
    summary="cite verified works at a place in a .docx, as tracked changes in a copy "
    "(example: insert-citation paper.docx --paragraph-id p3-ab12 --at 118 "
    "--doi 10.5555/saltlog.2020.1)",
    params=InsertParams,
    handler=_handle_insert,
)

VERBS: tuple[Verb, ...] = (FIND_VERB, INSERT_VERB)
