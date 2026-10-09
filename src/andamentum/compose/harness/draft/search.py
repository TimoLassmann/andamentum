"""``LiteratureSearch`` over the harness literature layer: find, then verify each work.

The one place ``draft`` meets ``harness.literature``: a query goes to
``find_works`` (library file, Zotero read-only, Crossref), every candidate goes
through ``resolve`` (exists, details agree, not retracted), and the answer
separates verified works from a line per work that could not be verified. A
source that cannot answer raises (``LiteratureError``); it is never an empty
result.

A verified work can back a fact only through its abstract, which comes from the
CSL ``abstract`` field (a library file or Zotero item that has one; JATS tags
are removed). Crossref's search does not return abstracts, so a work found only
there is verified but not quotable and the report lists it as not used.
"""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from pathlib import Path

from andamentum.compose.genres import Genre
from andamentum.compose.research import ResearchSetup, run_research
from andamentum.references import ReferenceHttp
from andamentum.references import Work as ScholarlyWork

from ..literature import Sources, Unverified, Work, find_works, resolve
from .cite import CANDIDATES_PER_SENTENCE, CiteTools, work_from_csl
from .works import (
    CitableWork,
    LiteratureFinding,
    LiteratureSearch,
    ResearchFinding,
    ResearchRun,
)

_TAG = re.compile(r"<[^>]+>")


def abstract_of(csl: dict[str, object]) -> str:
    return " ".join(_TAG.sub(" ", str(csl.get("abstract") or "")).split())


def literature_search(*, sources: Sources, http: ReferenceHttp) -> LiteratureSearch:
    """A search that returns verified works only (see the module docstring)."""

    async def search(query: str, *, limit: int) -> LiteratureFinding:
        finding = LiteratureFinding()
        for work in await find_works(query, sources=sources, limit=limit):
            verdict = await resolve(work, http=http)
            if isinstance(verdict, Unverified):
                year = f" ({verdict.work.year})" if verdict.work.year else ""
                finding.unverified.append(
                    f"{verdict.work.title}{year}: {verdict.reason}"
                )
                continue
            checked = verdict.work
            finding.works.append(
                CitableWork(
                    csl=checked.csl, abstract=abstract_of(checked.csl), uri=checked.uri
                )
            )
        return finding

    return search


def research_literature(
    *,
    setup: ResearchSetup,
    model: str,
    http: ReferenceHttp,
    progress: Callable[[str], None],
) -> ResearchRun:
    """A research run whose cited works pass the same verification as any other.

    The engine retraction-checks the works it reads; here each work that has a
    finding goes through ``resolve`` as well (DOI exists, details agree, not
    retracted), so a work is cited from Word only on the evidence a library or
    Crossref work needs. A work with no DOI cannot be verified that way: it is
    not cited and the notes say so.
    """

    async def run(task: str, *, genre: Genre, database: Path) -> ResearchFinding:
        result = await run_research(
            task,
            genre=genre,
            setup=setup,
            model=model,
            database=database,
            progress=progress,
        )
        finding = ResearchFinding(result=result)
        with_findings = {f.work_id for f in result.findings}
        for work in result.works:
            if work.work_id not in with_findings:
                continue
            if not work.doi:
                finding.notes.append(
                    f"work not used: {work.title or work.url or work.work_id} "
                    "(no DOI, so it cannot be verified or cited in Word)"
                )
                continue
            csl: dict[str, object] = dict(work.to_csl())
            verdict = await resolve(Work(csl=csl, source="crossref"), http=http)
            if isinstance(verdict, Unverified):
                year = f" ({verdict.work.year})" if verdict.work.year else ""
                finding.unverified.append(
                    f"{verdict.work.title}{year}: {verdict.reason}"
                )
                continue
            finding.works[work.work_id] = CitableWork(
                csl=verdict.work.csl, abstract="", uri=None
            )
        return finding

    return run


LIBRARY_PER_SENTENCE = 3  # library or Zotero works a sentence may take, best first


@asynccontextmanager
async def cite_tools(
    *,
    sources: Sources,
    http: ReferenceHttp,
    model: str,
    store: Path,
    cache: Path,
    mailto: str | None,
) -> AsyncIterator[CiteTools]:
    """The cite loop's tools over the real services (see ``cite.py``).

    *store* is the document store the papers read go into: an existing store is
    opened, a path that does not exist becomes a new store. *cache* is a
    research store whose HTTP cache the finder and the reader share. Library
    works come from the user's library file and Zotero (never Crossref, which
    gives no text to read); the finder is ``deep_research.find_works`` with one
    step of citation following; every work is verified by ``resolve``. API keys
    come from the environment (the command loads ``.env``), never from flags.
    """
    from andamentum.deep_research import (
        FINDER_SOURCES,
        READ_SOURCES,
        ResearchStore,
        WorkText,
        read_work,
    )
    from andamentum.deep_research import find_works as find_scholarly
    from andamentum.document_store import Store

    keys = {
        "openalex_api_key": os.environ.get("OPENALEX_API_KEY") or None,
        "semantic_scholar_api_key": os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None,
        "mailto": mailto or os.environ.get("OPENALEX_EMAIL") or None,
    }
    finder_sources = FINDER_SOURCES.model_copy(update=keys)
    read_sources = READ_SOURCES.model_copy(update=keys)
    own = replace(sources, crossref=None)
    has_library = own.library is not None or own.zotero is not None

    async def library(sentence: str) -> list[ScholarlyWork]:
        found = await find_works(sentence, sources=own, limit=LIBRARY_PER_SENTENCE)
        return [work_from_csl(w.csl) for w in found]

    source_notes: list[str] = []

    def note(prefix: str, lines: list[str]) -> None:
        for line in lines:
            full = f"{prefix}{line}"
            if full not in source_notes:
                source_notes.append(full)

    # Both are closed on every path, including when opening the second raises.
    async with AsyncExitStack() as stack:
        http_cache = ResearchStore(cache)
        stack.callback(http_cache.close)
        papers = await Store.open(store)
        stack.push_async_callback(papers.close)

        async def finder(sentence: str, *, paragraph: str) -> list[ScholarlyWork]:
            result = await find_scholarly(
                sentence,
                model=model,
                paragraph=paragraph,
                sources=finder_sources,
                http_cache=http_cache,
            )
            note("search outage: ", result.degraded)
            return result.works[:CANDIDATES_PER_SENTENCE]

        async def read(work: ScholarlyWork, *, query: str) -> WorkText:
            text = await read_work(
                work, query=query, sources=read_sources, http_cache=http_cache
            )
            note("read failed: ", text.failed)
            return text

        async def verify(work: ScholarlyWork) -> CitableWork | str:
            csl: dict[str, object] = dict(work.to_csl())
            verdict = await resolve(Work(csl=csl, source="crossref"), http=http)
            if isinstance(verdict, Unverified):
                year = f" ({verdict.work.year})" if verdict.work.year else ""
                return f"{verdict.work.title}{year}: {verdict.reason}"
            return CitableWork(csl=verdict.work.csl, abstract="", uri=verdict.work.uri)

        yield CiteTools(
            library=library if has_library else None,
            finder=finder,
            read=read,
            verify=verify,
            papers=papers,
            store_path=store,
            source_notes=source_notes,
        )
