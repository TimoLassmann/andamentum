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

import re
from collections.abc import Callable
from pathlib import Path

from andamentum.compose.genres import Genre
from andamentum.compose.research import ResearchSetup, run_research_v2
from andamentum.references import ReferenceHttp

from ..literature import Sources, Unverified, Work, find_works, resolve
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
        result = await run_research_v2(
            task,
            genre=genre,
            setup=setup,
            model=model,
            database=database,
            progress=progress,
        )
        finding = ResearchFinding(result=result, decision_model=setup.decision_model)
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
