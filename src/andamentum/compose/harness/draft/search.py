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

from andamentum.references import ReferenceHttp

from ..literature import Sources, Unverified, find_works, resolve
from .works import CitableWork, LiteratureFinding, LiteratureSearch

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
