"""Resolve one cited work from a DOI, or from its title, authors and year.

The entry-text lookup in :mod:`.lookup` takes a printed reference. A caller
that already holds the parts — a Zotero field, a parsed BibTeX entry — has no
printed reference to send. This builds one from the parts and runs the same
two stages: (1) fetch candidates, by DOI or by a bibliographic search; (2)
compare the candidates' title, year and first author with what was cited, with
the same rules (a returned work counts only when its title is in the entry).
"""

from __future__ import annotations

from urllib.parse import quote

from .http import ReferenceHttp
from .lookup import (
    _CANDIDATES,
    _SELECT,
    _judge,
    _to_match,
    title_guess,
    years_in,
)
from .models import LookupResult, ReferenceEntry


async def resolve_cited_work(
    *,
    http: ReferenceHttp,
    doi: str = "",
    title: str = "",
    authors: list[str] | None = None,
    year: int | None = None,
) -> LookupResult:
    """Does the cited work exist, and do its details agree?

    *authors* are family names or full names, first author first. With a DOI
    the record is fetched by it and compared when a title is also given; with
    none of DOI and title there is nothing to resolve, and the result is
    ``service_error`` (not checked) with the reason.
    """
    doi = doi.strip().removeprefix("https://doi.org/").removeprefix("doi:")
    text = _entry_text(title, authors or [], year)
    entry = ReferenceEntry(
        number=None,
        text=text,
        start=0,
        end=len(text),
        doi=doi,
        years=sorted(years_in(text)),
        year=year,
    )
    return await resolve_entry(
        entry,
        http=http,
        title=title,
        compare=bool(title),
        compare_authors=bool(authors),
    )


async def resolve_entry(
    entry: ReferenceEntry,
    *,
    http: ReferenceHttp,
    title: str = "",
    compare: bool = True,
    compare_authors: bool = True,
) -> LookupResult:
    """The same two stages for an entry already read from a reference list.

    *title* is the cited title when the caller knows it; otherwise the entry's
    most title-like part is guessed. With a DOI, *compare* False accepts the
    fetched record without comparing it to the entry's text (the caller had a
    DOI and nothing to compare). *compare_authors* False means the entry names
    no author, so the fetched record's first author is not held against it.
    """
    doi = entry.doi
    text = entry.text
    title = title or title_guess(text)
    if doi:
        result = await http.crossref(f"/works/{quote(doi, safe='/')}")
        if isinstance(result, str):
            return _not_checked(result)
        code, body = result
        if code == 404:
            return LookupResult(
                entry_number=None,
                status="not_found",
                detail=f"the DOI {doi} does not resolve",
            )
        match = _to_match(body.get("message", {}))
        if not compare:
            return LookupResult(entry_number=None, status="found", match=match)
        if not compare_authors:
            entry = entry.model_copy(
                update={"text": f"{match.first_author} {entry.text}"}
            )
        return _judge(entry, [match], by_doi=True)
    if not title:
        return _not_checked("neither a DOI nor a title was given")

    searches: list[dict[str, str]] = [{"query.bibliographic": text[:1000]}]
    if title:
        searches.append({"query.title": title})
    best: LookupResult | None = None
    for search in searches:
        result = await http.crossref(
            "/works", {**search, "rows": str(_CANDIDATES), "select": _SELECT}
        )
        if isinstance(result, str):
            return _not_checked(result)
        items = result[1].get("message", {}).get("items", [])
        judged = _judge(entry, [_to_match(i) for i in items], by_doi=False)
        if judged.status == "found":
            return judged
        if best is None or judged.status == "details_differ":
            best = judged
    assert best is not None
    return best


def _entry_text(title: str, authors: list[str], year: int | None) -> str:
    names = ", ".join(a for a in authors if a)
    return " ".join(
        part for part in (names, f"({year})" if year else "", title) if part
    )


def _not_checked(reason: str) -> LookupResult:
    return LookupResult(entry_number=None, status="service_error", detail=reason)
