"""Crossref works as CSL JSON: one by-DOI fetch, one search, one mapping.

``ReferenceHttp`` is the package's one Crossref client (polite user agent,
the caller's cache). Every by-DOI read in this package goes through
:func:`get_work` so a DOI is normalised in exactly one way; this module also
turns Crossref's records, which are close to CSL JSON, into CSL JSON.
"""

from __future__ import annotations

from urllib.parse import quote

from .errors import ReferencesError
from .http import ReferenceHttp
from .works import normalise_doi

_SELECT = (
    "DOI,title,author,editor,issued,container-title,volume,issue,page,"
    "publisher,type,URL,abstract"
)
_TYPE_MAP = {
    "journal-article": "article-journal",
    "proceedings-article": "paper-conference",
    "book-chapter": "chapter",
    "posted-content": "article",
    "monograph": "book",
    "report": "report",
    "dissertation": "thesis",
    "dataset": "dataset",
}


class CrossrefError(ReferencesError, RuntimeError):
    """Crossref could not answer, or has no usable record; never an empty result."""


async def get_work(doi: str, *, http: ReferenceHttp) -> tuple[int, dict] | str:
    """``(status, json)`` for Crossref's record of *doi* (a 200 or a 404), or
    a one-line failure reason. The one by-DOI request in this package.

    *doi* is normalised with :func:`works.normalise_doi`; text that is not a
    DOI is reported as such and never sent.
    """
    bare = normalise_doi(doi)
    if bare is None:
        return f"{doi!r} is not a DOI"
    return await http.crossref(f"/works/{quote(bare, safe='/')}")


async def fetch_work(doi: str, *, http: ReferenceHttp) -> dict[str, object]:
    """The Crossref record of *doi* as CSL JSON.

    Raises :class:`CrossrefError` when Crossref does not know the DOI, cannot
    be reached, or the record has no title; a DOI that does not resolve is
    never turned into a made-up work.
    """
    bare = normalise_doi(doi) or doi.strip()
    result = await get_work(bare, http=http)
    if isinstance(result, str):
        raise CrossrefError(f"Crossref lookup of {bare} failed: {result}")
    code, body = result
    if code == 404:
        raise CrossrefError(f"Crossref has no work with DOI {bare}")
    message = body.get("message")
    if code != 200 or not isinstance(message, dict):
        raise CrossrefError(f"Crossref answered {code} for DOI {bare}")
    csl = to_csl(message)
    if not csl.get("title"):
        raise CrossrefError(f"the Crossref record of {bare} has no title")
    return csl


async def search_crossref(
    query: str, *, http: ReferenceHttp, limit: int
) -> list[dict[str, object]]:
    """The works Crossref ranks best for *query* (a title, a citation or a
    claim), as CSL JSON; records without a title are dropped.

    The query text is sent; nothing else. A failed request raises
    :class:`CrossrefError` (never an empty list).
    """
    result = await http.crossref(
        "/works",
        {
            "query.bibliographic": " ".join(query.split())[:1000],
            "rows": str(limit),
            "select": _SELECT,
        },
    )
    if isinstance(result, str):
        raise CrossrefError(f"Crossref search failed: {result}")
    code, body = result
    if code != 200:
        raise CrossrefError(f"Crossref search answered {code}")
    items = body.get("message", {}).get("items", [])
    works = [to_csl(item) for item in items if isinstance(item, dict)]
    return [csl for csl in works if csl.get("title")]


def to_csl(item: dict) -> dict[str, object]:
    """One Crossref record as CSL JSON."""
    doi = str(item.get("DOI", "")).strip()
    csl: dict[str, object] = {
        "id": doi or "crossref",
        "type": _TYPE_MAP.get(str(item.get("type", "")), "article-journal"),
    }
    if doi:
        csl["DOI"] = doi
    for source, target in (("title", "title"), ("container-title", "container-title")):
        value = item.get(source)
        text = value[0] if isinstance(value, list) and value else value
        if isinstance(text, str) and text.strip():
            csl[target] = " ".join(text.split())
    for role in ("author", "editor"):
        people = [_person(p) for p in item.get(role) or [] if isinstance(p, dict)]
        people = [p for p in people if p]
        if people:
            csl[role] = people
    issued = item.get("issued")
    if isinstance(issued, dict) and issued.get("date-parts"):
        csl["issued"] = {"date-parts": issued["date-parts"]}
    for name in ("volume", "issue", "page", "publisher", "URL"):
        if item.get(name):
            csl[name] = item[name]
    abstract = item.get("abstract")
    if isinstance(abstract, str) and abstract.strip():
        csl["abstract"] = abstract.strip()  # JATS tags are removed where it is read
    return csl


def _person(entry: dict) -> dict[str, str]:
    if entry.get("family"):
        person = {"family": str(entry["family"])}
        if entry.get("given"):
            person["given"] = str(entry["given"])
        return person
    if entry.get("name"):
        return {"literal": str(entry["name"])}
    return {}
