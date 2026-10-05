"""Crossref search, through ``andamentum.references``' HTTP client.

``ReferenceHttp`` is the package's one Crossref client (polite user agent,
the project cache). This module only builds the search and turns Crossref's
records, which are close to CSL JSON, into CSL JSON.
"""

from __future__ import annotations

from urllib.parse import quote

from andamentum.references import ReferenceHttp

from .models import LiteratureError, Work

_SELECT = (
    "DOI,title,author,editor,issued,container-title,volume,issue,page,"
    "publisher,type,URL,abstract"
)
_TYPES = {
    "journal-article": "article-journal",
    "proceedings-article": "paper-conference",
    "book-chapter": "chapter",
    "posted-content": "article",
    "monograph": "book",
    "report": "report",
    "dissertation": "thesis",
    "dataset": "dataset",
}


async def search_crossref(query: str, *, http: ReferenceHttp, limit: int) -> list[Work]:
    """The works Crossref ranks best for *query* (a title, a citation or a claim).

    The query text is sent; nothing else. A failed request raises
    ``LiteratureError`` (never an empty list).
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
        raise LiteratureError(f"Crossref search failed: {result}")
    code, body = result
    if code != 200:
        raise LiteratureError(f"Crossref search answered {code}")
    items = body.get("message", {}).get("items", [])
    works = [to_csl(item) for item in items if isinstance(item, dict)]
    return [Work(csl=csl, source="crossref") for csl in works if csl.get("title")]


async def fetch_work(doi: str, *, http: ReferenceHttp) -> Work:
    """The Crossref record of *doi* as a work (it is not yet verified).

    Raises ``LiteratureError`` when Crossref does not know the DOI or cannot be
    reached; a DOI that does not resolve is never turned into a made-up work.
    """
    bare = doi.strip().removeprefix("https://doi.org/").removeprefix("doi:")
    result = await http.crossref(f"/works/{quote(bare, safe='/')}")
    if isinstance(result, str):
        raise LiteratureError(f"Crossref lookup of {bare} failed: {result}")
    code, body = result
    if code == 404:
        raise LiteratureError(f"Crossref has no work with DOI {bare}")
    message = body.get("message")
    if code != 200 or not isinstance(message, dict):
        raise LiteratureError(f"Crossref answered {code} for DOI {bare}")
    csl = to_csl(message)
    if not csl.get("title"):
        raise LiteratureError(f"the Crossref record of {bare} has no title")
    return Work(csl=csl, source="crossref")


def to_csl(item: dict) -> dict[str, object]:
    """One Crossref record as CSL JSON."""
    doi = str(item.get("DOI", "")).strip()
    csl: dict[str, object] = {
        "id": doi or "crossref",
        "type": _TYPES.get(str(item.get("type", "")), "article-journal"),
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
