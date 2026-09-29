"""Look each reference up in Crossref, and compare what comes back.

Crossref's ``query.bibliographic`` search takes a reference exactly as a paper
prints it and returns the works that best match, so the entry never has to be
parsed into fields — which is what makes one check work across citation styles.
An entry that carries a DOI is resolved by the DOI instead.

Crossref ALWAYS returns something: an invented reference comes back with the
nearest real work, an unrelated one. So a returned work counts as a match only
when its title appears in the entry (fuzzily — a typo or a dropped subtitle
still matches). Crossref's own relevance score is relative to the query and is
not used as a threshold.

What is sent: the text of each reference entry, and nothing else. Never the
manuscript's own text.

Every network failure is a ``service_error`` result: the entry was not checked,
and the report counts it as such. It is never turned into a claim about the
reference.
"""

from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from urllib.parse import quote

import httpx
from rapidfuzz import fuzz

from .models import LookupResult, Match, ReferenceEntry

logger = logging.getLogger("andamentum.references")

CROSSREF_API = "https://api.crossref.org"
_SELECT = "DOI,title,author,issued,published,published-print,published-online,type,volume,page"
_CANDIDATES = 10
_DATE_FIELDS = ("issued", "published", "published-print", "published-online")
_TITLE_MATCH = 90  # rapidfuzz partial_ratio, 0-100
_SHORT_TITLE = 15  # characters; below this, only an exact phrase counts
_RANK = {"not_found": 0, "details_differ": 1, "found": 2, "service_error": -1}
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
# A later repost or an earlier preprint of the same work carries a different
# year without contradicting the entry. Only these kinds are the version of
# record, whose year the entry should give.
_VERSION_OF_RECORD = frozenset(
    {"journal-article", "proceedings-article", "book-chapter", "book", "monograph"}
)


async def lookup_entries(
    entries: list[ReferenceEntry],
    *,
    client: httpx.AsyncClient | None = None,
    mailto: str | None = None,
    pause_seconds: float = 0.2,
) -> list[LookupResult]:
    """Look every entry up, one at a time.

    Sequential on purpose: Crossref asks public clients to be gentle, and a
    reference list is at most a few hundred calls. *mailto* joins Crossref's
    "polite pool" (faster, fewer refusals); it is sent as a query parameter
    and in the User-Agent, as Crossref asks.
    """
    http = client or httpx.AsyncClient(timeout=30.0)
    try:
        results = []
        for index, entry in enumerate(entries):
            if index and pause_seconds:
                await asyncio.sleep(pause_seconds)
            results.append(await lookup_entry(entry, client=http, mailto=mailto))
        return results
    finally:
        if client is None:
            await http.aclose()


async def lookup_entry(
    entry: ReferenceEntry, *, client: httpx.AsyncClient, mailto: str | None = None
) -> LookupResult:
    """One entry: by DOI when it has one, by its text otherwise."""
    headers = {"User-Agent": _user_agent(mailto)}
    params: dict[str, str] = {"mailto": mailto} if mailto else {}
    if entry.doi:
        response = await _get(
            client,
            f"{CROSSREF_API}/works/{quote(entry.doi, safe='/')}",
            params=params,
            headers=headers,
        )
        if isinstance(response, str):
            return LookupResult(
                entry_number=entry.number, status="service_error", detail=response
            )
        if response.status_code == 404:
            return LookupResult(
                entry_number=entry.number,
                status="not_found",
                detail=f"the DOI {entry.doi} does not resolve",
            )
        candidates = [_to_match(response.json().get("message", {}))]
        return _judge(entry, candidates, by_doi=True)

    # Crossref's ranking is uneven: a consortium author or a generic title can
    # push the right work out of the first page for the whole-entry search, and
    # a field search by title alone finds it first. So up to three searches,
    # stopping at the first that matches; "not found" means all three failed.
    # Measured on one manuscript's 51 entries: the whole-entry search alone
    # missed three papers in Nature and Science that are in Crossref.
    searches: list[dict[str, str]] = [{"query.bibliographic": entry.text[:1000]}]
    title = title_guess(entry.text)
    if title:
        searches.append({"query.title": title})
    years = sorted(years_in(entry.text))
    if years:
        searches.append(
            {
                "query.bibliographic": entry.text[:1000],
                "filter": f"from-pub-date:{years[0] - 1},until-pub-date:{years[-1] + 1}",
            }
        )
    # Keep searching while the best so far is only partial: a first page that
    # holds a same-titled work with another author is not proof that the
    # cited work is missing — a later search can still find it exactly.
    best: LookupResult | None = None
    for search in searches:
        response = await _get(
            client,
            f"{CROSSREF_API}/works",
            params={**params, **search, "rows": str(_CANDIDATES), "select": _SELECT},
            headers=headers,
        )
        if isinstance(response, str):
            return LookupResult(
                entry_number=entry.number, status="service_error", detail=response
            )
        items = response.json().get("message", {}).get("items", [])
        result = _judge(entry, [_to_match(item) for item in items], by_doi=False)
        if result.status == "found":
            return result
        if best is None or _RANK[result.status] > _RANK[best.status]:
            best = result
    assert best is not None
    return best


def _user_agent(mailto: str | None) -> str:
    base = "andamentum-references"
    return f"{base}; mailto:{mailto}" if mailto else base


async def _get(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, str],
    headers: dict[str, str],
) -> httpx.Response | str:
    """The response, or a one-line reason the call failed.

    One retry, after the wait the service asks for (capped), for a refusal or
    a server error. A 404 is an answer, not a failure, and is returned.
    """
    reason = "not attempted"
    for attempt in range(2):
        wait = 2.0
        try:
            response = await client.get(url, params=params, headers=headers)
        except httpx.HTTPError as exc:
            reason = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 404 or response.is_success:
                return response
            reason = f"HTTP {response.status_code}"
            if response.status_code not in _RETRY_STATUSES:
                break
            wait = _retry_after(response)
        if attempt == 0:
            await asyncio.sleep(wait)
    logger.warning("[references] lookup failed: %s (%s)", reason, url)
    return reason


def _retry_after(response: httpx.Response) -> float:
    try:
        return min(float(response.headers.get("Retry-After", "2")), 10.0)
    except ValueError:
        return 2.0


def _year_of(item: dict, field: str) -> int | None:
    parts = (item.get(field) or {}).get("date-parts") or [[None]]
    year = parts[0][0] if parts and parts[0] else None
    return year if isinstance(year, int) else None


def _to_match(item: dict) -> Match:
    titles = item.get("title") or []
    authors = item.get("author") or []
    years = [y for y in (_year_of(item, f) for f in _DATE_FIELDS) if y is not None]
    first = authors[0] if authors else {}
    return Match(
        title=" ".join(str(titles[0]).split()) if titles else "",
        year=_year_of(item, "issued"),
        years=sorted(set(years)),
        # Only a person's family name. A consortium author ("The 1000 Genomes
        # Project Consortium") is often cited by its first named member, so
        # comparing it with the entry would report a difference that is not
        # one; with no family name the author is simply not compared.
        first_author=str(first.get("family") or ""),
        doi=str(item.get("DOI", "")),
        kind=str(item.get("type", "")),
        volume=str(item.get("volume") or ""),
        first_page=re.split(r"[-–]", str(item.get("page") or ""))[0],
    )


# ── comparing a match with the entry (deterministic) ────────────────────────


def normalise(text: str) -> str:
    """Lower case, accents removed, punctuation to spaces."""
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", plain.lower()).split())


def title_in_entry(title: str, entry_text: str) -> bool:
    """Whether *title* is (nearly) written in the entry."""
    t, e = normalise(title), normalise(entry_text)
    if not t:
        return False
    if len(t) < _SHORT_TITLE:
        return f" {t} " in f" {e} "
    return fuzz.partial_ratio(t, e) >= _TITLE_MATCH


def years_in(entry_text: str) -> set[int]:
    return {int(y) for y in re.findall(r"(?<!\d)(1[89]\d\d|20\d\d)(?!\d)", entry_text)}


def title_guess(entry_text: str) -> str:
    """The entry's most title-like part, for a search by title alone.

    The longest piece, split at sentence ends and at a bracketed year, that
    has four or more words and does not look like an author list (commas
    between initials). Right for the common "Authors (year) Title. Journal"
    and "Authors. Title. Journal. Year" styles; when it guesses wrong the
    search simply finds nothing that matches, and the other searches stand.
    """
    parts = [
        p.strip()
        for p in re.split(r"(?<=[.?!])\s+|\(\d{4}[a-z]?\)\s*", entry_text)
        if p.strip()
    ]

    def author_list(part: str) -> bool:
        return part.count(",") >= 2 and len(re.findall(r"\b[A-Z]{1,3}\b", part)) >= 2

    candidates = [p for p in parts if len(p.split()) >= 4 and not author_list(p)]
    return max(candidates, key=len).rstrip(".") if candidates else ""


def author_in_entry(family: str, entry_text: str) -> bool:
    name = normalise(family)
    return not name or f" {name} " in f" {normalise(entry_text)} "


def _year_agrees(match: Match, years: set[int]) -> bool:
    return not years or bool(
        years & set(match.years or ([match.year] if match.year else []))
    )


def _numbers_agree(match: Match, entry_text: str) -> bool:
    """Whether the record's volume or first page appears in the entry.

    The one detail every citation style prints the same way — journal names
    are abbreviated differently in every style, numbers are not. Used to tell
    a real author error from a same-titled work.
    """
    numbers = set(re.findall(r"\d+", entry_text))
    return any(n and n in numbers for n in (match.volume, match.first_page))


def _namesake(match: Match, *, entry_text: str, years: set[int]) -> bool:
    """A work with the entry's title that is plainly another work.

    Generic titles repeat ("Malignant salivary gland tumors" is a book
    chapter, a review and an encyclopaedia entry). When a record agrees on the
    title but its year AND its first author both disagree with the entry, it
    is a different work with the same title — reporting it as "details
    differ" would tell the author to change a correct entry.
    """
    if not match.first_author or not years or not (match.years or match.year):
        return False
    return not _year_agrees(match, years) and not author_in_entry(
        match.first_author, entry_text
    )


def _judge(
    entry: ReferenceEntry, candidates: list[Match], *, by_doi: bool
) -> LookupResult:
    number = entry.number
    years = years_in(entry.text)
    matched = [
        c
        for c in candidates
        if c.title
        and title_in_entry(c.title, entry.text)
        and not _namesake(c, entry_text=entry.text, years=years)
    ]
    if not matched:
        if by_doi and candidates and candidates[0].title:
            return LookupResult(
                entry_number=number,
                status="details_differ",
                match=candidates[0],
                detail=f"the DOI points to a different work: “{candidates[0].title}”",
            )
        return LookupResult(entry_number=number, status="not_found")

    same_year = [c for c in matched if _year_agrees(c, years)]
    if same_year:
        # Prefer the candidate whose first author also agrees.
        best = next(
            (c for c in same_year if author_in_entry(c.first_author, entry.text)),
            same_year[0],
        )
        author_differs = best.first_author and not author_in_entry(
            best.first_author, entry.text
        )
        if author_differs and not (by_doi or _numbers_agree(best, entry.text)):
            # Same title, same year, another author, and nothing else that
            # ties the record to the entry: a namesake (measured: "Squamous
            # cell carcinoma of the conjunctiva", 2016, twice, by different
            # authors in different journals). Not a match.
            return LookupResult(entry_number=number, status="not_found")
        if author_differs:
            return LookupResult(
                entry_number=number,
                status="details_differ",
                match=best,
                detail=f"the first author is listed as {best.first_author}",
            )
        return LookupResult(entry_number=number, status="found", match=best)

    of_record = [c for c in matched if c.kind in _VERSION_OF_RECORD and c.years]
    if of_record:
        best = of_record[0]
        given = " or ".join(str(y) for y in best.years)
        return LookupResult(
            entry_number=number,
            status="details_differ",
            match=best,
            detail=f"the published record gives the year as {given}",
        )
    # Only reposts or preprints matched: their year says nothing about the
    # entry's, so the work is found and the year left unjudged.
    return LookupResult(entry_number=number, status="found", match=matched[0])
