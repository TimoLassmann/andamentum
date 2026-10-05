"""The citations check, over recorded fixtures and small mock transports (no network)."""

from __future__ import annotations

import httpx
import pytest

from andamentum.compose.harness.bench.fixture_http import (
    fixture_transport,
    scheme_only_url_check,
)
from andamentum.compose.harness.bench.planted import load_fixtures
from andamentum.compose.harness.checks.citations import CitationsCheck
from andamentum.compose.harness.kernel import CheckContext, select
from andamentum.references import ReferenceHttp

from ._helpers import field, make_view

FIXTURES = load_fixtures()
RETRACTED = "10.5555/am.bench.r001"
VALID = next(
    doi
    for doi, found in FIXTURES.crossref.works.items()
    if found.status == 200 and not found.body["message"].get("updated-by")  # type: ignore[index, union-attr]
)
FABRICATED = "10.5555/am.bench.nowhere"


def _title(doi: str) -> str:
    return FIXTURES.crossref.works[doi].body["message"]["title"][0]  # type: ignore[index]


def _entry(doi: str) -> str:
    work = FIXTURES.crossref.works[doi].body["message"]  # type: ignore[index]
    author = work["author"][0]
    return (
        f"{author['family']}, {author['given']} ({work['issued']['date-parts'][0][0]}). "
        f"{work['title'][0]}. {work['container-title'][0]}, "
        f"{work['volume']}({work['issue']}), {work['page']}. https://doi.org/{doi}"
    )


class Counting:
    """A transport that counts the requests it serves."""

    def __init__(self, inner: httpx.MockTransport) -> None:
        self.inner = inner
        self.urls: list[str] = []

    def transport(self) -> httpx.MockTransport:
        async def handle(request: httpx.Request) -> httpx.Response:
            self.urls.append(str(request.url))
            return await self.inner.handle_async_request(request)

        return httpx.MockTransport(handle)


class MemoryCache:
    def __init__(self) -> None:
        self.items: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self.items.get(key)

    def put(self, key: str, value: bytes) -> None:
        self.items[key] = value


def context(transport: httpx.AsyncBaseTransport) -> CheckContext:
    http = ReferenceHttp(
        client=httpx.AsyncClient(transport=transport),
        cache=MemoryCache(),
        url_check=scheme_only_url_check,
    )
    return CheckContext(store=None, offline=False, http=http)


async def run(view, transport=None):
    ctx = context(transport or fixture_transport(FIXTURES))
    return await CitationsCheck().run(view, ctx)


def test_registered_and_needs_network():
    (check,) = select(ids=["citations"])
    assert check.spec.needs == ["network"]
    assert select(needs=[], ids=None) == [] or "citations" not in [
        c.spec.id for c in select(needs=[])
    ]


async def test_requires_the_http_client():
    ctx = CheckContext(store=None, offline=False)
    with pytest.raises(ValueError, match="HTTP client"):
        await CitationsCheck().run(make_view(("x", None)), ctx)


async def test_valid_entry_has_no_findings():
    view = make_view(("References", 1), (_entry(VALID), None))
    assert await run(view) == []


async def test_retracted_work_is_an_error_at_the_entry():
    view = make_view(("References", 1), (_entry(RETRACTED), None))
    (d,) = await run(view)
    assert d.severity == "error"
    assert d.data["kind"] == "retraction"
    assert d.paragraph_id == "p1"
    assert (d.start, d.end) == (0, len(_entry(RETRACTED)))
    assert "retraction" in d.message
    assert d.data["source"] in {"retraction-watch", "publisher"}


async def test_fabricated_doi_is_a_warning_not_an_error():
    text = (
        "Nobody, A. (2020). A study that was never published anywhere at all. "
        f"Journal of Nothing, 1(1), 1-2. https://doi.org/{FABRICATED}"
    )
    view = make_view(("References", 1), (text, None))
    (d,) = await run(view)
    assert d.severity == "warning"
    assert d.data["kind"] == "unresolved"
    assert d.message.startswith("Could not verify")
    assert d.paragraph_id == "p1"


async def test_zotero_field_is_checked_at_the_field():
    data = {
        "citationItems": [
            {
                "itemData": {
                    "DOI": FABRICATED,
                    "title": "A study that was never published anywhere at all",
                    "author": [{"family": "Nobody", "given": "A."}],
                    "issued": {"date-parts": [[2020]]},
                }
            }
        ]
    }
    view = make_view(
        ("As shown (Nobody, 2020).", None),
        fields=[
            field(
                "citation",
                "zotero",
                data=data,
                display="(Nobody, 2020)",
                pid="p0",
            )
        ],
    )
    (d,) = await run(view)
    assert d.paragraph_id == "p0"
    assert (d.start, d.end) == (0, len("(Nobody, 2020)"))
    assert d.data["kind"] == "unresolved"


async def test_endnote_field_without_doi_resolves_by_title():
    title = _title(VALID)
    work = FIXTURES.crossref.works[VALID].body["message"]  # type: ignore[index]
    data = {
        "cites": [
            {
                "title": title,
                "authors": [
                    f"{work['author'][0]['family']}, {work['author'][0]['given']}"
                ],
                "record_year": str(work["issued"]["date-parts"][0][0]),
                "doi": "",
            }
        ]
    }
    view = make_view(
        ("x", None),
        fields=[field("citation", "endnote", data=data, display="(x)")],
    )
    assert await run(view) == []


async def test_a_work_cited_twice_is_looked_up_once_and_reported_twice():
    data = {
        "citationItems": [{"itemData": {"DOI": RETRACTED, "title": _title(RETRACTED)}}]
    }
    view = make_view(
        ("In text (R, 2019).", None),
        ("References", 1),
        (_entry(RETRACTED), None),
        fields=[field("citation", "zotero", data=data, display="(R, 2019)")],
    )
    counter = Counting(fixture_transport(FIXTURES))
    found = await run(view, counter.transport())
    assert [d.paragraph_id for d in found] == ["p0", "p2"]
    assert all(d.severity == "error" for d in found)
    assert [u for u in counter.urls if RETRACTED in u] == [
        f"https://api.crossref.org/works/{RETRACTED}"
    ]


async def test_unparseable_field_is_left_to_the_crossrefs_check():
    view = make_view(
        ("x", None),
        fields=[
            field(
                "citation",
                "zotero",
                data={"parse_error": "invalid JSON", "raw": "ADDIN ZOTERO_ITEM {"},
            )
        ],
    )
    assert await run(view) == []


def _url_transport(behaviour: dict[str, object]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        outcome = behaviour[str(request.url)]
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, int)
        return httpx.Response(outcome)

    return httpx.MockTransport(handle)


async def test_url_outcomes_have_their_own_wording_and_severity():
    base = "https://site.example/"
    paths = {
        "ok": 200,
        "gone": 404,
        "removed": 410,
        "broken": 503,
        "walled": 403,
        "dns": httpx.ConnectError("nodename nor servname provided, or not known"),
        "slow": httpx.ReadTimeout("timed out"),
    }
    behaviour = {base + k: v for k, v in paths.items()}
    text = " ".join(base + k for k in paths) + "."
    found = await run(make_view((text, None)), _url_transport(behaviour))
    by_status = {d.data["status"]: d for d in found}
    assert "ok" not in by_status
    assert by_status["not_found"].severity == "warning"
    assert (
        "404" in by_status["not_found"].message
        or "410" in by_status["not_found"].message
    )
    assert by_status["server_error"].severity == "warning"
    assert by_status["client_error"].severity == "info"
    assert by_status["dns_failure"].severity == "warning"
    assert "host name" in by_status["dns_failure"].message
    assert by_status["timeout"].severity == "warning"
    assert "in time" in by_status["timeout"].message
    assert len(found) == 6  # the two not-found URLs both report
    for d in found:
        paragraph_text = text
        assert paragraph_text[d.start : d.end].startswith(base)


async def test_doi_links_are_not_url_checked():
    view = make_view(
        ("See https://doi.org/10.5555/x and https://dx.doi.org/10.5555/y.", None)
    )
    counter = Counting(fixture_transport(FIXTURES))
    assert await run(view, counter.transport()) == []
    assert counter.urls == []


async def test_a_redirect_that_ends_well_is_fine():
    url = next(u for u, v in FIXTURES.urls.urls.items() if v.redirects)
    assert await run(make_view((f"Posted at {url}.", None))) == []


async def test_trailing_punctuation_is_not_part_of_the_url():
    url = next(
        u for u, v in FIXTURES.urls.urls.items() if v.status == 404 and not v.redirects
    )
    (d,) = await run(make_view((f"(see {url}).", None)))
    assert d.evidence[0] == url


async def test_a_failing_lookup_is_one_document_warning(monkeypatch):
    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr("andamentum.references.lookup.asyncio.sleep", no_wait)

    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={})

    view = make_view(
        ("References", 1), (_entry(VALID), None), (_entry(RETRACTED), None)
    )
    found = await run(view, httpx.MockTransport(refuse))
    (d,) = found
    assert d.severity == "warning"
    assert d.paragraph_id is None
    assert d.data["kind"] == "lookup_failed"
    assert "2 cited work" in d.message


async def test_no_reference_list_and_no_fields_means_nothing_to_check():
    assert await run(make_view(("Plain text with no citations.", None))) == []


async def test_a_failed_retraction_lookup_is_a_warning_not_a_clean_report(monkeypatch):
    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr("andamentum.references.lookup.asyncio.sleep", no_wait)
    inner = fixture_transport(FIXTURES)

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/10."):
            return httpx.Response(503, json={})
        return await inner.handle_async_request(request)

    # No DOI in the entry: the work resolves by title search, and only then
    # is its record fetched for notices, which is the request that fails.
    entry = _entry(RETRACTED).rsplit(" https://doi.org/", 1)[0]
    view = make_view(("References", 1), (entry, None))
    found = await run(view, httpx.MockTransport(handle))
    kinds = [d.data["kind"] for d in found]
    assert "lookup_failed" in kinds
    (d,) = [d for d in found if d.data["kind"] == "lookup_failed"]
    assert d.severity == "warning"
    assert "Retraction status" in d.message


def _typed(surname: str, year: int) -> object:
    return field(
        "citation",
        data={"works": [{"surname": surname, "year": year}]},
        display=f"{surname} ({year})",
    )


async def test_a_plain_citation_with_no_reference_list_is_a_warning():
    view = make_view(
        ("As Marlowe (2021) showed, logging helps.", None),
        fields=[_typed("Marlowe", 2021)],  # type: ignore[list-item]
    )
    (d,) = await run(view)
    assert d.severity == "warning" and d.data["kind"] == "unlisted"
    assert "no reference list" in d.message and d.paragraph_id == "p0"


async def test_a_plain_citation_is_quiet_when_an_entry_names_its_author():
    message = FIXTURES.crossref.works[VALID].body["message"]  # type: ignore[index, union-attr]
    surname = message["author"][0]["family"]
    year = message["issued"]["date-parts"][0][0]
    view = make_view(
        (f"As {surname} ({year}) showed, logging helps.", None),
        ("References", 1),
        (_entry(VALID), None),
        fields=[_typed(surname, year)],  # type: ignore[list-item]
    )
    assert await run(view) == []


async def test_a_plain_citation_no_entry_names_is_a_warning():
    view = make_view(
        ("As Marlowe (2021) showed, logging helps.", None),
        ("References", 1),
        (_entry(VALID), None),
        fields=[_typed("Marlowe", 2021)],  # type: ignore[list-item]
    )
    (d,) = await run(view)
    assert d.data["kind"] == "unlisted" and "no entry" in d.message
