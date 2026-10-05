"""Retraction lookup, cited-work resolution and URL health.

No network: Crossref answers are recorded JSON (synthetic works) served by a
mock transport; URL probes are served by a mock transport too.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import httpx
import pytest

from andamentum.references import (
    ReferenceHttp,
    lookup_retraction,
    resolve_cited_work,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


class DictCache:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.puts = 0

    def get(self, key: str) -> bytes | None:
        return self.data.get(key)

    def put(self, key: str, value: bytes) -> None:
        self.puts += 1
        self.data[key] = value


def _http(handler, cache=None, mailto=None) -> ReferenceHttp:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ReferenceHttp(client=client, cache=cache, mailto=mailto)


def _crossref_handler(routes: dict[str, dict], seen: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        for path, body in routes.items():
            if request.url.path == path:
                return httpx.Response(200, json=body)
        return httpx.Response(404, json={})

    return handler


# ── retraction ──────────────────────────────────────────────────────────────


async def test_retraction_reads_all_three_kinds_and_ignores_others() -> None:
    handler = _crossref_handler(
        {"/works/10.1000/example.retracted": _fixture("crossref_retracted_work.json")}
    )
    async with _http(handler) as http:
        status = await lookup_retraction("10.1000/example.retracted", http=http)
    assert status.status == "checked"
    assert status.retracted
    assert [n.kind for n in status.notices] == [
        "retraction",
        "correction",
        "expression_of_concern",
    ]
    first = status.notices[0]
    assert first.source == "retraction-watch"
    assert first.notice_doi == "10.1000/example.retraction-notice"
    assert first.date == "2021-03-04"
    assert status.notices[1].source == "publisher"


async def test_clean_work_has_no_notices() -> None:
    handler = _crossref_handler(
        {"/works/10.1000/example.clean": _fixture("crossref_clean_work.json")}
    )
    async with _http(handler) as http:
        status = await lookup_retraction(
            "https://doi.org/10.1000/example.clean", http=http
        )
    assert status.status == "checked" and status.notices == []
    assert not status.retracted and not status.is_notice


async def test_a_notice_doi_is_marked_as_a_notice() -> None:
    handler = _crossref_handler(
        {"/works/10.1000/example.retraction-notice": _fixture("crossref_notice.json")}
    )
    async with _http(handler) as http:
        status = await lookup_retraction("10.1000/example.retraction-notice", http=http)
    assert status.is_notice


async def test_unknown_doi_is_not_found_and_failure_is_service_error() -> None:
    async with _http(_crossref_handler({})) as http:
        assert (
            await lookup_retraction("10.1000/none", http=http)
        ).status == "not_found"

    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    async with _http(refuse) as http:
        status = await lookup_retraction("10.1000/x", http=http)
    assert status.status == "service_error" and "403" in status.detail


async def test_cache_answers_the_second_call_and_mailto_is_sent() -> None:
    seen: list[httpx.Request] = []
    handler = _crossref_handler(
        {"/works/10.1000/example.clean": _fixture("crossref_clean_work.json")}, seen
    )
    cache = DictCache()
    async with _http(handler, cache, mailto="me@example.org") as http:
        await lookup_retraction("10.1000/example.clean", http=http)
        await lookup_retraction("10.1000/example.clean", http=http)
    assert len(seen) == 1 and cache.puts == 1
    assert "mailto=me%40example.org" in str(seen[0].url)
    assert "me@example.org" in seen[0].headers["user-agent"]


async def test_a_service_error_is_not_cached() -> None:
    cache = DictCache()

    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400)

    async with _http(refuse, cache) as http:
        await lookup_retraction("10.1000/x", http=http)
    assert cache.puts == 0


# ── resolver ────────────────────────────────────────────────────────────────


def _search_handler(seen: list):
    return _crossref_handler(
        {
            "/works": _fixture("crossref_search_clean.json"),
            "/works/10.1000/example.clean": _fixture("crossref_clean_work.json"),
        },
        seen,
    )


async def test_resolve_by_title_searches_then_compares() -> None:
    seen: list = []
    async with _http(_search_handler(seen)) as http:
        result = await resolve_cited_work(
            http=http,
            title="Invented methods for sparse sampling",
            authors=["Roe"],
            year=2021,
        )
    assert result.status == "found"
    assert result.match is not None and result.match.doi == "10.1000/example.clean"
    assert seen[0].url.path == "/works"


async def test_resolve_by_title_flags_a_wrong_year() -> None:
    async with _http(_search_handler([])) as http:
        result = await resolve_cited_work(
            http=http,
            title="Invented methods for sparse sampling",
            authors=["Roe"],
            year=2015,
        )
    assert result.status == "details_differ"
    assert "2021" in result.detail


async def test_resolve_unmatched_title_is_not_found() -> None:
    async with _http(_search_handler([])) as http:
        result = await resolve_cited_work(
            http=http,
            title="A paper that does not exist anywhere in the index",
            year=2020,
        )
    assert result.status == "not_found"


async def test_resolve_by_doi_alone_and_with_a_different_title() -> None:
    async with _http(_search_handler([])) as http:
        alone = await resolve_cited_work(http=http, doi="10.1000/example.clean")
        wrong = await resolve_cited_work(
            http=http,
            doi="10.1000/example.clean",
            title="Something quite different entirely",
        )
        missing = await resolve_cited_work(http=http, doi="10.1000/fabricated")
    assert alone.status == "found"
    assert wrong.status == "details_differ" and "different work" in wrong.detail
    assert missing.status == "not_found"


async def test_resolve_with_nothing_is_not_checked() -> None:
    async with _http(_search_handler([])) as http:
        result = await resolve_cited_work(http=http)
    assert result.status == "service_error"


# ── URL health ──────────────────────────────────────────────────────────────

PUBLIC = "https://example.org/page"


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_safe_url resolves names; make every test name a public address."""
    monkeypatch.setattr(
        "andamentum.core.url_safety.is_safe_url",
        lambda url, **_: (True, ""),
    )
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url", lambda url, **_: (True, "")
    )


async def _health(handler, url: str = PUBLIC, cache=None):
    async with _http(handler, cache) as http:
        return await http.check_url(url)


async def test_head_ok() -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(200)

    health = await _health(handler)
    assert health.status == "ok" and methods == ["HEAD"]


async def test_head_refused_falls_back_to_get() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(405 if request.method == "HEAD" else 200)

    assert (await _health(handler)).status == "ok"


async def test_not_found_server_error_and_client_error() -> None:
    for code, status in (
        (404, "not_found"),
        (410, "not_found"),
        (503, "server_error"),
        (403, "client_error"),
    ):
        health = await _health(lambda r, c=code: httpx.Response(c))
        assert (health.status, health.http_status) == (status, code)


async def test_redirect_is_followed_and_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/new"})
        return httpx.Response(200)

    health = await _health(handler, "https://example.org/old")
    assert health.status == "ok" and health.final_url == "https://example.org/new"


async def test_timeout_dns_and_other_errors() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    def dns(request: httpx.Request) -> httpx.Response:
        error = httpx.ConnectError("lookup failed", request=request)
        error.__cause__ = socket.gaierror(8, "nodename nor servname provided")
        raise error

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    assert (await _health(timeout)).status == "timeout"
    assert (await _health(dns)).status == "dns_failure"
    assert (await _health(refused)).status == "error"


async def test_unsafe_url_is_blocked_without_a_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url",
        lambda url, **_: (False, "private address"),
    )
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        return httpx.Response(200)

    health = await _health(handler, "http://10.0.0.1/")
    assert health.status == "blocked" and calls == []


async def test_redirect_into_a_private_address_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url",
        lambda url, **_: ("169.254" not in url, "metadata address"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://169.254.169.254/"})

    assert (await _health(handler)).status == "blocked"


async def test_settled_url_status_is_cached_and_transient_is_not() -> None:
    cache = DictCache()
    calls: list[str] = []

    def ok(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        return httpx.Response(200)

    async with _http(ok, cache) as http:
        await http.check_url(PUBLIC)
        await http.check_url(PUBLIC)
    assert calls == ["HEAD"]

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    transient = DictCache()
    await _health(timeout, cache=transient)
    assert transient.puts == 0


# ── resolve_entry and the URL-safety hook ───────────────────────────────────


async def test_resolve_entry_reads_a_listed_entry_by_its_doi():
    from andamentum.references import ReferenceEntry, resolve_entry

    work = _fixture("crossref_clean_work.json")
    message = work["message"] if "message" in work else work
    title = message["title"][0]
    doi = message["DOI"]
    text = f"Someone, A. (2020). {title}. A Journal, 1(1), 1-2."
    entry = ReferenceEntry(
        number=1, text=text, start=0, end=len(text), doi=doi, years=[2020], year=2020
    )
    seen: list[httpx.Request] = []
    http = _http(_crossref_handler({f"/works/{doi}": work}, seen))
    result = await resolve_entry(entry, http=http, compare_authors=False)
    assert result.status in {"found", "details_differ"}
    assert [r.url.path for r in seen] == [f"/works/{doi}"]


async def test_cited_work_without_authors_does_not_hold_the_author_against_it():
    work = _fixture("crossref_clean_work.json")
    message = work["message"] if "message" in work else work
    http = _http(_crossref_handler({f"/works/{message['DOI']}": work}))
    result = await resolve_cited_work(
        http=http, doi=message["DOI"], title=message["title"][0]
    )
    assert result.status == "found"


async def test_a_custom_url_check_replaces_the_dns_based_one():
    calls: list[str] = []

    def allow(url: str) -> tuple[bool, str]:
        calls.append(url)
        return True, ""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    http = ReferenceHttp(client=client, url_check=allow)
    health = await http.check_url("https://no-such-host.example/page")
    assert health.status == "ok"
    assert calls == ["https://no-such-host.example/page"]
