"""Serve the recorded Crossref and URL fixtures as an ``httpx`` transport.

The offline benchmark and the tests hand this transport to the check verb, so
every Crossref lookup and URL probe is answered from ``bench/fixtures/`` and no
request leaves the machine. A request the fixtures do not cover raises
``FixtureMissing``: a gap in the recording is a failure, never a silent 404.

Crossref routes: ``GET /works/<doi>`` (the ``works`` fixtures; an unknown DOI is
a 404, as Crossref answers) and ``GET /works?query.title=...`` /
``query.bibliographic=...`` (the ``search`` fixtures, keyed by lower-cased
title; a bibliographic query matches the fixture whose title it contains).
Every other URL is looked up in ``urls``: a status, a recorded redirect, or an
error (``dns_failure`` raises a connect error, ``timeout`` a read timeout).
"""

from __future__ import annotations

from urllib.parse import unquote

import httpx

from .planted import Fixtures, load_fixtures

CROSSREF_HOST = "api.crossref.org"


class FixtureMissing(LookupError):
    """The request is not covered by the recorded fixtures."""


def scheme_only_url_check(url: str) -> tuple[bool, str]:
    """The URL-safety predicate for fixture runs: no DNS, http(s) only.

    The recorded hosts are ``.example`` names that cannot resolve, and nothing
    is requested over a real network, so the DNS-based check has nothing to
    protect here.
    """
    if url.lower().startswith(("http://", "https://")):
        return True, ""
    return False, f"unsupported scheme in {url}"


def fixture_transport(fixtures: Fixtures | None = None) -> httpx.MockTransport:
    recorded = fixtures or load_fixtures()

    def handle(request: httpx.Request) -> httpx.Response:
        url = request.url
        if url.host == CROSSREF_HOST:
            return _crossref(request, recorded)
        return _probe(request, recorded)

    return httpx.MockTransport(handle)


def _crossref(request: httpx.Request, recorded: Fixtures) -> httpx.Response:
    path = unquote(request.url.path)
    params = dict(request.url.params)
    if path.startswith("/works/"):
        doi = path[len("/works/") :]
        found = recorded.crossref.works.get(doi)
        if found is None:
            return httpx.Response(404, json={})
        return httpx.Response(found.status, json=found.body)
    if path == "/works":
        if "query.title" in params:
            items = recorded.crossref.search.get(params["query.title"].lower(), [])
        elif "query.bibliographic" in params:
            text = params["query.bibliographic"].lower()
            items = next(
                (
                    candidates
                    for title, candidates in recorded.crossref.search.items()
                    if title in text
                ),
                [],
            )
        elif "filter" in params:
            doi = params["filter"].removeprefix("updates:")
            found = recorded.crossref.updates.get(doi)
            if found is None:
                raise FixtureMissing(f"no updates fixture for {doi}")
            return httpx.Response(found.status, json=found.body)
        else:
            raise FixtureMissing(f"unrecognised Crossref query: {request.url}")
        return httpx.Response(200, json={"status": "ok", "message": {"items": items}})
    raise FixtureMissing(f"unrecognised Crossref path: {request.url}")


def _probe(request: httpx.Request, recorded: Fixtures) -> httpx.Response:
    url = str(request.url)
    entry = recorded.urls.urls.get(url) or recorded.urls.urls.get(url.rstrip("/"))
    if entry is None:
        for owner in recorded.urls.urls.values():  # the far end of a recorded redirect
            if owner.redirects and str(owner.redirects[0]["location"]) == url:
                assert owner.status is not None
                return httpx.Response(owner.status)
        raise FixtureMissing(f"no URL fixture for {url}")
    if entry.error == "dns_failure":
        raise httpx.ConnectError(
            "[Errno 8] nodename nor servname provided, or not known", request=request
        )
    if entry.error == "timeout":
        raise httpx.ReadTimeout("timed out", request=request)
    if entry.error is not None:
        raise FixtureMissing(f"unknown recorded error {entry.error!r} for {url}")
    if entry.redirects:
        hop = entry.redirects[0]
        return httpx.Response(
            int(str(hop["status"])), headers={"location": str(hop["location"])}
        )
    assert entry.status is not None
    return httpx.Response(entry.status)
