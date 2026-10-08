"""SourceClient: cache, spacing, retry, rate headers, documents, errors."""

from __future__ import annotations

import httpx
import pytest

from andamentum.core import fetch_gate, url_safety
from andamentum.deep_research.sources.base import (
    SourceClient,
    SourceConfig,
    SourceError,
    resolve_client,
)
from andamentum.deep_research.store import TTL_METADATA_S, ResearchStore
from andamentum.references import LICENCE_NO_REDISTRIBUTE

from .source_stub import SourceKit

BODY = b'{"answer": 42}'


def json_ok(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=BODY)


@pytest.fixture(autouse=True)
def _gate_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Documents: public hosts without DNS, and a fresh robots.txt cache."""
    fetch_gate._robots_cache_clear()
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))


async def test_a_second_identical_get_is_served_from_the_cache() -> None:
    kit = SourceKit(json_ok)
    first = await kit.client.get_json(
        "openalex", "https://api.example/x", params={"q": "a"}, ttl_s=TTL_METADATA_S
    )
    second = await kit.client.get_json(
        "openalex", "https://api.example/x", params={"q": "a"}, ttl_s=TTL_METADATA_S
    )
    assert first == second == {"answer": 42}
    assert len(kit.requests) == 1
    assert kit.store.cache_hits == 1


async def test_auth_params_and_headers_are_sent_but_never_keyed_or_stored() -> None:
    kit = SourceKit(json_ok)
    await kit.client.get(
        "openalex",
        "https://api.example/x",
        params={"q": "a"},
        auth_params={"api_key": "SECRET"},
        headers={"x-api-key": "SECRET2"},
        ttl_s=TTL_METADATA_S,
    )
    assert kit.param(0, "api_key") == "SECRET"
    assert kit.requests[0].headers["x-api-key"] == "SECRET2"
    # a caller with another key is served the same answer
    again = await kit.client.get(
        "openalex",
        "https://api.example/x",
        params={"q": "a"},
        auth_params={"api_key": "OTHER"},
        ttl_s=TTL_METADATA_S,
    )
    assert again.from_cache
    stored = list(kit.store.iter_cache())
    assert len(stored) == 1
    assert "SECRET" not in stored[0].url
    assert stored[0].url == "https://api.example/x?q=a"


async def test_a_validated_rejection_is_not_cached() -> None:
    kit = SourceKit(json_ok)

    def refuse(_body: bytes) -> None:
        raise SourceError("openalex", "an error body in a 200", kind="http")

    with pytest.raises(SourceError, match="error body"):
        await kit.client.get(
            "openalex", "https://api.example/x", ttl_s=TTL_METADATA_S, validate=refuse
        )
    assert list(kit.store.iter_cache()) == []


async def test_a_404_raises_with_its_status_and_is_not_cached() -> None:
    kit = SourceKit(lambda _r: httpx.Response(404, text="no such work"))
    with pytest.raises(SourceError) as caught:
        await kit.client.get("openalex", "https://api.example/x", ttl_s=TTL_METADATA_S)
    assert caught.value.status == 404
    assert caught.value.kind == "http"
    assert "no such work" in caught.value.reason
    assert list(kit.store.iter_cache()) == []
    assert len(kit.requests) == 1  # a 404 is not retried


async def test_a_429_is_retried_after_the_wait_the_server_names() -> None:
    answers = [
        httpx.Response(429, headers={"Retry-After": "2"}),
        httpx.Response(200, content=BODY),
    ]
    kit = SourceKit(lambda _r: answers.pop(0))
    result = await kit.client.get(
        "openalex", "https://api.example/x", ttl_s=TTL_METADATA_S
    )
    assert result.status == 200
    assert kit.time.sleeps == [2.0]
    assert len(kit.requests) == 2


async def test_a_server_error_that_persists_raises_after_the_retries() -> None:
    kit = SourceKit(lambda _r: httpx.Response(503, text="down"), retries=2)
    with pytest.raises(SourceError) as caught:
        await kit.client.get("arxiv", "https://api.example/x", ttl_s=TTL_METADATA_S)
    assert caught.value.status == 503
    assert len(kit.requests) == 3
    assert kit.time.sleeps == [1.0, 2.0]  # exponential backoff


async def test_a_retry_after_longer_than_the_cap_raises_at_once() -> None:
    kit = SourceKit(lambda _r: httpx.Response(429, headers={"Retry-After": "3600"}))
    with pytest.raises(SourceError) as caught:
        await kit.client.get(
            "semantic_scholar", "https://api.example/x", ttl_s=TTL_METADATA_S
        )
    assert caught.value.kind == "rate_limited"
    assert caught.value.retry_after_s == 3600
    assert len(kit.requests) == 1
    assert kit.time.sleeps == []


async def test_a_transport_failure_is_a_transport_error_after_the_retries() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    kit = SourceKit(boom, retries=1)
    with pytest.raises(SourceError) as caught:
        await kit.client.get("openalex", "https://api.example/x", ttl_s=TTL_METADATA_S)
    assert caught.value.kind == "transport"
    assert "ConnectTimeout" in caught.value.reason
    assert len(kit.requests) == 2


async def test_a_body_that_is_not_json_is_a_parse_error() -> None:
    kit = SourceKit(lambda _r: httpx.Response(200, text="<html>"))
    with pytest.raises(SourceError) as caught:
        await kit.client.get_json(
            "openalex", "https://api.example/x", ttl_s=TTL_METADATA_S
        )
    assert caught.value.kind == "parse"


async def test_requests_to_one_source_are_spaced_but_cached_ones_are_not() -> None:
    kit = SourceKit(json_ok, intervals={"arxiv": 3.0})
    await kit.client.get("arxiv", "https://api.example/x", params={"q": "1"}, ttl_s=60)
    assert kit.time.sleeps == []  # the first request waits for nothing
    await kit.client.get("arxiv", "https://api.example/x", params={"q": "2"}, ttl_s=60)
    assert kit.time.sleeps == [pytest.approx(3.0)]
    await kit.client.get("arxiv", "https://api.example/x", params={"q": "2"}, ttl_s=60)
    assert kit.time.sleeps == [pytest.approx(3.0)]  # from the cache: no wait


async def test_rate_and_cost_headers_are_read() -> None:
    kit = SourceKit(
        lambda _r: httpx.Response(
            200,
            content=BODY,
            headers={
                "x-ratelimit-remaining": "990",
                "x-ratelimit-limit": "1000",
                "x-ratelimit-reset": "36150",
                "x-ratelimit-cost-usd": "0.001",
                "x-ratelimit-credits-used": "10",
            },
        )
    )
    await kit.client.get(
        "openalex", "https://api.example/a", params={"q": "1"}, ttl_s=60
    )
    await kit.client.get(
        "openalex", "https://api.example/a", params={"q": "2"}, ttl_s=60
    )
    info = kit.client.rate_info["openalex"]
    assert (info.remaining, info.limit, info.reset_s, info.credits_used) == (
        990,
        1000,
        36150,
        10,
    )
    assert kit.client.spent_usd["openalex"] == pytest.approx(0.002)
    assert kit.client.requests["openalex"] == 2


async def test_a_spent_budget_is_not_asked_again_until_it_resets() -> None:
    kit = SourceKit(
        lambda _r: httpx.Response(
            200,
            content=BODY,
            headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "36000"},
        )
    )
    await kit.client.get(
        "openalex", "https://api.example/a", params={"q": "1"}, ttl_s=60
    )
    with pytest.raises(SourceError) as caught:
        await kit.client.get(
            "openalex", "https://api.example/a", params={"q": "2"}, ttl_s=60
        )
    assert caught.value.kind == "rate_limited"
    assert "API key" in caught.value.reason
    assert len(kit.requests) == 1  # the second call never left
    kit.time.now += 36001
    assert (
        await kit.client.get(
            "openalex", "https://api.example/a", params={"q": "2"}, ttl_s=60
        )
    ).status == 200


async def test_a_no_redistribute_answer_is_usable_but_never_exported() -> None:
    kit = SourceKit(json_ok)
    await kit.client.get(
        "semantic_scholar",
        "https://api.example/x",
        ttl_s=60,
        licence=LICENCE_NO_REDISTRIBUTE,
    )
    assert list(kit.store.iter_cache()) == []
    assert len(list(kit.store.iter_cache(exportable_only=False))) == 1


async def test_paced_holds_the_turn_and_spaces_the_next_call() -> None:
    kit = SourceKit(json_ok, intervals={"crossref": 1.0})
    async with kit.client.paced("crossref"):
        pass
    async with kit.client.paced("crossref"):
        pass
    assert kit.time.sleeps == [pytest.approx(1.0)]
    assert kit.client.requests["crossref"] == 2


# -- documents ---------------------------------------------------------------

PAGE = (
    "Sparse sampling recovered the lattice constant in 94 of 100 trials. " * 6
).encode()


def page(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=PAGE, headers={"content-type": "text/plain"})


async def test_a_document_is_fetched_extracted_and_cached_as_text() -> None:
    kit = SourceKit(page)
    first = await kit.client.fetch_document("web", "https://pages.example/paper")
    second = await kit.client.fetch_document("web", "https://pages.example/paper")
    assert first.text == second.text == PAGE.decode()
    assert first.access == "full_text"
    paper_requests = [r for r in kit.requests if r.url.path == "/paper"]
    assert len(paper_requests) == 1
    assert kit.client.fetched_bytes == len(PAGE)


async def test_a_page_with_almost_no_text_is_an_error_not_a_paper() -> None:
    kit = SourceKit(
        lambda _r: httpx.Response(
            200, content=b"Loading...", headers={"content-type": "text/plain"}
        )
    )
    with pytest.raises(SourceError) as caught:
        await kit.client.fetch_document("web", "https://pages.example/stub")
    assert caught.value.kind == "parse"
    assert list(kit.store.iter_cache()) == []


async def test_a_paywalled_publisher_is_refused_unless_a_licence_is_attested() -> None:
    kit = SourceKit(page)
    url = "https://www.nature.com/articles/x1"
    with pytest.raises(SourceError) as caught:
        await kit.client.fetch_document("openalex", url)
    assert caught.value.kind == "refused"
    assert [r for r in kit.requests if r.url.path == "/articles/x1"] == []
    allowed = await kit.client.fetch_document(
        "openalex", url, tdm_allowed_hosts=frozenset({"nature.com"})
    )
    assert allowed.text


async def test_robots_txt_can_refuse_a_fetch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private\n")
        return page(request)

    kit = SourceKit(handler)
    kit.client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(SourceError) as caught:
        await kit.client.fetch_document("web", "https://pages.example/private/p1")
    assert caught.value.kind == "refused"
    assert "robots.txt" in caught.value.reason


async def test_an_unsafe_address_is_refused_on_every_hop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()  # the real SSRF check
    fetch_gate._robots_cache_clear()
    kit = SourceKit(page)
    with pytest.raises(SourceError) as caught:
        await kit.client.fetch_document("web", "http://127.0.0.1/secret")
    assert caught.value.kind == "refused"
    assert [r for r in kit.requests if r.url.path == "/secret"] == []


async def test_a_redirect_into_a_private_address_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def only_public(url: str, **_kw: object) -> tuple[bool, str]:
        return ("169.254.169.254" not in url, "metadata endpoint")

    monkeypatch.setattr(url_safety, "is_safe_url", only_public)
    kit = SourceKit(
        lambda _r: httpx.Response(
            302, headers={"location": "http://169.254.169.254/latest"}
        )
    )
    with pytest.raises(SourceError) as caught:
        await kit.client.fetch_document("web", "https://pages.example/redirect")
    assert caught.value.kind == "refused"


async def test_an_http_error_on_a_document_carries_its_status() -> None:
    kit = SourceKit(lambda _r: httpx.Response(404, text="gone"))
    with pytest.raises(SourceError) as caught:
        await kit.client.fetch_document("web", "https://pages.example/missing")
    assert caught.value.status == 404


async def test_an_oversized_document_is_refused() -> None:
    kit = SourceKit(page)
    with pytest.raises(SourceError) as caught:
        await kit.client.fetch_document(
            "web", "https://pages.example/big", max_bytes=50
        )
    assert caught.value.kind == "refused"
    assert "cap" in caught.value.reason


async def test_fetches_from_one_host_are_spaced() -> None:
    kit = SourceKit(page)
    await kit.client.fetch_document("web", "https://pages.example/a")
    await kit.client.fetch_document("web", "https://pages.example/b")
    assert kit.time.sleeps == [pytest.approx(0.5)]


# -- wiring ------------------------------------------------------------------


def test_an_adapter_needs_a_store_or_a_client() -> None:
    with pytest.raises(SourceError) as caught:
        resolve_client("openalex", store=None, client=None)
    assert caught.value.kind == "config"


def test_adapters_built_from_one_store_share_one_client() -> None:
    store = ResearchStore.in_memory()
    assert resolve_client("a", store=store, client=None) is SourceClient.for_store(
        store
    )
    assert SourceClient.for_store(store) is not SourceClient.for_store(
        ResearchStore.in_memory()
    )


def test_source_config_defaults_search_the_free_scholarly_sources_by_keyword() -> None:
    config = SourceConfig()
    assert config.scholarly == ("openalex", "europepmc", "arxiv")
    assert config.openalex_search == "keyword"
    assert config.searxng_url is None
    assert config.arxiv_pdf_interval_s == 15.0
