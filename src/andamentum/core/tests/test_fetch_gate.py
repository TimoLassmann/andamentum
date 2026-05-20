"""Unit tests for core.fetch_gate.

Tests run against httpx.MockTransport — no live network. Both gate
concerns (robots.txt + paywall tripwire) are exercised end-to-end via
the public ``check_fetch_allowed`` entry point.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from andamentum.core.fetch_gate import (
    PaywallBlocked,
    RobotsBlocked,
    _robots_cache_clear,
    check_fetch_allowed,
    user_agent_for,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache_each_test() -> None:
    """Each test starts with an empty robots.txt cache."""
    _robots_cache_clear()


def _client_with(
    robots_responses: dict[str, tuple[int, str]] | None = None,
) -> httpx.AsyncClient:
    """Build an httpx client that serves canned robots.txt responses.

    Keys are host names ("example.com"); values are (status, body).
    Requests to non-robots URLs raise — the gate must never fetch the
    target URL itself.
    """
    canned = robots_responses or {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/robots.txt":
            raise AssertionError(
                f"fetch_gate must only hit /robots.txt, got {request.url}"
            )
        host = request.url.host
        if host not in canned:
            return httpx.Response(404)
        status, body = canned[host]
        return httpx.Response(status, text=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Paywall tripwire
# ---------------------------------------------------------------------------


async def test_paywalled_host_blocks_without_tdm() -> None:
    async with _client_with() as client:
        with pytest.raises(PaywallBlocked) as exc_info:
            await check_fetch_allowed(
                "https://www.nature.com/articles/foo",
                user_agent="test/1.0",
                client=client,
            )
    assert exc_info.value.host == "www.nature.com"
    assert "nature.com" in str(exc_info.value)


async def test_paywalled_host_allowed_via_tdm() -> None:
    async with _client_with({"www.nature.com": (404, "")}) as client:
        # 404 robots.txt → all paths allowed; tripwire disarmed via TDM.
        await check_fetch_allowed(
            "https://www.nature.com/articles/foo",
            user_agent="test/1.0",
            tdm_allowed_hosts=frozenset({"nature.com"}),
            client=client,
        )


async def test_paywall_tdm_emits_info_log(caplog: pytest.LogCaptureFixture) -> None:
    async with _client_with({"www.cell.com": (404, "")}) as client:
        with caplog.at_level(logging.INFO, logger="andamentum.fetch_gate"):
            await check_fetch_allowed(
                "https://www.cell.com/some/path",
                user_agent="test/1.0",
                tdm_allowed_hosts=frozenset({"cell.com"}),
                client=client,
            )
    info_records = [
        r for r in caplog.records if r.levelno == logging.INFO and "tdm_allowed" in r.message
    ]
    assert info_records, "expected an INFO log line on TDM-allowed fetch"
    assert "cell.com" in info_records[0].message


async def test_paywall_bare_host_match_works() -> None:
    """A paywalled host listed bare (no subdomain) is matched correctly."""
    async with _client_with() as client:
        with pytest.raises(PaywallBlocked):
            await check_fetch_allowed(
                "https://nature.com/article",
                user_agent="test/1.0",
                client=client,
            )


async def test_open_access_host_passes_paywall() -> None:
    """arxiv, biorxiv, etc. are NOT in the paywall seed list."""
    async with _client_with({"arxiv.org": (404, "")}) as client:
        await check_fetch_allowed(
            "https://arxiv.org/abs/2401.00001",
            user_agent="test/1.0",
            client=client,
        )


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------


async def test_robots_disallow_blocks() -> None:
    """robots.txt disallow for our UA prefix blocks the path.

    Python's RobotFileParser strips the UA at the first ``/`` and matches by
    case-insensitive substring, so a rule for ``andamentum-harvest`` applies
    to a real UA like ``andamentum-harvest/0.3.0 (+url)``.
    """
    robots_body = "User-agent: andamentum-harvest\nDisallow: /private/\n"
    async with _client_with({"example.com": (200, robots_body)}) as client:
        with pytest.raises(RobotsBlocked) as exc_info:
            await check_fetch_allowed(
                "https://example.com/private/secret",
                user_agent="andamentum-harvest/0.3.0 (+example)",
                client=client,
            )
    assert exc_info.value.user_agent == "andamentum-harvest/0.3.0 (+example)"
    assert "/private/secret" in exc_info.value.url


async def test_robots_wildcard_disallow_blocks() -> None:
    """A ``User-agent: *`` block matches any caller."""
    robots_body = "User-agent: *\nDisallow: /secret/\n"
    async with _client_with({"example.com": (200, robots_body)}) as client:
        with pytest.raises(RobotsBlocked):
            await check_fetch_allowed(
                "https://example.com/secret/page",
                user_agent="andamentum-harvest/0.3.0",
                client=client,
            )


async def test_robots_allow_permits() -> None:
    robots_body = "User-agent: *\nAllow: /\n"
    async with _client_with({"example.com": (200, robots_body)}) as client:
        await check_fetch_allowed(
            "https://example.com/public/page",
            user_agent="andamentum-harvest/0.3.0",
            client=client,
        )


async def test_robots_missing_treated_as_allow() -> None:
    """404 on /robots.txt → all paths allowed."""
    async with _client_with({"example.com": (404, "")}) as client:
        await check_fetch_allowed(
            "https://example.com/page",
            user_agent="test/1.0",
            client=client,
        )


async def test_robots_fetch_error_treated_as_allow(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Network error on robots.txt → all paths allowed (with WARNING log)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated network error")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with caplog.at_level(logging.WARNING, logger="andamentum.fetch_gate"):
            await check_fetch_allowed(
                "https://flaky-example.com/page",
                user_agent="test/1.0",
                client=client,
            )
    assert any("robots.txt fetch failed" in r.message for r in caplog.records)


async def test_robots_5xx_treated_as_allow(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Server error on robots.txt → all paths allowed (with WARNING log)."""
    async with _client_with({"example.com": (503, "")}) as client:
        with caplog.at_level(logging.WARNING, logger="andamentum.fetch_gate"):
            await check_fetch_allowed(
                "https://example.com/page",
                user_agent="test/1.0",
                client=client,
            )
    assert any("returned HTTP 503" in r.message for r in caplog.records)


async def test_robots_cache_reused_within_host() -> None:
    """Two checks against the same host fetch /robots.txt only once."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if request.url.path == "/robots.txt":
            call_count += 1
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        raise AssertionError("unexpected non-robots fetch")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await check_fetch_allowed(
            "https://example.com/one",
            user_agent="test/1.0",
            client=client,
        )
        await check_fetch_allowed(
            "https://example.com/two",
            user_agent="test/1.0",
            client=client,
        )
    assert call_count == 1, "robots.txt should be fetched once per host"


# ---------------------------------------------------------------------------
# Non-HTTP / edge cases
# ---------------------------------------------------------------------------


async def test_non_http_url_skips_gate() -> None:
    """file:// URLs and similar are not gated (no fetch will happen)."""
    async with _client_with() as client:
        await check_fetch_allowed(
            "file:///tmp/foo.pdf",
            user_agent="test/1.0",
            client=client,
        )


async def test_empty_host_skips_gate() -> None:
    """Malformed URL with no host → no gate; fetch helper will reject anyway."""
    async with _client_with() as client:
        await check_fetch_allowed(
            "http:///no-host",
            user_agent="test/1.0",
            client=client,
        )


# ---------------------------------------------------------------------------
# user_agent_for
# ---------------------------------------------------------------------------


def test_user_agent_for_includes_component_and_version() -> None:
    ua = user_agent_for("harvest")
    assert ua.startswith("andamentum-harvest/")
    assert "github.com" in ua


def test_user_agent_for_uses_andamentum_version() -> None:
    from andamentum import __version__

    ua = user_agent_for("research")
    assert __version__ in ua
