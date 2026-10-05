"""One HTTP client for the lookups that need the network.

``ReferenceHttp`` wraps an ``httpx.AsyncClient`` the caller may inject (a test
supplies a mock transport; a harness supplies its own) and an optional cache.
Crossref calls and URL probes both go through it, so a project that passes its
cache gets every answer remembered once.

The cache is a two-method protocol on bytes, so any store fits: a dict
wrapper, a file directory, a project cache. Only answers that will not change
within a session are cached (a Crossref 200 or 404, a settled URL status);
a timeout or a server error is never remembered, because asking again may
succeed.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Callable
from typing import Literal, Protocol
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel

from andamentum.core.url_safety import SsrfBlocked, is_safe_url

from .lookup import CROSSREF_API, _get, _user_agent


def _safe_url(url: str) -> tuple[bool, str]:
    """``is_safe_url``, looked up when called (not when a client is built)."""
    return is_safe_url(url)


class Cache(Protocol):
    """Bytes in, bytes out. ``get`` returns None on a miss."""

    def get(self, key: str) -> bytes | None: ...

    def put(self, key: str, value: bytes) -> None: ...


UrlStatus = Literal[
    "ok",  # a response below 400 (a redirect chain that ends in one)
    "not_found",  # 404 or 410
    "client_error",  # another 4xx: often a bot block (401, 403, 429), not a dead link
    "server_error",  # 5xx
    "timeout",
    "dns_failure",  # the host name does not resolve
    "blocked",  # refused by the SSRF check (private address, bad scheme)
    "error",  # any other failure to connect or read
]


class UrlHealth(BaseModel):
    url: str
    status: UrlStatus
    http_status: int | None = None
    final_url: str = ""  # where the redirects ended, when they moved
    detail: str = ""


_SETTLED: frozenset[str] = frozenset({"ok", "not_found", "client_error"})
_HEAD_NOT_TRUSTED = 400  # HEAD at or above this is retried as GET
_REDIRECTS = frozenset({301, 302, 303, 307, 308})


class ReferenceHttp:
    """Crossref GETs and URL probes over one client and one cache."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        cache: Cache | None = None,
        mailto: str | None = None,
        timeout: float = 20.0,
        url_check: Callable[[str], tuple[bool, str]] | None = None,
    ) -> None:
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=timeout)
        self.cache = cache
        self.mailto = mailto
        self.url_check = url_check or _safe_url

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def __aenter__(self) -> ReferenceHttp:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ── Crossref ────────────────────────────────────────────────────────────

    async def crossref(
        self, path: str, params: dict[str, str] | None = None
    ) -> tuple[int, dict] | str:
        """``(status, json)`` for a 200 or 404, or a one-line failure reason.

        *path* starts with ``/``. A 404 comes back as ``(404, {})``.
        """
        params = dict(params or {})
        if self.mailto:
            params["mailto"] = self.mailto
        key = f"crossref:{path}?{urlencode(sorted(params.items()))}"
        cached = self._read(key)
        if cached is not None:
            return int(cached["status"]), cached["body"]
        response = await _get(
            self.client,
            f"{CROSSREF_API}{path}",
            params=params,
            headers={"User-Agent": _user_agent(self.mailto)},
        )
        if isinstance(response, str):
            return response
        body = {} if response.status_code == 404 else response.json()
        self._write(key, {"status": response.status_code, "body": body})
        return response.status_code, body

    # ── URL health ──────────────────────────────────────────────────────────

    async def check_url(self, url: str) -> UrlHealth:
        """Is the URL reachable? HEAD first, then GET when HEAD fails.

        Many servers answer HEAD wrongly (405, 403, or a 404 the GET does not
        give), so a HEAD that is not a success is asked again as GET. No body
        is read in either case. Every hop of a redirect chain is checked with
        ``is_safe_url``, so a redirect into a private address is refused.
        """
        key = f"url:{url}"
        cached = self._read(key)
        if cached is not None:
            return UrlHealth.model_validate(cached)
        health = await self._probe(url, "HEAD")
        if health.status != "ok" and health.status not in ("blocked", "dns_failure"):
            health = await self._probe(url, "GET")
        if health.status in _SETTLED:
            self._write(key, health.model_dump())
        return health

    async def _probe(self, url: str, method: str) -> UrlHealth:
        current = url
        try:
            for _ in range(6):
                safe, reason = self.url_check(current)
                if not safe:
                    raise SsrfBlocked(f"URL blocked: {reason} ({current})")
                request = self.client.build_request(
                    method, current, headers={"User-Agent": _user_agent(self.mailto)}
                )
                response = await self.client.send(
                    request, stream=True, follow_redirects=False
                )
                await response.aclose()
                location = response.headers.get("location")
                if response.status_code in _REDIRECTS and location:
                    current = str(response.url.join(location))
                    continue
                return _classify(url, current, response.status_code)
            raise SsrfBlocked(f"too many redirects starting at {url}")
        except SsrfBlocked as exc:
            return UrlHealth(url=url, status="blocked", detail=str(exc))
        except httpx.TimeoutException as exc:
            return UrlHealth(url=url, status="timeout", detail=type(exc).__name__)
        except httpx.HTTPError as exc:
            status: UrlStatus = "dns_failure" if _is_dns_failure(exc) else "error"
            return UrlHealth(
                url=url, status=status, detail=f"{type(exc).__name__}: {exc}"
            )

    # ── cache ───────────────────────────────────────────────────────────────

    def _read(self, key: str) -> dict | None:
        if self.cache is None:
            return None
        raw = self.cache.get(key)
        if raw is None:
            return None
        try:
            value = json.loads(raw)
        except ValueError:
            return None  # a damaged entry is a miss
        return value if isinstance(value, dict) else None

    def _write(self, key: str, value: dict) -> None:
        if self.cache is not None:
            self.cache.put(key, json.dumps(value).encode())


def _classify(url: str, final: str, code: int) -> UrlHealth:
    moved = final if final != url else ""
    if code < 400:
        status: UrlStatus = "ok"
    elif code in (404, 410):
        status = "not_found"
    elif code >= 500:
        status = "server_error"
    else:
        status = "client_error"
    return UrlHealth(
        url=url, status=status, http_status=code, final_url=moved, detail=f"HTTP {code}"
    )


def _is_dns_failure(exc: BaseException) -> bool:
    seen: set[int] = set()
    link: BaseException | None = exc
    while link is not None and id(link) not in seen:
        seen.add(id(link))
        if isinstance(link, socket.gaierror):
            return True
        text = str(link).lower()
        if any(
            m in text
            for m in (
                "name or service not known",
                "nodename nor servname",
                "temporary failure in name resolution",
                "getaddrinfo failed",
                "no address associated",
            )
        ):
            return True
        link = link.__cause__ or link.__context__
    return False
