"""Pre-fetch gate: robots.txt + paywalled-publisher tripwire.

A single async check that callers run BEFORE fetching an HTTP(S) URL.
Raises one of two typed exceptions when the fetch should be refused.

The gate is fully explicit — no environment variables, no ambient state.
Configuration (the set of paywalled hosts the caller has TDM licences
for) is a keyword-only argument, propagated from the CLI / surface API.

Two independent checks:

  1. robots.txt — host-level. The gate fetches /robots.txt on first
     encounter, caches the parsed result in-process, and asks the
     stdlib RobotFileParser whether *user_agent* may visit *url*'s path.
     - Missing robots.txt (HTTP 404) → all paths allowed (RFC-aligned).
     - Fetch failure (timeout, network error) → allowed-with-warning;
       refusing real content because robots.txt itself is unreachable
       would be over-broad.

  2. Paywalled-publisher tripwire — hostname suffix match against the
     curated seed list (Elsevier, Springer Nature, Wiley, IEEE, ACM,
     NEJM, JAMA, Cell Press, Nature, Science / AAAS). When matched,
     refuse unless the caller has listed the host in *tdm_allowed_hosts*
     — the caller's attestation that they hold a text-and-data-mining
     licence for that publisher.

Cache: module-level dict keyed by (scheme, host). In-memory, per-process,
no disk persistence. A long-running deep_research session fetches each
host's robots.txt at most once.

Audit events: when *tdm_allowed_hosts* permits a fetch that the tripwire
would otherwise have blocked, the gate emits an INFO-level log record on
the ``andamentum.fetch_gate`` logger. Callers that want a persistent
audit trail install a handler on that logger.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger("andamentum.fetch_gate")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FetchGateError(Exception):
    """Base class for fetch-gate refusals."""


class RobotsBlocked(FetchGateError):
    """robots.txt disallows *user_agent* from fetching *url*."""

    def __init__(self, url: str, user_agent: str) -> None:
        self.url = url
        self.user_agent = user_agent
        super().__init__(
            f"robots.txt disallows {user_agent!r} from fetching {url}. "
            f"The publisher's robots.txt explicitly refuses this user-agent at "
            f"this path; respect the directive or contact the site owner."
        )


class PaywallBlocked(FetchGateError):
    """*host* is on the paywalled-publisher seed list and not allowed via TDM."""

    def __init__(self, url: str, host: str) -> None:
        self.url = url
        self.host = host
        super().__init__(
            f"refusing to fetch {url}: host {host!r} is a paywalled academic "
            f"publisher and is not in the caller's `tdm_allowed_hosts`. "
            f"Bulk extraction without a text-and-data-mining licence is "
            f"contractually prohibited by major publishers. If your "
            f"institution holds a TDM agreement for this host, pass it "
            f"explicitly via the surface API's `tdm_allowed_hosts` argument "
            f"(or the CLI's `--tdm-host` flag)."
        )


# ---------------------------------------------------------------------------
# Paywalled-publisher seed list
# ---------------------------------------------------------------------------

# Hostname suffix patterns. A URL host matches a pattern if the host
# equals the pattern OR ends with "." + pattern.
#
# Open-access homes (arxiv, biorxiv, medrxiv, europepmc, ncbi/PMC, plos,
# frontiers, elife, mdpi, f1000research) are intentionally NOT on this
# list. Public preprint servers and indexing services are intentionally
# NOT on this list (Google Scholar, Semantic Scholar, OpenAlex, etc.).
_PAYWALLED_HOSTS: frozenset[str] = frozenset(
    {
        "elsevier.com",
        "sciencedirect.com",
        "springernature.com",
        "springer.com",
        "nature.com",
        "wiley.com",
        "onlinelibrary.wiley.com",
        "ieee.org",
        "ieeexplore.ieee.org",
        "acm.org",
        "dl.acm.org",
        "nejm.org",
        "jamanetwork.com",
        "cell.com",
        "science.org",
        "sciencemag.org",
    }
)


def _host_matches_paywall(host: str) -> str | None:
    """Return the matching paywall pattern if *host* is paywalled, else None."""
    if not host:
        return None
    lowered = host.lower()
    for pattern in _PAYWALLED_HOSTS:
        if lowered == pattern or lowered.endswith("." + pattern):
            return pattern
    return None


# ---------------------------------------------------------------------------
# robots.txt cache
# ---------------------------------------------------------------------------


@dataclass
class _RobotsEntry:
    """Cached robots.txt result for one origin."""

    parser: RobotFileParser | None  # None when robots.txt was unreachable or 404
    fetched_ok: bool  # False if fetch errored or 404 — treat as "all allowed"


# Keyed by (scheme, host). Module-level, in-memory, per-process.
_ROBOTS_CACHE: dict[tuple[str, str], _RobotsEntry] = {}


def _robots_cache_clear() -> None:
    """Test helper: clear the in-process robots.txt cache."""
    _ROBOTS_CACHE.clear()


async def _fetch_robots_txt(
    scheme: str,
    host: str,
    *,
    user_agent: str,
    client: httpx.AsyncClient,
) -> _RobotsEntry:
    """Fetch /robots.txt for *scheme*://*host* and return a parsed entry.

    Network errors and 404s yield ``fetched_ok=False`` with no parser —
    callers treat that as "all paths allowed."
    """
    robots_url = f"{scheme}://{host}/robots.txt"
    try:
        resp = await client.get(
            robots_url,
            timeout=5.0,
            headers={"User-Agent": user_agent},
        )
    except httpx.HTTPError as exc:
        logger.warning(
            "robots.txt fetch failed for %s (%s); treating as all-allowed",
            host,
            exc,
        )
        return _RobotsEntry(parser=None, fetched_ok=False)

    if resp.status_code == 404:
        # No robots.txt → all paths allowed, per the de-facto standard.
        return _RobotsEntry(parser=None, fetched_ok=False)

    if resp.status_code >= 400:
        # Any other error code: log and treat permissively, same as a fetch error.
        logger.warning(
            "robots.txt for %s returned HTTP %d; treating as all-allowed",
            host,
            resp.status_code,
        )
        return _RobotsEntry(parser=None, fetched_ok=False)

    parser = RobotFileParser()
    parser.parse(resp.text.splitlines())
    return _RobotsEntry(parser=parser, fetched_ok=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def check_fetch_allowed(
    url: str,
    *,
    user_agent: str,
    tdm_allowed_hosts: frozenset[str] = frozenset(),
    client: httpx.AsyncClient | None = None,
) -> None:
    """Refuse the fetch if robots.txt or the paywall tripwire blocks it.

    Run before every HTTP(S) GET of an external URL.

    Parameters
    ----------
    url:
        The URL the caller is about to fetch.
    user_agent:
        The User-Agent the caller will send (and that the gate uses to
        query robots.txt).
    tdm_allowed_hosts:
        Hostnames the caller attests they hold a text-and-data-mining
        licence for. A match here disarms the paywall tripwire for that
        host. Pass an empty frozenset (default) to apply the tripwire
        fully.
    client:
        Optional httpx.AsyncClient. When None, a short-lived client is
        used for the robots.txt fetch. Re-using the caller's client is
        more efficient for high-volume callers.

    Raises
    ------
    PaywallBlocked
        Host is on the paywalled-publisher seed list and not in
        *tdm_allowed_hosts*.
    RobotsBlocked
        Host's robots.txt disallows *user_agent* at *url*'s path.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        # Non-HTTP URLs (file://, data:, ...) — nothing to gate.
        return
    host = parsed.netloc.lower()
    if not host:
        return

    # ── Paywall tripwire ──────────────────────────────────────────────────
    paywall_match = _host_matches_paywall(host)
    if paywall_match is not None:
        # Strip port if present, e.g. "example.com:8080" → "example.com"
        bare_host = host.split(":", 1)[0]
        normalised_allow = {h.lower().strip() for h in tdm_allowed_hosts}
        if (
            paywall_match not in normalised_allow
            and bare_host not in normalised_allow
        ):
            raise PaywallBlocked(url=url, host=bare_host)
        # Allowed via TDM acknowledgment — emit an audit-grade INFO line so
        # any logging handler installed by the caller can record it.
        logger.info(
            "tdm_allowed: fetching %s (host %s matched paywall pattern %s; "
            "explicitly allowed by caller)",
            url,
            bare_host,
            paywall_match,
        )

    # ── robots.txt ────────────────────────────────────────────────────────
    cache_key = (parsed.scheme, host)
    entry = _ROBOTS_CACHE.get(cache_key)
    if entry is None:
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(follow_redirects=True)
        try:
            entry = await _fetch_robots_txt(
                parsed.scheme, host, user_agent=user_agent, client=client
            )
        finally:
            if owns_client and client is not None:
                await client.aclose()
        _ROBOTS_CACHE[cache_key] = entry

    if entry.parser is not None and not entry.parser.can_fetch(user_agent, url):
        raise RobotsBlocked(url=url, user_agent=user_agent)


# ---------------------------------------------------------------------------
# Shared User-Agent helper
# ---------------------------------------------------------------------------


def user_agent_for(component: str) -> str:
    """Return the canonical andamentum User-Agent string for *component*.

    Examples
    --------
    >>> user_agent_for("harvest")
    'andamentum-harvest/<version> (+https://github.com/TimoLassmann/andamentum)'

    Identifies andamentum to remote hosts per RFC 9110 §10.1.5 so abuse
    desks can contact the project.
    """
    from andamentum import __version__ as _version

    return (
        f"andamentum-{component}/{_version} "
        f"(+https://github.com/TimoLassmann/andamentum)"
    )
