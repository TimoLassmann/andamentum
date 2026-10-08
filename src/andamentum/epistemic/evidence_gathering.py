"""Default evidence gathering implementation.

Provides a WebSearchGatherer that runs the ``deep_research`` research loop
over the open web and hands its findings on as evidence.

Architecture: Layer 1 (standalone package, depends on deep_research)
"""

from __future__ import annotations

import logging
import time
import urllib.request
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from andamentum.deep_research import Finding, Passage, ResearchResult

    from .preflight import CheckResult

from .operations import GatheredEvidence, EvidenceGatherer, GathererCapability
from .operations.base import ProviderRegistry

logger = logging.getLogger(__name__)

# One gather is a small research run: rounds of search, read and reflect per
# sub-question, and pages read in full per round. The loop stops early when the
# question is answered, so a simple fact usually takes one round.
GATHER_MAX_ROUNDS = 2
GATHER_READS_PER_ROUND = 3
# Characters of the page kept on each side of a finding's quote as its context.
CONTEXT_CHARS = 800


def _searxng_is_healthy(url: str, timeout: float = 3.0) -> bool:
    """Check if SearXNG is responding."""
    try:
        urllib.request.urlopen(f"{url}/search?q=test&format=json", timeout=timeout)
        return True
    except Exception:
        return False


def _browser_is_available() -> bool:
    """Check that podman, which runs the isolated Chrome, is installed."""
    from andamentum.deep_research.browser import BrowserUnavailable, require_podman

    try:
        require_podman()
    except BrowserUnavailable:
        return False
    return True


class WebSearchGatherer:
    """Evidence gatherer running the ``deep_research`` loop over the web.

    Satisfies the EvidenceGatherer protocol. One ``gather`` call is one small
    research run (``research(query, mode="web")``, ``GATHER_MAX_ROUNDS``
    rounds). Each finding whose quote the engine found in the page text becomes
    one ``GatheredEvidence``: the content is the page text around the quote,
    the source is the work's URL. A finding whose quote could not be verified is
    not evidence and is left out; the count is logged. ``quality_score`` stays
    ``None``: the engine gives no per-finding relevance score, and a number made up here
    would be a hidden judgement, so the downstream scorer decides.

    Search goes through the isolated Chrome (``browser``: a podman container
    each ``gather`` starts and removes) when it is set, else through the SearXNG
    at ``searxng_url``. One of the two is required.
    """

    def __init__(
        self,
        *,
        model: str,
        searxng_url: str | None = None,
        browser: bool = False,
    ):
        if not searxng_url and not browser:
            raise ValueError(
                "web search needs a SearXNG instance (searxng_url) or the "
                "isolated browser (browser=True)"
            )
        self.model = model
        self.searxng_url = searxng_url
        self.browser = browser

    async def check_health(self) -> "CheckResult":
        """Test that the search can start (podman for the browser, else SearXNG)."""
        from .preflight import CheckResult

        t0 = time.monotonic()
        if self.browser:
            healthy = _browser_is_available()
            where = "podman for the isolated Chrome"
        else:
            assert self.searxng_url is not None  # __init__ requires one address
            healthy = _searxng_is_healthy(self.searxng_url, timeout=5.0)
            where = f"SearXNG at {self.searxng_url}"
        elapsed = (time.monotonic() - t0) * 1000
        if healthy:
            return CheckResult(
                name="WebSearch",
                status="pass",
                message=f"{where} reachable ({elapsed:.0f}ms)",
                elapsed_ms=elapsed,
            )
        return CheckResult(
            name="WebSearch",
            status="fail",
            message=f"{where} not reachable",
            elapsed_ms=elapsed,
        )

    async def capable_of(self, capability: str) -> GathererCapability:
        """A live web search accepts a query framed any way — always
        capable, regardless of which capability string is asked about.
        """
        del capability
        return GathererCapability(capable=True, source_types=["web_search"])

    async def gather(self, source_type: str, query: str) -> list[GatheredEvidence]:
        """Gather evidence via one ``deep_research`` web run.

        Returns ``[]`` when the run produced no finding with a verified quote;
        the caller treats that as "no source found", not as evidence.
        """
        del source_type
        from andamentum.deep_research import Budget, SourceConfig, research

        result = await research(
            query,
            mode="web",
            model=self.model,
            budget=Budget(
                max_rounds=GATHER_MAX_ROUNDS, reads_per_round=GATHER_READS_PER_ROUND
            ),
            sources=SourceConfig(searxng_url=self.searxng_url, browser=self.browser),
        )
        for reason in result.degraded:
            logger.warning(
                "[WebSearchGatherer] degraded run for %.80s: %s", query, reason
            )
        return _evidence_from_result(result, query=query)


def _evidence_from_result(
    result: "ResearchResult", *, query: str
) -> list[GatheredEvidence]:
    """One ``GatheredEvidence`` per finding with a verified quote."""
    works = {w.work_id: w for w in result.works}
    passages = {p.passage_id: p for p in result.passages}
    gathered: list[GatheredEvidence] = []
    skipped = 0
    for finding in result.findings:
        if not finding.quote_verified:
            skipped += 1
            continue
        work = works.get(finding.work_id)
        if work is None:
            raise RuntimeError(
                f"finding {finding.finding_id} cites work {finding.work_id} "
                f"which is not among the run's works"
            )
        url = work.url or (f"https://doi.org/{work.doi}" if work.doi else None)
        if not url:
            raise RuntimeError(
                f"work {finding.work_id} of finding {finding.finding_id} "
                f"has neither a URL nor a DOI"
            )
        content = _context_around(finding, passages.get(finding.passage_id))
        gathered.append(
            GatheredEvidence(
                content=content,
                source_ref=url,
                source_type="web_search",
                evidence_kind="web_page",
                identifiers={"url": url},
                structured_data={
                    "claim": finding.claim,
                    "quote": finding.quote,
                    "quote_verified": finding.quote_verified,
                    "title": work.title,
                    "site": work.venue,
                },
                limitations=[
                    "Web source; passage taken from a larger page, not peer-reviewed"
                ],
            )
        )
    if skipped:
        logger.warning(
            "[WebSearchGatherer] %d of %d findings for %.80s had an unverified "
            "quote and were not passed on as evidence",
            skipped,
            len(result.findings),
            query,
        )
    return gathered


def _context_around(finding: "Finding", passage: "Passage | None") -> str:
    """The text of the passage around the finding's quote.

    Falls back to the quote itself when the passage is missing or the quote
    cannot be located in it (the engine verified it against the page, whose
    window may differ from the stored passage).
    """
    from andamentum.deep_research import locate_quote

    if passage is None:
        return finding.quote
    at = locate_quote(finding.quote, passage.text)
    if at is None:
        return finding.quote
    start, end = at
    return passage.text[max(0, start - CONTEXT_CHARS) : end + CONTEXT_CHARS].strip()


class CompositeGatherer:
    """Routes evidence queries to registered providers with web search fallback.

    Satisfies the EvidenceGatherer protocol. Providers are keyed by source_type.
    Unknown source_types fall back to web search.
    """

    def __init__(
        self,
        web_search: Any,  # accepts WebSearchGatherer or any compatible gatherer
        providers: Optional[ProviderRegistry] = None,
    ):
        self._web_search = web_search
        self._providers: dict[str, Any] = dict(providers) if providers else {}

    async def capable_of(self, capability: str) -> GathererCapability:
        """Every registered provider and the web-search fallback accept
        a natural-language query (providers/CONTRIBUTING.md: "the
        epistemic system passes natural language intent") — capability
        is about QUERY FORM, not network reach, so any registered
        source qualifies, including a closed corpus that happens to
        hold what a refutation-seeking query looks for. Web search
        stays first when reachable (via ``self._web_search`` or a
        "web_search"-keyed provider), preserving this gatherer's
        long-standing default route; other registered providers follow
        in registration order. Only a gatherer with nothing registered
        at all is incapable.
        """
        del capability
        source_types: list[str] = []
        if "web_search" in self._providers or self._web_search is not None:
            source_types.append("web_search")
        source_types.extend(k for k in self._providers if k != "web_search")
        if not source_types:
            return GathererCapability(
                capable=False,
                reason="no evidence sources registered on this gatherer",
            )
        return GathererCapability(capable=True, source_types=source_types)

    async def gather(self, source_type: str, query: str) -> list[GatheredEvidence]:
        """Gather evidence.

        Routing:
        - Specific provider registered for source_type → call it.
          Provider raises → propagate (no silent fallback).
          Provider returns empty → return empty (no silent fallback).
        - source_type == "all" → call every provider and web search,
          aggregate successes, log per-provider failures. Raise only
          if EVERY single call fails.
        - Unknown source_type → use web search (explicit default route).
        """
        logger.info(
            "[CompositeGatherer] source_type=%s providers=%s query=%.80s",
            source_type,
            list(self._providers.keys()),
            query,
        )

        if source_type == "all":
            all_results: list[GatheredEvidence] = []
            failures: list[tuple[str, Exception]] = []

            for name, prov in self._providers.items():
                try:
                    results = await prov.gather(query)
                    logger.info(
                        "[CompositeGatherer] Provider '%s' returned %d results",
                        name,
                        len(results),
                    )
                    all_results.extend(results)
                except Exception as e:
                    logger.warning(
                        "[CompositeGatherer] Provider '%s' failed during 'all': %s",
                        name,
                        e,
                    )
                    failures.append((name, e))
            # `None` means the caller asked for providers ONLY (see
            # get_default_gatherer's use_web_search). Skipped, not failed: a
            # source that was deliberately not consulted must not appear in
            # `failures`, or a clean offline run would report an error it did
            # not have.
            if self._web_search is not None:
                try:
                    web_results = await self._web_search.gather(source_type, query)
                    all_results.extend(web_results)
                except Exception as e:
                    logger.warning(
                        "[CompositeGatherer] Web search failed during 'all': %s", e
                    )
                    failures.append(("web_search", e))

            if not all_results and failures:
                raise RuntimeError(
                    f"All gather calls failed for 'all' source_type. "
                    f"Failures: {[(n, type(e).__name__, str(e)) for n, e in failures]}"
                )
            return all_results

        provider = self._providers.get(source_type)
        if provider:
            results = await provider.gather(query)
            logger.info(
                "[CompositeGatherer] Provider '%s' returned %d results",
                source_type,
                len(results),
            )
            return results

        # Unknown source_type → explicit web-search default route.
        # With web search opted out there is no fallback route, and an empty
        # list is the honest answer: no configured source covers this
        # source_type. Reaching for the web here would silently defeat the
        # opt-out the caller asked for.
        if self._web_search is None:
            logger.info(
                "[CompositeGatherer] no provider for source_type %r and web "
                "search is disabled — returning no results",
                source_type,
            )
            return []
        return await self._web_search.gather(source_type, query)


def get_default_gatherer(
    *,
    model: str,
    providers: Optional[ProviderRegistry] = None,
    use_web_search: bool = True,
    searxng_url: Optional[str] = None,
    browser: bool = False,
) -> EvidenceGatherer:
    """Create the default evidence gatherer.

    Args:
        model: LLM model string for web search analysis.
        providers: Optional dict of named providers (e.g., from
            ``andamentum.epistemic.providers.get_biomedical_providers()``).
            When provided, returns a CompositeGatherer that routes
            by source_type and falls back to web search.
        use_web_search: False for providers only (requires ``providers``).
        searxng_url: The address of a SearXNG instance for web search.
        browser: Search the web through the isolated Chrome (a podman
            container) instead of SearXNG. With ``use_web_search`` one of the
            two is required (``ValueError`` otherwise).

    Returns a ``CompositeGatherer`` when ``providers`` are given, else a
    ``WebSearchGatherer``.
    """
    # WEB SEARCH IS OPT-OUT, NOT ABSENT-BY-DEFAULT.
    #
    # Passing `providers` used to ADD to web search rather than replace it, so
    # there was no way to say "these sources and nothing else". A caller running
    # a closed, reproducible corpus therefore still made live network calls:
    # measured 2026-08-30 on a benchmark run that believed itself fully offline,
    # 6 of 2,696 evidence rows came from the web. 99.8% offline is not offline —
    # it is an unrepeatable run that looks repeatable.
    #
    # `use_web_search=False` requires `providers`, because a gatherer with
    # neither web search nor providers can gather nothing at all, and silently
    # returning such a thing would be the same class of failure one layer down.
    if not use_web_search:
        if not providers:
            raise ValueError(
                "use_web_search=False requires providers — a gatherer with "
                "neither web search nor any provider can gather nothing, and "
                "would report an empty pool as though it had searched."
            )
        return CompositeGatherer(None, providers)  # type: ignore[return-value]

    web = WebSearchGatherer(model=model, searxng_url=searxng_url, browser=browser)
    if providers:
        return CompositeGatherer(web, providers)  # type: ignore[return-value]
    return web
