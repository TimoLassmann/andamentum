"""General web-search evidence provider.

Restores general-domain (non-biomedical) evidence to the epistemic system. Web
search was the universal fallback in the legacy three-agent gather chain; the
2026-05-12 description-driven-dispatch refactor (commit 837f941) removed that
chain and, with it, web search as a *primary* evidence source — leaving the 10
biomedical APIs as the only registered providers. This re-introduces web search
as a first-class **dispatch provider**, so the dispatch agent can route to it
for claims no specialist provider covers (current events, general topics,
non-biomedical science), while abstaining on specialist biomedical claims that
the dedicated providers handle better.

Unlike the legacy fallback, this is a pure retrieval provider matching the
provider contract (CONTRIBUTING.md): no LLM synthesis, ``quality_score=None``
always, returns ``list[GatheredEvidence]`` (empty on error, never raises). It
composes the model-free ``deep_research`` web source (SearXNG or the isolated
Chrome search, safe page fetch/extract), which carries SSRF protection, robots/paywall
gating and safe redirects.

API docs: a SearXNG instance (``searxng_url``), or the isolated Chrome
(``browser``); one of the two is required to gather
Authentication: none

Architecture: Layer 1 (standalone; depends on deep_research, an allowed
evidence-gathering dependency).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

from ..operations import GatheredEvidence

if TYPE_CHECKING:
    from ..preflight import CheckResult

logger = logging.getLogger(__name__)


class WebSearchProvider:
    """Dispatch evidence provider backed by SearXNG web search.

    Pure retrieval: searches the web, fetches and extracts the top pages, and
    returns them as evidence. The dispatch agent decides per-claim whether to
    route here based on the description/examples below.
    """

    description = (
        "General-purpose web search for claims that the specialist biomedical "
        "and scholarly providers do not cover: current events and news, public "
        "and policy information, technology and industry topics, and "
        "non-biomedical science or general knowledge. It returns extracted text "
        "from the most relevant public web pages. Prefer a specialist provider "
        "(PubMed, OpenAlex, ClinicalTrials, etc.) whenever the claim is squarely "
        "in their domain; use web search when no specialist is a clear fit or "
        "the claim is about general, current, or non-academic information. "
        "Example claims it suits: 'electric-vehicle adoption is rising in "
        "Europe', 'the 4-day work week improves reported productivity', 'the "
        "James Webb telescope detected CO2 in an exoplanet atmosphere'."
    )

    query_guidance = (
        "The query is sent to a local SearXNG instance, which aggregates results "
        "from general web search engines. Use natural-language keyword queries — "
        "the strongest signal is a tight set of the claim's distinctive terms, "
        "not a full sentence. Supported styles that all work:\n"
        "- Plain keywords: electric vehicle adoption Europe 2024\n"
        '- Quoted phrases for exact matches: "four-day work week" productivity\n'
        "- Entity + attribute: James Webb telescope CO2 exoplanet\n"
        "- Topic + qualifier: remote work productivity meta-analysis\n"
        "- Comparison terms: nuclear vs solar cost per kilowatt-hour\n"
        "Drop stopwords and question phrasing ('what is', 'how does'); keep "
        "proper nouns, numbers, and domain terms. Avoid site: operators unless "
        "the claim names a specific source."
    )

    # (claim, native_query) pairs. None = the provider should abstain (a
    # specialist provider covers it). Includes both in-domain and out-of-domain
    # examples (required by the dispatch-quality fixtures + contract tests).
    query_examples: list[tuple[str, str | None]] = [
        (
            "Adoption of electric vehicles has risen sharply in Europe.",
            "electric vehicle adoption Europe statistics",
        ),
        (
            "The four-day work week improves employee productivity.",
            '"four-day work week" productivity outcomes',
        ),
        (
            "The James Webb Space Telescope detected CO2 in an exoplanet atmosphere.",
            "James Webb telescope CO2 exoplanet atmosphere",
        ),
        (
            "Remote work reduces overall team productivity.",
            "remote work productivity studies evidence",
        ),
        (
            "Global renewable energy capacity overtook coal in 2024.",
            "renewable energy capacity overtakes coal 2024",
        ),
        # Out-of-domain — specialist biomedical providers cover these → abstain.
        ("Interleukin-6 drives synovial inflammation in rheumatoid arthritis.", None),
        ("Imatinib inhibits BCR-ABL with low-nanomolar IC50.", None),
        ("The KEYNOTE-189 trial improved overall survival in NSCLC.", None),
    ]

    output_kind = "assertion_evidence"
    independence_group = "general_web"
    provider_contract_version = 1

    def __init__(
        self,
        *,
        searxng_url: str | None = None,
        browser: bool = False,
        max_results: int = 8,
        max_pages: int = 4,
    ) -> None:
        """Args:
        searxng_url: Base URL of a SearXNG instance. With neither this nor
            ``browser``, ``gather`` and ``check_health`` raise
            ``ValueError`` (a configuration error is not a failed search).
        browser: Search through the isolated Chrome (a podman container each
            call starts and removes) instead of SearXNG.
        max_results: Search results to request.
        max_pages: Top results to fetch + extract full content from (the rest
            contribute their snippet only).
        """
        self.searxng_url = searxng_url
        self.browser = browser
        self.max_results = max_results
        self.max_pages = max_pages

    def _require_address(self) -> None:
        if not self.searxng_url and not self.browser:
            raise ValueError(
                "web search needs a SearXNG instance (searxng_url) or the "
                "isolated browser (browser=True)"
            )

    async def _open(self, stack: AsyncExitStack) -> Any:
        """A web source over an in-memory store (and the isolated Chrome when
        ``browser`` is set); the stack closes what was opened."""
        from andamentum.deep_research import ResearchStore
        from andamentum.deep_research.browser import browser_session
        from andamentum.deep_research.sources import SourceClient, WebSource

        client = SourceClient(ResearchStore.in_memory())
        stack.push_async_callback(client.aclose)
        session = (
            await stack.enter_async_context(browser_session()) if self.browser else None
        )
        return WebSource(searxng_url=self.searxng_url, browser=session, client=client)

    async def check_health(self) -> "CheckResult":
        """Probe search reachability via the same search path ``gather`` uses."""
        import time

        from ..preflight import CheckResult

        self._require_address()
        t0 = time.monotonic()
        stack = AsyncExitStack()
        try:
            source = await self._open(stack)
            await source.search("health check", limit=1)
            elapsed = (time.monotonic() - t0) * 1000
            where = "isolated Chrome" if self.browser else "SearXNG"
            return CheckResult(
                name="WebSearchProvider",
                status="pass",
                message=f"{where} reachable ({elapsed:.0f}ms)",
                elapsed_ms=elapsed,
            )
        except Exception as e:  # noqa: BLE001
            elapsed = (time.monotonic() - t0) * 1000
            return CheckResult(
                name="WebSearchProvider",
                status="fail",
                message=str(e),
                elapsed_ms=elapsed,
            )
        finally:
            await stack.aclose()

    async def gather(self, query: str) -> list[GatheredEvidence]:
        """Search the web and return extracted page content as evidence.

        Returns ``[]`` on any failure (search down, no results, all fetches
        failed) — never raises, per the provider contract. A missing search
        address is a configuration error and raises ``ValueError``.
        """
        self._require_address()
        stack = AsyncExitStack()
        try:
            source = await self._open(stack)
            results = await source.search(query, limit=self.max_results)
            if not results:
                return []

            async def _fetch(candidate: Any) -> tuple[Any, Any]:
                try:
                    return candidate, await source.fetch_text(candidate.work)
                except Exception as e:  # noqa: BLE001
                    logger.debug(
                        "web_search fetch failed for %s: %s", candidate.work.url, e
                    )
                    return candidate, None

            fetched = await asyncio.gather(
                *(_fetch(c) for c in results[: self.max_pages])
            )

            gathered: list[GatheredEvidence] = []
            for candidate, page in fetched:
                work = candidate.work
                # Full extracted text when the fetch succeeded; otherwise fall
                # back to the search snippet so a fetch failure still
                # contributes something.
                body = page.text if page is not None else candidate.snippet
                if not body:
                    continue
                title = work.title or ""
                content = f"{title}\n\n{body}" if title else body
                gathered.append(
                    GatheredEvidence(
                        content=content,
                        source_ref=work.url or "",
                        source_type="web_search",
                        evidence_kind="web_page",
                        identifiers={"url": work.url or ""},
                        structured_data={
                            "title": title,
                            "url": work.url,
                            "domain": work.venue,
                            "snippet": candidate.snippet,
                            "fetched_full_content": page is not None,
                        },
                        quality_score=None,
                        quality_metadata={"domain": work.venue},
                        limitations=[
                            "Web source — not peer-reviewed; assess "
                            "credibility independently."
                        ],
                    )
                )
            return gathered
        except Exception as e:  # noqa: BLE001
            logger.warning("Web search query failed for '%s': %s", query, e)
            return []
        finally:
            await stack.aclose()
