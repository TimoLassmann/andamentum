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
composes the model-free deep_research backend (SearXNG search + safe page
fetch/extract), which carries SSRF protection, robots/paywall gating, safe
redirects, and a circuit breaker.

API docs: SearXNG (local instance; default http://127.0.0.1:4070)
Authentication: none (local SearXNG)

Architecture: Layer 1 (standalone; depends on deep_research, an allowed
evidence-gathering dependency).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from ..operations import GatheredEvidence

if TYPE_CHECKING:
    from ..preflight import CheckResult

logger = logging.getLogger(__name__)

_DEFAULT_SEARXNG_URL = "http://127.0.0.1:4070"


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
        "- Quoted phrases for exact matches: \"four-day work week\" productivity\n"
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
        ("Adoption of electric vehicles has risen sharply in Europe.",
         "electric vehicle adoption Europe statistics"),
        ("The four-day work week improves employee productivity.",
         "\"four-day work week\" productivity outcomes"),
        ("The James Webb Space Telescope detected CO2 in an exoplanet atmosphere.",
         "James Webb telescope CO2 exoplanet atmosphere"),
        ("Remote work reduces overall team productivity.",
         "remote work productivity studies evidence"),
        ("Global renewable energy capacity overtook coal in 2024.",
         "renewable energy capacity overtakes coal 2024"),
        # Out-of-domain — specialist biomedical providers cover these → abstain.
        ("Interleukin-6 drives synovial inflammation in rheumatoid arthritis.",
         None),
        ("Imatinib inhibits BCR-ABL with low-nanomolar IC50.",
         None),
        ("The KEYNOTE-189 trial improved overall survival in NSCLC.",
         None),
    ]

    output_kind = "assertion_evidence"
    independence_group = "general_web"
    provider_contract_version = 1

    def __init__(
        self,
        *,
        searxng_url: str = _DEFAULT_SEARXNG_URL,
        max_results: int = 8,
        max_pages: int = 4,
    ) -> None:
        """Args:
        searxng_url: Base URL of the local SearXNG instance.
        max_results: Search results to request from SearXNG.
        max_pages: Top results to fetch + extract full content from (the rest
            contribute their snippet only).
        """
        self.searxng_url = searxng_url
        self.max_results = max_results
        self.max_pages = max_pages

    async def check_health(self) -> "CheckResult":
        """Probe SearXNG reachability via the same search path ``gather`` uses."""
        import time

        from ..preflight import CheckResult

        t0 = time.monotonic()
        try:
            from andamentum.deep_research.backends import HttpxSearchBackend

            async with HttpxSearchBackend(searxng_url=self.searxng_url) as backend:
                await backend.search("health check", max_results=1)
            elapsed = (time.monotonic() - t0) * 1000
            return CheckResult(
                name="WebSearchProvider",
                status="pass",
                message=f"SearXNG reachable ({elapsed:.0f}ms)",
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

    async def gather(self, query: str) -> list[GatheredEvidence]:
        """Search the web and return extracted page content as evidence.

        Returns ``[]`` on any failure (SearXNG down, no results, all fetches
        failed) — never raises, per the provider contract.
        """
        try:
            from andamentum.deep_research.backends import HttpxSearchBackend
        except ImportError:
            logger.info("deep_research not available — web search disabled")
            return []

        try:
            async with HttpxSearchBackend(searxng_url=self.searxng_url) as backend:
                results = await backend.search(query, max_results=self.max_results)
                if not results:
                    return []

                to_fetch = results[: self.max_pages]

                async def _fetch(result: object) -> tuple[object, object | None]:
                    try:
                        page = await backend.fetch_page(result.url)  # type: ignore[attr-defined]
                        return result, page
                    except Exception as e:  # noqa: BLE001
                        logger.debug("web_search fetch failed for %s: %s", result.url, e)  # type: ignore[attr-defined]
                        return result, None

                fetched = await asyncio.gather(*(_fetch(r) for r in to_fetch))

                gathered: list[GatheredEvidence] = []
                for result, page in fetched:
                    # Full extracted content when the fetch succeeded; otherwise
                    # fall back to the search snippet so a fetch failure still
                    # contributes something.
                    body = (
                        page.content  # type: ignore[attr-defined]
                        if page is not None and getattr(page, "content", "")
                        else result.snippet  # type: ignore[attr-defined]
                    )
                    if not body:
                        continue
                    title = result.title  # type: ignore[attr-defined]
                    content = f"{title}\n\n{body}" if title else body
                    gathered.append(
                        GatheredEvidence(
                            content=content,
                            source_ref=result.url,  # type: ignore[attr-defined]
                            source_type="web_search",
                            evidence_kind="web_page",
                            identifiers={"url": result.url},  # type: ignore[attr-defined]
                            structured_data={
                                "title": title,
                                "url": result.url,  # type: ignore[attr-defined]
                                "domain": result.domain,  # type: ignore[attr-defined]
                                "snippet": result.snippet,  # type: ignore[attr-defined]
                                "fetched_full_content": page is not None,
                            },
                            quality_score=None,
                            quality_metadata={"domain": result.domain},  # type: ignore[attr-defined]
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
