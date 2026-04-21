"""Default evidence gathering implementation.

Provides a WebSearchGatherer that uses deep_research (optional dependency)
to fetch real evidence from the web. When deep_research is not installed,
the gatherer is not available and operations fall back to agent-only extraction.

Architecture: Layer 1 (standalone package, optional dependency on deep_research)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .preflight import CheckResult
    from andamentum.deep_research.models import PageSummary

from .operations import GatheredEvidence, EvidenceGatherer

logger = logging.getLogger(__name__)

# SearXNG defaults (match deep_research conventions)
_SEARXNG_PORT = int(os.getenv("SEARXNG_PORT", "4070"))
_SEARXNG_URL = os.getenv("SEARXNG_URL", f"http://127.0.0.1:{_SEARXNG_PORT}")
_SEARXNG_IMAGE = os.getenv("SEARXNG_IMAGE", "docker.io/searxng/searxng:latest")
_SEARXNG_CONTAINER = os.getenv("SEARXNG_CONTAINER", "mcp-searxng")


def _searxng_is_healthy(url: str = _SEARXNG_URL, timeout: float = 3.0) -> bool:
    """Check if SearXNG is responding."""
    try:
        urllib.request.urlopen(f"{url}/search?q=test&format=json", timeout=timeout)
        return True
    except Exception:
        return False


def ensure_searxng(url: str = _SEARXNG_URL, port: int = _SEARXNG_PORT) -> bool:
    """Ensure SearXNG is running, starting it via podman if needed.

    Returns True if SearXNG is available after this call.
    Uses the same container name and settings as the deep_research PodmanSearxngManager
    for compatibility.
    """
    if _searxng_is_healthy(url):
        return True

    if not shutil.which("podman"):
        logger.warning("SearXNG not running and podman not found")
        return False

    # Write minimal settings
    state_dir = Path(
        os.getenv("SEARXNG_STATE_DIR", Path.home() / ".cache" / "mcp-searxng")
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    settings_path = state_dir / "settings.yml"

    if not settings_path.exists():
        secret = os.urandom(32).hex()
        settings_path.write_text(
            "use_default_settings: true\n"
            "search:\n"
            "  formats:\n"
            "    - html\n"
            "    - json\n"
            "server:\n"
            f'  secret_key: "{secret}"\n'
            "  port: 8080\n"
            "  method: GET\n"
            "ui:\n"
            "  query_in_title: true\n",
            encoding="utf-8",
        )

    # Start container (--replace handles existing stopped containers)
    cmd = [
        "podman",
        "run",
        "--name",
        _SEARXNG_CONTAINER,
        "--replace",
        "-d",
        "-p",
        f"{port}:8080",
        "-v",
        f"{settings_path}:/etc/searxng/settings.yml:ro",
        _SEARXNG_IMAGE,
    ]
    logger.info(f"Starting SearXNG: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        logger.warning(f"Failed to start SearXNG: {result.stderr or result.stdout}")
        return False

    # Wait for it to come up
    for _ in range(15):
        time.sleep(2)
        if _searxng_is_healthy(url):
            logger.info("SearXNG started successfully")
            return True

    logger.warning("SearXNG started but not responding after 30s")
    return False


class WebSearchGatherer:
    """Evidence gatherer using deep_research for web search.

    Satisfies the EvidenceGatherer protocol. Uses `deep_research.orchestrator`
    for actual web search via SearXNG + LLM analysis.
    """

    def __init__(self, *, model: str, embedding_model: str | None = None):
        self.model = model
        self.embedding_model = embedding_model
        self._searxng_ensured = False

    async def check_health(self) -> "CheckResult":
        """Test SearXNG reachability."""
        from .preflight import CheckResult

        t0 = time.monotonic()
        healthy = _searxng_is_healthy(timeout=5.0)
        elapsed = (time.monotonic() - t0) * 1000
        if healthy:
            return CheckResult(
                name="WebSearch",
                status="pass",
                message=f"SearXNG reachable ({elapsed:.0f}ms)",
                elapsed_ms=elapsed,
            )
        return CheckResult(
            name="WebSearch",
            status="fail",
            message=f"SearXNG not reachable at {_SEARXNG_URL}",
            elapsed_ms=elapsed,
        )

    async def gather(self, source_type: str, query: str) -> list[GatheredEvidence]:
        """Gather evidence via deep_research web search."""
        from andamentum.deep_research.orchestrator import run_research

        # Ensure SearXNG is running on first call
        if not self._searxng_ensured:
            ensure_searxng()
            self._searxng_ensured = True

        result = await run_research(
            query=query,
            max_iterations=1,
            model=self.model,
            max_results=5,
            max_pages=3,
            verbose=False,
        )

        gathered: list[GatheredEvidence] = []

        # Build URL → PageSummary lookup for AI pre-analysis metadata
        summary_by_url: dict[str, "PageSummary"] = {}
        for ps in result.page_summaries:
            summary_by_url.setdefault(ps.url, ps)

        # Strategy 1: Passage extraction from fetched_pages (preferred)
        if result.fetched_pages:
            from .passage_extraction import extract_passages, PageData
            from .embeddings import _chunk_text, embed_texts as _embed_texts

            # Build PageData for each page that has a summary
            page_data_list: list[PageData] = []
            for page in result.fetched_pages:
                summary = summary_by_url.get(page.url)
                if summary is None:
                    continue
                page_data_list.append(
                    PageData(
                        url=page.url,
                        title=page.title,
                        content=page.content,
                        key_excerpts=list(summary.key_excerpts),
                        key_points=list(summary.key_points),
                        relevance_score=summary.relevance_score,
                    )
                )

            if page_data_list:
                # Cross-page findings from the synthesiser
                cross_findings = (
                    list(result.output.key_findings) if result.output else []
                )

                # Pre-compute chunk embeddings per page
                chunk_embeddings_by_url: dict[str, list[list[float]]] = {}
                for pd in page_data_list:
                    chunks = _chunk_text(pd.content)
                    if self.embedding_model:
                        try:
                            chunk_embeddings_by_url[pd.url] = await _embed_texts(
                                chunks, model=self.embedding_model
                            )
                        except RuntimeError:
                            logger.warning("Embedding failed for %s, skipping", pd.url)

                # Pre-compute cross-page finding embeddings
                cross_finding_embs: list[list[float]] | None = None
                if cross_findings and self.embedding_model:
                    try:
                        cross_finding_embs = await _embed_texts(
                            cross_findings, model=self.embedding_model
                        )
                    except RuntimeError:
                        logger.warning("Embedding failed for cross-page findings")

                # Extract focused passages
                passages = await extract_passages(
                    pages=page_data_list,
                    cross_page_findings=cross_findings,
                    cross_page_finding_embeddings=cross_finding_embs,
                    chunk_embeddings_by_url=chunk_embeddings_by_url,
                    embedding_model=self.embedding_model,
                )

                for passage in passages:
                    gathered.append(
                        GatheredEvidence(
                            content=passage.text,
                            source_ref=passage.page_url,
                            source_type="web_search",
                            evidence_kind="web_page",
                            structured_data={
                                "annotations": passage.annotations,
                                "page_title": passage.page_title,
                            },
                            limitations=[
                                "Web source; passage extracted from larger page"
                            ],
                            quality_score=next(
                                (
                                    pd.relevance_score
                                    for pd in page_data_list
                                    if pd.url == passage.page_url
                                ),
                                0.5,
                            ),
                            quality_metadata={
                                "title": passage.page_title,
                                "annotation_count": passage.annotation_count,
                            },
                        )
                    )

        # Strategy 2: Fall back to page_summaries (backward compat / empty fetched_pages)
        if not gathered and result.page_summaries:
            for ps in result.page_summaries:
                if ps.relevance_score < 0.2:
                    continue
                key_points = (
                    "\n".join(f"- {p}" for p in ps.key_points) if ps.key_points else ""
                )
                content = f"{ps.title}\n\n{ps.summary}"
                if key_points:
                    content += f"\n\nKey points:\n{key_points}"
                gathered.append(
                    GatheredEvidence(
                        content=content,
                        source_ref=ps.url,
                        source_type="web_search",
                        limitations=[
                            "AI-summarized content; original source text not available"
                        ],
                        quality_score=ps.relevance_score,
                        quality_metadata={
                            "relevance_score": ps.relevance_score,
                            "title": ps.title,
                        },
                    )
                )

        # Strategy 3: Fall back to synthesized EvidenceReport
        if not gathered:
            output = result.output
            if output:
                summary = output.evidence_summary or ""
                sources = output.sources or []
                if summary:
                    source_refs = [str(s) for s in sources] if sources else [query]
                    gathered.append(
                        GatheredEvidence(
                            content=summary,
                            source_ref="; ".join(source_refs[:3]),
                            source_type="web_search",
                            limitations=[
                                "AI-synthesized summary; no direct source text available"
                            ],
                            quality_score=0.4,
                        )
                    )

        # Strategy 4: No results
        if not gathered:
            gathered.append(
                GatheredEvidence(
                    content=f"Web search for '{query}' returned no usable results.",
                    source_ref=query,
                    source_type="web_search",
                    limitations=["No results found"],
                    quality_score=0.0,
                )
            )

        return gathered


class CompositeGatherer:
    """Routes evidence queries to registered providers with web search fallback.

    Satisfies the EvidenceGatherer protocol. Providers are keyed by source_type.
    Unknown source_types fall back to web search.
    """

    def __init__(
        self,
        web_search: Any,  # accepts WebSearchGatherer or any compatible gatherer
        providers: Optional[dict] = None,
    ):
        self._web_search = web_search
        self._providers: dict = dict(providers) if providers else {}

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

        # Unknown source_type → explicit web-search default route
        return await self._web_search.gather(source_type, query)


def get_default_gatherer(
    *,
    model: str,
    providers: Optional[dict] = None,
    embedding_model: Optional[str] = None,
) -> Optional[EvidenceGatherer]:
    """Create a default evidence gatherer if deep_research is available.

    Args:
        model: LLM model string for web search analysis.
        providers: Optional dict of named providers (e.g., from
            ``andamentum.epistemic.providers.get_biomedical_providers()``).
            When provided, returns a CompositeGatherer that routes
            by source_type and falls back to web search.
        embedding_model: Optional embedding model for passage extraction.

    Returns None if deep_research is not installed, allowing callers
    to fall back gracefully.
    """
    try:
        import andamentum.deep_research.orchestrator  # noqa: F401

        web = WebSearchGatherer(model=model, embedding_model=embedding_model)
        if providers:
            return CompositeGatherer(web, providers)  # type: ignore[return-value]
        return web
    except ImportError:
        logger.info("deep_research not available — evidence gathering disabled")
        return None
