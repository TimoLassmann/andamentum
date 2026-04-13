"""Evidence Router - Automatic source selection for evidence collection.

Uses SourceIndex to automatically select relevant evidence sources based on
the research query. Provides a unified interface for the workflow engine.

Usage:
    from andamentum.epistemic.evidence_router import EvidenceRouter

    router = await EvidenceRouter.create()

    # Auto-select sources for a query
    sources = await router.select_sources(
        query="What is the clinical significance of BRCA1 c.5266dupC?",
        top_k=3
    )
    # Returns: [
    #   {"provider": "knowledge_sources", "config": {"sources": ["clinvar"]}},
    #   {"provider": "web_search", "config": {}},
    # ]

    # Or get provider configs for workflow_engine
    evidence_strategy = await router.build_evidence_strategy(query)
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

try:
    from .source_index import SourceIndex, SourceMatch  # type: ignore[import-not-found]
    from .source_registry import SourceRegistry, BUILTIN_SOURCES  # type: ignore[import-not-found]
except ImportError:
    SourceIndex = None  # type: ignore[assignment,misc]
    SourceMatch = None  # type: ignore[assignment,misc]
    SourceRegistry = None  # type: ignore[assignment,misc]
    BUILTIN_SOURCES = []  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class EvidenceRouter:
    """Routes research queries to relevant evidence sources.

    Combines SourceRegistry (discovery) with SourceIndex (ranking) to
    automatically select the most relevant sources for any query.
    """

    def __init__(
        self,
        index: SourceIndex,
        registry: SourceRegistry,
    ):
        """Initialize with pre-built index and registry.

        Use EvidenceRouter.create() for async construction.
        """
        self.index = index
        self.registry = registry

    @classmethod
    async def create(
        cls,
        api_url: str = "http://localhost:8000",
        always_include_web: bool = True,
    ) -> "EvidenceRouter":
        """Create an EvidenceRouter with refreshed sources.

        Args:
            api_url: Knowledge Sources API URL
            always_include_web: Whether to always include web_search

        Returns:
            Ready-to-use EvidenceRouter
        """
        # Refresh registry from API
        registry = SourceRegistry(api_url=api_url)
        await registry.refresh()

        # Build index from all sources
        all_sources = registry.get_all_sources()
        if not all_sources:
            # Fallback to built-in only
            logger.warning("[EvidenceRouter] No dynamic sources, using built-in only")
            all_sources = list(BUILTIN_SOURCES)

        index = await SourceIndex.create(all_sources)

        logger.info(f"[EvidenceRouter] Created with {len(all_sources)} sources")
        return cls(index, registry)

    async def select_sources(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.01,
        always_include_web: bool = True,
    ) -> List[SourceMatch]:
        """Select relevant sources for a query.

        Args:
            query: Research question or claim
            top_k: Maximum sources to return
            min_score: Minimum relevance score
            always_include_web: Always include web_search if not selected

        Returns:
            List of SourceMatch objects
        """
        # Use async version for efficiency
        matches = await self.index.find_sources_async(
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

        # Ensure web_search is included if requested
        if always_include_web:
            web_ids = {m.source_id for m in matches if m.source_id == "web_search"}
            if not web_ids:
                # Add web_search at lower priority
                matches.append(
                    SourceMatch(
                        source_id="web_search",
                        score=0.01,  # Low but present
                        match_type="fallback",
                    )
                )

        return matches

    async def build_evidence_strategy(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """Build evidence_strategy for workflow engine.

        This is the main integration point. Returns a list of provider
        configurations that the workflow engine can execute directly.

        Args:
            query: Research question
            top_k: Maximum sources to use
            min_score: Minimum relevance score

        Returns:
            List of {"provider": str, "config": dict} for workflow engine
        """
        matches = await self.select_sources(
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

        strategy: List[Dict[str, Any]] = []
        knowledge_sources: List[str] = []

        for match in matches:
            source_id = match.source_id

            if source_id == "web_search":
                # Web search is its own provider
                strategy.append(
                    {
                        "provider": "web_search",
                        "config": {"depth": "standard"},
                    }
                )
            else:
                # Other sources go through knowledge_sources provider
                knowledge_sources.append(source_id)

        # Bundle knowledge sources into one provider call
        if knowledge_sources:
            strategy.insert(
                0,
                {
                    "provider": "knowledge_sources",
                    "config": {"sources": knowledge_sources},
                },
            )

        logger.info(
            f"[EvidenceRouter] Built strategy with {len(strategy)} providers: "
            f"{[s['provider'] for s in strategy]}"
        )

        return strategy

    def get_source_info(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get human-readable info about a source.

        Args:
            source_id: Source identifier

        Returns:
            Dict with name, description, keywords or None
        """
        source = self.registry.get_source_by_id(source_id)
        if not source:
            return None

        return {
            "id": source.id,
            "name": source.name,
            "description": source.description,
            "keywords": source.keywords,
            "entity_types": source.entity_types,
        }

    @property
    def available_sources(self) -> List[str]:
        """List of all available source IDs."""
        return [s.id for s in self.registry.get_all_sources()]

    def __repr__(self) -> str:
        return f"EvidenceRouter({len(self.index)} sources indexed)"
