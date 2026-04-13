"""Deep research system — models, agents, and orchestration for web research.

Standalone package: ``pip install mosaic-deep-research`` for core models.
Add ``[llm]`` extra for agent runner: ``pip install mosaic-deep-research[llm]``.
"""

from .state import ResearchState
from .models import (
    SearchQuery,
    SearchResult,
    SearchPlan,
    FetchedPage,
    FetchResults,
    FetchPlan,
    PageSummary,
    GapAnalysis,
    EvidenceItem,
    EvidenceReport,
    ResearchErrors,
    ResearchResult,
)
from .circuit_breaker import CircuitBreaker, CircuitOpenError, get_searxng_breaker
from .searxng import SearxngManager, check_health as check_searxng_health
from .verification import verify_sources

# Content extraction
from .content_extractor import extract_html, extract_pdf, extract_content, ExtractionError

# Novelty checking (submodule)
from .novelty import check_novelty, NoveltyReport, NoveltyAssessment, SimilarWork, Relevance

# Agent definitions (Python-native)
from .agents import AgentDefinition, AGENT_REGISTRY

__version__ = "0.1.0"

DEFAULT_MODEL = "ollama:gpt-oss:20b"

__all__ = [
    "ResearchState",
    "SearchQuery",
    "SearchResult",
    "SearchPlan",
    "FetchedPage",
    "FetchResults",
    "FetchPlan",
    "PageSummary",
    "GapAnalysis",
    "EvidenceItem",
    "EvidenceReport",
    "ResearchErrors",
    "ResearchResult",
    "CircuitBreaker",
    "CircuitOpenError",
    "get_searxng_breaker",
    "SearxngManager",
    "check_searxng_health",
    "verify_sources",
    # Novelty
    "check_novelty",
    "NoveltyReport",
    "NoveltyAssessment",
    "SimilarWork",
    "Relevance",
    # Agents
    "AgentDefinition",
    "AGENT_REGISTRY",
    "DEFAULT_MODEL",
    # Content extraction
    "extract_html",
    "extract_pdf",
    "extract_content",
    "ExtractionError",
]
