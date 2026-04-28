"""Deep research system — models, agents, and orchestration for web research."""

# === Functions you can wrap as agent tools ===
# `SearxngManager` and `CircuitBreaker` are classes — wrap their methods
# (`.start`, `.stop`, `.is_running`, `.allow_request`, `.record_failure` …) as tools.
from .circuit_breaker import CircuitBreaker, get_searxng_breaker
from .content_extractor import extract_content, extract_html, extract_pdf
from .novelty import check_novelty
from .searxng import SearxngManager, check_health as check_searxng_health
from .verification import verify_sources

# === Result/data types (returned by the above; not tools themselves) ===
from .agents import AGENT_REGISTRY, AgentDefinition
from .circuit_breaker import CircuitOpenError
from .content_extractor import ExtractionError
from .models import (
    EvidenceItem,
    EvidenceReport,
    FetchedPage,
    FetchPlan,
    FetchResults,
    GapAnalysis,
    GeneratorOutput,
    PageSummary,
    ResearchErrors,
    ResearchResult,
    SearchQuery,
    SearchResult,
    VerifierOutput,
)
from .novelty import (
    NoveltyAssessment,
    NoveltyReport,
    Relevance,
    SimilarWork,
)
from .state import ResearchState

__version__ = "0.1.0"

__all__ = [
    # Functions / callables
    "CircuitBreaker",
    "SearxngManager",
    "check_novelty",
    "check_searxng_health",
    "extract_content",
    "extract_html",
    "extract_pdf",
    "get_searxng_breaker",
    "verify_sources",
    # Data types
    "AGENT_REGISTRY",
    "AgentDefinition",
    "CircuitOpenError",
    "EvidenceItem",
    "EvidenceReport",
    "ExtractionError",
    "FetchPlan",
    "FetchResults",
    "FetchedPage",
    "GapAnalysis",
    "GeneratorOutput",
    "NoveltyAssessment",
    "NoveltyReport",
    "PageSummary",
    "Relevance",
    "ResearchErrors",
    "ResearchResult",
    "ResearchState",
    "SearchQuery",
    "SearchResult",
    "SimilarWork",
    "VerifierOutput",
]
