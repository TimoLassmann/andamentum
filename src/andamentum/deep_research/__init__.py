"""Deep research system — web research, novelty checking, and URL summarisation.

Three end-user one-shot entry points:

  • :func:`run_research` — multi-iteration search/fetch/synthesis against
    a research question.
  • :func:`run_novelty_check` — claim → web research → novelty verdict.
  • :func:`run_fetch` — single URL → structured summary.

Everything else (extraction primitives, SearxNG manager, circuit
breaker, agent registry) is internal infrastructure that the three
one-shots compose. They remain importable for advanced users who need
them.
"""

# === End-user one-shots ===
from .fetch import run_fetch
from .novelty import run_novelty_check
from .orchestrator import run_research

# === Internal infrastructure (advanced use) ===
from .circuit_breaker import CircuitBreaker, get_searxng_breaker
from .content_extractor import extract_content, extract_html, extract_pdf
from .searxng import SearxngManager, check_health as check_searxng_health
from .verification import verify_sources

# === Result / data types ===
from .agents import AGENT_REGISTRY, AgentDefinition
from .circuit_breaker import CircuitOpenError
from .content_extractor import ExtractionError
from .models import (
    EvidenceItem,
    EvidenceReport,
    FetchedPage,
    FetchPlan,
    FetchResults,
    FetchSummary,
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
    # End-user one-shots
    "run_fetch",
    "run_novelty_check",
    "run_research",
    # Internal infrastructure (advanced use)
    "CircuitBreaker",
    "SearxngManager",
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
    "FetchSummary",
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
