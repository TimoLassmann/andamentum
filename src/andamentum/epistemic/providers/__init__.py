"""Evidence providers for the epistemic system.

Registry-based provider discovery. Each provider implements:
- check_health() → CheckResult
- gather(query) → list[GatheredEvidence]

Usage:
    from epistemic.providers import get_all_providers, get_provider

    # Get all providers
    providers = get_all_providers()

    # Get a specific provider
    pubmed = get_provider("pubmed")
    results = await pubmed.gather("BRCA1 breast cancer")
"""

from __future__ import annotations

from typing import Any

from .openalex import OpenAlexProvider, OpenAlexQualityScorer
from .monarch import MonarchProvider
from .pubmed import PubMedProvider
from .biorxiv import BioRxivProvider
from .clinicaltrials import ClinicalTrialsProvider
from .chembl import ChEMBLProvider
from .open_targets import OpenTargetsProvider

# ── Provider Registry ────────────────────────────────────────────────────────

PROVIDER_REGISTRY: dict[str, type] = {}
PROVIDER_DESCRIPTIONS: dict[str, str] = {}


def register_provider(name: str, cls: type, description: str = "") -> None:
    """Register a provider class by name with an optional domain description."""
    PROVIDER_REGISTRY[name] = cls
    if description:
        PROVIDER_DESCRIPTIONS[name] = description


def get_source_catalogue() -> str:
    """Build a formatted catalogue of available providers for the planning agent.

    Returns a markdown-formatted list of provider names and domain descriptions.
    """
    lines = []
    for name in sorted(PROVIDER_REGISTRY):
        desc = PROVIDER_DESCRIPTIONS.get(name, "")
        if desc:
            lines.append(f"- **{name}**: {desc}")
        else:
            lines.append(f"- **{name}**")
    lines.append("- **web_search**: General-purpose web search with evidence synthesis (always available as fallback)")
    return "\n".join(lines)


def get_provider(name: str, **kwargs: Any) -> Any:
    """Get a provider instance by name."""
    cls = PROVIDER_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(PROVIDER_REGISTRY.keys()))
        raise KeyError(f"Unknown provider: {name}. Available: {available}")
    return cls(**kwargs)


def get_all_providers(**kwargs: Any) -> dict[str, Any]:
    """Get instances of all registered providers."""
    return {name: cls(**kwargs) for name, cls in PROVIDER_REGISTRY.items()}


def get_biomedical_providers() -> dict[str, Any]:
    """Get all providers suitable for biomedical research.

    Backward-compatible convenience function.
    """
    return get_all_providers()


# ── Register built-in providers ──────────────────────────────────────────────

register_provider("openalex", OpenAlexProvider, "General academic literature across all disciplines — papers, citations, abstracts")
register_provider("pubmed", PubMedProvider, "Biomedical and life sciences literature — clinical studies, reviews, trials")
register_provider("biorxiv", BioRxivProvider, "Biology and medicine preprints — not yet peer-reviewed research")
register_provider("clinicaltrials", ClinicalTrialsProvider, "Clinical trial registry — trial designs, endpoints, enrollment, results")
register_provider("chembl", ChEMBLProvider, "Drug compounds and bioactivity data — IC50, mechanisms, drug targets")
register_provider("monarch", MonarchProvider, "Gene-disease associations — curated links between genes, diseases, phenotypes")
register_provider("open_targets", OpenTargetsProvider, "Drug target evidence — genetic associations, pathways, known drugs for targets")


__all__ = [
    # Provider classes
    "OpenAlexProvider",
    "OpenAlexQualityScorer",
    "MonarchProvider",
    "PubMedProvider",
    "BioRxivProvider",
    "ClinicalTrialsProvider",
    "ChEMBLProvider",
    "OpenTargetsProvider",
    # Registry
    "PROVIDER_REGISTRY",
    "PROVIDER_DESCRIPTIONS",
    "register_provider",
    "get_provider",
    "get_all_providers",
    "get_biomedical_providers",
    "get_source_catalogue",
]
