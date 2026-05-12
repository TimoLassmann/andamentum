"""Evidence providers for the epistemic system.

Registry-based provider discovery. Each provider implements:
- check_health() → CheckResult
- gather(query) → list[GatheredEvidence]

Usage:
    from andamentum.epistemic.providers import get_all_providers, get_provider

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
from .europepmc import EuropePMCProvider
from .cochrane import CochraneProvider
from .arxiv import ArXivProvider

# ── Provider Registry ────────────────────────────────────────────────────────

PROVIDER_REGISTRY: dict[str, type] = {}


def register_provider(name: str, cls: type) -> None:
    """Register a provider class.

    Data — ``description``, ``query_guidance``, ``query_examples``,
    ``output_kind``, ``independence_group``, ``provider_contract_version`` —
    lives on the provider class as class attributes and is read directly
    from the class by the dispatch agent at runtime. This function only
    indexes the class under ``name`` in ``PROVIDER_REGISTRY``.
    """
    PROVIDER_REGISTRY[name] = cls


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

# OpenAlex — description, query_guidance, output_kind, independence_group,
# provider_contract_version live on the OpenAlexProvider class (post-refactor
# pattern). The shim in register_provider pulls them off the class.
register_provider("openalex", OpenAlexProvider)
# PubMed — data lives on PubMedProvider class (post-refactor pattern).
register_provider("pubmed", PubMedProvider)
# bioRxiv — data lives on BioRxivProvider class.
register_provider("biorxiv", BioRxivProvider)
# ClinicalTrials.gov — data lives on ClinicalTrialsProvider class.
register_provider("clinicaltrials", ClinicalTrialsProvider)
# ChEMBL — data lives on ChEMBLProvider class.
register_provider("chembl", ChEMBLProvider)
# Monarch — data lives on MonarchProvider class.
register_provider("monarch", MonarchProvider)
# Open Targets — data lives on OpenTargetsProvider class.
register_provider("open_targets", OpenTargetsProvider)
# Europe PMC — data lives on EuropePMCProvider class.
register_provider("europepmc", EuropePMCProvider)
# Cochrane — data lives on CochraneProvider class.
register_provider("cochrane", CochraneProvider)
# arXiv — data lives on ArXivProvider class.
register_provider("arxiv", ArXivProvider)


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
    "EuropePMCProvider",
    "CochraneProvider",
    "ArXivProvider",
    # Registry
    "PROVIDER_REGISTRY",
    "register_provider",
    "get_provider",
    "get_all_providers",
    "get_biomedical_providers",
]
