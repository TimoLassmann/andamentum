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
PROVIDER_DESCRIPTIONS: dict[str, str] = {}
PROVIDER_QUERY_GUIDANCE: dict[str, str] = {}


def register_provider(
    name: str,
    cls: type,
    description: str = "",
    query_guidance: str = "",
) -> None:
    """Register a provider class.

    ``description`` describes content scope and when to pick this provider —
    read by the ranker / dispatch agent. ``query_guidance`` describes the
    provider's native query language with a catalogue of valid styles — read
    by the legacy formulator (still load-bearing through Phases 1–4 of the
    description-driven-dispatch refactor, removed in Phase 5).

    **Migration shim (description-driven-dispatch Phase 1).** This function
    is the bridge between the legacy "data lives in the registry call" world
    and the post-refactor "data lives on the provider class" world. Both work:

    - If a provider class declares ``description`` and/or ``query_guidance``
      as class attributes, those are used preferentially (post-refactor
      providers do this).
    - If kwargs are passed explicitly to ``register_provider``, those win
      over class attributes (legacy migration not yet complete for this
      provider).
    - If neither, the registry-level value is empty.

    The dispatch-agent-readable contract — ``description``, ``query_examples``,
    ``output_kind``, ``independence_group``, ``provider_contract_version`` —
    lives entirely on the provider class. The shim does not consume those
    fields; they are read directly by the dispatch path in Phase 2 onwards.
    """
    PROVIDER_REGISTRY[name] = cls

    # Prefer class attributes (post-refactor pattern) over kwargs (legacy).
    effective_description = description or getattr(cls, "description", "")
    effective_query_guidance = query_guidance or getattr(cls, "query_guidance", "")

    if effective_description:
        PROVIDER_DESCRIPTIONS[name] = effective_description
    if effective_query_guidance:
        PROVIDER_QUERY_GUIDANCE[name] = effective_query_guidance


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
    lines.append(
        "- **web_search**: General-purpose web search with evidence synthesis (always available as fallback)"
    )
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


# PROVIDER_EXAMPLES was deleted as part of the description-driven-dispatch
# refactor (Phase 1, 2026-05-12). Its only consumer was the deleted
# `provider_routing.py` module. Per-provider example queries are now
# colocated on each provider class as ``query_examples`` (populated in
# Phase 2 when the dispatch agent is built).


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
    "PROVIDER_DESCRIPTIONS",
    "PROVIDER_QUERY_GUIDANCE",
    "register_provider",
    "get_provider",
    "get_all_providers",
    "get_biomedical_providers",
    "get_source_catalogue",
]
