"""Per-provider claim-and-query fixtures for the dispatch-quality benchmark.

This file is a thin re-export: the actual examples live on each provider
class as the ``query_examples`` class attribute (the dispatch agent reads
them as in-context teaching at runtime). The benchmark needs to iterate
across all providers and partition by in-domain vs out-of-domain, so this
module exposes those views.

Single source of truth: ``Provider.query_examples`` on each registered
provider class.
"""

from __future__ import annotations

from andamentum.epistemic.providers import PROVIDER_REGISTRY


def all_examples() -> dict[str, list[tuple[str, str | None]]]:
    """Return ``{provider_name: query_examples}`` for every registered provider.

    Reads ``query_examples`` from each registered provider class. Phase
    3.3 populates these; if any are empty, the benchmark reports zero
    in-domain and zero out-of-domain claims for that provider (a useful
    signal that the examples haven't been authored yet).
    """
    return {
        name: list(getattr(cls, "query_examples", []))
        for name, cls in PROVIDER_REGISTRY.items()
    }


def get_in_domain_claims(provider_name: str) -> list[str]:
    """Claims with non-None native queries — the in-domain set."""
    cls = PROVIDER_REGISTRY.get(provider_name)
    if cls is None:
        return []
    return [c for c, q in cls.query_examples if q is not None]


def get_out_of_domain_claims(provider_name: str) -> list[str]:
    """Claims with None queries — the abstain-expected set."""
    cls = PROVIDER_REGISTRY.get(provider_name)
    if cls is None:
        return []
    return [c for c, q in cls.query_examples if q is None]
