# Evidence Provider Specification

This document defines the exact interface, conventions, and patterns for
evidence providers in the epistemic system. Follow it precisely when adding
a new provider or modifying an existing one.

A provider retrieves evidence from an external source and structures it for
the epistemic pipeline. Providers never assess quality, never filter by
relevance, and never truncate content. They retrieve and structure — the
system does everything else.

## Interface

A provider is a class with exactly two async methods:

```python
class YourProvider:
    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    async def check_health(self) -> CheckResult:
        ...

    async def gather(self, query: str) -> list[GatheredEvidence]:
        ...
```

There is no base class. The system uses duck typing — if your class has
`check_health` and `gather` with the right signatures, it works.

## Module structure

```python
"""Your Provider Name.

One-line description of what it searches.

API docs: https://...
Authentication: None required / API key required / etc.

Architecture: Layer 1 (standalone package)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..preflight import CheckResult

from ..operations import GatheredEvidence

logger = logging.getLogger(__name__)

YOUR_API = "https://api.example.com/v1"
```

Heavy imports (`httpx`, `time`) go inside the methods that use them, not at
module level. This keeps import-time fast and avoids failures when optional
dependencies are missing.

## `gather()` method

### Signature and return type

```python
async def gather(self, query: str) -> list[GatheredEvidence]:
```

Always returns a list. On error, return an empty list — never `None`, never
raise. The outer try-except catches everything:

```python
async def gather(self, query: str) -> list[GatheredEvidence]:
    import httpx

    gathered: list[GatheredEvidence] = []

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # search, parse, build GatheredEvidence items
            ...
    except Exception as e:
        logger.warning(f"YourProvider query failed for '{query}': {e}")

    return gathered
```

### GatheredEvidence fields

Every item you return must be a `GatheredEvidence` with these fields:

**`content`** (required) — Human-readable summary. This is what LLM agents
read. Include the title, key authors or entities, and the substance of the
finding. Never truncate. Build it from parts:

```python
content_parts = [title]
if authors:
    if len(authors) > 5:
        content_parts.append(
            f"Authors: {', '.join(authors[:5])} (et al, {len(authors)} authors total)"
        )
    else:
        content_parts.append(f"Authors: {', '.join(authors)}")
if abstract:
    content_parts.append(f"\n{abstract}")
gathered.append(GatheredEvidence(
    content="\n".join(content_parts),
    ...
))
```

**`source_ref`** (required) — The single primary identifier for this evidence.
Use DOI when available, otherwise PMID, NCT ID, ChEMBL ID, or URL. Never
combine multiple identifiers into one string.

```python
# Good
source_ref="doi:10.1038/s41586-024-07487-w"
source_ref="PMID:38437170"
source_ref="NCT04381936"
source_ref="CHEMBL941"

# Bad — multi-part breaks deduplication
source_ref="PMID:38437170 DOI:10.1038/..."
```

**`source_type`** (required) — Your provider's lowercase name, matching the
key used in `register_provider()`. Examples: `"pubmed"`, `"chembl"`,
`"clinicaltrials"`, `"monarch_initiative"`, `"open_targets"`, `"openalex"`,
`"biorxiv"`.

**`evidence_kind`** — What type of evidence this is. Use one of the
established kinds:

| Kind | When to use |
|------|-------------|
| `"literature"` | Peer-reviewed published paper |
| `"preprint"` | Not-yet-peer-reviewed manuscript |
| `"clinical_trial"` | Clinical study record |
| `"bioactivity"` | Compound-target interaction data |
| `"genetic_association"` | Gene-disease or gene-phenotype link |
| `"entity_metadata"` | Entity description (gene, disease, etc.) |
| `"literature_mining"` | Text-mined from literature (not curated) |
| `"genetic_evidence"` | GWAS or genetic association evidence |
| `"somatic_evidence"` | Somatic mutation evidence |
| `"animal_model"` | Evidence from model organisms |
| `"clinical_evidence"` | Clinical/drug evidence |
| `"curated_genetic"` | Manually curated genetic evidence |
| `"database_record"` | Generic structured database entry |

If none fit, use `"database_record"` as default. New kinds can be added when
a provider genuinely needs one, but prefer reusing existing kinds.

**`identifiers`** — Dict of all cross-reference IDs you can extract. These
are used for deduplication across providers. Include every ID type available:

```python
identifiers={"pmid": "38437170", "doi": "10.1038/s41586-024-07487-w"}
identifiers={"nct_id": "NCT04381936"}
identifiers={"chembl_id": "CHEMBL941", "smiles": "CC(=O)Oc1ccccc1C(=O)O"}
identifiers={"subject_id": "HGNC:1100", "object_id": "MONDO:0007254"}
```

**`structured_data`** — Dict preserving provider-specific fields exactly as
the API returns them. This is where you put everything the system might need
downstream: trial phase, IC50 values, gene symbols, association scores,
publication dates, author lists. Don't rename or reshape API fields — preserve
them:

```python
structured_data={
    "title": title,
    "phase": phase,
    "status": status,
    "enrollment": enrollment,
    "primary_endpoints": primary_endpoints,
    "sponsor": sponsor,
}
```

**`quality_score`** — Always `None`. Providers do not assess quality.
The system's quality agent scores evidence after gathering. This is
non-negotiable.

```python
quality_score=None,  # Always. No exceptions.
```

**`quality_metadata`** — Dict of raw metadata the quality agent can use
to make its assessment. Include factual attributes like publication type,
journal name, trial phase, evidence count — things that inform quality
without judging it:

```python
quality_metadata={
    "publication_types": ["Randomized Controlled Trial"],
    "journal": "The Lancet",
}
```

**`limitations`** — List of caveats the system should know about. Only
include genuine limitations, not generic disclaimers. Empty list if none:

```python
# Good — specific, actionable
limitations=["Preprint — not peer-reviewed"]
limitations=["This is a NEGATED association"]
limitations=["Text-mined from literature; not manually curated"]

# Bad — generic noise
limitations=["Results may not be comprehensive"]
```

## `check_health()` method

Health checks verify API reachability using a real query that exercises the
same code path as `gather()`. This catches parameter format bugs that a
dummy ping would miss.

```python
async def check_health(self) -> "CheckResult":
    """Test API reachability."""
    import time
    import httpx
    from ..preflight import CheckResult

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Use the SAME parameter format as gather()
            response = await client.get(
                f"{YOUR_API}/search",
                params={"q": "test", "limit": 1},
            )
            elapsed = (time.monotonic() - t0) * 1000
            if response.status_code == 200:
                return CheckResult(
                    name="YourProvider",
                    status="pass",
                    message=f"API reachable ({elapsed:.0f}ms)",
                    elapsed_ms=elapsed,
                )
            return CheckResult(
                name="YourProvider",
                status="fail",
                message=f"HTTP {response.status_code}",
                elapsed_ms=elapsed,
            )
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="YourProvider",
            status="fail",
            message=str(e),
            elapsed_ms=elapsed,
        )
```

Key rules:
- Timeout: 10 seconds
- Use production parameter format (not a simplified test query)
- `CheckResult.name`: your provider class name (e.g., `"PubMedProvider"`)
- `CheckResult.status`: `"pass"` or `"fail"`
- Always return a `CheckResult`, never raise

## Registration

Register your provider in `providers/__init__.py`:

```python
from .your_provider import YourProvider

register_provider(
    "your_provider",
    YourProvider,
    description=(
        "Description of what this provider searches and when to use it. "
        "Be specific about the domain. Include 3-4 example queries. "
        "Example queries: 'query one', 'query two', 'query three'."
    ),
    query_guidance=(
        "How the query reaches the API (e.g., goes to `/search` `q` "
        "parameter). Native syntax supported (Boolean, field operators, "
        "phrase quoting, IDs). Catalogue of 5-7 syntactically distinct "
        "query styles that all work — frame as 'all of these work', not "
        "'this is optimal', so the formulator varies its output across "
        "calls. Note any operators that are silently ignored."
    ),
)
```

The description is shown to the LLM planning agent that decides which
providers to query. Write it to help the agent make good routing decisions —
be specific about what this provider covers and what it does not.

Also add 6-8 example queries to `PROVIDER_EXAMPLES` in the same file. These
are used by the semantic routing system for embedding-based provider matching.

## Error handling

1. **`gather()` catches all exceptions** and returns an empty list. Log at
   WARNING level: `logger.warning(f"Provider query failed for '{query}': {e}")`

2. **Private helper methods** (e.g., `_parse_result`, `_get_details`) can
   either propagate exceptions to the outer try-except or catch internally
   and return `None` / empty list. Both patterns are fine.

3. **`check_health()` catches all exceptions** and returns
   `CheckResult(status="fail", ...)`. Never raise from a health check.

4. **Never truncate content.** If the API returns a 5000-word abstract,
   include the full 5000-word abstract.

5. **HTTP errors return empty results.** If the API returns 500, return `[]`.
   Don't retry — the scheduler will re-run the operation if needed.

## Testing

Add tests in `tests/test_providers.py` using the existing `MockTransport`
pattern. Each provider needs at minimum:

1. **`test_gather_returns_gathered_evidence`** — Mock a successful API
   response and verify the returned `GatheredEvidence` objects have correct
   `source_type`, `quality_score is None`, and populated content.

2. **`test_health_check`** — Mock the API and verify `CheckResult` with
   `status="pass"`. Check that the health check uses the same parameter
   format as production.

3. **`test_error_returns_empty_list`** — Mock a 500 response and verify
   `gather()` returns `[]` without raising.

```python
class TestYourProvider:
    async def test_gather_returns_gathered_evidence(self, monkeypatch):
        transport = MockTransport(responses={
            "/api/search": {"results": [...]},
        })
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        provider = YourProvider(max_results=5)
        results = await provider.gather("test query")

        assert len(results) >= 1
        for r in results:
            assert isinstance(r, GatheredEvidence)
            assert r.source_type == "your_provider"
            assert r.quality_score is None
```

## Checklist

Before submitting a new provider:

- [ ] Class has `__init__(self, max_results: int = 10)`
- [ ] `check_health()` returns `CheckResult`, uses production params, timeout 10s
- [ ] `gather()` returns `list[GatheredEvidence]`, outer try-except, timeout 30s
- [ ] `quality_score=None` on every `GatheredEvidence`
- [ ] `source_ref` is a single identifier (not multi-part)
- [ ] `source_type` matches the registration key
- [ ] `content` is human-readable, never truncated
- [ ] `identifiers` includes all available cross-reference IDs
- [ ] `structured_data` preserves raw API fields
- [ ] `limitations` lists real caveats (or empty list)
- [ ] Heavy imports (`httpx`, `time`) inside methods, not at module level
- [ ] `CheckResult` imported under `TYPE_CHECKING`
- [ ] Registered in `providers/__init__.py` with description
- [ ] Example queries added to `PROVIDER_EXAMPLES`
- [ ] Tests added in `tests/test_providers.py`
- [ ] No content truncation anywhere
- [ ] No quality pre-computation anywhere
