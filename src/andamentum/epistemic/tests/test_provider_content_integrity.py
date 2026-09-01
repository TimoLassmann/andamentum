"""Provider content-integrity tests — both live and shape-validated.

This file exists because of SciFact case 781 v25, where a closed-loop
hallucination produced 6 phantom "supports" evidence pieces (source_ref =
search-query string, content = 100-230 chars paraphrasing the claim) that
outvoted 2 real "contradicts" pieces and gave a confidently-wrong verdict
on a CON-gold claim. Root cause was the agent-extraction fallback in
``ExtractEvidenceOperation`` (deleted in commit ee4bbb8). But the bug
class — providers returning placeholder content rather than honestly
empty results — could recur.

These tests guard against three failure modes per provider:

1. **Phantom-evidence on error**: when the API errors or returns no
   results, the provider must return ``[]``, not a placeholder
   ``GatheredEvidence`` with synthesised content (the monarch.py:147 bug,
   fixed in the same commit set).

2. **Search-query as source_ref**: ``source_ref`` should always be a
   real identifier (DOI, PMID, NCT ID, ChEMBL ID, URL, or paper title),
   never a search-query string. We assert ``source_ref`` does not equal
   ``query`` and matches a permissive ID-or-URL regex.

3. **Placeholder content**: ``content`` should be substantive (real
   abstract or summary), not a paraphrase of the query or a "search
   failed" notice. Cheap shape checks: minimum length and no exact
   query-equality.

Two test tiers:

* **Tier 1 (default-on, no internet)**: mock-based shape tests. Drive
  each provider with a mock transport returning empty / error
  responses; assert the provider returns ``[]``.

* **Tier 2 (gated by ``live_provider`` marker)**: hits the real APIs.
  Per provider, a known-good query that should return at least one
  result; assert source_ref shape, content length, content != query.

  Run with: ``uv run pytest -m live_provider``.

**Each tier is one parametrized test over a table of providers**, so a
new provider is a new row rather than a copied class. The contract is
written once; nobody can add a provider and forget to test it the same
way as the others.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field

import pytest

# Permissive regex covering the source_ref shapes all providers should
# produce: DOI / PMID / NCT / ChEMBL / arXiv / URL / paper-title (any
# string with a colon or slash, OR ≥ 10 chars without commas/parens
# typical of search-query syntax).
_REAL_ID_RE = re.compile(
    r"^("
    r"doi:.+|"
    r"PMID:\d+|"
    r"NCT\d+|"
    r"CHEMBL\d+|"
    r"arXiv:.+|"
    r"https?://.+|"
    r"http://.+|"
    r"[^\(\)\"]{15,}"  # paper title fallback (non-search-syntax, ≥15 chars)
    r")$"
)

_MIN_CONTENT_CHARS = 50  # realistic-content lower bound; phantoms in
# case 781 ranged 104–230 chars but were short and content-thin.
# Real abstracts run 500–3000+ chars. We're checking for "the
# provider returned SOMETHING substantive" rather than imposing a
# strict length floor.

#: Query that should match nothing anywhere.
_NO_MATCH_QUERY = "xyzzy_definitely_no_results_zzzqq_12345"

_MAX_RESULTS = 3


def _build_provider(module: str, cls: str):
    """Import a provider lazily and construct it.

    The import stays inside the call, as it did when every test did its
    own ``from ..providers.x import XProvider``: collecting this module
    must not import every provider.
    """
    imported = importlib.import_module(f"..providers.{module}", package=__package__)
    return getattr(imported, cls)(max_results=_MAX_RESULTS)


def _assert_real_evidence(item, query: str, source_kind: str) -> None:
    """Common assertions for any GatheredEvidence returned by any
    provider in a live test. Fails loudly if the item has phantom
    characteristics."""
    assert item.source_ref, f"{source_kind}: source_ref is empty"
    assert item.source_ref != query, (
        f"{source_kind}: source_ref equals the search query "
        f"({item.source_ref!r}) — provider returned a phantom"
    )
    assert _REAL_ID_RE.match(item.source_ref), (
        f"{source_kind}: source_ref does not look like a real identifier: "
        f"{item.source_ref!r}"
    )
    assert item.content, f"{source_kind}: content is empty"
    assert len(item.content) >= _MIN_CONTENT_CHARS, (
        f"{source_kind}: content too short ({len(item.content)} chars): "
        f"{item.content[:120]!r}"
    )
    assert item.content.strip() != query.strip(), (
        f"{source_kind}: content equals the search query — provider "
        f"returned a phantom paraphrase"
    )
    assert "search failed" not in item.content.lower(), (
        f"{source_kind}: content contains 'search failed' — provider "
        f"is returning error placeholders instead of []"
    )


# ─────────────────────────────────────────────────────────────────────
# Tier 2 — Live provider tests (gated)
#
# Each case hits the real API with a known-good query. Marked
# ``live_provider`` so they're deselected by default; run explicitly
# with ``uv run pytest -m live_provider``.
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LiveCase:
    """One provider's live contract."""

    name: str
    module: str
    cls: str
    query: str
    #: False where zero results is a legitimate live-API answer, so the
    #: test validates whatever came back instead of demanding some.
    expect_results: bool = True
    #: Extra shape the provider's own identifiers must have.
    source_ref_prefix: tuple[str, ...] = ()


LIVE_CASES: list[LiveCase] = [
    # PubMed carried an extra source_ref assertion of the form
    # `startswith((...)) or r.source_ref`, which `_assert_real_evidence`
    # already proves non-empty — it could never fail. Dropped rather
    # than tightened: strengthening a live assertion is a behaviour
    # change, not a refactor.
    LiveCase("pubmed", "pubmed", "PubMedProvider", "CRISPR Cas9 gene editing"),
    LiveCase(
        "openalex", "openalex", "OpenAlexProvider", "transformer attention mechanism"
    ),
    LiveCase("europepmc", "europepmc", "EuropePMCProvider", "BRCA1 breast cancer"),
    # Cochrane is filter-restricted; for some queries 0 results is legitimate.
    LiveCase(
        "cochrane",
        "cochrane",
        "CochraneProvider",
        "statin cardiovascular prevention",
        expect_results=False,
    ),
    LiveCase(
        "arxiv",
        "arxiv",
        "ArXivProvider",
        "attention mechanism neural networks",
        source_ref_prefix=("arXiv:",),
    ),
    # bioRxiv content varies; allow zero but if returned, validate.
    LiveCase(
        "biorxiv",
        "biorxiv",
        "BioRxivProvider",
        "CRISPR genome editing",
        expect_results=False,
    ),
    LiveCase(
        "clinicaltrials",
        "clinicaltrials",
        "ClinicalTrialsProvider",
        "diabetes type 2",
        source_ref_prefix=("NCT",),
    ),
    LiveCase(
        "chembl",
        "chembl",
        "ChEMBLProvider",
        "aspirin",
        source_ref_prefix=("CHEMBL",),
    ),
    # Open Targets resolution can fail for casual queries; if returned,
    # assert shape; if empty, accept (live API state).
    LiveCase(
        "open_targets",
        "open_targets",
        "OpenTargetsProvider",
        "BRCA1 breast cancer",
        expect_results=False,
    ),
    # Monarch's contract was buggy until commit 6b6f17b; previously
    # returned phantom on error. Now returns []. If results, real;
    # if empty, also fine.
    LiveCase(
        "monarch",
        "monarch",
        "MonarchProvider",
        "BRCA1 breast cancer",
        expect_results=False,
    ),
]


@pytest.mark.live_provider
@pytest.mark.parametrize("case", LIVE_CASES, ids=lambda c: c.name)
async def test_live_provider_returns_real_evidence(case: LiveCase) -> None:
    provider = _build_provider(case.module, case.cls)
    results = await provider.gather(case.query)
    if case.expect_results:
        assert results, f"{case.name} returned nothing for a query that should match"
    for item in results:
        _assert_real_evidence(item, case.query, case.name)
        if case.source_ref_prefix:
            assert item.source_ref.startswith(case.source_ref_prefix), (
                f"{case.name} source_ref unexpected shape: {item.source_ref!r}"
            )


#: Providers whose live no-result query must yield exactly ``[]``.
#: bioRxiv is absent on purpose — it has a date-range fallback that can
#: return real recent preprints for any query, so it gets its own test.
LIVE_NO_RESULT_CASES: list[LiveCase] = [
    case
    for case in LIVE_CASES
    if case.name not in ("biorxiv", "open_targets", "monarch")
]


@pytest.mark.live_provider
@pytest.mark.parametrize("case", LIVE_NO_RESULT_CASES, ids=lambda c: c.name)
async def test_live_provider_no_results_returns_empty(case: LiveCase) -> None:
    provider = _build_provider(case.module, case.cls)
    # ChEMBL indexes molecules, so the sentinel names one that cannot exist.
    query = (
        "xyzzy_definitely_no_molecule_zzzqq_12345"
        if case.name == "chembl"
        else _NO_MATCH_QUERY
    )
    results = await provider.gather(query)
    assert results == [], f"{case.name} should return [] for a no-result query"


@pytest.mark.live_provider
async def test_live_biorxiv_no_result_query_returns_no_phantoms() -> None:
    """bioRxiv falls back to recent preprints, so [] is not required.

    Accept either [] or real results from the fallback path; only fail
    on phantoms.
    """
    provider = _build_provider("biorxiv", "BioRxivProvider")
    results = await provider.gather(_NO_MATCH_QUERY)
    for item in results:
        _assert_real_evidence(item, _NO_MATCH_QUERY, "biorxiv")


# ─────────────────────────────────────────────────────────────────────
# Tier 1 — Mock-based no-result shape tests (default-on)
#
# These run in default CI without internet. We patch ``httpx.AsyncClient``
# at the provider's import point with a transport that returns
# ``200 OK`` with empty/no-result JSON. The contract being tested:
# providers MUST return ``[]`` rather than synthesising a placeholder.
# ─────────────────────────────────────────────────────────────────────


class _MockResponse:
    """Minimal httpx.Response stand-in for our test transport."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: dict | None = None,
        text_data: str | None = None,
    ):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text_data or ""

    def json(self) -> dict:
        return self._json_data


class _NoResultMockClient:
    """Mock httpx.AsyncClient that returns an empty-result response on
    every call. Used to simulate "search returned no hits" behaviour
    deterministically without internet."""

    def __init__(
        self,
        json_response: dict | None = None,
        text_response: str | None = None,
        status_code: int = 200,
        *args,
        **kwargs,
    ):
        self.json_response = json_response or {}
        self.text_response = text_response or ""
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def get(self, *_args, **_kwargs):
        return _MockResponse(
            status_code=self.status_code,
            json_data=self.json_response,
            text_data=self.text_response,
        )


#: arXiv returns Atom XML; an empty feed has no <entry> elements.
_EMPTY_ATOM = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<feed xmlns="http://www.w3.org/2005/Atom">\n'
    "  <title>arXiv Query: ...</title>\n"
    "  <opensearch:totalResults xmlns:opensearch="
    '"http://a9.com/-/spec/opensearch/1.1/">0</opensearch:totalResults>\n'
    "</feed>"
)


@dataclass(frozen=True)
class EmptyCase:
    """One provider's 'upstream returned nothing' contract."""

    name: str
    module: str
    cls: str
    #: Per-provider 'empty-result' JSON shape that the provider expects.
    json_response: dict = field(default_factory=dict)
    text_response: str = ""
    status_code: int = 200
    reason: str = ""


EMPTY_CASES: list[EmptyCase] = [
    EmptyCase(
        "pubmed",
        "pubmed",
        "PubMedProvider",
        json_response={"esearchresult": {"idlist": [], "count": "0"}},
        text_response="<PubmedArticleSet/>",
    ),
    EmptyCase(
        "cochrane",
        "cochrane",
        "CochraneProvider",
        json_response={"esearchresult": {"idlist": [], "count": "0"}},
    ),
    EmptyCase(
        "openalex",
        "openalex",
        "OpenAlexProvider",
        json_response={"results": [], "meta": {"count": 0}},
    ),
    EmptyCase(
        "europepmc",
        "europepmc",
        "EuropePMCProvider",
        json_response={"resultList": {"result": []}},
    ),
    EmptyCase(
        "clinicaltrials",
        "clinicaltrials",
        "ClinicalTrialsProvider",
        json_response={"studies": []},
    ),
    EmptyCase(
        "chembl",
        "chembl",
        "ChEMBLProvider",
        json_response={"molecules": []},
    ),
    EmptyCase("arxiv", "arxiv", "ArXivProvider", text_response=_EMPTY_ATOM),
    # Regression for monarch.py:147 — formerly returned a single phantom
    # GatheredEvidence with source_ref=query and synthesised 'search
    # failed' content when the API errored. Now returns []. This case
    # drives the ERROR path (500), not the empty-result path.
    EmptyCase(
        "monarch",
        "monarch",
        "MonarchProvider",
        status_code=500,
        reason="regression test for the phantom-evidence bug",
    ),
]


def _patch_httpx(monkeypatch, case: EmptyCase) -> None:
    import httpx as _httpx

    monkeypatch.setattr(
        _httpx,
        "AsyncClient",
        lambda *a, **kw: _NoResultMockClient(
            json_response=case.json_response,
            text_response=case.text_response,
            status_code=case.status_code,
        ),
    )


@pytest.mark.parametrize("case", EMPTY_CASES, ids=lambda c: c.name)
async def test_provider_returns_empty_on_no_results(monkeypatch, case: EmptyCase):
    """When the upstream API returns an empty result set (or errors), the
    provider must return ``[]``.

    No phantoms, no placeholders, no "search failed" stubs in the
    evidence pool. Default-on (runs in CI) so regressions land loudly.
    The deeper bugs (closed-loop hallucination) are caught by integration
    tests with the Strategy-2-deleted ExtractEvidenceOperation, but
    provider-level contract tests give us the cheapest possible
    regression surface.
    """
    _patch_httpx(monkeypatch, case)
    provider = _build_provider(case.module, case.cls)
    results = await provider.gather("any query")
    detail = f" ({case.reason})" if case.reason else ""
    assert results == [], (
        f"{case.name} must return [] on an empty/failed search{detail}; "
        f"got {len(results)} items"
    )


async def test_biorxiv_empty_search_does_not_return_phantom(monkeypatch) -> None:
    """bioRxiv has a date-range fallback that may return real recent
    preprints even when the keyword search is empty. We accept that
    behaviour but verify any returned items have real source_refs
    (DOIs/PMIDs), not query strings."""
    case = EmptyCase(
        "biorxiv",
        "biorxiv",
        "BioRxivProvider",
        json_response={"esearchresult": {"idlist": []}, "collection": []},
    )
    _patch_httpx(monkeypatch, case)
    provider = _build_provider(case.module, case.cls)
    results = await provider.gather("xyz_no_match_query")
    for item in results:
        assert item.source_ref != "xyz_no_match_query", (
            f"biorxiv source_ref equals query: {item.source_ref!r}"
        )
        assert item.source_ref, "biorxiv source_ref empty"
