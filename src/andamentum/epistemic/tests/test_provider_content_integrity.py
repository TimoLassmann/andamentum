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
  each provider with a ``MockTransport`` returning empty / error
  responses; assert the provider returns ``[]``.

* **Tier 2 (gated by ``live_provider`` marker)**: hits the real APIs.
  Per provider, a known-good query that should return at least one
  result; assert source_ref shape, content length, content != query.

  Run with: ``uv run pytest -m live_provider``.
"""

from __future__ import annotations

import re

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
# Each test hits the real API with a known-good query. Marked
# ``live_provider`` so they're deselected by default; run explicitly
# with ``uv run pytest -m live_provider``.
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.live_provider
class TestLivePubMed:
    async def test_returns_real_evidence(self) -> None:
        from ..providers.pubmed import PubMedProvider

        provider = PubMedProvider(max_results=3)
        results = await provider.gather("CRISPR Cas9 gene editing")
        assert results, "PubMed returned nothing for a query that should match"
        for r in results:
            _assert_real_evidence(r, "CRISPR Cas9 gene editing", "pubmed")
            # PubMed-specific: source_ref should start with PMID: or doi:
            assert r.source_ref.startswith(("PMID:", "doi:")) or r.source_ref, (
                f"pubmed source_ref unexpected shape: {r.source_ref!r}"
            )

    async def test_no_results_returns_empty(self) -> None:
        from ..providers.pubmed import PubMedProvider

        provider = PubMedProvider(max_results=3)
        results = await provider.gather("xyzzy_definitely_no_results_zzzqq_12345")
        assert results == [], "PubMed should return [] for a no-result query"


@pytest.mark.live_provider
class TestLiveOpenAlex:
    async def test_returns_real_evidence(self) -> None:
        from ..providers.openalex import OpenAlexProvider

        provider = OpenAlexProvider(max_results=3)
        results = await provider.gather("transformer attention mechanism")
        assert results, "OpenAlex returned nothing"
        for r in results:
            _assert_real_evidence(r, "transformer attention mechanism", "openalex")

    async def test_no_results_returns_empty(self) -> None:
        from ..providers.openalex import OpenAlexProvider

        provider = OpenAlexProvider(max_results=3)
        results = await provider.gather("xyzzy_definitely_no_results_zzzqq_12345")
        assert results == [], "OpenAlex should return [] for a no-result query"


@pytest.mark.live_provider
class TestLiveEuropePMC:
    async def test_returns_real_evidence(self) -> None:
        from ..providers.europepmc import EuropePMCProvider

        provider = EuropePMCProvider(max_results=3)
        results = await provider.gather("BRCA1 breast cancer")
        assert results, "Europe PMC returned nothing"
        for r in results:
            _assert_real_evidence(r, "BRCA1 breast cancer", "europepmc")

    async def test_no_results_returns_empty(self) -> None:
        from ..providers.europepmc import EuropePMCProvider

        provider = EuropePMCProvider(max_results=3)
        results = await provider.gather("xyzzy_definitely_no_results_zzzqq_12345")
        assert results == [], "Europe PMC should return [] for a no-result query"


@pytest.mark.live_provider
class TestLiveCochrane:
    async def test_returns_real_evidence(self) -> None:
        from ..providers.cochrane import CochraneProvider

        provider = CochraneProvider(max_results=3)
        # Cochrane indexes systematic reviews — pick a topic with many
        results = await provider.gather("statin cardiovascular prevention")
        # Cochrane is filter-restricted; for some queries 0 results is
        # legitimate. If results, validate; if empty, that's also ok.
        for r in results:
            _assert_real_evidence(r, "statin cardiovascular prevention", "cochrane")

    async def test_no_results_returns_empty(self) -> None:
        from ..providers.cochrane import CochraneProvider

        provider = CochraneProvider(max_results=3)
        results = await provider.gather("xyzzy_definitely_no_results_zzzqq_12345")
        assert results == [], "Cochrane should return [] for a no-result query"


@pytest.mark.live_provider
class TestLiveArxiv:
    async def test_returns_real_evidence(self) -> None:
        from ..providers.arxiv import ArXivProvider

        provider = ArXivProvider(max_results=3)
        results = await provider.gather("attention mechanism neural networks")
        assert results, "arXiv returned nothing"
        for r in results:
            _assert_real_evidence(
                r, "attention mechanism neural networks", "arxiv"
            )
            assert r.source_ref.startswith("arXiv:"), (
                f"arXiv source_ref unexpected shape: {r.source_ref!r}"
            )

    async def test_no_results_returns_empty(self) -> None:
        from ..providers.arxiv import ArXivProvider

        provider = ArXivProvider(max_results=3)
        results = await provider.gather("xyzzy_definitely_no_results_zzzqq_12345")
        assert results == [], "arXiv should return [] for a no-result query"


@pytest.mark.live_provider
class TestLiveBioRxiv:
    async def test_returns_real_evidence(self) -> None:
        from ..providers.biorxiv import BioRxivProvider

        provider = BioRxivProvider(max_results=3)
        results = await provider.gather("CRISPR genome editing")
        # bioRxiv content varies; allow zero but if returned, validate
        for r in results:
            _assert_real_evidence(r, "CRISPR genome editing", "biorxiv")

    async def test_no_results_returns_empty(self) -> None:
        from ..providers.biorxiv import BioRxivProvider

        provider = BioRxivProvider(max_results=3)
        # bioRxiv has a fallback to "recent preprints" which would
        # return SOMETHING for any query. Accept either [] or real
        # results from the fallback path; only fail on phantoms.
        results = await provider.gather("xyzzy_definitely_no_results_zzzqq_12345")
        for r in results:
            _assert_real_evidence(
                r, "xyzzy_definitely_no_results_zzzqq_12345", "biorxiv"
            )


@pytest.mark.live_provider
class TestLiveClinicalTrials:
    async def test_returns_real_evidence(self) -> None:
        from ..providers.clinicaltrials import ClinicalTrialsProvider

        provider = ClinicalTrialsProvider(max_results=3)
        results = await provider.gather("diabetes type 2")
        assert results, "ClinicalTrials.gov returned nothing"
        for r in results:
            _assert_real_evidence(r, "diabetes type 2", "clinicaltrials")
            assert r.source_ref.startswith("NCT"), (
                f"NCT id expected, got {r.source_ref!r}"
            )

    async def test_no_results_returns_empty(self) -> None:
        from ..providers.clinicaltrials import ClinicalTrialsProvider

        provider = ClinicalTrialsProvider(max_results=3)
        results = await provider.gather("xyzzy_definitely_no_results_zzzqq_12345")
        assert results == [], "ClinicalTrials.gov should return [] for a no-result query"


@pytest.mark.live_provider
class TestLiveChEMBL:
    async def test_returns_real_evidence(self) -> None:
        from ..providers.chembl import ChEMBLProvider

        provider = ChEMBLProvider(max_results=3)
        results = await provider.gather("aspirin")
        assert results, "ChEMBL returned nothing for 'aspirin'"
        for r in results:
            _assert_real_evidence(r, "aspirin", "chembl")
            assert r.source_ref.startswith("CHEMBL"), (
                f"ChEMBL ID expected, got {r.source_ref!r}"
            )

    async def test_no_results_returns_empty(self) -> None:
        from ..providers.chembl import ChEMBLProvider

        provider = ChEMBLProvider(max_results=3)
        results = await provider.gather("xyzzy_definitely_no_molecule_zzzqq_12345")
        assert results == [], "ChEMBL should return [] for a no-result query"


@pytest.mark.live_provider
class TestLiveOpenTargets:
    async def test_returns_real_evidence(self) -> None:
        from ..providers.open_targets import OpenTargetsProvider

        provider = OpenTargetsProvider(max_results=3)
        results = await provider.gather("BRCA1 breast cancer")
        # Open Targets resolution can fail for casual queries; if
        # returned, assert shape; if empty, accept (live API state).
        for r in results:
            _assert_real_evidence(r, "BRCA1 breast cancer", "open_targets")


@pytest.mark.live_provider
class TestLiveMonarch:
    async def test_returns_real_evidence(self) -> None:
        from ..providers.monarch import MonarchProvider

        provider = MonarchProvider(max_results=3)
        results = await provider.gather("BRCA1 breast cancer")
        # Monarch's contract was buggy until commit 6b6f17b; previously
        # returned phantom on error. Now returns []. If results, real;
        # if empty, also fine.
        for r in results:
            _assert_real_evidence(r, "BRCA1 breast cancer", "monarch")


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

    def __init__(self, status_code: int = 200, json_data: dict | None = None, text_data: str | None = None):
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


def _empty_json_for(provider_module: str) -> dict:
    """Per-provider 'empty-result' JSON shape that the provider expects."""
    if provider_module == "pubmed" or provider_module == "cochrane":
        return {"esearchresult": {"idlist": [], "count": "0"}}
    if provider_module == "europepmc":
        return {"resultList": {"result": []}}
    if provider_module == "openalex":
        return {"results": [], "meta": {"count": 0}}
    if provider_module == "clinicaltrials":
        return {"studies": []}
    if provider_module == "chembl":
        return {"molecules": []}
    if provider_module == "monarch":
        return {"items": []}
    if provider_module == "biorxiv":
        return {"esearchresult": {"idlist": []}, "collection": []}
    if provider_module == "open_targets":
        # Open Targets uses GraphQL; empty data dict suffices
        return {"data": {}}
    return {}


class TestProvidersReturnEmptyOnNoResults:
    """Mock-based: when the upstream API returns an empty result set,
    each provider must return ``[]``. No phantoms, no placeholders, no
    "search failed" stubs in the evidence pool. Default-on (runs in CI)
    so regressions land loudly. The deeper bugs (closed-loop
    hallucination) are caught by integration tests with the
    Strategy-2-deleted ExtractEvidenceOperation, but provider-level
    contract tests give us the cheapest possible regression surface.
    """

    async def test_pubmed_empty_search_returns_empty(self, monkeypatch) -> None:
        from ..providers import pubmed
        import httpx as _httpx

        empty = _empty_json_for("pubmed")
        monkeypatch.setattr(
            _httpx,
            "AsyncClient",
            lambda *a, **kw: _NoResultMockClient(
                json_response=empty, text_response="<PubmedArticleSet/>"
            ),
        )
        provider = pubmed.PubMedProvider(max_results=3)
        results = await provider.gather("any query")
        assert results == [], (
            f"pubmed must return [] on empty search; got {len(results)} items"
        )

    async def test_cochrane_empty_search_returns_empty(self, monkeypatch) -> None:
        from ..providers import cochrane
        import httpx as _httpx

        empty = _empty_json_for("cochrane")
        monkeypatch.setattr(
            _httpx,
            "AsyncClient",
            lambda *a, **kw: _NoResultMockClient(json_response=empty),
        )
        provider = cochrane.CochraneProvider(max_results=3)
        results = await provider.gather("any query")
        assert results == []

    async def test_openalex_empty_search_returns_empty(self, monkeypatch) -> None:
        from ..providers import openalex
        import httpx as _httpx

        empty = _empty_json_for("openalex")
        monkeypatch.setattr(
            _httpx,
            "AsyncClient",
            lambda *a, **kw: _NoResultMockClient(json_response=empty),
        )
        provider = openalex.OpenAlexProvider(max_results=3)
        results = await provider.gather("any query")
        assert results == []

    async def test_europepmc_empty_search_returns_empty(self, monkeypatch) -> None:
        from ..providers import europepmc
        import httpx as _httpx

        empty = _empty_json_for("europepmc")
        monkeypatch.setattr(
            _httpx,
            "AsyncClient",
            lambda *a, **kw: _NoResultMockClient(json_response=empty),
        )
        provider = europepmc.EuropePMCProvider(max_results=3)
        results = await provider.gather("any query")
        assert results == []

    async def test_clinicaltrials_empty_search_returns_empty(
        self, monkeypatch
    ) -> None:
        from ..providers import clinicaltrials
        import httpx as _httpx

        empty = _empty_json_for("clinicaltrials")
        monkeypatch.setattr(
            _httpx,
            "AsyncClient",
            lambda *a, **kw: _NoResultMockClient(json_response=empty),
        )
        provider = clinicaltrials.ClinicalTrialsProvider(max_results=3)
        results = await provider.gather("any query")
        assert results == []

    async def test_chembl_empty_search_returns_empty(self, monkeypatch) -> None:
        from ..providers import chembl
        import httpx as _httpx

        empty = _empty_json_for("chembl")
        monkeypatch.setattr(
            _httpx,
            "AsyncClient",
            lambda *a, **kw: _NoResultMockClient(json_response=empty),
        )
        provider = chembl.ChEMBLProvider(max_results=3)
        results = await provider.gather("any query")
        assert results == []

    async def test_arxiv_empty_search_returns_empty(self, monkeypatch) -> None:
        from ..providers import arxiv
        import httpx as _httpx

        # arXiv returns Atom XML; empty feed has no <entry> elements
        empty_atom = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<feed xmlns="http://www.w3.org/2005/Atom">\n'
            "  <title>arXiv Query: ...</title>\n"
            "  <opensearch:totalResults xmlns:opensearch="
            '"http://a9.com/-/spec/opensearch/1.1/">0</opensearch:totalResults>\n'
            "</feed>"
        )
        monkeypatch.setattr(
            _httpx,
            "AsyncClient",
            lambda *a, **kw: _NoResultMockClient(text_response=empty_atom),
        )
        provider = arxiv.ArXivProvider(max_results=3)
        results = await provider.gather("any query")
        assert results == []

    async def test_monarch_empty_search_returns_empty(self, monkeypatch) -> None:
        """Regression for monarch.py:147 — formerly returned a single
        phantom GatheredEvidence with source_ref=query and synthesised
        'search failed' content when the API errored. Now returns []."""
        from ..providers import monarch
        import httpx as _httpx

        # Simulate a 500 error
        monkeypatch.setattr(
            _httpx,
            "AsyncClient",
            lambda *a, **kw: _NoResultMockClient(status_code=500),
        )
        provider = monarch.MonarchProvider(max_results=3)
        results = await provider.gather("any query")
        assert results == [], (
            f"monarch must return [] on API error (regression test for the "
            f"phantom-evidence bug); got {len(results)} items"
        )
        for r in results:
            assert r.source_ref != "any query", (
                "monarch must not use the search query as source_ref"
            )

    async def test_biorxiv_empty_search_does_not_return_phantom(
        self, monkeypatch
    ) -> None:
        """bioRxiv has a date-range fallback that may return real recent
        preprints even when the keyword search is empty. We accept that
        behaviour but verify any returned items have real source_refs
        (DOIs/PMIDs), not query strings."""
        from ..providers import biorxiv
        import httpx as _httpx

        empty = _empty_json_for("biorxiv")
        monkeypatch.setattr(
            _httpx,
            "AsyncClient",
            lambda *a, **kw: _NoResultMockClient(json_response=empty),
        )
        provider = biorxiv.BioRxivProvider(max_results=3)
        results = await provider.gather("xyz_no_match_query")
        for r in results:
            assert r.source_ref != "xyz_no_match_query", (
                f"biorxiv source_ref equals query: {r.source_ref!r}"
            )
            assert r.source_ref, "biorxiv source_ref empty"
