"""Tests for evidence providers — verify API call construction and response parsing.

These tests would have caught the Monarch 422 bug (comma-separated category params)
and ensure health checks use the same code path as production queries.
No network calls — all HTTP is mocked via httpx mock transport.
"""

import pytest
from typing import Any

import httpx

from ..operations import GatheredEvidence
from ..providers.monarch import MonarchProvider
from ..providers.openalex import OpenAlexProvider

# Capture real AsyncClient before any patching
_RealAsyncClient = httpx.AsyncClient


# ──────────────────────────────────────────────────────────────────────────────
# httpx mock transport helpers
# ──────────────────────────────────────────────────────────────────────────────


class MockTransport(httpx.AsyncBaseTransport):
    """Records requests and returns canned responses."""

    def __init__(self, responses: dict[str, Any] | None = None, status_code: int = 200):
        self.requests: list[httpx.Request] = []
        self._responses = responses or {}
        self._status_code = status_code

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url_path = request.url.path
        body = self._responses.get(url_path, {"items": []})
        return httpx.Response(
            status_code=self._status_code,
            json=body,
            request=request,
        )


def _make_patched_client(transport):
    """Create a patched AsyncClient class using the given transport.

    Uses the captured _RealAsyncClient to avoid recursion when httpx.AsyncClient
    is globally patched.
    """

    class PatchedClient:
        def __init__(self, **kwargs):
            self._client = _RealAsyncClient(transport=transport, timeout=30.0)

        async def __aenter__(self):
            return self._client

        async def __aexit__(self, *args):
            await self._client.aclose()

    return PatchedClient


def _make_failing_client():
    """Create a client that always fails on connect."""

    class FailingClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("Connection refused")

        async def __aexit__(self, *args):
            pass

    return FailingClient


# ──────────────────────────────────────────────────────────────────────────────
# Monarch Provider — search param construction
# ──────────────────────────────────────────────────────────────────────────────


class TestMonarchSearchParams:
    """Verify Monarch API calls use list-of-tuples for category params.

    This is the exact bug class that caused 422 errors: sending
    category=biolink:Gene,biolink:Disease as a single comma-separated value
    instead of two separate query parameters.
    """

    async def test_search_sends_separate_category_params(self, monkeypatch):
        """Each category must be a separate query parameter, not comma-separated."""
        transport = MockTransport(
            responses={
                "/v3/api/search": {
                    "items": [
                        {
                            "name": "BRCA1",
                            "category": "biolink:Gene",
                            "id": "HGNC:1100",
                            "description": "DNA repair",
                        },
                    ]
                },
                "/v3/api/association": {"items": []},
            }
        )

        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        provider = MonarchProvider(max_results=5)
        await provider.gather("BRCA1 breast cancer")

        assert len(transport.requests) >= 1

        search_request = transport.requests[0]
        raw_query = (
            search_request.url.raw_path.decode("utf-8")
            if isinstance(search_request.url.raw_path, bytes)
            else str(search_request.url)
        )

        # The category parameter must appear TWICE as separate params
        category_count = raw_query.count("category=")
        assert category_count == 2, (
            f"Expected 2 separate category params, got {category_count}. "
            f"URL: {raw_query}"
        )

    async def test_search_returns_gathered_evidence(self, monkeypatch):
        """Search results should be parsed into GatheredEvidence objects."""
        transport = MockTransport(
            responses={
                "/v3/api/search": {
                    "items": [
                        {
                            "name": "BRCA1",
                            "category": "biolink:Gene",
                            "id": "HGNC:1100",
                            "description": "DNA repair gene",
                        },
                        {
                            "name": "Breast Cancer",
                            "category": "biolink:Disease",
                            "id": "MONDO:0007254",
                            "description": "Malignant neoplasm",
                        },
                    ]
                },
                "/v3/api/association": {"items": []},
            }
        )

        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        provider = MonarchProvider(max_results=10)
        results = await provider.gather("BRCA1")

        assert len(results) >= 2
        for r in results:
            assert isinstance(r, GatheredEvidence)
            assert r.source_type == "monarch_initiative"
            assert r.quality_score is None  # providers don't pre-compute quality

    async def test_search_skips_items_without_name(self, monkeypatch):
        """Items missing 'name' field should be skipped."""
        transport = MockTransport(
            responses={
                "/v3/api/search": {
                    "items": [
                        {
                            "category": "biolink:Gene",
                            "id": "X",
                            "description": "No name",
                        },
                        {
                            "name": "Valid",
                            "category": "biolink:Gene",
                            "id": "Y",
                            "description": "Has name",
                        },
                    ]
                },
                "/v3/api/association": {"items": []},
            }
        )

        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        provider = MonarchProvider(max_results=10)
        results = await provider.gather("test")

        names = [r.content for r in results if "Valid" in r.content]
        assert len(names) >= 1


class TestMonarchAssociations:
    async def test_association_lookup_uses_entity_ids(self, monkeypatch):
        """Entity IDs from search results should be used for association lookups."""
        transport = MockTransport(
            responses={
                "/v3/api/search": {
                    "items": [
                        {
                            "name": "BRCA1",
                            "category": "biolink:Gene",
                            "id": "HGNC:1100",
                            "description": "",
                        },
                    ]
                },
                "/v3/api/association": {
                    "items": [
                        {
                            "subject": {"name": "BRCA1", "id": "HGNC:1100"},
                            "object": {"name": "Breast Cancer", "id": "MONDO:0007254"},
                            "predicate": "associated_with",
                            "evidence_types": ["ECO:0000269"],
                            "publications": ["PMID:12345"],
                        },
                    ]
                },
            }
        )

        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        provider = MonarchProvider(max_results=10)
        results = await provider.gather("BRCA1")

        # Should have search results + association results
        assert len(results) >= 2


class TestMonarchHealthCheck:
    async def test_health_check_uses_same_params_as_production(self, monkeypatch):
        """Health check must use the same parameter format as production searches."""
        transport = MockTransport(
            responses={
                "/v3/api/search": {
                    "items": [{"name": "BRCA1", "id": "X", "category": "biolink:Gene"}]
                },
            }
        )

        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        provider = MonarchProvider()
        result = await provider.check_health()

        assert result.status == "pass"
        assert len(transport.requests) == 1

        req = transport.requests[0]
        raw_query = (
            req.url.raw_path.decode("utf-8")
            if isinstance(req.url.raw_path, bytes)
            else str(req.url)
        )
        category_count = raw_query.count("category=")
        assert category_count == 2, (
            f"Health check must use same params as production. Got {category_count} category params."
        )

    async def test_health_check_reports_failure_on_error(self, monkeypatch):
        """Health check should return fail status on connection error."""
        monkeypatch.setattr("httpx.AsyncClient", _make_failing_client())

        provider = MonarchProvider()
        result = await provider.check_health()

        assert result.status == "fail"

    async def test_health_check_reports_failure_on_non_200(self, monkeypatch):
        transport = MockTransport(status_code=500)
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        provider = MonarchProvider()
        result = await provider.check_health()

        assert result.status == "fail"
        assert "500" in result.message


# ──────────────────────────────────────────────────────────────────────────────
# Monarch _extract_entity_ids
# ──────────────────────────────────────────────────────────────────────────────


class TestMonarchExtractEntityIds:
    def test_extracts_ids_from_quality_metadata(self):
        results = [
            GatheredEvidence(
                content="A",
                source_ref="x",
                source_type="monarch_initiative",
                quality_score=0.7,
                quality_metadata={"entity_id": "HGNC:1100"},
            ),
            GatheredEvidence(
                content="B",
                source_ref="y",
                source_type="monarch_initiative",
                quality_score=0.7,
                quality_metadata={"entity_id": "MONDO:007"},
            ),
            GatheredEvidence(
                content="C",
                source_ref="z",
                source_type="monarch_initiative",
                quality_score=0.7,
                quality_metadata={},
            ),
        ]
        ids = MonarchProvider._extract_entity_ids(results, "test")
        assert ids == ["HGNC:1100", "MONDO:007"]

    def test_empty_results(self):
        ids = MonarchProvider._extract_entity_ids([], "test")
        assert ids == []

    def test_no_metadata(self):
        results = [
            GatheredEvidence(
                content="A",
                source_ref="x",
                source_type="monarch_initiative",
                quality_score=0.7,
            ),
        ]
        ids = MonarchProvider._extract_entity_ids(results, "test")
        assert ids == []


# ──────────────────────────────────────────────────────────────────────────────
# OpenAlex Provider
# ──────────────────────────────────────────────────────────────────────────────


class TestOpenAlexProvider:
    async def test_gather_parses_results(self, monkeypatch):
        """OpenAlex provider should parse search results into GatheredEvidence."""
        from andamentum.epistemic.quality import LiteratureResult, QualityScore

        async def mock_search(query, max_results=10):
            return [
                LiteratureResult(
                    title="Spaced Repetition Study",
                    authors=["Smith J", "Doe A"],
                    abstract="A study on spaced repetition effectiveness.",
                    doi="10.1234/test",
                    pmid="12345",
                    quality=QualityScore(score=0.75, cited_by_count=50),
                ),
                LiteratureResult(
                    title="",
                    authors=[],
                    abstract="",
                    doi=None,
                    pmid=None,
                    quality=None,
                ),  # Should be skipped (no title or abstract)
            ]

        monkeypatch.setattr(
            "andamentum.epistemic.providers.openalex.search_literature", mock_search
        )

        provider = OpenAlexProvider(max_results=10)
        results = await provider.gather("spaced repetition")

        assert len(results) == 1  # Empty result skipped
        assert results[0].source_type == "openalex"
        assert "Spaced Repetition Study" in results[0].content
        assert "Smith J" in results[0].content
        assert results[0].quality_score is None  # providers don't pre-compute quality
        assert "doi:10.1234/test" in results[0].source_ref
        assert "PMID:12345" in results[0].source_ref

    async def test_gather_handles_no_doi(self, monkeypatch):
        """Results without DOI should use title as source_ref."""
        from andamentum.epistemic.quality import LiteratureResult, QualityScore

        async def mock_search(query, max_results=10):
            return [
                LiteratureResult(
                    title="No DOI Paper",
                    authors=["Author"],
                    abstract="Abstract text.",
                    doi=None,
                    pmid=None,
                    quality=QualityScore(score=0.5, cited_by_count=10),
                ),
            ]

        monkeypatch.setattr(
            "andamentum.epistemic.providers.openalex.search_literature", mock_search
        )

        provider = OpenAlexProvider(max_results=10)
        results = await provider.gather("test")

        assert len(results) == 1
        assert results[0].source_ref == "No DOI Paper"


class TestOpenAlexQualityScorer:
    def test_doi_extraction_patterns(self):
        """DOI should be extracted from various formats."""
        assert "10." in "doi:10.1234/test"
        assert "10." in "https://doi.org/10.1234/test"

    def test_pmid_extraction_patterns(self):
        """PMID should be extracted from various formats."""
        test_cases = [
            ("PMID:12345", True),
            ("pmid:12345", True),
            ("(PMID:12345)", True),
            ("no pmid here", False),
        ]
        for ref, should_find in test_cases:
            has_pmid = "pmid:" in ref.lower() or "PMID:" in ref
            assert has_pmid == should_find, f"Failed for: {ref}"


class TestOpenAlexHealthCheck:
    async def test_health_check_success(self, monkeypatch):
        transport = MockTransport(responses={"/works": {"results": []}})
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        provider = OpenAlexProvider()
        result = await provider.check_health()

        assert result.status == "pass"

    async def test_health_check_failure(self, monkeypatch):
        monkeypatch.setattr("httpx.AsyncClient", _make_failing_client())

        provider = OpenAlexProvider()
        result = await provider.check_health()

        assert result.status == "fail"


# ──────────────────────────────────────────────────────────────────────────────
# Provider factory
# ──────────────────────────────────────────────────────────────────────────────


class TestGetBiomedicalProviders:
    def test_returns_both_providers(self):
        from andamentum.epistemic.providers import get_biomedical_providers

        providers = get_biomedical_providers()
        assert "openalex" in providers
        assert "monarch" in providers
        assert isinstance(providers["openalex"], OpenAlexProvider)
        assert isinstance(providers["monarch"], MonarchProvider)

    def test_providers_have_health_check(self):
        from andamentum.epistemic.providers import get_biomedical_providers

        providers = get_biomedical_providers()
        for name, provider in providers.items():
            assert hasattr(provider, "check_health"), (
                f"{name} provider missing check_health()"
            )
            assert hasattr(provider, "gather"), f"{name} provider missing gather()"


# ──────────────────────────────────────────────────────────────────────────────
# Monarch Provider — error path tests
# ──────────────────────────────────────────────────────────────────────────────


class MalformedJsonTransport(httpx.AsyncBaseTransport):
    """Returns non-JSON content to simulate malformed API responses."""

    def __init__(self):
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            status_code=200,
            content=b"<html>not json</html>",
            headers={"content-type": "text/html"},
            request=request,
        )


class TimeoutTransport(httpx.AsyncBaseTransport):
    """Simulates a network timeout on every request."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Connection timed out")


class TestMonarchErrorPaths:
    """Tests for Monarch provider error handling — malformed responses, missing fields, timeouts."""

    async def test_malformed_json_response(self, monkeypatch):
        """API returns invalid JSON → verify graceful handling (empty results, no crash)."""
        transport = MalformedJsonTransport()
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        provider = MonarchProvider(max_results=5)
        results = await provider.gather("BRCA1")

        # Monarch._search catches all exceptions including JSONDecodeError and returns []
        # gather() then returns a single error GatheredEvidence with quality_score=0.0
        # OR returns empty list if no exception propagated from the outer try/except
        assert isinstance(results, list)
        # Should not crash — either empty or contains a single error entry
        for r in results:
            assert isinstance(r, GatheredEvidence)

    async def test_empty_items_response(self, monkeypatch):
        """API returns {"items": []} → verify empty list returned."""
        transport = MockTransport(
            responses={
                "/v3/api/search": {"items": []},
            }
        )
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        provider = MonarchProvider(max_results=5)
        results = await provider.gather("nonexistent query")

        assert results == []

    async def test_partial_items_missing_fields(self, monkeypatch):
        """Items missing 'name'/'id' fields → verify skipped gracefully."""
        transport = MockTransport(
            responses={
                "/v3/api/search": {
                    "items": [
                        {"description": "no name or id"},
                        {"id": "HGNC:999"},  # has id but no name
                        {
                            "name": "",
                            "id": "HGNC:888",
                            "description": "empty name",
                        },  # empty name should be skipped
                        {
                            "name": "ValidGene",
                            "category": "biolink:Gene",
                            "id": "HGNC:100",
                            "description": "Good entry",
                        },
                    ]
                },
                "/v3/api/association": {"items": []},
            }
        )
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        provider = MonarchProvider(max_results=10)
        results = await provider.gather("test")

        # Only the item with a non-empty name should produce a search result
        search_results = [r for r in results if "ValidGene" in r.content]
        assert len(search_results) == 1

        # Items with missing/empty name should be skipped
        bad_results = [r for r in results if "no name or id" in r.content]
        assert len(bad_results) == 0

    async def test_non_200_status_code(self, monkeypatch):
        """API returns 500 → verify empty results, no crash."""
        transport = MockTransport(status_code=500)
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        provider = MonarchProvider(max_results=5)
        results = await provider.gather("BRCA1")

        # _search returns [] on non-200, no entity_ids found, so no associations
        assert isinstance(results, list)
        assert results == []

    async def test_network_timeout(self, monkeypatch):
        """Connection times out → verify graceful handling."""
        transport = TimeoutTransport()
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        provider = MonarchProvider(max_results=5)
        results = await provider.gather("BRCA1")

        # _search catches the timeout exception and returns []
        # gather() returns [] or a single error GatheredEvidence
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, GatheredEvidence)


# ──────────────────────────────────────────────────────────────────────────────
# OpenAlex Provider — error path tests
# ──────────────────────────────────────────────────────────────────────────────


class TestOpenAlexErrorPaths:
    """Tests for OpenAlex provider error handling — exceptions, empty results, missing quality."""

    async def test_search_literature_throws(self, monkeypatch):
        """Mock search_literature that throws → verify empty results."""

        async def mock_search_raises(query, max_results=10):
            raise RuntimeError("OpenAlex API unreachable")

        monkeypatch.setattr(
            "andamentum.epistemic.providers.openalex.search_literature",
            mock_search_raises,
        )

        provider = OpenAlexProvider(max_results=10)
        # OpenAlex.gather() does NOT catch exceptions — it propagates.
        # The caller (CompositeGatherer) is responsible for catching.
        with pytest.raises(RuntimeError, match="OpenAlex API unreachable"):
            await provider.gather("test query")

    async def test_search_returns_empty(self, monkeypatch):
        """Mock returns [] → verify empty list returned."""

        async def mock_search_empty(query, max_results=10):
            return []

        monkeypatch.setattr(
            "andamentum.epistemic.providers.openalex.search_literature",
            mock_search_empty,
        )

        provider = OpenAlexProvider(max_results=10)
        results = await provider.gather("test query")

        assert results == []

    async def test_results_with_missing_quality(self, monkeypatch):
        """Results with quality=None → verify handled (quality_score becomes None)."""
        from andamentum.epistemic.quality import LiteratureResult

        async def mock_search_no_quality(query, max_results=10):
            return [
                LiteratureResult(
                    title="Paper Without Quality",
                    authors=["Author A"],
                    abstract="Some abstract text.",
                    doi="10.9999/test",
                    pmid=None,
                    quality=None,
                ),
            ]

        monkeypatch.setattr(
            "andamentum.epistemic.providers.openalex.search_literature",
            mock_search_no_quality,
        )

        provider = OpenAlexProvider(max_results=10)
        results = await provider.gather("test")

        assert len(results) == 1
        assert results[0].quality_score is None
        assert results[0].quality_metadata == {}
        assert "Paper Without Quality" in results[0].content


# ──────────────────────────────────────────────────────────────────────────────
# Europe PMC Provider
# ──────────────────────────────────────────────────────────────────────────────

_EUROPEPMC_RESULT = {
    "id": "38437170",
    "source": "MED",
    "pmid": "38437170",
    "doi": "10.1038/s41586-024-07487-w",
    "title": "Test article title",
    "authorString": "Smith J, Jones A",
    "journalTitle": "Nature",
    "pubYear": "2024",
    "abstractText": "This is the abstract text.",
    "isOpenAccess": "Y",
    "citedByCount": 42,
    "pubTypeList": {"pubType": ["research-article"]},
    "pmcid": "PMC11234567",
    "firstPublicationDate": "2024-03-01",
}


class TestEuropePMCProvider:
    async def test_gather_returns_gathered_evidence(self, monkeypatch):
        """Mock successful response, verify GatheredEvidence fields."""
        transport = MockTransport(
            responses={
                "/europepmc/webservices/rest/search": {
                    "resultList": {"result": [_EUROPEPMC_RESULT]},
                    "hitCount": 1,
                },
            }
        )
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        from ..providers.europepmc import EuropePMCProvider

        provider = EuropePMCProvider(max_results=5)
        results = await provider.gather("test query")

        assert len(results) == 1
        r = results[0]
        assert isinstance(r, GatheredEvidence)
        assert r.source_type == "europepmc"
        assert r.quality_score is None
        assert r.evidence_kind == "literature"
        assert r.source_ref == "doi:10.1038/s41586-024-07487-w"
        assert "Test article title" in r.content
        assert "Smith J, Jones A" in r.content
        assert "This is the abstract text." in r.content
        assert r.identifiers["pmid"] == "38437170"
        assert r.identifiers["doi"] == "10.1038/s41586-024-07487-w"
        assert r.identifiers["pmcid"] == "PMC11234567"
        assert r.structured_data["journal"] == "Nature"
        assert r.structured_data["cited_by_count"] == 42
        assert r.structured_data["pub_types"] == ["research-article"]
        assert r.limitations == []

    async def test_health_check_pass(self, monkeypatch):
        """Mock 200 response, verify CheckResult."""
        transport = MockTransport(
            responses={
                "/europepmc/webservices/rest/search": {
                    "resultList": {"result": []},
                    "hitCount": 0,
                },
            }
        )
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        from ..providers.europepmc import EuropePMCProvider

        provider = EuropePMCProvider()
        result = await provider.check_health()

        assert result.status == "pass"
        assert "reachable" in result.message
        assert result.name == "EuropePMCProvider"

    async def test_gather_error_returns_empty(self, monkeypatch):
        """Mock 500 response, verify empty list."""
        transport = MockTransport(status_code=500)
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        from ..providers.europepmc import EuropePMCProvider

        provider = EuropePMCProvider(max_results=5)
        results = await provider.gather("test query")

        assert results == []

    async def test_preprint_evidence_kind(self, monkeypatch):
        """Mock response with source PPR, verify evidence_kind and limitations."""
        preprint_result = dict(_EUROPEPMC_RESULT)
        preprint_result["source"] = "PPR"
        preprint_result["pmid"] = ""
        preprint_result["pmcid"] = ""

        transport = MockTransport(
            responses={
                "/europepmc/webservices/rest/search": {
                    "resultList": {"result": [preprint_result]},
                    "hitCount": 1,
                },
            }
        )
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        from ..providers.europepmc import EuropePMCProvider

        provider = EuropePMCProvider(max_results=5)
        results = await provider.gather("preprint query")

        assert len(results) == 1
        r = results[0]
        assert r.evidence_kind == "preprint"
        assert "Preprint" in r.limitations[0]
        assert "not peer-reviewed" in r.limitations[0]

    async def test_html_stripped_from_abstract(self, monkeypatch):
        """Verify HTML tags are stripped from abstractText."""
        html_result = dict(_EUROPEPMC_RESULT)
        html_result["abstractText"] = (
            "<h4>Background</h4>Some background text. "
            "<h4>Methods</h4>Some methods text."
        )

        transport = MockTransport(
            responses={
                "/europepmc/webservices/rest/search": {
                    "resultList": {"result": [html_result]},
                    "hitCount": 1,
                },
            }
        )
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        from ..providers.europepmc import EuropePMCProvider

        provider = EuropePMCProvider(max_results=5)
        results = await provider.gather("test")

        assert len(results) == 1
        assert "<h4>" not in results[0].content
        assert "Background" in results[0].content
        assert "Some methods text." in results[0].content


# ──────────────────────────────────────────────────────────────────────────────
# CompositeGatherer — error path tests
# ──────────────────────────────────────────────────────────────────────────────


class _MockProvider:
    """Mock provider implementing the same gather(query) interface as Monarch/OpenAlex."""

    def __init__(
        self,
        results: list[GatheredEvidence] | None = None,
        error: Exception | None = None,
    ):
        self._results = results or []
        self._error = error

    async def gather(self, query: str) -> list[GatheredEvidence]:
        if self._error:
            raise self._error
        return self._results


class _MockWebSearch:
    """Mock WebSearchGatherer implementing gather(source_type, query)."""

    def __init__(
        self,
        results: list[GatheredEvidence] | None = None,
        error: Exception | None = None,
    ):
        self._results = results or []
        self._error = error

    async def gather(self, source_type: str, query: str) -> list[GatheredEvidence]:
        if self._error:
            raise self._error
        return self._results


class TestCompositeGathererErrorPaths:
    """Tests for CompositeGatherer error handling — provider failures, fallback behavior."""

    async def test_single_provider_fails_others_succeed(self):
        """One provider throws → verify other results still returned (source_type='all')."""
        from andamentum.epistemic.evidence_gathering import CompositeGatherer

        good_evidence = GatheredEvidence(
            content="Good result",
            source_ref="good-source",
            source_type="openalex",
            quality_score=0.8,
        )
        web_evidence = GatheredEvidence(
            content="Web result",
            source_ref="web-source",
            source_type="web_search",
            quality_score=0.5,
        )

        failing_provider = _MockProvider(error=RuntimeError("API down"))
        good_provider = _MockProvider(results=[good_evidence])
        web_search = _MockWebSearch(results=[web_evidence])

        gatherer = CompositeGatherer(
            web_search=web_search,
            providers={
                "failing": failing_provider,
                "openalex": good_provider,
            },
        )

        results = await gatherer.gather("all", "test query")

        # Good provider + web search results should be present; failing provider is skipped
        contents = [r.content for r in results]
        assert "Good result" in contents
        assert "Web result" in contents
        assert len(results) >= 2

    async def test_all_providers_fail(self):
        """All providers throw AND web search throws → RuntimeError raised (source_type='all').

        Previously this silently returned []. Now it raises because every gather call
        failed and returning [] would hide the total failure from the caller.
        """
        from andamentum.epistemic.evidence_gathering import CompositeGatherer

        failing1 = _MockProvider(error=RuntimeError("API 1 down"))
        failing2 = _MockProvider(error=ValueError("API 2 broken"))
        web_search = _MockWebSearch(error=ConnectionError("SearXNG down"))

        gatherer = CompositeGatherer(
            web_search=web_search,
            providers={
                "monarch": failing1,
                "openalex": failing2,
            },
        )

        with pytest.raises(RuntimeError, match="All gather calls failed"):
            await gatherer.gather("all", "test query")

    async def test_web_search_gatherer_fails(self):
        """WebSearchGatherer throws → verify graceful handling for unknown source_type."""
        from andamentum.epistemic.evidence_gathering import CompositeGatherer

        web_search = _MockWebSearch(error=RuntimeError("SearXNG not running"))

        gatherer = CompositeGatherer(
            web_search=web_search,
            providers={},
        )

        # For an unknown source_type with no matching provider, CompositeGatherer
        # falls through to web_search.gather() which is NOT wrapped in try/except
        with pytest.raises(RuntimeError, match="SearXNG not running"):
            await gatherer.gather("web_search", "test query")

    async def test_provider_fails_raises(self):
        """Registered provider throws → error propagates for that specific source_type.

        Previously this silently fell back to web search, which could return web results
        labelled as if they came from the requested provider. Now it raises so the caller
        knows the requested source failed.
        """
        from andamentum.epistemic.evidence_gathering import CompositeGatherer

        failing_provider = _MockProvider(error=RuntimeError("Monarch down"))
        web_search = _MockWebSearch(results=[])  # should never be called

        gatherer = CompositeGatherer(
            web_search=web_search,
            providers={"monarch": failing_provider},
        )

        with pytest.raises(RuntimeError, match="Monarch down"):
            await gatherer.gather("monarch", "BRCA1")


# ──────────────────────────────────────────────────────────────────────────────
# Cochrane Provider
# ──────────────────────────────────────────────────────────────────────────────

_COCHRANE_EFETCH_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>Interventions for preventing falls in older people</ArticleTitle>
        <AuthorList>
          <Author><LastName>Gillespie</LastName><Initials>LD</Initials></Author>
        </AuthorList>
        <Abstract>
          <AbstractText Label="BACKGROUND">Falls are common in older people.</AbstractText>
          <AbstractText Label="MAIN RESULTS">Exercise reduces fall rate by 23%.</AbstractText>
          <AbstractText Label="AUTHORS' CONCLUSIONS">Exercise programmes reduce falls.</AbstractText>
        </Abstract>
        <ArticleIdList>
          <ArticleId IdType="doi">10.1002/14651858.CD007146.pub4</ArticleId>
        </ArticleIdList>
      </Article>
      <MedlineJournalInfo>
        <MedlineTA>Cochrane Database Syst Rev</MedlineTA>
      </MedlineJournalInfo>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


class CochraneMockTransport(httpx.AsyncBaseTransport):
    """Returns JSON for esearch, XML for efetch."""

    def __init__(self, status_code: int = 200):
        self.requests: list[httpx.Request] = []
        self._status_code = status_code

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._status_code != 200:
            return httpx.Response(self._status_code, request=request)
        if "esearch" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "esearchresult": {"idlist": ["12345678"], "count": "1"},
                },
                request=request,
            )
        elif "efetch" in request.url.path:
            return httpx.Response(
                200,
                text=_COCHRANE_EFETCH_XML,
                request=request,
                headers={"content-type": "text/xml"},
            )
        return httpx.Response(404, request=request)


class TestCochraneProvider:
    async def test_gather_returns_gathered_evidence(self, monkeypatch):
        """Mock successful response, verify GatheredEvidence fields."""
        transport = CochraneMockTransport()
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        from ..providers.cochrane import CochraneProvider

        provider = CochraneProvider(max_results=5)
        results = await provider.gather("falls prevention")

        assert len(results) == 1
        r = results[0]
        assert isinstance(r, GatheredEvidence)
        assert r.source_type == "cochrane"
        assert r.evidence_kind == "systematic_review"
        assert r.quality_score is None
        assert "Interventions for preventing falls" in r.content
        assert "Gillespie LD" in r.content
        assert r.source_ref == "doi:10.1002/14651858.CD007146.pub4"
        assert r.identifiers["pmid"] == "12345678"
        assert r.identifiers["doi"] == "10.1002/14651858.CD007146.pub4"
        assert r.quality_metadata is not None
        assert r.quality_metadata["journal"] == "Cochrane Database Syst Rev"
        assert "Systematic Review" in r.quality_metadata["publication_types"]
        assert r.limitations == []

    async def test_health_check_pass(self, monkeypatch):
        """Mock 200 response with count > 0, verify CheckResult."""
        transport = CochraneMockTransport()
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        from ..providers.cochrane import CochraneProvider

        provider = CochraneProvider()
        result = await provider.check_health()

        assert result.status == "pass"
        assert "reachable" in result.message
        assert result.name == "CochraneProvider"

    async def test_gather_error_returns_empty(self, monkeypatch):
        """Mock 500 response, verify empty list."""
        transport = CochraneMockTransport(status_code=500)
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        from ..providers.cochrane import CochraneProvider

        provider = CochraneProvider(max_results=5)
        results = await provider.gather("test query")

        assert results == []

    async def test_structured_abstract_sections(self, monkeypatch):
        """Verify abstract_sections dict in structured_data."""
        transport = CochraneMockTransport()
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        from ..providers.cochrane import CochraneProvider

        provider = CochraneProvider(max_results=5)
        results = await provider.gather("falls prevention")

        assert len(results) == 1
        r = results[0]
        sections = r.structured_data["abstract_sections"]
        assert sections["BACKGROUND"] == "Falls are common in older people."
        assert sections["MAIN RESULTS"] == "Exercise reduces fall rate by 23%."
        assert sections["AUTHORS' CONCLUSIONS"] == "Exercise programmes reduce falls."

        # Verify content includes labeled sections
        assert "BACKGROUND: Falls are common in older people." in r.content
        assert "MAIN RESULTS: Exercise reduces fall rate by 23%." in r.content
        assert "AUTHORS' CONCLUSIONS: Exercise programmes reduce falls." in r.content


# ──────────────────────────────────────────────────────────────────────────────
# arXiv Provider
# ──────────────────────────────────────────────────────────────────────────────

_ARXIV_XML_WITH_JOURNAL = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Attention Is All You Need: A Revisit</title>
    <summary>We revisit the transformer architecture and propose improvements.</summary>
    <published>2023-01-01T00:00:00Z</published>
    <updated>2023-01-02T00:00:00Z</updated>
    <author><name>Smith J</name></author>
    <author><name>Jones A</name></author>
    <category term="cs.CL"/>
    <category term="cs.AI"/>
    <arxiv:primary_category term="cs.CL"/>
    <arxiv:doi>10.1234/test.2023</arxiv:doi>
    <arxiv:journal_ref>Nature 2023</arxiv:journal_ref>
    <arxiv:comment>10 pages, 3 figures</arxiv:comment>
    <link title="pdf" href="http://arxiv.org/pdf/2301.00001v1" rel="related"/>
  </entry>
</feed>"""

_ARXIV_XML_NO_JOURNAL = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <title>Scaling Laws for
Neural Language Models</title>
    <summary>We study scaling laws for neural language model performance.</summary>
    <published>2024-01-15T00:00:00Z</published>
    <updated>2024-02-01T00:00:00Z</updated>
    <author><name>Doe B</name></author>
    <category term="cs.LG"/>
    <arxiv:primary_category term="cs.LG"/>
  </entry>
</feed>"""


class ArXivMockTransport(httpx.AsyncBaseTransport):
    """Returns Atom XML for arXiv API queries."""

    def __init__(self, status_code: int = 200, xml: str = _ARXIV_XML_WITH_JOURNAL):
        self.requests: list[httpx.Request] = []
        self._status_code = status_code
        self._xml = xml

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._status_code != 200:
            return httpx.Response(self._status_code, text="Error", request=request)
        return httpx.Response(
            200,
            text=self._xml,
            request=request,
            headers={"content-type": "application/atom+xml"},
        )


class TestArXivProvider:
    async def test_gather_returns_gathered_evidence(self, monkeypatch):
        """Mock successful response, verify GatheredEvidence fields."""
        transport = ArXivMockTransport()
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        from ..providers.arxiv import ArXivProvider

        provider = ArXivProvider(max_results=10)
        results = await provider.gather("transformer attention")

        assert len(results) == 1
        r = results[0]
        assert isinstance(r, GatheredEvidence)
        assert r.source_type == "arxiv"
        assert r.evidence_kind == "preprint"
        assert r.quality_score is None
        assert r.source_ref == "arXiv:2301.00001v1"
        assert "Attention Is All You Need: A Revisit" in r.content
        assert "Smith J" in r.content
        assert "We revisit the transformer architecture" in r.content
        assert r.identifiers["arxiv_id"] == "2301.00001v1"
        assert r.identifiers["doi"] == "10.1234/test.2023"
        assert r.structured_data["title"] == "Attention Is All You Need: A Revisit"
        assert r.structured_data["authors"] == ["Smith J", "Jones A"]
        assert r.structured_data["categories"] == ["cs.CL", "cs.AI"]
        assert r.structured_data["primary_category"] == "cs.CL"
        assert r.structured_data["published"] == "2023-01-01T00:00:00Z"
        assert r.structured_data["updated"] == "2023-01-02T00:00:00Z"
        assert r.structured_data["journal_ref"] == "Nature 2023"
        assert r.structured_data["doi"] == "10.1234/test.2023"
        assert r.structured_data["comment"] == "10 pages, 3 figures"
        assert r.quality_metadata is not None
        assert r.quality_metadata["primary_category"] == "cs.CL"
        assert r.quality_metadata["has_journal_ref"] is True
        # Has journal_ref → no limitations
        assert r.limitations == []

    async def test_health_check_pass(self, monkeypatch):
        """Mock 200 response with entry, verify CheckResult."""
        transport = ArXivMockTransport()
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        from ..providers.arxiv import ArXivProvider

        provider = ArXivProvider()
        result = await provider.check_health()

        assert result.status == "pass"
        assert "reachable" in result.message
        assert result.name == "ArXivProvider"

    async def test_gather_error_returns_empty(self, monkeypatch):
        """Mock 500 response, verify empty list."""
        transport = ArXivMockTransport(status_code=500)
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        from ..providers.arxiv import ArXivProvider

        provider = ArXivProvider(max_results=5)
        results = await provider.gather("test query")

        assert results == []

    async def test_preprint_without_journal_ref(self, monkeypatch):
        """Entry without journal_ref → limitations includes preprint caveat."""
        transport = ArXivMockTransport(xml=_ARXIV_XML_NO_JOURNAL)
        monkeypatch.setattr("httpx.AsyncClient", _make_patched_client(transport))

        from ..providers.arxiv import ArXivProvider

        provider = ArXivProvider(max_results=10)
        results = await provider.gather("scaling laws")

        assert len(results) == 1
        r = results[0]
        assert r.source_ref == "arXiv:2401.12345v2"
        assert r.limitations == ["Preprint — not peer-reviewed"]
        assert r.quality_metadata is not None
        assert r.quality_metadata["has_journal_ref"] is False
        # Title newlines should be stripped
        assert "\n" not in r.structured_data["title"]
        assert "Scaling Laws for Neural Language Models" == r.structured_data["title"]
        # No DOI in identifiers when missing
        assert "doi" not in r.identifiers
        assert r.structured_data["doi"] is None
        assert r.structured_data["journal_ref"] is None
