"""Integration smoke tests — hit real APIs to verify response shapes.

These tests catch API schema changes between releases. They require network
access and are skipped by default. Run with:

    uv run pytest packages/epistemic/tests/test_integration_smoke.py -v -m integration

Marked @pytest.mark.integration so they don't run in CI or normal test runs.
"""

import pytest

pytestmark = pytest.mark.integration


class TestMonarchAPISmoke:
    """Verify Monarch Initiative API is reachable and returns expected shapes."""

    async def test_search_returns_items(self):
        """Hit real Monarch search endpoint with a known gene query."""
        from andamentum.epistemic.providers.monarch import MonarchProvider

        provider = MonarchProvider(max_results=3)
        results = await provider.gather("BRCA1")

        # We should get at least 1 result for a well-known gene
        assert len(results) >= 1, (
            "Monarch returned no results for BRCA1 — API may have changed"
        )

        # Verify result shape
        for r in results:
            assert r.content, "Result content should not be empty"
            assert r.source_type == "monarch"
            assert r.source_ref, "Result should have a source_ref"

    async def test_health_check_passes(self):
        """Monarch health check should pass against live API."""
        from andamentum.epistemic.providers.monarch import MonarchProvider

        provider = MonarchProvider()
        result = await provider.check_health()

        assert result.status == "pass", f"Monarch health check failed: {result.message}"

    async def test_search_params_produce_200(self):
        """Verify the production search params don't produce 422 (regression for the comma-separated category bug)."""
        import httpx

        from andamentum.epistemic.providers.monarch import MONARCH_API

        # Use the exact same params as production
        params = [
            ("q", "BRCA1"),
            ("limit", "3"),
            ("category", "biolink:Gene"),
            ("category", "biolink:Disease"),
        ]

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{MONARCH_API}/search", params=params)  # type: ignore[arg-type]
            assert resp.status_code == 200, (
                f"Monarch search returned {resp.status_code} — "
                f"category params may need updating. Response: {resp.text[:200]}"
            )
            data = resp.json()
            assert "items" in data, (
                f"Response missing 'items' key. Keys: {list(data.keys())}"
            )

    async def test_association_endpoint_reachable(self):
        """Verify the Monarch association endpoint is reachable with production params."""
        import httpx

        from andamentum.epistemic.providers.monarch import MONARCH_API

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{MONARCH_API}/association",
                params={"subject": "HGNC:1100", "limit": 2},
            )
            assert resp.status_code == 200, (
                f"Monarch association returned {resp.status_code}. Response: {resp.text[:200]}"
            )
            data = resp.json()
            assert "items" in data, (
                f"Response missing 'items' key. Keys: {list(data.keys())}"
            )


class TestOpenAlexAPISmoke:
    """Verify OpenAlex API is reachable and returns expected shapes."""

    async def test_search_returns_results(self):
        """Hit real OpenAlex with a known query."""
        from andamentum.epistemic.providers.openalex import OpenAlexProvider

        provider = OpenAlexProvider(max_results=3)
        results = await provider.gather("spaced repetition memory")

        assert len(results) >= 1, "OpenAlex returned no results — API may have changed"

        for r in results:
            assert r.content, "Result content should not be empty"
            assert r.source_type == "openalex"

    async def test_health_check_passes(self):
        """OpenAlex health check should pass against live API."""
        from andamentum.epistemic.providers.openalex import OpenAlexProvider

        provider = OpenAlexProvider()
        result = await provider.check_health()

        assert result.status == "pass", (
            f"OpenAlex health check failed: {result.message}"
        )

    async def test_works_endpoint_shape(self):
        """Verify the OpenAlex /works endpoint returns expected structure."""
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                "https://api.openalex.org/works",
                params={"search": "BRCA1", "per_page": "2"},
                headers={
                    "User-Agent": "andamentum-epistemic-test/0.1 (mailto:test@example.com)"
                },
            )
            assert resp.status_code == 200, f"OpenAlex returned {resp.status_code}"
            data = resp.json()
            assert "results" in data, (
                f"Response missing 'results'. Keys: {list(data.keys())}"
            )

            if data["results"]:
                work = data["results"][0]
                # Verify expected fields exist
                expected_fields = {"id", "title", "doi"}
                actual_fields = set(work.keys())
                missing = expected_fields - actual_fields
                assert not missing, f"OpenAlex work missing fields: {missing}"

    async def test_search_literature_returns_quality(self):
        """Verify search_literature returns results with pre-computed quality scores."""
        from andamentum.epistemic.quality import search_literature

        results = await search_literature("CRISPR gene editing", max_results=3)

        assert len(results) >= 1, "search_literature returned no results"

        for r in results:
            assert r.title, "LiteratureResult should have a title"
            if r.quality is not None:
                assert 0.0 <= r.quality.score <= 1.0, (
                    f"Quality score {r.quality.score} out of range"
                )
                assert r.quality.cited_by_count >= 0
                assert r.quality.source == "openalex"


class TestOpenAlexQualityScorerSmoke:
    """Verify OpenAlex quality scorer works with real DOIs."""

    async def test_score_known_doi(self):
        """Score a well-known paper by DOI."""
        from andamentum.epistemic.quality import score_source

        # Use a well-cited paper DOI (Nature paper on CRISPR)
        result = await score_source(
            doi="10.1038/nature12373",
            source_ref="doi:10.1038/nature12373",
            source_type="openalex",
        )

        assert result is not None, "score_source returned None for a known DOI"
        assert 0.0 <= result.score <= 1.0, f"Score {result.score} out of range"
        assert result.cited_by_count >= 0
        assert result.source == "openalex"

    async def test_score_unknown_doi_returns_none(self):
        """Non-existent DOI should return None."""
        from andamentum.epistemic.quality import score_source

        result = await score_source(
            doi="10.9999/nonexistent-fake-doi-12345",
            source_ref="fake",
            source_type="test",
        )

        # Should return None for a DOI that doesn't exist
        assert result is None, f"Unknown DOI should return None, got: {result}"

    async def test_scorer_class_extracts_doi(self):
        """OpenAlexQualityScorer should accept pre-extracted identifiers
        and look them up. Phase 3 of the efficiency plan moved
        identifier extraction upstream — the scorer no longer does its
        own primitive string parsing."""
        from andamentum.epistemic.operations.identifier_extraction import (
            extract_identifiers,
        )
        from andamentum.epistemic.providers.openalex import OpenAlexQualityScorer

        scorer = OpenAlexQualityScorer()
        identifiers = extract_identifiers("doi:10.1038/nature12373")
        assert identifiers.doi == "10.1038/nature12373"
        result = await scorer.score(identifiers, "doi:10.1038/nature12373", "openalex")

        if result is not None:
            assert 0.0 <= result.score <= 1.0, f"Score {result.score} out of range"
            assert result.cited_by_count >= 0

    async def test_scorer_class_handles_no_identifier(self):
        """OpenAlexQualityScorer with no DOI/PMID in identifiers should return None."""
        from andamentum.epistemic.operations.identifier_extraction import Identifiers
        from andamentum.epistemic.providers.openalex import OpenAlexQualityScorer

        scorer = OpenAlexQualityScorer()
        identifiers = Identifiers()  # all None
        result = await scorer.score(identifiers, "some random text", "web_search")

        assert result is None, "Should return None when identifiers are all None"


class TestHealthCheckProductionParity:
    """Structural assertion: health check params must match production params.

    This test reads the source code to verify the health check and production
    search use the same parameter format. This catches the class of bug where
    health checks test different code paths than production.
    """

    def test_monarch_health_check_uses_search_endpoint(self):
        """Monarch health check should call the /search endpoint — same as gather()."""
        import inspect

        from andamentum.epistemic.providers.monarch import MonarchProvider

        health_source = inspect.getsource(MonarchProvider.check_health)
        # Health check must hit the same /search endpoint
        assert "/search" in health_source, (
            "Health check should use /search endpoint for param parity"
        )

    def test_monarch_category_params_are_list_format(self):
        """Verify the Monarch provider uses list-of-tuples for category params (not comma-separated)."""
        import inspect

        from andamentum.epistemic.providers.monarch import MonarchProvider

        source = inspect.getsource(MonarchProvider)

        # Should NOT have comma-separated categories in a single string
        # The bug was: category="biolink:Gene,biolink:Disease"
        assert "biolink:Gene,biolink:Disease" not in source, (
            "Monarch provider still uses comma-separated categories — this causes 422 errors"
        )

    def test_openalex_health_check_uses_works_endpoint(self):
        """OpenAlex health check should call /works — same endpoint as search_literature()."""
        import inspect

        from andamentum.epistemic.providers.openalex import OpenAlexProvider

        health_source = inspect.getsource(OpenAlexProvider.check_health)
        assert "/works" in health_source, (
            "Health check should use /works endpoint for param parity"
        )
