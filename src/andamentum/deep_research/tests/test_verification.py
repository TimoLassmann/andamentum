"""Tests for deep research source verification."""

import pytest
from deep_research.verification import normalize_url, verify_sources


class TestURLNormalization:
    """Test URL normalization function."""

    def test_normalize_basic(self):
        assert normalize_url("https://example.com/page") == "https://example.com/page"

    def test_normalize_case(self):
        assert normalize_url("HTTPS://EXAMPLE.COM/PAGE") == "https://example.com/page"

    def test_normalize_trailing_slash(self):
        assert normalize_url("https://example.com/page/") == "https://example.com/page"

    def test_normalize_http_to_https(self):
        assert normalize_url("http://example.com/page") == "https://example.com/page"

    def test_normalize_whitespace(self):
        assert normalize_url("  https://example.com/page  ") == "https://example.com/page"

    def test_normalize_query_params(self):
        assert normalize_url("https://example.com/page?q=test") == "https://example.com/page?q=test"

    def test_normalize_empty(self):
        assert normalize_url("") == ""

    def test_normalize_complex_path(self):
        assert normalize_url("HTTP://Example.COM/Path/To/Page/") == "https://example.com/path/to/page"


class TestSourceVerification:
    """Test source verification function."""

    def test_all_verified_fetched(self):
        cited = ["https://example.com/page1", "https://example.com/page2"]
        fetched = {"https://example.com/page1", "https://example.com/page2"}
        result = verify_sources(cited, set(), fetched)
        assert result["total_cited"] == 2
        assert result["verified_count"] == 2
        assert result["verification_rate"] == 1.0
        assert len(result["unverified"]) == 0

    def test_all_verified_searched(self):
        cited = ["https://example.com/page1"]
        searched = {"https://example.com/page1"}
        result = verify_sources(cited, searched, set())
        assert result["verified_count"] == 1
        assert result["verification_rate"] == 1.0

    def test_hallucinated_source(self):
        cited = ["https://example.com/real", "https://example.com/fake"]
        fetched = {"https://example.com/real"}
        result = verify_sources(cited, set(), fetched)
        assert result["total_cited"] == 2
        assert result["verified_count"] == 1
        assert result["verification_rate"] == 0.5
        assert len(result["unverified"]) == 1
        assert "fake" in result["unverified"][0]

    def test_url_normalization_in_verification(self):
        cited = ["HTTPS://Example.com/Page/"]
        fetched = {"http://example.com/page"}
        result = verify_sources(cited, set(), fetched)
        assert result["verified_count"] == 1
        assert result["verification_rate"] == 1.0

    def test_accessed_not_cited(self):
        cited = ["https://example.com/page1"]
        fetched = {"https://example.com/page1", "https://example.com/page2"}
        result = verify_sources(cited, set(), fetched)
        assert len(result["accessed_not_cited"]) == 1
        assert "page2" in result["accessed_not_cited"][0]

    def test_empty_citations(self):
        result = verify_sources([], set(), set())
        assert result["total_cited"] == 0
        assert result["verified_count"] == 0
        assert result["verification_rate"] == 0.0

    def test_partial_verification(self):
        cited = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3",
            "https://example.com/fake",
        ]
        searched = {"https://example.com/page1"}
        fetched = {"https://example.com/page2", "https://example.com/page3"}
        result = verify_sources(cited, searched, fetched)
        assert result["total_cited"] == 4
        assert result["verified_count"] == 3
        assert result["verification_rate"] == 0.75
        assert len(result["unverified"]) == 1

    def test_searched_and_fetched_overlap(self):
        cited = ["https://example.com/page1"]
        searched = {"https://example.com/page1"}
        fetched = {"https://example.com/page1"}
        result = verify_sources(cited, searched, fetched)
        assert result["verified_count"] == 1
        assert result["verification_rate"] == 1.0

    def test_complex_url_variations(self):
        cited = ["HTTPS://EXAMPLE.COM/PAGE/", "http://example.com/other"]
        searched = {"https://example.com/page"}
        fetched = {"https://example.com/other"}
        result = verify_sources(cited, searched, fetched)
        assert result["verified_count"] == 2
        assert result["verification_rate"] == 1.0

    def test_query_params_preserved(self):
        cited = ["https://example.com/page?id=123"]
        fetched = {"https://example.com/page?id=123"}
        result = verify_sources(cited, set(), fetched)
        assert result["verified_count"] == 1

    def test_query_params_mismatch(self):
        cited = ["https://example.com/page?id=123"]
        fetched = {"https://example.com/page?id=456"}
        result = verify_sources(cited, set(), fetched)
        assert result["verified_count"] == 0
        assert len(result["unverified"]) == 1

    def test_all_hallucinated(self):
        cited = ["https://fake1.com", "https://fake2.com", "https://fake3.com"]
        result = verify_sources(cited, set(), set())
        assert result["total_cited"] == 3
        assert result["verified_count"] == 0
        assert result["verification_rate"] == 0.0
        assert len(result["unverified"]) == 3

    def test_verification_result_structure(self):
        result = verify_sources(["https://example.com/page1"], {"https://example.com/page1"}, set())
        assert isinstance(result["total_cited"], int)
        assert isinstance(result["verified_count"], int)
        assert isinstance(result["verified"], list)
        assert isinstance(result["unverified"], list)
        assert isinstance(result["accessed_not_cited"], list)
        assert isinstance(result["verification_rate"], float)

    def test_large_scale_verification(self):
        cited = [f"https://example.com/page{i}" for i in range(100)]
        searched = {f"https://example.com/page{i}" for i in range(50)}
        fetched = {f"https://example.com/page{i}" for i in range(50, 80)}
        result = verify_sources(cited, searched, fetched)
        assert result["total_cited"] == 100
        assert result["verified_count"] == 80
        assert result["verification_rate"] == 0.8
        assert len(result["unverified"]) == 20
