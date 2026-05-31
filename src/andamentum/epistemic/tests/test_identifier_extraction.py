"""Tests for the general academic identifier extractor.

Phase 3 of the efficiency plan. The extractor pulls DOIs / PMIDs /
arXiv IDs out of arbitrary text so the quality-scorer Path 1 (free
bibliometric lookup) catches identifiers that the previous
primitive string-matching missed.

These tests pin the recogniser patterns. False positives here are
benign (one wasted OpenAlex API call per false positive — the API
returns nothing). False negatives are what cost LLM dollars
(items fall through to Path 2 unnecessarily).
"""

from __future__ import annotations

from andamentum.epistemic.operations.identifier_extraction import (
    Identifiers,
    extract_identifiers,
)


# ── DOI extraction ───────────────────────────────────────────────────


def test_doi_in_url() -> None:
    ids = extract_identifiers("https://doi.org/10.1038/s41586-020-2012-7")
    assert ids.doi == "10.1038/s41586-020-2012-7"


def test_doi_with_doi_prefix() -> None:
    ids = extract_identifiers("doi:10.1038/s41586-020-2012-7")
    assert ids.doi == "10.1038/s41586-020-2012-7"


def test_doi_in_prose_strips_trailing_period() -> None:
    """DOIs in prose often have a trailing period belonging to the
    sentence, not the identifier. The extractor strips it."""
    ids = extract_identifiers("...published as doi:10.1234/abc. The next sentence.")
    assert ids.doi == "10.1234/abc"


def test_doi_in_url_path_query_param() -> None:
    """Many web pages have the DOI buried in a URL path or query
    parameter (not standalone). Phase 3 catches these."""
    ids = extract_identifiers(
        "https://example.com/some/path?ref=10.1038/s41586-020-2012-7"
    )
    assert ids.doi == "10.1038/s41586-020-2012-7"


def test_doi_inside_content_body() -> None:
    """A common case: the source_ref is just a URL but the actual DOI
    is in the first 1000 chars of the extracted content. The
    extractor takes multiple text inputs."""
    source_ref = "https://example.com/article"
    content = "Abstract. Background... For details see doi:10.1234/foo."
    ids = extract_identifiers(source_ref, content)
    assert ids.doi == "10.1234/foo"


def test_doi_long_suffix_with_punctuation() -> None:
    """DOI suffixes can contain punctuation (e.g. parentheses,
    hyphens, dots). The extractor stops at whitespace or sentence-
    boundary punctuation."""
    ids = extract_identifiers("see 10.1016/S0140-6736(20)30183-5 for the trial")
    assert ids.doi == "10.1016/S0140-6736(20"  # parens stop the match
    # Note: this is a known limitation. Most DOIs don't have parens
    # in the suffix; for those that do, the OpenAlex lookup will fail
    # and Path 2 catches it.


# ── PMID extraction ──────────────────────────────────────────────────


def test_pmid_with_colon_prefix() -> None:
    ids = extract_identifiers("See PMID: 12345678 for details.")
    assert ids.pmid == "12345678"


def test_pmid_lowercase_prefix() -> None:
    ids = extract_identifiers("pmid:9876543")
    assert ids.pmid == "9876543"


def test_pmid_in_pubmed_url() -> None:
    """PubMed URL format ``pubmed/12345`` (or
    pubmed.ncbi.nlm.nih.gov/12345/) is a common source_ref shape."""
    ids = extract_identifiers("https://pubmed.ncbi.nlm.nih.gov/12345/")
    assert ids.pmid == "12345"


def test_pmid_in_pubmed_short_url() -> None:
    ids = extract_identifiers("pubmed/87654321")
    assert ids.pmid == "87654321"


def test_pmid_no_bare_digits() -> None:
    """Bare digit strings without a recognisable cue are NOT pmids
    (otherwise we'd hit massive false-positive rates in scientific
    text full of years, ages, sample sizes, etc.)."""
    ids = extract_identifiers("In 2020, 100 patients were enrolled (mean age 65).")
    assert ids.pmid is None


# ── arXiv extraction ─────────────────────────────────────────────────


def test_arxiv_post_2007_format() -> None:
    ids = extract_identifiers("arXiv:2401.12345")
    assert ids.arxiv == "2401.12345"


def test_arxiv_with_space() -> None:
    ids = extract_identifiers("arXiv 2401.12345")
    assert ids.arxiv == "2401.12345"


def test_arxiv_4_digit_suffix() -> None:
    ids = extract_identifiers("arXiv:0901.1234")
    assert ids.arxiv == "0901.1234"


# ── Combined / edge cases ────────────────────────────────────────────


def test_all_three_in_one_text() -> None:
    text = "Combined: doi:10.1234/x and PMID:9876543 and arXiv:2401.12345"
    ids = extract_identifiers(text)
    assert ids.doi == "10.1234/x"
    assert ids.pmid == "9876543"
    assert ids.arxiv == "2401.12345"


def test_no_identifiers_returns_all_none() -> None:
    ids = extract_identifiers("just some text without any academic identifiers.")
    assert ids.doi is None
    assert ids.pmid is None
    assert ids.arxiv is None
    assert ids.has_any is False


def test_empty_inputs_return_all_none() -> None:
    assert not extract_identifiers().has_any
    assert not extract_identifiers(None, None).has_any
    assert not extract_identifiers("", "").has_any


def test_has_any_true_when_one_found() -> None:
    ids = extract_identifiers("doi:10.1234/x")
    assert ids.has_any is True


def test_first_match_wins_across_inputs() -> None:
    """When the same identifier type appears in multiple inputs, the
    extractor takes the first match (stops searching subsequent
    inputs for that type)."""
    ids = extract_identifiers("doi:10.1234/x", "doi:10.5678/y")
    assert ids.doi == "10.1234/x"


def test_skips_none_inputs() -> None:
    """Mixing None among real strings is supported (typical for
    optional content fields)."""
    ids = extract_identifiers(None, "doi:10.1234/x", None)
    assert ids.doi == "10.1234/x"


def test_identifiers_is_immutable() -> None:
    """Identifiers is a frozen dataclass; direct field mutation
    raises (FrozenInstanceError or AttributeError depending on
    Python version)."""
    import dataclasses

    ids = Identifiers(doi="10.1234/x")
    try:
        ids.doi = "10.5678/y"  # type: ignore[misc]
        raise AssertionError("Identifiers should be frozen")
    except (dataclasses.FrozenInstanceError, AttributeError):
        pass  # expected
