"""Lexical scoring: query terms and BM25 over a handful of passages."""

from __future__ import annotations

from ..lexical import BM25_B, BM25_K1, bm25_scores, lexical_terms


def test_terms_are_casefolded_long_enough_and_without_stop_words() -> None:
    assert lexical_terms("The Apollo cache is on a big disk") == [
        "apollo",
        "cache",
        "big",
        "disk",
    ]


def test_a_plural_s_is_folded_but_not_a_double_s() -> None:
    assert lexical_terms("measurements latencies glass") == [
        "measurement",
        "latencie",
        "glass",
    ]


def test_repeats_are_kept_so_frequencies_count() -> None:
    assert lexical_terms("cache cache cache") == ["cache"] * 3


def test_constants_keep_their_values() -> None:
    assert (BM25_K1, BM25_B) == (1.5, 0.75)


def test_the_passage_sharing_the_query_terms_scores_highest() -> None:
    docs = [
        "SaltLog writes verbose logs to disk.",
        "The Apollo cache cut median latency in the benchmark.",
        "Weather was mild during the benchmark.",
    ]
    scores = bm25_scores("Apollo cache latency", docs)
    assert scores[1] > scores[2] >= 0.0
    assert scores[0] == 0.0 and scores[2] == 0.0


def test_a_document_without_a_shared_term_scores_zero_and_none_is_empty() -> None:
    assert bm25_scores("zzz", ["alpha beta gamma"]) == [0.0]
    assert bm25_scores("alpha", []) == []


def test_a_rarer_term_weighs_more_than_a_common_one() -> None:
    docs = ["common rare", "common filler", "common padding", "common words"]
    scores = bm25_scores("common rare", docs)
    assert scores[0] == max(scores) and scores[0] > scores[1]
