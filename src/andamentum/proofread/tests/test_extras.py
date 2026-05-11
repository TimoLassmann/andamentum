"""Tests for duplicate words, adverb density, weak openers."""

from andamentum.proofread.extras import (
    adverb_stats,
    scan_duplicates,
    scan_weak_openers,
)
from andamentum.proofread.sentences import segment


def test_scan_duplicates_finds_the_the():
    text = "The result is that the the data is wrong."
    findings = scan_duplicates(text, segment(text))
    assert len(findings) == 1
    assert findings[0].word.lower() == "the"


def test_scan_duplicates_skips_allowlist():
    text = "He had had enough. I knew that that was wrong."
    assert scan_duplicates(text, segment(text)) == []


def test_scan_duplicates_case_insensitive():
    text = "We are The the same."
    findings = scan_duplicates(text, segment(text))
    assert len(findings) == 1


def test_adverb_stats_counts_ly_words():
    s = adverb_stats("The system quickly produced highly accurate results.")
    # quickly, highly → 2; "results" doesn't end in -ly
    assert s.adverb_count == 2
    assert s.adverb_density > 0


def test_adverb_stats_excludes_non_adverbs():
    s = adverb_stats("The only family was ugly.")
    # only, family, ugly are excluded
    assert s.adverb_count == 0


def test_adverb_stats_empty_text():
    s = adverb_stats("")
    assert s.adverb_count == 0
    assert s.adverb_density == 0.0


def test_weak_openers_there_is():
    text = "There is a problem. The cat sat."
    findings = scan_weak_openers(segment(text))
    assert len(findings) == 1
    assert findings[0].matched_text.lower() == "there is"
    assert findings[0].sentence_index == 0


def test_weak_openers_it_was():
    text = "Quiet for once. It was raining."
    findings = scan_weak_openers(segment(text))
    assert len(findings) == 1
    assert findings[0].sentence_index == 1


def test_weak_openers_offsets_roundtrip():
    text = "There are many dogs. It is clear."
    findings = scan_weak_openers(segment(text))
    for f in findings:
        assert text[f.span.start : f.span.end].lower() == f.matched_text.lower()
