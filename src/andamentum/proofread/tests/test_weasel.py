"""Tests for the weasel-word detector."""

from andamentum.proofread.sentences import segment
from andamentum.proofread.weasel import scan


def test_scan_finds_classic_weasels():
    text = "There are many studies. The result is very clearly significant."
    findings = scan(text, segment(text))
    words = [f.word.lower() for f in findings]
    assert "many" in words
    assert "very" in words
    assert "clearly" in words
    assert "significantly" not in words  # "significantly" not present


def test_scan_offsets_roundtrip():
    text = "The result was extremely surprising."
    sentences = segment(text)
    findings = scan(text, sentences)
    for f in findings:
        assert text[f.span.start : f.span.end].lower() == f.word.lower()


def test_scan_assigns_sentence_index():
    text = "Cats are nice. Many dogs bark loudly. The end."
    findings = scan(text, segment(text))
    by_word = {f.word.lower(): f.sentence_index for f in findings}
    assert by_word["many"] == 1


def test_scan_is_case_insensitive_but_preserves_casing():
    text = "Very interesting. EXTREMELY fast."
    findings = scan(text, segment(text))
    cases = [f.word for f in findings]
    assert "Very" in cases
    assert "EXTREMELY" in cases


def test_scan_returns_empty_for_clean_text():
    text = "The temperature rose by three degrees. The dataset has 100 rows."
    assert scan(text, segment(text)) == []
