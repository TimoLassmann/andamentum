"""Tests for the passive-voice heuristic."""

from andamentum.proofread.passive import scan
from andamentum.proofread.sentences import segment


def test_regular_past_participle_caught():
    text = "The cake was eaten by the dog."
    findings = scan(text, segment(text))
    assert len(findings) >= 1
    assert any("eaten" in f.matched_text.lower() for f in findings)


def test_regular_ed_participle_caught():
    text = "The data was processed yesterday."
    findings = scan(text, segment(text))
    assert any("processed" in f.matched_text.lower() for f in findings)


def test_active_voice_not_caught():
    text = "The dog ate the cake."
    assert scan(text, segment(text)) == []


def test_adverb_between_be_and_participle():
    text = "The result was quickly verified."
    findings = scan(text, segment(text))
    assert any("verified" in f.matched_text.lower() for f in findings)


def test_offsets_roundtrip():
    text = "The cake was eaten. The data was processed."
    sentences = segment(text)
    findings = scan(text, sentences)
    for f in findings:
        assert text[f.span.start : f.span.end] == f.matched_text


def test_sentence_index_correct():
    text = "Cats are happy. The cake was eaten yesterday. The end."
    findings = scan(text, segment(text))
    eaten = [f for f in findings if "eaten" in f.matched_text.lower()]
    assert eaten
    assert eaten[0].sentence_index == 1
