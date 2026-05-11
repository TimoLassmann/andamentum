"""Tests for the regex sentence segmenter."""

from andamentum.proofread.sentences import index_for_offset, segment


def test_segment_empty():
    assert segment("") == []


def test_segment_single_sentence():
    s = segment("Hello world.")
    assert len(s) == 1
    assert s[0].text == "Hello world."
    assert s[0].start == 0
    assert s[0].end == 12


def test_segment_three_sentences():
    text = "The cat sat. The dog ran! Is that all?"
    s = segment(text)
    assert [x.text for x in s] == [
        "The cat sat.",
        "The dog ran!",
        "Is that all?",
    ]
    # Offsets must round-trip via slicing.
    for sent in s:
        assert text[sent.start : sent.end] == sent.text


def test_segment_does_not_split_on_abbreviation():
    text = "Dr. Smith arrived. He was on time."
    s = segment(text)
    assert len(s) == 2
    assert s[0].text == "Dr. Smith arrived."
    assert s[1].text == "He was on time."


def test_segment_trailing_fragment_without_terminator():
    text = "Hello world. No terminator here"
    s = segment(text)
    assert len(s) == 2
    assert s[1].text == "No terminator here"


def test_index_for_offset_finds_sentence():
    text = "Alpha. Beta. Gamma."
    sentences = segment(text)
    assert index_for_offset(sentences, 0) == 0  # 'A'
    assert index_for_offset(sentences, 7) == 1  # 'B'
    assert index_for_offset(sentences, 14) == 2  # 'G'


def test_index_for_offset_empty():
    assert index_for_offset([], 0) == -1


def test_index_for_offset_in_whitespace_snaps_to_last_started():
    sentences = segment("Alpha. Beta.")
    # offset 5 is the period of "Alpha." → sentence 0
    # offset 6 is the space between → still in sentence 0's [start,end)? No,
    # sentence 0 ends at 6. So 6 lands in inter-sentence space, snaps to 0.
    assert index_for_offset(sentences, 6) == 0
