"""Tests for the readability wrapper."""

from andamentum.proofread.readability import compute


def test_compute_empty_returns_zeros():
    r = compute("")
    assert r.word_count == 0
    assert r.sentence_count == 0
    assert r.smog_index == 0.0
    assert r.flesch_reading_ease == 0.0


def test_compute_simple_paragraph():
    text = (
        "The cat sat on the mat. The dog jumped over the lazy fox. "
        "Cats and dogs are very common house pets."
    )
    r = compute(text)
    assert r.word_count > 0
    assert r.sentence_count == 3
    # SMOG on a simple text should be a small grade level.
    assert 0.0 <= r.smog_index < 12.0
    # Flesch ease for simple text is high (closer to 100).
    assert r.flesch_reading_ease > 50.0
    # Sanity: avg sentence length ≈ word_count / sentence_count
    assert abs(r.avg_sentence_length - r.word_count / r.sentence_count) < 0.01


def test_compute_returns_floats_for_grade_scores():
    r = compute("This is a sentence. Here is another one.")
    for v in (
        r.smog_index,
        r.flesch_kincaid_grade,
        r.flesch_reading_ease,
        r.gunning_fog,
        r.coleman_liau_index,
        r.automated_readability_index,
        r.avg_sentence_length,
        r.avg_syllables_per_word,
    ):
        assert isinstance(v, float)
